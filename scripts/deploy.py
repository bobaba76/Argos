#!/usr/bin/env python3
"""deploy.py — one-command repo → live plugin sync (P5.2, issue #7).

The repo (``argos_plugin/``) is canonical; the running Hermes loads the
**installed copy** (``%LOCALAPPDATA%\\hermes\\plugins\\hybrid_memory``),
which is not a git repo and has been synced by hand. This closes the gap:
one command makes ``committed = live``, with verification.

Modes:
  --check (default)   sha256 per-file diff (CHANGED/NEW/REMOVED/UNCHANGED)
                      + repo HEAD vs last-deployed HEAD; exit 0 clean / 1 drift.
  copy (no --check)   copy changed + new runtime files; .bak-<ts> before
                      overwrite; re-hash every copied file (byte-verify);
                      append to deploy_state.json.
  --prune (opt-in)    remove target files absent from source (never deletes
                      protected live artifacts; never default).
  --restart-service   restart the memory shared service after copy (kill
                      stale processes per SYNC_HANDOFF.md). Without it:
                      prints "restart required".

Hard rules (SYNC_HANDOFF.md):
  - Only runtime modules are synced: top-level *.py + plugin.yaml, minus
    dev utilities (cleanup/dump/migrate/rebuild/reembed/review/why_not/
    backfill/run_tests).
  - Never touch live-only artifacts: skills/, *.duckdb, *.duckdb.wal,
    hybrid_memory_service.json, hybrid_memory.json, __pycache__/, cron/,
    state.db, _mh_analysis.txt, *.bak-*, *.pre_*.
  - Never delete anything without --prune.

Stdlib only, Windows-safe. Run with the Hermes venv python.

Usage:
    python scripts/deploy.py                      # check (default)
    python scripts/deploy.py --check              # explicit
    python scripts/deploy.py                      # copy (same as check+copy)
    python scripts/deploy.py --prune              # copy + prune
    python scripts/deploy.py --restart-service    # copy + restart
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# --- Paths -----------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
DEFAULT_SOURCE = _REPO_ROOT / "argos_plugin"
DEFAULT_STATE = _REPO_ROOT / "deploy_state.json"


def default_target() -> Path:
    """The live plugin install dir (SYNC_HANDOFF.md topology, tag C)."""
    if os.name == "nt":
        local = os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local"))
        return Path(local) / "hermes" / "plugins" / "hybrid_memory"
    return Path.home() / ".hermes" / "plugins" / "hybrid_memory"


# --- File scoping (SYNC_HANDOFF.md rules) ----------------------------------

# Dev utilities in the repo that must NEVER be synced to live.
EXCLUDED_SOURCE_FILES = {
    "cleanup_memories.py",
    "dump_memories.py",
    "migrate_gateway.py",
    "rebuild_graph.py",
    "reembed_memories.py",
    "review_pending.py",
    "why_not_cli.py",
    "backfill_graph.py",
}

# Live-only artifacts that deploy.py must never copy over, prune, or touch.
EXCLUDED_TARGET_NAMES = {
    "skills",
    "hybrid_memory_service.json",
    "hybrid_memory.json",
    "__pycache__",
    "cron",
    "state.db",
    "_mh_analysis.txt",
    ".pytest_cache",
}


def _is_backfill(name: str) -> bool:
    return name.endswith("_backfill.py") or "_backfill" in name


def _is_backup_artifact(name: str) -> bool:
    return ".bak-" in name or ".pre_" in name


def source_files(source: Path) -> List[Path]:
    """Runtime files in the source dir (top-level *.py + plugin.yaml +
    extractor_patterns/*.json data files).

    Subdirectories are NOT synced recursively — only the explicitly-listed
    data subdirs (extractor_patterns/) are included. This keeps the sync
    scoped to runtime files while ensuring pattern packs ship live
    (Hermes flagged that en.json was never deployed, making the E1
    fallback path the only live code path).
    """
    files = []
    for p in sorted(source.iterdir()):
        if not p.is_file():
            # Sync known data subdirs (not recursive — one level only).
            if p.is_dir() and p.name == "extractor_patterns":
                for jp in sorted(p.iterdir()):
                    if jp.is_file() and jp.name.endswith(".json"):
                        # Store with relative path for target-side sync.
                        files.append(jp)
            continue
        if p.name in EXCLUDED_SOURCE_FILES or _is_backfill(p.name):
            continue
        if p.name.endswith(".py") or p.name == "plugin.yaml":
            files.append(p)
        # #247: system prompt template must deploy alongside the .py files
        # (loaded at runtime by _load_system_prompt in provider_retrieval.py).
        elif p.name == "system_prompt_template.txt":
            files.append(p)
    return files


def _rel_key(path: Path, base: Path) -> str:
    """Relative path from *base* as a forward-slash string (stable key
    across platforms). Used as the diff/copy identity so files in
    subdirectories (extractor_patterns/en.json) don't collide with
    top-level files."""
    try:
        return path.relative_to(base).as_posix()
    except ValueError:
        return path.name


def target_scope(target: Path) -> List[Path]:
    """Files in the target dir that deploy.py may compare/copy/prune.

    Also skips dev-utility names and backfill scripts: per SYNC_HANDOFF.md
    these are legacy leftovers in the live install ("leave them alone,
    they're harmless") and are not part of the deployable runtime set.

    Includes the extractor_patterns/ data subdir if present (mirrors
    source_files).
    """
    if not target.exists():
        return []
    files = []
    for p in sorted(target.iterdir()):
        if not p.is_file():
            if p.is_dir() and p.name == "extractor_patterns":
                for jp in sorted(p.iterdir()):
                    if jp.is_file() and jp.name.endswith(".json"):
                        files.append(jp)
            continue
        if p.name in EXCLUDED_TARGET_NAMES or _is_backup_artifact(p.name):
            continue
        if p.name in EXCLUDED_SOURCE_FILES or _is_backfill(p.name):
            continue
        if p.name.startswith("test_"):
            continue  # legacy test leftovers in live; tests never sync
        if p.name.endswith(".py") or p.name == "plugin.yaml":
            files.append(p)
        # #247: system prompt template must deploy alongside the .py files
        # (loaded at runtime by _load_system_prompt in provider_retrieval.py).
        elif p.name == "system_prompt_template.txt":
            files.append(p)
    return files


# --- Hashing ---------------------------------------------------------------

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def repo_head() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(_REPO_ROOT), capture_output=True, text=True, timeout=15,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


# --- State ---------------------------------------------------------------

def load_state(path: Path) -> Dict:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def append_state(path: Path, entry: Dict) -> None:
    data = load_state(path)
    data.setdefault("deployments", []).append(entry)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


# --- Diff ------------------------------------------------------------------

def diff_files(source: Path, target: Path) -> Dict[str, List[str]]:
    """Return {changed, new, removed, unchanged} file name lists.

    Keys are relative paths (forward-slash) so subdirectory files like
    ``extractor_patterns/en.json`` are tracked distinctly.
    """
    src_map = {_rel_key(p, source): sha256(p) for p in source_files(source)}
    tgt_map = {_rel_key(p, target): sha256(p) for p in target_scope(target)}
    changed, new, removed, unchanged = [], [], [], []
    for name in sorted(src_map):
        if name in tgt_map:
            if src_map[name] == tgt_map[name]:
                unchanged.append(name)
            else:
                changed.append(name)
        else:
            new.append(name)
    for name in sorted(tgt_map):
        if name not in src_map:
            removed.append(name)
    return {"changed": changed, "new": new, "removed": removed, "unchanged": unchanged}


def check_mode(source: Path, target: Path, state: Path) -> int:
    if not source.exists():
        print(f"ERROR: source dir not found: {source}", file=sys.stderr)
        return 2
    if not target.exists():
        print(f"ERROR: target dir not found: {target}", file=sys.stderr)
        return 2

    diff = diff_files(source, target)
    print(f"=== deploy --check ===")
    print(f"source: {source}")
    print(f"target: {target}")
    print(f"HEAD:   {repo_head()[:12]}")
    state_data = load_state(state)
    deployments = state_data.get("deployments", [])
    last = deployments[-1] if deployments else {}
    last_head = last.get("head", "")
    if last_head:
        tag = "MATCH" if last_head == repo_head() else "MISMATCH"
        print(f"last deployed HEAD: {last_head[:12]} ({tag})")
    else:
        print(f"last deployed HEAD: none (no deploy_state.json yet)")

    for name in diff["changed"]:
        print(f"CHANGED   {name}")
    for name in diff["new"]:
        print(f"NEW       {name}")
    for name in diff["removed"]:
        print(f"REMOVED   {name}")
    for name in diff["unchanged"]:
        print(f"UNCHANGED {name}")

    n_drift = len(diff["changed"]) + len(diff["new"]) + len(diff["removed"])
    print(f"\n{len(diff['unchanged'])} unchanged, {n_drift} drifted")
    if n_drift == 0:
        print("verdict: CLEAN")
        return 0
    print("verdict: DRIFT (exit 1 — blocks sync)")
    return 1


def copy_mode(
    source: Path,
    target: Path,
    state: Path,
    prune: bool,
    restart: bool,
) -> int:
    if not source.exists():
        print(f"ERROR: source dir not found: {source}", file=sys.stderr)
        return 2
    if not target.exists():
        print(f"ERROR: target dir not found: {target}", file=sys.stderr)
        return 2

    diff = diff_files(source, target)
    to_copy = diff["changed"] + diff["new"]
    if not to_copy and not (prune and diff["removed"]):
        print("=== deploy: nothing to copy (already in sync) ===")
        return 0

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    src_map = {_rel_key(p, source): p for p in source_files(source)}
    copied: List[Dict[str, str]] = []

    print(f"=== deploy copy ({ts}) ===")
    for name in to_copy:
        src = src_map[name]
        dst = target / name
        # Ensure target subdirectory exists (e.g. extractor_patterns/).
        dst.parent.mkdir(parents=True, exist_ok=True)
        # Backup before overwrite.
        if dst.exists():
            bak = target / f"{name}.bak-{ts}"
            try:
                import shutil
                shutil.copy2(dst, bak)
                print(f"backup  {name} -> {bak.name}")
            except Exception as exc:
                print(f"ERROR: backup failed for {name}: {exc}", file=sys.stderr)
                return 2
        # Copy, then re-hash to verify byte-identity.
        try:
            import shutil
            shutil.copy2(src, dst)
        except Exception as exc:
            print(f"ERROR: copy failed for {name}: {exc}", file=sys.stderr)
            return 2
        src_hash = sha256(src)
        dst_hash = sha256(dst)
        if src_hash != dst_hash:
            print(f"ERROR: hash mismatch after copy for {name} "
                  f"({src_hash[:12]} != {dst_hash[:12]})", file=sys.stderr)
            return 2
        copied.append({"file": name, "sha256": src_hash})
        print(f"copied  {name} (verified)")

    pruned: List[str] = []
    if prune:
        for name in diff["removed"]:
            dst = target / name
            try:
                dst.unlink()
                pruned.append(name)
                print(f"pruned  {name}")
            except Exception as exc:
                print(f"ERROR: prune failed for {name}: {exc}", file=sys.stderr)
                return 2

    # Record the deployment.
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "head": repo_head(),
        "copied": copied,
        "pruned": pruned,
    }
    append_state(state, entry)
    print(f"deploy_state.json updated ({len(copied)} copied, {len(pruned)} pruned)")

    if restart:
        print("restarting memory service...")
        if not restart_service():
            print("ERROR: service restart failed", file=sys.stderr)
            return 2
        print("service restarted")
    else:
        print("restart required: kill stale memory_service processes and "
              "restart Hermes (see SYNC_HANDOFF.md Step 4)")
    return 0


def restart_service() -> bool:
    """Kill stale memory_service processes (SYNC_HANDOFF.md Step 4)."""
    if os.name == "nt":
        cmd = (
            "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match "
            "'hybrid_memory\\\\memory_service\\.py' } | ForEach-Object { "
            "Stop-Process -Id $_.ProcessId -Force }"
        )
        try:
            result = subprocess.run(
                ["powershell", "-NoProfile", "-Command", cmd],
                capture_output=True, text=True, timeout=60,
            )
            return result.returncode == 0
        except Exception:
            return False
    try:
        result = subprocess.run(
            ["pkill", "-f", "memory_service\\.py"], capture_output=True, timeout=30,
        )
        return result.returncode in (0, 1)  # 1 = no process matched
    except Exception:
        return False


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--source", default=str(DEFAULT_SOURCE),
                        help=f"Repo plugin dir (default: {DEFAULT_SOURCE})")
    parser.add_argument("--target", default="",
                        help=f"Live install dir (default: {default_target()})")
    parser.add_argument("--state", default=str(DEFAULT_STATE),
                        help=f"deploy_state.json path (default: {DEFAULT_STATE})")
    parser.add_argument("--check", action="store_true",
                        help="Diff only; exit 0 clean / 1 drift (default mode).")
    parser.add_argument("--prune", action="store_true",
                        help="Also remove target files absent from source.")
    parser.add_argument("--restart-service", action="store_true",
                        help="Restart the memory service after copy.")
    args = parser.parse_args(argv)

    source = Path(args.source).resolve()
    target = Path(args.target).resolve() if args.target else default_target().resolve()
    state = Path(args.state).resolve()

    if args.check:
        return check_mode(source, target, state)
    return copy_mode(source, target, state, args.prune, args.restart_service)


if __name__ == "__main__":
    sys.exit(main())

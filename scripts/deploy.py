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
  --atomic-swap       #308: staging-then-atomic-rename. Copy ALL runtime
                      files to a versioned staged dir
                      (``hybrid_memory.staged-<ts>``), verify byte-parity,
                      preserve live-only artifacts (extractor_patterns/,
                      skills/, eval/, tests/, state files) into the staged
                      dir, then atomically swap the live directory reference
                      (rename old live → ``hybrid_memory.bak-<ts>``,
                      rename staged → ``hybrid_memory``). The previous
                      live dir is preserved for rollback.
                      HARD PRECONDITION: Hermes and the memory service
                      MUST be stopped — the service's CWD is the live
                      plugin dir, and renaming it while the process is
                      alive fails on Windows (WinError 32).
  --rollback          #308: roll back to the previous versioned live dir.
                      Reads deploy_state.json for the last backup dir,
                      renames current live → ``hybrid_memory.failed-<ts>``,
                      renames the backup back to ``hybrid_memory``.
  --list-versions     #308: list available versioned live dirs for rollback.
  --prune (opt-in)    remove target files absent from source (never deletes
                      protected live artifacts; never default).
  --restart-service   restart the memory shared service after copy (kill
                      stale processes per SYNC_HANDOFF.md). Without it:
                      prints "restart required".

Hard rules (SYNC_HANDOFF.md):
  - Only runtime modules are synced: top-level *.py + plugin.yaml, minus
    dev utilities (cleanup/dump/migrate/rebuild/reembed/review/why_not/
    backfill).
  - Never touch live-only artifacts: skills/, *.duckdb, *.duckdb.wal,
    hybrid_memory_service.json, hybrid_memory.json, __pycache__/, cron/,
    state.db, _mh_analysis.txt, *.bak-*, *.pre_*.
  - Never delete anything without --prune.

#308 atomic swap: the live dir is swapped via os.replace (atomic on the
same filesystem). The previous live dir is preserved as
``hybrid_memory.bak-<ts>`` for rollback. Only the most recent backup is
kept by --rollback; older backups can be cleaned manually.

Stdlib only, Windows-safe. Run with the Hermes venv python.

Usage:
    python scripts/deploy.py                      # check (default)
    python scripts/deploy.py --check              # explicit
    python scripts/deploy.py                      # copy (same as check+copy)
    python scripts/deploy.py --atomic-swap        # copy + atomic swap
    python scripts/deploy.py --rollback           # roll back to previous
    python scripts/deploy.py --list-versions      # list available versions
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


# --- #308: atomic swap / versioned rollback --------------------------------

def _copy_tree(src: Path, dst: Path, files: List[Path], source_base: Path) -> List[Dict[str, str]]:
    """Copy a list of files from src to dst, preserving relative structure.

    Returns a list of {file, sha256} dicts. Raises on copy or hash failure.
    """
    import shutil
    copied: List[Dict[str, str]] = []
    for p in files:
        rel = _rel_key(p, source_base)
        dst_path = dst / rel
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dst_path)
        src_hash = sha256(p)
        dst_hash = sha256(dst_path)
        if src_hash != dst_hash:
            raise RuntimeError(
                f"byte-parity verification failed for {rel} "
                f"({src_hash[:12]} != {dst_hash[:12]})"
            )
        copied.append({"file": rel, "sha256": src_hash})
    return copied


def _find_versioned_backups(target: Path) -> List[Path]:
    """Find versioned backup dirs (hybrid_memory.bak-*) sorted newest first."""
    parent = target.parent
    name = target.name
    backups = []
    if not parent.exists():
        return backups
    for p in sorted(parent.iterdir(), reverse=True):
        if p.is_dir() and p.name.startswith(f"{name}.bak-"):
            backups.append(p)
    return backups


def _is_service_running() -> bool:
    """Check if a memory_service.py process is alive."""
    if os.name == "nt":
        cmd = (
            "Get-CimInstance Win32_Process | Where-Object { "
            "$_.CommandLine -match 'hybrid_memory\\\\memory_service\\.py' } | "
            "Measure-Object | Select-Object -ExpandProperty Count"
        )
        try:
            result = subprocess.run(
                ["powershell", "-NoProfile", "-Command", cmd],
                capture_output=True, text=True, timeout=15,
            )
            return result.returncode == 0 and int(result.stdout.strip()) > 0
        except Exception:
            return False
    try:
        result = subprocess.run(
            ["pgrep", "-f", "memory_service\\.py"],
            capture_output=True, text=True, timeout=10,
        )
        return result.returncode == 0 and bool(result.stdout.strip())
    except Exception:
        return False


def _preserve_live_only_artifacts(target: Path, staged: Path) -> List[str]:
    """Copy live-only artifacts from the current live dir into the staged dir.

    Live-only artifacts are files/dirs that exist in the live install but
    are NOT in the repo source: extractor_patterns/ (loaded at runtime by
    extractor.py:112), skills/ (insight-log), eval/, tests/, .bak-* backups,
    local plugin.yaml edits, state files, etc.

    Without this, the atomic swap would evacuate these artifacts —
    extractor.py would silently degrade, skills would vanish, etc.
    """
    import shutil
    preserved: List[str] = []
    if not target.exists():
        return preserved

    # Items to skip during preservation — these are either regenerated
    # on import (__pycache__, .pytest_cache) or are deploy artifacts
    # (backup dirs, staged dirs, failed dirs). Unlike EXCLUDED_TARGET_NAMES,
    # this does NOT skip skills/, hybrid_memory.json, etc. — those ARE
    # preserved by the atomic swap.
    _PRESERVE_SKIP = {
        "__pycache__", ".pytest_cache",
        "hybrid_memory_service.json", "hybrid_memory_service.starting",
    }

    for item in sorted(target.iterdir()):
        name = item.name
        # Skip regenerated/cache dirs and deploy-internal files.
        if name in _PRESERVE_SKIP:
            continue
        # Skip .bak-* and .pre_* artifacts (backup dirs from previous swaps).
        if _is_backup_artifact(name):
            continue
        # Skip .staged-* dirs (from in-progress deploys).
        if ".staged-" in name:
            continue
        # Skip .failed-* dirs (from failed rollbacks).
        if ".failed-" in name:
            continue
        # Skip files that are in the source set — they'll be overwritten
        # by the source copy. We only preserve live-ONLY artifacts.
        # Source files are .py, plugin.yaml, system_prompt_template.txt.
        is_source_file = (
            item.is_file() and (
                name.endswith(".py") or name == "plugin.yaml" or
                name == "system_prompt_template.txt"
            )
        )
        if is_source_file:
            continue  # source will provide the new version
        # extractor_patterns/ is a source data dir, but it may also have
        # live-only locale files not in the repo. Preserve the whole dir
        # by merging: copy live-only files that aren't in the staged dir.
        if item.is_dir() and name == "extractor_patterns":
            staged_sub = staged / name
            if staged_sub.exists():
                # Merge: copy any live files not already staged.
                for jp in sorted(item.iterdir()):
                    if jp.is_file() and jp.name.endswith(".json"):
                        dst = staged_sub / jp.name
                        if not dst.exists():
                            shutil.copy2(jp, dst)
                            preserved.append(f"{name}/{jp.name}")
            else:
                # Not in source — copy the whole dir.
                shutil.copytree(item, staged_sub)
                preserved.append(name)
            continue
        # All other dirs (skills/, eval/, tests/, cron/, etc.) and
        # non-source files (state.db, *.json, etc.) are live-only.
        dst = staged / name
        if not dst.exists():
            try:
                if item.is_dir():
                    shutil.copytree(item, dst)
                else:
                    shutil.copy2(item, dst)
                preserved.append(name)
            except Exception as exc:
                print(f"WARNING: could not preserve {name}: {exc}",
                      file=sys.stderr)
    return preserved


def atomic_swap_mode(
    source: Path,
    target: Path,
    state: Path,
    restart: bool,
) -> int:
    """#308: staging-then-atomic-rename deploy.

    1. Check that the memory service is NOT running (hard precondition:
       on Windows, renaming a dir that is a running process's CWD fails
       with WinError 32. The service runs with cwd=plugin_dir.)
    2. Copy ALL runtime files to a versioned staged dir.
    3. Verify byte-parity between staged and source.
    4. Preserve live-only artifacts (extractor_patterns/, skills/, eval/,
       tests/, .bak-*, state files) by copying them into the staged dir.
    5. Atomically swap: rename old live → backup, rename staged → live.
    6. Record the deployment + backup dir in deploy_state.json.

    Hard precondition: Hermes MUST be stopped before --atomic-swap.
    The service's CWD is the live plugin dir; renaming it while the
    service is alive fails on Windows (WinError 32).
    """
    if not source.exists():
        print(f"ERROR: source dir not found: {source}", file=sys.stderr)
        return 2
    if not target.exists():
        print(f"ERROR: target dir not found: {target}", file=sys.stderr)
        return 2

    # Hard precondition: service must be stopped before swap.
    if _is_service_running():
        print(
            "ERROR: memory_service.py is running. --atomic-swap renames the\n"
            "  live plugin dir, which is the service's CWD — on Windows this\n"
            "  fails with WinError 32 (sharing violation). Stop Hermes and\n"
            "  the memory service first, then re-run:\n"
            "    python scripts/deploy.py --atomic-swap",
            file=sys.stderr,
        )
        return 2

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    parent = target.parent
    staged = parent / f"{target.name}.staged-{ts}"
    backup = parent / f"{target.name}.bak-{ts}"
    # Handle name collisions (e.g. two deploys in the same second).
    _counter = 0
    while backup.exists():
        _counter += 1
        backup = parent / f"{target.name}.bak-{ts}-{_counter}"
    _counter = 0
    while staged.exists():
        _counter += 1
        staged = parent / f"{target.name}.staged-{ts}-{_counter}"

    # Clean up any stale staged dir from a failed previous run.
    if staged.exists():
        import shutil
        shutil.rmtree(staged, ignore_errors=True)

    print(f"=== deploy --atomic-swap ({ts}) ===")
    print(f"source: {source}")
    print(f"target: {target}")
    print(f"staged: {staged}")

    # Phase 1: copy all runtime files to the staged dir.
    files = source_files(source)
    if not files:
        print("ERROR: no source files to deploy", file=sys.stderr)
        return 2

    staged.mkdir(parents=True, exist_ok=True)
    try:
        copied = _copy_tree(source, staged, files, source)
    except Exception as exc:
        print(f"ERROR: staging failed: {exc}", file=sys.stderr)
        import shutil
        shutil.rmtree(staged, ignore_errors=True)
        return 2

    print(f"staged {len(copied)} files (byte-parity verified)")

    # Phase 2: preserve live-only artifacts into the staged dir.
    # Without this, the swap would evacuate extractor_patterns/ (loaded
    # at runtime by extractor.py:112), skills/ (insight-log), eval/,
    # tests/, local state files, etc.
    preserved = _preserve_live_only_artifacts(target, staged)
    if preserved:
        print(f"preserved {len(preserved)} live-only artifacts: "
              f"{', '.join(preserved[:10])}"
              + (" ..." if len(preserved) > 10 else ""))

    # Phase 3: atomic swap.
    # On Windows, os.replace works for dirs on the same volume but only
    # if the target dir is empty or doesn't exist. We rename instead:
    #   target → backup, staged → target
    # If the rename of target → backup fails, we abort (no harm done).
    # If the rename of staged → target fails after target → backup
    # succeeded, we try to rename backup → target back (recovery).
    try:
        target.rename(backup)
        print(f"renamed live → {backup.name}")
    except Exception as exc:
        print(f"ERROR: cannot rename live dir to backup: {exc}", file=sys.stderr)
        print("  Hint: ensure Hermes and the memory service are stopped.",
              file=sys.stderr)
        import shutil
        shutil.rmtree(staged, ignore_errors=True)
        return 2

    try:
        staged.rename(target)
        print(f"renamed staged → {target.name}")
    except Exception as exc:
        print(f"ERROR: atomic swap failed during final rename: {exc}",
              file=sys.stderr)
        print("attempting recovery: renaming backup back to live...", file=sys.stderr)
        try:
            backup.rename(target)
            print("recovery: live dir restored")
        except Exception as exc2:
            print(f"FATAL: recovery failed: {exc2}", file=sys.stderr)
            print(f"live dir is at: {backup}", file=sys.stderr)
            print(f"manual recovery: rename {backup} → {target}", file=sys.stderr)
        return 2

    # Phase 4: record deployment.
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "head": repo_head(),
        "mode": "atomic-swap",
        "copied": copied,
        "preserved": preserved,
        "backup_dir": str(backup),
        "staged_ts": ts,
    }
    append_state(state, entry)
    print(f"deploy_state.json updated (atomic-swap, {len(copied)} files, "
          f"{len(preserved)} preserved)")
    print(f"backup: {backup}")
    print(f"rollback: python scripts/deploy.py --rollback")

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


def rollback_mode(target: Path, state: Path) -> int:
    """#308: roll back to the previous versioned live dir.

    Reads deploy_state.json for the last backup dir, renames current
    live → failed, renames backup → live.
    """
    if not target.exists():
        print(f"ERROR: target dir not found: {target}", file=sys.stderr)
        return 2

    state_data = load_state(state)
    deployments = state_data.get("deployments", [])
    # Find the last atomic-swap deployment with a backup_dir.
    last_backup = None
    for dep in reversed(deployments):
        if dep.get("mode") == "atomic-swap" and dep.get("backup_dir"):
            last_backup = Path(dep["backup_dir"])
            break

    if last_backup is None or not last_backup.exists():
        # Fall back to scanning for backup dirs.
        backups = _find_versioned_backups(target)
        if not backups:
            print("ERROR: no versioned backup found for rollback", file=sys.stderr)
            return 2
        last_backup = backups[0]
        print(f"WARNING: no deploy_state entry, using newest backup: {last_backup}")

    if not last_backup.exists():
        print(f"ERROR: backup dir does not exist: {last_backup}", file=sys.stderr)
        return 2

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    failed = target.parent / f"{target.name}.failed-{ts}"

    print(f"=== deploy --rollback ({ts}) ===")
    print(f"current live: {target}")
    print(f"rolling back to: {last_backup}")

    try:
        target.rename(failed)
        print(f"renamed current live → {failed.name}")
    except Exception as exc:
        print(f"ERROR: cannot rename current live dir: {exc}", file=sys.stderr)
        return 2

    try:
        last_backup.rename(target)
        print(f"renamed backup → {target.name}")
    except Exception as exc:
        print(f"ERROR: rollback rename failed: {exc}", file=sys.stderr)
        print("attempting recovery: renaming failed back to live...", file=sys.stderr)
        try:
            failed.rename(target)
            print("recovery: live dir restored (rollback aborted)")
        except Exception as exc2:
            print(f"FATAL: recovery failed: {exc2}", file=sys.stderr)
            print(f"live dir is at: {failed}", file=sys.stderr)
        return 2

    # Record the rollback.
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": "rollback",
        "rolled_back_to": str(last_backup),
        "failed_dir": str(failed),
    }
    append_state(state, entry)
    print(f"rollback complete. failed dir preserved at: {failed}")
    print("restart required: kill stale memory_service processes and "
          "restart Hermes (see SYNC_HANDOFF.md Step 4)")
    return 0


def list_versions_mode(target: Path, state: Path) -> int:
    """#308: list available versioned live dirs for rollback."""
    print(f"=== deploy --list-versions ===")
    print(f"live dir: {target}")
    print()

    backups = _find_versioned_backups(target)
    if not backups:
        print("No versioned backups found.")
        return 0

    state_data = load_state(state)
    deployments = state_data.get("deployments", [])

    print(f"Available backups (newest first):")
    for i, bak in enumerate(backups):
        # Try to find the matching deploy_state entry.
        head = ""
        ts_label = bak.name.split(".bak-", 1)[-1] if ".bak-" in bak.name else ""
        for dep in reversed(deployments):
            if dep.get("backup_dir") and Path(dep["backup_dir"]).resolve() == bak.resolve():
                head = dep.get("head", "")[:12]
                break
        head_str = f" HEAD:{head}" if head else ""
        print(f"  [{i}] {bak.name} (ts:{ts_label}){head_str}")

    # Also list failed dirs (from failed rollbacks).
    parent = target.parent
    failed_dirs = []
    if parent.exists():
        for p in sorted(parent.iterdir(), reverse=True):
            if p.is_dir() and p.name.startswith(f"{target.name}.failed-"):
                failed_dirs.append(p)
    if failed_dirs:
        print()
        print(f"Failed dirs (from rollbacks):")
        for i, fd in enumerate(failed_dirs):
            print(f"  [{i}] {fd.name}")

    return 0


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
    parser.add_argument("--atomic-swap", action="store_true",
                        help="#308: staging-then-atomic-rename deploy.")
    parser.add_argument("--rollback", action="store_true",
                        help="#308: roll back to the previous versioned live dir.")
    parser.add_argument("--list-versions", action="store_true",
                        help="#308: list available versioned live dirs for rollback.")
    parser.add_argument("--prune", action="store_true",
                        help="Also remove target files absent from source.")
    parser.add_argument("--restart-service", action="store_true",
                        help="Restart the memory service after copy.")
    args = parser.parse_args(argv)

    source = Path(args.source).resolve()
    target = Path(args.target).resolve() if args.target else default_target().resolve()
    state = Path(args.state).resolve()

    if args.list_versions:
        return list_versions_mode(target, state)
    if args.rollback:
        return rollback_mode(target, state)
    if args.atomic_swap:
        return atomic_swap_mode(source, target, state, args.restart_service)
    if args.check:
        return check_mode(source, target, state)
    return copy_mode(source, target, state, args.prune, args.restart_service)


if __name__ == "__main__":
    sys.exit(main())

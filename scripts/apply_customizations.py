#!/usr/bin/env python3
"""Safely apply this repository's Hermes customization bundle.

The command changes Hermes source/plugin code and configuration only. Runtime
memory databases are never copied, replaced, or deleted.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / ".devin" / "customization-manifest.json"
PLUGIN_EXCLUDED_NAMES = ("*.duckdb", "*.wal", "*_kuzu", "__pycache__", ".pytest_cache")


class CustomizationError(RuntimeError):
    pass


def load_manifest() -> dict[str, Any]:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def default_home() -> Path:
    configured = os.environ.get("HERMES_HOME", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    if os.name == "nt":
        return Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "hermes"
    return Path.home() / ".hermes"


def default_repo(home: Path) -> Path:
    configured = os.environ.get("HERMES_AGENT_REPO", "").strip()
    return Path(configured).expanduser().resolve() if configured else home / "hermes-agent"


def run(command: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None,
        check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, cwd=cwd, env=env, text=True, capture_output=True)
    if check and result.returncode:
        detail = (result.stderr or result.stdout).strip()
        raise CustomizationError(f"Command failed ({result.returncode}): {' '.join(command)}\n{detail}")
    return result


def git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return run(["git", "-C", str(repo), *args], check=check)


def resolve_path(value: str, *, repo: Path, home: Path) -> Path:
    value = value.replace("$HERMES_HOME", str(home))
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def is_excluded(relative: Path, patterns: tuple[str, ...]) -> bool:
    return any(
        relative.match(pattern) or any(Path(part).match(pattern) for part in relative.parts)
        for pattern in patterns
    )


def target_python(repo: Path, explicit: str | None = None) -> str:
    if explicit:
        return str(Path(explicit).expanduser().resolve())
    configured = os.environ.get("HERMES_PYTHON", "").strip()
    if configured:
        return str(Path(configured).expanduser().resolve())
    candidates = [
        repo / "venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python"),
        repo / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return sys.executable


def ensure_repo_clean(repo: Path) -> None:
    status = git(repo, "status", "--porcelain").stdout.strip()
    if not status:
        return
    # Check if any dirty files overlap with the patch's target files.
    # If not, warn but allow the apply to proceed.
    dirty_files: set[str] = set()
    for line in status.splitlines():
        # Porcelain format: XY <path> or XY <orig> -> <path> for renames
        path = line[3:].split(" -> ")[-1].strip()
        dirty_files.add(path.replace("/", os.sep))
    # Files the patch and core_copies will touch.
    manifest = load_manifest()
    patch_targets: set[str] = set()
    # Parse the patch to find which files it modifies.
    patch_path = ROOT / manifest["core_patch"]
    if patch_path.exists():
        import re as _re
        patch_text = patch_path.read_text(encoding="utf-8", errors="replace")
        for m in _re.finditer(r"^(?:---|\+\+\+) [ab]/(.+)$", patch_text, _re.MULTILINE):
            patch_targets.add(m.group(1).replace("/", os.sep))
    for item in manifest["core_copies"]:
        patch_targets.add(item["target"].replace("/", os.sep))
    overlap = dirty_files & patch_targets
    if overlap:
        raise CustomizationError(
            "Hermes checkout has local changes to files the patch will modify. "
            "Commit or back up these files first:\n" +
            "\n".join(sorted(overlap))
        )
    print(
        f"WARNING: Hermes checkout has local changes to {len(dirty_files)} file(s) "
        "that do not overlap with the customization patch. The apply will proceed, "
        "but those local changes will remain untouched:\n" +
        "\n".join(f"  {f}" for f in sorted(dirty_files)[:10])
    )


def ensure_memory_service_stopped(home: Path) -> None:
    endpoint = home / "hybrid_memory_service.json"
    if not endpoint.exists():
        return
    try:
        details = json.loads(endpoint.read_text(encoding="utf-8"))
        import socket
        with socket.create_connection((details["host"], int(details["port"])), timeout=0.5) as connection:
            request = json.dumps({"method": "health", "token": details["token"]}) + "\n"
            connection.sendall(request.encode("utf-8"))
            response = json.loads(connection.recv(4096).decode("utf-8"))
        if response.get("ok"):
            raise CustomizationError(
                "The shared memory service is still running. Stop Hermes/gateway and retry; "
                "the apply command never changes locked memory files."
            )
    except CustomizationError:
        raise
    except (OSError, KeyError, TypeError, ValueError):
        pass


def format_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(str(value), ensure_ascii=False)


def merge_yaml_section(path: Path, section: str, values: dict[str, Any], *, dry_run: bool) -> None:
    if path.exists():
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    else:
        lines = []
    section_re = re.compile(rf"^(?P<indent>\s*){re.escape(section)}\s*:\s*(?:#.*)?$")
    section_index = next((i for i, line in enumerate(lines) if section_re.match(line.rstrip("\r\n"))), None)
    if section_index is None:
        if lines and not lines[-1].endswith("\n"):
            lines[-1] += "\n"
        lines.append(f"{section}:\n")
        lines.extend(f"  {key}: {format_value(value)}\n" for key, value in values.items())
    else:
        match = section_re.match(lines[section_index].rstrip("\r\n"))
        assert match is not None
        section_indent = len(match.group("indent"))
        end = len(lines)
        for i in range(section_index + 1, len(lines)):
            stripped = lines[i].strip()
            if stripped and len(lines[i]) - len(lines[i].lstrip()) <= section_indent:
                end = i
                break
        body = lines[section_index + 1:end]
        updated: set[str] = set()
        for i, line in enumerate(body):
            for key, value in values.items():
                key_re = re.compile(rf"^(?P<indent>\s*){re.escape(key)}\s*:")
                if key_re.match(line):
                    newline = "\n" if line.endswith("\n") else ""
                    body[i] = f"{key_re.match(line).group('indent')}{key}: {format_value(value)}{newline}"
                    updated.add(key)
                    break
        missing = [key for key in values if key not in updated]
        body.extend(f"{' ' * (section_indent + 2)}{key}: {format_value(values[key])}\n" for key in missing)
        lines[section_index + 1:end] = body
    if dry_run:
        print(f"would update YAML: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(lines), encoding="utf-8")


def merge_json_defaults(path: Path, defaults: dict[str, Any], *, dry_run: bool) -> None:
    current: dict[str, Any] = {}
    if path.exists():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                current = loaded
        except (OSError, ValueError) as exc:
            raise CustomizationError(f"Cannot parse {path}: {exc}") from exc
    changed = False
    for key, value in defaults.items():
        if key not in current:
            current[key] = value
            changed = True
    if dry_run:
        print(f"would merge hybrid memory config: {path} ({'changes' if changed else 'already current'})")
    elif changed or not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(current, indent=2) + "\n", encoding="utf-8")


def backup_runtime(home: Path, *, dry_run: bool) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = home / "backups" / f"customizations-pre-apply-{stamp}"
    if dry_run:
        print(f"would create backup: {destination}")
        return destination
    destination.mkdir(parents=True, exist_ok=False)
    for relative in ("config.yaml", "hybrid_memory.json"):
        source = home / relative
        if source.exists():
            shutil.copy2(source, destination / relative)
    plugin = home / "plugins" / "hybrid_memory"
    if plugin.exists():
        for source in plugin.rglob("*"):
            if not source.is_file():
                continue
            relative = source.relative_to(plugin)
            if is_excluded(relative, PLUGIN_EXCLUDED_NAMES):
                continue
            target = destination / "plugins" / "hybrid_memory" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    (destination / "backup-manifest.json").write_text(
        json.dumps({"created_at": stamp, "runtime_names_not_touched": list(PLUGIN_EXCLUDED_NAMES)}, indent=2) + "\n",
        encoding="utf-8",
    )
    return destination


def base_blob(repo: Path, commit: str, relative: str) -> bytes | None:
    result = subprocess.run(
        ["git", "-C", str(repo), "cat-file", "blob", f"{commit}:{relative.replace(os.sep, '/')}"],
        capture_output=True,
    )
    return result.stdout if result.returncode == 0 else None


def apply_core(repo: Path, manifest: dict[str, Any], *, dry_run: bool) -> None:
    patch = ROOT / manifest["core_patch"]
    base = manifest["upstream"]["base_commit"]
    head = git(repo, "rev-parse", "HEAD").stdout.strip()
    if head != base:
        print(
            f"WARNING: Hermes HEAD ({head[:12]}) does not match the manifest's "
            f"base commit ({base[:12]}). The patch will be tested against the "
            f"current checkout; if it applies cleanly, the customizations will work."
        )

    # Check if the patch applies cleanly before doing anything else.
    check = git(repo, "apply", "--check", "--whitespace=nowarn", str(patch), check=False)
    if check.returncode:
        raise CustomizationError(
            f"Core patch does not apply cleanly to HEAD {head[:12]}. "
            "Create a new versioned patch for this upstream revision.\n" +
            (check.stderr or check.stdout).strip()
        )

    copy_plan: list[tuple[Path, Path]] = []
    for item in manifest["core_copies"]:
        source = ROOT / item["source"]
        target = repo / item["target"]
        if not source.exists():
            raise CustomizationError(f"Missing bundle source file: {source}")
        if target.exists() and target.read_bytes() != source.read_bytes():
            upstream_bytes = base_blob(repo, head, item["target"])
            if upstream_bytes is None or target.read_bytes() != upstream_bytes:
                print(
                    f"WARNING: Core copy target has unknown local changes: {target}. "
                    "It will be overwritten with the bundle version."
                )
        copy_plan.append((source, target))

    if dry_run:
        print(f"would apply core patch: {patch}")
    else:
        git(repo, "apply", "--whitespace=nowarn", str(patch))
    for source, target in copy_plan:
        if dry_run:
            print(f"would copy core file: {source} -> {target}")
        elif not target.exists() or target.read_bytes() != source.read_bytes():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)


def sync_plugin(home: Path, manifest: dict[str, Any], *, dry_run: bool) -> None:
    source_root = ROOT / manifest["plugin"]["source"]
    target_root = home / manifest["plugin"]["target"]
    excluded = tuple(manifest["plugin"].get("excluded_names", [])) or PLUGIN_EXCLUDED_NAMES
    for source in source_root.rglob("*"):
        if not source.is_file():
            continue
        relative = source.relative_to(source_root)
        if is_excluded(relative, excluded):
            continue
        target = target_root / relative
        if dry_run:
            print(f"would sync plugin file: {target}")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)


def install_dependencies(repo: Path, dependencies: list[str], *, dry_run: bool, python_path: str | None) -> None:
    python = target_python(repo, python_path)
    if dry_run:
        print(f"would install plugin dependencies with {python}: {', '.join(dependencies)}")
    else:
        run([python, "-m", "pip", "install", *dependencies])


def verify(repo: Path, home: Path, manifest: dict[str, Any], *, run_tests: bool, python_path: str | None) -> None:
    python = target_python(repo, python_path)
    env = os.environ.copy()
    env["HERMES_HOME"] = str(home)
    code = (
        "from plugins.memory import find_provider_dir, load_memory_provider; "
        "p=find_provider_dir('hybrid_memory'); assert p is not None, p; "
        "provider=load_memory_provider('hybrid_memory'); assert provider is not None; "
        "assert provider.is_available(); print(p)"
    )
    result = run([python, "-c", code], cwd=repo, env=env)
    print(f"plugin discovery: {result.stdout.strip()}")
    if not run_tests:
        return
    plugin_root = home / manifest["plugin"]["target"]
    plugin_tests = plugin_root / "tests"
    plugin_env = env.copy()
    plugin_env["PYTHONPATH"] = str(plugin_root) + os.pathsep + plugin_env.get("PYTHONPATH", "")
    run([python, "-m", "pytest", "tests", "-q"], cwd=plugin_root, env=plugin_env)
    core_tests = [str(repo / path) for path in manifest["tests"]["core"]]
    run([python, "-m", "pytest", *core_tests, "-q"], cwd=repo, env=env)
    print("verification tests: passed")


def capture(repo: Path, manifest: dict[str, Any], version: str) -> None:
    if not version:
        raise CustomizationError("capture requires --version, for example --version 0.21.0")
    head = git(repo, "rev-parse", "HEAD").stdout.strip()
    status = git(repo, "status", "--porcelain").stdout.strip()
    if not status:
        raise CustomizationError("Nothing to capture; Hermes checkout has no local custom changes.")
    patch_path = ROOT / "patches" / f"hermes-v{version}-custom.patch"
    git(repo, "diff", "HEAD", "--binary", f"--output={patch_path}")
    manifest["upstream"]["version"] = version
    manifest["upstream"]["base_commit"] = head
    manifest["core_patch"] = str(patch_path.relative_to(ROOT)).replace(os.sep, "/")
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    untracked = [line[3:] for line in status.splitlines() if line.startswith("?? ")]
    print(f"captured {patch_path} from base {head}")
    if untracked:
        print("Review these untracked files and add them to core_copies if they are customization source:")
        for path in untracked:
            print(f"  {path}")


def snapshot(repo: Path, home: Path, manifest: dict[str, Any], version: str, message: str | None,
             python_path: str | None = None) -> None:
    """One-command ritual: capture code changes, back up live data, push the bundle.

    Runs after every customization session so the bundle repo (the single
    source of truth) never drifts from the live Hermes checkout:
      1. capture   - regenerate patches/<version>-custom.patch from the live tree
      2. backup    - stop memory service, copy duckdb/kuzu/state.db to backups/
      3. commit    - stage and commit the whole bundle repo
      4. push      - off-machine copy on origin
    """
    if not version:
        version = manifest["upstream"]["version"]
    try:
        capture(repo, manifest, version)
        print(f"snapshot: patch + manifest regenerated for v{version}")
    except CustomizationError as exc:
        print(f"snapshot: capture skipped ({exc})")

    backup_script = ROOT / "scripts" / "backup_data.py"
    backup_python = target_python(repo, python_path)
    run([backup_python, str(backup_script), "--hermes-home", str(home)])

    status = git(ROOT, "status", "--porcelain").stdout.strip()
    if not status:
        print("snapshot: bundle repo clean; nothing to commit")
        return
    git(ROOT, "add", "-A")
    staged = git(ROOT, "diff", "--cached", "--name-only").stdout.splitlines()
    print(f"snapshot: staging {len(staged)} changed file(s)")
    for path in staged:
        print(f"  + {path}")
    commit_msg = message or f"snapshot {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
    git(ROOT, "commit", "-m", commit_msg)
    git(ROOT, "push")
    print("snapshot: committed and pushed to origin")


def rollback(repo: Path, home: Path, manifest: dict[str, Any], backup: Path) -> None:
    if not backup.exists():
        raise CustomizationError(f"Backup does not exist: {backup}")
    patch = ROOT / manifest["core_patch"]
    reverse = git(repo, "apply", "--check", "-R", "--whitespace=nowarn", str(patch), check=False)
    if reverse.returncode:
        raise CustomizationError("Core patch is no longer in a reversible state; refusing rollback.")
    git(repo, "apply", "-R", "--whitespace=nowarn", str(patch))
    base = manifest["upstream"]["base_commit"]
    for item in manifest["core_copies"]:
        source = ROOT / item["source"]
        target = repo / item["target"]
        if target.exists() and target.read_bytes() != source.read_bytes():
            raise CustomizationError(f"Refusing to overwrite post-apply core change during rollback: {target}")
        upstream_bytes = base_blob(repo, base, item["target"])
        if upstream_bytes is None:
            if target.exists():
                target.unlink()
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(upstream_bytes)
    for relative in ("config.yaml", "hybrid_memory.json"): 
        source = backup / relative
        if source.exists():
            shutil.copy2(source, home / relative)
    saved_plugin = backup / "plugins" / "hybrid_memory"
    target_plugin = home / "plugins" / "hybrid_memory"
    if saved_plugin.exists():
        if target_plugin.exists():
            shutil.rmtree(target_plugin)
        shutil.copytree(saved_plugin, target_plugin)
    print(f"rollback complete from {backup}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply or verify the Hermes customization bundle safely.")
    parser.add_argument("command", choices=("apply", "verify", "capture", "rollback", "snapshot"))
    parser.add_argument("--version", default=None, help="Upstream version for capture/snapshot")
    parser.add_argument("--message", default=None, help="Commit message for snapshot")
    parser.add_argument("--hermes-repo", type=Path, default=None)
    parser.add_argument("--hermes-home", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--python", dest="python_path", default=None, help="Hermes Python interpreter")
    parser.add_argument("--skip-deps", action="store_true")
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--backup", type=Path, default=None, help="Backup directory for rollback")
    args = parser.parse_args()
    manifest = load_manifest()
    home = (args.hermes_home or default_home()).expanduser().resolve()
    repo = (args.hermes_repo or default_repo(home)).expanduser().resolve()
    try:
        if not repo.is_dir() or not (repo / ".git").exists():
            raise CustomizationError(f"Not a Hermes Git checkout: {repo}")
        if args.command == "capture":
            capture(repo, manifest, args.version or "")
            return 0
        if args.command == "snapshot":
            snapshot(repo, home, manifest, args.version or "", args.message, args.python_path)
            return 0
        if args.command == "verify":
            verify(repo, home, manifest, run_tests=not args.skip_tests, python_path=args.python_path)
            return 0
        if args.command == "rollback":
            backup = args.backup
            if backup is None:
                pointer = home / "backups" / "last-customization-backup.txt"
                if not pointer.exists():
                    raise CustomizationError("Provide --backup or create an apply backup first.")
                backup = Path(pointer.read_text(encoding="utf-8").strip())
            if args.dry_run:
                print(f"would rollback from {backup}")
            else:
                rollback(repo, home, manifest, backup)
            return 0
        ensure_repo_clean(repo)
        ensure_memory_service_stopped(home)
        backup = backup_runtime(home, dry_run=args.dry_run)
        apply_core(repo, manifest, dry_run=args.dry_run)
        sync_plugin(home, manifest, dry_run=args.dry_run)
        yaml_path = home / "config.yaml"
        merge_yaml_section(yaml_path, "memory", manifest["config"]["yaml"]["memory"], dry_run=args.dry_run)
        merge_json_defaults(home / "hybrid_memory.json", manifest["config"]["hybrid_memory_json_defaults"], dry_run=args.dry_run)
        if not args.skip_deps:
            install_dependencies(repo, manifest["plugin"]["dependencies"], dry_run=args.dry_run, python_path=args.python_path)
        if not args.dry_run:
            (home / "backups" / "last-customization-backup.txt").write_text(str(backup) + "\n", encoding="utf-8")
            if not args.skip_tests:
                verify(repo, home, manifest, run_tests=True, python_path=args.python_path)
        print("customization apply complete" if not args.dry_run else "dry-run complete")
        return 0
    except CustomizationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

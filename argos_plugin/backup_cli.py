#!/usr/bin/env python
"""CLI for Argos memory backup and restore.

Backup  — connects to the running memory service (which owns the DB) and
          triggers a service-coordinated EXPORT.  No downtime, cross-platform.

Restore — standalone.  The service MUST be stopped first.  Refuses to run
          if the live DB is locked.  Imports the snapshot into a temp DB,
          verifies row counts, then atomically swaps it into place.

Usage:
    # Backup (service must be running):
    python backup_cli.py backup [--home PATH] [--dst-root PATH] [--retention N]

    # List snapshots:
    python backup_cli.py list [--home PATH] [--dst-root PATH]

    # Verify a snapshot without restoring:
    python backup_cli.py verify --snapshot-dir PATH

    # Restore (service must be STOPPED):
    python backup_cli.py restore --snapshot-dir PATH [--home PATH] [--force]

    # Restore the latest snapshot:
    python backup_cli.py restore --latest [--home PATH]

Exit codes:
    0 — success
    1 — error (missing args, service running on restore, verify failure, etc.)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Ensure the plugin package is importable when run as a standalone script.
_plugin_dir = Path(__file__).resolve().parent
if str(_plugin_dir) not in sys.path:
    sys.path.insert(0, str(_plugin_dir))


def _get_hermes_home(arg_home: str | None) -> Path:
    """Resolve HERMES_HOME the same way the plugin does."""
    if arg_home:
        return Path(arg_home)
    env = os.environ.get("HERMES_HOME")
    if env:
        return Path(env)
    if sys.platform == "win32":
        local = Path(os.path.expandvars(r"%LOCALAPPDATA%\hermes"))
        if local.exists():
            return local
    home = Path.home() / ".hermes"
    if home.exists():
        return home
    return Path(os.path.expandvars(r"%LOCALAPPDATA%\hermes"))


def _resolve_db_path(home: Path) -> Path:
    """Resolve the DuckDB path from hybrid_memory.json or default."""
    cfg_path = home / "hybrid_memory.json"
    db_name = "hybrid_memory.duckdb"
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            db_name = str(cfg.get("database_filename", db_name))
        except Exception:
            pass
    return home / db_name


def _resolve_dst_root(home: Path, arg_dst: str | None) -> Path:
    if arg_dst:
        return Path(arg_dst)
    cfg_path = home / "hybrid_memory.json"
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            backup_cfg = cfg.get("backup", {})
            if isinstance(backup_cfg, dict) and backup_cfg.get("dst_root"):
                return Path(backup_cfg["dst_root"])
        except Exception:
            pass
    return home / "backups" / "memory"


def cmd_backup(args: argparse.Namespace) -> int:
    from service_client import SharedMemoryStore, SharedMemoryServiceError

    home = _get_hermes_home(args.home)
    dst_root = _resolve_dst_root(home, args.dst_root)

    print(f"HERMES_HOME: {home}")
    print(f"Backup destination: {dst_root}")
    print()

    # Connect to the running service (auto-starts if not running).
    try:
        store = SharedMemoryStore(home, user_id="default_user", embedder=None)
    except SharedMemoryServiceError as e:
        print(f"ERROR: cannot connect to memory service: {e}", file=sys.stderr)
        return 1

    try:
        print("Triggering service-coordinated backup...")
        manifest = store._rpc.backup(
            dst_root=str(dst_root),
            retention=args.retention,
        )
        print()
        print("=== BACKUP COMPLETE ===")
        print(f"  Snapshot:    {manifest.get('snapshot_dir', 'N/A')}")
        print(f"  Timestamp:   {manifest.get('timestamp', 'N/A')}")
        print(f"  DuckDB:      {manifest.get('duckdb_version', 'N/A')}")
        print(f"  Tables:      {len(manifest.get('tables', []))}")
        print(f"  Total rows:  {sum(manifest.get('row_counts', {}).values())}")
        print(f"  Files:       {len(manifest.get('files', {}))}")
        print()
        print("Graph note: the Kuzu graph is derived data. After a restore,")
        print("run backfill_graph.py to rebuild it from the restored memories.")
        return 0
    except SharedMemoryServiceError as e:
        print(f"ERROR: backup failed: {e}", file=sys.stderr)
        return 1
    finally:
        try:
            store._rpc.stop_service()
        except Exception:
            pass


def cmd_list(args: argparse.Namespace) -> int:
    from backup import list_snapshots

    home = _get_hermes_home(args.home)
    dst_root = _resolve_dst_root(home, args.dst_root)

    snapshots = list_snapshots(dst_root)
    if not snapshots:
        print(f"No snapshots found in {dst_root}")
        return 0

    print(f"Snapshots in {dst_root} (newest first):")
    print()
    for i, s in enumerate(snapshots, 1):
        print(f"  {i}. {s.get('snapshot_dir', '?')}")
        print(f"     Timestamp: {s.get('timestamp', 'N/A')}")
        print(f"     Tables:    {len(s.get('tables', []))}")
        print(f"     Rows:      {sum(s.get('row_counts', {}).values())}")
        print()
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    from backup import verify_snapshot

    snap = Path(args.snapshot_dir)
    if not snap.exists():
        print(f"ERROR: snapshot directory not found: {snap}", file=sys.stderr)
        return 1

    print(f"Verifying snapshot: {snap}")
    try:
        report = verify_snapshot(snap)
        print()
        print("=== VERIFY PASSED ===")
        print(f"  Status:     {report['status']}")
        print(f"  Timestamp:  {report.get('timestamp', 'N/A')}")
        print(f"  Tables:     {len(report.get('tables', []))}")
        print(f"  Row counts: {report.get('row_counts', {})}")
        return 0
    except Exception as e:
        print(f"VERIFY FAILED: {e}", file=sys.stderr)
        return 1


def cmd_restore(args: argparse.Namespace) -> int:
    from backup import restore_store, list_snapshots

    home = _get_hermes_home(args.home)
    db_path = _resolve_db_path(home)

    if args.latest:
        dst_root = _resolve_dst_root(home, args.dst_root)
        snapshots = list_snapshots(dst_root)
        if not snapshots:
            print(f"ERROR: no snapshots found in {dst_root}", file=sys.stderr)
            return 1
        snap_dir = Path(snapshots[0]["snapshot_dir"])
        print(f"Latest snapshot: {snap_dir}")
    else:
        if not args.snapshot_dir:
            print("ERROR: --snapshot-dir or --latest is required", file=sys.stderr)
            return 1
        snap_dir = Path(args.snapshot_dir)

    if not snap_dir.exists():
        print(f"ERROR: snapshot directory not found: {snap_dir}", file=sys.stderr)
        return 1

    print(f"HERMES_HOME: {home}")
    print(f"Live DB:     {db_path}")
    print(f"Snapshot:    {snap_dir}")
    print()

    if not args.force:
        # Check that the service is not running.
        try:
            from service_client import _read_endpoint
            endpoint = _read_endpoint(home)
            if endpoint:
                print("WARNING: memory service endpoint file exists. "
                      "If the service is running, the restore will fail.", file=sys.stderr)
                print("         Stop the service first, or use --force to skip this check.",
                      file=sys.stderr)
                print()
        except Exception:
            pass

    try:
        print("Restoring...")
        report = restore_store(snap_dir, db_path, force=args.force)
        print()
        print("=== RESTORE COMPLETE ===")
        print(f"  Status:         {report['status']}")
        print(f"  Tables restored: {report['tables_restored']}")
        print(f"  Rows restored:   {report['rows_restored']}")
        print()
        print("NEXT: rebuild the Kuzu graph from the restored memories:")
        print(f"  python backfill_graph.py --home {home}")
        print()
        print("Then restart Hermes. The memory service will auto-respawn.")
        return 0
    except Exception as e:
        print(f"ERROR: restore failed: {e}", file=sys.stderr)
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Argos memory backup and restore (cross-platform, no downtime for backup)."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_backup = sub.add_parser("backup", help="Trigger a service-coordinated backup.")
    p_backup.add_argument("--home", default=None, help="Override HERMES_HOME.")
    p_backup.add_argument("--dst-root", default=None, help="Backup destination root.")
    p_backup.add_argument("--retention", type=int, default=6, help="Snapshots to keep (default 6).")
    p_backup.set_defaults(func=cmd_backup)

    p_list = sub.add_parser("list", help="List available snapshots.")
    p_list.add_argument("--home", default=None, help="Override HERMES_HOME.")
    p_list.add_argument("--dst-root", default=None, help="Backup destination root.")
    p_list.set_defaults(func=cmd_list)

    p_verify = sub.add_parser("verify", help="Verify a snapshot without restoring.")
    p_verify.add_argument("--snapshot-dir", required=True, help="Path to the snapshot directory.")
    p_verify.set_defaults(func=cmd_verify)

    p_restore = sub.add_parser("restore", help="Restore a snapshot (service must be stopped).")
    p_restore.add_argument("--snapshot-dir", default=None, help="Path to the snapshot directory.")
    p_restore.add_argument("--latest", action="store_true", help="Restore the newest snapshot.")
    p_restore.add_argument("--home", default=None, help="Override HERMES_HOME.")
    p_restore.add_argument("--dst-root", default=None, help="Backup destination root (for --latest).")
    p_restore.add_argument("--force", action="store_true", help="Skip the service-running check.")
    p_restore.set_defaults(func=cmd_restore)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

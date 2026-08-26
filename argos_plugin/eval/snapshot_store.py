#!/usr/bin/env python3
"""snapshot_store.py — versioned, snapshot-pinned copies of the live memory DB.

The self-corpus regression gate runs against a frozen copy of the user's
own store so live retrieval paths are never touched and the data cannot
change under the measurement.  The live DuckDB is held by the shared
memory service with an EXCLUSIVE lock — a live copy fails, so snapshots
must be taken with Hermes/the memory service stopped.  This module
refuses to snapshot while the service is running.

Snapshots live in ``eval/snapshots/<date>_<sha8>/`` with a sha256
manifest; the last 5 are kept.  Snapshots contain personal memory
content — never commit the snapshots directory.

Usage:
    python eval/snapshot_store.py take --db <live duckdb> [--endpoint <service json>]
    python eval/snapshot_store.py list [--snapshots-dir <dir>]
    python eval/snapshot_store.py prune [--keep 5] [--snapshots-dir <dir>]
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import shutil
import socket
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_PLUGIN_ROOT = Path(__file__).resolve().parents[1]
if str(_PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_ROOT))

DEFAULT_SNAPSHOTS_DIR = Path(__file__).resolve().parent / "snapshots"
KEEP_SNAPSHOTS = 5
ENDPOINT_NAME = "hybrid_memory_service.json"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def service_running(endpoint: Path) -> bool:
    """True if the shared memory service answers on the recorded endpoint."""
    if not Path(endpoint).exists():
        return False
    try:
        info = json.loads(Path(endpoint).read_text(encoding="utf-8"))
        host = str(info.get("host", "127.0.0.1"))
        port = int(info.get("port", 0))
        token = str(info.get("token", ""))
    except Exception:
        return False
    if port <= 0:
        return False
    try:
        sock = socket.create_connection((host, port), timeout=2.0)
        try:
            payload = (json.dumps({"token": token, "method": "health"}) + "\n").encode("utf-8")
            sock.sendall(payload)
            resp = b""
            while b"\n" not in resp:
                chunk = sock.recv(65536)
                if not chunk:
                    break
                resp += chunk
            if not resp:
                return False
            return json.loads(resp.splitlines()[0]).get("ok") is True
        finally:
            sock.close()
    except Exception:
        return False


def _record_count(db_path: Path) -> Optional[int]:
    try:
        import duckdb
        conn = duckdb.connect(str(db_path), read_only=True)
        try:
            row = conn.execute("SELECT COUNT(*) FROM memory_records").fetchone()
            return int(row[0]) if row else 0
        finally:
            conn.close()
    except Exception:
        return None


def take_snapshot(
    live_db: Path,
    snapshots_dir: Path = DEFAULT_SNAPSHOTS_DIR,
    endpoint: Optional[Path] = None,
    user_id: str = "default_user",
) -> Dict[str, Any]:
    """Copy the live DB into a versioned snapshot dir; return the manifest.

    Refuses while the shared memory service is running (exclusive DuckDB
    lock) or when the DB file itself is locked.  Prunes to the last
    ``KEEP_SNAPSHOTS`` snapshots.
    """
    live_db = Path(live_db)
    if not live_db.exists():
        raise FileNotFoundError(f"DB not found: {live_db}")
    if endpoint is not None and service_running(Path(endpoint)):
        raise RuntimeError(
            "The shared memory service is running and holds an exclusive "
            "lock on the live DB. Stop Hermes/the memory service before "
            "taking a snapshot."
        )
    # Belt & braces: a read-only connect catches a lock the endpoint check
    # missed (e.g. a direct-mode writer holding the file).
    try:
        import duckdb
        probe = duckdb.connect(str(live_db), read_only=True)
        probe.close()
    except Exception as exc:
        raise RuntimeError(
            f"Live DB is locked or unreadable ({exc}); stop the memory "
            "service before snapshotting."
        ) from exc

    snapshots_dir = Path(snapshots_dir)
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    digest = _sha256_file(live_db)[:8]
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    snapshot_id = f"{stamp}_{digest}"
    target = snapshots_dir / snapshot_id
    target.mkdir(parents=True, exist_ok=True)

    db_copy = target / live_db.name
    shutil.copy2(live_db, db_copy)
    wal = Path(str(live_db) + ".wal")
    if wal.exists():
        shutil.copy2(wal, target / (live_db.name + ".wal"))
    # Pin the retrieval config alongside the snapshot so the gate's
    # config_hash is stable across reruns.
    cfg = live_db.parent / "hybrid_memory.json"
    if cfg.exists():
        shutil.copy2(cfg, target / "hybrid_memory.json")

    manifest = {
        "snapshot_id": snapshot_id,
        "created_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "source": str(live_db),
        "db_filename": live_db.name,
        "db_sha256": _sha256_file(db_copy),
        "db_bytes": db_copy.stat().st_size,
        "record_count": _record_count(db_copy),
        "user_id": user_id,
    }
    (target / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    prune(snapshots_dir, keep=KEEP_SNAPSHOTS)
    return manifest


def list_snapshots(snapshots_dir: Path = DEFAULT_SNAPSHOTS_DIR) -> List[Dict[str, Any]]:
    """Return manifests of existing snapshots, newest first."""
    snapshots_dir = Path(snapshots_dir)
    if not snapshots_dir.exists():
        return []
    out: List[Dict[str, Any]] = []
    for d in snapshots_dir.iterdir():
        mpath = d / "manifest.json"
        if not d.is_dir() or not mpath.exists():
            continue
        try:
            out.append(json.loads(mpath.read_text(encoding="utf-8")))
        except Exception:
            continue
    out.sort(key=lambda m: m.get("snapshot_id", ""), reverse=True)
    return out


def prune(snapshots_dir: Path = DEFAULT_SNAPSHOTS_DIR, keep: int = KEEP_SNAPSHOTS) -> List[Path]:
    """Delete snapshots beyond the newest *keep*; return removed paths."""
    snapshots_dir = Path(snapshots_dir)
    if not snapshots_dir.exists():
        return []
    dirs = sorted(
        (
            d for d in snapshots_dir.iterdir()
            if d.is_dir() and (d / "manifest.json").exists()
        ),
        key=lambda d: d.name,
    )
    removed: List[Path] = []
    for old in dirs[:-keep] if keep > 0 else dirs:
        shutil.rmtree(old, ignore_errors=True)
        removed.append(old)
    return removed


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Snapshot manager for the self-corpus regression gate.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_take = sub.add_parser("take", help="Copy the live DB into a versioned snapshot.")
    p_take.add_argument("--db", required=True, help="Path to the live hybrid_memory.duckdb.")
    p_take.add_argument("--endpoint", default="", help="Path to hybrid_memory_service.json (refuses if the service answers).")
    p_take.add_argument("--snapshots-dir", default=str(DEFAULT_SNAPSHOTS_DIR))
    p_take.add_argument("--user-id", default="default_user")

    p_list = sub.add_parser("list", help="List existing snapshots.")
    p_list.add_argument("--snapshots-dir", default=str(DEFAULT_SNAPSHOTS_DIR))

    p_prune = sub.add_parser("prune", help="Delete snapshots beyond the newest N.")
    p_prune.add_argument("--keep", type=int, default=KEEP_SNAPSHOTS)
    p_prune.add_argument("--snapshots-dir", default=str(DEFAULT_SNAPSHOTS_DIR))

    args = parser.parse_args(argv)
    if args.command == "take":
        endpoint = Path(args.endpoint) if args.endpoint else None
        manifest = take_snapshot(
            Path(args.db), Path(args.snapshots_dir), endpoint=endpoint,
            user_id=args.user_id,
        )
        print(f"Snapshot taken: {manifest['snapshot_id']}")
        print(f"  records: {manifest['record_count']}  sha256: {manifest['db_sha256'][:16]}…")
        return 0
    if args.command == "list":
        for m in list_snapshots(Path(args.snapshots_dir)):
            print(f"{m['snapshot_id']}  records={m.get('record_count')}  sha={m['db_sha256'][:12]}…")
        return 0
    if args.command == "prune":
        removed = prune(Path(args.snapshots_dir), keep=args.keep)
        for r in removed:
            print(f"removed {r.name}")
        print(f"kept {args.keep}; removed {len(removed)}")
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())

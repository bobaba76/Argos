#!/usr/bin/env python3
"""Argos memory store backup - stop service, copy, verify, manifest.

Stops the shared memory service gracefully (RPC shutdown), copies the
DuckDB main + WAL and Kuzu graph + WAL to a timestamped snapshot dir,
verifies by replaying the WAL (open read-write with DuckDB), writes a
SHA256 manifest, and prunes old snapshots. The service auto-restarts
on next client use.

This replaces the VSS-based approach (vss_backup_memory.ps1) which was
blocked by antivirus on this machine. The tradeoff: ~5s of service
downtime during the copy instead of zero-downtime VSS. The copy is
still crash-consistent because the service checkpoints the WAL on
clean shutdown before we copy.

Usage:
    python scripts/backup_memory.py [--hermes-home PATH] [--dry-run] [--keep N]

Zero network calls. Zero LLM calls. No elevation required.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Files to copy (main + wal pairs, never split)
DB_NAME_DEFAULT = "hybrid_memory.duckdb"
KUZU_NAME_DEFAULT = "hybrid_memory_kuzu"

STOP_VERIFY_TRIES = 30  # 0.5s apart = up to 15s
COPY_RETRIES = 4
COPY_RETRY_DELAY = 0.5
MIN_RATIO = 0.95


def default_home() -> Path:
    configured = os.environ.get("HERMES_HOME", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    if os.name == "nt":
        return Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "hermes"
    return Path.home() / ".hermes"


def load_config(home: Path) -> dict:
    config_path = home / "hybrid_memory.json"
    if not config_path.exists():
        print(f"ERROR: hybrid_memory.json not found at {config_path}")
        sys.exit(1)
    return json.loads(config_path.read_text(encoding="utf-8-sig"))


def _service_healthy(endpoint: Path) -> bool:
    try:
        details = json.loads(endpoint.read_text(encoding="utf-8"))
        with socket.create_connection((details["host"], int(details["port"])), timeout=0.5) as conn:
            request = json.dumps({"method": "health", "token": details["token"]}) + "\n"
            conn.sendall(request.encode("utf-8"))
            response = json.loads(conn.recv(4096).decode("utf-8"))
            return bool(response.get("ok"))
    except Exception:
        return False


def stop_memory_service(home: Path, plugin_dir: Path) -> bool:
    """Graceful RPC shutdown. True only when the service is verifiably down."""
    endpoint = home / "hybrid_memory_service.json"
    if not endpoint.exists():
        return True  # nothing running

    code = (
        "import sys; sys.path.insert(0, r'%s'); "
        "from service_client import SharedMemoryStore; "
        "s = SharedMemoryStore(r'%s'); s._rpc.stop_service(); print('stopped')"
    ) % (plugin_dir, home)
    try:
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=30
        )
        if result.returncode != 0:
            print(f"  stop RPC error: {result.stderr.strip()[:400]}")
    except Exception as exc:
        print(f"  stop RPC exception: {exc}")

    for _ in range(STOP_VERIFY_TRIES):
        if not endpoint.exists():
            return True
        time.sleep(0.5)
    if _service_healthy(endpoint):
        print("ERROR: memory service still healthy after stop (may have auto-restarted).")
        return False
    print("  note: endpoint file stale but service is down")
    return True


def get_source_count(home: Path) -> int:
    """Get memory_records count via service RPC before stopping."""
    endpoint = home / "hybrid_memory_service.json"
    if not endpoint.exists():
        return -1
    try:
        details = json.loads(endpoint.read_text(encoding="utf-8"))
        with socket.create_connection((details["host"], int(details["port"])), timeout=2.0) as conn:
            request = json.dumps({"method": "count", "token": details["token"], "user_id": "default_user"}) + "\n"
            conn.sendall(request.encode("utf-8"))
            response = json.loads(conn.recv(4096).decode("utf-8"))
            if response.get("ok"):
                return int(response["result"])
    except Exception:
        pass
    return -1


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def safe_copy(src: Path, dst: Path) -> bool:
    src_size = src.stat().st_size if src.exists() else 0
    for attempt in range(COPY_RETRIES):
        try:
            shutil.copy2(src, dst)
        except Exception as exc:
            if attempt == COPY_RETRIES - 1:
                print(f"  FAIL {src.name}: {exc}")
                return False
            time.sleep(COPY_RETRY_DELAY)
            continue
        dst_size = dst.stat().st_size
        if dst_size >= max(1, src_size * MIN_RATIO):
            print(f"  OK   {src.name}: {src_size:,} -> {dst_size:,} bytes")
            return True
        print(f"  WARN {src.name}: short copy {dst_size:,}/{src_size:,}; retrying")
        time.sleep(COPY_RETRY_DELAY)
    print(f"  FAIL {src.name}: could not produce a full-size copy")
    return False


def verify_snapshot(db_path: Path, expected_count: int) -> dict:
    """Open the copy read-write to replay WAL, count rows."""
    import tempfile
    scratch = Path(tempfile.mkdtemp(prefix="argos_verify_"))
    try:
        db_path = Path(db_path)
        main_dst = scratch / db_path.name
        wal_src = Path(str(db_path) + ".wal")
        wal_dst = Path(str(main_dst) + ".wal")
        shutil.copy2(str(db_path), str(main_dst))
        if wal_src.exists():
            shutil.copy2(str(wal_src), str(wal_dst))
        import duckdb
        con = duckdb.connect(str(main_dst))  # read-write -> replays WAL
        row = con.execute("SELECT count(*) FROM memory_records WHERE valid_to IS NULL").fetchone()
        total = con.execute("SELECT count(*) FROM memory_records").fetchone()
        con.close()
        result = {"ok": True, "active_count": row[0], "total_count": total[0]}
        if expected_count >= 0 and row[0] != expected_count:
            result["ok"] = False
            result["error"] = f"count mismatch: copy={row[0]} source={expected_count}"
        return result
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hermes-home", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep", type=int, default=0, help="snapshots to retain (default: from config or 3)")
    args = parser.parse_args()

    home = (args.hermes_home or default_home()).expanduser().resolve()
    config = load_config(home)

    db_name = config.get("database_filename", DB_NAME_DEFAULT)
    kuzu_name = config.get("graph_dirname", KUZU_NAME_DEFAULT)

    db_path = home / db_name
    wal_path = Path(str(db_path) + ".wal")
    kuzu_path = home / kuzu_name
    kuzu_wal_path = Path(str(kuzu_path) + ".wal")

    # Resolve backup root
    backup_root = config.get("backup_root", "")
    if backup_root:
        backup_root = os.path.expandvars(backup_root)
    else:
        backup_root = str(home / "backups" / "memory")
    backup_root = Path(backup_root)

    # Retention
    retention = args.keep
    if retention == 0:
        retention = int(config.get("backup_retention_snapshots", "3"))

    # Files to copy. WAL files are not critical — a clean shutdown checkpoints
    # the WAL into main, so the .wal may not exist after stop. That's fine;
    # the main DB has all the data.
    store_files = [
        (db_name, db_path, True),
        (f"{db_name}.wal", wal_path, False),
    ]
    if kuzu_path.exists():
        store_files.append((kuzu_name, kuzu_path, True))
    if kuzu_wal_path.exists():
        store_files.append((f"{kuzu_name}.wal", kuzu_wal_path, False))

    # Same-volume warning
    try:
        store_vol = os.path.splitdrive(str(db_path))[0]
        backup_vol = os.path.splitdrive(str(backup_root))[0]
        if store_vol and backup_vol and store_vol.lower() == backup_vol.lower():
            print(f"WARNING: backup root is on the SAME volume ({store_vol}) as the store.")
            print("  This protects against corruption, NOT disk failure.")
            print("  Point it at D: / NAS / removable for full protection.")
    except Exception:
        pass

    print(f"Home:        {home}")
    print(f"BackupRoot:  {backup_root}")
    print(f"Retention:   {retention} snapshots")

    # Verify critical files exist
    for name, path, critical in store_files:
        if critical and not path.exists():
            print(f"ERROR: Critical store file missing: {path}")
            return 1

    if args.dry_run:
        print("\nDRY RUN - no stop, no copy.")
        for name, path, critical in store_files:
            exists = path.exists()
            size = path.stat().st_size if exists else 0
            print(f"  {name}: {size:,} bytes ({'present' if exists else 'absent'})")
        return 0

    # Get source count before stopping
    source_count = get_source_count(home)
    if source_count >= 0:
        print(f"Source count (via RPC): {source_count}")
    else:
        print("WARNING: Could not get source count via RPC - will verify opens + count only.")

    # Stop the service (clean shutdown checkpoints the WAL)
    plugin_dir = Path(__file__).resolve().parents[1] / "argos_plugin"
    if not plugin_dir.exists():
        plugin_dir = Path(__file__).resolve().parents[1] / "hybrid_memory_plugin"

    print("\nStopping memory service...")
    if not stop_memory_service(home, plugin_dir):
        print("BACKUP ABORTED: service could not be stopped cleanly.")
        return 1
    print("Service stopped.")

    # Create snapshot dir
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    snapshot_dir = backup_root / f"memory-{stamp}"
    backup_root.mkdir(parents=True, exist_ok=True)
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    # Copy files
    print(f"\nCopying to {snapshot_dir}...")
    copy_start = time.time()
    failures = []
    for name, src_path, critical in store_files:
        if not src_path.exists():
            if name.endswith(".wal"):
                # WAL is absent after a clean checkpoint — data is in main. This is fine.
                print(f"  SKIP {name} (not present — WAL was checkpointed on clean shutdown)")
                continue
            if critical:
                print(f"  FAIL {name}: file disappeared")
                failures.append(name)
            else:
                print(f"  SKIP {name} (not present)")
            continue
        # Retry on PermissionError — the service may still be releasing the lock
        copied = False
        for retry in range(COPY_RETRIES):
            try:
                if safe_copy(src_path, snapshot_dir / name):
                    copied = True
                    break
            except PermissionError:
                if retry < COPY_RETRIES - 1:
                    print(f"  WARN {name}: locked, retrying in {COPY_RETRY_DELAY}s...")
                    time.sleep(COPY_RETRY_DELAY)
                    continue
                raise
        if not copied:
            failures.append(name)

    if failures:
        print(f"\nBACKUP FAILED: {len(failures)} file(s) not backed up: {', '.join(failures)}")
        # Mark incomplete
        incomplete = snapshot_dir.parent / f"memory-{stamp}.INCOMPLETE"
        try:
            snapshot_dir.rename(incomplete)
        except Exception:
            pass
        return 1

    copy_elapsed = time.time() - copy_start
    print(f"Copy completed in {copy_elapsed:.1f}s")

    # Verify - replay WAL, count rows
    print("\nVerifying snapshot (WAL replay + row count)...")
    copied_db = snapshot_dir / db_name
    try:
        verify_result = verify_snapshot(copied_db, source_count)
    except Exception as exc:
        print(f"VERIFY ERROR: {exc}")
        return 1

    if not verify_result["ok"]:
        print(f"Verification FAILED: {verify_result.get('error', 'unknown')}")
        incomplete = snapshot_dir.parent / f"memory-{stamp}.INCOMPLETE"
        try:
            snapshot_dir.rename(incomplete)
        except Exception:
            pass
        return 1

    copy_count = verify_result["active_count"]
    total_count = verify_result["total_count"]
    print(f"Verification OK: active={copy_count} total={total_count} (source={source_count})")

    # Write manifest
    manifest = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "snapshot_dir": str(snapshot_dir),
        "source_count": source_count,
        "copy_count": copy_count,
        "total_count": total_count,
        "files": [],
    }
    for name, src_path, critical in store_files:
        dst_path = snapshot_dir / name
        if dst_path.exists():
            manifest["files"].append({
                "name": name,
                "sha256": sha256_file(dst_path),
                "size": dst_path.stat().st_size,
                "source_size": src_path.stat().st_size if src_path.exists() else 0,
            })

    manifest_path = snapshot_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Manifest written: {manifest_path}")

    # Retention - prune old snapshots (only after new one verifies)
    all_snapshots = sorted(
        [p for p in backup_root.iterdir() if p.is_dir() and p.name.startswith("memory-")
         and len(p.name) == 22],  # memory-YYYYMMDD-HHMMSS = 22 chars
        reverse=True,
    )
    if len(all_snapshots) > retention:
        for old in all_snapshots[retention:]:
            shutil.rmtree(old, ignore_errors=True)
            print(f"Pruned old snapshot: {old.name}")

    # Summary
    total_size = sum(
        (snapshot_dir / name).stat().st_size
        for name, _, _ in store_files
        if (snapshot_dir / name).exists()
    )
    total_mb = total_size / (1024 * 1024)
    print(f"\nBACKUP SUCCESS: {snapshot_dir} ({total_mb:.1f} MB, {copy_count} active memories, {copy_elapsed:.1f}s copy)")
    print("note: the memory service restarts automatically on next client use.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

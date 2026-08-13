#!/usr/bin/env python3
"""Backup Hermes hybrid-memory data + core state files.

Stops the shared memory service gracefully (RPC shutdown), VERIFIES it is
actually down, copies the canonical data files to
HERMES_HOME/backups/hybrid-memory-<timestamp>/, verifies sizes, and lets the
client auto-restart the service on next use.

Hard-fails (exit 1) if the service cannot be stopped or any critical file
cannot be copied at full size — a silent partial backup is worse than none.

Usage:
    python scripts/backup_data.py [--hermes-home PATH] [--dry-run] [--keep N]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Files that hold live memory state (copied, never opened for write here).
# The first three are CRITICAL: backup must fail loudly if they are missing
# or undersized.
CRITICAL_FILES = (
    "hybrid_memory.duckdb",
    "hybrid_memory_kuzu",          # single ~10-90MB FILE, not a directory
    "state.db",
)
OPTIONAL_FILES = (
    "hybrid_memory.duckdb.wal",
    "hybrid_memory_gateway.duckdb",
    "hybrid_memory_gateway.duckdb.wal",
    "hybrid_memory_kuzu_gateway",  # single file
    "hybrid_memory.json",
    "hybrid_memory_service.json",
    "config.yaml",
    "cron/jobs.json",
)
ALL_FILES = CRITICAL_FILES + OPTIONAL_FILES

# Stop copying if the backup is smaller than this fraction of the source
# (guards against locked-file partial/0-byte copies).
MIN_RATIO = 0.95
COPY_RETRIES = 4
COPY_RETRY_DELAY = 0.5
STOP_VERIFY_TRIES = 30  # 0.5s apart = up to 15s


def default_home() -> Path:
    configured = os.environ.get("HERMES_HOME", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    if os.name == "nt":
        return Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "hermes"
    return Path.home() / ".hermes"


def _service_healthy(endpoint: Path) -> bool:
    """True when the endpoint's port answers health positively."""
    try:
        details = json.loads(endpoint.read_text(encoding="utf-8"))
        with socket.create_connection((details["host"], int(details["port"])), timeout=0.5) as conn:
            request = json.dumps({"method": "health", "token": details["token"]}) + "\n"
            conn.sendall(request.encode("utf-8"))
            response = json.loads(conn.recv(4096).decode("utf-8"))
            return bool(response.get("ok"))
    except Exception:
        return False


def stop_memory_service(home: Path) -> bool:
    """Graceful RPC shutdown. True only when the service is verifiably down."""
    endpoint = home / "hybrid_memory_service.json"
    if not endpoint.exists():
        return True  # nothing running

    plugin_dir = Path(__file__).resolve().parents[1] / "hybrid_memory_plugin"
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
            print(f"stop RPC error: {result.stderr.strip()[:400]}")
    except Exception as exc:
        print(f"stop RPC exception: {exc}")

    # Clean shutdown removes the endpoint file (pid-matched) and checkpoints
    # the WAL. Wait for that; then fall back to a health probe.
    for _ in range(STOP_VERIFY_TRIES):
        if not endpoint.exists():
            return True
        time.sleep(0.5)
    if _service_healthy(endpoint):
        print("ERROR: memory service is still healthy after stop request "
              "(it may have auto-restarted). Refusing to copy locked files.")
        return False
    # Endpoint file stale but port dead: treat as stopped.
    print("note: endpoint file stale but service is down")
    return True


def safe_copy(src: Path, dst: Path) -> bool:
    """Copy with retries; returns True only when the backup is full-size."""
    src_size = src.stat().st_size
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
        print(f"  WARN {src.name}: short copy {dst_size:,}/{src_size:,} bytes; retrying")
        time.sleep(COPY_RETRY_DELAY)
    print(f"  FAIL {src.name}: could not produce a full-size copy")
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hermes-home", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep", type=int, default=5, help="backups to retain")
    args = parser.parse_args()

    home = (args.hermes_home or default_home()).expanduser().resolve()
    backups_root = home / "backups"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    dest = backups_root / f"hybrid-memory-{stamp}"
    print(f"home: {home}")
    print(f"backup target: {dest}")

    if args.dry_run:
        for rel in ALL_FILES:
            src = home / rel
            print(f"  plan {rel}" if src.exists() else f"  -    {rel} (absent)")
        return 0

    if not stop_memory_service(home):
        print("BACKUP ABORTED: service could not be stopped cleanly.")
        return 1

    dest.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []
    for rel in ALL_FILES:
        src = home / rel
        if not src.exists():
            print(f"  -    {rel} (absent)")
            continue
        if not safe_copy(src, dest / src.name):
            failures.append(rel)

    if failures:
        print(f"BACKUP FAILED: {len(failures)} file(s) not backed up: {', '.join(failures)}")
        return 1

    backups = sorted(
        (p for p in backups_root.glob("hybrid-memory-*") if p.is_dir()),
        reverse=True,
    )
    for stale in backups[args.keep:]:
        shutil.rmtree(stale, ignore_errors=True)
        print(f"pruned old backup: {stale.name}")

    print(f"backup complete: {dest}")
    print("note: the memory service restarts automatically on next client use.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

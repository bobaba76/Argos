"""Cross-platform backup and restore for the Argos memory store.

Service-coordinated logical backup via DuckDB's EXPORT/IMPORT DATABASE
(FORMAT PARQUET).  No filesystem snapshots, no elevation, no platform-specific
tricks — works identically on Windows, macOS, and Linux.

Design:
  Backup  — runs inside the memory service (the sole DB writer).  CHECKPOINT
            flushes the WAL into main, then EXPORT writes schema.sql + one
            .parquet per table.  Parquet round-trips DOUBLE[] (the embedding
            column) and JSON natively.  A manifest.json records per-table row
            counts, schema, source size, timestamp, and DuckDB version.  The
            manifest is verified by reopening each parquet in a fresh
            connection and re-counting.  Old snapshots are pruned *after* the
            new one verifies.

  Restore — standalone (the service MUST be stopped).  IMPORT DATABASE into
            a temp DuckDB, verify row counts against the manifest, then
            atomically rename over the live DB path.  The Kùzu graph is
            derived data (rebuilt via backfill_graph.py on next start); v1
            backs up only the DuckDB, which includes the entity_aliases table
            (the important manual graph data).

Safety invariants:
  - Backup never stops the service; restore explicitly requires it stopped.
  - Crash-consistent: CHECKPOINT before EXPORT gives a transactional view.
  - Verify before report: a partial/never-verified snapshot never passes.
  - Fail loudly (non-zero) on any mismatch.
  - Zero LLM calls, zero network calls.
  - Retention deletes only after the new snapshot verifies.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("argos.backup")

_SCHEMA_VERSION = 1
_MANIFEST_NAME = "manifest.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _list_tables(conn) -> List[str]:
    rows = conn.execute(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema = 'main' ORDER BY table_name"
    ).fetchall()
    return [r[0] for r in rows]


def _table_row_count(conn, table: str) -> int:
    # Table names come from information_schema — safe to interpolate.
    return int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])


def _duckdb_version(conn) -> str:
    try:
        return str(conn.execute("SELECT duckdb_version()").fetchone()[0])
    except Exception:
        return "unknown"


def _snapshot_dir(dst_root: Path, when: Optional[datetime] = None) -> Path:
    ts = (when or datetime.now(timezone.utc)).strftime("%Y%m%d-%H%M%S")
    return dst_root / f"memory-{ts}"


# ---------------------------------------------------------------------------
# Backup
# ---------------------------------------------------------------------------

def backup_store(
    conn,
    dst_root: str | Path,
    *,
    retention_snapshots: int = 6,
    source_db_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Back up the DuckDB store *conn* into *dst_root*.

    *conn* is an open DuckDB connection (the service's own — the service is
    the sole writer, so a CHECKPOINT here is safe).  The export is a logical
    dump (schema.sql + parquet per table), not a byte copy — it works live,
    cross-platform, with no file-lock fights.

    Returns the manifest dict.  Raises on any failure (verify-reject,
    export error, etc.).
    """
    dst_root = Path(dst_root)
    dst_root.mkdir(parents=True, exist_ok=True)
    snap = _snapshot_dir(dst_root)
    # If the timestamp collides (rapid re-backup), append a counter.
    counter = 0
    while snap.exists():
        counter += 1
        snap = dst_root / f"{snap.name}-{counter}"
    snap.mkdir(parents=True)

    tables: List[str] = []
    counts: Dict[str, int] = {}
    try:
        # 1. CHECKPOINT — flush WAL into main (service is sole writer, safe).
        conn.execute("CHECKPOINT")

        # 2. EXPORT DATABASE (FORMAT PARQUET) — schema.sql + load.sql + .parquet
        conn.execute(f"EXPORT DATABASE '{snap.as_posix()}' (FORMAT PARQUET)")

        # 3. Record per-table row counts from the source connection.
        tables = _list_tables(conn)
        for t in tables:
            counts[t] = _table_row_count(conn, t)

        # 4. Verify: reopen each parquet in a fresh read-only connection and
        #    re-count.  This catches corruption, partial writes, and type
        #    fidelity issues before we declare the snapshot good.
        import duckdb as _ddb
        verify_conn = _ddb.connect(":memory:")
        verify_conn.execute(f"IMPORT DATABASE '{snap.as_posix()}'")
        for t in tables:
            actual = _table_row_count(verify_conn, t)
            if actual != counts[t]:
                raise RuntimeError(
                    f"verify-reject: table '{t}' row count mismatch "
                    f"(source={counts[t]}, snapshot={actual})"
                )
        verify_conn.close()

        # 5. Write manifest.
        files: Dict[str, Dict[str, Any]] = {}
        for f in sorted(snap.iterdir()):
            if f.is_file():
                files[f.name] = {
                    "sha256": _sha256_file(f),
                    "size_bytes": f.stat().st_size,
                }

        manifest: Dict[str, Any] = {
            "schema_version": _SCHEMA_VERSION,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "duckdb_version": _duckdb_version(conn),
            "tables": tables,
            "row_counts": counts,
            "files": files,
            "source_db_path": str(source_db_path) if source_db_path else None,
            "source_db_size_bytes": (
                Path(source_db_path).stat().st_size if source_db_path and Path(source_db_path).exists() else None
            ),
            "format": "parquet",
            "graph_note": "Kuzu graph is derived data; rebuild via backfill_graph.py on restore.",
        }
        manifest_path = snap / _MANIFEST_NAME
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        # 6. Retention: prune oldest snapshots *after* the new one verifies.
        _prune_old_snapshots(dst_root, retention_snapshots)

        logger.info("backup complete: %s (%d tables, %d rows total)",
                     snap.name, len(tables), sum(counts.values()))
        manifest["snapshot_dir"] = str(snap)
        return manifest

    except Exception:
        # Clean up the partial snapshot so a failed backup never looks
        # like a good one to the retention pruner or the restore path.
        try:
            shutil.rmtree(snap, ignore_errors=True)
        except Exception:
            pass
        raise


def _prune_old_snapshots(dst_root: Path, keep: int) -> None:
    """Delete oldest snapshot dirs beyond *keep*, keeping the newest *keep*."""
    if keep <= 0:
        return
    snapshots = sorted(
        [d for d in dst_root.iterdir()
         if d.is_dir() and d.name.startswith("memory-") and (d / _MANIFEST_NAME).exists()],
        key=lambda d: d.name,
        reverse=True,
    )
    for old in snapshots[keep:]:
        try:
            shutil.rmtree(old, ignore_errors=True)
            logger.info("pruned old snapshot: %s", old.name)
        except Exception as exc:
            logger.warning("failed to prune %s: %s", old.name, exc)


def list_snapshots(dst_root: str | Path) -> List[Dict[str, Any]]:
    """List all valid snapshots in *dst_root*, newest first."""
    dst_root = Path(dst_root)
    if not dst_root.exists():
        return []
    out: List[Dict[str, Any]] = []
    for d in sorted(dst_root.iterdir(), key=lambda d: d.name, reverse=True):
        if not (d.is_dir() and d.name.startswith("memory-")):
            continue
        mpath = d / _MANIFEST_NAME
        if not mpath.exists():
            continue
        try:
            m = json.loads(mpath.read_text(encoding="utf-8"))
            m["snapshot_dir"] = str(d)
            out.append(m)
        except Exception:
            continue
    return out


# ---------------------------------------------------------------------------
# Restore
# ---------------------------------------------------------------------------

def _is_db_locked(db_path: Path) -> bool:
    """Return True if *db_path* is opened by another process (exclusive lock)."""
    import duckdb as _ddb
    try:
        test = _ddb.connect(str(db_path), read_only=False)
        test.close()
        return False
    except Exception as exc:
        msg = str(exc).lower()
        # Windows: "being used by another process" (cross-process lock)
        # DuckDB: "cannot access the file" (lock error variant)
        # DuckDB same-process: "different configuration than existing connections"
        #   — this means another connection is already open (service is running
        #   in this process; in a real restore the service is a separate process
        #   and the first two messages are what we'd see).
        return (
            "being used by another process" in msg
            or "cannot access the file" in msg
            or "different configuration" in msg
        )


def restore_store(
    snapshot_dir: str | Path,
    live_db_path: str | Path,
    *,
    force: bool = False,
) -> Dict[str, Any]:
    """Restore a snapshot into *live_db_path*.

    The service MUST be stopped (no live writer).  If the live DB is locked
    by a running service, the restore refuses unless *force* is True (which
    still won't bypass OS file locks — it only skips the pre-check).

    Steps: IMPORT into a temp DB → verify against manifest → atomic rename.

    Returns a summary dict.  Raises on any failure.
    """
    snapshot_dir = Path(snapshot_dir)
    live_db_path = Path(live_db_path)
    manifest_path = snapshot_dir / _MANIFEST_NAME

    if not manifest_path.exists():
        raise FileNotFoundError(f"no manifest in snapshot: {snapshot_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    # 1. Refuse if the live DB is locked (service is running).
    if not force and live_db_path.exists():
        if _is_db_locked(live_db_path):
            raise RuntimeError(
                f"live DB is locked (memory service is running). "
                f"Stop the service before restoring: {live_db_path}"
            )

    import duckdb as _ddb

    # 2. IMPORT into a temp DB (never touch the live path until verified).
    live_db_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_db = live_db_path.with_suffix(".restore-tmp.duckdb")
    # Clean up any stale temp from a previous failed restore.
    for suffix in ("", ".wal"):
        p = Path(str(tmp_db) + suffix) if suffix else tmp_db
        if p.exists():
            try:
                p.unlink()
            except Exception:
                pass

    try:
        conn = _ddb.connect(str(tmp_db))
        conn.execute(f"IMPORT DATABASE '{snapshot_dir.as_posix()}'")

        # 3. Verify row counts against the manifest.
        expected_counts = manifest.get("row_counts", {})
        for table, expected in expected_counts.items():
            actual = _table_row_count(conn, table)
            if actual != expected:
                raise RuntimeError(
                    f"restore verify-reject: table '{table}' "
                    f"expected {expected} rows, got {actual}"
                )
        conn.close()

        # 4. Atomic swap: rename live → .bak, temp → live, then delete .bak.
        backup_of_live = live_db_path.with_suffix(".pre-restore.duckdb")
        if live_db_path.exists():
            # Move the WAL too if it exists.
            for wal_suffix in (".wal",):
                wal = Path(str(live_db_path) + wal_suffix)
                if wal.exists():
                    wal_bak = Path(str(backup_of_live) + wal_suffix)
                    try:
                        wal.rename(wal_bak)
                    except Exception:
                        pass
            live_db_path.rename(backup_of_live)

        tmp_db.rename(live_db_path)

        # Clean up the pre-restore backup (restore is committed).
        for suffix in ("", ".wal"):
            p = Path(str(backup_of_live) + suffix) if suffix else backup_of_live
            if p.exists():
                try:
                    p.unlink()
                except Exception:
                    pass

        logger.info("restore complete: %s -> %s", snapshot_dir.name, live_db_path)
        return {
            "status": "restored",
            "snapshot_dir": str(snapshot_dir),
            "live_db_path": str(live_db_path),
            "tables_restored": len(expected_counts),
            "rows_restored": sum(expected_counts.values()),
            "manifest_timestamp": manifest.get("timestamp"),
        }

    except Exception:
        # Clean up the temp DB on failure.
        try:
            for suffix in ("", ".wal"):
                p = Path(str(tmp_db) + suffix) if suffix else tmp_db
                if p.exists():
                    p.unlink()
        except Exception:
            pass
        raise


def verify_snapshot(snapshot_dir: str | Path) -> Dict[str, Any]:
    """Verify a snapshot's integrity without restoring it.

    Re-imports into an in-memory DB and checks row counts + file hashes
    against the manifest.  Returns a report dict.  Raises on mismatch.
    """
    snapshot_dir = Path(snapshot_dir)
    manifest_path = snapshot_dir / _MANIFEST_NAME
    if not manifest_path.exists():
        raise FileNotFoundError(f"no manifest in snapshot: {snapshot_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    # Check file hashes.
    files = manifest.get("files", {})
    for fname, info in files.items():
        fpath = snapshot_dir / fname
        if not fpath.exists():
            raise FileNotFoundError(f"missing file in snapshot: {fname}")
        actual_hash = _sha256_file(fpath)
        if actual_hash != info.get("sha256"):
            raise RuntimeError(f"hash mismatch for {fname}: manifest={info.get('sha256')}, actual={actual_hash}")

    # Re-import and check row counts.
    import duckdb as _ddb
    conn = _ddb.connect(":memory:")
    try:
        conn.execute(f"IMPORT DATABASE '{snapshot_dir.as_posix()}'")
        expected_counts = manifest.get("row_counts", {})
        for table, expected in expected_counts.items():
            actual = _table_row_count(conn, table)
            if actual != expected:
                raise RuntimeError(
                    f"row count mismatch for table '{table}': "
                    f"manifest={expected}, actual={actual}"
                )
    finally:
        conn.close()

    return {
        "status": "verified",
        "snapshot_dir": str(snapshot_dir),
        "tables": manifest.get("tables", []),
        "row_counts": manifest.get("row_counts", {}),
        "timestamp": manifest.get("timestamp"),
    }

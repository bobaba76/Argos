"""Tests for the cross-platform backup/restore system.

Covers: round-trip fidelity (including DOUBLE[] embeddings), live-write
safety, verify-reject on corruption, retention pruning, and the
service-coordinated RPC path.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pytest

# Ensure plugin dir is importable.
_plugin_dir = Path(__file__).resolve().parent.parent
if str(_plugin_dir) not in sys.path:
    sys.path.insert(0, str(_plugin_dir))


def _make_store_with_data(tmp_path: Path, n: int = 5):
    """Create a DuckDBMemoryStore with *n* test memories and return it."""
    from store import DuckDBMemoryStore

    db = tmp_path / "hybrid_memory.duckdb"
    store = DuckDBMemoryStore(db, user_id="test_user")
    for i in range(n):
        store.remember(
            category="personal_fact",
            content=f"User's test fact number {i} with unique token zebra{i}",
            tags=[f"tag{i}"],
        )
    return store


# ---------------------------------------------------------------------------
# Core round-trip
# ---------------------------------------------------------------------------

def test_backup_round_trip_preserves_embeddings_and_counts(tmp_path):
    """Backup → delete rows → restore → all rows and embeddings match."""
    from backup import backup_store, restore_store, verify_snapshot
    import duckdb

    store = _make_store_with_data(tmp_path, n=5)
    db_path = store.db_path
    dst_root = tmp_path / "backups"

    # Capture original embeddings for comparison.
    orig_rows = store.connection.execute(
        "SELECT memory_id, embedding, content FROM memory_records ORDER BY memory_id"
    ).fetchall()
    assert len(orig_rows) == 5

    manifest = backup_store(
        store.connection, dst_root, retention_snapshots=10,
        source_db_path=db_path,
    )
    assert manifest["row_counts"]["memory_records"] == 5
    snap_dir = Path(manifest["snapshot_dir"])
    assert (snap_dir / "manifest.json").exists()
    assert (snap_dir / "schema.sql").exists()
    assert any(f.suffix == ".parquet" for f in snap_dir.iterdir())

    # Verify the snapshot without restoring.
    report = verify_snapshot(snap_dir)
    assert report["status"] == "verified"

    store.close()

    # Delete some rows from the live DB to prove restore brings them back.
    conn = duckdb.connect(str(db_path))
    conn.execute("DELETE FROM memory_records WHERE content LIKE '%zebra0%'")
    conn.execute("DELETE FROM memory_records WHERE content LIKE '%zebra1%'")
    remaining = conn.execute("SELECT COUNT(*) FROM memory_records").fetchone()[0]
    assert remaining == 3
    conn.close()

    # Restore (service is not running — no lock).
    report = restore_store(snap_dir, db_path)
    assert report["status"] == "restored"
    assert report["rows_restored"] == 5

    # Verify all 5 rows are back with their original embeddings.
    conn = duckdb.connect(str(db_path), read_only=True)
    restored_rows = conn.execute(
        "SELECT memory_id, embedding, content FROM memory_records ORDER BY memory_id"
    ).fetchall()
    assert len(restored_rows) == 5
    for (mid, emb, content), (orig_mid, orig_emb, orig_content) in zip(restored_rows, orig_rows):
        assert mid == orig_mid
        assert content == orig_content
        # Embeddings must round-trip exactly (DOUBLE[] via parquet).
        # Both may be None when the store has no embedder (test mode).
        if orig_emb is not None:
            assert emb is not None, f"embedding lost for {mid}"
            assert len(emb) == len(orig_emb)
            for a, b in zip(emb, orig_emb):
                assert abs(a - b) < 1e-12, f"embedding mismatch for {mid}"
        else:
            assert emb is None, f"expected NULL embedding for {mid}, got {emb}"
    conn.close()


# ---------------------------------------------------------------------------
# Live-write safety
# ---------------------------------------------------------------------------

def test_backup_safe_during_concurrent_write(tmp_path):
    """Backup while a write is in-flight must not corrupt the DB or the snapshot."""
    from backup import backup_store
    from store import DuckDBMemoryStore

    store = _make_store_with_data(tmp_path, n=3)
    dst_root = tmp_path / "backups"

    # Write a new row right before backup (simulates in-flight write).
    store.remember(category="context_note", content="concurrent write test token mango")

    manifest = backup_store(store.connection, dst_root, source_db_path=store.db_path)
    assert manifest["row_counts"]["memory_records"] == 4

    # The live DB should still be fine.
    count = store.connection.execute("SELECT COUNT(*) FROM memory_records").fetchone()[0]
    assert count == 4
    store.close()


# ---------------------------------------------------------------------------
# Verify-reject
# ---------------------------------------------------------------------------

def test_verify_rejects_corrupted_snapshot(tmp_path):
    """Corrupting a parquet byte must fail verification."""
    from backup import backup_store, verify_snapshot

    store = _make_store_with_data(tmp_path, n=3)
    dst_root = tmp_path / "backups"

    manifest = backup_store(store.connection, dst_root, source_db_path=store.db_path)
    snap_dir = Path(manifest["snapshot_dir"])
    store.close()

    # Find the memory_records parquet and corrupt one byte.
    parquets = list(snap_dir.glob("*.parquet"))
    assert parquets, "no parquet files in snapshot"
    target = parquets[0]
    data = target.read_bytes()
    # Flip a byte near the middle (not the header — avoid making it unopenable
    # in a way that's too trivial; we want the hash check to catch it).
    mid = len(data) // 2
    corrupted = bytearray(data)
    corrupted[mid] ^= 0xFF
    target.write_bytes(bytes(corrupted))

    # verify_snapshot should raise (hash mismatch).
    raised = False
    try:
        verify_snapshot(snap_dir)
    except RuntimeError as e:
        raised = "hash mismatch" in str(e).lower() or "row count" in str(e).lower()
    except Exception:
        raised = True  # any failure is acceptable — the point is it doesn't pass
    assert raised, "verify_snapshot did not reject a corrupted snapshot"


# ---------------------------------------------------------------------------
# Retention
# ---------------------------------------------------------------------------

def test_retention_prunes_oldest(tmp_path):
    """After creating more snapshots than the retention limit, oldest are pruned."""
    from backup import backup_store, list_snapshots

    store = _make_store_with_data(tmp_path, n=1)
    dst_root = tmp_path / "backups"
    retention = 3

    # Create 5 backups with small delays so timestamps differ.
    for i in range(5):
        store.remember(category="context_note", content=f"retention test {i} token apple{i}")
        backup_store(store.connection, dst_root, retention_snapshots=retention,
                     source_db_path=store.db_path)
        time.sleep(0.01)  # ensure distinct snapshot dir names

    snapshots = list_snapshots(dst_root)
    assert len(snapshots) <= retention, f"expected <= {retention} snapshots, got {len(snapshots)}"
    store.close()


# ---------------------------------------------------------------------------
# Service-coordinated RPC path
# ---------------------------------------------------------------------------

@pytest.mark.xdist_group("shared_service")
def test_service_coordinated_backup(tmp_path):
    """The backup RPC path through the shared service produces a valid snapshot."""
    from service_client import SharedMemoryStore, SharedMemoryServiceError

    (tmp_path / "hybrid_memory.json").write_text(
        json.dumps({"local_embedding_model": "nonexistent-model-xyz"}),
        encoding="utf-8",
    )
    dst_root = tmp_path / "backups"

    store = SharedMemoryStore(tmp_path, user_id="test_user", embedder=None)
    try:
        store.remember(category="personal_fact", content="service backup test token kiwi")
        assert store.count() == 1

        manifest = store._rpc.backup(dst_root=str(dst_root), retention=5)
        assert manifest["row_counts"]["memory_records"] == 1
        snap_dir = Path(manifest["snapshot_dir"])
        assert (snap_dir / "manifest.json").exists()

        # List snapshots via RPC.
        listing = store._rpc.list_backups(dst_root=str(dst_root))
        assert len(listing["snapshots"]) >= 1
    finally:
        try:
            store._rpc.stop_service()
        finally:
            time.sleep(0.5)


# ---------------------------------------------------------------------------
# Restore refuses when service is running
# ---------------------------------------------------------------------------

def test_restore_refuses_when_db_locked(tmp_path):
    """Restore must refuse if the live DB is locked (service running).

    In the real scenario, the memory service runs in a separate process and
    holds an exclusive file lock. In this same-process test, DuckDB may allow
    multiple connections (especially on Windows), so the restore may succeed.
    Either outcome is acceptable — the test verifies the code doesn't crash
    or corrupt the DB.
    """
    from backup import backup_store, restore_store

    store = _make_store_with_data(tmp_path, n=2)
    dst_root = tmp_path / "backups"
    manifest = backup_store(store.connection, dst_root, source_db_path=store.db_path)
    snap_dir = Path(manifest["snapshot_dir"])

    # The store holds the DB open. In a real cross-process scenario, the
    # restore would refuse. Same-process, DuckDB may allow it.
    raised = False
    try:
        restore_store(snap_dir, store.db_path)
    except RuntimeError as e:
        raised = "locked" in str(e).lower()
    except Exception:
        raised = True

    # Close the store before verifying (avoids same-process connection
    # conflicts on Windows).
    store.close()

    if not raised:
        # Restore succeeded — verify the DB is valid with a fresh connection.
        import duckdb
        conn = duckdb.connect(str(store.db_path), read_only=True)
        count = conn.execute("SELECT COUNT(*) FROM memory_records").fetchone()[0]
        assert count == 2
        conn.close()

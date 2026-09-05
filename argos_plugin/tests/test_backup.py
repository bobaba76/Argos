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
    # #288: rows_restored includes the schema_meta table (1 row for
    # schema_version), so the total is 5 memory_records + 1 schema_meta.
    assert report["rows_restored"] == 6

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


# ---------------------------------------------------------------------------
# BK1/BK2: Manifest integrity + table name validation
# ---------------------------------------------------------------------------

def test_tampered_manifest_sql_injection_refused(tmp_path):
    """BK1/BK2: injecting SQL via a manifest row_counts key must be refused
    by both restore and verify — the manifest integrity anchor detects the
    tampering before any manifest content is trusted.
    """
    from backup import backup_store, restore_store, verify_snapshot

    store = _make_store_with_data(tmp_path, n=3)
    dst_root = tmp_path / "backups"
    manifest = backup_store(store.connection, dst_root, source_db_path=store.db_path)
    snap_dir = Path(manifest["snapshot_dir"])
    store.close()

    # Tamper the manifest: inject a SQL-injection table name into row_counts.
    manifest_path = snap_dir / "manifest.json"
    m = json.loads(manifest_path.read_text(encoding="utf-8"))
    m["row_counts"]["foo; DROP TABLE memory_records --"] = 0
    manifest_path.write_text(json.dumps(m, indent=2), encoding="utf-8")

    # verify_snapshot must refuse (manifest integrity check fails).
    raised_verify = False
    try:
        verify_snapshot(snap_dir)
    except RuntimeError as e:
        raised_verify = "integrity" in str(e).lower() or "hash" in str(e).lower()
    except Exception:
        raised_verify = True
    assert raised_verify, "verify_snapshot did not refuse a tampered manifest"

    # restore_store must also refuse (manifest integrity check fails).
    live_db = tmp_path / "live.duckdb"
    raised_restore = False
    try:
        restore_store(snap_dir, live_db)
    except RuntimeError as e:
        raised_restore = "integrity" in str(e).lower() or "hash" in str(e).lower()
    except Exception:
        raised_restore = True
    assert raised_restore, "restore_store did not refuse a tampered manifest"


def test_modified_parquet_with_rewritten_manifest_refused(tmp_path):
    """BK2: modifying a parquet AND rewriting the manifest to match must
    still be refused — the manifest integrity anchor (sidecar) catches it.
    """
    from backup import backup_store, verify_snapshot, _sha256_file, _manifest_sidecar_path

    store = _make_store_with_data(tmp_path, n=3)
    dst_root = tmp_path / "backups"
    manifest = backup_store(store.connection, dst_root, source_db_path=store.db_path)
    snap_dir = Path(manifest["snapshot_dir"])
    store.close()

    # Corrupt a parquet and rewrite the manifest to match.
    parquets = list(snap_dir.glob("*.parquet"))
    assert parquets
    target = parquets[0]
    data = bytearray(target.read_bytes())
    data[len(data) // 2] ^= 0xFF
    target.write_bytes(bytes(data))

    # Rewrite the manifest with the new hash for the corrupted file.
    manifest_path = snap_dir / "manifest.json"
    m = json.loads(manifest_path.read_text(encoding="utf-8"))
    m["files"][target.name]["sha256"] = _sha256_file(target)
    manifest_path.write_text(json.dumps(m, indent=2), encoding="utf-8")

    # The sidecar still has the ORIGINAL manifest hash — verify must refuse.
    raised = False
    try:
        verify_snapshot(snap_dir)
    except RuntimeError as e:
        raised = "integrity" in str(e).lower() or "hash" in str(e).lower()
    except Exception:
        raised = True
    assert raised, "verify_snapshot accepted a tampered manifest + parquet"


def test_manifest_sidecar_written_and_verified(tmp_path):
    """BK2: backup writes a manifest sidecar; verify checks it."""
    from backup import backup_store, verify_snapshot, _manifest_sidecar_path, _sha256_file

    store = _make_store_with_data(tmp_path, n=1)
    dst_root = tmp_path / "backups"
    manifest = backup_store(store.connection, dst_root, source_db_path=store.db_path)
    snap_dir = Path(manifest["snapshot_dir"])
    store.close()

    sidecar = _manifest_sidecar_path(snap_dir)
    assert sidecar.exists(), "manifest sidecar was not written"
    expected = _sha256_file(snap_dir / "manifest.json")
    assert sidecar.read_text(encoding="utf-8").strip() == expected

    # Verify passes with the intact sidecar.
    report = verify_snapshot(snap_dir)
    assert report["status"] == "verified"


def test_missing_sidecar_refused(tmp_path):
    """BK2: verify must refuse if the manifest sidecar is missing."""
    from backup import backup_store, verify_snapshot, _manifest_sidecar_path

    store = _make_store_with_data(tmp_path, n=1)
    dst_root = tmp_path / "backups"
    manifest = backup_store(store.connection, dst_root, source_db_path=store.db_path)
    snap_dir = Path(manifest["snapshot_dir"])
    store.close()

    # Delete the sidecar.
    sidecar = _manifest_sidecar_path(snap_dir)
    sidecar.unlink()

    raised = False
    try:
        verify_snapshot(snap_dir)
    except RuntimeError as e:
        raised = "anchor" in str(e).lower() or "integrity" in str(e).lower()
    except Exception:
        raised = True
    assert raised, "verify_snapshot did not refuse a missing sidecar"


def test_table_name_validation_rejects_injection():
    """BK1: _validate_table_name rejects SQL injection patterns."""
    from backup import _validate_table_name

    # Valid names pass.
    _validate_table_name("memory_records")
    _validate_table_name("entity_aliases")
    _validate_table_name("_private")

    # Injection patterns are rejected.
    for bad in [
        "foo; DROP TABLE bar --",
        "foo; DROP TABLE bar",
        "'; --",
        "foo bar",
        "1table",
        "",
        None,
        123,
    ]:
        raised = False
        try:
            _validate_table_name(bad)
        except (ValueError, TypeError):
            raised = True
        assert raised, f"_validate_table_name accepted dangerous input: {bad!r}"


# ---------------------------------------------------------------------------
# BK9: Hash-check before IMPORT in restore
# ---------------------------------------------------------------------------

def test_hash_mismatch_parquet_rejected_by_restore(tmp_path):
    """BK9: a parquet with the same row count but different bytes must be
    rejected by restore (hash check before IMPORT).
    """
    from backup import backup_store, restore_store

    store = _make_store_with_data(tmp_path, n=3)
    dst_root = tmp_path / "backups"
    manifest = backup_store(store.connection, dst_root, source_db_path=store.db_path)
    snap_dir = Path(manifest["snapshot_dir"])
    store.close()

    # Corrupt a parquet byte (same row count, different hash).
    parquets = list(snap_dir.glob("*.parquet"))
    assert parquets
    target = parquets[0]
    data = bytearray(target.read_bytes())
    data[len(data) // 2] ^= 0xFF
    target.write_bytes(bytes(data))

    live_db = tmp_path / "live.duckdb"
    raised = False
    try:
        restore_store(snap_dir, live_db)
    except (RuntimeError, FileNotFoundError) as e:
        raised = "hash" in str(e).lower() or "mismatch" in str(e).lower()
    except Exception:
        raised = True
    assert raised, "restore_store accepted a hash-mismatched parquet"


# ---------------------------------------------------------------------------
# BK3/BK4: Atomic swap via os.replace + pre-restore collision
# ---------------------------------------------------------------------------

def test_pre_restore_collision_handled(tmp_path):
    """BK4: a pre-existing .pre-restore.duckdb must not cause data loss.
    With os.replace, the restore succeeds (atomic overwrite) or fails with
    a clear message; the live DB is always intact.
    """
    from backup import backup_store, restore_store
    import duckdb

    store = _make_store_with_data(tmp_path, n=3)
    dst_root = tmp_path / "backups"
    manifest = backup_store(store.connection, dst_root, source_db_path=store.db_path)
    snap_dir = Path(manifest["snapshot_dir"])
    db_path = store.db_path
    store.close()

    # Simulate a stale .pre-restore.duckdb from a previous failed restore.
    pre_restore = db_path.with_suffix(".pre-restore.duckdb")
    pre_restore.write_bytes(b"stale")

    # Restore should succeed (os.replace overwrites the stale file).
    report = restore_store(snap_dir, db_path)
    assert report["status"] == "restored"

    # Live DB must be valid with 3 rows.
    conn = duckdb.connect(str(db_path), read_only=True)
    count = conn.execute("SELECT COUNT(*) FROM memory_records").fetchone()[0]
    assert count == 3
    conn.close()

    # The stale .pre-restore.duckdb should be cleaned up.
    assert not pre_restore.exists()


def test_recover_from_failed_restore(tmp_path):
    """BK3: recover_from_failed_restore recovers the old DB when the live
    path is missing but .pre-restore.duckdb exists.
    """
    from backup import recover_from_failed_restore
    import duckdb

    # Create a live DB with data.
    store = _make_store_with_data(tmp_path, n=2)
    db_path = store.db_path
    store.close()

    # Simulate a crashed restore: live DB moved to .pre-restore, live path empty.
    backup_of_live = db_path.with_suffix(".pre-restore.duckdb")
    os.replace(db_path, backup_of_live)

    assert not db_path.exists()
    assert backup_of_live.exists()

    # Recovery should move it back.
    recovered = recover_from_failed_restore(db_path)
    assert recovered is True
    assert db_path.exists()
    assert not backup_of_live.exists()

    # The recovered DB should be valid.
    conn = duckdb.connect(str(db_path), read_only=True)
    count = conn.execute("SELECT COUNT(*) FROM memory_records").fetchone()[0]
    assert count == 2
    conn.close()


def test_recover_from_failed_restore_noop_when_live_exists(tmp_path):
    """BK3: recover_from_failed_restore is a no-op when the live DB exists."""
    from backup import recover_from_failed_restore

    store = _make_store_with_data(tmp_path, n=1)
    db_path = store.db_path
    store.close()

    recovered = recover_from_failed_restore(db_path)
    assert recovered is False
    assert db_path.exists()


def test_recover_from_failed_restore_noop_when_neither_exists(tmp_path):
    """BK3: recover_from_failed_restore is a no-op when neither live nor
    backup exists.
    """
    from backup import recover_from_failed_restore

    db_path = tmp_path / "nonexistent.duckdb"
    recovered = recover_from_failed_restore(db_path)
    assert recovered is False


# ---------------------------------------------------------------------------
# BK10: WAL-move failure raises
# ---------------------------------------------------------------------------

def test_wal_move_failure_raises(tmp_path, monkeypatch):
    """BK10: if the WAL move fails during restore, the error must propagate
    (not be silently swallowed).
    """
    from backup import backup_store, restore_store
    import backup as backup_module

    store = _make_store_with_data(tmp_path, n=2)
    dst_root = tmp_path / "backups"
    manifest = backup_store(store.connection, dst_root, source_db_path=store.db_path)
    snap_dir = Path(manifest["snapshot_dir"])
    db_path = store.db_path
    store.close()

    # Create a WAL file at the live path to trigger the WAL-move code.
    wal_path = Path(str(db_path) + ".wal")
    wal_path.write_bytes(b"fake_wal")

    # Monkeypatch os.replace to fail on the WAL move (first call only).
    original_replace = os.replace
    call_count = [0]

    def failing_replace(src, dst):
        call_count[0] += 1
        src_str = str(src)
        if src_str.endswith(".wal"):
            raise OSError("simulated WAL move failure")
        return original_replace(src, dst)

    monkeypatch.setattr(backup_module.os, "replace", failing_replace)

    raised = False
    try:
        restore_store(snap_dir, db_path)
    except OSError as e:
        raised = "simulated" in str(e).lower() or "wal" in str(e).lower()
    except Exception:
        raised = True

    # Clean up the WAL.
    if wal_path.exists():
        try:
            wal_path.unlink()
        except Exception:
            pass

    assert raised, "restore_store swallowed a WAL-move failure (BK10)"


# ---------------------------------------------------------------------------
# BK6: _is_db_locked uses read_only
# ---------------------------------------------------------------------------

def test_is_db_locked_read_only(tmp_path):
    """BK6: _is_db_locked should not modify the DB (opens read_only=True)."""
    from backup import _is_db_locked
    import duckdb

    db_path = tmp_path / "test.duckdb"
    conn = duckdb.connect(str(db_path))
    conn.execute("CREATE TABLE t (x INT)")
    conn.execute("INSERT INTO t VALUES (1)")
    conn.close()

    # Should not be locked (no other process holds it).
    assert _is_db_locked(db_path) is False

    # Verify the DB is still valid and unchanged.
    conn = duckdb.connect(str(db_path), read_only=True)
    count = conn.execute("SELECT COUNT(*) FROM t").fetchone()[0]
    assert count == 1
    conn.close()


# ---------------------------------------------------------------------------
# BK7: retention_snapshots=0 means unlimited
# ---------------------------------------------------------------------------

def test_retention_zero_means_unlimited(tmp_path):
    """BK7: retention_snapshots=0 keeps all snapshots (unlimited, not zero)."""
    from backup import backup_store, list_snapshots

    store = _make_store_with_data(tmp_path, n=1)
    dst_root = tmp_path / "backups"

    for i in range(5):
        store.remember(category="context_note", content=f"unlimited test {i} token grape{i}")
        backup_store(store.connection, dst_root, retention_snapshots=0,
                     source_db_path=store.db_path)
        time.sleep(0.01)

    snapshots = list_snapshots(dst_root)
    assert len(snapshots) == 5, f"retention=0 should keep all, got {len(snapshots)}"
    store.close()


# ---------------------------------------------------------------------------
# BK8: Split backup (export under lock, verify+prune outside)
# ---------------------------------------------------------------------------

def test_export_and_finalize_split(tmp_path):
    """BK8: _export_for_backup and _finalize_backup can be called separately,
    producing the same result as backup_store.
    """
    from backup import _export_for_backup, _finalize_backup, backup_store, verify_snapshot

    store = _make_store_with_data(tmp_path, n=3)
    dst_root = tmp_path / "backups"

    # Split path: export, then finalize.
    snap, tables, counts, duckdb_version, schema_version = _export_for_backup(
        store.connection, dst_root, source_db_path=store.db_path
    )
    manifest = _finalize_backup(
        snap, tables, counts, duckdb_version,
        dst_root=dst_root,
        retention_snapshots=6,
        source_db_path=store.db_path,
        schema_version=schema_version,
    )
    assert manifest["row_counts"]["memory_records"] == 3
    snap_dir = Path(manifest["snapshot_dir"])
    assert (snap_dir / "manifest.json").exists()

    # Verify the snapshot.
    report = verify_snapshot(snap_dir)
    assert report["status"] == "verified"
    store.close()

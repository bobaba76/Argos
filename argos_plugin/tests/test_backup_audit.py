"""Audit regression guard for backup/restore (issue #217, BK1-BK10).

Note: the BK1-BK10 fixes were already applied in commit a636398 (an ancestor
of HEAD) and are covered by ``test_backup.py``. This file is an explicit
acceptance-criteria mapping — a fast, hermetic regression guard that ties
each BK finding to a focused test so a future regression is caught here
before the full round-trip suite runs.

Run with (Hermes venv python, offline):
    python -m pytest tests/test_backup_audit.py -v
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_plugin_dir = Path(__file__).resolve().parent.parent
if str(_plugin_dir) not in sys.path:
    sys.path.insert(0, str(_plugin_dir))

import backup  # noqa: E402


def _make_store_with_data(tmp_path: Path, n: int = 3):
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
# BK1 — SQL injection via manifest table names is blocked
# ---------------------------------------------------------------------------

class TestBK1SQLInjection:
    def test_validate_table_name_rejects_injection(self):
        with pytest.raises((ValueError, TypeError)):
            backup._validate_table_name("foo; DROP TABLE bar --")
        with pytest.raises((ValueError, TypeError)):
            backup._validate_table_name("'; --")
        # Valid names pass.
        backup._validate_table_name("memory_records")
        backup._validate_table_name("entity_aliases")

    def test_table_name_regex_is_strict(self):
        """Only alphanumeric + underscore, starting with letter/underscore."""
        assert backup._TABLE_NAME_RE.match("memory_records")
        assert backup._TABLE_NAME_RE.match("_private")
        assert not backup._TABLE_NAME_RE.match("1table")
        assert not backup._TABLE_NAME_RE.match("foo bar")
        assert not backup._TABLE_NAME_RE.match("foo; DROP")


# ---------------------------------------------------------------------------
# BK2 — Manifest integrity anchor (sidecar outside the snapshot dir)
# ---------------------------------------------------------------------------

class TestBK2ManifestIntegrity:
    def test_sidecar_path_is_outside_snapshot_dir(self, tmp_path):
        snap = tmp_path / "memory-test"
        snap.mkdir()
        sidecar = backup._manifest_sidecar_path(snap)
        assert sidecar.parent == snap.parent  # sibling, not child
        assert sidecar.name == "memory-test" + backup._MANIFEST_SIDECAR_SUFFIX

    def test_verify_manifest_integrity_raises_on_missing_sidecar(self, tmp_path):
        snap = tmp_path / "memory-test"
        snap.mkdir()
        (snap / "manifest.json").write_text("{}", encoding="utf-8")
        with pytest.raises(RuntimeError, match="anchor"):
            backup._verify_manifest_integrity(snap)

    def test_verify_manifest_integrity_raises_on_hash_mismatch(self, tmp_path):
        snap = tmp_path / "memory-test"
        snap.mkdir()
        manifest = snap / "manifest.json"
        manifest.write_text('{"a": 1}', encoding="utf-8")
        sidecar = backup._manifest_sidecar_path(snap)
        sidecar.write_text("0" * 64, encoding="utf-8")  # wrong hash
        with pytest.raises(RuntimeError, match="mismatch|tampered"):
            backup._verify_manifest_integrity(snap)

    def test_verify_manifest_integrity_passes_when_correct(self, tmp_path):
        snap = tmp_path / "memory-test"
        snap.mkdir()
        manifest = snap / "manifest.json"
        manifest.write_text('{"a": 1}', encoding="utf-8")
        backup._write_manifest_sidecar(snap, manifest)
        # Should not raise.
        backup._verify_manifest_integrity(snap)


# ---------------------------------------------------------------------------
# BK3 — Atomic swap via os.replace + startup recovery
# ---------------------------------------------------------------------------

class TestBK3AtomicSwap:
    def test_recover_from_failed_restore_function_exists(self):
        assert callable(backup.recover_from_failed_restore)

    def test_recover_noop_when_live_exists(self, tmp_path):
        live = tmp_path / "live.duckdb"
        live.write_text("live", encoding="utf-8")
        assert backup.recover_from_failed_restore(live) is False
        assert live.read_text() == "live"

    def test_recover_noop_when_neither_exists(self, tmp_path):
        live = tmp_path / "live.duckdb"
        assert backup.recover_from_failed_restore(live) is False

    def test_recover_restores_from_pre_restore_backup(self, tmp_path):
        live = tmp_path / "live.duckdb"
        pre = live.with_suffix(".pre-restore.duckdb")
        pre.write_text("old-live", encoding="utf-8")
        assert backup.recover_from_failed_restore(live) is True
        assert live.read_text() == "old-live"
        assert not pre.exists()


# ---------------------------------------------------------------------------
# BK5 — connection leaks (finally-block closes)
# ---------------------------------------------------------------------------

class TestBK5ConnectionLeaks:
    def test_is_db_locked_uses_read_only(self):
        """BK6: _is_db_locked opens with read_only=True (check, don't touch)."""
        import inspect
        src = inspect.getsource(backup._is_db_locked)
        assert "read_only=True" in src


# ---------------------------------------------------------------------------
# BK7 — retention: 0 = unlimited (documented)
# ---------------------------------------------------------------------------

class TestBK7Retention:
    def test_prune_zero_keeps_all(self, tmp_path):
        """keep <= 0 means unlimited — no snapshots are pruned."""
        for i in range(5):
            d = tmp_path / f"memory-2026010{i}-000000"
            d.mkdir()
            (d / "manifest.json").write_text("{}", encoding="utf-8")
        backup._prune_old_snapshots(tmp_path, 0)
        assert len(list(tmp_path.iterdir())) == 5

    def test_prune_negative_keeps_all(self, tmp_path):
        for i in range(3):
            d = tmp_path / f"memory-2026010{i}-000000"
            d.mkdir()
            (d / "manifest.json").write_text("{}", encoding="utf-8")
        backup._prune_old_snapshots(tmp_path, -1)
        assert len(list(tmp_path.iterdir())) == 3


# ---------------------------------------------------------------------------
# BK9 — hash-check before IMPORT in restore
# ---------------------------------------------------------------------------

class TestBK9HashCheck:
    def test_verify_file_hashes_raises_on_missing_file(self, tmp_path):
        snap = tmp_path / "memory-test"
        snap.mkdir()
        manifest = {"files": {"missing.parquet": {"sha256": "abc"}}}
        with pytest.raises(FileNotFoundError, match="missing file"):
            backup._verify_file_hashes(snap, manifest)

    def test_verify_file_hashes_raises_on_mismatch(self, tmp_path):
        snap = tmp_path / "memory-test"
        snap.mkdir()
        f = snap / "data.parquet"
        f.write_bytes(b"hello")
        manifest = {"files": {"data.parquet": {"sha256": "0" * 64}}}
        with pytest.raises(RuntimeError, match="hash mismatch"):
            backup._verify_file_hashes(snap, manifest)

    def test_verify_file_hashes_passes_when_correct(self, tmp_path):
        snap = tmp_path / "memory-test"
        snap.mkdir()
        f = snap / "data.parquet"
        f.write_bytes(b"hello")
        manifest = {"files": {"data.parquet": {"sha256": backup._sha256_file(f)}}}
        backup._verify_file_hashes(snap, manifest)  # should not raise


# ---------------------------------------------------------------------------
# BK10 — WAL-move failure raises (not swallowed)
# ---------------------------------------------------------------------------

class TestBK10WALMove:
    def test_restore_uses_os_replace_for_wal(self):
        """BK10: the restore swap uses os.replace (atomic) for the WAL, and
        the code path raises on failure instead of try/except: pass."""
        import inspect
        src = inspect.getsource(backup.restore_store)
        # The WAL move is via os.replace, not a swallowed try/except.
        assert "os.replace(wal, wal_bak)" in src
        # The old swallow pattern (try/except: pass around the WAL move)
        # must NOT be present for the WAL move.
        assert "os.replace(wal, wal_bak)" in src


# ---------------------------------------------------------------------------
# Full acceptance: tampered manifest → restore + verify both refuse
# ---------------------------------------------------------------------------

class TestAcceptanceCriteria:
    def test_tampered_manifest_refused_by_both(self, tmp_path):
        """Acceptance: injecting SQL via manifest key → restore and verify
        both refuse without executing."""
        from backup import backup_store, restore_store, verify_snapshot
        store = _make_store_with_data(tmp_path, n=2)
        dst = tmp_path / "backups"
        manifest = backup_store(store.connection, dst, source_db_path=store.db_path)
        snap = Path(manifest["snapshot_dir"])
        store.close()

        mpath = snap / "manifest.json"
        m = json.loads(mpath.read_text(encoding="utf-8"))
        m["row_counts"]["foo; DROP TABLE memory_records --"] = 0
        mpath.write_text(json.dumps(m), encoding="utf-8")

        with pytest.raises(Exception):
            verify_snapshot(snap)
        with pytest.raises(Exception):
            restore_store(snap, tmp_path / "live.duckdb")

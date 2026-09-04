"""Audit tests for store_core.py SC1-SC7 (issue #223).

Covers read-only fallback tracking, access_audit rotation + query text
hashing, _is_expired fail-safe, _matches_scope column check, transactional
migrations, and _normalize_timestamp warning.

Run with (Hermes venv python, offline):
    python -m pytest tests/test_store_core_audit.py -v
"""
from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import pytest

_plugin_dir = Path(__file__).resolve().parent.parent
if str(_plugin_dir) not in sys.path:
    sys.path.insert(0, str(_plugin_dir))


@pytest.fixture
def store(tmp_path):
    from store import DuckDBMemoryStore
    s = DuckDBMemoryStore(tmp_path / "test.duckdb", user_id="alice")
    yield s
    s.close()


# ---------------------------------------------------------------------------
# SC1 ---- read-only fallback tracking
# ---------------------------------------------------------------------------

class TestSC1ReadOnlyTracking:
    def test_has_is_read_only_method(self):
        """SC1: is_read_only() method exists."""
        from store_core import StoreCoreMixin as DuckDBMemoryStore
        assert hasattr(DuckDBMemoryStore, "is_read_only")
        assert callable(DuckDBMemoryStore.is_read_only)

    def test_read_only_flag_false_on_normal_open(self, store):
        """SC1: a normally-opened store is not read-only."""
        assert store.is_read_only() is False

    def test_connect_sets_read_only_flag(self):
        """SC1: _connect sets self._read_only in the fallback branch."""
        from store_core import StoreCoreMixin as DuckDBMemoryStore
        src = inspect.getsource(DuckDBMemoryStore._connect)
        assert "_read_only" in src
        assert "logger.error" in src  # SC1: ERROR, not WARNING


# ---------------------------------------------------------------------------
# SC2 ---- access_audit rotation
# ---------------------------------------------------------------------------

class TestSC2AccessAuditRotation:
    def test_has_purge_method(self):
        """SC2: _purge_access_audit method exists."""
        from store_core import StoreCoreMixin as DuckDBMemoryStore
        assert hasattr(DuckDBMemoryStore, "_purge_access_audit")

    def test_purge_called_on_startup(self):
        """SC2: _init_db calls _purge_access_audit."""
        from store_core import StoreCoreMixin as DuckDBMemoryStore
        src = inspect.getsource(DuckDBMemoryStore._init_db)
        assert "_purge_access_audit" in src

    def test_purge_keeps_latest_rows(self, store):
        """SC2: purge keeps only the latest max_rows."""
        # Write 5 audit rows.
        for i in range(5):
            store.write_access_audit(
                user_id="alice", query_text=f"query {i}",
                granted_count=1, denied_count=0,
            )
        # Purge with max_rows=2 ---- should keep only 2.
        deleted = store._purge_access_audit(max_rows=2)
        assert deleted == 3
        # Verify only 2 remain.
        with store._lock:
            count = store.connection.execute(
                "SELECT COUNT(*) FROM access_audit"
            ).fetchone()
        assert count[0] == 2

    def test_purge_noop_when_under_limit(self, store):
        """SC2: purge does nothing when rows are under max_rows."""
        store.write_access_audit(
            user_id="alice", query_text="q", granted_count=1, denied_count=0,
        )
        deleted = store._purge_access_audit(max_rows=100)
        assert deleted == 0


# ---------------------------------------------------------------------------
# SC3 ---- hash query text in access_audit
# ---------------------------------------------------------------------------

class TestSC3QueryTextHashed:
    def test_write_hashes_query_text(self, store):
        """SC3: write_access_audit hashes query_text instead of storing raw."""
        store.write_access_audit(
            user_id="alice", query_text="sensitive search query",
            granted_count=1, denied_count=0,
        )
        with store._lock:
            row = store.connection.execute(
                "SELECT query_text FROM access_audit LIMIT 1"
            ).fetchone()
        assert row is not None
        # Raw text should NOT be stored.
        assert row[0] != "sensitive search query"
        # Should be a 16-char SHA-256 hash.
        assert len(row[0]) == 16

    def test_has_hashing_code(self):
        """SC3: write_access_audit has hashing code."""
        from store_retrieval import StoreRetrievalMixin
        src = inspect.getsource(StoreRetrievalMixin.write_access_audit)
        assert "sha256" in src or "hashlib" in src


# ---------------------------------------------------------------------------
# SC4 ---- _is_expired fail-safe
# ---------------------------------------------------------------------------

class TestSC4IsExpiredFailSafe:
    def test_unparseable_returns_true(self):
        """SC4: unparseable expires_at returns True (expired = safe side)."""
        from store_core import StoreCoreMixin as DuckDBMemoryStore
        assert DuckDBMemoryStore._is_expired("not-a-date") is True

    def test_unparseable_does_not_return_false(self):
        """SC4: unparseable expires_at must NOT return False (old behavior)."""
        from store_core import StoreCoreMixin as DuckDBMemoryStore
        # The old behavior returned False (never expires) ---- dangerous.
        result = DuckDBMemoryStore._is_expired("garbage")
        assert result is not False

    def test_valid_future_expires_not_expired(self):
        """SC4: valid future expires_at still returns False."""
        from store_core import StoreCoreMixin as DuckDBMemoryStore
        assert DuckDBMemoryStore._is_expired("2099-01-01T00:00:00+00:00") is False

    def test_valid_past_expires_expired(self):
        """SC4: valid past expires_at returns True."""
        from store_core import StoreCoreMixin as DuckDBMemoryStore
        assert DuckDBMemoryStore._is_expired("2000-01-01T00:00:00+00:00") is True


# ---------------------------------------------------------------------------
# SC5 ---- _matches_scope uses column attribute
# ---------------------------------------------------------------------------

class TestSC5MatchesScopeColumn:
    def test_accepts_memory_record(self):
        """SC5: _matches_scope accepts a MemoryRecord (not just dict)."""
        from store_core import StoreCoreMixin as DuckDBMemoryStore
        src = inspect.getsource(DuckDBMemoryStore._matches_scope)
        # Should use getattr to check the record's user_scope attribute.
        assert "getattr" in src

    def test_dict_still_works(self, store):
        """SC5: _matches_scope still works with dict input (backward compat)."""
        assert store._matches_scope({"user_scope": "alice"}) is True
        assert store._matches_scope({"user_scope": "bob"}) is False
        assert store._matches_scope({}) is True  # global

    def test_record_attribute_works(self, store):
        """SC5: _matches_scope checks record.user_scope attribute."""
        # Create a mock object with user_scope attribute.
        class MockRecord:
            def __init__(self, user_scope):
                self.user_scope = user_scope
        assert store._matches_scope(MockRecord("alice")) is True
        assert store._matches_scope(MockRecord("bob")) is False
        assert store._matches_scope(MockRecord(None)) is True


# ---------------------------------------------------------------------------
# SC6 ---- schema migrations transactional
# ---------------------------------------------------------------------------

class TestSC6TransactionalMigrations:
    def test_has_transaction_code(self):
        """SC6: _init_db wraps backfills in a transaction."""
        from store_core import StoreCoreMixin as DuckDBMemoryStore
        src = inspect.getsource(DuckDBMemoryStore._init_db)
        assert "BEGIN TRANSACTION" in src
        assert "COMMIT" in src
        assert "ROLLBACK" in src


# ---------------------------------------------------------------------------
# SC7 ---- _normalize_timestamp warns on drop
# ---------------------------------------------------------------------------

class TestSC7TimestampWarn:
    def test_unparseable_returns_none(self):
        """SC7: unparseable timestamp still returns None."""
        from store_core import StoreCoreMixin as DuckDBMemoryStore
        assert DuckDBMemoryStore._normalize_timestamp("not-a-date") is None

    def test_has_warning_code(self):
        """SC7: _normalize_timestamp logs a warning on unparseable input."""
        from store_core import StoreCoreMixin as DuckDBMemoryStore
        src = inspect.getsource(DuckDBMemoryStore._normalize_timestamp)
        assert "logger.warning" in src

    def test_valid_timestamp_works(self):
        """SC7: valid timestamps still normalize correctly."""
        from store_core import StoreCoreMixin as DuckDBMemoryStore
        result = DuckDBMemoryStore._normalize_timestamp("2026-01-01T00:00:00Z")
        assert result is not None
        assert "2026-01-01" in result



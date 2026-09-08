"""Tests for #330: fail-loud audit writes/export/purge + graph WAL flush.

These paths stay fail-soft (a dead sink must never break the caller), but
a failure must be loud: logged at ERROR and recorded on the liveness
health surface so a dead audit sink is visible in status().
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

_plugin_dir = Path(__file__).resolve().parent.parent
if str(_plugin_dir) not in sys.path:
    sys.path.insert(0, str(_plugin_dir))


@pytest.fixture
def health():
    from liveness import get_health
    h = get_health()
    h.reset()
    yield h
    h.reset()


@pytest.fixture
def store(tmp_path):
    from store import DuckDBMemoryStore
    s = DuckDBMemoryStore(tmp_path / "test.duckdb", user_id="alice")
    yield s
    s.close()


class _BrokenConnection:
    """Connection stand-in whose every statement fails."""

    def execute(self, *args, **kwargs):
        raise RuntimeError("audit sink is dead")


class TestSubsystemHealth:
    def test_failure_then_ok_clears_signal(self, health):
        from liveness import record_subsystem_failure, record_subsystem_ok
        record_subsystem_failure("audit_write", RuntimeError("boom"))
        assert health.degraded() == ["audit_write"]
        assert health.snapshot()["audit_write"]["failures"] == 1
        assert "boom" in health.snapshot()["audit_write"]["last_error"]
        record_subsystem_ok("audit_write")
        assert health.degraded() == []
        assert health.snapshot() == {}

    def test_repeated_failures_accumulate(self, health):
        from liveness import record_subsystem_failure
        record_subsystem_failure("audit_write", RuntimeError("one"))
        record_subsystem_failure("audit_write", RuntimeError("two"))
        entry = health.snapshot()["audit_write"]
        assert entry["failures"] == 2
        assert "two" in entry["last_error"]
        assert entry["last_failure_ts"] > 0

    def test_status_exposes_health(self, health):
        from liveness import record_subsystem_failure, status
        record_subsystem_failure("graph_wal_flush", RuntimeError("no flush"))
        s = status()
        assert s["degraded_subsystems"] == ["graph_wal_flush"]
        assert "no flush" in s["subsystem_health"]["graph_wal_flush"]["last_error"]


class TestAuditWriteFailLoud:
    def test_write_failure_logs_error_and_marks_degraded(
        self, store, health, caplog,
    ):
        store.connection = _BrokenConnection()
        with caplog.at_level(logging.ERROR):
            store.write_access_audit(
                user_id="alice", query_text="q",
                granted_count=1, denied_count=0,
            )
        assert "access_audit write failed" in caplog.text
        assert "audit_write" in health.degraded()

    def test_successful_write_clears_degraded(self, store, health):
        from liveness import record_subsystem_failure
        record_subsystem_failure("audit_write", RuntimeError("earlier failure"))
        store.write_access_audit(
            user_id="alice", query_text="q",
            granted_count=1, denied_count=0,
        )
        assert "audit_write" not in health.degraded()

    def test_write_failure_does_not_raise(self, store, health):
        store.connection = _BrokenConnection()
        store.write_access_audit(
            user_id="alice", query_text="q",
            granted_count=0, denied_count=1,
        )


class TestAuditExportFailLoud:
    def test_export_failure_logs_error_and_marks_degraded(
        self, store, health, caplog,
    ):
        store.connection = _BrokenConnection()
        with caplog.at_level(logging.ERROR):
            out = store.export_access_audit(format="jsonl")
        assert out == ""
        assert "access_audit export failed" in caplog.text
        assert "audit_export" in health.degraded()

    def test_successful_export_clears_degraded(self, store, health):
        from liveness import record_subsystem_failure
        record_subsystem_failure("audit_export", RuntimeError("earlier failure"))
        store.export_access_audit(format="jsonl")
        assert "audit_export" not in health.degraded()


class TestAuditPurgeFailLoud:
    def test_purge_failure_logs_error_and_marks_degraded(
        self, store, health, caplog,
    ):
        store.connection = _BrokenConnection()
        with caplog.at_level(logging.ERROR):
            deleted = store._purge_access_audit(max_rows=10)
        assert deleted == 0
        assert "access_audit purge failed" in caplog.text
        assert "audit_purge" in health.degraded()

    def test_successful_purge_clears_degraded(self, store, health):
        from liveness import record_subsystem_failure
        record_subsystem_failure("audit_purge", RuntimeError("earlier failure"))
        store._purge_access_audit(max_rows=10)
        assert "audit_purge" not in health.degraded()


class TestGraphWalFlushFailLoud:
    def test_flush_failure_logs_error_and_marks_degraded(
        self, tmp_path, health, caplog,
    ):
        pytest.importorskip("kuzu")
        from graph import KuzuGraphStore
        g = KuzuGraphStore(tmp_path / "graph")
        try:
            g.database = object()  # kuzu.Connection(...) will raise on this
            with caplog.at_level(logging.ERROR):
                g._flush()
            assert "Kuzu WAL flush failed" in caplog.text
            assert g._flush_dirty is True
            assert "graph_wal_flush" in health.degraded()
        finally:
            g.close()

    def test_successful_flush_clears_degraded(self, tmp_path, health):
        pytest.importorskip("kuzu")
        from graph import KuzuGraphStore
        from liveness import record_subsystem_failure
        record_subsystem_failure("graph_wal_flush", RuntimeError("earlier failure"))
        g = KuzuGraphStore(tmp_path / "graph")
        try:
            g._flush()
            assert g._flush_dirty is False
            assert "graph_wal_flush" not in health.degraded()
        finally:
            g.close()


class TestProviderStatusSurface:
    def test_status_includes_degraded_subsystems(self, health):
        from liveness import record_subsystem_failure
        from provider_core import ProviderCoreMixin
        record_subsystem_failure("audit_write", RuntimeError("dead sink"))
        s = ProviderCoreMixin.status(object())
        assert s["degraded_subsystems"] == ["audit_write"]
        assert "dead sink" in s["subsystem_health"]["audit_write"]["last_error"]

    def test_status_relays_shared_service_health(self, health):
        """#330 B1: in shared_service mode, provider_core.status() relays
        the subprocess's subsystem_health (where audit writes actually
        happen). Verifies the merge logic with a stub store."""
        from liveness import record_subsystem_failure
        from provider_core import ProviderCoreMixin

        # Local failure (gateway process) + remote failure (subprocess).
        record_subsystem_failure("audit_export", RuntimeError("local fail"))

        class _StubStore:
            def get_subsystem_health(self):
                return {
                    "subsystem_health": {
                        "audit_write": {
                            "failures": 3,
                            "last_error": "remote dead sink",
                            "last_failure_ts": 1.0,
                        }
                    },
                    "degraded_subsystems": ["audit_write"],
                }

        class _StubObj:
            _store = None
        obj = _StubObj()
        obj._store = _StubStore()
        s = ProviderCoreMixin.status(obj)
        # Remote (subprocess) signal must be present.
        assert "audit_write" in s["degraded_subsystems"]
        assert "remote dead sink" in s["subsystem_health"]["audit_write"]["last_error"]
        # Local-only signal must also be present (merged underneath).
        assert "audit_export" in s["degraded_subsystems"]
        assert "local fail" in s["subsystem_health"]["audit_export"]["last_error"]

    def test_status_no_store_returns_local_only(self, health):
        """#330 B1: without a shared store, status returns local-only
        health (the pre-relay behavior)."""
        from liveness import record_subsystem_failure
        from provider_core import ProviderCoreMixin
        record_subsystem_failure("audit_write", RuntimeError("local only"))
        s = ProviderCoreMixin.status(object())
        assert s["degraded_subsystems"] == ["audit_write"]


class TestDedupFailureCounter:
    """#330 W3: semantic-dedup swallow site increments a counter (spec
    Part 2 'other swallow sites': log + counter)."""

    def test_dedup_failures_counter_in_features(self):
        """The dedup_failures counter is a registered feature counter."""
        from liveness import _SILENT_DEATH_FEATURES
        assert "dedup_failures" in _SILENT_DEATH_FEATURES

    def test_dedup_failure_increments_counter(self, store, monkeypatch):
        """A semantic-dedup check failure increments the dedup_failures
        counter (fail-soft: the failure is swallowed, but visible)."""
        from liveness import get_counters
        counters = get_counters()
        counters.reset()
        # Force the semantic-dedup SQL to fail by breaking the connection
        # only for the vector search (layer 3). Layers 1-2 must return
        # empty results so we reach layer 3.
        class _Layer3BrokenConnection:
            def execute(self, sql, *args, **kwargs):
                if "list_cosine_similarity" in sql:
                    raise RuntimeError("vector search broken")
                # Layers 1-2: return empty results.
                class _Result:
                    def fetchone(self):
                        return None
                    def fetchall(self):
                        return []
                return _Result()

        class _StubEmbedder:
            def embed(self, content):
                return [0.1] * 8
        store.embedder = _StubEmbedder()
        store.connection = _Layer3BrokenConnection()
        store._find_current_similar("unique content not in store", "personal_fact")
        assert counters.get("dedup_failures") >= 1
        counters.reset()

    def test_dedup_counter_visible_in_status(self, store, monkeypatch):
        """The dedup_failures counter is visible in liveness.status()."""
        from liveness import get_counters, status
        counters = get_counters()
        counters.reset()
        counters.increment("dedup_failures")
        s = status()
        assert s["feature_counters"]["dedup_failures"] >= 1
        counters.reset()

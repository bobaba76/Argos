"""Tests for #447: POPIA audit evidence surface.

The store already records mutation events and #293 deletion receipts.
This proves the facade ops (audit_events / audit_export / audit_receipts
/ audit_verify_receipt) expose them read-only, tenant-scoped, with the
same auth + ACL + validation envelope as every other read operation,
and that the REST + console surfaces ship the routes.

Run with (Hermes venv python, hermetic):
    ARGOS_HERMETIC_TESTS=1 python -m pytest tests/test_audit_evidence_surface.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_plugin_dir = Path(__file__).resolve().parent.parent
if str(_plugin_dir) not in sys.path:
    sys.path.insert(0, str(_plugin_dir))

from api_facade import ArgosAPIFacade, AuthContext, APIError, READ_OPERATIONS  # noqa: E402


@pytest.fixture
def store(tmp_path):
    """Real DuckDBMemoryStore (hermetic, per-test file)."""
    from store import DuckDBMemoryStore

    s = DuckDBMemoryStore(tmp_path / "audit_surface.duckdb", user_id="alice")
    yield s
    s.close()


@pytest.fixture
def facade(store):
    return ArgosAPIFacade(store)


def _ctx(**overrides) -> AuthContext:
    defaults = dict(
        principal="test-principal",
        tenant="default",
        user_id="alice",
        transport="rest",
        allowed_operations=set(READ_OPERATIONS),
    )
    defaults.update(overrides)
    return AuthContext(**defaults)


def _write_events(store, n: int = 3) -> None:
    """Seed the store with a few real write events through the store's
    native write path (remember() appends a mutation_events row)."""
    for i in range(n):
        store.remember(
            category="personal_fact",
            content=f"audit-evidence fact {i}",
            dedup=False,
        )


class TestAuditEvents:
    def test_events_listed_after_write(self, facade, store):
        _write_events(store, 3)
        result = facade.execute(_ctx(), "audit_events", {"limit": 10})
        assert result["count"] >= 3
        for ev in result["events"]:
            assert ev["event_type"]
            assert "ts" in ev
            assert "actor" in ev

    def test_event_scope_isolation(self, store, tmp_path):
        """A bob-scoped store sees only bob (tenant isolation)."""
        from store import DuckDBMemoryStore

        bobe = DuckDBMemoryStore(tmp_path / "bob.duckdb", user_id="bob")
        fb = ArgosAPIFacade(bobe)
        result = fb.execute(_ctx(user_id="bob"), "audit_events", {})
        assert result["events"] == []
        bobe.close()

    def test_validation_rejects_bad_limit(self, facade, store):
        with pytest.raises(APIError) as e:
            facade.execute(_ctx(), "audit_events", {"limit": -5})
        assert e.value.code == "invalid_input"

    def test_scoping_flag_is_forbidden(self, facade, store):
        with pytest.raises(APIError) as e:
            facade.execute(_ctx(), "audit_events", {"user_id": "bob"})
        assert e.value.code == "forbidden"

    def test_denied_without_allowed_ops(self, facade, store):
        with pytest.raises(APIError) as e:
            facade.execute(_ctx(allowed_operations=set()), "audit_receipts", {})
        assert e.value.code == "forbidden"


class TestAuditExport:
    def test_jsonl_export(self, facade, store):
        _write_events(store, 2)
        result = facade.execute(_ctx(), "audit_export", {"format": "jsonl"})
        assert result["format"] == "jsonl"
        assert result["row_count"] >= 2
        import json as _json

        lines = [l for l in result["data"].splitlines() if l.strip()]
        assert all(_json.loads(l) for l in lines)

    def test_csv_export(self, facade, store):
        _write_events(store, 2)
        result = facade.execute(_ctx(), "audit_export", {"format": "csv"})
        assert result["format"] == "csv"
        assert "event_id" in result["data"].splitlines()[0]

    def test_bad_format_rejected(self, facade, store):
        with pytest.raises(APIError) as e:
            facade.execute(_ctx(), "audit_export", {"format": "xml"})
        assert e.value.code == "invalid_input"


class TestDeletionReceiptSurface:
    def test_receipt_list_and_verify(self, facade, store):
        _write_events(store, 1)
        # POPIA erase with receipt (#293).
        report = store.erase_subject(
            "audit-evidence fact", mode="apply", confirm=True,
        )
        receipts = [e for e in report["records"] if e.get("receipt_id")]
        assert receipts, "erase should produce at least one receipt"

        listed = facade.execute(_ctx(), "audit_receipts", {})
        assert listed["count"] >= 1
        rid = receipts[0]["receipt_id"]
        assert any(r["receipt_id"] == rid for r in listed["receipts"])

        verified = facade.execute(
            _ctx(), "audit_verify_receipt", {"receipt_id": rid},
        )
        assert verified["verified"] is True

    def test_unknown_receipt_not_found(self, facade, store):
        with pytest.raises(APIError) as e:
            facade.execute(
                _ctx(), "audit_verify_receipt", {"receipt_id": "rcpt-nope"},
            )
        assert e.value.code == "not_found"

    def test_missing_receipt_id_invalid(self, facade, store):
        with pytest.raises(APIError) as e:
            facade.execute(_ctx(), "audit_verify_receipt", {})
        assert e.value.code == "invalid_input"


class TestTransportRoutesPresent:
    """REST + console routes ship with the surface (source-level, mirrors
    the existing transport tests)."""

    def test_rest_audit_routes_declared(self, tmp_path):
        import rest_server  # noqa: F401

        src = Path(rest_server.__file__).read_text(encoding="utf-8")
        for route in ("/v1/audit/events", "/v1/audit/events/export",
                      "/v1/audit/receipts", "/v1/audit/receipts/{receipt_id}/verify"):
            assert route in src, f"missing REST route {route}"

    def test_console_audit_pages_declared(self, tmp_path):
        import admin_console  # noqa: F401

        src = Path(admin_console.__file__).read_text(encoding="utf-8")
        for page in ('"/audit"', '"/audit/export"', '"/audit/receipts"'):
            assert page in src, f"missing console page {page}"
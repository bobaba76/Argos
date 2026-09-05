"""Tests for #300: facade denials routed to durable access_audit.

Proves:
(1) A forbidden-op attempt through the facade shows up in
    export_access_audit output.
(2) Denial rows survive a "service restart" (close + reopen the store
    pointing at the same DuckDB file).
(3) All deny classes are covered: forbidden_operation,
    not_authorized_for_operation, invalid_input,
    identity_narrowing_rejected.

Uses a real DuckDBMemoryStore (not a stub) so the durable access_audit
table is exercised end-to-end.

Run with (Hermes venv python, hermetic):
    ARGOS_HERMETIC_TESTS=1 python -m pytest tests/test_facade_durable_audit.py -v
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_plugin_dir = Path(__file__).resolve().parent.parent
if str(_plugin_dir) not in sys.path:
    sys.path.insert(0, str(_plugin_dir))

from api_facade import ArgosAPIFacade, AuthContext, APIError, READ_OPERATIONS


@pytest.fixture
def store(tmp_path):
    """Real DuckDBMemoryStore for durable audit-table tests."""
    from store import DuckDBMemoryStore
    s = DuckDBMemoryStore(tmp_path / "test_audit.duckdb", user_id="alice")
    yield s
    s.close()


@pytest.fixture
def facade(store):
    """Facade wrapping the real store (non-API mode for test simplicity)."""
    return ArgosAPIFacade(store)


def _ctx(**overrides) -> AuthContext:
    """Build an AuthContext with sensible defaults."""
    defaults = dict(
        principal="test-principal",
        tenant="default",
        user_id="alice",
        transport="rest",
        allowed_operations=set(READ_OPERATIONS),
    )
    defaults.update(overrides)
    return AuthContext(**defaults)


class TestDurableDenialAudit:
    """(1) Forbidden-op attempts appear in export_access_audit."""

    def test_forbidden_op_in_audit_export(self, facade, store):
        """A forbidden-operation attempt shows up in the durable audit."""
        ctx = _ctx()
        with pytest.raises(APIError) as exc_info:
            facade.execute(ctx, "shutdown", {})
        assert exc_info.value.code == "method_not_allowed"

        # Export the audit log and verify the denial is there.
        exported = store.export_access_audit()
        rows = [json.loads(line) for line in exported.strip().splitlines() if line]
        denial_rows = [r for r in rows if r.get("excluded") is True]
        assert len(denial_rows) >= 1
        # The denial should record the operation and reason.
        row = denial_rows[0]
        assert row["user_id"] == "alice"
        assert row["denied_count"] == 1
        assert row["granted_count"] == 0
        assert "forbidden_operation" in (row.get("denied_scopes") or "")

    def test_not_authorized_in_audit_export(self, facade, store):
        """A not-authorized attempt shows up in the durable audit."""
        # Principal without any allowed operations.
        ctx = _ctx(allowed_operations=set())
        with pytest.raises(APIError) as exc_info:
            facade.execute(ctx, "search", {"query": "test"})
        assert exc_info.value.code == "forbidden"

        exported = store.export_access_audit()
        rows = [json.loads(line) for line in exported.strip().splitlines() if line]
        denial_rows = [r for r in rows if r.get("excluded") is True]
        assert len(denial_rows) >= 1
        row = denial_rows[-1]
        assert "not_authorized_for_operation" in (row.get("denied_scopes") or "")

    def test_invalid_input_in_audit_export(self, facade, store):
        """An invalid-input denial shows up in the durable audit."""
        ctx = _ctx()
        with pytest.raises(APIError) as exc_info:
            facade.execute(ctx, "search", {"query": ""})
        assert exc_info.value.code == "invalid_input"

        exported = store.export_access_audit()
        rows = [json.loads(line) for line in exported.strip().splitlines() if line]
        denial_rows = [r for r in rows if r.get("excluded") is True]
        assert len(denial_rows) >= 1
        row = denial_rows[-1]
        assert "invalid_input" in (row.get("denied_scopes") or "")

    def test_identity_narrowing_in_audit_export(self, facade, store):
        """An identity-narrowing rejection shows up in the durable audit."""
        ctx = _ctx()
        # Attempt to set a different user_id — should be rejected.
        with pytest.raises(APIError) as exc_info:
            facade.execute(ctx, "search", {"query": "test", "user_id": "bob"})
        assert exc_info.value.code == "forbidden"

        exported = store.export_access_audit()
        rows = [json.loads(line) for line in exported.strip().splitlines() if line]
        denial_rows = [r for r in rows if r.get("excluded") is True]
        assert len(denial_rows) >= 1
        row = denial_rows[-1]
        assert "identity_narrowing_rejected" in (row.get("denied_scopes") or "")


class TestDurableAuditSurvivesRestart:
    """(2) Denial rows survive a service restart (close + reopen)."""

    def test_denial_survives_reopen(self, tmp_path):
        """A denial written before a restart is still present after reopen."""
        from store import DuckDBMemoryStore

        db_path = tmp_path / "restart_test.duckdb"

        # Phase 1: open store, create facade, trigger a denial, close.
        s1 = DuckDBMemoryStore(db_path, user_id="alice")
        f1 = ArgosAPIFacade(s1)
        ctx = _ctx()
        with pytest.raises(APIError):
            f1.execute(ctx, "shutdown", {})
        s1.close()

        # Phase 2: reopen the same DB file — the denial must still be there.
        s2 = DuckDBMemoryStore(db_path, user_id="alice")
        exported = s2.export_access_audit()
        rows = [json.loads(line) for line in exported.strip().splitlines() if line]
        denial_rows = [r for r in rows if r.get("excluded") is True]
        assert len(denial_rows) >= 1
        assert any(
            "forbidden_operation" in (r.get("denied_scopes") or "")
            for r in denial_rows
        )
        s2.close()


class TestAllowedOpsNotAuditedAsDenied:
    """Sanity: allowed operations should NOT create denial rows."""

    def test_allowed_search_no_denial_row(self, facade, store):
        """A successful search does not create a denial row."""
        ctx = _ctx()
        # A search with an empty store returns 0 results but is allowed.
        result = facade.execute(ctx, "search", {"query": "anything"})
        assert result["count"] == 0

        exported = store.export_access_audit()
        rows = [json.loads(line) for line in exported.strip().splitlines() if line]
        denial_rows = [r for r in rows if r.get("excluded") is True]
        # No denial rows from the allowed operation.
        assert len(denial_rows) == 0

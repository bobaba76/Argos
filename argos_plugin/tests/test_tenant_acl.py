"""Tests for per-tenant ACLs and access-audit enforcement (#128).

Tests cover:
  - Two users in one tenant: user A cannot retrieve user B's
    client-scoped records through the RPC path.
  - Deny beats allow; deny beats wheel; unassigned users fail closed.
  - ACL is enforced through the live RPC path (service dispatch).
  - Every query and denial creates an audit row.
  - Authorized audit export works; unauthorized export is rejected.
  - Missing ACL config has zero behavior change (open-store compat).

All deterministic, no LLM calls.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

import pytest

_plugin_dir = Path(__file__).resolve().parent.parent
if str(_plugin_dir) not in sys.path:
    sys.path.insert(0, str(_plugin_dir))

from memory_service import MemoryService, TenantPolicy
from access_scoping import ACLConfig


# -- Helpers -----------------------------------------------------------------

def _make_acl_config(
    roles: dict | None = None,
    user_roles: dict | None = None,
    deny_lists: dict | None = None,
    enforcement_on: bool = True,
) -> dict:
    """Build an ACL config dict for tenant config."""
    acl = {
        "enforcement_on": enforcement_on,
    }
    if roles:
        acl["roles"] = roles
    if user_roles:
        acl["user_roles"] = user_roles
    if deny_lists:
        acl["deny_lists"] = deny_lists
    return acl


def _make_service_with_acl(tmp_path, acl_config: dict, users: list[str]) -> MemoryService:
    """Build a single-tenant MemoryService with the given ACL."""
    config = {
        "tenants": {
            "default": {
                "database_filename": "test_acl.duckdb",
                "graph_dirname": "test_acl_kuzu",
                "allowed_user_ids": users,
                "config": {
                    "acl": acl_config,
                },
            },
        },
    }
    (tmp_path / "hybrid_memory.json").write_text(
        json.dumps(config), encoding="utf-8"
    )
    return MemoryService(tmp_path)


def _remember(svc, user_id, content, client_scope=None, doc_class=None):
    """Write a memory through the service dispatch."""
    args = {"category": "personal_fact", "content": content}
    if client_scope:
        args["client_scope"] = client_scope
    if doc_class:
        args["doc_class"] = doc_class
    return svc.dispatch({
        "component": "store", "method": "remember",
        "user_id": user_id, "args": args,
    })


def _search(svc, user_id, query, limit=10):
    """Search through the service dispatch."""
    return svc.dispatch({
        "component": "store", "method": "search",
        "user_id": user_id,
        "args": {"query": query, "limit": limit},
    })


# ---------------------------------------------------------------------------
# Missing ACL config = open store (backward compat)
# ---------------------------------------------------------------------------

class TestMissingACLCompat:
    """Missing ACL config has zero behavior change."""

    def test_no_acl_config_is_open_store(self, tmp_path):
        """A tenant without an acl key has an open-store ACL."""
        config = {
            "tenants": {
                "default": {
                    "database_filename": "test_noacl.duckdb",
                    "graph_dirname": "test_noacl_kuzu",
                    "allowed_user_ids": ["user-a"],
                },
            },
        }
        (tmp_path / "hybrid_memory.json").write_text(
            json.dumps(config), encoding="utf-8"
        )
        svc = MemoryService(tmp_path)
        tenant = svc._tenants["default"]
        assert tenant.acl.is_open_store is True
        # Search returns all records (no filtering).
        _remember(svc, "user-a", "User likes apples")
        results = _search(svc, "user-a", "apples")
        assert len(results) >= 1

    def test_empty_acl_config_is_open_store(self, tmp_path):
        """An empty acl dict means open store."""
        svc = _make_service_with_acl(tmp_path, {}, ["user-a"])
        tenant = svc._tenants["default"]
        assert tenant.acl.is_open_store is True

    def test_acl_attached_to_store(self, tmp_path):
        """The store has _acl_config set."""
        svc = _make_service_with_acl(tmp_path, _make_acl_config(
            roles={"staff": {"client_scopes": ["acme"]}},
            user_roles={"user-a": "staff"},
        ), ["user-a", "user-b"])
        assert svc._tenants["default"].store._acl_config is not None
        assert not svc._tenants["default"].store._acl_config.is_open_store


# ---------------------------------------------------------------------------
# Two-user isolation: user A cannot see user B's client-scoped records
# ---------------------------------------------------------------------------

class TestTwoUserIsolation:
    """Two users in one tenant with different client scopes."""

    def _make_two_user_service(self, tmp_path) -> MemoryService:
        """Build a service with two users: alice (acme scope) and bob (beta scope)."""
        acl = _make_acl_config(
            roles={
                "acme_staff": {"client_scopes": ["acme"]},
                "beta_staff": {"client_scopes": ["beta"]},
            },
            user_roles={
                "alice": "acme_staff",
                "bob": "beta_staff",
            },
        )
        return _make_service_with_acl(tmp_path, acl, ["alice", "bob"])

    def test_alice_cannot_see_bob_client_scope(self, tmp_path):
        """Alice (acme) cannot retrieve records in bob's beta scope."""
        svc = self._make_two_user_service(tmp_path)
        # Bob writes a record in the beta scope.
        _remember(svc, "bob", "Beta Corp revenue is $10M", client_scope="beta")
        # Alice searches for it.
        results = _search(svc, "alice", "Beta Corp revenue")
        # Alice should see zero results — the record is in beta scope,
        # and alice only has acme scope.
        assert len(results) == 0, (
            f"Alice should not see beta-scoped records, got {len(results)}"
        )

    def test_bob_cannot_see_alice_client_scope(self, tmp_path):
        """Bob (beta) cannot retrieve records in alice's acme scope."""
        svc = self._make_two_user_service(tmp_path)
        _remember(svc, "alice", "Acme Corp revenue is $5M", client_scope="acme")
        results = _search(svc, "bob", "Acme Corp revenue")
        assert len(results) == 0

    def test_alice_can_see_own_scope(self, tmp_path):
        """Alice can see records in her own acme scope."""
        svc = self._make_two_user_service(tmp_path)
        _remember(svc, "alice", "Acme Corp revenue is $5M", client_scope="acme")
        results = _search(svc, "alice", "Acme Corp revenue")
        assert len(results) >= 1

    def test_wheel_user_sees_all_scopes(self, tmp_path):
        """A wheel user (principal) can see all client scopes."""
        acl = _make_acl_config(
            roles={
                "acme_staff": {"client_scopes": ["acme"]},
                "principal": {"wheel": True},
            },
            user_roles={
                "alice": "acme_staff",
                "carol": "principal",
            },
        )
        svc = _make_service_with_acl(tmp_path, acl, ["alice", "carol"])
        # Carol writes records in different client scopes — the ACL
        # filter checks client_scope, not user_scope. Carol (wheel)
        # should see all her own records regardless of client_scope.
        _remember(svc, "carol", "Acme Corp revenue is $5M", client_scope="acme")
        _remember(svc, "carol", "Beta Corp revenue is $10M", client_scope="beta")
        # Carol (wheel/principal) should see both.
        results = _search(svc, "carol", "revenue")
        assert len(results) >= 2, (
            f"Wheel user should see all scopes, got {len(results)}"
        )

    def test_fetch_by_id_filters_by_acl(self, tmp_path):
        """Fetch-by-ID also enforces ACL — alice can't fetch bob's record."""
        svc = self._make_two_user_service(tmp_path)
        rec = _remember(svc, "bob", "Beta Corp secret data", client_scope="beta")
        mid = rec["memory_id"]
        # Alice tries to fetch it.
        results = svc.dispatch({
            "component": "store", "method": "get_memories_by_ids",
            "user_id": "alice",
            "args": {"memory_ids": [mid]},
        })
        assert len(results) == 0, (
            f"Alice should not fetch beta-scoped records, got {len(results)}"
        )
        # Bob can fetch it.
        results = svc.dispatch({
            "component": "store", "method": "get_memories_by_ids",
            "user_id": "bob",
            "args": {"memory_ids": [mid]},
        })
        assert len(results) == 1


# ---------------------------------------------------------------------------
# Deny precedence: deny > allow > wheel
# ---------------------------------------------------------------------------

class TestDenyPrecedence:
    """Deny beats allow; deny beats wheel; unassigned users fail closed."""

    def test_deny_beats_allow(self, tmp_path):
        """A deny entry on a user's own scope hides the record."""
        acl = _make_acl_config(
            roles={"staff": {"client_scopes": ["acme"]}},
            user_roles={"alice": "staff"},
            deny_lists={"alice": [{"client_scope": "acme"}]},
        )
        svc = _make_service_with_acl(tmp_path, acl, ["alice"])
        _remember(svc, "alice", "Acme secret", client_scope="acme")
        results = _search(svc, "alice", "Acme secret")
        assert len(results) == 0, "Deny should hide the record from alice"

    def test_deny_beats_wheel(self, tmp_path):
        """A deny entry on a wheel user hides the record."""
        acl = _make_acl_config(
            roles={"principal": {"wheel": True}},
            user_roles={"carol": "principal"},
            deny_lists={"carol": [{"doc_class": "practice-internal"}]},
        )
        svc = _make_service_with_acl(tmp_path, acl, ["carol"])
        _remember(svc, "carol", "Partner profit share", client_scope="acme",
                  doc_class="practice-internal")
        results = _search(svc, "carol", "profit share")
        assert len(results) == 0, "Deny should hide practice-internal from carol"

    def test_unassigned_user_fails_closed(self, tmp_path):
        """An unassigned user under enforcement gets deny-all."""
        acl = _make_acl_config(
            roles={"staff": {"client_scopes": ["acme"]}},
            user_roles={"alice": "staff"},
            # bob is NOT in user_roles
        )
        svc = _make_service_with_acl(tmp_path, acl, ["alice", "bob"])
        _remember(svc, "alice", "Acme data", client_scope="acme")
        results = _search(svc, "bob", "Acme data")
        assert len(results) == 0, "Unassigned user should see nothing"

    def test_practice_internal_only_for_wheel(self, tmp_path):
        """practice-internal doc_class is only visible to wheel users."""
        acl = _make_acl_config(
            roles={
                "staff": {"client_scopes": ["acme"]},
                "principal": {"wheel": True},
            },
            user_roles={
                "alice": "staff",
                "carol": "principal",
            },
        )
        svc = _make_service_with_acl(tmp_path, acl, ["alice", "carol"])
        _remember(svc, "carol", "Payroll data", client_scope="acme",
                  doc_class="practice-internal")
        # Alice (staff) cannot see it.
        results = _search(svc, "alice", "Payroll data")
        assert len(results) == 0
        # Carol (principal/wheel) can see it.
        results = _search(svc, "carol", "Payroll data")
        assert len(results) >= 1


# ---------------------------------------------------------------------------
# Audit rows: every query and denial creates an audit row
# ---------------------------------------------------------------------------

class TestAccessAudit:
    """Every query and denial creates an audit row."""

    def test_search_writes_audit_row(self, tmp_path):
        """A search through the RPC path writes an audit row."""
        acl = _make_acl_config(
            roles={"staff": {"client_scopes": ["acme"]}},
            user_roles={"alice": "staff"},
        )
        svc = _make_service_with_acl(tmp_path, acl, ["alice"])
        _remember(svc, "alice", "Acme data", client_scope="acme")
        _search(svc, "alice", "Acme data")
        # Check the audit table.
        store = svc._tenants["default"].store
        with store._state.lock:
            rows = store.connection.execute(
                "SELECT user_id, query_text, granted_count, denied_count, tenant FROM access_audit"
            ).fetchall()
        assert len(rows) >= 1
        assert rows[-1][0] == "alice"
        assert rows[-1][4] == "default"  # tenant name
        assert rows[-1][2] >= 1  # granted_count

    def test_denial_writes_audit_row(self, tmp_path):
        """A denial (user sees fewer results due to ACL) writes an audit row."""
        acl = _make_acl_config(
            roles={
                "acme_staff": {"client_scopes": ["acme"]},
                "beta_staff": {"client_scopes": ["beta"]},
            },
            user_roles={
                "alice": "acme_staff",
                "bob": "beta_staff",
            },
        )
        svc = _make_service_with_acl(tmp_path, acl, ["alice", "bob"])
        # Alice writes a record in the beta scope (she can write it,
        # but the ACL filter should deny her from seeing it on search
        # since her mask is acme-only).
        _remember(svc, "alice", "Beta secret data", client_scope="beta")
        # Alice searches — the record is in her user_scope but outside
        # her ACL client_scope mask, so it should be denied.
        _search(svc, "alice", "Beta secret")
        store = svc._tenants["default"].store
        with store._state.lock:
            rows = store.connection.execute(
                "SELECT user_id, denied_count, excluded FROM access_audit WHERE user_id = 'alice'"
            ).fetchall()
        assert len(rows) >= 1
        assert rows[-1][1] >= 1  # denied_count
        assert rows[-1][2] is True  # excluded


# ---------------------------------------------------------------------------
# Audit export: restricted to authorized principals
# ---------------------------------------------------------------------------

class TestAuditExport:
    """Authorized audit export works; unauthorized export is rejected."""

    def test_wheel_user_can_export(self, tmp_path):
        """A wheel user (principal) can export the audit log."""
        acl = _make_acl_config(
            roles={
                "staff": {"client_scopes": ["acme"]},
                "principal": {"wheel": True},
            },
            user_roles={
                "alice": "staff",
                "carol": "principal",
            },
        )
        svc = _make_service_with_acl(tmp_path, acl, ["alice", "carol"])
        _remember(svc, "alice", "Acme data", client_scope="acme")
        _search(svc, "alice", "Acme data")
        # Carol (wheel) exports.
        result = svc.dispatch({
            "component": "store", "method": "export_access_audit",
            "user_id": "carol",
            "args": {"limit": 100, "format": "jsonl"},
        })
        assert result is not None
        assert isinstance(result, str)
        # Should contain at least one audit row.
        assert len(result) > 0

    def test_staff_user_cannot_export(self, tmp_path):
        """A non-wheel user (staff) cannot export the audit log."""
        acl = _make_acl_config(
            roles={
                "staff": {"client_scopes": ["acme"]},
                "principal": {"wheel": True},
            },
            user_roles={
                "alice": "staff",
                "carol": "principal",
            },
        )
        svc = _make_service_with_acl(tmp_path, acl, ["alice", "carol"])
        with pytest.raises(PermissionError):
            svc.dispatch({
                "component": "store", "method": "export_access_audit",
                "user_id": "alice",
                "args": {"limit": 100, "format": "jsonl"},
            })

    def test_unassigned_user_cannot_export(self, tmp_path):
        """An unassigned user cannot export the audit log."""
        acl = _make_acl_config(
            roles={"staff": {"client_scopes": ["acme"]}},
            user_roles={"alice": "staff"},
        )
        svc = _make_service_with_acl(tmp_path, acl, ["alice", "bob"])
        with pytest.raises(PermissionError):
            svc.dispatch({
                "component": "store", "method": "export_access_audit",
                "user_id": "bob",
                "args": {},
            })

    def test_open_store_export_allowed(self, tmp_path):
        """In an open store (no ACL), any user can export."""
        config = {
            "tenants": {
                "default": {
                    "database_filename": "test_open.duckdb",
                    "graph_dirname": "test_open_kuzu",
                    "allowed_user_ids": ["user-a"],
                },
            },
        }
        (tmp_path / "hybrid_memory.json").write_text(
            json.dumps(config), encoding="utf-8"
        )
        svc = MemoryService(tmp_path)
        _remember(svc, "user-a", "Test data")
        _search(svc, "user-a", "Test data")
        result = svc.dispatch({
            "component": "store", "method": "export_access_audit",
            "user_id": "user-a",
            "args": {},
        })
        assert result is not None


# ---------------------------------------------------------------------------
# get_status reports ACL status
# ---------------------------------------------------------------------------

class TestGetStatusACL:
    """get_status includes per-tenant ACL status."""

    def test_get_status_includes_acl_status(self, tmp_path):
        """get_status returns tenant_acl_status with enforcement info."""
        acl = _make_acl_config(
            roles={"staff": {"client_scopes": ["acme"]}},
            user_roles={"alice": "staff"},
        )
        svc = _make_service_with_acl(tmp_path, acl, ["alice"])
        status = svc.dispatch({"method": "get_status"})
        assert "tenant_acl_status" in status
        assert "default" in status["tenant_acl_status"]
        acl_status = status["tenant_acl_status"]["default"]
        assert acl_status["enforcement_on"] is True
        assert acl_status["is_open_store"] is False
        assert acl_status["role_count"] == 1
        assert acl_status["user_count"] == 1

    def test_get_status_open_store_acl(self, tmp_path):
        """get_status shows open store when no ACL is configured."""
        config = {
            "tenants": {
                "default": {
                    "database_filename": "test.duckdb",
                    "graph_dirname": "test_kuzu",
                    "allowed_user_ids": ["user-a"],
                },
            },
        }
        (tmp_path / "hybrid_memory.json").write_text(
            json.dumps(config), encoding="utf-8"
        )
        svc = MemoryService(tmp_path)
        status = svc.dispatch({"method": "get_status"})
        acl_status = status["tenant_acl_status"]["default"]
        assert acl_status["is_open_store"] is True
        assert acl_status["enforcement_on"] is False

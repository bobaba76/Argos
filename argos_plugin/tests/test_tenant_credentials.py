"""Tests for per-tenant credential binding and RPC identity (#129).

Tests cover:
  - A client cannot change user_id to access another tenant.
  - A credential for tenant A cannot read/write/backup tenant B.
  - An identity mismatch is rejected before dispatch.
  - Backup/list/admin operations follow the same authorization boundary.
  - Spoofed user, spoofed tenant, stale credential, missing credential.
  - Legacy trusted-local mode remains backward compatible.

All deterministic, no LLM calls.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_plugin_dir = Path(__file__).resolve().parent.parent
if str(_plugin_dir) not in sys.path:
    sys.path.insert(0, str(_plugin_dir))

from memory_service import MemoryService


# -- Helpers -----------------------------------------------------------------

_TENANT_A_TOKEN = "tenant-a-secret-token-aaaa"
_TENANT_B_TOKEN = "tenant-b-secret-token-bbbb"
_STALE_TOKEN = "this-token-was-revoked"


def _make_credential_config() -> dict:
    """Build a two-tenant config with per-tenant credentials."""
    return {
        "tenants": {
            "alpha": {
                "database_filename": "alpha_cred.duckdb",
                "graph_dirname": "alpha_cred_kuzu",
                "credentials": [
                    {"token": _TENANT_A_TOKEN, "user_id": "alice"},
                    {"token": _TENANT_A_TOKEN + "-bob", "user_id": "bob"},
                ],
                "config": {
                    "external_sources_require_confirmation": "true",
                },
            },
            "beta": {
                "database_filename": "beta_cred.duckdb",
                "graph_dirname": "beta_cred_kuzu",
                "credentials": [
                    {"token": _TENANT_B_TOKEN, "user_id": "carol"},
                ],
                "config": {
                    "external_sources_require_confirmation": "true",
                },
            },
        },
    }


def _make_credential_service(tmp_path) -> MemoryService:
    """Build a two-tenant MemoryService with credentials."""
    config = _make_credential_config()
    (tmp_path / "hybrid_memory.json").write_text(
        json.dumps(config), encoding="utf-8"
    )
    return MemoryService(tmp_path)


def _make_legacy_service(tmp_path) -> MemoryService:
    """Build a legacy single-tenant service without credentials."""
    config = {
        "tenants": {
            "default": {
                "database_filename": "legacy.duckdb",
                "graph_dirname": "legacy_kuzu",
                "allowed_user_ids": ["user-a", "user-b"],
            },
        },
    }
    (tmp_path / "hybrid_memory.json").write_text(
        json.dumps(config), encoding="utf-8"
    )
    return MemoryService(tmp_path)


def _dispatch(svc, credential=None, user_id=None, component="store",
              method="count", args=None):
    """Build a dispatch request with optional credential."""
    req = {"component": component, "method": method, "args": args or {}}
    if credential is not None:
        req["credential"] = credential
    if user_id is not None:
        req["user_id"] = user_id
    return svc.dispatch(req)


def _remember(svc, credential, content, client_scope=None):
    """Write a memory using a credential."""
    args = {"category": "personal_fact", "content": content}
    if client_scope:
        args["client_scope"] = client_scope
    return _dispatch(svc, credential=credential, method="remember", args=args)


def _search(svc, credential, query, limit=10):
    """Search using a credential."""
    return _dispatch(svc, credential=credential, method="search",
                     args={"query": query, "limit": limit})


# ---------------------------------------------------------------------------
# Credential mode: server derives identity
# ---------------------------------------------------------------------------

class TestCredentialModeBasic:
    """Credential mode basics — server derives identity from credential."""

    def test_credential_mode_enabled(self, tmp_path):
        """The service reports multi-user auth mode."""
        svc = _make_credential_service(tmp_path)
        assert svc._credential_mode is True
        assert len(svc._credential_map) == 3  # alice, bob, carol

    def test_valid_credential_works(self, tmp_path):
        """A valid credential allows dispatch."""
        svc = _make_credential_service(tmp_path)
        result = _dispatch(svc, credential=_TENANT_A_TOKEN, method="count")
        assert result is not None

    def test_get_status_reports_auth_mode(self, tmp_path):
        """get_status shows auth_mode=multi-user and credential_count."""
        svc = _make_credential_service(tmp_path)
        status = svc.dispatch({"method": "get_status"})
        assert status["auth_mode"] == "multi-user"
        assert status["credential_count"] == 3

    def test_credential_derives_correct_tenant(self, tmp_path):
        """Alice's credential routes to the alpha tenant."""
        svc = _make_credential_service(tmp_path)
        # Write with alice's credential, then count.
        _remember(svc, _TENANT_A_TOKEN, "Alice works at Acme")
        count_a = _dispatch(svc, credential=_TENANT_A_TOKEN, method="count")
        # Carol's credential should see a different (empty) tenant.
        count_c = _dispatch(svc, credential=_TENANT_B_TOKEN, method="count")
        assert count_a >= 1
        assert count_c == 0


# ---------------------------------------------------------------------------
# Missing credential
# ---------------------------------------------------------------------------

class TestMissingCredential:
    """In credential mode, a missing credential is rejected."""

    def test_no_credential_rejected(self, tmp_path):
        """A request without a credential is rejected in multi-user mode."""
        svc = _make_credential_service(tmp_path)
        with pytest.raises(PermissionError, match="Credential required"):
            _dispatch(svc, credential=None, method="count")

    def test_empty_credential_rejected(self, tmp_path):
        """An empty string credential is rejected."""
        svc = _make_credential_service(tmp_path)
        with pytest.raises(PermissionError, match="Credential required"):
            _dispatch(svc, credential="", method="count")


# ---------------------------------------------------------------------------
# Stale / invalid credential
# ---------------------------------------------------------------------------

class TestStaleCredential:
    """A stale or invalid credential is rejected."""

    def test_invalid_credential_rejected(self, tmp_path):
        """A completely invalid token is rejected."""
        svc = _make_credential_service(tmp_path)
        with pytest.raises(PermissionError, match="invalid or revoked"):
            _dispatch(svc, credential="not-a-real-token", method="count")

    def test_stale_credential_rejected(self, tmp_path):
        """A revoked/stale token is rejected."""
        svc = _make_credential_service(tmp_path)
        with pytest.raises(PermissionError, match="invalid or revoked"):
            _dispatch(svc, credential=_STALE_TOKEN, method="count")


# ---------------------------------------------------------------------------
# Spoofed user_id
# ---------------------------------------------------------------------------

class TestSpoofedUserId:
    """A client cannot change user_id to access another tenant."""

    def test_user_id_mismatch_rejected(self, tmp_path):
        """If the client supplies a user_id that doesn't match the
        credential's bound user_id, it's rejected."""
        svc = _make_credential_service(tmp_path)
        with pytest.raises(PermissionError, match="Identity mismatch"):
            _dispatch(svc, credential=_TENANT_A_TOKEN,
                      user_id="carol", method="count")

    def test_user_id_match_allowed(self, tmp_path):
        """If the client supplies the correct user_id, it's allowed."""
        svc = _make_credential_service(tmp_path)
        result = _dispatch(svc, credential=_TENANT_A_TOKEN,
                           user_id="alice", method="count")
        assert result is not None

    def test_no_user_id_with_credential_allowed(self, tmp_path):
        """If the client omits user_id, the server derives it."""
        svc = _make_credential_service(tmp_path)
        result = _dispatch(svc, credential=_TENANT_A_TOKEN,
                           user_id=None, method="count")
        assert result is not None


# ---------------------------------------------------------------------------
# Spoofed tenant: credential for tenant A cannot access tenant B
# ---------------------------------------------------------------------------

class TestTenantIsolation:
    """A credential for tenant A cannot read/write tenant B."""

    def test_alpha_credential_cannot_see_beta_data(self, tmp_path):
        """Alice's credential (alpha) cannot see carol's data (beta)."""
        svc = _make_credential_service(tmp_path)
        # Carol writes in beta.
        _remember(svc, _TENANT_B_TOKEN, "Carol works at Beta Corp")
        # Alice searches from alpha — should see nothing.
        results = _search(svc, _TENANT_A_TOKEN, "Beta Corp")
        assert len(results) == 0

    def test_beta_credential_cannot_see_alpha_data(self, tmp_path):
        """Carol's credential (beta) cannot see alice's data (alpha)."""
        svc = _make_credential_service(tmp_path)
        _remember(svc, _TENANT_A_TOKEN, "Alice works at Alpha Corp")
        results = _search(svc, _TENANT_B_TOKEN, "Alpha Corp")
        assert len(results) == 0

    def test_alpha_credential_count_isolated(self, tmp_path):
        """Alice's count only sees alpha's records."""
        svc = _make_credential_service(tmp_path)
        _remember(svc, _TENANT_A_TOKEN, "Alice lives at 123 Acme Street")
        _remember(svc, _TENANT_A_TOKEN, "Alice drives a blue Toyota Camry")
        _remember(svc, _TENANT_B_TOKEN, "Carol works at Beta Corporation downtown")
        alpha_count = _dispatch(svc, credential=_TENANT_A_TOKEN, method="count")
        beta_count = _dispatch(svc, credential=_TENANT_B_TOKEN, method="count")
        assert alpha_count == 2
        assert beta_count == 1

    def test_backup_scoped_to_credential_tenant(self, tmp_path):
        """In credential mode, backup is scoped to the credential's tenant."""
        svc = _make_credential_service(tmp_path)
        _remember(svc, _TENANT_A_TOKEN, "Alice fact 1")
        # Alice backs up — should be scoped to alpha.
        result = svc.dispatch({
            "method": "backup",
            "credential": _TENANT_A_TOKEN,
            "args": {"dst_root": str(tmp_path / "backups")},
        })
        assert result is not None
        # The backup should mention alpha as the tenant.
        assert "alpha" in str(result.get("tenant", "")) or result.get("tenant") == "alpha"

    def test_backup_without_credential_rejected(self, tmp_path):
        """In credential mode, backup without a credential is rejected."""
        svc = _make_credential_service(tmp_path)
        with pytest.raises(PermissionError, match="Credential required"):
            svc.dispatch({
                "method": "backup",
                "args": {"dst_root": str(tmp_path / "backups")},
            })


# ---------------------------------------------------------------------------
# Multiple users in same tenant
# ---------------------------------------------------------------------------

class TestMultipleUsersSameTenant:
    """Multiple credentials can map to the same tenant with different users."""

    def test_alice_and_bob_same_tenant(self, tmp_path):
        """Alice and bob are both in the alpha tenant, each sees their own
        records (user_scope defense-in-depth)."""
        svc = _make_credential_service(tmp_path)
        # Both can write — use distinct content to avoid dedup.
        _remember(svc, _TENANT_A_TOKEN, "Alice lives at 123 Acme Street")
        _remember(svc, _TENANT_A_TOKEN + "-bob", "Bob drives a red Ford Mustang")
        # Each sees their own records (user_scope filtering).
        alice_count = _dispatch(svc, credential=_TENANT_A_TOKEN, method="count")
        bob_count = _dispatch(svc, credential=_TENANT_A_TOKEN + "-bob", method="count")
        assert alice_count == 1
        assert bob_count == 1
        # But they're in the same tenant — carol (beta) sees zero.
        carol_count = _dispatch(svc, credential=_TENANT_B_TOKEN, method="count")
        assert carol_count == 0

    def test_bob_cannot_spoof_alice(self, tmp_path):
        """Bob cannot use his credential and claim to be alice."""
        svc = _make_credential_service(tmp_path)
        with pytest.raises(PermissionError, match="Identity mismatch"):
            _dispatch(svc, credential=_TENANT_A_TOKEN + "-bob",
                      user_id="alice", method="count")


# ---------------------------------------------------------------------------
# Legacy trusted-local mode (backward compat)
# ---------------------------------------------------------------------------

class TestTrustedLocalMode:
    """Legacy single-token mode remains backward compatible."""

    def test_legacy_mode_no_credentials(self, tmp_path):
        """A config without credentials is in trusted-local mode."""
        svc = _make_legacy_service(tmp_path)
        assert svc._credential_mode is False
        assert len(svc._credential_map) == 0

    def test_legacy_mode_uses_client_user_id(self, tmp_path):
        """In trusted-local mode, the client supplies user_id."""
        svc = _make_legacy_service(tmp_path)
        result = _dispatch(svc, credential=None, user_id="user-a", method="count")
        assert result is not None

    def test_legacy_get_status_reports_trusted_local(self, tmp_path):
        """get_status shows auth_mode=trusted-local."""
        svc = _make_legacy_service(tmp_path)
        status = svc.dispatch({"method": "get_status"})
        assert status["auth_mode"] == "trusted-local"
        assert status["credential_count"] == 0

    def test_legacy_credential_ignored(self, tmp_path):
        """In trusted-local mode, a credential field is ignored —
        the client-supplied user_id is used."""
        svc = _make_legacy_service(tmp_path)
        result = _dispatch(svc, credential="some-token",
                           user_id="user-a", method="count")
        assert result is not None

    def test_legacy_strict_routing_still_works(self, tmp_path):
        """Strict routing (#87) still rejects unknown user_ids."""
        svc = _make_legacy_service(tmp_path)
        with pytest.raises(PermissionError):
            _dispatch(svc, credential=None, user_id="unknown-user", method="count")


# ---------------------------------------------------------------------------
# Credential config validation
# ---------------------------------------------------------------------------

class TestCredentialConfigValidation:
    """Credential config is validated at startup."""

    def test_duplicate_token_across_tenants_rejected(self, tmp_path):
        """The same token in two tenants is a config error."""
        config = {
            "tenants": {
                "alpha": {
                    "database_filename": "a.duckdb",
                    "graph_dirname": "a_kuzu",
                    "credentials": [{"token": "shared-token", "user_id": "alice"}],
                },
                "beta": {
                    "database_filename": "b.duckdb",
                    "graph_dirname": "b_kuzu",
                    "credentials": [{"token": "shared-token", "user_id": "carol"}],
                },
            },
        }
        (tmp_path / "hybrid_memory.json").write_text(
            json.dumps(config), encoding="utf-8"
        )
        with pytest.raises(ValueError, match="credential conflict"):
            MemoryService(tmp_path)

    def test_empty_credentials_list_ignored(self, tmp_path):
        """An empty credentials list doesn't enable credential mode."""
        config = {
            "tenants": {
                "default": {
                    "database_filename": "test.duckdb",
                    "graph_dirname": "test_kuzu",
                    "credentials": [],
                },
            },
        }
        (tmp_path / "hybrid_memory.json").write_text(
            json.dumps(config), encoding="utf-8"
        )
        svc = MemoryService(tmp_path)
        assert svc._credential_mode is False

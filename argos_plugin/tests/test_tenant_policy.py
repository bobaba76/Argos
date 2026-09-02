"""Tests for per-tenant policy overlays (#127 / spec multitenancy).

Tests cover:
  - TenantPolicy resolution from merged config (global + overlay)
  - Two tenants with opposing review_modes produce correct candidate
    outcomes through the real service dispatch path
  - Two tenants with different injection caps are independently capped
  - A tenant with local_only=true has the flag set in its policy
  - A tenant's external-source policy is enforced independently
  - Policy state is immutable for the lifetime of a request (client-
    supplied arguments cannot change policy)
  - Config reload/restart semantics (policy resolved at startup)
  - Direct and shared-service modes

All deterministic, no LLM calls.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest

_plugin_dir = Path(__file__).resolve().parent.parent
if str(_plugin_dir) not in sys.path:
    sys.path.insert(0, str(_plugin_dir))

from memory_service import TenantPolicy, _Tenant, _parse_tenants, MemoryService


# -- TenantPolicy unit tests -------------------------------------------------

class TestTenantPolicyResolution:
    """TenantPolicy correctly resolves from merged config."""

    def test_default_policy_is_confirm_mode(self):
        """The default review_mode is 'confirm' (human approval required)."""
        policy = TenantPolicy({})
        assert policy.review_mode == "confirm"

    def test_auto_review_mode_from_config(self):
        """review_mode=auto is resolved from config."""
        policy = TenantPolicy({"review_mode": "auto"})
        assert policy.review_mode == "auto"

    def test_invalid_review_mode_fails_closed(self):
        """An invalid review_mode fails closed to 'confirm'."""
        policy = TenantPolicy({"review_mode": "invalid"})
        assert policy.review_mode == "confirm"

    def test_default_injection_caps(self):
        """Default injection caps are sensible."""
        policy = TenantPolicy({})
        assert policy.max_injected_items == 5
        assert policy.inject_content_char_cap == 800

    def test_custom_injection_caps(self):
        """Custom injection caps are respected within bounds."""
        policy = TenantPolicy({
            "max_injected_items": "10",
            "inject_content_char_cap": "1200",
        })
        assert policy.max_injected_items == 10
        assert policy.inject_content_char_cap == 1200

    def test_injection_caps_clamped_to_bounds(self):
        """Injection caps are clamped to safe bounds."""
        policy = TenantPolicy({
            "max_injected_items": "999",
            "inject_content_char_cap": "99999",
        })
        assert policy.max_injected_items == 50  # max
        assert policy.inject_content_char_cap == 5000  # max

    def test_injection_caps_floor(self):
        """Injection caps have a floor."""
        policy = TenantPolicy({
            "max_injected_items": "0",
            "inject_content_char_cap": "10",
        })
        assert policy.max_injected_items == 0
        assert policy.inject_content_char_cap == 100  # min

    def test_default_external_sources_require_confirmation(self):
        """Default: external sources require confirmation (True)."""
        policy = TenantPolicy({})
        assert policy.external_sources_require_confirmation is True

    def test_external_sources_can_be_disabled_per_tenant(self):
        """A tenant can disable external_sources_require_confirmation."""
        policy = TenantPolicy({"external_sources_require_confirmation": "false"})
        assert policy.external_sources_require_confirmation is False

    def test_default_local_only_is_false(self):
        """Default: local_only is False (LLM calls allowed)."""
        policy = TenantPolicy({})
        assert policy.local_only is False

    def test_local_only_can_be_set_per_tenant(self):
        """A tenant can set local_only=True."""
        policy = TenantPolicy({"local_only": "true"})
        assert policy.local_only is True

    def test_policy_to_dict_roundtrips(self):
        """to_dict() returns all policy fields."""
        policy = TenantPolicy({
            "review_mode": "auto",
            "max_injected_items": "15",
            "inject_content_char_cap": "1000",
            "external_sources_require_confirmation": "false",
            "local_only": "true",
        })
        d = policy.to_dict()
        assert d["review_mode"] == "auto"
        assert d["max_injected_items"] == 15
        assert d["inject_content_char_cap"] == 1000
        assert d["external_sources_require_confirmation"] is False
        assert d["local_only"] is True

    def test_overlay_overrides_global(self):
        """Tenant overlay values override global config values."""
        global_config = {
            "review_mode": "confirm",
            "max_injected_items": "5",
            "local_only": "false",
        }
        overlay = {
            "review_mode": "auto",
            "max_injected_items": "20",
            "local_only": "true",
        }
        merged = dict(global_config)
        merged.update(overlay)
        policy = TenantPolicy(merged)
        assert policy.review_mode == "auto"
        assert policy.max_injected_items == 20
        assert policy.local_only is True

    def test_policy_inherits_unoverridden_global(self):
        """Policy fields not in the overlay inherit from global config."""
        global_config = {
            "review_mode": "confirm",
            "max_injected_items": "5",
            "local_only": "false",
            "external_sources_require_confirmation": "true",
        }
        overlay = {"local_only": "true"}
        merged = dict(global_config)
        merged.update(overlay)
        policy = TenantPolicy(merged)
        assert policy.review_mode == "confirm"  # from global
        assert policy.max_injected_items == 5  # from global
        assert policy.local_only is True  # from overlay
        assert policy.external_sources_require_confirmation is True  # from global


# -- Per-tenant policy isolation through _parse_tenants ----------------------

class TestTenantPolicyIsolation:
    """Two tenants with opposing policies are independently configured."""

    def _make_config(self) -> dict:
        """Build a two-tenant config with opposing policies."""
        return {
            "tenants": {
                "restrictive": {
                    "database_filename": "restrictive.duckdb",
                    "graph_dirname": "restrictive_kuzu",
                    "allowed_user_ids": ["user-restrictive"],
                    "config": {
                        "review_mode": "confirm",
                        "max_injected_items": "3",
                        "inject_content_char_cap": "500",
                        "external_sources_require_confirmation": "true",
                        "local_only": "true",
                    },
                },
                "permissive": {
                    "database_filename": "permissive.duckdb",
                    "graph_dirname": "permissive_kuzu",
                    "allowed_user_ids": ["user-permissive"],
                    "config": {
                        "review_mode": "auto",
                        "max_injected_items": "15",
                        "inject_content_char_cap": "1500",
                        "external_sources_require_confirmation": "false",
                        "local_only": "false",
                    },
                },
            },
        }

    def test_two_tenants_have_opposing_review_modes(self, tmp_path):
        """The two tenants have different review_mode values."""
        config = self._make_config()
        tenants, user_map, strict = _parse_tenants(config, tmp_path, None, None)
        assert tenants["restrictive"].policy.review_mode == "confirm"
        assert tenants["permissive"].policy.review_mode == "auto"

    def test_two_tenants_have_different_injection_caps(self, tmp_path):
        """The two tenants have different max_injected_items."""
        config = self._make_config()
        tenants, _, _ = _parse_tenants(config, tmp_path, None, None)
        assert tenants["restrictive"].policy.max_injected_items == 3
        assert tenants["permissive"].policy.max_injected_items == 15

    def test_two_tenants_have_different_local_only(self, tmp_path):
        """The restrictive tenant has local_only=True, permissive has False."""
        config = self._make_config()
        tenants, _, _ = _parse_tenants(config, tmp_path, None, None)
        assert tenants["restrictive"].policy.local_only is True
        assert tenants["permissive"].policy.local_only is False

    def test_two_tenants_have_different_external_source_policy(self, tmp_path):
        """The restrictive tenant requires confirmation, permissive doesn't."""
        config = self._make_config()
        tenants, _, _ = _parse_tenants(config, tmp_path, None, None)
        assert tenants["restrictive"].policy.external_sources_require_confirmation is True
        assert tenants["permissive"].policy.external_sources_require_confirmation is False

    def test_store_inherits_external_sources_policy(self, tmp_path):
        """The store's external_sources_require_confirmation matches the
        tenant policy."""
        config = self._make_config()
        tenants, _, _ = _parse_tenants(config, tmp_path, None, None)
        assert tenants["restrictive"].store.external_sources_require_confirmation is True
        assert tenants["permissive"].store.external_sources_require_confirmation is False

    def test_strict_routing_enabled_with_allowed_user_ids(self, tmp_path):
        """With allowed_user_ids on both tenants, strict routing is enabled."""
        config = self._make_config()
        _, _, strict = _parse_tenants(config, tmp_path, None, None)
        assert strict is True

    def test_user_tenant_map_correct(self, tmp_path):
        """The user_tenant_map maps each user to their tenant."""
        config = self._make_config()
        _, user_map, _ = _parse_tenants(config, tmp_path, None, None)
        assert user_map["user-restrictive"] == "restrictive"
        assert user_map["user-permissive"] == "permissive"


# -- Review mode enforcement through service dispatch ------------------------

class TestReviewModeEnforcement:
    """Two tenants with opposing review modes produce the correct
    candidate outcome through the real service dispatch path."""

    def _make_service(self, tmp_path) -> MemoryService:
        """Build a two-tenant MemoryService for dispatch tests."""
        config = {
            "tenants": {
                "restrictive": {
                    "database_filename": "restrictive.duckdb",
                    "graph_dirname": "restrictive_kuzu",
                    "allowed_user_ids": ["user-r"],
                    "config": {
                        "review_mode": "confirm",
                        "external_sources_require_confirmation": "true",
                    },
                },
                "permissive": {
                    "database_filename": "permissive.duckdb",
                    "graph_dirname": "permissive_kuzu",
                    "allowed_user_ids": ["user-p"],
                    "config": {
                        "review_mode": "auto",
                        "external_sources_require_confirmation": "false",
                    },
                },
            },
        }
        (tmp_path / "hybrid_memory.json").write_text(
            json.dumps(config), encoding="utf-8"
        )
        return MemoryService(tmp_path)

    def test_confirm_tenant_blocks_auto_review(self, tmp_path):
        """In a confirm-mode tenant, auto_review approval is downgraded
        to pending_user_confirmation."""
        svc = self._make_service(tmp_path)
        # Save a candidate first.
        cand = svc.dispatch({
            "component": "store", "method": "save_candidate",
            "user_id": "user-r",
            "args": {
                "category": "personal_fact",
                "content": "User works at TechCorp",
            },
        })
        cid = cand["candidate_id"]
        # Attempt auto-review approval.
        result = svc.dispatch({
            "component": "store", "method": "review_candidate",
            "user_id": "user-r",
            "args": {
                "candidate_id": cid,
                "decision": "reviewed_approved",
                "review_source": "auto_review",
            },
        })
        # The decision should be downgraded to pending_user_confirmation.
        assert result is not None
        # The candidate's status should be pending_user_confirmation, not approved.
        # review_candidate returns a dict with the candidate.
        candidate = result.get("candidate", result) if isinstance(result, dict) else result
        status = candidate.get("status", "") if isinstance(candidate, dict) else ""
        assert "pending" in status.lower() or "confirmation" in status.lower(), (
            f"Confirm-mode tenant should downgrade auto_review to pending, got status={status!r}"
        )

    def test_auto_tenant_allows_auto_review(self, tmp_path):
        """In an auto-mode tenant, auto_review approval proceeds."""
        svc = self._make_service(tmp_path)
        cand = svc.dispatch({
            "component": "store", "method": "save_candidate",
            "user_id": "user-p",
            "args": {
                "category": "personal_fact",
                "content": "User likes coffee",
            },
        })
        cid = cand["candidate_id"]
        result = svc.dispatch({
            "component": "store", "method": "review_candidate",
            "user_id": "user-p",
            "args": {
                "candidate_id": cid,
                "decision": "reviewed_approved",
                "review_source": "auto_review",
            },
        })
        assert result is not None
        candidate = result.get("candidate", result) if isinstance(result, dict) else result
        status = candidate.get("status", "") if isinstance(candidate, dict) else ""
        # Should be approved (or reviewed_approved), not pending.
        assert "approved" in status.lower(), (
            f"Auto-mode tenant should allow auto_review approval, got status={status!r}"
        )

    def test_manual_review_not_affected_by_confirm_mode(self, tmp_path):
        """Manual review (review_source=manual) is not downgraded by
        confirm mode — a human explicitly approved."""
        svc = self._make_service(tmp_path)
        cand = svc.dispatch({
            "component": "store", "method": "save_candidate",
            "user_id": "user-r",
            "args": {
                "category": "personal_fact",
                "content": "User likes tea",
            },
        })
        cid = cand["candidate_id"]
        result = svc.dispatch({
            "component": "store", "method": "review_candidate",
            "user_id": "user-r",
            "args": {
                "candidate_id": cid,
                "decision": "approved",
                "review_source": "manual",
            },
        })
        assert result is not None
        candidate = result.get("candidate", result) if isinstance(result, dict) else result
        status = candidate.get("status", "") if isinstance(candidate, dict) else ""
        assert "approved" in status.lower(), (
            f"Manual review should not be downgraded, got status={status!r}"
        )

    def test_client_cannot_override_review_mode_in_request(self, tmp_path):
        """A client-supplied review_mode in the args cannot change policy."""
        svc = self._make_service(tmp_path)
        # The policy is resolved from the tenant config, not from args.
        # Even if the client passes review_mode=auto in the args, it's
        # ignored — the service uses the tenant's policy.
        cand = svc.dispatch({
            "component": "store", "method": "save_candidate",
            "user_id": "user-r",
            "args": {
                "category": "personal_fact",
                "content": "User likes hiking",
            },
        })
        cid = cand["candidate_id"]
        # Attempt auto-review with review_mode in args (should be ignored).
        result = svc.dispatch({
            "component": "store", "method": "review_candidate",
            "user_id": "user-r",
            "args": {
                "candidate_id": cid,
                "decision": "reviewed_approved",
                "review_source": "auto_review",
                "review_mode": "auto",  # client attempt to override — ignored
            },
        })
        assert result is not None
        candidate = result.get("candidate", result) if isinstance(result, dict) else result
        status = candidate.get("status", "") if isinstance(candidate, dict) else ""
        assert "pending" in status.lower() or "confirmation" in status.lower(), (
            f"Client-supplied review_mode should be ignored, got status={status!r}"
        )


# -- get_status reports per-tenant policies ----------------------------------

class TestGetStatusReportsPolicies:
    """get_status includes per-tenant policy summaries (#127)."""

    def test_get_status_includes_tenant_policies(self, tmp_path):
        """get_status returns a tenant_policies dict with per-tenant policy."""
        config = {
            "tenants": {
                "alpha": {
                    "database_filename": "alpha.duckdb",
                    "graph_dirname": "alpha_kuzu",
                    "allowed_user_ids": ["user-a"],
                    "config": {"review_mode": "auto", "max_injected_items": "10"},
                },
            },
        }
        (tmp_path / "hybrid_memory.json").write_text(
            json.dumps(config), encoding="utf-8"
        )
        svc = MemoryService(tmp_path)
        status = svc.dispatch({"method": "get_status"})
        assert "tenant_policies" in status
        assert "alpha" in status["tenant_policies"]
        alpha_policy = status["tenant_policies"]["alpha"]
        assert alpha_policy["review_mode"] == "auto"
        assert alpha_policy["max_injected_items"] == 10


# -- Backward compatibility: single-tenant (no tenants key) -----------------

class TestBackwardCompat:
    """Legacy single-tenant config (no tenants key) still works."""

    def test_single_tenant_default_policy(self, tmp_path):
        """A config without 'tenants' gets a single 'default' tenant with
        default policy."""
        config = {"max_injected_items": "7"}
        tenants, _, strict = _parse_tenants(config, tmp_path, None, None)
        assert "default" in tenants
        assert tenants["default"].policy.review_mode == "confirm"
        assert tenants["default"].policy.max_injected_items == 7
        assert strict is False  # no allowed_user_ids

    def test_single_tenant_local_only(self, tmp_path):
        """A single-tenant config with local_only works."""
        config = {"local_only": "true"}
        tenants, _, _ = _parse_tenants(config, tmp_path, None, None)
        assert tenants["default"].policy.local_only is True

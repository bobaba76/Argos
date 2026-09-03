"""Tests for provider_retrieval.py audit fixes."""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

_plugin_dir = Path(__file__).resolve().parent.parent
if str(_plugin_dir) not in sys.path:
    sys.path.insert(0, str(_plugin_dir))


def _stub_hermes_runtime():
    """Stub the Hermes runtime so we can import the provider."""
    if "agent" not in sys.modules:
        sys.modules["agent"] = types.ModuleType("agent")
    if "agent.memory_provider" not in sys.modules:
        _mp = types.ModuleType("agent.memory_provider")
        class MemoryProvider:
            pass
        _mp.MemoryProvider = MemoryProvider
        sys.modules["agent.memory_provider"] = _mp
    if "tools" not in sys.modules:
        sys.modules["tools"] = types.ModuleType("tools")
    if "tools.registry" not in sys.modules:
        _tr = types.ModuleType("tools.registry")
        def _tool_error(msg):
            return json.dumps({"error": str(msg)})
        _tr.tool_error = _tool_error
        sys.modules["tools.registry"] = _tr


# ---------------------------------------------------------------------------
# PR1 — ACL filter in prefetch path uses self.user_id (doesn't exist)
#        instead of self._user_id. The try/except swallows the AttributeError,
#        so the ACL re-validation never runs.
# ---------------------------------------------------------------------------


def test_acl_filter_in_prefetch_uses_correct_attribute(tmp_path):
    """PR1: The ACL defence-in-depth filter in the prefetch path must
    use self._user_id (the provider's actual attribute), not self.user_id
    (which doesn't exist on the provider and raises AttributeError).

    Before the fix, the try/except swallowed the AttributeError, so the
    ACL filter never ran — denied content could leak into injection.
    """
    _stub_hermes_runtime()

    from store import DuckDBMemoryStore
    from access_scoping import ACLConfig, filter_records_by_access
    try:
        import argos_plugin
    except ModuleNotFoundError:
        import argos as argos_plugin

    # Build an ACL config with two users: alice (acme scope) and bob (beta scope).
    acl = ACLConfig.from_dict({
        "enforcement_on": True,
        "roles": {
            "acme_staff": {"client_scopes": ["acme"]},
            "beta_staff": {"client_scopes": ["beta"]},
        },
        "user_roles": {
            "alice": "acme_staff",
            "bob": "beta_staff",
        },
    })

    store = DuckDBMemoryStore(tmp_path / "test.duckdb", user_id="alice")
    store._acl_config = acl

    # Alice saves a memory in the acme scope.
    store.remember(
        category="personal_fact",
        content="Acme Corp revenue is $5M",
        client_scope="acme",
    )
    # Alice saves a memory in the beta scope (should be hidden from alice).
    store.remember(
        category="personal_fact",
        content="Beta Inc revenue is $3M",
        client_scope="beta",
    )

    # Construct a provider as alice.
    provider = argos_plugin.ArgosProvider()
    provider._store = store
    provider._graph = None
    provider._acl_config = acl
    provider._user_id = "alice"
    provider._max_injected = 5
    provider._inject_cap = 800
    provider._injection_min_score = 0.0
    provider._context_aware_retrieval = False
    provider._query_expander = None
    provider._graph_aware_retrieval = False
    provider._chronological_injection = False
    provider._date_anchor_rerank = False
    provider._freshness_markers = False
    provider._skip_retrieval_on_trivial = False
    provider._client_scope = None
    provider._prefetch_lock = __import__("threading").Lock()
    provider._prefetch_query = ""
    provider._prefetch_result = ""
    provider._prefetch_done = False
    provider._prefetch_thread = None
    provider._context_lock = __import__("threading").Lock()
    provider._recent_user_messages = []
    provider._context_window_size = 10
    provider._context_max_chars = 1000

    # Run prefetch for a query that matches both memories.
    result = provider.prefetch("revenue")

    # The ACL filter should have hidden the beta-scope memory from alice.
    # If the ACL filter is dead code (self.user_id bug), both memories appear.
    assert "Acme Corp" in result, "Alice should see her own acme-scope memory"
    assert "Beta Inc" not in result, (
        "Alice should NOT see beta-scope memory — the ACL filter in the "
        "prefetch path is dead code (uses self.user_id instead of self._user_id)"
    )

    store.close()


def test_acl_filter_in_prefetch_open_store_no_filter(tmp_path):
    """Sanity: an open store (no ACL) doesn't filter — both memories appear."""
    _stub_hermes_runtime()

    from store import DuckDBMemoryStore
    from access_scoping import ACLConfig
    try:
        import argos_plugin
    except ModuleNotFoundError:
        import argos as argos_plugin

    # Open store (no enforcement).
    acl = ACLConfig.from_dict({})
    store = DuckDBMemoryStore(tmp_path / "test.duckdb", user_id="alice")
    store._acl_config = acl

    store.remember(
        category="personal_fact",
        content="Acme Corp revenue is $5M",
        client_scope="acme",
    )
    store.remember(
        category="personal_fact",
        content="Beta Inc revenue is $3M",
        client_scope="beta",
    )

    provider = argos_plugin.ArgosProvider()
    provider._store = store
    provider._graph = None
    provider._acl_config = acl
    provider._user_id = "alice"
    provider._max_injected = 5
    provider._inject_cap = 800
    provider._injection_min_score = 0.0
    provider._context_aware_retrieval = False
    provider._query_expander = None
    provider._graph_aware_retrieval = False
    provider._chronological_injection = False
    provider._date_anchor_rerank = False
    provider._freshness_markers = False
    provider._skip_retrieval_on_trivial = False
    provider._client_scope = None
    provider._prefetch_lock = __import__("threading").Lock()
    provider._prefetch_query = ""
    provider._prefetch_result = ""
    provider._prefetch_done = False
    provider._prefetch_thread = None
    provider._context_lock = __import__("threading").Lock()
    provider._recent_user_messages = []
    provider._context_window_size = 10
    provider._context_max_chars = 1000

    result = provider.prefetch("revenue")

    # Open store: both memories should appear.
    assert "Acme Corp" in result
    assert "Beta Inc" in result

    store.close()

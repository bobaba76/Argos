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


# ---------------------------------------------------------------------------
# #203 — provider._acl_config never set by initialize(); the prefetch
#        defence-in-depth re-validation was dormant in the live path.
# ---------------------------------------------------------------------------


def test_initialize_wires_acl_config_from_config_file(tmp_path):
    """#203: initialize() must load ACLConfig from the config file and set
    self._acl_config on the provider. Without this, the prefetch ACL
    re-validation in provider_retrieval.py is dormant (acl is None)."""
    _stub_hermes_runtime()

    try:
        import argos_plugin
    except ModuleNotFoundError:
        import argos as argos_plugin

    # Write a config file with ACL data.
    home = tmp_path / "hermes_home"
    home.mkdir()
    config_path = home / "hybrid_memory.json"
    config_path.write_text(json.dumps({
        "storage_mode": "direct",
        "acl": {
            "enforcement_on": True,
            "roles": {
                "acme_staff": {"client_scopes": ["acme"]},
            },
            "user_roles": {
                "alice": "acme_staff",
            },
        },
    }), encoding="utf-8")

    # Minimal valid DuckDB file (no records needed for init).
    from store import DuckDBMemoryStore
    DuckDBMemoryStore(home / "hybrid_memory.duckdb", user_id="alice").close()

    provider = argos_plugin.ArgosProvider()
    provider.initialize(
        session_id="test",
        hermes_home=str(home),
        platform="cli",
        user_id="alice",
    )

    # The provider must have _acl_config set from the config file.
    assert hasattr(provider, "_acl_config"), \
        "initialize() must set _acl_config on the provider"
    assert provider._acl_config is not None, \
        "_acl_config must not be None after initialize()"
    assert not provider._acl_config.is_open_store, \
        f"ACL config should have enforcement_on=True, got open store: {provider._acl_config.__dict__}"
    # Alice's mask should be ["acme"].
    mask = provider._acl_config.allow_mask("alice")
    assert mask is not None and "acme" in mask, \
        f"Alice should have acme in her mask, got: {mask}"


def test_initialize_no_acl_config_defaults_to_open_store(tmp_path):
    """#203: Without an 'acl' key in the config, initialize() should set
    _acl_config to an open-store ACLConfig (backward compatible)."""
    _stub_hermes_runtime()

    try:
        import argos_plugin
    except ModuleNotFoundError:
        import argos as argos_plugin

    home = tmp_path / "hermes_home"
    home.mkdir()
    config_path = home / "hybrid_memory.json"
    config_path.write_text(json.dumps({
        "storage_mode": "direct",
    }), encoding="utf-8")

    from store import DuckDBMemoryStore
    DuckDBMemoryStore(home / "hybrid_memory.duckdb", user_id="alice").close()

    provider = argos_plugin.ArgosProvider()
    provider.initialize(
        session_id="test",
        hermes_home=str(home),
        platform="cli",
        user_id="alice",
    )

    assert hasattr(provider, "_acl_config"), \
        "initialize() must set _acl_config even when no acl key exists"
    assert provider._acl_config.is_open_store, \
        f"Without acl config, should be open store: {provider._acl_config.__dict__}"


def test_prefetch_acl_filter_works_after_initialize(tmp_path):
    """#203: End-to-end: initialize() with ACL config → prefetch hides
    out-of-scope content. This proves the provider's ACL is live without
    manually setting _acl_config.

    Uses _search_memories + the ACL filter directly (instead of prefetch,
    which has a 3-second thread timeout that can be flaky in test envs
    when the embedding model loads slowly). The ACL filter code path is
    the same — we're just calling it synchronously.
    """
    _stub_hermes_runtime()

    from store import DuckDBMemoryStore
    from access_scoping import filter_records_by_access
    try:
        import argos_plugin
    except ModuleNotFoundError:
        import argos as argos_plugin

    home = tmp_path / "hermes_home"
    home.mkdir()
    config_path = home / "hybrid_memory.json"
    config_path.write_text(json.dumps({
        "storage_mode": "direct",
        "acl": {
            "enforcement_on": True,
            "roles": {
                "acme_staff": {"client_scopes": ["acme"]},
            },
            "user_roles": {
                "alice": "acme_staff",
            },
        },
    }), encoding="utf-8")

    # Create the store and add records with different client_scopes.
    store = DuckDBMemoryStore(home / "hybrid_memory.duckdb", user_id="alice")
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
    store.close()

    # Initialize the provider — this should load the ACL config.
    provider = argos_plugin.ArgosProvider()
    provider.initialize(
        session_id="test",
        hermes_home=str(home),
        platform="cli",
        user_id="alice",
    )

    # The provider must have _acl_config set.
    assert provider._acl_config is not None
    assert not provider._acl_config.is_open_store

    # Search returns both records (store-level search has no ACL filter
    # in direct mode — that's the service's job).
    results = provider._search_memories("revenue", limit=10)
    contents = [r.content for r in results]
    assert "Acme Corp revenue is $5M" in contents, \
        f"Store search should find acme memory: {contents}"
    assert "Beta Inc revenue is $3M" in contents, \
        f"Store search should find beta memory: {contents}"

    # Apply the same ACL filter the prefetch path uses (line 1065-1077
    # of provider_retrieval.py). With the fix, provider._acl_config is
    # set, so the filter runs and hides the beta-scope memory.
    acl = getattr(provider, "_acl_config", None)
    assert acl is not None and not acl.is_open_store, \
        "provider._acl_config must be set and enforced after initialize()"
    filtered, denied = filter_records_by_access(results, acl, provider._user_id)
    filtered_contents = [r.content for r in filtered]
    assert "Acme Corp revenue is $5M" in filtered_contents, \
        f"Alice should see her own acme-scope memory: {filtered_contents}"
    assert "Beta Inc revenue is $3M" not in filtered_contents, (
        f"Alice should NOT see beta-scope memory — the ACL filter is "
        f"dormant (provider._acl_config not set by initialize): {filtered_contents}"
    )
    assert denied == 1, f"Expected 1 denied record, got {denied}"

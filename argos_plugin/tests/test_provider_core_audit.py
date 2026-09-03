"""Tests for provider_core.py audit fixes (PC1-PC10)."""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

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
# PC1 — entity_aliases loading was dead code (ran before store was created)
# ---------------------------------------------------------------------------


def test_entity_aliases_loaded_after_initialize(tmp_path):
    """PC1: entity_aliases from config must be loaded into the store
    during initialize(). Before the fix, the alias-loading block ran
    before self._store was created, so the guard
    `if aliases_json and self._store:` was always False.
    """
    _stub_hermes_runtime()

    from store import DuckDBMemoryStore
    try:
        import argos_plugin
    except ModuleNotFoundError:
        import argos as argos_plugin

    home = tmp_path / "hermes_home"
    home.mkdir()
    config_path = home / "hybrid_memory.json"
    config_path.write_text(json.dumps({
        "storage_mode": "direct",
        "entity_aliases": json.dumps({
            "my role": "Entity-A",
            "my company": "Entity-B",
        }),
    }), encoding="utf-8")

    # Create a minimal valid DuckDB file.
    DuckDBMemoryStore(home / "hybrid_memory.duckdb", user_id="alice").close()

    provider = argos_plugin.ArgosProvider()
    provider.initialize(
        session_id="test",
        hermes_home=str(home),
        platform="cli",
        user_id="alice",
    )

    # The store should have the aliases loaded.
    aliases = provider._store.list_aliases()
    alias_map = {a.get("alias"): a.get("canonical_entity") for a in aliases}
    assert "my role" in alias_map, \
        f"entity_aliases not loaded — dead code bug (PC1): {alias_map}"
    assert alias_map["my role"] == "entity-a"
    assert "my company" in alias_map
    assert alias_map["my company"] == "entity-b"


def test_entity_aliases_empty_config_no_error(tmp_path):
    """Sanity: no entity_aliases in config → no aliases loaded, no error."""
    _stub_hermes_runtime()

    from store import DuckDBMemoryStore
    try:
        import argos_plugin
    except ModuleNotFoundError:
        import argos as argos_plugin

    home = tmp_path / "hermes_home"
    home.mkdir()
    config_path = home / "hybrid_memory.json"
    config_path.write_text(json.dumps({"storage_mode": "direct"}), encoding="utf-8")

    DuckDBMemoryStore(home / "hybrid_memory.duckdb", user_id="alice").close()

    provider = argos_plugin.ArgosProvider()
    provider.initialize(
        session_id="test",
        hermes_home=str(home),
        platform="cli",
        user_id="alice",
    )

    aliases = provider._store.list_aliases()
    assert len(aliases) == 0


# ---------------------------------------------------------------------------
# PC2 — reranker_enabled default mismatch (initialize default was "true",
#        config default was "false")
# ---------------------------------------------------------------------------


def test_reranker_default_is_false(tmp_path):
    """PC2: reranker_enabled default should be False (matching _load_config
    default of "false"), not True (the old initialize() default of "true").
    """
    _stub_hermes_runtime()

    from store import DuckDBMemoryStore
    try:
        import argos_plugin
    except ModuleNotFoundError:
        import argos as argos_plugin

    home = tmp_path / "hermes_home"
    home.mkdir()
    config_path = home / "hybrid_memory.json"
    config_path.write_text(json.dumps({"storage_mode": "direct"}), encoding="utf-8")

    DuckDBMemoryStore(home / "hybrid_memory.duckdb", user_id="alice").close()

    provider = argos_plugin.ArgosProvider()
    provider.initialize(
        session_id="test",
        hermes_home=str(home),
        platform="cli",
        user_id="alice",
    )

    assert provider._reranker_enabled is False, \
        "reranker_enabled default should be False (PC2)"


# ---------------------------------------------------------------------------
# PC4 — chain_unfold config default was "auto" but comment said "ships OFF"
# ---------------------------------------------------------------------------


def test_chain_unfold_default_is_off(tmp_path):
    """PC4: chain_unfold config default should be "off" (matching the
    comment 'ships OFF for the first wave — measure, then flip to auto'),
    not "auto".
    """
    _stub_hermes_runtime()

    from store import DuckDBMemoryStore
    try:
        import argos_plugin
    except ModuleNotFoundError:
        import argos as argos_plugin

    home = tmp_path / "hermes_home"
    home.mkdir()
    config_path = home / "hybrid_memory.json"
    config_path.write_text(json.dumps({"storage_mode": "direct"}), encoding="utf-8")

    DuckDBMemoryStore(home / "hybrid_memory.duckdb", user_id="alice").close()

    provider = argos_plugin.ArgosProvider()
    provider.initialize(
        session_id="test",
        hermes_home=str(home),
        platform="cli",
        user_id="alice",
    )

    assert provider._chain_unfold == "off", \
        f"chain_unfold default should be 'off' (PC4), got '{provider._chain_unfold}'"


# ---------------------------------------------------------------------------
# PC7 — inline boolean parsers now accept "on" (via _flag helper)
# ---------------------------------------------------------------------------


def test_flag_helper_accepts_on():
    """PC7: _flag() helper accepts "on" as a truthy value."""
    from provider_core import _flag
    assert _flag({"key": "on"}, "key", "false") is True
    assert _flag({"key": "ON"}, "key", "false") is True
    assert _flag({"key": "true"}, "key", "false") is True
    assert _flag({"key": "1"}, "key", "false") is True
    assert _flag({"key": "yes"}, "key", "false") is True
    assert _flag({"key": "false"}, "key", "false") is False
    assert _flag({"key": "off"}, "key", "false") is False
    assert _flag({}, "key", "true") is True
    assert _flag({}, "key", "false") is False


def test_graph_aware_retrieval_accepts_on(tmp_path):
    """PC7: graph_aware_retrieval="on" should be True after the _flag
    refactor. Before the fix, the inline parser only accepted
    "true"/"1"/"yes" and would return False for "on".
    """
    _stub_hermes_runtime()

    from store import DuckDBMemoryStore
    try:
        import argos_plugin
    except ModuleNotFoundError:
        import argos as argos_plugin

    home = tmp_path / "hermes_home"
    home.mkdir()
    config_path = home / "hybrid_memory.json"
    config_path.write_text(json.dumps({
        "storage_mode": "direct",
        "graph_aware_retrieval": "on",
    }), encoding="utf-8")

    DuckDBMemoryStore(home / "hybrid_memory.duckdb", user_id="alice").close()

    provider = argos_plugin.ArgosProvider()
    provider.initialize(
        session_id="test",
        hermes_home=str(home),
        platform="cli",
        user_id="alice",
    )

    assert provider._graph_aware_retrieval is True, \
        "graph_aware_retrieval='on' should be True after PC7 fix"

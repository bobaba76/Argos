"""Pytest tests for the argos plugin.

Run with:
    python -m pytest tests/test_argos.py -v

Or use the standalone script (no pytest needed):
    python tests/run_tests.py
"""
from __future__ import annotations

import sys
import tempfile
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pytest

# Ensure the plugin package is importable.
_plugin_dir = Path(__file__).resolve().parent.parent
if str(_plugin_dir.parent) not in sys.path:
    sys.path.insert(0, str(_plugin_dir.parent))


class TestKuzuGraph:
    def test_init_and_query(self, tmp_path):
        from graph import KuzuGraphStore

        graph = KuzuGraphStore(tmp_path / "test_kuzu", user_id="test_user")
        graph.add_relationship("user", "person", "married_to", "Sam", "person")
        graph.add_relationship("user", "person", "takes_medication", "FocusTool", "medication")

        edges = graph.query_graph("user")
        assert len(edges) == 2
        relations = {e["relation"] for e in edges}
        assert "married_to" in relations
        assert "takes_medication" in relations

        sam_edges = graph.search_graph("Sam")
        assert len(sam_edges) >= 1
        graph.close()

    def test_graph_scopes_isolate_entities(self, tmp_path):
        from graph import KuzuGraphStore

        graph_alice = KuzuGraphStore(tmp_path / "scoped_kuzu", user_id="alice")
        graph_bob = KuzuGraphStore(tmp_path / "scoped_kuzu", user_id="bob")
        graph_alice.add_relationship("user", "person", "knows", "AlicePrivate", "person")
        graph_bob.add_relationship("user", "person", "knows", "BobPrivate", "person")
        assert graph_alice.search_graph("BobPrivate") == []
        assert graph_bob.search_graph("AlicePrivate") == []
        assert graph_alice.search_graph("AlicePrivate")
        assert graph_bob.search_graph("BobPrivate")
        graph_alice.close()
        graph_bob.close()

    def test_graph_query_returns_memory_evidence(self, tmp_path):
        from graph import KuzuGraphStore

        graph = KuzuGraphStore(tmp_path / "test_kuzu", user_id="test_user")
        graph.index_memory(
            "mem-kubernetes",
            "personal_fact",
            "User uses Kubernetes for local deployments",
            ["devops"],
            use_llm=False,
        )
        ids = graph.memory_ids_for_query("Kubernetes", limit=10)
        assert "mem-kubernetes" in ids
        graph.close()

    def test_graph_traversal_walks_typed_relations(self, tmp_path):
        """Traversal must walk TYPED relations from a seed entity and
        return memory IDs attached to traversed edges — and must NOT
        traverse from the user hub or follow generic mentions edges."""
        from graph import KuzuGraphStore

        graph = KuzuGraphStore(tmp_path / "trav_kuzu", user_id="test_user")
        # A typed chain: user -has_wife-> Alex -works_at-> TechCorp.
        graph.add_relationship(
            "user", "person", "has_wife", "Alex", "person",
            {"memory_id": "mem-wife", "extractor": "llm"})
        graph.add_relationship(
            "Alex", "person", "works_at", "TechCorp", "organization",
            {"memory_id": "mem-alex-work", "extractor": "llm"})
        # Generic noise edge that must NOT be traversed.
        graph.add_relationship(
            "user", "person", "related_to", "noise thing", "concept",
            {"memory_id": "mem-noise", "extractor": "graph_patterns"})

        # Query mentioning Alex: traversal should reach the 2-hop chain
        # (mem-wife at hop 1, mem-alex-work at hop 2).
        ids = graph.traversal_memory_ids("Alex", depth=2, limit=20)
        assert "mem-wife" in ids, ids
        assert "mem-alex-work" in ids, ids
        # The generic related_to edge must not contribute.
        assert "mem-noise" not in ids, ids
        graph.close()

    def test_graph_traversal_ignores_user_hub(self, tmp_path):
        """Seeds grounded only to the 'user' node must yield no traversal
        (the hub connects to everything and carries no signal)."""
        from graph import KuzuGraphStore

        graph = KuzuGraphStore(tmp_path / "hub_kuzu", user_id="test_user")
        graph.add_relationship(
            "user", "person", "uses", "FocusTool", "tool",
            {"memory_id": "mem-ft", "extractor": "llm"})
        # The term 'user' grounds to the user hub only -> no seeds -> [].
        ids = graph.traversal_memory_ids("the user", depth=2, limit=10)
        assert ids == [], ids
        graph.close()

    def test_graph_update_removes_old_id_not_new(self, tmp_path):
        """Regression: memory_update must remove the OLD memory_id from the
        graph, not the new one. The old ID was indexed; the new ID was never
        indexed at the time of removal.

        Before the fix, the provider called remove_memory(rec.memory_id)
        where rec was the NEW record — so the old ID stayed in the graph
        as a zombie (resolving to 0 records), and the new ID was never
        removed (harmless but wrong).

        This test goes through the real provider path (handle_tool_call)
        with a real DuckDB store + Kuzu graph — not a simulation. If the
        arg swap in __init__.py is reverted, this test will fail.
        """
        import sys
        import types
        import json as _json

        # Stub the Hermes runtime so we can import the provider
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
                return _json.dumps({"error": str(msg)})
            _tr.tool_error = _tool_error
            sys.modules["tools.registry"] = _tr

        from store import DuckDBMemoryStore
        from graph import KuzuGraphStore
        try:
            import argos_plugin
        except ModuleNotFoundError:
            import argos as argos_plugin

        store = DuckDBMemoryStore(tmp_path / "test.duckdb", user_id="test_user")
        graph = KuzuGraphStore(tmp_path / "test_kuzu", user_id="test_user")

        # Create a memory and index it in the graph (simulating what
        # the provider does on memory_save)
        rec = store.remember(
            category="personal_fact",
            content="User uses Kubernetes for local deployments",
            tags=["devops"],
        )
        old_id = rec.memory_id
        graph.index_memory(
            old_id, rec.category, rec.content, rec.tags,
            rec.created_at, use_llm=False,
        )
        assert old_id in graph.memory_ids_for_query("Kubernetes", limit=10)

        # Construct a provider and attach the real store + graph
        provider = argos_plugin.ArgosProvider()
        provider._store = store
        provider._graph = graph

        # Call the REAL provider path — handle_tool_call("memory_update")
        # This exercises the exact code in __init__.py that had the bug.
        result = provider.handle_tool_call(
            "memory_update",
            {"memory_id": old_id, "content": "User uses Docker for local deployments"},
        )
        parsed = _json.loads(result)
        assert parsed.get("status") == "updated"
        new_id = parsed.get("memory_id")
        assert new_id != old_id, "update_memory should create a new version ID"

        # Old ID should be gone from the graph (removed by the provider)
        ids = graph.memory_ids_for_query("Kubernetes", limit=10)
        assert old_id not in ids, f"Old ID {old_id} leaked as graph zombie"

        # New ID should be present (indexed by the provider)
        new_ids = graph.memory_ids_for_query("Docker", limit=10)
        assert new_id in new_ids, f"New ID {new_id} not indexed in graph"

        store.close()
        graph.close()

    def test_purge_junk(self, tmp_path):
        from graph import KuzuGraphStore

        graph = KuzuGraphStore(tmp_path / "test_kuzu", user_id="test_user")
        graph.upsert_node("FocusTool", "medication")
        graph.upsert_node("the", "concept")
        graph.upsert_node("ab", "concept")

        before = graph.count_nodes()
        quarantined = graph.purge_junk_entities()
        after = graph.count_nodes()

        assert quarantined >= 2
        assert after == before
        nodes = graph.list_nodes()
        ids = {n["id"] for n in nodes}
        assert "FocusTool" in ids
        assert "the" not in ids
        graph.close()

    def test_clear_scope_removes_nodes_and_edges(self, tmp_path):
        """clear_scope should delete all nodes and edges for the current
        user scope while leaving other scopes untouched."""
        from graph import KuzuGraphStore

        graph_a = KuzuGraphStore(tmp_path / "scoped_kuzu", user_id="alice")
        graph_b = KuzuGraphStore(tmp_path / "scoped_kuzu", user_id="bob")

        graph_a.add_relationship("user", "person", "knows", "AliceFriend", "person")
        graph_b.add_relationship("user", "person", "knows", "BobFriend", "person")

        assert graph_a.count_nodes() >= 1
        assert graph_b.count_nodes() >= 1

        # Clear alice's scope
        remaining_a_nodes, remaining_a_edges = graph_a.clear_scope()
        assert remaining_a_nodes == 0
        assert remaining_a_edges == 0

        # Bob's scope should be untouched
        assert graph_b.count_nodes() >= 1
        assert graph_b.count_edges() >= 1

        graph_a.close()
        graph_b.close()

    def test_extraction_gate_rejects_sentence_payloads(self):
        from graph import _valid_graph_entity, extract_graph_relations

        assert _valid_graph_entity("Sam")
        assert _valid_graph_entity("know more about the watcher")
        assert not _valid_graph_entity(
            "expecting me to be loading new codes into this system"
        )
        assert not _valid_graph_entity("Location")
        relations = extract_graph_relations(
            "User goal: expecting me to be loading new codes into this system",
            category="goal",
        )
        assert all(len(item["target"].split()) < 6 for item in relations)



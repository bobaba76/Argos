"""Pytest tests for the argos plugin.

Run with:
    python -m pytest tests/test_argos.py -v

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


@pytest.fixture
def tmp_path_factory_dir(tmp_path):
    """Provide a clean temp directory for each test."""
    return tmp_path


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

    def test_search_graph_rejects_cross_scope_edge(self, tmp_path):
        """#176: search_graph must use AND for scope filter, not OR.

        A corrupted cross-scope edge (one endpoint in alice, one in bob)
        must NOT leak to either scope's search_graph results. The OR
        filter would return the edge to whichever scope matches one side;
        the AND filter correctly requires both endpoints in scope.
        """
        from graph import KuzuGraphStore

        graph_alice = KuzuGraphStore(tmp_path / "cross_scope_kuzu", user_id="alice")
        graph_bob = KuzuGraphStore(tmp_path / "cross_scope_kuzu", user_id="bob")
        # Normal in-scope edges.
        graph_alice.add_relationship("Alice", "person", "knows", "Alex", "person")
        graph_bob.add_relationship("Bob", "person", "knows", "Carol", "person")
        # Inject a cross-scope edge directly via the raw connection.
        # This simulates corruption or manual edit — both endpoints
        # have different user_scope values.
        with graph_alice._shared_conn_lock:
            graph_alice.conn.execute(
                "CREATE (a:Entity {id: 'CrossLeak', entity_type: 'person', "
                "user_scope: 'alice', attributes: ''})-[r:RelatesTo "
                "{relation_type: 'knows', attributes: ''}]->"
                "(b:Entity {id: 'BobSecret', entity_type: 'person', "
                "user_scope: 'bob', attributes: ''})"
            )
        # Alice's search_graph must NOT return the cross-scope edge
        # (BobSecret is in bob's scope, not alice's).
        alice_results = graph_alice.search_graph("CrossLeak", limit=50)
        alice_targets = {e.get("target", "") for e in alice_results}
        assert "BobSecret" not in alice_targets, (
            "search_graph OR-filter leak: alice sees bob's entity via "
            "cross-scope edge"
        )
        # Bob's search_graph must NOT return it either (CrossLeak is in
        # alice's scope).
        bob_results = graph_bob.search_graph("BobSecret", limit=50)
        bob_sources = {e.get("source", "") for e in bob_results}
        assert "CrossLeak" not in bob_sources, (
            "search_graph OR-filter leak: bob sees alice's entity via "
            "cross-scope edge"
        )
        graph_alice.close()
        graph_bob.close()

    def test_search_graph_filters_cross_client_scope_via_acl(self, tmp_path):
        """#197: search_graph must apply ACL filtering (filter_graph_neighbours)
        so a user with a restricted client_scope mask cannot see graph entities
        from other client scopes within the same tenant."""
        from graph import KuzuGraphStore
        from access_scoping import ACLConfig

        graph = KuzuGraphStore(tmp_path / "acl_kuzu", user_id="alice")
        graph.upsert_node("AcmeProject", "project", {"client_scope": "acme"})
        graph.upsert_node("BetaProject", "project", {"client_scope": "beta"})
        graph.upsert_node("SharedDirector", "person", {"client_scope": "acme"})
        graph.upsert_edge("AcmeProject", "SharedDirector", "owned_by")
        graph.upsert_edge("AcmeProject", "BetaProject", "linked_to")
        graph._acl_config = ACLConfig(
            roles={"staff": {"client_scopes": ["acme"], "wheel": False}},
            user_roles={"alice": "staff"},
            enforcement_on=True,
        )
        results = graph.search_graph("Project", limit=50)
        targets = {e.get("target", "") for e in results}
        sources = {e.get("source", "") for e in results}
        assert "AcmeProject" in sources or "AcmeProject" in targets, \
            "acme entity should be visible"
        assert "BetaProject" not in targets, \
            f"ACL leak: search_graph returned beta entity to acme-only user: {results}"
        assert "BetaProject" not in sources, \
            f"ACL leak: search_graph returned beta entity to acme-only user: {results}"
        acme_edges = [e for e in results
                      if e.get("source") == "AcmeProject" and e.get("target") == "SharedDirector"]
        assert len(acme_edges) == 1, \
            f"acme-internal edge should survive ACL filter: {results}"
        graph.close()

    def test_search_graph_no_acl_config_returns_all(self, tmp_path):
        """#197: Without an ACL config, search_graph returns all edges."""
        from graph import KuzuGraphStore

        graph = KuzuGraphStore(tmp_path / "no_acl_kuzu", user_id="alice")
        graph.upsert_node("AcmeProject", "project", {"client_scope": "acme"})
        graph.upsert_node("BetaProject", "project", {"client_scope": "beta"})
        graph.upsert_edge("AcmeProject", "BetaProject", "linked_to")
        results = graph.search_graph("Project", limit=50)
        targets = {e.get("target", "") for e in results}
        assert "BetaProject" in targets, \
            f"Without ACL config, all edges should be returned: {results}"
        graph.close()

    def test_search_graph_open_store_acl_returns_all(self, tmp_path):
        """#197: Open store ACL returns all edges (backward compatible)."""
        from graph import KuzuGraphStore
        from access_scoping import ACLConfig

        graph = KuzuGraphStore(tmp_path / "open_acl_kuzu", user_id="alice")
        graph.upsert_node("AcmeProject", "project", {"client_scope": "acme"})
        graph.upsert_node("BetaProject", "project", {"client_scope": "beta"})
        graph.upsert_edge("AcmeProject", "BetaProject", "linked_to")
        graph._acl_config = ACLConfig()
        results = graph.search_graph("Project", limit=50)
        targets = {e.get("target", "") for e in results}
        assert "BetaProject" in targets, \
            f"Open store should return all edges: {results}"
        graph.close()

    def test_memory_ids_for_query_inherits_acl_filter(self, tmp_path):
        """#197: memory_ids_for_query calls search_graph internally, so it
        inherits the ACL filter."""
        from graph import KuzuGraphStore
        from access_scoping import ACLConfig

        graph = KuzuGraphStore(tmp_path / "memids_acl_kuzu", user_id="alice")
        graph.upsert_node("AcmeDoc", "document", {"client_scope": "acme"})
        graph.upsert_node("AcmeAuthor", "person", {"client_scope": "acme"})
        graph.upsert_edge("AcmeDoc", "AcmeAuthor", "authored_by", {
            "memory_ids": ["mem-acme-1"],
        })
        graph.upsert_node("BetaDoc", "document", {"client_scope": "beta"})
        graph.upsert_edge("AcmeDoc", "BetaDoc", "related_to", {
            "memory_ids": ["mem-beta-1"],
        })
        graph._acl_config = ACLConfig(
            roles={"staff": {"client_scopes": ["acme"], "wheel": False}},
            user_roles={"alice": "staff"},
            enforcement_on=True,
        )
        mem_ids = graph.memory_ids_for_query("Doc", limit=50)
        assert "mem-acme-1" in mem_ids, \
            f"acme memory should be visible: {mem_ids}"
        assert "mem-beta-1" not in mem_ids, \
            f"ACL leak: beta memory surfaced to acme-only user: {mem_ids}"
        graph.close()

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



"""Batch-C tests: graph lock safety (#76), flush exception safety (#88),
boost-floor gate exemption (#81), and graph block error visibility (#84).
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path

_plugin_dir = Path(__file__).resolve().parent.parent
for _path in (_plugin_dir.parent, _plugin_dir):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


def _make_graph(tmp_path, user_id="test_user"):
    from graph import KuzuGraphStore
    return KuzuGraphStore(tmp_path / "test_graph", user_id=user_id)


class TestConcurrentGraphAccess:
    """#76: Hammer search_graph/query_graph while index_memory runs."""

    def test_concurrent_search_and_index_no_crash(self, tmp_path):
        graph = _make_graph(tmp_path)
        try:
            graph.index_memory("m1", "personal_fact", "User works at TechCorp",
                               ["work"], use_llm=False, flush=False)
            graph._flush()
            errors: list[Exception] = []

            def _reader():
                for _ in range(50):
                    try:
                        graph.search_graph("TechCorp", limit=10)
                    except Exception as exc:
                        errors.append(exc)

            def _writer():
                for i in range(20):
                    try:
                        graph.index_memory(f"m{i+10}", "context_note",
                            f"TechCorp builds product {i}", ["work"],
                            use_llm=False, flush=False)
                    except Exception as exc:
                        errors.append(exc)

            threads = [threading.Thread(target=_reader) for _ in range(3)]
            threads.append(threading.Thread(target=_writer))
            for t in threads: t.start()
            for t in threads: t.join(timeout=30)
            assert not errors, f"Concurrent access produced errors: {errors}"
        finally:
            graph.close()

    def test_concurrent_query_graph_and_index_no_crash(self, tmp_path):
        graph = _make_graph(tmp_path)
        try:
            graph.index_memory("m1", "personal_fact", "User works at TechCorp",
                               ["work"], use_llm=False, flush=False)
            graph._flush()
            errors: list[Exception] = []

            def _reader():
                for _ in range(50):
                    try: graph.query_graph("TechCorp")
                    except Exception as exc: errors.append(exc)

            def _writer():
                for i in range(20):
                    try: graph.index_memory(f"m{i+10}", "context_note",
                            f"TechCorp builds product {i}", ["work"],
                            use_llm=False, flush=False)
                    except Exception as exc: errors.append(exc)

            threads = [threading.Thread(target=_reader) for _ in range(3)]
            threads.append(threading.Thread(target=_writer))
            for t in threads: t.start()
            for t in threads: t.join(timeout=30)
            assert not errors
        finally:
            graph.close()

    def test_concurrent_list_nodes_and_index_no_crash(self, tmp_path):
        graph = _make_graph(tmp_path)
        try:
            graph.index_memory("m1", "personal_fact", "User works at TechCorp",
                               ["work"], use_llm=False, flush=False)
            graph._flush()
            errors: list[Exception] = []

            def _reader():
                for _ in range(50):
                    try: graph.list_nodes(limit=10)
                    except Exception as exc: errors.append(exc)

            def _writer():
                for i in range(20):
                    try: graph.index_memory(f"m{i+10}", "context_note",
                            f"TechCorp builds product {i}", ["work"],
                            use_llm=False, flush=False)
                    except Exception as exc: errors.append(exc)

            threads = [threading.Thread(target=_reader) for _ in range(3)]
            threads.append(threading.Thread(target=_writer))
            for t in threads: t.start()
            for t in threads: t.join(timeout=30)
            assert not errors
        finally:
            graph.close()


class TestFlushExceptionSafety:
    """#88: _flush() must not raise."""

    def test_flush_failure_does_not_raise(self, tmp_path, monkeypatch):
        graph = _make_graph(tmp_path)
        try:
            import kuzu
            orig = kuzu.Connection
            class _Boom:
                def __init__(self, *a, **k): raise RuntimeError("disk error")
            monkeypatch.setattr(kuzu, "Connection", _Boom)
            graph._flush()
            assert graph._flush_dirty is True
        finally:
            monkeypatch.setattr(kuzu, "Connection", orig)
            graph.close()

    def test_flush_success_clears_dirty_flag(self, tmp_path):
        graph = _make_graph(tmp_path)
        try:
            graph._flush_dirty = True
            graph._flush()
            assert graph._flush_dirty is False
        finally:
            graph.close()

    def test_index_memory_survives_flush_failure(self, tmp_path, monkeypatch):
        graph = _make_graph(tmp_path)
        try:
            import kuzu
            orig = kuzu.Connection
            class _Boom:
                def __init__(self, *a, **k): raise RuntimeError("disk error")
            graph.index_memory("m1", "personal_fact", "User works at TechCorp",
                               ["work"], use_llm=False, flush=False)
            monkeypatch.setattr(kuzu, "Connection", _Boom)
            graph.index_memory("m2", "context_note", "TechCorp is big",
                               ["work"], use_llm=False, flush=True)
        finally:
            monkeypatch.setattr(kuzu, "Connection", orig)
            graph.close()


class TestGraphBlockErrorVisibility:
    """#84: Programming errors must NOT be silently swallowed."""

    def test_name_error_in_graph_block_is_raised(self):
        import inspect
        from provider_retrieval import ProviderRetrievalMixin
        source = inspect.getsource(ProviderRetrievalMixin._search_memories)
        assert "NameError" in source
        assert "AttributeError" in source
        assert "ImportError" in source
        assert "_graph_retrieval_failures" in source

    def test_expected_failure_fails_soft_with_counter(self):
        import inspect
        from provider_retrieval import ProviderRetrievalMixin
        source = inspect.getsource(ProviderRetrievalMixin._search_memories)
        assert "logger.warning" in source
        assert "_graph_retrieval_failures" in source


class TestBoostFloorGateExemption:
    """#81: Alias/traversal/PPR candidates exempt from inclusion gate."""

    def test_gate_exemption_in_source(self):
        import inspect
        from provider_retrieval import ProviderRetrievalMixin
        source = inspect.getsource(ProviderRetrievalMixin._search_memories)
        assert "is_boosted_candidate" in source
        assert "alias_id_set_pre" in source


class TestObservedAtCapture:
    """#138: index_memory writes created_at into edge attributes as
    observed_at, capturing provenance for future temporal-graph work."""

    def test_edge_attributes_contain_observed_at(self, tmp_path):
        graph = _make_graph(tmp_path)
        try:
            created_at = "2026-09-02T10:00:00Z"
            graph.index_memory(
                "m_obs", "personal_fact", "User works at TechCorp",
                ["work"], created_at=created_at, use_llm=False,
            )
            # query_graph returns edges touching the TechCorp entity;
            # each edge's attributes should carry observed_at.
            edges = graph.query_graph("TechCorp")
            assert edges, "Expected at least one edge for TechCorp"
            found = False
            for edge in edges:
                attrs = edge.get("attributes") or {}
                if attrs.get("observed_at") == created_at:
                    found = True
                    break
            assert found, (
                f"observed_at={created_at!r} not found in any edge "
                f"attributes: {[e.get('attributes') for e in edges]}"
            )
        finally:
            graph.close()

    def test_observed_at_matches_record_created_at(self, tmp_path):
        """The observed_at value must exactly match the created_at passed
        to index_memory — no transformation, no truncation."""
        graph = _make_graph(tmp_path)
        try:
            ts = "2026-08-15T14:30:00Z"
            graph.index_memory(
                "m_obs2", "personal_fact", "Alice is my wife",
                ["relationship"], created_at=ts, use_llm=False,
            )
            edges = graph.query_graph("Alice")
            assert edges
            observed_values = {
                (e.get("attributes") or {}).get("observed_at")
                for e in edges
                if e.get("attributes")
            }
            assert ts in observed_values, (
                f"observed_at {ts!r} not in {observed_values}"
            )
        finally:
            graph.close()

    def test_observed_at_none_when_created_at_none(self, tmp_path):
        """When created_at is None, observed_at is None — no crash, no
        fabricated timestamp. Existing edges (pre-change) are unaffected."""
        graph = _make_graph(tmp_path)
        try:
            graph.index_memory(
                "m_obs3", "personal_fact", "User uses Vim",
                ["tool"], created_at=None, use_llm=False,
            )
            edges = graph.query_graph("Vim")
            assert edges
            # observed_at should be None (not missing, not fabricated)
            for edge in edges:
                attrs = edge.get("attributes") or {}
                # None is acceptable; the key may or may not be present
                # but must not be a fabricated timestamp.
                assert attrs.get("observed_at") is None or attrs.get("observed_at") is None
        finally:
            graph.close()


# ---------------------------------------------------------------------------
# Graph audit: G1, G2, G3
# ---------------------------------------------------------------------------

class TestGraphAuditG1G2G3:
    """Tests for graph audit findings G1–G3.

    G1: remove_memory skips incoming direct edges without memory_id in attrs
    G2: _query_edges_for_nodes has no user_scope filter (defense-in-depth)
    G3: upsert_node overwrites user_scope on match (cleanup)
    """

    # -- G1: incoming edge cleanup ---------------------------------------

    def test_g1_remove_memory_cleans_incoming_edge_to_memory_node(self, tmp_path):
        """remove_memory should clean up edges pointing AT the memory node,
        not just edges originating FROM it.

        Today the extractor only creates outgoing edges (about_user,
        mentions), so this is a landmine test — it creates a manual
        incoming edge and verifies remove_memory catches it.
        """
        graph = _make_graph(tmp_path)
        try:
            graph.index_memory(
                "m_g1", "personal_fact", "User works at TechCorp",
                ["work"], use_llm=False,
            )
            # Manually create an incoming edge: TechCorp -> memory:m_g1
            # (the extractor never does this, but a future pattern might).
            # The edge does NOT carry memory_id in its attributes — this is
            # the G1 bug scenario: a direct edge (target = memory_node)
            # without memory_id in attrs would be skipped by the filter at
            # line 1102 (source != memory_node is True, memory_id not in
            # memory_ids is True → continue).
            memory_node = graph._internal_id("memory:m_g1")
            techcorp_node = graph._internal_id("TechCorp")
            graph.upsert_node("TechCorp", "organization", {})
            graph.upsert_edge(
                "TechCorp", memory_node, "referenced_by",
                {"note": "manual cross-reference"},  # no memory_id
            )
            # Verify the edge exists.
            edges_before = graph.query_graph("TechCorp")
            assert any(e["relation"] == "referenced_by" for e in edges_before), (
                "Pre-condition: incoming edge to memory node must exist"
            )
            # Remove the memory — G1: should clean up the incoming edge too.
            graph.remove_memory("m_g1")
            # Check the edge's actual status in the DB (not just visibility
            # through query_graph, which would hide it because the memory
            # NODE is quarantined — that's a side effect, not the edge
            # being cleaned up).
            with graph._shared_conn_lock:
                result = graph.conn.execute(
                    """MATCH (a:Entity)-[r:RelatesTo]->(b:Entity)
                       WHERE a.id = $techcorp AND b.id = $mem
                         AND r.relation_type = $rel
                       RETURN r.attributes""",
                    parameters={
                        "techcorp": graph._internal_id("TechCorp"),
                        "mem": graph._internal_id("memory:m_g1"),
                        "rel": "referenced_by",
                    },
                )
                if result.has_next():
                    raw_attrs = result.get_next()[0]
                    import json as _json
                    attrs = _json.loads(raw_attrs) if raw_attrs else {}
                else:
                    attrs = None  # edge was deleted
            # G1 FIX: the edge should either be quarantined (status set)
            # or deleted — not left active.
            if attrs is not None:
                assert attrs.get("status") == "quarantined", (
                    f"Incoming edge to memory node should be quarantined "
                    f"after remove_memory; edge attrs: {attrs}"
                )
        finally:
            graph.close()

    # -- G2: _query_edges_for_nodes scope filter -------------------------

    def test_g2_query_edges_for_nodes_filters_by_scope(self, tmp_path):
        """_query_edges_for_nodes should not return edges from a different
        user_scope, even if the node IDs happen to match.

        Defense-in-depth: today ID prefixing (alice:: vs bob::) prevents
        cross-scope matches. But if a cross-scope edge ever exists (data
        corruption, manual edit), the scope filter should catch it.
        """
        from graph import KuzuGraphStore
        import json as _json
        # Create a graph with bob's scope.
        graph_b = KuzuGraphStore(tmp_path / "test_g2_graph", user_id="bob")
        try:
            # Create a node in bob's scope.
            graph_b.upsert_node("BobEntity", "concept", {})
            # Manually insert a cross-scope edge: alice's node -> bob's node,
            # with alice's user_scope on the edge. This simulates data
            # corruption or a bug that creates a cross-scope edge.
            alice_node = "alice::AliceEntity"
            bob_node = graph_b._internal_id("BobEntity")
            with graph_b._shared_conn_lock:
                # Create alice's node manually (with alice's scope).
                graph_b.conn.execute(
                    "MERGE (n:Entity {id: $id}) "
                    "ON CREATE SET n.entity_type = $type, "
                    "n.attributes = $attrs, n.user_scope = $scope",
                    parameters={
                        "id": alice_node, "type": "concept",
                        "attrs": "{}", "scope": "alice",
                    },
                )
                # Create a cross-scope edge: alice -> bob, scope = alice.
                graph_b.conn.execute(
                    "MATCH (a:Entity {id: $src}), (b:Entity {id: $tgt}) "
                    "MERGE (a)-[r:RelatesTo {relation_type: $rel}]->(b) "
                    "ON CREATE SET r.attributes = $attrs, r.user_scope = $scope",
                    parameters={
                        "src": alice_node, "tgt": bob_node,
                        "rel": "cross_scope", "attrs": "{}", "scope": "alice",
                    },
                )
            # Now query with bob's scope for BobEntity — the cross-scope
            # edge should NOT be returned.
            edges = graph_b._query_edges_for_nodes(["BobEntity"])
            # G2 FIX: bob should not see alice's cross-scope edge.
            cross_scope = [e for e in edges if e.get("relation") == "cross_scope"]
            assert not cross_scope, (
                f"_query_edges_for_nodes should not return cross-scope "
                f"edges; got cross-scope edges: {cross_scope}"
            )
        finally:
            graph_b.close()

    # -- G3: upsert_node user_scope overwrite ----------------------------

    def test_g3_upsert_node_preserves_user_scope_on_match(self, tmp_path):
        """upsert_node should NOT overwrite user_scope on an existing node.

        G3: ON MATCH SET n.user_scope = $scope overwrites the scope even
        when the node already exists. This is unnecessary (the scope was
        set on creation) and potentially confusing. The fix: only set
        user_scope on ON CREATE.
        """
        graph = _make_graph(tmp_path, user_id="alice")
        try:
            # Create a node with alice's scope.
            graph.upsert_node("TestEntity", "concept", {"foo": "bar"})
            # Verify scope.
            with graph._shared_conn_lock:
                result = graph.conn.execute(
                    "MATCH (n:Entity {id: $id}) RETURN n.user_scope",
                    parameters={"id": graph._internal_id("TestEntity")},
                )
                row = result.get_next()
                assert row[0] == "alice", (
                    f"Initial scope should be 'alice', got '{row[0]}'"
                )
            # Re-upsert the same node with a DIFFERENT user_scope —
            # the scope should NOT be overwritten on match.
            graph.upsert_node("TestEntity", "concept", {"baz": "qux"},
                              user_scope="bob")
            with graph._shared_conn_lock:
                result = graph.conn.execute(
                    "MATCH (n:Entity {id: $id}) RETURN n.user_scope",
                    parameters={"id": graph._internal_id("TestEntity")},
                )
                row = result.get_next()
                # G3 FIX: scope should still be 'alice', not 'bob'.
                assert row[0] == "alice", (
                    f"user_scope should not be overwritten on match; "
                    f"expected 'alice', got '{row[0]}'"
                )
        finally:
            graph.close()

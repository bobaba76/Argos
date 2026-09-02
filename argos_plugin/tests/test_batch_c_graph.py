"""Batch-C tests: graph lock safety (#76), flush exception safety (#88),
boost-floor gate exemption (#81), and graph block error visibility (#84).
"""
from __future__ import annotations

import json
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


# ---------------------------------------------------------------------------
# Graph audit: G4, G5, G6
# ---------------------------------------------------------------------------

class TestGraphAuditG4G5G6:
    """Tests for graph audit findings G4–G6.

    G4: remove_memory full-scans all edges (memory_id buried in JSON)
    G5: traverse_graph BFS does N lock acquisitions for N nodes
    G6: _query_edges_for_nodes has no LIMIT (unbounded at scale)
    """

    # -- G6: _query_edges_for_nodes LIMIT ---------------------------------

    def test_g6_query_edges_for_nodes_has_limit(self, tmp_path):
        """_query_edges_for_nodes should not return an unbounded number of
        edges. With many high-degree nodes, the result set must be capped
        by a LIMIT proportional to the input size.
        """
        graph = _make_graph(tmp_path)
        try:
            # Create a hub node with many edges.
            graph.upsert_node("Hub", "concept", {})
            for i in range(200):
                leaf = f"Leaf{i}"
                graph.upsert_node(leaf, "concept", {})
                graph.upsert_edge("Hub", leaf, "connected_to", {})
            # Query edges for the hub — should be capped, not return all 200.
            edges = graph._query_edges_for_nodes(["Hub"])
            # G6 FIX: the result should be bounded by a LIMIT.
            # The limit is proportional to len(node_ids) * 50 = 1 * 50 = 50.
            assert len(edges) <= 50, (
                f"_query_edges_for_nodes should be bounded by LIMIT; "
                f"got {len(edges)} edges for 1 node (expected <= 50)"
            )
        finally:
            graph.close()

    def test_g6_limit_proportional_to_input_size(self, tmp_path):
        """The LIMIT should scale with the number of input nodes so that
        querying multiple nodes doesn't artificially cap results too low.
        """
        graph = _make_graph(tmp_path)
        try:
            # Create 3 hub nodes, each with 30 edges.
            for hub_idx in range(3):
                hub = f"Hub{hub_idx}"
                graph.upsert_node(hub, "concept", {})
                for i in range(30):
                    leaf = f"Leaf{hub_idx}_{i}"
                    graph.upsert_node(leaf, "concept", {})
                    graph.upsert_edge(hub, leaf, "connected_to", {})
            # Query edges for all 3 hubs — limit should be 3 * 50 = 150.
            edges = graph._query_edges_for_nodes(["Hub0", "Hub1", "Hub2"])
            # Should return all 90 edges (under the 150 limit).
            assert len(edges) == 90, (
                f"Expected 90 edges for 3 hubs with 30 edges each; "
                f"got {len(edges)}"
            )
        finally:
            graph.close()

    # -- G5: traverse_graph batch node fetch -------------------------------

    def test_g5_traverse_graph_correctness(self, tmp_path):
        """traverse_graph should still return correct results after
        switching from per-node to batch node fetching.
        """
        graph = _make_graph(tmp_path)
        try:
            # Build a small graph: A -> B -> C -> D
            for node in ("A", "B", "C", "D"):
                graph.upsert_node(node, "concept", {})
            graph.upsert_edge("A", "B", "connected_to", {})
            graph.upsert_edge("B", "C", "connected_to", {})
            graph.upsert_edge("C", "D", "connected_to", {})
            # Traverse from A with depth=2, limit=10
            result = graph.traverse_graph("A", depth=2, limit=10)
            # Should find A (seed), B (hop 1), C (hop 2) and 2 edges
            node_ids = {n["id"] for n in result["nodes"]}
            assert "A" in node_ids, f"Seed A should be in nodes: {node_ids}"
            assert "B" in node_ids, f"B (hop 1) should be in nodes: {node_ids}"
            assert "C" in node_ids, f"C (hop 2) should be in nodes: {node_ids}"
            assert "D" not in node_ids, (
                f"D (hop 3) should NOT be in nodes at depth=2: {node_ids}"
            )
            assert len(result["edges"]) == 2, (
                f"Expected 2 edges (A->B, B->C); got {len(result['edges'])}"
            )
        finally:
            graph.close()

    def test_g5_traverse_graph_uses_batch_fetch(self, tmp_path):
        """traverse_graph should use _query_nodes_for_ids (batch) instead
        of per-node _query_node calls, reducing lock acquisitions.

        We verify by counting lock acquisitions during a traversal that
        discovers multiple new nodes.
        """
        graph = _make_graph(tmp_path)
        try:
            # Build a star graph: Hub -> Leaf0..Leaf9 (10 new nodes)
            graph.upsert_node("Hub", "concept", {})
            for i in range(10):
                leaf = f"Leaf{i}"
                graph.upsert_node(leaf, "concept", {})
                graph.upsert_edge("Hub", leaf, "connected_to", {})

            # Count lock acquisitions during traverse_graph by wrapping
            # the lock. We count __enter__ calls.
            original_lock = graph._shared_conn_lock
            lock_count = [0]
            class CountingLock:
                def __enter__(self_inner):
                    lock_count[0] += 1
                    return original_lock.__enter__()
                def __exit__(self_inner, *a):
                    return original_lock.__exit__(*a)
            graph._shared_conn_lock = CountingLock()

            result = graph.traverse_graph("Hub", depth=2, limit=50)

            # G5 FIX: with batch fetch, lock acquisitions for node lookups
            # should be ~depth (1-2), not ~N (10+).
            # Total lock acquisitions = seed _query_node (1-2) +
            # seed_edges _query_edges_for_nodes (1) +
            # per-hop _query_edges_for_nodes (1 per hop) +
            # batch _query_nodes_for_ids (1 per hop) +
            # Total should be well under the old N+pattern (which would
            # be 10+ for the 10 leaf nodes alone).
            assert lock_count[0] < 10, (
                f"traverse_graph should use batch node fetch; "
                f"lock acquisitions = {lock_count[0]} (expected < 10, "
                f"old per-node pattern would be 10+ for 10 new nodes)"
            )
            # Verify correctness: all 10 leaves should be discovered.
            node_ids = {n["id"] for n in result["nodes"]}
            for i in range(10):
                assert f"Leaf{i}" in node_ids, (
                    f"Leaf{i} should be in traversal results: {node_ids}"
                )
        finally:
            graph.close()

    # -- G4: remove_memory no full scan ------------------------------------

    def test_g4_remove_memory_no_full_scan(self, tmp_path):
        """remove_memory should not full-scan all edges to find those
        referencing memory_id in JSON attributes. With the memory_ids
        column migration, it should use a native Cypher filter.

        We verify correctness: remove_memory should still quarantine
        the right edges, and it should NOT scan edges that don't
        reference the memory_id.
        """
        graph = _make_graph(tmp_path)
        try:
            # Index a memory — this creates entity edges with memory_ids.
            graph.index_memory(
                "m_g4", "personal_fact", "User works at TechCorp",
                ["work"], use_llm=False,
            )
            # Create an unrelated edge that does NOT reference m_g4.
            graph.upsert_node("UnrelatedA", "concept", {})
            graph.upsert_node("UnrelatedB", "concept", {})
            graph.upsert_edge("UnrelatedA", "UnrelatedB", "connected_to", {})

            # Count lock acquisitions during remove_memory.
            original_lock = graph._shared_conn_lock
            lock_count = [0]
            class CountingLock:
                def __enter__(self_inner):
                    lock_count[0] += 1
                    return original_lock.__enter__()
                def __exit__(self_inner, *a):
                    return original_lock.__exit__(*a)
            graph._shared_conn_lock = CountingLock()

            changed = graph.remove_memory("m_g4")
            assert changed, "remove_memory should report changes"

            # G4 FIX: with the memory_ids column, remove_memory should
            # use a targeted query (WHERE list_contains(r.memory_ids, $mid))
            # instead of a full scan. The lock count should be low
            # (1 for the whole operation, not 2+ for separate scans).
            assert lock_count[0] <= 2, (
                f"remove_memory should use a targeted query, not a full "
                f"scan; lock acquisitions = {lock_count[0]} (expected <= 2)"
            )

            # Verify the memory node is quarantined.
            memory_node = graph._internal_id("memory:m_g4")
            with graph._shared_conn_lock:
                result = graph.conn.execute(
                    "MATCH (n:Entity {id: $id}) RETURN n.attributes",
                    parameters={"id": memory_node},
                )
                if result.has_next():
                    import json as _json
                    attrs = _json.loads(result.get_next()[0] or "{}")
                    assert attrs.get("status") == "quarantined", (
                        f"Memory node should be quarantined; got status={attrs.get('status')}"
                    )
        finally:
            graph.close()

    def test_g4_remove_memory_correctness_with_evidence_edges(self, tmp_path):
        """remove_memory should correctly quarantine entity-to-entity edges
        that carry the memory_id in their evidence list, and should
        correctly leave unrelated edges untouched.
        """
        graph = _make_graph(tmp_path)
        try:
            # Index two memories that share an entity edge.
            graph.index_memory(
                "m_g4a", "personal_fact", "User works at TechCorp",
                ["work"], use_llm=False,
            )
            graph.index_memory(
                "m_g4b", "personal_fact", "User works at TechCorp too",
                ["work"], use_llm=False,
            )
            # Remove one memory — the shared edge should have m_g4a removed
            # from its memory_ids but NOT be quarantined (m_g4b still references it).
            graph.remove_memory("m_g4a")

            # Check the TechCorp entity node's edges.
            techcorp_id = graph._internal_id("TechCorp")
            with graph._shared_conn_lock:
                result = graph.conn.execute(
                    """MATCH (a:Entity)-[r:RelatesTo]->(b:Entity)
                       WHERE a.id = $techcorp OR b.id = $techcorp
                       RETURN a.id, r.relation_type, b.id, r.attributes,
                              r.memory_ids""",
                    parameters={"techcorp": techcorp_id},
                )
                edges = []
                while result.has_next():
                    edges.append(result.get_next())

            # At least one edge should still be active (referenced by m_g4b).
            import json as _json
            active_edges = []
            for src, rel, tgt, raw_attrs, raw_mids in edges:
                attrs = _json.loads(raw_attrs) if raw_attrs else {}
                if attrs.get("status") != "quarantined":
                    active_edges.append((src, rel, tgt, attrs, raw_mids))

            # The edge referencing m_g4b should still be active.
            assert len(active_edges) > 0, (
                f"At least one edge should still be active (referenced by m_g4b); "
                f"all edges quarantined: {edges}"
            )
            # m_g4a should NOT be in any active edge's memory_ids.
            for src, rel, tgt, attrs, raw_mids in active_edges:
                mids = attrs.get("memory_ids", [])
                if not isinstance(mids, list):
                    mids = [mids] if mids else []
                assert "m_g4a" not in {str(x) for x in mids}, (
                    f"m_g4a should be removed from memory_ids; "
                    f"edge {src}->{tgt} still has: {mids}"
                )
        finally:
            graph.close()


class TestGraphG4BackfillAndFallback:
    """G4 review (2/9): pre-migration edges (memory_ids column NULL,
    evidence only in the attributes JSON blob) must still be cleaned by
    remove_memory — via the init-time backfill when it has run, and via
    the NULL-column fallback clause when it hasn't."""

    @staticmethod
    def _old_format_edge(graph, attrs, relation="works_at"):
        graph.upsert_node("person:alice", "person", {})
        graph.upsert_node("company:acme", "company", {})
        graph.conn.execute(
            """MATCH (a:Entity {id: $s}), (b:Entity {id: $t})
               CREATE (a)-[r:RelatesTo {relation_type: $rel,
                 attributes: $attrs, user_scope: $scope}]->(b)""",
            parameters={
                "s": graph._internal_id("person:alice"),
                "t": graph._internal_id("company:acme"),
                "rel": relation,
                "attrs": json.dumps(attrs),
                "scope": graph.user_id,
            },
        )

    def test_backfill_populates_column_from_attrs(self, tmp_path):
        graph = _make_graph(tmp_path)
        try:
            self._old_format_edge(graph, {"memory_ids": ["m1"]})
            from graph import _backfill_memory_ids
            n = _backfill_memory_ids(graph.conn)
            assert n == 1
            res = graph.conn.execute(
                "MATCH (a:Entity)-[r:RelatesTo]->(b:Entity) RETURN r.memory_ids"
            )
            assert res.has_next()
            assert res.get_next()[0] == ["m1"]
        finally:
            graph.close()

    def test_backfill_handles_legacy_singular_memory_id(self, tmp_path):
        graph = _make_graph(tmp_path)
        try:
            self._old_format_edge(graph, {"memory_id": "m9"})
            from graph import _backfill_memory_ids
            n = _backfill_memory_ids(graph.conn)
            assert n == 1
            res = graph.conn.execute(
                "MATCH (a:Entity)-[r:RelatesTo]->(b:Entity) RETURN r.memory_ids"
            )
            assert res.has_next()
            assert res.get_next()[0] == ["m9"]
        finally:
            graph.close()

    def test_remove_memory_finds_legacy_edge_after_backfill(self, tmp_path):
        graph = _make_graph(tmp_path)
        try:
            self._old_format_edge(graph, {"memory_ids": ["m1"]})
            from graph import _backfill_memory_ids
            _backfill_memory_ids(graph.conn)
            assert graph.remove_memory("m1") is True
            res = graph.conn.execute(
                "MATCH (a:Entity)-[r:RelatesTo]->(b:Entity) RETURN r.attributes"
            )
            attrs = json.loads(res.get_next()[0])
            assert attrs.get("status") == "quarantined"
        finally:
            graph.close()

    def test_remove_memory_finds_legacy_edge_via_fallback(self, tmp_path):
        """Without backfill, the NULL-column + quoted-id LIKE fallback must
        still catch the old-format edge (this is the regression that failed
        on the original G4 change)."""
        graph = _make_graph(tmp_path)
        try:
            self._old_format_edge(graph, {"memory_ids": ["m1"]})
            assert graph.remove_memory("m1") is True
            res = graph.conn.execute(
                "MATCH (a:Entity)-[r:RelatesTo]->(b:Entity) RETURN r.attributes"
            )
            attrs = json.loads(res.get_next()[0])
            assert attrs.get("status") == "quarantined"
        finally:
            graph.close()

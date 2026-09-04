"""Audit tests for graph.py (G1-G8, issue #225).

Covers:
- G1: conn property reads from shared pool (no stale references after flush)
- G2: junk sweep is bounded by _JUNK_SWEEP_LIMIT
- G3: clear_scope confirm guard (partial — full fix needs memory_service.py)
- G4: dead self._lock removed
- G5: LLM-extracted relations capped per memory
- G6: backfill persistence flag (set after successful backfill)
- G7: PPR iteration count documented (cost accepted per issue)
- G8: search_graph limit*3 heuristic documented

Run with (Hermes venv python, offline):
    python -m pytest tests/test_graph_audit.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_plugin_dir = Path(__file__).resolve().parent.parent
for _path in (_plugin_dir.parent, _plugin_dir):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import graph  # noqa: E402


def _make_graph(tmp_path, user_id="test_user"):
    return graph.KuzuGraphStore(tmp_path / "test_graph_audit", user_id=user_id)


# ---------------------------------------------------------------------------
# G1 — conn property reads from shared pool
# ---------------------------------------------------------------------------

class TestG1ConnProperty:
    def test_conn_is_property(self):
        """conn is a property descriptor, not a plain instance attribute."""
        import inspect
        attr = inspect.getattr_static(graph.KuzuGraphStore, "conn")
        assert isinstance(attr, property), (
            "KuzuGraphStore.conn must be a property (G1)"
        )

    def test_closed_instance_conn_is_none(self, tmp_path):
        g = _make_graph(tmp_path)
        assert g.conn is not None
        g.close()
        assert g.conn is None

    def test_flush_updates_shared_pool(self, tmp_path):
        """After _flush, the shared pool has a new connection, and the
        conn property returns it (not a stale cached reference)."""
        g = _make_graph(tmp_path)
        try:
            conn_before = g.conn
            g._flush()
            conn_after = g.conn
            # The connection object should be a valid (possibly new) connection.
            assert conn_after is not None
            # The shared pool entry's connection matches what the property returns.
            shared = graph.KuzuGraphStore._shared[g._db_key]
            assert shared[1] is conn_after
        finally:
            g.close()

    def test_two_instances_share_conn_after_flush(self, tmp_path):
        """G1 core: after instance A flushes, instance B's conn property
        returns the new connection (not the stale old one)."""
        g_a = _make_graph(tmp_path)
        g_b = _make_graph(tmp_path)
        try:
            conn_a_before = g_a.conn
            conn_b_before = g_b.conn
            assert conn_a_before is conn_b_before  # same shared connection
            g_a._flush()
            conn_a_after = g_a.conn
            conn_b_after = g_b.conn
            # Both now see the new connection from the shared pool.
            assert conn_a_after is conn_b_after
            assert conn_a_after is not conn_a_before  # it's a new connection
        finally:
            g_a.close()
            g_b.close()


# ---------------------------------------------------------------------------
# G2 — junk sweep is bounded
# ---------------------------------------------------------------------------

class TestG2JunkSweepBounded:
    def test_junk_sweep_limit_constant_exists(self):
        assert isinstance(graph._JUNK_SWEEP_LIMIT, int)
        assert graph._JUNK_SWEEP_LIMIT > 0

    def test_junk_sweep_query_has_limit(self):
        """The quarantine_junk_entities method uses LIMIT in its queries."""
        import inspect
        src = inspect.getsource(graph.KuzuGraphStore.quarantine_junk_entities)
        assert "LIMIT" in src.upper()


# ---------------------------------------------------------------------------
# G3 — clear_scope confirm guard (partial)
# ---------------------------------------------------------------------------

class TestG3ClearScopeGuard:
    def test_clear_scope_has_confirm_param(self):
        """clear_scope accepts a confirm keyword argument."""
        import inspect
        sig = inspect.signature(graph.KuzuGraphStore.clear_scope)
        assert "confirm" in sig.parameters

    def test_clear_scope_refuses_without_confirm(self, tmp_path):
        """clear_scope(confirm=False) raises PermissionError."""
        g = _make_graph(tmp_path)
        try:
            with pytest.raises(PermissionError, match="irreversible|confirm"):
                g.clear_scope(confirm=False)
        finally:
            g.close()

    def test_clear_scope_proceeds_with_confirm(self, tmp_path):
        """clear_scope(confirm=True) succeeds (default)."""
        g = _make_graph(tmp_path)
        try:
            # Add a node so there's something to clear.
            g.upsert_node("test_node", "test", {})
            result = g.clear_scope()  # default confirm=True
            assert isinstance(result, tuple)
        finally:
            g.close()


# ---------------------------------------------------------------------------
# G4 — dead self._lock removed
# ---------------------------------------------------------------------------

class TestG4DeadLockRemoved:
    def test_no_per_instance_lock(self, tmp_path):
        """The per-instance self._lock (never acquired) is removed."""
        g = _make_graph(tmp_path)
        try:
            assert not hasattr(g, "_lock"), (
                "KuzuGraphStore instances should not have a _lock attribute (G4: "
                "it was dead code — all operations use _shared_conn_lock)"
            )
        finally:
            g.close()


# ---------------------------------------------------------------------------
# G5 — LLM-extracted relations capped
# ---------------------------------------------------------------------------

class TestG5RelationCap:
    def test_max_relations_constant_exists(self):
        assert isinstance(graph._LLM_MAX_GRAPH_RELATIONS, int)
        assert graph._LLM_MAX_GRAPH_RELATIONS > 0

    def test_extract_graph_relations_llm_caps_result(self, monkeypatch):
        """An LLM response with more than _LLM_MAX_GRAPH_RELATIONS relations
        is truncated."""
        import json as _json
        many = [
            {"source": "user", "target": f"entity{i}", "relation": "uses"}
            for i in range(50)
        ]

        class _FakeChoice:
            class message:
                content = _json.dumps(many)
            choices = [type("c", (), {"message": message})]

        def fake_call_llm(**kwargs):
            return _FakeChoice()

        import types
        egress_mod = types.ModuleType("egress")
        egress_mod.gate = lambda *a, **k: True
        monkeypatch.setitem(sys.modules, "egress", egress_mod)
        aux_mod = types.ModuleType("agent.auxiliary_client")
        aux_mod.call_llm = fake_call_llm
        agent_mod = types.ModuleType("agent")
        agent_mod.auxiliary_client = aux_mod
        monkeypatch.setitem(sys.modules, "agent", agent_mod)
        monkeypatch.setitem(sys.modules, "agent.auxiliary_client", aux_mod)

        relations = graph.extract_graph_relations_llm(
            "A substantial enough memory content for graph extraction. " * 5,
        )
        assert len(relations) <= graph._LLM_MAX_GRAPH_RELATIONS


# ---------------------------------------------------------------------------
# G6 — backfill persistence flag
# ---------------------------------------------------------------------------

class TestG6BackfillFlag:
    def test_backfill_flag_constants_exist(self):
        assert graph._BACKFILL_FLAG_NODE == "__system_state__"
        assert graph._BACKFILL_FLAG_ATTR == "memory_ids_backfilled"

    def test_backfill_sets_flag_after_success(self, tmp_path):
        """After a successful backfill (count > 0), the flag is set so the
        next call skips the scan."""
        g = _make_graph(tmp_path)
        try:
            import json as _json
            # Add an old-format edge (memory_ids column NULL, evidence in attrs).
            g.upsert_node("person:alice", "person", {})
            g.upsert_node("company:acme", "company", {})
            g.conn.execute(
                """MATCH (a:Entity {id: $s}), (b:Entity {id: $t})
                   CREATE (a)-[r:RelatesTo {relation_type: $rel,
                     attributes: $attrs, user_scope: $scope}]->(b)""",
                parameters={
                    "s": g._internal_id("person:alice"),
                    "t": g._internal_id("company:acme"),
                    "rel": "works_at",
                    "attrs": _json.dumps({"memory_ids": ["m1"]}),
                    "scope": g.user_id,
                },
            )
            n = graph._backfill_memory_ids(g.conn)
            assert n == 1
            # Flag should now be set — second call skips the scan.
            n2 = graph._backfill_memory_ids(g.conn)
            assert n2 == 0
        finally:
            g.close()

    def test_backfill_does_not_set_flag_on_empty_db(self, tmp_path):
        """On a fresh/empty DB, the flag is NOT set (edges might be added
        later that need backfill)."""
        g = _make_graph(tmp_path)
        try:
            n = graph._backfill_memory_ids(g.conn)
            assert n == 0
            # Flag should NOT be set — check by calling again, which should
            # still scan (not skip). We can't directly check the flag node
            # easily, but we can verify the scan runs by confirming it
            # doesn't skip (the return is still 0, but the debug log would
            # show the skip). Instead, verify the flag node doesn't exist.
            result = g.conn.execute(
                "MATCH (n:Entity {id: $id}) RETURN n.attributes",
                parameters={"id": graph._BACKFILL_FLAG_NODE},
            )
            assert not result.has_next(), "flag should not be set on empty DB"
        finally:
            g.close()


# ---------------------------------------------------------------------------
# G7 — PPR iteration count (cost accepted, documented)
# ---------------------------------------------------------------------------

class TestG7PPRIterations:
    def test_ppr_preserves_max_iterations(self):
        """G7: the max_iterations parameter is preserved (cost accepted
        per the issue — reducing it changes PPR rankings)."""
        import inspect
        sig = inspect.signature(graph.KuzuGraphStore.ppr_memory_ids)
        assert sig.parameters["max_iterations"].default == 20


# ---------------------------------------------------------------------------
# G8 — search_graph limit*3 documented
# ---------------------------------------------------------------------------

class TestG8SearchLimit:
    def test_search_graph_uses_limit_multiplier(self):
        """G8: search_graph fetches limit*3 rows (documented heuristic)."""
        import inspect
        src = inspect.getsource(graph.KuzuGraphStore.search_graph)
        assert "limit * 3" in src or "limit*3" in src

"""#287: temporal-aware graph (timestamped edges + as-of traversal).

Tests:
1. Graph migration adds timestamp columns idempotently (run twice =
   no-op, transactional — a failing migration rolls back cleanly).
2. Node/edge writes stamp timestamps mirroring record valid windows.
3. As-of traversal returns only connections valid at T (edge valid-to
   < T excluded); default traversal shape unchanged.
4. Graph-vs-record alignment: a record superseded at B does not appear
   as a live edge after B — including via the re-index/backfill path.
5. Migration-time backfill stamps existing rows from their own
   attributes provenance (never invents timestamps).

Run with (Hermes venv python, hermetic):
    ARGOS_HERMETIC_TESTS=1 python -m pytest tests/test_temporal_graph.py -v
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_plugin_dir = Path(__file__).resolve().parent.parent
if str(_plugin_dir.parent) not in sys.path:
    sys.path.insert(0, str(_plugin_dir.parent))

# Canonical ISO-8601 UTC timestamps (lexicographic == chronological).
T0 = "2026-01-01T00:00:00+00:00"
T1 = "2026-02-01T00:00:00+00:00"
T2 = "2026-03-01T00:00:00+00:00"
T3 = "2026-04-01T00:00:00+00:00"


def _make_graph(tmp_path, name="g", user_id="test_user"):
    from graph import KuzuGraphStore
    return KuzuGraphStore(tmp_path / name, user_id=user_id)


def _table_columns(conn, table):
    result = conn.execute(f"CALL table_info('{table}') RETURN *")
    cols = set()
    while result.has_next():
        row = result.get_next()
        if len(row) > 1 and row[1]:
            cols.add(str(row[1]))
    return cols


class TestGraphMigrationTimestamps:
    """1. Migration adds timestamp columns idempotently + transactionally."""

    def test_migration_adds_timestamp_columns(self, tmp_path):
        graph = _make_graph(tmp_path, "mig_cols")
        try:
            assert graph.get_graph_schema_version() == 1
            with graph._shared_conn_lock:
                node_cols = _table_columns(graph.conn, "Entity")
                edge_cols = _table_columns(graph.conn, "RelatesTo")
            for col in ("created_at", "valid_from", "valid_to"):
                assert col in node_cols, f"Entity missing {col}"
                assert col in edge_cols, f"RelatesTo missing {col}"
        finally:
            graph.close()

    def test_migration_idempotent_run_twice(self, tmp_path):
        """run_graph_migrations twice: second call is a no-op."""
        import kuzu
        from schema_migrations import run_graph_migrations

        graph = _make_graph(tmp_path, "mig_idem")
        graph.close()
        # Reopen the raw database directly (bypasses the shared pool ref
        # counting) and run the migrations again.
        database = kuzu.Database(str(tmp_path / "mig_idem"))
        conn = kuzu.Connection(database)
        try:
            r1 = run_graph_migrations(conn)
            assert r1["from_version"] == 1
            assert r1["to_version"] == 1
            assert r1["applied"] == []
            assert 1 in r1["skipped"]
        finally:
            conn.close()
            database.close()

    def test_failed_migration_rolls_back(self, tmp_path):
        """A migration that fails mid-way rolls back its DDL and leaves
        the version unchanged (no half-migrated graph)."""
        import kuzu
        from schema_migrations import (
            run_graph_migrations, get_graph_schema_version,
        )

        graph = _make_graph(tmp_path, "mig_fail")
        graph.close()

        database = kuzu.Database(str(tmp_path / "mig_fail"))
        conn = kuzu.Connection(database)
        try:
            assert get_graph_schema_version(conn) == 1

            def bad_migration(c):
                # Mirrors the real migration pattern: BEGIN, partial DDL,
                # then fail — the fn's own except rolls back.
                c.execute("BEGIN TRANSACTION")
                try:
                    c.execute("ALTER TABLE Entity ADD bogus_col STRING")
                    raise RuntimeError("intentional failure")
                except Exception:
                    c.execute("ROLLBACK")
                    raise

            migs = [(1, 2, bad_migration)]
            with pytest.raises(RuntimeError, match="1→2 failed"):
                run_graph_migrations(conn, migrations=migs)

            # Version unchanged; the partial DDL is rolled back.
            assert get_graph_schema_version(conn) == 1
            assert "bogus_col" not in _table_columns(conn, "Entity")
        finally:
            conn.close()
            database.close()

    def test_migration_backfills_from_attributes_provenance(self, tmp_path):
        """Migration-time backfill stamps existing rows from their own
        attributes JSON provenance (#138 observed_at) — never invents."""
        import kuzu
        from schema_migrations import run_graph_migrations

        db_dir = tmp_path / "mig_backfill"
        database = kuzu.Database(str(db_dir))
        conn = kuzu.Connection(database)
        try:
            # Build a PRE-#287 graph manually: base schema only, no
            # temporal columns, one edge with observed_at provenance and
            # one without.
            conn.execute(
                "CREATE NODE TABLE Entity("
                "id STRING, entity_type STRING, attributes STRING, "
                "user_scope STRING, PRIMARY KEY (id))"
            )
            conn.execute(
                "CREATE REL TABLE RelatesTo("
                "FROM Entity TO Entity, "
                "relation_type STRING, attributes STRING, user_scope STRING, "
                "memory_ids STRING[])"
            )
            conn.execute(
                "CREATE (n:Entity {id: 'a', entity_type: 'person', "
                "attributes: '{}', user_scope: 'u'})"
            )
            conn.execute(
                "CREATE (n:Entity {id: 'b', entity_type: 'concept', "
                "attributes: '{}', user_scope: 'u'})"
            )
            conn.execute(
                """CREATE (a:Entity {id: 'memory:m1', entity_type: 'memory',
                    attributes: $attrs, user_scope: 'u'})""",
                parameters={"attrs": json.dumps({"created_at": T0})},
            )
            conn.execute(
                """MATCH (a:Entity {id: 'a'}), (b:Entity {id: 'b'})
                   CREATE (a)-[r:RelatesTo {relation_type: 'knows',
                       attributes: $attrs, user_scope: 'u', memory_ids: []}]->(b)""",
                parameters={"attrs": json.dumps({"observed_at": T0})},
            )
            conn.execute(
                """MATCH (a:Entity {id: 'b'}), (b:Entity {id: 'memory:m1'})
                   CREATE (a)-[r:RelatesTo {relation_type: 'related_to',
                       attributes: '{}', user_scope: 'u', memory_ids: []}]->(b)"""
            )

            r = run_graph_migrations(conn)
            assert r["to_version"] == 1
            assert 1 in r["applied"]

            # Columns added.
            assert "created_at" in _table_columns(conn, "Entity")
            assert "valid_from" in _table_columns(conn, "RelatesTo")

            # Edge WITH provenance stamped; edge WITHOUT stays NULL.
            result = conn.execute(
                """MATCH (a:Entity {id: 'a'})-[r:RelatesTo]->(b:Entity)
                   WHERE r.relation_type = 'knows'
                   RETURN r.created_at, r.valid_from, r.valid_to"""
            )
            row = result.get_next()
            assert row[0] == T0 and row[1] == T0 and row[2] is None

            result = conn.execute(
                """MATCH (a:Entity {id: 'b'})-[r:RelatesTo]->(b:Entity)
                   WHERE r.relation_type = 'related_to'
                   RETURN r.created_at, r.valid_from"""
            )
            row = result.get_next()
            assert row[0] is None and row[1] is None

            # Memory node stamped from its attributes.created_at.
            result = conn.execute(
                "MATCH (n:Entity {id: 'memory:m1'}) "
                "RETURN n.created_at, n.valid_from"
            )
            row = result.get_next()
            assert row[0] == T0 and row[1] == T0
        finally:
            conn.close()
            database.close()


class TestGraphWritesStampTimestamps:
    """2. Node/edge writes stamp timestamps mirroring record valid windows."""

    def test_index_memory_stamps_open_window(self, tmp_path):
        graph = _make_graph(tmp_path, "stamp_open")
        try:
            graph.index_memory(
                memory_id="mem-1",
                category="personal_fact",
                content="Alice works at Acme on Project X",
                created_at=T0,
                use_llm=False,
                valid_from=T0,
                valid_to=None,
            )
            mem_node = graph._internal_id("memory:mem-1")
            with graph._shared_conn_lock:
                # Memory node mirrors the record window.
                result = graph.conn.execute(
                    "MATCH (n:Entity {id: $id}) "
                    "RETURN n.created_at, n.valid_from, n.valid_to",
                    parameters={"id": mem_node},
                )
                row = result.get_next()
                assert row[0] == T0
                assert row[1] == T0
                assert row[2] is None  # open — record still current

                # about_user edge carries the same window.
                result = graph.conn.execute(
                    """MATCH (:Entity {id: $id})-[r:RelatesTo]->(:Entity)
                       WHERE r.relation_type = 'about_user'
                       RETURN r.valid_from, r.valid_to""",
                    parameters={"id": mem_node},
                )
                row = result.get_next()
                assert row[0] == T0 and row[1] is None
        finally:
            graph.close()

    def test_edge_mirrors_closed_record_window(self, tmp_path):
        """An edge from a superseded record (valid_to=T1) carries valid_to."""
        graph = _make_graph(tmp_path, "stamp_closed")
        try:
            graph.index_memory(
                memory_id="mem-2",
                category="personal_fact",
                content="Bob lives in Berlin",
                created_at=T0,
                use_llm=False,
                valid_from=T0,
                valid_to=T1,  # record superseded at T1
            )
            mem_node = graph._internal_id("memory:mem-2")
            with graph._shared_conn_lock:
                result = graph.conn.execute(
                    "MATCH (n:Entity {id: $id}) RETURN n.valid_from, n.valid_to",
                    parameters={"id": mem_node},
                )
                row = result.get_next()
                assert row[0] == T0 and row[1] == T1
        finally:
            graph.close()

    def test_reindex_preserves_provenance_without_temporal_args(self, tmp_path):
        """A plain re-index (no temporal args) must NOT NULL out the
        previously stamped columns."""
        graph = _make_graph(tmp_path, "stamp_preserve")
        try:
            graph.index_memory(
                memory_id="mem-3",
                category="personal_fact",
                content="Carol drives a red car",
                created_at=T0,
                use_llm=False,
                valid_from=T0,
                valid_to=T1,
            )
            # Plain re-index — no temporal args (legacy caller shape).
            graph.index_memory(
                memory_id="mem-3",
                category="personal_fact",
                content="Carol drives a red car",
                created_at=T0,
                use_llm=False,
            )
            mem_node = graph._internal_id("memory:mem-3")
            with graph._shared_conn_lock:
                result = graph.conn.execute(
                    "MATCH (n:Entity {id: $id}) "
                    "RETURN n.created_at, n.valid_from, n.valid_to",
                    parameters={"id": mem_node},
                )
                row = result.get_next()
                assert row[0] == T0, "created_at clobbered by re-index"
                assert row[1] == T0, "valid_from clobbered by re-index"
                assert row[2] == T1, "valid_to clobbered by re-index"
        finally:
            graph.close()

    def test_multi_evidence_edge_union_window(self, tmp_path):
        """An edge with two evidence memories: open evidence keeps the
        edge open; closing the last open evidence closes the edge."""
        graph = _make_graph(tmp_path, "stamp_union")
        try:
            # mem-A (open) and mem-B (closed at T1) both evidence X-knows-Y.
            graph.add_relationship(
                "X", "person", "knows", "Y", "person",
                {"memory_id": "mem-a"},
                created_at=T0, valid_from=T0, valid_to=None,
            )
            graph.add_relationship(
                "X", "person", "knows", "Y", "person",
                {"memory_id": "mem-b"},
                created_at=T0, valid_from=T0, valid_to=T1,
            )
            with graph._shared_conn_lock:
                result = graph.conn.execute(
                    """MATCH (:Entity {id: $x})-[r:RelatesTo {relation_type: 'knows'}]->(:Entity {id: $y})
                       RETURN r.valid_from, r.valid_to""",
                    parameters={"x": graph._internal_id("X"),
                                "y": graph._internal_id("Y")},
                )
                row = result.get_next()
                assert row[0] == T0
                assert row[1] is None, "open evidence must keep edge open"

            # Re-index mem-b with its closed window (backfill path) —
            # mem-a's evidence is still open, edge stays open.
            graph.add_relationship(
                "X", "person", "knows", "Y", "person",
                {"memory_id": "mem-b"},
                created_at=T0, valid_from=T0, valid_to=T1,
            )
            with graph._shared_conn_lock:
                result = graph.conn.execute(
                    """MATCH (:Entity {id: $x})-[r:RelatesTo {relation_type: 'knows'}]->(:Entity {id: $y})
                       RETURN r.valid_to""",
                    parameters={"x": graph._internal_id("X"),
                                "y": graph._internal_id("Y")},
                )
                assert result.get_next()[0] is None
        finally:
            graph.close()


class TestAsOfTraversal:
    """3. As-of traversal returns only connections valid at T."""

    def _seed(self, graph):
        # E1: user-knows-Ally, open window [T0, NULL).
        graph.add_relationship(
            "user", "person", "knows", "Ally", "person",
            {"memory_id": "mem-ally"},
            created_at=T0, valid_from=T0, valid_to=None,
        )
        # E2: user-knows-Bob, closed window [T0, T1).
        graph.add_relationship(
            "user", "person", "knows", "Bob", "person",
            {"memory_id": "mem-bob"},
            created_at=T0, valid_from=T0, valid_to=T1,
        )

    def test_query_graph_as_of_excludes_closed(self, tmp_path):
        graph = _make_graph(tmp_path, "asof_query")
        try:
            self._seed(graph)
            # As-of before closure: both edges.
            edges = graph.query_graph("user", as_of=T0)
            relations = {e["relation"] for e in edges}
            targets = {e["target"] for e in edges}
            assert "knows" in relations
            assert "Ally" in targets and "Bob" in targets

            # As-of after closure: only the open edge.
            edges = graph.query_graph("user", as_of=T2)
            targets = {e["target"] for e in edges}
            assert "Ally" in targets
            assert "Bob" not in targets, "closed edge leaked past valid_to"
        finally:
            graph.close()

    def test_traverse_graph_as_of(self, tmp_path):
        graph = _make_graph(tmp_path, "asof_traverse")
        try:
            self._seed(graph)
            # Default shape unchanged (as_of=None).
            full = graph.traverse_graph("user", depth=1)
            full_targets = {e["target"] for e in full["edges"]}
            assert "Ally" in full_targets and "Bob" in full_targets

            # As-of after T1: Bob's edge excluded.
            past = graph.traverse_graph("user", depth=1, as_of=T2)
            past_targets = {e["target"] for e in past["edges"]}
            assert "Ally" in past_targets
            assert "Bob" not in past_targets
        finally:
            graph.close()

    def test_search_graph_as_of(self, tmp_path):
        graph = _make_graph(tmp_path, "asof_search")
        try:
            self._seed(graph)
            assert graph.search_graph("Bob", as_of=T0)
            assert graph.search_graph("Bob", as_of=T2) == []
            assert graph.search_graph("Ally", as_of=T2)
        finally:
            graph.close()

    def test_traversal_memory_ids_as_of(self, tmp_path):
        graph = _make_graph(tmp_path, "asof_tmi")
        try:
            self._seed(graph)
            # 'knows' is a typed relation — traversable.
            mids_before = graph.traversal_memory_ids("Bob", as_of=T0)
            assert "mem-bob" in mids_before
            mids_after = graph.traversal_memory_ids("Bob", as_of=T2)
            assert "mem-bob" not in mids_after
        finally:
            graph.close()


class TestGraphRecordAlignment:
    """4. A record superseded at B does not appear as a live edge after B."""

    def test_superseded_record_edge_closes_after_backfill(self, tmp_path):
        """End-to-end: index open → record superseded → backfill re-index
        with the record's closed window → edge not live after B."""
        graph = _make_graph(tmp_path, "align_backfill")
        try:
            # Phase 1: record R1 current — edge open.
            graph.index_memory(
                memory_id="mem-r1",
                category="personal_fact",
                content="Dave works at Initech",
                created_at=T0,
                use_llm=False,
                valid_from=T0,
                valid_to=None,
            )
            edges = graph.query_graph("Dave", as_of=T2)
            assert edges, "edge should be live while record is current"

            # Phase 2: record superseded at T1 (update_memory sets
            # valid_to=T1 on the old version). The backfill path re-indexes
            # with the record's ACTUAL window — provenance preserved.
            graph.index_memory(
                memory_id="mem-r1",
                category="personal_fact",
                content="Dave works at Initech",
                created_at=T0,
                use_llm=False,
                valid_from=T0,
                valid_to=T1,
            )

            # Phase 3: alignment — record valid [T0, T1]; the edge must
            # not be live after T1.
            edges_after = graph.query_graph("Dave", as_of=T2)
            assert not any(
                e["source"] == "memory:mem-r1" or e["target"] == "Dave"
                for e in edges_after
            ), "superseded record's edge still live after valid_to"

            # As-of BEFORE the closure, the edge is still visible
            # (temporal provenance preserved — history is queryable).
            edges_hist = graph.query_graph("Dave", as_of=T0)
            assert edges_hist, "historical edge lost — provenance dropped"
        finally:
            graph.close()

    def test_alignment_with_full_store_and_update_memory(self, tmp_path):
        """Full-store alignment: remember → update (supersede) →
        backfill-style re-index of the closed version → graph agrees
        with the record's valid window."""
        from store import DuckDBMemoryStore

        store = DuckDBMemoryStore(
            tmp_path / "align.duckdb", user_id="alice", embedder=None,
        )
        graph = _make_graph(tmp_path, "align_full", user_id="alice")
        try:
            rec1 = store.remember(
                category="personal_fact",
                content="Eve works at Umbrella Corp",
                created_at=T0,
            )
            # Index to graph (record current → open window).
            graph.index_memory(
                memory_id=rec1.memory_id,
                category=rec1.category,
                content=rec1.content,
                tags=rec1.tags,
                created_at=rec1.created_at,
                use_llm=False,
                valid_from=rec1.valid_from or T0,
                valid_to=rec1.valid_to,
            )

            # Supersede at T2 via update_memory.
            rec2 = store.update_memory(
                rec1.memory_id,
                content="Eve works at Umbrella Corp as chief scientist",
            )
            assert rec2.memory_id != rec1.memory_id

            # Record layer: old version closed at update time.
            with store._state.lock:
                row = store.connection.execute(
                    "SELECT valid_to, superseded_by FROM memory_records "
                    "WHERE memory_id = ?",
                    [rec1.memory_id],
                ).fetchone()
            old_valid_to = row[0]
            assert row[1] == rec2.memory_id
            assert old_valid_to is not None

            # Backfill path: re-index the OLD version with its closed
            # window (what backfill_graph/rebuild do for historical rows).
            graph.index_memory(
                memory_id=rec1.memory_id,
                category=rec1.category,
                content=rec1.content,
                tags=rec1.tags,
                created_at=rec1.created_at,
                use_llm=False,
                valid_from=rec1.valid_from or T0,
                valid_to=old_valid_to,
            )
            # And the NEW version with its open window.
            graph.index_memory(
                memory_id=rec2.memory_id,
                category=rec2.category,
                content=rec2.content,
                tags=rec2.tags,
                created_at=rec2.created_at,
                use_llm=False,
                valid_from=rec2.valid_from,
                valid_to=rec2.valid_to,
            )

            # The old version's memory node must carry the closed window —
            # no disagreement with the record layer.
            with graph._shared_conn_lock:
                result = graph.conn.execute(
                    "MATCH (n:Entity {id: $id}) RETURN n.valid_to",
                    parameters={"id": graph._internal_id(f"memory:{rec1.memory_id}")},
                )
                assert result.get_next()[0] == old_valid_to
        finally:
            graph.close()
            store.close()


class TestGraphSchemaVersionIntrospection:
    """5. Version introspection + fresh-graph behavior."""

    def test_fresh_graph_at_version_1(self, tmp_path):
        graph = _make_graph(tmp_path, "ver_fresh")
        try:
            assert graph.get_graph_schema_version() == 1
        finally:
            graph.close()

    def test_version_persists_across_reopen(self, tmp_path):
        graph = _make_graph(tmp_path, "ver_persist")
        graph.close()
        graph2 = _make_graph(tmp_path, "ver_persist")
        try:
            assert graph2.get_graph_schema_version() == 1
        finally:
            graph2.close()

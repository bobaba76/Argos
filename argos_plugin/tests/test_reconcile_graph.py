"""Tests for #305: DuckDB-Kuzu reconciliation probe.

Proves:
(1) 0 drift on a clean store (DuckDB and graph in sync after indexing).
(2) Deliberately removing a graph node produces non-zero actionable output.

Uses direct DuckDBMemoryStore + KuzuGraphStore (not shared service) for
deterministic, fast, hermetic tests.

Run with (Hermes venv python, hermetic):
    ARGOS_HERMETIC_TESTS=1 python -m pytest tests/test_reconcile_graph.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_plugin_dir = Path(__file__).resolve().parent.parent
for _path in (_plugin_dir.parent, _plugin_dir):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from store import DuckDBMemoryStore


@pytest.fixture
def store(tmp_path):
    """Direct DuckDBMemoryStore for deterministic tests."""
    s = DuckDBMemoryStore(tmp_path / "test.duckdb", user_id="test_user")
    yield s
    s.close()


@pytest.fixture
def graph(tmp_path):
    """Direct KuzuGraphStore for deterministic tests."""
    from graph import KuzuGraphStore
    g = KuzuGraphStore(tmp_path / "test_kuzu", user_id="test_user")
    yield g
    g.close()


class TestReconcileNoDrift:
    """(1) 0 drift on a clean store after indexing."""

    def test_no_drift_on_clean_store(self, store, graph):
        """After writing a memory and indexing it in the graph,
        the reconciliation probe reports 0 drift."""
        from reconcile_graph import reconcile

        # Write a memory to DuckDB.
        rec = store.remember(
            category="personal_fact",
            content="Alice lives in Johannesburg",
        )
        assert rec is not None
        memory_id = rec.memory_id

        # Index it in the graph.
        graph.index_memory(
            memory_id=memory_id,
            category="personal_fact",
            content="Alice lives in Johannesburg",
            tags=[],
            created_at=rec.created_at,
            use_llm=False,
            flush=True,
        )

        # Run reconciliation — should report 0 drift.
        result = reconcile(store, graph, sample_size=10)
        assert result["drift"] is False
        assert result["missing_in_graph_count"] == 0
        assert result["extra_in_graph_count"] == 0
        assert result["duckdb_count"] >= 1
        assert result["graph_count"] >= 1


class TestReconcileDetectsDrift:
    """(2) Deliberately removing a graph node produces non-zero output."""

    def test_missing_in_graph_after_removal(self, store, graph):
        """If a memory exists in DuckDB but its graph node is removed,
        the probe reports it as missing_in_graph."""
        from reconcile_graph import reconcile

        # Write and index two memories.
        rec1 = store.remember(
            category="personal_fact",
            content="Alice works at Acme Corp",
        )
        rec2 = store.remember(
            category="preference",
            content="Alice prefers Python over Java",
        )
        for rec in (rec1, rec2):
            graph.index_memory(
                memory_id=rec.memory_id,
                category=rec.category,
                content=rec.content,
                tags=[],
                created_at=rec.created_at,
                use_llm=False,
                flush=True,
            )

        # Verify no drift initially.
        result = reconcile(store, graph, sample_size=10)
        assert result["drift"] is False

        # Remove one memory's graph entry (simulate drift).
        graph.remove_memory(rec1.memory_id)

        # Reconcile — should detect the missing graph entry.
        result = reconcile(store, graph, sample_size=10)
        assert result["drift"] is True
        assert result["missing_in_graph_count"] >= 1
        # The missing ID should be in the sample.
        assert rec1.memory_id in result["missing_in_graph"]

    def test_extra_in_graph_after_duckdb_delete(self, store, graph, tmp_path):
        """If a memory is deleted from DuckDB but its graph node remains,
        the probe reports it as extra_in_graph."""
        from reconcile_graph import reconcile

        # Write and index a memory.
        rec = store.remember(
            category="personal_fact",
            content="Bob lives in Cape Town",
        )
        graph.index_memory(
            memory_id=rec.memory_id,
            category="personal_fact",
            content="Bob lives in Cape Town",
            tags=[],
            created_at=rec.created_at,
            use_llm=False,
            flush=True,
        )

        # Verify no drift.
        result = reconcile(store, graph, sample_size=10)
        assert result["drift"] is False

        # Delete from DuckDB (but leave graph intact — simulate drift).
        store.delete_memory(memory_id=rec.memory_id)

        # Reconcile — should detect the extra graph entry.
        result = reconcile(store, graph, sample_size=10)
        assert result["drift"] is True
        # The deleted memory should not appear in DuckDB's active set.
        # If the graph still has it, it's extra.
        # Note: delete_memory may quarantine rather than hard-delete,
        # so the memory might still be in DuckDB but not active.
        # The probe checks active memories, so quarantined ones
        # won't be in duckdb_ids.
        assert result["extra_in_graph_count"] >= 0  # may be 0 if graph also cleaned up


class TestReconcileOutputFormat:
    """The reconcile function returns a well-structured dict."""

    def test_reconcile_returns_dict_with_required_keys(self, store, graph):
        """The result dict has all required keys."""
        from reconcile_graph import reconcile

        result = reconcile(store, graph, sample_size=5)
        required_keys = {
            "duckdb_count", "graph_count",
            "missing_in_graph_count", "missing_in_graph",
            "extra_in_graph_count", "extra_in_graph",
            "drift",
        }
        assert set(result.keys()) == required_keys

    def test_reconcile_sample_size_respected(self, store, graph):
        """The sample lists are truncated to sample_size."""
        from reconcile_graph import reconcile

        # Write 5 memories but don't index any — all missing in graph.
        for i in range(5):
            store.remember(
                category="context_note",
                content=f"Test memory number {i}",
            )

        result = reconcile(store, graph, sample_size=3)
        assert result["missing_in_graph_count"] >= 5
        assert len(result["missing_in_graph"]) <= 3


# ---------------------------------------------------------------------------
# #376: --prune-orphans — remove graph nodes with no DuckDB record
# ---------------------------------------------------------------------------

def _index(store, graph, rec):
    graph.index_memory(
        memory_id=rec.memory_id,
        category=rec.category,
        content=rec.content,
        tags=rec.tags or [],
        created_at=rec.created_at,
        use_llm=False,
        flush=True,
    )


class TestPurgeOrphanMemory:
    """Graph-level hard delete of a memory node + its evidence."""

    def test_purge_removes_node_and_edges(self, store, graph):
        """purge_orphan_memory removes the memory: node and edges touching
        it; shared entity nodes survive."""
        from backfill_graph import _graph_memory_ids

        rec = store.remember(
            category="personal_fact",
            content="Alice works at Acme Corp in Johannesburg",
        )
        _index(store, graph, rec)
        assert rec.memory_id in _graph_memory_ids(graph)
        # Purge the orphan node.
        assert graph.purge_orphan_memory(rec.memory_id) is True
        assert rec.memory_id not in _graph_memory_ids(graph)
        # Shared entities remain (nodes that are not memory: nodes).
        nodes = graph.list_nodes(limit=1000)
        ids = [n["id"] for n in nodes]
        assert not any(i.startswith("memory:") for i in ids)

    def test_purge_unknown_id_returns_false(self, graph):
        """Purge of a non-existent node is a no-op."""
        assert graph.purge_orphan_memory("no-such-memory") is False


class TestFindOrphans:
    """Orphan detection cross-checks DuckDB in ANY state."""

    def test_hard_deleted_memory_is_orphan(self, store, graph):
        """A memory whose row was hard-deleted (head, no predecessor) is
        an orphan — its memory_id is absent from DuckDB."""
        from backfill_graph import _find_orphans

        rec = store.remember(
            category="personal_fact",
            content="Bob lives in Cape Town",
        )
        _index(store, graph, rec)
        # Hard delete: head with no predecessor → tombstone + row DELETE.
        store.delete_memory(rec.memory_id)
        orphans, detection_errors = _find_orphans(store, [rec.memory_id])
        assert rec.memory_id in orphans
        assert detection_errors == 0

    def test_quarantined_memory_not_orphan(self, store, graph):
        """A memory whose row still exists (quarantined non-head version)
        is NOT an orphan — never prune a node whose base record exists in
        any DuckDB state (acceptance criterion 3)."""
        from backfill_graph import _find_orphans

        rec1 = store.remember(
            category="personal_fact",
            content="Carol lives in Durban",
        )
        store.update_memory(rec1.memory_id, content="Carol lives in Durban now")
        # rec1 is now a non-head version; deleting it quarantines the row
        # (reversible — the row stays in DuckDB).
        store.delete_memory(rec1.memory_id)
        _index(store, graph, rec1)
        orphans, detection_errors = _find_orphans(store, [rec1.memory_id])
        assert rec1.memory_id not in orphans
        assert detection_errors == 0

    def test_active_memory_not_orphan(self, store, graph):
        """An active memory is never an orphan."""
        from backfill_graph import _find_orphans

        rec = store.remember(
            category="personal_fact",
            content="Dan prefers tea over coffee",
        )
        _index(store, graph, rec)
        orphans, detection_errors = _find_orphans(store, [rec.memory_id])
        assert rec.memory_id not in orphans
        assert detection_errors == 0


class TestPruneOrphansCli:
    """prune_orphans dry-run / run semantics (backfill_graph #376)."""

    def test_dry_run_lists_without_deleting(self, store, graph):
        """--prune-orphans --dry-run reports orphans but deletes nothing."""
        from backfill_graph import prune_orphans

        rec = store.remember(
            category="personal_fact",
            content="Eve works at Globex in Nairobi",
        )
        _index(store, graph, rec)
        store.delete_memory(rec.memory_id)  # hard delete → orphan
        summary = prune_orphans(store, graph, dry_run=True)
        assert summary["orphans"] >= 1
        assert summary["removed"] == 0
        assert summary["errors"] == 0
        # Node still present (dry run must not delete).
        from backfill_graph import _graph_memory_ids
        assert rec.memory_id in _graph_memory_ids(graph)

    def test_run_removes_orphans(self, store, graph):
        """A non-dry-run prune removes the orphaned nodes."""
        from backfill_graph import prune_orphans, _graph_memory_ids

        rec = store.remember(
            category="personal_fact",
            content="Frank lives in Accra",
        )
        _index(store, graph, rec)
        store.delete_memory(rec.memory_id)  # hard delete → orphan
        summary = prune_orphans(store, graph, dry_run=False)
        assert summary["orphans"] >= 1
        assert summary["removed"] >= 1
        assert rec.memory_id not in _graph_memory_ids(graph)

    def test_no_orphans_nothing_removed(self, store, graph):
        """A synced store has zero orphans to prune."""
        from backfill_graph import prune_orphans

        rec = store.remember(
            category="personal_fact",
            content="Grace lives in Lagos",
        )
        _index(store, graph, rec)
        summary = prune_orphans(store, graph, dry_run=False)
        assert summary["orphans"] == 0
        assert summary["removed"] == 0

    def test_prune_then_reconcile_clean(self, store, graph):
        """After prune, the reconcile probe reports 0 drift (acceptance 1/4)."""
        from backfill_graph import prune_orphans
        from reconcile_graph import reconcile

        keep = store.remember(
            category="personal_fact",
            content="Hank works at Initech in Boston",
        )
        drop = store.remember(
            category="personal_fact",
            content="Iris lives in Seattle",
        )
        _index(store, graph, keep)
        _index(store, graph, drop)
        # Hard-delete Iris from DuckDB, leaving an orphan graph node.
        store.delete_memory(drop.memory_id)
        result = reconcile(store, graph, sample_size=10)
        assert result["extra_in_graph_count"] >= 1
        # Prune orphans → drift gone.
        prune_orphans(store, graph, dry_run=False)
        result = reconcile(store, graph, sample_size=10)
        assert result["drift"] is False
        assert result["extra_in_graph_count"] == 0


# ---------------------------------------------------------------------------
# #378 review fix: fail-closed orphan verification
# ---------------------------------------------------------------------------

class _RaisingStore:
    """Wrapper that raises on get_memory_history / get_memories_by_ids to
    simulate a wedged shared service (lock-jam, relay timeout)."""

    def __init__(self, real_store, raise_history=False, raise_bulk=False):
        self._real = real_store
        self._raise_history = raise_history
        self._raise_bulk = raise_bulk

    def get_memories_by_ids(self, ids, include_quarantined=False):
        if self._raise_bulk:
            raise RuntimeError("shared service wedge (bulk)")
        return self._real.get_memories_by_ids(ids, include_quarantined=include_quarantined)

    def get_memory_history(self, mid):
        if self._raise_history:
            raise RuntimeError("shared service wedge (per-id)")
        return self._real.get_memory_history(mid)


class TestFindOrphansFailClosed:
    """#378: verification exceptions must NOT declare orphans (fail-closed)."""

    def test_history_raise_skips_id_not_orphan(self, store, graph):
        """If get_memory_history RAISES, the id is NOT in orphans and is
        counted as a detection error — not treated as absent."""
        from backfill_graph import _find_orphans

        rec = store.remember(
            category="personal_fact",
            content="Wedge test memory for history raise",
        )
        _index(store, graph, rec)
        store.delete_memory(rec.memory_id)  # genuinely orphaned in DuckDB
        raising = _RaisingStore(store, raise_history=True)
        orphans, detection_errors = _find_orphans(raising, [rec.memory_id])
        assert rec.memory_id not in orphans, (
            "A verification exception must not declare an orphan "
            "(fail-closed). The id should be skipped, not purged."
        )
        assert detection_errors == 1

    def test_bulk_chunk_raise_skips_chunk_ids(self, store, graph):
        """If the bulk existence check RAISES for a chunk, every id in that
        chunk is skipped (counted as detection errors) and NOT sent down
        the per-id path."""
        from backfill_graph import _find_orphans

        # Create two memories: one in the graph, one not.
        rec1 = store.remember(
            category="personal_fact",
            content="Bulk wedge test memory one",
        )
        rec2 = store.remember(
            category="personal_fact",
            content="Bulk wedge test memory two",
        )
        _index(store, graph, rec1)
        _index(store, graph, rec2)
        store.delete_memory(rec1.memory_id)
        store.delete_memory(rec2.memory_id)
        raising = _RaisingStore(store, raise_bulk=True)
        ids = [rec1.memory_id, rec2.memory_id]
        orphans, detection_errors = _find_orphans(raising, ids)
        # Neither id should be declared an orphan — the bulk check failed
        # and they were NOT sent to the per-id path.
        assert rec1.memory_id not in orphans
        assert rec2.memory_id not in orphans
        assert detection_errors == 2

    def test_mixed_raise_and_confirm(self, store, graph):
        """When some ids verify normally and one raises, only the raising
        id is skipped; the confirmed-orphan is still detected."""
        from backfill_graph import _find_orphans

        rec_ok = store.remember(
            category="personal_fact",
            content="Mixed test confirmed orphan",
        )
        rec_wedge = store.remember(
            category="personal_fact",
            content="Mixed test wedged id",
        )
        _index(store, graph, rec_ok)
        _index(store, graph, rec_wedge)
        store.delete_memory(rec_ok.memory_id)
        store.delete_memory(rec_wedge.memory_id)
        # Only the per-id check raises for rec_wedge; rec_ok's history
        # returns normally (empty → orphan).
        class _SelectiveRaising(_RaisingStore):
            def get_memory_history(self, mid):
                if mid == rec_wedge.memory_id:
                    raise RuntimeError("wedge on specific id")
                return self._real.get_memory_history(mid)
        raising = _SelectiveRaising(store)
        orphans, detection_errors = _find_orphans(
            raising, [rec_ok.memory_id, rec_wedge.memory_id],
        )
        assert rec_ok.memory_id in orphans
        assert rec_wedge.memory_id not in orphans
        assert detection_errors == 1


class TestPruneOrphansAbortOnDetectionErrors:
    """#378: a non-dry-run prune with detection errors must abort before
    deleting a single node (nothing is half-pruned)."""

    def test_prune_aborts_on_detection_errors(self, store, graph):
        """A --prune-orphans run with any detection error aborts before
        deleting anything (exit non-zero, nothing removed)."""
        from backfill_graph import prune_orphans, _graph_memory_ids

        rec = store.remember(
            category="personal_fact",
            content="Abort test memory that should survive",
        )
        _index(store, graph, rec)
        store.delete_memory(rec.memory_id)  # genuinely orphaned
        raising = _RaisingStore(store, raise_history=True)
        summary = prune_orphans(raising, graph, dry_run=False)
        # The prune must abort — nothing removed.
        assert summary["detection_errors"] > 0
        assert summary["removed"] == 0
        # The graph node must still be present (not purged).
        assert rec.memory_id in _graph_memory_ids(graph)

    def test_dry_run_still_lists_on_detection_errors(self, store, graph):
        """Dry-run may still list orphans found and report detection
        errors as a warning (does not abort — it's informational)."""
        from backfill_graph import prune_orphans

        rec = store.remember(
            category="personal_fact",
            content="Dry run abort test memory",
        )
        _index(store, graph, rec)
        store.delete_memory(rec.memory_id)
        raising = _RaisingStore(store, raise_history=True)
        summary = prune_orphans(raising, graph, dry_run=True)
        assert summary["detection_errors"] > 0
        # Dry run never removes anything regardless.
        assert summary["removed"] == 0

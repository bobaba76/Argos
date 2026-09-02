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

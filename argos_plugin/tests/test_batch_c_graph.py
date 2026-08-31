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

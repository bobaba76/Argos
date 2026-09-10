"""2026-09-10: one bounded retry + a LOUD final failure for graph indexing.

The old path swallowed every failure at debug level; contention-timeout drops
went unnoticed by the drift watch for hours. Guards:
- transient failure then success => retried, no warning;
- persistent failure => retried once, then WARNING (fail-soft, not silent).
Also guards the LLM-import repair used by the shared service process.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

_plugin_dir = Path(__file__).resolve().parent.parent
for _path in (_plugin_dir.parent, _plugin_dir):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


class _FakeGraph:
    def __init__(self, fail_times: int) -> None:
        self.fail_times = fail_times
        self.calls = 0

    def index_memory(self, **kwargs):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RuntimeError("service timeout (simulated)")
        return 3


def _provider_with(graph):
    try:
        import argos_plugin
    except ModuleNotFoundError:
        import argos as argos_plugin

    provider = argos_plugin.ArgosProvider()
    provider._graph = graph
    provider._store = None
    return provider


def test_transient_failure_is_retried(caplog):
    graph = _FakeGraph(fail_times=1)
    provider = _provider_with(graph)
    with caplog.at_level(logging.WARNING):
        provider._index_memory_graph("mem-test-1", "personal_fact", "Alice uses Docker")
    assert graph.calls == 2
    assert not any(
        "Graph indexing failed" in r.getMessage() for r in caplog.records
    )


def test_persistent_failure_is_loud_and_fail_soft(caplog):
    graph = _FakeGraph(fail_times=99)
    provider = _provider_with(graph)
    with caplog.at_level(logging.WARNING):
        provider._index_memory_graph("mem-test-2", "personal_fact", "Alice uses Docker")
    assert graph.calls == 2
    messages = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("Graph indexing failed" in m for m in messages), messages


def test_llm_import_repair_degrades_to_none_when_agent_blocked(monkeypatch):
    """_load_host_call_llm must degrade to None (never raise) when the agent
    package is not importable even via the candidate roots."""
    monkeypatch.setitem(sys.modules, "agent", None)
    monkeypatch.setitem(sys.modules, "agent.auxiliary_client", None)
    from graph import _load_host_call_llm

    assert _load_host_call_llm() is None

"""Tests for the in-service DuckDB-to-Kuzu drift watch (#329)."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

_plugin_dir = Path(__file__).resolve().parent.parent
if str(_plugin_dir) not in sys.path:
    sys.path.insert(0, str(_plugin_dir))

import drift_watch  # noqa: E402
from drift_watch import (  # noqa: E402
    heal_missing,
    last_result,
    run_drift_cycle,
    run_drift_cycle_for_service,
)


def _record(mid, content="c", category="context_note", tags=None):
    return SimpleNamespace(
        memory_id=mid,
        content=content,
        category=category,
        tags=tags or [],
        created_at="2026-01-01T00:00:00+00:00",
    )


def _store(records):
    store = MagicMock()
    store.list_recent.return_value = records
    return store


def _graph(indexed_ids):
    # SimpleNamespace, not MagicMock: _get_graph_memory_ids() probes for a
    # `_rpc` attribute to pick the shared-service path, and MagicMock
    # auto-creates any attribute — the fake must NOT look like a proxy.
    g = SimpleNamespace()
    g.list_nodes = MagicMock(
        return_value=[
            {"id": f"memory:{mid}", "entity_type": "memory"}
            for mid in indexed_ids
        ]
    )
    g.index_memory = MagicMock(return_value=1)
    return g


def test_check_reports_drift_and_heals_missing():
    store = _store([_record("m1"), _record("m2", content="beta")])
    graph = _graph(["m1"])
    result = run_drift_cycle(store, graph, auto_heal=True)
    assert result["drift"] is True
    assert result["missing_in_graph_count"] == 1
    # m2 re-indexed through the same MERGE path the save flow uses.
    assert graph.index_memory.call_count == 1
    kwargs = graph.index_memory.call_args.kwargs
    assert kwargs["memory_id"] == "m2"
    assert kwargs["content"] == "beta"
    assert result["healed"] == 1
    assert result["heal_failed"] == 0
    assert result["checked_at"]


def test_auto_heal_off_reports_without_writing():
    store = _store([_record("m1"), _record("m2")])
    graph = _graph(["m1"])
    result = run_drift_cycle(store, graph, auto_heal=False)
    assert result["drift"] is True
    assert result["missing_in_graph_count"] == 1
    assert graph.index_memory.call_count == 0
    assert "healed" not in result


def test_no_drift_no_heal():
    store = _store([_record("m1")])
    graph = _graph(["m1"])
    result = run_drift_cycle(store, graph, auto_heal=True)
    assert result["drift"] is False
    assert graph.index_memory.call_count == 0


def test_heal_is_fail_soft():
    store = _store([_record("m1"), _record("m2")])
    graph = _graph([])
    graph.index_memory.side_effect = RuntimeError("kuzu boom")
    result = run_drift_cycle(store, graph, auto_heal=True)
    assert result["drift"] is True
    assert result["healed"] == 0
    assert result["heal_failed"] == 2


def test_heal_missing_skips_unknown_ids():
    store = _store([_record("m1")])
    graph = _graph([])
    healed, failed = heal_missing(store, graph, ["ghost"])
    assert healed == 0
    assert failed == 0
    assert graph.index_memory.call_count == 0


def test_service_cycle_covers_all_tenants_and_caches():
    t1s, t1g = _store([_record("m1")]), _graph(["m1"])
    t2s, t2g = _store([_record("x1")]), _graph([])
    service = SimpleNamespace(
        home=str(Path("nonexistent-home-329")),
        _tenants={
            "t1": SimpleNamespace(store=t1s, graph=t1g),
            "t2": SimpleNamespace(store=t2s, graph=t2g),
        },
    )
    summary = run_drift_cycle_for_service(service, auto_heal=True)
    assert set(summary["tenants"]) == {"t1", "t2"}
    assert summary["drift"] is True            # t2 drifts
    assert summary["tenants"]["t1"]["drift"] is False
    assert t2g.index_memory.call_count == 1    # x1 healed
    cached = last_result()
    assert cached["drift"] is True
    assert "checked_at" in cached


def test_failing_tenant_does_not_break_cycle():
    bad = SimpleNamespace(store=MagicMock(), graph=MagicMock())
    bad.store.list_recent.side_effect = RuntimeError("db gone")
    service = SimpleNamespace(home="", _tenants={"bad": bad, "empty": SimpleNamespace(store=None, graph=None)})
    summary = run_drift_cycle_for_service(service, auto_heal=True)
    assert "error" in summary["tenants"]["bad"]
    assert summary["drift"] is False  # errors are not drift
    assert "empty" not in summary["tenants"]  # store-less cells skipped


def test_interval_parsing():
    assert drift_watch._read_interval_min({}) == 360
    assert drift_watch._read_interval_min({"graph_drift_check_interval_min": "0"}) == 0
    assert drift_watch._read_interval_min({"graph_drift_check_interval_min": "45"}) == 45
    assert drift_watch._read_interval_min({"graph_drift_check_interval_min": 99999}) == 10080
    assert drift_watch._read_interval_min({"graph_drift_check_interval_min": "junk"}) == 360
    assert drift_watch._read_interval_min({"graph_drift_check_interval_min": -5}) == 360


def test_thread_disabled_when_interval_zero(tmp_path):
    (tmp_path / "hybrid_memory.json").write_text(
        '{"graph_drift_check_interval_min": 0}', encoding="utf-8",
    )
    svc = SimpleNamespace(home=str(tmp_path))
    assert drift_watch.start_drift_watch_thread(svc) is None


def test_reconcile_full_lists_opt_in():
    from reconcile_graph import reconcile

    store = _store([_record("m1"), _record("m2"), _record("m3")])
    graph = _graph([])
    small = reconcile(store, graph, sample_size=1)
    assert "missing_in_graph_all" not in small
    assert small["missing_in_graph_count"] == 3
    assert len(small["missing_in_graph"]) == 1  # sample only
    full = reconcile(store, graph, sample_size=1, full_lists=True)
    assert sorted(full["missing_in_graph_all"]) == ["m1", "m2", "m3"]

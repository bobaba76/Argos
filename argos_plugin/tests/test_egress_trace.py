"""LLM call trace tests (#454).

The optional trace wraps agent.auxiliary_client.call_llm (the single choke
point for plugin-owned LLM calls) and appends JSONL rows with token usage.
Off by default; install is idempotent and must never break a call.
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

_plugin_dir = Path(__file__).resolve().parent.parent
for _path in (_plugin_dir.parent, _plugin_dir):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import pytest

from argos_plugin import egress


@pytest.fixture()
def fake_agent(monkeypatch):
    """A fake agent.auxiliary_client with a recording call_llm."""
    calls = []

    def call_llm(*args, **kwargs):
        calls.append((args, kwargs))
        usage = SimpleNamespace(
            prompt_tokens=100,
            completion_tokens=20,
            prompt_tokens_details=SimpleNamespace(cached_tokens=50),
        )
        return SimpleNamespace(usage=usage)

    aux = types.ModuleType("agent.auxiliary_client")
    aux.call_llm = call_llm
    agent_mod = types.ModuleType("agent")
    agent_mod.auxiliary_client = aux
    monkeypatch.setitem(sys.modules, "agent", agent_mod)
    monkeypatch.setitem(sys.modules, "agent.auxiliary_client", aux)
    # Hermeticity: egress installs the trace at *import time* when the live
    # config already has llm_trace_enabled=true (egress.py module tail). That
    # import-time install wraps the import-time agent object — NOT this fresh
    # fixture fake — and the idempotent-guard in install_trace() then
    # early-returns, leaving the fake unwrapped. Always reset so each test
    # deterministically installs onto the fixture's fake.
    egress.uninstall_trace()
    yield aux
    egress.uninstall_trace()


def _trace_cfg(tmp_path, enabled=True):
    return {"llm_trace_enabled": enabled, "llm_trace_path": str(tmp_path / "trace.jsonl")}


def test_disabled_install_is_noop(fake_agent, monkeypatch, tmp_path):
    monkeypatch.setattr(egress, "load_config", lambda: _trace_cfg(tmp_path, enabled=False))
    assert egress.install_trace() is False
    # call_llm untouched: still the original function object.
    assert fake_agent.call_llm.__name__ == "call_llm"
    assert not (tmp_path / "trace.jsonl").exists()


def test_enabled_install_records_row(fake_agent, monkeypatch, tmp_path):
    monkeypatch.setattr(egress, "load_config", lambda: _trace_cfg(tmp_path))
    assert egress.install_trace() is True
    try:
        resp = fake_agent.call_llm(task="extraction", messages=[{"role": "user", "content": "x"}])
        assert resp.usage.prompt_tokens == 100
        rows = [json.loads(line) for line in (tmp_path / "trace.jsonl").read_text().splitlines()]
        assert len(rows) == 1
        row = rows[0]
        assert row["task"] == "extraction"
        assert row["kind"] == "extractor"
        assert row["prompt_tokens"] == 100
        assert row["completion_tokens"] == 20
        assert row["cached_tokens"] == 50
        assert row["ok"] is True
    finally:
        egress.uninstall_trace()
    # uninstall restores the original function.
    assert fake_agent.call_llm.__name__ == "call_llm"


def test_failure_row_and_propagation(fake_agent, monkeypatch, tmp_path):
    def boom(*args, **kwargs):
        raise RuntimeError("provider down")

    fake_agent.call_llm = boom
    monkeypatch.setattr(egress, "load_config", lambda: _trace_cfg(tmp_path))
    assert egress.install_trace() is True
    try:
        with pytest.raises(RuntimeError, match="provider down"):
            fake_agent.call_llm(task="conflict_judge")
        rows = [json.loads(line) for line in (tmp_path / "trace.jsonl").read_text().splitlines()]
        assert len(rows) == 1
        assert rows[0]["ok"] is False
        assert rows[0]["kind"] == "conflict_judge"
        assert "provider down" in rows[0]["error"]
    finally:
        egress.uninstall_trace()


def test_unknown_task_kind_falls_back_to_task_name(fake_agent, monkeypatch, tmp_path):
    monkeypatch.setattr(egress, "load_config", lambda: _trace_cfg(tmp_path))
    assert egress.install_trace() is True
    try:
        fake_agent.call_llm(task="brand_new_lane")
        row = json.loads((tmp_path / "trace.jsonl").read_text().splitlines()[0])
        assert row["kind"] == "brand_new_lane"
    finally:
        egress.uninstall_trace()


def test_append_never_raises_on_bad_path(monkeypatch, tmp_path):
    monkeypatch.setattr(egress, "_trace_path", lambda: str(tmp_path / "no" / "such" / "dir" / "t.jsonl"))
    # Must not raise even though the parent directory does not exist.
    egress._append_trace_row({"ts": "2026-09-12T00:00:00+00:00", "task": "x", "ok": True})
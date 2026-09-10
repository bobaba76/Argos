"""2026-09-10 drift-rebuild fixes: teardown & spawn-env regression guards.

Incident: the 06:30 drift-watch cron ran reconcile_graph.py, whose teardown
called store._rpc.stop_service() -- an unconditional shutdown RPC that killed
the shared service the app was using. Separately, a service spawned by an
env-stripped process could not import agent.auxiliary_client, silently
disabling LLM graph extraction.

Guards:
- stop_service() is a no-op unless (a) force=True, or (b) this process is
  recorded as the spawner in the endpoint file.
- _write_endpoint records the spawner PID from HERMES_SERVICE_SPAWNER_PID.
- _find_agent_root() locates a checkout containing agent/auxiliary_client.py.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_plugin_dir = Path(__file__).resolve().parent.parent
for _path in (_plugin_dir.parent, _plugin_dir):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


def _rpc_stub(monkeypatch, endpoint, sent):
    from service_client import _SharedRPC

    rpc = _SharedRPC.__new__(_SharedRPC)
    rpc.home = Path("unused-stub-home")
    monkeypatch.setattr("service_client._read_endpoint", lambda home: endpoint)

    def _record(request, timeout=None):
        sent.append(request)
        return {"ok": True}

    rpc._request = _record
    return rpc


class TestStopServiceOwnership:
    def test_stop_allowed_for_spawner_process(self, monkeypatch):
        sent = []
        rpc = _rpc_stub(monkeypatch, {"spawner_pid": os.getpid()}, sent)
        result = rpc.stop_service()
        assert result == {"ok": True}
        assert sent and sent[-1].get("method") == "shutdown"

    def test_stop_refused_for_other_process(self, monkeypatch):
        sent = []
        rpc = _rpc_stub(monkeypatch, {"spawner_pid": os.getpid() + 12345}, sent)
        result = rpc.stop_service()
        assert result == {"stopped": False, "reason": "not_started_by_this_process"}
        assert sent == []

    def test_stop_refused_when_spawner_unknown(self, monkeypatch):
        sent = []
        rpc = _rpc_stub(monkeypatch, {"host": "127.0.0.1", "port": 1}, sent)
        result = rpc.stop_service()
        assert result == {"stopped": False, "reason": "spawner_unknown"}
        assert sent == []

    def test_force_stop_bypasses_ownership(self, monkeypatch):
        sent = []
        rpc = _rpc_stub(monkeypatch, {"spawner_pid": os.getpid() + 12345}, sent)
        result = rpc.stop_service(force=True)
        assert result == {"ok": True}
        assert sent and sent[-1].get("method") == "shutdown"

    def test_stop_refused_when_service_down(self, monkeypatch):
        sent = []
        rpc = _rpc_stub(monkeypatch, None, sent)
        result = rpc.stop_service()
        assert result == {"stopped": False, "reason": "spawner_unknown"}
        assert sent == []


class TestEndpointSpawnerPid:
    def test_write_endpoint_records_spawner_pid(self, tmp_path, monkeypatch):
        from memory_service import _write_endpoint

        monkeypatch.setenv("HERMES_SERVICE_SPAWNER_PID", "4242")
        path = tmp_path / "hybrid_memory_service.json"
        _write_endpoint(path, 1234, "tok", "gs")
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["spawner_pid"] == 4242
        assert payload["pid"] == os.getpid()

    def test_write_endpoint_spawner_pid_none_without_env(self, tmp_path, monkeypatch):
        from memory_service import _write_endpoint

        monkeypatch.delenv("HERMES_SERVICE_SPAWNER_PID", raising=False)
        path = tmp_path / "hybrid_memory_service.json"
        _write_endpoint(path, 1234, "tok", "gs")
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["spawner_pid"] is None


class TestAgentRootDiscovery:
    def test_env_override_is_used(self, monkeypatch, tmp_path):
        from service_client import _find_agent_root

        root = tmp_path / "hermes-agent"
        (root / "agent").mkdir(parents=True)
        (root / "agent" / "auxiliary_client.py").write_text(
            "def call_llm(**kwargs):\n    return None\n", encoding="utf-8"
        )
        monkeypatch.setenv("HERMES_AGENT_ROOT", str(root))
        assert _find_agent_root() == str(root)

    def test_result_always_contains_agent_package(self, monkeypatch):
        from service_client import _find_agent_root

        monkeypatch.delenv("HERMES_AGENT_ROOT", raising=False)
        result = _find_agent_root()
        if result is not None:
            assert (Path(result) / "agent" / "auxiliary_client.py").is_file()

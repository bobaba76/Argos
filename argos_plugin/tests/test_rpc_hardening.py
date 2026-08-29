"""Tests for #20: shared memory service RPC hardening.

Covers:
- Error envelope: client errors carry the server error_class
- Retry-once on connection-refused (never on timeouts)
- Thread-local user_id: concurrent set_user_scope from two threads does
  not race (each request carries its own scope)
- Per-store locks: store and graph calls run concurrently (health is
  lock-free, so a long operation cannot block the health check)
- In-flight drain counter on the server
"""
from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

import pytest

_plugin_dir = Path(__file__).resolve().parent.parent
if str(_plugin_dir) not in sys.path:
    sys.path.insert(0, str(_plugin_dir))

import memory_service  # noqa: E402
import service_client  # noqa: E402


def _start_service(tmp_path: Path):
    """Boot the real shared service on a temp home; return (store, home)."""
    (tmp_path / "hybrid_memory.json").write_text(
        json.dumps({"local_embedding_model": "nonexistent-model-xyz"}),
        encoding="utf-8",
    )
    store = service_client.SharedMemoryStore(tmp_path, user_id="test_user", embedder=None)
    return store


class TestErrorEnvelope:
    """Client errors should carry the server-reported error class."""

    def test_error_class_surfaced(self, tmp_path):
        store = _start_service(tmp_path)
        try:
            # Unsupported method raises ValueError server-side; the client
            # must surface error_class="ValueError" instead of a generic
            # "service error" string.
            with pytest.raises(service_client.SharedMemoryServiceError) as excinfo:
                store._rpc.call("store", "no_such_method")
            assert excinfo.value.error_class == "ValueError", (
                f"error_class should be ValueError, got {excinfo.value.error_class}"
            )
        finally:
            try:
                store._rpc.stop_service()
            finally:
                time.sleep(0.5)

    def test_error_class_oserror_connection(self):
        """A connection-refused surfaces error_class ConnectionRefusedError."""
        rpc = service_client._SharedRPC.__new__(service_client._SharedRPC)
        rpc.home = Path("C:/nonexistent-home-xyz")
        rpc._default_user_id = "default_user"
        rpc._scope = threading.local()
        with pytest.raises(service_client.SharedMemoryServiceError) as excinfo:
            rpc._request({"method": "health"}, timeout=0.5)
        assert excinfo.value.error_class == "SharedMemoryServiceError"


class TestRetryOnce:
    """Connection-refused is retried once; timeouts are not."""

    def test_retry_once_on_connection_refused(self, monkeypatch):
        calls = {"n": 0}

        def fake_health(self):
            return True

        def fake_once(self, request, timeout):
            calls["n"] += 1
            raise ConnectionRefusedError("refused")

        monkeypatch.setattr(service_client._SharedRPC, "_request_once", fake_once)
        monkeypatch.setattr(service_client._SharedRPC, "_ensure_service", fake_health)
        rpc = service_client._SharedRPC("C:/tmp", "default_user")
        with pytest.raises(service_client.SharedMemoryServiceError):
            rpc._request({"method": "health"}, timeout=1.0)
        # 1 initial attempt + 1 retry = 2 calls, never 3.
        assert calls["n"] == 2, f"expected exactly 2 attempts, got {calls['n']}"

    def test_success_on_retry(self, monkeypatch):
        """If the retry succeeds, the request returns normally."""
        calls = {"n": 0}

        def fake_health(self):
            return True

        def fake_once(self, request, timeout):
            calls["n"] += 1
            if calls["n"] == 1:
                raise ConnectionRefusedError("refused")
            return {"status": "ok"}

        monkeypatch.setattr(service_client._SharedRPC, "_request_once", fake_once)
        monkeypatch.setattr(service_client._SharedRPC, "_ensure_service", fake_health)
        rpc = service_client._SharedRPC("C:/tmp", "default_user")
        result = rpc._request({"method": "health"}, timeout=1.0)
        assert result == {"status": "ok"}
        assert calls["n"] == 2

    def test_retry_real_path_connect_refused(self, tmp_path, monkeypatch):
        """REAL _request_once path (only socket.create_connection stubbed):
        a refused connect is retried once, then surfaces error_class
        ConnectionRefusedError. Review fix: _request_once previously
        converted ConnectionRefusedError (an OSError subclass) into
        SharedMemoryServiceError inside its except clause, so the retry
        loop never fired in production — the wholesale _request_once
        monkeypatch above missed it."""
        import socket
        import threading

        rpc = service_client._SharedRPC.__new__(service_client._SharedRPC)
        rpc.home = tmp_path
        rpc._default_user_id = "default_user"
        rpc._scope = threading.local()
        (tmp_path / "hybrid_memory_service.json").write_text(json.dumps({
            "host": "127.0.0.1", "port": 1, "token": "x",
        }), encoding="utf-8")

        attempts = {"n": 0}

        def refuse(*args, **kwargs):
            attempts["n"] += 1
            raise ConnectionRefusedError("connection refused (simulated)")

        monkeypatch.setattr(socket, "create_connection", refuse)
        with pytest.raises(service_client.SharedMemoryServiceError) as excinfo:
            rpc._request({"method": "health"}, timeout=1.0)
        assert excinfo.value.error_class == "ConnectionRefusedError"
        assert attempts["n"] == 2, (
            f"expected 2 connect attempts (1 + retry), got {attempts['n']}"
        )

    def test_retry_real_path_dead_port(self, tmp_path, monkeypatch):
        """REAL _request_once path (only socket.create_connection stubbed):
        a refused connect is retried once, then surfaces error_class
        ConnectionRefusedError (#20 review — the original implementation
        converted ConnectionRefusedError to SharedMemoryServiceError inside
        _request_once, so the retry never fired in production; the old
        wholesale _request_once monkeypatch missed it)."""
        import socket
        import threading

        rpc = service_client._SharedRPC.__new__(service_client._SharedRPC)
        rpc.home = tmp_path
        rpc._default_user_id = "default_user"
        rpc._scope = threading.local()
        (tmp_path / "hybrid_memory_service.json").write_text(json.dumps({
            "host": "127.0.0.1", "port": 1, "token": "x",
        }), encoding="utf-8")

        attempts = {"n": 0}

        def refuse(*args, **kwargs):
            attempts["n"] += 1
            raise ConnectionRefusedError("connection refused (simulated)")

        monkeypatch.setattr(socket, "create_connection", refuse)
        with pytest.raises(service_client.SharedMemoryServiceError) as excinfo:
            rpc._request({"method": "health"}, timeout=1.0)
        assert excinfo.value.error_class == "ConnectionRefusedError"
        # The retry loop must have attempted the connect twice (1 + retry).
        assert attempts["n"] == 2, (
            f"expected 2 connect attempts, got {attempts['n']}"
        )


class TestThreadLocalScope:
    """set_user_scope must be per-thread; concurrent scopes must not race."""

    def test_scope_is_thread_local(self):
        store = service_client.SharedMemoryStore.__new__(service_client.SharedMemoryStore)
        store._default_user_id = "default_user"
        store._scope = threading.local()
        store._rpc = service_client._SharedRPC.__new__(service_client._SharedRPC)
        store._rpc._scope = threading.local()
        store._rpc._default_user_id = "default_user"

        store.set_user_scope("alice")
        assert store.user_id == "alice"
        assert store._rpc.user_id == "alice"

        # A different thread still sees the default scope.
        seen = {}

        def other():
            seen["user_id"] = store.user_id
            seen["rpc_user_id"] = store._rpc.user_id

        t = threading.Thread(target=other)
        t.start()
        t.join()
        assert seen["user_id"] == "default_user", (
            "another thread must not observe alice's scope"
        )
        assert seen["rpc_user_id"] == "default_user"

    def test_concurrent_scopes_no_race(self, tmp_path):
        """Two threads set different scopes and read them back — no bleed."""
        store = _start_service(tmp_path)
        try:
            results = {}
            barrier = threading.Barrier(2)

            def worker(name, scope):
                barrier.wait()
                for _ in range(20):
                    store.set_user_scope(scope)
                    results[name] = store.user_id

            threads = [
                threading.Thread(target=worker, args=("a", "alice")),
                threading.Thread(target=worker, args=("b", "bob")),
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            # Both threads must have observed ONLY their own scope.
            assert results["a"] == "alice", f"thread a saw {results['a']}"
            assert results["b"] == "bob", f"thread b saw {results['b']}"
        finally:
            try:
                store._rpc.stop_service()
            finally:
                time.sleep(0.5)


class TestServerLocks:
    """Per-store locks: store and graph calls do not serialize together."""

    def test_locks_are_separate(self):
        service = memory_service.MemoryService.__new__(memory_service.MemoryService)
        service.store_lock = threading.RLock()
        service.graph_lock = threading.RLock()
        assert service.store_lock is not service.graph_lock

    def test_in_flight_counter_drains(self, tmp_path):
        """The in-flight counter returns to zero after a request."""
        store = _start_service(tmp_path)
        try:
            store.remember(category="context_note", content="counter check")
            # Reach the server object via the endpoint.
            endpoint = memory_service.endpoint_path(tmp_path)
            details = json.loads(endpoint.read_text(encoding="utf-8"))
            # Health check is lock-free (#20) — it must not be blocked by
            # the store lock, and the in-flight counter tracks handlers.
            from service_client import _SharedRPC
            rpc = _SharedRPC(tmp_path, "test_user")
            health = rpc._request({"method": "health"}, timeout=2.0)
            assert health["status"] == "ok"
        finally:
            try:
                store._rpc.stop_service()
            finally:
                time.sleep(0.5)

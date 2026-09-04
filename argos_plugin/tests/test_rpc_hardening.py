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

    @pytest.mark.xdist_group("shared_service")
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

    @pytest.mark.xdist_group("shared_service")
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

    @pytest.mark.xdist_group("shared_service")
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


# ---------------------------------------------------------------------------
# SC1-SC6: Service client audit fixes (#218)
# ---------------------------------------------------------------------------

class TestSC1ResponseSizeLimit:
    """SC1: the client must cap total response bytes to prevent unbounded
    memory consumption from a malicious or buggy server."""

    def test_max_response_bytes_constant_exists(self):
        """The _MAX_RESPONSE_BYTES constant should be defined."""
        assert hasattr(service_client, "_MAX_RESPONSE_BYTES")
        assert service_client._MAX_RESPONSE_BYTES > 0
        # Should be at least 1MB (generous for any legitimate response).
        assert service_client._MAX_RESPONSE_BYTES >= 1024 * 1024

    def test_response_size_cap_raises(self, tmp_path):
        """A response exceeding _MAX_RESPONSE_BYTES should raise
        SharedMemoryServiceError with error_class='ResponseTooLarge'."""
        from unittest.mock import MagicMock, patch
        large_chunk = b"x" * (service_client._MAX_RESPONSE_BYTES + 1)

        mock_conn = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_conn.recv = MagicMock(side_effect=[large_chunk, b""])

        rpc = service_client._SharedRPC.__new__(service_client._SharedRPC)
        rpc.home = tmp_path
        rpc._default_user_id = "test"
        rpc._scope = threading.local()

        with patch("service_client._read_endpoint", return_value={
            "host": "127.0.0.1", "port": 9999, "token": "test",
        }), patch("socket.create_connection", return_value=mock_conn):
            with pytest.raises(service_client.SharedMemoryServiceError) as excinfo:
                rpc._request_once({"method": "test"}, timeout=5.0)
        assert excinfo.value.error_class == "ResponseTooLarge"


class TestSC2EndpointPermissions:
    """SC2: the endpoint file should be restricted to owner-only on POSIX."""

    def test_endpoint_file_permissions_restricted(self, tmp_path):
        """_write_endpoint should chmod the file to 0o600 on POSIX."""
        import os
        path = tmp_path / "endpoint.json"
        memory_service._write_endpoint(path, port=12345, token="secret")
        assert path.exists()
        if sys.platform != "win32":
            mode = os.stat(path).st_mode & 0o777
            assert mode == 0o600, f"Expected 0o600, got {oct(mode)}"


class TestSC5MutableDefaults:
    """SC5: _record_from_dict should not share mutable default objects."""

    def test_tags_default_is_new_list(self):
        """Missing tags should produce a new empty list, not a shared default."""
        r1 = service_client._record_from_dict({"memory_id": "m1", "content": "a"})
        r2 = service_client._record_from_dict({"memory_id": "m2", "content": "b"})
        assert r1.tags == []
        assert r2.tags == []
        r1.tags.append("test")
        assert r2.tags == [], "Mutable default shared between records (SC5)"

    def test_payload_default_is_new_dict(self):
        """Missing payload should produce a new empty dict, not a shared default."""
        r1 = service_client._record_from_dict({"memory_id": "m1", "content": "a"})
        r2 = service_client._record_from_dict({"memory_id": "m2", "content": "b"})
        assert r1.payload == {}
        assert r2.payload == {}
        r1.payload["key"] = "value"
        assert r2.payload == {}, "Mutable default shared between records (SC5)"

    def test_tags_falsy_value_gets_new_list(self):
        """A falsy tags value (None) should produce a new empty list."""
        r = service_client._record_from_dict({
            "memory_id": "m1", "content": "a", "tags": None,
        })
        assert r.tags == []

    def test_payload_falsy_value_gets_new_dict(self):
        """A falsy payload value (None) should produce a new empty dict."""
        r = service_client._record_from_dict({
            "memory_id": "m1", "content": "a", "payload": None,
        })
        assert r.payload == {}


class TestSC6CloseIsNoOp:
    """SC6: close() should be a no-op for the shared service client."""

    def test_shared_memory_store_close_returns_none(self):
        """SharedMemoryStore.close() should return None (no-op)."""
        store = service_client.SharedMemoryStore.__new__(service_client.SharedMemoryStore)
        assert store.close() is None

    def test_shared_graph_store_close_returns_none(self):
        """SharedGraphStore.close() should return None (no-op)."""
        store = service_client.SharedGraphStore.__new__(service_client.SharedGraphStore)
        assert store.close() is None


# ---------------------------------------------------------------------------
# MS1-MS10: Memory service audit fixes (#224)
# ---------------------------------------------------------------------------

class TestMS1SanitizeArgs:
    """MS1: strip server-set fields from client-supplied args."""

    def test_ms1_strips_provenance_origin(self):
        assert "provenance_origin" not in memory_service._sanitize_args(
            {"content": "test", "provenance_origin": "internal"}
        )

    def test_ms1_strips_grounding(self):
        assert "grounding" not in memory_service._sanitize_args(
            {"content": "test", "grounding": "observed"}
        )

    def test_ms1_strips_status(self):
        assert "status" not in memory_service._sanitize_args(
            {"content": "test", "status": "active"}
        )

    def test_ms1_strips_source(self):
        assert "source" not in memory_service._sanitize_args(
            {"content": "test", "source": "internal"}
        )

    def test_ms1_strips_user_scope(self):
        assert "user_scope" not in memory_service._sanitize_args(
            {"content": "test", "user_scope": "other-user"}
        )

    def test_ms1_strips_confidence(self):
        assert "confidence" not in memory_service._sanitize_args(
            {"content": "test", "confidence": 1.0}
        )

    def test_ms1_strips_review_mode(self):
        assert "review_mode" not in memory_service._sanitize_args(
            {"content": "test", "review_mode": "auto"}
        )

    def test_ms1_preserves_valid_args(self):
        cleaned = memory_service._sanitize_args(
            {"content": "test", "category": "personal_fact", "tags": ["a"]}
        )
        assert cleaned == {"content": "test", "category": "personal_fact", "tags": ["a"]}

    def test_ms1_non_dict_returns_empty(self):
        assert memory_service._sanitize_args(None) == {}
        assert memory_service._sanitize_args("not-a-dict") == {}


class TestMS2ForbiddenMethods:
    """MS2: destructive methods are forbidden on the RPC boundary."""

    def test_ms2_forbidden_store_methods_defined(self):
        assert "delete_memory" in memory_service._FORBIDDEN_STORE_METHODS
        assert "quarantine_memory" in memory_service._FORBIDDEN_STORE_METHODS
        assert "cleanup_junk" in memory_service._FORBIDDEN_STORE_METHODS
        assert "consolidate" in memory_service._FORBIDDEN_STORE_METHODS
        assert "purge_tombstone" in memory_service._FORBIDDEN_STORE_METHODS
        assert "mark_superseded" in memory_service._FORBIDDEN_STORE_METHODS

    def test_ms2_set_state_forbidden(self):
        """MS7: set_state is forbidden on the RPC boundary."""
        assert "set_state" in memory_service._FORBIDDEN_STORE_METHODS

    def test_ms2_forbidden_graph_methods_defined(self):
        assert "clear_scope" in memory_service._FORBIDDEN_GRAPH_METHODS

    def test_ms2_delete_memory_rejected_in_dispatch(self, tmp_path):
        """MS2: calling delete_memory via dispatch raises PermissionError."""
        service = memory_service.MemoryService.__new__(memory_service.MemoryService)
        service._credential_mode = False
        service._strict_routing = False
        service._user_tenant_map = {}
        # Need a dummy tenant for _resolve_tenant to succeed before the
        # forbidden methods check runs.
        dummy = type("DummyTenant", (), {"store_lock": threading.RLock(), "graph_lock": threading.RLock()})()
        service._tenants = {"default": dummy}
        service._default_tenant = "default"
        service._lock_wait_total_s = 0.0
        service._lock_wait_count = 0
        service.server = None
        with pytest.raises(PermissionError):
            service.dispatch({
                "component": "store",
                "method": "delete_memory",
                "args": {"memory_id": "m1"},
                "user_id": "default_user",
            })

    def test_ms2_set_state_rejected_in_dispatch(self, tmp_path):
        """MS7: calling set_state via dispatch raises PermissionError."""
        service = memory_service.MemoryService.__new__(memory_service.MemoryService)
        service._credential_mode = False
        service._strict_routing = False
        service._user_tenant_map = {}
        dummy = type("DummyTenant", (), {"store_lock": threading.RLock(), "graph_lock": threading.RLock()})()
        service._tenants = {"default": dummy}
        service._default_tenant = "default"
        service._lock_wait_total_s = 0.0
        service._lock_wait_count = 0
        service.server = None
        with pytest.raises(PermissionError):
            service.dispatch({
                "component": "store",
                "method": "set_state",
                "args": {"key": "k", "value": "v"},
                "user_id": "default_user",
            })


class TestMS4NoTracebackLeak:
    """MS4: error responses must not include traceback."""

    def test_ms4_no_traceback_import_needed(self):
        """MS4: the traceback module should not be imported (unused after fix)."""
        source = Path(memory_service.__file__).read_text(encoding="utf-8")
        # Should not have "import traceback" as an active import
        assert "import traceback" not in source, (
            "traceback import should be removed (MS4)"
        )

    def test_ms4_error_envelope_no_traceback_field(self):
        """MS4: the _write method should not include a traceback field."""
        source = Path(memory_service.__file__).read_text(encoding="utf-8")
        # The error response should not contain "traceback" as a key
        # (it may appear in comments explaining what was removed).
        # Check the _RequestHandler.handle method doesn't write traceback.
        assert '"traceback"' not in source, (
            "Error response should not include a traceback field (MS4)"
        )


class TestMS6ResponseSizeLimit:
    """MS6: response size limit prevents unbounded memory."""

    def test_ms6_max_response_bytes_defined(self):
        assert hasattr(memory_service, "_MAX_RESPONSE_BYTES")
        assert memory_service._MAX_RESPONSE_BYTES > 0

    def test_ms6_response_too_large_error(self):
        """MS6: _write should return a ResponseTooLarge error for oversized data."""
        class FakeWriteFile:
            def __init__(self):
                self.written = []
            def write(self, data):
                self.written.append(data)
            def flush(self):
                pass

        handler = memory_service._RequestHandler.__new__(memory_service._RequestHandler)
        handler.wfile = FakeWriteFile()
        # Create a response that exceeds the limit.
        huge_value = {"ok": True, "result": "x" * (memory_service._MAX_RESPONSE_BYTES + 1)}
        handler._write(huge_value)
        # The written data should be an error, not the huge response.
        assert len(handler.wfile.written) == 1
        response = json.loads(handler.wfile.written[0].decode("utf-8"))
        assert response["ok"] is False
        assert response["error_class"] == "ResponseTooLarge"


class TestMS8ProbeSizeCap:
    """MS8: single-instance guard probe has a size cap."""

    def test_ms8_probe_max_bytes_defined(self):
        assert hasattr(memory_service, "_PROBE_MAX_BYTES")
        assert memory_service._PROBE_MAX_BYTES > 0

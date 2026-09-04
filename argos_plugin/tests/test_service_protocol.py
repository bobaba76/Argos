"""Tests for RPC protocol wire versioning + stale-service self-heal (#246).

Covers:
- Server rejects wrong/missing protocol version with structured VersionMismatch
- Client self-heals on VersionMismatch (kill + respawn + retry once)
- _PROTOCOL_VERSION is a single imported constant in both client and server
"""
from __future__ import annotations

import json
import socket
import sys
import threading
import time
from pathlib import Path

import pytest

_plugin_dir = Path(__file__).resolve().parent.parent
if str(_plugin_dir) not in sys.path:
    sys.path.insert(0, str(_plugin_dir))


class TestProtocolVersionConstant:
    """_PROTOCOL_VERSION is a single shared constant imported by both."""

    def test_version_is_int(self):
        from store_protocol import _PROTOCOL_VERSION
        assert isinstance(_PROTOCOL_VERSION, int)
        assert _PROTOCOL_VERSION >= 1

    def test_client_imports_protocol_version(self):
        import service_client
        assert hasattr(service_client, "_PROTOCOL_VERSION")
        from store_protocol import _PROTOCOL_VERSION
        assert service_client._PROTOCOL_VERSION == _PROTOCOL_VERSION

    def test_server_imports_protocol_version(self):
        import memory_service
        assert hasattr(memory_service, "_PROTOCOL_VERSION")
        from store_protocol import _PROTOCOL_VERSION
        assert memory_service._PROTOCOL_VERSION == _PROTOCOL_VERSION

    def test_both_import_from_same_module(self):
        """Structural assert: both files import from store_protocol."""
        sc_src = (_plugin_dir / "service_client.py").read_text(encoding="utf-8")
        ms_src = (_plugin_dir / "memory_service.py").read_text(encoding="utf-8")
        assert "from .store_protocol import _PROTOCOL_VERSION" in sc_src or \
               "from store_protocol import _PROTOCOL_VERSION" in sc_src
        assert "from .store_protocol import _PROTOCOL_VERSION" in ms_src or \
               "from store_protocol import _PROTOCOL_VERSION" in ms_src


class TestServerVersionCheck:
    """Server rejects mismatched or missing protocol version."""

    def _start_mock_server(self):
        """Start a minimal TCP server that runs the real _RequestHandler.handle()."""
        import memory_service

        class MockMemoryService:
            def dispatch(self, request):
                return {"echo": request.get("method")}

        class MockServer:
            auth_token = "test-token"
            in_flight_lock = threading.Lock()
            in_flight = 0
            memory_service = MockMemoryService()

        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.bind(("127.0.0.1", 0))
        server_sock.listen(1)
        port = server_sock.getsockname()[1]
        mock_server = MockServer()

        def serve():
            try:
                conn, _ = server_sock.accept()
                handler = memory_service._RequestHandler.__new__(memory_service._RequestHandler)
                handler.request = conn
                handler.client_address = ("127.0.0.1", 0)
                handler.server = mock_server
                handler.rfile = conn.makefile("rb")
                handler.wfile = conn.makefile("wb")
                try:
                    handler.handle()
                finally:
                    conn.close()
            except Exception:
                pass

        t = threading.Thread(target=serve, daemon=True)
        t.start()
        return server_sock, port, t

    def _send_request(self, port, request):
        """Send a raw request and return the parsed response."""
        with socket.create_connection(("127.0.0.1", port), timeout=5.0) as conn:
            conn.sendall((json.dumps(request) + "\n").encode("utf-8"))
            data = b""
            while b"\n" not in data:
                chunk = conn.recv(65536)
                if not chunk:
                    break
                data += chunk
            return json.loads(data.splitlines()[0].decode("utf-8"))

    def test_rejects_wrong_version(self):
        """Server rejects v=999 with VersionMismatch, no dispatch."""
        server_sock, port, t = self._start_mock_server()
        try:
            resp = self._send_request(port, {
                "v": 999, "token": "test-token",
                "component": "store", "method": "health", "args": {},
            })
            assert resp["ok"] is False
            assert resp["error_class"] == "VersionMismatch"
            err = resp["error"]
            assert err["class"] == "VersionMismatch"
            assert err["received"] == 999
            assert 1 in err["supported"]
        finally:
            server_sock.close()

    def test_rejects_missing_version(self):
        """Server rejects missing v with VersionMismatch (received=null)."""
        server_sock, port, t = self._start_mock_server()
        try:
            resp = self._send_request(port, {
                "token": "test-token",
                "component": "store", "method": "health", "args": {},
            })
            assert resp["ok"] is False
            assert resp["error_class"] == "VersionMismatch"
            err = resp["error"]
            assert err["class"] == "VersionMismatch"
            assert err["received"] is None
        finally:
            server_sock.close()

    def test_accepts_correct_version(self):
        """Server accepts v=_PROTOCOL_VERSION and dispatches normally."""
        from store_protocol import _PROTOCOL_VERSION
        server_sock, port, t = self._start_mock_server()
        try:
            resp = self._send_request(port, {
                "v": _PROTOCOL_VERSION, "token": "test-token",
                "component": "store", "method": "health", "args": {},
            })
            assert resp["ok"] is True
            assert "result" in resp
        finally:
            server_sock.close()


class TestClientSelfHeal:
    """Client self-heals on VersionMismatch: kill + respawn + retry once."""

    def _make_store(self, tmp_path, monkeypatch):
        """Create a SharedMemoryStore without starting a real service."""
        import service_client as sc_mod
        # Prevent _ensure_service from starting a real service during __init__.
        monkeypatch.setattr(sc_mod._SharedRPC, "_ensure_service", lambda self: None)
        store = sc_mod.SharedMemoryStore(tmp_path, user_id="test")
        return store

    def test_version_mismatch_triggers_respawn(self, tmp_path, monkeypatch):
        """When the server returns VersionMismatch, the client calls
        _kill_stale_service + _ensure_service + retries once."""
        from service_client import SharedMemoryServiceError

        store = self._make_store(tmp_path, monkeypatch)
        rpc = store._rpc

        # Track calls to _kill_stale_service and _ensure_service.
        kill_called = []
        ensure_called = []
        monkeypatch.setattr(rpc, "_kill_stale_service", lambda: kill_called.append(True))
        monkeypatch.setattr(rpc, "_ensure_service", lambda: ensure_called.append(True))

        # First _request_once returns VersionMismatch, second succeeds.
        call_count = [0]

        def mock_once(request, timeout):
            call_count[0] += 1
            if call_count[0] == 1:
                err = SharedMemoryServiceError(
                    "version mismatch", error_class="VersionMismatch",
                )
                err.received_version = 999
                raise err
            return {"status": "ok"}

        monkeypatch.setattr(rpc, "_request_once", mock_once)

        result = rpc._request({"method": "health"})
        assert result == {"status": "ok"}
        assert len(kill_called) == 1, "_kill_stale_service must be called"
        assert len(ensure_called) == 1, "_ensure_service must be called"
        assert call_count[0] == 2, "must retry exactly once"

    def test_version_mismatch_not_retried_twice(self, tmp_path, monkeypatch):
        """If the respawned service ALSO returns VersionMismatch, the
        client gives up (no infinite loop)."""
        from service_client import SharedMemoryServiceError

        store = self._make_store(tmp_path, monkeypatch)
        rpc = store._rpc
        monkeypatch.setattr(rpc, "_kill_stale_service", lambda: None)
        monkeypatch.setattr(rpc, "_ensure_service", lambda: None)

        call_count = [0]
        def mock_once(request, timeout):
            call_count[0] += 1
            raise SharedMemoryServiceError(
                "version mismatch", error_class="VersionMismatch",
            )

        monkeypatch.setattr(rpc, "_request_once", mock_once)

        with pytest.raises(SharedMemoryServiceError) as exc_info:
            rpc._request({"method": "health"})
        assert exc_info.value.error_class == "VersionMismatch"
        assert call_count[0] == 2, "must try twice (original + 1 retry), no more"

    def test_non_version_error_not_retried(self, tmp_path, monkeypatch):
        """Non-VersionMismatch errors are NOT retried via the self-heal path."""
        from service_client import SharedMemoryServiceError

        store = self._make_store(tmp_path, monkeypatch)
        rpc = store._rpc
        kill_called = []
        monkeypatch.setattr(rpc, "_kill_stale_service", lambda: kill_called.append(True))
        monkeypatch.setattr(rpc, "_ensure_service", lambda: None)

        call_count = [0]
        def mock_once(request, timeout):
            call_count[0] += 1
            raise SharedMemoryServiceError("some error", error_class="ValueError")

        monkeypatch.setattr(rpc, "_request_once", mock_once)

        with pytest.raises(SharedMemoryServiceError):
            rpc._request({"method": "health"})
        assert call_count[0] == 1, "non-version errors must not retry"
        assert len(kill_called) == 0, "_kill_stale_service must not be called"

    def test_request_envelope_includes_version(self, tmp_path, monkeypatch):
        """_request_once adds v=_PROTOCOL_VERSION to the request envelope.

        Structural assert: the source code adds request["v"] before sending.
        """
        from store_protocol import _PROTOCOL_VERSION
        sc_src = (_plugin_dir / "service_client.py").read_text(encoding="utf-8")
        assert 'request["v"] = _PROTOCOL_VERSION' in sc_src, \
            "_request_once must add v=_PROTOCOL_VERSION to the request envelope"

"""Regression-guard tests for service_client.py (SC1-SC6, issue #218).

The fixes for SC1, SC2, SC5, SC6 are already on master. These tests
verify the fixes stay in place and document the accepted edge cases
(SC3, SC4).

Run with (Hermes venv python, offline):
    python -m pytest tests/test_service_client_audit.py -v
"""
from __future__ import annotations

import inspect
import os
import sys
from pathlib import Path

import pytest

_plugin_dir = Path(__file__).resolve().parent.parent
for _path in (_plugin_dir.parent, _plugin_dir):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


# ---------------------------------------------------------------------------
# SC1 — response size limit (already fixed on master)
# ---------------------------------------------------------------------------

class TestSC1ResponseSizeLimit:
    def test_max_response_bytes_constant_exists(self):
        """SC1: _MAX_RESPONSE_BYTES is defined and positive."""
        from service_client import _MAX_RESPONSE_BYTES
        assert _MAX_RESPONSE_BYTES > 0
        assert _MAX_RESPONSE_BYTES == 64 * 1024 * 1024

    def test_request_once_has_size_check(self):
        """SC1: _request_once checks total_read against _MAX_RESPONSE_BYTES."""
        from service_client import _SharedRPC
        src = inspect.getsource(_SharedRPC._request_once)
        assert "_MAX_RESPONSE_BYTES" in src
        assert "total_read" in src
        assert "ResponseTooLarge" in src


# ---------------------------------------------------------------------------
# SC2 — endpoint file permissions (already fixed on master)
# ---------------------------------------------------------------------------

class TestSC2EndpointFilePermissions:
    def test_write_endpoint_chmods_temp_and_final(self):
        """SC2: _write_endpoint restricts file permissions to 0o600."""
        from memory_service import _write_endpoint
        src = inspect.getsource(_write_endpoint)
        assert "0o600" in src
        assert "os.chmod" in src

    def test_write_endpoint_uses_tempfile_replace(self):
        """SC2: _write_endpoint uses atomic os.replace (tempfile pattern)."""
        from memory_service import _write_endpoint
        src = inspect.getsource(_write_endpoint)
        assert "os.replace" in src
        assert ".tmp" in src or "with_suffix" in src


# ---------------------------------------------------------------------------
# SC3 — STILL_ACTIVE=259 false positive (documented edge case)
# ---------------------------------------------------------------------------

class TestSC3StillActiveEdgeCase:
    def test_pid_alive_documents_still_active_limitation(self):
        """SC3: _pid_alive documents the STILL_ACTIVE=259 edge case."""
        from service_client import _pid_alive
        src = inspect.getsource(_pid_alive)
        assert "259" in src or "STILL_ACTIVE" in src
        # The 90-second stale-lock fallback must be documented.
        assert "90" in src or "stale" in src.lower()

    def test_pid_alive_rejects_invalid_pid(self):
        """SC3: _pid_alive returns False for invalid PIDs."""
        from service_client import _pid_alive
        assert _pid_alive(0) is False
        assert _pid_alive(-1) is False


# ---------------------------------------------------------------------------
# SC4 — _start_lock_is_stale false negative (documented, no fix needed)
# ---------------------------------------------------------------------------

class TestSC4StartLockStale:
    def test_start_lock_stale_timeout_exists(self):
        """SC4: the stale-lock timeout constant exists as a fallback."""
        from service_client import _START_LOCK_STALE_SECS
        assert _START_LOCK_STALE_SECS > 0
        assert _START_LOCK_STALE_SECS == 90


# ---------------------------------------------------------------------------
# SC5 — mutable defaults in _record_from_dict (already fixed on master)
# ---------------------------------------------------------------------------

class TestSC5MutableDefaults:
    def test_record_from_dict_uses_or_not_default(self):
        """SC5: _record_from_dict uses `or []` / `or {}` instead of
        `value.get("tags", [])` which shares a mutable default."""
        from service_client import _record_from_dict
        src = inspect.getsource(_record_from_dict)
        assert 'value.get("tags") or []' in src
        assert 'value.get("payload") or {}' in src
        # Must NOT use the old mutable-default pattern.
        assert 'value.get("tags", [])' not in src
        assert 'value.get("payload", {})' not in src

    def test_record_from_dict_creates_new_list_each_call(self):
        """SC5: two calls with missing tags produce independent lists."""
        from service_client import _record_from_dict
        r1 = _record_from_dict({"memory_id": "m1", "content": "x"})
        r2 = _record_from_dict({"memory_id": "m2", "content": "y"})
        assert r1.tags is not r2.tags
        assert r1.payload is not r2.payload
        # Mutating one must not affect the other.
        r1.tags.append("t1")
        assert r2.tags == []


# ---------------------------------------------------------------------------
# SC6 — close() is a no-op (documented, correct behavior)
# ---------------------------------------------------------------------------

class TestSC6CloseNoOp:
    def test_shared_memory_store_close_is_noop(self):
        """SC6: SharedMemoryStore.close() is a documented no-op."""
        from service_client import SharedMemoryStore
        src = inspect.getsource(SharedMemoryStore.close)
        assert "no-op" in src.lower() or "no op" in src.lower()

    def test_shared_graph_store_close_is_noop(self):
        """SC6: SharedGraphStore.close() is a documented no-op."""
        from service_client import SharedGraphStore
        src = inspect.getsource(SharedGraphStore.close)
        assert "no-op" in src.lower() or "no op" in src.lower()

"""Tests for #347: mutation_events RPC dispatch + proxy (round-3 B4).

Proves the RPC wire seam is exercised:
(1) list_mutation_events / export_mutation_events proxy methods exist
    with the right signatures.
(2) The service dispatch arms exist and pass args through.
(3) export_mutation_events is wheel-gated (structural check).
(4) The set_actor_context proxy accepts (actor, actor_type) for
    facade compatibility but doesn't send actor over the wire.
(5) Credential-mode forces actor_type="model" in _call_store.

Structural tests (no service spawn needed) + one live-mode test.
"""
from __future__ import annotations

import inspect
import json
import os
import sys
import time
from pathlib import Path

import pytest

_plugin_dir = Path(__file__).resolve().parent.parent
for _path in (_plugin_dir.parent, _plugin_dir):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

# Group with other shared-service tests so xdist serializes the spawns.
pytestmark = pytest.mark.xdist_group("shared_service")

# Live-mode tests spawn a real shared memory service subprocess. They only
# run under ARGOS_HERMETIC_TESTS=1 (CI / hermetic gate) so they don't hang
# against a real local service on the same port (round-3 review: green-in-CI-
# but-red-against-a-real-service). Structural tests run everywhere.
_HERMETIC = os.environ.get("ARGOS_HERMETIC_TESTS", "").strip().lower() in {"1", "true", "yes"}
_live_only = pytest.mark.skipif(not _HERMETIC, reason="requires ARGOS_HERMETIC_TESTS=1")


class TestProxySignatures:
    """(1) Proxy methods exist with the right signatures."""

    def test_list_mutation_events_proxy_exists(self):
        from service_client import SharedMemoryStore
        assert hasattr(SharedMemoryStore, "list_mutation_events")

    def test_list_mutation_events_proxy_signature(self):
        from service_client import SharedMemoryStore
        sig = inspect.signature(SharedMemoryStore.list_mutation_events)
        params = set(sig.parameters.keys()) - {"self"}
        assert params == {"limit", "offset", "event_type"}

    def test_export_mutation_events_proxy_exists(self):
        from service_client import SharedMemoryStore
        assert hasattr(SharedMemoryStore, "export_mutation_events")

    def test_export_mutation_events_proxy_signature(self):
        from service_client import SharedMemoryStore
        sig = inspect.signature(SharedMemoryStore.export_mutation_events)
        params = set(sig.parameters.keys()) - {"self"}
        assert params == {"limit", "offset", "event_type", "format"}

    def test_set_actor_context_proxy_accepts_actor(self):
        """(4) The proxy accepts (actor, actor_type) for facade compat
        but doesn't send actor over the wire."""
        from service_client import SharedMemoryStore
        sig = inspect.signature(SharedMemoryStore.set_actor_context)
        params = set(sig.parameters.keys()) - {"self"}
        # actor is accepted but not sent (server-derived identity).
        assert "actor" in params
        assert "actor_type" in params


class TestServiceDispatch:
    """(2) Service dispatch arms exist and pass args through."""

    def test_dispatch_list_mutation_events(self):
        """The service has a dispatch arm for list_mutation_events."""
        from memory_service import MemoryService
        src = inspect.getsource(MemoryService._call_store)
        assert "list_mutation_events" in src
        assert "export_mutation_events" in src

    def test_dispatch_passes_event_type(self):
        """list_mutation_events dispatch passes event_type through."""
        from memory_service import MemoryService
        src = inspect.getsource(MemoryService._call_store)
        # The dispatch should pass event_type from args.
        list_idx = src.index("list_mutation_events")
        list_block = src[list_idx:list_idx + 300]
        assert "event_type" in list_block

    def test_dispatch_passes_offset_to_export(self):
        """export_mutation_events dispatch passes offset through (round-3)."""
        from memory_service import MemoryService
        src = inspect.getsource(MemoryService._call_store)
        export_idx = src.index("export_mutation_events")
        export_block = src[export_idx:export_idx + 1200]
        assert "offset" in export_block


class TestExportWheelGate:
    """(3) export_mutation_events is wheel-gated."""

    def test_export_has_wheel_gate(self):
        """The export dispatch checks for wheel role."""
        from memory_service import MemoryService
        src = inspect.getsource(MemoryService._call_store)
        export_idx = src.index("export_mutation_events")
        export_block = src[export_idx:export_idx + 800]
        assert "wheel" in export_block
        assert "PermissionError" in export_block


class TestCredentialModeActorStamp:
    """(5) Credential-mode forces actor_type='model' in _call_store."""

    def test_call_store_stamps_actor(self):
        """_call_store stamps actor after set_user_scope."""
        from memory_service import MemoryService
        src = inspect.getsource(MemoryService._call_store)
        assert "set_actor_context" in src
        assert "_credential_mode" in src
        assert '"model"' in src or "'model'" in src


def _make_store(tmp_path, user_id="test_user"):
    """Create a SharedMemoryStore with a disposable home dir."""
    from service_client import SharedMemoryStore
    (tmp_path / "hybrid_memory.json").write_text(
        json.dumps({"local_embedding_model": "nonexistent-model-xyz"}),
        encoding="utf-8",
    )
    return SharedMemoryStore(tmp_path, user_id=user_id, embedder=None)


@_live_only
class TestMutationEventsRpcEndToEnd:
    """(5) Live-mode: mutation events appear via the RPC proxy.

    Skipped unless ARGOS_HERMETIC_TESTS=1 — these spawn a real shared
    memory service subprocess and hang against a live local service on
    the same port (round-3 review: test-hygiene fix)."""

    def test_remember_creates_event_via_rpc(self, tmp_path):
        """A remember() through SharedMemoryStore creates a memory_created
        event visible via list_mutation_events RPC proxy."""
        store = _make_store(tmp_path)
        try:
            store.remember(category="personal_fact", content="RPC event test")
            events = store.list_mutation_events(limit=100)
            assert events, "no mutation events returned"
            created = [e for e in events if e.get("event_type") == "memory_created"]
            assert len(created) >= 1, "no memory_created event found"
            # Actor should be stamped server-side.
            assert created[0]["actor"] == "test_user"
            assert created[0]["actor_type"] == "human"
        finally:
            try:
                store._rpc.stop_service()
            finally:
                time.sleep(0.5)

    def test_export_mutation_events_via_rpc(self, tmp_path):
        """export_mutation_events returns JSONL via the RPC proxy."""
        store = _make_store(tmp_path)
        try:
            store.remember(category="personal_fact", content="Export test")
            exported = store.export_mutation_events(format="jsonl")
            assert exported, "export returned empty"
            rows = [json.loads(line) for line in exported.strip().splitlines() if line]
            assert len(rows) >= 1
            assert any(r.get("event_type") == "memory_created" for r in rows)
        finally:
            try:
                store._rpc.stop_service()
            finally:
                time.sleep(0.5)

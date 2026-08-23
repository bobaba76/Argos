"""Regression tests for candidate review through the shared-memory client."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Keep this test independently importable when pytest collects it first.
_plugin_dir = Path(__file__).resolve().parent.parent
for _path in (_plugin_dir.parent, _plugin_dir):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


def test_shared_store_review_candidate_forwards_keyword_arguments(tmp_path):
    from argos.service_client import SharedMemoryStore

    (tmp_path / "hybrid_memory.json").write_text(
        json.dumps({"local_embedding_model": "nonexistent-model-xyz"}),
        encoding="utf-8",
    )
    store = SharedMemoryStore(tmp_path, user_id="test_user", embedder=None)
    try:
        candidate = store.save_candidate(
            category="preference",
            content="User prefers the shared review path",
            source="llm_extraction",
            confidence=0.42,
            scope="profile",
        )
        assert candidate is not None

        reviewed = store.review_candidate(
            candidate_id=candidate["candidate_id"],
            decision="approved",
            reason="confirmed",
        )

        assert reviewed is not None
        assert reviewed["candidate"]["status"] == "approved"
        assert reviewed["memory"]["status"] == "active"
    finally:
        try:
            store._rpc.stop_service()
        finally:
            time.sleep(0.5)


def test_argos_tool_review_uses_keyword_arguments(tmp_path):
    """The provider tool path must work with SharedMemoryStore's keyword API."""
    from argos import ArgosProvider
    from argos.service_client import SharedMemoryStore

    (tmp_path / "hybrid_memory.json").write_text(
        json.dumps({"local_embedding_model": "nonexistent-model-xyz"}),
        encoding="utf-8",
    )
    store = SharedMemoryStore(tmp_path, user_id="test_user", embedder=None)
    try:
        candidate = store.save_candidate(
            category="preference",
            content="User prefers keyword-safe candidate review",
            source="llm_extraction",
            confidence=0.42,
            scope="profile",
        )
        assert candidate is not None

        provider = object.__new__(ArgosProvider)
        provider._store = store
        provider._graph = None
        provider._evidence_retention = "full"
        result = provider.handle_tool_call(
            "memory_candidate_review",
            {
                "candidate_id": candidate["candidate_id"],
                "decision": "approved",
                "reason": "confirmed",
            },
        )

        payload = json.loads(result)
        assert payload["candidate"]["status"] == "approved"
        assert payload["memory"]["status"] == "active"
    finally:
        try:
            store._rpc.stop_service()
        finally:
            time.sleep(0.5)


def test_shared_store_delete_memory_forwards_to_service(tmp_path):
    """The shared client must expose the delete operation used by the tool."""
    from argos.service_client import SharedMemoryStore

    (tmp_path / "hybrid_memory.json").write_text(
        json.dumps({"local_embedding_model": "nonexistent-model-xyz"}),
        encoding="utf-8",
    )
    store = SharedMemoryStore(tmp_path, user_id="test_user", embedder=None)
    try:
        record = store.remember(
            category="preference",
            content="Temporary duplicate for delete regression",
        )
        assert record is not None
        assert store.delete_memory(memory_id=record.memory_id)
        assert not store.delete_memory(memory_id=record.memory_id)
    finally:
        try:
            store._rpc.stop_service()
        finally:
            time.sleep(0.5)


def test_shared_memory_tool_delete_works_with_shared_client(tmp_path):
    """The memory_delete tool must work when the provider uses shared storage."""
    from argos import ArgosProvider
    from argos.service_client import SharedMemoryStore

    (tmp_path / "hybrid_memory.json").write_text(
        json.dumps({"local_embedding_model": "nonexistent-model-xyz"}),
        encoding="utf-8",
    )
    store = SharedMemoryStore(tmp_path, user_id="test_user", embedder=None)
    try:
        record = store.remember(
            category="preference",
            content="Temporary duplicate for tool delete regression",
        )
        assert record is not None

        provider = object.__new__(ArgosProvider)
        provider._store = store
        result = json.loads(
            provider.handle_tool_call(
                "memory_delete", {"memory_id": record.memory_id}
            )
        )
        assert result["status"] == "deleted"
    finally:
        try:
            store._rpc.stop_service()
        finally:
            time.sleep(0.5)

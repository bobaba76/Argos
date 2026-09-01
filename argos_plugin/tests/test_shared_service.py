"""End-to-end tests for the single-owner shared memory service."""
from __future__ import annotations

import json
import time

import pytest

# Every test here spawns the shared memory service subprocess. Group them
# onto a single xdist worker (pytest -n auto --dist loadgroup) so parallel
# runs serialize the spawns instead of racing them (#98).
pytestmark = pytest.mark.xdist_group("shared_service")


def test_shared_service_store_and_graph(tmp_path):
    from service_client import SharedGraphStore, SharedMemoryStore

    (tmp_path / "hybrid_memory.json").write_text(
        json.dumps({"local_embedding_model": "nonexistent-model-xyz"}),
        encoding="utf-8",
    )
    store = SharedMemoryStore(tmp_path, user_id="test_user", embedder=None)
    try:
        record = store.remember(
            category="preference",
            content="User prefers the shared memory service",
        )
        assert record is not None
        assert store.count() == 1
        assert store.search("shared memory service", limit=5)

        graph = SharedGraphStore(tmp_path, user_id="test_user")
        graph.add_relationship(
            source="user",
            source_type="person",
            relation="uses",
            target="Hermes",
            target_type="agent",
        )
        assert graph.search_graph("Hermes")
        graph.close()
    finally:
        try:
            store._rpc.stop_service()
        finally:
            time.sleep(0.5)


def test_shared_service_search_forwards_provider_kwargs(tmp_path):
    """Regression: provider._search_memories ALWAYS passes include_expired on
    every search (memory_search tool + prefetch). The RPC client must accept
    and forward it — a closed signature here TypeErrors on every search in
    shared_service mode (the production default)."""
    from service_client import SharedMemoryStore

    (tmp_path / "hybrid_memory.json").write_text(
        json.dumps({"local_embedding_model": "nonexistent-model-xyz"}),
        encoding="utf-8",
    )
    store = SharedMemoryStore(tmp_path, user_id="test_user", embedder=None)
    try:
        active = store.remember(
            category="context_note",
            content="User remembers the timed service memory",
        )
        expired = store.remember(
            category="context_note",
            content="Timed service memory expired long ago",
            expires_at="2000-01-01T00:00:00+00:00",
        )
        assert active is not None and expired is not None

        # The provider's exact calling convention (search tool + prefetch path).
        hidden = store.search(
            "timed service memory", limit=5,
            category_filter=None, project_id=None,
            suppress_retrieval=True, include_expired=False,
        )
        visible = store.search(
            "timed service memory", limit=5,
            category_filter=None, project_id=None,
            suppress_retrieval=True, include_expired=True,
        )
        hidden_ids = {r.memory_id for r in hidden}
        visible_ids = {r.memory_id for r in visible}
        assert active.memory_id in hidden_ids
        assert expired.memory_id not in hidden_ids
        assert expired.memory_id in visible_ids
    finally:
        try:
            store._rpc.stop_service()
        finally:
            time.sleep(0.5)


def test_shared_service_search_forwards_include_closed(tmp_path):
    """Regression (25/8 incident): history-at-current-time threads include_closed
    through _search_memories on EVERY prefetch. The proxy rejected it (TypeError)
    and the service dispatch silently dropped it, so under storage_mode=
    shared_service every live search failed fail-soft into EMPTY injections.
    Locks the kwarg end-to-end: client signature -> wire -> dispatch -> store,
    including that a closed (superseded) version actually comes back labelled."""
    from service_client import SharedMemoryStore

    (tmp_path / "hybrid_memory.json").write_text(
        json.dumps({"local_embedding_model": "nonexistent-model-xyz"}),
        encoding="utf-8",
    )
    store = SharedMemoryStore(tmp_path, user_id="test_user", embedder=None)
    try:
        v1 = store.remember(
            category="personal_fact",
            content="Alex lives in Johannesburg",
        )
        # Proxy convention: memory_id must be passed as a keyword.
        v2 = store.update_memory(memory_id=v1.memory_id, content="Alex lives in Centurion")
        assert v1 is not None and v2 is not None

        # Default: closed version stays hidden.
        current = store.search("where does Alex live", limit=10)
        contents = [r.content for r in current]
        assert "Alex lives in Centurion" in contents
        assert all("Johannesburg" not in c for c in contents)

        # Widened: both versions return; the closed row must carry valid_to
        # set so the injection formatter can label it "(previously)".
        widened = store.search(
            "where does Alex live", limit=10,
            category_filter=None, project_id=None,
            suppress_retrieval=True, include_expired=False,
            include_closed=True,
        )
        by_content = {r.content: r for r in widened}
        assert "Alex lives in Johannesburg" in by_content, (
            "closed version missing after RPC round trip"
        )
        assert "Alex lives in Centurion" in by_content
        assert by_content["Alex lives in Johannesburg"].valid_to is not None
        assert by_content["Alex lives in Centurion"].valid_to is None
    finally:
        try:
            store._rpc.stop_service()
        finally:
            time.sleep(0.5)


def test_shared_graph_index_and_traverse_round_trip(tmp_path):
    """Priority 3 graph indexing/traversal must work through shared RPC."""
    from service_client import SharedGraphStore

    (tmp_path / "hybrid_memory.json").write_text(
        json.dumps({"local_embedding_model": "nonexistent-model-xyz"}),
        encoding="utf-8",
    )
    graph = SharedGraphStore(tmp_path, user_id="test_user")
    try:
        indexed = graph.index_memory(
            memory_id="shared-m1",
            category="insight",
            content="I just realized shame affects my work patterns",
            tags=["insight", "shame", "work"],
        )
        assert indexed >= 1
        result = graph.traverse_graph("work", depth=2)
        node_ids = {node["id"] for node in result["nodes"]}
        assert "memory:shared-m1" in node_ids
        assert graph.remove_memory("shared-m1") is True
        assert graph.traverse_graph("work", depth=2)["nodes"] == []
    finally:
        try:
            graph._rpc.stop_service()
        finally:
            time.sleep(0.5)


def test_shared_store_review_candidate_forwards_keyword_arguments(tmp_path):
    from service_client import SharedMemoryStore

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
        reviewed = store.review_candidate(
            candidate_id=candidate["candidate_id"],
            decision="approved",
            reason="confirmed",
        )
        assert reviewed["candidate"]["status"] == "approved"
        assert reviewed["memory"]["status"] == "active"

        assert store.delete_memory(memory_id=reviewed["memory"]["memory_id"])
        assert not store.delete_memory(memory_id=reviewed["memory"]["memory_id"])
    finally:
        try:
            store._rpc.stop_service()
        finally:
            time.sleep(0.5)
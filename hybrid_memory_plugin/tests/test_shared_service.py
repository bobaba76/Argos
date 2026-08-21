"""End-to-end tests for the single-owner shared memory service."""
from __future__ import annotations

import json
import time


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

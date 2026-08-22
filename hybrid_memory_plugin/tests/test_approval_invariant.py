"""Approval-boundary invariant tests (Perseus review point 4).

The storage layer must enforce that an unsupervised automatic review can
never write the "approved" transition — that status is reserved for the
agent-facing confirmation tool (memory_candidate_review) and manual
callers. The ceiling for auto-review approval is "reviewed_approved"
(LLM-approved, awaiting user confirmation).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Keep this test independently importable when pytest collects it first.
_plugin_dir = Path(__file__).resolve().parent.parent
for _path in (_plugin_dir.parent, _plugin_dir):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import pytest


def _make_store(tmp_path):
    from hybrid_memory.store import DuckDBMemoryStore

    store = DuckDBMemoryStore(tmp_path / "test.duckdb", user_id="test_user")
    return store


def _add_pending(store, content="Alex prefers flat white coffee"):
    candidate = store.save_candidate(
        category="preference",
        content=content,
        source="llm_extraction",
        confidence=0.9,
        scope="profile",
        evidence_text="My name is Alex and I prefer flat white coffee.",
    )
    assert candidate is not None
    return candidate["candidate_id"]


def test_auto_review_cannot_set_approved(tmp_path):
    """The core invariant: storage refuses approved from an auto source."""
    store = _make_store(tmp_path)
    try:
        cid = _add_pending(store)
        with pytest.raises(ValueError, match="approval invariant"):
            store.review_candidate(
                candidate_id=cid,
                decision="approved",
                reason="auto said yes",
                review_source="auto_review",
            )
        # Candidate is untouched — still pending.
        cands = store.list_candidates(candidate_id=cid, limit=1)
        assert cands[0]["status"] == "pending"
    finally:
        store.close()


def test_auto_review_can_set_reviewed_approved(tmp_path):
    """Auto-review approval lands at the reviewed_approved ceiling."""
    store = _make_store(tmp_path)
    try:
        cid = _add_pending(store)
        result = store.review_candidate(
            candidate_id=cid,
            decision="reviewed_approved",
            reason="high confidence, non-sensitive",
            review_source="auto_review",
        )
        assert result is not None
        assert result["candidate"]["status"] == "reviewed_approved"
        assert result["memory"] is not None
        # The memory exists and is active — but labelled reviewer-approved.
        mem = store.get_memories_by_ids([result["memory"]["memory_id"]])
        assert mem and mem[0].status == "active"
    finally:
        store.close()


def test_tool_path_can_set_approved(tmp_path):
    """The agent-facing confirmation path may still write approved."""
    store = _make_store(tmp_path)
    try:
        cid = _add_pending(store)
        result = store.review_candidate(
            candidate_id=cid,
            decision="approved",
            reason="user confirmed",
        )
        assert result is not None
        assert result["candidate"]["status"] == "approved"
        assert result["memory"] is not None
    finally:
        store.close()


def test_auto_path_maps_approve_to_reviewed_approved(tmp_path, monkeypatch):
    """The provider's auto-reviewer maps reviewer 'approve' to reviewed_approved."""
    import importlib

    hm = importlib.import_module("hybrid_memory_plugin")

    fake_review = {
        "decision": "approve",
        "reason": "confident and non-sensitive",
        "confidence": 0.95,
        "review_model": "memory_review",
        "durability": "durable",
        "scope": "profile",
    }
    monkeypatch.setattr(
        hm, "review_candidate_with_llm", lambda *a, **k: fake_review
    )

    store = _make_store(tmp_path)
    try:
        cid = _add_pending(store)

        class _StubProvider(hm.HybridMemoryProvider):
            def __init__(self, store):
                self._store = store
                self._llm_model = None
                self._llm_provider = None
                self._evidence_retention = "full"
                self._expiry_auto_suggest = False

            def initialize(self):
                pass

            def is_available(self):
                return True

            def name(self):
                return "stub"

            def get_tool_schemas(self):
                return []

        provider = _StubProvider(store)

        candidate = store.list_candidates(candidate_id=cid, limit=1)[0]
        provider._review_candidate(candidate)

        cands = store.list_candidates(candidate_id=cid, limit=1)
        assert cands[0]["status"] == "reviewed_approved", cands[0]["status"]
    finally:
        store.close()
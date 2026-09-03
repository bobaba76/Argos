"""Guarded confirmation surfacing (#99 rework, 3/9).

The prefetch path surfaces ONE pending user-confirmation per turn:
- genuine confirmation needs only (reviewer failures are never
  "should this be saved?" prompts)
- a candidate that was already surfaced is never shown again
  (ledger persisted in system_state, survives restarts)
- one candidate max per call — a turn can never dump the whole backlog
"""
from __future__ import annotations

import json

import pytest

from confirmation import (
    build_confirmation_block,
    choose_confirmation_block,
)
from provider_retrieval import ProviderRetrievalMixin


def _cand(cid: str, review_model: str = "memory_review", content: str = "Some fact worth saving") -> dict:
    return {
        "candidate_id": cid,
        "category": "context_note",
        "content": content,
        "review_reason": "needs human decision",
        "review_model": review_model,
    }


class TestChooseConfirmationBlock:
    def test_returns_nothing_when_no_candidates(self):
        assert choose_confirmation_block([]) == ("", None)
        assert choose_confirmation_block(None) == ("", None)

    def test_picks_first_genuine_candidate(self):
        cands = [
            _cand("c2", review_model="memory_review"),
            _cand("c1", review_model="memory_review"),
        ]
        block, cid = choose_confirmation_block(cands)
        assert cid == "c2"
        assert "c2" in block
        assert "untrusted memory data" in block

    def test_skips_reviewer_failures(self):
        cands = [
            _cand("bad1", review_model="reviewer_unavailable"),
            _cand("bad2", review_model="evidence_gate"),
            _cand("bad3", review_model="egress_gate_unavailable"),
            _cand("good", review_model="memory_review"),
        ]
        block, cid = choose_confirmation_block(cands)
        assert cid == "good"
        assert "bad1" not in block and "bad2" not in block and "bad3" not in block

    def test_all_reviewer_failures_returns_nothing(self):
        block, cid = choose_confirmation_block(
            [_cand("x", review_model="egress_gate_unavailable")]
        )
        assert (block, cid) == ("", None)

    def test_never_resurfaces_already_surfaced(self):
        cands = [_cand("seen1"), _cand("fresh")]
        block, cid = choose_confirmation_block(cands, already_surfaced=["seen1"])
        assert cid == "fresh"

    def test_exhausted_backlog_returns_nothing(self):
        cands = [_cand("seen1"), _cand("seen2")]
        block, cid = choose_confirmation_block(
            cands, already_surfaced=["seen1", "seen2"]
        )
        assert (block, cid) == ("", None)

    def test_skips_blank_candidate_id(self):
        block, cid = choose_confirmation_block([_cand(""), _cand("real")])
        assert cid == "real"

    def test_one_candidate_max_per_call(self):
        cands = [_cand("a"), _cand("b"), _cand("c")]
        block, cid = choose_confirmation_block(cands)
        assert cid == "a"
        assert block.count("candidate_id") == 1

    def test_block_is_bounded(self):
        long_content = "x" * 5000
        block, cid = choose_confirmation_block([_cand("a", content=long_content)])
        # build_confirmation_block truncates content to 1200 chars
        assert "x" * 1200 in block
        assert "x" * 1201 not in block


class _FakeStore:
    def __init__(self) -> None:
        self.kv: dict = {}

    def get_state(self, key: str) -> str | None:
        return self.kv.get(key)

    def set_state(self, key: str, value: str) -> None:
        self.kv[key] = value

    def list_candidates(self, **kwargs):
        return []


class TestSurfacingLedger:
    def test_roundtrip_across_instances(self):
        store = _FakeStore()
        p1 = ProviderRetrievalMixin()
        p1._store = store
        p2 = ProviderRetrievalMixin()
        p2._store = store

        assert p1._surfaced_confirmation_ids() == set()
        p1._mark_surfaced_confirmation("cand-abc")
        assert p2._surfaced_confirmation_ids() == {"cand-abc"}
        p1._mark_surfaced_confirmation("cand-def")
        assert p2._surfaced_confirmation_ids() == {"cand-abc", "cand-def"}
        assert store.kv["surfaced_confirmation_ids"] == json.dumps(
            ["cand-abc", "cand-def"]
        )

    def test_mark_none_or_blank_is_noop(self):
        store = _FakeStore()
        p = ProviderRetrievalMixin()
        p._store = store
        p._mark_surfaced_confirmation(None)
        p._mark_surfaced_confirmation("")
        assert store.kv == {}

    def test_fail_soft_when_store_missing(self):
        p = ProviderRetrievalMixin()
        p._store = None
        assert p._surfaced_confirmation_ids() == set()
        p._mark_surfaced_confirmation("x")  # must not raise
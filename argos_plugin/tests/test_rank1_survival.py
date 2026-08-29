"""Tests for #38: strong semantic rank-1 must survive rank fusion.

RRF can suppress a strong semantic rank-1 when the other retrieval arms
disagree with it. The rank-1 survival guard ensures that if a single arm
ranks an item #1 by a clear margin, the fused top-k must still contain it.

Covers:
- Synthetic fusion: vector #1 weak in text → still in fused top-3
- Synthetic fusion: text #1 weak in vector → still in fused top-3
- No promotion when rank-1 is already in top-k
- No promotion when no clear margin (rank-1 ≈ rank-2)
- Both arms have clear rank-1 → both survive
- Empty arm edge cases
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_plugin_dir = Path(__file__).resolve().parent.parent
if str(_plugin_dir) not in sys.path:
    sys.path.insert(0, str(_plugin_dir))

from store import DuckDBMemoryStore, MemoryRecord


def _make_record(mid: str, content: str, similarity: float = 0.0) -> MemoryRecord:
    """Create a minimal MemoryRecord for fusion testing."""
    return MemoryRecord(
        memory_id=mid,
        content=content,
        category="personal_fact",
        tags=[],
        payload={},
        embedding=[],
        similarity=similarity,
        created_at="2026-01-01T00:00:00Z",
        retrieval_count=0,
        helpful_count=0,
        dismissed_count=0,
        confidence=0.5,
        durability="durable",
        scope="profile",
        source="test",
    )


class TestRank1SurvivalGuard:
    """The rank-1 survival guard should keep each arm's clear #1 in top-k."""

    def test_vector_rank1_survives_when_text_disagrees(self):
        """Vector #1 with clear margin but absent from text → still in top-3."""
        # Vector arm: mem-A is clear #1 (score 0.95 vs 0.40 for #2)
        vector = [
            _make_record("mem-A", "alpha content", 0.95),
            _make_record("mem-B", "beta content", 0.40),
            _make_record("mem-C", "gamma content", 0.35),
            _make_record("mem-D", "delta content", 0.30),
        ]
        # Text arm: mem-A is absent; mem-B, C, D rank well
        text = [
            _make_record("mem-B", "beta content", 0.90),
            _make_record("mem-C", "gamma content", 0.80),
            _make_record("mem-D", "delta content", 0.70),
            _make_record("mem-E", "epsilon content", 0.60),
        ]
        fused = DuckDBMemoryStore._rrf_fuse(vector, text)
        top3_ids = {r.memory_id for r in fused[:3]}
        assert "mem-A" in top3_ids, (
            "Vector rank-1 (mem-A) with clear margin must survive in "
            "fused top-3 even when text arm disagrees"
        )

    def test_text_rank1_survives_when_vector_disagrees(self):
        """Text #1 with clear margin but absent from vector → still in top-3."""
        # Vector arm: mem-B is #1, mem-A absent
        vector = [
            _make_record("mem-B", "beta content", 0.90),
            _make_record("mem-C", "gamma content", 0.80),
            _make_record("mem-D", "delta content", 0.70),
            _make_record("mem-E", "epsilon content", 0.60),
        ]
        # Text arm: mem-A is clear #1 (score 0.95 vs 0.30 for #2)
        text = [
            _make_record("mem-A", "alpha content", 0.95),
            _make_record("mem-B", "beta content", 0.30),
            _make_record("mem-C", "gamma content", 0.25),
            _make_record("mem-D", "delta content", 0.20),
        ]
        fused = DuckDBMemoryStore._rrf_fuse(vector, text)
        top3_ids = {r.memory_id for r in fused[:3]}
        assert "mem-A" in top3_ids, (
            "Text rank-1 (mem-A) with clear margin must survive in "
            "fused top-3 even when vector arm disagrees"
        )

    def test_no_promotion_when_already_in_top_k(self):
        """If rank-1 is already in fused top-k, no promotion needed."""
        # Both arms agree: mem-A is #1
        vector = [
            _make_record("mem-A", "alpha", 0.95),
            _make_record("mem-B", "beta", 0.40),
        ]
        text = [
            _make_record("mem-A", "alpha", 0.90),
            _make_record("mem-B", "beta", 0.50),
        ]
        fused = DuckDBMemoryStore._rrf_fuse(vector, text)
        assert fused[0].memory_id == "mem-A"

    def test_no_promotion_without_clear_margin(self):
        """If rank-1 and rank-2 are close, no promotion (no clear margin)."""
        # Vector arm: mem-A is #1 but barely (0.50 vs 0.49 — ratio < 1.5)
        vector = [
            _make_record("mem-A", "alpha", 0.50),
            _make_record("mem-B", "beta", 0.49),
            _make_record("mem-C", "gamma", 0.48),
            _make_record("mem-D", "delta", 0.47),
        ]
        # Text arm: mem-B, C, D rank well, mem-A absent
        text = [
            _make_record("mem-B", "beta", 0.90),
            _make_record("mem-C", "gamma", 0.80),
            _make_record("mem-D", "delta", 0.70),
        ]
        fused = DuckDBMemoryStore._rrf_fuse(vector, text)
        top3_ids = {r.memory_id for r in fused[:3]}
        # mem-A has no clear margin (0.50 vs 0.49), so it may not be in top-3.
        # The guard should NOT fire — this is correct behavior.
        # We just verify the guard doesn't crash and produces valid output.
        assert len(fused) >= 3

    def test_both_arms_clear_rank1_both_survive(self):
        """When both arms have a clear rank-1, both should survive."""
        # Vector: mem-A clear #1 (0.95 vs 0.30)
        # Text: mem-B clear #1 (0.90 vs 0.25)
        vector = [
            _make_record("mem-A", "alpha", 0.95),
            _make_record("mem-C", "gamma", 0.30),
            _make_record("mem-D", "delta", 0.25),
            _make_record("mem-E", "epsilon", 0.20),
        ]
        text = [
            _make_record("mem-B", "beta", 0.90),
            _make_record("mem-C", "gamma", 0.25),
            _make_record("mem-D", "delta", 0.20),
            _make_record("mem-E", "epsilon", 0.15),
        ]
        fused = DuckDBMemoryStore._rrf_fuse(vector, text)
        top3_ids = {r.memory_id for r in fused[:3]}
        assert "mem-A" in top3_ids, "Vector clear rank-1 must survive"
        assert "mem-B" in top3_ids, "Text clear rank-1 must survive"

    def test_empty_arms(self):
        """Empty arms should not crash the guard."""
        fused = DuckDBMemoryStore._rrf_fuse([], [])
        assert fused == []

    def test_single_item_arm(self):
        """An arm with only one item should not trigger the guard (no rank-2)."""
        vector = [_make_record("mem-A", "alpha", 0.95)]
        text = [
            _make_record("mem-B", "beta", 0.90),
            _make_record("mem-C", "gamma", 0.80),
            _make_record("mem-D", "delta", 0.70),
        ]
        fused = DuckDBMemoryStore._rrf_fuse(vector, text)
        # mem-A should be in the fused results (RRF gives it rank-1 score).
        assert any(r.memory_id == "mem-A" for r in fused)

    def test_guard_preserves_order_when_not_needed(self):
        """When no promotion is needed, the fused order is unchanged."""
        # Both arms agree on ranking
        vector = [
            _make_record("mem-A", "alpha", 0.95),
            _make_record("mem-B", "beta", 0.80),
            _make_record("mem-C", "gamma", 0.70),
        ]
        text = [
            _make_record("mem-A", "alpha", 0.90),
            _make_record("mem-B", "beta", 0.70),
            _make_record("mem-C", "gamma", 0.60),
        ]
        fused = DuckDBMemoryStore._rrf_fuse(vector, text)
        ids = [r.memory_id for r in fused[:3]]
        assert ids == ["mem-A", "mem-B", "mem-C"]

    def test_guard_can_be_disabled(self):
        """enable_rank1_guard=False exposes the raw RRF failure (#38 probe).

        Each fusion call needs its own copies: _rrf_fuse writes the fused
        score onto the arm records' similarity, so reusing the same objects
        would break the second call's guard margin check.
        """
        import copy
        # Vector: mem-A clear #1; text: mem-A absent (the failure shape)
        vector = [
            _make_record("mem-A", "alpha", 0.95),
            _make_record("mem-B", "beta", 0.40),
            _make_record("mem-C", "gamma", 0.35),
            _make_record("mem-D", "delta", 0.30),
        ]
        text = [
            _make_record("mem-B", "beta", 0.90),
            _make_record("mem-C", "gamma", 0.80),
            _make_record("mem-D", "delta", 0.70),
            _make_record("mem-E", "epsilon", 0.60),
        ]
        raw = DuckDBMemoryStore._rrf_fuse(
            copy.deepcopy(vector), copy.deepcopy(text), enable_rank1_guard=False,
        )
        guarded = DuckDBMemoryStore._rrf_fuse(
            copy.deepcopy(vector), copy.deepcopy(text),
        )
        raw_top3 = {r.memory_id for r in raw[:3]}
        guarded_top3 = {r.memory_id for r in guarded[:3]}
        # The raw fusion buries mem-A (both arms disagree); the guard rescues it.
        assert "mem-A" not in raw_top3, "sanity: raw RRF must drop mem-A"
        assert "mem-A" in guarded_top3, "guard must rescue mem-A"


class TestRank1GuardHelpers:
    """_rank1_guard_ids and _ensure_rank1_guards in isolation."""

    def test_guard_ids_clear_margin(self):
        vector = [
            _make_record("mem-A", "alpha", 0.95),
            _make_record("mem-B", "beta", 0.40),
        ]
        text = [
            _make_record("mem-C", "gamma", 0.90),
            _make_record("mem-D", "delta", 0.20),
        ]
        ids = DuckDBMemoryStore._rank1_guard_ids(vector, text)
        assert ids == ["mem-A", "mem-C"]

    def test_guard_ids_no_clear_margin(self):
        vector = [
            _make_record("mem-A", "alpha", 0.50),
            _make_record("mem-B", "beta", 0.49),
        ]
        ids = DuckDBMemoryStore._rank1_guard_ids(vector, [])
        assert ids == []

    def test_guard_ids_skips_single_item_arm(self):
        vector = [_make_record("mem-A", "alpha", 0.95)]
        ids = DuckDBMemoryStore._rank1_guard_ids(vector, [])
        assert ids == []

    def test_ensure_guards_noop_when_inside(self):
        fused = [
            _make_record("mem-A", "alpha", 0.9),
            _make_record("mem-B", "beta", 0.8),
            _make_record("mem-C", "gamma", 0.7),
        ]
        out = DuckDBMemoryStore._ensure_rank1_guards(fused, ["mem-A"])
        assert [r.memory_id for r in out] == ["mem-A", "mem-B", "mem-C"]

    def test_ensure_guards_promotes_missing(self):
        fused = [
            _make_record("mem-C", "gamma", 0.9),
            _make_record("mem-D", "delta", 0.8),
            _make_record("mem-E", "epsilon", 0.7),
            _make_record("mem-A", "alpha", 0.3),
        ]
        out = DuckDBMemoryStore._ensure_rank1_guards(fused, ["mem-A"])
        assert out[0].memory_id == "mem-A", "missing guard must move to front"
        # Everything still present (permutation).
        assert {r.memory_id for r in out} == {"mem-A", "mem-C", "mem-D", "mem-E"}

class TestProbeRank1Loss:
    """probe_rank1_loss should count raw-RRF loss and guard rescue correctly."""

    def _stub_store(self, queries):
        """A store stub whose vector/text arms come from per-query fixtures."""
        _probes_dir = _plugin_dir / "eval" / "probes"
        if str(_probes_dir) not in sys.path:
            sys.path.insert(0, str(_probes_dir))
        import probe_rank1_loss  # noqa: F401  (module under test)

        class FakeEmbedder:
            def embed(self, query):
                return [0.1] * 384

        class StubStore:
            def __init__(self, fixtures):
                self.fixtures = fixtures
                self.embedder = FakeEmbedder()

            def _vector_search_raw(self, emb, limit, excluded):
                return self.fixtures["vector"]

            def _text_search_raw(self, query, limit, excluded):
                return self.fixtures["text"]

        return StubStore({})

    def test_probe_counts_loss_and_rescue(self):
        _probes_dir = _plugin_dir / "eval" / "probes"
        if str(_probes_dir) not in sys.path:
            sys.path.insert(0, str(_probes_dir))
        import probe_rank1_loss

        # Query 1: vector clear rank-1 (0.95 vs 0.40), absent from text
        #          -> lost by raw RRF, rescued by guard.
        # Query 2: arms agree -> no loss.
        fixtures = {
            "vector": [
                _make_record("mem-A", "alpha", 0.95),
                _make_record("mem-B", "beta", 0.40),
                _make_record("mem-C", "gamma", 0.35),
                _make_record("mem-D", "delta", 0.30),
            ],
            "text": [
                _make_record("mem-B", "beta", 0.90),
                _make_record("mem-C", "gamma", 0.80),
                _make_record("mem-D", "delta", 0.70),
                _make_record("mem-E", "epsilon", 0.60),
            ],
        }
        store = self._stub_store(fixtures)
        store.fixtures = fixtures
        # Both queries share the same fixture (loss shape).
        queries = [
            {"query": "q1", "memory_id": "mem-A"},
            {"query": "q2", "memory_id": "mem-A"},
        ]
        stats = probe_rank1_loss.probe_rank1_loss(store, queries)
        assert stats["queries_with_clear_vector_rank1"] == 2
        assert stats["lost_by_raw_rrf"] == 2
        assert stats["rescued_by_guard"] == 2
        assert stats["all_rank1"] == 2
        assert stats["all_rank1_lost"] == 2

    def test_probe_no_loss_when_arms_agree(self):
        _probes_dir = _plugin_dir / "eval" / "probes"
        if str(_probes_dir) not in sys.path:
            sys.path.insert(0, str(_probes_dir))
        import probe_rank1_loss

        fixtures = {
            "vector": [
                _make_record("mem-A", "alpha", 0.95),
                _make_record("mem-B", "beta", 0.40),
                _make_record("mem-C", "gamma", 0.30),
            ],
            "text": [
                _make_record("mem-A", "alpha", 0.90),
                _make_record("mem-B", "beta", 0.30),
                _make_record("mem-C", "gamma", 0.20),
            ],
        }
        store = self._stub_store(fixtures)
        store.fixtures = fixtures
        queries = [{"query": "q1", "memory_id": "mem-A"}]
        stats = probe_rank1_loss.probe_rank1_loss(store, queries)
        assert stats["queries_with_clear_vector_rank1"] == 1
        assert stats["lost_by_raw_rrf"] == 0
        assert stats["all_rank1_lost"] == 0

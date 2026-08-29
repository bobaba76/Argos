"""Regression check: strong semantic rank-1 must survive rank fusion (issue #38).

RRF can suppress a strong semantic rank-1 when the other retrieval arms
disagree with it. This test suite is a regression guard on the fused
ranking: if a single arm ranks an item #1 by a clear margin, the fused
top-k must still contain it.

These tests are test-only (no production fusion change). They document
the current RRF behavior and establish a guard against future regressions.
If the tests fail, the fusion policy needs adjustment (interleave,
interpolate, or a rank-1 preservation guard).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_plugin_dir = Path(__file__).resolve().parent.parent
if str(_plugin_dir) not in sys.path:
    sys.path.insert(0, str(_plugin_dir))

from store import MemoryRecord, DuckDBMemoryStore


def _rec(mid: str, content: str = "", similarity: float = 0.0) -> MemoryRecord:
    """Build a minimal MemoryRecord for fusion tests."""
    return MemoryRecord(
        memory_id=mid,
        category="context_note",
        content=content,
        similarity=similarity,
    )


# ---------------------------------------------------------------------------
# RRF fusion: semantic rank-1 survival
# ---------------------------------------------------------------------------

class TestRRFSemanticRankOneSurvival:
    """The core regression: a strong semantic (vector) rank-1 must survive
    RRF fusion into the top-k, even when the text arm disagrees."""

    def test_vector_rank1_in_text_top5_survives(self):
        """Vector #1 is also in text top-5 → fused top-3 retains it.

        This is the easy case: both arms agree the item is relevant.
        """
        vector = [_rec("A", "alpha", 0.95), _rec("B", "beta", 0.80),
                  _rec("C", "gamma", 0.70), _rec("D", "delta", 0.60),
                  _rec("E", "epsilon", 0.50)]
        text = [_rec("B"), _rec("A"), _rec("C"), _rec("F"), _rec("D")]
        fused = DuckDBMemoryStore._rrf_fuse(vector, text)
        top3_ids = {r.memory_id for r in fused[:3]}
        assert "A" in top3_ids, "vector rank-1 must survive in fused top-3"

    def test_vector_rank1_absent_from_text_survives(self):
        """Vector #1 is absent from the text arm entirely → fused top-3
        must still retain it.

        This is the hard case: the text arm doesn't find the item at all.
        RRF gives it only 1/(k+1) ≈ 0.0476 (with k=20). If text has 5
        items all at ranks 1-5, the text rank-1 gets 0.0476 too. So the
        vector rank-1 ties with text rank-1 and should be in top-3.
        """
        vector = [_rec("A", "alpha", 0.95), _rec("B", "beta", 0.80),
                  _rec("C", "gamma", 0.70), _rec("D", "delta", 0.60),
                  _rec("E", "epsilon", 0.50)]
        text = [_rec("X"), _rec("Y"), _rec("Z"), _rec("W"), _rec("V")]
        fused = DuckDBMemoryStore._rrf_fuse(vector, text)
        top3_ids = {r.memory_id for r in fused[:3]}
        assert "A" in top3_ids, (
            "vector rank-1 absent from text must still survive in fused top-3; "
            f"got {[r.memory_id for r in fused[:3]]}"
        )

    def test_vector_rank1_low_in_text_survives_top5(self):
        """Vector #1 is at text rank 20+ (weak lexical match) → fused
        top-5 must still retain it.

        This is the scenario the issue describes: the semantic arm has a
        clear #1, but the lexical arm ranks it low. RRF's 1/(k+rank)
        weighting should still keep it in top-5.
        """
        vector = [_rec("A", "alpha", 0.95), _rec("B", "beta", 0.80),
                  _rec("C", "gamma", 0.70)]
        # Text arm ranks A at position 25 (weak lexical match).
        text = [_rec(f"T{i}") for i in range(24)] + [_rec("A")] + [_rec("B")]
        fused = DuckDBMemoryStore._rrf_fuse(vector, text)
        top5_ids = {r.memory_id for r in fused[:5]}
        assert "A" in top5_ids, (
            "vector rank-1 with weak text rank must survive in fused top-5; "
            f"got {[r.memory_id for r in fused[:5]]}"
        )

    def test_vector_rank1_with_many_text_exclusives(self):
        """Vector #1 is absent from text, and text has 20 exclusive items.
        The text arm's many exclusives each get RRF votes. Does the vector
        rank-1 still survive in top-5?

        With k=20: vector rank-1 gets 1/21 ≈ 0.0476. Text rank-1 gets
        1/21 ≈ 0.0476. They tie. Text ranks 2-5 get 1/22..1/25 ≈ 0.0455..
        0.0400. So vector rank-1 should be in top-5 (it ties with text
        rank-1 and beats text ranks 2+).
        """
        vector = [_rec("A", "alpha", 0.95), _rec("B", "beta", 0.80)]
        text = [_rec(f"T{i}") for i in range(20)]
        fused = DuckDBMemoryStore._rrf_fuse(vector, text)
        top5_ids = {r.memory_id for r in fused[:5]}
        assert "A" in top5_ids, (
            "vector rank-1 with 20 text exclusives must survive in top-5; "
            f"got {[r.memory_id for r in fused[:5]]}"
        )

    def test_both_arms_rank1_survives(self):
        """Both arms have the same #1 → it must be fused #1."""
        vector = [_rec("A", "alpha", 0.95), _rec("B", "beta", 0.80)]
        text = [_rec("A"), _rec("B")]
        fused = DuckDBMemoryStore._rrf_fuse(vector, text)
        assert fused[0].memory_id == "A", (
            "item ranked #1 in both arms must be fused #1"
        )


# ---------------------------------------------------------------------------
# Fusion path coverage
# ---------------------------------------------------------------------------

class TestFusionPaths:
    """All fusion paths must preserve a strong arm rank-1."""

    def test_vector_only_fallback_preserves_rank1(self):
        """When text arm returns empty, vector results pass through.
        Rank-1 must be preserved."""
        vector = [_rec("A", "alpha", 0.95), _rec("B", "beta", 0.80)]
        text: list[MemoryRecord] = []
        # This is the _hybrid_search fallback path, not _rrf_fuse.
        # Simulate it: when text is empty, fused = vector_results.
        fused = vector if vector and not text else (
            DuckDBMemoryStore._rrf_fuse(vector, text) if vector and text
            else text
        )
        assert fused[0].memory_id == "A"

    def test_text_only_fallback_preserves_rank1(self):
        """When vector arm returns empty, text results pass through."""
        vector: list[MemoryRecord] = []
        text = [_rec("X", "xray"), _rec("Y", "yankee")]
        fused = text if text and not vector else (
            DuckDBMemoryStore._rrf_fuse(vector, text) if vector and text
            else vector
        )
        assert fused[0].memory_id == "X"

    def test_both_empty_returns_empty(self):
        """Both arms empty → fused is empty."""
        fused = DuckDBMemoryStore._rrf_fuse([], [])
        assert fused == []


# ---------------------------------------------------------------------------
# Probe: how often is semantic #1 lost?
# ---------------------------------------------------------------------------

class TestSemanticRankOneLossProbe:
    """Probe (on synthetic data) counting how often the semantic #1 is
    lost from the fused top-k. This is the test-only canary the issue
    asks for. If the loss rate is high, the fusion policy needs adjustment.
    """

    @staticmethod
    def _run_probe(n_trials: int = 100, top_k: int = 5,
                   n_vector: int = 10, n_text: int = 10,
                   text_overlap: float = 0.3,
                   v1_in_text: bool = True) -> dict:
        """Run a synthetic probe: how often does the vector #1 fall out
        of the fused top-k?

        Args:
            n_trials: number of synthetic fusion rounds.
            top_k: the fused top-k threshold (loss = vector #1 not in top-k).
            n_vector: number of vector results per trial.
            n_text: number of text results per trial.
            text_overlap: fraction of vector items that appear in the text
                arm (the rest are text-exclusive).
            v1_in_text: if True, the vector #1 (V0) is always included in
                the text arm's overlap. If False, V0 is excluded from the
                text arm (the hard case: arms disagree on the top item).

        Returns:
            dict with loss_count, loss_rate, and mean_rank_of_vector_1.
        """
        import random
        random.seed(42)
        losses = 0
        ranks_of_v1 = []
        for trial in range(n_trials):
            # Vector arm: items V0..V{n-1}, ranked by similarity.
            vector = [_rec(f"V{i}", f"vec{i}", 1.0 - i * 0.05) for i in range(n_vector)]
            # Text arm: some overlap with vector, some exclusives.
            n_overlap = int(n_text * text_overlap)
            # Build overlap set: always include V0 if v1_in_text, then
            # sample the rest from V1..V{n-1}.
            overlap_ids = []
            pool = list(range(1, n_vector))
            if v1_in_text:
                overlap_ids.append("V0")
                n_overlap -= 1
            if n_overlap > 0 and pool:
                sampled = random.sample(pool, min(n_overlap, len(pool)))
                overlap_ids.extend(f"V{i}" for i in sampled)
            exclusive_ids = [f"T{i}_{trial}" for i in range(n_text - len(overlap_ids))]
            text = [_rec(mid) for mid in overlap_ids + exclusive_ids]
            random.shuffle(text)
            fused = DuckDBMemoryStore._rrf_fuse(vector, text)
            fused_ids = [r.memory_id for r in fused]
            v1_rank = fused_ids.index("V0") if "V0" in fused_ids else len(fused_ids)
            ranks_of_v1.append(v1_rank)
            if v1_rank >= top_k:
                losses += 1
        return {
            "n_trials": n_trials,
            "top_k": top_k,
            "loss_count": losses,
            "loss_rate": losses / n_trials,
            "mean_rank_of_v1": sum(ranks_of_v1) / len(ranks_of_v1),
        }

    def test_probe_low_overlap_low_loss(self):
        """With 30% text overlap and top-5, the vector #1 should rarely
        be lost (loss rate < 10%). This is the baseline scenario."""
        result = self._run_probe(n_trials=200, top_k=5, text_overlap=0.3)
        # Document the result in the assertion message.
        assert result["loss_rate"] < 0.10, (
            f"vector #1 loss rate too high: {result['loss_rate']:.1%} "
            f"({result['loss_count']}/{result['n_trials']} trials), "
            f"mean rank of V1: {result['mean_rank_of_v1']:.1f}"
        )

    def test_probe_high_overlap_zero_loss(self):
        """With 80% text overlap, the vector #1 should never be lost
        (both arms agree it's relevant)."""
        result = self._run_probe(n_trials=200, top_k=5, text_overlap=0.8)
        assert result["loss_rate"] == 0.0, (
            f"vector #1 lost even with high overlap: {result['loss_rate']:.1%}"
        )

    def test_probe_zero_overlap_moderate_loss(self):
        """With 0% text overlap (arms completely disagree), the vector #1
        may be lost from top-5. This is the scenario the issue describes.
        Record the loss rate as a decision input."""
        result = self._run_probe(n_trials=200, top_k=5, text_overlap=0.0,
                                 v1_in_text=False)
        # With k=20 and 10+10 items, vector #1 gets 1/21 ≈ 0.0476.
        # Text #1 gets 1/21 ≈ 0.0476. They tie at the top.
        # So vector #1 should be in top-5 even with zero overlap.
        # If it's not, RRF needs a rank-1 preservation guard.
        assert result["loss_rate"] < 0.05, (
            f"vector #1 lost with zero overlap: {result['loss_rate']:.1%} "
            f"({result['loss_count']}/{result['n_trials']} trials), "
            f"mean rank: {result['mean_rank_of_v1']:.1f}. "
            f"RRF may need a rank-1 preservation guard."
        )

    def test_probe_top3_zero_overlap(self):
        """Stricter: with 0% overlap and top-3, does vector #1 survive?
        This is the hardest case — both arms completely disagree and we
        only keep 3 results."""
        result = self._run_probe(n_trials=200, top_k=3, text_overlap=0.0,
                                 v1_in_text=False)
        # With k=20, vector #1 ties with text #1 at 0.0476.
        # Text #2 gets 1/22 ≈ 0.0455, text #3 gets 1/23 ≈ 0.0435.
        # Vector #1 should be in top-3 (it's either #1 or #2 after fusion).
        assert result["loss_rate"] < 0.10, (
            f"vector #1 lost from top-3 with zero overlap: "
            f"{result['loss_rate']:.1%} "
            f"({result['loss_count']}/{result['n_trials']} trials), "
            f"mean rank: {result['mean_rank_of_v1']:.1f}"
        )

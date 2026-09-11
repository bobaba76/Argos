"""Tests for #445: per-retriever union before the CE cut + bounded CE-rescue.

The fused top-N cut can lose a record that is a strong VECTOR rank-1 but has
ZERO lexical overlap (RRF rewards two-arm presence; the rank-1 survival guard
needs a clear margin, which a flat semantic head does not produce). The fix:

1. Union pool: the rerank pool appends each arm's head beyond the fused cut,
   so the cross-encoder actually sees the best semantic candidate.
2. Bounded CE-rescue: a union-pool record with a near-perfect normalized
   cross-encoder match (>= CE_PROMOTE_MIN) gets rescued INTO the window —
   at most CE_PROMOTE_MAX records. The strict ranking lane's strongest
   members stay put; rescued records are flagged `_ce_promoted`.

Deterministic harness: real DuckDBMemoryStore on a temp file; retrieval arms
stubbed with crafted lists; a stub reranker that scores "semantic" documents
10.0 (-> normalized CE 1.0) and everything else by pool index.
"""
from __future__ import annotations

import math
import os
import tempfile

import pytest

try:
    from store import DuckDBMemoryStore, MemoryRecord
except ImportError:  # pragma: no cover
    pass


def _rec(mid: str, content: str, similarity: float) -> MemoryRecord:
    return MemoryRecord(
        memory_id=mid, content=content, category="personal_fact",
        created_at="2026-01-01T00:00:00Z", similarity=similarity,
    )


class _SemanticReranker:
    """CE score: 10.0 for "semantic" content (-> norm 1.0), else index."""

    def score(self, query, documents):
        return [10.0 if "semantic" in d else float(i)
                for i, d in enumerate(documents)]


class _HashEmbedder:
    """Deterministic token-hash embedder (no model): keeps the vector arm
    alive so the crafted vector arm is actually consulted."""

    def __init__(self, dim: int = 64) -> None:
        self._dim = dim
        self.dimension = dim

    def embed(self, text, *, is_query=False):
        import hashlib
        v = [0.0] * self._dim
        for tok in text.lower().split():
            h = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
            idx = int.from_bytes(h[:2], "little") % self._dim
            sign = 1.0 if (h[2] & 1) == 0 else -1.0
            v[idx] += sign
        n = math.sqrt(sum(x * x for x in v))
        return [x / n for x in v] if n > 0 else v


def _build(vector_arm, text_arm, *, limit=3, top_n=2, rescue_enabled=True):
    tmp = tempfile.TemporaryDirectory()

    class _S(DuckDBMemoryStore):
        def _vector_search_raw(self, *args, **kwargs):
            return list(vector_arm)

        def _text_search_raw(self, *args, **kwargs):
            return list(text_arm)

    store = _S(os.path.join(tmp.name, "r.duckdb"), user_id="test",
               embedder=_HashEmbedder())
    store.reranker = _SemanticReranker()
    store._reranker_top_n = top_n
    store._ce_rescue_enabled = rescue_enabled
    return store, tmp


class TestCeRescue:
    @staticmethod
    def _arms():
        # Fused order (RRF K=20): B, D, E (two-arm), then A (vector rank-1,
        # ABSENT from text, flat head 0.95 vs 0.94 -> no rank-1 guard), then C.
        vector = [
            _rec("mem-A", "alpha semantic content", 0.95),
            _rec("mem-B", "beta content", 0.94),
            _rec("mem-D", "delta content", 0.93),
            _rec("mem-E", "epsilon content", 0.92),
        ]
        text = [
            _rec("mem-B", "beta content", 0.90),
            _rec("mem-D", "delta content", 0.80),
            _rec("mem-E", "epsilon content", 0.70),
            _rec("mem-C", "gamma content", 0.60),
        ]
        return vector, text

    def test_answer_rescued_into_window(self):
        vector, text = self._arms()
        store, tmp = _build(vector, text, limit=3, top_n=2)
        with tmp:
            results = store.search("alpha beta gamma", limit=3)
            mids = [r.memory_id for r in results]
            assert "mem-A" in mids, (
                "vector rank-1 with zero lexical overlap must be rescued "
                "into the window via the union pool + CE-rescue"
            )
            rescued = [r for r in results if getattr(r, "_ce_promoted", False)]
            assert len(rescued) == 1 and rescued[0].memory_id == "mem-A"
            # Strict lane must never be displaced by a rescue: the rescued
            # record lands at the WINDOW TAIL, never at the head.
            assert results[0].memory_id != "mem-A"
            assert results[-1].memory_id == "mem-A"
            assert len(results) == 3

    def test_no_rescue_without_fix(self):
        vector, text = self._arms()
        store, tmp = _build(vector, text, limit=3, top_n=2, rescue_enabled=False)
        with tmp:
            results = store.search("alpha beta gamma", limit=3)
            assert "mem-A" not in {r.memory_id for r in results}, (
                "pre-fix behavior reproducible when rescue is disabled"
            )

    def test_rescue_bounded_at_two(self):
        vector = [
            _rec("mem-A", "alpha semantic content", 0.95),
            _rec("mem-S1", "s1 semantic content", 0.90),
            _rec("mem-S2", "s2 semantic content", 0.89),
            _rec("mem-S3", "s3 semantic content", 0.88),
        ]
        text = [
            _rec("mem-B", "beta content", 0.90),
            _rec("mem-C", "gamma content", 0.80),
        ]
        store, tmp = _build(vector, text, limit=3, top_n=1)
        with tmp:
            results = store.search("alpha beta gamma", limit=3)
            promoted = [r for r in results if getattr(r, "_ce_promoted", False)]
            assert len(promoted) <= 2
            assert len(results) == 3

    def test_low_ce_not_rescued(self):
        vector, text = self._arms()
        store, tmp = _build(vector, text, limit=3, top_n=2)
        store._CE_PROMOTE_MIN = 2.0  # floor above any possible CE norm
        with tmp:
            results = store.search("alpha beta gamma", limit=3)
            assert all(not getattr(r, "_ce_promoted", False) for r in results)
            assert "mem-A" not in {r.memory_id for r in results}
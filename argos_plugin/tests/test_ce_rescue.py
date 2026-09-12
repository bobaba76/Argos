"""Tests for #445: recall band + bounded CE-rescue + template-dialect probes.

The fused top-N cut can lose a record that is a strong VECTOR match but has
ZERO lexical overlap (RRF rewards two-arm presence; the rank-1 survival guard
needs a clear margin, which a flat semantic head does not produce). Round-2:
live data showed the answering record even below the vector head (rank ~13),
so the recall band is wider than the pool, and rescue eligibility is a pure
CE-vs-window-tail comparison (no normalized floor — an exact-lexical record
can poison pool-max normalization).

Round-3 (11/9): the cross-encoder itself is verb-phrase locked on the
canonical "User's ..." record template (measured 0.0046 for the true answer
vs 0.99 with the query spoken in the record's dialect). Template-dialect
probes fix the INPUT side: matching intent families add probe queries that
ARE in the record's dialect, and each record keeps the MAX CE score.

Deterministic harness: real DuckDBMemoryStore on a temp file; retrieval arms
stubbed; a query-aware keyword reranker (strong keyword present in BOTH
probe/query and doc -> 10.0; "gamma" -> -5.0 weak; else 0.0) so CE behavior
is fully controlled.
"""
from __future__ import annotations

import math
import os
import tempfile

try:
    from store import DuckDBMemoryStore, MemoryRecord
except ImportError:  # pragma: no cover
    pass

try:
    from store_retrieval import _vector_probe_queries
except ImportError:  # pragma: no cover
    _vector_probe_queries = None


def _rec(mid: str, content: str, similarity: float) -> MemoryRecord:
    return MemoryRecord(
        memory_id=mid, content=content, category="personal_fact",
        created_at="2026-01-01T00:00:00Z", similarity=similarity,
    )


class _KeywordReranker:
    """Query-AWARE stub: a doc scores 10.0 only when a *strong* keyword
    appears in BOTH the (possibly alias) probe and the doc; explicit weak
    marker ("gamma") -> -5.0; otherwise 0.0."""

    def __init__(self, strong=("semantic",)):
        self._strong = strong

    def score(self, query, documents):
        out = []
        for d in documents:
            if any(k in query and k in d for k in self._strong):
                out.append(10.0)
            elif "gamma" in d:  # explicit weak marker ("gamma content")
                out.append(-5.0)
            else:
                out.append(0.0)
        return out


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


def _build(vector_arm, text_arm, *, limit=3, top_n=2, rescue_enabled=True,
           strong=("semantic",)):
    tmp = tempfile.TemporaryDirectory()

    class _S(DuckDBMemoryStore):
        def _vector_search_raw(self, *args, **kwargs):
            return list(vector_arm)

        def _vector_probe_search(self, *args, **kwargs):
            # #460: the probe loop now uses the lightweight (memory_id, sim)
            # scan — mirror the crafted arm so probe semantics are unchanged.
            return [(r.memory_id, r.similarity) for r in vector_arm]

        def _text_search_raw(self, *args, **kwargs):
            return list(text_arm)

    store = _S(os.path.join(tmp.name, "r.duckdb"), user_id="test",
               embedder=_HashEmbedder())
    store.reranker = _KeywordReranker(strong=strong)
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
        store, tmp = _build(vector, text, limit=3, top_n=2, strong=("alpha",))
        with tmp:
            results = store.search("alpha beta gamma", limit=3)
            mids = [r.memory_id for r in results]
            assert "mem-A" in mids, (
                "vector rank-1 with zero lexical overlap must be rescued "
                "into the window via the recall band + CE-rescue"
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
        store, tmp = _build(vector, text, limit=3, top_n=2,
                            rescue_enabled=False, strong=("alpha",))
        with tmp:
            results = store.search("alpha beta gamma", limit=3)
            assert "mem-A" not in {r.memory_id for r in results}, (
                "pre-fix behavior reproducible when rescue is disabled"
            )

    def test_rescue_bounded_at_two(self):
        # Two band-only candidates that both carry a query keyword.
        vector = [
            _rec("mem-A", "alpha semantic content", 0.95),
            _rec("mem-S1", "s1 beta semantic content", 0.90),
            _rec("mem-S2", "s2 semantic content", 0.89),
            _rec("mem-S3", "s3 semantic content", 0.88),
        ]
        text = [
            _rec("mem-B", "beta content", 0.90),
            _rec("mem-C", "gamma content", 0.80),
        ]
        store, tmp = _build(vector, text, limit=3, top_n=1,
                            strong=("alpha", "beta"))
        with tmp:
            results = store.search("alpha beta gamma", limit=3)
            promoted = [r for r in results if getattr(r, "_ce_promoted", False)]
            assert len(promoted) <= 2, (
                "rescue must be bounded at CE_PROMOTE_MAX, never flood"
            )
            assert len(results) == 3

    def test_low_ce_not_rescued(self):
        # Window members (B/D/E) are the STRONG CE matches (10.0); the recall
        # band candidate A is a weak match (0.0) -> below the window's worst
        # CE -> must NOT be rescued.
        vector, text = self._arms()
        store, tmp = _build(vector, text, limit=3, top_n=2,
                            strong=("beta", "delta", "epsilon"))
        with tmp:
            results = store.search("alpha beta gamma", limit=3)
            assert all(not getattr(r, "_ce_promoted", False) for r in results)
            assert "mem-A" not in {r.memory_id for r in results}


def test_initial_pool_member_rescued():
        # Regression (11/9): the rescue gate tested _ce_pool_member, but that
        # flag was only set on band-union additions. A record that makes the
        # INITIAL top-N fused slice (high vector rank via probe, ZERO lexical
        # overlap) was in the pool and CE-scored ~perfect, yet ineligible for
        # rescue — its blended similarity (0.8*RRF-sim + 0.2*CE) kept it
        # buried. A sits at fused position #2 (inside fused[:top_n=2]) and
        # must be flagged + rescued into the limit-1 window.
        vector = [
            _rec("mem-A", "alpha semantic content", 0.99),
            _rec("mem-B", "beta content", 0.50),
        ]
        text = [
            _rec("mem-B", "beta content", 0.99),
        ]
        store, tmp = _build(vector, text, limit=1, top_n=2,
                            strong=("alpha",))
        with tmp:
            results = store.search("alpha beta gamma", limit=1)
            assert results[0].memory_id == "mem-A", (
                "initial-pool (top-N slice) member with near-perfect CE must "
                "be rescue-eligible — the flag must cover the whole pool"
            )
            assert getattr(results[0], "_ce_promoted", False)


class TestTemplateDialectProbes:
    @staticmethod
    def _arms():
        # The record answers "what do I currently work as?" but ONLY in the
        # record's own dialect ("User's job title is ...") — the natural
        # query shares no CE-strong keyword with it. This mirrors the live
        # 11/9 measurement (0.0046 natural vs 0.99 dialect).
        vector = [
            _rec("mem-A",
                 "User's job title is 'National Product Manager' at an "
                 "electronic security distribution company.", 0.95),
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

    def test_probe_lifts_record_in_record_dialect(self):
        # "job title" is CE-strong ONLY via the work-family probe ("what is
        # user's job title") — the natural query never contains it. The
        # template-dialect probe must lift the record into rescue range.
        vector, text = self._arms()
        store, tmp = _build(vector, text, limit=3, top_n=2,
                            strong=("job title",))
        with tmp:
            results = store.search("what do I currently work as", limit=3)
            assert "mem-A" in {r.memory_id for r in results}, (
                "work-intent query must rescue the answer record via the "
                "template-dialect probe"
            )
            rescued = [r for r in results if getattr(r, "_ce_promoted", False)]
            assert len(rescued) == 1 and rescued[0].memory_id == "mem-A"
            assert results[-1].memory_id == "mem-A"

    def test_probe_family_gated_by_intent_regex(self):
        # Same arms; the query is location-family ("live") — the work probes
        # must NOT fire, and no CE-strong kw matches -> no rescue.
        vector, text = self._arms()
        store, tmp = _build(vector, text, limit=3, top_n=2,
                            strong=("job title",))
        with tmp:
            results = store.search("where does the user live", limit=3)
            assert "mem-A" not in {r.memory_id for r in results}, (
                "unrelated intent family must not trigger work probes"
            )
            assert all(not getattr(r, "_ce_promoted", False) for r in results)


class TestVectorArmProbes:
    def test_probe_selection(self):
        assert _vector_probe_queries is not None
        # Work-family phrasings all yield the canonical stored-keyword probe.
        for q in ("What do I currently work as?",
                  "What is my current role?",
                  "my job title"):
            assert _vector_probe_queries(q) == ["what is user's job title"], q
        # Location family yields its own dialect probes.
        assert _vector_probe_queries("Where do I currently live?") == [
            "where does user live", "what is user's address"]
        # Unrelated query -> zero extra work (no probes).
        assert _vector_probe_queries("alpha beta gamma") == []

    def test_probe_wiring_fires_vector_searches(self):
        # A work-family query must issue 1 primary vector search + 1
        # lightweight probe scan (the template probe); an unrelated query
        # exactly 1 primary and zero probe scans (mirrors the pre-#460
        # wiring, where the probe loop issued full _vector_search_raw calls).
        class _Counting(DuckDBMemoryStore):
            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                self.raw = 0
                self.probes = 0

            def _vector_search_raw(self, *args, **kwargs):
                self.raw += 1
                return []

            def _vector_probe_search(self, *args, **kwargs):
                self.probes += 1
                return []

        tmp = tempfile.TemporaryDirectory()
        store = _Counting(os.path.join(tmp.name, "r.duckdb"), user_id="test",
                          embedder=_HashEmbedder())
        with tmp:
            store.search("what do I currently work as", limit=3)
            assert store.raw == 1, "primary vector search"
            assert store.probes == 1, "primary + 1 work probe scan"
            store.raw = 0
            store.probes = 0
            store.search("alpha beta gamma", limit=3)
            assert store.raw == 1, "primary only for unrelated query"
            assert store.probes == 0, "no probes for an unrelated query"
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

    def score_multi(self, queries, documents):
        out = []
        for q in queries:
            row = []
            for d in documents:
                if any(k in q and k in d for k in self._strong):
                    row.append(10.0)
                elif "gamma" in d:
                    row.append(-5.0)
                else:
                    row.append(0.0)
            out.append(row)
        return out


class _CountingReranker(_KeywordReranker):
    """Counts predict entries: one per score() call vs one per score_multi()
    call (the #460 batched-CE regression: a work-family query must issue ONE
    score_multi with query+probe, not two separate score passes)."""

    def __init__(self):
        super().__init__()
        self.score_calls = 0
        self.multi_calls = 0
        self.multi_queries = []

    def score(self, query, documents):
        self.score_calls += 1
        return super().score(query, documents)

    def score_multi(self, queries, documents):
        self.multi_calls += 1
        self.multi_queries.append(list(queries))
        return super().score_multi(queries, documents)


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
           strong=("semantic",), reranker=None):
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
    store.reranker = reranker if reranker is not None else _KeywordReranker(strong=strong)
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


class _RawReranker:
    """Assigns fixed per-content CE raws (query-independent), for testing
    the rescue displacement rules with full control over raw ordering."""

    def __init__(self, raw_markers: dict):
        self._markers = raw_markers  # content-substring -> raw

    def _raws(self, documents):
        out = []
        for d in documents:
            raw = 0.0
            for marker, v in self._markers.items():
                if marker in d:
                    raw = v
                    break
            out.append(raw)
        return out

    def score(self, query, documents):
        return self._raws(documents)

    def score_multi(self, queries, documents):
        return [self._raws(documents) for _ in queries]


class TestLimitInvariantRescue:
    """#467: the rescue must never evict an in-window member on POSITION —
    displacement is by CE-raw. Same query at limits 3..15: the answer stays
    present at every limit (rescued into small windows, at its true rank in
    wide ones) and the strict head is constant."""

    def _store(self, rescue_enabled=True):
        # Fused order (RRF): H1,H2,W1 (two-arm) then W2,F1,F2,F3, then the
        # answer A at slot 8 (the old tail-eviction point at limit 8),
        # then band-only B1/B2 (CE-strong, outside any <=8 window).
        vector = [
            _rec("mem-H1", "H1MARK alpha", 0.95),
            _rec("mem-H2", "H2MARK beta", 0.90),
            _rec("mem-W1", "W1MARK gamma", 0.88),
            _rec("mem-W2", "W2MARK delta", 0.78),
            _rec("mem-F1", "F1MARK one", 0.75),
            _rec("mem-F2", "F2MARK two", 0.72),
            _rec("mem-F3", "F3MARK three", 0.70),
            _rec("mem-A", "AMARK semantic", 0.69),
            _rec("mem-B1", "B1MARK epsilon", 0.60),
            _rec("mem-B2", "B2MARK zeta", 0.59),
            _rec("mem-F4", "F4MARK four", 0.55),
        ]
        text = [
            _rec("mem-H1", "H1MARK alpha", 0.80),
            _rec("mem-H2", "H2MARK beta", 0.80),
            _rec("mem-W1", "W1MARK gamma", 0.80),
        ]
        rr = _RawReranker({
            "H1MARK": 0.98, "H2MARK": 0.97, "W1MARK": 0.40, "W2MARK": 0.35,
            "F1MARK": 0.30, "F2MARK": 0.30, "F3MARK": 0.30, "F4MARK": 0.33,
            "AMARK": 0.99, "B1MARK": 0.96, "B2MARK": 0.95,
        })
        store, tmp = _build(vector, text, limit=3, top_n=2, reranker=rr)
        store._ce_rescue_enabled = rescue_enabled
        return store, tmp

    def test_answer_present_at_every_limit(self):
        q = "alpha beta gamma delta epsilon zeta semantic"
        store, tmp = self._store()
        with tmp:
            base_off, tmp_off = self._store(rescue_enabled=False)
            with tmp_off:
                off8 = [r.memory_id for r in base_off.search(q, limit=8)]
            pos_off8 = off8.index("mem-A") + 1 if "mem-A" in off8 else None
            for lim in (3, 6, 8, 15):
                rows = store.search(q, limit=lim)
                ids = [r.memory_id for r in rows]
                assert "mem-A" in ids, f"answer evicted at limit {lim} (#467)"
                assert ids[0] == "mem-H1", f"strict head displaced at limit {lim}"
            # wide window: answer at its true rank, never worse than
            # rescue-off (a rescue may pull it UP, never push it down).
            rows8 = store.search(q, limit=8)
            pos8 = [r.memory_id for r in rows8].index("mem-A") + 1
            assert pos_off8 is not None and pos8 <= pos_off8, (
                f"rescue made the answer WORSE: on={pos8} off={pos_off8}"
            )

    def test_small_window_rescue_lands_at_tail(self):
        q = "alpha beta gamma delta epsilon zeta semantic"
        store, tmp = self._store()
        with tmp:
            rows = store.search(q, limit=3)
            ids = [r.memory_id for r in rows]
            # Head survives; the strong-raw band answer rescues INTO the
            # 3-window and is flagged; the rescued slot is never the head.
            assert ids[0] == "mem-H1", f"head displaced: {ids}"
            assert "mem-A" in ids, f"answer not rescued: {ids}"
            a_row = rows[ids.index("mem-A")]
            assert getattr(a_row, "_ce_promoted", False), "answer not flagged"
            promoted = [r.memory_id for r in rows if getattr(r, "_ce_promoted", False)]
            assert set(promoted) <= {"mem-A", "mem-B1", "mem-B2"}, promoted


class TestBatchedCeProbes:
    """#460: work-family queries batch query+probe into ONE score_multi
    predict call instead of two separate CE passes."""

    def _arms(self):
        # D is a strong vector+text record ranking 2nd; a work-family probe
        # must reach the CE block (not just the vector probe prepend).
        vector = [
            _rec("A", "alpha beta", similarity=0.95),
            _rec("D", "semantic job title record", similarity=0.80),
        ]
        text = [_rec("D", "semantic job title record", similarity=0.80)]
        return vector, text

    def test_work_family_uses_one_batched_predict(self):
        vector, text = self._arms()
        rr = _CountingReranker()
        store, tmp = _build(vector, text, top_n=2, reranker=rr)
        with tmp:
            store.search("what do I currently work as", limit=3)
            assert rr.multi_calls >= 1, "batched path must be used"
            assert rr.score_calls == 0, "no unbatched per-query pass"
            # query + the single work-family star probe, exactly
            assert rr.multi_queries[-1][0] == "what do I currently work as"
            assert len(rr.multi_queries[-1]) == 2, "query + 1 probe"

    def test_unrelated_query_no_probe_pass(self):
        vector, text = self._arms()
        rr = _CountingReranker()
        store, tmp = _build(vector, text, top_n=2, reranker=rr)
        with tmp:
            store.search("alpha beta gamma", limit=3)
            assert rr.multi_calls == 0, "no family -> plain score path"
            assert rr.score_calls >= 1, "main query still CE-scored"

    def test_score_multi_chunking(self):
        # Real CrossEncoderReranker shape: one predict over (queries x docs)
        # pairs, re-chunked per query in the ORIGINAL order.
        from embeddings import CrossEncoderReranker, _SHARED_RERANKERS

        class _FakeModel:
            def predict(self, pairs, show_progress_bar=False):
                # 2 queries x 3 docs (first call) then 1 query x 2 docs
                # (delegated score() call): accept any pair count, return
                # index-scaled floats so chunk order is checkable.
                return [float(10 * i) for i in range(len(pairs))]

        _SHARED_RERANKERS["bge-test"] = _FakeModel()
        try:
            rr = CrossEncoderReranker("bge-test",
                                      hermes_home=r"C:/Users/michael/AppData/Local/hermes")
            out = rr.score_multi(["q1", "q2"], ["d1", "d2", "d3"])
            assert len(out) == 2 and all(len(o) == 3 for o in out), "2x3 shape"
            assert out[0] == [0.0, 10.0, 20.0], "query 1 rows in order"
            assert out[1] == [30.0, 40.0, 50.0], "query 2 rows in order"
            # score() delegates to the same batched path
            one = rr.score("q1", ["d1", "d2"])
            assert one == [0.0, 10.0]
        finally:
            _SHARED_RERANKERS.pop("bge-test", None)
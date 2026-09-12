"""Tests for #460: search-latency cuts (CE input truncation, lightweight
vector probe scans, single-star CE probe aliases).

Perf contract under test:
1. CE input documents are truncated to CE_MAX_DOC_CHARS BEFORE the
   tokenizer (the full record content is preserved everywhere else).
2. The vector-arm probe loop uses the lightweight (memory_id, sim) scan;
   probe hits outside the primary pool still enter via by-id fetch
   (#456 pool-entry class preserved).
3. The probe scan respects the same filters as the primary arm.
4. Each intent family carries exactly ONE CE probe (every extra probe is a
   full CE pass — the dominant per-query cost, measured 11/9).
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
    from store_common import _tokenize
except ImportError:  # pragma: no cover
    pass

try:
    from tuning import CE_MAX_DOC_CHARS, CE_PROBE_ALIASES
except ImportError:  # pragma: no cover
    pass

try:
    from store_retrieval import _vector_probe_queries
except ImportError:  # pragma: no cover
    _vector_probe_queries = None


class _HashEmbedder:
    """Deterministic token-hash embedder (no model)."""

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


def _rec(mid: str, content: str, similarity: float) -> MemoryRecord:
    return MemoryRecord(
        memory_id=mid, content=content, category="personal_fact",
        created_at="2026-01-01T00:00:00Z", similarity=similarity,
    )


class _RecordingReranker:
    """Captures every (query, documents) call; returns neutral scores."""

    def __init__(self, score=0.5):
        self.calls = []
        self._score = score

    def score(self, query, documents):
        self.calls.append((query, list(documents)))
        return [self._score] * len(documents)


class TestCeInputTruncation:
    def test_ce_input_truncated_to_cap_but_record_full(self):
        long = "word " * 4000  # ~20k chars, far beyond the 512-token window
        long = long.strip()
        vector = [
            _rec("mem-A", "alpha content", 0.95),
            _rec("mem-B", long, 0.94),
        ]
        text = [_rec("mem-B", long, 0.90)]
        tmp = tempfile.TemporaryDirectory()

        class _S(DuckDBMemoryStore):
            def _vector_search_raw(self, *args, **kwargs):
                return list(vector)

            def _vector_probe_search(self, *args, **kwargs):
                return []

            def _text_search_raw(self, *args, **kwargs):
                return list(text)

        store = _S(os.path.join(tmp.name, "r.duckdb"), user_id="test",
                   embedder=_HashEmbedder())
        rerank = _RecordingReranker()
        store.reranker = rerank
        store._reranker_top_n = 2
        with tmp:
            results = store.search("alpha long content", limit=2)
            assert rerank.calls, "reranker must fire"
            for _q, docs in rerank.calls:
                for d in docs:
                    assert len(d) <= CE_MAX_DOC_CHARS, (
                        "CE input must be capped at CE_MAX_DOC_CHARS "
                        f"(got {len(d)})"
                    )
            # The RECORD content survives intact outside the CE input.
            rec_b = next(r for r in results if r.memory_id == "mem-B")
            assert rec_b.content == long, "record content must not be truncated"


class TestVectorProbeScan:
    def test_probe_hit_outside_primary_pool_enters_via_id_fetch(self):
        # #456 pool-entry class: a probe can rank a record #1 that the
        # primary pool cut off — the record must still enter the pool.
        tmp = tempfile.TemporaryDirectory()
        store = DuckDBMemoryStore(os.path.join(tmp.name, "r.duckdb"),
                                  user_id="test", embedder=_HashEmbedder())
        kept = store.remember("personal_fact", "user works as a software engineer",
                              tags=[], dedup=False, scope="profile")
        buried = store.remember("personal_fact", "user lives in Roodepoort",
                                tags=[], dedup=False, scope="profile")
        assert kept and buried

        class _S(DuckDBMemoryStore):
            def _vector_search_raw(self, *args, **kwargs):
                return [_rec(kept.memory_id, kept.content, 0.9)]

            def _vector_probe_search(self, *args, **kwargs):
                return [(buried.memory_id, 0.99), (kept.memory_id, 0.9)]

        s2 = _S(os.path.join(tmp.name, "r.duckdb"), user_id="test",
                embedder=_HashEmbedder())
        s2._reranker_top_n = 2
        with tmp:
            results = s2.search("where does user live", limit=2)
            mids = {r.memory_id for r in results}
            assert buried.memory_id in mids, (
                "probe hit outside the primary pool must enter via by-id fetch"
            )

    def test_probe_scan_respects_filters_and_limit(self):
        tmp = tempfile.TemporaryDirectory()
        store = DuckDBMemoryStore(os.path.join(tmp.name, "r.duckdb"),
                                  user_id="test", embedder=_HashEmbedder())
        a = store.remember("context_note", "user's job title is product manager",
                           tags=[], dedup=False, scope="profile")
        b = store.remember("personal_fact", "user's job title used to be analyst",
                           tags=[], dedup=False, scope="profile")
        c = store.remember("personal_fact", "user's job title history includes clerk",
                           tags=[], dedup=False, scope="profile")
        assert a and b and c
        with tmp:
            probe_emb = _HashEmbedder().embed("what is user's job title",
                                              is_query=True)
            hits = store._vector_probe_search(
                probe_emb, limit=2, excluded={"context_note"})
            assert all(isinstance(h[0], str) and isinstance(h[1], float)
                       for h in hits), "must return (memory_id, sim) pairs"
            assert len(hits) <= 2, "limit must be honored"
            assert not any(h[0] == a.memory_id for h in hits), (
                "excluded category must not appear in probe scan results"
            )
            assert {h[0] for h in hits} <= {b.memory_id, c.memory_id}


class TestCeProbeBudget:
    def test_one_star_probe_per_family(self):
        # #460 perf contract: every extra CE alias is a full extra CE pass
        # over the pool (the dominant per-query cost). Keep ONE per family.
        assert len(CE_PROBE_ALIASES) >= 1
        for fam, (_regex, probes) in CE_PROBE_ALIASES.items():
            assert len(probes) == 1, (
                f"family {fam!r} must carry exactly one CE probe (perf: "
                "each extra probe = one full CE pass)"
            )

    def test_ce_probe_pass_count(self):
        # work-family query: primary CE pass + exactly 1 probe pass;
        # unrelated query: exactly 1 pass.
        vector = [_rec("mem-A", "job title content", 0.95),
                  _rec("mem-B", "beta content", 0.90)]
        text = [_rec("mem-A", "job title content", 0.90)]
        tmp = tempfile.TemporaryDirectory()

        class _S(DuckDBMemoryStore):
            def _vector_search_raw(self, *args, **kwargs):
                return list(vector)

            def _vector_probe_search(self, *args, **kwargs):
                return []

            def _text_search_raw(self, *args, **kwargs):
                return list(text)

        store = _S(os.path.join(tmp.name, "r.duckdb"), user_id="test",
                   embedder=_HashEmbedder())
        rerank = _RecordingReranker()
        store.reranker = rerank
        store._reranker_top_n = 2
        with tmp:
            store.search("what do I currently work as", limit=2)
            assert len(rerank.calls) == 2, (
                "work family: primary + 1 star probe pass, got "
                f"{len(rerank.calls)}"
            )
            rerank.calls = []
            store.search("alpha beta gamma delta", limit=2)
            assert len(rerank.calls) == 1, (
                "unrelated query: exactly 1 CE pass, got "
                f"{len(rerank.calls)}"
            )
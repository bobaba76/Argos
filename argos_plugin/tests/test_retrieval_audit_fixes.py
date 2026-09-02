"""Regression tests for the 2026-09-02 retrieval audit fixes (B1-B5).

Covered:
  B1  raw_similarity is captured BEFORE the cross-encoder blend, so the
      query-expansion gate sees pure retrieval strength (not a reranker-
      lifted score).
  B3  P2C overlap uses _tokenize() (punctuation-aware) like the rest of
      the pipeline, not whitespace .split().
  B4  Semantic dedup returns the MOST similar record above threshold
      (ORDER BY sim DESC), consistent with find_semantic_duplicate.
  B5  Query-expansion RRF k is linked to StoreRetrievalMixin._RRF_K so
      tuning the fusion constant can't silently desync the merge.

  B2 (provider graph-candidate cosine dimension guard) is a defensive
  guard on a path that requires a dimension-mismatched embedding to
  exist; not unit-testable without heavy stubbing.
"""
import math
import os
import tempfile
from types import SimpleNamespace

import pytest

try:
    from store import DuckDBMemoryStore
    from store_retrieval import StoreRetrievalMixin
    from provider_retrieval import ProviderRetrievalMixin
except ImportError:  # pragma: no cover - test collection fallback
    pass

_DIM = 384


def _unit(idx: int, dim: int = _DIM) -> list[float]:
    """Basis vector e_idx of length dim."""
    v = [0.0] * dim
    v[idx] = 1.0
    return v


class _HashEmbedder:
    """Hermetic token-hash embedder: deterministic, no model, no cache."""

    def __init__(self, dim: int = 128) -> None:
        self._dim = dim
        self.dimension = dim

    def embed(self, text, *, is_query=False):
        import hashlib
        v = [0.0] * self._dim
        for tok in text.lower().split():
            h = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
            idx = int.from_bytes(h[:4], "little") % self._dim
            sign = 1.0 if (h[4] & 1) == 0 else -1.0
            v[idx] += sign
        norm = math.sqrt(sum(x * x for x in v))
        return [x / norm for x in v] if norm > 0 else v


class _StubReranker:
    """Cross-encoder stand-in: highest score for the LAST document."""

    def score(self, query, documents):
        return [float(i) for i in range(len(documents))]


class TestB1RawSimilarityBeforeReranker:
    def test_raw_similarity_uncontaminated_by_reranker_blend(self):
        """raw_similarity must equal the pre-reranker value even when the
        reranker is on and moves final similarity."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = DuckDBMemoryStore(
                os.path.join(tmpdir, "b1.duckdb"),
                user_id="test", embedder=_HashEmbedder(),
                reranker=None,
            )
            store.remember(category="context_note",
                           content="alpha document about apples", dedup=False)
            store.remember(category="context_note",
                           content="beta document about bananas", dedup=False)
            store.remember(category="context_note",
                           content="gamma document about grapes", dedup=False)

            # Run A: reranker off.
            a = store.search("alpha beta gamma", limit=3)
            raw_a = {r.memory_id: r.raw_similarity for r in a}

            # Run B: reranker on (reverses order, blends 20% CE score).
            store.reranker = _StubReranker()
            store._reranker_top_n = 20
            b = store.search("alpha beta gamma", limit=3)
            raw_b = {r.memory_id: r.raw_similarity for r in b}

            # The reranker must actually have engaged (else the test is
            # vacuous): final similarity changed for at least one record.
            sim_a = {r.memory_id: r.similarity for r in a}
            sim_b = {r.memory_id: r.similarity for r in b}
            assert any(
                abs(sim_a[mid] - sim_b[mid]) > 1e-9 for mid in sim_a
            ), "reranker did not engage — test is vacuous"

            # raw_similarity is identical across both runs (pre-blend).
            for mid, raw in raw_a.items():
                assert mid in raw_b
                assert raw_b[mid] == pytest.approx(raw, abs=1e-9), (
                    f"raw_similarity for {mid} was contaminated by the "
                    "reranker blend (B1 regression)"
                )


class TestB3P2CTokenization:
    def test_p2c_overlap_is_punctuation_aware(self):
        """'Cape Town.' and 'Cape Town' are the same token set via
        _tokenize() — overlap is 1.0, not 2/3."""
        overlap = StoreRetrievalMixin._p2c_overlap(
            "I live in Cape Town.", "I live in Cape Town"
        )
        assert overlap == pytest.approx(1.0)

    def test_p2c_promotes_newer_when_punctuation_differs(self, monkeypatch):
        """With .split() the 'Cape Town.'/'Cape Town' pair scores 0.333 and
        never demotes; with _tokenize() it scores 1.0 and the newer member
        clears the older (B3 regression guard)."""
        older = SimpleNamespace(memory_id="old", content="Cape Town.",
                                created_at="2026-01-01T00:00:00+00:00",
                                similarity=0.90)
        newer = SimpleNamespace(memory_id="new", content="Cape Town",
                                created_at="2026-02-01T00:00:00+00:00",
                                similarity=0.50)
        monkeypatch.setattr(StoreRetrievalMixin, "_P2C_ENABLED", True)
        try:
            out = [older, newer]
            StoreRetrievalMixin._apply_p2c(out)
        finally:
            monkeypatch.setattr(StoreRetrievalMixin, "_P2C_ENABLED", False)
        assert out[0].memory_id == "new"
        assert newer.similarity == pytest.approx(min(1.0, 0.90 + 0.005))


class TestB4SemanticDedupOrdering:
    def test_semantic_dedup_returns_most_similar(self):
        """Layer-3 dedup must return the record with the highest cosine
        above the threshold, not an arbitrary one (B4 regression guard)."""
        v_new = _unit(0)
        v_high = [0.99, math.sqrt(1.0 - 0.99 ** 2)] + [0.0] * (_DIM - 2)
        v_low = [0.86, math.sqrt(1.0 - 0.86 ** 2)] + [0.0] * (_DIM - 2)

        class _FixedEmbedder:
            dimension = _DIM

            def embed(self, text, *, is_query=False):
                if text == "brand new fact to check":
                    return v_new
                if text.startswith("high similarity record"):
                    return v_high
                if text.startswith("low similarity record"):
                    return v_low
                return [0.5, 0.5] + [0.0] * (_DIM - 2)

        with tempfile.TemporaryDirectory() as tmpdir:
            store = DuckDBMemoryStore(
                os.path.join(tmpdir, "b4.duckdb"),
                user_id="test", embedder=_FixedEmbedder(), reranker=None,
            )
            # Insert LOW-sim FIRST: without ORDER BY, DuckDB's arbitrary
            # LIMIT 1 scans insertion order and would return THIS record
            # (0.86), making the test fail red on the old code.
            store.remember(category="context_note",
                           content="low similarity record two", dedup=False)
            store.remember(category="context_note",
                           content="high similarity record one", dedup=False)

            hit = store.search("high similarity record one", limit=1,
                               suppress_retrieval=True)
            high_id = hit[0].memory_id
            hit = store.search("low similarity record two", limit=1,
                               suppress_retrieval=True)
            low_id = hit[0].memory_id

            # Both stored vectors are > 0.85 vs the new content; the dedup
            # must pick the 0.99 one (was: arbitrary LIMIT 1).
            mid, reason = store._find_current_similar(
                "brand new fact to check", "context_note"
            )
            assert reason == "semantic"
            assert mid == high_id, (
                f"semantic dedup returned {mid} instead of the most "
                f"similar record {high_id} (B4 regression)"
            )


class TestB5ExpansionRRFLinksStoreConstant:
    def test_expansion_merge_uses_store_rrf_k(self, monkeypatch):
        """Tuning StoreRetrievalMixin._RRF_K must change the expansion
        merge's RRF scores (was: hardcoded 20 in the provider)."""
        class StubExpander:
            enabled = True

            def expand(self, query):
                return ["expanded subquery"]

        class StubStore:
            def search(self, query, limit, **kwargs):
                if "expanded" in query:
                    return [
                        SimpleNamespace(memory_id="m1", similarity=0.1,
                                        content="sub one"),
                        SimpleNamespace(memory_id="m2", similarity=0.1,
                                        content="sub two"),
                    ]
                return [SimpleNamespace(memory_id="m0", similarity=0.3,
                                        content="original")]

        prov = object.__new__(ProviderRetrievalMixin)
        prov._query_expander = StubExpander()
        prov._store = StubStore()
        original = [SimpleNamespace(memory_id="m0", similarity=0.3,
                                    content="original")]

        monkeypatch.setattr(StoreRetrievalMixin, "_RRF_K", 40)
        try:
            merged = prov._expand_and_merge(
                "weak query", None, None, 10, original
            )
        finally:
            monkeypatch.setattr(StoreRetrievalMixin, "_RRF_K", 20)

        by_id = {r.memory_id: r for r in merged}
        assert "m2" in by_id
        # With k=40: m2 = 1/42, max = 1/41 -> normalized 41/42 (~0.9762).
        # With a hardcoded k=20 it would be 21/22 (~0.9545).
        assert by_id["m2"].similarity == pytest.approx(41.0 / 42.0, abs=1e-6), (
            "expansion RRF k is not linked to the store constant (B5)"
        )
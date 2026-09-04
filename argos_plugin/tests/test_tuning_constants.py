"""Tests for tuning constants consolidation (#248).

Asserts that tuning constants exist in tuning.py with sane values,
and that the code paths use them instead of inline literals.
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest

_plugin_dir = Path(__file__).resolve().parent.parent
if str(_plugin_dir) not in sys.path:
    sys.path.insert(0, str(_plugin_dir))


# ---------------------------------------------------------------------------
# Constants exist with sane values
# ---------------------------------------------------------------------------

class TestTuningConstantsExist:
    def test_bm25_constants(self):
        import tuning
        assert tuning.BM25_K1 == 2.2
        assert tuning.BM25_B == 0.75

    def test_dedup_threshold(self):
        import tuning
        assert tuning.DEDUP_SIMILARITY_THRESHOLD == 0.85

    def test_max_embedding_dim(self):
        import tuning
        assert tuning.MAX_EMBEDDING_DIM == 4096

    def test_alias_cache_ttl(self):
        import tuning
        assert tuning.ALIAS_CACHE_TTL_SECONDS == 60.0

    def test_graph_circuit_breaker(self):
        import tuning
        assert tuning.GRAPH_CIRCUIT_BREAKER_THRESHOLD == 5
        assert tuning.GRAPH_CIRCUIT_BREAKER_COOLDOWN == 300.0


# ---------------------------------------------------------------------------
# Code paths use the constants (no inline literals)
# ---------------------------------------------------------------------------

class TestCodeUsesTuningConstants:
    def test_text_search_uses_bm25_constants(self):
        """#248: _text_search_raw references BM25_K1/BM25_B, not inline 2.2/0.75."""
        from store_retrieval import StoreRetrievalMixin
        src = inspect.getsource(StoreRetrievalMixin._text_search_raw)
        assert "BM25_K1" in src
        assert "BM25_B" in src
        # No inline literals (2.2 or 0.75 as standalone numbers in the scoring).
        # Note: 0.5 appears in the IDF formula, not BM25 — that's fine.
        assert "2.2 *" not in src
        assert "0.75 *" not in src

    def test_dedup_threshold_from_tuning(self):
        """#248: _DEDUP_SIMILARITY_THRESHOLD references tuning.py."""
        from store_retrieval import StoreRetrievalMixin
        # Class attr should alias the tuning constant.
        import tuning
        assert StoreRetrievalMixin._DEDUP_SIMILARITY_THRESHOLD == tuning.DEDUP_SIMILARITY_THRESHOLD

    def test_max_embedding_dim_from_tuning(self):
        """#248: _MAX_EMBEDDING_DIM references tuning.py."""
        from store_retrieval import StoreRetrievalMixin
        import tuning
        assert StoreRetrievalMixin._MAX_EMBEDDING_DIM == tuning.MAX_EMBEDDING_DIM

    def test_alias_cache_ttl_from_tuning(self):
        """#248: _ALIAS_CACHE_TTL_SECONDS references tuning.py."""
        from provider_retrieval import ProviderRetrievalMixin
        import tuning
        assert ProviderRetrievalMixin._ALIAS_CACHE_TTL_SECONDS == tuning.ALIAS_CACHE_TTL_SECONDS

    def test_graph_circuit_breaker_from_tuning(self):
        """#248: graph circuit breaker constants reference tuning.py."""
        from provider_retrieval import ProviderRetrievalMixin
        import tuning
        assert ProviderRetrievalMixin._GRAPH_CIRCUIT_BREAKER_THRESHOLD == tuning.GRAPH_CIRCUIT_BREAKER_THRESHOLD
        assert ProviderRetrievalMixin._GRAPH_CIRCUIT_BREAKER_COOLDOWN == tuning.GRAPH_CIRCUIT_BREAKER_COOLDOWN

    def test_store_retrieval_imports_tuning(self):
        """#248: store_retrieval.py imports from tuning.py."""
        import store_retrieval
        src = inspect.getsource(store_retrieval)
        assert "from" in src and "tuning" in src
        assert "BM25_K1" in src

    def test_provider_retrieval_imports_tuning(self):
        """#248: provider_retrieval.py imports from tuning.py."""
        import provider_retrieval
        src = inspect.getsource(provider_retrieval)
        assert "from" in src and "tuning" in src
        assert "ALIAS_CACHE_TTL" in src or "_TUNING_ALIAS_CACHE_TTL" in src

"""Audit tests for store_retrieval SR13 (issue #265).

Covers embedding dimension validation in semantic dedup / vector search.

Run with (Hermes venv python, offline):
    python -m pytest tests/test_store_retrieval_audit2.py -v
"""
from __future__ import annotations

import inspect
import logging
import sys
from pathlib import Path

import pytest

_plugin_dir = Path(__file__).resolve().parent.parent
if str(_plugin_dir) not in sys.path:
    sys.path.insert(0, str(_plugin_dir))


@pytest.fixture
def store(tmp_path):
    from store import DuckDBMemoryStore
    s = DuckDBMemoryStore(tmp_path / "test.duckdb", user_id="alice")
    yield s
    s.close()


# ---------------------------------------------------------------------------
# SR13 -- embedding dimension validation
# ---------------------------------------------------------------------------

class TestSR13EmbeddingDimValidation:
    def test_max_embedding_dim_constant(self):
        """SR13: _MAX_EMBEDDING_DIM constant exists."""
        from store_retrieval import StoreRetrievalMixin
        assert hasattr(StoreRetrievalMixin, "_MAX_EMBEDDING_DIM")
        assert StoreRetrievalMixin._MAX_EMBEDDING_DIM > 0

    def test_vector_search_validates_dim_in_source(self):
        """SR13: _vector_search_raw validates embedding dim before SQL."""
        from store_retrieval import StoreRetrievalMixin
        src = inspect.getsource(StoreRetrievalMixin._vector_search_raw)
        assert "_MAX_EMBEDDING_DIM" in src
        assert "invalid dim" in src

    def test_find_current_similar_validates_dim_in_source(self):
        """SR13: _find_current_similar validates embedding dim before SQL."""
        from store_retrieval import StoreRetrievalMixin
        src = inspect.getsource(StoreRetrievalMixin._find_current_similar)
        assert "_MAX_EMBEDDING_DIM" in src

    def test_empty_embedding_returns_empty(self, store, caplog):
        """SR13: empty embedding returns [] from _vector_search_raw
        without raising."""
        with caplog.at_level(logging.WARNING):
            result = store._vector_search_raw([], 10, set())
        assert result == []
        assert any("invalid dim" in r.message for r in caplog.records)

    def test_oversized_embedding_returns_empty(self, store, caplog):
        """SR13: oversized embedding returns [] from _vector_search_raw
        without raising."""
        huge_emb = [0.1] * 5000  # > _MAX_EMBEDDING_DIM (4096)
        with caplog.at_level(logging.WARNING):
            result = store._vector_search_raw(huge_emb, 10, set())
        assert result == []
        assert any("invalid dim" in r.message for r in caplog.records)

    def test_find_current_similar_empty_embedding_no_raise(self, store, caplog):
        """SR13: _find_current_similar with oversized embedding returns
        (None, None) without raising."""
        class FakeEmbedder:
            def embed(self, content):
                return [0.1] * 5000  # oversized
        store.embedder = FakeEmbedder()
        with caplog.at_level(logging.WARNING):
            mid, reason = store._find_current_similar("test content", "insight")
        assert mid is None
        assert reason is None

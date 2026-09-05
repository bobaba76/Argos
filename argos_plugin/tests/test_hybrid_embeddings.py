"""Pytest tests for the argos plugin.

Run with:
    python -m pytest tests/test_argos.py -v

"""
from __future__ import annotations

import sys
import tempfile
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pytest

# Ensure the plugin package is importable.
_plugin_dir = Path(__file__).resolve().parent.parent
if str(_plugin_dir.parent) not in sys.path:
    sys.path.insert(0, str(_plugin_dir.parent))


@pytest.fixture
def tmp_path_factory_dir(tmp_path):
    """Provide a clean temp directory for each test."""
    return tmp_path


class TestEmbeddings:
    def test_embed_returns_list_never_raises(self):
        from embeddings import LocalEmbedder

        emb = LocalEmbedder("nonexistent-model-xyz")
        result = emb.embed("test text")
        assert isinstance(result, list)

    def test_embed_empty_string(self):
        from embeddings import LocalEmbedder

        emb = LocalEmbedder()
        assert emb.embed("") == []

    def test_embed_accepts_is_query_flag(self):
        """is_query must be accepted without error, even if model fails to load."""
        from embeddings import LocalEmbedder

        emb = LocalEmbedder("nonexistent-model-xyz")
        # Should not raise — just returns [] if model unavailable.
        result = emb.embed("test text", is_query=True)
        assert isinstance(result, list)

    def test_embed_batch_accepts_is_query_flag(self):
        from embeddings import LocalEmbedder

        emb = LocalEmbedder("nonexistent-model-xyz")
        result = emb.embed_batch(["text one", "text two"], is_query=True)
        assert isinstance(result, list)
        assert len(result) == 2

    def test_bge_model_gets_query_prefix(self):
        """BGE models must apply a query instruction when is_query=True."""
        from embeddings import LocalEmbedder, _query_instruction_for

        instruction = _query_instruction_for("BAAI/bge-small-en-v1.5")
        assert instruction != "", "BGE model should have a query instruction"
        assert "searching" in instruction.lower() or "query" in instruction.lower()

    def test_symmetric_model_gets_no_prefix(self):
        """multi-qa-MiniLM (the old default) is symmetric — no query prefix."""
        from embeddings import _query_instruction_for

        instruction = _query_instruction_for("sentence-transformers/multi-qa-MiniLM-L6-cos-v1")
        assert instruction == "", "Symmetric model should have no query instruction"

    def test_prepare_text_query_vs_document(self):
        """_prepare_text must prefix queries but not documents for BGE."""
        from embeddings import LocalEmbedder

        emb = LocalEmbedder("BAAI/bge-small-en-v1.5")
        doc_text = emb._prepare_text("hello world", is_query=False)
        query_text = emb._prepare_text("hello world", is_query=True)
        assert doc_text == "hello world"
        assert query_text != "hello world"
        assert "hello world" in query_text  # the original text is still there

    def test_default_model_is_bge_small(self):
        """The default model must be bge-small-en-v1.5 (the upgrade)."""
        from embeddings import _DEFAULT_MODEL

        assert "bge-small-en-v1.5" in _DEFAULT_MODEL

    def test_none_model_name_coerces_to_default(self):
        """LocalEmbedder(model_name=None) must not crash (issue #45).

        A None model_name used to crash deferredly inside
        _query_instruction_for -> model_name.lower(), silently emptying
        all retrieval. It must coerce to the default instead.
        """
        from embeddings import LocalEmbedder, _DEFAULT_MODEL

        emb = LocalEmbedder(model_name=None)
        assert emb._model_name == _DEFAULT_MODEL
        # _prepare_text must not raise on the query path:
        result = emb._prepare_text("test query", is_query=True)
        assert isinstance(result, str)

    def test_embed_query_with_none_model_does_not_raise(self):
        """embed(is_query=True) must not raise even with a None-origin
        model (issue #45): the None is coerced to the default at init,
        so the query-instruction path never sees a None model_name."""
        from embeddings import LocalEmbedder

        emb = LocalEmbedder(model_name=None)
        result = emb.embed("test query", is_query=True)
        assert isinstance(result, list)



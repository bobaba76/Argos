"""Deterministic, dependency-free test embedder (issue #98).

Mirrors the ``LocalEmbedder`` interface (``embed`` / ``embed_batch``) but
computes character n-gram hashing vectors instead of loading a real model.

Why: the suite's embedding-heavy tests (distillation, dedup, extraction)
seed dozens of records per test.  The real BGE model costs ~0.3-1.9s per
encode on this machine (the GPU path is not meaningfully engaged), which
makes those files the dominant share of the suite's wall time.  A
deterministic embedder derives similarity from lexical overlap -- exactly
what those tests need (near-identical seeded clusters must cluster;
unrelated texts must not) -- at microseconds per encode, with no model
load and no ``HF_HUB_OFFLINE`` requirement.

Vectors are L2-normalized (like ``LocalEmbedder`` with
``normalize_embeddings=True``), so cosine similarity equals the dot
product.  ``is_query`` is ignored: the embedder is symmetric, which is the
behavior the tests assume for stored content and search alike.
"""
from __future__ import annotations

import re
import zlib
from typing import List

_DIM = 384          # same as BGE-small, so record vectors are shape-compatible
_NGRAM = 3          # character trigrams -- good precision for near-duplicates
_WS = re.compile(r"\s+")


def _normalize(text: str) -> str:
    """Lowercase and collapse whitespace before n-gram extraction."""
    return _WS.sub(" ", text).strip().lower()


def _ngrams(text: str) -> List[str]:
    """Character n-grams of the normalized text."""
    if len(text) < _NGRAM:
        return [text] if text else []
    return [text[i : i + _NGRAM] for i in range(len(text) - _NGRAM + 1)]


class DeterministicTestEmbedder:
    """LocalEmbedder-compatible embedder using character n-gram hashing."""

    def __init__(self, model_name: str = "deterministic-test"):
        self._model_name = model_name

    def _vector(self, text: str) -> List[float]:
        vec = [0.0] * _DIM
        for gram in _ngrams(_normalize(text)):
            idx = zlib.crc32(gram.encode("utf-8")) % _DIM
            vec[idx] += 1.0
        norm = sum(v * v for v in vec) ** 0.5
        if norm == 0.0:
            return []
        return [v / norm for v in vec]

    def embed(self, text: str, *, is_query: bool = False) -> List[float]:
        """Return a deterministic embedding vector, or [] for blank text."""
        if not text or not text.strip():
            return []
        return self._vector(text)

    def embed_batch(self, texts: List[str], *, is_query: bool = False) -> List[List[float]]:
        """Embed multiple texts at once."""
        if not texts:
            return []
        return [self.embed(t, is_query=is_query) for t in texts]

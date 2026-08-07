"""Local sentence-transformers embedding wrapper with graceful fallback.

Loads ``sentence-transformers/bge-small-en-v1.5`` lazily on first use.
If the model or library is unavailable, ``embed()`` returns an empty list and
the DuckDB store falls back to text search transparently.

Uses a process-level shared model so that multiple HybridMemoryProvider
instances (Hermes creates one per agent/session) all reuse the same loaded
model instead of reloading the ~130MB model on every new session.

Query-prefix convention
-----------------------
Some models (notably BGE/E5) are *asymmetric*: queries must be prefixed with
an instruction while stored documents are embedded raw.  The old default
``multi-qa-MiniLM-L6-cos-v1`` is symmetric and needs no prefix.  ``embed()``
and ``embed_batch()`` accept ``is_query=True`` to apply the correct prefix at
search time.  Callers embedding stored content (the default) pass no flag.
"""
from __future__ import annotations

import logging
import threading
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "sentence-transformers/bge-small-en-v1.5"

# Process-level shared model cache: {model_name: (model, dim)}
_SHARED_MODELS: Dict[str, Tuple[object, int]] = {}
_SHARED_LOCK = threading.Lock()

# Query instructions for asymmetric models.  Keys are matched by substring
# so that local cache paths (e.g. "models--sentence-transformers--bge-small-en-v1.5")
# also match.  Models not listed here are treated as symmetric (no prefix).
_QUERY_INSTRUCTIONS: Dict[str, str] = {
    "bge-small-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "bge-base-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "bge-large-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "e5-small-v2": "query: ",
    "e5-base-v2": "query: ",
    "e5-large-v2": "query: ",
}


def _query_instruction_for(model_name: str) -> str:
    """Return the query prefix for *model_name*, or '' if symmetric."""
    name_lower = model_name.lower()
    for key, instruction in _QUERY_INSTRUCTIONS.items():
        if key in name_lower:
            return instruction
    return ""


class LocalEmbedder:
    """Thin wrapper around sentence-transformers with lazy init + fallback.

    The underlying SentenceTransformer model is shared across all instances
    in the same process via a class-level cache. This avoids reloading the
    ~130MB model every time Hermes creates a new agent/session.
    """

    def __init__(self, model_name: str = _DEFAULT_MODEL) -> None:
        self._model_name = model_name
        self._loaded = False
        self._load_failed = False
        self._lock = threading.Lock()
        self._dim: Optional[int] = None

    @property
    def is_available(self) -> bool:
        """True if the embedding model loaded successfully."""
        # Check shared cache first.
        shared = _SHARED_MODELS.get(self._model_name)
        if shared is not None:
            return True
        return self._loaded and _SHARED_MODELS.get(self._model_name) is not None

    @property
    def dimension(self) -> Optional[int]:
        shared = _SHARED_MODELS.get(self._model_name)
        if shared is not None:
            return shared[1]
        return self._dim

    def _ensure_loaded(self) -> None:
        """Lazy-load the model on first call. Thread-safe.

        Uses a process-level shared cache so the model is only loaded once
        per process, regardless of how many LocalEmbedder instances exist.
        """
        if self._loaded or self._load_failed:
            return
        with self._lock:
            if self._loaded or self._load_failed:
                return
            # Check if another instance already loaded it.
            with _SHARED_LOCK:
                shared = _SHARED_MODELS.get(self._model_name)
                if shared is not None:
                    self._dim = shared[1]
                    self._loaded = True
                    logger.debug(
                        "Embedding model reused from cache (dim=%d)", self._dim
                    )
                    return
            try:
                from sentence_transformers import SentenceTransformer

                logger.info("Loading embedding model: %s", self._model_name)
                model = SentenceTransformer(self._model_name)
                # Probe dimension with a dummy encode.
                test = model.encode(["dimension probe"], normalize_embeddings=True)
                dim = len(test[0])
                # Store in shared cache.
                with _SHARED_LOCK:
                    _SHARED_MODELS[self._model_name] = (model, dim)
                self._dim = dim
                self._loaded = True
                logger.info("Embedding model loaded (dim=%d)", self._dim)
            except Exception as e:
                self._load_failed = True
                logger.warning(
                    "Embedding model '%s' unavailable — falling back to text search. "
                    "Reason: %s",
                    self._model_name, e,
                )

    def _prepare_text(self, text: str, is_query: bool) -> str:
        """Apply the query instruction prefix if the model requires it."""
        if not is_query:
            return text
        instruction = _query_instruction_for(self._model_name)
        if instruction:
            return instruction + text
        return text

    def embed(self, text: str, *, is_query: bool = False) -> List[float]:
        """Return an embedding vector for *text*, or [] if unavailable.

        When *is_query* is True, the model's query instruction is prepended
        (for asymmetric models like BGE/E5).  Stored content should use the
        default (is_query=False) so documents are embedded raw.
        """
        if not text or not text.strip():
            return []
        self._ensure_loaded()
        with _SHARED_LOCK:
            shared = _SHARED_MODELS.get(self._model_name)
        if shared is None:
            return []
        model = shared[0]
        prepared = self._prepare_text(text, is_query)
        try:
            vec = model.encode([prepared], normalize_embeddings=True)
            return [float(x) for x in vec[0]]
        except Exception as e:
            logger.debug("Embedding failed for text (%s): %s", text[:60], e)
            return []

    def embed_batch(self, texts: List[str], *, is_query: bool = False) -> List[List[float]]:
        """Embed multiple texts at once. Returns [] per-item if unavailable.

        When *is_query* is True, the query instruction is applied to each text.
        """
        if not texts:
            return []
        self._ensure_loaded()
        with _SHARED_LOCK:
            shared = _SHARED_MODELS.get(self._model_name)
        if shared is None:
            return [[] for _ in texts]
        model = shared[0]
        prepared = [self._prepare_text(t, is_query) for t in texts]
        try:
            vecs = model.encode(prepared, normalize_embeddings=True)
            return [[float(x) for x in v] for v in vecs]
        except Exception as e:
            logger.debug("Batch embedding failed: %s", e)
            return [[] for _ in texts]

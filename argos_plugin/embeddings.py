"""Local sentence-transformers embedding wrapper with graceful fallback.

Loads ``BAAI/bge-small-en-v1.5`` lazily on first use.
If the model or library is unavailable, ``embed()`` returns an empty list and
the DuckDB store falls back to text search transparently.

Uses a process-level shared model so that multiple ArgosProvider
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
import os
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"

# Process-level shared model cache: {model_name: (model, dim)}
_SHARED_MODELS: Dict[str, Tuple[object, int]] = {}
_SHARED_LOCK = threading.Lock()

# Bounded query-embedding cache (2026-08-22): searches re-embed the same
# natural-language query strings repeatedly across turns ("what did we do in
# August").  Query vectors are small (384 floats) and stable per text, so a
# small FIFO cache removes the per-search CPU embed call (~100ms+ on-device).
# Content embeddings (is_query=False) are NEVER cached — stored text is not
# immutable.  Key: "<model_name>::<prepared text>"; value is a tuple so the
# cached vector cannot be mutated by callers.
_QUERY_EMBED_CACHE: "OrderedDict[str, Tuple[float, ...]]" = OrderedDict()
_QUERY_EMBED_CACHE_MAX = 512
_QUERY_EMBED_LOCK = threading.Lock()


def clear_query_embed_cache() -> None:
    """Drop all cached query embeddings (tests, model swaps)."""
    with _QUERY_EMBED_LOCK:
        _QUERY_EMBED_CACHE.clear()

# Query instructions for asymmetric models.  Keys are matched by substring
# so that local cache paths (e.g. "models--sentence-transformers--bge-small-en-v1.5")
# also match.  Models not listed here are treated as symmetric (no prefix).
_QUERY_INSTRUCTIONS: Dict[str, str] = {
    "bge-small-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "bge-base-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "bge-large-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "bge-m3": "Represent this sentence for searching relevant passages: ",
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


def _resolve_embedding_model_path(
    model_name: str, hermes_home: Optional[Path] = None
) -> str:
    """Resolve a configured model name to a fully local path when possible.

    Priority: explicit existing path > ``<hermes_home>/models/<name>`` > HF
    cache lookup (resolves bare names like ``bge-small-en-v1.5`` to full hub
    names like ``BAAI/bge-small-en-v1.5``) > hub name.  Local-first avoids
    the network HEAD-check sentence-transformers performs when loading by
    hub name — which fails behind firewalls and previously degraded to
    silent text-only search on fresh installs.
    """
    name = (model_name or "").strip() or _DEFAULT_MODEL
    p = Path(name)
    if p.is_dir():
        return str(p)
    if hermes_home is not None:
        candidate = Path(hermes_home) / "models" / Path(name).name
        if candidate.is_dir():
            return str(candidate)
    # Bare name (no org prefix): try to resolve to a full hub name via the
    # HF cache. This fixes "bge-small-en-v1.5" → "BAAI/bge-small-en-v1.5"
    # so sentence-transformers can find it in the cache.
    if "/" not in name:
        for org in ("BAAI", "sentence-transformers"):
            full = f"{org}/{name}"
            if _is_in_hf_cache(full):
                return full
    return name


def _is_in_hf_cache(hub_name: str) -> bool:
    """Check if a HuggingFace hub model name is present in the local cache.

    sentence-transformers downloads models to the HuggingFace hub cache
    (``~/.cache/huggingface/hub/models--<org>--<name>/snapshots/...``).
    When a model is already cached, we can load it with
    ``local_files_only=True`` to avoid the network HEAD-check that
    fails behind firewalls and adds latency on every load.
    """
    if not hub_name or "/" not in hub_name:
        # Short names like "bge-small-en-v1.5" (no org prefix) — check
        # both the bare name and common org prefixes.
        candidates = [
            f"models--BAAI--{hub_name}",
            f"models--sentence-transformers--{hub_name}",
        ]
    else:
        org, _, model = hub_name.partition("/")
        candidates = [f"models--{org}--{model}"]
    cache_root = Path.home() / ".cache" / "huggingface" / "hub"
    for candidate in candidates:
        snap_dir = cache_root / candidate / "snapshots"
        if snap_dir.is_dir() and any(snap_dir.iterdir()):
            return True
    return False


def _ensure_offline_mode(resolved: str) -> None:
    """Set HF_HUB_OFFLINE=1 when the model is in the local cache.

    sentence-transformers' ``local_files_only=True`` flag does not fully
    suppress network HEAD-checks in all versions — the underlying
    huggingface_hub client still tries to reach huggingface.co for
    metadata. Setting ``HF_HUB_OFFLINE=1`` (and the older
    ``TRANSFORMERS_OFFLINE=1``) forces true offline behavior. We only
    set it when we know the model is cached, so a genuinely uncached
    model can still be downloaded on first use.
    """
    if not os.environ.get("HF_HUB_OFFLINE"):
        if Path(resolved).is_dir() or _is_in_hf_cache(resolved):
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"


class LocalEmbedder:
    """Thin wrapper around sentence-transformers with lazy init + fallback.

    The underlying SentenceTransformer model is shared across all instances
    in the same process via a class-level cache. This avoids reloading the
    ~130MB model every time Hermes creates a new agent/session.
    """

    def __init__(self, model_name: str = _DEFAULT_MODEL,
                 hermes_home: Optional[Path] = None) -> None:
        # Coerce None to the default (issue #45): a None model_name crashed
        # deferredly inside _query_instruction_for -> model_name.lower(),
        # silently emptying all retrieval. Treat None as "use the default"
        # so a caller that passes None (e.g. an unset CLI flag) gets a
        # working embedder, not a crash-at-query-time trap.
        self._model_name = model_name or _DEFAULT_MODEL
        self._hermes_home = hermes_home
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
    def load_failed(self) -> bool:
        """True after a failed load attempt (degraded, not just pending)."""
        return self._load_failed

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

                resolved = _resolve_embedding_model_path(
                    self._model_name, self._hermes_home
                )
                use_local_only = Path(resolved).is_dir() or _is_in_hf_cache(resolved)
                _ensure_offline_mode(resolved)
                logger.info(
                    "Loading embedding model: %s (local_files_only=%s)",
                    resolved, use_local_only,
                )
                model = SentenceTransformer(resolved, local_files_only=use_local_only)
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
                logger.error(
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
        (for asymmetric models like BGE/E5), and the result is served from
        a bounded FIFO cache — search queries repeat far more often than
        they change.  Stored content should use the default (is_query=False)
        so documents are embedded raw and never cached.
        """
        if not text or not text.strip():
            return []
        prepared = self._prepare_text(text, is_query)
        if is_query:
            cache_key = f"{self._model_name}::{prepared}"
            with _QUERY_EMBED_LOCK:
                cached = _QUERY_EMBED_CACHE.get(cache_key)
            if cached is not None:
                return list(cached)
        self._ensure_loaded()
        with _SHARED_LOCK:
            shared = _SHARED_MODELS.get(self._model_name)
        if shared is None:
            return []
        model = shared[0]
        try:
            vec = model.encode([prepared], normalize_embeddings=True)
            result = [float(x) for x in vec[0]]
            if is_query:
                with _QUERY_EMBED_LOCK:
                    if len(_QUERY_EMBED_CACHE) >= _QUERY_EMBED_CACHE_MAX:
                        _QUERY_EMBED_CACHE.popitem(last=False)
                    _QUERY_EMBED_CACHE[cache_key] = tuple(result)
            return result
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


# ---------------------------------------------------------------------------
# Cross-encoder reranker
# ---------------------------------------------------------------------------

_DEFAULT_RERANKER = "BAAI/bge-reranker-base"

# Process-level shared reranker cache: {model_name: model}
_SHARED_RERANKERS: Dict[str, object] = {}
_SHARED_RERANKER_LOCK = threading.Lock()


class CrossEncoderReranker:
    """Cross-encoder reranker for second-stage relevance scoring.

    Unlike bi-encoders (LocalEmbedder), a cross-encoder reads the query
    and document *together* with full bidirectional attention, producing
    a relevance score that captures subtle semantic matches the bi-encoder
    misses. This is the standard trick for improving top-k ranking quality
    in RAG pipelines.

    Lazy-loaded and process-shared (same pattern as LocalEmbedder).
    Falls back gracefully — if the model is unavailable, ``rerank()``
    returns the input unchanged.
    """

    def __init__(self, model_name: str = _DEFAULT_RERANKER,
                 hermes_home: Optional[Path] = None) -> None:
        self._model_name = model_name
        self._hermes_home = hermes_home
        self._loaded = False
        self._load_failed = False
        self._lock = threading.Lock()

    @property
    def is_available(self) -> bool:
        shared = _SHARED_RERANKERS.get(self._model_name)
        if shared is not None:
            return True
        return self._loaded and _SHARED_RERANKERS.get(self._model_name) is not None

    def _ensure_loaded(self) -> None:
        """Lazy-load the reranker model on first call. Thread-safe."""
        if self._loaded or self._load_failed:
            return
        with self._lock:
            if self._loaded or self._load_failed:
                return
            with _SHARED_RERANKER_LOCK:
                shared = _SHARED_RERANKERS.get(self._model_name)
                if shared is not None:
                    self._loaded = True
                    logger.debug("Reranker reused from cache: %s", self._model_name)
                    return
            try:
                from sentence_transformers import CrossEncoder

                logger.info("Loading reranker model: %s", self._model_name)
                resolved = _resolve_embedding_model_path(
                    self._model_name, self._hermes_home
                )
                use_local_only = Path(resolved).is_dir() or _is_in_hf_cache(resolved)
                _ensure_offline_mode(resolved)
                model = CrossEncoder(
                    resolved, max_length=512,
                    local_files_only=use_local_only,
                )
                with _SHARED_RERANKER_LOCK:
                    _SHARED_RERANKERS[self._model_name] = model
                self._loaded = True
                logger.info("Reranker loaded: %s", self._model_name)
            except Exception as e:
                self._load_failed = True
                logger.warning(
                    "Reranker model '%s' unavailable — falling back to bi-encoder ranking. "
                    "Reason: %s",
                    self._model_name, e,
                )

    def score(self, query: str, documents: List[str]) -> List[float]:
        """Score (query, document) pairs. Returns one float per document.

        Higher score = more relevant. If the reranker is unavailable,
        returns empty list (caller should fall back to existing ranking).
        """
        if not query or not documents:
            return []
        self._ensure_loaded()
        with _SHARED_RERANKER_LOCK:
            model = _SHARED_RERANKERS.get(self._model_name)
        if model is None:
            return []
        try:
            pairs = [(query, doc) for doc in documents]
            scores = model.predict(pairs)
            return [float(s) for s in scores]
        except Exception as e:
            logger.debug("Reranker scoring failed: %s", e)
            return []

"""Tuning constants for retrieval, dedup, and circuit-breaker behavior.

#248: consolidated from inline literals and scattered class attributes
so tuning parameters are visible and adjustable from one place. Only
clear tuning constants live here — path strings, counts, and sizes that
are structural (not tuning) stay inline at their use sites.
"""
from __future__ import annotations

# -- BM25-lite retrieval (store_retrieval._text_search_raw) ------------------
# Standard BM25 parameters: k1 controls term-frequency saturation,
# b controls length normalization. Tuned for short-to-medium memory
# records (avg 50-200 tokens).
BM25_K1 = 2.2
BM25_B = 0.75

# -- Semantic dedup (store_retrieval._find_current_similar) ------------------
# Cosine similarity above this means "same fact" — used to gate the
# semantic dedup layer in save_candidate / ingest_versioned.
DEDUP_SIMILARITY_THRESHOLD = 0.85

# -- Embedding dimension cap (store_retrieval._vector_search_raw) -------------
# Embeddings with 0 dims or above this cap are rejected before SQL
# interpolation (SR13). 4096 covers all common embedding models.
MAX_EMBEDDING_DIM = 4096

# -- Alias cache (provider_retrieval.ProviderRetrievalMixin) ------------------
# TTL in seconds for the resolved-alias cache. After this, resolve_aliases
# re-queries the store.
ALIAS_CACHE_TTL_SECONDS = 60.0

# -- Graph circuit breaker (provider_retrieval.ProviderRetrievalMixin) --------
# After this many consecutive failures, graph lookups are short-circuited
# (fail-closed) for the cooldown period to avoid stalling the hot path.
GRAPH_CIRCUIT_BREAKER_THRESHOLD = 5
GRAPH_CIRCUIT_BREAKER_COOLDOWN = 300.0  # 5 minutes

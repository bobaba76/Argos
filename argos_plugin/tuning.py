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

# -- Reciprocal Rank Fusion (store_retrieval._fuse_rrf) ----------------------
# RRF_K is the smoothing constant in the RRF score: 1/(k + rank).
# Lower k sharpens relevance discrimination (favors top ranks more).
# Monkeypatched in tests (test_retrieval_audit_fixes.py:221-227).
RRF_K = 20

# #445: bounded CE-rescue slots (store_retrieval reranker block).
#   CE_PROMOTE_MAX — max rescued records per query (never displaces the
#                    strict ranking lane beyond the window tail; rescue
#                    eligibility is CE-vs-window-tail, no absolute floor)
CE_PROMOTE_MAX = 2

# #445: VECTOR-ARM template-dialect probes. The bi-encoder is even more
# phrasing-locked than the CE (measured 11/9: 'what is user's job title'
# ranks the answering record #5, while 'what is user's role' / 'occupation'
# / 'do for work' are all ≥#128 or absent — the record literally stores
# "job title", so only probes carrying the STORED keyword hit). These
# probes are embedded and merged into the vector candidate set so a
# paraphrased query still surfaces the record to the ranker at all.
VECTOR_PROBE_ALIASES = {
    "work": (
        r"\b(work as|job title|current role|job|role|employed|occupation|"
        r"profession|position)\b",
        ("what is user's job title",),
    ),
    "location": (
        r"\b(live|lives|living|address|resid(?:e|ence)|located|stay(?:ing)?)\b",
        ("where does user live", "what is user's address"),
    ),
}

# #445: query-side template-dialect probes for the cross-encoder.
# bge-reranker-* is verb-phrase locked: a natural NL query ("What do I
# currently work as?") scores Argos's canonical record template ("User's
# job title is ...") ~0.00, while the SAME query in the record's dialect
# ("What is user's job title?") scores it 0.99 (measured 11/9). Each
# family: (intent regex, probe list). Probes are scored per record and
# the MAX is kept — unrelated queries pay nothing (no regex hit).
# Perf (#460, 11/9): trimmed to ONE measured-star probe per family —
# every extra probe is a full extra CE pass over the pool (each pass is
# the dominant search cost; the vector-arm probes already guarantee pool
# entry, so the CE probe only needs to bridge the phrasing gap, which the
# star probe does).
CE_PROBE_ALIASES = {
    "work": (
        r"\b(work as|job title|current role|job|role|employed|occupation|"
        r"profession|position)\b",
        ("what is user's job title",),
    ),
    "location": (
        r"\b(live|lives|living|address|resid(?:e|ence)|located|stay(?:ing)?)\b",
        ("where does user live",),
    ),
}

# #460: CE input truncation cap (chars). The cross-encoder tokenizer
# (SentencePiece, Python-side) processes the FULL document string before
# the model's own max_length=512 truncation — on real memory content the
# tokenizer pass, not the GPU matmul, dominates the per-pass cost
# (measured 11/9: ~30us/char; a 56-doc pool with a few 2.5k-char records
# cost 1.07s, truncating at 700 chars halved it). The cap bounds input
# before tokenization; records shorter than the cap are untouched.
CE_MAX_DOC_CHARS = 700

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

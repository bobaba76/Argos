# Tuning

Argos ships with sensible defaults. This page covers the knobs that matter most for retrieval quality, latency, and cost — and points to the tuning matrix from issue #283 (embedder/reranker benchmarks) once it's available.

## Embeddings

The default embedding model is `BAAI/bge-small-en-v1.5` (~130MB, sentence-transformers, offline). It's a good balance of quality and size for a local-first system.

### When to change

- **Different language:** swap to a multilingual model (e.g. `paraphrase-multilingual-MiniLM-L12-v2`). Update `local_embedding_model` in `hybrid_memory.json`.
- **Higher quality, more storage:** a larger model (e.g. `BAAI/bge-large-en-v1.5`, ~1.3GB) may improve recall on complex queries. The store records `embedding_dim` per record, so mixed-dimension records are detected and a text fallback is used for incompatible queries (see #286).
- **No model available (offline):** if the model fails to load, Argos falls back to text-only search (keyword + RRF). Vector search is skipped — recall drops, but the system stays functional.

### Re-embedding

If you change the embedding model, existing records keep their old embeddings. Use the re-embedding CLI (`reembed_store()`) to backfill. The store tracks `embedder_id` and `embedded_at` per record so you can audit which model produced which embeddings.

## Reranker

The reranker is a cross-encoder (`BAAI/bge-reranker-base`, ~420MB) that re-ranks the top-N candidates. It's **off by default** — on CPU it adds ~8s per query, which is too slow for interactive use. Enable it only with CUDA torch.

### When to enable

```json
{
  "reranker_enabled": "true",
  "reranker_model": "BAAI/bge-reranker-base",
  "reranker_top_n": "10"
}
```

- **Precision-critical use cases** where latency is acceptable (batch jobs, offline analysis).
- **CUDA torch available.** CPU-only deployments should leave it off.

### Tuning matrix (#283)

The full embedder/reranker benchmark matrix — recall@k, latency, storage size per model combination — is tracked in [issue #283](https://github.com/bobaba76/Argos/issues/283). It is **not yet built**. When it lands, the numbers will be published here and in the repo's eval snapshots.

Until then, the defaults are the validated configuration:

| Component | Default | Notes |
|-----------|---------|-------|
| Embedder | `BAAI/bge-small-en-v1.5` | ~130MB, offline, good recall/size balance. |
| Reranker | off | Enable only with CUDA torch. |
| `injection_min_score` | `0.30` | Validated relevance floor. |
| `max_injected_items` | `20` | Tuned for context window budget. |

## Retrieval knobs

### Relevance floor

`injection_min_score` (default `0.30`) drops items below this similarity. Raise it for precision (fewer false positives); lower it for recall (more context, more noise).

### Context-aware retrieval

`context_aware_retrieval` (default `true`) prepends recent conversation context to queries with pronouns/references. Zero latency, no LLM calls. Leave it on.

### Query expansion

`query_expansion_enabled` (default `true`) rewrites weak queries (top hit below `query_expansion_similarity_floor`) into sub-queries for better recall. It's an LLM call — fail-soft, cached 1h. Disable it for air-gapped deployments or to save cost.

### Graph boost

`graph_retrieval_boost` (default `0.0`) is a no-op at default. Raise it (up to `0.5`) to boost memories supported by graph entities. `graph_boost_min_similarity` (default `0.15`) is the floor — a memory must clear this semantic similarity to receive the graph boost.

### Graph traversal

`graph_traversal_enabled` (default `true`) with `graph_traversal_depth` (default `2`) and `graph_traversal_boost` (default `0.60`) controls multi-hop retrieval. The boost is applied to candidates found by traversing the graph from entities in the query.

### Personalized PageRank

`graph_ppr_enabled` (default `false`) is experimental. Enable it for graph-heavy use cases where traversal boost is insufficient. Tune `graph_ppr_damping` (default `0.5`) and `graph_ppr_boost` (default `0.0`).

## Chains (version evolution)

`chain_unfold` (default `auto`) injects a compact version arc on change-intent queries. The three modes:

| Mode | Behavior |
|------|----------|
| `off` | No auto-unfold. Use `memory_fetch_history` on demand. |
| `auto` | Unfold when the query has change-intent ("what changed?", "update"). |
| `always` | Unfold every chained result. Higher context cost. |

`chain_unfold_min_similarity` (default `0.30`) is the precision guard — only candidates above this similarity trigger unfold. Raise for precision; lower for recall.

## Conflict surfacing

`conflict_surfacing` (default `true`) injects an explicit conflict note when two active records disagree on the same subject. Keep it on for trust-critical use cases; turn it off if the notes add noise.

## Quick tuning recipes

### Precision-focused (fewer false positives)

```json
{
  "injection_min_score": "0.40",
  "chain_unfold_min_similarity": "0.35",
  "reranker_enabled": "true",
  "reranker_top_n": "20"
}
```

### Recall-focused (more context)

```json
{
  "injection_min_score": "0.20",
  "max_injected_items": "30",
  "chain_unfold": "auto",
  "graph_inject_candidates": "true"
}
```

### Air-gapped (no LLM calls)

```json
{
  "local_only": "true",
  "query_expansion_enabled": "false",
  "reranker_enabled": "false"
}
```

## Next steps

- [Configuration](configuration.md) — the full settings reference.
- [API reference](api/index.md) — MCP and REST.

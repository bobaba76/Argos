# Configuration

Settings live in `hybrid_memory.json` in the Hermes home directory (`~/.hermes/` on Linux/macOS, `%LOCALAPPDATA%\hermes\` on Windows). The file is created on first run with defaults. The settings UI reads the same file — what you see is what's live.

This page mirrors [`CONFIG_REFERENCE.md`](https://github.com/bobaba76/Argos/blob/master/CONFIG_REFERENCE.md) in the repo. If the two disagree, the repo file is authoritative (it's generated from the live config loader).

## Storage

| Setting | Default | Description |
|---------|---------|-------------|
| `storage_mode` | `shared_service` | `shared_service` (RPC service owns the DB, multi-process safe) or `direct` (plugin opens DuckDB directly, single-process; diagnostics only). |
| `database_filename` | `hybrid_memory.duckdb` | DuckDB filename (in HERMES_HOME). Must be relative. |
| `graph_dirname` | `hybrid_memory_kuzu` | Kùzu graph file base name (in HERMES_HOME). |
| `local_only` | `false` | Egress gate: restrict plugin-owned LLM calls to local-only models. |
| `external_sources_require_confirmation` | `true` | Memory-safety gate: external-source candidates can never auto-activate. |
| `evidence_retention` *(JSON only)* | `full` | `full` keeps all evidence; `minimal` keeps only the grounding level. |
| `deployment_mode` | `cloud_pilot` | `cloud_pilot` or `local_sku`. Must be consistent with `data_residency`. |
| `data_residency` | `cloud` | `cloud` or `local`. Must be consistent with `deployment_mode`. |
| `acl` *(JSON only)* | *(empty)* | Optional ACL config dict for per-user access scoping. |

## Embeddings

| Setting | Default | Description |
|---------|---------|-------------|
| `local_embedding_model` | `BAAI/bge-small-en-v1.5` | Sentence-transformers model (~130MB, offline). Falls back to text search if it fails to load. |
| `freshness_markers` | `true` | Append as-of date markers to injected memories with date anchors. |
| `reranker_enabled` | `false` | Cross-encoder re-ranking. ~8s/query on CPU — needs CUDA torch. Experimental. |
| `reranker_model` | `BAAI/bge-reranker-base` | HuggingFace reranker model (~420MB). |
| `reranker_top_n` | `10` | Number of top candidates to re-rank (5–100). |

## Retrieval

| Setting | Default | Description |
|---------|---------|-------------|
| `max_injected_items` | `20` | Max memories auto-injected as context before each turn. |
| `inject_content_char_cap` | `800` | Per-item char cap in injected context. |
| `skip_retrieval_on_trivial` | `true` | Skip retrieval for trivial/filler turns. |
| `injection_min_score` | `0.30` | Relevance floor for injected items. |
| `context_aware_retrieval` | `true` | Prepend recent conversation context to queries with pronouns/references. |
| `context_window_size` | `3` | Recent user messages used as context (1–10). |
| `context_max_chars` | `500` | Max total chars of context to prepend (100–2000). |
| `query_expansion_enabled` | `true` | LLM rewrites weak queries into sub-queries. Fail-soft, cached 1h. |
| `query_expansion_similarity_floor` | `0.3` | Trigger expansion when top hit similarity is below this. |
| `phrase_lift_alpha` *(JSON only)* | `0.0` | Exact-phrase lift strength. No-op at default. |
| `phrase_lift_pool` *(JSON only)* | `200` | Candidate pool scanned for phrase lift. |
| `chronological_injection` | `true` | Chronological re-sort of injected items on temporal turns. |
| `date_anchor_rerank` | `true` | Date-expression re-ranking for temporal queries. |
| `history_at_current_time` | `true` | Widen retrieval to superseded versions on historical queries. |
| `conflict_surfacing` | `true` | Inject an explicit conflict note when two active records disagree. |

## Graph

| Setting | Default | Description |
|---------|---------|-------------|
| `graph_aware_retrieval` | `true` | Boost memories supported by graph entities during normal search. |
| `graph_retrieval_boost` | `0.0` | Max similarity boost for graph-supported memories (0.0–0.5). |
| `graph_boost_min_similarity` | `0.15` | Minimum semantic similarity for a memory to receive the graph boost. |
| `graph_inject_candidates` | `false` | Inject memories found only by the graph. Off by default (adds noise). |
| `graph_traversal_enabled` | `true` | Enable graph-traversal boost for multi-hop retrieval. **Unproven:** the only A/B (#139) ran on a regex-built graph where traversal never engaged, so there is no evidence for or against it (#364). Reported `false` in the live deployment config (#364). |
| `graph_traversal_depth` | `2` | Traversal depth for graph-traversal boost. |
| `graph_traversal_boost` | `0.60` | Boost strength for graph-traversal candidates. |
| `alias_expansion_boost` | `0.7` | Similarity floor for alias-expanded candidates. |
| `graph_ppr_enabled` *(JSON only)* | `false` | Enable Personalized PageRank graph search. Experimental. |
| `graph_ppr_damping` *(JSON only)* | `0.5` | PPR damping factor (0.0–1.0). |
| `graph_ppr_boost` *(JSON only)* | `0.0` | PPR boost strength (0.0–1.0). No-op at default. |
| `entity_aliases` | *(empty)* | JSON mapping of aliases → canonical entity names. |
| `role_words` | *(empty)* | Extra role words for alias extraction. JSON array of strings. |

## Chains (version evolution)

| Setting | Default | Description |
|---------|---------|-------------|
| `chain_unfold` | `auto` | Auto-inject a compact version arc on change-intent queries. `off`, `auto`, or `always`. |
| `chain_unfold_min_similarity` | `0.30` | Per-candidate similarity floor for unfold. |
| `chain_unfold_arc_min_similarity` *(JSON only)* | `0.15` | Semantic-arc floor for the unfolded chain. |
| `chain_unfold_top_k` | `3` | How many top results to scan for a chain anchor (1–20). |
| `chain_unfold_query_fallback` | `false` | Search deeper for a chain matching the query. |
| `chain_max_versions` | `3` | Max versions to inject per chain unfold (1–10). |
| `chain_max_inject` | `150` | Soft token cap per chain injection. |

## Extraction

See [`CONFIG_REFERENCE.md`](https://github.com/bobaba76/Argos/blob/master/CONFIG_REFERENCE.md) for the full extraction, dedup, expiry, LLM, maintenance, lifecycle, watcher, backup, scale, distillation, and router sections — they're kept in the repo file to avoid drift between this site and the source of truth.

## Representative config

```json
{
  "storage_mode": "shared_service",
  "max_injected_items": "20",
  "local_embedding_model": "BAAI/bge-small-en-v1.5",
  "auto_extract": "true",
  "llm_fallback": "true",
  "auto_review": "true",
  "graph_aware_retrieval": "true",
  "graph_retrieval_boost": "0.0",
  "alias_expansion_boost": "0.7",
  "context_aware_retrieval": "true",
  "query_expansion_enabled": "true",
  "query_expansion_similarity_floor": "0.3",
  "chain_unfold": "auto",
  "chain_unfold_min_similarity": "0.30",
  "chain_unfold_top_k": "3",
  "chain_unfold_query_fallback": "false",
  "chain_max_versions": "3",
  "chain_max_inject": "150",
  "reranker_enabled": "false",
  "consolidation_enabled": "false"
}
```

## Next steps

- [Tuning](tuning.md) — embedder and reranker guidance.
- [API reference](api/index.md) — MCP and REST endpoints.

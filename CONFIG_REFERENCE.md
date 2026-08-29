# Argos configuration reference

Full settings tables for `hybrid_memory.json` in the Hermes home directory. Defaults below are the **runtime defaults** from `_load_config()` — what runs when no config file is present. The settings UI reads the same file; what you see is what's live.

A few knobs are not surfaced in the UI yet — edit the JSON directly. Those are marked *(JSON only)*.

## Storage

| Setting | Default | Description |
|---------|---------|-------------|
| `storage_mode` | `shared_service` | `shared_service` (RPC service owns the DB, multi-process safe) or `direct` (plugin opens DuckDB directly, single-process; diagnostics only). |
| `database_filename` | `hybrid_memory.duckdb` | DuckDB filename (in HERMES_HOME). |
| `graph_dirname` | `hybrid_memory_kuzu` | Kùzu graph file base name (in HERMES_HOME). A single file, not a directory; a `.wal` sibling exists while the service holds the graph. |
| `local_only` | `false` | Egress gate: restrict plugin-owned LLM calls (extraction, review, distillation, router sub-calls) to local-only models. |
| `external_sources_require_confirmation` | `true` | Memory-safety gate: candidates tagged `external_source` (email/web/import) can never auto-activate — the reviewer short-circuits to `pending_user_confirmation` (no LLM call) and the store's `auto_review` transition is downgraded at the storage boundary. A human confirmation or a `manual`/`tool` review is required. Inbound scanning of external evidence runs regardless and always routes blocked content to pending. |

## Embeddings

| Setting | Default | Description |
|---------|---------|-------------|
| `local_embedding_model` | `BAAI/bge-small-en-v1.5` | Sentence-transformers model (~130MB, offline). Falls back to text search if it fails to load. |
| `reranker_enabled` | `false` | Cross-encoder re-ranking of top candidates. ~8s/query on CPU — needs CUDA torch to be production-viable. Experimental. |
| `reranker_model` | `BAAI/bge-reranker-base` | HuggingFace reranker model (~420MB, downloaded on first use). |
| `reranker_top_n` | `10` | Number of top candidates to re-rank (5–100). |

## Retrieval

| Setting | Default | Description |
|---------|---------|-------------|
| `max_injected_items` | `20` | Max memories auto-injected as context before each turn. |
| `inject_content_char_cap` | `800` | Per-item char cap in injected context. |
| `injection_min_score` *(JSON only)* | `0.0` | Relevance floor for injected items — items below this are dropped. |
| `skip_retrieval_on_trivial` *(JSON only)* | `false` | Skip retrieval for trivial/filler turns. Cost lever. |
| `context_aware_retrieval` | `true` | Prepend recent conversation context to queries with pronouns/references. Zero latency, no LLM calls. |
| `context_window_size` | `3` | Recent user messages used as context (1–10). |
| `context_max_chars` | `500` | Max total chars of context to prepend (100–2000). |
| `query_expansion_enabled` | `true` | LLM rewrites weak queries into sub-queries for better recall. Fires only when top hit is below the similarity floor; cached 1h; fail-soft. |
| `query_expansion_similarity_floor` | `0.3` | Trigger expansion when top hit similarity is below this (0.0–1.0). |
| `phrase_lift_alpha` *(JSON only)* | `0.0` | Exact-phrase lift strength in ranking. No-op at default. |
| `phrase_lift_pool` *(JSON only)* | `200` | Candidate pool scanned for phrase lift. |
| `chronological_injection` | `false` | Chronological re-sort of injected items. |
| `date_anchor_rerank` | `false` | Date-expression re-ranking for temporal queries. |
| `history_at_current_time` | `true` | Widen retrieval to superseded versions on historical queries; injected with a "(previously)" label. |

## Graph

| Setting | Default | Description |
|---------|---------|-------------|
| `graph_aware_retrieval` | `true` | Boost memories supported by graph entities during normal search. |
| `graph_retrieval_boost` | `0.0` | Max similarity boost for graph-supported memories (0.0–0.5). No-op at default. |
| `graph_boost_min_similarity` | `0.15` | Minimum semantic similarity for a memory to receive the graph boost. |
| `graph_inject_candidates` | `false` | Inject memories found *only* by the graph (not semantic search). Off by default — adds noise that hurt recall in benchmarks. |
| `graph_traversal_enabled` | `true` | Enable graph-traversal boost for multi-hop retrieval. |
| `graph_traversal_depth` | `2` | Traversal depth for graph-traversal boost. |
| `graph_traversal_boost` | `0.60` | Boost strength for graph-traversal candidates. |
| `alias_expansion_boost` | `0.7` | Similarity floor for alias-expanded candidates. |
| `entity_aliases` | *(empty)* | JSON mapping of aliases → canonical entity names, e.g. `{"my role": "Entity-A"}`. |
| `role_words` | *(empty)* | Extra role words for "my X is Name" alias extraction (comma-separated or JSON array). 40+ defaults built in; LLM-learned words auto-added. |

## Chains (version evolution)

| Setting | Default | Description |
|---------|---------|-------------|
| `chain_unfold` | `auto` | Auto-inject a compact version arc on change-intent queries. `off` (on-demand only), `auto` (change-intent trigger), or `always` (every chained result). |
| `chain_unfold_min_similarity` | `0.30` | Per-candidate similarity floor for unfold (the precision guard). |
| `chain_unfold_top_k` | `3` | How many top results to scan for a chain anchor (1–20). |
| `chain_unfold_query_fallback` | `false` | Search deeper for a chain matching the query if no top-K result has one. |
| `chain_max_versions` | `3` | Max versions to inject per chain unfold (1–10). |
| `chain_max_inject` | `150` | Soft token cap per chain injection (~4 chars/token estimate). |

## Extraction

| Setting | Default | Description |
|---------|---------|-------------|
| `auto_extract` | `true` | Extract memory proposals from each conversation turn. |
| `llm_fallback` | `true` | Use the host LLM to create proposals when regex misses. Adds latency + token cost; proposals stay pending until reviewed. |
| `extraction_shadow_diff` | `false` | Run LLM extraction in parallel with regex and log the diff (validation mode; no proposal changes). |
| `auto_review` | `true` | Auxiliary LLM reviews new proposals; obvious junk quarantined, sensitive ones stay pending. |
| `stale_review_sweep_enabled` | `true` | Periodically re-review proposals pending too long. |
| `stale_review_interval_min` | `15` | Sweep interval in minutes. |
| `stale_review_min_age_min` | `30` | Min proposal age (minutes) before it's eligible for the stale sweep. |
| `stale_review_max_batch` | `25` | Max proposals re-reviewed per sweep. |
| `role_alias_llm_fallback` | `true` | Ask the LLM if an unknown "my X is Name" word is a person-role; learned words persist to `role_words`. |
| `extraction_dup_threshold` | *(from schema)* | Dedup threshold for extraction proposals. |

## Dedup

| Setting | Default | Description |
|---------|---------|-------------|
| `duplicate_min_similarity` | `0.88` | Embedding similarity threshold for semantic dedup during consolidation. |
| `duplicate_semantic_max_pairs` | `20000` | Max pairs compared in semantic dedup pass. |

## Expiry / TTL

| Setting | Default | Description |
|---------|---------|-------------|
| `expiry_enabled` | `false` | Enable time-based expiry of memories by category. |
| `expiry_ttl_days` | `{"context_note":30,"event":180,"goal":180}` | Per-category TTL in days. Categories not listed use `expiry_default_days`. |
| `expiry_default_days` | `90` | Default TTL for categories not in `expiry_ttl_days`. |
| `expiry_auto_suggest` | `false` | Suggest expiry for memories that look time-limited. |

## LLM

| Setting | Default | Description |
|---------|---------|-------------|
| `llm_model` | *(empty)* | Model ID for auxiliary LLM tasks (extraction, review, query expansion). Empty = host auxiliary default. |
| `llm_provider` | *(empty)* | Provider override for auxiliary tasks (`openrouter`, `openai`, `anthropic`, …). Empty = host default. |

## Maintenance

| Setting | Default | Description |
|---------|---------|-------------|
| `consolidation_enabled` | `false` | Reversible maintenance of expired/duplicate memories at session end. Preview with `memory_maintenance` `dry_run=true` first. |
| `consolidation_min_age_days` | `30` | Age threshold for stale temporary memories. |
| `consolidation_max_actions` | `25` | Max records to quarantine per run. |

## Distillation (the dream)

| Setting | Default | Description |
|---------|---------|-------------|
| `distillation_enabled` | `false` | Enable the gated LLM distillation pass at session boundaries (session end + chat rotation). Proposals only — nothing enters active memory without review. |
| `distillation_min_new_records` | `20` | Novelty gate: minimum new/updated records since the last run to fire. |
| `distillation_cooldown_hours` | `24` | Cooldown gate: minimum hours between runs. |
| `distillation_max_records_per_run` | `100` | Budget: max records considered per run. |
| `distillation_max_calls` | `10` | Budget: max LLM calls per run (1 per cluster + 1 feedback scan). |

> The distillation pass reuses the auxiliary LLM (`llm_model`/`llm_provider`
> above). To route it to a cheaper model without touching the plugin, set
> `auxiliary.distillation.model` in the host's config — the auxiliary client
> picks up per-task overrides automatically.

## Router (answerer routing)

Routes temporal/multi-hop queries to a smarter model; trivial turns stay on the cheap default. The router deliberately returns None for non-smart turns so it never stomps a user-selected session model.

| Setting | Code default | Description |
|---------|-------------|-------------|
| `router_enabled` | `false` | Enable answerer routing. |
| `router_subcall_enabled` | `false` | One trimmed sub-call injects a dated-memory hint before routing (fail-soft). |
| `router_default_model` | *(empty)* | Cheap default answerer model. |
| `router_default_provider` | *(empty)* | Cheap default answerer provider. |
| `router_smart_model` | *(empty)* | Smart answerer for routed queries. |
| `router_smart_provider` | *(empty)* | Smart answerer provider. |

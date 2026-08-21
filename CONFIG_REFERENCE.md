# Argos configuration reference

This file holds the full settings tables (moved out of the README so the README can stay readable).

## Configuration

All settings live in `hybrid_memory.json` in the Hermes home directory. The
defaults below are the **runtime defaults** from `_load_config()` (what runs
when no config file is present). The desktop settings UI (`config_schema.py`)
shows its own display defaults in a couple of cases — those are footnoted.

> The settings panel reads the *same* `hybrid_memory.json` the running
> service uses — what you see in the UI is what's live (no separate,
> diverging store).

### Storage

| Setting | Default | Description |
|---------|---------|-------------|
| `storage_mode` | `shared_service` | `shared_service` (RPC service owns the DB, multi-process safe) or `direct` (plugin opens DuckDB directly, single-process; diagnostics only). |
| `database_filename` | `hybrid_memory.duckdb` | DuckDB filename (in HERMES_HOME). |
| `graph_dirname` | `hybrid_memory_kuzu` | Kùzu graph directory (in HERMES_HOME). |

### Embeddings

| Setting | Default | Description |
|---------|---------|-------------|
| `local_embedding_model` | `BAAI/bge-small-en-v1.5` | Sentence-transformers model (~130MB, offline). Falls back to text search if it fails to load. |
| `reranker_enabled` | `false` | Cross-encoder re-ranking of top candidates (~300ms/query on CPU). Experimental — evaluate before relying on it. |
| `reranker_model` | `BAAI/bge-reranker-base` | HuggingFace reranker model (~420MB, downloaded on first use). |
| `reranker_top_n` | `10` | Number of top candidates to re-rank (5–100). |

### Retrieval

| Setting | Default | Description |
|---------|---------|-------------|
| `max_injected_items` | `20` ¹ | Max memories auto-injected as context before each turn. |
| `context_aware_retrieval` | `true` | Prepend recent conversation context to queries with pronouns/references ("that", "he", "the thing"). Zero latency, no LLM calls. |
| `context_window_size` | `3` | Recent user messages used as context (1–10). |
| `context_max_chars` | `500` | Max total chars of context to prepend (100–2000). |
| `query_expansion_enabled` | `true` | LLM rewrites weak queries into sub-queries for better recall. Fires only when top hit is below the similarity floor; cached 1h; fail-soft. |
| `query_expansion_similarity_floor` | `0.3` | Trigger expansion when top hit similarity is below this (0.0–1.0). |

### Graph

| Setting | Default | Description |
|---------|---------|-------------|
| `graph_aware_retrieval` | `true` | Boost memories supported by graph entities during normal search. |
| `graph_retrieval_boost` | `0.0` ² | Max similarity boost for graph-supported memories (0.0–0.5). |
| `graph_boost_min_similarity` | `0.15` | Minimum semantic similarity for a memory to receive the graph boost. |
| `graph_inject_candidates` | `false` | Inject memories found *only* by the graph (not semantic search). Off by default — adds noise that hurt recall in benchmarks. |
| `graph_traversal_enabled` | `true` | Enable graph-traversal boost for multi-hop retrieval. |
| `graph_traversal_depth` | `2` | Traversal depth for graph-traversal boost. |
| `graph_traversal_boost` | `0.60` | Boost strength for graph-traversal candidates. |
| `alias_expansion_boost` | `0.7` | Similarity floor for alias-expanded candidates. |
| `entity_aliases` | *(empty)* | JSON mapping of aliases → canonical entity names, e.g. `{"my role": "Entity-A"}`. |
| `role_words` | *(empty)* | Extra role words for "my X is Name" alias extraction (comma-separated or JSON array). 40+ defaults built in; LLM-learned words auto-added. |

### Chains (version evolution)

| Setting | Default | Description |
|---------|---------|-------------|
| `chain_unfold` | `auto` ³ | Auto-inject a compact version arc on change-intent queries. `off` (on-demand only), `auto` (change-intent trigger), or `always` (every chained result). |
| `chain_unfold_min_similarity` | `0.30` | Per-candidate similarity floor for unfold (the precision guard). |
| `chain_unfold_top_k` | `3` | How many top results to scan for a chain anchor (1–20). |
| `chain_unfold_query_fallback` | `false` | Search deeper for a chain matching the query if no top-K result has one. |
| `chain_max_versions` | `3` | Max versions to inject per chain unfold (1–10). |
| `chain_max_inject` | `150` | Soft token cap per chain injection (~4 chars/token estimate). |

### Extraction

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

### LLM

| Setting | Default | Description |
|---------|---------|-------------|
| `llm_model` | *(empty)* | Model ID for auxiliary LLM tasks (extraction, review, query expansion). Empty = host auxiliary default. |
| `llm_provider` | *(empty)* | Provider override for auxiliary tasks (`openrouter`, `openai`, `anthropic`, …). Empty = host default. |

### Maintenance

| Setting | Default | Description |
|---------|---------|-------------|
| `consolidation_enabled` | `false` | Reversible maintenance of expired/duplicate memories at session end. Preview with `memory_maintenance` `dry_run=true` first. |
| `consolidation_min_age_days` | `30` | Age threshold for stale temporary memories. |
| `consolidation_max_actions` | `25` | Max records to quarantine per run. |

> ¹ The desktop UI schema shows `8` as its display default; the runtime
> default is `20`. Your saved `hybrid_memory.json` wins either way.
> ² The desktop UI schema shows `0.05`; the runtime default is `0.0`.
> ³ The desktop UI schema shows `off` ("ships OFF"); the runtime default
> is `auto`. The `memory_chain` tool is always available on-demand
> regardless of this setting.

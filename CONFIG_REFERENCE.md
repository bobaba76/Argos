# Argos configuration reference

Full settings tables for `hybrid_memory.json` in the Hermes home directory. Defaults below are the **runtime defaults** from `_load_config()` — what runs when no config file is present. The settings UI reads the same file; what you see is what's live.

A few knobs are not surfaced in the UI yet — edit the JSON directly. Those are marked *(JSON only)*.

## Storage

| Setting | Default | Description |
|---------|---------|-------------|
| `storage_mode` | `shared_service` | `shared_service` (RPC service owns the DB, multi-process safe) or `direct` (plugin opens DuckDB directly, single-process; diagnostics only). **When to change:** use `direct` only for diagnostics or single-process testing; production should always use `shared_service`. |
| `database_filename` | `hybrid_memory.duckdb` | DuckDB filename (in HERMES_HOME). Must be relative — no absolute path, drive letter, UNC prefix, or `..`; unsafe values fall back to the default. **When to change:** only if you need a separate DB for testing or a different tenant. |
| `graph_dirname` | `hybrid_memory_kuzu` | Kùzu graph file base name (in HERMES_HOME). A single file, not a directory; a `.wal` sibling exists while the service holds the graph. Same path rules as `database_filename`. **When to change:** only if you need a separate graph for testing. |
| `local_only` | `false` | Egress gate: restrict plugin-owned LLM calls (extraction, review, distillation, router sub-calls) to local-only models. **When to change:** set to `true` for air-gapped or privacy-critical deployments. |
| `external_sources_require_confirmation` | `true` | Memory-safety gate: candidates tagged `external_source` (email/web/import) can never auto-activate — the reviewer short-circuits to `pending_user_confirmation` (no LLM call) and the store's `auto_review` transition is downgraded at the storage boundary. A human confirmation or a `manual`/`tool` review is required. Inbound scanning of external evidence runs regardless and always routes blocked content to pending. **When to change:** rarely — this is a safety gate, not a tuning knob. |
| `evidence_retention` *(JSON only)* | `full` | Controls how much provenance/evidence metadata is retained per memory record. `full` keeps all evidence (quotes, source URLs, timestamps); `minimal` keeps only the grounding level. **When to change:** set to `minimal` for privacy-constrained deployments where provenance detail is not needed. |
| `deployment_mode` | `cloud_pilot` | Deployment mode for POPIA compliance: `cloud_pilot` (default, cloud-hosted pilot) or `local_sku` (local-only SKU). **When to change:** switch to `local_sku` for fully local deployments. Must be consistent with `data_residency`. |
| `data_residency` | `cloud` | Data residency mode: `cloud` (data may leave the local machine for LLM calls) or `local` (all data stays local). **When to change:** set to `local` for air-gapped or data-sovereignty requirements. Must be consistent with `deployment_mode`. |
| `acl` *(JSON only)* | *(empty)* | Optional ACL config dict for per-user access scoping. Format: `{"users": {"alice": {"scopes": ["personal"]}}}`. When absent, the store is open (backward compatible). **When to change:** when deploying multi-tenant with per-user access restrictions. |

## Embeddings

| Setting | Default | Description |
|---------|---------|-------------|
| `local_embedding_model` | `BAAI/bge-small-en-v1.5` | Sentence-transformers model (~130MB, offline). Falls back to text search if it fails to load. **When to change:** to use a different embedding model (must be sentence-transformers compatible). |
| `freshness_markers` | `true` | Append an as-of date marker to injected memories whose content carries a date anchor. Append-only text; ranking and retrieval untouched. **When to change:** set to `false` if the markers add noise to your injected context. |
| `reranker_enabled` | `false` | Cross-encoder re-ranking of top candidates. ~8s/query on CPU — needs CUDA torch to be production-viable. Experimental. **When to change:** enable only with CUDA torch and when precision matters more than latency. |
| `reranker_model` | `BAAI/bge-reranker-base` | HuggingFace reranker model (~420MB, downloaded on first use). **When to change:** to use a different cross-encoder model. |
| `reranker_top_n` | `10` | Number of top candidates to re-rank (5–100). **When to change:** increase for deeper re-ranking (higher latency); decrease for speed. |

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
| `history_at_current_time` | `true` | Widen retrieval to superseded versions on historical queries; injected with a "(previously)" label. **When to change:** set to `false` if you only want current-version results. |
| `conflict_surfacing` | `true` | When the injected set contains two active records that conflict on the same subject (differing values, or one asserting a rule vs a later discontinuation/scoping), inject an explicit conflict note so the answerer surfaces the disagreement instead of smoothing it. **When to change:** set to `false` if conflict notes add noise; keep `true` for trust-critical use cases. |

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
| `alias_expansion_boost` | `0.7` | Similarity floor for alias-expanded candidates. **When to change:** lower if alias-expanded results are noisy; raise if valid aliases are being dropped. |
| `graph_ppr_enabled` *(JSON only)* | `false` | Enable Personalized PageRank graph search for multi-hop retrieval. Experimental. **When to change:** enable for graph-heavy use cases where traversal boost is insufficient. |
| `graph_ppr_damping` *(JSON only)* | `0.5` | PPR damping factor (0.0–1.0). Controls how much PPR spreads vs. stays local. **When to change:** lower for more local focus; higher for broader spread. |
| `graph_ppr_boost` *(JSON only)* | `0.0` | PPR boost strength for graph-supported memories (0.0–1.0). No-op at default. **When to change:** increase to boost PPR-found memories in ranking. |
| `entity_aliases` | *(empty)* | JSON mapping of aliases → canonical entity names, e.g. `{"my role": "Entity-A"}`. **When to change:** add aliases when entities have multiple names in conversation. |
| `role_words` | *(empty)* | Extra role words for "my X is Name" alias extraction. Canonical format is a JSON array of strings (`["wife", "boss"]`); a comma-separated string is accepted and rewritten to the array form. 40+ defaults built in; LLM-learned words auto-added. **When to change:** add custom role words for domain-specific relationships. |

## Chains (version evolution)

| Setting | Default | Description |
|---------|---------|-------------|
| `chain_unfold` | `auto` | Auto-inject a compact version arc on change-intent queries. `off` (on-demand only), `auto` (change-intent trigger), or `always` (every chained result). |
| `chain_unfold_min_similarity` | `0.30` | Per-candidate similarity floor for unfold (the precision guard). **When to change:** raise for precision (fewer false triggers); lower for recall. |
| `chain_unfold_arc_min_similarity` *(JSON only)* | `0.15` | Semantic-arc floor: cosine(query, current-version content) that the unfolded chain must clear before the arc is injected. Precision guard — filters false triggers while keeping top-K recall. **When to change:** raise if false chain triggers are common; lower if valid chains are being filtered. |
| `chain_unfold_top_k` | `3` | How many top results to scan for a chain anchor (1–20). **When to change:** increase for broader chain discovery; decrease for speed. |
| `chain_unfold_query_fallback` | `false` | Search deeper for a chain matching the query if no top-K result has one. |
| `chain_max_versions` | `3` | Max versions to inject per chain unfold (1–10). |
| `chain_max_inject` | `150` | Soft token cap per chain injection (~4 chars/token estimate). |

## Extraction

| Setting | Default | Description |
|---------|---------|-------------|
| `auto_extract` | `true` | Extract memory proposals from each conversation turn. **When to change:** set to `false` for manual-only memory capture. |
| `llm_fallback` | `true` | Use the host LLM to create proposals when regex misses. Adds latency + token cost; proposals stay pending until reviewed. **When to change:** set to `false` for regex-only proposals (no LLM cost). |
| `extraction_shadow_diff` | `false` | Run LLM extraction in parallel with regex and log the diff (validation mode; no proposal changes). **When to change:** enable temporarily to evaluate LLM-first extraction quality. |
| `auto_review` | `true` | Auxiliary LLM reviews new proposals; obvious junk quarantined, sensitive ones stay pending. **When to change:** set to `false` for manual-only review. |
| `confirmation_surfacing` | `true` | Surface one pending memory proposal per non-trivial turn for in-session confirmation. Reviewer-failure outcomes are never surfaced; a proposal is never asked about twice (ledger survives restarts). **When to change:** set to `false` if confirmations are disruptive; keep `true` to prevent pending pileup. |
| `stale_review_sweep_enabled` | `true` | Periodically re-review proposals pending too long. **When to change:** set to `false` only if you review manually. |
| `stale_review_interval_min` | `15` | Sweep interval in minutes. **When to change:** lower for faster re-review; raise to reduce background activity. |
| `stale_review_min_age_min` | `30` | Min proposal age (minutes) before it's eligible for the stale sweep. **When to change:** lower to catch stale proposals sooner. |
| `stale_review_max_batch` | `25` | Max proposals re-reviewed per sweep. **When to change:** raise for faster backlog clearing; lower to reduce burst load. |
| `role_alias_llm_fallback` | `true` | Ask the LLM if an unknown "my X is Name" word is a person-role; learned words persist to `role_words`. **When to change:** set to `false` for regex-only role detection. |
| `extraction_dup_threshold` | `0.88` | Minimum embedding cosine similarity for a proposed fact to be skipped as already covered by an active memory (0.0–1.0). Set to 1.0 to disable semantic dedupe. **When to change:** raise to reduce duplicate proposals; lower to allow more proposals through. |
| `extraction_llm_model` | *(empty)* | Model ID for extraction-specific LLM calls. Empty = fall back to `llm_model`, then auxiliary client default. **When to change:** to use a cheaper/different model for extraction than for other tasks. |
| `extraction_llm_provider` | *(empty)* | Provider override for extraction LLM calls. Empty = fall back to `llm_provider`. **When to change:** to route extraction to a different provider. |

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
| `llm_model` | *(empty)* | Model ID for auxiliary LLM tasks (extraction, review, query expansion). Empty = host auxiliary default. **When to change:** to use a specific model for all auxiliary tasks. |
| `llm_provider` | *(empty)* | Provider override for auxiliary tasks (`openrouter`, `openai`, `anthropic`, …). Empty = host default. **When to change:** to route auxiliary tasks to a specific provider. |
| `answering_llm_model` | *(empty)* | Model ID for the answering LLM (the model that generates user-facing responses). Empty = fall back to `llm_model`, then auxiliary client default. **When to change:** to use a different model for answering than for auxiliary tasks. |
| `answering_llm_provider` | *(empty)* | Provider override for the answering LLM. Empty = fall back to `llm_provider`. **When to change:** to route answering to a different provider. |

## Maintenance

| Setting | Default | Description |
|---------|---------|-------------|
| `consolidation_enabled` | `false` | Reversible maintenance of expired/duplicate memories at session end. Preview with `memory_maintenance` `dry_run=true` first. **When to change:** enable when the store accumulates stale duplicates. |
| `consolidation_min_age_days` | `30` | Age threshold for stale temporary memories. **When to change:** lower to clean sooner; raise to keep records longer. |
| `consolidation_max_actions` | `25` | Max records to quarantine per run. **When to change:** raise for faster cleanup; lower to reduce per-run impact. |
| `consolidation_auto_apply` *(JSON only)* | `true` | Auto-apply consolidation actions without manual confirmation. When `false`, consolidation produces proposals only. **When to change:** set to `false` for manual review of all consolidation actions. |

## Lifecycle (archival, forgetting, rollups)

Three independent phases, all ship OFF by default. All are reversible (quarantine, not delete).

| Setting | Default | Description |
|---------|---------|-------------|
| `archive_enabled` | `false` | Records older than `archive_after_days` with no retrievals/feedback are tiered to `archived` (out of injection pool, searchable via `include_archived=True`). **When to change:** enable for long-running stores with stale records. |
| `archive_after_days` | `180` | Age threshold (days) for archival. **When to change:** lower to archive sooner. |
| `forget_enabled` | `false` | Auto-quarantine (reversible, never delete) of `context_note`/`event`/`goal` older than `forget_after_days`. **When to change:** enable for privacy-conscious deployments. |
| `forget_after_days` | `365` | Age threshold (days) for forgetting. **When to change:** lower to forget sooner. |
| `rollup_enabled` | `false` | Monthly LLM pass emitting profile-style proposals from accumulated records (reuses distillation seam). **When to change:** enable for long-horizon pattern discovery. |
| `rollup_interval_days` | `30` | Interval between rollup runs (days). **When to change:** lower for more frequent rollups. |
| `rollup_max_records_per_run` | `100` | Max records considered per rollup run. **When to change:** raise for larger stores. |
| `compaction_enabled` | `false` | Schedule-aware self-compaction of stale/duplicate/low-value memories. Reversible quarantine (never hard-deletes). Zero-LLM. Runs on session-end, cooldown-gated by `compaction_interval_days`. **When to change:** enable to control injection token bloat on long-running stores. |
| `compaction_interval_days` | `7` | Minimum days between compaction runs (cooldown gate). **When to change:** lower for more frequent compaction. |
| `compaction_aggressiveness` | `1.0` | Compaction aggressiveness scalar: `1.0` = conservative (fewer candidates, higher dedup threshold), `2.0` = aggressive (more candidates, lower threshold). Interpolated between. **When to change:** raise toward `2.0` if token budget is tight. |
| `compaction_auto_apply` | `false` | When `false` (default), compaction runs in dry-run (report-only) mode — candidates are identified but NOT quarantined. When `true`, compaction auto-quarantines candidates on each scheduled run. Mirrors the `consolidation_auto_apply` safety default. **When to change:** set to `true` once you've reviewed a dry-run report and are confident in the candidate selection. |

## Watcher (file-system monitoring)

| Setting | Default | Description |
|---------|---------|-------------|
| `watcher_enabled` | `false` | Enable the file-system watcher thread for document catalog, extraction, and freshness. No watcher config = zero behaviour change (thread not started). **When to change:** enable for document-aware memory (spec-07). |
| `watcher_scan_roots` | *(empty)* | List of directories to scan. JSON array of strings or comma-separated. **When to change:** add directories to monitor. |
| `watcher_interval_min` | `30` | Scan interval in minutes. **When to change:** lower for faster detection; raise to reduce I/O. |

## Backup (service-coordinated)

| Setting | Default | Description |
|---------|---------|-------------|
| `backup_enabled` | `false` | Enable service-coordinated backups. The backup config is a nested dict in the live JSON. **When to change:** enable for production deployments. |
| `backup_dst_root` | *(empty)* | Destination root directory for backups. **When to change:** set to a local or network backup path. |
| `backup_retention_snapshots` | `6` | Number of backup snapshots to retain (1–100). **When to change:** raise for longer retention. |
| `backup` *(JSON only)* | *(empty)* | Nested dict for backup configuration: `{"dst_root": "/path", "retention_snapshots": 10}`. Scalar keys above are kept as model fields for schema parity. **When to change:** set the nested dict for full backup config. |

## Scale triggers

| Setting | Default | Description |
|---------|---------|-------------|
| `scale_warn_latency_ms` *(JSON only)* | `300.0` | Latency warning threshold in milliseconds. When search latency exceeds this, a scale warning is logged. **When to change:** lower for stricter latency monitoring. |
| `scale_warn_records` *(JSON only)* | `5000` | Record count warning threshold. When the store exceeds this count, a scale warning is logged. **When to change:** raise for larger stores. |

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
| `router_temporal_threshold` *(JSON only)* | `0.5` | Similarity threshold for routing temporal queries to the smart model (0.1–1.0). Higher = fewer queries routed. **When to change:** lower to route more temporal queries; raise for precision. |
| `router_multihop_threshold` *(JSON only)* | `0.5` | Similarity threshold for routing multi-hop queries to the smart model (0.1–1.0). Higher = fewer queries routed. **When to change:** lower to route more multi-hop queries; raise for precision. |

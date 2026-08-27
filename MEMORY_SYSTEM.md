# Hybrid Memory System

Deep dive into how Argos works. For installation see [SETUP_GUIDE.md](SETUP_GUIDE.md); for the front-door overview see [README.md](README.md).

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  Provider Layer                          │
│   16 memory_* tools + pre_llm_call ambient hook +       │
│    insight-log skill + /ilog, /revisit, /neg commands   │
├─────────────────────────────────────────────────────────┤
│                  Retrieval Pipeline                      │
│   prefetch → context-aware enrichment →                 │
│   store: text+vector search → RRF fusion →              │
│     optional reranker → phrase lift →                   │
│     feedback/recency → truncate →                       │
│   provider: query expansion → alias expansion →         │
│     graph boost → graph traversal →                     │
│   chain annotation + unfold (memory_search tool only)   │
├──────────────────┬──────────────────────────────────────┤
│   DuckDB Store   │         Kùzu Graph                   │
│  (memory records,│   (entities, relations, aliases,      │
│   embeddings,    │    memory-to-entity links,            │
│   version chains,│    graph-traversal boost)             │
│   candidates,    │                                      │
│   feedback)      │                                      │
└──────────────────┴──────────────────────────────────────┘
```

## Tools

See [README.md](README.md) for the grouped tool table. All 16 tools are agent-callable; the two graph tools are always in the schema even before the store initializes (they return a clear error at call time if the store isn't ready).

## Core concepts

### Memory records

Each memory is a typed record with a category, content text, tags, embedding vector, and metadata. Categories: `personal_fact`, `context_note`, `insight`, `event`, `relationship`, `goal`, `preference`, `negative`. Records are scoped by user and optionally by project (`project_id` filter on search).

### Hybrid search

The store-level search (`store._hybrid_search`):

1. **Text search** — ILIKE keyword matching for exact-term hits.
2. **Vector search** — cosine similarity between query and memory embeddings (BGE-small-en-v1.5, runs fully offline/local).
3. **RRF fusion** — merges the two ranked lists into one.
4. **Cross-encoder re-ranking** (optional, `reranker_enabled`) — re-ranks top-N with BGE-reranker-base (~420MB; ~8s/query on CPU — needs CUDA torch to be production-viable). Blends CE scores 20/80 with similarity. Experimental.
5. **Phrase lift** (`phrase_lift_alpha > 0`) — exact-phrase lift in ranking. No-op at default (0.0).
6. **Feedback + recency** — recency, retrieval frequency, and helpful/dismissed feedback adjust the final ranking.
7. **P2C** (`_P2C_ENABLED`) — demote older of a near-duplicate pair. Disabled at default; no config key.

Provider-level, after the store returns results:

8. **Query expansion** — if top hit similarity is below `query_expansion_similarity_floor`, the LLM rewrites the query into sub-queries and re-searches. Cached 1h, fail-soft. Runs *after* the initial store read, not before.
9. **Alias expansion** — bidirectional: querying an alias finds the canonical entity's memories, and vice versa.
10. **Graph boost** (`graph_aware_retrieval`) — bumps similarity score of memories supported by graph entities. No-op at default (`graph_retrieval_boost=0.0`).
11. **Graph traversal** (`graph_traversal_enabled`) — multi-hop boost for candidates reached by traversing the graph from query entities.

**Chain annotation and chain-unfold run only in the `memory_search` tool path**, not in the ambient per-turn injection. The injected context does not carry chain info.

### Knowledge graph

Memories are indexed into a Kùzu graph at save time:

- **Entity nodes** — people, places, concepts, organizations extracted from memory content.
- **Typed relations** — `married_to`, `works_at`, `uses`, `has_attribute`, etc.
- **Alias mappings** — "my role" → "Entity-A" so a query for either finds memories mentioning the other.
- **Memory-to-entity links** — each memory node links to the entities it mentions.

Alias resolution is bidirectional: querying an alias finds the canonical entity's memories, and querying the canonical name also finds memories that use only the alias.

### Memory evolution (version chains)

When a memory is updated, the old version is superseded (not deleted). The store tracks `valid_from`, `valid_to`, and `superseded_by` fields, forming a chronological chain of versions for each fact.

- **`memory_chain` tool** — retrieves the full version arc for any memory ID (modes: `arc`, `versions`, `diff`).
- **Chain annotation** — search results carry a `chain` field indicating whether a memory has version history.
- **Chain-unfold** — when `chain_unfold=auto`, a change-intent query ("why did I switch...", "what changed...") triggers automatic injection of a compact version arc into the search results. Only runs in the `memory_search` tool path.
- **Head-deletion promotion** — deleting the current version promotes the predecessor to current (and re-indexes it in the graph).
- **History-at-current-time** (`history_at_current_time=true`) — widens retrieval to superseded versions on historical queries; injected with a "(previously)" label.

### Date-anchored retrieval

Time expressions ("two weeks ago", "last Friday", "that Valentine's") are extracted by regex, resolved relative to the question date, and used to re-rank the result slice by recency. Enabled via `date_anchor_rerank`.

### Candidate review

New memories go through a dedup + review pipeline:

- **Semantic dedup** — if a new memory is embedding-similar to an existing one, it's flagged rather than stored blindly.
- **Approve-with-supersede** — when a new memory contradicts/replaces an existing one (same category, high similarity), the reviewer can chain it behind the existing record as a new version.

### Auto-extraction pipeline

Every turn (`sync_turn`) is processed by a background worker:

1. **Regex extraction** — pattern-based fact detection (fast, zero cost).
2. **LLM fallback** (`llm_fallback=true`) — when regex misses, the auxiliary LLM proposes facts. Adds latency + token cost; proposals stay pending.
3. **Shadow-diff** (`extraction_shadow_diff=true`) — runs LLM extraction alongside regex and logs what each found that the other missed. Validation mode only.
4. **Auto-review** (`auto_review=true`) — the auxiliary LLM reviews each new proposal: obvious junk quarantined, sensitive/contextless proposals stay pending, clear facts approved.
5. **Stale-review sweep** (`stale_review_sweep_enabled=true`) — periodically re-reviews proposals pending too long.
6. **Role-word learning** (`role_alias_llm_fallback=true`) — when an unknown word appears in "my X is Name", the LLM is asked if X is a person-role; learned words persist to `role_words`.

Proposals are never active memory until reviewed. The agent can also save explicitly via `memory_save`, bypassing the proposal queue.

### Maintenance and quarantine

- **`memory_maintenance`** — previews (`dry_run=true`, default) or applies reversible quarantine of stale temporary memories and low-quality duplicates. Never hard-deletes.
- **`consolidation_enabled`** — runs the same maintenance automatically at session end.
- **`memory_restore`** — brings a quarantined memory back to active retrieval (and re-indexes it in the graph).
- **`memory_feedback`** with `incorrect` — detaches the memory from the graph.
- **Junk-entity purge** — at session end, the graph purges orphaned junk entities.
- **Semantic merge** — `consolidate()`'s duplicate leg uses embedding-similarity merging: records within a high similarity threshold of an existing active record are consolidated (newest wins, older version chained or appended deterministically). Reversible, never hard-deletes.

### Expiry / TTL

When `expiry_enabled=true`, memories expire by category based on `expiry_ttl_days` (default: `context_note`=30, `event`=180, `goal`=180; other categories use `expiry_default_days`=90). `expiry_auto_suggest` suggests expiry for memories that look time-limited. Disabled by default.

### Distillation pass (the dream)

A bounded, LLM-assisted consolidation pass that turns accumulated records + feedback into *proposed* insights, guardrails, and contradiction warnings. Disabled by default (`distillation_enabled: false`).

- **Trigger** — runs at session boundaries: `on_session_end` and `on_session_switch` (every chat rotation). Both hooks call the same self-gating method.
- **Egress gate** — all LLM calls respect the `local_only` egress gate; if enabled, distillation is skipped when no local model is available.
- **Gates (all store-side, before any LLM call)** — LLM client availability; novelty: ≥ 20 new or updated records since last run; cooldown: ≥ 24h; budget: ≤ 100 records and ≤ 10 LLM calls per run.
- **Cluster scan** (free, deterministic) — seed-star greedy grouping: the newest record seeds a cluster, members are records with cosine ≥ 0.75 *to the seed only* (no transitive chaining), capped at 8 per cluster.
- **LLM distill** — one call per cluster with a strict JSON prompt: insights, contradictions (`a_id`/`b_id`/`reason`), and guardrails. Contradictions are honored only for IDs the model was actually shown. One additional call scans high-feedback records for lessons.
- **Proposals only** — every output is saved via `save_candidate()` (`source="distillation"`, `dedup=True`) with grounding (`evidence_text` + `payload.sources`). Pending until the auto-review pipeline and the user approve them. The pass never writes, edits, or deletes active memory.
- **Run state** — values live in a `system_state` KV table (`distillation_last_run`, `distillation_last_count`). Advances only on *completed* runs; fail-soft aborts leave it unadvanced so the next clean boundary retries.
- **Fail-soft throughout** — LLM error, bad JSON, or client import failure skips the affected leg; session lifecycle is never blocked.

### Ambient context

A `pre_llm_call` hook runs on every turn and injects short, fail-soft hints into the user-message context (via the native `plugin_user_context` path — no core source patch). Each hint is built independently so a failure in one never suppresses the others:

- **Time** — `Current time: Friday 2026-07-31 19:55 SAST`, via `hermes_time` (respects `HERMES_TIMEZONE` → `config.yaml` `timezone` → server-local).
- **Location** — `Location: City-X`, resolved from `HERMES_LOCATION` → `config.yaml` `location`.
- **Weather** — `Weather: 14°C, light rain`, via `hermes_weather` (geocodes location, fetches Open-Meteo; cached 20 min; free, no API key).
- **File activity** — `Last edited: ~/project/foo.py (4 min ago)`, via `hermes_file_activity` (scans the working directory; cached 5 min).

A conditional coder-MCP directive is also injected when the turn looks code-adjacent, steering the agent toward `mcp__coder__*` tools. Non-coding turns cost nothing.

### Insight log

The `insight-log` skill captures personal realizations verbatim (no sanitizing, no judgment) when the user shares a "I just realised…" moment, saving them as `insight`-category memories. In future sessions it proactively resurfaces relevant insights when the topic overlaps.

Three slash commands (registered by the plugin):

- **`/ilog [tag]`** — list saved insights, newest first (up to 20). Optional tag filter.
- **`/revisit`** — surface a random older insight (≥3 days old when available).
- **`/neg <claim>`** — store an explicit exclusion (e.g. `/neg I do not drink coffee`), injected with a `[negative]` label so the model answers grounded nos instead of guessing.

### Prefetch

At `on_turn_start`, a background thread kicks off recall for the incoming message before the LLM call. By the time the agent needs context, the results are usually already cached. If the prefetch hasn't finished, the agent waits up to a short timeout then falls back to a synchronous search.

## Configuration

All settings live in `hybrid_memory.json` in the Hermes home directory. Full settings tables: [CONFIG_REFERENCE.md](CONFIG_REFERENCE.md).

## Storage modes

- **`shared_service`** (default) — a standalone RPC service holds the database; multiple Hermes processes (CLI + gateway + desktop) share one canonical store safely. Prevents DuckDB writer locks and split-brain.
- **`direct`** — the plugin opens the DuckDB file directly. Single-process only; use for debugging/diagnostics.

## Files

```
argos_plugin/
├── __init__.py              Provider: 16 tools, retrieval pipeline, hooks
├── store.py                 DuckDB store: CRUD, search, version chains,
│                            evidence, candidates, feedback, consolidation
├── graph.py                 Kùzu graph: entities, relations, aliases,
│                            traversal, junk-entity purge
├── embeddings.py            Local embeddings (BGE-small) + optional
│                            cross-encoder reranker (BGE-reranker-base)
├── retriever.py             Retrieval seam (pluggable retriever protocol)
├── extractor.py             Regex + LLM fact/relation extraction
├── reviewer.py              Candidate review + dedup + supersede logic
├── distillation.py          Gated LLM distillation pass ("the dream"):
│                            seed-star clustering + proposal emission
├── intent_router.py         Answerer routing for temporal/multi-hop queries
├── query_expander.py        LLM query rewriting for weak results
├── config_schema.py         Config fields surfaced in the setup UI
├── egress.py                Egress gating for plugin-owned LLM calls
├── memory_service.py        Shared-service RPC server
├── service_client.py        RPC client facade
├── confirmation.py          Pending-confirmation block builder
├── routing.py               Provider routing helpers
├── hermes_location.py       Ambient: location resolution
├── hermes_weather.py        Ambient: Open-Meteo weather (cached 20 min)
├── hermes_file_activity.py  Ambient: recently-edited file scan (cached 5 min)
├── backfill_graph.py        Rebuild graph from existing memories
├── rebuild_graph.py         Graph rebuild helper
├── reembed_memories.py      Re-embed all memories (model swap)
├── cleanup_memories.py      Manual cleanup utility
├── dump_memories.py         Export memories for inspection
├── migrate_gateway.py       Gateway migration helper
├── review_pending.py        Bulk review helper
├── plugin.yaml              Plugin manifest
├── skills/insight-log/      Insight-log skill (capture + recall)
├── tests/                   pytest suite (see Testing below)
└── eval/                    Evaluation harness
```

## Testing

```bash
cd argos_plugin
python -m pytest tests/ -v
```

Test files:

- `test_hybrid_memory.py` — store CRUD, search, dedup, version chains, cycle guards, scope isolation, graph extraction, alias resolution, chain-unfold trigger precision/recall.
- `test_shared_service.py` — shared-service RPC path.
- `test_candidate_review_integration.py` — candidate review + supersede flow.
- `test_semantic_dedup.py` — semantic merge / duplicate consolidation.
- `test_distillation.py` — distillation gates, clustering, proposal shape, contradiction ID validation, run-state semantics. Run with `HF_HUB_OFFLINE=1` — loading the embedder by name performs a network HEAD-check that hangs otherwise.
- `test_ambient_location.py`, `test_ambient_weather.py`, `test_ambient_file_activity.py` — ambient-context modules.

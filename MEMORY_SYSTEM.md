# Hybrid Memory System

A hybrid retrieval-augmented memory store for AI agents. Combines dense vector
search (DuckDB + sentence-transformers) with a knowledge graph (Kùzu) for
entity-aware retrieval, alias resolution, and memory evolution tracking — plus
an ambient-context layer, an auto-extraction + candidate-review pipeline, and
an insight-log skill.

This is the deep dive. For installation see **[SETUP_GUIDE.md](SETUP_GUIDE.md)**;
for the front-door overview see **[README.md](README.md)**.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  Provider Layer                          │
│   (12 memory_* tools + pre_llm_call ambient hook +       │
│    insight-log skill + /ilog, /revisit commands)         │
├─────────────────────────────────────────────────────────┤
│                  Retrieval Pipeline                      │
│   prefetch → context-aware enrichment → query expansion  │
│   → vector + text search → RRF fusion → importance       │
│   boosts → graph boost → alias expansion → optional      │
│   cross-encoder reranker → chain annotation →            │
│   chain-unfold injection                                 │
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

Twelve agent-callable tools:

| Tool | Purpose |
|------|---------|
| `memory_search` | Search by meaning (vector + text + RRF + boosts). `category` and `project_id` filters. |
| `memory_save` | Store a durable fact explicitly (with reasoning, not just conclusions). |
| `memory_update` | Update by ID — creates a new version, superseding the old. |
| `memory_delete` | Delete by ID; head deletion promotes the predecessor. |
| `memory_chain` | Walk a memory's version arc. Modes: `arc`, `versions`, `diff`. |
| `memory_graph_search` | Search the relationship graph for edges matching an entity term. |
| `memory_graph_query` | Traverse the graph around an entity (depth 1–4). |
| `memory_candidate_list` | List pending extraction proposals by status. |
| `memory_candidate_review` | Approve / reject / quarantine a proposal; `supersedes_memory_id` chains a replacement. |
| `memory_restore` | Restore a quarantined memory to active retrieval. |
| `memory_feedback` | Mark a memory `helpful`, `dismissed`, or `incorrect`. |
| `memory_maintenance` | Preview (`dry_run=true`) or apply reversible quarantine of stale/duplicate memories. Never hard-deletes. |

## Core concepts

### Memory records

Each memory is a typed record with a category, content text, tags, embedding
vector, and metadata. Categories include `personal_fact`, `context_note`,
`insight`, `event`, `relationship`, `goal`, and `preference`. Records are
scoped by user and optionally by project (`project_id` filter on search).

### Hybrid search

Retrieval combines:
- **Vector similarity** — cosine similarity between query and memory embeddings
  (BGE-small-en-v1.5 by default, runs fully offline/local).
- **Text search** — ILIKE keyword matching for exact-term hits.
- **Reciprocal Rank Fusion (RRF)** — merges the two ranked lists into one.
- **Importance boosts** — recency, retrieval frequency, and helpful/dismissed
  feedback adjust the final ranking.
- **Cross-encoder re-ranking** (optional, `reranker_enabled`) — re-ranks the
  top-N candidates with BGE-reranker-base (~420MB, ~300ms/query on CPU).
  Experimental; evaluate before relying on it.

### Knowledge graph

Memories are indexed into a Kùzu graph at save time. The graph stores:
- **Entity nodes** — people, places, concepts, organizations extracted from
  memory content.
- **Typed relations** — `married_to`, `works_at`, `uses`, `has_attribute`, etc.
- **Alias mappings** — "my role" → "Entity-A" so a query for either finds
  memories mentioning the other.
- **Memory-to-entity links** — each memory node links to the entities it
  mentions, enabling graph-supported retrieval.

Alias resolution is bidirectional: querying an alias finds the canonical
entity's memories, and querying the canonical name also finds memories that
use only the alias.

Graph retrieval has two modes:
- **Graph boost** (`graph_aware_retrieval`) — bumps the similarity score of
  memories supported by graph entities during normal search.
- **Graph traversal** (`graph_traversal_enabled`) — multi-hop boost for
  candidates reached by traversing the graph from query entities.

### Memory evolution (version chains)

When a memory is updated, the old version is superseded (not deleted). The
store tracks `valid_from`, `valid_to`, and `superseded_by` fields, forming a
chronological chain of versions for each fact.

- **`memory_chain` tool** — retrieves the full version arc for any memory ID
  (modes: `arc`, `versions`, `diff`).
- **Chain annotation** — search results carry a `chain` field indicating
  whether a memory has version history.
- **Chain-unfold** — when `chain_unfold=auto`, a change-intent query
  ("why did I switch...", "what changed...") triggers automatic injection of
  a compact version arc into the search results, so the agent can answer
  "how did this fact evolve?" without an extra tool call.
- **Head-deletion promotion** — deleting the current version promotes the
  predecessor to current (and re-indexes it in the graph).

### Candidate review

New memories go through a dedup + review pipeline:
- **Semantic dedup** — if a new memory is embedding-similar to an existing
  one, it's flagged rather than stored blindly.
- **Approve-with-supersede** — when a new memory contradicts/replaces an
  existing one (same category, high similarity), the reviewer can chain it
  behind the existing record as a new version.

### Auto-extraction pipeline

Every turn (`sync_turn`) is processed by a background worker:

1. **Regex extraction** — pattern-based fact detection (fast, zero cost).
2. **LLM fallback** (`llm_fallback=true`) — when regex misses, the auxiliary
   LLM proposes facts. Adds latency + token cost; proposals stay pending.
3. **Shadow-diff** (`extraction_shadow_diff=true`) — runs LLM extraction
   alongside regex and logs what each found that the other missed. Validation
   mode only; does not change proposals.
4. **Auto-review** (`auto_review=true`) — the auxiliary LLM reviews each new
   proposal: obvious junk quarantined, sensitive/contextless proposals stay
   `pending_user_confirmation`, clear facts approved.
5. **Stale-review sweep** (`stale_review_sweep_enabled=true`) — periodically
   re-reviews proposals pending too long.
6. **Role-word learning** (`role_alias_llm_fallback=true`) — when an unknown
   word appears in "my X is Name", the LLM is asked if X is a person-role;
   learned words persist to `role_words` so future occurrences are regex-fast.

Proposals are never active memory until reviewed. The agent can also save
explicitly via `memory_save`, bypassing the proposal queue.

### Maintenance and quarantine

- **`memory_maintenance`** — previews (`dry_run=true`, default) or applies
  reversible quarantine of stale temporary memories and low-quality
  duplicates. Never hard-deletes.
- **`consolidation_enabled`** — runs the same maintenance automatically at
  session end.
- **`memory_restore`** — brings a quarantined memory back to active retrieval
  (and re-indexes it in the graph).
- **`memory_feedback`** with `incorrect` — detaches the memory from the graph.
- **Junk-entity purge** — at session end, the graph purges orphaned junk
  entities (cheap maintenance).

### Ambient context

A `pre_llm_call` hook runs on every turn and injects short, fail-soft hints
into the user-message context (via the native `plugin_user_context` path —
no core source patch). Each hint is built independently so a failure in one
never suppresses the others:

- **Time** — `Current time: Friday 2026-07-31 19:55 SAST`, via `hermes_time`
  (respects `HERMES_TIMEZONE` → `config.yaml` `timezone` → server-local).
- **Location** — `Location: City-X`, resolved fresh each turn from
  `HERMES_LOCATION` → `config.yaml` `location`.
- **Weather** — `Weather: 14°C, light rain`, via `hermes_weather`
  (geocodes location, fetches Open-Meteo; cached ~20 min; free, no API key).
- **File activity** — `Last edited: ~/project/foo.py (4 min ago)`, via
  `hermes_file_activity` (scans the working directory; cached ~5 min).

A conditional coder-MCP directive is also injected when the turn looks
code-adjacent, steering the agent toward `mcp__coder__*` tools. Non-coding
turns cost nothing.

### Insight log

The `insight-log` skill captures personal realizations verbatim (no
sanitizing, no judgment) when the user shares a "I just realised…" moment,
saving them as `insight`-category memories. In future sessions it
proactively resurfaces relevant insights when the topic overlaps.

Two slash commands (registered by the plugin):

- **`/ilog [tag]`** — list saved insights, newest first (up to 20). Optional
  tag filter.
- **`/revisit`** — surface a random older insight (≥3 days old when
  available) for re-engagement.

### Prefetch

At `on_turn_start`, a background thread kicks off recall for the incoming
message before the LLM call. By the time the agent needs context, the
results are usually already cached. If the prefetch hasn't finished, the
agent waits up to a short timeout then falls back to a synchronous search.

## Configuration

All settings live in `hybrid_memory.json` (in the Hermes home directory).
Defaults below are the **runtime defaults** from `_load_config()`. The
desktop settings UI (`config_schema.py`) shows its own display defaults in
a couple of cases — those are footnoted.

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
| `reranker_enabled` | `false` | Cross-encoder re-ranking of top candidates (~300ms/query on CPU). Experimental. |
| `reranker_model` | `BAAI/bge-reranker-base` | HuggingFace reranker model (~420MB, downloaded on first use). |
| `reranker_top_n` | `10` | Number of top candidates to re-rank (5–100). |

### Retrieval

| Setting | Default | Description |
|---------|---------|-------------|
| `max_injected_items` | `20` ¹ | Max memories auto-injected as context before each turn. |
| `context_aware_retrieval` | `true` | Prepend recent conversation context to queries with pronouns/references. Zero latency, no LLM calls. |
| `context_window_size` | `3` | Recent user messages used as context (1–10). |
| `context_max_chars` | `500` | Max total chars of context to prepend (100–2000). |
| `query_expansion_enabled` | `true` | LLM rewrites weak queries into sub-queries. Fires only below the similarity floor; cached 1h; fail-soft. |
| `query_expansion_similarity_floor` | `0.3` | Trigger expansion when top hit similarity is below this (0.0–1.0). |

### Graph

| Setting | Default | Description |
|---------|---------|-------------|
| `graph_aware_retrieval` | `true` | Boost memories supported by graph entities during normal search. |
| `graph_retrieval_boost` | `0.0` ² | Max similarity boost for graph-supported memories (0.0–0.5). |
| `graph_boost_min_similarity` | `0.15` | Minimum semantic similarity for a memory to receive the graph boost. |
| `graph_inject_candidates` | `false` | Inject memories found *only* by the graph. Off by default — adds noise that hurt recall in benchmarks. |
| `graph_traversal_enabled` | `true` | Enable graph-traversal boost for multi-hop retrieval. |
| `graph_traversal_depth` | `2` | Traversal depth for graph-traversal boost. |
| `graph_traversal_boost` | `0.60` | Boost strength for graph-traversal candidates. |
| `alias_expansion_boost` | `0.7` | Similarity floor for alias-expanded candidates. |
| `entity_aliases` | *(empty)* | JSON mapping of aliases → canonical entity names, e.g. `{"my role": "Entity-A"}`. |
| `role_words` | *(empty)* | Extra role words for "my X is Name" alias extraction. 40+ defaults built in; LLM-learned words auto-added. |

### Chains

| Setting | Default | Description |
|---------|---------|-------------|
| `chain_unfold` | `auto` ³ | Auto-inject a compact version arc on change-intent queries. `off`, `auto`, or `always`. |
| `chain_unfold_min_similarity` | `0.30` | Per-candidate similarity floor for unfold (the precision guard). |
| `chain_unfold_top_k` | `3` | How many top results to scan for a chain anchor (1–20). |
| `chain_unfold_query_fallback` | `false` | Search deeper for a chain matching the query if no top-K result has one. |
| `chain_max_versions` | `3` | Max versions to inject per chain unfold (1–10). |
| `chain_max_inject` | `150` | Soft token cap per chain injection (~4 chars/token estimate). |

### Extraction

| Setting | Default | Description |
|---------|---------|-------------|
| `auto_extract` | `true` | Extract memory proposals from each conversation turn. |
| `llm_fallback` | `true` | Use the host LLM to create proposals when regex misses. |
| `extraction_shadow_diff` | `false` | Run LLM extraction in parallel with regex and log the diff (validation mode). |
| `auto_review` | `true` | Auxiliary LLM reviews new proposals; obvious junk quarantined. |
| `stale_review_sweep_enabled` | `true` | Periodically re-review proposals pending too long. |
| `stale_review_interval_min` | `15` | Sweep interval in minutes. |
| `stale_review_min_age_min` | `30` | Min proposal age (minutes) before eligible for the stale sweep. |
| `stale_review_max_batch` | `25` | Max proposals re-reviewed per sweep. |
| `role_alias_llm_fallback` | `true` | Ask the LLM if an unknown "my X is Name" word is a person-role; learned words persist. |

### LLM

| Setting | Default | Description |
|---------|---------|-------------|
| `llm_model` | *(empty)* | Model ID for auxiliary LLM tasks. Empty = host auxiliary default. |
| `llm_provider` | *(empty)* | Provider override (`openrouter`, `openai`, `anthropic`, …). Empty = host default. |

### Maintenance

| Setting | Default | Description |
|---------|---------|-------------|
| `consolidation_enabled` | `false` | Reversible maintenance of expired/duplicate memories at session end. |
| `consolidation_min_age_days` | `30` | Age threshold for stale temporary memories. |
| `consolidation_max_actions` | `25` | Max records to quarantine per run. |

> ¹ The desktop UI schema shows `8` as its display default; the runtime
> default is `20`. Your saved `hybrid_memory.json` wins either way.
> ² The desktop UI schema shows `0.05`; the runtime default is `0.0`.
> ³ The desktop UI schema shows `off` ("ships OFF"); the runtime default
> is `auto`. The `memory_chain` tool is always available on-demand
> regardless of this setting.

## Storage modes

- **`shared_service`** (default) — a standalone RPC service holds the
  database; multiple Hermes processes (CLI + gateway + desktop) share one
  canonical store safely. Prevents DuckDB writer locks and split-brain.
- **`direct`** — the plugin opens the DuckDB file directly. Single-process
  only; use for debugging/diagnostics.

## Files

```
hybrid_memory_plugin/
├── __init__.py              Provider: 12 tools, retrieval pipeline, hooks
├── store.py                 DuckDB store: CRUD, search, version chains,
│                            evidence, candidates, feedback, consolidation
├── graph.py                 Kùzu graph: entities, relations, aliases,
│                            traversal, junk-entity purge
├── embeddings.py            Local embeddings (BGE-small) + optional
│                            cross-encoder reranker (BGE-reranker-base)
├── retriever.py             Retrieval seam (pluggable retriever protocol)
├── extractor.py             Regex + LLM fact/relation extraction
├── reviewer.py              Candidate review + dedup + supersede logic
├── query_expander.py        LLM query rewriting for weak results
├── config_schema.py         Config fields surfaced in the setup UI
├── memory_service.py        Shared-service RPC server
├── service_client.py        RPC client facade
├── confirmation.py          Pending-confirmation block builder
├── routing.py               Provider routing helpers
├── hermes_location.py       Ambient: location resolution
├── hermes_weather.py        Ambient: Open-Meteo weather (cached)
├── hermes_file_activity.py  Ambient: recently-edited file scan
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
cd hybrid_memory_plugin
python -m pytest tests/ -v
```

Test files and coverage:

- `test_hybrid_memory.py` — store CRUD, search, dedup, version chains, cycle
  guards, scope isolation, graph extraction, alias resolution, chain-unfold
  trigger precision/recall.
- `test_shared_service.py` — shared-service RPC path.
- `test_candidate_review_integration.py` — candidate review + supersede flow.
- `test_reviewer.py` — reviewer unit tests.
- `test_ambient_location.py`, `test_ambient_weather.py`,
  `test_ambient_file_activity.py` — ambient-context modules.

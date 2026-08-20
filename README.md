# Argos 👁️

> **Argos** — from the hundred-eyed watchman of Greek myth who never sleeps
> and never misses a thing. A fitting name for a memory system that watches
> your whole life, keeps it, and recalls it faithfully.

Argos is a **hybrid memory system** for AI agents. It combines a dense
vector store (DuckDB) with a knowledge graph (Kùzu) to give an assistant
persistent, entity-aware, self-evolving memory — plus an ambient-context
layer that feeds the agent live time, location, weather, and file-activity
every turn, and an insight-log skill that captures and resurfaces personal
realizations.

**Where data lives:** all memory records, embeddings, and the relationship
graph are stored **locally** on your machine — flat files in the Hermes home
directory. They're never shipped to a hosted memory vendor *(your memory
store is local — though the text that becomes memory still transits the
cloud LLM during processing)*.

**Where the LLM is:** Argos calls an LLM for extraction, candidate review,
and query expansion. Today that's the configured cloud model — there is **no
native local-LLM support yet**. (Local embeddings via BGE-small are offline;
the *storage* is fully local; the *LLM plumbing* is cloud until a local runtime
is wired in.)

---

## What it does

- **Persistent memory** across sessions — the agent remembers who you are,
  what you care about, and your *history*, not just this conversation.
- **Hybrid search** — dense vector similarity + keyword text search, fused
  by Reciprocal Rank Fusion, with recency / retrieval-frequency /
  helpful-dismissed importance boosts, optional cross-encoder re-ranking,
  and graph-supported retrieval.
- **Relationship graph** — memories are indexed into a Kùzu graph at save
  time: entities (people, places, concepts), typed relations
  (`married_to`, `works_at`, `uses`, …), and bidirectional alias resolution
  (so "my partner" and their name both find the same memories).
- **Memory evolution** — facts aren't deleted, they're *versioned*. Updates
  supersede old versions into chronological chains; `memory_chain` walks the
  arc, and `chain_unfold` auto-injects a compact history when you ask
  "what changed?".
- **Auto-extraction + candidate review** — every turn is mined for durable
  facts (regex first, optional LLM fallback). Proposals are *pending*, not
  active — an auxiliary LLM auto-reviews them, obvious junk is quarantined,
  and the rest wait for explicit approval. Nothing becomes memory silently.
- **Reversible maintenance** — stale temporary memories and low-quality
  duplicates can be *quarantined* (never hard-deleted) via `memory_maintenance`,
  and restored with `memory_restore`.
- **Ambient context** — a `pre_llm_call` hook injects per-turn hints: current
  time (IANA-timezone-aware), configured location, weather (Open-Meteo, cached
  ~20 min), and recently edited files. Each hint is independent and fail-soft.
- **Insight log** — an `insight-log` skill captures personal realizations
  verbatim and resurfaces them contextually; `/ilog` browses the log and
  `/revisit` brings back an older insight for re-engagement.

## Tools it exposes

Twelve agent-callable tools (verified with `hermes tools`):

| Tool | Purpose |
|------|---------|
| `memory_search` | Search memory by meaning (vector + text + RRF + boosts). Supports `category` and `project_id` filters. |
| `memory_save` | Store a durable fact explicitly (with reasoning, not just conclusions). |
| `memory_update` | Update a memory by ID — creates a new version, superseding the old. |
| `memory_delete` | Delete a memory by ID; head deletion promotes the predecessor. |
| `memory_chain` | Walk a memory's version arc. Modes: `arc` (compact), `versions` (full), `diff` (per-step deltas). |
| `memory_graph_search` | Search the relationship graph for edges matching an entity term. |
| `memory_graph_query` | Traverse the graph around an entity (depth 1–4), returning connected nodes, relations, and supporting memories. |
| `memory_candidate_list` | List pending extraction proposals by status (`pending`, `approved`, `rejected`, `quarantined`, `pending_user_confirmation`, …). |
| `memory_candidate_review` | Approve / reject / quarantine a proposal. Pass `supersedes_memory_id` to chain a replacement behind an existing fact. |
| `memory_restore` | Restore a quarantined memory to active retrieval. |
| `memory_feedback` | Mark an active memory `helpful`, `dismissed`, or `incorrect` (incorrect also detaches it from the graph). |
| `memory_maintenance` | Preview (`dry_run=true`, default) or apply reversible quarantine of stale/duplicate memories. Never hard-deletes. |

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

Two storage modes: **`shared_service`** (default — a standalone RPC service
owns the DB; the plugin connects over a socket; multi-process safe) and
**`direct`** (plugin opens the DuckDB file directly, single-process;
diagnostics only).

## Ambient context

A `pre_llm_call` hook runs on every turn and injects short, fail-soft hints
into the user-message context (via the native `plugin_user_context` path —
no core source patch). Each hint is built independently so a failure in one
(weather network timeout, missing location) never suppresses the others:

- **Time** — `Current time: Friday 2026-07-31 19:55 SAST`, via `hermes_time`
  (respects `HERMES_TIMEZONE` → `config.yaml` `timezone` → server-local).
- **Location** — `Location: City-X`, resolved fresh each turn from
  `HERMES_LOCATION` → `config.yaml` `location`.
- **Weather** — `Weather: 14°C, light rain`, via `hermes_weather`
  (geocodes location, fetches Open-Meteo; cached ~20 min; free, no API key).
- **File activity** — `Last edited: ~/project/foo.py (4 min ago)`, via
  `hermes_file_activity` (scans the working directory for recently modified
  files; cached ~5 min).

A conditional coder-MCP directive is also injected when the turn looks
code-adjacent (indexed repo names or code terms), steering the agent toward
the `mcp__coder__*` tools. Non-coding turns cost nothing.

## Insight log

The `insight-log` skill captures personal realizations verbatim (no
sanitizing, no judgment) when the user shares a "I just realised…" moment,
saving them as `insight`-category memories. In future sessions it
proactively resurfaces relevant insights when the topic overlaps.

Two slash commands (registered by the plugin):

- **`/ilog [tag]`** — list saved insights, newest first (up to 20). Optional
  tag filter (`/ilog work`, `/ilog ex shame`).
- **`/revisit`** — surface a random older insight (≥3 days old when
  available) for re-engagement.

## Auto-extraction pipeline

Every turn (`sync_turn`) is processed by a background worker:

1. **Regex extraction** — pattern-based fact detection (fast, zero cost).
2. **LLM fallback** (optional, `llm_fallback=true`) — when regex misses, the
   auxiliary LLM proposes facts. Adds latency + token cost; proposals stay
   pending until reviewed.
3. **Shadow-diff** (optional, `extraction_shadow_diff=true`) — runs LLM
   extraction alongside regex and logs what each found that the other
   missed. Validation mode only; does not change proposals.
4. **Auto-review** (`auto_review=true`) — the auxiliary LLM reviews each new
   proposal: obvious junk is quarantined, sensitive/contextless proposals
   stay `pending_user_confirmation`, clear facts are approved.
5. **Stale-review sweep** — periodically re-reviews proposals that have sat
   pending too long.
6. **Role-word learning** (`role_alias_llm_fallback=true`) — when an unknown
   word appears in "my X is Name", the LLM is asked if X is a person-role;
   learned words are persisted to `role_words` so future occurrences are
   regex-fast.

Proposals are never active memory until reviewed. The agent can also save
explicitly via `memory_save`, bypassing the proposal queue.

## Repository layout

```
├── hybrid_memory_plugin/        The Argos provider plugin (core of this repo)
│   ├── __init__.py              Provider: 12 tools, retrieval pipeline, hooks
│   ├── store.py                 DuckDB store: CRUD, search, chains, evidence,
│   │                            candidates, feedback, consolidation
│   ├── graph.py                 Kùzu graph: entities, relations, aliases,
│   │                            traversal, junk-entity purge
│   ├── embeddings.py            Local embeddings (BGE-small) + optional
│   │                            cross-encoder reranker (BGE-reranker-base)
│   ├── retriever.py             Retrieval seam (pluggable retriever protocol)
│   ├── extractor.py             Regex + LLM fact/relation extraction
│   ├── reviewer.py              Candidate review + dedup + supersede logic
│   ├── query_expander.py        LLM query rewriting for weak results
│   ├── config_schema.py         Config fields surfaced in the setup UI
│   ├── memory_service.py        Shared-service RPC server
│   ├── service_client.py        RPC client facade
│   ├── confirmation.py          Pending-confirmation block builder
│   ├── routing.py               Provider routing helpers
│   ├── hermes_location.py       Ambient: location resolution
│   ├── hermes_weather.py        Ambient: Open-Meteo weather (cached)
│   ├── hermes_file_activity.py  Ambient: recently-edited file scan
│   ├── backfill_graph.py        Rebuild graph from existing memories
│   ├── rebuild_graph.py         Graph rebuild helper
│   ├── reembed_memories.py      Re-embed all memories (model swap)
│   ├── cleanup_memories.py      Manual cleanup utility
│   ├── dump_memories.py         Export memories for inspection
│   ├── migrate_gateway.py       Gateway migration helper
│   ├── review_pending.py        Bulk review helper
│   ├── plugin.yaml              Plugin manifest
│   ├── skills/insight-log/      Insight-log skill (capture + recall)
│   ├── tests/                   pytest suite (see Testing below)
│   └── eval/                    Evaluation harness
├── ambient_context/             Ambient-context source modules + tests
├── tool_compression/            Compressed tool definitions (browser, code,
│                                terminal, delegate, memory, session_search…)
├── desktop_plugins/             Desktop-side plugins (aux-models)
├── desktop_fix/                 Desktop fix patches
├── scripts/                     Helper/dev scripts (apply_customizations,
│                                backup_data, embedding-check, memory-check,
│                                state_db_maintenance, fork-switchover)
├── patches/                     Customization patches applied to Hermes
├── database/                    Local DuckDB + Kùzu files (gitignored data)
├── data/                        Runtime data (duckdb, kuzu, lancedb, manifests)
├── SETUP_GUIDE.md               How to install & configure (start here)
├── REINSTALL.md                 Wipe / reinstall / recovery & migration
└── MEMORY_SYSTEM.md             Deep dive: architecture, concepts, config
```

## Quick start

Full instructions are in **[SETUP_GUIDE.md](SETUP_GUIDE.md)**. The short
version:

1. Place `hybrid_memory_plugin/` in your Hermes plugins directory
   (`%LOCALAPPDATA%\hermes\plugins\hybrid_memory\` on Windows,
   `~/.hermes/plugins/hybrid_memory/` on Linux/macOS).
2. Restart Hermes — the plugin auto-installs its dependencies
   (`duckdb`, `kuzu`, `sentence-transformers`) on first load.
3. Verify with `hermes tools` — the twelve `memory_*` tools should appear.
4. Configure via `hybrid_memory.json` in the Hermes home directory, or in
   the Hermes settings UI under **Memory → Argos (Local)**.

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

## Who is this for?

Argos keeps all of its memory data local to your machine — you don't hand
your memory over to a hosted memory vendor. Note that Argos itself still
calls an LLM (today, the cloud model you configure) for extraction and
review; native local-LLM support isn't here yet. If you want an agent that
actually remembers you across sessions — with memory stored under your own
control — this is the idea.

---

📖 Start with **[SETUP_GUIDE.md](SETUP_GUIDE.md)** · Deep dive in
**[MEMORY_SYSTEM.md](MEMORY_SYSTEM.md)** · Recovery in
**[REINSTALL.md](REINSTALL.md)**

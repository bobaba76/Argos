# Hybrid Memory System

A hybrid retrieval-augmented memory store for AI agents. Combines dense vector
search (DuckDB + sentence-transformers) with a knowledge graph (Kùzu) for
entity-aware retrieval, alias resolution, and memory evolution tracking — plus
an ambient-context layer, an auto-extraction + candidate-review pipeline, an
insight-log skill, and a gated LLM distillation pass ("the dream") that
proposes insights and guardrails from accumulated records.

This is the deep dive. For installation see **[SETUP_GUIDE.md](SETUP_GUIDE.md)**;
for the front-door overview see **[README.md](README.md)**.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  Provider Layer                          │
│   (16 memory_* tools + pre_llm_call ambient hook +       │
│    insight-log skill + /ilog, /revisit, /neg commands)         │
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

Fourteen agent-callable tools:

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
| `memory_why_not` | Diagnose why a memory missed retrieval: deterministic, free (no LLM), read-only. |
| `memory_fetch_full` | Fetch the full stored text of a memory by ID when the injected preview was truncated. |
| `memory_tombstones` | List deletion tombstones: fingerprints of hard-deleted memories that are blocked from being re-created. Use to answer "what was permanently deleted" or diagnose why saving a fact silently does nothing. Read-only. |
| `memory_tombstone_purge` | Escape hatch: lift a deletion tombstone so a previously hard-deleted fact may be saved again. Requires the exact original content + category (matching is case/whitespace-normalized). Explicit user request only. |

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
  top-N candidates with BGE-reranker-base (~420MB; measured ~8s/query on CPU, 26/8 A/B — needs CUDA torch to be production-viable).
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
  - **Measured (2026-08-20, eval harness):** ~93% recall and ~93% precision
    on change-intent questions after widening the intent matcher. The
    `Arc(0.15)` + `anchor(0.30)` similarity floors are *pure precision
    gates* — sweeps showed they have zero recall cost.
  - **Diagnosed ceiling:** the residual false positives sit just inside the
    true-positive similarity band (one FP at 0.548), so no cosine threshold
    separates them; ~93% precision is floor-independent. The trigger matcher
    is the lever that moved recall, not the thresholds.
- **Head-deletion promotion** — deleting the current version promotes the
  predecessor to current (and re-indexes it in the graph).

### Date-anchored retrieval

Time expressions ("two weeks ago", "last Friday", "that Valentine's") are
extracted by regex, resolved relative to the question date (shift-preserved
for multi-hop references), and used to re-rank the result slice by recency —
so a question like "what did we decide at last week's meeting?" weights the
matching records by their true date position instead of pure similarity.
Added 2026-08-21; evaluation bucket covers 133 temporal questions.

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
- **Semantic merge (P4.1)** — `consolidate()`'s duplicate leg was upgraded
  from exact/containment string matching to embedding-similarity merging:
  records within a high similarity threshold of an existing active record
  are consolidated (newest wins, older version chained or appended
  deterministically). Reversible, never hard-deletes.

### Distillation pass (the dream)

A bounded, LLM-assisted consolidation pass that turns accumulated records +
feedback into *proposed* insights, guardrails, and contradiction warnings.
Enabled by default? **No** in code (`distillation_enabled: false`) — but
  the maintainer's production config ships it **enabled**, because the
  measured 2×2 (26/8) showed the distill store is load-bearing for the
  flash answerer (+38.6 pts) and mildly harmful for a gpt-4o-class
  answerer (−5.6 pts).

- **Trigger** — runs at session boundaries. In the desktop app, sessions are
  resumable tiles and true `on_session_end` boundaries don't exist under the
  default reset policy, so the pass also fires from `on_session_switch`
  (every chat rotation). Both hooks call the same self-gating method.
- **Gates (all store-side, before any LLM call)** — novelty: ≥ 20 new or
  updated records since the last run; cooldown: ≥ 24h since the last run;
  budget: ≤ 100 records and ≤ 10 LLM calls per run.
- **Cluster scan (free, deterministic)** — seed-star greedy grouping over
  the eligible records: the newest record seeds a cluster, members are
  records with cosine ≥ 0.75 *to the seed only* (no transitive chaining, so
  subject-dense stores can't collapse into one giant cluster), capped at 8
  per cluster.
- **LLM distill** — one call per cluster with a strict JSON prompt:
  insights, contradictions (`a_id`/`b_id`/`reason`), and guardrails; short
  items only; contradictions are honored only for IDs the model was actually
  shown in that cluster. One additional call scans high-feedback records
  (helpful/dismissed counters) for lessons.
- **Proposals only** — every output is saved via `save_candidate()`
  (`source="distillation"`, `dedup=True`) with grounding (`evidence_text` +
  `payload.sources` listing source memory IDs). Proposals are pending —
  invisible to retrieval — until the existing auto-review pipeline and the
  user approve them. The pass never writes, edits, or deletes active memory.
- **Run state** — values live in a small `system_state` KV table
  (`distillation_last_run`, `distillation_last_count`). Advances only on
  *completed* runs (including zero-proposal runs); fail-soft aborts leave it
  unadvanced so the next clean boundary retries.
- **Fail-soft throughout** — LLM error, bad JSON, or client import failure
  skips the affected leg; session lifecycle is never blocked.
- **Auxiliary LLM** — reuses the same model/provider as other auxiliary
  tasks; can be pointed at a cheaper model via the host's
  `auxiliary.distillation.model` config without any plugin change.

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

Three slash commands (registered by the plugin):

- **`/ilog [tag]`** — list saved insights, newest first (up to 20). Optional
  tag filter.
- **`/revisit`** — surface a random older insight (≥3 days old when
  available) for re-engagement.
- **`/neg <claim>`** — store an explicit exclusion (e.g. `/neg I do not
  drink coffee`), injected with a `[negative]` label so the model answers
  grounded nos instead of guessing.

### Prefetch

At `on_turn_start`, a background thread kicks off recall for the incoming
message before the LLM call. By the time the agent needs context, the
results are usually already cached. If the prefetch hasn't finished, the
agent waits up to a short timeout then falls back to a synchronous search.

## Configuration

All settings live in `hybrid_memory.json` in the Hermes home directory
(runtime defaults from `_load_config()`). The full settings tables — every
option, default, and the runtime-vs-UI-default footnotes — live in
**[CONFIG_REFERENCE.md](CONFIG_REFERENCE.md)**. This file covers behavior,
not every knob.

## Storage modes

- **`shared_service`** (default) — a standalone RPC service holds the
  database; multiple Hermes processes (CLI + gateway + desktop) share one
  canonical store safely. Prevents DuckDB writer locks and split-brain.
- **`direct`** — the plugin opens the DuckDB file directly. Single-process
  only; use for debugging/diagnostics.

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
cd argos_plugin
python -m pytest tests/ -v
```

Test files and coverage:

- `test_hybrid_memory.py` — store CRUD, search, dedup, version chains, cycle
  guards, scope isolation, graph extraction, alias resolution, chain-unfold
  trigger precision/recall.
- `test_shared_service.py` — shared-service RPC path.
- `test_candidate_review_integration.py` — candidate review + supersede flow.
- `test_semantic_dedup.py` — semantic merge / duplicate consolidation (P4.1).
- `test_distillation.py` — distillation gates, clustering, proposal shape,
  contradiction ID validation, run-state semantics (P4.2). **Run with
  `HF_HUB_OFFLINE=1`** — loading the embedder by name performs a network
  HEAD-check that hangs for minutes otherwise.
- `test_ambient_location.py`, `test_ambient_weather.py`,
  `test_ambient_file_activity.py` — ambient-context modules.

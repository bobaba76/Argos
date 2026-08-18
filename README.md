# Argos 👁️

> **Argos** — from the hundred-eyed watchman of Greek myth who never sleeps
> and never misses a thing. A fitting name for a memory system that watches
> your whole life, keeps it, and recalls it faithfully.

Argos is a **hybrid memory system** for AI agents. It combines a dense
vector store (DuckDB) with a knowledge graph (Kùzu) to give an assistant
persistent, entity-aware, self-evolving memory.

**Where data lives:** all memory records, embeddings, and the relationship
graph are stored **locally** on your machine — flat files in the Hermes home
directory. They're never shipped to a hosted memory vendor.

**Where the LLM is:** Argos calls an LLM for extraction, candidate review,
and query expansion. Today that's the configured cloud model — there is **no
native local-LLM support yet**. (Local embeddings via BGE-small are offline;
the *storage* is fully local; the *LLM plumbing* is cloud until a local runtime
is wired in.)

It started life as the "Hermes Memory" / "hybrid memory" plugin; the name is
now **Argos**. The internal identifiers (`hybrid_memory`, the database files,
plugin keys) are kept as the stable plumbing — the visible name is Argos.

---

## What it does

- **Persistent memory** across sessions — the agent remembers who you are,
  what you care about, and your *history*, not just this conversation.
- **Hybrid search** — dense vector similarity + keyword text search, fused
  by Reciprocal Rank Fusion, with recency / retrieval-frequency /
  helpful-dismissed importance boosts.
- **Relationship graph** — memories are indexed into a Kùzu graph at save
  time: entities (people, places, concepts), typed relations
  (`married_to`, `works_at`, `uses`, …), and bidirectional alias resolution
  (so "my wife" and her name both find the same memories).
- **Memory evolution** — facts aren't deleted, they're *versioned*. Updates
  supersede old versions into chronological chains; `memory_chain` walks the
  arc, and `chain_unfold` auto-injects a compact history when you ask
  "what changed?".
- **Candidate review** — new memories pass through semantic dedup and a
  review step before they're trusted, so duplicates and stale contradictions
  don't pile up.

## Tools it exposes

`memory_search` · `memory_save` · `memory_update` · `memory_chain` ·
`memory_delete` (+ internal candidate-review, extraction, and retrieval
machinery).

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  Provider Layer                          │
│   (tool interface: memory_search / save / update /       │
│    chain / delete)                                       │
├─────────────────────────────────────────────────────────┤
│                  Retrieval Pipeline                      │
│   query expansion → vector search → RRF fusion →         │
│   graph boost → alias expansion → chain annotation →     │
│   chain-unfold injection                                 │
├──────────────────┬──────────────────────────────────────┤
│   DuckDB Store   │         Kùzu Graph                   │
│  (memory records,│   (entities, relations, aliases,      │
│   embeddings,    │    memory-to-entity links)            │
│   version chains)│                                      │
└──────────────────┴──────────────────────────────────────┘
```

Two storage modes: **`direct`** (plugin opens the DuckDB file directly,
single-process) and **`shared_service`** (a standalone RPC service owns the
DB; the plugin connects over a socket — multi-process safe, the default).

## Repository layout

```
├── hybrid_memory_plugin/   The Argos provider plugin (core of this repo)
│   ├── __init__.py         Provider: tool interface, retrieval pipeline
│   ├── store.py            DuckDB storage: CRUD, search, chains, evidence
│   ├── graph.py            Kùzu graph: entity extraction, alias resolution
│   ├── embeddings.py       Local embedding model (BGE-small-en-v1.5, offline)
│   ├── retriever.py        Retrieval seam (pluggable retriever)
│   ├── extractor.py        Fact / relation extraction from text
│   ├── reviewer.py         Candidate review + dedup + supersede
│   ├── query_expander.py   LLM query rewriting on weak results
│   ├── config_schema.py    Config fields surfaced in the setup UI
│   ├── memory_service.py / service_client.py   Shared-service RPC + client
│   └── plugin.yaml         Plugin manifest
├── SETUP_GUIDE.md          How to install & configure (start here)
├── REINSTALL.md            Wipe / reinstall / recovery & migration
├── MEMORY_SYSTEM.md        Deep dive: architecture, concepts, config table
├── scripts/                Helper/dev scripts
└── patches/                Customization patches applied to Hermes
```

## Quick start

Full instructions are in **[SETUP_GUIDE.md](SETUP_GUIDE.md)**. The short
version:

1. Place `hybrid_memory_plugin/` in your Hermes plugins directory
   (`%LOCALAPPDATA%\hermes\plugins\hybrid_memory\` on Windows,
   `~/.hermes/plugins/hybrid_memory/` on Linux/macOS).
2. Restart Hermes — the plugin auto-installs its dependencies
   (`duckdb`, `kuzu`, `sentence-transformers`) on first load.
3. Verify with `hermes tools` — the five `memory_*` tools should appear.
4. Configure via `hybrid_memory.json` in the Hermes home directory, or in
   the Hermes settings UI under **Memory → Argos (Local)**.

## Configuration

All settings live in `hybrid_memory.json` in the Hermes home directory. Key
knobs (see [MEMORY_SYSTEM.md](MEMORY_SYSTEM.md) for the full table):

| Setting | Default | Description |
|---------|---------|-------------|
| `max_injected_items` | `96` | How many memories are injected into context per turn |
| `chain_unfold` | `auto` | Auto-inject version arcs on change-intent queries |
| `graph_aware_retrieval` | `true` | Enable graph-supported search |
| `query_expansion_enabled` | `true` | LLM query rewriting on weak results |
| `context_aware_retrieval` | `true` | Enrich referential queries with conversation context |
| `local_embedding_model` | *(local path)* | Offline embedding model |

> The settings panel reads the *same* `hybrid_memory.json` the running
> service uses — what you see in the UI is what's live (no separate,
> diverging store).

## Testing

```bash
cd hybrid_memory_plugin
python -m pytest tests/test_hybrid_memory.py -v
```

Tests cover: store CRUD, search, dedup, version chains, cycle guards, scope
isolation, graph extraction, alias resolution, chain-unfold trigger
precision/recall, and the shared-service RPC path.

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

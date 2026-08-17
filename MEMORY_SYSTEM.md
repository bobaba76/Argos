# Hybrid Memory System

A hybrid retrieval-augmented memory store for AI agents. Combines dense vector
search (DuckDB + sentence-transformers) with a knowledge graph (KuzuDB) for
entity-aware retrieval, alias resolution, and memory evolution tracking.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Provider Layer                     │
│   (tool interface: memory_search, memory_save,       │
│    memory_update, memory_chain, memory_delete)        │
├─────────────────────────────────────────────────────┤
│              Retrieval Pipeline                       │
│   query expansion → vector search → RRF fusion →     │
│   graph boost → alias expansion → reranker →          │
│   chain annotation → chain-unfold injection           │
├──────────────────┬──────────────────────────────────┤
│   DuckDB Store   │         KuzuDB Graph              │
│  (memory records,│   (entities, relations, aliases,  │
│   embeddings,    │    memory-to-entity links)         │
│   version chains)│                                   │
└──────────────────┴──────────────────────────────────┘
```

## Core concepts

### Memory records

Each memory is a typed record with a category, content text, tags, embedding
vector, and metadata. Categories include `personal_fact`, `context_note`,
`insight`, `relationship`, `preference`, and others. Records are scoped by
user and optionally by project.

### Hybrid search

Retrieval combines:
- **Vector similarity** — cosine similarity between query and memory embeddings
  (BGE-small-en-v1.5 by default, runs fully offline/local).
- **Text search** — ILIKE keyword matching for exact-term hits.
- **Reciprocal Rank Fusion (RRF)** — merges the two ranked lists into one.
- **Importance boosts** — recency, retrieval frequency, and helpful/dismissed
  feedback adjust the final ranking.

### Knowledge graph

Memories are indexed into a KuzuDB graph at save time. The graph stores:
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

### Candidate review

New memories go through a dedup + review pipeline:
- **Semantic dedup** — if a new memory is embedding-similar to an existing
  one, it's flagged rather than stored blindly.
- **Approve-with-supersede** — when a new memory contradicts/replaces an
  existing one (same category, high similarity), the reviewer can chain it
  behind the existing record as a new version.

## Configuration

All settings live in `hybrid_memory.json` (in the Hermes home directory).
Key knobs:

| Setting | Default | Description |
|---------|---------|-------------|
| `chain_unfold` | `auto` | Auto-inject version arcs on change-intent queries |
| `chain_unfold_min_similarity` | `0.30` | Per-candidate similarity floor for unfold |
| `chain_unfold_top_k` | `3` | How many top results to scan for a chain anchor |
| `chain_unfold_query_fallback` | `true` | Search deeper if no top-K result has a chain |
| `chain_max_versions` | `3` | Max versions to inject per chain |
| `chain_max_inject` | `150` | Soft token cap per chain injection |
| `graph_aware_retrieval` | `true` | Enable graph-supported search |
| `alias_expansion_boost` | `0.7` | Similarity floor for alias-expanded candidates |
| `query_expansion_enabled` | `true` | LLM query rewriting on weak results |
| `context_aware_retrieval` | `true` | Enrich referential queries with conversation context |

## Storage modes

- **`direct`** — the plugin opens the DuckDB file directly (single-process).
- **`shared_service`** — a standalone RPC service holds the DB; the plugin
  connects via a socket client (multi-process safe).

## Files

```
hybrid_memory_plugin/
├── __init__.py          Provider: tool interface, retrieval pipeline
├── store.py             DuckDB store: CRUD, search, version chains, evidence
├── graph.py             KuzuDB graph: entity extraction, alias resolution
├── embeddings.py        Local embedding model (BGE-small-en-v1.5)
├── retriever.py         Retrieval seam (pluggable retriever protocol)
├── extractor.py         Fact/relation extraction from text
├── reviewer.py          Candidate review + dedup + supersede logic
├── query_expander.py    LLM-powered query rewriting for weak results
├── config_schema.py     Config field definitions for the setup UI
├── memory_service.py    Shared-service RPC server
├── service_client.py    RPC client facade
└── plugin.yaml          Plugin manifest
```

## Testing

```bash
cd hybrid_memory_plugin
python -m pytest tests/test_hybrid_memory.py -v
```

Tests cover: store CRUD, search, dedup, version chains, cycle guards, scope
isolation, graph extraction, alias resolution, chain-unfold trigger
precision/recall, and the shared-service RPC path.

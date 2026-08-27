# Setup Guide

## Prerequisites

- Python 3.11+
- Hermes (the agent framework this plugin extends)
- pip packages auto-installed on first load: `duckdb`, `kuzu`, `sentence-transformers`
- Optional: `BAAI/bge-reranker-base` (~420MB, downloaded on first use if `reranker_enabled=true`)

## Installation

### Option A: Automatic (via Hermes plugin system)

1. Place `argos_plugin/` in your Hermes plugins folder:
   ```
   ~/.hermes/plugins/hybrid_memory/            (Linux/macOS)
   %LOCALAPPDATA%\hermes\plugins\hybrid_memory\ (Windows)
   ```
2. Restart Hermes. Pip dependencies install on first load.
3. Verify: `hermes tools` — you should see the 16 `memory_*` tools.

### Option B: Manual pip install

```bash
pip install duckdb kuzu sentence-transformers
```

## Configuration

Settings live in `hybrid_memory.json` in the Hermes home directory (`~/.hermes/` on Linux/macOS, `%LOCALAPPDATA%\hermes\` on Windows). Created on first run with defaults. Representative subset:

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

The settings UI reads the same `hybrid_memory.json` the service uses — what you see is what's live. Full settings reference: [CONFIG_REFERENCE.md](CONFIG_REFERENCE.md).

## Storage modes

- **`shared_service`** (recommended) — a standalone RPC service owns the database; multiple Hermes processes (CLI + gateway + desktop) share one store safely.
- **`direct`** — the plugin opens the DuckDB file directly. Single-process only; for debugging.

## Embedding model

BGE-small-en-v1.5, loaded locally (no API calls, no cloud). Cached at `~/.hermes/models/bge-small-en-v1.5/`. Falls back to text-only search if the model is unavailable.

## Verifying it works

`hermes tools` confirms the plugin loaded (16 `memory_*` tools listed). To test save/search/graph-query, start a chat and ask the agent to use the tools — that's the only runtime path. Examples:

- "Remember that I use Vim as my primary editor" — triggers `memory_save`.
- "What editor do I use?" — triggers `memory_search`.
- "Show me the graph around Vim" — triggers `memory_graph_query`.

Ambient context (time/location/weather/file-activity) injects automatically every turn via a `pre_llm_call` hook. Set `location` in `~/.hermes/config.yaml` (or `HERMES_LOCATION`) for location/weather hints.

Insight log: share a realization in chat ("I just realised…") and it's captured as an `insight`-category memory. Browse with `/ilog` or surface a random older one with `/revisit`.

## Troubleshooting

- **"Kuzu graph unavailable"** — graph database failed to initialize. Check write permissions on `~/.hermes/hybrid_memory_kuzu`.
- **Embedding model not found** — ensure `sentence-transformers` is installed and the model is cached. Falls back to text search if embeddings are unavailable.
- **DuckDB writer lock** — another process has the DB open in direct mode. Switch to `shared_service`.

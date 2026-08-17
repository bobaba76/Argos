# Setup Guide

## Prerequisites

- **Python 3.11+**
- **Hermes** (the agent framework this plugin extends)
- **pip** packages installed automatically by the plugin system:
  - `duckdb` — vector + text search storage
  - `kuzu` — relationship graph database
  - `sentence-transformers` — local embedding model (BGE-small-en-v1.5)

## Installation

### Option A: Automatic (via Hermes plugin system)

1. Place the `hybrid_memory_plugin/` directory in your Hermes plugins folder:
   ```
   ~/.hermes/plugins/hybrid_memory/        (Linux/macOS)
   %LOCALAPPDATA%\hermes\plugins\hybrid_memory\  (Windows)
   ```

2. Restart Hermes. The plugin auto-installs its pip dependencies on first load.

3. Verify the plugin loaded:
   ```
   hermes tools
   ```
   You should see `memory_search`, `memory_save`, `memory_update`, `memory_chain`, and `memory_delete` in the tool list.

### Option B: Manual pip install

If auto-install fails, install dependencies manually:
```bash
pip install duckdb kuzu sentence-transformers
```

## Configuration

The config file lives at `~/.hermes/hybrid_memory.json` (created on first run with defaults). Key settings:

```json
{
  "storage_mode": "shared_service",
  "chain_unfold": "auto",
  "chain_unfold_min_similarity": 0.30,
  "chain_unfold_top_k": 3,
  "chain_unfold_query_fallback": true,
  "chain_max_versions": 3,
  "chain_max_inject": 150,
  "graph_aware_retrieval": true,
  "alias_expansion_boost": 0.7,
  "query_expansion_enabled": true,
  "context_aware_retrieval": true
}
```

See `MEMORY_SYSTEM.md` for a full description of each setting.

## Storage modes

- **`shared_service`** (recommended) — a standalone RPC service holds the database; multiple Hermes processes (CLI + gateway + desktop) share one canonical store safely.
- **`direct`** — the plugin opens the DuckDB file directly. Single-process only; use for debugging.

## Embedding model

The plugin uses `BGE-small-en-v1.5` by default, loaded locally (no API calls, no cloud). The model is cached at `~/.hermes/models/bge-small-en-v1.5/`. If the model is unavailable, the system gracefully falls back to text-only search.

## Verifying it works

1. Save a memory:
   ```
   hermes -m "Remember: I use Vim as my primary editor"
   ```

2. Search for it:
   ```
   hermes -m "What editor do I use?"
   ```
   The system should retrieve the memory via vector similarity.

3. Check the graph:
   ```
   hermes tools memory_graph_query --entity "Vim"
   ```
   Should show the entity node and its relationships.

## Troubleshooting

- **"Kuzu graph unavailable"** — the graph database failed to initialize. Check write permissions on `~/.hermes/hybrid_memory_kuzu/`.
- **Embedding model not found** — ensure `sentence-transformers` is installed and the model cached. The system will fall back to text search if embeddings are unavailable.
- **DuckDB writer lock** — another process has the DB open in direct mode. Switch to `shared_service` mode.

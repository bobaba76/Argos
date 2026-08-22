# Setup Guide

## Prerequisites

- **Python 3.11+**
- **Hermes** (the agent framework this plugin extends)
- **pip** packages installed automatically by the plugin system:
  - `duckdb` — vector + text search storage
  - `kuzu` — relationship graph database
  - `sentence-transformers` — local embedding model (BGE-small-en-v1.5)
- *(Optional)* `BAAI/bge-reranker-base` — cross-encoder reranker, downloaded
  on first use only if `reranker_enabled=true` (~420MB).

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
   You should see all fourteen `memory_*` tools: `memory_search`,
      `memory_save`, `memory_update`, `memory_delete`, `memory_chain`,
      `memory_graph_search`, `memory_graph_query`, `memory_candidate_list`,
      `memory_candidate_review`, `memory_restore`, `memory_feedback`,
      `memory_maintenance`, `memory_why_not`, and `memory_fetch_full`.

### Option B: Manual pip install

If auto-install fails, install dependencies manually:
```bash
pip install duckdb kuzu sentence-transformers
```

## Configuration

The config file lives at `~/.hermes/hybrid_memory.json` (created on first run with defaults). A representative subset:

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

A few settings have different display defaults in the desktop UI schema than
in the runtime defaults above: `max_injected_items` (UI `8` vs runtime `20`),
`graph_retrieval_boost` (UI `0.05` vs runtime `0.0`), and `chain_unfold`
(UI `off` vs runtime `auto`). Your saved `hybrid_memory.json` wins either way.

See `MEMORY_SYSTEM.md` for the full config table and a description of every
setting.

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

4. Ambient context (time/location/weather/file-activity) is injected
   automatically every turn via a `pre_llm_call` hook — no setup needed
   beyond setting `location` in `~/.hermes/config.yaml` (or
   `HERMES_LOCATION`) if you want the location/weather hints.

5. Insight log: share a realization in chat ("I just realised…") and it's
   captured as an `insight`-category memory. Browse with `/ilog` or surface
   a random older one with `/revisit`.

## Troubleshooting

- **"Kuzu graph unavailable"** — the graph database failed to initialize. Check write permissions on `~/.hermes/hybrid_memory_kuzu/`.
- **Embedding model not found** — ensure `sentence-transformers` is installed and the model cached. The system will fall back to text search if embeddings are unavailable.
- **DuckDB writer lock** — another process has the DB open in direct mode. Switch to `shared_service` mode.

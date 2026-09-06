# Installation

Argos is a Hermes plugin. The repo (`argos_plugin/`) is canonical; the running Hermes loads an installed copy. This page documents the real install path — verified against `plugin.yaml` and `scripts/deploy.py`.

## Prerequisites

- **Python 3.11+** (Hermes requirement)
- **Hermes** — the agent framework this plugin extends
- Pip dependencies auto-install on first load (declared in `plugin.yaml`):
    - `duckdb==1.5.5`
    - `kuzu==0.11.3`
    - `sentence-transformers==5.6.1`
    - `pandas==2.2.3`
    - `fastapi==0.133.1`
    - `uvicorn==0.41.0`
    - `pydantic==2.13.4`
    - `PyPDF2==3.0.1`
    - `openpyxl==3.1.5`
    - `python-docx==1.2.0`
    - `PyYAML==6.0.3`
- Optional: `BAAI/bge-reranker-base` (~420MB, downloaded on first use if `reranker_enabled=true`)

## Option A: Automatic (via Hermes plugin system)

1. Place `argos_plugin/` in your Hermes plugins folder:

   ```
   %LOCALAPPDATA%\hermes\plugins\hybrid_memory\   (Windows)
   ~/.hermes/plugins/hybrid_memory/               (Linux/macOS)
   ```

2. Restart Hermes. Pip dependencies install on first load (from `plugin.yaml`).
3. Verify: `hermes tools` — you should see the `memory_*` tools.

## Option B: Manual pip install

If you want to pre-install dependencies (e.g. in a venv without network on first load):

```bash
pip install duckdb==1.5.5 kuzu==0.11.3 sentence-transformers==5.6.1 \
  pandas==2.2.3 fastapi==0.133.1 uvicorn==0.41.0 pydantic==2.13.4
```

## Option C: Repo → live sync (for developers)

If you cloned the repo and want to sync changes to the live plugin install, use the deploy script. **This is for development only — never run it against a production Hermes instance without reading [`SYNC_HANDOFF.md`](https://github.com/bobaba76/Argos/blob/master/SYNC_HANDOFF.md).**

```bash
# Check drift between repo and live install (default mode)
python scripts/deploy.py --check

# Copy changed runtime files to the live install
python scripts/deploy.py copy

# Atomic swap (staging + rename) — requires Hermes stopped
python scripts/deploy.py --atomic-swap

# Roll back to the previous versioned live dir
python scripts/deploy.py --rollback
```

The deploy script syncs only runtime modules (top-level `*.py` + `plugin.yaml`). It never touches live-only artifacts (the DuckDB database, Kùzu graph, config JSON, skills, eval, tests, state files).

## Plugin manifest

The plugin manifest (`argos_plugin/plugin.yaml`) declares the plugin name, version, dependencies, and hooks:

```yaml
name: argos
version: 1.0.0
description: "Argos — local-first shared memory service..."
pip_dependencies:
  - duckdb==1.5.5
  - kuzu==0.11.3
  - sentence-transformers==5.6.1
  - pandas==2.2.3
  - fastapi==0.133.1
  - uvicorn==0.41.0
  - pydantic==2.13.4
  - PyPDF2==3.0.1
  - openpyxl==3.1.5
  - python-docx==1.2.0
  - PyYAML==6.0.3
hooks:
  - on_turn_start
  - sync_turn
  - on_session_end
  - on_session_switch
```

## Storage modes

Argos supports two storage modes (configured in `hybrid_memory.json`):

| Mode | When to use |
|------|-------------|
| `shared_service` (default) | Production. An RPC service owns the DuckDB file; multi-process safe. |
| `direct` | Diagnostics or single-process testing only. The plugin opens DuckDB directly. |

See [Configuration](configuration.md) for the full settings reference.

## Verify the install

After installation:

1. `hermes tools` — confirm the `memory_*` tools appear.
2. Start a conversation and tell Argos a fact. Check the review queue (`memory_candidate_list`).
3. Approve a candidate, then search for it. It should surface.
4. Optional: start the REST server and `curl /v1/health` — should return `{"status": "ok"}`.

## Troubleshooting

- **`ModuleNotFoundError: No module named 'duckdb'`** — pip dependencies didn't install. Restart Hermes (it installs on first load), or install manually (Option B).
- **`memory_*` tools don't appear** — check the Hermes log for import errors. Usually a missing or version-mismatched dependency.
- **Vector search returns nothing** — the embedding model (`BAAI/bge-small-en-v1.5`, ~130MB) downloads on first use. If you're offline, it falls back to text-only search. Pre-download it or set `local_embedding_model` to a model you already have.
- **Database locked** — another Hermes process is holding the DuckDB file. Use `shared_service` mode (default) and ensure only one memory service is running.

## Next steps

- [Quickstart](quickstart.md) — five-minute walkthrough.
- [Configuration](configuration.md) — every setting.
- [API reference](api/index.md) — MCP and REST.

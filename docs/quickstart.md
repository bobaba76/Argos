# Quickstart

Get Argos running in five minutes. This assumes you already have [Hermes](https://github.com/NousResearch/hermes-agent) installed. If not, see [Installation](installation.md) for the full path.

## 1. Install the plugin

Copy `argos_plugin/` into your Hermes plugins directory:

```
%LOCALAPPDATA%\hermes\plugins\hybrid_memory\   (Windows)
~/.hermes/plugins/hybrid_memory/               (Linux/macOS)
```

Restart Hermes. Pip dependencies (declared in `plugin.yaml`) install on first load:

- `duckdb`, `kuzu`, `sentence-transformers`, `fastapi`, `uvicorn`, `pydantic`, `pandas`
- Optional: `PyPDF2`, `openpyxl`, `python-docx` (for structured ingestion)

## 2. Verify the tools loaded

```bash
hermes tools
```

You should see the `memory_*` tools listed. If they don't appear, check the Hermes log for import errors (usually a missing dependency).

## 3. Tell Argos something

In a Hermes conversation:

> I live in Cape Town and I prefer dark mode for coding.

Argos extracts facts and creates *proposals* — they enter the review queue, they don't become active memory until you approve them.

## 4. Search what Argos remembers

> What do you know about where I live?

Argos searches the store (vector + keyword fusion) and injects the relevant memories as context for the answer.

## 5. Review the proposal queue

Use the `memory_candidate_list` tool (or the CLI review tool) to see pending proposals. Approve the ones you want to keep; reject the rest. Approved proposals become active memories with full provenance.

## 6. Optional: start the external API

Argos works with **any MCP or REST client** — not just Hermes. The MCP and
REST transports boot **read + write enabled** by default (spec-11): propose
candidates, save/update memories, and manage collections without any env-var
setup.

**From a source checkout?** Add the repo root and `argos_plugin/` to `PYTHONPATH` first — or run the server scripts directly from `argos_plugin/`. See [Running from a source checkout](integration.md#running-from-a-source-checkout).

```bash
# REST server (loopback-only, token-authenticated, read + write by default)
ARGOS_REST_TOKEN=<your-token> python -m argos_plugin.rest_server --home <hermes-home> --port 8732

# MCP server (stdio, read + write by default)
python -m argos_plugin.mcp_server --home <hermes-home>
```

**Trust model:**
| Class | Operations | Default | When to turn off |
|-------|-----------|---------|-------------------|
| A (propose) | `memory_propose`, `ingest` | ON | Conservative deployments: `ARGOS_API_READ_ONLY=1` |
| C (direct write) | `memory_save`, `memory_update`, collection writes | ON (loopback) | Non-loopback: `ARGOS_API_NO_LOOPBACK=1` |
| B (approve) | `review_candidate` | OFF (model denied) | Never for model-driven clients |

**Read-only escape hatch:** `ARGOS_API_READ_ONLY=1` restores the spec-09
read-only default for conservative or shared deployments.

See [Integration guide](integration.md) for MCP client registration JSON
(OpenWebUI, Claude Desktop, Cursor) and [API reference](api/index.md) for
the full operation set.

## Next steps

- [Configuration](configuration.md) — tune retrieval, embeddings, graph, and chains.
- [API reference](api/index.md) — MCP and REST endpoints.
- [Tuning](tuning.md) — embedder and reranker guidance.

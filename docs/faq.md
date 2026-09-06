# FAQ

## General

### Is Argos cloud-based?

No. Argos is local-first. The memory database (DuckDB + Kùzu graph) lives on your machine. Embeddings run locally via sentence-transformers. The external API binds to `127.0.0.1` only. There are no cloud calls, no telemetry, no third-party embeddings API.

Set `local_only: true` in `hybrid_memory.json` to additionally gate plugin-owned LLM calls (extraction, review, distillation) to local-only models.

### What's the trust model?

Nothing becomes a memory silently. Extraction produces *proposals* — pending until you approve them. `memory_save` is the explicit exception: it writes directly to active memory (an intentional agent action). Updates chain versions instead of overwriting. Cleanup quarantines instead of deleting. Distillation proposes but never writes. Provenance and grounding are tracked per record.

See the [README trust model section](https://github.com/bobaba76/Argos/blob/master/README.md#trust-model) for the full statement.

### What license is Argos under?

Business Source License 1.1 (BSL 1.1): free for personal and non-production use; production or commercial use requires a license. Converts to Apache 2.0 on August 21, 2030. Full terms: [`LICENSE.md`](https://github.com/bobaba76/Argos/blob/master/LICENSE.md).

## Storage

### What database does Argos use?

DuckDB for the memory store (records, candidates, audit, tombstones, receipts). Kùzu for the relationship graph (entities, relations, aliases). DuckDB is the source of truth; Kùzu is a derived graph index.

### What storage mode should I use?

`shared_service` (default) for production — an RPC service owns the DuckDB file, multi-process safe. `direct` for diagnostics or single-process testing only.

### Where is my data stored?

In the Hermes home directory: `hybrid_memory.duckdb` (DuckDB) and `hybrid_memory_kuzu` (Kùzu graph). On Windows: `%LOCALAPPDATA%\hermes\`. On Linux/macOS: `~/.hermes/`.

## Retrieval

### Why didn't my memory surface?

Use the `memory_why_not` MCP tool or the `POST /v1/memory/explain-retrieval` REST endpoint. It returns the rank, per-stage diagnostics (vector/text/status/scope), and human-readable reasons. Deterministic, read-only, zero-LLM.

### How does versioning work?

When a fact is updated, Argos chains a new version onto the old one. The old version is marked superseded (not deleted). `memory_fetch_history` shows the full chain. `chain_unfold` (default `auto`) injects a compact version arc on change-intent queries.

### What happens when two memories conflict?

If `conflict_surfacing` is `true` (default), Argos injects an explicit conflict note when two active records disagree on the same subject. The answerer surfaces the disagreement instead of smoothing it.

### Can I use a different embedding model?

Yes. Set `local_embedding_model` in `hybrid_memory.json` to any sentence-transformers-compatible model. The store tracks `embedding_dim` per record, so mixed-dimension records are detected and a text fallback is used for incompatible queries (see #286). Use the re-embedding CLI to backfill existing records.

## API

### Can I access Argos from a non-Hermes agent?

Yes. Start the REST server (`python -m argos_plugin.rest_server --home <hermes-home>`) or the MCP server (`python -m argos_plugin.mcp_server --home <hermes-home>`) and connect from any HTTP or MCP client. See [Integration guides](integration.md).

### Can I write memories over the API?

You can *propose* memories over the API (`memory_propose` MCP tool, or the proposal tier). Proposals enter the review queue — they don't become active memory until a human approves them. This is by design: the API is a trust boundary, not a trusted writer.

### Why does the REST server only bind to localhost?

Security. Argos is local-first. The REST server is for same-machine integrations (scripts, local agents). If you need remote access, use an SSH tunnel — do not change the bind address to `0.0.0.0`.

## Troubleshooting

### The `memory_*` tools don't appear in Hermes

Check the Hermes log for import errors. Usually a missing or version-mismatched dependency. Try a manual pip install (see [Installation](installation.md#option-b-manual-pip-install)).

### Vector search returns nothing

The embedding model (`BAAI/bge-small-en-v1.5`, ~130MB) downloads on first use. If you're offline, it falls back to text-only search. Pre-download the model or set `local_embedding_model` to a model you already have.

### Database is locked

Another Hermes process is holding the DuckDB file. Use `shared_service` mode (default) and ensure only one memory service is running.

### The REST server won't start

It needs a token. Set `ARGOS_REST_TOKEN` or `rest_token` in the Hermes home config. The server refuses to start without one (fail-closed).

## More help

- [README](https://github.com/bobaba76/Argos/blob/master/README.md) — the canonical short-form reference.
- [CONFIG_REFERENCE.md](https://github.com/bobaba76/Argos/blob/master/CONFIG_REFERENCE.md) — every setting.
- [MEMORY_SYSTEM.md](https://github.com/bobaba76/Argos/blob/master/MEMORY_SYSTEM.md) — how the system works under the hood.
- [REINSTALL.md](https://github.com/bobaba76/Argos/blob/master/REINSTALL.md) — reinstall, migration, graph rebuild.
- [Issues](https://github.com/bobaba76/Argos/issues) — bug reports and feature requests.

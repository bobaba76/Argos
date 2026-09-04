# Argos

Persistent memory for AI agents, on your own machine.

## What it is

Argos is a memory layer for AI agents. It ships as a Hermes plugin, and it
speaks MCP and REST — so an MCP-capable agent or a plain script can read the
same store without running inside Hermes.

What it buys you is continuity. Your agent learns something in one session and
still knows it three weeks later, including what changed in between: updating a
fact chains a new version onto the old one instead of overwriting it, so
"what did this used to say?" has an answer.

Your data sits in files in your agent's home directory — a DuckDB store, the
embeddings, and a Kùzu relationship graph. No hosted memory service. Embeddings
run on your own hardware (BAAI/bge-small-en-v1.5, under 130MB). Extraction,
candidate review, and distillation call your configured cloud model; there is
no local-LLM runtime yet.

## What it does

- **Recall across sessions.** "What did we settle on for the budget last week?"
  comes back with the decision and the versions behind it.
- **Hybrid search.** Vector and keyword legs fused with RRF, date-anchored
  temporal handling, and an optional GPU cross-encoder reranker.
- **A relationship graph.** Entities, relations, and aliases in Kùzu, for
  multi-hop questions like "who works with my sister?"
- **Ambient context.** Time, location, weather, and recent file activity are
  injected each turn, so the agent doesn't open every conversation blind.
- **Insight capture.** "I just realised…" moments are logged verbatim. Browse
  them with `/ilog`, resurface one with `/revisit`, exclude a claim with `/neg`.
- **Review before anything sticks.** Turns are mined for facts, but the output
  is a proposal in a queue until you approve it. `memory_save` is the
  exception — an explicit tool call, not passive ingestion. Daily distillation
  proposes patterns under a cost cap and never writes on its own.
- **Cleanup you can undo.** Maintenance quarantines stale and duplicate records
  instead of destroying them. Deletes are chain-aware: multi-version records
  promote the predecessor, single-version records get tombstoned so they can't
  quietly reappear.
- **As-of queries.** Records carry `valid_from` and `valid_to`. Retrieval shows
  the current view by default, or a past state on request.
- **Multitenant cells.** Per-tenant stores behind one shared service, isolated
  and concurrency-gated.

## Tools

| Group | Tools |
|---|---|
| Store and search | `memory_search`, `memory_save`, `memory_fetch_full` |
| Version chains | `memory_update`, `memory_delete`, `memory_chain` |
| Graph | `memory_graph_search`, `memory_graph_query` |
| Review and restore | `memory_candidate_list`, `memory_candidate_review`, `memory_restore` |
| Feedback and maintenance | `memory_feedback`, `memory_maintenance` |
| Diagnostics | `memory_why_not`, `memory_tombstones`, `memory_tombstone_purge` |

## External API

A read tier over two transports. Both bind to loopback only, require a bearer
token, and go through one facade (auth → ACL → validation → audit) with an
explicit operation allowlist — no raw RPC passthrough, so internal operations
like shutdown, backup, and purge are never reachable.

```bash
# REST — /v1/health, /v1/ready, /v1/capabilities, POST /v1/memory/search,
#         /v1/memories/{id}, /v1/memories/{id}/history
ARGOS_REST_TOKEN=<token> python -m argos_plugin.rest_server --home <hermes-home> --port 8732

# MCP (stdio) — memory_search, memory_fetch, memory_fetch_history, memory_capabilities
python -m argos_plugin.mcp_server --home <hermes-home>


Writes over the API go through the same facade when they land. Not yet.

## Quick start

1. Copy `argos_plugin/` into your Hermes plugins directory
   (`%LOCALAPPDATA%\hermes\plugins\hybrid_memory` on Windows).
2. Restart Hermes. Dependencies install on first load.
3. Run `hermes tools` and confirm 16 `memory_*` tools appear.
4. Tune it in `hybrid_memory.json` or the settings UI (Memory → Argos).

Full walkthrough: [SETUP_GUIDE.md](SETUP_GUIDE.md).

## Where it stands

On LongMemEval_S it scores 89.8% (449/500) — self-measured, with a
GLM-5.3-flash answerer and a gpt-4o judge. Protocol, dataset hashes, judged
outputs, and the caveats that go with the number:
[BENCHMARK_REPRODUCIBILITY.md](eval/repro/BENCHMARK_REPRODUCIBILITY.md).

Known limits:

- The external API reads only. No remote writes yet.
- Extraction, review, and distillation need your cloud model. Embeddings are
  local; the LLM isn't.
- The reranker wants a GPU. Without one, search falls back to
  similarity-only.
- Developed and tested against the maintainer's Hermes build. It uses stock
  plugin APIs, but a stock upstream build hasn't been through the test suite.

## Docs

- [SETUP_GUIDE.md](SETUP_GUIDE.md) — install, start to finish
- [MEMORY_SYSTEM.md](MEMORY_SYSTEM.md) — how it works underneath
- [CONFIG_REFERENCE.md](CONFIG_REFERENCE.md) — every setting and default
- [REINSTALL.md](REINSTALL.md) — reinstall, migration, graph rebuild
- [CLAIMS-AUDIT.md](CLAIMS-AUDIT.md) — each claim mapped to its evidence

## License

BSL 1.1 — free for personal and non-production use; commercial or production
use needs a license. Converts to Apache 2.0 on 21 August 2030.
[LICENSE.md](LICENSE.md)
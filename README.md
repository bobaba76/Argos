# Argos

Persistent memory for AI agents, running on your own machine.

## What Argos is

Argos gives your AI agent a memory that actually sticks. It's a plugin for the Hermes agent framework (a standalone server is also on the table), and it remembers what your agent learns across sessions instead of forgetting the moment the conversation ends. Facts get versioned, not deleted, so you can always see how something changed. Everything is searchable, and everything is auditable.

Your data stays where you can see it: records, embeddings, and the relationship graph are all local files in your agent's home directory. No hosted memory vendor, no data leaving your machine for storage. Embeddings run locally too, using BGE-small-en-v1.5, a compact model under 130MB. The one honest exception: extraction, candidate review, and query expansion currently call your configured cloud model. There's no native local-LLM runtime yet.

## What it does for you

- Facts persist across sessions. Ask "what did we agree about the budget last week?" and get a real answer, not a shrug.
- Changes are versioned, never erased. Ask "what was the old value before I updated it?" and Argos shows you the chain.
- Semantic search fuses vector and keyword matching with RRF, so "find that thing about the car service" actually finds it.
- A relationship graph (built on Kùzu) answers multi-hop questions like "who works with my sister?"
- Ambient context — time, weather, location, recent files — gets injected automatically every turn, so your agent isn't starting cold.
- Insight capture logs "I just realised..." moments verbatim. Browse them with /ilog, bring them back with /revisit.
- Once a day, cost-capped, Argos proposes patterns distilled from accumulated records. Nothing lands in memory without your approval.
- A read-tier external API, over MCP (stdio) and REST, lets any MCP-capable agent or script search, fetch, and inspect history against the same store, behind auth, ACL, validation, and audit.
- Multitenant cells keep per-tenant stores separate behind a shared service.
- Temporal as-of queries: records carry valid_from and valid_to, retrieval defaults to the current view, and you can ask for a past state or include closed records.
- Cleanup is reversible. Maintenance quarantines stale or duplicate records rather than destroying them, and explicit deletes are chain-aware — promoting the predecessor, quarantining the middle, tombstoning single versions so they can't quietly reappear.

## How it earns trust

Nothing becomes a memory silently. Every turn gets mined for facts, but the result is a proposal sitting in a queue until you approve it. The one exception is memory_save, which is an explicit, intentional action by the agent, not a background guess.

Updates chain versions instead of overwriting history, so "what changed?" is always answerable. Cleanup quarantines instead of deleting, and explicit deletes are chain-aware with tombstones, so a deleted record can't be silently recreated.

Every capability claim in this README is backed by a committed, re-runnable artifact in CLAIMS-AUDIT.md. A reproducibility gate re-derives every headline number and fails the build if the numbers drift.

## The tools

Argos ships 16 memory_* tools. The MCP and REST surface exposes the read subset.

| Group | Tools |
|---|---|
| Store and search | memory_search, memory_save, memory_fetch_full |
| Version chains | memory_update, memory_delete, memory_chain |
| Graph | memory_graph_search, memory_graph_query |
| Review and restore | memory_candidate_list, memory_candidate_review, memory_restore |
| Feedback and maintenance | memory_feedback, memory_maintenance |
| Diagnostics | memory_why_not, memory_tombstones, memory_tombstone_purge |

## What it can't do yet

- The external API is read-tier only. No remote writes yet.
- LLM calls go through your configured cloud model. There's no native local-LLM runtime yet.
- The optional GPU reranker (BGE-reranker-base) needs a GPU, or it falls back to similarity-only search.
- Argos has been tested on the maintainer's build of Hermes. A stock upstream build hasn't been through the full suite yet.
- Argos ships as a Hermes plugin, but the external API (MCP and REST) means any MCP-capable agent or custom script can use the same store without running inside Hermes.

## Quick start

1. Drop the argos_plugin/ folder into your Hermes plugins directory.
2. Restart Hermes. Dependencies install automatically.
3. Check `hermes tools` and confirm you see 16 memory_* tools.
4. Configure settings in hybrid_memory.json or the settings UI.

Full walkthrough in SETUP_GUIDE.md.

## Docs

- SETUP_GUIDE.md — full install walkthrough
- MEMORY_SYSTEM.md — how it works under the hood
- CONFIG_REFERENCE.md — every setting, default, and description
- REINSTALL.md — reinstall, migration, graph rebuild
- CLAIMS-AUDIT.md — every claim mapped to evidence
- BENCHMARK_REPRODUCIBILITY.md — how to re-derive every headline number

## License

BSL 1.1. Free for personal and non-production use. Commercial or production use needs a license. Converts to Apache 2.0 on August 21, 2030.
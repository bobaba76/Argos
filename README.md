# Argos

Persistent memory for AI agents, on your own machine. A Hermes plugin: hybrid vector + graph store with local embeddings.

## Capabilities

- **Facts persist across sessions.** Tell it once, ask weeks later.
- **Changes are versioned, not erased.** Updating a fact chains a new version onto the old one. Ask "what changed?" and get the history.
- **Semantic search.** Vector + keyword fusion (RRF) finds memories by meaning, not just exact words.
- **Relationship graph.** A Kùzu graph of entities, relations, and aliases powers multi-hop queries ("who works with my sister?").
- **Ambient context.** Time, location, weather, and recent file activity inject every turn via a `pre_llm_call` hook.
- **Insight capture.** "I just realised…" moments are logged verbatim. Browse with `/ilog`, resurface with `/revisit`, store exclusions with `/neg <claim>`.
- **Gated distillation.** Once a day, cost-capped, Argos proposes distilled patterns from accumulated records. Nothing lands in memory without your approval.
- **Reversible cleanup.** Stale or duplicate memories are quarantined, not deleted. Restore if needed.

## Trust model

Nothing becomes a memory silently. Every turn is mined for facts (regex first, LLM fallback), but the output is a *proposal* — pending until you approve it. Updates chain versions instead of overwriting. Cleanup quarantines instead of deleting. Distillation proposes but never writes. Every feature is gated by a measurement in the eval harness; ideas that didn't help were turned off.

## Tools

Sixteen `memory_*` tools, grouped:

| Group | Tools |
|-------|-------|
| Store & search | `memory_search`, `memory_save`, `memory_fetch_full` |
| Version chains | `memory_update`, `memory_delete`, `memory_chain` |
| Graph | `memory_graph_search`, `memory_graph_query` |
| Review & restore | `memory_candidate_list`, `memory_candidate_review`, `memory_restore` |
| Feedback & maintenance | `memory_feedback`, `memory_maintenance` |
| Diagnostics | `memory_why_not`, `memory_tombstones`, `memory_tombstone_purge` |

## Numbers

| Metric | Result | Protocol |
|--------|--------|----------|
| LongMemEval_S (best config) | 89.8% (449/500) | GLM-5.3-flash answerer, gpt-4o judge |
| LongMemEval_S (baseline) | 70.4% (352/500) | gpt-4o judge, default answerer |
| Chain-unfold (change-intent) | 93% recall / 93% precision | canonical eval harness |
| Temporal questions | 88.7% (118/133) | full-bank, text-leg hardening |
| Recall@96 | 99.6% | answer-bearing memories reaching top-96 |

Protocols, dataset SHA-256, per-category denominators, model versions, prompts, exact commands, and judged outputs: [eval/repro/BENCHMARK_REPRODUCIBILITY.md](eval/repro/BENCHMARK_REPRODUCIBILITY.md).

## What it can't do

Memory data, embeddings, and graph live locally as flat files — no hosted vendor. LLM calls for extraction, review, and distillation go through your configured cloud model; no native local-LLM support yet. Only embeddings are offline.

## Quick start

1. Copy `argos_plugin/` to your Hermes plugins directory (`%LOCALAPPDATA%\hermes\plugins\hybrid_memory` on Windows).
2. Restart Hermes.
3. Run `hermes tools` — confirm the 16 `memory_*` tools appear.
4. Configure in `hybrid_memory.json` or the settings UI (Memory → Argos).

Full walkthrough: [SETUP_GUIDE.md](SETUP_GUIDE.md).

Argos is developed and tested on the maintainer's build of Hermes. It uses only stock plugin APIs (memory provider, pre-call hook, user-context injection), so it should work on a plain install, but a stock upstream build hasn't been through the test suite yet.

## License

Business Source License 1.1 (BSL 1.1): free for personal and non-production use; production or commercial use requires a license. Converts to Apache 2.0 on August 21, 2030. Full terms: [LICENSE.md](LICENSE.md).

## More docs

- [CONFIG_REFERENCE.md](CONFIG_REFERENCE.md) — every setting, default, and description
- [MEMORY_SYSTEM.md](MEMORY_SYSTEM.md) — how the system works under the hood
- [REINSTALL.md](REINSTALL.md) — reinstall, migration, graph rebuild
- Tests: `python -m pytest tests/ -v` (run from `argos_plugin/`)

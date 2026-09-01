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
- **Reversible cleanup.** Maintenance and consolidation quarantine stale or duplicate memories, not delete them. Explicit deletion (`memory_delete`) is chain-aware: multi-version records promote the predecessor or quarantine the middle version; single-version records are hard-deleted and tombstoned against re-creation.

## Trust model

Nothing becomes a memory silently. Every turn is mined for facts (regex first, LLM fallback), but the output is a *proposal* — pending until you approve it. `memory_save` is the explicit exception: it writes directly to active memory, an intentional agent action rather than passive ingestion. Updates chain versions instead of overwriting. Cleanup quarantines instead of deleting; explicit `memory_delete` on a single-version record hard-deletes and tombstones it (re-creation is blocked until the tombstone is purged). Distillation proposes but never writes. Every capability listed above is measured by the eval harness or structurally verified against source — see [Verification](#verification).

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

## Verification

Every number and capability statement above is backed by a committed, re-runnable artifact — nothing is quoted without evidence.

- **Claims audit** — [CLAIMS-AUDIT.md](CLAIMS-AUDIT.md) maps every claim to its evidence and separates three tiers: *measured* (committed judged artifacts), *structural* (checked against source), and *aspirational* (not claims yet). It is updated whenever a claim changes.
- **Reproducibility gate** — the Numbers table is re-derived from committed artifacts by `./eval/repro/verify_repro.sh`, which fails on any drift. Run it before quoting a number.
- **Test suite** — 1007 tests across 50+ modules (as of 2026-09-01), covering the store, retrieval, security gates, and eval harness; runs hermetically on a fresh clone without a live Hermes runtime. The gate forces hermetic mode (`ARGOS_HERMETIC_TESTS=1` in the runner) so unmocked LLM-path tests can never make real calls, even in venvs that resolve the Hermes runtime. Run it in a visible, live-updating window (GPU venv + bounded parallel workers with the shared-service grouping, per #98): `powershell -File scripts/run_tests_visible.ps1`. Manual equivalent from `argos_plugin/`: `"C:/Users/michael/AppData/Local/hermes/hermes-agent/venv-cuda/Scripts/python.exe" -m pytest tests/ -q -n 4 --dist loadgroup`.

Honest boundaries that travel with the claims:

- Benchmark numbers are self-measured on the maintainer's stack and answerer-conditional (the 89.8% headline uses a GLM-5.3-flash answerer with a gpt-4o judge); small-n bands are indicative.
- Some measurements are internal-only, not yet banked as committed artifacts — they are deliberately excluded from the public claim set.
- The plugin is developed and tested on the maintainer's build of Hermes. It uses only stock plugin APIs (memory provider, pre-call hook, user-context injection), but a stock upstream build hasn't been through the test suite yet.

## What it can't do

Memory data, embeddings, and graph live locally as flat files — no hosted vendor. LLM calls for extraction, review, and distillation go through your configured cloud model; no native local-LLM support yet. Only embeddings are offline.

## Quick start

1. Copy `argos_plugin/` to your Hermes plugins directory (`%LOCALAPPDATA%\hermes\plugins\hybrid_memory` on Windows).
2. Restart Hermes.
3. Run `hermes tools` — confirm the 16 `memory_*` tools appear.
4. Configure in `hybrid_memory.json` or the settings UI (Memory → Argos).

Full walkthrough: [SETUP_GUIDE.md](SETUP_GUIDE.md). Compatibility boundary: see [Verification](#verification).

## License

Business Source License 1.1 (BSL 1.1): free for personal and non-production use; production or commercial use requires a license. Converts to Apache 2.0 on August 21, 2030. Full terms: [LICENSE.md](LICENSE.md).

## More docs

- [CONFIG_REFERENCE.md](CONFIG_REFERENCE.md) — every setting, default, and description
- [MEMORY_SYSTEM.md](MEMORY_SYSTEM.md) — how the system works under the hood
- [REINSTALL.md](REINSTALL.md) — reinstall, migration, graph rebuild

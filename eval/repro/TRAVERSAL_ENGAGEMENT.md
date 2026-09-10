# Traversal engagement proof — 2026-09-10

**Scope:** does graph traversal engage at all on current data, or is the arm dormant?
(#407; follow-up to the withdrawn #139 verdict, #364.)

This is the $0 half of the traversal question: a regex-graph build (`use_llm=False`)
has no LLM spend. The typed-graph (LLM-supplemented) A/B remains deferred — tracked
in #424.

## Protocol

- Harness: `argos_plugin/eval/run_eval_provider.py` @ `9f9e3bf` (working tree).
- Snapshot: `20260906_162404_018897_1d9c5253` (current reviewed baseline pair, with gold_v3).
- Eval set: `eval_set_traversal.json` — 300-query stratified sample; arms `baseline` + `traversal_on`.
- Graph build: regex-only (`build_snapshot_graph(use_llm=False)`) — the "current graph" variant.
- Engagement is measured through the production-mirroring gates
  (`KuzuGraphStore.traversal_engagement` — same gates as `traversal_memory_ids`);
  per-query counters aggregated by `summarize_engagement` (`engaged` = traversal
  returned >=1 id for that query).
- Cost: $0 LLM (local embeddings, GPU). Wall time ~23 min for both arms including
  per-arm graph builds.

## Result — engagement counters (`traversal_on`)

| counter | value |
| --- | --- |
| n_queries | 300 |
| seeds_resolved | 298 |
| non_concept_seeds | 131 |
| **engaged** | **97** |
| **engaged_fraction** | **0.3233** |

Metric deltas (`traversal_on` - `baseline`): nDCG@5/10/20 **+0.0014**, MRR **+0.0018**;
precision/recall unchanged (0.0). Full per-metric table: `traversal_engagement_summary.json`.

## Verdict

- **Not dormant on current data.** 97/300 queries (32.3%) clear the gates and receive
  traversal-sourced candidate ids on the *regex-only* graph — the strong reading of
  "a regex graph can never fire traversal" does not hold for the 2026-09-06 snapshot.
- **Contribution in this arm configuration ~nil**: <=0.2pp metric movement; ranking and
  recall are carried by the non-traversal path.
- **Still unmeasured:** the typed-graph (production-like hybrid) variant and the
  production injection path — the A/B remains deferred (#424).

## Notes & caveats

- Not directly comparable to the 2026-09-02 run (0826-era snapshot, prior pipeline):
  baseline MRR moved 0.90 -> 0.5169 between runs (different snapshot composition,
  11 further days of store writes, pipeline changes since). Within-run arm-vs-arm
  comparison is the valid one here.
- `id_coverage`: 297/300 eval relevant ids present in the snapshot (3 since TTL-expired).
- 9 search events during the run hit stale-embedding records (vector dimension mismatch,
  text fallback). Snapshot hygiene follow-up candidate — consider a re-embed pass before
  the next snapshot freeze.
- Raw per-query outputs (contain query text) stay local under the gitignored
  `argos_plugin/eval/bench/traversal_ab/`; this summary is the committed aggregate.

## Re-run

```
python argos_plugin/eval/run_eval_provider.py <snapshot.duckdb> \
  argos_plugin/eval/bench/rerank_ab/eval_set_traversal.json <out_dir>
```

Requires the maintainer env: Python with the repo + hermes-agent on `sys.path`,
local model cache (embedder), GPU recommended.

## Change history

- 2026-09-10 — this run: counters banked; regex-graph arm engages (97/300). (#407)
- 2026-09-08 — #364: the #139 "measured-and-flat" verdict withdrawn; engagement counters
  added to the harness.
- 2026-09-02 — #139 A/B (0826 snapshot): all 10 metrics byte-identical; vacuous.

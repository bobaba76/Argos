# Reranker A/B — measured result (2026-08-26)

Deterministic ranking A/B: **baseline (no cross-encoder) vs reranker_on**
(`bge-reranker-base`, `reranker_top_n=50`, CE blend 0.8/0.2) on a 300-query
stratified sample (seed 42), evaluated
with `argos_plugin/eval/run_eval_provider.py` against the frozen snapshot
`20260826_170626_224920_4d612e0f`.

## Result (committed aggregate)

`reranker_ab_summary.json` (this directory). Headline deltas:

| Metric | baseline | reranker_on | Δ |
|---|---|---|---|
| MRR | 0.9058 | 0.9372 | **+0.0314** |
| nDCG@20 | 0.9217 | 0.9462 | **+0.0245** |
| R@5 | 0.9567 | 0.9633 | +0.0066 |
| R@10 | 0.9700 | 0.9700 | 0.0000 |
| R@20 | 0.9700 | 0.9733 | **+0.0033** |
| nDCG@5 | 0.9174 | 0.9432 | +0.0258 |
| P@5 | 0.1913 | 0.1927 | +0.0014 |

The committed artifact is the aggregate summary and this protocol.

## Verdict

- **Ranking lever, NOT a recall lever.** MRR +3.1pp and nDCG +2.5pp are
  real; recall@20 only +0.3pp because the true miss gap sits OUTSIDE the
  50-pool (dual-encoder misses ~2.7% of relevant memories before the reranker
  ever sees them).
- **k-descent stays locked at 96.** The recall ceiling is pool depth, not
  ranking, so shrinking k would give up retrieval coverage the reranker
  cannot win back.
- **Latency blocks prod ship.** CPU-only torch here measures ~8-13 s/query
  (rerank arm) — unviable. Enabling the reranker in production requires a
  CUDA-torch install into the service venv first, then a latency re-measure
  (also flips the "CPU ~300ms" doc promise).
- Reranker stays `false` in prod config until the CUDA prerequisite lands.

## Re-run steps

```
1. Build eval set: from paraphrase_all.jsonl (dedupe keep-last; ~995 natural
   queries) → 300-strat sample, seed 42 → eval_set_300.json.
2. python eval/run_eval_provider.py <snapshot>/hybrid_memory.duckdb \
     eval_set_300.json <outdir>   # both arms, ~45 min CPU
3. Compare <outdir>/provider_summary.json arms.
```

## Change history

- 2026-08-26 — A/B measured (300-strat seed 42). Committed summary + protocol.
- 2026-08-27 — claims audit; graduated from "finding" to "claim".
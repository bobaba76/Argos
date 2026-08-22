# ANN swap protocol (exact scan -> approximate index)

One-page, measured recipe for swapping the vector-search engine behind the
`Retriever` seam (`store.set_retriever(...)`) without regressing recall.
Read this BEFORE touching an ANN library.

## Current baseline (measured 2026-08-22, ~1k records, live service)

| Component | Before | After |
|---|---|---|
| Full live search (96 results) | 1.2 - 2.1 s | ~0.5 s |
| Pipeline: embed + scan + RRF + importance | 0.75 - 1.1 s | 0.21 - 0.26 s |
| Retrieval recording (bookkeeping) | 0.5 - 1.3 s | ~0.29 s |
| Brute-force cosine scan | **~1.0 s** (!) | **~0.01 s** |

The scan was the hidden monster: `list_cosine_similarity(embedding, ?::DOUBLE[])`
with a Python-list parameter binds through an interpreted per-row path
(~1ms/row — ~1s at 1k rows). Passing the vector as a string and casting it as a
fixed-size constant (`CAST(? AS DOUBLE[384])`) lets the planner materialize it
once and scan natively. Measured scale: ~7-14ms at 1k-5k rows, linear ~280ms at
100k. Bit-identical ranking verified (same IDs, same similarity values)
before and after.

Pre-ANN cheap wins (shipped 2026-08-22):
1. String-cast vector constant in `_vector_search_raw` AND the write-path
   semantic dedup (the dedup scan was equally slow).
2. `_record_retrieval` batched into one `UPDATE ... WHERE memory_id IN (...)`.
3. Query embeddings cached (bounded FIFO) in `LocalEmbedder.embed(is_query=True)`.
4. This protocol. Next candidate when the store grows: a real lexical index
   (BM25 / DuckDB FTS) for the `ILIKE %token%` leg — the ILIKE leg can never
   use an index and its scan cost grows linearly with store size.

## When ANN is justified (trigger, not vibes)

Swap to ANN only when the brute-force scan is measurably dominant:

- Probe: time the scan component separately (query embed + scan, no recording,
  no text leg) at p95 over 20 varied queries against a live store copy.
- Gate: scan p95 > ~50 ms — with the string-cast fix that lands around
  ~100k records (≈280ms at 100k; prior to the fix it was already ~1s at 1k).
  Below that, ANN adds approximate-noise for speed you do not need.
- Secondary trigger: a hard injection-latency budget you have actually set
  and are exceeding end-to-end.
- If the scan *does* become dominant before you want ANN, the intermediate
  step is a numpy exact scan (fetch the embedding column once, `M @ q` in
  numpy, argpartition top-k): still exact, ~65ms at 5k rows, and keeps every
  retrieval-semantics guarantee intact.

## A/B procedure (non-negotiable gates)

1. **Baseline (exact engine)**: run the LongMemEval harness
   (`eval/run_eval.py`) AND the short-fact recall check
   (`diagnose_ranking.py` — lives in the Argos benchmark env) on the SAME
   snapshot. Record recall + precision numbers.
2. **Implement behind the seam**: the new engine must honor
   scope/status/validity/expiry semantics and `suppress_retrieval`, exactly
   like `DuckDBRetriever`. Keep the exact engine reachable via a config flag
   (default stays exact).
3. **A/B**: identical harness runs with the ANN engine. Gates:
   - LongMemEval: recall@N and precision **no worse than exact** (not "close",
     not "similar" — no worse; the whole point of the seam is zero-regression).
   - Short-fact recall (`diagnose_ranking.py`): **no worse** — this is the
     real-use check. Tuning to the benchmark alone has historically regressed
     short-fact retrieval; ANN at small top-k on 384-dim vectors silently
     drops exact matches, which is exactly what this check catches.
   - 20 hand-picked real-store queries: eyeball-inspect the top 10 of each.
4. **Ship behind the flag**, run live A/B for one week with the flag
   half-on if the runtime supports it, then flip the default and keep the
   exact engine as the rollback flag.

## Notes

- HNSW-class indexes need `ef`/`M` tuning for recall@96 on 384-dim bge
  vectors — tune against the gates above, not against latency.
- The index must stay consistent on the write path (1-2 writes/turn
  incremental inserts are fine; bulk re-ingest must trigger a rebuild).
- Never delete the exact engine until the ANN engine has survived a real
  store growth cycle. The seam was built so this swap is a config change,
  not a rewrite.
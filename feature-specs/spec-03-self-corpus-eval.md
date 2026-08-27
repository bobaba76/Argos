# Spec 3 — `eval_self_corpus.py`: regression-test retrieval on the real store

> **Status: IMPLEMENTED — superseded in form.** Historical design record
> (Aug 2026). The standalone script evolved into the snapshot/gold/gate
> toolchain (`snapshot_store` / `build_gold` / `run_gate` / `weekly_recon` /
> `personal_bench`) — see SYNC_HANDOFF.md and the benchmark runbook.


## 1. Why this exists
Argos is benchmarked on LongMemEval — a 500-question lab corpus of large synthetic documents (turns up to ~76k chars). The user's **real store is short-fact-dense** (median memory ~120 chars). Tuning against the benchmark can silently regress real-world recall, and the reverse: a benchmark win can be meaningless for the real store. We need a **free, deterministic, repeatable** way to score retrieval against the user's own memories and catch regressions before they matter.

This mirrors the design from diagnostic tooling already present (`LongMemEval\Evaluation\diagnose_ranking.py` — hit/catchable/retriever-miss classification) but:
- runs on the **live store** (real memories), not a cache file
- generates its own probe queries (no external question set)
- costs **zero API money** by default (template queries, no LLM judge, no LLM paraphrase unless explicitly opted in)
- is **strictly read-only** and cannot pollute ranking.

## 2. Tool contract — standalone script
`C:\Users\<user>\Documents\Github\Hermes\argos_plugin\eval\eval_self_corpus.py`

```
python eval/eval_self_corpus.py
    --db <path to hybrid_memory.duckdb>          (required; the live store or a copy)
    --limit 100                                   (memories sampled; default 100, max 500)
    --seed 42                                     (reproducible sampling)
    --top-k-ladder "5,20,96"                      (recall windows; 96 = production injection window)
    --out eval_self_corpus_out.jsonl              (append-only results; never commit to public repos)
    --baseline <path to a prior --out file>       (optional; enables PASS/FAIL regression verdict)
    --embedder-model BAAI/bge-small-en-v1.5       (must resolve from local HF cache — see README env notes)
    --llm-paraphrase                              (OFF by default; see §5)
    --threads 4                                   (embedding workers only; keep modest, laptop-friendly)
    --resume                                      (skip memory_ids already present in --out)
    --verbose
```
Exit code 0 = run complete (PASS or FAIL printed); nonzero = crashed. Add a `--help` describing everything.

## 3. Pipeline

### 3.1 Sampling (stratified)
- Connect **read-only**: open the DuckDB file with a second connection `read_only=True` (the desktop app holds the writer; DuckDB allows concurrent readers). If the file is locked, fall back to copying it to a temp path and reading the copy — never write to the live store from this script.
- Target population: `status='active' AND valid_to IS NULL AND (expires_at IS NULL OR expires_at > now)` — the same set production retrieval sees (this is Spec 1 territory; if expiry is not merged yet, the clause is a no-op safety net).
- Stratify by `category` (personal_fact, preference, insight, event, relationship, goal, context_note) proportional to the store's distribution, then by recency (half from the newest third, half from the rest) so the sample isn't all old or all new. `--seed` makes it reproducible.

### 3.2 Query generation (deterministic, free, per-template)
For each sampled memory, produce **exactly one** probe query chosen by a round-robin over templates so every template is exercised *within* the sample:
- `direct` — subject + attribute style: take the first content noun-phrase (strip stopwords, take first 1–3 non-stopword tokens, cap 4 tokens) and ask "what is <subject> <rest-of-content-clipped-to-24-chars>?" (This is intentionally imperfect; it mimics how a user half-remembers a fact. The hit rule in §3.4 is what keeps it fair.)
- `temporal` — only for memories with a date expression (reuse `date_anchor.py` patterns): "when did <subject> <content-clipped>?" 
- `preference` — category=preference: "what does the user prefer about <subject>?"
- `entity` — if the record has tags matching graph entities: "<subject> <first-verb-ish token>?"
- `negation` — "who is NOT <subject>" (the target should still rank; tests robustness to distractors)
- `synonym` — apply a tiny local synonym map (car→vehicle, job→work, bond→home loan, boss→manager, wife→partner... keep the map in the script, domain-agnostic) to one content token and rebuild the direct template. Local-only; never commit real content.
Templates live in the script; they must be transparent in the output JSONL (`"template": "direct"` per probe).

### 3.3 Retrieval execution
- Load `DuckDBMemoryStore` from the fork's plugin package (the harness imports the fork copy — same convention as the LongMemEval harness).
- For each probe: `store.search(query, top_k=max(ladder), suppress_retrieval=True)` **plus** a wide pool probe `store.search(query, top_k=512, suppress_retrieval=True)` to classify pool entry (mirror `diagnose_ranking.py`'s pool-cap insight: the provider's candidate pool is capped; a target absent from the wide run is a true retriever miss).
- Record per probe: query, template, target memory_id, per-window hit flags, wide-pool rank (or `not_in_pool`), top-5 returned ids + similarities (for eyeball review), config_hash (see §3.5), timestamp.
- **Never** call `_record_retrieval` paths — `suppress_retrieval=True` everywhere. A previous lesson: internal/diagnostic searches without that flag inflated `retrieval_count` and distorted ranking.

### 3.4 Hit rule (fair to versioned memories)
A probe is a hit for its target if ANY of:
- target `memory_id` is in the result set, or
- a result row is an older version of the same fact (`result.superseded_by == target.memory_id` or same chain root: follow `superseded_by` links — treat a chain as one fact), or
- token-Jaccard(result.content, target.content) ≥ 0.75 (catches re-saved near-duplicates; prevents version churn from manufacturing misses).
Do not require exact string match — the point is *recall of the fact*, not the row.

### 3.5 Metrics & regression verdict
Per run, group recall@K for K in the ladder: overall, by category, by template, by content-length bucket (<100 / 100–500 / >500 chars), by age bucket (<30d / 30–180d / >180d). Also report:
- `not_in_pool_rate` (share of probes whose target never entered the wide pool — the retriever-miss mirror)
- `avg_rank_of_hits` (pool rank, not window rank)
- sample size + category distribution + config_hash.

`config_hash` = sha256 over the retrieval-relevant knobs (embedder model name, max_injected_items, inject cap, graph boost values, query-expander enabled, date-anchor enabled, chain floors, `_RRF_K`, importance weights) so a future run can be compared only against runs with the same hash. When `--baseline` is given: **FAIL if recall@20 overall drops more than 3 percentage points vs the baseline run with the same config_hash**; PASS otherwise. Print a one-line verdict: `PASS recall@20=91.2% (baseline 92.0%)` or `FAIL recall@20=88.1% (baseline 92.0%, -3.9pp)`.

### 3.6 Output & resume
Append one JSON object per probe to `--out` (append-only; `--resume` skips memory_ids already present — same pattern as the LongMemEval harness `eval_longmemeval_hybrid.py` which is append-only with `--resume`). A summary block (the metrics above) is appended at the end of a run as a `{"run_summary": {...}}` line. Keep `--out` files OUT of any repo (add `.gitignore` entry if needed).

## 4. Optional LLM paraphrase (OFF by default)
`--llm-paraphrase` additionally synthesizes one paraphrased probe per sampled memory using the auxiliary model configured in `hybrid_memory.json` (LLM group keys — read them, don't hardcode). Constraints, mirroring `query_expander.py`'s established discipline:
- gated behind the explicit flag,
- bounded by a day budget (`llm_*` budget config if present; else hard cap of e.g. 50 calls/run — print cost estimate before starting and stop at the cap),
- never used for judging (hit rule stays deterministic),
- results tagged `"template": "llm_paraphrase"` so they're separable in the report.
Default behavior spends **zero rand**.

## 5. Tests (`tests/test_eval_self_corpus.py`, imported by `tests/run_tests.py`)
Use a temp DuckDB store built with the existing test fixture pattern:
1. Build ~20 known records (categories spread, one superseded chain of 2, one near-duplicate pair, one expired row) + run the pipeline on it.
2. Assert: sampling respects category stratification and seed reproducibility (same seed → same sample).
3. Assert: hit rule accepts (a) exact id, (b) older-version id, (c) near-duplicate (Jaccard ≥ 0.75).
4. Assert: recall math per group is correct on the fixture (hand-computed expected values).
5. Assert: `suppress_retrieval=True` → `retrieval_count` of all fixture records unchanged after a full run.
6. Assert: read-only mode (no writes attempted; run against a copy if needed).
7. Assert: `--resume` skips completed ids (out file unchanged for those).
8. Assert: `--baseline` PASS/FAIL verdict logic (feed a fabricated baseline with a known delta).
9. Assert: no LLM calls without `--llm-paraphrase` (monkeypatch the LLM client to raise if touched).

## 6. Acceptance criteria
- On the live store: a full `--limit 100` run completes with no API spend, in reasonable time (target < 5 min on a 4GB laptop GPU), and prints grouped recall@5/20/96 that match the known ballpark of the production system's measured behavior (evidence recall has consistently been ~99% at window 96 in benchmarks; real-store numbers will differ — that's the point of the tool).
- A deliberately broken retrieval knob (e.g. query-expander disabled in a temp store variant) produces a measurable recall drop that the `--baseline` verdict flags.
- The script is fully documented via `--help`; JSONL output is self-describing.
- All tests pass. The script itself runs from the fork; no deploy required (it's a dev tool, not a runtime feature — do NOT copy it to the deployed plugin).

## 7. Out of scope
- No LLM judge/answerer — this is a pure retrieval-recall tool (the LongMemEval harness already covers end-to-end answer quality).
- No changes to LongMemEval or its harness.
- No auto-scheduled runs (a cron job can wrap the CLI later; not this spec).
- No writes to the live store, ever.
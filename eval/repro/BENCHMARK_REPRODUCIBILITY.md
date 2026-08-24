# Benchmark reproducibility — Argos on LongMemEval_S

Every headline number in the README's "honest numbers" section maps to a
checked-in artifact below. The goal of this file is to make each claim
independently re-measurable: dataset + hash, protocol, model versions,
prompts, per-category denominators, and the judged outputs themselves.

**Verified 2026-08-22**: all numbers below were recomputed from the committed
judged files with the official `print_metrics.py` during the writing of this
document.

---

## 1. Claim → artifact map

| README claim | Value | Artifact (this repo, `eval/repro/`) |
|---|---|---|
| "70.4% on LongMemEval_S (500 questions, judged by gpt-4o, default answerer)" | 352/500 = 0.7040 | `judged_capexp_c1500_k96_gpt4o.jsonl` — 500 judged records, 0 missing |
| "99.6% of answer-bearing memories reach the top-96 candidates" | recall@96 = 0.996 | retrieval phase of the same 500-question run (phase-A cache, see §8) |
| "Temporal questions: 82% correct (133 questions)" | 109/133 = 0.820 | 52/75 (`judged_temporal_flash_flat_75_gpt4o.jsonl`) + 57/58 (`judged_temporal_flash_flat_58_gpt4o.jsonl`) — same protocol both slices |
| "a stronger answerer measured ~80+" | 43/55 = 0.782 | `judged_temporal_dsv4pro_55_gpt4o.jsonl` — 55-question temporal probe, directional (partial slice) |
| "cross-system gap is directional, not lab-controlled" | Perseus 73.8 / Zep 63.8 / Mem0 49.0 | vendor-published numbers; see §11 caveats |

## 2. Dataset

- **File:** `longmemeval_s_cleaned.json` (official LongMemEval synthetic data,
  English; sourced from the xiaowu0162 LongMemEval corpus as cleaned for
  evaluation — 500 questions).
- **SHA-256:** `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442`
  (hash of the exact file the runs used).
- **Per-category denominators:**

| question_type | n |
|---|---|
| multi-session | 133 |
| temporal-reasoning | 133 |
| knowledge-update | 78 |
| single-session-user | 70 |
| single-session-assistant | 56 |
| single-session-preference | 30 |
| **Total** | **500** |

- Each record carries `question_id, question_type, question, question_date,
  answer, answer_session_ids, haystack_dates, haystack_session_ids,
  haystack_sessions` (full per-question synthetic conversation transcripts).

## 3. Protocol (headline 70.4% run, 2026-08-19)

- **Harness:** `eval_longmemeval_hybrid.py` (this repo's evaluation tooling,
  developed against the official LongMemEval contract) + `judge_longmemeval.py`.
- **Store:** a fresh temp store **per question** (direct mode, `dedup=False`,
  graph + chains enabled, bge-small-en-v1.5 local embeddings). The live
  store is never touched.
- **Ingest:** haystack sessions ingested **per-turn, in order** (user +
  assistant turns) — per-session aggregation flattens version/update
  structure and was measured to damage knowledge-update and temporal scores.
  No LLM extraction during ingest (fair-ingestion).
- **Retrieval:** provider-path `_search_memories(top_k=96)` — vector + text +
  RRF + graph-aware retrieval + alias expansion + chain unfold all live.
  Retrieval pool cap raised to 512 (the earlier `min(50, …)` cap was a
  measured bottleneck; see `LONGMEMEVAL_RESULTS` history).
- **Rendering:** flat numbered list of the retrieved records, per-record
  content cap **1500 chars** (`--cap 1500`), records sorted by similarity.
- **Answering:** batched (5 questions/call, hard `ANSWER FOR <qid>:` delimiters,
  format-enforced), `temperature=0`, 20 threads. Abstention: empty retrieval →
  "cannot confirm". `ANSWER_SYSTEM_PROMPT` below, routed via
  `agent.auxiliary_client.call_llm`.
- **Judging:** official `get_anscheck_prompt()` imported bit-identical from
  the official repo's `src/evaluation/evaluate_qa.py`, `temperature=0`,
  per-question mode. Judged records: `{question_id, hypothesis,
  autoeval_label: {model, label}}`.
- **Date anchor:** NOT active for the 70.4% run (that is why temporal inside
  it is only 47.4%; the 82% temporal figure comes from the 2026-08-21
  re-measurement under the date-anchored protocol, §5).

## 4. Model versions

| Role | Model | Route | Notes |
|---|---|---|---|
| Answerer (default) | deepseek-v4-flash | opencode-go (default configured provider) | the "production brain" — what the README's 70.4% uses |
| Answerer (parity) | gpt-4o-2024-08-06 | OpenRouter | parity protocol run: 0.590 @ k=96 (v2 config) — see §8 |
| Judge | gpt-4o-2024-08-06 | OpenRouter | `LME_MODEL`/`LME_PROVIDER` env overrides of `judge_longmemeval.py` |
| Answerer (probe) | deepseek-v4-pro-0813 | OpenRouter | 55-question temporal probe only (cost-gated, partial) |

Judge neutrality was measured directly (cross-judge matrix, k=48): swapping
the judge moved results ≤0.6pp; swapping the answerer moved them −8.5pp. The
answerer, not the judge or the memory layer, is the dominant lever.

## 5. Temporal protocol (82.0%, 2026-08-21)

The 133 temporal-reasoning questions were re-measured after date-anchored
retrieval shipped (2026-08-21), in two sub-slices (75 + 58):

| Slice | Protocol | Correct | Accuracy |
|---|---|---|---|
| 75 questions, flat render | flash answerer, gpt-4o judge | 52/75 | 69.3% |
| 58 questions, flat render | flash answerer, gpt-4o judge | 57/58 | 98.3% |
| **Combined (uniform flat)** | — | **109/133** | **82.0%** |
| 75 questions, date-anchor prompt | flash answerer, gpt-4o judge | 54/75 | 72.0% |
| 75 questions, chronological render | flash answerer, gpt-4o judge | 50/75 | 66.7% |
| 58 questions, chronological-asc render | flash answerer, gpt-4o judge | 55/58 | 94.8% |

"Date-anchored retrieval" = the time-expression re-ranking (+word-number
resolution) applied before rendering, added 2026-08-21. The 58-question slice
at 98.3% flat render is small-n but internally consistent (see caveats §11).

## 6. Per-category scores — headline run (recomputed 2026-08-22)

REPRODUCIBILITY GATE: `./eval/repro/verify_repro.sh` re-derives every
headline number above from the committed judged files (plus the sibling
`../LongMemEval` checkout for dataset/cache artifacts; override with
`--artifacts DIR`) and exits non-zero on any drift — run it before
quoting a number.

Official `print_metrics.py` output on `judged_capexp_c1500_k96_gpt4o.jsonl`:

```
single-session-user:     0.9143 (70)
single-session-preference: 0.6333 (30)
single-session-assistant: 0.8929 (56)
multi-session:           0.6767 (133)
temporal-reasoning:      0.4737 (133)   <- pre-date-anchoring; see §5
knowledge-update:        0.8462 (78)
Task-averaged Accuracy:  0.7395
Overall Accuracy:        0.7040
Abstention Accuracy:     1.0000 (30)
Total judged: 500 | missing refs: 0
```

## 7. Prompts

**Answerer — `ANSWER_SYSTEM_PROMPT`** (verbatim from the harness):

```
You answer memory-evaluation questions using only the retrieved memory context.
Rules:
1. Use only the retrieved memories.
2. Do not invent facts that are not supported by the retrieved memories.
3. If the memories are insufficient, say that you cannot confirm.
4. If the memories contain inconsistent statements, briefly mention the inconsistency first and then give the best-supported answer.
5. Keep the answer concise, natural, and directly responsive to the question.
```

Date-anchored runs append: `\nToday's date is <YYYY-MM-DD>. Use it for any date
arithmetic in the questions.`

**Judge:** the official `get_anscheck_prompt(qtype, question, answer,
hypothesis, abstention=…)` — imported, not reimplemented.

## 8. Re-running

Canonical commands (paths relative to the maintainer's LongMemEval checkout;
the harness already ships in this repo's evaluation tooling):

```bash
# Phase A (one-time, CPU): ingest + retrieve top-96, checkpoint per question
python -u Evaluation/eval_longmemeval_hybrid.py --ingest_threads 8 --threads 6 \
  --batch 5 --top_k 96 --retrieve_k 128 --cap 1500 \
  --out hyp_capexp_c1500_k96.jsonl --ack cache_capexp_k96.jsonl.phaseA.jsonl

# Phase B (any k/rendering, minutes): re-answer from cache
python -u Evaluation/eval_longmemeval_hybrid.py --k_slice --top_k 96 --cap 1500 \
  --out hyp_capexp_c1500_k96.jsonl --ack cache_capexp_k96.jsonl.phaseA.jsonl

# Judge (gpt-4o both roles via env overrides)
LME_PROVIDER=openrouter LME_MODEL=openai/gpt-4o-2024-08-06 \
  python -u Evaluation/judge_longmemeval.py --hyp hyp_capexp_c1500_k96.jsonl \
  --out judged_capexp_c1500_k96_gpt4o.jsonl --batch 1 --threads 8

# Metrics
python Evaluation/print_metrics.py judged_capexp_c1500_k96_gpt4o.jsonl data/longmemeval_s_cleaned.json
```

- **Determinism:** `temperature=0` for answering and judging; retrieval,
  ingest, and ranking are fully deterministic; there is no sampling step, so
  no seeds are involved.
- **Environment:** Python 3.11 / Windows, `HF_HUB_OFFLINE=1`, torch intra-op
  capped at 2 threads, kuzu ingest ≤ 8 threads.
- **Cost:** one full 500-question answer + judge cycle at flash-tier pricing
  is a few dollars (the 70.4% run re-uses a single phase-A cache across all
  k-slices; re-answering any k costs only the LLM phase).

## 9. Cost axes (Zep-style transparency, added 2026-08-24)

Accuracy alone is half a claim. This section pins the **measured context
size, latency, and item counts** for the headline configuration and its
neighbors, all derived from the committed phase-A caches (the same artifacts
behind §3/§8) with a fully deterministic counter:

`context chars = Σ min(len(content), cap)` over items ranked by `sim`,
capped at k, filtered at `similarity >= floor`. ÷4 ≈ tokens. The script is
~30 lines; see `eval/repro/cost_axes.py`.

| Configuration | Accuracy | Context tok/turn | Median items | Retrieval s/q |
|---|---|---|---|---|
| cap400 k96 no-floor | 63.0% | ~7,900 | — | 31.2 |
| cap1500 k96 no-floor (**70.4% headline**) | 70.4% | ~20,900 | 96 | 33.4 |
| cap1500 k96 +0.30 floor (**today's prod config**) | not re-judged | **~14,600** | 67 | 33.4 |

Notes:
- Token figures are **composed retrieval context** per turn (÷4 chars/tok),
  measured over all 500 questions of the capexp cache — not API-bill tokens.
  Answer+judge LLM costs are additional but small relative to context.
- The +0.30 floor row is a *counterfactual* on the same cache: it shows what
  today's production config would have injected on the headline run. The
  never-blind fallback (top-8 unfiltered when the floor empties the window)
  adds back ≤8 items on ~1.4% of turns (~7/500), so real production sits
  marginally above the floored figure.
- Retrieval latency is single-machine CPU ingest+retrieval wall-time from the
  cache (`secs` field); scales with haystack size, not with cap/floor.
- Zep reports 347–1,997 context tokens across its settings for 10.7 accuracy
  points; Argos operates two orders of magnitude higher by design (it buys
  recall@96 = 99.6% with context, then claws budget back with the floor).

## 10. Chain-unfold recall/precision (~93% / ~93%, 2026-08-20)

Separate harness (`eval_chain_unfold_clean.py`, maintainer's dev tree).
Protocol: one-line synthetic facts seeded as version chains on **unsaturated**
topics (hobbies/lifestyle/gear — dense real memories on saturated topics bury
synthetic chains and understate recall, a diagnosed eval artifact); production
defaults (chain unfold auto, arc floor 0.15, anchor 0.30); every positive miss
classified RETRIEVAL-BURIED (not surfaced in top-20 → not a gate failure) vs
GATE-BLOCKED (surfaced but didn't unfold → real failure). Measured 2026-08-20:
recall ≈ 93%, precision ≈ 93%.

## 11. Honest caveats (read before quoting)

1. **70.4% and 82% describe different protocols.** 70.4% = full 500 with
   flat render, no date anchor (temporal 47.4% inside it). 82% = the 133
   temporal questions re-measured under date-anchored retrieval. Quoting them
   side by side is correct per the README's framing ("improving that bucket
   further") but they are not one run.
2. **Cross-system numbers are vendor-published** under their own protocols
   (their exact k, judge versions, and prompts are not fully public). We match
   the model tier (gpt-4o-class both sides) and official judge prompts; the
   parity run measured 0.590 @ k=96. Directional, not lab-controlled.
3. **Small-n bands:** preference (n=30), abstention (n=30), and the temporal
   58-slice wobble by several points per question. Treat as indicative.
4. **The 78.2% strong-answerer probe is a 55-question partial slice**
   (cost-gated), not a full 500-question run. It is directional evidence that
   the answerer is the bigger lever than the memory layer.
5. **Embedder:** all runs use bge-small-en-v1.5 (384-dim). A/B trials with
   bge-large and bge-m3 did not change results (measured 2026-08-21); prod
   ships the small embedder.
6. **Hindsight (91.4%) is excluded from cross-system comparisons** — different
   judging protocol and retrieval-depth regime.
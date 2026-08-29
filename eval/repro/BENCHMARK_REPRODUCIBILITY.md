# Benchmark reproducibility — Argos on LongMemEval_S

Every headline number in the README's "honest numbers" section maps to a
checked-in artifact below. The goal of this file is to make each claim
independently re-measurable: dataset + hash, protocol, model versions,
prompts, per-category denominators, and the judged outputs themselves.

**Verified 2026-08-22**: all numbers below were recomputed from the committed
judged files with the official `print_metrics.py` during the writing of this
document. **Extended 2026-08-26**: the answerer×distillation matrix (§12) and
the grounding composite + GLM direct 449/500 arm (§13) were added, each wired
into `verify_repro.sh`. **Refreshed 2026-08-27:** temporal headline now the full-bank 88.7% census (§5).

---

## 1. Claim → artifact map

| README claim | Value | Artifact (this repo, `eval/repro/`) |
|---|---|---|
| "70.4% on LongMemEval_S (500 questions, judged by gpt-4o, default answerer)" | 352/500 = 0.7040 | `judged_capexp_c1500_k96_gpt4o.jsonl` — 500 judged records, 0 missing |
| "99.6% of answer-bearing memories reach the top-96 candidates" | recall@96 = 0.996 | retrieval phase of the same 500-question run (phase-A cache, see §8) |
| "Chain-unfold, change-intent: ~93% recall / ~93% precision" | 92.9% prec / 92.9% raw / 100% fair recall, reproduced 27/8 | `argos_plugin/eval/eval_chain_unfold_clean.py` + `CHAIN_UNFOLD_RESULTS.md` — §10 |
| "Temporal questions: 88.7% correct" (118/133, full bank) | 118/133 = 0.887 | full-bank census under the 2026-08-23 text-leg hardening; per-slice trail in [DRIP_LOG.md](../../docs/archive/DRIP_LOG.md); earlier uniform-flat slices 109/133 = 0.820 (§5) |
| "a stronger answerer measured ~80+" | 43/55 = 0.782 | `judged_temporal_dsv4pro_55_gpt4o.jsonl` — 55-question temporal probe, directional (partial slice) |
| "Answerer × distillation, measured: gpt-4o 82.2% → 76.6% with the distill store; flash 48.0% → 86.6%" | 2×2 matrix | `judged_fullbank_v4_gpt4o_REAL.jsonl` (411/500), `judged_gpt4o500_v2.jsonl` (383/500), `judged_flash500_nodistill.jsonl` (240/500), `judged_full500_distill.jsonl` (433/500) — §12 |
| "Best current config: 89.8% (449/500), grounding + distill" | 449/500 = 0.8980 | direct: `judged_glm500_final.jsonl` (GLM answerer); flash equivalent: `composite_449.py` composition over the matrix + grounding files — §13 |

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

## 5. Temporal protocol (88.7% full bank — census 2026-08-23)

The 133 temporal-reasoning questions are **fully measured** — every question,
no projection — under the BM25-lite text-leg hardening of 2026-08-23:
**118/133 = 88.7%** (was 109/133 = 82.0% uniform flat / 66.2% original
protocol). Miss recovery 12/24 (50%); the 12 survivors are answerer-reasoning
gaps, not retrieval failures (autopsied in the drip log). The step-by-step
verification trail lives in [docs/archive/DRIP_LOG.md](../../docs/archive/DRIP_LOG.md).

Protocol history (same 133-question bank, flash answerer, gpt-4o judge):

| Slice | Protocol | Correct | Accuracy |
|---|---|---|---|
| 75 questions, flat render | pre-hardening | 52/75 | 69.3% |
| 58 questions, flat render | pre-hardening | 57/58 | 98.3% |
| **Combined (uniform flat)** | — | **109/133** | **82.0%** |
| 75 questions, date-anchor prompt | 21/8 protocol | 54/75 | 72.0% |
| 58 questions, chronological-asc render | 21/8 protocol | 55/58 | 94.8% |
| **Full bank, hardened store (census)** | 23/8 protocol | **118/133** | **88.7%** |

"Date-anchored retrieval" = the time-expression re-ranking (+word-number
resolution) applied before rendering, added 2026-08-21; the census then ran
the whole bank through the hardened text leg. The 58-question slices are
small-n but internally consistent (see caveats §11).

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

**ORPHAN SWEEP FIRST (Windows bash-wrapper gotcha — recurring 25/8, 28/8).**
Hermes terminal backgrounds runs through an MSYS bash wrapper: killing the
session kills the bash, NOT the python child. The orphan keeps running
(API spend), holds DB/CPU contention (a relaunched run then hangs silently —
banner prints, no `embedder warm` line, 0 progress), and can contaminate the
output file. Strays come from any earlier harness (venv-cuda or plain
Python311). Run `sweep_lme_orphans.ps1` (repo root, untracked local tool)
**before every launch and after every abort**:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\sweep_lme_orphans.ps1
# -DryRun to inspect first. The sweep also removes stale *.lock files whose
# PID is dead, and never touches hermes serve/gateway/memory-service processes.
```

Liveness check after launch: the run must print `embedder warm` within ~60
seconds; if it prints the banner but stalls there, sweep again and relaunch
(applies to the single-instance guard's stale-lock takeover too). Aborting a
run = kill it via the sweep, never rely on the launcher kill alone.

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

## 9. Cost axes (added 2026-08-24)

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
- Some hosted memory vendors publish context budgets in the hundreds to low
  thousands of tokens per turn; Argos operates two orders of magnitude higher
  by design (it buys recall@96 = 99.6% with context, then claws budget back
  with the floor).

## 10. Chain-unfold recall/precision (~93% / ~93%, 2026-08-20)

Separate harness (`eval_chain_unfold_clean.py`, maintainer's dev tree).
Protocol: one-line synthetic facts seeded as version chains on **unsaturated**
topics (hobbies/lifestyle/gear — dense real memories on saturated topics bury
synthetic chains and understate recall, a diagnosed eval artifact); production
defaults (chain unfold auto, arc floor 0.15, anchor 0.30); every positive miss
classified RETRIEVAL-BURIED (not surfaced in top-20 → not a gate failure) vs
GATE-BLOCKED (surfaced but didn't unfold → real failure). Measured 2026-08-20:
recall ≈ 93%, precision ≈ 93%.

> **Artifact (2026-08-27):** the canonical harness
> (`argos_plugin/eval/eval_chain_unfold_clean.py`) plus a committed reference
> run (`eval/repro/CHAIN_UNFOLD_RESULTS.md`) are now in-repo. The 20/8 figure
> **reproduces on current code**: precision 92.9%, raw recall 92.9%, fair
> recall 100% (arc 0.15 == arc 0.00 bit-identical); the single buried miss
> (`gym`) is a retrieval artifact, not a gate failure. The original diagnostic
> probes (PII-bearing) remain gitignored by rule. Note: the earlier crude
> harness (raw recall over all positives incl. buried, removed 27/8 as a
> footgun) read ~80/80 — that was the same feature under eval-artifact burden,
> **not** a regression and never the headline figure.

## 11. Honest caveats (read before quoting)

1. **70.4% and 88.7% describe different protocols.** 70.4% = full 500 with
   flat render, no date anchor (temporal 47.4% inside it). 88.7% = the 133
   temporal questions re-measured under date-anchored retrieval + the 23/8
   text-leg hardening (full-bank census). Quoting them side by side is
   correct per the README's framing but they are not one run.
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
7. **The 89.8% headline is answerer-conditional.** It is *directly measured*
   with the GLM-5.3-flash answerer (`judged_glm500_final.jsonl`); for the
   production flash answerer it is a zero-overlap composition (distill store
   on 338 qids + grounding A/B on 162), not an end-to-end flash run — §13.
   Grounding was A/B'd on only 162 of 500 qids for flash; everything beyond
   that on the flash side is composed, and the two full-bank numbers landing
   on the same 449 is the corroboration, not a substitute for an end-to-end
   flash run.
8. **Matrix arms mix eras.** The gpt-4o no-distill arm is a "banked" baseline
   (its judged file predates the other arms). All four are 500-qid,
   same-judge, same test set, so the pairwise *deltas* are the claim — treat
   the absolute percentages as era-specific.

## 12. Answerer × distillation matrix (2×2, 2026-08-25)

All four arms judged by **openai/gpt-4o-2024-11-20**, all on the **same 500
unique qids** (identical test set), `temperature=0` judging. Files committed
in this directory.

| Answerer | Distill store OFF | Distill store ON | Δ |
|---|---|---|---|
| gpt-4o-2024-11-20 | 411/500 = 82.2% (`judged_fullbank_v4_gpt4o_REAL.jsonl`, banked) | 383/500 = 76.6% (`judged_gpt4o500_v2.jsonl`) | −5.6 pts |
| deepseek-v4-flash | 240/500 = 48.0% (`judged_flash500_nodistill.jsonl`) | 433/500 = 86.6% (`judged_full500_distill.jsonl`) | +38.6 pts |

The distill store is **answerer-interactive**: load-bearing for the flash
answerer (+38.6 pts), mildly harmful for gpt-4o (−5.6 pts). Distillation
ships enabled; a gpt-4o-class answerer should disable it. This supersedes
the "stronger answerer ~80+" probe (§4) — the measured answerer effect is
the matrix above. Judge neutrality was itself measured: judge swap ≤0.6pp,
answerer swap −8.5pp (§4).

## 13. Grounding (LME_GROUNDING=1) and the 449/500 headline (2026-08-25/26)

Grounding = answerer prompt grounded in the retrieved evidence
(`LME_GROUNDING=1`). Judge is **openai/gpt-4o-2024-11-20** on every row of
every file in this section.

**Flash composite (production answerer).** A composition with zero qid
overlap (derivation: `eval/repro/composite_449.py`, wired into
`verify_repro.sh`):

| Component | Protocol | Correct |
|---|---|---|
| 338 qids | flash + distill store, no grounding (`judged_full500_distill.jsonl` minus the 162 A/B qids) | 308/338 |
| 30 qids (preference) | flash + grounding (`judged_pref30_grounding.jsonl`) | 25/30 |
| 132 qids (multi-session) | flash + grounding (`judged_multisession_grounding.jsonl`) | 116/132 |
| **Composite** | — | **449/500 = 89.8%** |

Grounding deltas on the A/B'd qids vs the banked no-grounding baselines
(25/8 run ledger; the committed files carry the post-values):
preference 17/30 → 25/30 (+8), multi-session 108/132 → 116/132 (+8).

**GLM direct arm.** Same 500 qids (identical test set), grounding on all
500, answerer `z-ai/glm-5.3-flash` — answerer stamped per-row in
`hyp_glm500_v1.dedup.jsonl`; judged in `judged_glm500_final.jsonl`:
**449/500 = 89.8%**, the direct full-bank measurement (26/8, ~US$1.9).

Head-to-head on the same 162 grounded qids: flash 141 vs GLM 140 (net −1) —
the flash composite landing on 449 is corroborated by the GLM direct run,
and the answerer is not the ceiling (both answerers tie at 449 under the
grounded protocol).

**Fix trail (documented so 424 ≠ 449 never reads as drift):** the first GLM
run produced ~34/500 garbage rows (batch truncation at `effort=max` left
dangling rows that passed truthiness checks); 25 rows were re-answered in
small batches (workspace-only `hyp_glm500_fix*.jsonl` fragments) and 35 rows
re-judged (`judged_glm500_rejudge35.jsonl`, committed). `lme_phaseB` was
hardened against the truncation mode. The earlier judged run
(`judged_glm500_gpt4o.jsonl`, 424/500) was left in the workspace for audit
and is superseded by `judged_glm500_final.jsonl`.
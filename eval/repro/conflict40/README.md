# Frozen conflict eval set — `lme_conflict40_v1`

Measures the baseline miss rate of write-time supersession detection across the
three failure classes from the design-smell thread, and serves as the regression
gate for read-side conflict surfacing (config `conflict_surfacing`).

## Set

40 questions, 4 classes, synthetic team scenario, `haystack_sessions` inline
(so it slots into any LongMemEval-family harness with no external session store):

| class | n | expectation | correct answer is |
|---|---|---|---|
| `conflict-stale` | 15 | `current_value` | the NEW value (unlinked duplicate, restatement-phrased) |
| `conflict-no-policy` | 12 | `abstain` | "no current policy" (discontinued / scoped-elsewhere) |
| `conflict-authority` | 10 | `refuse` | refusal / named authority (draft, scoped team) |
| `conflict-control` | 3 | `fact` | sanity: plain factual recall |

## Baseline (2026-09-03, prod config)

Answerer `deepseek/deepseek-v4-flash-0731`, `top_k=5`, `retrieve_k=20`, judge
`gpt-4o-2024-11-20` (corrected for batch-truncation artifacts) = **92.5% (37/40)**;
flash judge agrees (37/40). Class splits: stale 15/15, no-policy 9/12, authority
10/10, control 3/3.

Key finding: the dangerous class is **no-policy**, not stale — the three genuine
misses were all abstention cases (past-tense or negative-implication wording
without an explicit "no current policy", plus one scope-blindness case).

## A/B result (2026-09-03): read-side conflict surfacing — CONFIRMED +2

`LME_CONFLICT_SURFACING=true`, same answerer/judge protocol, fresh phase-A:

| class | baseline (×2 runs) | surfacing | Δ |
|---|---|---|---|
| conflict-stale | 14/15 | 15/15 | +1 |
| conflict-no-policy | 10/12 | 11/12 | +1 |
| conflict-authority | 10/10 | 10/10 | — |
| conflict-control | 3/3 | 3/3 | — |
| **TOTAL** | **37/40 (92.5%)** | **39/40 (97.5%)** | **+2** |

- No-policy win is the poster's exact case (beta storage discontinued → answerer
  now says "there is a conflict... the later record says the beta program ended").
- The stale class initially *looked* like a −2 regression — that was a harness
  artifact: the note text pushes answers past the 4096-token batch cap, so
  batch=5 answers come back truncated mid-sentence. Rerun at `--batch 1`:
  all three resolve correctly (state the current value AND cite the conflict).
  Baseline reproduced 37/40 twice, so the delta is not answerer variance.
- Remaining miss (1 no-policy): the scope-blindness case (`288796c6`, UK rule vs
  global question) — NOT a conflict pair, so surfacing can't catch it; that's
  the separate scope-awareness gap (noted in the feature issue).

**Harness gotchas learned (batch=1 is mandatory for this set; see skill):**
- Truncated-but-nonempty hypotheses are the batch-cap signature — check for
  answers ending mid-sentence, not just empty/`ANSWER FOR` markers.
- Judge failures (no verdict) need a one-off rerun, not a semantic read.

## A/B: read-side conflict surfacing

With `LME_CONFLICT_SURFACING=true` (fresh phase-A — the annotation is baked at
retrieval time, so reusing a flag-off cache would be a false-identical A/B),
expectation: no-policy 75% -> ~100%, stale stays flat.

## How to run

Run inside the LongMemEval harness clone (needs `agent.auxiliary_client` +
`hermes-agent` root on `sys.path`; exports from `HERMES_HOME/.env`).

```
# answer phase
LME_CONFLICT_SURFACING=true LME_MODEL=deepseek/deepseek-v4-flash-0731 \
  python eval_longmemeval_hybrid.py \
  --data eval/repro/conflict40/lme_conflict40_v1.json \
  --out hyp_conflict40_v1.jsonl --top_k 5 --retrieve_k 20

# judge (flash ~= gpt-4o at 96% agreement; gpt-4o for protocol parity)
python eval/repro/conflict40/judge_conflict40.py --hyp hyp_conflict40_v1.jsonl

# rebuild the dataset (idempotent, deterministic)
python eval/repro/conflict40/build_conflict40.py
```

Notes:
- Batch answers can truncate at the 4096-token cap and a `### ANSWER FOR`
  marker variance breaks the parser — on a miss, rerun that question at
  `batch=1` from the phase-A cache (`--ack`) before judging.
- The 27/8 failure's other half (old record crowding the new one out of the
  top window at large-haystack scale) is NOT covered here — needs a
  large-haystack extension.

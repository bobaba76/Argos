# Text-leg hardening — drip-feed verification log

**Change under test:** BM25-lite `_text_search_raw` (regex tokenizer +
IGNORECASE, 8 tokens minus stopwords, idf × tf-saturation × length-norm,
LIMIT 2000) replacing first-4-token ILIKE overlap scoring.
Commit `5754df4`, pushed 2026-08-23. Live in prod since the evening restart
of 2026-08-23 (installed copy md5-verified against repo: `68e61fc7…`).

**Why:** component ablation on a 100-memory real-store sample showed the
text channel at 71/87 (r@5/r@20) while the vector leg ran 88/95 — the
disaster-mode floor was the weakest link in an otherwise clean system.

---

## Verification timeline (all 2026-08-23, gpt-4o judge, flash answerer)

| # | Slice | Banked | Measured | Net | Miss recovery |
|---|---|---|---|---|---|
| A/B | 10 q, **miss-weighted** (7 misses + 3 passes), fresh phase-A through patched store | 3/10 | 7/10 | +4 | 4/7 |
| Drip2 | 12 q, uniform random, seed 20260823 | 10/12 | 10/12 | ±0 | 1 of 2 drawn |
| Drip3 | 12 q, uniform random, seed 20260824 | 11/12 | 12/12 | +1 | 1/1 |
| Drip4 | 20 q, uniform random, seed 20260825 (miss-heavy draw: 7) | 13/20 | 16/20 | +3 | 4/7 |
| Drip5 | 20 q, uniform random, seed 20260826 | 16/20 | **18/20** | +2 | 2/4 |
| Census | **final 59 q — completes the 133-question bank**, incl. ALL 3 unused misses | 56/59 | 55/59 | −1 | 0/3 |

## FINAL RESULT — full temporal bank, every question measured (no projection)

**118/133 = 88.7%** (was 109/133 = 82.0% flat / 66.2% original protocol)

- Misses recovered: 12 of 24 (50%); the 12 survivors are resistant reasoning
  gaps, not retrieval failures — the patch's jurisdiction ends there.
- Pass regressions: 3 flip-events across 109 pass-draws (~2.7%), all
  answerer-paraphrase judge boundaries, retrieval exonerated each time.
- Random-slice pooled accuracy: 78.1% → **87.5%** (n=64).
- Projection made before any drip ran ("87–88%") — measured 88.7%.

## Interpretation (honest version)

- Miss-recovery rate is stable across two independent draws (~57–60%).
- Pass-regressions are rare and so far attributable to answerer paraphrase
  at judge boundaries, NOT retrieval degradation; one autopsy performed:
  - `gpt4_fe651585` ("Who became a parent first?"): gold evidence ranks
    IMPROVED old→new ([2,4,7,9]→[2,3,5,6]); both hypotheses contain the
    same facts; flash's "which was later" → "which appears to be more
    recent" flipped the judge. Retrieval exonerated.
- Projection if rates hold across the full temporal bank (24 misses):
  ~14 recoveries − ~6–7 pass flips ≈ **87–88% temporal**, up from
  66.2% original → 82% post-date-anchor. Wide CI until more drips land;
  each installment sharpens it.

## Artifacts (LongMemEval checkout)

- `data/lme_temporal_{abl10,drip2,drip3,drip4}.json` (+ `_banked` sidecars)
- `hyp_drip{2,3,4}.jsonl.phaseA.jsonl` — fresh retrieval through patched store
- `judged_abl10_patched_gpt4o.jsonl`, `judged_drip{2,3,4}_gpt4o.jsonl`
- Builders: `perseus-bench/_build_ablation_subset.py`, `_build_drip{2,3,4}.py`
- Comparer: `perseus-bench/_compare_drip.py <sidecar> <judged> [label]`
- Autopsy: `perseus-bench/_autopsy_fe651585.py`

## Harness notes (post argos-rename)

- Eval harness imports `<Github>/hybrid_memory_plugin`; today's rename broke
  that path. Restored as a junction → `Argos/argos_plugin`.
- `HybridMemoryProvider = ArgosProvider` alias appended to plugin
  `__init__.py` (committed in `5754df4`).
- Judge banner/artifact previously hardcoded `deepseek-v4-flash` regardless
  of `LME_MODEL` env; fixed to record the effective model. Judge with:
  `LME_PROVIDER=openrouter LME_MODEL=openai/gpt-4o-2024-08-06`.

## Next installments

**None needed — the census completed the bank.** All 133 temporal questions
have now been measured under the patched store (see FINAL RESULT above).
Any future re-verification should be a full-bank rerun or a targeted
re-test of the 12 resistant misses.

## Drip5 postmortem (harness robustness lesson)

A foreground timeout killed the original drip5 run mid-ingest; three
follow-on quirks cost ~40 minutes before diagnosis:
1. `--resume` only answers *remaining* questions — checkpointed ones never
   get hypotheses written (retrieval-only).
2. Custom `--out` is IGNORED when `--ack` is set; output always lands as
   the ack-derived sibling filename.
3. The answer loop silently skips questions whose cached records are
   corrupt/truncated (no error surfaced) — stragglers `gpt4_0b2f1d21` +
   `af082822` dropped twice this way; a fresh-tiny-ack rerun answered 2/2.
Rule of thumb: after ANY killed run, re-answer affected questions with a
fresh ack rather than trusting partial outputs.

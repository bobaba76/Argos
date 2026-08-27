# Phrase-lift (exact-phrase ranking lift) — measured result (2026-08-24 / reproduced 2026-08-27)

The `phrase_lift_alpha` lever (prod `0.25`, pool 200) rewards contiguous
query bigrams present verbatim in the memory. It exists for the class of
query where the gold memory shares the exact phrase but was ranked low
because unigram token overlap tied it with merely-similar content.

## Reproduction (2026-08-27, committed artifacts)

Deterministic harness `argos_plugin/eval/eval_phrase_lift_clean.py`:
fresh temp store per run, bge-small-en-v1.5 local embeddings, 8 exact-phrase
cases (gold memory contains the verbatim phrase; 3-4 token-overlap
distractors tie it) + 3 negative controls. No LLM, no network.

| | MRR | h@1 |
|---|---|---|
| α = 0.00 (control) | 0.7292 | 4/8 |
| α = 0.25 (prod) | **0.9375** | **7/8** |
| Δ | **+0.2083** | **+3** |

- Zero regressions from control rank-1 cases.
- Reproduces the real-store effect measured 24/8 (MRR .66 → .82, h@1 4→6/8
  on real queries): same class, same direction.

## Verdict

- Phrase-lift is a **ranking repair for exact-phrase queries**, not a recall
  lever and not a reranker substitute. Keep α = 0.25, pool 200 (prod config).
- No regression surface measured (all control rank-1 cases stay rank-1).

## Re-run

```
cd argos_plugin
env -u PYTHONPATH HF_HUB_OFFLINE=1 <hermes-venv-python> \
  eval/eval_phrase_lift_clean.py
```

Exit 0 = improved with no regression.

## Change history

- 2026-08-24 — R0 validation on real queries (MRR .66 → .82, h@1 4→6/8).
- 2026-08-27 — sanitized reproducible harness committed; result reproduced
  (MRR 0.7292 → 0.9375, h@1 4/8 → 7/8).
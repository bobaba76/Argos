# Chain-unfold: repro artifact (2026-08-27)

Two harnesses exist, and they measure **different things**. This file documents
both so nobody repeats the earlier misreading (reporting the crude harness as a
regression when it is the same feature under a harsher metric).

**Verdict up front: no regression.** The canonical headline — precision ≈ 93%,
fair recall 100% on the 20/8 protocol — **reproduces exactly on current code**
against the same frozen snapshot (`memory-eval-clean.duckdb`, mtime
2026-08-12, unchanged since the 20/8 measurement).

## Canonical eval (the number to quote) — `eval_chain_unfold_clean.py`

Seeds **14 chains on unsaturated topics** (hobbies/lifestyle/gear) where
retrieval can surface them, plus 12 negative queries. Every positive miss is
classified **RETRIEVAL-BURIED** (current version not in top-20 → eval artifact,
real memories on dense topics bury synthetic chains) vs **GATE-BLOCKED**
(surfaced but didn't unfold → real feature miss). Recall is reported two ways:
**raw** (over all positives) and **fair** (over positives that surfaced).

### Run metadata

| Field | Value |
|---|---|
| Date | 2026-08-27 |
| Repo | `bobaba76/Argos` master |
| Python | 3.11.9 (Hermes `hermes-agent/venv`) |
| Embedder | `bge-small-en-v1.5` (local cache, offline — `HF_HUB_OFFLINE=1`) |
| Corpus snapshot | `%LOCALAPPDATA%\hermes\memory-eval-clean.duckdb` (frozen 2026-08-12, sha256 prefix `861118e6`) |
| Mode | provider-layer, direct storage, temp home per run (never touches live data); seeds sanitized (generic names/employers/meds/locations) |
| Cost | 0 LLM calls — deterministic precision/recall vs expected flags; no billing path touched |

### Results (current code, repo HEAD at measurement time)

```
PROD-DEFAULTS (arc 0.15 / anchor 0.30):  precision 92.9% (13/14)  raw 92.9%  fair 100.0%  TP13 FP1 FN1 TN11
ARC OFF (0.00):                          precision 92.9% (13/14)  raw 92.9%  fair 100.0%  TP13 FP1 FN1 TN11
IN BAND (prec >= 90 & fair recall >= 90): True
```

- **Fair recall 100% (13/13 surfaced)** — every change-intent question whose
  chain actually surfaced unfolded correctly. No GATE-BLOCKED (real feature
  misses) at all.
- **1 RETRIEVAL-BURIED: `gym`** — "do I still work out at home" didn't surface
  the seeded chain in top-20. Eval artifact, not a gate failure.
- **Precision 92.9%**: 1 FP among negatives (a change-phrased query that got
  an arc). This is a single toggle away from 100% on this set — treat 93% as
  "≈93%", not a precise population rate (26 queries).
- **The arc floor is inert for recall**: arc 0.15 == arc 0.00 exactly. The
  gate is a pure precision knob, as documented — it never blocks a real saga.

## Crude eval (why it read lower) — `eval_chain_unfold.py`

The older, harsher harness (3 saturated chains — property/meds — plus real
distractor memories; **raw recall over all positives; no RETRIEVAL-BURIED
excise; no fair-recall metric**). Its numbers (best: 80% / 80% with
query-fallback) are a **different, stricter measurement of the same feature**,
not a regression signal:

- The property-plan chain sits on a saturated topic (the snapshot is dense
  with real property memories), so it is RETRIEVAL-BURIED and counts as a raw
  miss. The clean protocol would classify it exactly that way and exclude it
  from fair recall.
- Keep it if you want the "what does real dense-store recall look like, the
  hard way" number; it is *not* the headline and never was.

## What changed vs 20/8? (nothing material)

- Snapshot: identical (frozen 12/8).
- Canonical eval (hook + gate): unchanged since `c610b4f` (20/8) — only
  unrelated provider work landed since (dream/P4.2, TTL, semantic dedup,
  intent-router v3, date-anchor).
- Result: 92.9% / 92.9% / 100% matches the 20/8 ≈93% claims.

## Repro

```bash
env -u PYTHONPATH HF_HUB_OFFLINE=1 <hermes-venv-python> \
  argos_plugin/eval/eval_chain_unfold_clean.py
```

Exit code 0 iff both rows are IN BAND (prec ≥ 90 & fair recall ≥ 90).

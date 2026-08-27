# Claims Audit — Argos

**Date:** 2026-08-27 · **Audited:** README.md and docs vs `src/` and committed eval
artifacts on `master` · **Branch HEAD:** `7d14697`

This file is the living index of every claim Argos makes. The rule: **anything
quoted publicly must map to a committed, re-runnable artifact — or be marked
aspirational.** This mirrors the audit discipline of comparable memory systems
(see Perseus Vault's own `CLAIMS-AUDIT.md`) and is the honest-evidence culture
this repo runs on.

Verification has two tiers:

- **Measured claims** — numbers behind the README "Numbers" table. These are
  backed by committed judged artifacts in `eval/repro/` and re-derived by the
  runnable gate `./eval/repro/verify_repro.sh` (fails loudly on any drift).
  Run it before quoting a number.
- **Structural claims** — capability statements in the README/docs. Audited
  directly against source below (dated).

---

## 1. Measured claims (LongMemEval_S + feature evals)

Every row is verified by `verify_repro.sh` against the listed committed
artifact. Protocol, dataset SHA-256, denominators, model versions, prompts,
and exact commands: [`eval/repro/BENCHMARK_REPRODUCIBILITY.md`](eval/repro/BENCHMARK_REPRODUCIBILITY.md).

| README claim | Value | Committed artifact (`eval/repro/`) | Gate |
|---|---|---|---|
| LongMemEval_S (best config) | 89.8% (449/500) | `judged_glm500_final.jsonl` (GLM direct, full bank); `composite_449.py` (flash composition) | ✓ |
| LongMemEval_S (baseline) | 70.4% (352/500) | `judged_capexp_c1500_k96_gpt4o.jsonl` | ✓ |
| Chain-unfold (change-intent) | 93% recall / 93% precision | `CHAIN_UNFOLD_RESULTS.md` + canonical harness `argos_plugin/eval/eval_chain_unfold_clean.py` | ✓ (27/8 reproduced: 92.9/92.9/100 fair) |
| Temporal questions | 88.7% (118/133) | full-bank census under 23/8 text-leg hardening (repro §5); earlier slices `judged_temporal_flash_*.jsonl` | ✓ |
| Recall@96 | 99.6% | retrieval phase of the 500-question capexp run (repro §1/§8) | ✓ |

**Caveats that travel with the numbers** (full list: repro §11): 70.4% and
88.7% are different protocols, not one run; cross-system numbers are
vendor-published under their own protocols (parity run: 0.590 @ k=96);
small-n bands (preference n=30, abstention n=30) are indicative; the 89.8%
headline is answerer-conditional (GLM direct / flash composed).

---

## 2. Structural claims (audited against source, 2026-08-27)

| Claim (README/docs) | Status | Evidence |
|---|---|---|
| "Sixteen `memory_*` tools" | ✓ verified | `handle_tool_call` routes exactly the 16 named tools; grouping matches the README table (3+3+2+3+2+3). |
| Facts persist across sessions | ✓ | `memory_save` → DuckDB store (`src`); persistence covered by `tests/test_hybrid_memory.py`. |
| Changes versioned, not erased | ✓ | `memory_update` chains new versions onto old; `memory_chain`; tested. |
| Semantic search — vector + keyword fusion (RRF) | ✓ | `_search_memories` (vector + text + RRF + graph + alias + chain-unfold); recall@96 99.6% measured (§1). |
| Relationship graph (Kùzu) | ✓ | `memory_graph_search` / `memory_graph_query`; Kùzu backend in `src`. |
| Ambient context injects per turn | ✓ | `pre_llm_call` hook; `tests/test_ambient_*` (weather/location/file activity). |
| Insight capture + `/ilog`, `/revisit`, `/neg` | ✓ | insight-log tool; negative-memory exclusions tested (`test_negative_memory.py`). |
| Gated distillation — cost-capped, proposes only | ✓ | `test_distillation.py`; nothing lands without approval (approval invariant, `test_approval_invariant.py`). |
| Reversible cleanup — quarantine, not delete | ✓ | tombstones + `memory_restore`; `test_deletion_tombstones.py`, `test_ttl_expiry.py`. |
| Local embeddings, offline | ✓ | `bge-small-en-v1.5`, local-first cache-path resolution (no network HEAD-check); `embeddings.py`. |
| LLM calls via configured cloud model only; no native local-LLM | ✓ | consistent with egress gating (`tests/test_egress.py`, `SITES` registry). |
| License: BSL 1.1 → Apache-2.0 on 2030-08-21 | ✓ | `LICENSE.md` (BSL 1.1, MariaDB text); production/commercial use requires a licence (per BSL terms). |
| Test suite | ✓ | 26 test modules in `argos_plugin/tests/` (gate 21, egress, adversarial chains, contradiction matrix, shared-service RPC). |
| No personal data in public repo | ✓ verified | gold JSONL, snapshots, bench outputs, gate baselines gitignored by rule; only gold freeze sha committed (`eval/gold/README.md`). |

---

## 3. Claims flagged — measured but not yet public evidence

These are real measurements that are **not yet committed as re-runnable
artifacts**, so they are *not* quotable as public claims until they are:

| Claim | Status | What's missing |
|---|---|---|
| Phrase/proximity lift (α = 0.25) — MRR .66 → .82, h@1 4→6/8, zero regressions (measured 24/8) | measured, kept in maintainer notes | committed eval set + judged/frozen artifact + gate row |
| Reranker A/B (26/8, 300-strat seed 42): MRR +3.1pp (0.906→0.937), ndcg20 +2.5pp, recall@20 +0.3pp | measured, kept in maintainer notes | committed eval set + artifact; verdict is *ranking* lever, not recall lever (k-descent stays locked); CPU-only torch ~8s/q blocks prod ship until CUDA-torch |
| Personal self-corpus gate + personal bench | intentionally **private** | gold/snapshots/baselines hold personal memory content — gitignored by design; never public evidence |

Rule of thumb for this section: if it landed in memory or a session log but
not in `eval/repro/`, it is a **finding**, not a **claim** — and cannot be
quoted as a public number until it has an artifact.

---

## 4. Aspirational statements (not claims)

| Statement | Why it's not a claim yet |
|---|---|
| README Trust model: "Every feature is gated by a measurement in the eval harness" | Aspirational. Feature *areas* are tested, but not every shipped knob has a committed before/after measurement. The measured subset is §1; everything else is structural verification (§2). Treat the sentence as engineering intent, not an audited fact. |

---

## 5. History

- **2026-08-27** — first audit. Structural claims verified against `master`
  `7d14697`. Benchmark family already backed by `verify_repro.sh` (from the
  22/8 reproducibility work). Flagged the uncommitted measurements (§3) and
  the aspirational trust-model sentence (§4) so the honest-evidence boundary
  is explicit.

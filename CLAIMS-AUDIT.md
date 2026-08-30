# Claims Audit — Argos

**Date:** 2026-08-28 · **Audited:** README.md and docs vs `src/` and committed eval
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
| Phrase-lift (exact-phrase ranking) | α=0.25: MRR .7292 → .9375, h@1 4/8 → 7/8, zero regressions | `PHRASE_LIFT_RESULTS.md` + harness `argos_plugin/eval/eval_phrase_lift_clean.py` | ✓ (27/8 reproduced) |
| Reranker A/B | MRR +3.1pp (0.9058→0.9372), nDCG@20 +2.5pp, R@20 +0.3pp (300-strat seed 42) | `RERANKER_AB.md` + committed aggregate `reranker_ab_summary.json` | ✓ (aggregate) |

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
| Trust model — "nothing becomes a memory silently" | ✓ (scoped) | Auto-extraction → proposal queue (pending until reviewed). `memory_save` is the explicit exception: writes directly to active memory, bypassing the proposal queue (intentional agent action, not passive ingestion). `MEMORY_SYSTEM.md` documents the exception. |
| Reversible cleanup — quarantine, not delete | ✓ (scoped) | `memory_maintenance` / `consolidate()` quarantine only, never hard-delete. `memory_delete` is chain-aware: head-with-predecessor promotes + hard-deletes the row; non-head quarantines; **single-version hard-deletes + tombstones** (re-creation blocked until `memory_tombstone_purge`). `test_deletion_tombstones.py`, `test_ttl_expiry.py`. |
| Local embeddings, offline | ✓ | `bge-small-en-v1.5`, local-first cache-path resolution (no network HEAD-check); `embeddings.py`. |
| LLM calls via configured cloud model only; no native local-LLM | ✓ | consistent with egress gating (`tests/test_egress.py`, `SITES` registry). |
| License: BSL 1.1 → Apache-2.0 on 2030-08-21 | ✓ | `LICENSE.md` (BSL 1.1, MariaDB text); production/commercial use requires a licence (per BSL terms). |
| Test suite | ✓ | 26 test modules in `argos_plugin/tests/` (gate 21, egress, adversarial chains, contradiction matrix, shared-service RPC). |
| Public repo contains no personal data | ✓ verified | gold freeze sha documented in `eval/gold/README.md`. |

---

## 3. Claims flagged — measured internally, no public artifact yet

These measurements are not part of the public claim set until they have a
committed, re-runnable artifact:

| Claim | Status | What's missing |
|---|---|---|
| Self-corpus gate + personal bench | internal | maintained for the weekly recon |
| MemConflict 16-question slice — turn-level ingest (28/8 vs 30/8) | internal | Same 16 questions (persona 0, sessions 0–9), same answerer (deepseek-v4-flash via OpenRouter), same levers/prompt. AA 0.406 → 0.344 (one question, n=16), UOCS 0.188 → 0.438, CRS 0.188 → 0.125. Not comparable to the 13/8 180-question baseline (different harness, ingest, scorer handling). Artifacts live in the benchmark clone (`hermes-memconflict-fork`, `argosvault/Results/`), untracked — not banked, so not a public claim yet. |

---

## 4. Aspirational statements (not claims)

| Statement | Why it's not a claim yet |
|---|---|
| README Trust model: "Every feature is gated by a measurement in the eval harness" | Aspirational. Feature *areas* are tested, but not every shipped knob has a committed before/after measurement. The measured subset is §1; everything else is structural verification (§2). Treat the sentence as engineering intent, not an audited fact. |
| MEMORY_SYSTEM.md:102 + CONFIG_REFERENCE.md:79-82 + UI label "Stale-pending sweep": "periodically re-reviews proposals pending too long" | **Configured-but-unimplemented.** Four keys (`stale_review_sweep_enabled`, `stale_review_interval_min`, `stale_review_min_age_min`, `stale_review_max_batch`) are declared in `config_schema.py`, documented as a live feature, and parsed in `__init__.py:942-964` into `self._stale_review_*` attributes — and never read again. No scheduler, timer, cron, or hook consumes them. `review_pending.py` is the manual CLI that would be the sweep's engine (default limit 25 matches `stale_review_max_batch`), but nothing schedules it. `git log --all -S '_stale_review'` shows no consumer was ever added. The docs overstate a shipped feature; either implement the sweep or mark this section aspirational. |
| MEMORY_SYSTEM.md:103 + CONFIG_REFERENCE.md:83: "Role-word learning (role_alias_llm_fallback=true) — when an unknown word appears in 'my X is Name', the LLM is asked if X is a person-role; learned words persist to role_words" | **Implemented for alias minting, not for fact categorization.** The feature works end-to-end in `__init__.py` (`_extract_role_aliases` + `_llm_classify_role_word` + `_persist_learned_role_word`), and the docs accurately describe the mechanism. The gap: `extractor.py` has a separate 12-word hardcoded role lexicon that decides fact *category* (relationship vs personal_fact), while `graph.py` has a 46+ word seed list that decides *alias minting*. LLM learning only feeds the second set. A learned word like "doula" instantly fixes alias retrieval and persists, but the underlying fact is still stored as `personal_fact` forever — extractor.py never consults learned words. Category wrong, alias right; the system papers over the categorization gap at the alias layer. |

---

## 5. History

- **2026-08-27** — first audit. Structural claims verified against `master`
  `7d14697`. Benchmark family already backed by `verify_repro.sh` (from the
  22/8 reproducibility work). Flagged the uncommitted measurements (§3) and
  the aspirational trust-model sentence (§4) so the honest-evidence boundary
  is explicit.
- **2026-08-27 (same day)** — phrase-lift and reranker A/B graduated from
  findings (§3) to measured claims (§1): phrase-lift gained a sanitized
  re-runnable harness (`eval_phrase_lift_clean.py`) with the result
  reproduced (MRR .7292 → .9375); the reranker A/B aggregate summary was
  committed (`reranker_ab_summary.json`).
- **2026-08-28** — scoped two trust-model claims that were overstated (issues #12, #13):
  the "nothing becomes a memory silently" claim now notes the `memory_save` explicit-save
  exception; the "reversible cleanup — quarantine, not delete" claim now distinguishes
  maintenance/consolidation (quarantine only) from `memory_delete` (chain-aware: promote,
  quarantine, or hard-delete + tombstone for single-version records). Added a structural
  claim row for the trust-model paragraph. No code behavior changed.
- **2026-08-28 (deep review)** — added two aspirational entries to §4:
  (a) the stale-review sweep is configured-but-unimplemented (four parsed keys never
  consumed; `review_pending.py` is the orphaned manual engine; docs describe it as live);
  (b) role-word LLM learning works for alias minting but not for fact categorization
  (two hardcoded lexicons that never converge — extractor.py's 12-word list vs graph.py's
  46+ word seed set). Filed issues #23 (inbound security fail-open on import error,
  `reviewer.py:206-212`) and #24 (egress gate returns True for unknown kind,
  `egress.py:225-227`) — both are fail-closed fixes. No code behavior changed.
- **2026-08-30** — recorded the MemConflict 16-question re-run pair (issue #8) in §3 as an
  internal measurement: 28/8 vs 30/8 on the same slice, answerer, and levers shows AA within
  ±1 question (0.406 → 0.344), UOCS 0.188 → 0.438 (update-order awareness — the
  turn-level-ingest differentiator), CRS 0.188 → 0.125. The 30/8 prompt-v2 follow-up slice
  (answer-form + status-vs-details rules) was interrupted mid-ingest and remains pending.
  Kept in §3 (not §1) until the benchmark-clone artifacts are banked/committed.

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
| Test suite | ✓ | 59 test modules in `argos_plugin/tests/` (1169 `def test_` definitions; full suite green 2026-08-30 via `pytest tests/ -q -n 4`). Covers gate verdicts, egress, inbound security, adversarial chains, contradiction matrix, shared-service RPC, multitenant Cells. |
| Public repo contains no personal data | ✓ verified | gold freeze sha documented in `eval/gold/README.md`. |

---

## 3. Claims flagged — measured internally, no public artifact yet

These measurements are not part of the public claim set until they have a
committed, re-runnable artifact:

| Claim | Status | What's missing |
|---|---|---|
| Self-corpus gate + personal bench | internal | maintained for the weekly recon |
| MemConflict 16-question slice — turn-level ingest (28/8 vs 30/8) | internal | Same 16 questions (persona 0, sessions 0–9), same answerer (deepseek-v4-flash via OpenRouter), same levers/prompt. AA 0.406 → 0.344 (one question, n=16), UOCS 0.188 → 0.438, CRS 0.188 → 0.125. **Mechanism caveat (chain-verify 30/8):** store-level chains are absent — 0 of 595 (and 0 of 884 in the 30/8 DB) records have `valid_to`/`superseded_by` set; ingest is `remember(dedup=False)`, so supersession/versioning never fires and chain-unfold has nothing to walk. The UOCS delta is the answerer's timestamp reasoning over the chronologically rendered list (prompt rule "latest wins"), not store update-arithmetic — the #8 store-level intent remains unproven (filed as #74). Not comparable to the 13/8 180-question baseline (different harness, ingest, scorer handling). Artifacts live in the benchmark clone (`hermes-memconflict-fork`, `argosvault/Results/`), untracked — not banked, so not a public claim yet. |

---

## 4. Aspirational statements (not claims)

| Statement | Why it's not a claim yet |
|---|---|
| README Trust model: "Every feature is gated by a measurement in the eval harness" | Aspirational. Feature *areas* are tested, but not every shipped knob has a committed before/after measurement. The measured subset is §1; everything else is structural verification (§2). Treat the sentence as engineering intent, not an audited fact. |
| MEMORY_SYSTEM.md:102 + CONFIG_REFERENCE.md:79-82 + UI label "Stale-pending sweep": "periodically re-reviews proposals pending too long" | **Implemented (#10, 2026-09-01).** The four config keys are now consumed by `stale_review_sweep.py` — a daemon thread that runs every `stale_review_interval_min`, re-reviews only `pending` candidates older than `stale_review_min_age_min`, caps at `stale_review_max_batch`, and preserves the no-auto-promotion invariant (decision map identical to `review_pending.py`). Fail-soft on LLM error. Started by the provider after initialization; stopped on shutdown. |
| MEMORY_SYSTEM.md:103 + CONFIG_REFERENCE.md:83: "Role-word learning (role_alias_llm_fallback=true) — when an unknown word appears in 'my X is Name', the LLM is asked if X is a person-role; learned words persist to role_words" | **Implemented end-to-end (#14, closed 2026-08-29).** The two lexicons now converge: `extractor.set_role_words()` is called from `provider_core.py` at init (with the graph's defaults + config) and from `provider_session.py` each time the LLM ambiguity gate learns a new word, so `extractor._all_role_words()` (base + extra) and `graph._get_role_words()` (defaults + override + learned) stay in sync. A learned word like "doula" now correctly categorizes as `relationship` in the extractor *and* mints the alias in the graph. Verified by 16 `set_role_words`/`_set_role_words_override` test references in `test_hybrid_memory.py` (override extends the set, learned word extends the set, LLM ambiguity gate accepts/rejects, fallback-disabled skips the gate). The earlier "category wrong, alias right" gap described in the 2026-08-28 deep review is closed. |

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
- **2026-08-30 (chain-verify correction)** — the mechanism attribution above was corrected:
  store-level version chains are absent across every run DB (0 of 2,424 records with
  `valid_to`/`superseded_by`; ingest is `remember(dedup=False)`, so supersession never
  fires and chain-unfold has nothing to walk). The UOCS gain is the answerer's timestamp
  reasoning over the chronologically rendered list, not store update-arithmetic — the #8
  store-level intent remains unproven. Filed as #74.
- **2026-08-30 (sync)** — §2 test-suite row refreshed: was "26 test modules" (27/8); now
  48 modules / 877 tests, re-verified green on the refactor working tree (12:17, 0 failures).
  The README Verification section cites the same counts.
- **2026-09-01 (deep-review refresh)** — three entries updated to stop describing closed
  gaps as open, after verifying code + issue state on `master` HEAD `562f024`:
  (a) §2 test-suite row refreshed again — 59 test modules / 1169 `def test_` definitions
      (was 48 / 877). The README Verification section still cites "1007 tests across 50+
      modules" and is now stale relative to both the audit and the source; flagged for a
      README refresh.
  (b) §4 role-word entry rewritten — issue #14 (closed 2026-08-29) converged the two
      lexicons via `extractor.set_role_words()` called from `provider_core.py` (init) and
      `provider_session.py` (per learned word). The "category wrong, alias right" gap
      described in the 2026-08-28 deep review is closed; 16 test references verify the
      convergence. The entry now records the implemented state, not the pre-fix gap.
  (c) Issues #23 (inbound security fail-open, closed 2026-08-28) and #24 (egress gate
      unknown-kind, closed 2026-08-29) are fixed in code: `reviewer.py:225-233` now returns
      `pending_user_confirmation` with `review_model: "inbound_security_unavailable"` on
      scanner import failure (fail-closed to human review); `egress.py:244-246` now returns
      `False` and logs a warning for unknown `kind` (fail-closed). The 2026-08-28 deep-review
      history entry above is left as-is (it records the *filing*); this entry records the
      *resolution* so the audit no longer implies either gap is open.
  No code behavior changed — this is a documentation-only refresh of the living index.

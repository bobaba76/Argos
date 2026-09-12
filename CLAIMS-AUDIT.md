# Claims Audit — Argos

**Date:** 2026-09-05 · **Audited:** README.md and docs vs `src/` and committed eval
artifacts on `master` · **Branch HEAD:** `8f4c860`

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
| Test suite | ✓ | 171 test modules in `argos_plugin/tests/` (3,347 `def test_` definitions; counts generated via AST 2026-09-12 by `scripts/count_test_fns.py`, guarded by `test_claims_audit_parity.py`; last recorded full-suite green run 2026-08-30 via `pytest tests/ -q -n 4` — not re-run for this refresh). Covers gate verdicts, egress, inbound security, adversarial chains, contradiction matrix, shared-service RPC, multitenant Cells, mutation_events audit log. |
| Public repo contains no personal data | ✓ verified | gold freeze sha documented in `eval/gold/README.md`. |

---

## 3. Claims flagged — measured internally, no public artifact yet

These measurements are not part of the public claim set until they have a
committed, re-runnable artifact:

| Claim | Status | What's missing |
|---|---|---|
| Self-corpus gate + personal bench | internal | maintained for the weekly recon |
| MemConflict 16-question slice — turn-level ingest (28/8 vs 30/8) | internal | Same 16 questions (persona 0, sessions 0–9), same answerer (deepseek-v4-flash via OpenRouter), same levers/prompt. AA 0.406 → 0.344 (one question, n=16), UOCS 0.188 → 0.438, CRS 0.188 → 0.125. **Mechanism caveat (chain-verify 30/8):** store-level chains are absent — 0 of 595 (and 0 of 884 in the 30/8 DB) records have `valid_to`/`superseded_by` set; ingest is `remember(dedup=False)`, so supersession/versioning never fires and chain-unfold has nothing to walk. The UOCS delta is the answerer's timestamp reasoning over the chronologically rendered list (prompt rule "latest wins"), not store update-arithmetic — the #8 store-level intent remains unproven at the benchmark level (filed as #74; #74 is now **closed** — `store.ingest_versioned()` landed 2026-09-01 and is unit-tested, but no chained LongMemEval run has been banked, so the caveat stands — see the 2026-09-07 resolution entry). Not comparable to the 13/8 180-question baseline (different harness, ingest, scorer handling). Artifacts live in the benchmark clone (`hermes-memconflict-fork`, `argosvault/Results/`), untracked — not banked, so not a public claim yet. |

---

## 4. Aspirational statements (not claims)

| Statement | Why it's not a claim yet |
|---|---|
| README Trust model: "Every feature is gated by a measurement in the eval harness" | Aspirational. Feature *areas* are tested, but not every shipped knob has a committed before/after measurement. The measured subset is §1; everything else is structural verification (§2). Treat the sentence as engineering intent, not an audited fact. |
| MEMORY_SYSTEM.md:102 + CONFIG_REFERENCE.md:79-82 + UI label "Stale-pending sweep": "periodically re-reviews proposals pending too long" | **Implemented (#10, 2026-09-01).** The four config keys are now consumed by `stale_review_sweep.py` — a daemon thread that runs every `stale_review_interval_min`, re-reviews only `pending` candidates older than `stale_review_min_age_min`, caps at `stale_review_max_batch`, and preserves the no-auto-promotion invariant (decision map identical to `review_pending.py`). Fail-soft on LLM error. Started by the provider after initialization; stopped on shutdown. |
| MEMORY_SYSTEM.md:103 + CONFIG_REFERENCE.md:83: "Role-word learning (role_alias_llm_fallback=true) — when an unknown word appears in 'my X is Name', the LLM is asked if X is a person-role; learned words persist to role_words" | **Implemented end-to-end (#14, closed 2026-08-29).** The two lexicons now converge: `extractor.set_role_words()` is called from `provider_core.py` at init (with the graph's defaults + config) and from `provider_session.py` each time the LLM ambiguity gate learns a new word, so `extractor._all_role_words()` (base + extra) and `graph._get_role_words()` (defaults + override + learned) stay in sync. A learned word like "doula" now correctly categorizes as `relationship` in the extractor *and* mints the alias in the graph. Verified by 16 `set_role_words`/`_set_role_words_override` test references in `test_hybrid_memory.py` (override extends the set, learned word extends the set, LLM ambiguity gate accepts/rejects, fallback-disabled skips the gate). The earlier "category wrong, alias right" gap described in the 2026-08-28 deep review is closed. |
| `config_model.py:74` + CONFIG_REFERENCE: `graph_traversal_enabled` (default **true** in code; the live `hybrid_memory.json` sets it false), `graph_traversal_boost=0.60`, `graph_traversal_depth=2` — BFS traversal from query seed entities | **Unproven — the #139 A/B was vacuous (#364, 2026-09-08).** The 2026-09-02 `traversal_on` vs `baseline` run on the self-corpus snapshot (1201 records, 300 queries) returned all 10 metrics byte-identical (MRR 0.9, nDCG@5 0.9119, nDCG@10 0.9166, nDCG@20 0.9174, deltas 0.0). Inspection shows the arm could not have moved anything: the harness builds its graph regex-only (`run_eval_provider.py` `build_snapshot_graph(use_llm=False)`), while `traversal_memory_ids` walks TYPED/LLM relations only (generic regex edges excluded via `_GRAPH_GENERIC_RELATIONS`) and returns `[]` unless >=1 grounded seed is a non-concept entity. A regex-only graph is dominated by generic edges and concept nodes, so traversal was expected to return `[]` on essentially every query — the A/B measured the regex graph, not the production hybrid (LLM-supplemented) graph. **Verdict: the production-like (typed-graph) variant remains unmeasured; the regex-graph variant engages without moving metrics.** The 2026-09-10 counter proof shows 97/300 queries engage on the regex graph with <=0.2pp deltas. The harness now records per-query engagement counters (seeds resolved / non-concept seeds / traversal_ids>0) in any traversal-enabled arm and warns when an arm never engaged; a typed-graph A/B (~5h+ LLM) is deferred until traversal is ever enabled in production (currently `graph_traversal_enabled=false` and `graph_inject_candidates=false` live → path dormant). The 2026-09-02 raw results were never committed (local-only under the gitignored `eval/bench/traversal_ab/`); the 2026-09-10 counter proof is banked — `eval/repro/TRAVERSAL_ENGAGEMENT.md` + `eval/repro/traversal_engagement_summary.json`. The typed-graph A/B remains deferred (#424). |

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
  store-level intent remains unproven. Filed as #74 (resolution recorded in the 2026-09-07
  entry below).
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
- **2026-09-07 (canary catch)** — §2 test-suite row refreshed: was 137 modules / 2,493 (6/9); now
  150 modules /  ​2,842 `def test_` definitions, counted via AST (`ast.walk`, UTF-8 BOM
  tolerated) by `scripts/count_test_fns.py`. The README Verification section quotes the same
  refresh (was 2,350 /  ​131 as of 2026-09-05). The parity canary had gone **red**
  (`test_claims_audit_parity.py` failed both params: module drift 8.7%, test drift
    12.3%, both exceeding the 5% tolerance) — test additions since the last recorded full-suite green
  run (30/8) merged without a suite run. Canary re-checked **green** after this refresh. This
  entry records the refresh and the catch, so the audit's living index stays honest.

- **2026-09-07 (#354 — #74 resolution entry)** — issue #74 ("benchmark ingest must fire
  store-level versioning") is **CLOSED**: the store-side fix landed 2026-09-01 (`6f06460`,
  follow-up `b982dd1`) as `store.ingest_versioned()` (`store_write.py:366`), which detects
  restatements (exact/substring/semantic) and routes them through `update_memory` so
  `valid_to`/`superseded_by` chains form; `tests/test_ingest_versioning.py` covers
  insert/duplicate/supersede/chain-walk/tombstone-block mechanics. **Proof-at-scale is NOT
  yet banked**: no LongMemEval/MemConflict adapter in this tree calls `ingest_versioned`
  (the benchmark adapters live in the sibling checkout), and no chained-run artifact
  (judged files + `verify_repro` entry showing >0 rows with `valid_to`/`superseded_by`) has
  been committed. So the §3 caveat above narrows from "machinery absent" to "machinery
  present, run not banked" and **stands until a chained run is committed**. This entry
  records the *resolution* (the 2026-08-30 chain-verify entry records the *filing*),
  mirroring the #23/#24 pattern, so the audit no longer implies #74 is an open gap.
  Hygiene note: when refreshing this audit, `gh issue view` every "filed as #N" reference
  (#10, #14, #74, #139, #325) before quoting it, and add a resolution line in the same pass
  when an issue has closed with a fix. No code behavior changed.

- **2026-09-07 (parity guard extended + wired into CI)** —the canary now
  guards **both** docs rows (CLAIMS-AUDIT §2 and the README Verification "Test suite"
  bullet) and runs on **every PR** as a tier-0 merge gate (alongside the retrieval
  smoke gate,.github/workflows/ci.yml)). Was: audit-row-only, and not part of CI —
  a red canary sat unnoticed from  ​30/8 to 7/9. The README row formerly had no guard.

- **2026-09-06 (#325)** — §2 test-suite row refreshed: was 59 modules / 1169 tests (01/9);
  now 137 modules / 2,493 `def test_` definitions, counted via AST (`ast.walk`, UTF-8 BOM
  tolerated) by the new `scripts/count_test_fns.py`. The README Verification section quotes
  2,350 / 131 (as of 2026-09-05). To stop this row rotting again,
  `tests/test_claims_audit_parity.py` parses the row and fails when the quoted counts drift
  more than 5% from the on-disk counts. `pytest --collect-only -q` remains authoritative for
  parametrized totals; the AST count is the definition floor.
- **2026-09-10 (#407)** — $0 regex-graph engagement proof banked:
  `eval/repro/TRAVERSAL_ENGAGEMENT.md` + `traversal_engagement_summary.json`. Counters:
  seeds resolved 298, non-concept seeds 131, **engaged 97/300 (32.3%)**; metric deltas
  <=0.2pp. The regex-graph arm is NOT dormant on the 2026-09-06 snapshot; contribution in
  that arm config ~nil. The typed-graph A/B stays deferred, tracked in #424. §4 row updated.
- **2026-09-08 (#364)** — the #139 "measured-and-flat" traversal verdict is **withdrawn**:
  the `traversal_on` arm never engaged. The harness graph is regex-only, traversal walks
  TYPED/LLM edges only and needs a non-concept seed, so `traversal_memory_ids` returned `[]`
  on essentially every query — byte-identical metrics are the no-op signature, not a
  measurement. §4 row rewritten to "unproven; only the regex-graph variant was tested and
  it never engaged". `run_eval_provider.py` now records per-query engagement counters
  (seeds resolved / non-concept seeds / traversal_ids>0) for traversal-enabled arms and
  warns on zero engagement; the base arm config pins `graph_traversal_enabled=false`
  (since #272 `MemoryConfig` defaults it to true, so an unpinned baseline would be
  config-identical to `traversal_on`). Also fixed the default contradiction: code default
  is `true` (`config_model.py:74`), not `false` as this file said. Typed-graph A/B deferred
  until traversal is enabled in production. #139 stays closed. Alias expansion
  (`provider_retrieval.py`, injectable regardless of `graph_inject_candidates`) is the only
  live graph retrieval contribution and is untouched.
- **2026-09-02** — `graph_traversal_enabled` A/B (#139) *(verdict withdrawn 2026-09-08, see
  #364 above)*: added `traversal_on` arm to
  `run_eval_provider.py`, ran on the self-corpus snapshot (1201 records, 300 queries)
  vs `baseline`. Result: **flat** — all 10 metrics identical (MRR 0.9, nDCG@5 0.9119,
  nDCG@10 0.9166, nDCG@20 0.9174, deltas 0.0). Recorded in §4 as measured-and-flat.
  Verdict: BFS traversal does not earn its keep; the graph is a boost signal, not a
  traversal engine. spec-10 Layer 2/3 (temporal traversal) should not be invested in.
  Results committed to `eval/bench/traversal_ab/`. Also: #138 `observed_at` capture
  landed (one-line in `graph.py:1035`), preserving the provenance option for future
  temporal work even though traversal itself is flat.
- **2026-09-02 (evening) — 2/9 review fixes landed**: Hermes re-audited the same-day
  G4/D6 changes after merge and found two gaps that the merged tests missed.
  (a) **G4 recheck**: the `memory_ids STRING[]` column migration (ALTER) never
  backfilled existing rows, and the old JSON-scan fallback was deleted — pre-migration
  edges became invisible to `remove_memory` (probe: old-format edge returned
  changed=False). Fixed: `_backfill_memory_ids()` one-time idempotent migration at
  first DB open (folds attrs `memory_ids` + legacy singular `memory_id` into the
  column), a NULL-column + quoted-id CONTAINS fallback clause in `remove_memory`, and
  singular-key folding in the cleanup loop. 4 new red-green tests.
  (b) **D6 recheck**: the token pre-filter + `LIMIT 50` conflict scan could drop the
  one conflicting row when >50 active records match any token (no ORDER BY made the
  window nondeterministic). Fixed: deterministic `ORDER BY created_at DESC, memory_id
  DESC` + full-scan escalation on cap hit — the answer no longer changes with row
  count; common path stays bounded. 2 new red-green tests (red proven on pre-fix code).
- **2026-09-02 (evening, part 2) — facade fail-closed doc/code mismatch fixed**: the
  class/`__init__`/module docstrings claimed API mode "fails closed (deny-all)" on an
  invalid ACL, while the code started with the open store and only warned. Worse, the
  real hole was in `ACLConfig.from_file`: a corrupted/unreadable config file silently
  degraded to an open store with NO way to distinguish it from a deliberately absent
  config. Fixed: `ACLConfig.parse_error` flag (set on unreadable/structurally-invalid
  configs; absent file stays clearly absent); API mode now refuses to start
  (ValueError) when `parse_error` is set — fail closed; absent config keeps the v1
  open-store + startup warning behavior (pinned by existing test). Docstrings updated
  to describe the actual contract. 4 new tests.
- **2026-09-05 — spec-10 PR-3/3: transports (MCP+REST) read+write, collections, docs**:
  The external API is now read+write (spec-09 read tier + spec-10 write tier).
  MCP stdio (`mcp_server.py`) exposes 16 tools: 6 read, 1 propose, 1 review, 2 write
  (class C loopback), 6 collection (2 read + 4 write). REST (`rest_server.py`) exposes
  the full read surface + `POST /v1/memories` (class C on loopback, class A propose
  otherwise), `POST /v1/candidates/{id}/decision` (class B human-only), `POST /v1/memories/{id}/feedback`,
  and collection CRUD (`POST/GET/PATCH/DELETE /v1/collections[/{id}/items[/{item_id}]]`).
  `Idempotency-Key` required on all mutations; CAS via `If-Match` → 409 on conflict.
  `principal_type` ("human"|"model") and `is_loopback` are wired by both transports —
  a model principal is denied class B (no self-approval); class C writes require loopback.
  The default `principal_type` is "model" (fail-closed): a transport that forgets to set
  `ARGOS_API_PRINCIPAL_TYPE` is treated as a model agent and cannot approve candidates.
  A human-driven UI must explicitly set `ARGOS_API_PRINCIPAL_TYPE=human` to unlock class B.
  The HMAC gate from PR-2 extends through transports: collection writes go through
  `facade → store proxy → call_gated() → _gate_hmac` (verified by the service with the
  boot-time `gate_secret`). Acceptance tests T1-T12: 39 tests in `test_spec10_transports.py`,
  all deterministic, no LLM calls. README updated from "read tier today" to "read+write tier."
- **2026-09-07 (#347) — mutation_events: append-only, actor-attributed mutation log**:
  Closes the audit/provenance gap identified in the Agent Memory Atlas review. A new
  `mutation_events` table (schema v5, migration `_migration_4_to_5`) records one event
  row per committed store mutation, written in the SAME DuckDB transaction as the
  mutation (fail-loud #330: a failed event write rolls back the mutation). Event types:
  `memory_created`, `candidate_created`, `candidate_approved`, `candidate_reviewed`,
  `candidate_downgraded`, `candidate_rejected`, `auto_approval_refused`,
  `memory_updated`, `memory_deleted`, `memory_restored`, `memory_erased`,
  `refeed_refused`, `tombstone_purged`, `rejection_purged`, `rejection_imported`,
  `ingest_versioned`, `conflict_resolved`. Actor identity is server-derived
  (`AuthContext.principal` + `principal_type` via `set_actor_context`; local paths
  default to `user_id`/"human"; credential mode forces "model" #341/#344). The ledgers
  (`deletion_tombstones`, `rejection_ledger`) STAY `INSERT OR REPLACE` — one row per
  key is load-bearing for the tombstone_check/rejection_check gates; `mutation_events`
  is where the history lives (two rejects of the same claim slot with different reasons
  preserve BOTH reasons, even though the ledger keeps only the last). `mutation_events`
  NEVER rotates — no startup purge, no cap (unlike `access_audit` which keeps its
  100k operational rotation for read stats). Denials are NOT routed into
  `mutation_events` — they scale with read volume (every denied search), not
  mutation volume, and the never-rotating log would grow unboundedly on a perm-fail
  path. Denials stay in `access_audit` (which rotates at 100k) for operational
  telemetry. Imported rows use their native event types (`memory_created`,
  `candidate_created`, `rejection_imported`, etc.) with `reason="import_portable:<rtype>"`;
  there is no `import_portable` event type (round-2 review: the round-1 denial-routing
  design was dropped for this reason). Read surface: `list_mutation_events`
  (paginated, scope-filtered, optional event_type filter, `ORDER BY ts DESC, seq DESC`
  for deterministic ordering) + `export_mutation_events` (JSONL/CSV, wheel/principals
  gate mirroring `export_access_audit`, offset paging for full history). RPC threaded
  through `memory_service.py` dispatch + `service_client.py` proxy (#312 precedent).
  Provenance begins at deployment/shipping date; no historical backfill is possible.
  30 tests in `test_mutation_events.py`: schema/migration, event coverage per mutation
  path, same-transaction atomicity (rollback removes event), ledger history preserved,
  no rotation at 100k+, actor context, scope filtering, denial routing, export formats,
  candidate_created per proposal, ingest_versioned per version-chain write,
  conflict_resolved per resolution (keep_old/keep_new/remove_both/manual).

- **2026-09-11 (#404)** — §2 test-suite row + README Verification bullet refreshed:
  was 156 modules / 2,952 `def test_` definitions (08/9); now 162 / 3,105, counted the
  same way (AST via `scripts/count_test_fns.py::count_suite`). The increase covers the
  merge waves since the last refresh (#427/#428, #429, #425, #404 — trust-model,
  tenant-policy, sweep-coordination, RPC and egress tests). No measurement claims
  changed; the row's "last recorded full-suite green run" line is unchanged.

- **2026-09-11 (#393 S1)** — §2 test-suite row + README Verification bullet refreshed
  again: 162 modules / 3,105 `def test_` definitions (this morning's refresh) now
  164 / 3,162, counted the same way (AST via `scripts/count_test_fns.py::count_suite`).
  The increase covers #439's in-service drift-watch suite and #393 S1's write-policy
  suite (`test_approval_mode.py`, plus approval-mode dispatch coverage in
  `test_tenant_policy.py`). No measurement claims changed.

- **2026-09-11 (#393 S2)** — §2 test-suite row + README Verification bullet refreshed
  again: 164 modules / 3,162 `def test_` definitions (the S1 refresh) now 165 / 3,201,
  counted the same way (AST via `scripts/count_test_fns.py::count_suite`). The increase
  covers the conflict-note precision follow-ups (#446/#449) and #393 S2's observability
  suite (`test_unreviewed_observability.py`: trust-class search filter, `unreviewed_stats`,
  RPC dispatch threading, facade scoping/restore, MCP + REST surfaces). No measurement
  claims changed.

- **2026-09-12 (#393 S3 + wave merges)** — §2 test-suite row + README Verification bullet
  refreshed again: 165 modules / 3,201 `def test_` definitions (the S2 refresh) now
  168 / 3,264, counted the same way (AST via `scripts/count_test_fns.py::count_suite`).
  The increase covers the overnight pipeline wave (#455-#470: retrieval-rescue, cost,
  and search-latency suites) and #393 S3's resolution suite (`test_review_memory.py`:
  promote/dismiss store semantics, rejection-ledger fingerprinting with the honest
  slot-less flag, resolution invariant, RPC class derivation, facade/model-principal
  gates, MCP + REST surfaces). No measurement claims changed.
- **2026-09-12 (#460 close-out)** — section 2 test-suite row + README Verification
  bullet refreshed: 168 / 3,265 (+1: `test_scale_warning_fires_once_per_crossing`).
  The #460 latency claims (warm p50=76ms / p95=202ms, warnings_fired=0 live) are
  runtime measurements, not suite counts; recorded in the issue thread.

- **2026-09-12 (#459)** - §2 test-suite row + README Verification bullet refreshed
  again: 168 modules / 3,265 `def test_` definitions (the #460 refresh) now
  169 / 3,267 (adds `test_pollution_guard.py`: +2, pinning the conftest
  import-state leak guard). #459's acceptance is a runtime property (a green
  full-suite run), not a count - evidence lives in the issue/PR thread. No
  measurement claims changed.

- **2026-09-12 (#386, Spec-12)** — §2 test-suite row + README Verification
  bullet refreshed: 169 / 3,267 → 170 / 3,279 (+1 module, +12: new
  `test_spec12_parity.py` pinning transport retrieval parity — MCP/REST vs the
  direct store, byte-identical ordered IDs — and the ingest wiring gates:
  apply is loopback-only (class-C posture), preview writes nothing, idempotency
  key required on both transports). No measurement claims changed.

- **2026-09-12 (#387, Spec-12)** — §2 test-suite row + README Verification
  bullet refreshed: 170 / 3,279 → 171 / 3,337 (+1 module, +58: new
  `test_api_credentials.py` pinning credential-backed transport identity —
  `api_credential.json` parsing with hash-at-rest, granular operation
  classes (erase/ingest/review never implied by propose), never-widen env
  posture, REST live revocation + expiry + 401/500 mapping, MCP fail-closed
  startup, and the class-B human-vs-model facade gates). No measurement
  claims changed.

- **2026-09-12 (#390, Spec-12)** — §2 test-suite row + README Verification
  bullet refreshed: 171 / 3,337 → 171 / 3,347 (+10 in
  `test_admin_console.py`): console credential auth (#387 machinery reused):
  human credential renders review buttons + approve routes through the
  facade; model credential refused class B (no buttons, 403, no store call);
  read-class sees no actions; expired/revoked/malformed credential → 401/500;
  legacy env path preserved (human by default, explicit `model` narrows).
  No measurement claims changed.

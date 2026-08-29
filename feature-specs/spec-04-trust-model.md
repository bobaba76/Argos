# Spec 4 — Trust-model cluster (batch-2): provenance taint, grounding, one-way ladder

Status: **IMPLEMENTED** (batch-2 cluster — issues #43, #40, #39, #35).

Three record-level metadata fields that gate **what a memory may do or become**,
designed together as one schema pass per the trust-model cluster consolidation.
A fourth issue (#35, quote verification) feeds #40's grounding field.

Reference design: [Agent Memory Atlas](https://neoneye.github.io/agent-memory-atlas/)
— the question that matters when the store is an injection surface is "what may
this memory be allowed to *do*?", and the labels are permanent from ingest.

## Schema (one additive migration)

`memory_records` and `memory_candidates` each gain two columns:

| Column | Values | Default (records / candidates) | Fail-closed to |
|---|---|---|---|
| `provenance_origin` (#43) | `internal` / `external` | `internal` / `internal` | `external` (stricter) |
| `grounding` (#40) | `observed` / `extracted` / `inferred` / `speculative` | `observed` / `extracted` | `speculative` (strictest) |

A new `rejection_ledger` table (#39) keys on `(subject, predicate, user_scope)`
— the **claim slot**, not the exact value — so paraphrased re-assertions of a
rejected fact are also blocked. Distinct from `deletion_tombstones` (exact
content hash). Reversible via `purge_rejection()`.

Backfill derives both labels from the existing `source` / `payload.external_source`
for pre-existing records (distill-derived and external-origin → `inferred`;
`llm_extraction` → `extracted`; `explicit`/`user`/`manual` → `observed`).

## #43 — Provenance taint (per-record, fail-closed)

- `normalize_provenance()` parses to `internal`/`external`; **unknown/corrupt →
  `external`** (the stricter class). A missing or tampered label can never widen
  a memory's blast radius.
- Set on write (`remember`/`save_candidate`): explicit label wins; otherwise
  derived from `payload.external_source`.
- Gate: external-origin memories cannot auto-activate. `review_candidate` with
  `review_source="auto_review"` downgrades an external-origin approval to
  `pending_user_confirmation` (mirrors the existing payload check but keys on the
  **permanent column**, which survives payload stripping/sanitization).
- Invariant: **sanitization/redaction does not alter the taint label.**
  `sanitize_content()` touches content only; `provenance_origin` is a column, not
  payload, so scrubbing content leaves the taint untouched.

## #40 — Grounding ceiling (4-level, monotonic)

- Ladder (low→high): `speculative` < `inferred` < `extracted` < `observed`.
- Defaults per write path (`default_grounding_for_write`):
  - direct user statement (`explicit`/`user`/`manual`) → `observed`
  - parsed/extracted (`llm_extraction`) → `extracted`
  - distill/model-derived (`distillation`) → `inferred`
  - external-origin ingest → `inferred`
  - anything unresolved → `speculative` (strictest)
- Ceiling per grounding caps the reachable trust class:
  - `speculative` → `pending_user_confirmation`
  - `inferred` → `reviewed_approved`
  - `extracted` / `observed` → `approved`
- Promotion/confirmation may raise a record's class but **never above its
  grounding**. **User confirmation lifts the grounding** (and the ceiling) to the
  minimum required for the requested class — the ceiling moves with grounding,
  not with use. Auto-review is capped and downgrades instead.
- **Recall counts are not verification**: `retrieval_count`/`helpful_count`
  increments never change `status` or `grounding` (asserted by test).
- Supersession keeps the record's class — correction lives on the status axis
  (no demotion-as-punishment).
- Distill-store artifacts ground as `inferred` (covers the live incident where a
  distilled third-person extract ranked into the top retrieval window).

## #39 — One-way trust ladder (rejection ledger)

### Audit (every status-changing path → verdict)

| Path | Moves toward active/verified | Checked against ledger? | Verdict |
|---|---|---|---|
| `save_candidate` | pending (proposal) | **Yes** — `rejection_check` blocks re-proposal of a rejected slot | Blocked at gate |
| `remember` | active (direct write) | **Yes** — `rejection_check` blocks re-creation | Blocked at gate |
| `review_candidate` approve | approved/reviewed_approved | Candidate never reaches review (blocked at `save_candidate`); new memory via `remember` re-checks | Blocked at gate |
| `review_candidate` reject | rejected | **Writes** the ledger (`record_rejection`) | Records |
| `review_candidate` supersede | new version chains | New memory via `remember` re-checks the slot | Blocked at gate |
| `restore_memory` | quarantined → active | Restores a *quarantined* record, not a rejected value; quarantined records were never ledger-rejected. Re-creation paths (`remember`/`save_candidate`) remain gated. | N/A (different axis) |
| `delete_memory` | hard-delete | Writes `deletion_tombstones` (exact-value ledger, separate) | Records (tombstone) |

**Result: no path can resurrect a rejected value without a NEW record passing
the gates.** The ledger keys on the claim slot, so a paraphrased re-assertion of
the same `(subject, predicate, scope)` is blocked; the only way back is
`purge_rejection()` (the user-facing escape hatch) or a new record whose slot
differs.

### Self-promotion guard
An agent may not self-promote its own candidates through any tool surface:
`review_candidate` with `review_source="auto_review"` can never write
`"approved"` (the existing approval invariant — `auto_review` is capped at
`reviewed_approved`, and the grounding ceiling further caps inferred/speculative
candidates). The `"approved"` transition is reserved for the agent-facing
confirmation tool (`review_source="tool"`) and manual callers, which represent
**user** confirmation, not agent self-promotion.

### Companion rules (held)
- Recall bookkeeping is not belief mutation — retrieval/helpful/dismissed
  counters never change `status` or `grounding` (test-asserted).
- Transaction atomicity (#9) wraps the review path so a crash mid-supersession
  cannot orphan the ledger entry.

## #35 — Quote verification (feeds #40)

`verify_quote_against_source()` (in `extractor.py`) deterministically greps a
claimed verbatim quote against the source transcript before labelling a memory
`verbatim`. On miss, the item is downgraded from `verbatim` to `inferred` and
the failure is logged with a countable counter surfaced in review. The
downgrade lands on #40's `grounding` field (the structural home). Cheap, zero
LLM. Normalization: whitespace/case-insensitive substring match with a small
tolerance for near-misses.

## Acceptance criteria mapping

- **#43**: fail-closed parse test ✓; external-origin cannot auto-activate but
  surfaces in retrieval ✓; redaction leaves taint unchanged ✓; backfill by
  source ✓.
- **#40**: grounding column with correct defaults ✓; invariant test
  (speculative/inferred cannot reach top class via promotion or recall) ✓;
  supersession keeps class ✓; distill artifacts grounded `inferred` ✓.
- **#39**: audit recorded (table above) ✓; no path resurrects a rejected value
  without a new record ✓; negative retrieval test after re-assertion ✓; agent
  cannot self-promote ✓.
- **#35**: verbatim only when quote found in source ✓; misses downgrade + logged
  with counter ✓; no behavior change for passing evidence ✓; found/missed/near
  -miss test cases ✓.

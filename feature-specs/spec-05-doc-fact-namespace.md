# Spec 5 — Doc-fact namespace: domain separation for document-sourced facts

Status: **APPROVED 2026-08-30** (design complete; implementation folded into the
god-file refactor window — see Decisions below).

## Problem

The commercial document tier (catalog → extraction → facts) will produce facts at a
volume and cadence that dwarfs conversational memories. Three failure modes follow:

1. **Injection cannibalization** — document-facts crowd conversational memories out
   of the 96-slot injection budget (and vice versa on queries that mention shared
   entities like banks, amounts, dates).
2. **Trust semantics differ** — document-facts carry a source document, extraction
   method (text vs OCR), and a freshness state tied to the file's lifecycle.
   Conversational memories carry none of that. Mixing them in one pool makes trust
   labels meaningless.
3. **Client scope** — inside a practice tenant (Cells, #49), every document-fact
   belongs to a client folder. Retrieval must be able to scope to a client without
   breaking global queries.

## The axis is SOURCE, not "personal vs business"

Correction caught in review (Michael, 30/8): the namespace is **where the fact came
from** (a conversation, or a document), NOT the user domain. Both deployment shapes
use it:

- **Personal store (Michael's):** conversation-sourced = personal-life facts;
  document-sourced = facts from his own files.
- **Practice store (accountant):** conversation-sourced = business chat notes
  ("we agreed to push Acme's year-end to Friday"); document-sourced = facts from
  client files.

The cannibalization problem exists in BOTH — the practice store will be flooded
with document-facts, and the small pool of decision-notes from conversations is
exactly what must survive.

## Design

### D1 — Same table, two new columns (mirror the proven `project_id` pattern)

`memory_records` and `memory_candidates` each gain:

| Column | Values | Default | Meaning |
|---|---|---|---|
| `namespace` | `conversation` / `document` | `conversation` | Source of the fact |
| `client_scope` | nullable VARCHAR | NULL | Client/folder id within a practice tenant; NULL = global |

Exactly one additive migration, same shape as the existing `project_id` column
(dataclass field → row mapping → save param → search filter). All existing machinery
— versioning, supersession, graph indexing, evidence rows, TTL, tombstones, trust
labels (spec-04) — applies unchanged. No new table, no forked query paths.

Rationale vs a separate table: a second table forks every query path and doubles the
RPC surface (the 25/8 include_closed lesson). A column is where `project_id` already
proved the path.

### D2 — Retrieval filters (backward compatible)

- `search()` gains optional `namespace` and `client_scope` kwargs; `None` = no
  filter (today's behaviour, nothing breaks).
- **Both** `service_client.py` (proxy signature) and `memory_service.py` (dispatch)
  gain the kwargs — prefetch fail-soft swallows TypeErrors as silent empty
  injections, so the RPC path is covered by tests (test_shared_service.py).
- `memory_search` default stays `namespace=None` (all); scoping is applied by the
  injection layer, not the base search.

### D3 — Injection partition (presence-aware, not a fixed split)

Total cap stays 96; item cap stays 800. **Floors reserve nothing for an empty
namespace** — they only bite when both sides are populated:

- **Mixed store (both namespaces non-empty):** `floor_conversation = min(24,
  available)`, `floor_document = min(24, available)`, remaining slots by unified
  score. Client-scoped queries invert: document floor 40, conversation floor 12.
- **Single-namespace store (e.g. a practice store before any chat notes exist):**
  no reservation — all 96 slots go to the populated side. Zero waste.
- **v2 (gated):** dynamic floors, tuned **only after** real document-facts exist
  and recall is measured both ways (measure-before-ship rule, 13/8). v1 floors are
  explicitly labelled untuned.

Deterministic, no LLM calls, no new deps.

### D4 — Trust/freshness fields: DEFERRED, but tracked (cannot be forgotten)

Part (b) — `source_doc_id`, `source_loc`, `extraction_method`, `verified_state`,
`extracted_at` — is deliberately out of v1 (their semantics belong to the
watcher/extraction tier, which is not designed yet). **Forget-proofing:** a parked
GitHub issue on the Feature Roadmap records that D4 activates when the watcher spec
lands. Issue status, not memory, is the tracking surface (Michael's convention).

## Explicitly OUT of scope

- Cross-client graph-entity leakage (shared directors/suppliers connecting client
  folders) — the access-control workstream, not the namespace.
- Watcher, catalog, ingestion, OCR — separate feature.
- Per-user ACLs (payroll/partner-only docs) — separate feature.

## Tests (cheapest falsifying first — deterministic, no LLM calls)

1. Save document-fact + conversation fact → namespace filter returns only that
   namespace (records *and* candidates tables).
2. `client_scope` filter correctness incl. NULL (global) rows.
3. Partition math unit tests: floors, remainder fill, client-scoped inversion,
   **empty-namespace short-circuit** (single-namespace store gets all 96).
4. RPC regression: shared-service proxy + dispatch with new kwargs (the 25/8
   failure class).
5. Full suite green (617 tests) — zero behaviour change when namespace is not used.

## Effort

Small-medium: schema × 2 tables + dataclass + row map + save param + 2 search
kwargs + proxy/dispatch threading + partition logic + ~12–15 tests. No new deps, no
LLM calls, additive migration only.

## Decisions (Michael, 30/8)

1. **v1 scope:** `namespace` + `client_scope` only; trust/freshness fields (D4)
   deferred → **parked as a tracked issue** so they cannot be forgotten.
2. **Floors:** static, **presence-aware** floors v1; dynamic v2 gated on
   measurement. Axis renamed `conversation`/`document` (source), not
   personal/business — his review caught the original naming bug.
3. **Sequencing:** implementation **folded into the god-file refactor window**
   (after #20 RPC-hardening and #49 Cells land), per his call — not a standalone
   PR now. Design is complete and parked; nothing is lost by waiting.
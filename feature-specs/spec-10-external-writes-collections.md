# Spec 10 — External writes (classes A/B/C) + Collections (exhaustive store)

Status: **DRAFT 2026-09-03** — plan, not shipped behavior. Implementation is
tracked via GitHub issues on the Feature Roadmap (#2); issue numbers are
assigned at open. Contents are read as the plan, never as a description of
live behavior.

## Problem

Spec-09 shipped the external read tier (MCP stdio + REST loopback, #123–#126).
Two gaps remain, and they block the same goal: making Argos usable *by other
agents*, not just readable by them.

1. **Read-only is a half loop.** An external agent (Claude Code, Cursor, a
   script, a bot) can search Argos but cannot save to it. "Save now, retrieve
   months later" requires the write half. Every write class, idempotency
   registry, and audit shape needed for this was already designed in spec-09
   (D1, D2, D5, D7) — this spec lands them behind the same facade, nothing new
   to invent.
2. **No exhaustive storage path.** Ranked retrieval can drop item 9 of 9.
   Backlogs, task lists, checklists, and "what is still open?" questions need
   a path where **missing one item is a failure** — a collection store with
   no top-N cutoff. Without the write tier, collections would only ever be
   populated by the native plugin, so the two land together.

Verified current state (3/9, git log receipts in brackets):

- Facade (`api_facade.py`), MCP stdio (`mcp_server.py`), REST
  (`rest_server.py`) are live on master (b782691, #166).
- The internal dispatch still exposes privileged ops (`shutdown`, `backup`,
  `set_state`, `mark_superseded`, `purge_tombstone`, `clear_scope`,
  `cleanup_junk`) — the facade allowlist remains the only public surface.
- Provider-level side effects (graph indexing, structural-loss guard,
  candidate promotion, superseded-evidence removal) live in
  `provider_session.py` — writes over the API must route through provider
  semantics, never the raw store (spec-09 problem #2).
- Cells (per-tenant stores) is merged and end-to-end tested (#130/#131).
  Collections carry the same tenant/scope columns and isolation gates.

## Design

### W1 — Write classes over the facade (spec-09 policy, now implemented)

The three classes from spec-09, enforced server-side by the facade:

| Class | Meaning | Allowed principals | Effect |
|---|---|---|---|
| A | propose — never active | any authenticated external client | always creates a *candidate*, never an active memory |
| B | human-approve | human principals only (admin token / native UI) | decides candidates; **no model self-approval**, even with `review_source="tool"` |
| C | trusted-local privileged | loopback + server-derived identity only | direct `memory_save`-equivalent writes, same semantics as the native path |

- **Server-derived identity stays mandatory.** Client-supplied `user_id`,
  `tenant`, `project`, `client_scope` fields are rejected or narrowed to the
  principal's server-derived scope (spec-09 test #2 semantics).
- **API ACL fails closed** (spec-09 decision #3): missing/corrupt ACL →
  refused readiness, never an open store.
- **Idempotency keys + compare-and-set on every write op** — this time full
  coverage, not graduated: writes are the milestone. Key registry + CAS
  table/logic already built in #123.
- **Audit rows** for every write, denied included (spec-09 D10 shape).
- **No raw RPC passthrough, ever.** The allowlist grows, not the boundary.

### W2 — Public write surface

MCP (stdio, same tool tiers as spec-09 D2):

- `memory_save` → class C on loopback; class A (candidate) from any external
  principal otherwise.
- `memory_update` → class C only (external updates are a B decision path via
  candidates in v1).
- `memory_candidate_review` → class B, human principal required.
- Collection tools (below) → class C on loopback; class A/B otherwise.

REST (loopback, `127.0.0.1` only; same hardening as spec-09 D7):

```
POST /v1/memories                # class A: propose a memory (candidate)
GET  /v1/candidates              # class B: list pending (human approver)
POST /v1/candidates/{id}/decision  # class B: approve/reject (human only)
POST /v1/memories/{id}/feedback  # class C loopback / class A external
POST /v1/collections             # class C / class A
POST /v1/collections/{id}/items  # class C / class A
GET  /v1/collections/{id}/items  # read tier, exhaustive (no top-N)
PATCH /v1/collections/{id}/items/{item_id}  # class C / class A
```

- Idempotency header `Idempotency-Key` required on all POSTs; CAS via
  `If-Match` / `expected_version` on PATCH → `409` on conflict.
- **Never public:** delete/quarantine/restore, backup, shutdown, graph
  mutation, tombstone purge, maintenance (unchanged from spec-09).
- No general list/export endpoint (spec-09 D7) — collection reads are the
  only exhaustive walks, and only over the caller's own scope.

### C1 — Collections: exhaustive, structural, no ranking

The guarantee is **structural**: items are stored in their own tables and
returned by plain filtered SQL — no similarity score, no top-K, no cutoff.
"Exhaustive" is a storage-class property, not a ranking tweak.

Tables (same DuckDB, same shared service owner — never a second writer):

```
collections (
  collection_id   TEXT PRIMARY KEY,   -- server-minted
  name            TEXT NOT NULL,
  template        TEXT,               -- 'backlog' | 'todo' | 'reading-list' | ... optional
  schema          JSON,               -- optional field validation (name/type/required); v1 minimal
  status          TEXT DEFAULT 'active',
  user_scope      TEXT,               -- Cells isolation (spec-06 typed scope)
  tenant          TEXT,
  created_at / updated_at
)
collection_items (
  item_id         TEXT PRIMARY KEY,   -- server-minted
  collection_id   TEXT REFERENCES collections,
  fields          JSON,               -- free-form v1; validated when schema is set
  status          TEXT DEFAULT 'open',  -- open | done | parked (enum, configurable later)
  user_scope / tenant,
  created_at / updated_at / archived_at
)
```

- **Explicit writes only in v1.** `memory_collection_create / add / items /
  update / remove`. No extractor, no candidate pipeline for collections —
  exactly like `memory_save` being the explicit exception in the trust model.
  An agent that wants a backlog item says so in plain language.
- **Exhaustive reads:** `memory_collection_items` returns every item matching
  the filter, every time. No ranking, no top-N. Filter by status/scope only.
- **Zero regression risk:** `_search_memories` and all tuned retrieval paths
  are untouched. Collections are additive tables + new facade ops.
- **Isolation:** collection reads/writes are scope-filtered like everything
  else; the Cells end-to-end isolation gate (#131) extends to collections.
- **No graph edges on items in v1.** Linking a task to the decision that
  spawned it is a nice-to-have; Kuzu edges for items are deferred until a
  named consumer needs them.
- **Custom schemas deferred.** Field validation beyond a minimal
  name/type/required check is out of v1 (the store keeps the JSON; the
  validation layer is a later, additive step).

### P1 — Packaging tie-in (context, not scope)

This milestone is the prerequisite for the standalone-server play (`argos
serve` / single artifact / Docker): an agent wired to a standalone Argos must
be able to write, or the artifact is a read-only window. Packaging itself is a
**separate milestone** after this lands and its acceptance tests pass.

- Hermes plugin path is unchanged and remains the richest surface; the
  standalone server is peer, not replacement.

## Explicitly OUT of scope (v1)

- MCP client (spec-09 D12 — Themis phase).
- `memory://` resources; raw graph tools over the API; list/export endpoints.
- Custom collection schemas beyond minimal validation; graph edges on items.
- Delete/quarantine/restore over the API; admin ops.
- Streamable HTTP; new HTTP framework (stdlib behind the facade, spec-09 #10).
- Packaging/binary/Docker (separate milestone, gated on this one).

## Tests (cheapest falsifying first — deterministic, no LLM calls)

Runnable against a disposable HERMES_HOME with no personal data. The
foundational issue is not done until these pass:

1. **Allowlist:** `shutdown`, `backup`, `set_state`, `clear_scope`,
   `purge_tombstone`, `mark_superseded` remain unreachable through MCP/REST.
2. **Identity spoof:** principal A submits B's `user_id`/`tenant`/`scope` —
   denied or narrowed; never widened.
3. **Malformed ACL:** corrupt ACL file → API fails closed or refuses
   readiness.
4. **Idempotent ingest:** same `Idempotency-Key` twice, including a simulated
   client timeout → exactly one candidate, no duplicate.
5. **No model self-approval:** MCP/model principal cannot approve its own
   candidate, even with `review_source="tool"`.
6. **Class A never active:** external `POST /v1/memories` always produces a
   candidate; nothing enters active memory without a class-B human decision
   (or loopback class C).
7. **CAS conflict:** PATCH with stale `expected_version` → 409, no write.
8. **Provider side effects:** a class-C write through the API indexes the
   graph and chains versions exactly like the native path (compare against a
   native-path control).
9. **Collection exhaustiveness:** 9-item backlog, filter `status=open` →
   all 9 returned, no cutoff; 100-item list → all 100.
10. **Collection isolation:** tenant A's items never visible to tenant B,
    including counts and existence.
11. **Error envelope:** force a store failure → stable error code + request
    ID, never traceback/path/token/SQL detail.
12. **MCP stdio discipline:** write call transcript — every stdout line valid
    MCP JSON-RPC; no banners.

## Handoff (Devin-ready)

- **STATE:** Spec-09 facade/tests landed (master). Read tier live. This spec
  adds the write tier + collections behind the *same* facade; the spec is the
  plan. Read-only remains until this lands.
- **TASKS (build order):**
  1. Circuit-breaker probe: 20 "what's still open?" prompts against a
     100-item synthetic backlog via ranked search — confirm a miss exists
     (falsifiable gate; if zero misses, stop and report — do not build).
  2. Facade write ops (W1/W2): classes A/B/C + server-derived identity +
     idempotency/CAS full coverage + audit rows (extend #123 infra).
  3. Collections tables + tools (C1) with tenant/scope columns.
  4. MCP tool additions + REST endpoints (W2), stdio/loopback discipline.
  5. Acceptance tests 1–12; hermetic gate.
  6. Update CLAIMS-AUDIT.md and README (API becomes read+write; collections
     documented; README lag is a known liability — fix in the same PR).
- **GOTCHAS:**
  - New facade ops must be threaded through **both** `service_client.py`
    (proxy signature) **and** `memory_service.py` (dispatch), and stubs must
    accept `**kw` — a kwargs gap TypeErrors at the RPC seam and prefetch
    fail-soft swallows it into silent empty injections (25/8 incident).
  - Writes must reuse provider-level entry points (`provider_session`
    semantics) — raw-store calls skip graph indexing, the structural-loss
    guard, and candidate promotion silently.
  - Hermetic tests: `ARGOS_HERMETIC_TESTS=1`; LLM-path tests must stub
    `sys.modules['agent']` and pass config explicitly; never edit source
    mid-run.
  - Pre-commit hook blocks commits when `argos_plugin/` drifts from the live
    install — work in the repo, sync separately; never edit the live copy.
  - ACL must fail closed in API mode; the personal-mode open-store fallback
    is native-only (spec-09 decision #3) — do not flatten it.
  - Embedding by-name loads hang on network probes — any new model is loaded
    by local cached snapshot path only.
  - License stays BSL 1.1; no competitor names in the tracked tree
    (neutral wording only).
  - No personal content in tests/fixtures — use synthetic personas.

## Effort

| Direction | Complexity |
|---|---|
| Facade write ops (classes, idempotency, CAS, audit) | Medium (infra exists) |
| Collection tables + tools | Low |
| MCP write tools | Low–medium |
| REST write endpoints + hardening | Medium |
| Acceptance tests 1–12 | Medium |

## Decisions (maintainer, for sign-off)

1. **Collections are explicit-write only in v1** (no extractor, no candidate
   pipeline for items).
2. **Write-class mapping** as W1: external MCP/REST = class A propose;
   decisions = class B human-only; loopback + server-derived identity =
   class C.
3. **No new HTTP framework** — stdlib behind the facade (spec-09 #10).
4. **Idempotency full coverage now** (not graduated): writes are this
   milestone.
5. **Collection items are plain SQL, no embeddings/ranking; graph edges on
   items deferred.**
6. **Read-only stays until this lands** — no partial "writes on REST, not
   MCP" shipping.
7. **Packaging is a separate milestone**, gated on this one's acceptance
   tests.
8. **Issue mapping:** one foundational issue (facade writes + collections +
   tests as a unit) or split by layer (facade → collections → transports) —
   maintainer's call at open; status lives on the Feature Roadmap (#2).
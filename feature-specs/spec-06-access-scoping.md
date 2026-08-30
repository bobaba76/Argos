# Spec 6 — Access scoping: per-user, per-client access control inside a practice tenant

Status: **APPROVED 2026-08-30** (design + decisions signed off; implementation
parked behind Cells (#49) and spec-05 (#67) — see Decisions below).

## Problem

The commercial sale dies if any staff member can query any client's facts.
Inside a practice tenant, payroll and partner-only files sit next to everything
else. The deal-killer threat model, five concrete cases:

1. **Client-bound facts** — a staffer asks about Client X's VAT number. Fine if
   X is in their assignment; invisible if not.
2. **Practice-internal files** — the practice's own payroll and partner
   profit-share. Principals-only; no staff, ever.
3. **Cross-client graph leak** — Client A and Client B share a director. A
   query about that director inside A's scope must not surface B's facts
   through the relationship graph.
4. **"Must never know this client exists"** — a hostile-divorce file, a client
   suing the practice. No hint in results, no hint in language-model output.
5. **Audit** — who asked what, when, including denied attempts. Exportable.
   This is the trust win that sells the practice.

## The one master rule

**Facts inherit access from their source document.** A fact extracted from a
payroll PDF inside Client X's folder carries that folder's access. Never more
permissive than the source. This single rule collapses most of the design.

## Design

### D1 — Model: role-based allow masks + hidden deny list

- **Allow (the normal world):** each staff role holds an allow mask = the set
  of client scopes it may query. Principals/partners default to **wheel**
  (all client scopes). Staff roles are assigned scopes.
- **Deny (the exception world):** a per-user or per-role deny list for content
  that must not exist for that user. **Precedence: deny > allow > wheel.**
- **Deny semantics are hidden:** excluded content never appears — not in
  results, not as a hint in any response. The access event IS recorded in the
  audit log (`excluded: true`). Logged, withheld.
- **Admin surface:** per-tenant ACL config file (JSON/YAML sidecar, edited by
  the principal). No UI in v1; a management UI is a later product shape.
- Deterministic evaluation, zero LLM calls.

### D2 — Inheritance: facts ← docs ← folders

- **Catalog pass** assigns each document `client_scope` + `doc_class` from the
  folder→(client, class) mapping convention (same pass that builds the
  catalog; the folder layout IS the ACL).
- **Extraction** writes facts carrying their source doc's `client_scope` and
  `doc_class` (spec-05 columns; never more permissive than the source).
- **Practice-internal** = reserved `doc_class` `practice-internal`
  (client_scope NULL) → principals-only default.
- **Explicit override sidecar** for files that don't fit the folder convention.

### D3 — Enforcement points (three, all deterministic)

1. **Retrieval:** query arrives with user identity → effective mask → docs
   *and* facts filtered **before** ranking (records + candidates; `namespace=`
   `document`, `client_scope` in mask, `doc_class` policy).
2. **Graph traversal:** every entity expansion is pruned **post-traversal** —
   a shared director's neighbours in client B are dropped for a user without
   B. The graph is the leak vector; it gets its own filter, not just the
   retrieval filter.
3. **Injection/answer defence-in-depth:** injected facts are re-validated
   against the mask; citations can only point at permitted docs (automatic
   once retrieval is filtered, but asserted, not assumed).

### D4 — Audit log (append-only, from pilot day one)

- New table `access_audit`: `ts`, `tenant`, `user`, `query` (text), result
  counts, `granted`/`denied`, denied scope. Exportable via service API
  (CSV/JSONL).
- **Logged:** every query, and every deny attempt (content withheld, event
  recorded — the `excluded: true` rule).
- **Seeded in the pilot even though the pilot is single-user.** Retrofitting
  "who" into logs later is painful; the pilot's audit trail is also the demo
  artifact for the commercial pitch ("every question is logged").
- The audit log itself contains client data (query text) → it is
  **practice-internal, principals-only** to read, and rotates (configurable
  window, default 90 days online). Retention is the ops workstream's clock;
  no auto-purge of events in v1 beyond rotation.

### D5 — Identity prerequisite (typed `user_scope`)

`user_scope` is currently stored as JSON-extract. Enforcement needs a **typed
column + composite index** (flagged 12/8). This is a prerequisite for
production enforcement, lands with the spec-05 migration family, and
defaults to today's behaviour when unset.

### D6 — Phasing

- **Pilot (single-user, the principal):** audit skeleton + folder→(client,
  class) mapping at catalog. **No enforcement** — the sole user is the
  principal; the mappings and audit trail are what prove the design.
- **v2 (second staff user joins):** enforcement ON — retrieval filters, graph
  guard, deny list, ACL config file. This is the hard commercial gate: it
  flips on when the deployment stops being single-person.
- Depends on: Cells (#49, the between-tenant wall — this spec is the
  inside-tenant wall), spec-05 columns (#67), typed `user_scope` (D5).

## Explicitly OUT of scope

- **Cloud-vs-local boundary** (client data leaving the premises to the LLM
  API; POPIA cross-border) — its own session; practices ask "where does the
  data go" before they ask about features.
- **Retention/purge policy** — the ops workstream (catalog gives the
  calendar; never auto-delete).
- **ACL management UI** — config file v1, UI later.
- **Between-tenant isolation** — that is Cells (#49) itself, not this spec.

## Tests (cheapest falsifying first — deterministic, no LLM calls)

1. Mask evaluation unit tests: allow-only user, wheel user, deny-beats-allow,
   deny-beats-wheel, NULL-scope rows.
2. Retrieval filter: document-facts outside the mask never returned (records
   *and* candidates).
3. Graph guard: shared-entity expansion drops cross-scope neighbours.
4. Audit: every query and every deny writes a row; export works; single-user
   pilot mode still records identity.
5. Full suite green — **no ACL config = today's behaviour** (backward
   compatible; absence of a config file opens the store exactly as now).

## Effort

Medium. Rides the spec-05 migration family: typed `user_scope` + `doc_class`
columns, ACL config loader, retrieval + graph filters, `access_audit` table,
~15–20 tests. No new deps, no LLM calls, additive only.

## Decisions (Michael, 30/8 — signed off)

1. **Allow + deny, both.** Role masks for normal work; hidden deny list that
   wins over everything, including principals.
2. **Audit seeded in the pilot, day one** — identity + timestamp + outcome on
   every query, including denies; exportable.
3. **Identity typed now** — `user_scope` becomes a typed, indexed column as a
   prerequisite; nothing ships with JSON-extract identity.
4. **Sequencing:** implementation parked behind Cells (#49) and spec-05
   (#67); the pilot ships only the audit skeleton + folder mapping, so the
   second staff user just flips enforcement on.
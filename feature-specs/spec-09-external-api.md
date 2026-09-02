# Spec 9 — External API: MCP server, REST, MCP client (trust-boundary facade)

Status: **DRAFT 2026-09-02** — plan, not shipped behavior. Implementation is
tracked as issues #123–#126 and lands behind this spec's Gate. Contents are
read as the plan, never as a description of live behavior.

## Problem

Argos is reachable today by exactly one owner: the local shared service
(`memory_service.py` TCP JSON-lines on `127.0.0.1`, via `service_client.py`).
Adding MCP and REST is not a transport problem — it is opening a **new trust
boundary** over a store that already holds powerful internal capability.
Verified current state (2/9, `git log` receipts in brackets):

1. **The internal dispatch exposes privileged operations** to any caller that
   reaches it: `shutdown`, `backup`, `set_state`, `mark_superseded`,
   `purge_tombstone`, `clear_scope`, `cleanup_junk`, plus graph mutation
   (`memory_service.py:344–517`). A public adapter that forwards to this
   dispatch is a security hole, not an API.
2. **Provider behavior lives above the raw store.** `provider_session.py`
   indexes the graph after memory writes, applies the structural-loss guard on
   update, indexes promoted candidates and removes superseded graph evidence,
   and purges junk entities at session end (`provider_session.py:486–490,
   655–695`). A handler that calls the raw store directly skips these side
   effects silently.
3. **Error envelopes ship a `traceback` fragment** to clients
   (`memory_service.py:624–629`) — fine for one owner, unacceptable across a
   boundary.
4. **Identity is caller-supplied today.** The endpoint-token holder can spoof
   any `user_id`; when strict routing is off, unknown users fall back to the
   default tenant (`memory_service.py:158–161, 240–251`).
5. **The ACL defaults to an open store** when config is missing or unreadable
   (`access_scoping.py:41, 66–74, 91, 108`). Backward-compatible for personal
   mode; dangerous as an API default.
6. **Graph access is filtered post-traversal** (`access_scoping.py:229–248`) —
   better than nothing, not good enough to expose raw graph tools to external
   callers.

## The one master rule

**MCP and REST are not aliases for the existing provider tools.** Every
external operation passes through a canonical application facade that applies
identity, authorization, validation, idempotency, audit, and redaction before
anything touches `service_client.py`. There is no raw RPC passthrough, no raw
SQL, no arbitrary graph-method forwarding.

## Design

### D1 — Canonical application facade

```
MCP stdio / REST / (future Streamable HTTP)
        ↓
transport adapter        (protocol only: framing, errors, limits)
        ↓
Argos application facade
  · authentication context      (who is calling, from what credential)
  · authorization / ACL         (operation allowlist + spec-06 masks/denies)
  · input validation            (strict schemas, bounds, enums)
  · idempotency                 (key registry, compare-and-set)
  · audit event                 (one row per operation, denied included)
  · candidate/review policy     (write classes A/B/C, never bypass review)
  · output redaction            (provenance metadata, no evidence by default)
        ↓
service_client.py → memory_service.py → store / graph
```

- Both MCP and REST call the **same facade operations** — never independently
  reproduce provider behavior.
- Facade operations wrap the same provider-level entry points that carry the
  graph/side-effect behavior (`provider_session` semantics), not raw store
  methods.
- **Single policy boundary (design note):** the facade is also the natural
  future home for the Hermes-native plugin path, so the API never gets safer
  than the native path. Not a v1 requirement; noted so the boundary is drawn
  once.

### D2 — Public operation allowlist

The facade exposes an explicit, enumerated set of operations. Anything not on
the allowlist returns `410 gone` / `method_not_allowed` — never a forwarding
attempt.

| Tier | Example operations | Default for external principals |
|---|---|---|
| Read | `search`, `fetch` (by id/history), `capabilities` | available |
| Proposal | `memory_propose` / `memory_ingest` (class A) | with write scope |
| Feedback | `helpful` / `dismissed` | separately scoped |
| Human review | `approve` / `reject` (class B) | hidden or human-only |
| Destructive/admin | `delete`, `quarantine`, `restore`, tombstone purge, maintenance, backup | **not exposed to agents**; privileged or absent |
| Graph | ACL-filtered graph-enriched search only | deferred until per-hop scope filtering |

**Never exposed on the public boundary** (internal-only, remain where they
are): `shutdown`, `backup`, `set_state`, `clear_scope`, `purge_tombstone`,
`mark_superseded` (except through the review path), `cleanup_junk`, raw graph
mutation, maintenance operations, arbitrary RPC forwarding, raw SQL.

Tool annotations such as "read-only" or "destructive" are hints for clients,
**not enforcement** — permissions are enforced server-side in the facade.

### D3 — Identity, scope, provenance, audit-actor

These are separate concepts and are modeled separately:

| Concept | Meaning | Source |
|---|---|---|
| Authentication | which client/process is calling | verified credential (token file), **never request body** |
| Principal | which user/service/tenant/worker it represents | server-derived from credential |
| Authorization | which operations the principal may perform | allowlist + ACL |
| Data scope | which project/client namespace/doc class it may access | server-derived maximum; caller filters may only narrow |
| Provenance | where content came from, how trustworthy | server-set at ingest |
| Audit actor | who performed the operation | server-derived principal + client id |

**Normative rule: the authenticated principal, tenant, user scope, and maximum
permitted data scope are server-derived. Client-supplied identity or tenant
fields are rejected or ignored; they must never widen access.** A request
body containing `user_id`, `tenant`, `project_id`, or `client_scope` that
differs from the credential-derived scope is narrowed to the server scope or
rejected outright (`403`).

Token separation: the internal service endpoint token is **never** reused as
the public API credential and never exposed to clients. External API mode uses
its own key material.

**API-mode ACL is fail-closed:**
- strict tenant routing is **mandatory** when API mode is enabled;
- unknown principals are **denied**, never mapped to a default tenant;
- API mode **refuses to start** if the ACL config is invalid/unreadable
  (no silent fallback to the open store);
- requires a small Argos-side change: a fail-closed flag in `access_scoping.py`
  (today: absent/unreadable ACL → open store; API mode inverts that default).

### D4 — Write classes and provenance

Three classes, exactly one of which is ever exposed per principal:

- **A — Proposal ingest:** external caller → candidate → security scan →
  review queue → human decision. **Never creates active memory.** The exposed
  operation is named `memory_propose` / `memory_ingest`, *not* `memory_save`,
  so the name cannot imply immediate activation.
- **B — Human-confirmed approval:** approval requires a genuinely
  human-authorized principal (local UI/CLI confirmation, or a short-lived
  single-use approval handle issued after a human action), or a separately
  authenticated human principal. **An MCP/REST model principal can never
  approve its own candidate** — `{"decision": "approved", "review_source":
  "tool"}` is rejected by policy. Approval tools are hidden from ordinary
  agents.
- **C — Trusted local explicit write:** the current personal-store
  `memory_save` behavior, preserved only as a **privileged, separately
  documented local-user operation**. "API" is never synonymous with "trusted
  explicit user write."

Server-set provenance fields on every external write (caller may not claim
these):

```
text source            = "api"
transport              = "mcp-stdio" | "rest"
authenticated_client_id= <credential-derived>
request_id             = <server-assigned>
provenance_origin      = "external" (models may never claim "internal")
grounding              = "observed" | "extracted" | "inferred" | "speculative" — caller may not claim observed
```

`source="api"` alone is not provenance; the full tuple is mandatory.

### D5 — Idempotency and optimistic concurrency

The internal service deliberately never retries timeouts (a timed-out request
may have completed). External clients, proxies, and hosts **will** retry. So:

- Every mutation carries an idempotency key: `Idempotency-Key` header (REST)
  or explicit `idempotency_key` argument (MCP stdio). The JSON-RPC `id` is
  **not** a durable mutation identity across restarts.
- New table `api_idempotency`: `idempotency_key`, `authenticated_principal`,
  `operation`, `request_hash`, `created_at`, `result/status`.
- Semantics:
  - same key + same request → return original result, no duplicate mutation;
  - same key + different body → `409 conflict`;
  - no key + connection timeout → caller must query
    `operation/status` or accept at-least-once.
- Applies to: candidate ingest, approve/reject decisions, updates,
  delete/quarantine/restore, bulk operations.
- **Optimistic concurrency:** candidate decisions are compare-and-set state
  transitions — approving an already-decided candidate returns a stable
  conflict result, never a silent overwrite. Updates support
  `expected_version` (REST: `If-Match`) when writes land.
- Graduated scope: v1 (MCP propose + REST read) requires idempotency on
  proposal ingest and (when they ship) decisions; full coverage lands with the
  REST write slice. The table and core logic are built in the foundational
  issue regardless.

### D6 — MCP server transport

- **Protocol era — explicit choice (Decision 2):**
  - Option 1 (recommended): modern-only, current protocol revision
    (2026-07-28 line): stateless per-request metadata, `server/discover`,
    current Streamable HTTP vocabulary when remote is needed later.
  - Option 2: dual-era server (modern + legacy initialize/session).
  - Do **not** copy the Engram skeleton blindly — Engram pins `mcp>=1.9,<2.0`
    (modern SDK, verified 2/9), but its transport code still needs a
    conformance check against the revision we adopt before reuse.
- **stdio discipline (mandatory, tested):**
  - stdout carries only valid MCP JSON-RPC messages;
  - logs go to stderr;
  - UTF-8, newline framing;
  - no startup banners on stdout;
  - bounded message size;
  - explicit process exit/restart behavior;
  - configured HERMES_HOME/profile — never an arbitrary caller-selected data
    path;
  - deterministic tool ordering.
- **Tool tiers** as in D2's table; enforcement server-side.
- **Strict input schemas:** `additionalProperties: false`; max string
  lengths; max result counts; enum validation; bounded arrays; **no
  caller-controlled internal flags** (`include_quarantined`,
  `include_archived`, `include_expired`) unless separately authorized.
- **Output contract:** `outputSchema` + structured results; never the
  provider's raw JSON strings as the API contract.
- **No `memory://` resources in v1.** Tools are sufficient; a resource model
  adds a second URI authorization + redaction surface for no v1 consumer.

### D7 — REST surface

Binding and browser-protection rules:

- Bind `127.0.0.1` by default; explicit IPv6 loopback handling; **no
  `0.0.0.0`**; no automatic tailnet-interface binding. Remote access only via
  an explicitly configured SSH/tunnel/reverse-proxy mode.
- Loopback does **not** protect against malicious local processes, browser
  cross-site requests, DNS rebinding, proxy misconfiguration, or a shared
  machine with weak file/IPC permissions — so:
  - validate `Host`; validate `Origin`;
  - no CORS by default; exact configured origins only; no wildcard CORS with
    credentials;
  - CSRF protection if cookies are ever introduced;
  - never accept tokens in query parameters;
  - `Cache-Control: no-store` on memory responses;
  - never log request bodies or authorization headers.

Initial read slice (nothing else):

```
GET  /v1/health
GET  /v1/capabilities
POST /v1/memory/search
GET  /v1/memories/{memory_id}
GET  /v1/memories/{memory_id}/history
```

(Optional, dedicated permission only: `GET /v1/memories/{id}/provenance`.)

- **No general list/export endpoint** — a caller could otherwise walk the
  whole store one page at a time.
- Later, with a named consumer: `POST /v1/candidates`,
  `GET /v1/candidates`, `POST /v1/candidates/{id}/decision`,
  `POST /v1/memories/{id}/feedback`.
- High-risk operations stay separate and privileged:
  `DELETE /v1/memories/{id}`, `POST /v1/memories/{id}/quarantine`,
  `POST /v1/memories/{id}/restore`.
- **Never public:** backup, shutdown, graph mutation, tombstone purge,
  maintenance.

Stable error envelope (no tracebacks, no internal paths, no SQL detail):

```json
{ "error": { "code": "candidate_already_reviewed",
             "message": "The candidate has already been decided.",
             "request_id": "..." } }
```

| HTTP | Meaning |
|---|---|
| 400 | malformed request |
| 401 | missing/invalid authentication |
| 403 | authenticated but insufficient permission |
| 404 | not found — including hidden records (no existence oracle) |
| 409 | idempotency/version/state conflict |
| 413 | request too large |
| 422 | valid JSON, invalid business input |
| 429 | rate limit |
| 503 | service not ready |
| 504 | bounded operation timeout |

Limits (table form in implementation, values decided at build): request body
bytes; query length; candidate content length; number of memory IDs; result
count; response bytes; JSON nesting/depth; concurrent requests per principal;
total concurrent requests; header size; read timeout / slowloris protection.
`ThreadingHTTPServer` and the internal `ThreadingTCPServer` create unbounded
handler threads — add a semaphore/bounded executor and return 429/503 under
pressure.

### D8 — Graph isolation (release gate)

A query about a shared director or supplier must not reveal, across client
scopes: another client's nodes, edges, memory IDs, counts that reveal
another client's existence, or timing/error differences distinguishing
"not found" from "denied".

- Current post-traversal filtering (`access_scoping.py:229–248`) is an
  interim guard only.
- v1 exposes **only ACL-filtered graph-enriched memory search** through the
  facade — never `query_graph`, `traverse_graph`, `list_nodes`, graph counts,
  or raw traversal.
- Direct graph tools require a dedicated graph scope and pre-hop/per-hop
  filtering; both are **outside v1**.

### D9 — Prompt injection & output safety (three directions)

1. **Incoming content → Argos memory:** scan evidence before any LLM call
   (`inbound_security.py` patterns); preserve external provenance; quarantine
   injection-pattern hits; **no weakening for "trusted" senders**; retain raw
   evidence only under an explicit retention policy; never replay raw
   untrusted evidence as instructions.
2. **Argos memory → caller:** mark memory text as *data*, not instructions;
   structured fields over generated prose where possible; include
   trust/provenance/freshness metadata; never return hidden ACL decisions,
   denied scope names, or raw internal paths; no full evidence by default;
   sanitize/suppress HTML and markup where clients may render it.
3. **External MCP tool output → Argos/Themis:** tool descriptions,
   annotations, and outputs are untrusted; external output is **never
   auto-promoted to memory**; if it enters Argos it becomes
   `provenance_origin="external"` through the candidate pipeline; record
   server identity, tool name, call ID, timestamp, content hash; external
   output must never invoke approval/admin operations.

### D10 — Audit & metrics

- Extend spec-06 `access_audit` for API use (fields below). No bearer tokens
  in logs, ever.
  `timestamp, tenant, authenticated_principal, client_id, transport,
  operation, request_id, idempotency_key_hash, memory_id/candidate_id,
  effective project/client scope, decision (allowed/denied/conflict/error),
  denied_reason_internal, query_hash (or redacted preview), result_count,
  response_bytes, latency_ms, policy_version`.
- **Query storage is deliberate:** default = normalized query hash + length +
  redacted preview; principal/audit role may store raw query under explicit
  retention policy. Query text may contain personal/client data.
- **Tamper evidence:** an append-only table in DuckDB is application-append-
  only, not tamper-proof against someone who can modify the database. The
  Argos API audit is **operational access telemetry**; the **Themis approval
  ledger remains the governance-grade ledger** (hash chain/signed export live
  there, not here). Stated explicitly so the commercial security claim is not
  oversold.
- Metrics (aggregates, never per-principal to unauthenticated callers):
  request count, denied count, 4xx/5xx, timeouts, idempotency replays, lock
  wait, queue depth, candidate creation/review counts, graph failures,
  embedding readiness, LLM/tool egress count, response bytes, p50/p95 latency.

### D11 — Health & readiness (separate concepts)

- **Liveness:** process alive.
- **Readiness:** service endpoint reachable; store opened; embedding model
  ready; schema version compatible; ACL/policy loaded; graph available or
  explicitly degraded.
- **Never report "healthy" if the first search would trigger a model load or
  network HEAD check.** Embedding readiness uses the **resolved local model
  path** (by-name loads are known to hang on network probes — existing Argos
  failure mode); embedding degradation is reported loudly, not silently.

### D12 — MCP client (deferred — Themis phase)

Rating: medium-high/high. Lives beside Themis, not in the initial slice.
Rules recorded now so the boundary is drawn before implementation:

- **Never** connect to arbitrary URLs supplied by a model; launch arbitrary
  commands/package managers; accept tool servers from prompt text; auto-fetch
  external `$ref` schemas; follow unvalidated redirects; pass through
  credentials/tokens; automatically trust tool descriptions; save external
  tool output as active memory.
- Server registry: `server_id, transport, exact executable/argv or exact URL,
  allowed hosts, allowed tools, credential reference, tool-schema
  hash/version, network policy, data classification allowed to leave Argos,
  timeout/concurrency limits, approval status`.
- stdio servers: invoke argv directly (never through a shell); no
  model-controlled executable paths; no `npx`/package auto-download; restrict
  environment variables; fixed configured data profile; log lifecycle events;
  bounded restart circuit breaker.
- Remote servers: HTTPS by default; validate DNS; block
  private/link-local/metadata addresses unless explicitly allowed; validate
  every redirect; DNS-rebinding defense; restricted OAuth metadata discovery;
  if OAuth later: validate audience/resource indicators, no token passthrough.
- **Egress policy is a separate class from the LLM egress inventory** —
  LLM destinations and arbitrary tool servers are different policy classes.

### D13 — Capacity & performance

No ANN/BM25 infrastructure is added just because REST exists. Add a capacity
test at current and projected corpus sizes measuring: direct store latency;
service round-trip latency; serialization overhead; graph enrichment overhead;
lock wait; p50/p95 under concurrent readers; behavior during one writer plus
many readers. (The single-owner service is correctness-friendly but still
serializes competing store operations per tenant — numbers are required before
any concurrency claim.)

## Explicitly OUT of scope (v1)

- REST writes without a named consumer (issues/park, not build).
- MCP client (D12) — deferred to Themis phase.
- `memory://` resources; raw graph tools; list/export endpoints.
- Public hosting, `0.0.0.0`, automatic tunnel binding, CORS by default.
- Admin/maintenance ops on the public boundary.
- New HTTP framework dependency (FastAPI/uvicorn) for the first local slice —
  stdlib `ThreadingHTTPServer` behind the facade is sufficient and keeps
  Argos dep-light.
- **NOT in Themis:** this is Argos-generic transport + policy boundary.
  Themis gates sit on top (governance-only layer, unchanged).

## Tests (cheapest falsifying first — deterministic, no LLM calls)

All runnable against a disposable HERMES_HOME with no personal data, no paid
LLM calls. These are the spec's **first acceptance gate** — the foundational
issue is not done until they pass:

1. **Public-method allowlist:** attempt `shutdown`, `backup`, `set_state`,
   `clear_scope`, raw graph mutation, arbitrary RPC forwarding through the
   adapter — all unavailable.
2. **Identity spoof:** authenticate as principal A, submit `user_id`,
   `tenant`, `project`, `client_scope` values for B — denied or narrowed to
   A's server-derived scope.
3. **Malformed ACL:** corrupt the ACL file — external API fails closed or
   refuses readiness; never an open store.
4. **Candidate idempotency:** same ingest twice with same idempotency key,
   including a simulated client timeout — exactly one candidate.
5. **Human-approval:** an MCP/model principal cannot approve its own
   candidate, even with `review_source="tool"`.
6. **Poisoning:** API/MCP candidate with instruction override + hidden
   Unicode — quarantined, no LLM call triggered, never active.
7. **Graph isolation:** shared-entity query from client A — client B's
   nodes/edges/IDs/counts/existence never exposed.
8. **Error-leak:** force a store failure — stable error code + request ID,
   never traceback/path/token/SQL detail.
9. **MCP stdio transcript:** launch, discover/list tools, search — every
   stdout line valid MCP JSON-RPC; any banner/log is a failure.
10. **Restart/timeout:** kill/restart around a mutation — retry with same
    idempotency key does not duplicate the write.

## Effort

| Direction | Complexity (revised) |
|---|---|
| MCP server, read-only stdio | Low-medium |
| MCP candidate ingest (class A) | Medium |
| REST health/read with auth, ACL, audit, limits | Medium |
| REST writes with idempotency/concurrency/audit | High |
| MCP client for external servers | Medium-high/high |

Build order: **0** cross-cutting facade + policy context + audit +
idempotency + schemas + conform. harness (#123) → **1** MCP stdio read-only
(#124) → **2** MCP candidate proposal path (#125) → **3** REST health/read
(#126) → **4** named-consumer REST writes → **5** MCP client beside Themis.

## Decisions (Michael, for sign-off)

1. **Facade-first build order** — foundational issue #123 lands before any
   transport (#124–#126), and its conformance tests are the gate.
2. **MCP protocol era:** Option 1 modern-only (2026-07-28 line) recommended;
   Option 2 dual-era only if a named legacy client exists.
3. **API-mode ACL fail-closed** by default (small `access_scoping.py` flag;
   personal mode unchanged).
4. **Graduated idempotency** — v1: ingest + decisions; full coverage with
   REST writes. Core table/logic built in #123 regardless.
5. **Provenance endpoint** (`/provenance`) requires a dedicated permission —
   not in the base read slice.
6. **Single policy boundary** — facade designed so the native plugin path can
   later route through it too (noted, not v1).
7. **Strict tenant routing mandatory** in API mode; unknown principals denied.
8. **No list/export endpoints; no `memory://` resources** in v1.
9. **Graph tools not exposed** until per-hop filtering exists; v1 facade-only
   graph-enriched search.
10. **Issue mapping:** #123 foundational · #124 MCP read · #125 MCP propose ·
    #126 REST read. REST writes + MCP client parked behind named consumers.
11. **Synchronous v1, by design.** The service is `ThreadingTCPServer` with
    blocking I/O and a single dispatch lock; that is accepted for v1 (bounded
    threads + semaphore → 429/503, per D7). No async runtime. An async/server
    model (e.g. FastAPI or anyio workers behind the facade) is a *later,
    measured* decision: only when the D13 capacity test shows the sync path is
    the bottleneck, or a named consumer requires it. The facade keeps
    transports swappable so this stays a cheap swap, not a redesign.
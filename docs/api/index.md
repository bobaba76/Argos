# API reference

Argos exposes a read tier (and a proposal tier) over MCP (stdio) and REST (HTTP). Both transports bind to loopback only and enforce a bearer token. The operation set is an explicit allowlist behind `ArgosAPIFacade` — auth context → ACL → validation → audit. No raw RPC passthrough; internal operations (shutdown, backup, set_state, purge, etc.) are never exposed.

## Facade operation tiers

The facade (`argos_plugin/api_facade.py`) defines three tiers:

| Tier | Operations | Description |
|------|-----------|-------------|
| **READ** | `search`, `fetch`, `fetch_history`, `capabilities`, `explain`, `explain_retrieval` | Available to all authenticated principals. |
| **PROPOSAL** | `memory_propose`, `ingest`, `erase_request` | External caller → candidate → security scan → review queue. Never creates active memory directly. `erase_request` is destructive but shares the proposal tier for idempotency mechanics. |
| **FEEDBACK** | `record_feedback` | Separately scoped. |

The union of all three tiers is `PUBLIC_OPERATIONS`. Operations not in this set (and explicitly in `FORBIDDEN_OPERATIONS`) return `method_not_allowed` — they are internal-only.

## Error envelope

Every error carries a stable code and a `request_id`. The facade never leaks tracebacks, internal paths, SQL detail, or tokens.

| Code | HTTP status | Meaning |
|------|-------------|---------|
| `malformed_request` | 400 | Malformed JSON or headers. |
| `unauthenticated` | 401 | Missing or invalid credentials. |
| `forbidden` | 403 | Principal not authorized for this operation. |
| `not_found` | 404 | Memory/candidate not found (or out of scope). |
| `conflict` | 409 | Idempotency key collision. |
| `request_too_large` | 413 | Body or field exceeds limit. |
| `invalid_input` | 422 | Validation failed. |
| `rate_limited` | 429 | Too many concurrent requests. |
| `not_ready` | 503 | Store not ready. |
| `timeout` | 504 | Request timed out. |
| `internal_error` | 500 | Unexpected failure. |
| `method_not_allowed` | 405 | Operation not on the public boundary. |

## Transports

- [MCP (stdio)](mcp.md) — JSON-RPC 2.0 over stdio. Tools: `memory_search`, `memory_fetch`, `memory_fetch_history`, `memory_explain`, `memory_why_not`, `memory_capabilities`, `memory_propose`.
- [REST (HTTP)](rest.md) — FastAPI on `127.0.0.1`. Endpoints for health, ready, capabilities, search, fetch, history, explain, explain-retrieval.

## Security model

- **Loopback only.** Both servers bind to `127.0.0.1`. Never `0.0.0.0`, never tunnel binding.
- **Bearer token.** Required on every request. Verified via `hmac.compare_digest`. The token is separate from the internal service token — loaded from `ARGOS_REST_TOKEN` or `api_credential.json` in the Hermes home.
- **Server-derived identity.** The facade does not accept client-supplied user identity. `AuthContext` is built from env vars / credential, not from the request body.
- **ACL enforcement.** `scope_check` enforces project, client-scope, and namespace restrictions per principal.
- **Audit.** Every facade operation is logged (no bearer tokens in logs; query text is hashed). Denied operations are routed to the durable `access_audit` table when the store exposes it.
- **Idempotency.** Proposal-tier operations accept an idempotency key. Same key + same body → returns original result. Same key + different body → 409 conflict.

## Limits

| Limit | Value |
|-------|-------|
| `MAX_QUERY_LENGTH` | 2000 chars |
| `MAX_MEMORY_ID_LENGTH` | 256 chars |
| `MAX_CONTENT_LENGTH` | 10000 chars |
| `MAX_MEMORY_IDS` | 50 |
| `MAX_LIMIT` | 50 |
| `MIN_LIMIT` | 1 |
| `MAX_TAGS` | 50 |
| `MAX_PAYLOAD_BYTES` | 4096 bytes |
| `MAX_INGEST_BYTES` | 256 KiB |

## Forbidden client flags

These client-supplied flags are rejected on the public API (they're internal-only):

- `include_quarantined`
- `include_archived`
- `include_expired`
- `include_closed`
- `suppress_retrieval`

# REST server (HTTP)

The REST server (`argos_plugin/rest_server.py`) exposes Argos over HTTP on `127.0.0.1`. FastAPI-based, token-authenticated, loopback-only.

## Start

```bash
ARGOS_REST_TOKEN=<token> python -m argos_plugin.rest_server --home <hermes-home> --port 8732
```

**CLI flags** (verified against `rest_server.py:main`):

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--home` | yes | — | Path to the Hermes home directory. |
| `--port` | no | `8732` | Port to bind. |
| `--max-concurrent` | no | `20` | Maximum concurrent requests. |

The token is loaded from `ARGOS_REST_TOKEN` (env var) or `rest_token` in the Hermes home config. The server refuses to start without a token (fail-closed).

## Auth

Every request must carry:

```
Authorization: Bearer <token>
```

Verified via `hmac.compare_digest`. Missing or invalid → `401 unauthenticated`.

## Endpoints

### `GET /v1/health`

Liveness check. No auth required.

**Response:** `{ "status": "ok" }`

### `GET /v1/ready`

Readiness check. Verifies the store is reachable.

**Response:** `{ "ready": true }` or `{ "ready": false }`

### `GET /v1/capabilities`

List the operations available to the authenticated principal.

**Auth:** required

**Response:**

```json
{
  "operations": ["search", "fetch", "fetch_history", "capabilities", "explain", "explain_retrieval"],
  "transport": "rest",
  "principal": "local"
}
```

### `POST /v1/memory/search`

Search memories by natural-language query.

**Auth:** required

**Body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `query` | string | yes | Natural-language query (max 2000 chars). |
| `limit` | integer | no | Max results (1–50, default 10). |
| `category_filter` | string | no | Filter to a category. |
| `project_id` | string | no | Narrow to a project. |
| `namespace` | string | no | Narrow to a namespace. |
| `client_scope` | string | no | Narrow to a client scope. |

**Response:**

```json
{
  "results": [
    {
      "memory_id": "mem-1",
      "category": "personal_fact",
      "content": "User lives in Cape Town",
      "tags": [],
      "similarity": 0.87,
      "created_at": "2026-01-01T00:00:00+00:00",
      "updated_at": "2026-01-01T00:00:00+00:00",
      "status": "active",
      "scope": "profile"
    }
  ],
  "count": 1
}
```

### `GET /v1/memories/{memory_id}`

Fetch a single memory by ID. ACL-enforced — out-of-scope memories return `404 not_found` (not `403`, to avoid leaking existence).

**Auth:** required

**Response:** the memory record (same fields as search results, plus `scope`).

### `GET /v1/memories/{memory_id}/history`

Fetch the version history for a memory. ACL-enforced.

**Auth:** required

**Response:**

```json
{
  "history": [
    {
      "memory_id": "mem-1",
      "content": "User lives in Cape Town",
      "created_at": "2026-01-01T00:00:00+00:00",
      "status": "active"
    }
  ]
}
```

### `GET /v1/memories/{memory_id}/explain`

Explain why a memory was retrieved — the provenance walk (#280). Returns evidence chain, version chain, conflict notes, blend score, confidence, and gates fired. Read-only, zero-LLM, fail-soft.

**Auth:** required

**Response:**

```json
{
  "memory_id": "mem-1",
  "content": "User lives in Cape Town",
  "category": "personal_fact",
  "evidence": { "source": "user_turn", "text": "..." },
  "version_chain": [{ "memory_id": "mem-1", "version": 1 }],
  "conflict_note": "",
  "blend_score": { "vector": 0.87, "text": 0.0, "graph": 0.0 },
  "confidence": 0.85,
  "provenance_origin": "user_turn",
  "grounding": "extracted",
  "gates_fired": []
}
```

### `POST /v1/memory/explain-retrieval`

Diagnose why a memory did NOT surface in retrieval (#320). Deterministic, read-only, zero-LLM.

**Auth:** required

**Body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `query` | string | yes | The query that should have surfaced the memory. |
| `memory_id` | string | yes | The memory ID that did not surface. |
| `top_k` | integer | no | Diagnostic window (1–50, default 20). |

**Response:**

```json
{
  "expected_memory_id": "mem-42",
  "expected": { "memory_id": "mem-42", "content": "..." },
  "found_in_results": false,
  "rank": null,
  "top_results": [{ "memory_id": "mem-1", "content": "...", "similarity": 0.91 }],
  "reasons": ["not_found"],
  "diagnostics": { "vector": { "similarity": 0.12 }, "status": "active" }
}
```

## Error responses

All errors use the stable envelope from the [API overview](index.md#error-envelope):

```json
{
  "error": {
    "code": "not_found",
    "message": "Memory not found.",
    "request_id": "550e8400-e29b-41d4-a716-446655440000"
  }
}
```

## Security

- **Loopback only.** Binds to `127.0.0.1`. Never `0.0.0.0`.
- **Bearer token.** Required on every endpoint except `/v1/health`.
- **Content-length check.** Oversized bodies rejected with `413`.
- **Concurrency limit.** `--max-concurrent` gates in-flight requests; excess returns `429`.
- **Cache-Control: no-store** on all responses.
- **No tokens in logs.** The audit log hashes query text and never logs bearer tokens.

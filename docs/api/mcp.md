# MCP server (stdio)

The MCP server (`argos_plugin/mcp_server.py`) exposes Argos over JSON-RPC 2.0 on stdio. Register it with any MCP-capable agent.

## Start

```bash
python -m argos_plugin.mcp_server --home <hermes-home>
```

**CLI flags** (verified against `mcp_server.py:main`):

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--home` | yes | — | Path to the Hermes home directory. |

The server logs to stderr only (no banners on stdout — stdout is the JSON-RPC transport).

## Tools

The MCP server exposes these tools (defined in `TOOL_DEFINITIONS`, sorted by name). Each tool maps to a facade operation via `TOOL_TO_OPERATION`.

### `memory_capabilities`

List the operations available to the authenticated principal.

- **Facade op:** `capabilities`
- **Input:** none
- **Output:** `{ operations: [string], transport: string, principal: string }`

### `memory_explain`

Explain why a memory was retrieved — provenance view. Returns evidence row, version chain, conflict note (if any), blend score, confidence, and gates fired. Read-only, zero-LLM, fail-soft. ACL-enforced.

- **Facade op:** `explain`
- **Input:**

    | Field | Type | Required | Description |
    |-------|------|----------|-------------|
    | `memory_id` | string | yes | The memory ID to explain. |

- **Output:** `{ memory_id, content, category, evidence, version_chain, conflict_note, blend_score, confidence, provenance_origin, grounding, gates_fired }`

### `memory_fetch`

Fetch a single memory by its ID.

- **Facade op:** `fetch`
- **Input:**

    | Field | Type | Required | Description |
    |-------|------|----------|-------------|
    | `memory_id` | string | yes | The memory ID to fetch. |

- **Output:** `{ memory_id, category, content, tags, created_at, updated_at, status, scope }`

### `memory_fetch_history`

Fetch the version history for a memory.

- **Facade op:** `fetch_history`
- **Input:**

    | Field | Type | Required | Description |
    |-------|------|----------|-------------|
    | `memory_id` | string | yes | The memory ID to fetch version history for. |

- **Output:** `{ history: [{ memory_id, content, created_at, status }], count }`

### `memory_propose`

Propose a new memory for human review. The candidate enters the review queue — it does NOT become active memory until a human approves it. An idempotency key is required.

- **Facade op:** `memory_propose`
- **Input:**

    | Field | Type | Required | Description |
    |-------|------|----------|-------------|
    | `content` | string | yes | The fact or observation to propose (max 10000 chars). |
    | `category` | string | no | Memory category (defaults to `context_note`). |
    | `tags` | [string] | no | Optional tags. |
    | `idempotency_key` | string | yes | Client-generated unique key (1–256 chars). Same key + same body → returns original result. |

- **Output:** `{ candidate_id, status, reason, scan_summary }`
- **Statuses:** `pending`, `quarantined`, `error`

### `memory_search`

Search memories by natural-language query.

- **Facade op:** `search`
- **Input:**

    | Field | Type | Required | Description |
    |-------|------|----------|-------------|
    | `query` | string | yes | Natural-language search query (max 2000 chars). |
    | `limit` | integer | no | Maximum results (1–50, default 10). |
    | `category_filter` | string | no | Filter to a specific memory category. |
    | `trust_class` | string | no | Filter by write-policy class: `unreviewed` or `clean` (#393 S2). |

- **Output:** `{ results: [{ memory_id, category, content, tags, similarity, created_at, updated_at, status, scope }], count }`

### `memory_unreviewed`

Report the live unreviewed trust-class backlog: how many memories currently carry the `unreviewed` marker (server-stamped for medium-risk saves under `approval_mode: auto`) and how old the oldest one is. Read-only, scoped to the authenticated principal.

- **Facade op:** `unreviewed`
- **Input:** none
- **Output:** `{ count, oldest_created_at, oldest_age_days, scope }`

### `memory_why_not`

Diagnose why a memory did NOT surface in retrieval — rank, per-stage diagnostics (vector/text/status/scope), and human-readable reasons. Deterministic, read-only, zero-LLM. ACL-enforced.

- **Facade op:** `explain_retrieval`
- **Input:**

    | Field | Type | Required | Description |
    |-------|------|----------|-------------|
    | `query` | string | yes | The search query that should have surfaced the memory (max 2000 chars). |
    | `memory_id` | string | yes | The memory ID that did not surface (max 256 chars). |
    | `top_k` | integer | no | Diagnostic window size (1–50, default 20). |

- **Output:** `{ expected_memory_id, expected, found_in_results, rank, top_results, reasons, diagnostics }`

## Strict input schemas

Every tool's `inputSchema` sets `additionalProperties: false`. Unknown fields are rejected — the caller cannot smuggle internal flags (e.g. `include_quarantined`, `suppress_retrieval`) through the MCP surface.

## Provenance fields are server-set

`memory_propose` does NOT accept `source`, `provenance_origin`, or `grounding` — these are server-set. The facade rejects them if the caller attempts to set them. This is the D4 (server-derived identity) invariant: the caller cannot forge provenance.

## Registration

Register with any MCP client. The server speaks JSON-RPC 2.0 over stdio and implements the standard MCP `initialize` → `tools/list` → `tools/call` handshake.

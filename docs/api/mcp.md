# MCP server (stdio)

The MCP server (`argos_plugin/mcp_server.py`) exposes Argos over JSON-RPC 2.0 on stdio. Register it with any MCP-capable agent.

## Start

```bash
python -m argos_plugin.mcp_server --home <hermes-home>
```

**From a source checkout?** The `-m` command needs the repo root and `argos_plugin/` on `PYTHONPATH` — or run script-mode: `cd argos_plugin && python mcp_server.py --home <hermes-home>`. See [Running from a source checkout](../integration.md#running-from-a-source-checkout).

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

### `memory_ingest`

Structured ingestion (#289): JSON/CSV rows become memories with first-class provenance (server-set `source=structured_ingest`, `grounding=extracted`). Preview (default) validates and reports — writes nothing. Apply materializes records via the candidate/approval machinery and requires the literal `confirm: true`; it is a **class-C (trusted-local) operation** — the facade denies apply on non-loopback transports (`403 forbidden`), so bridge/remote callers should use `memory_propose` for external writes. An idempotency key is required.

- **Facade op:** `ingest`
- **Input:**

    | Field | Type | Required | Description |
    |-------|------|----------|-------------|
    | `data` | string | yes | Raw JSON array or CSV text (UTF-8, max 262144 bytes). |
    | `fmt` | string | yes | `json` or `csv`. |
    | `source_name` | string | yes | Source file/feed name — provenance + ingest namespace (max 200 chars). |
    | `mapping` | object | yes | Field mapping: `category` + `content_template` (required); `tags`, `key_field`, per-field mappings (optional). |
    | `mode` | string | no | `preview` (default, writes nothing) or `apply`. |
    | `confirm` | boolean | no | Human-in-loop gate. Apply requires the literal `true`. |
    | `client_scope` / `doc_class` / `project_id` | string | no | Optional scoping metadata. |
    | `idempotency_key` | string | yes | Client-generated unique key (1–256 chars). |

- **Output:** `{ mode, source, mapping_id, total_rows, valid_rows, error_rows, inserted, superseded, duplicates, quarantined, blocked, rows, errors, wrote }`

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

### `memory_review`

Resolve a memory saved under the `unreviewed` trust class (Spec-13 S3). `promote` vouches it — the class marker is cleared and the bounded rank penalty is removed; `dismiss` quarantines it and fingerprints its claim slot in the rejection ledger (`reassertion_blocked` reports whether re-assertion is actually blocked — slot-less records cannot be fingerprinted by design, so only the quarantine applies).

- **Facade op:** `review_memory`
- **Class:** B — human principal only (model principals are denied; no self-vouch). Resolution is audit-logged.
- **Input:**

    | Field | Type | Required | Description |
    |-------|------|----------|-------------|
    | `memory_id` | string | yes | The memory to resolve. |
    | `decision` | string | yes | `promote` or `dismiss`. |
    | `reason` | string | no | Reason for the decision (max 2000 chars). |
    | `idempotency_key` | string | yes | Client-generated unique key. |

- **Output:** `{ memory_id, decision, changed, previous_trust_class, previous_status, reassertion_blocked, reviewer }`

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

`memory_propose` and `memory_ingest` do NOT accept `source`, `provenance_origin`, or `grounding` — these are server-set. The facade rejects them if the caller attempts to set them. This is the D4 (server-derived identity) invariant: the caller cannot forge provenance. (`memory_ingest` additionally server-stamps `source=structured_ingest` and an `ingest:<source_name>` namespace on every row.)

## Environment

| Variable | Default | Effect |
|----------|---------|--------|
| `ARGOS_API_USER_ID` | `default_user` | Store user scope the server reads/writes. Set it to your agent's user id (e.g. in the `mcpServers` env block) so external clients see the same memories as the native agent. |
| `ARGOS_API_READ_ONLY` | unset | `1` restores the read-only surface (search/fetch/explain only). |
| `ARGOS_API_PRINCIPAL_TYPE` | `model` | `human` enables class-B review ops (`memory_review`). |
| `ARGOS_API_NO_LOOPBACK` | unset | `1` denies class-C ops — including `memory_ingest` apply. |
| `ARGOS_API_CREDENTIAL` | unset | Name of a per-principal credential in `api_credential.json` (#387). When set, principal/tenant/user_id/principal_type/classes come from the credential file — the identity variables above are ignored — and the server refuses to start if the credential is missing, expired, or the file is invalid (fail-closed). |
| `ARGOS_API_CREDENTIAL_FILE` | `<home>/api_credential.json` | Override the credential file path. |

**Credentials (#387).** `api_credential.json` holds named entries
(`name`, `token_sha256`, `tenant`, `user_id`, `principal_type`,
`allowed_classes`, optional `expires_at`). Classes: `read`, `propose`,
`ingest`, `erase`, `review` (class B — also requires
`principal_type: human`), `write`, `feedback`, `collection_read`,
`collection_write`; class C classes additionally require the loopback
posture. On stdio the spawner *selects* one named credential — stdio has
no per-request auth surface — so use a credential whose classes and
user scope fit the client you are registering. Mint with
`python scripts/mint_api_credential.py`; revoke by deleting the entry
(MCP resolves at startup).

Tier model in full: [Integration](../integration.md).

## Retrieval pipeline

Retrieval runs entirely **service-side**: the server proxies every stage (embedding, reranker/blend, chains) to the shared memory service over RPC, so MCP search returns the same vector-backed pipeline as the native agent — verified byte-identical for the same store/user (#386; pinned by `tests/test_spec12_parity.py`). There is no client-side embedder and no opt-in needed.

## Registration

Register with any MCP client. The server speaks JSON-RPC 2.0 over stdio and implements the standard MCP `initialize` → `tools/list` → `tools/call` handshake.

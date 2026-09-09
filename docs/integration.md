# Integration guides

Argos is a Hermes plugin, but the external API (MCP + REST) lets any agent or script read and propose to the memory store. This page tracks integration patterns and adapters.

## Built-in transports

### MCP (stdio)

The MCP server is the primary integration path for MCP-capable agents (Claude Desktop, Cursor, custom MCP clients). It speaks JSON-RPC 2.0 over stdio and implements the standard MCP handshake (`initialize` → `tools/list` → `tools/call`).

See [MCP API reference](api/mcp.md) for the full tool list and schemas.

### REST (HTTP)

The REST server is the primary integration path for scripts, web apps, and non-MCP agents. It's loopback-only and token-authenticated.

See [REST API reference](api/rest.md) for the full endpoint list.

## Adapter patterns

### Python script

```python
import requests

BASE = "http://127.0.0.1:8732"
TOKEN = "<your-token>"
HEADERS = {"Authorization": f"Bearer {TOKEN}"}

# Search
r = requests.post(f"{BASE}/v1/memory/search",
                  headers=HEADERS,
                  json={"query": "where does the user live?", "limit": 5})
results = r.json()["results"]
for hit in results:
    print(hit["memory_id"], hit["content"], hit["similarity"])

# Fetch provenance for the first result (if any)
if results:
    first = results[0]
    r = requests.get(f"{BASE}/v1/memories/{first['memory_id']}/explain",
                     headers=HEADERS)
    print(r.json())
```

### curl

```bash
# Search
curl -s http://127.0.0.1:8732/v1/memory/search \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query": "where does the user live?", "limit": 5}'

# Explain (provenance walk)
curl -s http://127.0.0.1:8732/v1/memories/mem-1/explain \
  -H "Authorization: Bearer $TOKEN"
```

### MCP client registration

Register the Argos MCP server with your MCP client's config. The server
is a stdio JSON-RPC 2.0 process spawned on demand by the client.

A minimal read-only registration (search, fetch, explain, collections,
export — the default principal is `local`, read-only):

```json
{
  "mcpServers": {
    "argos": {
      "command": "python",
      "args": ["-m", "argos_plugin.mcp_server", "--home", "/path/to/hermes-home"],
      "cwd": "/path/to/Argos",
      "env": {
        "PYTHONPATH": "/path/to/Argos;/path/to/Argos/argos_plugin"
      }
    }
  }
}
```

Notes on the env block:

- **`PYTHONPATH`** must include the repo root and `argos_plugin/` so
  `python -m argos_plugin.mcp_server` can import the package. Without it,
  a fresh checkout that isn't pip-installed will fail to start.
- **`PYTHONIOENCODING=utf-8`** is recommended on Windows. The server
  reconfigures its own stdio to UTF-8 at startup, but setting it in the
  env block is belt-and-suspenders for older clients that may intercept
  the stream before the server reconfigures it.
- The `command` should point at a Python interpreter that has the
  project's dependencies installed (e.g. the Hermes agent venv). A bare
  `"python"` may resolve to a system interpreter without `duckdb`,
  `jsonschema`, etc.

### Write tier (enabled by default — spec-11)

As of spec-11 (9/9), the MCP and REST transports boot **write-enabled** by
default. Class A (propose → human review queue) and Class C (direct write,
loopback-only) are ON out of the box — no env vars required. Class B
(candidate approval) stays OFF (model self-approval, never).

A minimal MCP registration with write support:

```json
{
  "mcpServers": {
    "argos": {
      "command": "python",
      "args": ["-m", "argos_plugin.mcp_server", "--home", "/path/to/hermes/home"],
      "env": {
        "PYTHONPATH": "/path/to/Argos;/path/to/Argos/argos_plugin"
      }
    }
  }
}
```

That's it — `memory_propose`, `memory_save`, `memory_update`, and collection
writes are all available. No `ARGOS_API_CAN_PROPOSE` or `ARGOS_API_CAN_WRITE`
needed.

**Read-only escape hatch:** Set `ARGOS_API_READ_ONLY=1` to restore the
spec-09 read-only default (reads + collection reads only). Use this for
conservative or shared deployments where you don't want external clients
writing to the store.

```json
{
  "env": {
    "PYTHONPATH": "/path/to/Argos;/path/to/Argos/argos_plugin",
    "ARGOS_API_READ_ONLY": "1"
  }
}
```

**Class B (candidate approval):** `ARGOS_API_PRINCIPAL_TYPE` controls
`review_candidate`. The default is `model` (fail-closed): a model principal
cannot approve its own candidates. **Do not set this to `human` for
model-driven clients** — it unlocks self-approval, the exact hole spec-09
closes. Only set `human` for a single-user, local, human-driven UI where a
human is actually at the keyboard. Generic MCP clients should omit it
entirely; candidate review flows through a human via class A proposals.

**Non-loopback deployments:** `ARGOS_API_NO_LOOPBACK=1` disables class C
direct writes even with write tiers ON — class C requires loopback
regardless of the default flip. Class A (propose) still works.

## Adapters roadmap (#277)

Dedicated adapters for specific agent frameworks (LangChain, AutoGen, CrewAI, etc.) are tracked in [issue #277](https://github.com/bobaba76/Argos/issues/277). They are **not yet built**. The patterns above work today with any HTTP or MCP client.

When the adapters land, this page will host per-framework walkthroughs. Until then, the REST and MCP surfaces are stable and documented in the [API reference](api/index.md).

## Security notes for integrators

- **Loopback only.** Both servers bind to `127.0.0.1`. If you need remote access, use an SSH tunnel or a reverse proxy with auth — do not change the bind address.
- **Token auth.** Every request needs a bearer token. The token is separate from the internal service token. Store it in an env var (`ARGOS_REST_TOKEN`), not in your script's source.
- **Server-derived identity.** You cannot pass a user identity in the request body. The facade derives identity from the credential / env vars. This is by design — it prevents identity spoofing.
- **No raw store access.** The facade is the only path. Internal operations (shutdown, backup, purge) are not exposed.

## Next steps

- [API reference](api/index.md) — MCP and REST.
- [Quickstart](quickstart.md) — get started in five minutes.

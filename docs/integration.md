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

Register the Argos MCP server with your MCP client's config:

```json
{
  "mcpServers": {
    "argos": {
      "command": "python",
      "args": ["-m", "argos_plugin.mcp_server", "--home", "/path/to/hermes-home"]
    }
  }
}
```

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

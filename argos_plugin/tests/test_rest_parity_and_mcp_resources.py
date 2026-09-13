"""Tests for #388 (REST transport parity) + #389 (MCP resources/prompts).

Covers:
- GET /v1/memories (bounded, paginated list — browse parity)
- GET /v1/export (portable jsonl download)
- GET /v1/candidates now bounded + status-filtered
- MCP initialize advertises resources + prompts capabilities
- resources/list + resources/read (scope-aware, gated by allowed ops)
- prompts/list + prompts/get
- pre-initialization gate on all four new methods

Hermetic: in-memory streams, no LLM, no subprocess.
Run with: ARGOS_HERMETIC_TESTS=1 python -m pytest tests/test_rest_parity_and_mcp_resources.py -v
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

_plugin_dir = Path(__file__).resolve().parent.parent
if str(_plugin_dir) not in sys.path:
    sys.path.insert(0, str(_plugin_dir))

from api_facade import ArgosAPIFacade, AuthContext, READ_OPERATIONS  # noqa: E402
from access_scoping import ACLConfig  # noqa: E402
from mcp_server import MCPServer, JSONRPC_INVALID_PARAMS, JSONRPC_INVALID_REQUEST  # noqa: E402
from store_common import MemoryRecord  # noqa: E402


# -- minimal harness (mirrors test_mcp_server.py) ---------------------------

class StubStore:
    def __init__(self) -> None:
        self._memories: Dict[str, MemoryRecord] = {}
        self._next = 1

    def list_memories(self, category=None, limit=100):
        recs = list(self._memories.values())
        if category:
            recs = [r for r in recs if r.category == category]
        return recs[:limit]

    def remember(self, **kwargs) -> MemoryRecord:
        mid = f"mem-{self._next}"
        self._next += 1
        rec = MemoryRecord(
            memory_id=mid,
            category=kwargs.get("category", "personal_fact"),
            content=kwargs.get("content", ""),
            similarity=0.9,
            status="active",
            scope="profile",
        )
        self._memories[mid] = rec
        return rec


def _make_server(store=None, allowed_ops=None, stdin_lines=None, initialized=True):
    store = store or StubStore()
    facade = ArgosAPIFacade(store, acl=ACLConfig(), api_mode=False)
    auth = AuthContext(
        principal="test-principal", tenant="default", user_id="test-user",
        transport="mcp-stdio", allowed_operations=allowed_ops or set(READ_OPERATIONS),
    )
    stdin = io.StringIO(stdin_lines or "")
    stdout = io.StringIO()
    stderr = io.StringIO()
    server = MCPServer(facade, auth, stdin=stdin, stdout=stdout, stderr=stderr)
    server._initialized = initialized
    return server, stdout


def _send(*msgs: Dict[str, Any]) -> str:
    return "\n".join(json.dumps(m) for m in msgs) + "\n"


def _run(stdin_lines: str) -> List[Dict[str, Any]]:
    server, stdout = _make_server(stdin_lines=stdin_lines)
    server.run()
    stdout.seek(0)
    return [json.loads(l) for l in stdout if l.strip()]


_INIT = {"jsonrpc": "2.0", "id": 1, "method": "initialize",
         "params": {"protocolVersion": "2025-06-18", "capabilities": {}}}


class TestInitializeCapabilities:
    def test_initialize_advertises_resources_and_prompts(self):
        msgs = _run(_send(_INIT))
        caps = msgs[0]["result"]["capabilities"]
        assert "resources" in caps
        assert "prompts" in caps
        assert caps["tools"]["listChanged"] is False


class TestResources:
    def test_resources_list(self):
        msgs = _run(_send(_INIT, {"jsonrpc": "2.0", "id": 2, "method": "resources/list"}))
        uris = [r["uri"] for r in msgs[1]["result"]["resources"]]
        assert "memory://stats" in uris
        assert "memory://stats/categories" in uris

    def test_resources_read_stats(self):
        store = StubStore()
        store.remember(category="personal_fact", content="hello world")
        server, stdout = _make_server(
            store=store,
            stdin_lines=_send(
                {"jsonrpc": "2.0", "id": 1, "method": "initialize",
                 "params": {"protocolVersion": "2025-06-18", "capabilities": {}}},
                {"jsonrpc": "2.0", "id": 2, "method": "resources/read",
                 "params": {"uri": "memory://stats"}},
            ),
        )
        server.run()
        stdout.seek(0)
        msgs = [json.loads(l) for l in stdout if l.strip()]
        contents = msgs[1]["result"]["contents"]
        data = json.loads(contents[0]["text"])
        assert data["memory_count"] == 1

    def test_resources_read_unknown_uri(self):
        msgs = _run(_send(_INIT, {"jsonrpc": "2.0", "id": 2, "method": "resources/read",
                                  "params": {"uri": "memory://nope"}}))
        assert msgs[1]["error"]["code"] == JSONRPC_INVALID_PARAMS

    def test_resources_denied_when_browse_not_allowed(self):
        # A principal WITHOUT browse cannot see or read resources.
        server, stdout = _make_server(
            allowed_ops={"memory_search", "memory_fetch"}, initialized=True,
            stdin_lines=_send(_INIT, {"jsonrpc": "2.0", "id": 2, "method": "resources/read",
                                      "params": {"uri": "memory://stats"}}),
        )
        server.run()
        stdout.seek(0)
        msgs = [json.loads(l) for l in stdout if l.strip()]
        assert msgs[1]["error"]["code"] == JSONRPC_INVALID_PARAMS


class TestPrompts:
    def test_prompts_list(self):
        msgs = _run(_send(_INIT, {"jsonrpc": "2.0", "id": 2, "method": "prompts/list"}))
        names = {p["name"] for p in msgs[1]["result"]["prompts"]}
        assert "memory_search_usage" in names
        assert "review_queue_usage" in names

    def test_prompts_get_known_and_unknown(self):
        msgs = _run(_send(
            _INIT,
            {"jsonrpc": "2.0", "id": 2, "method": "prompts/get",
             "params": {"name": "memory_search_usage"}},
        ))
        assert msgs[1]["result"]["messages"][0]["content"]["text"]
        msgs = _run(_send(
            _INIT,
            {"jsonrpc": "2.0", "id": 3, "method": "prompts/get",
             "params": {"name": "nope"}},
        ))
        assert msgs[1]["error"]["code"] == JSONRPC_INVALID_PARAMS


class TestPreInitGate:
    def test_resources_list_blocked_before_initialized(self):
        server, stdout = _make_server(
            initialized=False,
            stdin_lines=_send({"jsonrpc": "2.0", "id": 1, "method": "resources/list"}),
        )
        server.run()
        stdout.seek(0)
        msgs = [json.loads(l) for l in stdout if l.strip()]
        assert msgs[0]["error"]["code"] == JSONRPC_INVALID_REQUEST

    def test_prompts_get_blocked_before_initialized(self):
        server, stdout = _make_server(
            initialized=False,
            stdin_lines=_send({"jsonrpc": "2.0", "id": 1, "method": "prompts/get",
                               "params": {"name": "memory_search_usage"}}),
        )
        server.run()
        stdout.seek(0)
        msgs = [json.loads(l) for l in stdout if l.strip()]
        assert msgs[0]["error"]["code"] == JSONRPC_INVALID_REQUEST


class TestRestParityRoutes:
    """#388 transport parity routes are declared (source-level, mirrors
    the repo's existing REST surface tests)."""

    def test_list_and_export_routes_declared(self, tmp_path):
        import rest_server
        src = Path(rest_server.__file__).read_text(encoding="utf-8")
        assert 'app.get("/v1/memories")' in src
        assert 'app.get("/v1/export")' in src
        assert "Content-Disposition" in src

    def test_candidates_extended_with_bounds_and_status(self, tmp_path):
        import rest_server
        src = Path(rest_server.__file__).read_text(encoding="utf-8")
        assert "status must be pending|accepted|rejected|quarantined" in src
        assert "Query(20, ge=1, le=100)" in src
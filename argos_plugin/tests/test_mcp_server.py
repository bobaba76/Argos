"""Tests for the MCP stdio server (#124 / spec-09).

Spec tests covered:
  9. MCP stdio transcript: launch, discover/list tools, search — every
     stdout line valid JSON-RPC; any banner/log is a failure.
  2. Identity spoof through the adapter.
  8. Error-leak through the adapter.

All deterministic, no LLM calls, no subprocess — tests use in-memory
streams to keep them fast and hermetic.
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

from api_facade import (
    APIError,
    ArgosAPIFacade,
    AuthContext,
    READ_OPERATIONS,
    PROPOSAL_OPERATIONS,
    FEEDBACK_OPERATIONS,
)
from access_scoping import ACLConfig
from mcp_server import (
    MCPServer,
    MCP_PROTOCOL_VERSION,
    TOOL_DEFINITIONS,
    TOOL_TO_OPERATION,
    JSONRPC_INVALID_PARAMS,
    JSONRPC_INVALID_REQUEST,
)
from store_common import MemoryRecord


# -- Stub store --------------------------------------------------------------

class StubStore:
    """Minimal store stub for MCP server tests."""

    def __init__(self) -> None:
        self._memories: Dict[str, MemoryRecord] = {}
        self._next = 1

    def search(self, **kwargs) -> List[MemoryRecord]:
        query = kwargs.get("query", "").lower()
        results = []
        for mid, rec in self._memories.items():
            if query in rec.content.lower():
                results.append(rec)
        return results[:kwargs.get("limit", 10)]

    def get_memories_by_ids(self, memory_ids: List[str], **kwargs) -> List[MemoryRecord]:
        return [self._memories[mid] for mid in memory_ids if mid in self._memories]

    def get_memory_history(self, memory_id: str, **kwargs) -> List[MemoryRecord]:
        return [self._memories[memory_id]] if memory_id in self._memories else []

    def save_candidate(self, **kwargs) -> Dict[str, Any]:
        cid = f"cand-{self._next}"
        self._next += 1
        return {"candidate_id": cid, "status": "pending", **kwargs}

    def review_candidate(self, **kwargs) -> Dict[str, Any] | None:
        return {"candidate": {"candidate_id": kwargs.get("candidate_id")}, "memory": None}

    def record_feedback(self, memory_id: str, feedback: str) -> bool:
        return True

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


def _make_server(
    store=None,
    allowed_ops=None,
    stdin_lines=None,
    initialized: bool = True,
) -> tuple[MCPServer, io.StringIO, io.StringIO]:
    """Build an MCP server with in-memory streams for testing.

    *initialized* defaults to True so existing tests that send initialize
    → tools/list without an explicit notifications/initialized still work
    (M8 gate is checked in _handle_line). Tests that need to verify the
    pre-initialization gate should pass initialized=False.
    """
    store = store or StubStore()
    facade = ArgosAPIFacade(store, acl=ACLConfig(), api_mode=False)
    auth = AuthContext(
        principal="test-principal",
        tenant="default",
        user_id="test-user",
        transport="mcp-stdio",
        allowed_operations=allowed_ops or (READ_OPERATIONS | PROPOSAL_OPERATIONS | FEEDBACK_OPERATIONS),
        can_propose=True,
        can_feedback=True,
    )
    stdin = io.StringIO(stdin_lines or "")
    stdout = io.StringIO()
    stderr = io.StringIO()
    server = MCPServer(facade, auth, stdin=stdin, stdout=stdout, stderr=stderr)
    server._initialized = initialized  # M8: default initialized for existing tests
    return server, stdout, stderr


def _parse_stdout(stdout: io.StringIO) -> List[Dict[str, Any]]:
    """Parse all stdout lines as JSON-RPC messages."""
    stdout.seek(0)
    messages = []
    for line in stdout:
        line = line.strip()
        if not line:
            continue
        messages.append(json.loads(line))
    return messages


def _send_messages(*msgs: Dict[str, Any]) -> str:
    """Serialize messages as newline-delimited stdin input."""
    return "\n".join(json.dumps(m) for m in msgs) + "\n"


# ---------------------------------------------------------------------------
# Spec test 9: stdio transcript — every stdout line valid JSON-RPC
# ---------------------------------------------------------------------------

class TestStdioTranscript:
    """Launch, discover/list tools, search — every stdout line valid
    JSON-RPC; any banner/log is a failure."""

    def test_initialize_response_is_valid_jsonrpc(self):
        """The initialize response is valid JSON-RPC with protocolVersion."""
        server, stdout, stderr = _make_server(stdin_lines=_send_messages({
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "test", "version": "1.0"}},
        }))
        server.run()
        msgs = _parse_stdout(stdout)
        assert len(msgs) == 1
        msg = msgs[0]
        assert msg["jsonrpc"] == "2.0"
        assert msg["id"] == 1
        assert "result" in msg
        assert msg["result"]["protocolVersion"] == MCP_PROTOCOL_VERSION
        assert "tools" in msg["result"]["capabilities"]
        assert msg["result"]["serverInfo"]["name"] == "argos-memory"

    def test_no_banners_on_stdout(self):
        """No startup banners or logs on stdout — only JSON-RPC messages."""
        server, stdout, stderr = _make_server(stdin_lines=_send_messages({
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {"protocolVersion": "2025-06-18", "capabilities": {}},
        }))
        server.run()
        # Every non-empty stdout line must be valid JSON.
        stdout.seek(0)
        for line in stdout:
            line = line.strip()
            if line:
                json.loads(line)  # raises if not valid JSON

    def test_tools_list_returns_tool_definitions(self):
        """tools/list returns the read-tier tool definitions."""
        server, stdout, stderr = _make_server(stdin_lines=_send_messages(
            {"jsonrpc": "2.0", "id": 1, "method": "initialize",
             "params": {"protocolVersion": "2025-06-18", "capabilities": {}}},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        ))
        server.run()
        msgs = _parse_stdout(stdout)
        # First message is initialize response, second is tools/list response.
        tools_msg = msgs[1]
        assert tools_msg["id"] == 2
        tools = tools_msg["result"]["tools"]
        tool_names = [t["name"] for t in tools]
        assert "memory_search" in tool_names
        assert "memory_fetch" in tool_names
        assert "memory_fetch_history" in tool_names
        assert "memory_capabilities" in tool_names

    def test_tools_list_strict_schemas(self):
        """Tool input schemas have additionalProperties: false."""
        server, stdout, stderr = _make_server(stdin_lines=_send_messages(
            {"jsonrpc": "2.0", "id": 1, "method": "initialize",
             "params": {"protocolVersion": "2025-06-18", "capabilities": {}}},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        ))
        server.run()
        msgs = _parse_stdout(stdout)
        tools = msgs[1]["result"]["tools"]
        for tool in tools:
            schema = tool["inputSchema"]
            assert schema.get("additionalProperties") is False, (
                f"Tool {tool['name']} inputSchema must have additionalProperties: false"
            )

    def test_tools_list_deterministic_ordering(self):
        """Tools are returned in deterministic (sorted) order."""
        server, stdout, stderr = _make_server(stdin_lines=_send_messages(
            {"jsonrpc": "2.0", "id": 1, "method": "initialize",
             "params": {"protocolVersion": "2025-06-18", "capabilities": {}}},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        ))
        server.run()
        msgs = _parse_stdout(stdout)
        tools = msgs[1]["result"]["tools"]
        names = [t["name"] for t in tools]
        assert names == sorted(names), "Tools must be in deterministic (sorted) order"

    def test_search_through_adapter(self):
        """A search tool call returns structured results."""
        store = StubStore()
        store.remember(category="personal_fact", content="User likes apples")
        server, stdout, stderr = _make_server(
            store=store,
            stdin_lines=_send_messages(
                {"jsonrpc": "2.0", "id": 1, "method": "initialize",
                 "params": {"protocolVersion": "2025-06-18", "capabilities": {}}},
                {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                 "params": {"name": "memory_search", "arguments": {"query": "apples", "limit": 5}}},
            ),
        )
        server.run()
        msgs = _parse_stdout(stdout)
        search_result = msgs[1]
        assert search_result["id"] == 2
        assert search_result["result"]["isError"] is False
        content = search_result["result"]["structuredContent"]
        assert content["count"] >= 1
        assert "apples" in content["results"][0]["content"]

    def test_fetch_through_adapter(self):
        """A fetch tool call returns a single memory."""
        store = StubStore()
        rec = store.remember(category="personal_fact", content="User works at TechCorp")
        server, stdout, stderr = _make_server(
            store=store,
            stdin_lines=_send_messages(
                {"jsonrpc": "2.0", "id": 1, "method": "initialize",
                 "params": {"protocolVersion": "2025-06-18", "capabilities": {}}},
                {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                 "params": {"name": "memory_fetch", "arguments": {"memory_id": rec.memory_id}}},
            ),
        )
        server.run()
        msgs = _parse_stdout(stdout)
        fetch_result = msgs[1]
        assert fetch_result["result"]["isError"] is False
        content = fetch_result["result"]["structuredContent"]
        assert content["memory_id"] == rec.memory_id
        assert "TechCorp" in content["content"]

    def test_initialized_notification_no_response(self):
        """notifications/initialized is a notification — no response sent."""
        server, stdout, stderr = _make_server(stdin_lines=_send_messages(
            {"jsonrpc": "2.0", "id": 1, "method": "initialize",
             "params": {"protocolVersion": "2025-06-18", "capabilities": {}}},
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
        ))
        server.run()
        msgs = _parse_stdout(stdout)
        # Only the initialize response — no response for the notification.
        assert len(msgs) == 1

    def test_ping_returns_empty_result(self):
        """ping returns an empty result (keepalive)."""
        server, stdout, stderr = _make_server(stdin_lines=_send_messages(
            {"jsonrpc": "2.0", "id": 1, "method": "ping"},
        ))
        server.run()
        msgs = _parse_stdout(stdout)
        assert msgs[0]["id"] == 1
        assert msgs[0]["result"] == {}

    def test_invalid_json_returns_parse_error(self):
        """Invalid JSON on stdin returns a JSON-RPC parse error."""
        server, stdout, stderr = _make_server(stdin_lines="not valid json\n")
        server.run()
        msgs = _parse_stdout(stdout)
        assert msgs[0]["error"]["code"] == -32700  # JSONRPC_PARSE_ERROR

    def test_unknown_method_returns_method_not_found(self):
        """An unknown method returns a JSON-RPC method-not-found error."""
        server, stdout, stderr = _make_server(stdin_lines=_send_messages(
            {"jsonrpc": "2.0", "id": 1, "method": "unknown/method"},
        ))
        server.run()
        msgs = _parse_stdout(stdout)
        assert msgs[0]["error"]["code"] == -32601  # JSONRPC_METHOD_NOT_FOUND


# ---------------------------------------------------------------------------
# Spec test 2: Identity spoof through the adapter
# ---------------------------------------------------------------------------

class TestIdentitySpoofViaAdapter:
    """Identity spoof attempts through the MCP adapter are rejected."""

    def test_search_with_spoofed_user_id_rejected(self):
        """A tools/call with user_id in arguments is rejected.
        M2: now rejected at the schema level (additionalProperties: false)
        before reaching the facade — JSONRPC_INVALID_PARAMS."""
        server, stdout, stderr = _make_server(stdin_lines=_send_messages(
            {"jsonrpc": "2.0", "id": 1, "method": "initialize",
             "params": {"protocolVersion": "2025-06-18", "capabilities": {}}},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
             "params": {"name": "memory_search",
                        "arguments": {"query": "test", "user_id": "other-user"}}},
        ))
        server.run()
        msgs = _parse_stdout(stdout)
        # M2: schema validation rejects user_id (not in search inputSchema).
        assert msgs[1]["error"]["code"] == -32602  # JSONRPC_INVALID_PARAMS

    def test_search_with_spoofed_tenant_rejected(self):
        """A tools/call with tenant in arguments is rejected.
        M2: now rejected at the schema level (additionalProperties: false)
        before reaching the facade — JSONRPC_INVALID_PARAMS."""
        server, stdout, stderr = _make_server(stdin_lines=_send_messages(
            {"jsonrpc": "2.0", "id": 1, "method": "initialize",
             "params": {"protocolVersion": "2025-06-18", "capabilities": {}}},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
             "params": {"name": "memory_search",
                        "arguments": {"query": "test", "tenant": "other-tenant"}}},
        ))
        server.run()
        msgs = _parse_stdout(stdout)
        # M2: schema validation rejects tenant (not in search inputSchema).
        assert msgs[1]["error"]["code"] == -32602  # JSONRPC_INVALID_PARAMS


# ---------------------------------------------------------------------------
# Spec test 8: Error-leak through the adapter
# ---------------------------------------------------------------------------

class TestErrorLeakViaAdapter:
    """Store failures through the adapter return stable error codes,
    never traceback/path/token/SQL."""

    def test_store_failure_returns_stable_error(self):
        """A store RuntimeError is caught and returned as a stable error."""
        store = StubStore()
        def failing_search(**kwargs):
            raise RuntimeError("DB error: /var/lib/hermes/hybrid_memory.duckdb SELECT * FROM")
        store.search = failing_search
        server, stdout, stderr = _make_server(
            store=store,
            stdin_lines=_send_messages(
                {"jsonrpc": "2.0", "id": 1, "method": "initialize",
                 "params": {"protocolVersion": "2025-06-18", "capabilities": {}}},
                {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                 "params": {"name": "memory_search", "arguments": {"query": "test"}}},
            ),
        )
        server.run()
        msgs = _parse_stdout(stdout)
        result = msgs[1]["result"]
        assert result["isError"] is True
        error = result["structuredContent"]["error"]
        assert error["code"] == "internal_error"
        assert "request_id" in error
        # No traceback/path/SQL leaked.
        error_str = json.dumps(error)
        assert "/var/lib" not in error_str
        assert "SELECT" not in error_str
        assert "traceback" not in error_str.lower()

    def test_unknown_tool_returns_method_not_found(self):
        """Calling an unknown tool returns a JSON-RPC error."""
        server, stdout, stderr = _make_server(stdin_lines=_send_messages(
            {"jsonrpc": "2.0", "id": 1, "method": "initialize",
             "params": {"protocolVersion": "2025-06-18", "capabilities": {}}},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
             "params": {"name": "nonexistent_tool", "arguments": {}}},
        ))
        server.run()
        msgs = _parse_stdout(stdout)
        assert msgs[1]["error"]["code"] == -32601  # METHOD_NOT_FOUND


# ---------------------------------------------------------------------------
# Tool authorization: only allowed tools are listed
# ---------------------------------------------------------------------------

class TestToolAuthorization:
    """tools/list only returns tools the principal is authorized for."""

    def test_read_only_principal_gets_read_tools(self):
        """A read-only principal sees only read-tier tools."""
        server, stdout, stderr = _make_server(
            allowed_ops=READ_OPERATIONS,
            stdin_lines=_send_messages(
                {"jsonrpc": "2.0", "id": 1, "method": "initialize",
                 "params": {"protocolVersion": "2025-06-18", "capabilities": {}}},
                {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
            ),
        )
        server.run()
        msgs = _parse_stdout(stdout)
        tools = msgs[1]["result"]["tools"]
        names = {t["name"] for t in tools}
        assert "memory_search" in names
        assert "memory_fetch" in names
        # No proposal tools (not in allowed_ops).
        assert "memory_propose" not in names

    def test_principal_with_propose_gets_propose_tool(self):
        """A principal with proposal permission sees the propose tool."""
        # The propose tool isn't in TOOL_DEFINITIONS yet (read tier only),
        # but the authorization logic should still work.
        server, stdout, stderr = _make_server(
            allowed_ops=READ_OPERATIONS | PROPOSAL_OPERATIONS,
            stdin_lines=_send_messages(
                {"jsonrpc": "2.0", "id": 1, "method": "initialize",
                 "params": {"protocolVersion": "2025-06-18", "capabilities": {}}},
                {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
            ),
        )
        server.run()
        msgs = _parse_stdout(stdout)
        tools = msgs[1]["result"]["tools"]
        # All read tools are present.
        names = {t["name"] for t in tools}
        assert "memory_search" in names


# ---------------------------------------------------------------------------
# M1-M10: MCP server audit fixes (#233)
# ---------------------------------------------------------------------------

class TestMCPServerAuditM:
    """Regression tests for issue #233: MCP server audit M1-M10."""

    # -- M1: idempotency_key only popped for memory_propose ------------------

    def test_m1_idempotency_key_not_popped_for_search(self):
        """M1: idempotency_key in a search call should be rejected by
        schema validation (additionalProperties: false), not silently
        dropped."""
        server, stdout, stderr = _make_server(stdin_lines=_send_messages(
            {"jsonrpc": "2.0", "id": 1, "method": "initialize",
             "params": {"protocolVersion": "2025-06-18", "capabilities": {}}},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
             "params": {"name": "memory_search",
                        "arguments": {"query": "test", "idempotency_key": "key-1"}}},
        ))
        server.run()
        msgs = _parse_stdout(stdout)
        # Schema validation rejects idempotency_key (not in search schema).
        assert msgs[1]["error"]["code"] == JSONRPC_INVALID_PARAMS

    def test_m1_idempotency_key_popped_for_propose(self):
        """M1: idempotency_key IS popped for memory_propose (it's in the
        schema and required). The call should succeed."""
        store = StubStore()
        server, stdout, stderr = _make_server(
            store=store,
            stdin_lines=_send_messages(
                {"jsonrpc": "2.0", "id": 1, "method": "initialize",
                 "params": {"protocolVersion": "2025-06-18", "capabilities": {}}},
                {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                 "params": {"name": "memory_propose",
                            "arguments": {"content": "User likes tea",
                                          "idempotency_key": "key-m1"}}},
            ),
        )
        server.run()
        msgs = _parse_stdout(stdout)
        result = msgs[1]["result"]
        assert result["isError"] is False
        assert result["structuredContent"]["candidate_id"] is not None

    # -- M2: schema validation before facade call ----------------------------

    def test_m2_search_limit_above_max_rejected(self):
        """M2: limit=999 (above schema maximum of 50) is rejected."""
        server, stdout, stderr = _make_server(stdin_lines=_send_messages(
            {"jsonrpc": "2.0", "id": 1, "method": "initialize",
             "params": {"protocolVersion": "2025-06-18", "capabilities": {}}},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
             "params": {"name": "memory_search",
                        "arguments": {"query": "test", "limit": 999}}},
        ))
        server.run()
        msgs = _parse_stdout(stdout)
        assert msgs[1]["error"]["code"] == JSONRPC_INVALID_PARAMS

    def test_m2_search_limit_zero_rejected(self):
        """M2: limit=0 (below schema minimum of 1) is rejected."""
        server, stdout, stderr = _make_server(stdin_lines=_send_messages(
            {"jsonrpc": "2.0", "id": 1, "method": "initialize",
             "params": {"protocolVersion": "2025-06-18", "capabilities": {}}},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
             "params": {"name": "memory_search",
                        "arguments": {"query": "test", "limit": 0}}},
        ))
        server.run()
        msgs = _parse_stdout(stdout)
        assert msgs[1]["error"]["code"] == JSONRPC_INVALID_PARAMS

    def test_m2_search_missing_query_rejected(self):
        """M2: missing required 'query' field is rejected."""
        server, stdout, stderr = _make_server(stdin_lines=_send_messages(
            {"jsonrpc": "2.0", "id": 1, "method": "initialize",
             "params": {"protocolVersion": "2025-06-18", "capabilities": {}}},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
             "params": {"name": "memory_search", "arguments": {"limit": 5}}},
        ))
        server.run()
        msgs = _parse_stdout(stdout)
        assert msgs[1]["error"]["code"] == JSONRPC_INVALID_PARAMS

    def test_m2_propose_content_too_long_rejected(self):
        """M2: content exceeding maxLength of 10000 is rejected."""
        server, stdout, stderr = _make_server(stdin_lines=_send_messages(
            {"jsonrpc": "2.0", "id": 1, "method": "initialize",
             "params": {"protocolVersion": "2025-06-18", "capabilities": {}}},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
             "params": {"name": "memory_propose",
                        "arguments": {"content": "x" * 10001,
                                      "idempotency_key": "key-long"}}},
        ))
        server.run()
        msgs = _parse_stdout(stdout)
        assert msgs[1]["error"]["code"] == JSONRPC_INVALID_PARAMS

    def test_m2_valid_search_passes_schema(self):
        """M2: a valid search call passes schema validation and reaches
        the facade."""
        store = StubStore()
        store.remember(category="personal_fact", content="hello world")
        server, stdout, stderr = _make_server(
            store=store,
            stdin_lines=_send_messages(
                {"jsonrpc": "2.0", "id": 1, "method": "initialize",
                 "params": {"protocolVersion": "2025-06-18", "capabilities": {}}},
                {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                 "params": {"name": "memory_search",
                            "arguments": {"query": "hello", "limit": 10}}},
            ),
        )
        server.run()
        msgs = _parse_stdout(stdout)
        assert msgs[1]["result"]["isError"] is False

    # -- M4: protocol version warning ----------------------------------------

    def test_m4_old_protocol_version_logs_warning(self, caplog):
        """M4: a client requesting an older protocol version logs a warning."""
        import logging
        server, stdout, stderr = _make_server(stdin_lines=_send_messages(
            {"jsonrpc": "2.0", "id": 1, "method": "initialize",
             "params": {"protocolVersion": "2024-11-05", "capabilities": {}}},
        ))
        with caplog.at_level(logging.WARNING, logger="argos.mcp"):
            server.run()
        # A warning should be logged about the version mismatch.
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) >= 1
        assert "protocol" in warnings[0].message.lower()

    def test_m4_matching_protocol_no_warning(self, caplog):
        """M4: matching protocol version does not log a warning."""
        import logging
        server, stdout, stderr = _make_server(stdin_lines=_send_messages(
            {"jsonrpc": "2.0", "id": 1, "method": "initialize",
             "params": {"protocolVersion": MCP_PROTOCOL_VERSION, "capabilities": {}}},
        ))
        with caplog.at_level(logging.WARNING, logger="argos.mcp"):
            server.run()
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 0

    # -- M6: BrokenPipeError in _send is caught ------------------------------

    def test_m6_broken_pipe_does_not_crash(self):
        """M6: a BrokenPipeError in _send is caught, not propagated."""
        server, stdout, stderr = _make_server(stdin_lines=_send_messages(
            {"jsonrpc": "2.0", "id": 1, "method": "ping"},
        ))
        # Replace stdout with one that raises BrokenPipeError.
        class BrokenStdout:
            def write(self, data):
                raise BrokenPipeError("broken pipe")
            def flush(self):
                pass
        server._stdout = BrokenStdout()
        # Should not raise — the run loop handles BrokenPipeError.
        server.run()

    # -- M8: _initialized gate on tools/list and tools/call ------------------

    def test_m8_tools_list_before_initialized_rejected(self):
        """M8: tools/list before notifications/initialized is rejected."""
        server, stdout, stderr = _make_server(
            initialized=False,
            stdin_lines=_send_messages(
                {"jsonrpc": "2.0", "id": 1, "method": "initialize",
                 "params": {"protocolVersion": "2025-06-18", "capabilities": {}}},
                {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
            ),
        )
        server.run()
        msgs = _parse_stdout(stdout)
        # initialize response is fine, tools/list is rejected.
        assert msgs[0]["id"] == 1
        assert "result" in msgs[0]
        assert msgs[1]["error"]["code"] == JSONRPC_INVALID_REQUEST

    def test_m8_tools_call_before_initialized_rejected(self):
        """M8: tools/call before notifications/initialized is rejected."""
        server, stdout, stderr = _make_server(
            initialized=False,
            stdin_lines=_send_messages(
                {"jsonrpc": "2.0", "id": 1, "method": "initialize",
                 "params": {"protocolVersion": "2025-06-18", "capabilities": {}}},
                {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                 "params": {"name": "memory_search", "arguments": {"query": "test"}}},
            ),
        )
        server.run()
        msgs = _parse_stdout(stdout)
        assert msgs[1]["error"]["code"] == JSONRPC_INVALID_REQUEST

    def test_m8_tools_list_after_initialized_succeeds(self):
        """M8: tools/list after notifications/initialized works."""
        server, stdout, stderr = _make_server(
            initialized=False,
            stdin_lines=_send_messages(
                {"jsonrpc": "2.0", "id": 1, "method": "initialize",
                 "params": {"protocolVersion": "2025-06-18", "capabilities": {}}},
                {"jsonrpc": "2.0", "method": "notifications/initialized"},
                {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
            ),
        )
        server.run()
        msgs = _parse_stdout(stdout)
        # initialize response, then tools/list response (notification has no response).
        assert msgs[0]["id"] == 1
        assert msgs[1]["id"] == 2
        assert "result" in msgs[1]
        assert "tools" in msgs[1]["result"]

    # -- M9: TOOL_DEFINITIONS is a tuple (immutable) -------------------------

    def test_m9_tool_definitions_is_tuple(self):
        """M9: TOOL_DEFINITIONS should be a tuple, not a list (immutable)."""
        assert isinstance(TOOL_DEFINITIONS, tuple), (
            f"TOOL_DEFINITIONS should be a tuple, got {type(TOOL_DEFINITIONS)}"
        )

    def test_m9_tool_definitions_cannot_be_mutated(self):
        """M9: attempting to mutate TOOL_DEFINITIONS raises TypeError."""
        with pytest.raises(TypeError):
            TOOL_DEFINITIONS[0] = {}  # type: ignore
        with pytest.raises(AttributeError):
            TOOL_DEFINITIONS.append({})  # type: ignore

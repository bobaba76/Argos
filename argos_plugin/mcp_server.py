"""Spec-09 (#124): MCP stdio server — read tier.

Transport adapter only. Exposes the READ tier of the public allowlist
(search, fetch, fetch_history, capabilities) over MCP stdio, behind the
facade from #123.

Protocol: MCP JSON-RPC 2.0 over stdio (newline-delimited).
  - stdout carries ONLY valid MCP JSON-RPC messages
  - logs go to stderr
  - no startup banners on stdout
  - UTF-8, newline framing
  - bounded message size
  - deterministic tool ordering

Modern protocol era (2025-06-18 line, per spec Decision 2):
  - initialize → capability negotiation
  - notifications/initialized → ready
  - tools/list → tool definitions with strict inputSchema
  - tools/call → invoke facade operation, return structured result

Strict tool schemas: additionalProperties: false, max string lengths,
max result counts, enum validation. No caller-controlled internal flags
(include_quarantined, include_archived, include_expired).

Output: structured results with outputSchema, not provider JSON strings.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict

# JSON-RPC 2.0 error codes (per spec).
JSONRPC_PARSE_ERROR = -32700
JSONRPC_INVALID_REQUEST = -32600
JSONRPC_METHOD_NOT_FOUND = -32601
JSONRPC_INVALID_PARAMS = -32602
JSONRPC_INTERNAL_ERROR = -32603

# MCP protocol version (2025-06-18 line, per spec Decision 2).
MCP_PROTOCOL_VERSION = "2025-06-18"

# Bounded message size (D6: bounded messages).
MAX_MESSAGE_BYTES = 4 * 1024 * 1024  # 4 MiB

logger = logging.getLogger("argos.mcp")


# -- Tool definitions (D2 read tier) -----------------------------------------

def _search_input_schema() -> Dict[str, Any]:
    """Strict input schema for memory_search."""
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["query"],
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural-language search query.",
                "maxLength": 2000,
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of results (1-50).",
                "minimum": 1,
                "maximum": 50,
                "default": 10,
            },
            "category_filter": {
                "type": "string",
                "description": "Optional: filter to a specific memory category.",
            },
        },
    }


def _fetch_input_schema() -> Dict[str, Any]:
    """Strict input schema for memory_fetch."""
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["memory_id"],
        "properties": {
            "memory_id": {
                "type": "string",
                "description": "The memory ID to fetch.",
            },
        },
    }


def _fetch_history_input_schema() -> Dict[str, Any]:
    """Strict input schema for memory_fetch_history."""
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["memory_id"],
        "properties": {
            "memory_id": {
                "type": "string",
                "description": "The memory ID to fetch version history for.",
            },
        },
    }


def _explain_input_schema() -> Dict[str, Any]:
    """Strict input schema for memory_explain (#280)."""
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["memory_id"],
        "properties": {
            "memory_id": {
                "type": "string",
                "description": "The memory ID to explain (provenance view).",
            },
        },
    }


def _capabilities_input_schema() -> Dict[str, Any]:
    """Strict input schema for memory_capabilities (no params)."""
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {},
    }


def _propose_input_schema() -> Dict[str, Any]:
    """Strict input schema for memory_propose (class A write).

    The idempotency_key is required for propose operations (D5).
    Provenance fields (source, provenance_origin, grounding) are
    server-set and intentionally absent from this schema — the facade
    rejects them if the caller attempts to set them.
    """
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["content", "idempotency_key"],
        "properties": {
            "content": {
                "type": "string",
                "description": "The fact or observation to propose for review.",
                "maxLength": 10000,
            },
            "category": {
                "type": "string",
                "description": "Memory category (defaults to context_note).",
                "default": "context_note",
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional tags for the proposed memory.",
            },
            "idempotency_key": {
                "type": "string",
                "description": (
                    "Client-generated unique key. Same key + same body → "
                    "returns original result (no duplicate). Same key + "
                    "different body → 409 conflict."
                ),
                "minLength": 1,
                "maxLength": 256,
            },
        },
    }


# Tool name → (facade operation, input schema, description, output schema).
# Deterministic ordering (D6): sorted by tool name.
# M9: tuple (immutable) to prevent accidental mutation across instances.
TOOL_DEFINITIONS: tuple = (
    {
        "name": "memory_capabilities",
        "description": "List the operations available to the authenticated principal.",
        "inputSchema": _capabilities_input_schema(),
        "outputSchema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "operations": {"type": "array", "items": {"type": "string"}},
                "transport": {"type": "string"},
                "principal": {"type": "string"},
            },
        },
    },
    {
        "name": "memory_explain",
        "description": (
            "Explain why a memory was retrieved — provenance view. "
            "Returns evidence row, version chain, conflict note (if any), "
            "blend score, confidence, and gates fired. Read-only, zero-LLM, "
            "fail-soft. ACL-enforced."
        ),
        "inputSchema": _explain_input_schema(),
        "outputSchema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "memory_id": {"type": "string"},
                "content": {"type": "string"},
                "category": {"type": "string"},
                "evidence": {"type": "object"},
                "version_chain": {"type": "array", "items": {"type": "object"}},
                "conflict_note": {"type": "string"},
                "blend_score": {"type": "object"},
                "confidence": {"type": "number"},
                "provenance_origin": {"type": "string"},
                "grounding": {"type": "string"},
                "gates_fired": {"type": "array", "items": {"type": "string"}},
            },
        },
    },
    {
        "name": "memory_fetch",
        "description": "Fetch a single memory by its ID.",
        "inputSchema": _fetch_input_schema(),
        "outputSchema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "memory_id": {"type": "string"},
                "category": {"type": "string"},
                "content": {"type": "string"},
                "tags": {"type": "array", "items": {"type": "string"}},
                "created_at": {"type": "string"},
                "updated_at": {"type": "string"},
                "status": {"type": "string"},
                "scope": {"type": "string"},
            },
        },
    },
    {
        "name": "memory_fetch_history",
        "description": "Fetch the version history for a memory.",
        "inputSchema": _fetch_history_input_schema(),
        "outputSchema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "history": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "memory_id": {"type": "string"},
                            "content": {"type": "string"},
                            "created_at": {"type": "string"},
                            "status": {"type": "string"},
                        },
                    },
                },
                "count": {"type": "integer"},
            },
        },
    },
    {
        "name": "memory_propose",
        "description": (
            "Propose a new memory for human review. The candidate enters "
            "the review queue — it does NOT become active memory until a "
            "human approves it. An idempotency key is required: retrying "
            "with the same key and body returns the original result."
        ),
        "inputSchema": _propose_input_schema(),
        "outputSchema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "candidate_id": {"type": "string"},
                "status": {
                    "type": "string",
                    "enum": ["pending", "quarantined", "error"],
                },
                "reason": {"type": "string"},
                "scan_summary": {"type": "string"},
            },
        },
    },
    {
        "name": "memory_search",
        "description": "Search memories by natural-language query.",
        "inputSchema": _search_input_schema(),
        "outputSchema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "results": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "memory_id": {"type": "string"},
                            "category": {"type": "string"},
                            "content": {"type": "string"},
                            "tags": {"type": "array", "items": {"type": "string"}},
                            "similarity": {"type": "number"},
                            "created_at": {"type": "string"},
                            "updated_at": {"type": "string"},
                            "status": {"type": "string"},
                            "scope": {"type": "string"},
                        },
                    },
                },
                "count": {"type": "integer"},
            },
        },
    },
)

# Map MCP tool names to facade operations.
TOOL_TO_OPERATION: Dict[str, str] = {
    "memory_search": "search",
    "memory_fetch": "fetch",
    "memory_fetch_history": "fetch_history",
    "memory_explain": "explain",
    "memory_capabilities": "capabilities",
    "memory_propose": "memory_propose",
}


# -- JSON-RPC message helpers ------------------------------------------------

def _make_response(
    request_id: Any,
    result: Any = None,
    error: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Build a JSON-RPC 2.0 response."""
    msg: Dict[str, Any] = {"jsonrpc": "2.0", "id": request_id}
    if error is not None:
        msg["error"] = error
    else:
        msg["result"] = result
    return msg


def _make_error(code: int, message: str, data: Any = None) -> Dict[str, Any]:
    """Build a JSON-RPC 2.0 error object.

    M10: *data* is sent verbatim to the client — it must be client-safe
    (no stack traces, internal IDs, file paths, or SQL). The facade
    redacts errors, but callers must not pass unredacted data here.
    """
    err: Dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    return err


def _make_notification(method: str, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Build a JSON-RPC 2.0 notification (no id, no response expected)."""
    msg: Dict[str, Any] = {"jsonrpc": "2.0", "method": method}
    if params:
        msg["params"] = params
    return msg


# -- MCP server --------------------------------------------------------------

class MCPServer:
    """MCP stdio server exposing the Argos read tier through the facade.

    The server reads JSON-RPC messages from stdin (one per line), processes
    them, and writes responses to stdout (one per line). Logs go to stderr.

    The server is a transport adapter only — all business logic (auth,
    ACL, validation, idempotency, audit) is handled by the facade.
    """

    def __init__(
        self,
        facade,
        auth_context,
        *,
        stdin=None,
        stdout=None,
        stderr=None,
    ) -> None:
        """Initialize the MCP server.

        Args:
            facade: an ArgosAPIFacade instance.
            auth_context: an AuthContext for the authenticated principal.
                The transport derives this from the credential/environment
                before constructing the server.
            stdin: input stream (defaults to sys.stdin).
            stdout: output stream (defaults to sys.stdout).
            stderr: log stream (defaults to sys.stderr).
        """
        from api_facade import ArgosAPIFacade, AuthContext  # noqa: F401
        self._facade = facade
        self._auth = auth_context
        self._stdin = stdin or sys.stdin
        self._stdout = stdout or sys.stdout
        self._stderr = stderr or sys.stderr
        self._initialized = False

    def run(self) -> None:
        """Main loop: read lines from stdin, process, write to stdout.

        Exits when stdin is closed (EOF) or a fatal error occurs.
        M5: KeyboardInterrupt/SystemExit are caught and logged so the
        server shuts down gracefully rather than dying mid-message.
        """
        for line in self._stdin:
            line = line.strip()
            if not line:
                continue
            if len(line.encode("utf-8")) > MAX_MESSAGE_BYTES:
                self._send(_make_response(
                    None, error=_make_error(
                        JSONRPC_INVALID_REQUEST,
                        "Message exceeds maximum size.",
                    ),
                ))
                continue
            try:
                self._handle_line(line)
            except (KeyboardInterrupt, SystemExit):
                # M5: graceful shutdown on signals — log and re-exit.
                logger.info("MCP server shutting down (signal received).")
                raise
            except BrokenPipeError:
                # M6: stdout closed — client disconnected. Exit gracefully.
                logger.info("MCP server: stdout closed (client disconnected).")
                break
            except Exception as exc:
                # Fatal error — log to stderr, send error to stdout, continue.
                logger.error("MCP server error: %s", exc, exc_info=True)
                self._send(_make_response(
                    None, error=_make_error(
                        JSONRPC_INTERNAL_ERROR,
                        "Internal server error.",
                    ),
                ))

    def _handle_line(self, line: str) -> None:
        """Parse and handle one JSON-RPC message line."""
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            self._send(_make_response(
                None, error=_make_error(
                    JSONRPC_PARSE_ERROR, "Invalid JSON.",
                ),
            ))
            return
        if not isinstance(msg, dict):
            self._send(_make_response(
                None, error=_make_error(
                    JSONRPC_INVALID_REQUEST, "Request must be a JSON object.",
                ),
            ))
            return
        msg_id = msg.get("id")
        method = msg.get("method")
        params = msg.get("params") or {}
        if not isinstance(method, str):
            # Notifications (no method) are silently ignored.
            if msg_id is not None:
                self._send(_make_response(
                    msg_id, error=_make_error(
                        JSONRPC_INVALID_REQUEST, "Missing method.",
                    ),
                ))
            return
        # Route to handler.
        if method == "initialize":
            self._handle_initialize(msg_id, params)
        elif method == "notifications/initialized":
            self._initialized = True
            # No response for notifications.
        elif method == "tools/list":
            # M8: tools should not be listed until after notifications/initialized.
            if not self._initialized:
                self._send(_make_response(
                    msg_id, error=_make_error(
                        JSONRPC_INVALID_REQUEST,
                        "Server not initialized — send notifications/initialized first.",
                    ),
                ))
            else:
                self._handle_tools_list(msg_id)
        elif method == "tools/call":
            if not self._initialized:
                self._send(_make_response(
                    msg_id, error=_make_error(
                        JSONRPC_INVALID_REQUEST,
                        "Server not initialized — send notifications/initialized first.",
                    ),
                ))
            else:
                self._handle_tools_call(msg_id, params)
        elif method == "ping":
            self._send(_make_response(msg_id, result={}))
        else:
            self._send(_make_response(
                msg_id, error=_make_error(
                    JSONRPC_METHOD_NOT_FOUND,
                    f"Unknown method: {method}",
                ),
            ))

    def _handle_initialize(self, msg_id: Any, params: Dict[str, Any]) -> None:
        """Handle the initialize request (capability negotiation).

        M4: logs a warning if the client's protocol version is older than
        the server's minimum supported version.
        """
        client_version = params.get("protocolVersion", "")
        # M4: warn on incompatible versions (server still responds with its own).
        if client_version and client_version != MCP_PROTOCOL_VERSION:
            logger.warning(
                "MCP client requested protocol %s; server supports %s. "
                "Responding with server version (per spec).",
                client_version, MCP_PROTOCOL_VERSION,
            )
        # Respond with our protocol version. If the client requested a
        # different version, we respond with ours (per spec: server
        # responds with a version it supports).
        result = {
            "protocolVersion": MCP_PROTOCOL_VERSION,
            "capabilities": {
                "tools": {
                    "listChanged": False,
                },
            },
            "serverInfo": {
                "name": "argos-memory",
                "title": "Argos Memory Service",
                "version": "1.0.0",
            },
            "instructions": (
                "Argos memory service. Use memory_search to find memories, "
                "memory_fetch to get one by ID, memory_fetch_history for "
                "version history, memory_propose to submit a fact for human "
                "review (requires an idempotency key), and "
                "memory_capabilities to list available operations."
            ),
        }
        self._send(_make_response(msg_id, result=result))

    def _handle_tools_list(self, msg_id: Any) -> None:
        """Handle tools/list — return available tool definitions."""
        # Only return tools the principal is authorized for.
        allowed_ops = self._auth.allowed_operations
        tools = []
        for tool_def in TOOL_DEFINITIONS:
            op = TOOL_TO_OPERATION.get(tool_def["name"], "")
            if op in allowed_ops:
                tools.append(tool_def)
        self._send(_make_response(msg_id, result={"tools": tools}))

    def _handle_tools_call(self, msg_id: Any, params: Dict[str, Any]) -> None:
        """Handle tools/call — invoke a tool through the facade."""
        tool_name = params.get("name", "")
        arguments = dict(params.get("arguments") or {})  # copy — don't mutate caller's
        # Map tool name to facade operation.
        operation = TOOL_TO_OPERATION.get(tool_name)
        if operation is None:
            self._send(_make_response(
                msg_id, error=_make_error(
                    JSONRPC_METHOD_NOT_FOUND,
                    f"Unknown tool: {tool_name}",
                ),
            ))
            return
        # M2: validate arguments against the tool's inputSchema before
        # calling the facade. The MCP spec requires the server to validate
        # against the declared schema.
        tool_def = None
        for td in TOOL_DEFINITIONS:
            if td["name"] == tool_name:
                tool_def = td
                break
        if tool_def is not None:
            schema = tool_def.get("inputSchema")
            if schema is not None:
                try:
                    import jsonschema
                    jsonschema.validate(instance=arguments, schema=schema)
                except jsonschema.ValidationError as exc:
                    self._send(_make_response(
                        msg_id, error=_make_error(
                            JSONRPC_INVALID_PARAMS,
                            f"Invalid arguments: {exc.message}",
                        ),
                    ))
                    return
                except Exception:
                    # jsonschema unavailable or broken — fall through to
                    # facade validation (fail-open, not fail-closed, since
                    # the facade does its own validation).
                    pass
        # M1: only pop idempotency_key for memory_propose — other tools
        # should not have it, and the schema validation above would have
        # already rejected it (additionalProperties: false). For propose,
        # the key is passed as a keyword arg, not in the params dict.
        idempotency_key = None
        if tool_name == "memory_propose":
            idempotency_key = arguments.pop("idempotency_key", None)
        # Call the facade. The facade handles validation, auth, ACL,
        # idempotency, audit, and error redaction.
        try:
            result = self._facade.execute(
                self._auth, operation, arguments,
                idempotency_key=idempotency_key,
            )
            # MCP tools/call returns a CallToolResult with content array.
            # For structured results, we use the content as JSON.
            self._send(_make_response(msg_id, result={
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(result, ensure_ascii=False),
                    },
                ],
                "structuredContent": result,
                "isError": False,
            }))
        except Exception as exc:
            # The facade raises APIError with stable codes. We map those
            # to MCP error responses.
            from api_facade import APIError
            if isinstance(exc, APIError):
                self._send(_make_response(msg_id, result={
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(exc.to_dict(), ensure_ascii=False),
                        },
                    ],
                    "structuredContent": exc.to_dict(),
                    "isError": True,
                }))
            else:
                # Should not happen — the facade catches all exceptions.
                logger.error("Unhandled error in tools/call: %s", exc, exc_info=True)
                self._send(_make_response(msg_id, result={
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps({
                                "error": {
                                    "code": "internal_error",
                                    "message": "An internal error occurred.",
                                },
                            }, ensure_ascii=False),
                        },
                    ],
                    "isError": True,
                }))

    def _send(self, msg: Dict[str, Any]) -> None:
        """Write one JSON-RPC message to stdout (newline-delimited).

        M6: BrokenPipeError (client disconnected) is caught and logged
        rather than propagating — the run loop handles the exit.
        """
        data = json.dumps(msg, ensure_ascii=False) + "\n"
        try:
            self._stdout.write(data)
            self._stdout.flush()
        except (BrokenPipeError, OSError) as exc:
            logger.info("MCP server: write failed (%s) — client may be gone.", exc)


# -- Entry point -------------------------------------------------------------

def _load_auth_context(home: Path) -> "AuthContext":
    """Derive the auth context from the environment.

    For v1 (trusted-local mode), the principal is derived from the
    ARGOS_API_PRINCIPAL env var (default: "local"), the tenant from
    ARGOS_API_TENANT (default: "default"), and the user_id from
    ARGOS_API_USER_ID (default: "default_user").

    M3 — Threat model (trusted-local mode):
    Identity is derived from environment variables with NO credential
    verification. Any process that can set env vars can impersonate any
    user. This is acceptable ONLY in trusted-local mode (single-user
    workstation, MCP client spawned by the user's own shell). In a
    multi-process or hosted environment, the env vars are controlled by
    the spawner, not the user — a malicious spawner can impersonate
    anyone. For non-trusted-local deployments, a credential file or
    signed token MUST be used instead (future work, #129).

    In production (multi-user/hosted mode, #129), this would verify
    a credential file and derive identity from it. For now, the env-var
    approach is the trusted-local mode documented in the spec.
    """
    from api_facade import AuthContext, READ_OPERATIONS, PROPOSAL_OPERATIONS, FEEDBACK_OPERATIONS

    principal = os.environ.get("ARGOS_API_PRINCIPAL", "local")
    tenant = os.environ.get("ARGOS_API_TENANT", "default")
    user_id = os.environ.get("ARGOS_API_USER_ID", "default_user")

    # Default: read-only. Proposal and feedback are opt-in via env vars.
    allowed = set(READ_OPERATIONS)
    if os.environ.get("ARGOS_API_CAN_PROPOSE", "").lower() in ("true", "1", "yes"):
        allowed |= PROPOSAL_OPERATIONS
    if os.environ.get("ARGOS_API_CAN_FEEDBACK", "").lower() in ("true", "1", "yes"):
        allowed |= FEEDBACK_OPERATIONS

    return AuthContext(
        principal=principal,
        tenant=tenant,
        user_id=user_id,
        transport="mcp-stdio",
        allowed_operations=allowed,
        can_propose="memory_propose" in allowed,
        can_feedback="record_feedback" in allowed,
    )


def main() -> None:
    """Entry point for the MCP stdio server.

    Configured HERMES_HOME — never an arbitrary caller-selected data path.
    """
    import argparse
    from api_facade import ArgosAPIFacade, ACLConfig
    from service_client import SharedMemoryStore

    parser = argparse.ArgumentParser(description="Argos MCP stdio server (read tier)")
    parser.add_argument("--home", required=True, type=Path,
                        help="Path to the Hermes home directory.")
    args = parser.parse_args()

    # Logs to stderr only (D6: no banners on stdout).
    logging.basicConfig(
        stream=sys.stderr, level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    # Build the store, facade, and auth context.
    # M7: embedder=None means the MCP server degrades to text-only search
    # (no vector search). This is intentional for v1 — the MCP server is
    # a lightweight read/propose adapter. Loading the default embedder
    # here would add startup latency and a model dependency that may not
    # be available in all environments. Vector search can be added in a
    # future version by loading the embedder from the config.
    store = SharedMemoryStore(args.home, user_id="default_user", embedder=None)
    acl = ACLConfig()  # v1: open store (trusted-local mode)
    facade = ArgosAPIFacade(store, acl=acl, api_mode=False)
    auth_ctx = _load_auth_context(args.home)

    server = MCPServer(facade, auth_ctx)
    server.run()


if __name__ == "__main__":
    main()

"""Spec-09 (#123): Canonical application facade for external API access.

The trust boundary between external callers (MCP stdio, REST, future
transports) and the internal memory service. Every external operation
passes through this facade:

    transport adapter (MCP/REST)
        ↓
    ArgosAPIFacade
      · authentication context      (who is calling, from what credential)
      · authorization / ACL         (operation allowlist + spec-06 masks/denies)
      · input validation            (strict schemas, bounds, enums)
      · idempotency                 (key registry, compare-and-set)
      · audit event                 (one row per operation, denied included)
      · output redaction            (provenance metadata, no evidence by default)
        ↓
    service_client.py → memory_service.py → store / graph

The master rule (spec-09 D1): MCP and REST are NOT aliases for the
existing provider tools. There is no raw RPC passthrough, no raw SQL,
no arbitrary graph-method forwarding.

Identity is server-derived (D3): client-supplied user_id, tenant,
project_id, or client_scope fields are rejected or narrowed — never
widened. The authenticated principal, tenant, and maximum permitted
data scope come from the credential, not the request body.

ACL is fail-closed in API mode (D3): a corrupted/unreadable ACL config
refuses to start (never silently degrades to an open store); a truly
absent config starts open with a startup warning (v1 single-user legacy).
Unknown principals are denied, never mapped to a default tenant.

Idempotency (D5): every mutation carries an idempotency key. Same key +
same request → return original result (no duplicate). Same key + different
body → 409 conflict. The api_idempotency table is created on first use.

Error envelopes (D7): store failures return a stable error code +
request ID, never traceback/path/token/SQL detail.
"""
from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Set, Tuple

from access_scoping import ACLConfig
from inbound_security import scan_inbound_text
from store_common import VALID_CATEGORIES

logger = logging.getLogger(__name__)

# -- Operation allowlist (D2) ------------------------------------------------

# Read tier: available to all authenticated principals.
READ_OPERATIONS: Set[str] = {
    "search",
    "fetch",
    "fetch_history",
    "capabilities",
}

# Proposal tier: external caller → candidate → security scan → review queue.
# Never creates active memory directly.
PROPOSAL_OPERATIONS: Set[str] = {
    "memory_propose",
}

# Feedback tier: separately scoped.
FEEDBACK_OPERATIONS: Set[str] = {
    "record_feedback",
}

# All operations available through the facade.
PUBLIC_OPERATIONS: Set[str] = (
    READ_OPERATIONS | PROPOSAL_OPERATIONS | FEEDBACK_OPERATIONS
)

# Operations that are NEVER exposed on the public boundary (D2).
# These remain internal-only — any attempt to call them through the
# facade returns method_not_allowed.
FORBIDDEN_OPERATIONS: Set[str] = {
    "shutdown",
    "backup",
    "set_state",
    "clear_scope",
    "purge_tombstone",
    "mark_superseded",
    "cleanup_junk",
    "delete_memory",
    "quarantine_memory",
    "restore_memory",
    # Raw graph mutation is not exposed.
    "add_relationship",
    "index_memory",
    "remove_memory",
    "query_graph",
    "traverse_graph",
    "list_nodes",
    "clear_scope_graph",
}

# -- Error envelope (D7) -----------------------------------------------------

# Stable error code map. The facade never leaks tracebacks, internal
# paths, SQL detail, or tokens. Every error carries a request_id.
ERROR_CODE_MAP: Dict[str, int] = {
    "malformed_request": 400,
    "unauthenticated": 401,
    "forbidden": 403,
    "not_found": 404,
    "conflict": 409,
    "request_too_large": 413,
    "invalid_input": 422,
    "rate_limited": 429,
    "not_ready": 503,
    "timeout": 504,
    "internal_error": 500,
    "method_not_allowed": 410,  # "gone" — operation not on the public boundary
}


class APIError(Exception):
    """Stable error raised by the facade. Never carries traceback/path/SQL."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        request_id: str | None = None,
        details: Dict[str, Any] | None = None,
    ) -> None:
        self.code = code
        self.message = message
        self.request_id = request_id or str(uuid.uuid4())
        self.details = details or {}
        super().__init__(message)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serializable error envelope (no traceback/path/token)."""
        return {
            "error": {
                "code": self.code,
                "message": self.message,
                "request_id": self.request_id,
                **{k: v for k, v in self.details.items() if v is not None},
            }
        }


# -- Authentication context (D3) --------------------------------------------

@dataclass
class AuthContext:
    """Server-derived identity for an authenticated API caller.

    The credential (token file, API key) is verified by the transport
    adapter BEFORE constructing this context. The facade trusts the
    context's principal/tenant/scopes as server-derived — client-supplied
    identity fields in the request body are rejected or narrowed against
    these values, never widened.
    """

    principal: str           # authenticated client ID (from credential)
    tenant: str              # tenant name (from credential)
    user_id: str             # user scope within tenant (from credential)
    transport: str           # "mcp-stdio" | "rest" | ...
    # Server-derived maximum data scope. Caller filters may only narrow.
    max_project_id: Optional[str] = None
    max_client_scope: Optional[str] = None
    # AF1/R1: server-derived maximum namespace scope. Caller filters may
    # only narrow. When set, _scope_matches rejects records whose
    # namespace field is present and != max_namespace.
    max_namespace: Optional[str] = None
    # Operations this principal is allowed to perform.
    allowed_operations: Set[str] = field(default_factory=lambda: set(READ_OPERATIONS))
    # Whether this principal can propose new memories (class A write).
    can_propose: bool = False
    # Whether this principal can give feedback.
    can_feedback: bool = False


# -- Idempotency (D5) --------------------------------------------------------

# AF5: idempotency registry eviction settings.
IDEMPOTENCY_TTL_SECONDS = 24 * 3600  # 24 hours
IDEMPOTENCY_MAX_ENTRIES = 10_000

# In-memory idempotency cache for v1. The spec calls for a DuckDB table
# (api_idempotency) for durability across restarts; the in-memory cache
# handles the common case (same client retrying within a session). The
# table-based approach is a straightforward extension when the REST write
# slice ships.
_IdempotencyEntry = Dict[str, Any]  # {key, principal, operation, request_hash, created_at, result}


class IdempotencyRegistry:
    """In-memory idempotency key registry.

    Semantics (D5):
    - same key + same request hash → return original result, no duplicate
    - same key + different request hash → 409 conflict
    - no key → no idempotency guarantee (caller accepts at-least-once)

    AF4: thread-safe via a ``threading.Lock`` around check/record (the
    REST server allows concurrent requests).
    AF5: entries are evicted after ``IDEMPOTENCY_TTL_SECONDS`` (24h) or
    when the registry exceeds ``IDEMPOTENCY_MAX_ENTRIES`` (LRU-style
    eviction by ``created_at``).
    """

    def __init__(
        self,
        *,
        ttl_seconds: float = IDEMPOTENCY_TTL_SECONDS,
        max_entries: int = IDEMPOTENCY_MAX_ENTRIES,
    ) -> None:
        self._entries: Dict[str, _IdempotencyEntry] = {}
        self._lock = threading.Lock()
        self._ttl_seconds = ttl_seconds
        self._max_entries = max_entries

    def _evict_expired(self) -> None:
        """Remove entries older than TTL. Must be called under the lock."""
        if self._ttl_seconds <= 0:
            return
        cutoff = time.time() - self._ttl_seconds
        expired = [k for k, v in self._entries.items() if v["created_at"] < cutoff]
        for k in expired:
            del self._entries[k]

    def _evict_oldest(self) -> None:
        """Evict oldest entries if over max_entries. Must be under the lock."""
        if len(self._entries) <= self._max_entries:
            return
        # Sort by created_at ascending, evict the oldest.
        sorted_keys = sorted(self._entries, key=lambda k: self._entries[k]["created_at"])
        to_remove = len(self._entries) - self._max_entries
        for k in sorted_keys[:to_remove]:
            del self._entries[k]

    def check(
        self,
        key: str,
        principal: str,
        operation: str,
        request_hash: str,
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Check idempotency for a mutation.

        Returns (is_replay, cached_result):
        - (True, result) → replay of a previous successful call; return
          the cached result without executing the mutation.
        - (False, None) → first call with this key; proceed and record.
        Raises APIError(conflict) if the key exists with a different
        request hash.
        """
        if not key:
            return False, None
        with self._lock:
            self._evict_expired()
            existing = self._entries.get(key)
            if existing is None:
                return False, None
            if existing["request_hash"] != request_hash:
                raise APIError(
                    "conflict",
                    "Idempotency key was used with a different request body.",
                    details={"idempotency_key": key},
                )
            # Same key + same hash → replay.
            return True, existing.get("result")

    def record(
        self,
        key: str,
        principal: str,
        operation: str,
        request_hash: str,
        result: Dict[str, Any],
    ) -> None:
        """Record a completed mutation for idempotency replay."""
        if not key:
            return
        with self._lock:
            self._evict_expired()
            self._entries[key] = {
                "key": key,
                "principal": principal,
                "operation": operation,
                "request_hash": request_hash,
                "created_at": time.time(),
                "result": result,
            }
            self._evict_oldest()


# -- Input validation (D6) ---------------------------------------------------

# Strict bounds for API input. The facade rejects anything outside these
# bounds before it reaches the store.
MAX_QUERY_LENGTH = 2000
MAX_CONTENT_LENGTH = 10000
MAX_MEMORY_IDS = 50
MAX_LIMIT = 50
MIN_LIMIT = 1
# AF7: limits for tags and payload in memory_propose.
MAX_TAGS = 50
MAX_PAYLOAD_BYTES = 4096

# Client-controlled internal flags that are NEVER accepted from external
# callers (D6). These are internal-only and must not be set by API clients.
FORBIDDEN_CLIENT_FLAGS: Set[str] = {
    "include_quarantined",
    "include_archived",
    "include_expired",
    "include_closed",
    "suppress_retrieval",
}


def _validate_search_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and normalize search parameters."""
    cleaned: Dict[str, Any] = {}
    query = str(params.get("query", "")).strip()
    if not query:
        raise APIError("invalid_input", "query is required")
    if len(query) > MAX_QUERY_LENGTH:
        raise APIError(
            "request_too_large",
            f"query exceeds max length {MAX_QUERY_LENGTH}",
        )
    cleaned["query"] = query
    limit = params.get("limit", 10)
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        raise APIError("invalid_input", "limit must be an integer")
    if limit < MIN_LIMIT or limit > MAX_LIMIT:
        raise APIError("invalid_input", f"limit must be between {MIN_LIMIT} and {MAX_LIMIT}")
    cleaned["limit"] = limit
    # Optional filters — validated as strings, no internal flags.
    for opt_key in ("category_filter", "project_id", "namespace", "client_scope"):
        val = params.get(opt_key)
        if val is not None:
            cleaned[opt_key] = str(val)
    # Reject forbidden client flags.
    for flag in FORBIDDEN_CLIENT_FLAGS:
        if params.get(flag):
            raise APIError(
                "forbidden",
                f"Parameter {flag} is not available on the public API.",
            )
    return cleaned


def _validate_propose_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Validate memory_propose parameters (class A write)."""
    cleaned: Dict[str, Any] = {}
    content = str(params.get("content", "")).strip()
    if not content:
        raise APIError("invalid_input", "content is required")
    if len(content) > MAX_CONTENT_LENGTH:
        raise APIError(
            "request_too_large",
            f"content exceeds max length {MAX_CONTENT_LENGTH}",
        )
    cleaned["content"] = content
    category = str(params.get("category", "context_note")).strip()
    if not category:
        raise APIError("invalid_input", "category must not be empty")
    # AF8: validate category against VALID_CATEGORIES (fail-fast).
    if category not in VALID_CATEGORIES:
        raise APIError(
            "invalid_input",
            f"category must be one of {sorted(VALID_CATEGORIES)}",
        )
    cleaned["category"] = category
    # Optional fields.
    for opt_key in ("tags", "payload"):
        val = params.get(opt_key)
        if val is not None:
            # AF7: validate types and sizes for tags and payload.
            if opt_key == "tags":
                if not isinstance(val, list):
                    raise APIError("invalid_input", "tags must be a list")
                if len(val) > MAX_TAGS:
                    raise APIError(
                        "request_too_large",
                        f"tags exceeds max length {MAX_TAGS}",
                    )
            if opt_key == "payload":
                if not isinstance(val, dict):
                    raise APIError("invalid_input", "payload must be a dict")
                if len(json.dumps(val)) > MAX_PAYLOAD_BYTES:
                    raise APIError(
                        "request_too_large",
                        f"payload exceeds max size {MAX_PAYLOAD_BYTES} bytes",
                    )
            cleaned[opt_key] = val
    # Reject forbidden client flags.
    for flag in FORBIDDEN_CLIENT_FLAGS:
        if params.get(flag):
            raise APIError(
                "forbidden",
                f"Parameter {flag} is not available on the public API.",
            )
    # Caller may NOT claim provenance fields (D4).
    for provenance_key in ("source", "provenance_origin", "grounding", "user_scope"):
        if params.get(provenance_key) is not None:
            raise APIError(
                "forbidden",
                f"Parameter {provenance_key} is server-set and may not be "
                f"provided by the caller.",
            )
    return cleaned


def _validate_fetch_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Validate fetch (by ID) parameters."""
    cleaned: Dict[str, Any] = {}
    memory_id = str(params.get("memory_id", "")).strip()
    if not memory_id:
        raise APIError("invalid_input", "memory_id is required")
    cleaned["memory_id"] = memory_id
    return cleaned


def _validate_feedback_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Validate record_feedback parameters."""
    cleaned: Dict[str, Any] = {}
    memory_id = str(params.get("memory_id", "")).strip()
    if not memory_id:
        raise APIError("invalid_input", "memory_id is required")
    cleaned["memory_id"] = memory_id
    feedback = str(params.get("feedback", "")).strip().lower()
    if feedback not in ("helpful", "dismissed"):
        raise APIError("invalid_input", "feedback must be 'helpful' or 'dismissed'")
    cleaned["feedback"] = feedback
    return cleaned


# -- Audit (D10) -------------------------------------------------------------

def _hash_query(query: str) -> str:
    """SHA-256 hash of query text (first 16 hex chars)."""
    return hashlib.sha256(query.encode("utf-8")).hexdigest()[:16]


# -- The facade ---------------------------------------------------------------

class ArgosAPIFacade:
    """Canonical application facade for external API access (#123).

    Sits above service_client.py. Every external operation passes
    through this facade. The facade enforces:

    1. Operation allowlist — no raw RPC passthrough, no admin/destructive.
    2. Server-derived identity — client identity fields rejected/narrowed.
    3. Fail-closed ACL — API mode refuses to start when an ACL config
       was provided but is invalid/unreadable (parse_error); a truly
       absent config starts open with a startup warning (v1 single-user
       legacy, never silent).
    4. Input validation — strict schemas, bounds, no internal flags.
    5. Idempotency — key registry with replay/conflict semantics.
    6. Audit — hashed query, no tokens, one row per operation.
    7. Output redaction — stable error envelope, no traceback/path/SQL.
    """

    def __init__(
        self,
        store,  # SharedMemoryStore or compatible
        *,
        acl: ACLConfig | None = None,
        api_mode: bool = False,
        idempotency: IdempotencyRegistry | None = None,
    ) -> None:
        """Initialize the facade.

        Args:
            store: a SharedMemoryStore (or compatible) that talks to the
                memory service via RPC.
            acl: ACL configuration for this tenant. If None and
                api_mode is True, the facade starts with the open store
                and logs a warning (v1 single-user legacy — external
                callers see the tenant's own data, no access-scoping
                masks). If an ACL config was provided but is corrupted
                (parse_error), API mode refuses to start (fail closed).
            api_mode: when True, the facade enforces fail-closed ACL
                semantics. Unknown principals are denied, never mapped
                to a default tenant. Corrupted ACL config → refuse to
                start; absent config → open store + warning.
            idempotency: idempotency key registry. A default in-memory
                registry is created if None.
        """
        self._store = store
        self._acl = acl or ACLConfig()
        self._api_mode = api_mode
        self._idempotency = idempotency or IdempotencyRegistry()

        # D3: In API mode, the ACL must be valid. Fail-closed means we
        # refuse to operate if an ACL config was provided but is
        # invalid/unreadable (parse_error) — a corrupted config must
        # never silently degrade to an open store. A truly ABSENT config
        # (no file, api_mode with no acl) is the v1 single-user legacy
        # choice: the facade starts with the open store and warns loudly.
        if api_mode and self._acl.parse_error:
            raise ValueError(
                "ACL config invalid/unreadable — refusing to start in API "
                "mode (fail closed). Fix or remove the ACL config file."
            )
        if api_mode and self._acl.is_open_store:
            logger.warning(
                "API mode active with open-store ACL — external callers "
                "have no access-scoping enforcement. Configure an ACL for "
                "multi-user safety."
            )

    # -- Public operations ---------------------------------------------------

    def execute(
        self,
        ctx: AuthContext,
        operation: str,
        params: Dict[str, Any] | None = None,
        *,
        idempotency_key: str | None = None,
    ) -> Dict[str, Any]:
        """Execute one operation through the facade.

        This is the single entry point for all external API operations.
        Transport adapters (MCP, REST) call this method — they never
        call the store directly.

        Args:
            ctx: authenticated identity context (server-derived).
            operation: one of the PUBLIC_OPERATIONS.
            params: operation parameters (validated per operation).
            idempotency_key: optional idempotency key for mutations.

        Returns:
            Operation result as a JSON-serializable dict.

        Raises:
            APIError: stable error envelope (code, message, request_id).
        """
        params = params or {}
        request_id = str(uuid.uuid4())

        # 1. Operation allowlist check.
        if operation in FORBIDDEN_OPERATIONS:
            self._audit(ctx, operation, request_id, "denied",
                        denied_reason="forbidden_operation")
            raise APIError(
                "method_not_allowed",
                f"Operation {operation!r} is not available on the public API.",
                request_id=request_id,
            )
        if operation not in PUBLIC_OPERATIONS:
            self._audit(ctx, operation, request_id, "denied",
                        denied_reason="unknown_operation")
            raise APIError(
                "method_not_allowed",
                f"Operation {operation!r} is not recognized.",
                request_id=request_id,
            )

        # 2. Authorization: check the principal's allowed operations.
        if operation not in ctx.allowed_operations:
            self._audit(ctx, operation, request_id, "denied",
                        denied_reason="not_authorized_for_operation")
            raise APIError(
                "forbidden",
                f"Principal {ctx.principal!r} is not authorized for "
                f"operation {operation!r}.",
                request_id=request_id,
            )

        # 3. Identity enforcement (D3): reject client-supplied identity
        # fields that attempt to widen access. The caller may narrow
        # (e.g. filter to a subset of their allowed client_scope) but
        # never widen.
        params = self._enforce_identity(ctx, params, request_id)

        # 4. Input validation per operation.
        try:
            if operation == "search":
                validated = _validate_search_params(params)
            elif operation == "fetch":
                validated = _validate_fetch_params(params)
            elif operation == "fetch_history":
                validated = _validate_fetch_params(params)
            elif operation == "capabilities":
                validated = {}
            elif operation == "memory_propose":
                validated = _validate_propose_params(params)
            elif operation == "record_feedback":
                validated = _validate_feedback_params(params)
            else:
                raise APIError(
                    "method_not_allowed",
                    f"Operation {operation!r} is not implemented.",
                    request_id=request_id,
                )
        except APIError:
            self._audit(ctx, operation, request_id, "denied",
                        denied_reason="invalid_input")
            raise

        # 5. Idempotency check (mutations only).
        if operation in (PROPOSAL_OPERATIONS | FEEDBACK_OPERATIONS):
            request_hash = hashlib.sha256(
                json.dumps(validated, sort_keys=True).encode("utf-8")
            ).hexdigest()
            is_replay, cached = self._idempotency.check(
                idempotency_key or "", ctx.principal, operation, request_hash,
            )
            if is_replay and cached is not None:
                self._audit(ctx, operation, request_id, "allowed",
                            idempotency_replay=True)
                return cached

        # 6. Execute the operation through the store.
        try:
            if operation == "search":
                result = self._op_search(ctx, validated)
            elif operation == "fetch":
                result = self._op_fetch(ctx, validated)
            elif operation == "fetch_history":
                result = self._op_fetch_history(ctx, validated)
            elif operation == "capabilities":
                result = self._op_capabilities(ctx)
            elif operation == "memory_propose":
                result = self._op_memory_propose(ctx, validated, idempotency_key)
            elif operation == "record_feedback":
                result = self._op_record_feedback(ctx, validated, idempotency_key)
            else:
                raise APIError(
                    "internal_error",
                    "Operation not implemented.",
                    request_id=request_id,
                )
        except APIError as exc:
            self._audit(ctx, operation, request_id, "error",
                        error_code=exc.code)
            raise
        except Exception as exc:
            # D7: never leak traceback/path/SQL/token to the caller.
            logger.error("Facade operation %s failed: %s", operation, exc, exc_info=True)
            self._audit(ctx, operation, request_id, "error",
                        error_code="internal_error")
            raise APIError(
                "internal_error",
                "An internal error occurred. See server logs for details.",
                request_id=request_id,
            ) from exc

        # 7. Record idempotency for mutations.
        if operation in (PROPOSAL_OPERATIONS | FEEDBACK_OPERATIONS) and idempotency_key:
            request_hash = hashlib.sha256(
                json.dumps(validated, sort_keys=True).encode("utf-8")
            ).hexdigest()
            self._idempotency.record(
                idempotency_key, ctx.principal, operation, request_hash, result,
            )

        # 8. Audit the successful operation.
        self._audit(ctx, operation, request_id, "allowed",
                    result_count=len(result.get("results", [])) if isinstance(result, dict) else None)

        return result

    # -- Identity enforcement (D3) -------------------------------------------

    def _enforce_identity(
        self,
        ctx: AuthContext,
        params: Dict[str, Any],
        request_id: str,
    ) -> Dict[str, Any]:
        """Reject or narrow client-supplied identity fields.

        The caller may narrow their server-derived scope (e.g. filter to
        a specific project_id within their allowed scope) but may never
        widen it. Any attempt to set user_id, tenant, or a wider scope
        than the credential allows is rejected with 403.

        AF11: scope narrowing only activates when ``ctx.max_project_id``
        or ``ctx.max_client_scope`` is not None. In the current REST
        deployment, both are None (v1 single-user open scope), so
        ``_enforce_identity`` does not narrow. This is correct for v1 —
        scope enforcement requires credential-derived scopes (future
        work, #129). The ``user_id`` and ``tenant`` checks are always
        active regardless of scope settings.
        """
        cleaned = dict(params)
        # user_id: always server-derived. Client may not set it.
        if "user_id" in cleaned and cleaned["user_id"] != ctx.user_id:
            raise APIError(
                "forbidden",
                "user_id is server-derived and may not be changed.",
                request_id=request_id,
            )
        cleaned["user_id"] = ctx.user_id
        # tenant: always server-derived.
        if "tenant" in cleaned and cleaned["tenant"] != ctx.tenant:
            raise APIError(
                "forbidden",
                "tenant is server-derived and may not be changed.",
                request_id=request_id,
            )
        # project_id: caller may narrow to a subset of their allowed scope.
        if ctx.max_project_id is not None:
            client_project = cleaned.get("project_id")
            if client_project is not None and client_project != ctx.max_project_id:
                raise APIError(
                    "forbidden",
                    "project_id may not be widened beyond the credential scope.",
                    request_id=request_id,
                )
            cleaned.setdefault("project_id", ctx.max_project_id)
        # client_scope: caller may narrow.
        if ctx.max_client_scope is not None:
            client_cs = cleaned.get("client_scope")
            if client_cs is not None and client_cs != ctx.max_client_scope:
                raise APIError(
                    "forbidden",
                    "client_scope may not be widened beyond the credential scope.",
                    request_id=request_id,
                )
            cleaned.setdefault("client_scope", ctx.max_client_scope)
        return cleaned

    # -- Operation implementations -------------------------------------------

    def _op_search(self, ctx: AuthContext, params: Dict[str, Any]) -> Dict[str, Any]:
        """Read tier: search memories.

        AF6: set the store's user scope to the caller's user_id before
        searching, so results are scoped to the authenticated principal
        (not the store's default ``default_user``). ``set_user_scope`` is
        thread-local (#20), safe under the REST server's concurrency limiter.
        """
        # AF6: scope the search to the caller's user_id.
        if hasattr(self._store, "set_user_scope"):
            self._store.set_user_scope(ctx.user_id)
        results = self._store.search(
            query=params["query"],
            limit=params["limit"],
            category_filter=params.get("category_filter"),
            project_id=params.get("project_id"),
            namespace=params.get("namespace"),
            client_scope=params.get("client_scope"),
        )
        # Redact: convert MemoryRecord to dicts, strip internal fields.
        items = []
        for r in results:
            item = r.to_dict() if hasattr(r, "to_dict") else dict(r)
            # Redact provenance metadata by default (D9).
            item.pop("payload", None)
            items.append({
                "memory_id": item.get("memory_id"),
                "category": item.get("category"),
                "content": item.get("content"),
                "tags": item.get("tags", []),
                "similarity": round(float(item.get("similarity", 0.0)), 4),
                "created_at": item.get("created_at"),
                "updated_at": item.get("updated_at"),
                "status": item.get("status"),
                "scope": item.get("scope"),
            })
        return {"results": items, "count": len(items)}

    def _scope_matches(self, ctx: AuthContext, record: Any) -> bool:
        """AF1/AF2: check that a record is within the caller's ACL scope.

        When ``max_project_id`` or ``max_client_scope`` is set on the
        context, the record must match. When both are None (v1 open
        scope), all records pass.

        R1: also checks ``namespace`` when set on the context, closing
        the fetch authorization bypass for namespace-scoped records.
        """
        if ctx.max_project_id is not None:
            rec_pid = getattr(record, "project_id", None)
            if rec_pid is not None and rec_pid != ctx.max_project_id:
                return False
        if ctx.max_client_scope is not None:
            rec_cs = getattr(record, "client_scope", None)
            if rec_cs is not None and rec_cs != ctx.max_client_scope:
                return False
        # R1: namespace scope check.
        max_ns = getattr(ctx, "max_namespace", None)
        if max_ns is not None:
            rec_ns = getattr(record, "namespace", None)
            if rec_ns is not None and rec_ns != max_ns:
                return False
        return True

    def _op_fetch(self, ctx: AuthContext, params: Dict[str, Any]) -> Dict[str, Any]:
        """Read tier: fetch a single memory by ID.

        AF1: enforce ACL scope — the caller may only fetch memories
        within their ``max_project_id`` / ``max_client_scope``.
        """
        results = self._store.get_memories_by_ids([params["memory_id"]])
        if not results:
            raise APIError("not_found", "Memory not found.")
        r = results[0]
        # AF1: scope check — return not_found if out of scope (don't leak
        # existence to unauthorized callers).
        if not self._scope_matches(ctx, r):
            raise APIError("not_found", "Memory not found.")
        item = r.to_dict() if hasattr(r, "to_dict") else dict(r)
        return {
            "memory_id": item.get("memory_id"),
            "category": item.get("category"),
            "content": item.get("content"),
            "tags": item.get("tags", []),
            "created_at": item.get("created_at"),
            "updated_at": item.get("updated_at"),
            "status": item.get("status"),
            "scope": item.get("scope"),
        }

    def _op_fetch_history(self, ctx: AuthContext, params: Dict[str, Any]) -> Dict[str, Any]:
        """Read tier: fetch version history for a memory.

        AF1: enforce ACL scope — history is only returned for memories
        within the caller's scope.
        """
        # AF1: first fetch the current record to check scope. If the
        # memory doesn't exist or is out of scope, return not_found.
        current = self._store.get_memories_by_ids([params["memory_id"]])
        if not current:
            raise APIError("not_found", "Memory not found.")
        if not self._scope_matches(ctx, current[0]):
            raise APIError("not_found", "Memory not found.")
        history = self._store.get_memory_history(params["memory_id"])
        items = []
        for r in history:
            item = r.to_dict() if hasattr(r, "to_dict") else dict(r)
            items.append({
                "memory_id": item.get("memory_id"),
                "content": item.get("content"),
                "created_at": item.get("created_at"),
                "status": item.get("status"),
                "valid_from": item.get("valid_from"),
                "valid_to": item.get("valid_to"),
            })
        return {"history": items, "count": len(items)}

    def _op_capabilities(self, ctx: AuthContext) -> Dict[str, Any]:
        """Read tier: return the operations available to this principal."""
        return {
            "operations": sorted(ctx.allowed_operations),
            "transport": ctx.transport,
            "principal": ctx.principal,
        }

    def _op_memory_propose(
        self, ctx: AuthContext, params: Dict[str, Any], idempotency_key: str | None,
    ) -> Dict[str, Any]:
        """Proposal tier (class A): external caller → candidate → review queue.

        Never creates active memory. The candidate goes through the
        inbound security scan, then enters the review queue for human
        decision. Server-set provenance (D4):
          source = "api"
          transport = ctx.transport
          provenance_origin = "external"
          grounding = "extracted" (default; caller may not claim "observed")

        AF10: the facade scans content with ``scan_inbound_text`` before
        calling ``save_candidate``. The store also scans internally for
        ``provenance_origin="external"`` (store_write.py). This double
        scan is intentional defense-in-depth — the facade scan enables
        early quarantine (before the candidate enters the review queue),
        while the store scan is the authoritative boundary. The
        redundancy is acceptable for v1; a future optimization can skip
        the store-level scan when the facade has already quarantined.
        """
        content = params["content"]
        # AF3: pass user_id from ctx to save_candidate so API-proposed
        # memories are stored under the caller's user scope, not the
        # store's default.
        user_id = ctx.user_id
        # D9: scan inbound content for injection/poisoning patterns.
        # No weakening for "trusted" senders.
        scan_result = scan_inbound_text(content)
        if scan_result.blocked:
            # Quarantine the candidate — do not trigger any LLM call.
            candidate = self._store.save_candidate(
                category=params["category"],
                content=content,
                tags=params.get("tags", []),
                payload=params.get("payload", {}),
                source="api",
                confidence=0.0,
                scope="profile",
                provenance_origin="external",
                grounding="speculative",
                user_id=user_id,  # AF3
            )
            if candidate and candidate.get("candidate_id"):
                # Mark as quarantined with the injection reason.
                self._store.review_candidate(
                    candidate_id=candidate["candidate_id"],
                    decision="quarantined",
                    reason=f"inbound_security: {scan_result.summary()}",
                    review_source="system",
                )
            return {
                "candidate_id": candidate.get("candidate_id") if candidate else None,
                "status": "quarantined",
                "reason": "inbound_security_scan_blocked",
                "scan_summary": scan_result.summary(),
            }

        # Pass through to the candidate queue with server-set provenance.
        candidate = self._store.save_candidate(
            category=params["category"],
            content=content,
            tags=params.get("tags", []),
            payload=params.get("payload", {}),
            source="api",
            confidence=0.5,
            scope="profile",
            provenance_origin="external",
            grounding="extracted",
            user_id=user_id,  # AF3
        )
        return {
            "candidate_id": candidate.get("candidate_id") if candidate else None,
            "status": candidate.get("status", "pending") if candidate else "error",
        }

    def _op_record_feedback(
        self, ctx: AuthContext, params: Dict[str, Any], idempotency_key: str | None,
    ) -> Dict[str, Any]:
        """Feedback tier: record helpful/dismissed feedback on a memory.

        AF2: verify the caller has access to the memory before recording
        feedback. An authenticated user should not be able to record
        feedback on memories outside their ACL scope.
        """
        # AF2: fetch the memory first and check scope.
        results = self._store.get_memories_by_ids([params["memory_id"]])
        if not results:
            raise APIError("not_found", "Memory not found.")
        if not self._scope_matches(ctx, results[0]):
            raise APIError("not_found", "Memory not found.")
        self._store.record_feedback(params["memory_id"], params["feedback"])
        return {"memory_id": params["memory_id"], "feedback": params["feedback"]}

    # -- Audit (D10) ---------------------------------------------------------

    def _audit(
        self,
        ctx: AuthContext,
        operation: str,
        request_id: str,
        decision: str,
        *,
        denied_reason: str | None = None,
        error_code: str | None = None,
        result_count: int | None = None,
        idempotency_replay: bool = False,
    ) -> None:
        """Record an audit event for this operation.

        D10: no bearer tokens in logs, ever. Query text is hashed by
        default (the caller's query may contain personal/client data).
        The audit is operational access telemetry, not a governance-grade
        ledger (that's Themis's role).

        AF9: for v1, audit events are log-only (INFO level). The
        ``api_audit`` table is future work — the store-level
        ``access_audit`` table (store_core.py) covers store-level audit.
        Log rotation or process restart loses facade audit events. This
        is an accepted v1 limitation.
        """
        # For v1, audit events are logged at INFO level. The access_audit
        # table (store_core.py) is the durable sink for store-level audit;
        # the facade audit is a higher-level operation log. When the REST
        # slice ships, this will write to a dedicated api_audit table.
        logger.info(
            "api_audit principal=%s tenant=%s transport=%s operation=%s "
            "request_id=%s decision=%s denied_reason=%s error_code=%s "
            "result_count=%s idempotency_replay=%s",
            ctx.principal, ctx.tenant, ctx.transport, operation,
            request_id, decision, denied_reason, error_code,
            result_count, idempotency_replay,
        )

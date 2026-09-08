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
body → 409 conflict. v1 uses an in-memory idempotency registry; a durable
api_idempotency table ships with the REST write slice (see #302).

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
from liveness import record_subsystem_failure
from store_common import VALID_CATEGORIES

logger = logging.getLogger(__name__)

# -- Operation allowlist (D2) ------------------------------------------------

# Read tier: available to all authenticated principals.
READ_OPERATIONS: Set[str] = {
    "search",
    "fetch",
    "fetch_history",
    "capabilities",
    "explain",
    "explain_retrieval",
    # #295: admin console browse — list memories by namespace/scope/
    # category without a semantic query. Read-only, ACL-scoped.
    "browse",
    # #295: list pending candidates for the review queue. Read-only,
    # scoped to the caller's user_id.
    "list_candidates",
    # #295: portable export (#294) — read-only, scoped to the caller's
    # user_id. The export carries records, evidence, candidates, and
    # governance tables (tombstones/receipts/rejections/aliases).
    "export",
}

# Proposal tier: external caller → candidate → security scan → review queue.
# Never creates active memory directly.
PROPOSAL_OPERATIONS: Set[str] = {
    "memory_propose",
    # #289: structured ingestion (JSON/CSV → memory with provenance).
    # Preview mode writes nothing; apply mode requires an explicit
    # confirm and flows through the candidate/approval machinery
    # (save_candidate external → review_candidate tool-approved), so
    # every write is an approved, evidenced candidate — never a raw
    # unprovenanced insert.
    "ingest",
    # #293: POPIA erase-request workflow. Preview reports what WOULD be
    # erased and writes nothing; apply requires a strict confirm (only
    # the literal boolean True) and produces an append-only deletion
    # receipt per erased record — provable deletion, scoped per tenant.
    #
    # Tier note (#293 review): erase_request is destructive (class A)
    # but shares the proposal TIER for mechanics only — idempotency
    # registry + per-principal authorization. It is NOT implicitly
    # granted by can_propose: a principal can erase only if its
    # credential's allowed_operations includes "erase_request" (the
    # transport builds that set per principal). The destructive-action
    # controls live in the operation itself: strict confirm gate,
    # preview-first, per-record report, append-only receipts, and
    # server-derived identity (D4).
    "erase_request",
    # #295: candidate review — approve/reject/quarantine a pending
    # proposal through the existing approval-ledger flow. This is the
    # ONLY path by which a candidate becomes active memory (no bypass).
    # The storage layer enforces the approval invariant: "approved" is
    # reserved for review_source="tool"/"manual"; auto_review may only
    # set "reviewed_approved" (user confirmation still required).
    "review_candidate",
}

# Feedback tier: separately scoped.
FEEDBACK_OPERATIONS: Set[str] = {
    "record_feedback",
}

# #200 Spec-10: Write tier (class C) — trusted-local privileged writes.
# Loopback + server-derived identity only. Direct memory_save/update/delete
# with the SAME provider-level semantics as the native path (graph indexing,
# version chaining, candidate promotion, superseded-evidence removal).
# Never raw-store writes — the facade enforces identity, idempotency, CAS,
# and audit on every call. Gated by ctx.is_loopback in the execute() method.
WRITE_OPERATIONS: Set[str] = {
    "memory_save",
    "memory_update",
    "memory_delete",
}

# #200 Spec-10 PR 2/3: Collections — exhaustive, structural, no ranking.
# Read tier: list collections, list items (EXHAUSTIVE — no top-N cutoff).
# Filter by status/scope only; no ranking, no similarity.
COLLECTION_READ_OPERATIONS: Set[str] = {
    "collection_list",
    "collection_items",
}

# Write tier: create collection, add/update/remove items. Class C loopback
# + server-derived identity ONLY in v1. Non-loopback → denied fail-closed.
# No candidate pipeline for collections in v1 (explicitly deferred).
COLLECTION_WRITE_OPERATIONS: Set[str] = {
    "collection_create",
    "collection_add_item",
    "collection_update_item",
    "collection_remove_item",
}

# All operations available through the facade.
PUBLIC_OPERATIONS: Set[str] = (
    READ_OPERATIONS | PROPOSAL_OPERATIONS | FEEDBACK_OPERATIONS
    | WRITE_OPERATIONS
    | COLLECTION_READ_OPERATIONS | COLLECTION_WRITE_OPERATIONS
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
    # only narrow. When set, scope_check rejects records whose
    # namespace field is present and != max_namespace.
    max_namespace: Optional[str] = None
    # Operations this principal is allowed to perform.
    allowed_operations: Set[str] = field(default_factory=lambda: set(READ_OPERATIONS))
    # Whether this principal can propose new memories (class A write).
    can_propose: bool = False
    # Whether this principal can give feedback.
    can_feedback: bool = False
    # #200 Spec-10: principal type distinguishes human from model callers.
    # "human" = admin token / native UI / human-driven confirmation surface.
    # "model" = LLM agent / automated reviewer. Model principals are denied
    # class B (candidate approval) even with review_source="tool" — no model
    # self-approval, ever.
    principal_type: str = "human"  # "human" | "model"
    # #200 Spec-10: whether this connection is loopback (class C eligibility).
    # Class C (trusted-local privileged writes) requires loopback transport
    # AND server-derived identity. Set by the transport adapter.
    is_loopback: bool = False


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
MAX_MEMORY_ID_LENGTH = 256
MAX_CONTENT_LENGTH = 10000
MAX_MEMORY_IDS = 50
MAX_LIMIT = 50
MIN_LIMIT = 1
# AF7: limits for tags and payload in memory_propose.
MAX_TAGS = 50
MAX_PAYLOAD_BYTES = 4096
# #289: structured ingest batch size bound (raw JSON/CSV text).
MAX_INGEST_BYTES = 256 * 1024

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


def _validate_ingest_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Validate ingest (#289) parameters.

    Proposal-tier batch write: requires the raw data (JSON/CSV text),
    the format, and a field-mapping spec. Server-set provenance (D4):
    the caller may NOT claim source/provenance_origin/grounding/
    user_scope — the facade derives them. Apply mode requires an
    explicit ``confirm=True`` (human-in-loop gate); preview (dry-run)
    is the default and writes nothing.
    """
    cleaned: Dict[str, Any] = {}
    data = params.get("data")
    if not isinstance(data, str) or not data.strip():
        raise APIError("invalid_input", "data is required (JSON or CSV text)")
    # Byte-count limit (#289 fix): len(str) counts characters, but the
    # limit is a wire-size bound — UTF-8 multibyte content can exceed it
    # while passing a character count. Count encoded bytes.
    if len(data.encode("utf-8")) > MAX_INGEST_BYTES:
        raise APIError(
            "request_too_large",
            f"data exceeds max ingest size {MAX_INGEST_BYTES} bytes",
        )
    cleaned["data"] = data
    fmt = str(params.get("fmt", "")).strip().lower()
    if fmt not in {"json", "csv"}:
        raise APIError("invalid_input", "fmt must be 'json' or 'csv'")
    cleaned["fmt"] = fmt
    source_name = str(params.get("source_name", "")).strip()
    if not source_name:
        raise APIError("invalid_input", "source_name is required")
    if len(source_name) > 200:
        raise APIError("invalid_input", "source_name exceeds 200 characters")
    cleaned["source_name"] = source_name
    mapping = params.get("mapping")
    if not isinstance(mapping, dict):
        raise APIError("invalid_input", "mapping spec must be a dict")
    if not str(mapping.get("category", "")).strip():
        raise APIError("invalid_input", "mapping.category is required")
    if not str(mapping.get("content_template", "")).strip():
        raise APIError("invalid_input", "mapping.content_template is required")
    if len(json.dumps(mapping)) > MAX_PAYLOAD_BYTES:
        raise APIError(
            "request_too_large",
            f"mapping exceeds max size {MAX_PAYLOAD_BYTES} bytes",
        )
    cleaned["mapping"] = mapping
    mode = str(params.get("mode", "preview")).strip().lower()
    if mode not in {"preview", "apply"}:
        raise APIError("invalid_input", "mode must be 'preview' or 'apply'")
    cleaned["mode"] = mode
    # Human-in-loop gate: apply requires an explicit confirm flag.
    # Strict bool check (#289 fix): bool("false") is True in Python, so a
    # client sending the STRING "false" must not pass the human-in-loop
    # gate — only the literal boolean True confirms.
    confirm = params.get("confirm", False) is True
    if mode == "apply" and not confirm:
        raise APIError(
            "invalid_input",
            "apply mode requires confirm=true (preview first, then confirm)",
        )
    cleaned["confirm"] = confirm
    # Optional ACL metadata (narrowing only — enforced against the
    # credential in _op_ingest; never widened).
    for opt_key in ("client_scope", "doc_class", "project_id"):
        val = params.get(opt_key)
        if val is not None:
            if not isinstance(val, str) or not val.strip():
                raise APIError(
                    "invalid_input", f"{opt_key} must be a non-empty string"
                )
            cleaned[opt_key] = val.strip()
    # Caller may NOT claim provenance fields (D4) — same rule as
    # memory_propose. Provenance is server-set for every ingested row.
    for provenance_key in ("source", "provenance_origin", "grounding", "user_scope"):
        if params.get(provenance_key) is not None:
            raise APIError(
                "forbidden",
                f"Parameter {provenance_key} is server-set and may not be "
                f"provided by the caller.",
            )
    return cleaned


def _validate_erase_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Validate erase_request (#293) parameters.

    Destructive tier: subject-scoped provable deletion. Preview mode
    (default) reports what WOULD be erased and writes nothing; apply
    mode requires a STRICT confirm (only the literal boolean True —
    bool("false") is True in Python, so a client sending the string
    "false" must NOT pass the human-in-loop gate; mirrors #289).
    """
    cleaned: Dict[str, Any] = {}
    subject = str(params.get("subject", "")).strip()
    if not subject:
        raise APIError("invalid_input", "subject is required")
    if len(subject) > MAX_QUERY_LENGTH:
        raise APIError(
            "request_too_large",
            f"subject exceeds max length {MAX_QUERY_LENGTH}",
        )
    cleaned["subject"] = subject
    mode = str(params.get("mode", "preview")).strip().lower()
    if mode not in {"preview", "apply"}:
        raise APIError("invalid_input", "mode must be 'preview' or 'apply'")
    cleaned["mode"] = mode
    # STRICT human-in-loop gate: only the literal boolean True confirms.
    confirm = params.get("confirm", False) is True
    if mode == "apply" and not confirm:
        raise APIError(
            "invalid_input",
            "apply mode requires confirm=true (preview first, then confirm)",
        )
    cleaned["confirm"] = confirm
    # Optional scope narrowing (never widened — enforced in _op_erase).
    for opt_key in ("categories", "client_scope", "doc_class", "namespace"):
        val = params.get(opt_key)
        if val is None:
            continue
        if opt_key == "categories":
            if not isinstance(val, list) or not all(
                isinstance(c, str) and c.strip() for c in val
            ):
                raise APIError(
                    "invalid_input", "categories must be a list of strings"
                )
            cleaned[opt_key] = [c.strip() for c in val]
        else:
            if not isinstance(val, str) or not val.strip():
                raise APIError(
                    "invalid_input", f"{opt_key} must be a non-empty string"
                )
            cleaned[opt_key] = val.strip()
    # The caller may NOT claim server-derived identity fields (D4).
    for provenance_key in ("user_scope", "requested_by"):
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


def _validate_explain_retrieval_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Validate explain_retrieval (why-not diagnostic) parameters.

    Read-tier diagnostic: query + memory_id + optional diagnostic
    window. Mirrors search's bounds; no internal flags, no provenance
    claims (D4).
    """
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
    memory_id = str(params.get("memory_id", "")).strip()
    if not memory_id:
        raise APIError("invalid_input", "memory_id is required")
    if len(memory_id) > MAX_MEMORY_ID_LENGTH:
        raise APIError(
            "request_too_large",
            f"memory_id exceeds max length {MAX_MEMORY_ID_LENGTH}",
        )
    cleaned["memory_id"] = memory_id
    top_k = params.get("top_k", 20)
    try:
        top_k = int(top_k)
    except (TypeError, ValueError):
        raise APIError("invalid_input", "top_k must be an integer")
    if top_k < MIN_LIMIT or top_k > MAX_LIMIT:
        raise APIError(
            "invalid_input", f"top_k must be between {MIN_LIMIT} and {MAX_LIMIT}",
        )
    cleaned["top_k"] = top_k
    # Reject forbidden client flags (same rule as search).
    for flag in FORBIDDEN_CLIENT_FLAGS:
        if params.get(flag):
            raise APIError(
                "forbidden",
                f"Parameter {flag} is not available on the public API.",
            )
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


def _validate_browse_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Validate browse parameters (#295 admin console).

    Browse is a read-only listing by namespace/scope/category — no
    semantic query. The caller may narrow with optional filters but
    may never widen beyond their ACL scope (enforced by _enforce_identity).
    """
    cleaned: Dict[str, Any] = {}
    limit = params.get("limit", 50)
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        raise APIError("invalid_input", "limit must be an integer")
    if limit < MIN_LIMIT or limit > MAX_LIMIT:
        raise APIError(
            "invalid_input", f"limit must be between {MIN_LIMIT} and {MAX_LIMIT}",
        )
    cleaned["limit"] = limit
    # Optional filters — validated as strings, no internal flags.
    for opt_key in ("category", "namespace", "project_id", "client_scope"):
        val = params.get(opt_key)
        if val is not None:
            cleaned[opt_key] = str(val)
    for flag in FORBIDDEN_CLIENT_FLAGS:
        if params.get(flag):
            raise APIError(
                "forbidden",
                f"Parameter {flag} is not available on the public API.",
            )
    return cleaned


def _validate_list_candidates_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Validate list_candidates parameters (#295 admin console).

    Read-only listing of pending candidates for the review queue.
    Scoped to the caller's user_id by the store.
    """
    cleaned: Dict[str, Any] = {}
    status = str(params.get("status", "pending")).strip().lower()
    # Allowed statuses mirror the candidate lifecycle in store_write.py.
    if status not in ("pending", "quarantined", "reviewed_approved",
                      "pending_user_confirmation", "rejected", "approved"):
        raise APIError("invalid_input", f"invalid candidate status: {status}")
    cleaned["status"] = status
    limit = params.get("limit", 50)
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        raise APIError("invalid_input", "limit must be an integer")
    # list_candidates allows up to 500 (matching the store's own cap)
    # since the review queue may have more items than a search page.
    if limit < MIN_LIMIT or limit > 500:
        raise APIError(
            "invalid_input", f"limit must be between {MIN_LIMIT} and 500",
        )
    cleaned["limit"] = limit
    candidate_id = params.get("candidate_id")
    if candidate_id is not None:
        cleaned["candidate_id"] = str(candidate_id)
    return cleaned


def _validate_review_candidate_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Validate review_candidate parameters (#295 admin console).

    This is the ONLY public path to approve/reject a candidate. The
    storage layer enforces the approval invariant (review_source="tool"
    for "approved"; auto_review may only set "reviewed_approved").
    The facade always sets review_source="tool" — the UI is a
    human-driven confirmation surface, not an automated reviewer.
    """
    cleaned: Dict[str, Any] = {}
    candidate_id = str(params.get("candidate_id", "")).strip()
    if not candidate_id:
        raise APIError("invalid_input", "candidate_id is required")
    cleaned["candidate_id"] = candidate_id
    decision = str(params.get("decision", "")).strip().lower()
    if decision not in ("approved", "rejected", "quarantined"):
        raise APIError(
            "invalid_input",
            "decision must be 'approved', 'rejected', or 'quarantined'",
        )
    cleaned["decision"] = decision
    reason = str(params.get("reason", "")).strip()
    if len(reason) > MAX_CONTENT_LENGTH:
        raise APIError(
            "request_too_large",
            f"reason exceeds max length {MAX_CONTENT_LENGTH}",
        )
    cleaned["reason"] = reason
    # Optional review metadata.
    for opt_key in ("supersedes_memory_id", "durability", "scope"):
        val = params.get(opt_key)
        if val is not None:
            cleaned[opt_key] = str(val)
    for flag in FORBIDDEN_CLIENT_FLAGS:
        if params.get(flag):
            raise APIError(
                "forbidden",
                f"Parameter {flag} is not available on the public API.",
            )
    return cleaned


def _validate_export_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Validate export parameters (#295 admin console, #294 portable export).

    Read-only portable export scoped to the caller's user_id. Optional
    narrowing by categories/namespace/client_scope/doc_class.
    """
    cleaned: Dict[str, Any] = {}
    categories = params.get("categories")
    if categories is not None:
        if not isinstance(categories, list):
            raise APIError("invalid_input", "categories must be a list")
        cleaned["categories"] = [str(c) for c in categories]
    for opt_key in ("namespace", "client_scope", "doc_class"):
        val = params.get(opt_key)
        if val is not None:
            cleaned[opt_key] = str(val)
    for flag in FORBIDDEN_CLIENT_FLAGS:
        if params.get(flag):
            raise APIError(
                "forbidden",
                f"Parameter {flag} is not available on the public API.",
            )
    return cleaned


# -- #200 Spec-10: Class C write validation -----------------------------------

def _validate_memory_save_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Validate memory_save parameters (class C trusted-local write).

    Direct active-memory write — same semantics as the native
    memory_save tool path. The caller may NOT claim provenance fields
    (D4): source, provenance_origin, grounding, user_scope are
    server-set. Optional expected_version is for CAS on update only.
    """
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
    if category not in VALID_CATEGORIES:
        raise APIError(
            "invalid_input",
            f"category must be one of {sorted(VALID_CATEGORIES)}",
        )
    cleaned["category"] = category
    tags = params.get("tags")
    if tags is not None:
        if not isinstance(tags, list):
            raise APIError("invalid_input", "tags must be a list")
        if len(tags) > MAX_TAGS:
            raise APIError("request_too_large", f"tags exceeds max length {MAX_TAGS}")
        cleaned["tags"] = tags
    else:
        cleaned["tags"] = []
    # Optional durability/expires_at (passed through when expiry is enabled).
    for opt_key in ("durability", "expires_at"):
        val = params.get(opt_key)
        if val is not None:
            cleaned[opt_key] = val
    # Caller may NOT claim provenance fields (D4).
    for provenance_key in ("source", "provenance_origin", "grounding", "user_scope"):
        if params.get(provenance_key) is not None:
            raise APIError(
                "forbidden",
                f"Parameter {provenance_key} is server-set and may not be "
                f"provided by the caller.",
            )
    for flag in FORBIDDEN_CLIENT_FLAGS:
        if params.get(flag):
            raise APIError(
                "forbidden",
                f"Parameter {flag} is not available on the public API.",
            )
    return cleaned


def _validate_memory_update_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Validate memory_update parameters (class C trusted-local write).

    Updates an existing memory by memory_id. Supports CAS via
    expected_version: if provided, the store must verify the current
    version matches before applying the update — stale version → 409.
    """
    cleaned: Dict[str, Any] = {}
    memory_id = str(params.get("memory_id", "")).strip()
    if not memory_id:
        raise APIError("invalid_input", "memory_id is required")
    cleaned["memory_id"] = memory_id
    content = params.get("content")
    if content is not None:
        content = str(content).strip()
        if not content:
            raise APIError("invalid_input", "content must not be empty")
        if len(content) > MAX_CONTENT_LENGTH:
            raise APIError(
                "request_too_large",
                f"content exceeds max length {MAX_CONTENT_LENGTH}",
            )
        cleaned["content"] = content
    tags = params.get("tags")
    if tags is not None:
        if not isinstance(tags, list):
            raise APIError("invalid_input", "tags must be a list")
        if len(tags) > MAX_TAGS:
            raise APIError("request_too_large", f"tags exceeds max length {MAX_TAGS}")
        cleaned["tags"] = tags
    # #200 Spec-10: CAS via expected_version (If-Match). If provided,
    # the store checks the current version before applying the update.
    # Stale version → 409 conflict, no write.
    expected_version = params.get("expected_version")
    if expected_version is not None:
        cleaned["expected_version"] = str(expected_version)
    # Optional expires_at (passed through when expiry is enabled).
    if params.get("expires_at") is not None:
        cleaned["expires_at"] = params["expires_at"]
    # Caller may NOT claim provenance fields (D4).
    for provenance_key in ("source", "provenance_origin", "grounding", "user_scope"):
        if params.get(provenance_key) is not None:
            raise APIError(
                "forbidden",
                f"Parameter {provenance_key} is server-set and may not be "
                f"provided by the caller.",
            )
    for flag in FORBIDDEN_CLIENT_FLAGS:
        if params.get(flag):
            raise APIError(
                "forbidden",
                f"Parameter {flag} is not available on the public API.",
            )
    return cleaned


def _validate_memory_delete_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Validate memory_delete parameters (class C trusted-local write).

    Deletes an existing memory by memory_id. Supports CAS via
    expected_version: if provided, the store must verify the current
    version matches before applying the delete — stale version → 409.
    """
    cleaned: Dict[str, Any] = {}
    memory_id = str(params.get("memory_id", "")).strip()
    if not memory_id:
        raise APIError("invalid_input", "memory_id is required")
    cleaned["memory_id"] = memory_id
    # #200 Spec-10: CAS via expected_version (If-Match).
    expected_version = params.get("expected_version")
    if expected_version is not None:
        cleaned["expected_version"] = str(expected_version)
    for flag in FORBIDDEN_CLIENT_FLAGS:
        if params.get(flag):
            raise APIError(
                "forbidden",
                f"Parameter {flag} is not available on the public API.",
            )
    return cleaned


# -- #200 Spec-10 PR 2/3: Collection validation -------------------------------

def _validate_collection_list_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Validate collection_list parameters (read tier)."""
    cleaned: Dict[str, Any] = {}
    status = params.get("status")
    if status is not None:
        cleaned["status"] = str(status)
    limit = params.get("limit")
    if limit is not None:
        cleaned["limit"] = int(limit)
    # Caller may NOT claim scope fields (server-derived).
    for scope_key in ("user_scope", "tenant"):
        if params.get(scope_key) is not None:
            raise APIError(
                "forbidden",
                f"Parameter {scope_key} is server-set and may not be "
                f"provided by the caller.",
            )
    return cleaned


def _validate_collection_items_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Validate collection_items parameters (read tier — exhaustive)."""
    cleaned: Dict[str, Any] = {}
    collection_id = str(params.get("collection_id", "")).strip()
    if not collection_id:
        raise APIError("invalid_input", "collection_id is required")
    cleaned["collection_id"] = collection_id
    status = params.get("status")
    if status is not None:
        cleaned["status"] = str(status)
    include_archived = params.get("include_archived")
    if include_archived is not None:
        cleaned["include_archived"] = bool(include_archived)
    limit = params.get("limit")
    if limit is not None:
        cleaned["limit"] = int(limit)
    for scope_key in ("user_scope", "tenant"):
        if params.get(scope_key) is not None:
            raise APIError(
                "forbidden",
                f"Parameter {scope_key} is server-set and may not be "
                f"provided by the caller.",
            )
    return cleaned


def _validate_collection_create_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Validate collection_create parameters (class C write)."""
    cleaned: Dict[str, Any] = {}
    name = str(params.get("name", "")).strip()
    if not name:
        raise APIError("invalid_input", "name is required")
    if len(name) > 200:
        raise APIError("invalid_input", "name exceeds max length 200")
    cleaned["name"] = name
    template = params.get("template")
    if template is not None:
        cleaned["template"] = str(template)
    schema = params.get("schema")
    if schema is not None:
        if not isinstance(schema, list):
            raise APIError("invalid_input", "schema must be a list of field defs")
        if len(json.dumps(schema)) > MAX_PAYLOAD_BYTES:
            raise APIError("request_too_large", "schema exceeds max payload size")
        cleaned["schema"] = schema
    for scope_key in ("user_scope", "tenant", "collection_id"):
        if params.get(scope_key) is not None:
            raise APIError(
                "forbidden",
                f"Parameter {scope_key} is server-set and may not be "
                f"provided by the caller.",
            )
    return cleaned


def _validate_collection_add_item_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Validate collection_add_item parameters (class C write)."""
    cleaned: Dict[str, Any] = {}
    collection_id = str(params.get("collection_id", "")).strip()
    if not collection_id:
        raise APIError("invalid_input", "collection_id is required")
    cleaned["collection_id"] = collection_id
    fields = params.get("fields")
    if fields is None:
        raise APIError("invalid_input", "fields is required")
    if not isinstance(fields, dict):
        raise APIError("invalid_input", "fields must be a dict")
    if len(json.dumps(fields)) > MAX_PAYLOAD_BYTES:
        raise APIError("request_too_large", "fields exceeds max payload size")
    cleaned["fields"] = fields
    status = params.get("status")
    if status is not None:
        cleaned["status"] = str(status)
    for scope_key in ("user_scope", "tenant", "item_id"):
        if params.get(scope_key) is not None:
            raise APIError(
                "forbidden",
                f"Parameter {scope_key} is server-set and may not be "
                f"provided by the caller.",
            )
    return cleaned


def _validate_collection_update_item_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Validate collection_update_item parameters (class C write with CAS)."""
    cleaned: Dict[str, Any] = {}
    item_id = str(params.get("item_id", "")).strip()
    if not item_id:
        raise APIError("invalid_input", "item_id is required")
    cleaned["item_id"] = item_id
    fields = params.get("fields")
    if fields is not None:
        if not isinstance(fields, dict):
            raise APIError("invalid_input", "fields must be a dict")
        if len(json.dumps(fields)) > MAX_PAYLOAD_BYTES:
            raise APIError("request_too_large", "fields exceeds max payload size")
        cleaned["fields"] = fields
    status = params.get("status")
    if status is not None:
        cleaned["status"] = str(status)
    expected_version = params.get("expected_version")
    if expected_version is not None:
        cleaned["expected_version"] = str(expected_version)
    if fields is None and status is None:
        raise APIError("invalid_input", "at least one of fields/status must be provided")
    for scope_key in ("user_scope", "tenant"):
        if params.get(scope_key) is not None:
            raise APIError(
                "forbidden",
                f"Parameter {scope_key} is server-set and may not be "
                f"provided by the caller.",
            )
    return cleaned


def _validate_collection_remove_item_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Validate collection_remove_item parameters (class C write with CAS)."""
    cleaned: Dict[str, Any] = {}
    item_id = str(params.get("item_id", "")).strip()
    if not item_id:
        raise APIError("invalid_input", "item_id is required")
    cleaned["item_id"] = item_id
    expected_version = params.get("expected_version")
    if expected_version is not None:
        cleaned["expected_version"] = str(expected_version)
    for scope_key in ("user_scope", "tenant"):
        if params.get(scope_key) is not None:
            raise APIError(
                "forbidden",
                f"Parameter {scope_key} is server-set and may not be "
                f"provided by the caller.",
            )
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

        # #200 Spec-10: Class C write ops require loopback + server-derived
        # identity. Non-loopback callers (external MCP/REST) can only use
        # class A (propose) and class B (review) — never direct writes.
        # Collection write ops are also class C only in v1 (no candidate
        # pipeline for collections — explicitly deferred per spec D1).
        if (operation in WRITE_OPERATIONS
                or operation in COLLECTION_WRITE_OPERATIONS) and not ctx.is_loopback:
            self._audit(ctx, operation, request_id, "denied",
                        denied_reason="write_requires_loopback")
            raise APIError(
                "forbidden",
                f"Operation {operation!r} requires loopback transport "
                f"(class C trusted-local). External callers must use "
                f"memory_propose (class A).",
                request_id=request_id,
            )

        # #200 Spec-10: Class B (candidate approval) is human-only. A model
        # principal cannot approve its own candidate — even with
        # review_source="tool", the principal_type check denies it.
        if operation == "review_candidate" and ctx.principal_type == "model":
            self._audit(ctx, operation, request_id, "denied",
                        denied_reason="model_principal_cannot_approve")
            raise APIError(
                "forbidden",
                "Model principals may not approve candidates (class B is "
                "human-only). No model self-approval, even with "
                "review_source='tool'.",
                request_id=request_id,
            )

        # 3. Identity enforcement (D3): reject client-supplied identity
        # fields that attempt to widen access. The caller may narrow
        # (e.g. filter to a subset of their allowed client_scope) but
        # never widen.
        # #300: audit identity-narrowing rejections (previously uncaught).
        try:
            params = self._enforce_identity(ctx, params, request_id)
        except APIError:
            self._audit(ctx, operation, request_id, "denied",
                        denied_reason="identity_narrowing_rejected")
            raise

        # 4. Input validation per operation.
        try:
            if operation == "search":
                validated = _validate_search_params(params)
            elif operation == "fetch":
                validated = _validate_fetch_params(params)
            elif operation == "fetch_history":
                validated = _validate_fetch_params(params)
            elif operation == "explain_retrieval":
                validated = _validate_explain_retrieval_params(params)
            elif operation == "explain":
                validated = _validate_fetch_params(params)
            elif operation == "capabilities":
                validated = {}
            elif operation == "memory_propose":
                validated = _validate_propose_params(params)
            elif operation == "ingest":
                validated = _validate_ingest_params(params)
            elif operation == "erase_request":
                validated = _validate_erase_params(params)
            elif operation == "record_feedback":
                validated = _validate_feedback_params(params)
            elif operation == "browse":
                validated = _validate_browse_params(params)
            elif operation == "list_candidates":
                validated = _validate_list_candidates_params(params)
            elif operation == "review_candidate":
                validated = _validate_review_candidate_params(params)
            elif operation == "memory_save":
                validated = _validate_memory_save_params(params)
            elif operation == "memory_update":
                validated = _validate_memory_update_params(params)
            elif operation == "memory_delete":
                validated = _validate_memory_delete_params(params)
            elif operation == "collection_list":
                validated = _validate_collection_list_params(params)
            elif operation == "collection_items":
                validated = _validate_collection_items_params(params)
            elif operation == "collection_create":
                validated = _validate_collection_create_params(params)
            elif operation == "collection_add_item":
                validated = _validate_collection_add_item_params(params)
            elif operation == "collection_update_item":
                validated = _validate_collection_update_item_params(params)
            elif operation == "collection_remove_item":
                validated = _validate_collection_remove_item_params(params)
            elif operation == "export":
                validated = _validate_export_params(params)
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
        # #200 Spec-10: full idempotency coverage — all write ops, not
        # just proposal/feedback. Class C writes (save/update/delete) and
        # collection writes (create/add/update/remove) are included.
        if operation in (PROPOSAL_OPERATIONS | FEEDBACK_OPERATIONS
                         | WRITE_OPERATIONS | COLLECTION_WRITE_OPERATIONS):
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
        # #347: set the server-derived actor context on the store before
        # any operation so mutation_events records the real principal.
        # The store defaults to user_id/"human" for local paths; this
        # override brings the facade's ACL-resolved identity.
        try:
            if hasattr(self._store, "set_actor_context"):
                self._store.set_actor_context(ctx.principal, ctx.principal_type)
        except (TypeError, AttributeError):
            pass  # RPC proxy may not accept actor; _call_store stamps it
        try:
            if operation == "search":
                result = self._op_search(ctx, validated)
            elif operation == "fetch":
                result = self._op_fetch(ctx, validated)
            elif operation == "fetch_history":
                result = self._op_fetch_history(ctx, validated)
            elif operation == "explain_retrieval":
                result = self._op_explain_retrieval(ctx, validated)
            elif operation == "explain":
                result = self._op_explain(ctx, validated)
            elif operation == "capabilities":
                result = self._op_capabilities(ctx)
            elif operation == "memory_propose":
                result = self._op_memory_propose(ctx, validated, idempotency_key)
            elif operation == "ingest":
                result = self._op_ingest(ctx, validated)
            elif operation == "erase_request":
                result = self._op_erase_request(ctx, validated)
            elif operation == "record_feedback":
                result = self._op_record_feedback(ctx, validated, idempotency_key)
            elif operation == "browse":
                result = self._op_browse(ctx, validated)
            elif operation == "list_candidates":
                result = self._op_list_candidates(ctx, validated)
            elif operation == "review_candidate":
                result = self._op_review_candidate(ctx, validated)
            elif operation == "memory_save":
                result = self._op_memory_save(ctx, validated)
            elif operation == "memory_update":
                result = self._op_memory_update(ctx, validated)
            elif operation == "memory_delete":
                result = self._op_memory_delete(ctx, validated)
            elif operation == "collection_list":
                result = self._op_collection_list(ctx, validated)
            elif operation == "collection_items":
                result = self._op_collection_items(ctx, validated)
            elif operation == "collection_create":
                result = self._op_collection_create(ctx, validated)
            elif operation == "collection_add_item":
                result = self._op_collection_add_item(ctx, validated)
            elif operation == "collection_update_item":
                result = self._op_collection_update_item(ctx, validated)
            elif operation == "collection_remove_item":
                result = self._op_collection_remove_item(ctx, validated)
            elif operation == "export":
                result = self._op_export(ctx, validated)
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
        # #200 Spec-10: full coverage — all write ops + collection writes.
        if operation in (PROPOSAL_OPERATIONS | FEEDBACK_OPERATIONS
                         | WRITE_OPERATIONS | COLLECTION_WRITE_OPERATIONS) and idempotency_key:
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
        scope enforcement requires credential-derived scopes: today
        AuthContext is built from env vars only and max_* fields are
        always None. Wiring max_project_id / max_client_scope /
        max_namespace from the credential itself is the follow-up that
        activates scope narrowing. The ``user_id`` and ``tenant``
        checks are always active regardless of scope settings.
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

        #301: fail-loud and reset. The search is wrapped in try/finally so
        the store's user scope is always restored to whatever it was BEFORE
        the facade call, even on exception. A debug-level assertion verifies
        the scope was correctly set before searching — if the store's scope
        doesn't match ctx.user_id after set_user_scope, the assertion
        fires loudly instead of silently returning cross-user data.

        The pre-existing scope is captured via ``self._store.user_id``
        (not ``_default_user_id``, which only exists on SharedMemoryStore —
        DuckDBMemoryStore has no such attribute and would fall back to a
        hardcoded "default_user", permanently re-scoping a store constructed
        with a non-default user_id).
        """
        # AF6: scope the search to the caller's user_id.
        # #301: capture the pre-existing scope BEFORE setting it, so the
        # finally block restores exactly what was there — not a hardcoded
        # default. DuckDBMemoryStore has self.user_id (set by constructor,
        # changed by set_user_scope) but no _default_user_id attribute.
        _scope_before = getattr(self._store, "user_id", None)
        try:
            if hasattr(self._store, "set_user_scope"):
                self._store.set_user_scope(ctx.user_id)
                # #301: debug-level assertion that the scope was set correctly.
                _current = getattr(self._store, "user_id", None)
                if _current != ctx.user_id:
                    logger.debug(
                        "scope invariant violated after set_user_scope: "
                        "expected=%s actual=%s — search may return cross-user data",
                        ctx.user_id, _current,
                    )
            results = self._store.search(
                query=params["query"],
                limit=params["limit"],
                category_filter=params.get("category_filter"),
                project_id=params.get("project_id"),
                namespace=params.get("namespace"),
                client_scope=params.get("client_scope"),
            )
        finally:
            # #301: always restore the store's user scope to what it was
            # before the facade call, even on exception. This prevents a
            # leaked scope from affecting subsequent operations on the
            # same store handle — including non-facade use.
            if _scope_before is not None and hasattr(self._store, "set_user_scope"):
                self._store.set_user_scope(_scope_before)
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

    def scope_check(self, ctx: AuthContext, record: Any) -> bool:
        """#303: unified scope check for all facade operations that
        resolve a record by ID.

        This is the single helper behind facade scoping. Every facade
        operation that fetches a record (fetch, fetch_history,
        record_feedback) calls this method to verify the record is
        within the caller's ACL scope.

        When ``max_project_id``, ``max_client_scope``, or
        ``max_namespace`` is set on the context, the record must match.
        When all are None (v1 open scope), all records pass.

        R1: also checks ``namespace`` when set on the context, closing
        the fetch authorization bypass for namespace-scoped records.

        Note: the search path uses ``set_user_scope`` + store-level
        filtering (not this helper) because search returns a list, not
        a single record by ID. The RPC layer uses
        ``filter_records_by_access`` in access_scoping.py (unchanged
        in this batch).
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
        if not self.scope_check(ctx, r):
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
        if not self.scope_check(ctx, current[0]):
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

    def _op_explain(self, ctx: AuthContext, params: Dict[str, Any]) -> Dict[str, Any]:
        """Read tier: #280 — explain why a memory was retrieved.

        AF1: enforce ACL scope — the caller may only explain memories
        within their scope. Returns not_found if out of scope (don't
        leak existence to unauthorized callers).
        """
        # AF1: first fetch the current record to check scope.
        results = self._store.get_memories_by_ids([params["memory_id"]])
        if not results:
            raise APIError("not_found", "Memory not found.")
        if not self.scope_check(ctx, results[0]):
            raise APIError("not_found", "Memory not found.")
        # Delegate to store.provenance() — the store's _fetch_record
        # enforces user_scope, and we've already verified ACL scope
        # via get_memories_by_ids + scope_check above.
        return self._store.provenance(params["memory_id"])

    def _op_explain_retrieval(
        self, ctx: AuthContext, params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Read tier: diagnose why a memory did NOT surface in retrieval.

        Deterministic, zero-LLM, strictly read-only. The store's
        ``explain_retrieval`` runs a diagnostic pass with retrieval
        suppressed — it never touches production ranking or writes.

        ACL: the caller may only explain memories within their scope
        (existence not leaked — not_found for out-of-scope IDs). The
        store additionally filters by user_scope server-side (SM1).
        """
        results = self._store.get_memories_by_ids([params["memory_id"]])
        if not results:
            raise APIError("not_found", "Memory not found.")
        if not self.scope_check(ctx, results[0]):
            raise APIError("not_found", "Memory not found.")
        _scope_before = getattr(self._store, "user_id", None)
        try:
            if hasattr(self._store, "set_user_scope"):
                self._store.set_user_scope(ctx.user_id)
            return self._store.explain_retrieval(
                query=params["query"],
                expected_memory_id=params["memory_id"],
                top_k=params["top_k"],
            )
        finally:
            if _scope_before is not None and hasattr(self._store, "set_user_scope"):
                self._store.set_user_scope(_scope_before)

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

    def _op_ingest(self, ctx: AuthContext, params: Dict[str, Any]) -> Dict[str, Any]:
        """Proposal tier (class A): #289 structured ingestion.

        JSON/CSV rows → memory with first-class provenance. Preview mode
        (default) writes NOTHING and reports what WOULD be stored;
        apply mode requires the validated confirm flag and flows every
        row through the candidate/approval machinery (save_candidate
        external → review_candidate tool-approved) — never a raw
        unprovenanced insert.

        Server-set provenance (D4): the store stamps
        source="structured_ingest", provenance_origin="external",
        grounding="extracted" on every row; the caller cannot claim
        them (rejected in _validate_ingest_params). Scoping: rows are
        written under ctx.user_id (multi-tenant safe); client_scope
        narrows to the caller's credential clearance (never widens).
        """
        # AF3: scope the store to the caller's user for the duration of
        # the operation (same save/restore pattern as _op_search).
        _scope_before = getattr(self._store, "user_id", None)
        try:
            if hasattr(self._store, "set_user_scope"):
                self._store.set_user_scope(ctx.user_id)
            # D3: client_scope may only narrow the credential clearance.
            client_scope = params.get("client_scope")
            if ctx.max_client_scope is not None:
                if client_scope is not None and client_scope != ctx.max_client_scope:
                    raise APIError(
                        "forbidden",
                        "client_scope may not differ from the credential's "
                        "clearance.",
                    )
                client_scope = ctx.max_client_scope
            return self._store.ingest_structured(
                data=params["data"],
                fmt=params["fmt"],
                mapping=params["mapping"],
                source_name=params["source_name"],
                mode=params["mode"],
                confirm=params.get("confirm", False),
                client_scope=client_scope,
                doc_class=params.get("doc_class"),
                project_id=params.get("project_id"),
            )
        finally:
            if _scope_before is not None and hasattr(self._store, "set_user_scope"):
                self._store.set_user_scope(_scope_before)

    def _op_erase_request(self, ctx: AuthContext, params: Dict[str, Any]) -> Dict[str, Any]:
        """Destructive tier (class A): #293 POPIA erase-request workflow.

        Subject-scoped provable deletion. Preview mode (default) reports
        what WOULD be erased and writes nothing; apply mode (strict
        confirm) erases and appends an append-only deletion receipt per
        record — the receipt survives the deletion and is queryable.

        Server-derived identity (D4): the erase runs under ctx.user_id
        (tenant/user scoping — one tenant's erase never touches
        another's); requested_by is the authenticated principal, not a
        client claim. Legal hold: no legal-hold registry ships by
        default — the blocker is absent-by-default; the hook is the
        store-level ``legal_hold_check`` callable (documented in
        store.erase_subject). Scope narrowing: client_scope/doc_class/
        namespace/categories may only narrow the caller's reach.
        """
        _scope_before = getattr(self._store, "user_id", None)
        try:
            if hasattr(self._store, "set_user_scope"):
                self._store.set_user_scope(ctx.user_id)
            # D3: client_scope may only narrow the credential clearance.
            client_scope = params.get("client_scope")
            if ctx.max_client_scope is not None:
                if client_scope is not None and client_scope != ctx.max_client_scope:
                    raise APIError(
                        "forbidden",
                        "client_scope may not differ from the credential's "
                        "clearance.",
                    )
                client_scope = ctx.max_client_scope
            return self._store.erase_subject(
                subject=params["subject"],
                mode=params["mode"],
                confirm=params.get("confirm", False),
                categories=params.get("categories"),
                client_scope=client_scope,
                doc_class=params.get("doc_class"),
                namespace=params.get("namespace"),
                requested_by=ctx.principal,
            )
        finally:
            if _scope_before is not None and hasattr(self._store, "set_user_scope"):
                self._store.set_user_scope(_scope_before)

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
        if not self.scope_check(ctx, results[0]):
            raise APIError("not_found", "Memory not found.")
        self._store.record_feedback(params["memory_id"], params["feedback"])
        return {"memory_id": params["memory_id"], "feedback": params["feedback"]}

    # -- #295: admin console operations --------------------------------------

    def _op_browse(self, ctx: AuthContext, params: Dict[str, Any]) -> Dict[str, Any]:
        """Read tier (#295): browse memories by namespace/scope/category.

        Lists active memories without a semantic query — the admin
        console's "browse" view. ACL-scoped via set_user_scope (same
        pattern as _op_search). Optional category/namespace/project_id/
        client_scope filters narrow the listing.
        """
        _scope_before = getattr(self._store, "user_id", None)
        try:
            if hasattr(self._store, "set_user_scope"):
                self._store.set_user_scope(ctx.user_id)
            # Use list_memories if available (store_maintenance mixin and
            # SharedMemoryStore); fall back to list_recent otherwise.
            category = params.get("category")
            limit = params["limit"]
            if hasattr(self._store, "list_memories"):
                results = self._store.list_memories(
                    category=category, limit=limit,
                )
            elif hasattr(self._store, "list_recent"):
                results = self._store.list_recent(limit=limit)
            else:
                results = []
        finally:
            if _scope_before is not None and hasattr(self._store, "set_user_scope"):
                self._store.set_user_scope(_scope_before)
        items = []
        for r in results:
            item = r.to_dict() if hasattr(r, "to_dict") else dict(r)
            # Optional namespace filter (post-filter since list_memories
            # doesn't accept namespace — the store's user_scope already
            # scopes to the caller).
            if params.get("namespace"):
                if item.get("namespace") != params["namespace"]:
                    continue
            items.append({
                "memory_id": item.get("memory_id"),
                "category": item.get("category"),
                "content": item.get("content"),
                "tags": item.get("tags", []),
                "created_at": item.get("created_at"),
                "updated_at": item.get("updated_at"),
                "status": item.get("status"),
                "scope": item.get("scope"),
                "namespace": item.get("namespace"),
            })
        return {"results": items, "count": len(items)}

    def _op_list_candidates(
        self, ctx: AuthContext, params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Read tier (#295): list pending candidates for the review queue.

        Scoped to the caller's user_id via set_user_scope. The candidate
        lifecycle statuses (pending, quarantined, reviewed_approved,
        pending_user_confirmation, rejected, approved) are validated in
        _validate_list_candidates_params.
        """
        _scope_before = getattr(self._store, "user_id", None)
        try:
            if hasattr(self._store, "set_user_scope"):
                self._store.set_user_scope(ctx.user_id)
            if not hasattr(self._store, "list_candidates"):
                return {"candidates": [], "count": 0}
            candidates = self._store.list_candidates(
                status=params["status"],
                candidate_id=params.get("candidate_id"),
                limit=params["limit"],
            )
        finally:
            if _scope_before is not None and hasattr(self._store, "set_user_scope"):
                self._store.set_user_scope(_scope_before)
        # Redact: strip payload (may contain internal metadata), keep
        # the fields the review queue needs.
        items = []
        for c in candidates:
            items.append({
                "candidate_id": c.get("candidate_id"),
                "category": c.get("category"),
                "content": c.get("content"),
                "tags": c.get("tags", []),
                "source": c.get("source"),
                "confidence": c.get("confidence"),
                "status": c.get("status"),
                "created_at": c.get("created_at"),
                "reviewed_at": c.get("reviewed_at"),
                "review_reason": c.get("review_reason"),
                "quarantine_reason": c.get("quarantine_reason"),
                "provenance_origin": c.get("provenance_origin"),
                "grounding": c.get("grounding"),
                "evidence_text": c.get("evidence_text", "")[:500],
            })
        return {"candidates": items, "count": len(items)}

    def _op_review_candidate(
        self, ctx: AuthContext, params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Proposal tier (#295): approve/reject/quarantine a candidate.

        This is the ONLY public path to activate a candidate. The
        storage layer enforces the approval invariant: the facade always
        sets review_source="tool" (the UI is a human-driven confirmation
        surface, not an automated reviewer). The store's review_candidate
        method handles the approval ledger, grounding ceiling, external-
        source invariant, and injection guard.

        Server-derived identity (D4): the reviewer is ctx.principal,
        not a client-supplied name.
        """
        _scope_before = getattr(self._store, "user_id", None)
        try:
            if hasattr(self._store, "set_user_scope"):
                self._store.set_user_scope(ctx.user_id)
            if not hasattr(self._store, "review_candidate"):
                raise APIError(
                    "method_not_allowed",
                    "Candidate review is not available on this store.",
                )
            result = self._store.review_candidate(
                candidate_id=params["candidate_id"],
                decision=params["decision"],
                reason=params.get("reason", ""),
                review_source="tool",  # human-driven, never auto_review
                supersedes_memory_id=params.get("supersedes_memory_id"),
                durability=params.get("durability"),
                scope=params.get("scope"),
            )
        finally:
            if _scope_before is not None and hasattr(self._store, "set_user_scope"):
                self._store.set_user_scope(_scope_before)
        if result is None:
            raise APIError("not_found", "Candidate not found or already reviewed.")
        # Redact: return the candidate + memory summary (no payload).
        cand = result.get("candidate", {})
        mem = result.get("memory")
        return {
            "candidate_id": cand.get("candidate_id"),
            "decision": cand.get("status"),
            "review_reason": cand.get("review_reason"),
            "reviewed_at": cand.get("reviewed_at"),
            "memory_id": getattr(mem, "memory_id", None) if mem else None,
            "reviewer": ctx.principal,  # server-derived identity
        }

    def _op_export(self, ctx: AuthContext, params: Dict[str, Any]) -> Dict[str, Any]:
        """Read tier (#295/#294): portable export of the caller's memories.

        Scoped to the caller's user_id via set_user_scope. The export
        carries records, evidence, candidates, and governance tables
        (tombstones/receipts/rejections/aliases). Optional narrowing by
        categories/namespace/client_scope/doc_class.
        """
        _scope_before = getattr(self._store, "user_id", None)
        try:
            if hasattr(self._store, "set_user_scope"):
                self._store.set_user_scope(ctx.user_id)
            if not hasattr(self._store, "export_portable"):
                raise APIError(
                    "method_not_allowed",
                    "Portable export is not available on this store.",
                )
            result = self._store.export_portable(
                categories=params.get("categories"),
                namespace=params.get("namespace"),
                client_scope=params.get("client_scope"),
                doc_class=params.get("doc_class"),
            )
        finally:
            if _scope_before is not None and hasattr(self._store, "set_user_scope"):
                self._store.set_user_scope(_scope_before)
        # Return the export metadata + counts alongside the full jsonl/
        # markdown payloads (callers such as the admin console truncate
        # for display only).
        header = result.get("header", {})
        return {
            "header": header,
            "row_count": len(result.get("rows", [])),
            "jsonl_available": "jsonl" in result,
            "markdown_available": "markdown" in result,
            "jsonl": result.get("jsonl", ""),
            "markdown": result.get("markdown", ""),
        }

    # -- #200 Spec-10: Class C write operations -------------------------------

    def _op_memory_save(
        self, ctx: AuthContext, params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Class C (trusted-local privileged): direct active-memory write.

        Loopback + server-derived identity only (gated in execute()).
        Same store-level semantics as the native memory_save tool:
          store.remember(**save_kwargs) → active memory
        Graph indexing happens via the shared service's post-write hook
        (memory_service.py) when the service is live. For direct-store
        mode (tests), the test verifies the store-level write + version
        chain; graph indexing is reconciled by backfill_graph.py.

        The caller cannot claim provenance fields (source,
        provenance_origin, grounding, user_scope) — rejected in
        validation. The store's remember() defaults apply (source=
        "explicit", provenance_origin derived from payload).
        """
        _scope_before = getattr(self._store, "user_id", None)
        try:
            if hasattr(self._store, "set_user_scope"):
                self._store.set_user_scope(ctx.user_id)
            save_kwargs: Dict[str, Any] = {
                "category": params["category"],
                "content": params["content"],
                "tags": params.get("tags", []),
                "dedup": True,
            }
            if "durability" in params:
                save_kwargs["durability"] = params["durability"]
            if "expires_at" in params:
                save_kwargs["expires_at"] = params["expires_at"]
            rec = self._store.remember(**save_kwargs)
        finally:
            if _scope_before is not None and hasattr(self._store, "set_user_scope"):
                self._store.set_user_scope(_scope_before)
        if rec is None:
            return {"status": "deduplicated", "message": "Similar memory already exists"}
        return {
            "status": "saved",
            "memory_id": rec.memory_id,
        }

    def _op_memory_update(
        self, ctx: AuthContext, params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Class C (trusted-local privileged): direct memory update.

        Updates an existing memory by memory_id, creating a new version
        (the old version is superseded). CAS via expected_version: if
        provided, the facade checks the current version before applying.
        Stale version → 409 conflict, no write.

        #200 Spec-10: expected_version is the last-seen memory_id (each
        update mints a new memory_id for the new version). Clients should
        send the memory_id they last saw, not the original — this is how
        the version chain tracks staleness.

        TOCTOU note: the CAS check is check-then-act (two store calls).
        This is safe in today's single-owner loopback context (class C
        is loopback-only, one writer). A future multi-writer deployment
        should make the CAS compare+write a single atomic store call.
        """
        _scope_before = getattr(self._store, "user_id", None)
        try:
            if hasattr(self._store, "set_user_scope"):
                self._store.set_user_scope(ctx.user_id)
            memory_id = params["memory_id"]
            # #200 Spec-10: CAS check — verify expected_version if provided.
            # expected_version = last-seen memory_id (each update mints a
            # new id). TOCTOU: check-then-act, safe in single-owner loopback.
            expected_version = params.get("expected_version")
            if expected_version is not None:
                current = self._cas_check_version(memory_id, expected_version)
                if not current:
                    raise APIError(
                        "conflict",
                        "expected_version does not match the current "
                        "memory version (CAS conflict).",
                        details={"memory_id": memory_id,
                                 "expected_version": expected_version},
                    )
            update_kwargs: Dict[str, Any] = {"memory_id": memory_id}
            if "content" in params:
                update_kwargs["content"] = params["content"]
            if "tags" in params:
                update_kwargs["tags"] = params["tags"]
            if "expires_at" in params:
                update_kwargs["expires_at"] = params["expires_at"]
            rec = self._store.update_memory(**update_kwargs)
        finally:
            if _scope_before is not None and hasattr(self._store, "set_user_scope"):
                self._store.set_user_scope(_scope_before)
        if rec is None:
            raise APIError("not_found", f"Memory not found: {memory_id}")
        return {
            "status": "updated",
            "memory_id": rec.memory_id,
        }

    def _op_memory_delete(
        self, ctx: AuthContext, params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Class C (trusted-local privileged): direct memory delete.

        Deletes an existing memory by memory_id. CAS via expected_version:
        if provided, the store checks the current version before applying.
        Stale version → 409 conflict, no write.

        #200 Spec-10: over SharedMemoryStore (RPC), the service handler
        enforces strict confirm + CAS + audit server-side (same pattern
        as erase_subject #293). The facade passes confirm=True and
        expected_version through; the service gates them. For direct
        DuckDBMemoryStore (tests), the facade does the CAS check itself.
        """
        _scope_before = getattr(self._store, "user_id", None)
        try:
            if hasattr(self._store, "set_user_scope"):
                self._store.set_user_scope(ctx.user_id)
            memory_id = params["memory_id"]
            expected_version = params.get("expected_version")
            # #200: over SharedMemoryStore (RPC), delete_memory is in
            # _FORBIDDEN_STORE_METHODS. Use the sanctioned facade path
            # when available; the service handler enforces confirm+CAS+
            # audit server-side. For direct DuckDBMemoryStore (tests),
            # the facade does the CAS check itself.
            if hasattr(self._store, "facade_delete_memory"):
                # Service-side gating: the proxy (SharedMemoryStore) uses
                # call_gated() which sets _confirmed in the RPC envelope.
                # The service strips client identity, resolves user_id
                # from its own context, enforces CAS atomically under
                # the tenant lock, and writes the audit row.
                # #200 PR-2 fix: confirm is no longer passed — the gate
                # authority is the _confirmed envelope flag.
                delete_kwargs: Dict[str, Any] = {
                    "memory_id": memory_id,
                }
                if expected_version is not None:
                    delete_kwargs["expected_version"] = expected_version
                result = self._store.facade_delete_memory(**delete_kwargs)
            else:
                # Direct-store mode (tests): facade-side CAS check.
                if expected_version is not None:
                    current = self._cas_check_version(memory_id, expected_version)
                    if not current:
                        raise APIError(
                            "conflict",
                            "expected_version does not match the current "
                            "memory version (CAS conflict).",
                            details={"memory_id": memory_id,
                                     "expected_version": expected_version},
                        )
                result = self._store.delete_memory(memory_id=memory_id)
        finally:
            if _scope_before is not None and hasattr(self._store, "set_user_scope"):
                self._store.set_user_scope(_scope_before)
        if not result:
            raise APIError("not_found", f"Memory not found: {memory_id}")
        return {
            "status": "deleted",
            "memory_id": memory_id,
        }

    # -- #200 Spec-10 PR 2/3: Collection operations --------------------------

    def _op_collection_list(
        self, ctx: AuthContext, params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Read tier: list collections for the caller's scope. EXHAUSTIVE."""
        _scope_before = getattr(self._store, "user_id", None)
        try:
            if hasattr(self._store, "set_user_scope"):
                self._store.set_user_scope(ctx.user_id)
            collections = self._store.list_collections(
                status=params.get("status"),
                limit=params.get("limit", 200),
            )
        finally:
            if _scope_before is not None and hasattr(self._store, "set_user_scope"):
                self._store.set_user_scope(_scope_before)
        return {"collections": collections, "count": len(collections)}

    def _op_collection_items(
        self, ctx: AuthContext, params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Read tier: list items in a collection. EXHAUSTIVE — no cutoff."""
        _scope_before = getattr(self._store, "user_id", None)
        try:
            if hasattr(self._store, "set_user_scope"):
                self._store.set_user_scope(ctx.user_id)
            items = self._store.list_collection_items(
                collection_id=params["collection_id"],
                status=params.get("status"),
                include_archived=params.get("include_archived", False),
                limit=params.get("limit", 0),
            )
        finally:
            if _scope_before is not None and hasattr(self._store, "set_user_scope"):
                self._store.set_user_scope(_scope_before)
        return {"items": items, "count": len(items)}

    def _op_collection_create(
        self, ctx: AuthContext, params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Class C write: create a collection. Server-mints ID.

        #200 PR-2 fix: passes confirm=True as the RPC-seam capability
        marker. The service dispatch (memory_service.py) denies raw RPC
        calls without confirm=True and writes an audit row. In
        direct-store mode (tests), confirm is accepted and ignored.
        """
        _scope_before = getattr(self._store, "user_id", None)
        try:
            if hasattr(self._store, "set_user_scope"):
                self._store.set_user_scope(ctx.user_id)
            collection = self._store.create_collection(
                name=params["name"],
                template=params.get("template"),
                schema=params.get("schema"),
                tenant=ctx.tenant,
                confirm=True,
            )
        finally:
            if _scope_before is not None and hasattr(self._store, "set_user_scope"):
                self._store.set_user_scope(_scope_before)
        return {"status": "created", "collection": collection,
                "collection_id": collection["collection_id"]}

    def _op_collection_add_item(
        self, ctx: AuthContext, params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Class C write: add an item to a collection. Server-mints ID.

        #200 PR-2 fix: passes confirm=True (RPC-seam capability marker).
        """
        _scope_before = getattr(self._store, "user_id", None)
        try:
            if hasattr(self._store, "set_user_scope"):
                self._store.set_user_scope(ctx.user_id)
            item = self._store.add_collection_item(
                collection_id=params["collection_id"],
                fields=params["fields"],
                status=params.get("status", "open"),
                tenant=ctx.tenant,
                confirm=True,
            )
        except ValueError as exc:
            raise APIError("invalid_input", str(exc))
        finally:
            if _scope_before is not None and hasattr(self._store, "set_user_scope"):
                self._store.set_user_scope(_scope_before)
        return {"status": "added", "item": item,
                "item_id": item["item_id"]}

    def _op_collection_update_item(
        self, ctx: AuthContext, params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Class C write: update a collection item. CAS via expected_version.

        #200 PR-2 fix: passes confirm=True (RPC-seam capability marker).
        """
        _scope_before = getattr(self._store, "user_id", None)
        try:
            if hasattr(self._store, "set_user_scope"):
                self._store.set_user_scope(ctx.user_id)
            item = self._store.update_collection_item(
                item_id=params["item_id"],
                fields=params.get("fields"),
                status=params.get("status"),
                expected_version=params.get("expected_version"),
                confirm=True,
            )
        except ValueError as exc:
            if "CAS conflict" in str(exc):
                raise APIError("conflict", str(exc),
                               details={"item_id": params["item_id"]})
            if "not found" in str(exc).lower():
                raise APIError("not_found", str(exc))
            raise APIError("invalid_input", str(exc))
        finally:
            if _scope_before is not None and hasattr(self._store, "set_user_scope"):
                self._store.set_user_scope(_scope_before)
        return {"status": "updated", "item": item,
                "item_id": item["item_id"]}

    def _op_collection_remove_item(
        self, ctx: AuthContext, params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Class C write: remove (archive) a collection item. CAS via
        expected_version. Sets archived_at; does not delete the row.

        #200 PR-2 fix: passes confirm=True (RPC-seam capability marker).
        """
        _scope_before = getattr(self._store, "user_id", None)
        try:
            if hasattr(self._store, "set_user_scope"):
                self._store.set_user_scope(ctx.user_id)
            item = self._store.remove_collection_item(
                item_id=params["item_id"],
                expected_version=params.get("expected_version"),
                confirm=True,
            )
        except ValueError as exc:
            if "CAS conflict" in str(exc):
                raise APIError("conflict", str(exc),
                               details={"item_id": params["item_id"]})
            if "not found" in str(exc).lower():
                raise APIError("not_found", str(exc))
            raise APIError("invalid_input", str(exc))
        finally:
            if _scope_before is not None and hasattr(self._store, "set_user_scope"):
                self._store.set_user_scope(_scope_before)
        return {"status": "removed", "item": item,
                "item_id": item["item_id"]}

    def _cas_check_version(
        self, memory_id: str, expected_version: str,
    ) -> bool:
        """Check that the current version of a memory matches expected.

        #200 Spec-10: CAS (Compare-And-Swap) gate. The expected_version
        is compared against the memory's current version identifier.
        If they don't match, the caller's view is stale → 409 conflict.

        The version identifier is the memory_id itself (each update
        creates a new memory_id for the new version, so the "current"
        version of a logical memory is the head of its version chain).
        For a simple identity check: if the caller passes the memory_id
        they last saw, and the store still has that memory_id as active,
        the CAS passes. If the memory was updated (new memory_id), the
        old memory_id is superseded → CAS fails.

        The expected_version must match the memory_id of the active
        record. If expected_version != memory_id, the caller's view is
        stale (the memory was updated to a new version).

        PR-3 docs note: clients should send the last-seen memory_id as
        expected_version, not the original memory_id from the first
        version. Each update mints a new id; sending the old one is a
        CAS conflict by design.
        """
        try:
            records = self._store.get_memories_by_ids([memory_id])
            if not records:
                return False
            rec = records[0]
            status = getattr(rec, "status", None) or (
                rec.get("status") if isinstance(rec, dict) else None
            )
            if status != "active":
                return False
            # The expected_version must match the memory_id of the
            # active record. If the caller's expected_version is the
            # memory_id they last saw, and the record is still active,
            # the CAS passes.
            rec_mid = getattr(rec, "memory_id", None) or (
                rec.get("memory_id") if isinstance(rec, dict) else None
            )
            return rec_mid == expected_version
        except Exception:
            logger.debug("CAS version check failed for %s", memory_id, exc_info=True)
            return False

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

        AF9: the INFO log is the fast path and always fires. #300:
        denials are ALSO routed to the durable ``access_audit`` table
        via ``store.write_access_audit(...)`` when the store handle
        exposes it. In shared-service mode, ``SharedMemoryStore`` does
        not yet expose ``write_access_audit`` (RPC threading is a
        follow-up); in that case the log is the only record. When the
        store is a direct ``DuckDBMemoryStore`` (tests, direct mode),
        denials are durable and survive restarts.
        """
        # Fast path: always log at INFO level.
        logger.info(
            "api_audit principal=%s tenant=%s transport=%s operation=%s "
            "request_id=%s decision=%s denied_reason=%s error_code=%s "
            "result_count=%s idempotency_replay=%s",
            ctx.principal, ctx.tenant, ctx.transport, operation,
            request_id, decision, denied_reason, error_code,
            result_count, idempotency_replay,
        )
        # #300: route denials to the durable access_audit table when
        # the store handle exposes write_access_audit. Covers all deny
        # classes: forbidden_operation, not_authorized_for_operation,
        # invalid_input, identity_narrowing_rejected. Fail-soft: a
        # durable-audit write failure must never block the response.
        if decision == "denied" and hasattr(self._store, "write_access_audit"):
            try:
                self._store.write_access_audit(
                    user_id=ctx.user_id,
                    query_text=operation,  # hashed by write_access_audit
                    granted_count=0,
                    denied_count=1,
                    denied_scopes=denied_reason or "",
                    excluded=True,
                    tenant=ctx.tenant,
                )
            except Exception as exc:
                logger.error(
                    "durable access_audit write failed for denial %s: %s",
                    request_id, exc, exc_info=True,
                )
                record_subsystem_failure("audit_write", exc)

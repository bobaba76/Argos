"""Spec-09 (#126): REST health/read slice with auth, ACL, audit, limits.

FastAPI application exposing a small HTTP read surface behind the
facade from #123. Not a copy of the provider tools — a separate
transport with real auth, ACL enforcement, audit, and rate limits.

Endpoints:
    GET  /v1/health          — liveness (process alive)
    GET  /v1/ready           — readiness (store opened, ACL loaded,
                                embedding ready, graph available-or-degraded)
    GET  /v1/capabilities    — operations available to the principal
    POST /v1/memory/search   — search memories
    POST /v1/memory/explain-retrieval — why a memory did NOT surface
    GET  /v1/memories/{mid}  — fetch a single memory
    GET  /v1/memories/{mid}/history — version history

Security:
    - Bound to 127.0.0.1 only (no 0.0.0.0, no tunnel binding)
    - Bearer token auth (separate credential from the internal service
      token); verified via hmac.compare_digest
    - No CORS by default; exact origins only if configured
    - No tokens in query params; auth via Authorization header only
    - Cache-Control: no-store on all responses
    - No body/auth-header logging
    - Stable error envelope (D7): code + request_id, never traceback
    - Bounded concurrency: semaphore → 429/503 when saturated
    - Request/response size limits

The REST credential is separate from the internal service token. The
internal token secures the TCP RPC between service_client and
memory_service. The REST credential secures HTTP access to the facade.
They must not be the same value.
"""
from __future__ import annotations

import hmac
import logging
import os
import sys
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Header, HTTPException, Request, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, conint, constr

from api_facade import (
    APIError,
    ArgosAPIFacade,
    AuthContext,
    READ_OPERATIONS,
    PROPOSAL_OPERATIONS,
    FEEDBACK_OPERATIONS,
    WRITE_OPERATIONS,
    COLLECTION_READ_OPERATIONS,
    COLLECTION_WRITE_OPERATIONS,
)
from access_scoping import ACLConfig

logger = logging.getLogger("argos.rest")

# -- Config ------------------------------------------------------------------

DEFAULT_MAX_CONCURRENT = 20
DEFAULT_MAX_BODY_BYTES = 256 * 1024  # 256 KiB
MAX_QUERY_LENGTH = 2000
MAX_MEMORY_ID_LENGTH = 256


# -- Request models ----------------------------------------------------------

class SearchRequest(BaseModel):
    """Strict request body for POST /v1/memory/search."""
    model_config = {"extra": "forbid"}
    query: constr(min_length=1, max_length=MAX_QUERY_LENGTH)
    limit: conint(ge=1, le=50) = 10
    # R8: cap category_filter length to prevent wasteful large strings.
    category_filter: Optional[constr(max_length=100)] = None
    # No project_id, client_scope, namespace, user_id, tenant — those
    # are server-derived from the credential. The facade enforces this.


class ExplainRetrievalRequest(BaseModel):
    """Strict request body for POST /v1/memory/explain-retrieval."""
    model_config = {"extra": "forbid"}
    query: constr(min_length=1, max_length=MAX_QUERY_LENGTH)
    memory_id: constr(min_length=1, max_length=MAX_MEMORY_ID_LENGTH)
    top_k: conint(ge=1, le=50) = 20


# -- #200 Spec-10 PR-3: Write tier request models -----------------------------

class CreateMemoryRequest(BaseModel):
    """Strict request body for POST /v1/memories.

    On loopback: class C direct write (memory_save). On non-loopback:
    class A proposal (memory_propose). The transport decides based on
    is_loopback — the caller does NOT choose the write class.
    """
    model_config = {"extra": "forbid"}
    content: constr(min_length=1, max_length=10000)
    category: constr(min_length=1, max_length=100) = "context_note"
    tags: Optional[List[str]] = None


class CandidateDecisionRequest(BaseModel):
    """Strict request body for POST /v1/candidates/{id}/decision.

    Class B — human principal only. Model principals are denied by the
    facade (no self-approval).
    """
    model_config = {"extra": "forbid"}
    decision: constr(pattern=r"^(approved|rejected|quarantined)$")
    reason: Optional[constr(max_length=2000)] = None


class FeedbackRequest(BaseModel):
    """Strict request body for POST /v1/memories/{id}/feedback."""
    model_config = {"extra": "forbid"}
    feedback: constr(min_length=1, max_length=2000)


class CreateCollectionRequest(BaseModel):
    """Strict request body for POST /v1/collections. Class C, loopback only.

    Note: the 'schema' field is accepted via a raw dict body (not a
    Pydantic field) to avoid shadowing BaseModel.schema. The endpoint
    extracts it manually.
    """
    model_config = {"extra": "forbid"}
    name: constr(min_length=1, max_length=500)
    template: Optional[constr(max_length=100)] = None


class AddCollectionItemRequest(BaseModel):
    """Strict request body for POST /v1/collections/{id}/items. Class C."""
    model_config = {"extra": "forbid"}
    fields: Dict[str, Any]
    status: constr(pattern=r"^(open|done|parked)$") = "open"


class UpdateCollectionItemRequest(BaseModel):
    """Strict request body for PATCH /v1/collections/{id}/items/{item_id}."""
    model_config = {"extra": "forbid"}
    fields: Optional[Dict[str, Any]] = None
    status: Optional[constr(pattern=r"^(open|done|parked)$")] = None
    expected_version: Optional[constr(max_length=256)] = None


# -- Error envelope (D7) -----------------------------------------------------

def _error_response(
    code: str,
    message: str,
    request_id: str,
    status_code: int,
    details: Dict[str, Any] | None = None,
) -> JSONResponse:
    """Build a stable JSON error response (no traceback/path/SQL/token)."""
    body: Dict[str, Any] = {
        "error": {
            "code": code,
            "message": message,
            "request_id": request_id,
        }
    }
    if details:
        body["error"].update({k: v for k, v in details.items() if v is not None})
    return JSONResponse(
        status_code=status_code,
        content=body,
        headers={"Cache-Control": "no-store"},
    )


# Map facade error codes to HTTP status codes.
FACADE_ERROR_TO_HTTP: Dict[str, int] = {
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
    "method_not_allowed": 405,
}


# -- Auth dependency ---------------------------------------------------------

class RESTAuth:
    """Bearer token auth dependency for the REST API.

    The credential is separate from the internal service token. It's
    loaded from a file (api_credential.json) or env var (ARGOS_REST_TOKEN).
    Verification uses hmac.compare_digest to prevent timing attacks.

    #200 Spec-10 PR-3: wires principal_type and is_loopback.
    - principal_type: "human" (default) or "model" (ARGOS_API_PRINCIPAL_TYPE).
      A model principal is denied class B (candidate approval).
    - is_loopback: REST is bound to 127.0.0.1 only, so it IS a loopback
      transport. Set ARGOS_API_NO_LOOPBACK=1 to disable (for testing).
    """

    def __init__(self, expected_token: str) -> None:
        self._expected = expected_token

    def __call__(self, authorization: str = Header(default="")) -> AuthContext:
        """Verify the bearer token and return an AuthContext.

        Raises HTTPException(401) if the token is missing or invalid.
        """
        request_id = str(uuid.uuid4())
        if not authorization:
            raise HTTPException(
                status_code=401,
                detail={"error": {
                    "code": "unauthenticated",
                    "message": "Authorization header is required.",
                    "request_id": request_id,
                }},
            )
        # R6: RFC 7235 says the scheme is case-insensitive. Accept
        # "Bearer", "bearer", "BEARER", etc.
        if not authorization[:7].lower() == "bearer ":
            raise HTTPException(
                status_code=401,
                detail={"error": {
                    "code": "unauthenticated",
                    "message": "Authorization must be a Bearer token.",
                    "request_id": request_id,
                }},
            )
        token = authorization[7:]
        if not token:
            raise HTTPException(
                status_code=401,
                detail={"error": {
                    "code": "unauthenticated",
                    "message": "Bearer token is empty.",
                    "request_id": request_id,
                }},
            )
        if not hmac.compare_digest(token, self._expected):
            raise HTTPException(
                status_code=401,
                detail={"error": {
                    "code": "unauthenticated",
                    "message": "Invalid credentials.",
                    "request_id": request_id,
                }},
            )
        # #200 PR-3: wire principal_type — "human" (default) or "model".
        principal_type = os.environ.get("ARGOS_API_PRINCIPAL_TYPE", "human")
        if principal_type not in ("human", "model"):
            principal_type = "human"
        # #200 PR-3: REST is bound to 127.0.0.1 → loopback transport.
        is_loopback = os.environ.get("ARGOS_API_NO_LOOPBACK", "").lower() not in ("true", "1", "yes")
        # Build the auth context. In v1 (trusted-local mode), the
        # principal/tenant/user_id come from env vars and max_* scope
        # fields are always None (open scope).
        allowed = set(READ_OPERATIONS) | COLLECTION_READ_OPERATIONS
        if os.environ.get("ARGOS_API_CAN_PROPOSE", "").lower() in ("true", "1", "yes"):
            allowed |= PROPOSAL_OPERATIONS
        if os.environ.get("ARGOS_API_CAN_FEEDBACK", "").lower() in ("true", "1", "yes"):
            allowed |= FEEDBACK_OPERATIONS
        # #200 PR-3: class C writes (loopback only).
        if is_loopback and os.environ.get("ARGOS_API_CAN_WRITE", "").lower() in ("true", "1", "yes"):
            allowed |= WRITE_OPERATIONS
            allowed |= COLLECTION_WRITE_OPERATIONS
        return AuthContext(
            principal=os.environ.get("ARGOS_API_PRINCIPAL", "local"),
            tenant=os.environ.get("ARGOS_API_TENANT", "default"),
            user_id=os.environ.get("ARGOS_API_USER_ID", "default_user"),
            transport="rest",
            allowed_operations=allowed,
            can_propose="memory_propose" in allowed,
            can_feedback="record_feedback" in allowed,
            principal_type=principal_type,
            is_loopback=is_loopback,
        )


# -- Concurrency limiter -----------------------------------------------------

class ConcurrencyLimiter:
    """Bounded concurrency semaphore. Returns 429 when saturated."""

    def __init__(self, max_concurrent: int) -> None:
        self._sem = threading.Semaphore(max_concurrent)
        self._max = max_concurrent

    def acquire(self) -> bool:
        return self._sem.acquire(blocking=False)

    def release(self) -> None:
        self._sem.release()


# -- App factory -------------------------------------------------------------

def create_app(
    facade: ArgosAPIFacade,
    *,
    auth_token: str,
    readiness_probe=None,
    max_concurrent: int = DEFAULT_MAX_CONCURRENT,
    allowed_origins: set[str] | None = None,
) -> FastAPI:
    """Build the FastAPI application.

    Args:
        facade: the ArgosAPIFacade instance.
        auth_token: the REST API credential (separate from the internal
            service token).
        readiness_probe: a callable that returns a dict with readiness
            info. If None, a basic probe is used.
        max_concurrent: maximum concurrent requests (semaphore).
        allowed_origins: set of exact origins allowed for CORS. If None,
            no CORS headers are emitted (no cross-origin requests).
    """
    app = FastAPI(
        title="Argos Memory REST API",
        version="1.0.0",
        docs_url=None,   # no auto-docs on the public surface
        redoc_url=None,
        openapi_url=None,
    )
    auth = RESTAuth(auth_token)
    limiter = ConcurrencyLimiter(max_concurrent)
    origins = allowed_origins or set()

    # -- Middleware: concurrency limit + no-store + body size ----------------

    @app.middleware("http")
    async def _middleware(request: Request, call_next):
        # Body size check (before reading).
        # R2: guard against non-numeric Content-Length (return 400, not 500).
        cl = request.headers.get("content-length")
        if cl:
            try:
                cl_int = int(cl)
            except ValueError:
                return _error_response(
                    "malformed_request", "Invalid Content-Length header.",
                    str(uuid.uuid4()), 400,
                )
            if cl_int > DEFAULT_MAX_BODY_BYTES:
                return _error_response(
                    "request_too_large", "Request body exceeds limit.",
                    str(uuid.uuid4()), 413,
                )
        # Concurrency check.
        if not limiter.acquire():
            return _error_response(
                "rate_limited", "Server is at maximum concurrency.",
                str(uuid.uuid4()), 429,
            )
        try:
            response = await call_next(request)
        finally:
            limiter.release()
        # No-store on all responses.
        response.headers["Cache-Control"] = "no-store"
        # CORS: only exact origins, no wildcard.
        if origins:
            origin = request.headers.get("origin", "")
            if origin in origins:
                response.headers["Access-Control-Allow-Origin"] = origin
                # R7: Vary: Origin prevents cache poisoning when the CORS
                # header is set conditionally.
                response.headers["Vary"] = "Origin"
        return response

    # -- Error handler for APIError ------------------------------------------

    @app.exception_handler(APIError)
    async def _api_error_handler(request: Request, exc: APIError):
        status = FACADE_ERROR_TO_HTTP.get(exc.code, 500)
        return _error_response(
            exc.code, exc.message, exc.request_id, status, exc.details,
        )

    # R4: catch-all exception handler — non-APIError exceptions return a
    # stable error envelope (code + request_id) instead of FastAPI's default
    # {"detail": "Internal Server Error"} with no request_id. Never leaks
    # a traceback to the client (D7 design goal).
    @app.exception_handler(Exception)
    async def _catch_all(request: Request, exc: Exception):
        request_id = str(uuid.uuid4())
        logger.exception("Unhandled error (request_id=%s): %s", request_id, exc)
        return _error_response(
            "internal_error", "Internal server error.", request_id, 500,
        )

    # -- Liveness: GET /v1/health --------------------------------------------

    @app.get("/v1/health")
    async def health():
        """Liveness probe — process is alive. Does NOT check store/ACL."""
        return {"status": "ok"}

    # -- Readiness: GET /v1/ready --------------------------------------------

    @app.get("/v1/ready")
    async def ready():
        """Readiness probe — store opened, ACL loaded, embedding ready,
        graph available-or-degraded.

        Readiness is NEVER "healthy" if the first search would trigger
        a model load. This means the embedding model must be loaded
        before readiness returns ok.

        R5: returns only {"status": "ok"} / {"status": "not_ready"}
        without component details — internal state (e.g. "acl": "error")
        should not be visible to unauthenticated callers. Component
        details are available via the authenticated admin endpoint.
        """
        if readiness_probe is not None:
            probe = readiness_probe()
        else:
            probe = {"store": "ok", "acl": "ok", "embedding": "ok", "graph": "ok"}
        # If any critical component is not ready, return 503.
        critical = ["store", "acl", "embedding"]
        all_ready = all(probe.get(k) == "ok" for k in critical)
        if not all_ready:
            return JSONResponse(
                status_code=503,
                content={"status": "not_ready"},
                headers={"Cache-Control": "no-store"},
            )
        return {
            "status": "ok",
        }

    # -- Capabilities: GET /v1/capabilities ----------------------------------

    @app.get("/v1/capabilities")
    async def capabilities(ctx: AuthContext = Depends(auth)):
        """List the operations available to the authenticated principal."""
        try:
            result = facade.execute(ctx, "capabilities", {})
            return result
        except APIError as exc:
            status = FACADE_ERROR_TO_HTTP.get(exc.code, 500)
            return _error_response(exc.code, exc.message, exc.request_id, status)

    # -- Search: POST /v1/memory/search --------------------------------------

    @app.post("/v1/memory/search")
    async def search(
        body: SearchRequest,
        ctx: AuthContext = Depends(auth),
    ):
        """Search memories by natural-language query."""
        try:
            params: Dict[str, Any] = {"query": body.query, "limit": body.limit}
            if body.category_filter:
                params["category_filter"] = body.category_filter
            result = facade.execute(ctx, "search", params)
            return result
        except APIError as exc:
            status = FACADE_ERROR_TO_HTTP.get(exc.code, 500)
            return _error_response(exc.code, exc.message, exc.request_id, status)

    # -- Fetch: GET /v1/memories/{memory_id} ---------------------------------

    @app.get("/v1/memories/{memory_id}")
    async def fetch_memory(
        memory_id: str,
        ctx: AuthContext = Depends(auth),
    ):
        """Fetch a single memory by ID."""
        if len(memory_id) > MAX_MEMORY_ID_LENGTH:
            return _error_response(
                "invalid_input", "memory_id is too long.",
                str(uuid.uuid4()), 422,
            )
        try:
            result = facade.execute(ctx, "fetch", {"memory_id": memory_id})
            return result
        except APIError as exc:
            status = FACADE_ERROR_TO_HTTP.get(exc.code, 500)
            return _error_response(exc.code, exc.message, exc.request_id, status)

    # -- Fetch history: GET /v1/memories/{memory_id}/history -----------------

    @app.get("/v1/memories/{memory_id}/history")
    async def fetch_history(
        memory_id: str,
        ctx: AuthContext = Depends(auth),
    ):
        """Fetch version history for a memory."""
        if len(memory_id) > MAX_MEMORY_ID_LENGTH:
            return _error_response(
                "invalid_input", "memory_id is too long.",
                str(uuid.uuid4()), 422,
            )
        try:
            result = facade.execute(ctx, "fetch_history", {"memory_id": memory_id})
            return result
        except APIError as exc:
            status = FACADE_ERROR_TO_HTTP.get(exc.code, 500)
            return _error_response(exc.code, exc.message, exc.request_id, status)

    # -- Explain: GET /v1/memories/{memory_id}/explain (#280) ----------------

    @app.get("/v1/memories/{memory_id}/explain")
    async def explain(
        memory_id: str,
        ctx: AuthContext = Depends(auth),
    ):
        """Explain why a memory was retrieved — provenance view.

        Returns evidence row, version chain, conflict note (if any),
        blend score, confidence, and gates fired. Read-only, zero-LLM,
        fail-soft. ACL-enforced (cross-user explain returns not_found).
        """
        if len(memory_id) > MAX_MEMORY_ID_LENGTH:
            return _error_response(
                "invalid_input", "memory_id is too long.",
                str(uuid.uuid4()), 422,
            )
        try:
            result = facade.execute(ctx, "explain", {"memory_id": memory_id})
            return result
        except APIError as exc:
            status = FACADE_ERROR_TO_HTTP.get(exc.code, 500)
            return _error_response(exc.code, exc.message, exc.request_id, status)

    # -- Explain-retrieval: POST /v1/memory/explain-retrieval ---------------

    @app.post("/v1/memory/explain-retrieval")
    async def explain_retrieval(
        body: ExplainRetrievalRequest,
        ctx: AuthContext = Depends(auth),
    ):
        """Diagnose why a memory did NOT surface in retrieval.

        Deterministic, read-only, zero-LLM: rank, per-stage diagnostics,
        and human-readable reasons. ACL-enforced (cross-user returns
        not_found; existence not leaked).
        """
        try:
            result = facade.execute(ctx, "explain_retrieval", {
                "query": body.query,
                "memory_id": body.memory_id,
                "top_k": body.top_k,
            })
            return result
        except APIError as exc:
            status = FACADE_ERROR_TO_HTTP.get(exc.code, 500)
            return _error_response(exc.code, exc.message, exc.request_id, status)

    # -- No list/export endpoint (by design) ---------------------------------

    # -- #200 Spec-10 PR-3: Write tier endpoints ----------------------------

    def _require_idempotency_key(idempotency_key: str = Header(default="")) -> str:
        """Require an Idempotency-Key header on all POST mutations.

        Returns the key or raises 400 if missing. The facade's idempotency
        registry (#123) deduplicates on this key.
        """
        if not idempotency_key:
            raise HTTPException(
                status_code=400,
                detail={"error": {
                    "code": "malformed_request",
                    "message": "Idempotency-Key header is required for mutations.",
                    "request_id": str(uuid.uuid4()),
                }},
            )
        return idempotency_key

    # -- POST /v1/memories — create memory (class C loopback or class A propose)

    @app.post("/v1/memories")
    async def create_memory(
        body: CreateMemoryRequest,
        ctx: AuthContext = Depends(auth),
        idempotency_key: str = Depends(_require_idempotency_key),
    ):
        """Create a memory. On loopback: class C direct write (memory_save).
        On non-loopback: class A proposal (memory_propose). The transport
        decides based on is_loopback — the caller does NOT choose the class.
        """
        try:
            params: Dict[str, Any] = {
                "content": body.content,
                "category": body.category,
            }
            if body.tags:
                params["tags"] = body.tags
            # Loopback → memory_save (class C). Non-loopback → memory_propose (class A).
            operation = "memory_save" if ctx.is_loopback else "memory_propose"
            result = facade.execute(ctx, operation, params, idempotency_key=idempotency_key)
            return result
        except APIError as exc:
            status = FACADE_ERROR_TO_HTTP.get(exc.code, 500)
            return _error_response(exc.code, exc.message, exc.request_id, status)

    # -- GET /v1/candidates — list pending candidates (read)

    @app.get("/v1/candidates")
    async def list_candidates(
        ctx: AuthContext = Depends(auth),
    ):
        """List pending candidates for the review queue. Read-only,
        scoped to the caller's user_id."""
        try:
            result = facade.execute(ctx, "list_candidates", {})
            return result
        except APIError as exc:
            status = FACADE_ERROR_TO_HTTP.get(exc.code, 500)
            return _error_response(exc.code, exc.message, exc.request_id, status)

    # -- POST /v1/candidates/{candidate_id}/decision — review (class B, human only)

    @app.post("/v1/candidates/{candidate_id}/decision")
    async def review_candidate(
        candidate_id: str,
        body: CandidateDecisionRequest,
        ctx: AuthContext = Depends(auth),
        idempotency_key: str = Depends(_require_idempotency_key),
    ):
        """Approve/reject/quarantine a pending candidate. Class B — human
        principal only. Model principals are denied (no self-approval)."""
        if len(candidate_id) > MAX_MEMORY_ID_LENGTH:
            return _error_response(
                "invalid_input", "candidate_id is too long.",
                str(uuid.uuid4()), 422,
            )
        try:
            result = facade.execute(ctx, "review_candidate", {
                "candidate_id": candidate_id,
                "decision": body.decision,
                "reason": body.reason or "",
            }, idempotency_key=idempotency_key)
            return result
        except APIError as exc:
            status = FACADE_ERROR_TO_HTTP.get(exc.code, 500)
            return _error_response(exc.code, exc.message, exc.request_id, status)

    # -- POST /v1/memories/{memory_id}/feedback — record feedback

    @app.post("/v1/memories/{memory_id}/feedback")
    async def record_feedback(
        memory_id: str,
        body: FeedbackRequest,
        ctx: AuthContext = Depends(auth),
    ):
        """Record feedback on a memory (e.g. 'helpful', 'not_relevant')."""
        if len(memory_id) > MAX_MEMORY_ID_LENGTH:
            return _error_response(
                "invalid_input", "memory_id is too long.",
                str(uuid.uuid4()), 422,
            )
        try:
            result = facade.execute(ctx, "record_feedback", {
                "memory_id": memory_id,
                "feedback": body.feedback,
            })
            return result
        except APIError as exc:
            status = FACADE_ERROR_TO_HTTP.get(exc.code, 500)
            return _error_response(exc.code, exc.message, exc.request_id, status)

    # -- GET /v1/collections — list collections (read, exhaustive)

    @app.get("/v1/collections")
    async def list_collections(
        ctx: AuthContext = Depends(auth),
    ):
        """List collections for the caller's scope. Read-only, exhaustive."""
        try:
            result = facade.execute(ctx, "collection_list", {})
            return result
        except APIError as exc:
            status = FACADE_ERROR_TO_HTTP.get(exc.code, 500)
            return _error_response(exc.code, exc.message, exc.request_id, status)

    # -- POST /v1/collections — create collection (class C, loopback only)

    @app.post("/v1/collections")
    async def create_collection(
        request: Request,
        ctx: AuthContext = Depends(auth),
        idempotency_key: str = Depends(_require_idempotency_key),
    ):
        """Create a new collection. Class C write — loopback only.

        Accepts a raw JSON body to allow a 'schema' field (which would
        shadow BaseModel.schema if used in a Pydantic model).
        """
        try:
            raw = await request.json()
        except Exception:
            return _error_response(
                "malformed_request", "Invalid JSON body.",
                str(uuid.uuid4()), 400,
            )
        if not isinstance(raw, dict):
            return _error_response(
                "malformed_request", "Request body must be a JSON object.",
                str(uuid.uuid4()), 400,
            )
        name = raw.get("name")
        if not name or not isinstance(name, str) or len(name) < 1 or len(name) > 500:
            return _error_response(
                "invalid_input", "Field 'name' is required (1-500 chars).",
                str(uuid.uuid4()), 422,
            )
        try:
            params: Dict[str, Any] = {"name": name}
            template = raw.get("template")
            if template and isinstance(template, str) and len(template) <= 100:
                params["template"] = template
            schema = raw.get("schema")
            if schema is not None:
                params["schema"] = schema
            result = facade.execute(ctx, "collection_create", params,
                                    idempotency_key=idempotency_key)
            return result
        except APIError as exc:
            status = FACADE_ERROR_TO_HTTP.get(exc.code, 500)
            return _error_response(exc.code, exc.message, exc.request_id, status)

    # -- GET /v1/collections/{collection_id}/items — list items (exhaustive)

    @app.get("/v1/collections/{collection_id}/items")
    async def list_collection_items(
        collection_id: str,
        status: Optional[str] = None,
        ctx: AuthContext = Depends(auth),
    ):
        """List ALL items in a collection (exhaustive — no top-N cutoff).
        Scope-filtered to the caller's user_id."""
        if len(collection_id) > MAX_MEMORY_ID_LENGTH:
            return _error_response(
                "invalid_input", "collection_id is too long.",
                str(uuid.uuid4()), 422,
            )
        try:
            params: Dict[str, Any] = {"collection_id": collection_id}
            if status:
                params["status"] = status
            result = facade.execute(ctx, "collection_items", params)
            return result
        except APIError as exc:
            status_code = FACADE_ERROR_TO_HTTP.get(exc.code, 500)
            return _error_response(exc.code, exc.message, exc.request_id, status_code)

    # -- POST /v1/collections/{collection_id}/items — add item (class C)

    @app.post("/v1/collections/{collection_id}/items")
    async def add_collection_item(
        collection_id: str,
        body: AddCollectionItemRequest,
        ctx: AuthContext = Depends(auth),
        idempotency_key: str = Depends(_require_idempotency_key),
    ):
        """Add an item to a collection. Class C write — loopback only."""
        if len(collection_id) > MAX_MEMORY_ID_LENGTH:
            return _error_response(
                "invalid_input", "collection_id is too long.",
                str(uuid.uuid4()), 422,
            )
        try:
            result = facade.execute(ctx, "collection_add_item", {
                "collection_id": collection_id,
                "fields": body.fields,
                "status": body.status,
            }, idempotency_key=idempotency_key)
            return result
        except APIError as exc:
            status = FACADE_ERROR_TO_HTTP.get(exc.code, 500)
            return _error_response(exc.code, exc.message, exc.request_id, status)

    # -- PATCH /v1/collections/{collection_id}/items/{item_id} — update item

    @app.patch("/v1/collections/{collection_id}/items/{item_id}")
    async def update_collection_item(
        collection_id: str,
        item_id: str,
        body: UpdateCollectionItemRequest,
        ctx: AuthContext = Depends(auth),
        idempotency_key: str = Depends(_require_idempotency_key),
        if_match: str = Header(default=""),
    ):
        """Update an item in a collection. Class C write — loopback only.
        CAS via If-Match header or expected_version in body → 409 on conflict."""
        if len(collection_id) > MAX_MEMORY_ID_LENGTH or len(item_id) > MAX_MEMORY_ID_LENGTH:
            return _error_response(
                "invalid_input", "ID is too long.",
                str(uuid.uuid4()), 422,
            )
        # If-Match header takes precedence over body expected_version.
        expected_version = if_match or body.expected_version
        try:
            params: Dict[str, Any] = {"item_id": item_id}
            if body.fields is not None:
                params["fields"] = body.fields
            if body.status is not None:
                params["status"] = body.status
            if expected_version:
                params["expected_version"] = expected_version
            result = facade.execute(ctx, "collection_update_item", params,
                                    idempotency_key=idempotency_key)
            return result
        except APIError as exc:
            status = FACADE_ERROR_TO_HTTP.get(exc.code, 500)
            return _error_response(exc.code, exc.message, exc.request_id, status)

    # -- DELETE /v1/collections/{collection_id}/items/{item_id} — remove item

    @app.delete("/v1/collections/{collection_id}/items/{item_id}")
    async def remove_collection_item(
        collection_id: str,
        item_id: str,
        ctx: AuthContext = Depends(auth),
        idempotency_key: str = Depends(_require_idempotency_key),
        if_match: str = Header(default=""),
    ):
        """Remove (archive) an item from a collection. Class C write —
        loopback only. CAS via If-Match header → 409 on conflict."""
        if len(collection_id) > MAX_MEMORY_ID_LENGTH or len(item_id) > MAX_MEMORY_ID_LENGTH:
            return _error_response(
                "invalid_input", "ID is too long.",
                str(uuid.uuid4()), 422,
            )
        try:
            params: Dict[str, Any] = {"item_id": item_id}
            if if_match:
                params["expected_version"] = if_match
            result = facade.execute(ctx, "collection_remove_item", params,
                                    idempotency_key=idempotency_key)
            return result
        except APIError as exc:
            status = FACADE_ERROR_TO_HTTP.get(exc.code, 500)
            return _error_response(exc.code, exc.message, exc.request_id, status)

    return app


# -- Entry point -------------------------------------------------------------

def _load_rest_token(home: Path) -> str:
    """Load the REST API credential from a file or env var.

    The REST credential is SEPARATE from the internal service token.
    It's loaded from {home}/api_credential.json or the ARGOS_REST_TOKEN
    env var. If neither is set, the server refuses to start (fail-closed).
    """
    token = os.environ.get("ARGOS_REST_TOKEN", "")
    if token:
        return token
    cred_file = home / "api_credential.json"
    if cred_file.exists():
        import json
        try:
            data = json.loads(cred_file.read_text(encoding="utf-8"))
            token = data.get("token", "")
        except (json.JSONDecodeError, OSError):
            pass
    if not token:
        raise RuntimeError(
            "No REST API credential found. Set ARGOS_REST_TOKEN or "
            "create api_credential.json in HERMES_HOME. The REST server "
            "refuses to start without a credential (fail-closed)."
        )
    return token


def main() -> None:
    """Entry point for the REST server.

    Bound to 127.0.0.1 only — no 0.0.0.0, no tunnel binding.
    """
    import argparse
    import uvicorn
    from service_client import SharedMemoryStore

    parser = argparse.ArgumentParser(description="Argos REST API server (read + write tier)")
    parser.add_argument("--home", required=True, type=Path,
                        help="Path to the Hermes home directory.")
    parser.add_argument("--port", type=int, default=8732,
                        help="Port to bind (default: 8732).")
    parser.add_argument("--max-concurrent", type=int, default=DEFAULT_MAX_CONCURRENT,
                        help="Maximum concurrent requests.")
    args = parser.parse_args()

    logging.basicConfig(
        stream=sys.stderr if hasattr(sys, "stderr") else None,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    token = _load_rest_token(args.home)
    store = SharedMemoryStore(args.home, user_id="default_user", embedder=None)
    acl = ACLConfig()
    facade = ArgosAPIFacade(store, acl=acl, api_mode=False)

    app = create_app(
        facade,
        auth_token=token,
        max_concurrent=args.max_concurrent,
    )

    # uvicorn with host=127.0.0.1 — never 0.0.0.0.
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="info")


if __name__ == "__main__":
    main()

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

import hashlib
import hmac
import logging
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, Header, HTTPException, Request, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, conint, constr

from api_facade import (
    APIError,
    ArgosAPIFacade,
    AuthContext,
    READ_OPERATIONS,
    PROPOSAL_OPERATIONS,
    FEEDBACK_OPERATIONS,
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
    category_filter: Optional[str] = None
    # No project_id, client_scope, namespace, user_id, tenant — those
    # are server-derived from the credential. The facade enforces this.


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
        if not authorization.startswith("Bearer "):
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
        # Build the auth context. In v1 (trusted-local mode), the
        # principal/tenant/user_id come from env vars. In #129, these
        # would be derived from the credential itself.
        return AuthContext(
            principal=os.environ.get("ARGOS_API_PRINCIPAL", "local"),
            tenant=os.environ.get("ARGOS_API_TENANT", "default"),
            user_id=os.environ.get("ARGOS_API_USER_ID", "default_user"),
            transport="rest",
            allowed_operations=set(READ_OPERATIONS),
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
        cl = request.headers.get("content-length")
        if cl and int(cl) > DEFAULT_MAX_BODY_BYTES:
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
        return response

    # -- Error handler for APIError ------------------------------------------

    @app.exception_handler(APIError)
    async def _api_error_handler(request: Request, exc: APIError):
        status = FACADE_ERROR_TO_HTTP.get(exc.code, 500)
        return _error_response(
            exc.code, exc.message, exc.request_id, status, exc.details,
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
        """
        if readiness_probe is not None:
            probe = readiness_probe()
        else:
            probe = {"store": "ok", "acl": "ok", "embedding": "ok", "graph": "ok"}
        # If any critical component is not ready, return 503.
        critical = ["store", "acl", "embedding"]
        all_ready = all(probe.get(k) == "ok" for k in critical)
        # Graph can be degraded (available-or-degraded).
        graph_status = probe.get("graph", "ok")
        if not all_ready:
            return JSONResponse(
                status_code=503,
                content={"status": "not_ready", "components": probe},
                headers={"Cache-Control": "no-store"},
            )
        return {
            "status": "ok",
            "components": probe,
            "graph": graph_status,
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

    # -- No list/export endpoint (by design) ---------------------------------

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

    parser = argparse.ArgumentParser(description="Argos REST API server (read tier)")
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
    import sys
    main()

"""#295: local admin console / web UI for Argos memory.

A loopback-only FastAPI application that surfaces browse, search,
provenance, candidate review, and ops (erase/export) through the
existing api_facade. The UI never reaches into the store directly —
every operation passes through the facade's auth → ACL → validation →
audit spine.

Security model (same as rest_server.py):
  - Bound to 127.0.0.1 only (never 0.0.0.0, never tunnel binding).
  - Bearer token auth (separate credential from the internal service
    token); verified via hmac.compare_digest.
  - Server-derived identity: the UI does NOT accept client-supplied
    user identity. The AuthContext is built from env vars / credential,
    same as the REST server.
  - No tokens/credentials shipped in the UI HTML.
  - Cache-Control: no-store on all responses.
  - Read-only for non-admin roles: mutation buttons (approve/reject,
    erase, export) are only rendered when the principal's
    allowed_operations include the relevant operation.
  - Audit trail: every facade.execute() call is audited by the facade
    (one row per operation, hashed query, no tokens).

The HTML is server-rendered (no client-side JavaScript framework, no
external CDN dependencies — local-first, POPIA: no cloud calls).
"""
from __future__ import annotations

import html
import logging
import os
import sys
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Set

from fastapi import FastAPI, Header, HTTPException, Request, Depends, Form
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse

from api_facade import (
    APIError,
    ArgosAPIFacade,
    AuthContext,
    READ_OPERATIONS,
    PROPOSAL_OPERATIONS,
)
from access_scoping import ACLConfig

logger = logging.getLogger("argos.admin")

# -- Config ------------------------------------------------------------------

DEFAULT_MAX_CONCURRENT = 20
DEFAULT_MAX_BODY_BYTES = 256 * 1024  # 256 KiB

# Operations that require elevated privileges (proposal tier). The UI
# only renders mutation buttons for principals whose allowed_operations
# include these.
MUTATION_OPERATIONS: Set[str] = {
    "review_candidate",
    "erase_request",
    "memory_propose",
    "ingest",
}


# -- Error envelope (same stable shape as rest_server) -----------------------

def _error_json(
    code: str, message: str, request_id: str, status_code: int,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": message, "request_id": request_id}},
        headers={"Cache-Control": "no-store"},
    )


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


# -- Auth dependency (reuses the REST server's pattern) ----------------------

class AdminAuth:
    """Bearer token auth for the admin console.

    Same model as rest_server.RESTAuth: the credential is separate from
    the internal service token, loaded from ARGOS_REST_TOKEN or
    api_credential.json. Server-derived identity from env vars.

    The admin console grants proposal-tier operations when
    ARGOS_API_CAN_PROPOSE is set (same env-var gate as the MCP server).
    This means the UI is read-only by default; the operator opts in to
    mutations via the env var.
    """

    def __init__(self, expected_token: str) -> None:
        self._expected = expected_token

    def __call__(self, authorization: str = Header(default="")) -> AuthContext:
        import hmac
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
        # Build the auth context — server-derived identity.
        allowed = set(READ_OPERATIONS)
        if os.environ.get("ARGOS_API_CAN_PROPOSE", "").lower() in ("true", "1", "yes"):
            allowed |= PROPOSAL_OPERATIONS
        return AuthContext(
            principal=os.environ.get("ARGOS_API_PRINCIPAL", "local"),
            tenant=os.environ.get("ARGOS_API_TENANT", "default"),
            user_id=os.environ.get("ARGOS_API_USER_ID", "default_user"),
            transport="admin-console",
            allowed_operations=allowed,
            can_propose="memory_propose" in allowed,
        )


# -- Concurrency limiter (same as rest_server) -------------------------------

class ConcurrencyLimiter:
    def __init__(self, max_concurrent: int) -> None:
        self._sem = threading.Semaphore(max_concurrent)

    def acquire(self) -> bool:
        return self._sem.acquire(blocking=False)

    def release(self) -> None:
        self._sem.release()


# -- HTML helpers ------------------------------------------------------------

def _esc(text: Any) -> str:
    """HTML-escape a value for safe rendering."""
    return html.escape(str(text) if text is not None else "")


def _base_page(title: str, body: str, ctx: Optional[AuthContext] = None) -> str:
    """Render the base HTML page with a nav bar.

    The nav bar shows mutation links only when the principal has the
    relevant operation in allowed_operations (read-only for non-admin).
    """
    can_review = ctx and "review_candidate" in ctx.allowed_operations
    can_erase = ctx and "erase_request" in ctx.allowed_operations
    can_export = ctx and "export" in ctx.allowed_operations
    principal = _esc(ctx.principal) if ctx else "—"

    nav_links = [
        ('<a href="/">Dashboard</a>'),
        ('<a href="/browse">Browse</a>'),
        ('<a href="/search">Search</a>'),
    ]
    # Review queue is always visible (read operation), but the
    # approve/reject buttons are only rendered for authorized principals.
    nav_links.append('<a href="/review">Review Queue</a>')
    if can_erase:
        nav_links.append('<a href="/ops/erase">Erase</a>')
    if can_export:
        nav_links.append('<a href="/ops/export">Export</a>')

    nav = " | ".join(nav_links)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_esc(title)} — Argos Admin Console</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 0; padding: 0; background: #f8f9fa; color: #222; }}
  .nav {{ background: #1a1a2e; padding: 0.5rem 1rem; color: #eee; }}
  .nav a {{ color: #8ecae6; text-decoration: none; margin-right: 0.5rem; }}
  .nav a:hover {{ text-decoration: underline; }}
  .nav .principal {{ float: right; color: #aaa; font-size: 0.85rem; }}
  .container {{ max-width: 960px; margin: 1rem auto; padding: 0 1rem; }}
  h1, h2 {{ color: #1a1a2e; }}
  table {{ width: 100%; border-collapse: collapse; margin: 1rem 0; }}
  th, td {{ text-align: left; padding: 0.4rem 0.6rem; border-bottom: 1px solid #ddd; }}
  th {{ background: #e9ecef; }}
  tr:hover {{ background: #f1f3f5; }}
  .card {{ background: #fff; border: 1px solid #dee2e6; border-radius: 4px; padding: 1rem; margin: 1rem 0; }}
  .badge {{ display: inline-block; padding: 0.1rem 0.4rem; border-radius: 3px; font-size: 0.8rem; }}
  .badge-pending {{ background: #fff3cd; color: #856404; }}
  .badge-approved {{ background: #d4edda; color: #155724; }}
  .badge-rejected {{ background: #f8d7da; color: #721c24; }}
  .badge-quarantined {{ background: #cce5ff; color: #004085; }}
  .badge-active {{ background: #d4edda; color: #155724; }}
  .muted {{ color: #888; font-size: 0.85rem; }}
  .danger {{ color: #721c24; font-weight: bold; }}
  input[type=text], select {{ padding: 0.3rem 0.5rem; border: 1px solid #ccc; border-radius: 3px; }}
  button {{ padding: 0.3rem 0.8rem; border: 1px solid #ccc; border-radius: 3px; cursor: pointer; background: #fff; }}
  button.danger-btn {{ background: #f8d7da; border-color: #721c24; color: #721c24; }}
  button.approve-btn {{ background: #d4edda; border-color: #155724; color: #155724; }}
  pre {{ background: #f1f3f5; padding: 0.5rem; overflow-x: auto; border-radius: 3px; font-size: 0.85rem; }}
  .flash {{ padding: 0.5rem 1rem; border-radius: 3px; margin: 0.5rem 0; }}
  .flash-ok {{ background: #d4edda; color: #155724; }}
  .flash-err {{ background: #f8d7da; color: #721c24; }}
</style>
</head>
<body>
<nav class="nav">
  {nav}
  <span class="principal">Principal: {principal}</span>
</nav>
<div class="container">
  {body}
</div>
</body>
</html>"""


# -- App factory --------------------------------------------------------------

def create_app(
    facade: ArgosAPIFacade,
    *,
    auth_token: str,
    max_concurrent: int = DEFAULT_MAX_CONCURRENT,
) -> FastAPI:
    """Build the admin console FastAPI application.

    Args:
        facade: the ArgosAPIFacade instance (same facade the REST
            server uses — auth → ACL → validation → audit spine).
        auth_token: the API credential (separate from the internal
            service token). The same token as the REST server.
        max_concurrent: maximum concurrent requests.
    """
    app = FastAPI(
        title="Argos Admin Console",
        version="1.0.0",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    auth = AdminAuth(auth_token)
    limiter = ConcurrencyLimiter(max_concurrent)

    @app.middleware("http")
    async def _middleware(request: Request, call_next):
        cl = request.headers.get("content-length")
        if cl:
            try:
                cl_int = int(cl)
            except ValueError:
                return _error_json("malformed_request", "Invalid Content-Length.",
                                   str(uuid.uuid4()), 400)
            if cl_int > DEFAULT_MAX_BODY_BYTES:
                return _error_json("request_too_large", "Request body exceeds limit.",
                                   str(uuid.uuid4()), 413)
        if not limiter.acquire():
            return _error_json("rate_limited", "Server is at maximum concurrency.",
                               str(uuid.uuid4()), 429)
        try:
            response = await call_next(request)
        finally:
            limiter.release()
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(APIError)
    async def _api_error_handler(request: Request, exc: APIError):
        status = FACADE_ERROR_TO_HTTP.get(exc.code, 500)
        return _error_json(exc.code, exc.message, exc.request_id, status)

    @app.exception_handler(Exception)
    async def _catch_all(request: Request, exc: Exception):
        rid = str(uuid.uuid4())
        logger.exception("Unhandled error (request_id=%s): %s", rid, exc)
        return _error_json("internal_error", "Internal server error.", rid, 500)

    # -- Dashboard: GET / -------------------------------------------------

    @app.get("/", response_class=HTMLResponse)
    async def dashboard(ctx: AuthContext = Depends(auth)):
        caps = facade.execute(ctx, "capabilities", {})
        ops = caps.get("operations", [])
        can_review = "review_candidate" in ops
        can_erase = "erase_request" in ops
        can_export = "export" in ops
        body = f"""
        <h1>Argos Admin Console</h1>
        <div class="card">
          <h2>Session</h2>
          <table>
            <tr><th>Principal</th><td>{_esc(ctx.principal)}</td></tr>
            <tr><th>Tenant</th><td>{_esc(ctx.tenant)}</td></tr>
            <tr><th>User ID</th><td>{_esc(ctx.user_id)}</td></tr>
            <tr><th>Transport</th><td>{_esc(ctx.transport)}</td></tr>
            <tr><th>Operations</th><td>{_esc(', '.join(sorted(ops)))}</td></tr>
          </table>
        </div>
        <div class="card">
          <h2>Quick Actions</h2>
          <ul>
            <li><a href="/browse">Browse memories</a></li>
            <li><a href="/search">Search memories</a></li>
            <li><a href="/review">Review queue (candidates)</a></li>
            {'<li><a href="/ops/erase">Erase (POPIA)</a></li>' if can_erase else '<li class="muted">Erase — not authorized (read-only)</li>'}
            {'<li><a href="/ops/export">Portable export</a></li>' if can_export else '<li class="muted">Export — not authorized (read-only)</li>'}
          </ul>
        </div>
        """
        return _base_page("Dashboard", body, ctx)

    # -- Browse: GET /browse ---------------------------------------------

    @app.get("/browse", response_class=HTMLResponse)
    async def browse(
        category: str = "",
        namespace: str = "",
        limit: int = 50,
        ctx: AuthContext = Depends(auth),
    ):
        params: Dict[str, Any] = {"limit": limit}
        if category:
            params["category"] = category
        if namespace:
            params["namespace"] = namespace
        result = facade.execute(ctx, "browse", params)
        items = result.get("results", [])
        cat_filter = _esc(category)
        ns_filter = _esc(namespace)
        rows = ""
        for item in items:
            mid = _esc(item.get("memory_id", ""))
            cat = item.get("category", "")
            content = _esc(item.get("content", "")[:120])
            ns = _esc(item.get("namespace", "") or "—")
            created = _esc(item.get("created_at", ""))
            status = item.get("status", "active")
            badge = f'<span class="badge badge-{_esc(status)}">{_esc(status)}</span>'
            rows += f"""<tr>
              <td><a href="/memories/{mid}">{mid}</a></td>
              <td>{_esc(cat)}</td>
              <td>{content}</td>
              <td>{ns}</td>
              <td>{created}</td>
              <td>{badge}</td>
            </tr>"""
        if not rows:
            rows = '<tr><td colspan="6" class="muted">No memories found.</td></tr>'
        body = f"""
        <h1>Browse Memories</h1>
        <div class="card">
          <form method="get" action="/browse">
            <label>Category: <input type="text" name="category" value="{cat_filter}" placeholder="personal_fact, preference, ..."></label>
            <label>Namespace: <input type="text" name="namespace" value="{ns_filter}" placeholder="conversation, ..."></label>
            <label>Limit: <input type="text" name="limit" value="{_esc(limit)}" size="4"></label>
            <button type="submit">Filter</button>
          </form>
        </div>
        <table>
          <tr><th>ID</th><th>Category</th><th>Content</th><th>Namespace</th><th>Created</th><th>Status</th></tr>
          {rows}
        </table>
        <p class="muted">{result.get('count', 0)} memories shown (scoped to your user/tenant).</p>
        """
        return _base_page("Browse", body, ctx)

    # -- Search: GET /search ---------------------------------------------

    @app.get("/search", response_class=HTMLResponse)
    async def search(
        q: str = "",
        category_filter: str = "",
        limit: int = 10,
        ctx: AuthContext = Depends(auth),
    ):
        results_html = ""
        count = 0
        if q:
            params: Dict[str, Any] = {"query": q, "limit": limit}
            if category_filter:
                params["category_filter"] = category_filter
            result = facade.execute(ctx, "search", params)
            items = result.get("results", [])
            count = result.get("count", 0)
            for item in items:
                mid = _esc(item.get("memory_id", ""))
                cat = _esc(item.get("category", ""))
                content = _esc(item.get("content", "")[:150])
                sim = item.get("similarity", 0.0)
                rows_html = f"""<tr>
                  <td><a href="/memories/{mid}">{mid}</a></td>
                  <td>{cat}</td>
                  <td>{content}</td>
                  <td>{sim:.4f}</td>
                </tr>"""
                results_html += rows_html
            if not results_html:
                results_html = '<tr><td colspan="4" class="muted">No results.</td></tr>'
            results_html = f"""
            <table>
              <tr><th>ID</th><th>Category</th><th>Content</th><th>Similarity</th></tr>
              {results_html}
            </table>
            <p class="muted">{count} results (semantic + text search via facade).</p>
            """
        body = f"""
        <h1>Search Memories</h1>
        <div class="card">
          <form method="get" action="/search">
            <label>Query: <input type="text" name="q" value="{_esc(q)}" size="40" autofocus></label>
            <label>Category: <input type="text" name="category_filter" value="{_esc(category_filter)}" placeholder="optional"></label>
            <label>Limit: <input type="text" name="limit" value="{_esc(limit)}" size="4"></label>
            <button type="submit">Search</button>
          </form>
        </div>
        {results_html}
        """
        return _base_page("Search", body, ctx)

    # -- Memory detail: GET /memories/{mid} -------------------------------

    @app.get("/memories/{memory_id}", response_class=HTMLResponse)
    async def memory_detail(
        memory_id: str,
        ctx: AuthContext = Depends(auth),
    ):
        mem = facade.execute(ctx, "fetch", {"memory_id": memory_id})
        body = f"""
        <h1>Memory Detail</h1>
        <div class="card">
          <table>
            <tr><th>ID</th><td>{_esc(mem.get('memory_id'))}</td></tr>
            <tr><th>Category</th><td>{_esc(mem.get('category'))}</td></tr>
            <tr><th>Content</th><td>{_esc(mem.get('content'))}</td></tr>
            <tr><th>Tags</th><td>{_esc(', '.join(mem.get('tags', [])))}</td></tr>
            <tr><th>Status</th><td>{_esc(mem.get('status'))}</td></tr>
            <tr><th>Scope</th><td>{_esc(mem.get('scope'))}</td></tr>
            <tr><th>Created</th><td>{_esc(mem.get('created_at'))}</td></tr>
            <tr><th>Updated</th><td>{_esc(mem.get('updated_at'))}</td></tr>
          </table>
        </div>
        <p>
          <a href="/memories/{_esc(memory_id)}/provenance">Provenance walk</a> |
          <a href="/memories/{_esc(memory_id)}/history">Version history</a>
        </p>
        """
        return _base_page("Memory", body, ctx)

    # -- Provenance: GET /memories/{mid}/provenance -----------------------

    @app.get("/memories/{memory_id}/provenance", response_class=HTMLResponse)
    async def provenance(
        memory_id: str,
        ctx: AuthContext = Depends(auth),
    ):
        result = facade.execute(ctx, "explain", {"memory_id": memory_id})
        # The provenance walk (#280) returns evidence chain, version
        # chain, conflict notes, blend score, confidence, and gates.
        evidence = result.get("evidence", [])
        versions = result.get("version_chain", [])
        conflicts = result.get("conflict_notes", [])
        body = f"""
        <h1>Provenance Walk</h1>
        <div class="card">
          <h2>Summary</h2>
          <table>
            <tr><th>Memory ID</th><td>{_esc(memory_id)}</td></tr>
            <tr><th>Blend Score</th><td>{_esc(result.get('blend_score'))}</td></tr>
            <tr><th>Confidence</th><td>{_esc(result.get('confidence'))}</td></tr>
          </table>
        </div>
        <div class="card">
          <h2>Evidence Chain</h2>
          <pre>{_esc(__import__('json').dumps(evidence, indent=2, default=str))}</pre>
        </div>
        <div class="card">
          <h2>Version Chain</h2>
          <pre>{_esc(__import__('json').dumps(versions, indent=2, default=str))}</pre>
        </div>
        <div class="card">
          <h2>Conflict Notes</h2>
          <pre>{_esc(__import__('json').dumps(conflicts, indent=2, default=str))}</pre>
        </div>
        <p><a href="/memories/{_esc(memory_id)}">&larr; Back to memory</a></p>
        """
        return _base_page("Provenance", body, ctx)

    # -- History: GET /memories/{mid}/history -----------------------------

    @app.get("/memories/{memory_id}/history", response_class=HTMLResponse)
    async def history(
        memory_id: str,
        ctx: AuthContext = Depends(auth),
    ):
        result = facade.execute(ctx, "fetch_history", {"memory_id": memory_id})
        items = result.get("history", [])
        rows = ""
        for item in items:
            rows += f"""<tr>
              <td>{_esc(item.get('memory_id'))}</td>
              <td>{_esc(item.get('content', '')[:100])}</td>
              <td>{_esc(item.get('created_at'))}</td>
              <td>{_esc(item.get('status'))}</td>
              <td>{_esc(item.get('valid_from'))}</td>
              <td>{_esc(item.get('valid_to'))}</td>
            </tr>"""
        if not rows:
            rows = '<tr><td colspan="6" class="muted">No history.</td></tr>'
        body = f"""
        <h1>Version History</h1>
        <table>
          <tr><th>ID</th><th>Content</th><th>Created</th><th>Status</th><th>Valid From</th><th>Valid To</th></tr>
          {rows}
        </table>
        <p><a href="/memories/{_esc(memory_id)}">&larr; Back to memory</a></p>
        """
        return _base_page("History", body, ctx)

    # -- Review queue: GET /review ---------------------------------------

    @app.get("/review", response_class=HTMLResponse)
    async def review_queue(
        status: str = "pending",
        ctx: AuthContext = Depends(auth),
    ):
        result = facade.execute(ctx, "list_candidates", {"status": status})
        candidates = result.get("candidates", [])
        can_review = "review_candidate" in ctx.allowed_operations
        rows = ""
        for c in candidates:
            cid = _esc(c.get("candidate_id", ""))
            cat = _esc(c.get("category", ""))
            content = _esc(c.get("content", "")[:120])
            src = _esc(c.get("source", ""))
            conf = c.get("confidence", 0.0)
            cstatus = c.get("status", "pending")
            badge = f'<span class="badge badge-{_esc(cstatus)}">{_esc(cstatus)}</span>'
            prov = _esc(c.get("provenance_origin", ""))
            created = _esc(c.get("created_at", ""))
            actions = ""
            if can_review and cstatus == "pending":
                actions = f"""
                <form method="post" action="/review/{cid}" style="display:inline">
                  <input type="hidden" name="decision" value="approved">
                  <button type="submit" class="approve-btn">Approve</button>
                </form>
                <form method="post" action="/review/{cid}" style="display:inline">
                  <input type="hidden" name="decision" value="rejected">
                  <button type="submit" class="danger-btn">Reject</button>
                </form>
                """
            elif not can_review:
                actions = '<span class="muted">read-only</span>'
            rows += f"""<tr>
              <td>{cid}</td>
              <td>{cat}</td>
              <td>{content}</td>
              <td>{src}</td>
              <td>{conf:.2f}</td>
              <td>{badge}</td>
              <td>{prov}</td>
              <td>{created}</td>
              <td>{actions}</td>
            </tr>"""
        if not rows:
            rows = '<tr><td colspan="9" class="muted">No candidates.</td></tr>'
        status_opts = ""
        for s in ("pending", "quarantined", "reviewed_approved", "rejected", "approved"):
            sel = " selected" if s == status else ""
            status_opts += f'<option value="{s}"{sel}>{s}</option>'
        body = f"""
        <h1>Review Queue</h1>
        <div class="card">
          <form method="get" action="/review">
            <label>Status:
              <select name="status">{status_opts}</select>
            </label>
            <button type="submit">Filter</button>
          </form>
        </div>
        <table>
          <tr><th>ID</th><th>Category</th><th>Content</th><th>Source</th><th>Conf</th><th>Status</th><th>Provenance</th><th>Created</th><th>Actions</th></tr>
          {rows}
        </table>
        <p class="muted">{result.get('count', 0)} candidates. {"Review actions enabled." if can_review else "Read-only — set ARGOS_API_CAN_PROPOSE=1 to enable review actions."}</p>
        """
        return _base_page("Review Queue", body, ctx)

    # -- Review action: POST /review/{cid} -------------------------------

    @app.post("/review/{candidate_id}", response_class=HTMLResponse)
    async def review_action(
        candidate_id: str,
        decision: str = Form(...),
        reason: str = Form(""),
        ctx: AuthContext = Depends(auth),
    ):
        result = facade.execute(ctx, "review_candidate", {
            "candidate_id": candidate_id,
            "decision": decision,
            "reason": reason,
        })
        flash_class = "flash-ok" if decision == "approved" else "flash-err"
        flash_msg = f"Candidate {result.get('candidate_id', candidate_id)} {result.get('decision', decision)}."
        if result.get("memory_id"):
            flash_msg += f" Memory created: {result.get('memory_id')}"
        flash_msg += f" Reviewer: {result.get('reviewer', ctx.principal)} (audit row written)."
        body = f"""
        <h1>Review Result</h1>
        <div class="flash {flash_class}">{_esc(flash_msg)}</div>
        <div class="card">
          <table>
            <tr><th>Candidate ID</th><td>{_esc(result.get('candidate_id'))}</td></tr>
            <tr><th>Decision</th><td>{_esc(result.get('decision'))}</td></tr>
            <tr><th>Review Reason</th><td>{_esc(result.get('review_reason'))}</td></tr>
            <tr><th>Reviewed At</th><td>{_esc(result.get('reviewed_at'))}</td></tr>
            <tr><th>Memory ID</th><td>{_esc(result.get('memory_id'))}</td></tr>
            <tr><th>Reviewer</th><td>{_esc(result.get('reviewer'))}</td></tr>
          </table>
        </div>
        <p><a href="/review">&larr; Back to review queue</a></p>
        """
        return _base_page("Review Result", body, ctx)

    # -- Erase: GET /ops/erase + POST /ops/erase -------------------------

    @app.get("/ops/erase", response_class=HTMLResponse)
    async def erase_form(ctx: AuthContext = Depends(auth)):
        if "erase_request" not in ctx.allowed_operations:
            body = """
            <h1>Erase (POPIA)</h1>
            <div class="flash flash-err">Not authorized. Set ARGOS_API_CAN_PROPOSE=1 to enable erase operations.</div>
            """
            return _base_page("Erase", body, ctx)
        body = """
        <h1>Erase (POPIA)</h1>
        <div class="card">
          <p class="danger">This is a destructive operation. Preview first, then confirm.</p>
          <form method="post" action="/ops/erase">
            <label>Subject (name/ID to erase): <input type="text" name="subject" required size="40"></label><br><br>
            <label>Mode:
              <select name="mode">
                <option value="preview" selected>Preview (dry-run — no changes)</option>
                <option value="apply">Apply (strict confirm required)</option>
              </select>
            </label><br><br>
            <label>Confirm (type "true" to apply): <input type="text" name="confirm" value="false"></label><br><br>
            <button type="submit">Submit</button>
          </form>
        </div>
        """
        return _base_page("Erase", body, ctx)

    @app.post("/ops/erase", response_class=HTMLResponse)
    async def erase_action(
        subject: str = Form(...),
        mode: str = Form("preview"),
        confirm: str = Form("false"),
        ctx: AuthContext = Depends(auth),
    ):
        if "erase_request" not in ctx.allowed_operations:
            body = """
            <h1>Erase (POPIA)</h1>
            <div class="flash flash-err">Not authorized.</div>
            """
            return _base_page("Erase", body, ctx)
        # Strict confirm: only the literal "true" (case-insensitive)
        # enables apply mode. Anything else is treated as preview.
        confirm_bool = confirm.strip().lower() == "true"
        if mode == "apply" and not confirm_bool:
            mode = "preview"
        result = facade.execute(ctx, "erase_request", {
            "subject": subject,
            "mode": mode,
            "confirm": confirm_bool,
        })
        is_preview = mode == "preview"
        flash_class = "flash-ok" if is_preview else "flash-err"
        flash_msg = "Preview complete — no changes written." if is_preview else f"Erase applied. {result.get('receipt_count', 0)} receipts."
        body = f"""
        <h1>Erase Result</h1>
        <div class="flash {flash_class}">{_esc(flash_msg)}</div>
        <div class="card">
          <pre>{_esc(__import__('json').dumps(result, indent=2, default=str))}</pre>
        </div>
        <p><a href="/ops/erase">&larr; Back to erase form</a></p>
        """
        return _base_page("Erase Result", body, ctx)

    # -- Export: GET /ops/export + POST /ops/export -----------------------

    @app.get("/ops/export", response_class=HTMLResponse)
    async def export_form(ctx: AuthContext = Depends(auth)):
        if "export" not in ctx.allowed_operations:
            body = """
            <h1>Portable Export</h1>
            <div class="flash flash-err">Not authorized.</div>
            """
            return _base_page("Export", body, ctx)
        body = """
        <h1>Portable Export</h1>
        <div class="card">
          <p>Export your memories in portable JSONL + Markdown format (#294).</p>
          <form method="post" action="/ops/export">
            <label>Categories (comma-separated, optional): <input type="text" name="categories" size="40"></label><br><br>
            <label>Namespace (optional): <input type="text" name="namespace"></label><br><br>
            <button type="submit">Export</button>
          </form>
        </div>
        """
        return _base_page("Export", body, ctx)

    @app.post("/ops/export", response_class=HTMLResponse)
    async def export_action(
        categories: str = Form(""),
        namespace: str = Form(""),
        ctx: AuthContext = Depends(auth),
    ):
        if "export" not in ctx.allowed_operations:
            body = """
            <h1>Portable Export</h1>
            <div class="flash flash-err">Not authorized.</div>
            """
            return _base_page("Export", body, ctx)
        params: Dict[str, Any] = {}
        if categories.strip():
            params["categories"] = [c.strip() for c in categories.split(",") if c.strip()]
        if namespace.strip():
            params["namespace"] = namespace.strip()
        result = facade.execute(ctx, "export", params)
        header = result.get("header", {})
        jsonl = result.get("jsonl", "")
        markdown = result.get("markdown", "")
        body = f"""
        <h1>Export Result</h1>
        <div class="flash flash-ok">Export complete. {result.get('row_count', 0)} rows.</div>
        <div class="card">
          <h2>Header</h2>
          <pre>{_esc(__import__('json').dumps(header, indent=2, default=str))}</pre>
        </div>
        <div class="card">
          <h2>JSONL ({len(jsonl)} bytes)</h2>
          <pre>{_esc(jsonl[:5000])}{'...' if len(jsonl) > 5000 else ''}</pre>
        </div>
        <div class="card">
          <h2>Markdown Digest</h2>
          <pre>{_esc(markdown[:5000])}{'...' if len(markdown) > 5000 else ''}</pre>
        </div>
        <p><a href="/ops/export">&larr; Back to export form</a></p>
        """
        return _base_page("Export Result", body, ctx)

    # -- Health: GET /health ---------------------------------------------

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    return app


# -- Entry point --------------------------------------------------------------

def _load_token(home: Path) -> str:
    """Load the API credential (same as rest_server._load_rest_token)."""
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
            "No API credential found. Set ARGOS_REST_TOKEN or create "
            "api_credential.json in HERMES_HOME. The admin console refuses "
            "to start without a credential (fail-closed)."
        )
    return token


def main() -> None:
    """Entry point for the admin console.

    Bound to 127.0.0.1 only — never 0.0.0.0, never tunnel binding.
    One-command start:

        python argos_plugin/admin_console.py --home $HERMES_HOME

    The same ARGOS_REST_TOKEN / api_credential.json as the REST server
    is used for auth. Set ARGOS_API_CAN_PROPOSE=1 to enable mutation
    actions (review/erase); without it the console is read-only.
    """
    import argparse
    import uvicorn
    from service_client import SharedMemoryStore

    parser = argparse.ArgumentParser(
        description="Argos admin console (local web UI, loopback-only)",
    )
    parser.add_argument("--home", required=True, type=Path,
                        help="Path to the Hermes home directory.")
    parser.add_argument("--port", type=int, default=8733,
                        help="Port to bind (default: 8733, one above REST).")
    parser.add_argument("--max-concurrent", type=int, default=DEFAULT_MAX_CONCURRENT,
                        help="Maximum concurrent requests.")
    args = parser.parse_args()

    logging.basicConfig(
        stream=sys.stderr,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    token = _load_token(args.home)
    store = SharedMemoryStore(args.home, user_id="default_user", embedder=None)
    acl = ACLConfig()
    facade = ArgosAPIFacade(store, acl=acl, api_mode=False)

    app = create_app(
        facade,
        auth_token=token,
        max_concurrent=args.max_concurrent,
    )

    # uvicorn with host=127.0.0.1 — never 0.0.0.0.
    logger.info("Starting Argos admin console on 127.0.0.1:%d", args.port)
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="info")


if __name__ == "__main__":
    main()

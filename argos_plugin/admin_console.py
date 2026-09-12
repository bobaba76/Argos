"""#295: local admin console / web UI for Argos memory.

A loopback-only FastAPI application that surfaces browse, search,
provenance, candidate review, and ops (erase/export) through the
existing api_facade. The UI never reaches into the store directly —
every operation passes through the facade's auth → ACL → validation →
audit spine.

Security model (same as rest_server.py):
  - Bound to 127.0.0.1 only (never 0.0.0.0, never tunnel binding).
  - Bearer token auth (separate credential from the internal service
    token); verified via hmac.compare_digest. The same token can be
    exchanged for a browser session at /login (#482): an HttpOnly,
    SameSite=Strict session cookie — tokens never appear in URLs or
    HTML. Bearer header auth remains for API clients. The bearer may be the
    legacy transport token (env-derived context - an explicitly-local
    trusted UI) or a per-principal credential from api_credential.json
    (#387): principal, tenant, user_id, principal_type, and operation
    classes come from the credential entry. A human credential
    (principal_type: human) is the recommended identity for review
    actions; model principals are refused class B by the facade.
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
import secrets
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Set

from fastapi import FastAPI, Header, HTTPException, Request, Depends, Form
from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
    PlainTextResponse,
    RedirectResponse,
)

from api_facade import (
    APIError,
    ArgosAPIFacade,
    AuthContext,
    READ_OPERATIONS,
    PROPOSAL_OPERATIONS,
)
from access_scoping import ACLConfig
from api_credentials import (
    CredentialFileError,
    VALID_CLASSES,
    build_context as build_credential_context,
    credential_file_path,
    parse_credentials_file,
    resolve_by_token,
    write_credential,
)

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

# Browser session cookie (#482): sessions live in this process only — a
# console restart means signing in again. SameSite=Strict + HttpOnly keep
# the cookie out of cross-site requests and the reach of page scripts.
SESSION_COOKIE_NAME = "argos_admin_session"
SESSION_TTL_SECONDS = 30 * 24 * 3600  # 30 days


class BrowserLoginRequired(Exception):
    """A browser navigation that needs the sign-in flow (#482/#484).

    ``target`` is the redirect destination: ``/login`` for a missing
    session in a configured console, ``/setup`` in first-run setup
    mode (#484).
    """

    def __init__(self, target: str = "/login") -> None:
        super().__init__(target)
        self.target = target


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

    Same model as rest_server.RESTAuth (#387): the bearer token is
    either the legacy transport token (env-derived context - an
    explicitly-local, human-driven UI) or a per-principal credential
    from {home}/api_credential.json. A human credential
    (principal_type: human) is the recommended identity for review
    actions; model principals are refused class B by the facade.

    The admin console grants proposal-tier operations by default
    (spec-11). ARGOS_API_READ_ONLY=1 makes it read-only (spec-09
    default). ARGOS_API_PRINCIPAL_TYPE=model narrows an env-context
    principal (no review actions); it never upgrades a credential.
    """

    def __init__(self, expected_token: Optional[str], home: Optional[Path] = None) -> None:
        # #484: expected_token=None means "no legacy transport token" —
        # the legacy compare is skipped and auth runs on per-principal
        # credentials alone (first-run setup mode when nothing exists).
        self._expected = expected_token
        self._home = Path(home) if home is not None else None
        self._cred_cache: Optional[tuple] = None
        # #482: browser sessions — session_id -> {"token", "expires"}.
        self._sessions: Dict[str, Dict[str, Any]] = {}
        self._session_lock = threading.Lock()

    def _credentials(self):
        """Load (legacy_token, credentials), cached on mtime+size.

        Raises api_credentials.CredentialFileError on a malformed file
        (fail closed - never fall back to the legacy token path).
        """
        if self._home is None:
            return (None, [])
        path = credential_file_path(self._home)
        try:
            stat = path.stat()
            key = (stat.st_mtime_ns, stat.st_size)
        except OSError:
            self._cred_cache = None
            return (None, [])
        if self._cred_cache is not None and self._cred_cache[0] == key:
            return (self._cred_cache[1], self._cred_cache[2])
        token, creds = parse_credentials_file(path)
        self._cred_cache = (key, token, creds)
        return (token, creds)

    def setup_required(self) -> bool:
        """True when NO credential exists anywhere — first-run setup (#484).

        Setup mode serves only /setup and /health; every other route
        refuses. Recomputed per request (the credential file is cached
        on mtime+size), so the console leaves setup mode the moment the
        bootstrap key is created. A malformed file raises
        CredentialFileError — never treated as "empty" (fail closed).
        """
        if self._expected:
            return False
        legacy_token, creds = self._credentials()
        return not legacy_token and not creds

    def credential_path(self) -> Path:
        """Path of {home}/api_credential.json (setup mode requires home)."""
        if self._home is None:
            raise RuntimeError("admin console setup requires --home")
        return credential_file_path(self._home)

    def _legacy_context(self) -> AuthContext:
        """Env-derived context for the legacy transport token.

        Spec-11 (9/9): propose ON by default (class A).
        ARGOS_API_READ_ONLY=1 restores the spec-09 read-only default.
        The local console is an explicitly-local, human-driven UI; an
        explicit ARGOS_API_PRINCIPAL_TYPE=model narrows it (review
        actions off) -- env never widens anything.
        """
        is_read_only = os.environ.get("ARGOS_API_READ_ONLY", "").lower() in ("true", "1", "yes")
        allowed = set(READ_OPERATIONS)
        if not is_read_only:
            allowed |= PROPOSAL_OPERATIONS
        if os.environ.get("ARGOS_API_CAN_PROPOSE", "").lower() in ("true", "1", "yes"):
            allowed |= PROPOSAL_OPERATIONS
        principal_type = "human"
        if os.environ.get("ARGOS_API_PRINCIPAL_TYPE", "").strip().lower() == "model":
            principal_type = "model"
        return AuthContext(
            principal=os.environ.get("ARGOS_API_PRINCIPAL", "local"),
            tenant=os.environ.get("ARGOS_API_TENANT", "default"),
            user_id=os.environ.get("ARGOS_API_USER_ID", "default_user"),
            transport="admin-console",
            allowed_operations=allowed,
            can_propose="memory_propose" in allowed,
            principal_type=principal_type,
        )

    def __call__(self, request: Request, authorization: str = Header(default="")) -> AuthContext:
        """Authenticate a request: bearer header first, then the browser session."""
        request_id = str(uuid.uuid4())
        # #484: first-run setup mode — no credential exists yet, so no
        # request can be authenticated (there is nothing to match
        # against). Route humans to /setup and refuse everything else.
        if self.setup_required():
            if self._browser_navigation(request):
                raise BrowserLoginRequired(target="/setup")
            raise HTTPException(
                status_code=403,
                detail={"error": {
                    "code": "setup_required",
                    "message": "The admin console is in setup mode. "
                               "Create an admin key at /setup.",
                    "request_id": request_id,
                }},
            )
        if authorization:
            return self._bearer_context(authorization, request_id)
        session_ctx = self._context_from_session(request)
        if session_ctx is not None:
            return session_ctx
        if self._browser_navigation(request):
            # #482: send humans to the login form instead of a raw 401.
            raise BrowserLoginRequired()
        # No header and no session — fall through to the canonical 401.
        return self._bearer_context(authorization, request_id)

    def _bearer_context(self, authorization: str, request_id: str) -> AuthContext:
        """Validate an Authorization header value (pre-#482 behavior, byte-for-byte)."""
        import hmac
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
        if self._expected is not None and hmac.compare_digest(token, self._expected):
            return self._legacy_context()
        # #387/#390: per-principal credential lookup (same as RESTAuth).
        try:
            _, creds = self._credentials()
        except CredentialFileError as exc:
            raise HTTPException(
                status_code=500,
                detail={"error": {
                    "code": "invalid_credential_config",
                    "message": f"Credential file is invalid: {exc}",
                    "request_id": request_id,
                }},
            )
        cred, expired = resolve_by_token(creds, token)
        if cred is not None:
            return build_credential_context(
                cred,
                transport="admin-console",
                is_loopback=True,  # console is loopback-only by construction
                env_principal_type=os.environ.get("ARGOS_API_PRINCIPAL_TYPE", ""),
                is_read_only=os.environ.get("ARGOS_API_READ_ONLY", "").lower() in ("true", "1", "yes"),
            )
        if expired:
            raise HTTPException(
                status_code=401,
                detail={"error": {
                    "code": "unauthenticated",
                    "message": "Credential expired.",
                    "request_id": request_id,
                }},
            )
        raise HTTPException(
            status_code=401,
            detail={"error": {
                "code": "unauthenticated",
                "message": "Invalid credentials.",
                "request_id": request_id,
            }},
        )

    # -- Browser sessions (#482) ----------------------------------------------

    def _browser_navigation(self, request: Request) -> bool:
        """True for an HTML GET navigation — the human browser path."""
        accepts = (request.headers.get("accept") or "").lower()
        return request.method == "GET" and "text/html" in accepts

    def _context_from_session(self, request: Request) -> Optional[AuthContext]:
        """Resolve a session cookie to an AuthContext (None when absent/expired).

        The session's token is re-validated on every request, so credential
        file edits (revocation, expiry, class changes) apply on the next call.
        """
        sid = request.cookies.get(SESSION_COOKIE_NAME)
        if not sid:
            return None
        with self._session_lock:
            entry = self._sessions.get(sid)
        if entry is None:
            return None
        if entry["expires"] <= time.time():
            with self._session_lock:
                self._sessions.pop(sid, None)
            return None
        try:
            return self._bearer_context(f"Bearer {entry['token']}", str(uuid.uuid4()))
        except HTTPException:
            with self._session_lock:
                self._sessions.pop(sid, None)
            return None

    def create_session(self, token: str) -> str:
        """Start a browser session for a validated token; returns the session id."""
        sid = secrets.token_urlsafe(32)
        with self._session_lock:
            self._sessions[sid] = {"token": token, "expires": time.time() + SESSION_TTL_SECONDS}
        return sid

    def destroy_session(self, sid: str) -> None:
        with self._session_lock:
            self._sessions.pop(sid, None)

    def check_token(self, token: str) -> Optional[str]:
        """Validate a raw token for the login form.

        Returns None when valid, else a human-readable message mirroring
        the header path's 401 texts.
        """
        try:
            self._bearer_context(f"Bearer {token}", str(uuid.uuid4()))
            return None
        except HTTPException as exc:
            detail = exc.detail if isinstance(exc.detail, dict) else {}
            return detail.get("error", {}).get("message", "Invalid credentials.")

    def is_authenticated(self, request: Request, authorization: str = "") -> bool:
        """True when the request carries a valid header or a valid session."""
        if authorization:
            try:
                self._bearer_context(authorization, str(uuid.uuid4()))
                return True
            except HTTPException:
                return False
        return self._context_from_session(request) is not None


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
    can_review = ctx and "review_candidate" in ctx.allowed_operations and ctx.principal_type == "human"
    can_erase = ctx and "erase_request" in ctx.allowed_operations
    can_export = ctx and "export" in ctx.allowed_operations
    principal = _esc(ctx.principal) if ctx else "—"
    signout = ' · <a href="/logout">sign out</a>' if ctx else ""

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
  <span class="principal">Principal: {principal}{signout}</span>
</nav>
<div class="container">
  {body}
</div>
</body>
</html>"""


def _login_page(error: str = "") -> str:
    """Sign-in page (#482) — no token material is ever rendered."""
    err = f'<div class="flash flash-err">{_esc(error)}</div>' if error else ""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sign in — Argos Admin Console</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 0; padding: 0; background: #f8f9fa; color: #222; }}
  .nav {{ background: #1a1a2e; padding: 0.5rem 1rem; color: #eee; }}
  .container {{ max-width: 480px; margin: 3rem auto; padding: 0 1rem; }}
  h1 {{ color: #1a1a2e; font-size: 1.3rem; }}
  .card {{ background: #fff; border: 1px solid #dee2e6; border-radius: 4px; padding: 1rem; margin: 1rem 0; }}
  .flash {{ padding: 0.5rem 1rem; border-radius: 3px; margin: 0.5rem 0; }}
  .flash-err {{ background: #f8d7da; color: #721c24; }}
  .muted {{ color: #888; font-size: 0.85rem; }}
  label {{ font-size: 0.9rem; }}
  input[type=password] {{ width: 100%; padding: 0.4rem 0.5rem; border: 1px solid #ccc; border-radius: 3px; box-sizing: border-box; margin: 0.3rem 0 0.6rem; }}
  button {{ padding: 0.4rem 1rem; border: 1px solid #ccc; border-radius: 3px; cursor: pointer; background: #1a1a2e; color: #eee; }}
</style>
</head>
<body>
<nav class="nav">Argos Admin Console</nav>
<div class="container">
  <h1>Sign in</h1>
  {err}
  <div class="card">
    <form method="post" action="/login" autocomplete="off">
      <label for="token">API credential</label>
      <input type="password" id="token" name="token" autofocus autocomplete="off">
      <button type="submit">Sign in</button>
    </form>
    <p class="muted">Paste the token from <code>api_credential.json</code> in your
    Hermes home (or the value of <code>ARGOS_REST_TOKEN</code>). It is exchanged
    for a local session cookie and never appears in URLs or page source.</p>
  </div>
</div>
</body>
</html>"""


def _setup_page(error: str = "") -> str:
    """First-run setup page (#484) — create the bootstrap admin key.

    Rendered only while NO credential exists (setup mode). Same visual
    language as the sign-in page; no token material (none exists yet).
    """
    err = f'<div class="flash flash-err">{_esc(error)}</div>' if error else ""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Setup — Argos Admin Console</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 0; padding: 0; background: #f8f9fa; color: #222; }}
  .nav {{ background: #1a1a2e; padding: 0.5rem 1rem; color: #eee; }}
  .container {{ max-width: 520px; margin: 3rem auto; padding: 0 1rem; }}
  h1 {{ color: #1a1a2e; font-size: 1.3rem; }}
  .card {{ background: #fff; border: 1px solid #dee2e6; border-radius: 4px; padding: 1rem; margin: 1rem 0; }}
  .flash {{ padding: 0.5rem 1rem; border-radius: 3px; margin: 0.5rem 0; }}
  .flash-err {{ background: #f8d7da; color: #721c24; }}
  .muted {{ color: #888; font-size: 0.85rem; }}
  label {{ font-size: 0.9rem; }}
  input[type=text] {{ width: 100%; padding: 0.4rem 0.5rem; border: 1px solid #ccc; border-radius: 3px; box-sizing: border-box; margin: 0.3rem 0 0.6rem; }}
  button {{ padding: 0.4rem 1rem; border: 1px solid #ccc; border-radius: 3px; cursor: pointer; background: #1a1a2e; color: #eee; }}
</style>
</head>
<body>
<nav class="nav">Argos Admin Console</nav>
<div class="container">
  <h1>First-run setup</h1>
  {err}
  <div class="card">
    <p>No credential exists on this machine yet — nothing can sign in.</p>
    <p>Create your <b>admin key</b>: it signs you in here and works as the
    bearer token for the REST API and MCP on this machine.</p>
    <form method="post" action="/setup" autocomplete="off">
      <label for="name">Key name</label>
      <input type="text" id="name" name="name" value="admin" autofocus autocomplete="off">
      <button type="submit">Create my admin key</button>
    </form>
    <p class="muted">The key is stored hashed at rest and shown exactly once,
    right after you create it. This page closes permanently once a key
    exists (re-arm only by deleting the credential file by hand).</p>
  </div>
</div>
</body>
</html>"""


def _setup_key_page(principal: str, token: str) -> str:
    """Show-once page (#484) — the ONLY rendering of the plaintext key.

    Delivered as the POST response to /setup, with the session cookie
    riding along. Never reachable again: /setup redirects as soon as
    the credential exists, and all responses are Cache-Control:
    no-store.
    """
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Your admin key — Argos</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 0; padding: 0; background: #f8f9fa; color: #222; }}
  .nav {{ background: #1a1a2e; padding: 0.5rem 1rem; color: #eee; }}
  .container {{ max-width: 640px; margin: 3rem auto; padding: 0 1rem; }}
  h1 {{ color: #1a1a2e; font-size: 1.3rem; }}
  .card {{ background: #fff; border: 1px solid #dee2e6; border-radius: 4px; padding: 1rem; margin: 1rem 0; }}
  .keybox {{ background: #f1f3f5; border: 1px solid #ced4da; border-radius: 4px; padding: 0.8rem; margin: 0.8rem 0; font-family: ui-monospace, Consolas, monospace; font-size: 0.95rem; word-break: break-all; }}
  .muted {{ color: #888; font-size: 0.85rem; }}
  button {{ padding: 0.4rem 1rem; border: 1px solid #ccc; border-radius: 3px; cursor: pointer; background: #1a1a2e; color: #eee; }}
</style>
</head>
<body>
<nav class="nav">Argos Admin Console</nav>
<div class="container">
  <h1>Admin key created — copy it now</h1>
  <div class="card">
    <p>Key for <b>{_esc(principal)}</b>. <b>This is the only time it will be
    shown</b> — the server keeps only a hash of it.</p>
    <div class="keybox" id="key">{_esc(token)}</div>
    <p>
      <button type="button" onclick="navigator.clipboard.writeText(document.getElementById('key').textContent)">Copy key</button>
      &nbsp;<a href="/">Continue to the console &rarr;</a>
    </p>
    <p class="muted">Store it in your password manager now. You can mint
    additional keys later, but this one cannot be recovered — only
    replaced.</p>
  </div>
</div>
</body>
</html>"""


# -- App factory --------------------------------------------------------------

def create_app(
    facade: ArgosAPIFacade,
    *,
    auth_token: Optional[str],
    max_concurrent: int = DEFAULT_MAX_CONCURRENT,
    home: Optional[Path] = None,
) -> FastAPI:
    """Build the admin console FastAPI application.

    Args:
        facade: the ArgosAPIFacade instance (same facade the REST
            server uses — auth → ACL → validation → audit spine).
        auth_token: the legacy transport token (same as the REST
            server). None means "no legacy token" — auth runs on
            per-principal credentials, and an empty home starts in
            first-run setup mode (#484).
        max_concurrent: maximum concurrent requests.
        home: Hermes home directory (per-principal credentials, #387).
            When omitted, only the legacy transport token is accepted.
    """
    app = FastAPI(
        title="Argos Admin Console",
        version="1.0.0",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    auth = AdminAuth(auth_token, home=home)
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

    @app.exception_handler(BrowserLoginRequired)
    async def _login_required_handler(request: Request, exc: BrowserLoginRequired):
        return RedirectResponse(url=exc.target, status_code=303)

    # -- Browser login (#482): GET/POST /login, GET /logout --------------

    @app.get("/login", response_class=HTMLResponse)
    async def login_form(request: Request, authorization: str = Header(default="")):
        if auth.setup_required():
            # #484: nothing to sign in to yet — create the first key.
            return RedirectResponse(url="/setup", status_code=303)
        if auth.is_authenticated(request, authorization):
            return RedirectResponse(url="/", status_code=303)
        return _login_page()

    @app.post("/login", response_class=HTMLResponse)
    async def login_submit(token: str = Form(default="")):
        if auth.setup_required():
            # #484: no credential exists; the setup page is the door.
            return RedirectResponse(url="/setup", status_code=303)
        token = token.strip()
        if not token:
            return HTMLResponse(_login_page(error="Enter the API credential token."), status_code=401)
        err = auth.check_token(token)
        if err is not None:
            return HTMLResponse(_login_page(error=err), status_code=401)
        sid = auth.create_session(token)
        response = RedirectResponse(url="/", status_code=303)
        response.set_cookie(
            SESSION_COOKIE_NAME,
            sid,
            max_age=SESSION_TTL_SECONDS,
            path="/",
            httponly=True,
            samesite="strict",
        )
        return response

    @app.get("/logout")
    async def logout(request: Request):
        sid = request.cookies.get(SESSION_COOKIE_NAME)
        if sid:
            auth.destroy_session(sid)
        response = RedirectResponse(url="/login", status_code=303)
        response.delete_cookie(SESSION_COOKIE_NAME, path="/")
        return response

    # -- Setup mode (#484): first-run key creation -------------------------

    @app.get("/setup", response_class=HTMLResponse)
    async def setup_form():
        if not auth.setup_required():
            # Self-destructed: a credential exists, so this route is
            # closed. Re-arm only by removing the credential file by
            # hand and restarting.
            return RedirectResponse(url="/login", status_code=303)
        return HTMLResponse(_setup_page())

    @app.post("/setup", response_class=HTMLResponse)
    async def setup_submit(name: str = Form(default="")):
        if not auth.setup_required():
            # Someone refreshed the success page after the key was
            # created — never mint a second bootstrap credential.
            return RedirectResponse(url="/login", status_code=303)
        principal = (name or "").strip() or "admin"
        token, cred = write_credential(
            auth.credential_path(),
            name=principal,
            principal_type="human",
            allowed_classes=sorted(VALID_CLASSES),
            user_id="default_user",
        )
        # Auto sign-in: the session cookie rides on the show-once page,
        # so the dashboard is one click away with no second paste.
        sid = auth.create_session(token)
        response = HTMLResponse(_setup_key_page(cred.name, token))
        response.set_cookie(
            SESSION_COOKIE_NAME,
            sid,
            max_age=SESSION_TTL_SECONDS,
            path="/",
            httponly=True,
            samesite="strict",
        )
        logger.info("setup: created bootstrap credential %r", cred.name)
        return response

    # -- Dashboard: GET / -------------------------------------------------

    @app.get("/", response_class=HTMLResponse)
    async def dashboard(ctx: AuthContext = Depends(auth)):
        caps = facade.execute(ctx, "capabilities", {})
        ops = caps.get("operations", [])
        can_review = "review_candidate" in ops and ctx.principal_type == "human"
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
        can_review = "review_candidate" in ctx.allowed_operations and ctx.principal_type == "human"
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
        <p class="muted">{result.get('count', 0)} candidates. {"Review actions enabled." if can_review else "Read-only — review actions require a human identity (see docs/api)."}</p>
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
            <div class="flash flash-err">Not authorized. Unset ARGOS_API_READ_ONLY (or set it to 0) to enable erase operations.</div>
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

def _load_token(home: Path) -> Optional[str]:
    """Load the legacy transport token; None when only credentials exist.

    #484: the console also starts in first-run SETUP MODE — None with
    no credential file at all means every route except /setup and
    /health refuses until the browser flow creates the bootstrap key.
    A malformed credential file fails loud (never masked as "empty").
    """
    token = os.environ.get("ARGOS_REST_TOKEN", "")
    if token:
        return token
    try:
        legacy_token, _creds = parse_credentials_file(
            credential_file_path(home)
        )
    except CredentialFileError as exc:
        raise RuntimeError(
            f"api_credential.json is invalid: {exc}"
        ) from exc
    return legacy_token


def main() -> None:
    """Entry point for the admin console.

    Bound to 127.0.0.1 only — never 0.0.0.0, never tunnel binding.
    One-command start:

        python argos_plugin/admin_console.py --home $HERMES_HOME

    The same ARGOS_REST_TOKEN / api_credential.json as the REST server
    is used for auth. First run with no credential at all starts in
    SETUP MODE: only /setup and /health are served until the browser
    flow creates the bootstrap admin key (#484). Mutation actions
    (review/erase) are enabled by
    default (spec-11); set ARGOS_API_READ_ONLY=1 to make the console
    read-only. Review actions should use a human credential (#387):
    mint one with scripts/mint_api_credential.py --principal-type
    human --classes read,review; the legacy env path remains for
    explicitly-local trusted UIs.
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
    if token is None:
        try:
            _legacy, _creds = parse_credentials_file(
                credential_file_path(args.home)
            )
        except CredentialFileError:
            _creds = []
        if _creds:
            logger.info(
                "No legacy transport token — authenticating with "
                "per-principal credentials (#387)."
            )
        else:
            logger.info(
                "No credential found — starting in SETUP MODE. Open "
                "http://127.0.0.1:%d in a browser to create the admin key.",
                args.port,
            )
    store = SharedMemoryStore(args.home, user_id="default_user", embedder=None)
    acl = ACLConfig()
    facade = ArgosAPIFacade(store, acl=acl, api_mode=False)

    app = create_app(
        facade,
        auth_token=token,
        max_concurrent=args.max_concurrent,
        home=args.home,
    )

    # uvicorn with host=127.0.0.1 — never 0.0.0.0.
    logger.info("Starting Argos admin console on 127.0.0.1:%d", args.port)
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="info")


if __name__ == "__main__":
    main()

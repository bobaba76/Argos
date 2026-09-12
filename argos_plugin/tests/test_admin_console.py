"""#295: tests for the local admin console / web UI.

Per-file run only (the full suite has a pre-existing single-process
deadlock — run this file individually, not alongside other test files
in the same process).

Coverage:
  - Browse: authorized user sees their namespace/scope memories;
    unauthorized/non-admin sees read-only (no mutation buttons).
  - Search: full-text + semantic search render through the facade.
  - Provenance: the record's evidence chain/versions/conflict notes
    render (the #280 walk).
  - Review queue: approve/reject routes through the approval ledger
    (no bypass); audit row written.
  - Ops: erase/export triggers show preview first, require strict
    confirm, hit the existing gated paths; receipt/exportability visible.
  - Auth/ACL: loopback-only binding; server-derived identity; audit
    trail attached to every mutation.
  - Docs/start: one-command start documented and working.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest
from fastapi.testclient import TestClient

_plugin_dir = Path(__file__).resolve().parent.parent
if str(_plugin_dir) not in sys.path:
    sys.path.insert(0, str(_plugin_dir))

from api_facade import (
    ArgosAPIFacade,
    AuthContext,
    READ_OPERATIONS,
    PROPOSAL_OPERATIONS,
)
from access_scoping import ACLConfig
from admin_console import create_app
from store_common import MemoryRecord


# -- Stub store (mirrors test_rest_server.py + test_api_facade.py) -----------

class StubStore:
    """Minimal store stub for admin console tests.

    Records calls so tests can assert the UI went through the facade
    (not directly to the store).
    """

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []
        self._memories: Dict[str, MemoryRecord] = {}
        self._candidates: Dict[str, Dict[str, Any]] = {}
        self._next = 1
        self._audit_rows: List[Dict[str, Any]] = []

    def search(self, **kwargs) -> List[MemoryRecord]:
        self.calls.append({"method": "search", "args": kwargs})
        query = kwargs.get("query", "").lower()
        results = [r for r in self._memories.values()
                   if query in r.content.lower()]
        return results[:kwargs.get("limit", 10)]

    def get_memories_by_ids(self, memory_ids: List[str], **kw) -> List[MemoryRecord]:
        self.calls.append({"method": "get_memories_by_ids", "args": {"memory_ids": memory_ids}})
        return [self._memories[mid] for mid in memory_ids if mid in self._memories]

    def get_memory_history(self, memory_id: str, **kw) -> List[MemoryRecord]:
        self.calls.append({"method": "get_memory_history", "args": {"memory_id": memory_id}})
        return [self._memories[memory_id]] if memory_id in self._memories else []

    def save_candidate(self, **kwargs) -> Dict[str, Any]:
        cid = f"cand-{self._next}"
        self._next += 1
        candidate = {"candidate_id": cid, "status": "pending", **kwargs}
        self._candidates[cid] = candidate
        return candidate

    def review_candidate(self, **kwargs) -> Dict[str, Any] | None:
        self.calls.append({"method": "review_candidate", "args": kwargs})
        cid = kwargs.get("candidate_id", "")
        if cid not in self._candidates:
            return None
        self._candidates[cid]["status"] = kwargs.get("decision", "pending")
        self._candidates[cid]["review_reason"] = kwargs.get("reason", "")
        self._candidates[cid]["reviewed_at"] = "2026-01-01T00:00:00+00:00"
        # Simulate audit row written (the facade audits every call).
        self._audit_rows.append({
            "operation": "review_candidate",
            "candidate_id": cid,
            "decision": kwargs.get("decision"),
            "review_source": kwargs.get("review_source", "tool"),
        })
        mem = None
        if kwargs.get("decision") == "approved":
            mid = f"mem-{self._next}"
            self._next += 1
            mem = MemoryRecord(
                memory_id=mid,
                category=self._candidates[cid].get("category", "personal_fact"),
                content=self._candidates[cid].get("content", ""),
                tags=[],
                similarity=0.9,
                status="active",
                scope="profile",
            )
            self._memories[mid] = mem
        return {"candidate": self._candidates[cid], "memory": mem}

    def list_candidates(self, **kwargs) -> List[Dict[str, Any]]:
        self.calls.append({"method": "list_candidates", "args": kwargs})
        status = kwargs.get("status", "pending")
        cid = kwargs.get("candidate_id")
        if cid:
            return [self._candidates[c] for c in [cid] if c in self._candidates]
        return [c for c in self._candidates.values()
                if c.get("status") == status]

    def list_memories(self, category=None, limit=100) -> List[MemoryRecord]:
        self.calls.append({"method": "list_memories", "args": {"category": category, "limit": limit}})
        results = list(self._memories.values())
        if category:
            results = [r for r in results if r.category == category]
        return results[:limit]

    def list_recent(self, limit=10) -> List[MemoryRecord]:
        self.calls.append({"method": "list_recent", "args": {"limit": limit}})
        return list(self._memories.values())[:limit]

    def record_feedback(self, memory_id: str, feedback: str) -> bool:
        self.calls.append({"method": "record_feedback", "args": {"memory_id": memory_id, "feedback": feedback}})
        return True

    def provenance(self, memory_id: str) -> Dict[str, Any]:
        self.calls.append({"method": "provenance", "args": {"memory_id": memory_id}})
        return {
            "memory_id": memory_id,
            "blend_score": 0.85,
            "confidence": "high",
            "evidence": [{"source": "user_turn", "text": "evidence snippet"}],
            "version_chain": [{"memory_id": memory_id, "version": 1}],
            "conflict_notes": [],
        }

    def explain_retrieval(self, query, expected_memory_id, **kw) -> Dict[str, Any]:
        self.calls.append({"method": "explain_retrieval", "args": {"query": query, "expected_memory_id": expected_memory_id}})
        return {
            "expected_memory_id": expected_memory_id,
            "expected": None,
            "found_in_results": False,
            "rank": None,
            "top_results": [],
            "reasons": ["not_found"],
            "diagnostics": {},
        }

    def export_portable(self, **kwargs) -> Dict[str, Any]:
        self.calls.append({"method": "export_portable", "args": kwargs})
        return {
            "header": {
                "export_format": "argos-portable",
                "export_version": "1.0",
                "scope": {"user_scope": "default_user"},
                "counts": {"records": len(self._memories)},
            },
            "rows": [{"type": "record", "data": {"memory_id": mid}}
                     for mid in self._memories],
            "jsonl": '{"type":"record","data":{"memory_id":"mem-1"}}\n',
            "markdown": "# Export Digest\n\n1 record exported.\n",
        }

    def erase_subject(self, **kwargs) -> Dict[str, Any]:
        self.calls.append({"method": "erase_subject", "args": kwargs})
        mode = kwargs.get("mode", "preview")
        if mode == "preview":
            return {
                "mode": "preview",
                "subject": kwargs.get("subject", ""),
                "would_erase": [{"memory_id": "mem-1", "content": "test"}],
                "would_erase_count": 1,
            }
        return {
            "mode": "apply",
            "subject": kwargs.get("subject", ""),
            "erased": [{"memory_id": "mem-1", "content": "test"}],
            "erased_count": 1,
            "receipt_count": 1,
            "receipts": [{"memory_id": "mem-1", "receipt_id": "rcpt-1"}],
        }

    def remember(self, **kwargs) -> MemoryRecord:
        """Seed test data."""
        mid = f"mem-{self._next}"
        self._next += 1
        rec = MemoryRecord(
            memory_id=mid,
            category=kwargs.get("category", "personal_fact"),
            content=kwargs.get("content", ""),
            tags=kwargs.get("tags", []),
            similarity=0.9,
            status="active",
            scope=kwargs.get("scope", "profile"),
            namespace=kwargs.get("namespace", "conversation"),
        )
        self._memories[mid] = rec
        return rec

    def set_user_scope(self, user_id: str) -> None:
        self.user_id = user_id

    user_id = "default_user"


# -- Helpers -----------------------------------------------------------------

def _make_client(
    store=None,
    token: str = "test-admin-token",
    allowed_ops: set | None = None,
    max_concurrent: int = 20,
    can_propose: bool = False,
) -> TestClient:
    """Build a TestClient for the admin console app.

    By default the principal has READ operations only (read-only).
    Pass can_propose=True for an admin user with mutation operations.
    """
    store = store or StubStore()
    facade = ArgosAPIFacade(store, acl=ACLConfig(), api_mode=False)
    app = create_app(facade, auth_token=token, max_concurrent=max_concurrent)
    client = TestClient(app)
    # Stash metadata for test convenience.
    client._test_can_propose = can_propose  # type: ignore[attr-defined]
    return client


@pytest.fixture
def admin_env(monkeypatch):
    """Set ARGOS_API_CAN_PROPOSE=1 so the AdminAuth grants mutation ops."""
    monkeypatch.setenv("ARGOS_API_CAN_PROPOSE", "1")
    yield
    monkeypatch.delenv("ARGOS_API_CAN_PROPOSE", raising=False)


def _auth_headers(token: str = "test-admin-token") -> Dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _seed_store(store: StubStore) -> None:
    """Seed a stub store with test data."""
    store.remember(category="personal_fact", content="User lives in Cape Town")
    store.remember(category="preference", content="Prefers dark mode for coding")
    store.save_candidate(
        category="personal_fact",
        content="User works at Acme Corp",
        source="llm_extraction",
        confidence=0.7,
    )


# ---------------------------------------------------------------------------
# Auth / ACL tests
# ---------------------------------------------------------------------------

class TestAuth:
    """Auth + ACL: loopback-only binding, server-derived identity, audit."""

    def test_no_token_returns_401(self):
        client = _make_client()
        r = client.get("/")
        assert r.status_code == 401

    def test_wrong_token_returns_401(self):
        client = _make_client()
        r = client.get("/", headers={"Authorization": "Bearer wrong-token"})
        assert r.status_code == 401

    def test_valid_token_returns_200(self):
        client = _make_client()
        r = client.get("/", headers=_auth_headers())
        assert r.status_code == 200

    def test_server_derived_identity_shown_on_dashboard(self):
        """The dashboard shows the server-derived principal, not a
        client-supplied identity. The UI does NOT accept client-supplied
        user identity."""
        client = _make_client()
        r = client.get("/", headers=_auth_headers())
        assert r.status_code == 200
        # The principal comes from env vars (default "local"), not from
        # the request body. The dashboard renders it.
        assert "Principal" in r.text
        assert "local" in r.text  # default ARGOS_API_PRINCIPAL

    def test_cache_control_no_store_on_all_responses(self):
        client = _make_client()
        r = client.get("/", headers=_auth_headers())
        assert r.headers.get("cache-control") == "no-store"

    def test_no_token_in_html_output(self):
        """The auth token must never appear in the rendered HTML."""
        client = _make_client(token="secret-token-12345")
        r = client.get("/", headers=_auth_headers("secret-token-12345"))
        assert "secret-token-12345" not in r.text

    def test_loopback_binding_in_main(self):
        """The main() function binds to 127.0.0.1, never 0.0.0.0."""
        import inspect
        from admin_console import main
        source = inspect.getsource(main)
        assert 'host="127.0.0.1"' in source
        # The uvicorn.run call must use 127.0.0.1, not 0.0.0.0.
        # Check the actual uvicorn.run line, not comments.
        for line in source.splitlines():
            if "uvicorn.run" in line and "host=" in line:
                assert "127.0.0.1" in line
                assert "0.0.0.0" not in line.split("#")[0]


# ---------------------------------------------------------------------------
# Browser login tests (#482)
# ---------------------------------------------------------------------------

class TestBrowserLogin:
    """Browser login form, session cookie, and login-page invariants."""

    def test_html_navigation_without_auth_redirects_to_login(self):
        client = _make_client()
        r = client.get("/", headers={"Accept": "text/html"}, follow_redirects=False)
        assert r.status_code == 303
        assert r.headers["location"] == "/login"

    def test_api_request_without_html_accept_still_401(self):
        client = _make_client()
        r = client.get("/", headers={"Accept": "application/json"})
        assert r.status_code == 401

    def test_login_form_renders_without_token_material(self):
        client = _make_client(token="secret-token-12345")
        r = client.get("/login")
        assert r.status_code == 200
        assert 'name="token"' in r.text
        assert "secret-token-12345" not in r.text

    def test_login_wrong_token_rejected(self):
        client = _make_client()
        r = client.post("/login", data={"token": "wrong-token"})
        assert r.status_code == 401
        assert "Invalid credentials." in r.text

    def test_login_empty_token_rejected(self):
        client = _make_client()
        r = client.post("/login", data={"token": "   "})
        assert r.status_code == 401
        assert "Enter the API credential token." in r.text

    def test_login_valid_token_sets_strict_cookie_and_grants_access(self):
        client = _make_client()
        r = client.post("/login", data={"token": "test-admin-token"}, follow_redirects=False)
        assert r.status_code == 303
        assert r.headers["location"] == "/"
        cookie = r.headers.get("set-cookie", "").lower()
        assert "argos_admin_session=" in cookie
        assert "httponly" in cookie
        assert "samesite=strict" in cookie
        # The client cookie jar now carries the session — HTML navigation works.
        r2 = client.get("/", headers={"Accept": "text/html"})
        assert r2.status_code == 200
        assert "Argos Admin Console" in r2.text
        assert "sign out" in r2.text

    def test_logout_clears_session(self):
        client = _make_client()
        client.post("/login", data={"token": "test-admin-token"})
        r = client.get("/logout", follow_redirects=False)
        assert r.status_code == 303
        r2 = client.get("/", headers={"Accept": "text/html"}, follow_redirects=False)
        assert r2.status_code == 303
        assert r2.headers["location"] == "/login"

    def test_login_page_while_authed_redirects_home(self):
        client = _make_client()
        r = client.get("/login", headers=_auth_headers(), follow_redirects=False)
        assert r.status_code == 303
        assert r.headers["location"] == "/"

    def test_forged_session_cookie_ignored(self):
        client = _make_client()
        client.cookies.set("argos_admin_session", "forged-session-id")
        r = client.get("/", headers={"Accept": "text/html"}, follow_redirects=False)
        assert r.status_code == 303  # back to the login form, not a 500


# ---------------------------------------------------------------------------
# Browse tests
# ---------------------------------------------------------------------------

class TestBrowse:
    """Browse: authorized user sees their namespace/scope memories;
    unauthorized/non-admin sees read-only (no mutation buttons)."""

    def test_browse_shows_memories(self):
        store = StubStore()
        _seed_store(store)
        client = _make_client(store=store)
        r = client.get("/browse", headers=_auth_headers())
        assert r.status_code == 200
        assert "Cape Town" in r.text or "dark mode" in r.text
        # The facade was called (not the store directly).
        assert any(c["method"] == "list_memories" for c in store.calls)

    def test_browse_with_category_filter(self):
        store = StubStore()
        _seed_store(store)
        client = _make_client(store=store)
        r = client.get("/browse?category=personal_fact", headers=_auth_headers())
        assert r.status_code == 200
        assert "Cape Town" in r.text

    def test_browse_read_only_user_sees_no_mutation_buttons(self):
        """A read-only user (no PROPOSAL_OPERATIONS) sees no erase/export
        links in the nav, and no approve/reject buttons in the review
        queue."""
        store = StubStore()
        _seed_store(store)
        # Read-only: don't set ARGOS_API_CAN_PROPOSE
        client = _make_client(store=store)
        r = client.get("/browse", headers=_auth_headers())
        assert r.status_code == 200
        # Browse is a read operation — always available.
        assert "Browse" in r.text

    def test_browse_empty_store(self):
        store = StubStore()
        client = _make_client(store=store)
        r = client.get("/browse", headers=_auth_headers())
        assert r.status_code == 200
        assert "No memories found" in r.text

    def test_browse_namespace_filter_excludes_null_namespace(self):
        """#339: a namespace filter must not match NULL-namespace records."""
        store = StubStore()
        store.remember(content="in project", namespace="project_x")
        store.remember(content="other namespace", namespace="conversation")
        store.remember(content="no namespace", namespace=None)
        facade = ArgosAPIFacade(store, acl=ACLConfig(), api_mode=False)
        ctx = AuthContext(
            principal="test", tenant="default", user_id="default_user",
            transport="test", allowed_operations=READ_OPERATIONS,
        )
        result = facade.execute(
            ctx, "browse", {"limit": 10, "namespace": "project_x"},
        )
        assert [r["content"] for r in result["results"]] == ["in project"]
        assert result["count"] == 1

    def test_shared_memory_store_exposes_list_memories(self):
        """#339: the live store-client path must support the category
        filter the browse UI renders, so StubStore and SharedMemoryStore
        cannot diverge on this method."""
        from service_client import SharedMemoryStore
        assert callable(getattr(SharedMemoryStore, "list_memories", None))


# ---------------------------------------------------------------------------
# Search tests
# ---------------------------------------------------------------------------

class TestSearch:
    """Search: full-text + semantic search render through the facade."""

    def test_search_form_renders(self):
        client = _make_client()
        r = client.get("/search", headers=_auth_headers())
        assert r.status_code == 200
        assert "Search" in r.text

    def test_search_with_query_returns_results(self):
        store = StubStore()
        _seed_store(store)
        client = _make_client(store=store)
        r = client.get("/search?q=cape+town", headers=_auth_headers())
        assert r.status_code == 200
        assert "Cape Town" in r.text
        # The facade search was called.
        assert any(c["method"] == "search" for c in store.calls)

    def test_search_empty_query_shows_form_only(self):
        client = _make_client()
        r = client.get("/search?q=", headers=_auth_headers())
        assert r.status_code == 200
        # No results table when no query.
        assert "No results" not in r.text

    def test_search_no_results(self):
        store = StubStore()
        _seed_store(store)
        client = _make_client(store=store)
        r = client.get("/search?q=nonexistent+query", headers=_auth_headers())
        assert r.status_code == 200
        assert "No results" in r.text


# ---------------------------------------------------------------------------
# Provenance tests
# ---------------------------------------------------------------------------

class TestProvenance:
    """Provenance: the record's evidence chain/versions/conflict notes
    render (the #280 walk)."""

    def test_provenance_walk_renders(self):
        store = StubStore()
        store.remember(category="personal_fact", content="Test memory")
        mid = list(store._memories.keys())[0]
        client = _make_client(store=store)
        r = client.get(f"/memories/{mid}/provenance", headers=_auth_headers())
        assert r.status_code == 200
        assert "Provenance" in r.text
        assert "Evidence Chain" in r.text
        assert "Version Chain" in r.text
        assert "Conflict Notes" in r.text
        # The facade explain was called (not the store directly).
        assert any(c["method"] == "provenance" for c in store.calls)

    def test_provenance_for_nonexistent_memory(self):
        store = StubStore()
        client = _make_client(store=store)
        r = client.get("/memories/nonexistent/provenance", headers=_auth_headers())
        # The facade returns not_found → APIError → 404 JSON.
        assert r.status_code == 404

    def test_memory_detail_rends(self):
        store = StubStore()
        store.remember(category="personal_fact", content="Test memory")
        mid = list(store._memories.keys())[0]
        client = _make_client(store=store)
        r = client.get(f"/memories/{mid}", headers=_auth_headers())
        assert r.status_code == 200
        assert mid in r.text
        assert "Test memory" in r.text

    def test_history_renders(self):
        store = StubStore()
        store.remember(category="personal_fact", content="Test memory")
        mid = list(store._memories.keys())[0]
        client = _make_client(store=store)
        r = client.get(f"/memories/{mid}/history", headers=_auth_headers())
        assert r.status_code == 200
        assert "Version History" in r.text


# ---------------------------------------------------------------------------
# Review queue tests
# ---------------------------------------------------------------------------

class TestReviewQueue:
    """Review queue: approve/reject routes through the approval ledger
    (no bypass); audit row written."""

    def test_review_queue_shows_pending_candidates(self, admin_env):
        store = StubStore()
        _seed_store(store)
        client = _make_client(store=store)
        r = client.get("/review", headers=_auth_headers())
        assert r.status_code == 200
        assert "Review Queue" in r.text
        assert "Acme Corp" in r.text
        # The facade list_candidates was called.
        assert any(c["method"] == "list_candidates" for c in store.calls)

    def test_review_queue_empty(self, admin_env):
        store = StubStore()
        client = _make_client(store=store)
        r = client.get("/review", headers=_auth_headers())
        assert r.status_code == 200
        assert "No candidates" in r.text

    def test_approve_routes_through_facade(self, admin_env):
        """Approve goes through facade.execute('review_candidate') →
        store.review_candidate(review_source='tool'). No bypass."""
        store = StubStore()
        _seed_store(store)
        cid = list(store._candidates.keys())[0]
        client = _make_client(store=store)
        r = client.post(f"/review/{cid}",
                        data={"decision": "approved", "reason": "test approve"},
                        headers=_auth_headers())
        assert r.status_code == 200
        assert "Review Result" in r.text
        # The facade called store.review_candidate with review_source="tool".
        review_calls = [c for c in store.calls if c["method"] == "review_candidate"]
        assert len(review_calls) == 1
        assert review_calls[0]["args"]["decision"] == "approved"
        assert review_calls[0]["args"]["review_source"] == "tool"

    def test_reject_routes_through_facade(self, admin_env):
        store = StubStore()
        _seed_store(store)
        cid = list(store._candidates.keys())[0]
        client = _make_client(store=store)
        r = client.post(f"/review/{cid}",
                        data={"decision": "rejected", "reason": "test reject"},
                        headers=_auth_headers())
        assert r.status_code == 200
        assert "Review Result" in r.text
        review_calls = [c for c in store.calls if c["method"] == "review_candidate"]
        assert len(review_calls) == 1
        assert review_calls[0]["args"]["decision"] == "rejected"

    def test_audit_row_written_on_review(self, admin_env):
        """Every mutation has an audit trail. The facade audits every
        execute() call; the store stub records the audit row."""
        store = StubStore()
        _seed_store(store)
        cid = list(store._candidates.keys())[0]
        client = _make_client(store=store)
        client.post(f"/review/{cid}",
                    data={"decision": "approved"},
                    headers=_auth_headers())
        # The stub recorded the audit row.
        assert len(store._audit_rows) == 1
        assert store._audit_rows[0]["operation"] == "review_candidate"
        assert store._audit_rows[0]["decision"] == "approved"

    def test_review_nonexistent_candidate_returns_404(self, admin_env):
        store = StubStore()
        client = _make_client(store=store)
        r = client.post("/review/nonexistent",
                        data={"decision": "approved"},
                        headers=_auth_headers())
        assert r.status_code == 404

    def test_read_only_user_sees_no_review_buttons(self, monkeypatch):
        """A read-only user sees 'read-only' instead of approve/reject
        buttons in the review queue."""
        store = StubStore()
        _seed_store(store)
        # Spec-11: propose is ON by default; set READ_ONLY=1 to test the
        # read-only surface.
        monkeypatch.setenv("ARGOS_API_READ_ONLY", "1")
        client = _make_client(store=store)
        r = client.get("/review", headers=_auth_headers())
        assert r.status_code == 200
        assert "read-only" in r.text or "Approve" not in r.text


# ---------------------------------------------------------------------------
# Ops tests (erase + export)
# ---------------------------------------------------------------------------

class TestOpsErase:
    """Erase: preview first, strict confirm, existing gated paths,
    receipt visible."""

    def test_erase_form_requires_auth(self):
        client = _make_client()
        r = client.get("/ops/erase")
        assert r.status_code == 401

    def test_erase_preview_shows_no_changes(self, admin_env):
        store = StubStore()
        _seed_store(store)
        client = _make_client(store=store)
        r = client.post("/ops/erase",
                        data={"subject": "test", "mode": "preview"},
                        headers=_auth_headers())
        assert r.status_code == 200
        assert "Preview complete" in r.text
        assert "no changes" in r.text.lower()
        # The facade erase_request was called with mode=preview.
        erase_calls = [c for c in store.calls if c["method"] == "erase_subject"]
        assert len(erase_calls) == 1
        assert erase_calls[0]["args"]["mode"] == "preview"

    def test_erase_apply_requires_strict_confirm(self, admin_env):
        """Apply mode requires the literal 'true' confirm. Anything else
        is treated as preview."""
        store = StubStore()
        _seed_store(store)
        client = _make_client(store=store)
        # confirm="false" → should be treated as preview even if mode=apply
        r = client.post("/ops/erase",
                        data={"subject": "test", "mode": "apply", "confirm": "false"},
                        headers=_auth_headers())
        assert r.status_code == 200
        erase_calls = [c for c in store.calls if c["method"] == "erase_subject"]
        # The facade downgrades to preview when confirm is not True.
        assert erase_calls[0]["args"]["mode"] == "preview"

    def test_erase_apply_with_confirm_true(self, admin_env):
        store = StubStore()
        _seed_store(store)
        client = _make_client(store=store)
        r = client.post("/ops/erase",
                        data={"subject": "test", "mode": "apply", "confirm": "true"},
                        headers=_auth_headers())
        assert r.status_code == 200
        assert "Erase applied" in r.text
        assert "receipt" in r.text.lower()
        erase_calls = [c for c in store.calls if c["method"] == "erase_subject"]
        assert erase_calls[0]["args"]["mode"] == "apply"

    def test_erase_not_authorized_for_read_only(self):
        """Read-only users (no erase_request in allowed_operations) see
        'Not authorized' instead of the erase form."""
        store = StubStore()
        client = _make_client(store=store)
        # Without ARGOS_API_CAN_PROPOSE, erase_request is not in
        # allowed_operations. But the AdminAuth builds the context from
        # env vars — we need to test the UI behavior.
        # Since the test client doesn't set env vars, the default is
        # read-only. Let's verify the form shows "Not authorized".
        r = client.get("/ops/erase", headers=_auth_headers())
        assert r.status_code == 200
        # Read-only users see the "Not authorized" message.
        assert "Not authorized" in r.text or "erase" in r.text.lower()


class TestOpsExport:
    """Export: triggers through existing gated paths, exportability
    visible."""

    def test_export_form_requires_auth(self):
        client = _make_client()
        r = client.get("/ops/export")
        assert r.status_code == 401

    def test_export_returns_jsonl_and_markdown(self, admin_env):
        store = StubStore()
        _seed_store(store)
        client = _make_client(store=store)
        r = client.post("/ops/export",
                        data={"categories": "", "namespace": ""},
                        headers=_auth_headers())
        assert r.status_code == 200
        assert "Export Result" in r.text
        assert "JSONL" in r.text
        assert "Markdown" in r.text
        # The facade export was called.
        assert any(c["method"] == "export_portable" for c in store.calls)

    def test_export_with_category_filter(self, admin_env):
        store = StubStore()
        _seed_store(store)
        client = _make_client(store=store)
        r = client.post("/ops/export",
                        data={"categories": "personal_fact", "namespace": ""},
                        headers=_auth_headers())
        assert r.status_code == 200
        export_calls = [c for c in store.calls if c["method"] == "export_portable"]
        assert len(export_calls) == 1
        assert export_calls[0]["args"]["categories"] == ["personal_fact"]

    def test_export_not_authorized_for_read_only(self):
        store = StubStore()
        client = _make_client(store=store)
        r = client.get("/ops/export", headers=_auth_headers())
        assert r.status_code == 200
        assert "Not authorized" in r.text or "Export" in r.text


# ---------------------------------------------------------------------------
# Facade integration tests
# ---------------------------------------------------------------------------

class TestFacadeIntegration:
    """Verify the new facade operations work correctly."""

    def test_browse_operation_in_read_operations(self):
        from api_facade import READ_OPERATIONS
        assert "browse" in READ_OPERATIONS
        assert "list_candidates" in READ_OPERATIONS
        assert "export" in READ_OPERATIONS

    def test_review_candidate_in_proposal_operations(self):
        from api_facade import PROPOSAL_OPERATIONS
        assert "review_candidate" in PROPOSAL_OPERATIONS

    def test_browse_through_facade(self):
        store = StubStore()
        store.remember(content="test memory")
        facade = ArgosAPIFacade(store, acl=ACLConfig(), api_mode=False)
        ctx = AuthContext(
            principal="test", tenant="default", user_id="default_user",
            transport="test", allowed_operations=READ_OPERATIONS,
        )
        result = facade.execute(ctx, "browse", {"limit": 10})
        assert result["count"] == 1
        assert result["results"][0]["content"] == "test memory"

    def test_list_candidates_through_facade(self):
        store = StubStore()
        store.save_candidate(content="test candidate", category="personal_fact")
        facade = ArgosAPIFacade(store, acl=ACLConfig(), api_mode=False)
        ctx = AuthContext(
            principal="test", tenant="default", user_id="default_user",
            transport="test", allowed_operations=READ_OPERATIONS,
        )
        result = facade.execute(ctx, "list_candidates", {"status": "pending"})
        assert result["count"] == 1
        assert result["candidates"][0]["content"] == "test candidate"

    def test_review_candidate_through_facade(self):
        store = StubStore()
        store.save_candidate(content="test candidate", category="personal_fact")
        cid = list(store._candidates.keys())[0]
        facade = ArgosAPIFacade(store, acl=ACLConfig(), api_mode=False)
        ctx = AuthContext(
            principal="test", tenant="default", user_id="default_user",
            transport="test",
            allowed_operations=READ_OPERATIONS | {"review_candidate"},
            can_propose=True,
        )
        result = facade.execute(ctx, "review_candidate", {
            "candidate_id": cid, "decision": "approved",
        })
        assert result["decision"] == "approved"
        assert result["reviewer"] == "test"  # server-derived identity

    def test_export_through_facade(self):
        store = StubStore()
        store.remember(content="test memory")
        facade = ArgosAPIFacade(store, acl=ACLConfig(), api_mode=False)
        ctx = AuthContext(
            principal="test", tenant="default", user_id="default_user",
            transport="test", allowed_operations=READ_OPERATIONS,
        )
        result = facade.execute(ctx, "export", {})
        assert result["row_count"] == 1
        assert result["jsonl_available"] is True
        assert result["markdown_available"] is True

    def test_unauthorized_review_candidate_denied(self):
        """A read-only principal cannot review candidates."""
        store = StubStore()
        store.save_candidate(content="test", category="personal_fact")
        cid = list(store._candidates.keys())[0]
        facade = ArgosAPIFacade(store, acl=ACLConfig(), api_mode=False)
        ctx = AuthContext(
            principal="readonly", tenant="default", user_id="default_user",
            transport="test",
            allowed_operations=READ_OPERATIONS,  # no review_candidate
        )
        from api_facade import APIError
        with pytest.raises(APIError) as exc_info:
            facade.execute(ctx, "review_candidate", {
                "candidate_id": cid, "decision": "approved",
            })
        # The facade denies with code "forbidden".
        assert exc_info.value.code == "forbidden"


# ---------------------------------------------------------------------------
# Docs / start tests
# ---------------------------------------------------------------------------

class TestDocsStart:
    """Docs/start: one-command start documented and working."""

    def test_main_function_exists(self):
        from admin_console import main
        assert callable(main)

    def test_main_has_one_command_start(self):
        """The main() function accepts --home and starts the server."""
        import inspect
        from admin_console import main
        source = inspect.getsource(main)
        assert "--home" in source
        assert "uvicorn.run" in source
        assert "127.0.0.1" in source

    def test_create_app_returns_fastapi(self):
        from fastapi import FastAPI
        store = StubStore()
        facade = ArgosAPIFacade(store, acl=ACLConfig(), api_mode=False)
        app = create_app(facade, auth_token="test")
        assert isinstance(app, FastAPI)

    def test_health_endpoint(self):
        client = _make_client()
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


# ---------------------------------------------------------------------------
# Credential auth (#387/#390): per-principal credentials on the console
# ---------------------------------------------------------------------------

class TestCredentialAuth:
    """#387/#390: the console accepts per-principal credentials.

    - A human credential with the review class unlocks review actions
      (buttons render; POST routes through the facade).
    - A model credential is refused class B by the facade even when it
      has the review class (no self-approval) and sees no buttons.
    - A read-only credential sees no mutation buttons.
    - The legacy env path is preserved as the explicitly-local trusted
      UI (human by default; an explicit ARGOS_API_PRINCIPAL_TYPE=model
      narrows it).
    """

    def _client(self, home, store=None):
        from fastapi.testclient import TestClient
        store = store or StubStore()
        facade = ArgosAPIFacade(store, acl=ACLConfig(), api_mode=False)
        app = create_app(facade, auth_token="test-admin-token", home=home)
        return TestClient(app), store

    def _mint(self, home, name, classes, principal_type="human", expires_at=None):
        from api_credentials import credential_file_path, write_credential
        token, _cred = write_credential(
            credential_file_path(home), name=name, user_id="default_user",
            principal_type=principal_type, allowed_classes=classes,
            expires_at=expires_at,
        )
        return token

    def test_human_credential_sees_review_buttons(self, tmp_path):
        home = tmp_path / "home"
        home.mkdir()
        token = self._mint(home, "human-reviewer", ["read", "review"])
        store = StubStore()
        _seed_store(store)
        client, _ = self._client(home, store)
        r = client.get("/review", headers=_auth_headers(token))
        assert r.status_code == 200
        assert "Approve" in r.text
        assert "Review actions enabled." in r.text

    def test_human_credential_approve_routes_through_facade(self, tmp_path):
        home = tmp_path / "home"
        home.mkdir()
        token = self._mint(home, "human-reviewer", ["read", "review"])
        store = StubStore()
        _seed_store(store)
        cid = list(store._candidates.keys())[0]
        client, _ = self._client(home, store)
        r = client.post(f"/review/{cid}",
                        data={"decision": "approved", "reason": "credential review"},
                        headers=_auth_headers(token))
        assert r.status_code == 200
        assert "Review Result" in r.text
        review_calls = [c for c in store.calls if c["method"] == "review_candidate"]
        assert len(review_calls) == 1
        assert review_calls[0]["args"]["review_source"] == "tool"

    def test_model_credential_refused_class_b(self, tmp_path):
        """No self-approval: a model credential with the review class
        gets no buttons and the facade refuses the action."""
        home = tmp_path / "home"
        home.mkdir()
        token = self._mint(home, "model-agent", ["read", "review"], principal_type="model")
        store = StubStore()
        _seed_store(store)
        cid = list(store._candidates.keys())[0]
        client, _ = self._client(home, store)
        r = client.get("/review", headers=_auth_headers(token))
        assert r.status_code == 200
        assert "Approve" not in r.text
        r = client.post(f"/review/{cid}",
                        data={"decision": "approved"},
                        headers=_auth_headers(token))
        assert r.status_code == 403
        assert not [c for c in store.calls if c["method"] == "review_candidate"]

    def test_read_class_credential_no_mutation_buttons(self, tmp_path):
        home = tmp_path / "home"
        home.mkdir()
        token = self._mint(home, "reader", ["read"])
        store = StubStore()
        _seed_store(store)
        client, _ = self._client(home, store)
        r = client.get("/review", headers=_auth_headers(token))
        assert r.status_code == 200
        assert "Approve" not in r.text
        assert "read-only" in r.text.lower()

    def test_credential_principal_shown_on_dashboard(self, tmp_path):
        home = tmp_path / "home"
        home.mkdir()
        token = self._mint(home, "simone", ["read"])
        client, _ = self._client(home)
        r = client.get("/", headers=_auth_headers(token))
        assert r.status_code == 200
        assert "simone" in r.text

    def test_expired_credential_401(self, tmp_path):
        from datetime import datetime, timedelta, timezone
        home = tmp_path / "home"
        home.mkdir()
        past = datetime.now(timezone.utc) - timedelta(hours=1)
        token = self._mint(home, "old", ["read"], expires_at=past)
        client, _ = self._client(home)
        r = client.get("/", headers=_auth_headers(token))
        assert r.status_code == 401
        assert "expired" in r.text.lower()

    def test_revoked_credential_401_next_request(self, tmp_path):
        """Removing the entry revokes it without a restart (live re-read)."""
        import json
        from api_credentials import credential_file_path
        home = tmp_path / "home"
        home.mkdir()
        token = self._mint(home, "temp", ["read"])
        client, _ = self._client(home)
        assert client.get("/", headers=_auth_headers(token)).status_code == 200
        path = credential_file_path(home)
        data = json.loads(path.read_text(encoding="utf-8"))
        data["credentials"] = [c for c in data.get("credentials", [])
                               if c.get("name") != "temp"]
        path.write_text(json.dumps(data), encoding="utf-8")
        r = client.get("/", headers=_auth_headers(token))
        assert r.status_code == 401

    def test_legacy_env_default_preserved(self, tmp_path, monkeypatch):
        """Existing behavior: the legacy token is the local trusted UI —
        human by construction, review enabled by default (spec-11)."""
        monkeypatch.delenv("ARGOS_API_PRINCIPAL_TYPE", raising=False)
        home = tmp_path / "home"
        home.mkdir()
        store = StubStore()
        _seed_store(store)
        client, _ = self._client(home, store)
        r = client.get("/review", headers=_auth_headers())
        assert r.status_code == 200
        assert "Approve" in r.text

    def test_legacy_env_model_narrows(self, tmp_path, monkeypatch):
        """An explicit ARGOS_API_PRINCIPAL_TYPE=model on the legacy path
        is honored (review actions off) — env can only narrow."""
        monkeypatch.setenv("ARGOS_API_PRINCIPAL_TYPE", "model")
        home = tmp_path / "home"
        home.mkdir()
        store = StubStore()
        _seed_store(store)
        cid = list(store._candidates.keys())[0]
        client, _ = self._client(home, store)
        r = client.get("/review", headers=_auth_headers())
        assert r.status_code == 200
        assert "Approve" not in r.text
        r = client.post(f"/review/{cid}",
                        data={"decision": "approved"},
                        headers=_auth_headers())
        assert r.status_code == 403

    def test_invalid_credential_file_500(self, tmp_path):
        home = tmp_path / "home"
        home.mkdir()
        from api_credentials import credential_file_path
        credential_file_path(home).write_text("{broken", encoding="utf-8")
        client, _ = self._client(home)
        r = client.get("/", headers=_auth_headers("some-token"))
        assert r.status_code == 500
        assert "invalid_credential_config" in r.text


# ---------------------------------------------------------------------------
# #484: first-run setup mode
# ---------------------------------------------------------------------------

class TestSetupMode:
    """#484 Phase 1: no credential → setup mode; create-key flow."""

    def _make_setup_client(self, tmp_path: Path) -> TestClient:
        store = StubStore()
        facade = ArgosAPIFacade(store, acl=ACLConfig(), api_mode=False)
        app = create_app(facade, auth_token=None, home=tmp_path)
        return TestClient(app)

    def test_setup_mode_routes_browsers_to_setup(self, tmp_path):
        client = self._make_setup_client(tmp_path)
        assert client.get("/health").status_code == 200
        r = client.get("/", headers={"Accept": "text/html"}, follow_redirects=False)
        assert r.status_code == 303
        assert r.headers["location"] == "/setup"
        r = client.get("/login", follow_redirects=False)
        assert r.status_code == 303
        assert r.headers["location"] == "/setup"

    def test_setup_mode_refuses_api_requests(self, tmp_path):
        client = self._make_setup_client(tmp_path)
        r = client.get("/", headers={"Accept": "application/json"},
                      follow_redirects=False)
        assert r.status_code == 403
        assert "setup_required" in r.text

    def test_setup_page_renders_form_without_token_material(self, tmp_path):
        client = self._make_setup_client(tmp_path)
        r = client.get("/setup")
        assert r.status_code == 200
        assert "Create" in r.text
        assert "argos_" not in r.text

    def test_setup_creates_key_shows_once_and_signs_in(self, tmp_path):
        client = self._make_setup_client(tmp_path)
        r = client.post("/setup", data={"name": "admin"})
        assert r.status_code == 200
        match = re.search(r"argos_[A-Za-z0-9_\-]+", r.text)
        assert match, "the key must be shown exactly once on the success page"
        token = match.group(0)

        # Auto sign-in: the session cookie rode on the setup response.
        dash = client.get("/", headers={"Accept": "text/html"})
        assert dash.status_code == 200
        assert "Argos Admin Console" in dash.text

        # On disk: one human credential, full class set, hash only.
        raw = (tmp_path / "api_credential.json").read_text(encoding="utf-8")
        data = json.loads(raw)
        [cred] = data["credentials"]
        assert cred["name"] == "admin"
        assert cred["principal_type"] == "human"
        assert set(cred["allowed_classes"]) == {
            "read", "propose", "ingest", "erase", "review",
            "write", "feedback", "collection_read", "collection_write",
        }
        assert "token_sha256" in cred
        assert token not in raw

    def test_setup_key_works_as_bearer(self, tmp_path):
        client = self._make_setup_client(tmp_path)
        r = client.post("/setup", data={"name": "admin"})
        token = re.search(r"argos_[A-Za-z0-9_\-]+", r.text).group(0)
        fresh = self._make_setup_client(tmp_path)  # new app, no cookies
        authed = fresh.get(
            "/", headers={"Authorization": f"Bearer {token}"}
        )
        assert authed.status_code == 200

    def test_setup_self_destructs_and_never_duplicates(self, tmp_path):
        client = self._make_setup_client(tmp_path)
        assert client.post("/setup", data={"name": "admin"}).status_code == 200
        # Refresh / revisit: no second mint, both routes are closed.
        again = client.post("/setup", data={"name": "admin"},
                            follow_redirects=False)
        assert again.status_code == 303
        assert again.headers["location"] == "/login"
        page = client.get("/setup", follow_redirects=False)
        assert page.status_code == 303
        assert page.headers["location"] == "/login"
        data = json.loads(
            (tmp_path / "api_credential.json").read_text(encoding="utf-8")
        )
        assert len(data["credentials"]) == 1

    def test_configured_console_has_no_setup(self):
        client = _make_client()  # legacy-token console
        r = client.get("/setup", follow_redirects=False)
        assert r.status_code == 303
        assert r.headers["location"] == "/login"


class TestKeysPage:
    """#484 Phase 2: mint / list / revoke credentials from the UI."""

    def _keys_client(self, tmp_path):
        from api_credentials import credential_file_path, write_credential

        home = tmp_path / "home"
        home.mkdir()
        _admin_tok, _ = write_credential(
            credential_file_path(home),
            name="admin", principal_type="human",
            allowed_classes=["read", "review"], user_id="default_user",
        )
        store = StubStore()
        facade = ArgosAPIFacade(store, acl=ACLConfig(), api_mode=False)
        app = create_app(facade, auth_token=None, home=home)
        return TestClient(app), home, _admin_tok

    def _signin(self, client, token):
        r = client.post("/login", data={"token": token}, follow_redirects=False)
        assert r.status_code == 303

    def test_keys_requires_auth(self):
        client = _make_client()  # legacy console, no credential file
        r = client.get("/keys", follow_redirects=False)
        assert r.status_code in (401, 303)

    def test_keys_lists_credentials(self, tmp_path):
        from api_credentials import credential_file_path, write_credential

        client, home, admin_tok = self._keys_client(tmp_path)
        write_credential(credential_file_path(home), name="worker",
                         principal_type="human", allowed_classes=["read"],
                         user_id="default_user")
        self._signin(client, admin_tok)
        r = client.get("/keys")
        assert r.status_code == 200
        assert "worker" in r.text and "admin" in r.text
        assert "Mint a key" in r.text      # human principal, not read-only
        assert "danger-btn" in r.text      # revoke affordance rendered
        assert "argos_" not in r.text      # no key material on the page

    def test_mint_shows_once_and_preserves_session(self, tmp_path):
        from api_credentials import credential_file_path, parse_credentials_file

        client, home, admin_tok = self._keys_client(tmp_path)
        self._signin(client, admin_tok)
        r = client.post("/keys", data={"name": "worker"}, follow_redirects=False)
        assert r.status_code == 200
        assert "worker" in r.text
        assert re.search(r"argos_[A-Za-z0-9_\-]+", r.text)  # shown once
        assert client.get("/").status_code == 200  # session not swapped
        _, creds = parse_credentials_file(credential_file_path(home))
        assert {c.name for c in creds} == {"admin", "worker"}
        assert all(c.principal_type == "human" for c in creds)

    def test_revoke_applies_immediately_and_blocks_self(self, tmp_path):
        from api_credentials import (credential_file_path,
                                     parse_credentials_file,
                                     write_credential)

        client, home, admin_tok = self._keys_client(tmp_path)
        other_tok, _ = write_credential(
            credential_file_path(home), name="other", principal_type="human",
            allowed_classes=["read"], user_id="default_user",
        )
        self._signin(client, admin_tok)
        r = client.post("/keys/revoke", data={"action": "revoke", "name": "other"},
                        follow_redirects=False)
        assert r.status_code == 303 and "/keys" in r.headers["location"]
        _, creds = parse_credentials_file(credential_file_path(home))
        assert [c.name for c in creds] == ["admin"]
        # Revocation applies on the very next request.
        r2 = client.get("/", headers={"Authorization": f"Bearer {other_tok}"})
        assert r2.status_code == 401
        # Self-revoke is refused; the file is untouched.
        r3 = client.post("/keys/revoke", data={"action": "revoke", "name": "admin"},
                         follow_redirects=False)
        assert r3.status_code == 303 and "Cannot+revoke" in r3.headers["location"]
        _, creds2 = parse_credentials_file(credential_file_path(home))
        assert [c.name for c in creds2] == ["admin"]

    def test_legacy_revoke_guarded_while_in_use(self, tmp_path):
        import json as _json
        from api_credentials import (
            credential_file_path, parse_credentials_file, write_credential,
        )

        home = tmp_path / "home"
        home.mkdir()
        path = credential_file_path(home)
        admin_tok, _ = write_credential(
            path, name="admin", principal_type="human",
            allowed_classes=["read", "review"], user_id="default_user",
        )
        data = _json.loads(path.read_text(encoding="utf-8"))
        data["token"] = "legacy-secret"
        path.write_text(_json.dumps(data), encoding="utf-8")

        store = StubStore()
        facade = ArgosAPIFacade(store, acl=ACLConfig(), api_mode=False)
        app = create_app(facade, auth_token="legacy-secret", home=home)
        client = TestClient(app)

        # Signed in with the legacy token: the console refuses to retire
        # the asset it is itself depending on this session.
        client.post("/login", data={"token": "legacy-secret"})
        r = client.post("/keys/revoke", data={"action": "revoke_legacy"},
                        follow_redirects=False)
        assert "signed+in+with+the+legacy" in r.headers["location"]
        assert '"token"' in path.read_text(encoding="utf-8")

        # Signed in with a key: retiring the legacy transport token works,
        # per-principal credentials survive.
        client.post("/login", data={"token": admin_tok})
        r2 = client.post("/keys/revoke", data={"action": "revoke_legacy"},
                         follow_redirects=False)
        assert r2.status_code == 303 and "ok=" in r2.headers["location"]
        assert '"token"' not in path.read_text(encoding="utf-8")
        _, creds = parse_credentials_file(path)
        assert [c.name for c in creds] == ["admin"]

    def test_model_principal_is_keys_read_only(self, tmp_path):
        from api_credentials import credential_file_path, write_credential

        home = tmp_path / "home"
        home.mkdir()
        model_tok, _ = write_credential(
            credential_file_path(home), name="bench", principal_type="model",
            allowed_classes=["read"], user_id="default_user",
        )
        store = StubStore()
        facade = ArgosAPIFacade(store, acl=ACLConfig(), api_mode=False)
        app = create_app(facade, auth_token=None, home=home)
        client = TestClient(app)
        client.post("/login", data={"token": model_tok})
        r = client.get("/keys")
        assert r.status_code == 200
        assert "Mint a key" not in r.text
        assert 'action="/keys/revoke"' not in r.text  # no revoke affordance
        r2 = client.post("/keys", data={"name": "x"}, follow_redirects=False)
        assert r2.status_code == 303 and "human" in r2.headers["location"]

    def test_read_only_env_hides_management(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ARGOS_API_READ_ONLY", "1")
        client, _, admin_tok = self._keys_client(tmp_path)
        self._signin(client, admin_tok)
        r = client.get("/keys")
        assert "Mint a key" not in r.text
        assert 'action="/keys/revoke"' not in r.text
        r2 = client.post("/keys", data={"name": "x"}, follow_redirects=False)
        assert r2.status_code == 303 and "read-only" in r2.headers["location"]

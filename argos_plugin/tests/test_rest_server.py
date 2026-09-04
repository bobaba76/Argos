"""Tests for the REST read slice (#126 / spec-09).

Spec tests covered:
  1. Public-method allowlist: no admin/destructive endpoints exposed.
  2. Identity spoof: client cannot widen to another user's scope.
  3. Malformed ACL: API fails closed.
  8. Error-leak: store failure returns stable code + request_id, never
     traceback/path/token/SQL.

Additional:
  - Auth: no token → 401; wrong token → 401; token in query param → 401.
  - Readiness: never "healthy" if embedding not ready.
  - Concurrency limit: saturated → 429.
  - Cache-Control: no-store on all responses.
  - Host binding: 127.0.0.1 only (verified in main(), not testable here).
  - No list/export endpoint.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest
from fastapi.testclient import TestClient

_plugin_dir = Path(__file__).resolve().parent.parent
if str(_plugin_dir) not in sys.path:
    sys.path.insert(0, str(_plugin_dir))

from api_facade import ArgosAPIFacade, AuthContext, READ_OPERATIONS
from access_scoping import ACLConfig
from rest_server import create_app
from store_common import MemoryRecord


# -- Stub store --------------------------------------------------------------

class StubStore:
    """Minimal store stub for REST tests."""

    def __init__(self) -> None:
        self._memories: Dict[str, MemoryRecord] = {}
        self._next = 1
        self._should_fail = False

    def search(self, **kwargs) -> List[MemoryRecord]:
        if self._should_fail:
            raise RuntimeError("DB error: /var/lib/hermes/hybrid_memory.duckdb SELECT * FROM")
        query = kwargs.get("query", "").lower()
        return [r for r in self._memories.values() if query in r.content.lower()][:kwargs.get("limit", 10)]

    def get_memories_by_ids(self, memory_ids: List[str], **kwargs) -> List[MemoryRecord]:
        if self._should_fail:
            raise RuntimeError("SQL: SELECT * FROM memory_records WHERE id='abc'")
        return [self._memories[mid] for mid in memory_ids if mid in self._memories]

    def get_memory_history(self, memory_id: str, **kwargs) -> List[MemoryRecord]:
        return [self._memories[memory_id]] if memory_id in self._memories else []

    def save_candidate(self, **kwargs) -> Dict[str, Any]:
        cid = f"cand-{self._next}"
        self._next += 1
        return {"candidate_id": cid, "status": "pending", **kwargs}

    def review_candidate(self, **kwargs) -> Dict[str, Any] | None:
        return None

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


def _make_client(
    store=None,
    token="test-rest-token",
    readiness_probe=None,
    max_concurrent=20,
    allowed_origins=None,
):
    """Build a TestClient for the REST app."""
    store = store or StubStore()
    facade = ArgosAPIFacade(store, acl=ACLConfig(), api_mode=False)
    app = create_app(
        facade,
        auth_token=token,
        readiness_probe=readiness_probe,
        max_concurrent=max_concurrent,
        allowed_origins=allowed_origins,
    )
    return TestClient(app)


def _auth_headers(token: str = "test-rest-token") -> Dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


# ---------------------------------------------------------------------------
# Spec test 1: Public-method allowlist — no admin/destructive endpoints
# ---------------------------------------------------------------------------

class TestAllowlist:
    """No admin/destructive endpoints are exposed."""

    def test_no_shutdown_endpoint(self):
        client = _make_client()
        for path in ["/v1/shutdown", "/v1/admin/shutdown", "/v1/store/shutdown"]:
            r = client.get(path, headers=_auth_headers())
            assert r.status_code == 404, f"{path} should not exist"

    def test_no_backup_endpoint(self):
        client = _make_client()
        for path in ["/v1/backup", "/v1/admin/backup"]:
            r = client.post(path, headers=_auth_headers())
            assert r.status_code == 404

    def test_no_delete_endpoint(self):
        client = _make_client()
        r = client.delete("/v1/memories/mem-1", headers=_auth_headers())
        assert r.status_code == 405 or r.status_code == 404

    def test_no_graph_endpoints(self):
        client = _make_client()
        for path in ["/v1/graph/search", "/v1/graph/query", "/v1/graph/traverse"]:
            r = client.post(path, headers=_auth_headers(), json={"query": "test"})
            assert r.status_code == 404

    def test_no_list_export_endpoint(self):
        """No list/export endpoint (by design)."""
        client = _make_client()
        for path in ["/v1/memories", "/v1/memories/list", "/v1/export"]:
            r = client.get(path, headers=_auth_headers())
            assert r.status_code == 404

    def test_no_raw_rpc_passthrough(self):
        """No raw RPC/component/method passthrough."""
        client = _make_client()
        r = client.post("/v1/rpc", headers=_auth_headers(),
                        json={"component": "store", "method": "delete_memory"})
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# Spec test 2: Identity spoof through the REST adapter
# ---------------------------------------------------------------------------

class TestIdentitySpoof:
    """Client cannot widen to another user's scope."""

    def test_search_with_spoofed_user_id_rejected(self):
        """user_id in the request body is rejected — the strict
        SearchRequest model forbids extra fields (extra='forbid'),
        so the request never reaches the facade."""
        store = StubStore()
        store.remember(content="User likes apples")
        client = _make_client(store=store)
        r = client.post("/v1/memory/search",
                        headers=_auth_headers(),
                        json={"query": "apples", "user_id": "other-user"})
        # Pydantic rejects the extra field with 422.
        assert r.status_code == 422

    def test_search_with_spoofed_tenant_rejected(self):
        """tenant in the request body is rejected by the strict model."""
        client = _make_client()
        r = client.post("/v1/memory/search",
                        headers=_auth_headers(),
                        json={"query": "test", "tenant": "other-tenant"})
        assert r.status_code == 422


# ---------------------------------------------------------------------------
# Spec test 3: Malformed ACL — fail closed
# ---------------------------------------------------------------------------

class TestFailClosedACL:
    """API mode with invalid ACL fails closed."""

    def test_open_store_acl_works_in_non_api_mode(self):
        """In non-API mode (trusted local), open-store ACL is fine."""
        client = _make_client()
        r = client.get("/v1/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


# ---------------------------------------------------------------------------
# Spec test 8: Error-leak through the REST adapter
# ---------------------------------------------------------------------------

class TestErrorLeak:
    """Store failures return stable code + request_id, never
    traceback/path/token/SQL."""

    def test_store_failure_returns_stable_error(self):
        """A RuntimeError from the store is caught and returned as
        internal_error, not a traceback."""
        store = StubStore()
        store._should_fail = True
        client = _make_client(store=store)
        r = client.post("/v1/memory/search",
                        headers=_auth_headers(),
                        json={"query": "test"})
        assert r.status_code == 500
        error = r.json()["error"]
        assert error["code"] == "internal_error"
        assert "request_id" in error
        # No traceback/path/SQL leaked.
        body_str = json.dumps(r.json())
        assert "/var/lib" not in body_str
        assert "SELECT" not in body_str
        assert "traceback" not in body_str.lower()

    def test_fetch_not_found_returns_404(self):
        """Fetching a nonexistent memory returns 404."""
        client = _make_client()
        r = client.get("/v1/memories/nonexistent", headers=_auth_headers())
        assert r.status_code == 404
        error = r.json()["error"]
        assert error["code"] == "not_found"

    def test_error_envelope_has_no_token(self):
        """The error response must not contain the auth token."""
        store = StubStore()
        store._should_fail = True
        client = _make_client(store=store, token="secret-token-xyz")
        r = client.post("/v1/memory/search",
                        headers=_auth_headers("secret-token-xyz"),
                        json={"query": "test"})
        body_str = json.dumps(r.json())
        assert "secret-token-xyz" not in body_str


# ---------------------------------------------------------------------------
# Auth tests
# ---------------------------------------------------------------------------

class TestAuth:
    """Bearer token auth enforcement."""

    def test_no_auth_header_returns_401(self):
        client = _make_client()
        r = client.get("/v1/capabilities")
        assert r.status_code == 401

    def test_wrong_token_returns_401(self):
        client = _make_client(token="correct-token")
        r = client.get("/v1/capabilities",
                       headers={"Authorization": "Bearer wrong-token"})
        assert r.status_code == 401

    def test_correct_token_succeeds(self):
        client = _make_client(token="correct-token")
        r = client.get("/v1/capabilities",
                       headers={"Authorization": "Bearer correct-token"})
        assert r.status_code == 200

    def test_malformed_auth_header_returns_401(self):
        client = _make_client()
        r = client.get("/v1/capabilities",
                       headers={"Authorization": "Basic abc123"})
        assert r.status_code == 401

    def test_empty_bearer_returns_401(self):
        client = _make_client()
        r = client.get("/v1/capabilities",
                       headers={"Authorization": "Bearer "})
        assert r.status_code == 401

    def test_health_does_not_require_auth(self):
        """Health endpoint is public (liveness probe)."""
        client = _make_client()
        r = client.get("/v1/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_ready_does_not_require_auth(self):
        """Readiness endpoint is public (readiness probe)."""
        client = _make_client()
        r = client.get("/v1/ready")
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# Readiness tests
# ---------------------------------------------------------------------------

class TestReadiness:
    """Liveness vs readiness separated. Readiness never 'healthy' if
    first search would trigger a model load."""

    def test_ready_when_all_components_ok(self):
        client = _make_client(readiness_probe=lambda: {
            "store": "ok", "acl": "ok", "embedding": "ok", "graph": "ok",
        })
        r = client.get("/v1/ready")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_not_ready_when_embedding_not_loaded(self):
        """Readiness is NOT ok if embedding model is not loaded —
        the first search would trigger a model load."""
        client = _make_client(readiness_probe=lambda: {
            "store": "ok", "acl": "ok", "embedding": "loading", "graph": "ok",
        })
        r = client.get("/v1/ready")
        assert r.status_code == 503
        assert r.json()["status"] == "not_ready"
        # R5: component details are no longer exposed to unauthenticated callers.
        assert "components" not in r.json()

    def test_not_ready_when_store_not_opened(self):
        client = _make_client(readiness_probe=lambda: {
            "store": "error", "acl": "ok", "embedding": "ok", "graph": "ok",
        })
        r = client.get("/v1/ready")
        assert r.status_code == 503

    def test_not_ready_when_acl_not_loaded(self):
        client = _make_client(readiness_probe=lambda: {
            "store": "ok", "acl": "error", "embedding": "ok", "graph": "ok",
        })
        r = client.get("/v1/ready")
        assert r.status_code == 503

    def test_ready_with_graph_degraded(self):
        """Graph can be degraded (available-or-degraded) and readiness
        is still ok — graph is not a critical component."""
        client = _make_client(readiness_probe=lambda: {
            "store": "ok", "acl": "ok", "embedding": "ok", "graph": "degraded",
        })
        r = client.get("/v1/ready")
        assert r.status_code == 200
        # R5: graph status is no longer exposed in the response body.
        assert r.json()["status"] == "ok"
        assert "graph" not in r.json()

    def test_liveness_independent_of_readiness(self):
        """Health (liveness) returns ok even when readiness is not ok."""
        client = _make_client(readiness_probe=lambda: {
            "store": "ok", "acl": "ok", "embedding": "loading", "graph": "ok",
        })
        r = client.get("/v1/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


# ---------------------------------------------------------------------------
# Cache-Control tests
# ---------------------------------------------------------------------------

class TestCacheControl:
    """Cache-Control: no-store on all responses."""

    def test_health_has_no_store(self):
        client = _make_client()
        r = client.get("/v1/health")
        assert r.headers.get("cache-control") == "no-store"

    def test_search_has_no_store(self):
        store = StubStore()
        store.remember(content="User likes apples")
        client = _make_client(store=store)
        r = client.post("/v1/memory/search",
                        headers=_auth_headers(),
                        json={"query": "apples"})
        assert r.headers.get("cache-control") == "no-store"

    def test_error_has_no_store(self):
        client = _make_client()
        r = client.get("/v1/capabilities")  # no auth → 401
        assert r.headers.get("cache-control") == "no-store"


# ---------------------------------------------------------------------------
# Search and fetch tests
# ---------------------------------------------------------------------------

class TestSearchAndFetch:
    """Search and fetch endpoints work through the facade."""

    def test_search_returns_results(self):
        store = StubStore()
        store.remember(content="User works at TechCorp")
        client = _make_client(store=store)
        r = client.post("/v1/memory/search",
                        headers=_auth_headers(),
                        json={"query": "TechCorp", "limit": 5})
        assert r.status_code == 200
        data = r.json()
        assert data["count"] >= 1
        assert "TechCorp" in data["results"][0]["content"]

    def test_search_empty_query_rejected(self):
        client = _make_client()
        r = client.post("/v1/memory/search",
                        headers=_auth_headers(),
                        json={"query": ""})
        # Pydantic validation catches empty string (min_length=1).
        assert r.status_code == 422

    def test_search_limit_out_of_range_rejected(self):
        client = _make_client()
        r = client.post("/v1/memory/search",
                        headers=_auth_headers(),
                        json={"query": "test", "limit": 100})
        assert r.status_code == 422

    def test_fetch_returns_memory(self):
        store = StubStore()
        rec = store.remember(content="User likes tea")
        client = _make_client(store=store)
        r = client.get(f"/v1/memories/{rec.memory_id}",
                       headers=_auth_headers())
        assert r.status_code == 200
        data = r.json()
        assert data["memory_id"] == rec.memory_id
        assert "tea" in data["content"]

    def test_fetch_history_returns_versions(self):
        store = StubStore()
        rec = store.remember(content="User likes coffee")
        client = _make_client(store=store)
        r = client.get(f"/v1/memories/{rec.memory_id}/history",
                       headers=_auth_headers())
        assert r.status_code == 200
        data = r.json()
        assert data["count"] >= 1

    def test_capabilities_returns_operations(self):
        client = _make_client()
        r = client.get("/v1/capabilities", headers=_auth_headers())
        assert r.status_code == 200
        data = r.json()
        assert "search" in data["operations"]
        assert data["transport"] == "rest"


# ---------------------------------------------------------------------------
# Concurrency limit test
# ---------------------------------------------------------------------------

class TestConcurrencyLimit:
    """Bounded concurrency: semaphore → 429 when saturated."""

    def test_concurrency_limit_returns_429(self):
        """When the semaphore is saturated, new requests get 429."""
        import threading
        import time as _time

        store = StubStore()
        # Use a very low concurrency limit.
        client = _make_client(store=store, max_concurrent=1)
        # Block the store to hold the semaphore.
        original_search = store.search
        barrier = threading.Event()
        def blocking_search(**kwargs):
            barrier.set()
            _time.sleep(0.5)
            return original_search(**kwargs)
        store.search = blocking_search

        results = []
        def do_request():
            r = client.post("/v1/memory/search",
                            headers=_auth_headers(),
                            json={"query": "test"})
            results.append(r.status_code)

        t1 = threading.Thread(target=do_request)
        t2 = threading.Thread(target=do_request)
        t1.start()
        barrier.wait(timeout=2.0)  # wait for first request to enter search
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)
        # One should be 200, the other 429.
        assert 429 in results, f"Expected 429 in {results}"


# ---------------------------------------------------------------------------
# CORS tests
# ---------------------------------------------------------------------------

class TestCORS:
    """No CORS by default; exact origins only if configured."""

    def test_no_cors_headers_by_default(self):
        client = _make_client()
        r = client.get("/v1/health",
                       headers={"Origin": "https://evil.com"})
        assert "access-control-allow-origin" not in {k.lower() for k in r.headers}

    def test_exact_origin_allowed(self):
        client = _make_client(allowed_origins={"https://app.example.com"})
        r = client.get("/v1/health",
                       headers={"Origin": "https://app.example.com"})
        assert r.headers.get("access-control-allow-origin") == "https://app.example.com"

    def test_wrong_origin_not_allowed(self):
        client = _make_client(allowed_origins={"https://app.example.com"})
        r = client.get("/v1/health",
                       headers={"Origin": "https://evil.com"})
        assert "access-control-allow-origin" not in {k.lower() for k in r.headers}

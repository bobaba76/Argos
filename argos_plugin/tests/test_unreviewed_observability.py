"""Spec-13 (#393 slice S2): unreviewed-class observability tests.

Covers:
  - ``_build_memory_where`` trust_class clause + normalization helper
  - store search trust_class filter ('unreviewed' | 'clean', fail loud)
  - ``unreviewed_stats`` (count + oldest age, global override)
  - RPC dispatch: search kwarg threading + the unreviewed_stats branch
  - facade: op registration, enum validation, caller scoping, restore
  - MCP surface: tool mapping + strict schema
  - REST surface: /v1/memory/unreviewed route + search filter field

S1/#443 lesson applied: exercise the FULL seam (store -> service ->
client -> facade -> transports), not a single layer - a field can
survive the DB and still be dropped by an intermediate whitelist.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

_plugin_dir = Path(__file__).resolve().parent.parent
for _path in (_plugin_dir.parent, _plugin_dir):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


def _make_store(tmp_path):
    from argos.store import DuckDBMemoryStore

    return DuckDBMemoryStore(tmp_path / "test.duckdb", user_id="test_user")


UNREVIEWED_TEXT = "Michael keeps a spare garage door remote in the kitchen drawer"
CLEAN_TEXT = "Michael prefers the window seat on long flights to Cape Town"


def _seed(store) -> None:
    store.remember(
        category="personal_fact",
        content=UNREVIEWED_TEXT,
        source="explicit",
        trust_class="unreviewed",
    )
    store.remember(
        category="preference",
        content=CLEAN_TEXT,
        source="explicit",
    )


# ===========================================================================
# Builder
# ===========================================================================

class TestBuilderFilter:
    def test_unreviewed_clause(self):
        from store_retrieval import _build_memory_where

        sql, params = _build_memory_where(trust_class="unreviewed")
        assert "AND trust_class = ?" in sql
        assert "unreviewed" in params

    def test_clean_clause_is_marker_absence(self):
        from store_retrieval import _build_memory_where

        sql, params = _build_memory_where(trust_class="clean")
        assert "COALESCE(trust_class, '') != 'unreviewed'" in sql
        assert "unreviewed" not in params

    def test_none_adds_no_clause(self):
        from store_retrieval import _build_memory_where

        sql, _ = _build_memory_where()
        assert "trust_class" not in sql

    def test_case_and_whitespace_normalized(self):
        from store_retrieval import _build_memory_where

        sql, params = _build_memory_where(trust_class="  UNReviewED ")
        assert "AND trust_class = ?" in sql
        assert "unreviewed" in params

    def test_invalid_value_fails_loud(self):
        from store_retrieval import _build_memory_where

        with pytest.raises(ValueError):
            _build_memory_where(trust_class="clean-ish")


# ===========================================================================
# Store: filter + stats
# ===========================================================================

class TestStoreFilter:
    def test_unreviewed_filter_returns_only_stamped(self, tmp_path):
        store = _make_store(tmp_path)
        _seed(store)
        hits = store.search("Michael", limit=10, trust_class="unreviewed")
        assert len(hits) == 1
        assert hits[0].trust_class == "unreviewed"
        assert "garage" in hits[0].content

    def test_clean_filter_excludes_stamped(self, tmp_path):
        store = _make_store(tmp_path)
        _seed(store)
        hits = store.search("Michael", limit=10, trust_class="clean")
        assert hits
        assert all((h.trust_class or "") != "unreviewed" for h in hits)

    def test_unfiltered_search_still_sees_both(self, tmp_path):
        store = _make_store(tmp_path)
        _seed(store)
        hits = store.search("Michael", limit=10)
        assert len(hits) == 2

    def test_filter_invalid_value_raises(self, tmp_path):
        store = _make_store(tmp_path)
        _seed(store)
        with pytest.raises(ValueError):
            store.search("Michael", limit=10, trust_class="junk")


class TestUnreviewedStats:
    def test_count_and_oldest_age(self, tmp_path):
        store = _make_store(tmp_path)
        _seed(store)
        stats = store.unreviewed_stats()
        assert stats["count"] == 1
        assert stats["oldest_created_at"]
        assert stats["oldest_age_days"] is not None
        assert stats["oldest_age_days"] >= 0.0

    def test_empty_store_is_zero(self, tmp_path):
        store = _make_store(tmp_path)
        stats = store.unreviewed_stats()
        assert stats["count"] == 0
        assert stats["oldest_created_at"] is None
        assert stats["oldest_age_days"] is None

    def test_global_override_includes_all_rows(self, tmp_path):
        store = _make_store(tmp_path)
        _seed(store)
        stats = store.unreviewed_stats(user_scope=None)
        assert stats["count"] == 1

    def test_clean_rows_not_counted(self, tmp_path):
        store = _make_store(tmp_path)
        store.remember(category="preference", content=CLEAN_TEXT, source="explicit")
        stats = store.unreviewed_stats()
        assert stats["count"] == 0


# ===========================================================================
# RPC dispatch (_call_store): kwarg threading + new branch
# ===========================================================================

class _RecordingStore:
    def __init__(self) -> None:
        self.calls: List[tuple] = []
        self.user_id = "boot_user"

    def set_user_scope(self, user_id: str) -> None:
        self.user_id = user_id

    def search(self, query: str = "", **kwargs) -> List[Any]:
        self.calls.append(("search", dict(kwargs, query=query)))
        return []

    def unreviewed_stats(self, user_scope: str | None = None) -> Dict[str, Any]:
        self.calls.append(("unreviewed_stats", user_scope))
        return {"count": 5, "oldest_created_at": None, "oldest_age_days": None}


class TestServiceDispatch:
    def _svc(self):
        from memory_service import MemoryService

        return MemoryService.__new__(MemoryService)

    def test_dispatch_threads_trust_class(self):
        store = _RecordingStore()
        self._svc()._call_store(
            "search", {"query": "x", "trust_class": "unreviewed"}, "u9", store
        )
        assert store.user_id == "u9"
        search_calls = [c for c in store.calls if c[0] == "search"]
        assert search_calls
        assert search_calls[0][1].get("trust_class") == "unreviewed"

    def test_dispatch_unreviewed_stats_scoped_to_caller(self):
        store = _RecordingStore()
        out = self._svc()._call_store("unreviewed_stats", {}, "u9", store)
        assert out["count"] == 5
        assert ("unreviewed_stats", "u9") in store.calls


# ===========================================================================
# Facade: registration, validation, scoping
# ===========================================================================

class TestFacadeOp:
    def _ctx(self, user_id: str = "test_user"):
        from api_facade import AuthContext

        return AuthContext(
            principal="test-principal",
            tenant="default",
            user_id=user_id,
            transport="rest",
        )

    def test_op_registered_read_tier(self):
        from api_facade import READ_OPERATIONS

        assert "unreviewed" in READ_OPERATIONS

    def test_validate_search_trust_class_normalized(self):
        from api_facade import _validate_search_params

        cleaned = _validate_search_params({"query": "x", "trust_class": "UNReviewed"})
        assert cleaned["trust_class"] == "unreviewed"

    def test_validate_search_trust_class_rejects_junk(self):
        from api_facade import APIError, _validate_search_params

        with pytest.raises(APIError):
            _validate_search_params({"query": "x", "trust_class": "junk"})

    def test_op_returns_scoped_report(self, tmp_path):
        from api_facade import ArgosAPIFacade
        from access_scoping import ACLConfig

        store = _make_store(tmp_path)
        _seed(store)
        facade = ArgosAPIFacade(store, acl=ACLConfig(), api_mode=False)
        result = facade.execute(self._ctx(), "unreviewed", {})
        assert result["count"] == 1
        assert result["scope"] == "test_user"
        assert "oldest_created_at" in result

    def test_op_restores_store_scope(self, tmp_path):
        from api_facade import ArgosAPIFacade
        from access_scoping import ACLConfig

        store = _make_store(tmp_path)
        _seed(store)
        facade = ArgosAPIFacade(store, acl=ACLConfig(), api_mode=False)
        facade.execute(self._ctx(user_id="someone_else"), "unreviewed", {})
        assert store.user_id == "test_user"

    def test_op_restores_scope_on_failure(self, tmp_path):
        from api_facade import ArgosAPIFacade
        from access_scoping import ACLConfig

        class _BoomStore:
            def __init__(self):
                self.user_id = "boot_user"

            def set_user_scope(self, user_id):
                self.user_id = user_id

            def unreviewed_stats(self):
                raise RuntimeError("boom")

        from api_facade import APIError

        store = _BoomStore()
        facade = ArgosAPIFacade(store, acl=ACLConfig(), api_mode=False)
        with pytest.raises(APIError) as excinfo:
            facade.execute(self._ctx(), "unreviewed", {})
        assert excinfo.value.code == "internal_error"
        assert store.user_id == "boot_user"

    def test_callers_without_op_are_denied(self, tmp_path):
        from api_facade import APIError, ArgosAPIFacade, AuthContext
        from access_scoping import ACLConfig

        store = _make_store(tmp_path)
        facade = ArgosAPIFacade(store, acl=ACLConfig(), api_mode=False)
        ctx = AuthContext(
            principal="test-principal",
            tenant="default",
            user_id="test_user",
            transport="rest",
            allowed_operations={"search"},
        )
        with pytest.raises(APIError):
            facade.execute(ctx, "unreviewed", {})


# ===========================================================================
# MCP surface
# ===========================================================================

class TestMcpSurface:
    def test_tool_mapping(self):
        from mcp_server import TOOL_TO_OPERATION

        assert TOOL_TO_OPERATION.get("memory_unreviewed") == "unreviewed"

    def test_search_schema_exposes_trust_class(self):
        from mcp_server import TOOL_DEFINITIONS

        entry = next(d for d in TOOL_DEFINITIONS if d["name"] == "memory_search")
        prop = entry["inputSchema"]["properties"]["trust_class"]
        assert prop["enum"] == ["unreviewed", "clean"]

    def test_tool_definition_strict_schema(self):
        from mcp_server import TOOL_DEFINITIONS

        entries = [d for d in TOOL_DEFINITIONS if d["name"] == "memory_unreviewed"]
        assert len(entries) == 1
        schema = entries[0]["inputSchema"]
        assert schema["type"] == "object"
        assert schema["additionalProperties"] is False
        assert schema["properties"] == {}


# ===========================================================================
# REST surface
# ===========================================================================

class _RestStubStore:
    def __init__(self) -> None:
        self.user_id = "boot_user"
        self.search_calls: List[Dict[str, Any]] = []

    def set_user_scope(self, user_id: str) -> None:
        self.user_id = user_id

    def search(self, **kwargs) -> List[Any]:
        self.search_calls.append(dict(kwargs))
        return []

    def unreviewed_stats(self, user_scope: str | None = None) -> Dict[str, Any]:
        return {
            "count": 4,
            "oldest_created_at": "2026-09-11T05:00:00+00:00",
            "oldest_age_days": 0.6,
        }


def _make_client(store=None, token="test-rest-token"):
    from api_facade import ArgosAPIFacade
    from access_scoping import ACLConfig
    from rest_server import create_app
    from fastapi.testclient import TestClient

    store = store or _RestStubStore()
    facade = ArgosAPIFacade(store, acl=ACLConfig(), api_mode=False)
    app = create_app(facade, auth_token=token)
    return TestClient(app), store


def _auth_headers(token: str = "test-rest-token") -> Dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


class TestRestSurface:
    def test_route_returns_report(self):
        client, _ = _make_client()
        r = client.get("/v1/memory/unreviewed", headers=_auth_headers())
        assert r.status_code == 200
        body = r.json()
        assert body["count"] == 4
        assert body["oldest_age_days"] == 0.6

    def test_route_requires_auth(self):
        client, _ = _make_client()
        r = client.get("/v1/memory/unreviewed")
        assert r.status_code == 401

    def test_search_filter_forwarded(self):
        client, store = _make_client()
        r = client.post(
            "/v1/memory/search",
            headers=_auth_headers(),
            json={"query": "x", "trust_class": "unreviewed"},
        )
        assert r.status_code == 200
        assert store.search_calls
        assert store.search_calls[-1].get("trust_class") == "unreviewed"

    def test_search_filter_invalid_rejected(self):
        client, _ = _make_client()
        r = client.post(
            "/v1/memory/search",
            headers=_auth_headers(),
            json={"query": "x", "trust_class": "junk"},
        )
        assert r.status_code == 422

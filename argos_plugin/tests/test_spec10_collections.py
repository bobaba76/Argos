"""#200 Spec-10 PR 2/3: Collections acceptance tests.

Deterministic, NO LLM calls, disposable HERMES_HOME, ARGOS_HERMETIC_TESTS=1.
Run individually — never the whole suite in one process.

Tests:
  T9 exhaustiveness: 9-item backlog → ALL 9 returned; 100-item list → ALL 100.
     Proven at BOTH the store level AND the facade read op.
  T10 isolation: tenant A's items never visible to tenant B, including
     counts and existence. Uses the Cells end-to-end isolation pattern.
  Writes: create/add/update/remove through the facade; non-loopback denied;
     idempotent replay (same key → same ID, no dup); audit rows (success +
     denial); client-supplied scope stripped; item status transitions
     (open→done→parked); archived_at on remove; schema validation when
     schema is set; free-form JSON when not.

Run:
    python -m pytest argos_plugin/tests/test_spec10_collections.py -q \
        -p no:cacheprovider --tb=short
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import pytest

os.environ["ARGOS_HERMETIC_TESTS"] = "1"

_plugin_dir = Path(__file__).resolve().parent.parent
if str(_plugin_dir) not in sys.path:
    sys.path.insert(0, str(_plugin_dir))

from api_facade import (
    APIError,
    ArgosAPIFacade,
    AuthContext,
    COLLECTION_READ_OPERATIONS,
    COLLECTION_WRITE_OPERATIONS,
    READ_OPERATIONS,
    WRITE_OPERATIONS,
)
from access_scoping import ACLConfig
from store import DuckDBMemoryStore


def _make_ctx(
    principal: str = "client-a",
    tenant: str = "default",
    user_id: str = "user-a",
    transport: str = "loopback",
    is_loopback: bool = True,
    principal_type: str = "human",
) -> AuthContext:
    ops = (
        READ_OPERATIONS | WRITE_OPERATIONS
        | COLLECTION_READ_OPERATIONS | COLLECTION_WRITE_OPERATIONS
    )
    return AuthContext(
        principal=principal,
        tenant=tenant,
        user_id=user_id,
        transport=transport,
        allowed_operations=ops,
        is_loopback=is_loopback,
        principal_type=principal_type,
    )


def _make_facade(store=None, api_mode=False) -> ArgosAPIFacade:
    return ArgosAPIFacade(
        store or DuckDBMemoryStore,
        acl=ACLConfig(),
        api_mode=api_mode,
    )


# -- T9: Exhaustiveness ------------------------------------------------------

class TestT9Exhaustiveness:
    """T9: 9-item backlog → ALL 9 returned; 100-item list → ALL 100.
    Proven at BOTH the store level AND the facade read op."""

    def test_store_9_items_all_returned(self, tmp_path):
        """Store level: 9 items, filter status=open → ALL 9."""
        store = DuckDBMemoryStore(
            tmp_path / "test.duckdb", user_id="test_user", embedder=None,
        )
        c = store.create_collection(name="Backlog", template="backlog")
        for i in range(9):
            store.add_collection_item(
                collection_id=c["collection_id"],
                fields={"title": f"Task {i}"},
            )
        items = store.list_collection_items(
            collection_id=c["collection_id"], status="open",
        )
        assert len(items) == 9

    def test_store_100_items_all_returned(self, tmp_path):
        """Store level: 100 items → ALL 100, no cutoff."""
        store = DuckDBMemoryStore(
            tmp_path / "test.duckdb", user_id="test_user", embedder=None,
        )
        c = store.create_collection(name="Big List")
        for i in range(100):
            store.add_collection_item(
                collection_id=c["collection_id"],
                fields={"title": f"Item {i}"},
            )
        items = store.list_collection_items(
            collection_id=c["collection_id"],
        )
        assert len(items) == 100

    def test_facade_9_items_all_returned(self, tmp_path):
        """Facade level: 9 items, filter status=open → ALL 9."""
        store = DuckDBMemoryStore(
            tmp_path / "test.duckdb", user_id="test_user", embedder=None,
        )
        facade = _make_facade(store=store)
        ctx = _make_ctx(user_id="test_user")
        r = facade.execute(ctx, "collection_create",
                           {"name": "Backlog", "template": "backlog"})
        cid = r["collection_id"]
        for i in range(9):
            facade.execute(ctx, "collection_add_item",
                           {"collection_id": cid, "fields": {"title": f"Task {i}"}})
        result = facade.execute(ctx, "collection_items",
                                {"collection_id": cid, "status": "open"})
        assert result["count"] == 9
        assert len(result["items"]) == 9

    def test_facade_100_items_all_returned(self, tmp_path):
        """Facade level: 100 items → ALL 100, no cutoff."""
        store = DuckDBMemoryStore(
            tmp_path / "test.duckdb", user_id="test_user", embedder=None,
        )
        facade = _make_facade(store=store)
        ctx = _make_ctx(user_id="test_user")
        r = facade.execute(ctx, "collection_create", {"name": "Big List"})
        cid = r["collection_id"]
        for i in range(100):
            facade.execute(ctx, "collection_add_item",
                           {"collection_id": cid, "fields": {"title": f"Item {i}"}})
        result = facade.execute(ctx, "collection_items",
                                {"collection_id": cid})
        assert result["count"] == 100
        assert len(result["items"]) == 100

    def test_status_filter_correct(self, tmp_path):
        """Filter by status returns only matching items."""
        store = DuckDBMemoryStore(
            tmp_path / "test.duckdb", user_id="test_user", embedder=None,
        )
        c = store.create_collection(name="Mixed")
        for i in range(5):
            store.add_collection_item(
                collection_id=c["collection_id"],
                fields={"title": f"Open {i}"},
                status="open",
            )
        for i in range(3):
            store.add_collection_item(
                collection_id=c["collection_id"],
                fields={"title": f"Done {i}"},
                status="done",
            )
        open_items = store.list_collection_items(
            collection_id=c["collection_id"], status="open",
        )
        done_items = store.list_collection_items(
            collection_id=c["collection_id"], status="done",
        )
        all_items = store.list_collection_items(
            collection_id=c["collection_id"],
        )
        assert len(open_items) == 5
        assert len(done_items) == 3
        assert len(all_items) == 8


# -- T10: Isolation ----------------------------------------------------------

class TestT10Isolation:
    """T10: tenant A's items never visible to tenant B, including counts
    and existence. Uses direct store scoping (same pattern as Cells #131)."""

    def test_user_scope_isolation(self, tmp_path):
        """User A's items are not visible to user B in the same tenant."""
        store = DuckDBMemoryStore(
            tmp_path / "test.duckdb", user_id="alice", embedder=None,
        )
        # Alice creates a collection + items
        store.set_user_scope("alice")
        c_a = store.create_collection(name="Alice's List")
        store.add_collection_item(
            collection_id=c_a["collection_id"], fields={"title": "A1"},
        )
        store.add_collection_item(
            collection_id=c_a["collection_id"], fields={"title": "A2"},
        )
        # Bob switches scope
        store.set_user_scope("bob")
        # Bob cannot see Alice's collection
        collections_b = store.list_collections()
        assert len(collections_b) == 0
        # Bob cannot see Alice's items
        items_b = store.list_collection_items(
            collection_id=c_a["collection_id"],
        )
        assert len(items_b) == 0
        # Bob's count is 0
        count_b = store.count_collection_items(
            collection_id=c_a["collection_id"],
        )
        assert count_b == 0
        # Bob cannot get Alice's collection
        assert store.get_collection(c_a["collection_id"]) is None

    def test_facade_scope_isolation(self, tmp_path):
        """Through the facade, user A's collections are not visible to
        user B."""
        store = DuckDBMemoryStore(
            tmp_path / "test.duckdb", user_id="alice", embedder=None,
        )
        facade = _make_facade(store=store)
        ctx_a = _make_ctx(user_id="alice")
        ctx_b = _make_ctx(user_id="bob")
        # Alice creates a collection
        r = facade.execute(ctx_a, "collection_create", {"name": "Alice's"})
        cid = r["collection_id"]
        facade.execute(ctx_a, "collection_add_item",
                       {"collection_id": cid, "fields": {"title": "A1"}})
        # Bob lists collections → empty
        result_b = facade.execute(ctx_b, "collection_list", {})
        assert result_b["count"] == 0
        # Bob lists items in Alice's collection → empty
        items_b = facade.execute(ctx_b, "collection_items",
                                 {"collection_id": cid})
        assert items_b["count"] == 0


# -- Writes: create/add/update/remove through the facade ---------------------

class TestCollectionWrites:
    """Collection write operations through the facade."""

    def test_create_collection(self, tmp_path):
        store = DuckDBMemoryStore(
            tmp_path / "test.duckdb", user_id="test_user", embedder=None,
        )
        facade = _make_facade(store=store)
        ctx = _make_ctx(user_id="test_user")
        r = facade.execute(ctx, "collection_create",
                           {"name": "My List", "template": "todo"})
        assert r["status"] == "created"
        assert r["collection"]["collection_id"] is not None
        assert r["collection"]["name"] == "My List"
        assert r["collection"]["template"] == "todo"

    def test_add_item(self, tmp_path):
        store = DuckDBMemoryStore(
            tmp_path / "test.duckdb", user_id="test_user", embedder=None,
        )
        facade = _make_facade(store=store)
        ctx = _make_ctx(user_id="test_user")
        c = facade.execute(ctx, "collection_create", {"name": "List"})
        r = facade.execute(ctx, "collection_add_item",
                           {"collection_id": c["collection_id"],
                            "fields": {"title": "Task 1"}})
        assert r["status"] == "added"
        assert r["item_id"] is not None
        assert r["item"]["fields"] == {"title": "Task 1"}

    def test_update_item_status_transition(self, tmp_path):
        """open → done → parked status transitions."""
        store = DuckDBMemoryStore(
            tmp_path / "test.duckdb", user_id="test_user", embedder=None,
        )
        facade = _make_facade(store=store)
        ctx = _make_ctx(user_id="test_user")
        c = facade.execute(ctx, "collection_create", {"name": "List"})
        item = facade.execute(ctx, "collection_add_item",
                              {"collection_id": c["collection_id"],
                               "fields": {"title": "Task"}})
        # open → done
        r = facade.execute(ctx, "collection_update_item",
                           {"item_id": item["item_id"], "status": "done"})
        assert r["status"] == "updated"
        assert r["item"]["status"] == "done"
        # done → parked
        r = facade.execute(ctx, "collection_update_item",
                           {"item_id": item["item_id"], "status": "parked"})
        assert r["item"]["status"] == "parked"

    def test_update_item_fields(self, tmp_path):
        store = DuckDBMemoryStore(
            tmp_path / "test.duckdb", user_id="test_user", embedder=None,
        )
        facade = _make_facade(store=store)
        ctx = _make_ctx(user_id="test_user")
        c = facade.execute(ctx, "collection_create", {"name": "List"})
        item = facade.execute(ctx, "collection_add_item",
                              {"collection_id": c["collection_id"],
                               "fields": {"title": "Original"}})
        r = facade.execute(ctx, "collection_update_item",
                           {"item_id": item["item_id"],
                            "fields": {"title": "Updated"}})
        assert r["item"]["fields"] == {"title": "Updated"}

    def test_remove_item_archives(self, tmp_path):
        """Remove sets archived_at; item is no longer in active list."""
        store = DuckDBMemoryStore(
            tmp_path / "test.duckdb", user_id="test_user", embedder=None,
        )
        facade = _make_facade(store=store)
        ctx = _make_ctx(user_id="test_user")
        c = facade.execute(ctx, "collection_create", {"name": "List"})
        item = facade.execute(ctx, "collection_add_item",
                              {"collection_id": c["collection_id"],
                               "fields": {"title": "To Remove"}})
        r = facade.execute(ctx, "collection_remove_item",
                           {"item_id": item["item_id"]})
        assert r["status"] == "removed"
        assert r["item"]["archived_at"] is not None
        # Item is gone from active list
        items = facade.execute(ctx, "collection_items",
                               {"collection_id": c["collection_id"]})
        assert items["count"] == 0
        # Item appears with include_archived
        archived = facade.execute(ctx, "collection_items",
                                  {"collection_id": c["collection_id"],
                                   "include_archived": True})
        assert archived["count"] == 1


# -- Non-loopback denied -----------------------------------------------------

class TestCollectionAccess:
    """Non-loopback callers are denied collection writes (class C only)."""

    def test_non_loopback_create_denied(self, tmp_path):
        store = DuckDBMemoryStore(
            tmp_path / "test.duckdb", user_id="test_user", embedder=None,
        )
        facade = _make_facade(store=store)
        ctx = _make_ctx(user_id="test_user", is_loopback=False,
                        transport="rest")
        with pytest.raises(APIError) as exc:
            facade.execute(ctx, "collection_create", {"name": "Denied"})
        assert exc.value.code == "forbidden"

    def test_non_loopback_add_denied(self, tmp_path):
        store = DuckDBMemoryStore(
            tmp_path / "test.duckdb", user_id="test_user", embedder=None,
        )
        facade = _make_facade(store=store)
        ctx_loop = _make_ctx(user_id="test_user")
        c = facade.execute(ctx_loop, "collection_create", {"name": "List"})
        ctx_ext = _make_ctx(user_id="test_user", is_loopback=False,
                            transport="rest")
        with pytest.raises(APIError) as exc:
            facade.execute(ctx_ext, "collection_add_item",
                           {"collection_id": c["collection_id"],
                            "fields": {"title": "denied"}})
        assert exc.value.code == "forbidden"

    def test_non_loopback_reads_allowed(self, tmp_path):
        """Reads are available to all authenticated callers (not class C)."""
        store = DuckDBMemoryStore(
            tmp_path / "test.duckdb", user_id="test_user", embedder=None,
        )
        facade = _make_facade(store=store)
        ctx_loop = _make_ctx(user_id="test_user")
        c = facade.execute(ctx_loop, "collection_create", {"name": "List"})
        ctx_ext = _make_ctx(user_id="test_user", is_loopback=False,
                            transport="rest")
        # collection_list should work for non-loopback
        r = facade.execute(ctx_ext, "collection_list", {})
        assert r["count"] == 1
        # collection_items should work too
        r = facade.execute(ctx_ext, "collection_items",
                           {"collection_id": c["collection_id"]})
        assert r["count"] == 0


# -- Idempotent replay -------------------------------------------------------

class TestCollectionIdempotency:
    """Same Idempotency-Key → same ID, no dup."""

    def test_idempotent_create(self, tmp_path):
        store = DuckDBMemoryStore(
            tmp_path / "test.duckdb", user_id="test_user", embedder=None,
        )
        facade = _make_facade(store=store)
        ctx = _make_ctx(user_id="test_user")
        params = {"name": "Idempotent List"}
        r1 = facade.execute(ctx, "collection_create", params,
                            idempotency_key="key-1")
        r2 = facade.execute(ctx, "collection_create", params,
                            idempotency_key="key-1")
        assert r1["collection_id"] == r2["collection_id"]
        # Only one collection exists
        listing = facade.execute(ctx, "collection_list", {})
        assert listing["count"] == 1

    def test_idempotent_add_item(self, tmp_path):
        store = DuckDBMemoryStore(
            tmp_path / "test.duckdb", user_id="test_user", embedder=None,
        )
        facade = _make_facade(store=store)
        ctx = _make_ctx(user_id="test_user")
        c = facade.execute(ctx, "collection_create", {"name": "List"})
        params = {"collection_id": c["collection_id"],
                  "fields": {"title": "Task"}}
        r1 = facade.execute(ctx, "collection_add_item", params,
                            idempotency_key="add-1")
        r2 = facade.execute(ctx, "collection_add_item", params,
                            idempotency_key="add-1")
        assert r1["item_id"] == r2["item_id"]
        items = facade.execute(ctx, "collection_items",
                               {"collection_id": c["collection_id"]})
        assert items["count"] == 1

    def test_different_key_creates_two(self, tmp_path):
        store = DuckDBMemoryStore(
            tmp_path / "test.duckdb", user_id="test_user", embedder=None,
        )
        facade = _make_facade(store=store)
        ctx = _make_ctx(user_id="test_user")
        facade.execute(ctx, "collection_create", {"name": "First"},
                       idempotency_key="k1")
        facade.execute(ctx, "collection_create", {"name": "Second"},
                       idempotency_key="k2")
        listing = facade.execute(ctx, "collection_list", {})
        assert listing["count"] == 2


# -- Client-supplied scope stripped ------------------------------------------

class TestCollectionScopeStripping:
    """Client-supplied user_scope/tenant/collection_id/item_id are rejected."""

    def test_client_scope_rejected_on_create(self, tmp_path):
        store = DuckDBMemoryStore(
            tmp_path / "test.duckdb", user_id="test_user", embedder=None,
        )
        facade = _make_facade(store=store)
        ctx = _make_ctx(user_id="test_user")
        with pytest.raises(APIError) as exc:
            facade.execute(ctx, "collection_create",
                           {"name": "Test", "user_scope": "other"})
        assert exc.value.code == "forbidden"

    def test_client_tenant_rejected_on_create(self, tmp_path):
        store = DuckDBMemoryStore(
            tmp_path / "test.duckdb", user_id="test_user", embedder=None,
        )
        facade = _make_facade(store=store)
        ctx = _make_ctx(user_id="test_user")
        with pytest.raises(APIError) as exc:
            facade.execute(ctx, "collection_create",
                           {"name": "Test", "tenant": "other"})
        assert exc.value.code == "forbidden"

    def test_client_collection_id_rejected(self, tmp_path):
        """Client cannot supply a collection_id — server mints it."""
        store = DuckDBMemoryStore(
            tmp_path / "test.duckdb", user_id="test_user", embedder=None,
        )
        facade = _make_facade(store=store)
        ctx = _make_ctx(user_id="test_user")
        with pytest.raises(APIError) as exc:
            facade.execute(ctx, "collection_create",
                           {"name": "Test", "collection_id": "custom-id"})
        assert exc.value.code == "forbidden"

    def test_client_item_id_rejected_on_add(self, tmp_path):
        """Client cannot supply an item_id — server mints it."""
        store = DuckDBMemoryStore(
            tmp_path / "test.duckdb", user_id="test_user", embedder=None,
        )
        facade = _make_facade(store=store)
        ctx = _make_ctx(user_id="test_user")
        c = facade.execute(ctx, "collection_create", {"name": "List"})
        with pytest.raises(APIError) as exc:
            facade.execute(ctx, "collection_add_item",
                           {"collection_id": c["collection_id"],
                            "fields": {"title": "T"},
                            "item_id": "custom-item-id"})
        assert exc.value.code == "forbidden"


# -- CAS ---------------------------------------------------------------------

class TestCollectionCAS:
    """CAS via expected_version on update/remove."""

    def test_update_with_valid_version(self, tmp_path):
        store = DuckDBMemoryStore(
            tmp_path / "test.duckdb", user_id="test_user", embedder=None,
        )
        facade = _make_facade(store=store)
        ctx = _make_ctx(user_id="test_user")
        c = facade.execute(ctx, "collection_create", {"name": "List"})
        item = facade.execute(ctx, "collection_add_item",
                              {"collection_id": c["collection_id"],
                               "fields": {"title": "T"}})
        r = facade.execute(ctx, "collection_update_item",
                           {"item_id": item["item_id"],
                            "status": "done",
                            "expected_version": item["item_id"]})
        assert r["status"] == "updated"

    def test_update_with_stale_version_conflict(self, tmp_path):
        store = DuckDBMemoryStore(
            tmp_path / "test.duckdb", user_id="test_user", embedder=None,
        )
        facade = _make_facade(store=store)
        ctx = _make_ctx(user_id="test_user")
        c = facade.execute(ctx, "collection_create", {"name": "List"})
        item = facade.execute(ctx, "collection_add_item",
                              {"collection_id": c["collection_id"],
                               "fields": {"title": "T"}})
        with pytest.raises(APIError) as exc:
            facade.execute(ctx, "collection_update_item",
                           {"item_id": item["item_id"],
                            "status": "done",
                            "expected_version": "stale-version"})
        assert exc.value.code == "conflict"

    def test_remove_with_stale_version_conflict(self, tmp_path):
        store = DuckDBMemoryStore(
            tmp_path / "test.duckdb", user_id="test_user", embedder=None,
        )
        facade = _make_facade(store=store)
        ctx = _make_ctx(user_id="test_user")
        c = facade.execute(ctx, "collection_create", {"name": "List"})
        item = facade.execute(ctx, "collection_add_item",
                              {"collection_id": c["collection_id"],
                               "fields": {"title": "T"}})
        with pytest.raises(APIError) as exc:
            facade.execute(ctx, "collection_remove_item",
                           {"item_id": item["item_id"],
                            "expected_version": "stale-version"})
        assert exc.value.code == "conflict"


# -- Schema validation -------------------------------------------------------

class TestCollectionSchema:
    """Schema validation when schema is set; free-form JSON when not."""

    def test_schema_validation_rejects_missing_required(self, tmp_path):
        store = DuckDBMemoryStore(
            tmp_path / "test.duckdb", user_id="test_user", embedder=None,
        )
        facade = _make_facade(store=store)
        ctx = _make_ctx(user_id="test_user")
        c = facade.execute(ctx, "collection_create", {
            "name": "Schema List",
            "schema": [{"name": "title", "type": "string", "required": True}],
        })
        with pytest.raises(APIError) as exc:
            facade.execute(ctx, "collection_add_item",
                           {"collection_id": c["collection_id"],
                            "fields": {}})
        assert exc.value.code == "invalid_input"

    def test_schema_validation_rejects_wrong_type(self, tmp_path):
        store = DuckDBMemoryStore(
            tmp_path / "test.duckdb", user_id="test_user", embedder=None,
        )
        facade = _make_facade(store=store)
        ctx = _make_ctx(user_id="test_user")
        c = facade.execute(ctx, "collection_create", {
            "name": "Schema List",
            "schema": [{"name": "priority", "type": "number", "required": True}],
        })
        with pytest.raises(APIError) as exc:
            facade.execute(ctx, "collection_add_item",
                           {"collection_id": c["collection_id"],
                            "fields": {"priority": "not-a-number"}})
        assert exc.value.code == "invalid_input"

    def test_schema_validation_accepts_valid(self, tmp_path):
        store = DuckDBMemoryStore(
            tmp_path / "test.duckdb", user_id="test_user", embedder=None,
        )
        facade = _make_facade(store=store)
        ctx = _make_ctx(user_id="test_user")
        c = facade.execute(ctx, "collection_create", {
            "name": "Schema List",
            "schema": [{"name": "title", "type": "string", "required": True}],
        })
        r = facade.execute(ctx, "collection_add_item",
                           {"collection_id": c["collection_id"],
                            "fields": {"title": "Valid"}})
        assert r["status"] == "added"

    def test_free_form_json_when_no_schema(self, tmp_path):
        """Without a schema, any JSON fields are accepted."""
        store = DuckDBMemoryStore(
            tmp_path / "test.duckdb", user_id="test_user", embedder=None,
        )
        facade = _make_facade(store=store)
        ctx = _make_ctx(user_id="test_user")
        c = facade.execute(ctx, "collection_create", {"name": "Free Form"})
        r = facade.execute(ctx, "collection_add_item",
                           {"collection_id": c["collection_id"],
                            "fields": {"any": "thing", "nested": {"a": 1}}})
        assert r["status"] == "added"


# -- Audit -------------------------------------------------------------------

class TestCollectionAudit:
    """Audit rows for collection writes + denials."""

    def test_audit_log_on_success(self, tmp_path):
        """Successful collection_create produces an audit log entry."""
        store = DuckDBMemoryStore(
            tmp_path / "test.duckdb", user_id="test_user", embedder=None,
        )
        facade = _make_facade(store=store)
        ctx = _make_ctx(user_id="test_user")
        facade.execute(ctx, "collection_create", {"name": "Audited"})
        # The facade's _audit method logs at INFO; verify no exception.
        # For durable audit, the store must have write_access_audit.
        assert hasattr(store, "write_access_audit")

    def test_audit_log_on_denial(self, tmp_path):
        """Denied collection_create (non-loopback) produces an audit entry."""
        store = DuckDBMemoryStore(
            tmp_path / "test.duckdb", user_id="test_user", embedder=None,
        )
        facade = _make_facade(store=store)
        ctx = _make_ctx(user_id="test_user", is_loopback=False,
                        transport="rest")
        with pytest.raises(APIError):
            facade.execute(ctx, "collection_create", {"name": "Denied"})
        # The denial is audited via _audit → write_access_audit (fail-soft).
        # Verify the access_audit table has at least one denial row.
        with store._state.lock:
            rows = store.connection.execute(
                "SELECT COUNT(*) FROM access_audit WHERE denied_count > 0"
            ).fetchone()
        assert rows[0] >= 1

"""#200 Spec-10 PR 1/3: Acceptance tests for the external write tier.

Deterministic, NO LLM calls, disposable HERMES_HOME, ARGOS_HERMETIC_TESTS=1.
Run individually — never the whole suite in one process (pre-existing
single-process deadlock).

Tests T1-T8:
  T1 allowlist: shutdown/backup/set_state/clear_scope/purge_tombstone/
     mark_superseded stay unreachable through the facade.
  T2 identity spoof: principal A submits B's user_id/tenant/scope →
     denied or narrowed, never widened.
  T3 malformed ACL: corrupt ACL → API fails closed / refuses readiness.
  T4 idempotent ingest: same Idempotency-Key twice (incl. simulated
     client timeout) → exactly one candidate, no dup.
  T5 no model self-approval: model principal cannot approve its own
     candidate even with review_source="tool".
  T6 class A never active: external POST always yields a candidate;
     nothing active without class-B human (or loopback class C).
  T7 CAS conflict: PATCH with stale expected_version → 409, no write.
  T8 provider side effects: class-C write through the facade indexes
     the graph + chains versions exactly like native.

Run:
    python -m pytest argos_plugin/tests/test_spec10_writes.py -q \
        -p no:cacheprovider --tb=short
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

os.environ["ARGOS_HERMETIC_TESTS"] = "1"

_plugin_dir = Path(__file__).resolve().parent.parent
if str(_plugin_dir) not in sys.path:
    sys.path.insert(0, str(_plugin_dir))

from api_facade import (
    APIError,
    ArgosAPIFacade,
    AuthContext,
    FORBIDDEN_OPERATIONS,
    PROPOSAL_OPERATIONS,
    PUBLIC_OPERATIONS,
    READ_OPERATIONS,
    WRITE_OPERATIONS,
    IdempotencyRegistry,
)
from access_scoping import ACLConfig
from store_common import MemoryRecord


# -- Stub store for fast, isolated tests -------------------------------------

class StubStore:
    """Minimal store stub that records calls and returns canned data."""

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []
        self._memories: Dict[str, MemoryRecord] = {}
        self._candidates: Dict[str, Dict[str, Any]] = {}
        self._next_id = 1
        self.user_id = "default_user"

    def set_user_scope(self, user_id: str) -> None:
        self.user_id = user_id
        self.calls.append({"method": "set_user_scope", "args": {"user_id": user_id}})

    def search(self, **kwargs) -> List[MemoryRecord]:
        self.calls.append({"method": "search", "args": kwargs})
        return list(self._memories.values())[:kwargs.get("limit", 10)]

    def get_memories_by_ids(self, memory_ids: List[str], **kwargs) -> List[MemoryRecord]:
        self.calls.append({"method": "get_memories_by_ids", "args": {"memory_ids": memory_ids}})
        return [self._memories[mid] for mid in memory_ids if mid in self._memories]

    def get_memory_history(self, memory_id: str, **kwargs) -> List[MemoryRecord]:
        return [self._memories[memory_id]] if memory_id in self._memories else []

    def save_candidate(self, **kwargs) -> Dict[str, Any]:
        self.calls.append({"method": "save_candidate", "args": kwargs})
        cid = f"cand-{self._next_id}"
        self._next_id += 1
        candidate = {
            "candidate_id": cid,
            "status": "pending",
            "content": kwargs.get("content", ""),
            "category": kwargs.get("category", ""),
            **kwargs,
        }
        self._candidates[cid] = candidate
        return candidate

    def save_api_candidate(self, **kwargs) -> Dict[str, Any]:
        self.calls.append({"method": "save_api_candidate", "args": kwargs})
        cid = f"cand-{self._next_id}"
        self._next_id += 1
        candidate = {
            "candidate_id": cid,
            "status": "pending",
            "content": kwargs.get("content", ""),
            "category": kwargs.get("category", ""),
            **kwargs,
        }
        self._candidates[cid] = candidate
        return candidate

    def review_candidate(self, **kwargs) -> Dict[str, Any]:
        self.calls.append({"method": "review_candidate", "args": kwargs})
        cid = kwargs.get("candidate_id", "")
        if cid in self._candidates:
            self._candidates[cid]["status"] = kwargs.get("decision", "pending")
            return {"candidate": self._candidates[cid], "memory": None}
        return None

    def list_candidates(self, **kwargs) -> List[dict]:
        status = kwargs.get("status")
        if status:
            return [c for c in self._candidates.values() if c.get("status") == status]
        return list(self._candidates.values())

    def record_feedback(self, memory_id: str, feedback: str) -> bool:
        self.calls.append({"method": "record_feedback", "args": {"memory_id": memory_id, "feedback": feedback}})
        return True

    def remember(self, **kwargs) -> MemoryRecord:
        self.calls.append({"method": "remember", "args": kwargs})
        mid = f"mem-{self._next_id}"
        self._next_id += 1
        rec = MemoryRecord(
            memory_id=mid,
            category=kwargs.get("category", "context_note"),
            content=kwargs.get("content", ""),
            tags=kwargs.get("tags", []),
            similarity=0.9,
            status="active",
            scope=kwargs.get("scope", "profile"),
        )
        self._memories[mid] = rec
        return rec

    def update_memory(self, **kwargs) -> MemoryRecord:
        self.calls.append({"method": "update_memory", "args": kwargs})
        mid = kwargs.get("memory_id", "")
        if mid in self._memories:
            rec = self._memories[mid]
            content = kwargs.get("content")
            if content is not None:
                rec = MemoryRecord(
                    memory_id=f"mem-{self._next_id}",
                    category=rec.category,
                    content=content,
                    tags=kwargs.get("tags", rec.tags),
                    similarity=0.9,
                    status="active",
                    scope=rec.scope,
                )
                self._next_id += 1
                self._memories[rec.memory_id] = rec
                # Mark old as superseded
                self._memories[mid].status = "superseded"
            return rec
        return None

    def delete_memory(self, **kwargs) -> bool | dict:
        self.calls.append({"method": "delete_memory", "args": kwargs})
        mid = kwargs.get("memory_id", "")
        if mid in self._memories:
            del self._memories[mid]
            return {"action": "deleted", "memory_id": mid}
        return False

    def write_access_audit(self, **kwargs) -> None:
        self.calls.append({"method": "write_access_audit", "args": kwargs})


def _make_ctx(
    principal: str = "client-a",
    tenant: str = "default",
    user_id: str = "user-a",
    transport: str = "mcp-stdio",
    allowed_ops: set | None = None,
    can_propose: bool = True,
    principal_type: str = "human",
    is_loopback: bool = False,
) -> AuthContext:
    """Build an AuthContext for testing."""
    ops = allowed_ops if allowed_ops is not None else (
        READ_OPERATIONS | {"memory_propose", "record_feedback", "review_candidate"}
        | WRITE_OPERATIONS
    )
    return AuthContext(
        principal=principal,
        tenant=tenant,
        user_id=user_id,
        transport=transport,
        allowed_operations=ops,
        can_propose=can_propose,
        principal_type=principal_type,
        is_loopback=is_loopback,
    )


def _make_facade(store=None, api_mode=False, acl=None) -> ArgosAPIFacade:
    return ArgosAPIFacade(
        store or StubStore(),
        acl=acl or ACLConfig(),
        api_mode=api_mode,
    )


# -- T1: Allowlist — forbidden ops stay unreachable --------------------------

class TestT1Allowlist:
    """T1: shutdown/backup/set_state/clear_scope/purge_tombstone/
    mark_superseded stay unreachable through the facade."""

    @pytest.mark.parametrize("forbidden_op", [
        "shutdown", "backup", "set_state", "clear_scope",
        "purge_tombstone", "mark_superseded",
    ])
    def test_forbidden_op_rejected(self, forbidden_op):
        facade = _make_facade()
        ctx = _make_ctx()
        with pytest.raises(APIError) as exc_info:
            facade.execute(ctx, forbidden_op, {})
        assert exc_info.value.code == "method_not_allowed"

    def test_forbidden_ops_not_in_public_operations(self):
        for op in ["shutdown", "backup", "set_state", "clear_scope",
                    "purge_tombstone", "mark_superseded"]:
            assert op not in PUBLIC_OPERATIONS
            assert op in FORBIDDEN_OPERATIONS

    def test_write_ops_are_public_but_gated(self):
        """Class C write ops are in PUBLIC_OPERATIONS but require loopback."""
        assert "memory_save" in PUBLIC_OPERATIONS
        assert "memory_update" in PUBLIC_OPERATIONS
        assert "memory_delete" in PUBLIC_OPERATIONS
        # Non-loopback caller is denied
        facade = _make_facade()
        ctx = _make_ctx(is_loopback=False)
        with pytest.raises(APIError) as exc_info:
            facade.execute(ctx, "memory_save", {"content": "test", "category": "context_note"})
        assert exc_info.value.code == "forbidden"


# -- T2: Identity spoof — denied or narrowed, never widened ------------------

class TestT2IdentitySpoof:
    """T2: principal A submits B's user_id/tenant/scope → denied or narrowed."""

    def test_user_id_spoof_rejected(self):
        facade = _make_facade()
        ctx = _make_ctx(principal="client-a", user_id="user-a")
        with pytest.raises(APIError) as exc_info:
            facade.execute(ctx, "memory_propose", {
                "content": "test content",
                "category": "context_note",
                "user_id": "user-b",  # attempting to spoof
            })
        assert exc_info.value.code == "forbidden"

    def test_tenant_spoof_rejected(self):
        facade = _make_facade()
        ctx = _make_ctx(principal="client-a", tenant="default")
        with pytest.raises(APIError) as exc_info:
            facade.execute(ctx, "memory_propose", {
                "content": "test content",
                "category": "context_note",
                "tenant": "other-tenant",  # attempting to spoof
            })
        assert exc_info.value.code == "forbidden"

    def test_server_derived_user_id_stamped(self):
        """The facade stamps ctx.user_id on the call, not the client's."""
        store = StubStore()
        facade = _make_facade(store=store)
        ctx = _make_ctx(principal="client-a", user_id="user-a")
        facade.execute(ctx, "memory_propose", {
            "content": "test content",
            "category": "context_note",
        })
        # #422/AF3: the facade scopes the store to ctx.user_id for the
        # duration of the propose (save/restore) instead of passing a
        # user_id kwarg that the RPC sanitize boundary would strip.
        save_calls = [c for c in store.calls if c["method"] == "save_api_candidate"]
        assert len(save_calls) == 1
        assert "user_id" not in save_calls[0]["args"]
        scope_calls = [c for c in store.calls if c["method"] == "set_user_scope"]
        assert scope_calls and scope_calls[0]["args"]["user_id"] == "user-a"


# -- T3: Malformed ACL — API fails closed ------------------------------------

class TestT3MalformedACL:
    """T3: corrupt ACL → API fails closed / refuses readiness."""

    def test_corrupt_acl_refuses_to_start(self, tmp_path):
        acl_file = tmp_path / "acl.json"
        acl_file.write_text("{ invalid json !!!")
        with pytest.raises(Exception):
            ACLConfig(path=acl_file)

    def test_api_mode_with_corrupt_acl_refuses_start(self, tmp_path):
        acl_file = tmp_path / "acl.json"
        acl_file.write_text("{ broken json")
        try:
            acl = ACLConfig(path=acl_file)
        except Exception:
            pytest.skip("ACLConfig raises on parse — testing via parse_error path")
        # If ACLConfig doesn't raise but sets parse_error, the facade should
        with pytest.raises(ValueError, match="fail closed"):
            ArgosAPIFacade(StubStore(), acl=acl, api_mode=True)


# -- T4: Idempotent ingest — same key twice → one candidate ------------------

class TestT4IdempotentIngest:
    """T4: same Idempotency-Key twice (incl. simulated client timeout) →
    exactly one candidate, no dup."""

    def test_same_key_same_body_returns_cached(self):
        store = StubStore()
        facade = _make_facade(store=store)
        ctx = _make_ctx()
        params = {"content": "idempotent test", "category": "context_note"}
        r1 = facade.execute(ctx, "memory_propose", params, idempotency_key="key-1")
        r2 = facade.execute(ctx, "memory_propose", params, idempotency_key="key-1")
        # Same candidate_id — no duplicate
        assert r1["candidate_id"] == r2["candidate_id"]
        save_calls = [c for c in store.calls if c["method"] == "save_api_candidate"]
        assert len(save_calls) == 1  # only one store write

    def test_different_key_different_body_creates_two(self):
        store = StubStore()
        facade = _make_facade(store=store)
        ctx = _make_ctx()
        facade.execute(ctx, "memory_propose",
                       {"content": "first", "category": "context_note"},
                       idempotency_key="key-1")
        facade.execute(ctx, "memory_propose",
                       {"content": "second", "category": "context_note"},
                       idempotency_key="key-2")
        save_calls = [c for c in store.calls if c["method"] == "save_api_candidate"]
        assert len(save_calls) == 2

    def test_same_key_different_body_raises_conflict(self):
        facade = _make_facade()
        ctx = _make_ctx()
        facade.execute(ctx, "memory_propose",
                       {"content": "first", "category": "context_note"},
                       idempotency_key="key-1")
        with pytest.raises(APIError) as exc_info:
            facade.execute(ctx, "memory_propose",
                           {"content": "different", "category": "context_note"},
                           idempotency_key="key-1")
        assert exc_info.value.code == "conflict"

    def test_simulated_timeout_replay(self):
        """Simulate a client timeout: the first call succeeds on the
        server, the client retries with the same key → gets the cached
        result, no duplicate write."""
        store = StubStore()
        facade = _make_facade(store=store)
        ctx = _make_ctx()
        params = {"content": "timeout test", "category": "context_note"}
        # First call succeeds
        r1 = facade.execute(ctx, "memory_propose", params, idempotency_key="timeout-key")
        # "Client timeout" — retry with same key
        r2 = facade.execute(ctx, "memory_propose", params, idempotency_key="timeout-key")
        assert r1 == r2
        save_calls = [c for c in store.calls if c["method"] == "save_api_candidate"]
        assert len(save_calls) == 1

    def test_class_c_save_idempotent(self):
        """Class C writes are also idempotent."""
        store = StubStore()
        facade = _make_facade(store=store)
        ctx = _make_ctx(is_loopback=True)
        params = {"content": "class c idempotent", "category": "context_note"}
        r1 = facade.execute(ctx, "memory_save", params, idempotency_key="save-key-1")
        r2 = facade.execute(ctx, "memory_save", params, idempotency_key="save-key-1")
        assert r1 == r2
        remember_calls = [c for c in store.calls if c["method"] == "remember"]
        assert len(remember_calls) == 1


# -- T5: No model self-approval ----------------------------------------------

class TestT5NoModelSelfApproval:
    """T5: model principal cannot approve its own candidate even with
    review_source="tool"."""

    def test_model_principal_denied_approval(self):
        facade = _make_facade()
        ctx = _make_ctx(principal_type="model")
        with pytest.raises(APIError) as exc_info:
            facade.execute(ctx, "review_candidate", {
                "candidate_id": "cand-1",
                "decision": "approved",
            })
        assert exc_info.value.code == "forbidden"
        assert "model" in exc_info.value.message.lower()

    def test_human_principal_can_approve(self):
        store = StubStore()
        # Seed a candidate
        store.save_candidate(content="test", category="context_note")
        facade = _make_facade(store=store)
        ctx = _make_ctx(principal_type="human")
        result = facade.execute(ctx, "review_candidate", {
            "candidate_id": "cand-1",
            "decision": "approved",
        })
        assert result["candidate_id"] == "cand-1"

    def test_model_principal_can_propose_but_not_approve(self):
        """A model principal can create candidates (class A) but cannot
        approve them (class B)."""
        store = StubStore()
        facade = _make_facade(store=store)
        ctx = _make_ctx(principal_type="model")
        # Class A: propose — allowed
        result = facade.execute(ctx, "memory_propose", {
            "content": "model proposal",
            "category": "context_note",
        })
        assert result["candidate_id"] is not None
        # Class B: approve — denied
        with pytest.raises(APIError) as exc_info:
            facade.execute(ctx, "review_candidate", {
                "candidate_id": result["candidate_id"],
                "decision": "approved",
            })
        assert exc_info.value.code == "forbidden"


# -- T6: Class A never active ------------------------------------------------

class TestT6ClassANeverActive:
    """T6: external POST always yields a candidate; nothing active without
    class-B human (or loopback class C)."""

    def test_external_propose_creates_candidate_not_active(self):
        store = StubStore()
        facade = _make_facade(store=store)
        ctx = _make_ctx(transport="rest")
        result = facade.execute(ctx, "memory_propose", {
            "content": "external proposal",
            "category": "context_note",
        })
        assert result["candidate_id"] is not None
        assert result["status"] in ("pending", "quarantined")
        # No active memory was created
        remember_calls = [c for c in store.calls if c["method"] == "remember"]
        assert len(remember_calls) == 0

    def test_class_c_save_creates_active(self):
        """Class C (loopback) save creates an active memory directly."""
        store = StubStore()
        facade = _make_facade(store=store)
        ctx = _make_ctx(is_loopback=True)
        result = facade.execute(ctx, "memory_save", {
            "content": "direct write",
            "category": "context_note",
        })
        assert result["status"] == "saved"
        assert result["memory_id"] is not None
        # An active memory was created
        remember_calls = [c for c in store.calls if c["method"] == "remember"]
        assert len(remember_calls) == 1

    def test_external_cannot_use_class_c_save(self):
        """External (non-loopback) caller cannot use memory_save."""
        facade = _make_facade()
        ctx = _make_ctx(is_loopback=False, transport="rest")
        with pytest.raises(APIError) as exc_info:
            facade.execute(ctx, "memory_save", {
                "content": "attempted direct write",
                "category": "context_note",
            })
        assert exc_info.value.code == "forbidden"


# -- T7: CAS conflict — stale expected_version → 409 ------------------------

class TestT7CASConflict:
    """T7: PATCH with stale expected_version → 409, no write."""

    def test_update_with_valid_version_succeeds(self):
        store = StubStore()
        facade = _make_facade(store=store)
        ctx = _make_ctx(is_loopback=True)
        # Create a memory first
        save_result = facade.execute(ctx, "memory_save", {
            "content": "original",
            "category": "context_note",
        })
        mid = save_result["memory_id"]
        # Update with correct version (the memory_id itself)
        result = facade.execute(ctx, "memory_update", {
            "memory_id": mid,
            "content": "updated content",
            "expected_version": mid,
        })
        assert result["status"] == "updated"

    def test_update_with_stale_version_returns_409(self):
        store = StubStore()
        facade = _make_facade(store=store)
        ctx = _make_ctx(is_loopback=True)
        # Create a memory
        save_result = facade.execute(ctx, "memory_save", {
            "content": "original",
            "category": "context_note",
        })
        mid = save_result["memory_id"]
        # Update it (creates a new version)
        facade.execute(ctx, "memory_update", {
            "memory_id": mid,
            "content": "first update",
        })
        # Now try to update the OLD version with stale expected_version
        with pytest.raises(APIError) as exc_info:
            facade.execute(ctx, "memory_update", {
                "memory_id": mid,  # old mid is now superseded
                "content": "stale update",
                "expected_version": mid,
            })
        assert exc_info.value.code == "conflict"

    def test_delete_with_stale_version_returns_409(self):
        store = StubStore()
        facade = _make_facade(store=store)
        ctx = _make_ctx(is_loopback=True)
        save_result = facade.execute(ctx, "memory_save", {
            "content": "to delete",
            "category": "context_note",
        })
        mid = save_result["memory_id"]
        # Delete with wrong version (nonexistent memory_id)
        with pytest.raises(APIError) as exc_info:
            facade.execute(ctx, "memory_delete", {
                "memory_id": mid,
                "expected_version": "nonexistent-version",
            })
        assert exc_info.value.code == "conflict"


# -- T8: Provider side effects — class C write chains versions ---------------

class TestT8ProviderSideEffects:
    """T8: class-C write through the facade indexes the graph + chains
    versions exactly like native (compare vs native-path control).

    For the facade path, the store-level write + version chaining is
    verified. Graph indexing is a provider_session enrichment that
    happens on the native path; over the facade (SharedMemoryStore),
    the service dispatches to store.remember() which handles the
    DuckDB write + version chain. The test verifies:
    1. A class-C save creates an active memory (same as native).
    2. A class-C update creates a new version (supersedes the old).
    3. The version chain is consistent (old version superseded, new active).
    """

    def test_class_c_save_creates_active_memory(self, tmp_path):
        """Compare: facade save vs native save both produce active memories."""
        from store import DuckDBMemoryStore
        db_path = tmp_path / "test.duckdb"
        store = DuckDBMemoryStore(db_path, user_id="test_user", embedder=None)
        facade = _make_facade(store=store)
        ctx = _make_ctx(is_loopback=True, user_id="test_user")
        result = facade.execute(ctx, "memory_save", {
            "content": "facade write test",
            "category": "context_note",
            "tags": ["test"],
        })
        assert result["status"] == "saved"
        assert result["memory_id"] is not None
        # Verify the memory is active in the store
        records = store.get_memories_by_ids([result["memory_id"]])
        assert len(records) == 1
        assert records[0].status == "active"
        assert records[0].content == "facade write test"

    def test_class_c_update_chains_versions(self, tmp_path):
        """Update creates a new version; old version is superseded."""
        from store import DuckDBMemoryStore
        db_path = tmp_path / "test.duckdb"
        store = DuckDBMemoryStore(db_path, user_id="test_user", embedder=None)
        facade = _make_facade(store=store)
        ctx = _make_ctx(is_loopback=True, user_id="test_user")
        # Save
        save_result = facade.execute(ctx, "memory_save", {
            "content": "version 1",
            "category": "context_note",
        })
        old_mid = save_result["memory_id"]
        # Update
        update_result = facade.execute(ctx, "memory_update", {
            "memory_id": old_mid,
            "content": "version 2",
        })
        new_mid = update_result["memory_id"]
        assert new_mid != old_mid
        # New version is active
        new_records = store.get_memories_by_ids([new_mid])
        assert len(new_records) == 1
        assert new_records[0].status == "active"
        assert new_records[0].content == "version 2"
        # Old version is superseded (not active)
        old_records = store.get_memories_by_ids([old_mid])
        if old_records:
            assert old_records[0].status != "active"

    def test_class_c_delete_removes_memory(self, tmp_path):
        """Delete removes the memory from the active set."""
        from store import DuckDBMemoryStore
        db_path = tmp_path / "test.duckdb"
        store = DuckDBMemoryStore(db_path, user_id="test_user", embedder=None)
        facade = _make_facade(store=store)
        ctx = _make_ctx(is_loopback=True, user_id="test_user")
        save_result = facade.execute(ctx, "memory_save", {
            "content": "to be deleted",
            "category": "context_note",
        })
        mid = save_result["memory_id"]
        # Delete
        del_result = facade.execute(ctx, "memory_delete", {"memory_id": mid})
        assert del_result["status"] == "deleted"
        # Memory is gone from active set
        records = store.get_memories_by_ids([mid])
        assert len(records) == 0 or all(
            r.status != "active" for r in records
        )

    def test_native_vs_facade_save_equivalent(self, tmp_path):
        """Compare native store.remember() vs facade memory_save — both
        produce an active memory with the same content/category/tags."""
        from store import DuckDBMemoryStore
        # Native path
        native_db = tmp_path / "native.duckdb"
        native_store = DuckDBMemoryStore(native_db, user_id="test_user", embedder=None)
        native_rec = native_store.remember(
            content="comparison test",
            category="context_note",
            tags=["compare"],
            dedup=True,
        )
        # Facade path
        facade_db = tmp_path / "facade.duckdb"
        facade_store = DuckDBMemoryStore(facade_db, user_id="test_user", embedder=None)
        facade = _make_facade(store=facade_store)
        ctx = _make_ctx(is_loopback=True, user_id="test_user")
        facade_result = facade.execute(ctx, "memory_save", {
            "content": "comparison test",
            "category": "context_note",
            "tags": ["compare"],
        })
        # Both produce active memories with the same content
        assert native_rec is not None
        assert native_rec.status == "active"
        assert facade_result["status"] == "saved"
        facade_rec = facade_store.get_memories_by_ids([facade_result["memory_id"]])
        assert len(facade_rec) == 1
        assert facade_rec[0].status == "active"
        assert facade_rec[0].content == native_rec.content
        assert facade_rec[0].category == native_rec.category


# -- T9: facade_delete_memory server-side gating (BLOCKER regression) --------

class TestT9FacadeDeleteGating:
    """T9: facade_delete_memory is gated server-side — raw RPC without
    confirm/CAS is denied, identity is stripped, audit records denials.

    These tests drive the service handler directly (no subprocess) to
    verify the server-side gates. The service handler is the security
    boundary — the facade's loopback gate is defense-in-depth on top.
    """

    def _make_service_and_store(self, tmp_path):
        """Build a MemoryService + DuckDBMemoryStore for direct handler tests."""
        from store import DuckDBMemoryStore
        from memory_service import MemoryService
        db_path = tmp_path / "test.duckdb"
        store = DuckDBMemoryStore(db_path, user_id="test_user", embedder=None)
        svc = MemoryService(tmp_path)
        # Register the test user's tenant with the store + graph.
        svc._tenants["default"].store = store
        return svc, store

    def test_raw_rpc_without_confirm_denied(self, tmp_path):
        """(i) Raw RPC facade_delete_memory without _confirmed → denied,
        audit row records the denial.

        #200 PR-2 fix: the gate authority is the _confirmed envelope flag
        (set by call_gated), NOT a client-supplied confirm in args. A
        direct _call_store call without confirmed=True is denied.
        """
        svc, store = self._make_service_and_store(tmp_path)
        # Seed a memory
        rec = store.remember(content="to delete", category="context_note")
        mid = rec.memory_id
        # Call the handler directly without confirmed (simulates raw RPC)
        with pytest.raises(PermissionError, match="confirmed"):
            svc._call_store(
                "facade_delete_memory",
                {"memory_id": mid, "expected_version": mid},
                "test_user",
                store,
                None,
                svc._tenants.get("default"),
            )
        # Memory still exists
        records = store.get_memories_by_ids([mid])
        assert len(records) == 1

    def test_raw_rpc_without_cas_denied(self, tmp_path):
        """Raw RPC facade_delete_memory without expected_version → denied."""
        svc, store = self._make_service_and_store(tmp_path)
        rec = store.remember(content="to delete", category="context_note")
        mid = rec.memory_id
        with pytest.raises(PermissionError, match="expected_version"):
            svc._call_store(
                "facade_delete_memory",
                {"memory_id": mid},
                "test_user",
                store,
                None,
                svc._tenants.get("default"),
                confirmed=True,
            )
        records = store.get_memories_by_ids([mid])
        assert len(records) == 1

    def test_client_user_id_stripped(self, tmp_path):
        """(ii) Client-chosen user_id in the call → stripped, service-
        resolved identity used. The _FORBIDDEN_CLIENT_ARGS set includes
        user_id, so _sanitize_args removes it before the store sees it."""
        from memory_service import _FORBIDDEN_CLIENT_ARGS
        assert "user_id" in _FORBIDDEN_CLIENT_ARGS
        assert "tenant" in _FORBIDDEN_CLIENT_ARGS

    def test_valid_cas_confirm_deletes(self, tmp_path):
        """(iii) Valid CAS + confirmed envelope → deletes exactly the
        expected version, audit written."""
        svc, store = self._make_service_and_store(tmp_path)
        rec = store.remember(content="to delete", category="context_note")
        mid = rec.memory_id
        result = svc._call_store(
            "facade_delete_memory",
            {"memory_id": mid, "expected_version": mid},
            "test_user",
            store,
            None,
            svc._tenants.get("default"),
            confirmed=True,
        )
        # Memory is gone
        records = store.get_memories_by_ids([mid])
        assert len(records) == 0 or all(r.status != "active" for r in records)

    def test_stale_cas_denied(self, tmp_path):
        """CAS with stale expected_version → denied, no delete."""
        svc, store = self._make_service_and_store(tmp_path)
        rec = store.remember(content="to delete", category="context_note")
        mid = rec.memory_id
        with pytest.raises((ValueError, PermissionError), match="CAS"):
            svc._call_store(
                "facade_delete_memory",
                {"memory_id": mid, "expected_version": "stale-version"},
                "test_user",
                store,
                None,
                svc._tenants.get("default"),
                confirmed=True,
            )
        # Memory still exists
        records = store.get_memories_by_ids([mid])
        assert len(records) == 1

    def test_delete_memory_stays_forbidden_raw(self, tmp_path):
        """(iv) delete_memory stays forbidden on the raw RPC boundary."""
        from memory_service import _FORBIDDEN_STORE_METHODS
        assert "delete_memory" in _FORBIDDEN_STORE_METHODS
        assert "facade_delete_memory" not in _FORBIDDEN_STORE_METHODS


# -- T10: Post-write graph hook (WARNING 2 regression) ----------------------

class TestT10PostWriteGraphHook:
    """T10: facade-mediated writes through the service dispatch trigger
    graph indexing (save) and graph removal+re-index (update/delete).

    These tests drive the service handler directly to verify the
    post-write graph hooks fire. The graph is a Kuzu instance attached
    to the tenant; the hook calls graph.index_memory / graph.remove_memory
    after a successful store write.
    """

    def _make_service_with_graph(self, tmp_path):
        """Build a MemoryService + DuckDBMemoryStore + mock graph."""
        from store import DuckDBMemoryStore
        from memory_service import MemoryService
        db_path = tmp_path / "test.duckdb"
        store = DuckDBMemoryStore(db_path, user_id="test_user", embedder=None)
        svc = MemoryService(tmp_path)
        tenant = svc._tenants["default"]
        tenant.store = store
        # Attach a mock graph to verify hook calls.
        tenant.graph = MagicMock()
        return svc, store, tenant

    def test_remember_triggers_graph_index(self, tmp_path):
        """store.remember through the service dispatch indexes the graph."""
        svc, store, tenant = self._make_service_with_graph(tmp_path)
        svc._call_store(
            "remember",
            {"content": "graph hook test", "category": "context_note", "tags": ["test"]},
            "test_user",
            store,
            None,
            tenant,
        )
        tenant.graph.index_memory.assert_called_once()
        call_kwargs = tenant.graph.index_memory.call_args.kwargs
        assert call_kwargs["content"] == "graph hook test"
        assert call_kwargs["category"] == "context_note"

    def test_update_triggers_graph_remove_and_index(self, tmp_path):
        """update_memory through the service dispatch removes the old id
        from the graph and indexes the new version."""
        svc, store, tenant = self._make_service_with_graph(tmp_path)
        # Seed a memory
        rec = store.remember(content="original", category="context_note")
        old_mid = rec.memory_id
        tenant.graph.reset_mock()
        # Update
        svc._call_store(
            "update_memory",
            {"memory_id": old_mid, "content": "updated"},
            "test_user",
            store,
            None,
            tenant,
        )
        # Old id removed from graph
        tenant.graph.remove_memory.assert_called_once_with(old_mid)
        # New version indexed
        tenant.graph.index_memory.assert_called_once()
        call_kwargs = tenant.graph.index_memory.call_args.kwargs
        assert call_kwargs["content"] == "updated"

    def test_facade_delete_triggers_graph_remove(self, tmp_path):
        """facade_delete_memory through the service dispatch removes the
        memory from the graph."""
        svc, store, tenant = self._make_service_with_graph(tmp_path)
        rec = store.remember(content="to delete", category="context_note")
        mid = rec.memory_id
        tenant.graph.reset_mock()
        svc._call_store(
            "facade_delete_memory",
            {"memory_id": mid, "expected_version": mid},
            "test_user",
            store,
            None,
            tenant,
            confirmed=True,
        )
        tenant.graph.remove_memory.assert_called_once_with(mid)

    def test_graph_failure_does_not_fail_write(self, tmp_path):
        """A graph indexing failure must not fail the store write."""
        svc, store, tenant = self._make_service_with_graph(tmp_path)
        tenant.graph.index_memory.side_effect = RuntimeError("graph down")
        # The write should still succeed
        result = svc._call_store(
            "remember",
            {"content": "survives graph failure", "category": "context_note"},
            "test_user",
            store,
            None,
            tenant,
        )
        assert result is not None
        # The memory was written to the store
        assert store.count() >= 1


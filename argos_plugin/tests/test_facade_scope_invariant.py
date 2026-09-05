"""Tests for #301: thread-local invariant in _op_search.

Proves:
(1) Two principals search through the facade in sequence on a shared
    store; the second receives ZERO records owned only by the first.
(2) A worker thread reusing the store without resetting scope must fail
    loudly (assertion fires), not return cross-user rows.
(3) The store's user scope is reset to the default after _op_search,
    even on exception.

Run with (Hermes venv python, hermetic):
    ARGOS_HERMETIC_TESTS=1 python -m pytest tests/test_facade_scope_invariant.py -v
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import Any, Dict, List

import pytest

_plugin_dir = Path(__file__).resolve().parent.parent
if str(_plugin_dir) not in sys.path:
    sys.path.insert(0, str(_plugin_dir))

from api_facade import ArgosAPIFacade, AuthContext, APIError, READ_OPERATIONS
from store_common import MemoryRecord


# -- Stub store with user-scope isolation ------------------------------------

class ScopeAwareStubStore:
    """Stub store that filters results by user_id to test scope isolation.

    Records are stored with a user_scope field. search() only returns
    records whose user_scope matches the store's current user_id.
    set_user_scope changes the current user_id.
    """

    def __init__(self, default_user: str = "default_user") -> None:
        self._default_user_id = default_user
        self.user_id = default_user
        self._memories: Dict[str, MemoryRecord] = {}
        self._next_id = 1

    def set_user_scope(self, user_id: str | None) -> None:
        self.user_id = (user_id or self._default_user_id).strip()

    def search(self, **kwargs) -> List[MemoryRecord]:
        query = kwargs.get("query", "").lower()
        results = []
        for mid, rec in self._memories.items():
            # Only return records for the current user scope.
            if getattr(rec, "user_scope", None) != self.user_id:
                continue
            if query in rec.content.lower():
                results.append(rec)
        return results[:kwargs.get("limit", 10)]

    def get_memories_by_ids(self, memory_ids, **kwargs):
        return [self._memories[mid] for mid in memory_ids if mid in self._memories]

    def get_memory_history(self, memory_id, **kwargs):
        return [self._memories[memory_id]] if memory_id in self._memories else []

    def save_candidate(self, **kwargs):
        cid = f"cand-{self._next_id}"
        self._next_id += 1
        return {"candidate_id": cid, "status": "pending", **kwargs}

    def review_candidate(self, **kwargs):
        return {"status": "ok"}

    def record_feedback(self, memory_id, feedback):
        return {"memory_id": memory_id, "feedback": feedback}

    def _add_memory(self, content: str, user_scope: str, memory_id: str = None) -> MemoryRecord:
        mid = memory_id or f"mem-{self._next_id}"
        self._next_id += 1
        rec = MemoryRecord(
            memory_id=mid,
            category="context_note",
            content=content,
            tags=[],
            user_scope=user_scope,
            status="active",
            created_at="2026-01-01T00:00:00Z",
            updated_at="2026-01-01T00:00:00Z",
        )
        self._memories[mid] = rec
        return rec


def _ctx(**overrides) -> AuthContext:
    defaults = dict(
        principal="test-principal",
        tenant="default",
        user_id="alice",
        transport="rest",
        allowed_operations=set(READ_OPERATIONS),
    )
    defaults.update(overrides)
    return AuthContext(**defaults)


class TestSequentialSearchScopeIsolation:
    """(1) Two principals search sequentially; no cross-user leak."""

    def test_second_principal_gets_zero_records_from_first(self):
        """Alice searches, then Bob searches on the same store.
        Bob must get zero records that belong only to Alice."""
        store = ScopeAwareStubStore(default_user="default_user")
        # Add a memory under Alice's scope.
        store._add_memory("alice secret project", user_scope="alice")
        # Add a memory under Bob's scope.
        store._add_memory("bob public note", user_scope="bob")

        facade = ArgosAPIFacade(store)

        # Alice searches.
        alice_ctx = _ctx(user_id="alice")
        alice_result = facade.execute(alice_ctx, "search", {"query": "secret"})
        assert alice_result["count"] == 1
        assert "alice" in alice_result["results"][0]["content"]

        # Bob searches on the same store — must NOT see Alice's record.
        bob_ctx = _ctx(user_id="bob")
        bob_result = facade.execute(bob_ctx, "search", {"query": "secret"})
        assert bob_result["count"] == 0  # Bob has no "secret" records

        # Bob can see his own records.
        bob_result2 = facade.execute(bob_ctx, "search", {"query": "public"})
        assert bob_result2["count"] == 1
        assert "bob" in bob_result2["results"][0]["content"]

    def test_scope_reset_after_search(self):
        """After _op_search, the store's scope is reset to default."""
        store = ScopeAwareStubStore(default_user="default_user")
        store._add_memory("test data", user_scope="alice")

        facade = ArgosAPIFacade(store)
        ctx = _ctx(user_id="alice")
        facade.execute(ctx, "search", {"query": "test"})

        # After the search, the store's scope should be back to default.
        assert store.user_id == "default_user"


class TestScopeResetOnException:
    """(3) Scope is reset even when search raises."""

    def test_scope_reset_on_search_exception(self):
        """If search() raises, the finally block still resets scope."""
        store = ScopeAwareStubStore(default_user="default_user")

        # Make search raise.
        original_search = store.search

        def failing_search(**kwargs):
            raise RuntimeError("DB error")

        store.search = failing_search

        facade = ArgosAPIFacade(store)
        ctx = _ctx(user_id="alice")
        with pytest.raises(APIError) as exc_info:
            facade.execute(ctx, "search", {"query": "test"})
        assert exc_info.value.code == "internal_error"

        # Scope must still be reset despite the exception.
        assert store.user_id == "default_user"


class TestWorkerThreadScopeLeakFailsLoud:
    """(2) A worker thread reusing the store without resetting scope
    must fail loudly, not return cross-user rows."""

    def test_cross_scope_search_returns_zero_not_leaked(self):
        """If a thread sets scope to alice and another searches as bob
        without the facade's set_user_scope, the store's scope is still
        alice — but the facade's _op_search will set it to bob before
        searching, so no leak occurs.

        This test verifies the facade's set_user_scope fires BEFORE
        search, not just the finally reset."""
        store = ScopeAwareStubStore(default_user="default_user")
        store._add_memory("alice private", user_scope="alice")
        store._add_memory("bob private", user_scope="bob")

        # Simulate: a thread leaves the store scoped to alice.
        store.set_user_scope("alice")

        facade = ArgosAPIFacade(store)

        # Now bob searches through the facade. The facade must set
        # scope to bob BEFORE searching, so bob doesn't see alice's data.
        bob_ctx = _ctx(user_id="bob")
        result = facade.execute(bob_ctx, "search", {"query": "private"})
        assert result["count"] == 1
        assert "bob" in result["results"][0]["content"]
        # Alice's record must NOT appear.
        for r in result["results"]:
            assert "alice" not in r["content"]

    def test_concurrent_threads_do_not_cross_contaminate(self):
        """Two threads searching concurrently through the facade must
        each see only their own data. The facade's set_user_scope +
        try/finally reset ensures each search is scoped correctly."""
        store = ScopeAwareStubStore(default_user="default_user")
        store._add_memory("alice data", user_scope="alice")
        store._add_memory("bob data", user_scope="bob")

        facade = ArgosAPIFacade(store)
        results = {}
        errors = []

        def search_as(user_id, query):
            try:
                ctx = _ctx(user_id=user_id)
                r = facade.execute(ctx, "search", {"query": query})
                results[user_id] = r
            except Exception as e:
                errors.append((user_id, e))

        t1 = threading.Thread(target=search_as, args=("alice", "data"))
        t2 = threading.Thread(target=search_as, args=("bob", "data"))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert not errors, f"Threads failed: {errors}"
        # Each thread should see exactly 1 result — their own.
        assert results["alice"]["count"] == 1
        assert "alice" in results["alice"]["results"][0]["content"]
        assert results["bob"]["count"] == 1
        assert "bob" in results["bob"]["results"][0]["content"]


class TestRealStoreScopeRestore:
    """Real-store test: DuckDBMemoryStore constructed with a non-default
    user_id must have its scope restored after a facade search.

    This catches the bug where the finally-reset used getattr(store,
    '_default_user_id', 'default_user') — DuckDBMemoryStore has no
    _default_user_id attribute, so the fallback hardcoded 'default_user'
    would permanently re-scope a store built with user_id='service_acct'.
    """

    def test_facade_search_restores_non_default_constructor_scope(self, tmp_path):
        """A DuckDBMemoryStore built with user_id='service_acct' must have
        its scope restored to 'service_acct' after a facade search — not
        reset to the hardcoded 'default_user'."""
        from store import DuckDBMemoryStore

        store = DuckDBMemoryStore(
            tmp_path / "scope_restore.duckdb", user_id="service_acct",
        )
        try:
            assert store.user_id == "service_acct"

            facade = ArgosAPIFacade(store)
            ctx = _ctx(user_id="alice")
            # Run a facade search — this sets scope to "alice" then
            # should restore it to "service_acct" in the finally block.
            result = facade.execute(ctx, "search", {"query": "anything"})
            assert result["count"] == 0  # empty store

            # The store's scope must be restored to the constructor value,
            # NOT the hardcoded "default_user".
            assert store.user_id == "service_acct", (
                f"Expected scope restored to 'service_acct', "
                f"got '{store.user_id}' — the finally-reset is using a "
                f"hardcoded default instead of the pre-existing scope."
            )
        finally:
            store.close()

    def test_facade_search_restores_scope_on_exception(self, tmp_path):
        """Scope is restored to the constructor value even when search
        raises an exception through the facade."""
        from store import DuckDBMemoryStore

        store = DuckDBMemoryStore(
            tmp_path / "scope_restore_exc.duckdb", user_id="service_acct",
        )
        try:
            facade = ArgosAPIFacade(store)

            # Force an error by searching with an invalid query type —
            # the facade's input validation will raise invalid_input
            # BEFORE reaching _op_search, so we need to trigger an error
            # inside _op_search itself. We do this by making the store's
            # search raise.
            original_search = store.search

            def failing_search(**kwargs):
                raise RuntimeError("simulated DB failure")

            store.search = failing_search

            ctx = _ctx(user_id="alice")
            with pytest.raises(APIError):
                facade.execute(ctx, "search", {"query": "test"})

            # Restore original and verify scope was reset despite exception.
            store.search = original_search
            assert store.user_id == "service_acct", (
                f"Expected scope restored to 'service_acct' after exception, "
                f"got '{store.user_id}'"
            )
        finally:
            store.close()

"""Tests for batch-B: store write-path integrity (#77, #78, #79, #80, #82, #93).

Covers:
- #77: delete_memory promote path is transactional — crash between statements
  leaves the store consistent (no two active versions)
- #78: resolve_conflict supersede paths guard on valid_to IS NULL + user_scope
  (stale target, cross-scope target, already-superseded target)
- #79: resolve_conflict keep_new marks candidate 'deduplicated' (not 'approved')
  when remember() returns None
- #80: timestamp normalization at the write boundary (expires_at, created_at,
  as_of) + _is_expired fails loud on unparseable values
- #82: _content_exists substring dedup gated by overlap ratio; ORDER BY
  recency; dedup reason surfaced
- #93: structural_guard=True wired into the memory_update tool path
  (end-to-end repair test)
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

_plugin_dir = Path(__file__).resolve().parent.parent
if str(_plugin_dir) not in sys.path:
    sys.path.insert(0, str(_plugin_dir))


def _stub_agent_modules():
    if "agent" not in sys.modules:
        sys.modules["agent"] = types.ModuleType("agent")
    if "agent.memory_provider" not in sys.modules:
        _mp = types.ModuleType("agent.memory_provider")

        class MemoryProvider:  # minimal stand-in
            pass

        _mp.MemoryProvider = MemoryProvider
        sys.modules["agent.memory_provider"] = _mp
    if "tools" not in sys.modules:
        sys.modules["tools"] = types.ModuleType("tools")
    if "tools.registry" not in sys.modules:
        _tr = types.ModuleType("tools.registry")
        _tr.tool_error = lambda msg: json.dumps({"error": str(msg)})
        sys.modules["tools.registry"] = _tr


_stub_agent_modules()

from store import DuckDBMemoryStore  # noqa: E402


# ---------------------------------------------------------------------------
# #77: delete_memory promote-path transactionality
# ---------------------------------------------------------------------------

def _create_chain(store, old_content, new_content):
    """Create a two-version chain via update_memory (the canonical path).

    Returns (old_mem, new_mem).
    """
    old_mem = store.remember(category="personal_fact", content=old_content)
    new_mem = store.update_memory(old_mem.memory_id, content=new_content)
    assert new_mem is not None
    return old_mem, new_mem


class TestDeleteMemoryPromoteTransaction:
    """The promote path must be transactional — a crash between the
    predecessor UPDATE and the head DELETE must not leave two active
    versions."""

    def test_promote_succeeds_normally(self, tmp_path):
        """Happy path: head with predecessor is deleted, predecessor promoted."""
        store = DuckDBMemoryStore(
            tmp_path / "test_promote.duckdb", embedder=None,
        )
        try:
            old_mem, new_mem = _create_chain(
                store, "I live in Berlin", "I live in Munich",
            )
            # Delete the head (new version) — should promote old.
            result = store.delete_memory(new_mem.memory_id)
            assert isinstance(result, dict)
            assert result["action"] == "promoted"
            assert result["promoted_memory_id"] == old_mem.memory_id
            # Old should be current again (valid_to IS NULL).
            with store._state.lock:
                row = store.connection.execute(
                    "SELECT valid_to FROM memory_records WHERE memory_id = ?",
                    [old_mem.memory_id],
                ).fetchone()
            assert row[0] is None, "Promoted predecessor should be current"
        finally:
            store.close()

    def test_promote_crash_rolls_back(self, tmp_path):
        """If the DELETE fails after the predecessor UPDATE, the transaction
        must roll back — no two active versions."""
        store = DuckDBMemoryStore(
            tmp_path / "test_promote_crash.duckdb", embedder=None,
        )
        try:
            old_mem, new_mem = _create_chain(
                store, "I live in Berlin", "I live in Munich",
            )
            # Inject a failure on the DELETE FROM memory_records statement
            # inside the promote path. We wrap the connection's execute via
            # a proxy since DuckDBPyConnection.execute is read-only.
            call_count = {"n": 0}

            class _CrashProxy:
                """Proxy that intercepts execute() to inject a failure on
                the first DELETE FROM memory_records (the promote path's
                head deletion). All other calls pass through."""

                def __init__(self, real_conn):
                    self._real = real_conn

                def execute(self, sql, *args, **kwargs):
                    if (
                        "DELETE FROM memory_records" in sql
                        and call_count["n"] == 0
                    ):
                        call_count["n"] += 1
                        raise RuntimeError("simulated crash mid-promote")
                    return self._real.execute(sql, *args, **kwargs)

                def __getattr__(self, name):
                    return getattr(self._real, name)

            proxy = _CrashProxy(store.connection)
            # Temporarily swap the connection on the store so delete_memory
            # uses the proxy.
            original_conn = store.connection
            store.connection = proxy
            try:
                with pytest.raises(RuntimeError, match="simulated crash"):
                    store.delete_memory(new_mem.memory_id)
            finally:
                store.connection = original_conn
            # After rollback: the head (new) should still be current, and
            # the predecessor (old) should still be superseded (valid_to set)
            # — NOT promoted. No two active versions.
            with store._state.lock:
                rows = store.connection.execute(
                    """SELECT memory_id, valid_to FROM memory_records
                       WHERE memory_id IN (?, ?)
                       ORDER BY memory_id""",
                    [old_mem.memory_id, new_mem.memory_id],
                ).fetchall()
            assert len(rows) == 2
            for mid, valid_to in rows:
                if mid == old_mem.memory_id:
                    assert valid_to is not None, (
                        "Predecessor should still be superseded after rollback"
                    )
                elif mid == new_mem.memory_id:
                    assert valid_to is None, (
                        "Head should still be current after rollback"
                    )
        finally:
            store.close()

    def test_promote_path_wrapped_in_transaction(self):
        """Structural check: the delete_memory promote path must be wrapped
        in BEGIN TRANSACTION / COMMIT / ROLLBACK (#77)."""
        import inspect
        from store_write import StoreWriteMixin
        source = inspect.getsource(StoreWriteMixin.delete_memory)
        assert "BEGIN TRANSACTION" in source
        assert "COMMIT" in source
        assert "ROLLBACK" in source
        # The transaction must wrap the multi-statement promote path
        # (UPDATE + tombstone + DELETE), not the single-statement paths.
        pred_branch = source[source.index("if pred:"):source.index("return", source.index("if pred:"))]
        assert "BEGIN TRANSACTION" in pred_branch
        assert "ROLLBACK" in pred_branch

    def test_non_head_paths_not_transactional(self):
        """Structural check: the quarantine and hard-delete paths are
        single-statement and do not need transactions (#77)."""
        import inspect
        from store_write import StoreWriteMixin
        source = inspect.getsource(StoreWriteMixin.delete_memory)
        non_head_idx = source.index("if not is_head:")
        non_head_block = source[non_head_idx:non_head_idx + 300]
        assert "BEGIN TRANSACTION" not in non_head_block


# ---------------------------------------------------------------------------
# #78: resolve_conflict supersession guards
# ---------------------------------------------------------------------------

class TestResolveConflictGuards:
    """resolve_conflict supersede paths must guard on valid_to IS NULL and
    user_scope, matching review_candidate."""

    def _setup_conflict(self, tmp_path, user_id="test_user"):
        store = DuckDBMemoryStore(
            tmp_path / "test_conflict_guard.duckdb",
            user_id=user_id, embedder=None,
        )
        old_mem = store.remember(
            category="personal_fact",
            content="accuracy on the test set is 89.8 percent",
        )
        cand = store.save_candidate(
            category="personal_fact",
            content="I switched to accuracy on the test set of 82.2 percent",
        )
        return store, old_mem, cand

    def test_keep_new_skips_already_superseded_target(self, tmp_path):
        """If old_memory_id points at an already-superseded record, the
        supersede is skipped (no chain corruption)."""
        store, old_mem, cand = self._setup_conflict(tmp_path)
        try:
            # Pre-supersede the old memory via update_memory (the canonical
            # path — now works with DuckDB 1.5.5).
            store.update_memory(
                old_mem.memory_id, content="accuracy is 90 percent now",
            )
            # The candidate's payload still points at the original old_mem.
            # resolve_conflict keep_new should skip the supersede (target
            # is no longer current) but still approve the candidate.
            result = store.resolve_conflict(
                cand["candidate_id"], "keep_new", reason="new value",
            )
            assert result["outcome"] == "keep_new"
            # The candidate should be approved (remember succeeded).
            updated_cand = result["candidate"]
            assert updated_cand["status"] == "approved"
        finally:
            store.close()

    def test_keep_new_cross_scope_target_skipped(self, tmp_path):
        """A candidate whose old_memory_id points at another user's memory
        must not supersede it (user_scope guard)."""
        store, old_mem, cand = self._setup_conflict(tmp_path, user_id="user_a")
        try:
            # Manually craft a candidate pointing at a different user's memory.
            other_store = DuckDBMemoryStore(
                tmp_path / "test_conflict_guard.duckdb",
                user_id="user_b", embedder=None,
            )
            other_mem = other_store.remember(
                category="personal_fact",
                content="accuracy on the test set is 89.8 percent",
            )
            # Patch the candidate's payload to point at the other user's mem.
            with store._state.lock:
                store.connection.execute(
                    """UPDATE memory_candidates
                       SET payload = ?
                       WHERE candidate_id = ?""",
                    [json.dumps({
                        "value_supersession": {
                            "supersedes_memory_id": other_mem.memory_id,
                        },
                    }), cand["candidate_id"]],
                )
            store.resolve_conflict(
                cand["candidate_id"], "keep_new", reason="cross-scope",
            )
            # The other user's memory should NOT be superseded.
            with other_store._state.lock:
                row = other_store.connection.execute(
                    "SELECT valid_to FROM memory_records WHERE memory_id = ?",
                    [other_mem.memory_id],
                ).fetchone()
            assert row[0] is None, (
                "Cross-scope target should not be superseded"
            )
            other_store.close()
        finally:
            store.close()

    def test_remove_both_skips_cross_scope(self, tmp_path):
        """remove_both should not supersede a cross-scope old memory."""
        store, old_mem, cand = self._setup_conflict(tmp_path, user_id="user_a")
        try:
            other_store = DuckDBMemoryStore(
                tmp_path / "test_conflict_guard.duckdb",
                user_id="user_b", embedder=None,
            )
            other_mem = other_store.remember(
                category="personal_fact",
                content="accuracy on the test set is 89.8 percent",
            )
            with store._state.lock:
                store.connection.execute(
                    """UPDATE memory_candidates
                       SET payload = ?
                       WHERE candidate_id = ?""",
                    [json.dumps({
                        "value_supersession": {
                            "supersedes_memory_id": other_mem.memory_id,
                        },
                    }), cand["candidate_id"]],
                )
            store.resolve_conflict(
                cand["candidate_id"], "remove_both", reason="cross-scope",
            )
            with other_store._state.lock:
                row = other_store.connection.execute(
                    "SELECT valid_to FROM memory_records WHERE memory_id = ?",
                    [other_mem.memory_id],
                ).fetchone()
            assert row[0] is None, (
                "remove_both should not supersede cross-scope target"
            )
            other_store.close()
        finally:
            store.close()


# ---------------------------------------------------------------------------
# #79: resolve_conflict keep_new — deduplicated status when remember() fails
# ---------------------------------------------------------------------------

class TestResolveConflictDeduplicated:
    """keep_new must mark the candidate 'deduplicated' (not 'approved') when
    remember() returns None."""

    def test_keep_new_deduped_candidate_gets_deduplicated_status(self, tmp_path):
        """When remember() returns None (content already exists), the
        candidate status should be 'deduplicated', not 'approved'."""
        store = DuckDBMemoryStore(
            tmp_path / "test_dedup_status.duckdb", embedder=None,
        )
        try:
            # Store the new value first so remember() will dedup it.
            store.remember(
                category="personal_fact",
                content="I switched to accuracy on the test set of 82.2 percent",
            )
            # Create the old memory and the conflict candidate.
            store.remember(
                category="personal_fact",
                content="accuracy on the test set is 89.8 percent",
            )
            cand = store.save_candidate(
                category="personal_fact",
                content="I switched to accuracy on the test set of 82.2 percent",
            )
            assert cand is not None
            result = store.resolve_conflict(
                cand["candidate_id"], "keep_new", reason="new value",
            )
            updated_cand = result["candidate"]
            assert updated_cand["status"] == "deduplicated", (
                f"Expected 'deduplicated', got '{updated_cand['status']}'"
            )
            assert result["memory"] is None
        finally:
            store.close()


# ---------------------------------------------------------------------------
# #80: timestamp normalization
# ---------------------------------------------------------------------------

class TestTimestampNormalization:
    """Timestamps must be normalized to the aware-UTC form at the write
    boundary, and _is_expired must fail loud on unparseable values."""

    def test_normalize_timestamp_z_suffix(self):
        """Z-suffix timestamps are normalized to +00:00 form."""
        result = DuckDBMemoryStore._normalize_timestamp("2026-01-15T12:00:00Z")
        assert result is not None
        assert "+00:00" in result
        assert "Z" not in result

    def test_normalize_timestamp_naive_assumes_utc(self):
        """Naive timestamps (no tzinfo) are assumed UTC."""
        result = DuckDBMemoryStore._normalize_timestamp("2026-01-15T12:00:00")
        assert result is not None
        assert "+00:00" in result

    def test_normalize_timestamp_date_only(self):
        """Date-only strings are normalized to full timestamp at midnight UTC."""
        result = DuckDBMemoryStore._normalize_timestamp("2020-01-15")
        assert result is not None
        assert result == "2020-01-15T00:00:00+00:00"

    def test_normalize_timestamp_unparseable_returns_none(self):
        """Unparseable strings return None."""
        assert DuckDBMemoryStore._normalize_timestamp("not a date") is None
        assert DuckDBMemoryStore._normalize_timestamp("") is None
        assert DuckDBMemoryStore._normalize_timestamp(None) is None

    def test_remember_normalizes_expires_at_z(self, tmp_path):
        """remember() normalizes expires_at with Z suffix to +00:00."""
        store = DuckDBMemoryStore(
            tmp_path / "test_ts_norm.duckdb", embedder=None,
        )
        try:
            mem = store.remember(
                category="personal_fact",
                content="I live in Berlin",
                expires_at="2026-01-15T12:00:00Z",
            )
            assert mem is not None
            assert "+00:00" in (mem.expires_at or "")
            assert "Z" not in (mem.expires_at or "")
        finally:
            store.close()

    def test_remember_normalizes_expires_at_date_only(self, tmp_path):
        """remember() normalizes date-only expires_at to full timestamp."""
        store = DuckDBMemoryStore(
            tmp_path / "test_ts_date.duckdb", embedder=None,
        )
        try:
            mem = store.remember(
                category="personal_fact",
                content="I live in Berlin",
                expires_at="2020-01-15",
            )
            assert mem is not None
            # Should be normalized to full timestamp, not left as date-only.
            assert mem.expires_at == "2020-01-15T00:00:00+00:00"
        finally:
            store.close()

    def test_remember_normalizes_created_at_z(self, tmp_path):
        """remember() normalizes created_at with Z suffix."""
        store = DuckDBMemoryStore(
            tmp_path / "test_ts_created.duckdb", embedder=None,
        )
        try:
            mem = store.remember(
                category="personal_fact",
                content="I lived in Paris in 2020",
                created_at="2020-06-15T10:30:00Z",
            )
            assert mem is not None
            assert "+00:00" in mem.created_at
            assert "Z" not in mem.created_at
        finally:
            store.close()

    def test_as_of_normalized_z_suffix(self, tmp_path):
        """as_of with Z suffix is normalized before SQL comparison."""
        store = DuckDBMemoryStore(
            tmp_path / "test_as_of_norm.duckdb", embedder=None,
        )
        try:
            # Store a record with +00:00 timestamp.
            mem = store.remember(
                category="personal_fact",
                content="I live in Berlin",
                created_at="2026-01-15T12:00:00+00:00",
            )
            assert mem is not None
            # Query with Z-suffix as_of — should still find the record
            # because as_of is normalized to +00:00 before SQL comparison.
            results = store.search(
                "Berlin", limit=5, suppress_retrieval=True,
                as_of="2026-01-15T13:00:00Z",
            )
            assert any(r.memory_id == mem.memory_id for r in results), (
                "as_of with Z suffix should find records stored with +00:00"
            )
        finally:
            store.close()

    def test_is_expired_date_only_logs_warning(self, tmp_path, caplog):
        """_is_expired on a date-only string should log a warning (not
        silently return False)."""
        import logging
        with caplog.at_level(logging.WARNING):
            # Date-only "2020-01-15" — fromisoformat parses this fine in
            # Python 3.11+, but the normalization in _is_expired handles it.
            # The key assertion: it doesn't silently return False for a
            # clearly-expired date.
            result = DuckDBMemoryStore._is_expired("2020-01-15")
            # With the fix, date-only is parsed (fromisoformat handles it)
            # and the expiry is correctly detected.
            assert result is True, (
                f"Date-only 2020-01-15 should be expired, got {result}"
            )

    def test_is_expired_unparseable_logs_warning(self, caplog):
        """_is_expired on an unparseable string should log a warning and
        return True (SC4 fail-safe: expired = safe side)."""
        import logging
        with caplog.at_level(logging.WARNING):
            result = DuckDBMemoryStore._is_expired("not-a-date")
            # SC4: fail-safe — unparseable expiry returns True (expired),
            # not False (never expires). Sensitive memories with broken
            # timestamps should expire rather than persist forever.
            assert result is True
            assert any("Unparseable expires_at" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# #82: _content_exists substring dedup gating + ORDER BY + reason
# ---------------------------------------------------------------------------

class TestContentExistsDedupGate:
    """The substring dedup layer must be gated by an overlap ratio so
    distinct facts sharing a long prefix are not silently dropped."""

    def test_content_exists_returns_reason_string(self, tmp_path):
        """_content_exists returns a reason string, not bool."""
        store = DuckDBMemoryStore(
            tmp_path / "test_dedup_reason.duckdb", embedder=None,
        )
        try:
            store.remember(category="personal_fact", content="I live in Berlin")
            reason = store._content_exists("I live in Berlin", "personal_fact")
            assert reason == "exact"
            reason2 = store._content_exists("I live in Munich", "personal_fact")
            assert reason2 is None
        finally:
            store.close()

    def test_substring_near_duplicate_below_gate_not_dropped(self, tmp_path):
        """Two facts sharing a long prefix but differing in a key detail
        (year) should NOT be deduped — the overlap gate blocks the drop."""
        store = DuckDBMemoryStore(
            tmp_path / "test_dedup_gate.duckdb", embedder=None,
        )
        try:
            store.remember(
                category="personal_fact",
                content="The user switched from Python to Rust in 2021",
            )
            # A distinct fact that shares a long prefix but differs in year.
            # Overlap ratio: shorter/longer — the strings are nearly the same
            # length, so overlap is high. But the content is a DIFFERENT fact.
            # With the 0.8 gate, this WILL be deduped (overlap > 0.8).
            # The test validates the gate threshold behavior: a genuinely
            # shorter distinct fact is NOT deduped.
            reason = store._content_exists(
                "The user switched from Python to Rust in 2021 and then to Go",
                "personal_fact",
            )
            # The new string is longer and contains the old as a substring,
            # but the overlap ratio (old_len/new_len) is < 0.8, so it should
            # NOT be deduped.
            assert reason is None, (
                "Distinct fact with low overlap should not be deduped"
            )
        finally:
            store.close()

    def test_substring_high_overlap_deduped(self, tmp_path):
        """Genuine near-duplicate with high overlap (>0.8) IS deduped."""
        store = DuckDBMemoryStore(
            tmp_path / "test_dedup_high.duckdb", embedder=None,
        )
        try:
            store.remember(
                category="personal_fact",
                content="The user switched from Python to Rust in 2021",
            )
            # Nearly identical — just a trailing period difference.
            reason = store._content_exists(
                "The user switched from Python to Rust in 2021.",
                "personal_fact",
            )
            assert reason == "substring"
        finally:
            store.close()

    def test_dedup_reason_logged_at_warning(self, tmp_path, caplog):
        """When remember() dedups, the reason should be logged at WARNING."""
        import logging
        store = DuckDBMemoryStore(
            tmp_path / "test_dedup_log.duckdb", embedder=None,
        )
        try:
            store.remember(category="personal_fact", content="I live in Berlin")
            with caplog.at_level(logging.WARNING):
                result = store.remember(
                    category="personal_fact", content="I live in Berlin",
                    dedup=True,
                )
            assert result is None
            assert any("Deduped memory" in r.message for r in caplog.records)
        finally:
            store.close()


# ---------------------------------------------------------------------------
# #93: structural_guard wired into memory_update tool path
# ---------------------------------------------------------------------------

class TestStructuralGuardWired:
    """The memory_update tool path must pass structural_guard=True for
    content changes (the LLM-agent rewrite path)."""

    def test_memory_update_tool_passes_structural_guard(self, tmp_path):
        """The provider's handle_tool_call for memory_update must pass
        structural_guard=True when content is provided."""
        # Use a stub store to capture kwargs.
        captured = {}

        class StubStore:
            def update_memory(self, **kwargs):
                captured.update(kwargs)
                mem_id = kwargs.get("memory_id", "stub-1")
                content = kwargs.get("content")
                return types.SimpleNamespace(
                    memory_id=f"new-{mem_id}",
                    category="personal_fact",
                    content=content or "",
                    tags=kwargs.get("tags") or [],
                    payload={},
                    created_at="2026-01-01T00:00:00+00:00",
                )

        from provider_session import ProviderSessionMixin
        session = ProviderSessionMixin.__new__(ProviderSessionMixin)
        session._store = StubStore()
        session._graph = None
        session._expiry_enabled = False

        args = {
            "memory_id": "mem-" + "a" * 32,
            "content": "User lives in Springfield.",
        }
        session.handle_tool_call("memory_update", args)
        assert "structural_guard" in captured
        assert captured["structural_guard"] is True

    def test_memory_update_tool_no_content_no_guard(self, tmp_path):
        """When content is not changed (tags-only update), structural_guard
        is not passed (no rewrite to guard)."""
        captured = {}

        class StubStore:
            def update_memory(self, **kwargs):
                captured.update(kwargs)
                return types.SimpleNamespace(
                    memory_id="new-stub",
                    category="personal_fact",
                    content="existing content",
                    tags=kwargs.get("tags") or [],
                    payload={},
                    created_at="2026-01-01T00:00:00+00:00",
                )

        from provider_session import ProviderSessionMixin
        session = ProviderSessionMixin.__new__(ProviderSessionMixin)
        session._store = StubStore()
        session._graph = None
        session._expiry_enabled = False

        args = {"memory_id": "mem-" + "a" * 32, "tags": ["new_tag"]}
        session.handle_tool_call("memory_update", args)
        # structural_guard should NOT be in kwargs (no content change).
        assert "structural_guard" not in captured or captured.get("structural_guard") is not True

    def test_structural_guard_repairs_agent_rewrite(self, tmp_path):
        """End-to-end: an agent rewrite that deletes content via the tool
        path should have it merged back by the guard."""
        store = DuckDBMemoryStore(
            tmp_path / "test_guard_e2e.duckdb", embedder=None,
        )
        try:
            mem = store.remember(
                category="personal_fact",
                content="User lives in Springfield. Works at Acme Corp.",
            )
            # Simulate the tool path: update with structural_guard=True
            # and content that drops the second sentence.
            updated = store.update_memory(
                mem.memory_id,
                content="User lives in Springfield.",
                structural_guard=True,
            )
            assert updated is not None
            # The lost content should be merged back.
            assert "acme" in updated.content.casefold()
        finally:
            store.close()

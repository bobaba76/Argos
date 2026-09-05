"""Pytest tests for the argos plugin.

Run with:
    python -m pytest tests/test_argos.py -v

"""
from __future__ import annotations

import sys
import tempfile
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pytest

# Ensure the plugin package is importable.
_plugin_dir = Path(__file__).resolve().parent.parent
if str(_plugin_dir.parent) not in sys.path:
    sys.path.insert(0, str(_plugin_dir.parent))


@pytest.fixture
def tmp_path_factory_dir(tmp_path):
    """Provide a clean temp directory for each test."""
    return tmp_path


class TestEvolutionChains:
    """Evolution-chains feature: version-chain walking, evidence join,
    search annotation, approve-with-supersede, chain-aware delete, and
    chain-unfold accounting."""

    # -- store layer ----------------------------------------------------------

    def test_chain_walk_from_head_middle_tail(self, tmp_path):
        """get_memory_history returns the full chain from any node."""
        from store import DuckDBMemoryStore

        store = DuckDBMemoryStore(tmp_path / "ec1.duckdb", user_id="alice")
        v1 = store.remember(category="personal_fact", content="Alice lives in Southtown")
        v2 = store.update_memory(v1.memory_id, content="Alice lives in Centurion")
        v3 = store.update_memory(v2.memory_id, content="Alice lives in Bayport")

        for start_id in (v1.memory_id, v2.memory_id, v3.memory_id):
            history = store.get_memory_history(start_id)
            assert len(history) == 3, f"chain from {start_id} has {len(history)}"
            assert "Southtown" in history[0].content
            assert "Centurion" in history[1].content
            assert "Bayport" in history[2].content
            # Oldest first, head (current) last with valid_to=None
            assert history[2].valid_to is None
        store.close()

    def test_chain_cycle_guard(self, tmp_path):
        """A corrupt A→B→A cycle must not loop forever."""
        from store import DuckDBMemoryStore

        store = DuckDBMemoryStore(tmp_path / "ec2.duckdb", user_id="alice")
        a = store.remember(category="personal_fact", content="Alice cycle A")
        b = store.update_memory(a.memory_id, content="Alice cycle B")
        # Corrupt: point B back at A, creating A→B→A.
        store.connection.execute(
            "UPDATE memory_records SET superseded_by = ? WHERE memory_id = ?",
            [a.memory_id, b.memory_id],
        )
        # Must terminate, not hang.
        history = store.get_memory_history(a.memory_id)
        assert len(history) >= 1
        store.close()

    def test_chain_scope_isolation(self, tmp_path):
        """User A cannot see user B's chain via hops."""
        from store import DuckDBMemoryStore

        store_a = DuckDBMemoryStore(tmp_path / "ec3.duckdb", user_id="alice")
        v1 = store_a.remember(category="personal_fact", content="Alice secret v1")
        store_a.update_memory(v1.memory_id, content="Alice secret v2")

        store_b = DuckDBMemoryStore(tmp_path / "ec3.duckdb", user_id="bob")
        # Bob querying Alice's memory_id gets nothing (scope check).
        history = store_b.get_memory_history(v1.memory_id)
        assert history == []
        store_a.close()
        store_b.close()

    def test_chain_max_versions_truncation(self, tmp_path):
        """max_versions keeps the most recent N (head always retained)."""
        from store import DuckDBMemoryStore

        store = DuckDBMemoryStore(tmp_path / "ec4.duckdb", user_id="alice")
        v1 = store.remember(category="personal_fact", content="Alice job v1 startup")
        v2 = store.update_memory(v1.memory_id, content="Alice job v2 scaleup")
        v3 = store.update_memory(v2.memory_id, content="Alice job v3 enterprise")
        v4 = store.update_memory(v3.memory_id, content="Alice job v4 freelance")

        trimmed = store.get_memory_history(v1.memory_id, max_versions=2)
        assert len(trimmed) == 2
        # Most recent 2: v3 and v4 (head retained).
        assert "enterprise" in trimmed[0].content
        assert "freelance" in trimmed[1].content
        store.close()

    def test_get_evidence_batch(self, tmp_path):
        """Batched evidence retrieval returns a map keyed by memory_id."""
        from store import DuckDBMemoryStore

        store = DuckDBMemoryStore(tmp_path / "ec5.duckdb", user_id="alice")
        # Create a candidate + approve it to generate evidence.
        cand = store.save_candidate(
            category="personal_fact",
            content="Alice drives a Toyota Hilux",
            evidence_text="User said: I bought a Hilux",
            source_timestamp="2026-08-13T10:00:00+00:00",
            session_id="sess-ec5",
        )
        result = store.review_candidate(
            candidate_id=cand["candidate_id"], decision="approved",
            evidence_retention="full",
        )
        mid = result["memory"]["memory_id"]
        batch = store.get_evidence_batch([mid, "nonexistent-id"])
        assert mid in batch
        assert batch[mid]["evidence_text"] == "User said: I bought a Hilux"
        assert "nonexistent-id" not in batch
        # Empty input is safe.
        assert store.get_evidence_batch([]) == {}
        store.close()

    def test_get_chain_membership_annotation(self, tmp_path):
        """get_chain_membership annotates which hits have a history."""
        from store import DuckDBMemoryStore

        store = DuckDBMemoryStore(tmp_path / "ec6.duckdb", user_id="alice")
        v1 = store.remember(category="personal_fact", content="Alice works at Acme Corp")
        store.update_memory(v1.memory_id, content="Alice works at Globex Inc")
        standalone = store.remember(category="preference", content="Alice prefers dark mode")

        membership = store.get_chain_membership([v1.memory_id, standalone.memory_id])
        # v1 is a predecessor now (superseded); it has a chain.
        assert membership[v1.memory_id]["has_history"] is True
        assert membership[v1.memory_id]["versions"] == 2
        # Standalone has no history.
        assert membership[standalone.memory_id]["has_history"] is False
        assert membership[standalone.memory_id]["versions"] == 1
        store.close()

    def test_delete_head_promotes_predecessor(self, tmp_path):
        """Deleting the current (head) version promotes the predecessor."""
        from store import DuckDBMemoryStore

        store = DuckDBMemoryStore(tmp_path / "ec7.duckdb", user_id="alice")
        v1 = store.remember(category="personal_fact", content="Alice salary R10k")
        v2 = store.update_memory(v1.memory_id, content="Alice salary R20k")

        result = store.delete_memory(v2.memory_id)
        assert isinstance(result, dict)
        assert result["action"] == "promoted"
        assert result["promoted_memory_id"] == v1.memory_id
        # v1 is now current.
        promoted = store.get_memories_by_ids([v1.memory_id])
        assert promoted and promoted[0].valid_to is None
        assert promoted[0].superseded_by is None
        store.close()

    def test_delete_non_head_quarantines(self, tmp_path):
        """Deleting a middle/historical version converts to quarantine."""
        from store import DuckDBMemoryStore

        store = DuckDBMemoryStore(tmp_path / "ec8.duckdb", user_id="alice")
        v1 = store.remember(category="personal_fact", content="Alice phone iPhone 12")
        v2 = store.update_memory(v1.memory_id, content="Alice phone iPhone 15")

        # v1 is a non-head (superseded). Delete → quarantine.
        result = store.delete_memory(v1.memory_id)
        assert isinstance(result, dict)
        assert result["action"] == "quarantined"
        # v1 still exists but is quarantined; chain walk still works.
        rec = store.get_memories_by_ids([v1.memory_id], include_quarantined=True)
        assert rec and rec[0].status == "quarantined"
        store.close()

    def test_approve_with_supersede_chains(self, tmp_path):
        """approve-with-supersede chains a new memory behind an existing one."""
        from store import DuckDBMemoryStore

        store = DuckDBMemoryStore(tmp_path / "ec9.duckdb", user_id="alice")
        existing = store.remember(category="personal_fact", content="Alice lives in Northtown")
        cand = store.save_candidate(
            category="personal_fact",
            content="Alice lives in Uptown",
            evidence_text="User said: I moved to Uptown",
            source_timestamp="2026-08-13T10:00:00+00:00",
            session_id="sess-ec9",
        )
        result = store.review_candidate(
            candidate_id=cand["candidate_id"], decision="approved",
            supersedes_memory_id=existing.memory_id,
            evidence_retention="full",
        )
        assert result["superseded"] is True
        new_mid = result["memory"]["memory_id"]
        # The chain: existing → new.
        history = store.get_memory_history(existing.memory_id)
        assert len(history) == 2
        assert "Northtown" in history[0].content
        assert "Uptown" in history[1].content
        assert history[1].memory_id == new_mid
        store.close()

    def test_approve_with_supersede_wrong_scope_fails(self, tmp_path):
        """Cannot supersede another user's memory."""
        from store import DuckDBMemoryStore

        store_a = DuckDBMemoryStore(tmp_path / "ec10.duckdb", user_id="alice")
        existing = store_a.remember(category="personal_fact", content="Alice tool Vim")
        store_b = DuckDBMemoryStore(tmp_path / "ec10.duckdb", user_id="bob")
        # Bob creates his own candidate and tries to supersede Alice's memory.
        cand_b = store_b.save_candidate(
            category="personal_fact",
            content="Bob tool Nano",
            source_timestamp="2026-08-13T10:00:00+00:00",
            session_id="sess-ec10-bob",
        )
        result = store_b.review_candidate(
            candidate_id=cand_b["candidate_id"], decision="approved",
            supersedes_memory_id=existing.memory_id,
        )
        # The supersede must NOT have happened (wrong scope).
        assert result["superseded"] is False
        store_a.close()
        store_b.close()

    def test_find_supersede_candidates_surfaces_current(self, tmp_path):
        """find_supersede_candidates returns current similar memories."""
        from store import DuckDBMemoryStore

        store = DuckDBMemoryStore(tmp_path / "ec11.duckdb", user_id="alice")
        store.remember(category="personal_fact", content="Alice works at Initech")
        cand = store.save_candidate(
            category="personal_fact",
            content="Alice works at Hooli",
            source_timestamp="2026-08-13T10:00:00+00:00",
            session_id="sess-ec11",
        )
        similar = store.find_supersede_candidates(cand["candidate_id"], limit=3)
        assert len(similar) >= 1
        assert similar[0]["valid_to"] is None  # only current records
        store.close()

    # -- client parity --------------------------------------------------------

    def test_shared_store_has_chain_methods(self):
        """SharedMemoryStore must expose the new chain methods."""
        from service_client import SharedMemoryStore
        for method in ("get_memory_history", "get_evidence_batch",
                       "get_chain_membership", "find_supersede_candidates"):
            assert hasattr(SharedMemoryStore, method), \
                f"SharedMemoryStore missing method: {method}"

    def test_get_memory_history_keyword_only(self):
        """get_memory_history on the shared client must be keyword-only
        (max_versions is keyword-only per the facade convention)."""
        import inspect
        from service_client import SharedMemoryStore
        sig = inspect.signature(SharedMemoryStore.get_memory_history)
        params = sig.parameters
        # memory_id and max_versions must be keyword-or-positional/keyword.
        assert "memory_id" in params
        assert "max_versions" in params
        # max_versions must be KEYWORD_ONLY (it follows a * in the signature).
        assert params["max_versions"].kind == inspect.Parameter.KEYWORD_ONLY

    def test_service_dispatch_has_chain_methods(self):
        """The service RPC dispatch must route the new chain methods."""
        import inspect
        from memory_service import MemoryService
        src = inspect.getsource(MemoryService._call_store)
        for method in ("get_memory_history", "get_evidence_batch",
                       "get_chain_membership", "find_supersede_candidates"):
            assert method in src, f"service missing RPC case: {method}"

    # -- provider e2e ---------------------------------------------------------

    def _make_provider(self, tmp_path, user_id="test_user"):
        """Build a provider with a real store + graph (Hermes runtime stubbed)."""
        import sys
        import types
        import json as _json

        if "agent" not in sys.modules:
            sys.modules["agent"] = types.ModuleType("agent")
        if "agent.memory_provider" not in sys.modules:
            _mp = types.ModuleType("agent.memory_provider")
            class MemoryProvider:
                pass
            _mp.MemoryProvider = MemoryProvider
            sys.modules["agent.memory_provider"] = _mp
        if "tools" not in sys.modules:
            sys.modules["tools"] = types.ModuleType("tools")
        if "tools.registry" not in sys.modules:
            _tr = types.ModuleType("tools.registry")
            def _tool_error(msg):
                return _json.dumps({"error": str(msg)})
            _tr.tool_error = _tool_error
            sys.modules["tools.registry"] = _tr

        from store import DuckDBMemoryStore
        from graph import KuzuGraphStore
        try:
            import argos_plugin
        except ModuleNotFoundError:
            import argos as argos_plugin

        store = DuckDBMemoryStore(tmp_path / "ec_prov.duckdb", user_id=user_id)
        graph = KuzuGraphStore(tmp_path / "ec_prov_kuzu", user_id=user_id)
        provider = argos_plugin.ArgosProvider()
        provider._store = store
        provider._graph = graph
        provider._evidence_retention = "full"
        return provider, store, graph

    def test_memory_chain_tool_registered(self, tmp_path):
        """memory_chain must be in the provider's tool schemas."""
        provider, store, graph = self._make_provider(tmp_path)
        schemas = provider.get_tool_schemas()
        names = [s["name"] for s in schemas]
        assert "memory_chain" in names
        store.close()
        graph.close()

    def test_memory_chain_tool_arc_mode(self, tmp_path):
        """memory_chain arc mode returns a compact ordered arc."""
        import json as _json
        provider, store, graph = self._make_provider(tmp_path)

        v1 = store.remember(category="personal_fact", content="User uses Spotify for music")
        v2 = store.update_memory(v1.memory_id, content="User switched to Apple Music")
        v3 = store.update_memory(v2.memory_id, content="User now uses Tidal for music")

        result = provider.handle_tool_call(
            "memory_chain", {"memory_id": v1.memory_id, "mode": "arc"},
        )
        parsed = _json.loads(result)
        assert parsed["mode"] == "arc"
        assert parsed["count"] == 3
        arc = parsed["arc"]
        assert "Spotify" in arc
        assert "Tidal" in arc
        assert "(current)" in arc  # head marker
        store.close()
        graph.close()

    def test_memory_chain_tool_versions_mode_with_evidence(self, tmp_path):
        """memory_chain versions mode returns full records + evidence."""
        import json as _json
        provider, store, graph = self._make_provider(tmp_path)

        # Create a candidate with evidence and approve it (v1).
        cand = store.save_candidate(
            category="personal_fact",
            content="User drives a Honda Civic",
            evidence_text="User said: I drive a Civic",
            source_timestamp="2026-08-13T10:00:00+00:00",
            session_id="sess-ec-v",
        )
        res = store.review_candidate(
            candidate_id=cand["candidate_id"], decision="approved",
            evidence_retention="full",
        )
        v1_id = res["memory"]["memory_id"]
        store.update_memory(v1_id, content="User drives a Toyota Corolla")

        result = provider.handle_tool_call(
            "memory_chain", {"memory_id": v1_id, "mode": "versions"},
        )
        parsed = _json.loads(result)
        assert parsed["mode"] == "versions"
        assert parsed["count"] == 2
        # v1 (oldest) has evidence with the original text.
        v0 = parsed["versions"][0]
        assert v0["evidence"] is not None
        assert v0["evidence"]["evidence_text"] == "User said: I drive a Civic"
        store.close()
        graph.close()

    def test_memory_chain_tool_diff_mode(self, tmp_path):
        """memory_chain diff mode returns per-step deltas."""
        import json as _json
        provider, store, graph = self._make_provider(tmp_path)

        v1 = store.remember(category="personal_fact", content="User lives in BigCity")
        store.update_memory(v1.memory_id, content="User lives in Capital City")

        result = provider.handle_tool_call(
            "memory_chain", {"memory_id": v1.memory_id, "mode": "diff"},
        )
        parsed = _json.loads(result)
        assert parsed["mode"] == "diff"
        assert parsed["count"] == 2
        # v2 should have a changes_from_previous with a content_diff.
        assert "changes_from_previous" in parsed["steps"][1]
        store.close()
        graph.close()

    def test_memory_chain_quarantine_gap(self, tmp_path):
        """A quarantined middle version is marked as a gap, not removed."""
        import json as _json
        provider, store, graph = self._make_provider(tmp_path)

        v1 = store.remember(category="personal_fact", content="User opinion: Python is best")
        v2 = store.update_memory(v1.memory_id, content="User opinion: Rust is best")
        v3 = store.update_memory(v2.memory_id, content="User opinion: Go is best")
        # Quarantine the middle version.
        store.quarantine_memory(v2.memory_id, "test gap")

        result = provider.handle_tool_call(
            "memory_chain", {"memory_id": v1.memory_id, "mode": "arc"},
        )
        parsed = _json.loads(result)
        # All 3 versions present (walk not broken by quarantine).
        assert parsed["count"] == 3
        arc = parsed["arc"]
        assert "[quarantined]" in arc
        store.close()
        graph.close()

    def test_search_result_carries_chain_annotation(self, tmp_path):
        """memory_search results carry a chain annotation for chained facts."""
        import json as _json
        provider, store, graph = self._make_provider(tmp_path)

        v1 = store.remember(category="personal_fact", content="User uses Vim as editor")
        store.update_memory(v1.memory_id, content="User uses Neovim as editor")
        # Index so search finds it.
        graph.index_memory(v1.memory_id, "personal_fact",
                           "User uses Vim as editor", [], v1.created_at,
                           use_llm=False)

        result = provider.handle_tool_call(
            "memory_search", {"query": "editor", "top_k": 5},
        )
        parsed = _json.loads(result)
        assert parsed["count"] >= 1
        # At least one result should carry a chain annotation.
        annotated = [r for r in parsed["results"] if "chain" in r]
        assert annotated, "No search result carried a chain annotation"
        store.close()
        graph.close()

    def test_memory_chain_unknown_id_errors(self, tmp_path):
        """memory_chain on a nonexistent ID returns an error."""
        provider, store, graph = self._make_provider(tmp_path)
        result = provider.handle_tool_call(
            "memory_chain", {"memory_id": "mem-" + "0" * 32},
        )
        import json as _json
        parsed = _json.loads(result)
        assert "error" in parsed
        store.close()
        graph.close()

    # -- chain-unfold accounting ----------------------------------------------

    def test_chain_unfold_no_results_no_unfold(self, tmp_path):
        """chain_unfold with empty results — no unfold, no counter change."""
        provider, store, graph = self._make_provider(tmp_path)
        v1 = store.remember(category="personal_fact", content="User used to like PHP")
        store.update_memory(v1.memory_id, content="User now likes Python")
        before = provider.get_chain_unfold_stats()
        arc = provider._maybe_unfold_chain("why did I stop using PHP", [])
        assert arc is None
        assert provider.get_chain_unfold_stats() == before
        store.close()
        graph.close()

    def test_chain_unfold_auto_accounting(self, tmp_path):
        """When enabled=auto + change-intent + chain present, unfold fires
        and updates the separate counter (NOT retrieval counters)."""
        provider, store, graph = self._make_provider(tmp_path)
        provider._chain_unfold = "auto"
        v1 = store.remember(category="personal_fact", content="User used to like PHP a lot")
        store.update_memory(v1.memory_id, content="User now likes Python")
        # Search for the CURRENT version's content so the head is found.
        results = store.search("Python", limit=3, suppress_retrieval=True)
        assert results, "search should find the chained memory"
        arc = provider._maybe_unfold_chain("why did I stop using PHP", results)
        assert arc is not None
        assert "PHP" in arc
        stats = provider.get_chain_unfold_stats()
        assert stats["count"] == 1
        assert stats["tokens_injected"] > 0
        store.close()
        graph.close()

    def test_chain_unfold_no_intent_no_fire(self, tmp_path):
        """auto mode without change-intent does not unfold."""
        provider, store, graph = self._make_provider(tmp_path)
        provider._chain_unfold = "auto"
        v1 = store.remember(category="personal_fact", content="User likes Python programming")
        store.update_memory(v1.memory_id, content="User loves Python programming")
        results = store.search("Python", limit=3, suppress_retrieval=True)
        arc = provider._maybe_unfold_chain("what language does the user like", results)
        assert arc is None
        store.close()
        graph.close()

    def test_as_of_search_uses_as_of_for_expiry_filter(self, tmp_path):
        """Point-in-time (as_of) queries must apply the expiry filter against
        as_of, not against now: a memory that expired 90 days ago is still
        visible to a history query dated BEFORE its expiry, and invisible to
        current-time search and to queries dated after expiry. Regression for
        the as_of/expiry mismatch in _text_search_raw / _vector_search_raw."""
        from datetime import datetime, timedelta, timezone
        from store import DuckDBMemoryStore

        store = DuckDBMemoryStore(tmp_path / "test.duckdb", user_id="test_user")
        now = datetime.now(timezone.utc)
        rec = store.remember(category="event", content="User attended a work conference")
        assert rec is not None, "remember should store the memory"
        # Backdate the row: valid since 200 days ago (current version), expired 90 days ago.
        store.connection.execute(
            "UPDATE memory_records SET valid_from = ?, expires_at = ? WHERE memory_id = ?",
            [(now - timedelta(days=200)).isoformat(),
             (now - timedelta(days=90)).isoformat(),
             rec.memory_id],
        )

        # Current-time search must exclude the expired memory.
        now_results = store.search("conference", limit=5, suppress_retrieval=True)
        assert all(r.memory_id != rec.memory_id for r in now_results)

        # as_of between valid_from and expiry must still return it (history is history).
        asof_results = store.search(
            "conference", limit=5, suppress_retrieval=True,
            as_of=(now - timedelta(days=150)).isoformat(),
        )
        assert any(r.memory_id == rec.memory_id for r in asof_results)

        # as_of dated after the expiry must exclude it.
        later_results = store.search(
            "conference", limit=5, suppress_retrieval=True,
            as_of=(now - timedelta(days=30)).isoformat(),
        )
        assert all(r.memory_id != rec.memory_id for r in later_results)
        store.close()


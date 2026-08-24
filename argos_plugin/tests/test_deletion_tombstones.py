"""Deletion tombstones (shipped 2026-08-24).

Enforcement version of the atlas deletion canary. The observational test
(test_contradiction_matrix.py::test_deletion_step_5_6_refeed_resurrection)
proved on 24/8 that hard-delete + re-feed resurrects the fact — the atlas's
predicted step-6 failure. The tombstone design is now decided and shipped:

- delete_memory() fingerprints hard-deleted content into deletion_tombstones
- remember() blocks re-creation of tombstoned content (returns None, logged)
- purge_tombstone() is the explicit user escape hatch
- quarantined chain versions are untouched: they were never resurrectable

Design notes:
- hash is case/whitespace-insensitive (normalised before sha256)
- scope = (content_hash, category, user_scope)
- promoted-predecessor path also tombstones the removed head's content,
  while history stays intact for as_of queries
"""
import sys
import types

import pytest


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
        import json as _json
        _tr = types.ModuleType("tools.registry")
        _tr.tool_error = lambda msg: _json.dumps({"error": str(msg)})
        sys.modules["tools.registry"] = _tr


_stub_agent_modules()

from store import DuckDBMemoryStore  # noqa: E402


CANARY = "Alex lives in Berlin."


@pytest.fixture()
def store(tmp_path):
    # embedder=None keeps these deterministic and fast: exact/substring dedup
    # layers still run; the tombstone check is pure SQL, no embeddings needed.
    s = DuckDBMemoryStore(tmp_path / "tomb.duckdb", user_id="test_user", embedder=None)
    yield s
    s.close()


def _canary(s):
    rec = s.remember(category="personal_fact", content=CANARY)
    assert rec is not None
    return rec


class TestTombstoneBasics:
    def test_delete_creates_tombstone(self, store):
        rec = _canary(store)
        result = store.delete_memory(rec.memory_id)
        assert result == {"deleted": True, "action": "deleted"}
        hit = store.tombstone_check(CANARY, "personal_fact")
        assert hit is not None, "hard delete must fingerprint content"
        assert hit["reason"] == "user_delete"

    def test_refeed_blocked_after_delete(self, store):
        """The enforcing flip of the observational resurrection canary."""
        rec = _canary(store)
        store.delete_memory(rec.memory_id)

        refed = store.remember(category="personal_fact", content=CANARY,
                               dedup=True, source="explicit")
        assert refed is None, "re-feed must be blocked by the tombstone"

        hits = store.search("Berlin", limit=10)
        assert all(CANARY != h.content for h in hits), \
            "deleted fact must stay gone after re-feed attempt"

    def test_refeed_blocked_case_whitespace_insensitive(self, store):
        rec = store.remember(category="personal_fact", content=CANARY)
        store.delete_memory(rec.memory_id)
        variant = "alex   lives in BERLIN."
        assert store.remember(category="personal_fact", content=variant) is None

    def test_other_content_and_category_unaffected(self, store):
        rec = _canary(store)
        store.delete_memory(rec.memory_id)
        other = store.remember(
            category="personal_fact",
            content="Alex moved to Lisbon in March 2026.",
        )
        assert other is not None, "unrelated content must pass"
        same_text_other_cat = store.remember(
            category="context_note", content=CANARY,
        )
        assert same_text_other_cat is not None, \
            "same text in a different category must pass"

    def test_purge_tombstone_allows_refeed(self, store):
        rec = _canary(store)
        store.delete_memory(rec.memory_id)
        assert store.remember(category="personal_fact", content=CANARY) is None

        assert store.purge_tombstone(CANARY, "personal_fact") is True
        refed = store.remember(category="personal_fact", content=CANARY)
        assert refed is not None, "purged tombstone must allow re-creation"

    def test_tombstone_scoped_by_user(self, tmp_path):
        """A tombstone in user A's scope must not block user B."""
        from store import DuckDBMemoryStore as S
        sa = S(tmp_path / "scope.duckdb", user_id="alice", embedder=None)
        sb = S(tmp_path / "scope2.duckdb", user_id="bob", embedder=None)
        rec_a = sa.remember(category="personal_fact", content=CANARY)
        sa.delete_memory(rec_a.memory_id)
        rec_b = sb.remember(category="personal_fact", content=CANARY)
        assert rec_b is not None, "bob must be unaffected by alice's tombstone"
        sa.close()
        sb.close()


class TestTombstoneChainPaths:
    def test_promoted_path_tombstones_removed_head(self, store):
        """delete(head-with-predecessor): predecessor promoted, head content
        tombstoned so re-feed of the HEAD value stays blocked."""
        v1 = _canary(store)
        v2 = store.update_memory(v1.memory_id, content="Alex lives in Lisbon.")
        assert v2 is not None
        result = store.delete_memory(v2.memory_id)
        assert result["action"] == "promoted"
        # predecessor came back to life
        assert store.tombstone_check("Alex lives in Lisbon.", "personal_fact") \
            is not None, "removed head must be tombstoned"
        assert store.tombstone_check(CANARY, "personal_fact") is None, \
            "promoted predecessor must NOT be tombstoned"
        refed_head = store.remember(category="personal_fact",
                                    content="Alex lives in Lisbon.")
        assert refed_head is None

    def test_quarantined_path_no_tombstone_needed(self, store):
        """Middle-of-chain delete -> quarantine; quarantine was never
        resurrectable via remember(), so no tombstone must be recorded."""
        v1 = _canary(store)
        v2 = store.update_memory(v1.memory_id, content="Alex lives in Lisbon.")
        v3 = store.update_memory(v2.memory_id, content="Alex lives in Porto.")
        result = store.delete_memory(v2.memory_id)  # middle version
        assert result["action"] == "quarantined"
        assert store.tombstone_check("Alex lives in Lisbon.", "personal_fact") \
            is None
        # current version still retrievable
        hits = store.search("Porto", limit=5)
        assert any(v3.content == h.content for h in hits)

    def test_missing_memory_returns_false(self, store):
        assert store.delete_memory("mem-nonexistent") is False

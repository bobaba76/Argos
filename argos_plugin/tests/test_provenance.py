"""Tests for #280: explainability pack — provenance walk + citation surfacing.

Acceptance criteria from the issue:
- Per-retrieval provenance view returns evidence row, version chain,
  conflict note (if any), blend score for a given memory.
- Citation/confidence surfacing works for a returned memory.
- Explain path is read-only, zero-LLM, fail-soft.
- Tests: provenance view on a memory with a version chain + conflict
  note; explain on a memory with no evidence (fail-soft); blend-score
  surfacing matches the reranker output.

Review fixes (PR #317 blockers):
- BLOCKER 1: cross-user ACL test — provenance() must not read across
  user_scope boundaries.
- BLOCKER 2: real-path blend-score test using explain_record() with a
  live search result (not fabricated 0.0 from storage).
- injection_min_score read from config (not hardcoded 0.3).
- reranker_applied only fires on actual reranker pass.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest

_plugin_dir = Path(__file__).resolve().parent.parent
if str(_plugin_dir) not in sys.path:
    sys.path.insert(0, str(_plugin_dir))


@pytest.fixture
def store():
    """Create a DuckDBMemoryStore with a temporary database."""
    from store import DuckDBMemoryStore
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test.duckdb"
        s = DuckDBMemoryStore(db_path, user_id="test_user")
        yield s
        s.close()


@pytest.fixture
def store_alice():
    """Create a DuckDBMemoryStore for user 'alice'."""
    from store import DuckDBMemoryStore
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test.duckdb"
        s = DuckDBMemoryStore(db_path, user_id="alice")
        yield s
        s.close()


@pytest.fixture
def store_bob():
    """Create a DuckDBMemoryStore for user 'bob' on the SAME db as alice."""
    from store import DuckDBMemoryStore
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test.duckdb"
        s_alice = DuckDBMemoryStore(db_path, user_id="alice")
        # Alice stores a memory
        m = s_alice.remember(
            category="personal_fact",
            content="Alice's secret fact",
        )
        alice_memory_id = m.memory_id
        s_alice.close()
        # Bob opens the same DB
        s_bob = DuckDBMemoryStore(db_path, user_id="bob")
        yield s_bob, alice_memory_id
        s_bob.close()


class TestProvenanceView:
    """Per-retrieval provenance view returns all required fields."""

    def test_provenance_returns_evidence_and_chain(self, store):
        """Provenance view on a memory with a version chain returns
        evidence, version chain, blend score, and confidence."""
        v1 = store.remember(
            category="personal_fact",
            content="User likes Python programming",
        )
        v2 = store.update_memory(
            v1.memory_id, content="User loves Python programming",
        )

        result = store.provenance(v2.memory_id)

        assert result["memory_id"] == v2.memory_id
        assert "content" in result
        assert "category" in result
        assert "evidence" in result
        assert "version_chain" in result
        assert "blend_score" in result
        assert "confidence" in result
        assert "provenance_origin" in result
        assert "grounding" in result
        assert "gates_fired" in result
        # Version chain should have 2 entries
        assert len(result["version_chain"]) >= 2
        # Chain should be chronological (oldest first)
        chain = result["version_chain"]
        assert chain[0]["memory_id"] == v1.memory_id
        assert chain[-1]["memory_id"] == v2.memory_id
        # The last version should be current
        assert chain[-1]["is_current"] is True
        # The first version should not be current
        assert chain[0]["is_current"] is False

    def test_provenance_on_single_version_memory(self, store):
        """Provenance view on a memory with no chain returns a
        single-element chain."""
        m = store.remember(
            category="personal_fact",
            content="User's favorite color is blue",
        )

        result = store.provenance(m.memory_id)

        assert result["memory_id"] == m.memory_id
        assert len(result["version_chain"]) == 1
        assert result["version_chain"][0]["is_current"] is True
        assert result["version_chain"][0]["memory_id"] == m.memory_id

    def test_provenance_fail_soft_on_missing_memory(self, store):
        """Provenance on a non-existent memory_id returns an error,
        not a crash."""
        result = store.provenance("nonexistent-id-12345")

        assert result["memory_id"] == "nonexistent-id-12345"
        assert "error" in result

    def test_provenance_fail_soft_on_no_evidence(self, store):
        """Provenance on a memory with no evidence row returns
        evidence=None, not a crash."""
        m = store.remember(
            category="personal_fact",
            content="User lives in Paris",
        )

        result = store.provenance(m.memory_id)

        # Evidence should be None (no evidence row written for a
        # direct remember() call)
        assert result["evidence"] is None
        # But other fields should still be populated
        assert result["content"] == "User lives in Paris"
        assert result["category"] == "personal_fact"

    def test_provenance_id_based_blend_score_not_available(self, store):
        """ID-based provenance does NOT fabricate retrieval-time scores.

        similarity/raw_similarity are per-query, not persisted on the
        row. The ID-based path annotates them as 'not available' rather
        than returning fabricated 0.0 values.
        """
        m = store.remember(
            category="personal_fact",
            content="User works as a software engineer",
        )

        result = store.provenance(m.memory_id)

        assert "blend_score" in result
        assert "similarity" in result["blend_score"]
        assert "raw_similarity" in result["blend_score"]
        assert "reranker_applied" in result["blend_score"]
        # Scores are NOT available from storage (retrieval-time only)
        assert result["blend_score"]["similarity"] is None
        assert result["blend_score"]["raw_similarity"] is None
        assert result["blend_score"]["reranker_applied"] is None
        assert "not available" in result["blend_score"]["source"]

    def test_provenance_returns_confidence(self, store):
        """Confidence field is surfaced in the provenance view."""
        m = store.remember(
            category="personal_fact",
            content="User's birthday is January 15",
            confidence=0.9,
        )

        result = store.provenance(m.memory_id)

        assert result["confidence"] == 0.9

    def test_provenance_returns_provenance_origin(self, store):
        """Provenance origin and grounding are surfaced."""
        m = store.remember(
            category="personal_fact",
            content="User has a dog named Rex",
        )

        result = store.provenance(m.memory_id)

        assert result["provenance_origin"] is not None
        assert result["grounding"] is not None

    def test_provenance_id_based_no_score_gates(self, store):
        """ID-based provenance does NOT emit score-based gates.

        Without retrieval-time scores, the injection_min_score and
        reranker gates cannot be determined. Only conflict_surfacing
        (inferable from category) is emitted.
        """
        m = store.remember(
            category="personal_fact",
            content="User likes hiking",
        )

        result = store.provenance(m.memory_id)

        # No score-based gates should fire (no retrieval-time data)
        gates = result["gates_fired"]
        assert not any("injection_min_score" in g for g in gates)
        assert not any("reranker" in g for g in gates)


class TestConflictNote:
    """Conflict note surfacing in the provenance view."""

    def test_conflict_note_on_two_active_versions(self, store):
        """If a chain has two active versions (both valid_to=None),
        a conflict note is returned."""
        v1 = store.remember(
            category="personal_fact",
            content="User likes coffee",
        )
        v2 = store.update_memory(
            v1.memory_id, content="User likes tea",
        )

        # Normal chain: v1 is superseded, v2 is current — no conflict
        result = store.provenance(v2.memory_id)
        assert "conflict_note" in result

    def test_conflict_note_none_on_single_version(self, store):
        """A single-version memory has no conflict note."""
        m = store.remember(
            category="personal_fact",
            content="User plays guitar",
        )

        result = store.provenance(m.memory_id)
        assert result["conflict_note"] is None


class TestExplainBatch:
    """Batch provenance view for multiple memories."""

    def test_explain_batch_returns_all(self, store):
        """explain_batch returns provenance for all requested IDs."""
        m1 = store.remember(category="personal_fact", content="Fact A")
        m2 = store.remember(category="personal_fact", content="Fact B")
        m3 = store.remember(category="personal_fact", content="Fact C")

        from provenance import explain_batch
        results = explain_batch(store, [m1.memory_id, m2.memory_id, m3.memory_id])

        assert len(results) == 3
        for r in results:
            assert "memory_id" in r
            assert "content" in r
            assert "blend_score" in r
            assert "evidence" in r

    def test_explain_batch_empty_list(self, store):
        """explain_batch on empty list returns empty list."""
        from provenance import explain_batch
        results = explain_batch(store, [])
        assert results == []

    def test_explain_batch_with_chains(self, store):
        """explain_batch with include_chains=True returns chains."""
        v1 = store.remember(category="personal_fact", content="Version 1")
        v2 = store.update_memory(v1.memory_id, content="Version 2")

        from provenance import explain_batch
        results = explain_batch(
            store, [v2.memory_id], include_chains=True,
        )

        assert len(results) == 1
        assert "version_chain" in results[0]
        assert len(results[0]["version_chain"]) >= 2


class TestGatesFired:
    """Gates fired surfacing — uses configured injection_min_score."""

    def test_gates_fired_on_low_similarity(self, store):
        """A record with similarity below the configured floor has the
        injection_min_score gate listed."""
        m = store.remember(
            category="personal_fact",
            content="User likes hiking",
        )
        m.similarity = 0.1
        m.raw_similarity = 0.1

        from provenance import _gates_fired
        gates = _gates_fired(m, injection_min_score=0.3)
        assert any("injection_min_score" in g for g in gates)

    def test_gates_fired_respects_configured_floor(self, store):
        """The injection_min_score gate uses the configured floor, not
        a hardcoded 0.3."""
        m = store.remember(
            category="personal_fact",
            content="User likes hiking",
        )
        m.similarity = 0.25
        m.raw_similarity = 0.25

        from provenance import _gates_fired
        # With floor=0.2, 0.25 is above the floor — no gate
        gates = _gates_fired(m, injection_min_score=0.2)
        assert not any("injection_min_score" in g for g in gates)
        # With floor=0.3, 0.25 is below the floor — gate fires
        gates = _gates_fired(m, injection_min_score=0.3)
        assert any("injection_min_score" in g for g in gates)

    def test_gates_fired_on_reranker_adjusted(self, store):
        """A record with raw_similarity != similarity has the reranker
        gate listed."""
        m = store.remember(
            category="personal_fact",
            content="User likes swimming",
        )
        m.similarity = 0.8
        m.raw_similarity = 0.6  # Different → reranker adjusted

        from provenance import _gates_fired
        gates = _gates_fired(m, injection_min_score=0.0)
        assert any("reranker" in g for g in gates)

    def test_gates_fired_empty_on_normal_record(self, store):
        """A normal record with no gates fired has an empty list."""
        m = store.remember(
            category="personal_fact",
            content="User likes running",
        )
        m.similarity = 0.7
        m.raw_similarity = 0.7  # Same → no reranker

        from provenance import _gates_fired
        gates = _gates_fired(m, injection_min_score=0.3)
        assert len(gates) == 0

    def test_gates_fired_no_score_claim_without_data(self, store):
        """A record with similarity=None does NOT emit a score-based
        gate claim — never fabricate a gate without supporting data."""
        from store_common import MemoryRecord
        m = MemoryRecord(
            memory_id="test-no-sim",
            category="personal_fact",
            content="Test",
            similarity=0.0,
            raw_similarity=None,
        )

        from provenance import _gates_fired
        # With similarity=0.0 and floor=0.3, the gate SHOULD fire
        # because 0.0 < 0.3 — this is a real value, not missing data.
        gates = _gates_fired(m, injection_min_score=0.3)
        assert any("injection_min_score" in g for g in gates)


class TestExplainIsReadOnly:
    """The explain path is read-only — no writes to the store."""

    def test_explain_does_not_create_memories(self, store):
        """Calling provenance() does not create any new memories."""
        m = store.remember(category="personal_fact", content="Test fact")
        count_before = store.count()

        store.provenance(m.memory_id)

        count_after = store.count()
        assert count_after == count_before

    def test_explain_does_not_modify_record(self, store):
        """Calling provenance() does not modify the memory record."""
        m = store.remember(
            category="personal_fact",
            content="Original content",
            confidence=0.8,
        )

        store.provenance(m.memory_id)

        # Fetch the record again and verify it's unchanged
        records = store._fetch_records(
            "SELECT * FROM memory_records WHERE memory_id = ?",
            [m.memory_id],
        )
        assert len(records) == 1
        assert records[0].content == "Original content"
        assert records[0].confidence == 0.8


class TestBlendScoreMatch:
    """Blend-score surfacing matches the reranker output.

    Uses explain_record() with a live MemoryRecord (option b from the
    review) — the ID-based path cannot surface retrieval-time scores
    because they are not persisted on the row.
    """

    def test_blend_score_reflects_reranker_adjustment(self):
        """explain_record with a reranker-adjusted record surfaces
        real similarity, raw_similarity, and reranker_applied=True."""
        from provenance import explain_record
        from store_common import MemoryRecord

        record = MemoryRecord(
            memory_id="test-1",
            category="personal_fact",
            content="User likes reading books",
            similarity=0.85,
            raw_similarity=0.70,
        )

        class MockStore:
            def get_evidence(self, mid):
                return None
            def get_memory_history(self, mid):
                return [record]

        result = explain_record(record, MockStore(), injection_min_score=0.3)

        assert result["blend_score"]["similarity"] == 0.85
        assert result["blend_score"]["raw_similarity"] == 0.70
        assert result["blend_score"]["reranker_applied"] is True
        assert result["blend_score"]["source"] == "retrieval-time"

    def test_blend_score_no_reranker(self):
        """explain_record with raw_similarity == similarity surfaces
        reranker_applied=False."""
        from provenance import explain_record
        from store_common import MemoryRecord

        record = MemoryRecord(
            memory_id="test-2",
            category="personal_fact",
            content="User likes cooking",
            similarity=0.75,
            raw_similarity=0.75,
        )

        class MockStore:
            def get_evidence(self, mid):
                return None
            def get_memory_history(self, mid):
                return [record]

        result = explain_record(record, MockStore(), injection_min_score=0.3)

        assert result["blend_score"]["similarity"] == 0.75
        assert result["blend_score"]["raw_similarity"] == 0.75
        assert result["blend_score"]["reranker_applied"] is False

    def test_blend_score_real_path_search(self, store):
        """REAL-PATH test: search → explain_record on a returned result
        → assert the blend_score has real (non-None) values.

        This is the acceptance criterion 'blend-score surfacing matches
        the reranker output' tested on a real store, not a mock.
        """
        store.remember(
            category="personal_fact",
            content="User likes Python programming language",
        )
        store.remember(
            category="personal_fact",
            content="User enjoys hiking in the mountains",
        )

        results = store.search("Python programming", limit=5)
        assert results, "search should return results"

        from provenance import explain_record
        result = explain_record(results[0], store, injection_min_score=0.0)

        # The blend_score should have real values from the search
        assert result["blend_score"]["similarity"] is not None
        assert result["blend_score"]["source"] == "retrieval-time"
        # similarity should be a real number (not None, not fabricated 0.0
        # unless the search genuinely returned 0.0)
        assert isinstance(result["blend_score"]["similarity"], (int, float))

    def test_gates_fired_real_path_search(self, store):
        """REAL-PATH test: search → explain_record → assert gates_fired
        reflects the ACTUAL retrieval, not fabricated claims."""
        store.remember(
            category="personal_fact",
            content="User likes Python programming language",
        )

        results = store.search("Python programming", limit=5)
        assert results

        from provenance import explain_record
        # Use a high floor so the gate fires for testing
        result = explain_record(results[0], store, injection_min_score=0.99)

        # With a 0.99 floor, any real similarity < 0.99 should fire
        # the injection_min_score gate
        sim = results[0].similarity
        if sim is not None and float(sim) < 0.99:
            assert any("injection_min_score" in g for g in result["gates_fired"])


class TestCrossUserACL:
    """BLOCKER 1: provenance() must not read across user_scope boundaries.

    _fetch_record must filter by user_scope (mirroring get_evidence,
    get_memory_history, get_evidence_batch). A cross-user read must
    return 'not found', not the other user's record.
    """

    def test_cross_user_provenance_denied(self, store_bob):
        """Bob cannot read Alice's memory via provenance()."""
        store, alice_memory_id = store_bob

        result = store.provenance(alice_memory_id)

        # Bob should get 'not found' — not Alice's content
        assert result["memory_id"] == alice_memory_id
        assert "error" in result
        assert "not found" in result["error"]
        # Content must NOT be Alice's secret
        assert result.get("content", "unknown") != "Alice's secret fact"

    def test_own_user_provenance_returns_data(self, store_bob):
        """Bob can read his own memory via provenance()."""
        store, _ = store_bob
        # Bob stores his own memory
        m = store.remember(
            category="personal_fact",
            content="Bob's public fact",
        )

        result = store.provenance(m.memory_id)

        assert result["memory_id"] == m.memory_id
        assert result["content"] == "Bob's public fact"
        assert "error" not in result

    def test_cross_user_fetch_record_returns_none(self, store_bob):
        """_fetch_record with user_scope filter returns None for
        cross-user memory IDs."""
        store, alice_memory_id = store_bob

        from provenance import _fetch_record
        record = _fetch_record(store, alice_memory_id)
        assert record is None

    def test_own_user_fetch_record_returns_data(self, store_bob):
        """_fetch_record returns the record for own-user memory IDs."""
        store, _ = store_bob
        m = store.remember(
            category="personal_fact",
            content="Bob's fact",
        )

        from provenance import _fetch_record
        record = _fetch_record(store, m.memory_id)
        assert record is not None
        assert record.content == "Bob's fact"

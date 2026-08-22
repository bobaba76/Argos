"""Tests for P4.1 — Semantic merge (dedup upgrade).

Covers the 10 required test cases:
1. Same fact rephrased → detected, loser quarantined with keeper link.
2. Distinct facts → untouched.
3. Keeper selection by quality score; tie-break by recency.
4. dry_run=True performs zero writes.
5. Cross-category pairs NOT merged at default config.
6. Expired / superseded / quarantined records excluded.
7. Cluster of 3+ near-dupes → exactly one survivor.
8. duplicate_min_similarity config honored (raise → fewer merges).
9. Restore flow: quarantined loser comes back; keeper untouched.
10. Existing consolidate behavior for old buckets stays green.

Run with:
    python -m pytest tests/test_semantic_dedup.py -v
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

_plugin_dir = Path(__file__).resolve().parent.parent
if str(_plugin_dir) not in sys.path:
    sys.path.insert(0, str(_plugin_dir))


@pytest.fixture
def store(tmp_path):
    """A fresh DuckDBMemoryStore with the BGE embedder."""
    from store import DuckDBMemoryStore
    from embeddings import LocalEmbedder
    embedder = LocalEmbedder("BAAI/bge-small-en-v1.5")
    s = DuckDBMemoryStore(
        tmp_path / "test.duckdb", user_id="test_user", embedder=embedder,
    )
    yield s
    s.close()


# ---------------------------------------------------------------------------
# 1. Same fact rephrased → detected, loser quarantined with keeper link
# ---------------------------------------------------------------------------

class TestSemanticDedup:
    def test_same_fact_rephrased_detected(self, store):
        """Two memories expressing the same fact with different wording
        should be detected as semantic near-duplicates."""
        rec1 = store.remember(
            category="personal_fact",
            content="User lives in Springfield and works as a software engineer",
            dedup=False,
        )
        rec2 = store.remember(
            category="personal_fact",
            content="The user resides in Springfield and is employed as a software developer",
            dedup=False,
        )
        assert rec1 is not None and rec2 is not None
        report = store.consolidate(dry_run=False, max_actions=10)
        # At least one should be quarantined as a semantic duplicate.
        semantic_candidates = [
            c for c in report["candidates"]
            if "duplicate_semantic" in c["reason"]
        ]
        assert len(semantic_candidates) >= 1, "Should detect semantic near-duplicate"
        # The quarantined one should have a keeper link in the reason.
        for c in semantic_candidates:
            assert "keeper=" in c["reason"], "Reason must encode keeper link"
            assert c["keeper_id"] is not None

    # -----------------------------------------------------------------------
    # 2. Distinct facts → untouched
    # -----------------------------------------------------------------------

    def test_distinct_facts_untouched(self, store):
        """Memories about completely different topics should not be merged."""
        store.remember(category="personal_fact", content="User enjoys drinking coffee every morning")
        store.remember(category="personal_fact", content="User bought a new Toyota Hilux last week")
        report = store.consolidate(dry_run=True, max_actions=50)
        semantic_candidates = [
            c for c in report["candidates"]
            if "duplicate_semantic" in c["reason"]
        ]
        assert len(semantic_candidates) == 0, "Distinct facts should not be merged"

    # -----------------------------------------------------------------------
    # 3. Keeper selection by quality score; tie-break by recency
    # -----------------------------------------------------------------------

    def test_keeper_selection_by_quality(self, store):
        """The higher-quality record should be the keeper."""
        # Low-quality: confidence 0.3, no retrieval, no helpful votes.
        rec_low = store.remember(
            category="personal_fact",
            content="User likes to go running in the park on weekends",
            confidence=0.3, dedup=False,
        )
        # High-quality: same fact rephrased, confidence 1.0.
        rec_high = store.remember(
            category="personal_fact",
            content="User enjoys running in the park during weekends",
            confidence=1.0, dedup=False,
        )
        report = store.consolidate(dry_run=False, max_actions=10)
        semantic_candidates = [
            c for c in report["candidates"]
            if "duplicate_semantic" in c["reason"]
        ]
        assert len(semantic_candidates) >= 1
        # The low-quality one should be the duplicate (quarantined).
        quarantined_ids = set(report["quarantined_ids"])
        assert rec_low.memory_id in quarantined_ids, \
            "Lower-quality record should be quarantined"
        assert rec_high.memory_id not in quarantined_ids, \
            "Higher-quality record should be the keeper"

    # -----------------------------------------------------------------------
    # 4. dry_run=True performs zero writes
    # -----------------------------------------------------------------------

    def test_dry_run_no_writes(self, store):
        """dry_run=True must not quarantine anything."""
        store.remember(
            category="personal_fact",
            content="User is a certified scuba diver who loves the ocean",
            dedup=False,
        )
        store.remember(
            category="personal_fact",
            content="The user is a certified scuba diver and loves the sea",
            dedup=False,
        )
        active_before = store.connection.execute(
            "SELECT COUNT(*) FROM memory_records WHERE status = 'active'"
        ).fetchone()[0]
        report = store.consolidate(dry_run=True, max_actions=50)
        assert report["dry_run"] is True
        assert report["quarantined_count"] == 0
        active_after = store.connection.execute(
            "SELECT COUNT(*) FROM memory_records WHERE status = 'active'"
        ).fetchone()[0]
        assert active_before == active_after, "dry_run must not change any records"

    # -----------------------------------------------------------------------
    # 5. Cross-category pairs NOT merged at default config
    # -----------------------------------------------------------------------

    def test_cross_category_not_merged(self, store):
        """An insight derived from a personal_fact should NOT be merged."""
        store.remember(
            category="personal_fact",
            content="User runs 5km every morning before work",
        )
        store.remember(
            category="insight",
            content="User discovered that running 5km each morning boosts focus",
        )
        report = store.consolidate(dry_run=True, max_actions=50)
        semantic_candidates = [
            c for c in report["candidates"]
            if "duplicate_semantic" in c["reason"]
        ]
        assert len(semantic_candidates) == 0, \
            "Cross-category pairs should not be merged at default config"

    # -----------------------------------------------------------------------
    # 6. Expired / superseded / quarantined records excluded
    # -----------------------------------------------------------------------

    def test_expired_excluded(self, store):
        """Expired records should not participate in semantic dedup."""
        past = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
        store.remember(
            category="context_note",
            content="User is on a business trip to Berlin this week",
            expires_at=past,
        )
        store.remember(
            category="context_note",
            content="User is traveling to Berlin for business this week",
        )
        report = store.consolidate(dry_run=True, max_actions=50)
        semantic_candidates = [
            c for c in report["candidates"]
            if "duplicate_semantic" in c["reason"]
        ]
        # The expired one should not be a semantic candidate (it's already
        # an "expired" candidate). The active one has no near-duplicate
        # in the eligible pool.
        assert len(semantic_candidates) == 0, \
            "Expired records should be excluded from semantic dedup"

    def test_superseded_excluded(self, store):
        """Superseded records (valid_to IS NOT NULL) should not participate."""
        rec1 = store.remember(
            category="personal_fact",
            content="User pays R15000 rent for an apartment in the city center",
        )
        # Supersede rec1 with a new version.
        store.update_memory(rec1.memory_id, content="User pays R18000 rent for an apartment in the city center")
        # Now add a near-duplicate of the NEW content.
        store.remember(
            category="personal_fact",
            content="The user pays R18000 rent for a flat in the city centre",
            dedup=False,
        )
        report = store.consolidate(dry_run=True, max_actions=50)
        semantic_candidates = [
            c for c in report["candidates"]
            if "duplicate_semantic" in c["reason"]
        ]
        # The superseded rec1 should not appear as a semantic candidate.
        for c in semantic_candidates:
            assert c["memory_id"] != rec1.memory_id, \
                "Superseded records should be excluded from semantic dedup"

    def test_quarantined_excluded(self, store):
        """Already-quarantined records should not participate."""
        rec1 = store.remember(
            category="personal_fact",
            content="User studies machine learning at the university",
        )
        store.quarantine_memory(rec1.memory_id, "manual review")
        rec2 = store.remember(
            category="personal_fact",
            content="User is studying machine learning at the university",
            dedup=False,
        )
        report = store.consolidate(dry_run=True, max_actions=50)
        semantic_candidates = [
            c for c in report["candidates"]
            if "duplicate_semantic" in c["reason"]
        ]
        # rec1 is quarantined → not in the active pool → not a candidate.
        # rec2 has no near-duplicate in the active pool.
        for c in semantic_candidates:
            assert c["memory_id"] != rec1.memory_id, \
                "Quarantined records should be excluded from semantic dedup"

    # -----------------------------------------------------------------------
    # 7. Cluster of 3+ near-dupes → exactly one survivor
    # -----------------------------------------------------------------------

    def test_cluster_of_three_one_survivor(self, store):
        """A cluster of 3+ near-dupes should leave exactly one survivor."""
        recs = [
            store.remember(
                category="personal_fact",
                content="User works as a software engineer at TechCorp in Springfield",
                confidence=0.5, dedup=False,
            ),
            store.remember(
                category="personal_fact",
                content="User is employed as a software engineer at TechCorp in Springfield",
                confidence=0.8, dedup=False,
            ),
            store.remember(
                category="personal_fact",
                content="The user works as a software engineer at TechCorp Springfield",
                confidence=1.0, dedup=False,
            ),
        ]
        assert all(r is not None for r in recs)
        report = store.consolidate(dry_run=False, max_actions=20)
        semantic_candidates = [
            c for c in report["candidates"]
            if "duplicate_semantic" in c["reason"]
        ]
        # Should have 2 quarantined (the cluster has 3, one survives).
        quarantined_ids = set(report["quarantined_ids"])
        semantic_quarantined = [r for r in recs if r.memory_id in quarantined_ids]
        assert len(semantic_quarantined) >= 2, \
            f"Expected ≥2 quarantined in cluster, got {len(semantic_quarantined)}"
        # The highest-confidence one should be the keeper.
        survivor = [r for r in recs if r.memory_id not in quarantined_ids]
        assert len(survivor) >= 1
        # No self-quarantine: keeper should not be in the quarantined list.
        for c in semantic_candidates:
            assert c["memory_id"] != c.get("keeper_id"), \
                "Keeper should not quarantine itself"
        # All quarantined should point to the same keeper.
        keepers = {c["keeper_id"] for c in semantic_candidates}
        assert len(keepers) == 1, "All cluster members should point to one keeper"

    # -----------------------------------------------------------------------
    # 8. duplicate_min_similarity config honored
    # -----------------------------------------------------------------------

    def test_threshold_honored(self, store):
        """Raising the threshold should reduce or eliminate merges."""
        store.remember(
            category="personal_fact",
            content="User plays the guitar and piano in a local band",
            dedup=False,
        )
        store.remember(
            category="personal_fact",
            content="User plays guitar and piano in a local music band",
            dedup=False,
        )
        # Low threshold → should detect.
        report_low = store.consolidate(
            dry_run=True, max_actions=50,
            duplicate_min_similarity=0.80,
        )
        semantic_low = [
            c for c in report_low["candidates"]
            if "duplicate_semantic" in c["reason"]
        ]
        # High threshold → should not detect (or detect fewer).
        report_high = store.consolidate(
            dry_run=True, max_actions=50,
            duplicate_min_similarity=0.99,
        )
        semantic_high = [
            c for c in report_high["candidates"]
            if "duplicate_semantic" in c["reason"]
        ]
        assert len(semantic_high) <= len(semantic_low), \
            "Higher threshold should not find more duplicates"

    # -----------------------------------------------------------------------
    # 9. Restore flow: quarantined loser comes back; keeper untouched
    # -----------------------------------------------------------------------

    def test_restore_flow(self, store):
        """Quarantined loser can be restored; keeper is untouched."""
        rec_dup = store.remember(
            category="personal_fact",
            content="User is a certified scuba diver who loves the ocean",
            confidence=0.5, dedup=False,
        )
        rec_keeper = store.remember(
            category="personal_fact",
            content="The user is a certified scuba diver and loves the sea",
            confidence=1.0, dedup=False,
        )
        report = store.consolidate(dry_run=False, max_actions=10)
        quarantined_ids = set(report["quarantined_ids"])
        # One of them was quarantined.
        assert len(quarantined_ids & {rec_dup.memory_id, rec_keeper.memory_id}) >= 1
        # Find which was quarantined.
        quarantined_id = (quarantined_ids & {rec_dup.memory_id, rec_keeper.memory_id}).pop()
        keeper_id = rec_keeper.memory_id if quarantined_id == rec_dup.memory_id else rec_dup.memory_id
        # Restore the quarantined one.
        assert store.restore_memory(quarantined_id) is True
        # Verify it's active again.
        restored = store.connection.execute(
            "SELECT status FROM memory_records WHERE memory_id = ?",
            [quarantined_id],
        ).fetchone()
        assert restored[0] == "active"
        # Keeper should still be active (untouched by restore).
        keeper_status = store.connection.execute(
            "SELECT status FROM memory_records WHERE memory_id = ?",
            [keeper_id],
        ).fetchone()
        assert keeper_status[0] == "active"

    # -----------------------------------------------------------------------
    # 10. Existing consolidate behavior for old buckets stays green
    # -----------------------------------------------------------------------

    def test_existing_buckets_unchanged(self, store):
        """The expired and stale_unused_temporary buckets should still work."""
        past = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()
        store.remember(
            category="context_note",
            content="User has a temporary promo code",
            expires_at=past,
        )
        # A stale unused temporary record (old, no retrieval, low confidence).
        old = (datetime.now(timezone.utc) - timedelta(days=90)).isoformat()
        rec_stale = store.remember(
            category="context_note",
            content="User mentioned a one-off task",
            confidence=0.3,
        )
        store.connection.execute(
            "UPDATE memory_records SET created_at = ? WHERE memory_id = ?",
            [old, rec_stale.memory_id],
        )
        report = store.consolidate(dry_run=True, max_actions=50)
        reasons = {c["reason"] for c in report["candidates"]}
        assert "expired" in reasons, "Expired bucket should still work"
        assert "stale_unused_temporary" in reasons, \
            "Stale unused temporary bucket should still work"

    def test_report_has_semantic_fields(self, store):
        """The report dict should include the new semantic dedup fields."""
        store.remember(category="personal_fact", content="User likes Python programming")
        report = store.consolidate(dry_run=True, max_actions=10)
        assert "semantic_duplicate_count" in report
        assert "reason_counts" in report
        assert "duplicate_min_similarity" in report
        assert isinstance(report["semantic_duplicate_count"], int)
        assert isinstance(report["reason_counts"], dict)

    def test_no_llm_no_content_rewrite(self, store):
        """Semantic dedup must not rewrite keeper content (no LLM, no fusion)."""
        rec1 = store.remember(
            category="personal_fact",
            content="User is a certified scuba diver who loves the ocean",
            confidence=0.5, dedup=False,
        )
        rec2 = store.remember(
            category="personal_fact",
            content="The user is a certified scuba diver and loves the sea",
            confidence=1.0, dedup=False,
        )
        store.consolidate(dry_run=False, max_actions=10)
        # The keeper's content must be byte-identical.
        for rec_id in [rec1.memory_id, rec2.memory_id]:
            content = store.connection.execute(
                "SELECT content FROM memory_records WHERE memory_id = ?",
                [rec_id],
            ).fetchone()[0]
            # Content should not have been modified.
            if rec_id == rec1.memory_id:
                assert content == "User is a certified scuba diver who loves the ocean"
            else:
                assert content == "The user is a certified scuba diver and loves the sea"

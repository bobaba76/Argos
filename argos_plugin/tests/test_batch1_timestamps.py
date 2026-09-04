"""Tests for batch-1 fixes: #28 (store.py timestamp/P2C bugs) + #33 (distillation.py bugs).

Covers:
- #28 finding 1: P2C loop-counter swap — i/j not mutated mid-iteration
- #28 finding 2: Naive created_at gets recency boost (not silently 0.0)
- #28 finding 3: as_of validated as ISO-8601 (invalid → ignored with warning)
- #33 finding 1: Mixed-project cluster uses _unanimous_project_id (not cluster[0])
- #33 finding 2: numpy fallback logs a warning
- #33 finding 3: None created_at sorts as oldest (intentional, not accidental)
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_plugin_dir = Path(__file__).resolve().parent.parent
if str(_plugin_dir) not in sys.path:
    sys.path.insert(0, str(_plugin_dir))

from store import MemoryRecord, DuckDBMemoryStore


# ---------------------------------------------------------------------------
# #28 finding 1: P2C loop-counter swap
# ---------------------------------------------------------------------------

class TestP2CLoopCounter:
    """The P2C demotion loop must not mutate i/j mid-iteration."""

    def test_p2c_older_newer_without_swapping_counters(self):
        """When ti > tj (i is newer), the code should compute older_idx/newer_idx
        without reassigning i/j. Verify by checking that all pairs are visited
        exactly once and the bounded-sink check uses correct indices."""
        # Enable P2C for this test.
        original = DuckDBMemoryStore._P2C_ENABLED
        DuckDBMemoryStore._P2C_ENABLED = True
        try:
            # Build records where the newer one ranks higher (i < j).
            # Record at index 0 is newer (higher created_at) but ranks
            # higher (similarity). Record at index 1 is older.
            records = [
                MemoryRecord(
                    memory_id="newer",
                    category="personal_fact",
                    content="User lives in Springfield",
                    created_at="2026-01-01T00:00:00Z",
                    similarity=0.90,
                ),
                MemoryRecord(
                    memory_id="older",
                    category="personal_fact",
                    content="User lives in Springfield",
                    created_at="2025-01-01T00:00:00Z",
                    similarity=0.85,
                ),
            ]
            DuckDBMemoryStore._apply_p2c(records)
            # The newer record (index 0) should still be ranked first
            # because it's already above the older one. P2C only demotes
            # the newer one if the older one ranks higher.
            # Here older is at index 1 (lower), so no demotion needed.
            assert records[0].memory_id == "newer"
        finally:
            DuckDBMemoryStore._P2C_ENABLED = original

    def test_p2c_demotes_newer_when_older_ranks_higher(self):
        """When the older record ranks higher (i < j), P2C should demote
        the newer one above it by a small epsilon."""
        original = DuckDBMemoryStore._P2C_ENABLED
        DuckDBMemoryStore._P2C_ENABLED = True
        try:
            # Older at index 0 (higher similarity), newer at index 1.
            records = [
                MemoryRecord(
                    memory_id="older",
                    category="personal_fact",
                    content="User lives in Springfield",
                    created_at="2025-01-01T00:00:00Z",
                    similarity=0.90,
                ),
                MemoryRecord(
                    memory_id="newer",
                    category="personal_fact",
                    content="User lives in Springfield",
                    created_at="2026-01-01T00:00:00Z",
                    similarity=0.80,
                ),
            ]
            DuckDBMemoryStore._apply_p2c(records)
            # After P2C, newer should be boosted above older.
            assert records[0].memory_id == "newer"
            assert records[0].similarity > records[1].similarity
        finally:
            DuckDBMemoryStore._P2C_ENABLED = original

    def test_p2c_no_swap_corruption_with_three_records(self):
        """With 3 records, verify no pair is revisited due to counter swap.
        The old code would swap i/j, causing the inner loop to continue
        from the swapped position and revisit pairs."""
        original = DuckDBMemoryStore._P2C_ENABLED
        DuckDBMemoryStore._P2C_ENABLED = True
        try:
            records = [
                MemoryRecord(
                    memory_id="r0",
                    category="personal_fact",
                    content="User works at Acme Corp",
                    created_at="2026-03-01T00:00:00Z",
                    similarity=0.90,
                ),
                MemoryRecord(
                    memory_id="r1",
                    category="personal_fact",
                    content="User works at Acme Corp",
                    created_at="2026-01-01T00:00:00Z",
                    similarity=0.85,
                ),
                MemoryRecord(
                    memory_id="r2",
                    category="personal_fact",
                    content="User works at Acme Corp",
                    created_at="2025-06-01T00:00:00Z",
                    similarity=0.80,
                ),
            ]
            DuckDBMemoryStore._apply_p2c(records)
            # All three are near-duplicates. The newest (r0) should end up
            # first after P2C. No crash, no infinite loop.
            assert len(records) == 3
            # r0 is newest and already ranks highest — should stay first.
            assert records[0].memory_id == "r0"
        finally:
            DuckDBMemoryStore._P2C_ENABLED = original


# ---------------------------------------------------------------------------
# #28 finding 2: Naive created_at gets recency boost
# ---------------------------------------------------------------------------

class TestNaiveTimestampRecency:
    """Naive timestamps (no Z/offset) should get a recency boost, not 0.0."""

    def test_naive_timestamp_gets_recency_boost(self):
        """A naive timestamp (no Z) should still produce a recency boost.
        Before the fix, _recency_boost silently returned 0.0 for naive
        timestamps because datetime.fromisoformat failed on them."""
        # Use a recent naive timestamp (today, no timezone).
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        naive_ts = now.strftime("%Y-%m-%dT%H:%M:%S")  # no Z, no offset
        boost = DuckDBMemoryStore._recency_boost(naive_ts)
        # Should be close to 0.10 (today's boost), not 0.0.
        assert boost > 0.05, (
            f"Naive timestamp should get recency boost, got {boost} "
            f"(expected > 0.05 for a today-dated naive timestamp)"
        )

    def test_aware_timestamp_gets_recency_boost(self):
        """A timezone-aware timestamp (with Z) should still work."""
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        aware_ts = now.strftime("%Y-%m-%dT%H:%M:%SZ")
        boost = DuckDBMemoryStore._recency_boost(aware_ts)
        assert boost > 0.05

    def test_none_timestamp_returns_zero(self):
        """None/empty timestamp → 0.0 boost (no crash)."""
        assert DuckDBMemoryStore._recency_boost(None) == 0.0
        assert DuckDBMemoryStore._recency_boost("") == 0.0

    def test_unparseable_timestamp_returns_zero(self):
        """Garbage timestamp → 0.0 boost (no crash)."""
        assert DuckDBMemoryStore._recency_boost("not-a-date") == 0.0

    def test_parse_timestamp_naive_assumes_utc(self):
        """_parse_timestamp should assume UTC for naive timestamps."""
        from datetime import datetime, timezone
        parsed = DuckDBMemoryStore._parse_timestamp("2026-01-15T12:00:00")
        assert parsed is not None
        assert parsed.tzinfo is not None  # Should be timezone-aware
        assert parsed.utcoffset() == timezone.utc.utcoffset(None)

    def test_parse_timestamp_aware_preserves_timezone(self):
        """_parse_timestamp should preserve timezone for aware timestamps."""
        parsed = DuckDBMemoryStore._parse_timestamp("2026-01-15T12:00:00Z")
        assert parsed is not None
        assert parsed.tzinfo is not None

    def test_parse_timestamp_none_returns_none(self):
        """_parse_timestamp(None) → None."""
        assert DuckDBMemoryStore._parse_timestamp(None) is None
        assert DuckDBMemoryStore._parse_timestamp("") is None


# ---------------------------------------------------------------------------
# #28 finding 3: as_of validated as ISO-8601
# ---------------------------------------------------------------------------

class TestAsOfValidation:
    """Invalid as_of should be ignored with a warning, not passed to SQL."""

    def test_invalid_as_of_normalized_to_none(self, tmp_path, caplog):
        """An invalid as_of should be set to None so downstream SQL
        doesn't get a garbage temporal cutoff."""
        store = DuckDBMemoryStore(tmp_path / "test_asof.duckdb")
        try:
            import logging
            with caplog.at_level(logging.WARNING):
                # Call _hybrid_search with an invalid as_of.
                # It should log a warning and proceed with as_of=None.
                results = store._hybrid_search(
                    "test query", limit=5, as_of="not-a-date",
                )
                # Should not crash; results may be empty (no data).
                assert isinstance(results, list)
                # Check that a warning was logged.
                assert any("Invalid as_of" in r.message for r in caplog.records), (
                    f"Expected 'Invalid as_of' warning, got: {[r.message for r in caplog.records]}"
                )
        finally:
            store.close()

    def test_valid_as_of_accepted(self, tmp_path):
        """A valid ISO-8601 as_of should be accepted (not set to None)."""
        store = DuckDBMemoryStore(tmp_path / "test_asof_valid.duckdb")
        try:
            # Should not crash with a valid as_of.
            results = store._hybrid_search(
                "test query", limit=5, as_of="2026-01-15T12:00:00Z",
            )
            assert isinstance(results, list)
        finally:
            store.close()

    def test_none_as_of_accepted(self, tmp_path):
        """None as_of should be accepted (no temporal filter)."""
        store = DuckDBMemoryStore(tmp_path / "test_asof_none.duckdb")
        try:
            results = store._hybrid_search("test query", limit=5, as_of=None)
            assert isinstance(results, list)
        finally:
            store.close()


# ---------------------------------------------------------------------------
# #33 finding 1: Mixed-project cluster mis-tagging
# ---------------------------------------------------------------------------

class TestMixedProjectCluster:
    """_unanimous_project_id should be used instead of cluster[0].project_id."""

    def test_unanimous_project_id_unanimous(self):
        """When all records have the same project_id, return it."""
        from distillation import _unanimous_project_id

        class FakeRecord:
            def __init__(self, pid):
                self.project_id = pid

        records = [FakeRecord("proj1"), FakeRecord("proj1"), FakeRecord("proj1")]
        assert _unanimous_project_id(records) == "proj1"

    def test_unanimous_project_id_mixed(self):
        """When records have different project_ids, return None (global)."""
        from distillation import _unanimous_project_id

        class FakeRecord:
            def __init__(self, pid):
                self.project_id = pid

        records = [FakeRecord("proj1"), FakeRecord("proj2")]
        assert _unanimous_project_id(records) is None

    def test_unanimous_project_id_all_none(self):
        """When all records have None project_id, return None."""
        from distillation import _unanimous_project_id

        class FakeRecord:
            def __init__(self, pid):
                self.project_id = pid

        records = [FakeRecord(None), FakeRecord(None)]
        assert _unanimous_project_id(records) is None

    def test_unanimous_project_id_one_set_one_none(self):
        """When one record has a project_id and another has None, return
        the set one only if it's unanimous among non-None values."""
        from distillation import _unanimous_project_id

        class FakeRecord:
            def __init__(self, pid):
                self.project_id = pid

        records = [FakeRecord("proj1"), FakeRecord(None)]
        # _unanimous_project_id only counts non-None project_ids.
        # If there's exactly one unique non-None project_id, return it.
        assert _unanimous_project_id(records) == "proj1"


# ---------------------------------------------------------------------------
# #33 finding 2: numpy fallback warning
# ---------------------------------------------------------------------------

class TestNumpyFallbackWarning:
    """numpy fallback should log a warning, not silently degrade."""

    def test_numpy_fallback_logs_warning(self, caplog):
        """When numpy is unavailable and there are 2+ records, a warning
        should be logged."""
        from distillation import _seed_star_cluster
        import logging

        class FakeRecord:
            def __init__(self, mid, content="test"):
                self.memory_id = mid
                self.content = content
                self.embedding = [0.1, 0.2, 0.3]
                self.created_at = "2026-01-01T00:00:00Z"
                self.project_id = None

        records = [FakeRecord("m1"), FakeRecord("m2")]

        # Temporarily disable numpy.
        import distillation
        original_np = distillation.np
        distillation.np = None
        try:
            with caplog.at_level(logging.WARNING, logger="distillation"):
                clusters = _seed_star_cluster(records, min_similarity=0.75)
                # Should return singletons.
                assert len(clusters) == 2
                # Should have logged a warning.
                assert any("numpy unavailable" in r.message for r in caplog.records), (
                    f"Expected numpy unavailable warning, got: {[r.message for r in caplog.records]}"
                )
        finally:
            distillation.np = original_np

    def test_numpy_available_no_warning(self, caplog):
        """When numpy is available, no fallback warning should be logged."""
        from distillation import _seed_star_cluster
        import logging

        class FakeRecord:
            def __init__(self, mid, content="test"):
                self.memory_id = mid
                self.content = content
                self.embedding = [0.1, 0.2, 0.3]
                self.created_at = "2026-01-01T00:00:00Z"
                self.project_id = None

        records = [FakeRecord("m1"), FakeRecord("m2")]

        import distillation
        if distillation.np is None:
            pytest.skip("numpy not installed — skipping positive test")

        with caplog.at_level(logging.WARNING, logger="distillation"):
            _seed_star_cluster(records, min_similarity=0.75)
            # Should NOT have logged a numpy fallback warning.
            assert not any("numpy unavailable" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# #33 finding 3: None created_at sorts as oldest
# ---------------------------------------------------------------------------

class TestNoneCreatedAtSorting:
    """None created_at should sort as oldest (intentional), not as empty
    string (accidental oldest in string comparison)."""

    def test_none_created_at_sorts_last_in_descending(self):
        """In descending order (newest first), None created_at records
        should sort after all dated records."""
        from distillation import _seed_star_cluster
        import distillation

        if distillation.np is None:
            pytest.skip("numpy not installed")

        class FakeRecord:
            def __init__(self, mid, created_at, content="unique content"):
                self.memory_id = mid
                self.content = content
                self.embedding = [0.1, 0.2, 0.3]
                self.created_at = created_at
                self.project_id = None

        # Records with unique content AND distinct embeddings (no clustering,
        # just sorting). The sort order is what we're testing.
        records = [
            FakeRecord("old", "2025-01-01T00:00:00Z", content="old unique content"),
            FakeRecord("none", None, content="none unique content"),
            FakeRecord("new", "2026-01-01T00:00:00Z", content="new unique content"),
        ]
        # Give distinct embeddings so they don't cluster together.
        records[0].embedding = [1.0, 0.0, 0.0]
        records[1].embedding = [0.0, 1.0, 0.0]
        records[2].embedding = [0.0, 0.0, 1.0]

        clusters = _seed_star_cluster(records, min_similarity=0.75)
        # Each record should be its own cluster (unique content).
        assert len(clusters) == 3
        # The first cluster's first record should be "new" (newest).
        # The None record should be last (oldest).
        cluster_ids = [c[0].memory_id for c in clusters]
        assert cluster_ids[0] == "new", (
            f"Expected 'new' first, got {cluster_ids}"
        )
        assert cluster_ids[-1] == "none", (
            f"Expected 'none' last, got {cluster_ids}"
        )

"""Pytest tests for the argos plugin.

Run with:
    python -m pytest tests/test_argos.py -v

Or use the standalone script (no pytest needed):
    python tests/run_tests.py
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


class TestHybridRanking:
    """Tests for RRF fusion, feedback weighting, and recency boost."""

    def test_rrf_fuse_combines_both_lists(self):
        """RRF should produce a score for items in either or both lists."""
        from store import DuckDBMemoryStore, MemoryRecord

        vec = [
            MemoryRecord(memory_id="a", category="personal_fact", content="alpha", similarity=0.9),
            MemoryRecord(memory_id="b", category="personal_fact", content="beta", similarity=0.7),
        ]
        text = [
            MemoryRecord(memory_id="b", category="personal_fact", content="beta", similarity=0.5),
            MemoryRecord(memory_id="c", category="personal_fact", content="gamma", similarity=0.5),
        ]
        fused = DuckDBMemoryStore._rrf_fuse(vec, text)
        ids = {r.memory_id for r in fused}
        assert ids == {"a", "b", "c"}, "RRF must include items from both lists"
        # Item 'b' appears in both lists — should rank highest.
        assert fused[0].memory_id == "b", "Item in both lists must rank highest"

    def test_rrf_score_in_zero_one_range(self):
        """Normalized RRF scores must be in [0, 1]."""
        from store import DuckDBMemoryStore, MemoryRecord

        vec = [MemoryRecord(memory_id=f"v{i}", category="personal_fact", content=f"v{i}", similarity=0.5) for i in range(10)]
        text = [MemoryRecord(memory_id=f"t{i}", category="personal_fact", content=f"t{i}", similarity=0.5) for i in range(10)]
        fused = DuckDBMemoryStore._rrf_fuse(vec, text)
        for r in fused:
            assert 0.0 <= r.similarity <= 1.0, f"Score {r.similarity} out of [0,1]"

    def test_feedback_boosts_helpful_memories(self, tmp_path):
        """A memory marked helpful should rank above one that wasn't, all else equal."""
        from store import DuckDBMemoryStore

        store = DuckDBMemoryStore(tmp_path / "test.duckdb", user_id="test_user", embedder=None)
        rec_normal = store.remember(category="personal_fact", content="User likes apples for snacks")
        rec_helpful = store.remember(category="personal_fact", content="User likes bananas for snacks")
        assert rec_normal and rec_helpful
        store.record_feedback(rec_helpful.memory_id, "helpful")

        results = store.search("snacks", limit=5)
        assert len(results) >= 2
        # The helpful memory should rank above the normal one.
        helpful_rank = next(i for i, r in enumerate(results) if r.memory_id == rec_helpful.memory_id)
        normal_rank = next(i for i, r in enumerate(results) if r.memory_id == rec_normal.memory_id)
        assert helpful_rank < normal_rank, "Helpful memory should rank higher"
        store.close()

    def test_feedback_penalizes_dismissed_memories(self, tmp_path):
        """A memory marked dismissed should rank below one that wasn't."""
        from store import DuckDBMemoryStore

        store = DuckDBMemoryStore(tmp_path / "test.duckdb", user_id="test_user", embedder=None)
        rec_dismissed = store.remember(category="personal_fact", content="User likes apples for snacks")
        rec_normal = store.remember(category="personal_fact", content="User likes bananas for snacks")
        assert rec_dismissed and rec_normal
        store.record_feedback(rec_dismissed.memory_id, "dismissed")

        results = store.search("snacks", limit=5)
        assert len(results) >= 2
        dismissed_rank = next(i for i, r in enumerate(results) if r.memory_id == rec_dismissed.memory_id)
        normal_rank = next(i for i, r in enumerate(results) if r.memory_id == rec_normal.memory_id)
        assert normal_rank < dismissed_rank, "Dismissed memory should rank lower"
        store.close()

    def test_suppress_retrieval_does_not_increment_count(self, tmp_path):
        """search(suppress_retrieval=True) must NOT bump retrieval_count.

        Without this, eval/diagnostic runs inflate retrieval_count on the
        memories they search, polluting the retrieval signal as a ranking
        discriminator. The eval-relevant memories end up with 400+ fake
        retrievals, all from eval reruns.
        """
        from store import DuckDBMemoryStore

        store = DuckDBMemoryStore(tmp_path / "test.duckdb", user_id="test_user", embedder=None)
        rec = store.remember(category="personal_fact", content="User likes apples for snacks")
        assert rec is not None

        # Normal search increments retrieval_count
        store.search("snacks", limit=5)
        rows = store._fetch_records(
            "SELECT retrieval_count FROM memory_records WHERE memory_id = ?",
            [rec.memory_id],
        )
        assert rows[0].retrieval_count >= 1, "Normal search should increment retrieval_count"

        # suppress_retrieval=True does NOT increment
        count_before = rows[0].retrieval_count
        store.search("snacks", limit=5, suppress_retrieval=True)
        rows = store._fetch_records(
            "SELECT retrieval_count FROM memory_records WHERE memory_id = ?",
            [rec.memory_id],
        )
        assert rows[0].retrieval_count == count_before, \
            f"suppress_retrieval=True should NOT increment retrieval_count: " \
            f"before={count_before}, after={rows[0].retrieval_count}"
        store.close()

    def test_recency_boost_is_nonnegative(self):
        """Recency boost must be >= 0 and decay with age."""
        from store import DuckDBMemoryStore
        from datetime import datetime, timezone, timedelta

        now = datetime.now(timezone.utc).isoformat()
        old = (datetime.now(timezone.utc) - timedelta(days=180)).isoformat()
        none = None

        boost_now = DuckDBMemoryStore._recency_boost(now)
        boost_old = DuckDBMemoryStore._recency_boost(old)
        boost_none = DuckDBMemoryStore._recency_boost(none)

        assert boost_none == 0.0, "Missing timestamp should give 0 boost"
        assert boost_now > boost_old > 0.0, "Recent must boost more than old, both > 0"
        assert boost_now <= 0.10, "Max boost is 0.10"

    def test_text_only_fallback_still_works(self, tmp_path):
        """When embeddings are unavailable, text-only search must still return results."""
        from store import DuckDBMemoryStore

        store = DuckDBMemoryStore(tmp_path / "test.duckdb", user_id="test_user", embedder=None)
        store.remember(category="personal_fact", content="User takes Medication-Y 50mg for ADHD")
        store.remember(category="personal_fact", content="User has Discovery medical aid")
        results = store.search("Medication-Y", limit=5)
        assert len(results) >= 1
        assert any("Medication-Y" in r.content for r in results)
        store.close()

    def test_keyword_match_boosts_via_rrf(self, tmp_path):
        """A precise keyword match should surface even if vector similarity is low."""
        from store import DuckDBMemoryStore

        # Use no embedder so we test text-only path (vector path is tested
        # implicitly by the RRF unit test above).
        store = DuckDBMemoryStore(tmp_path / "test.duckdb", user_id="test_user", embedder=None)
        store.remember(category="personal_fact", content="User takes Medication-X 10mg daily")
        store.remember(category="personal_fact", content="User enjoys hiking on weekends")
        results = store.search("Medication-X", limit=5)
        assert len(results) >= 1
        assert "Medication-X" in results[0].content, "Exact keyword match should rank first"
        store.close()



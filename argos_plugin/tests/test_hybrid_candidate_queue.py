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


class TestCandidateQueue:
    def test_pending_candidate_is_not_searchable_until_approved(self, tmp_path):
        from store import DuckDBMemoryStore

        store = DuckDBMemoryStore(tmp_path / "test.duckdb", user_id="test_user", embedder=None)
        candidate = store.save_candidate(
            category="preference",
            content="User prefers concise technical explanations",
            source="llm_extraction",
            confidence=0.42,
            scope="profile",
        )
        assert candidate is not None
        assert candidate["status"] == "pending"
        assert store.search("concise technical explanations", limit=5) == []

        reviewed = store.review_candidate(
            candidate_id=candidate["candidate_id"],
            decision="approved",
            reason="confirmed",
        )
        assert reviewed is not None
        assert reviewed["candidate"]["status"] == "approved"
        assert reviewed["memory"]["status"] == "active"
        assert store.search("concise technical explanations", limit=5)
        store.close()

    def test_reviewed_approved_promotes_with_reviewer_classification(self, tmp_path):
        from store import DuckDBMemoryStore

        store = DuckDBMemoryStore(tmp_path / "test.duckdb", user_id="test_user", embedder=None)
        candidate = store.save_candidate(
            category="context_note",
            content="User is temporarily working from a client office",
            source="llm_extraction",
            confidence=0.42,
            durability="temporary",
            scope="profile",
        )
        result = store.review_candidate(
            candidate_id=candidate["candidate_id"],
            decision="reviewed_approved",
            reason="reviewed",
            durability="durable",
            scope="project",
        )
        assert result is not None
        assert result["memory"] is not None
        assert result["memory"]["durability"] == "durable"
        assert result["memory"]["scope"] == "project"
        store.close()

    def test_quarantine_hides_without_deleting(self, tmp_path):
        from store import DuckDBMemoryStore

        store = DuckDBMemoryStore(tmp_path / "test.duckdb", user_id="test_user", embedder=None)
        rec = store.remember(category="goal", content="User goal: stop you")
        assert rec is not None
        assert store.quarantine_memory(rec.memory_id, "assistant instruction fragment")
        assert store.search("stop you", limit=5) == []
        rows = store._fetch_records(
            "SELECT status, quarantine_reason FROM memory_records WHERE memory_id = ?",
            [rec.memory_id],
        )
        assert rows[0].status == "quarantined"
        assert rows[0].quarantine_reason == "assistant instruction fragment"
        assert store.count() == 1
        store.close()

    def test_feedback_updates_usage_and_incorrect_quarantines(self, tmp_path):
        from store import DuckDBMemoryStore

        store = DuckDBMemoryStore(tmp_path / "test.duckdb", user_id="test_user", embedder=None)
        rec = store.remember(category="preference", content="User prefers direct answers")
        assert rec is not None
        assert store.search("direct answers", limit=5)
        assert store.record_feedback(rec.memory_id, "helpful")
        assert store.record_feedback(rec.memory_id, "incorrect")
        rows = store._fetch_records(
            "SELECT status, retrieval_count, helpful_count, dismissed_count "
            "FROM memory_records WHERE memory_id = ?",
            [rec.memory_id],
        )
        assert rows[0].status == "quarantined"
        assert rows[0].retrieval_count >= 1
        assert rows[0].helpful_count == 1
        assert rows[0].dismissed_count == 1
        assert store.search("direct answers", limit=5) == []
        assert store.restore_memory(rec.memory_id)
        assert store.search("direct answers", limit=5)
        store.close()

    def test_cleanup_quarantines_known_bad_shape(self, tmp_path):
        from store import DuckDBMemoryStore

        store = DuckDBMemoryStore(tmp_path / "test.duckdb", user_id="test_user", embedder=None)
        rec = store.remember(category="goal", content="User goal: stop you")
        assert rec is not None
        assert store.cleanup_junk() == 1
        assert store.count() == 1
        assert store.search("stop you", limit=5) == []
        store.close()

    def test_short_lived_categories_get_default_expiry(self, tmp_path):
        from store import DuckDBMemoryStore

        store = DuckDBMemoryStore(tmp_path / "test.duckdb", user_id="test_user", embedder=None)
        rec = store.remember(category="context_note", content="User is working from a temporary location")
        assert rec is not None
        assert rec.expires_at is not None
        assert rec.payload["expires_at"] == rec.expires_at
        assert rec.to_dict()["expires_at"] == rec.expires_at
        store.close()



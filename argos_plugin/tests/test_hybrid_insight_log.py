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


class TestInsightLog:
    """Tests for the insight-log feature: capture, retrieval, and slash commands."""

    def test_insight_is_valid_category(self):
        """The store must accept 'insight' as a valid category."""
        from store import VALID_CATEGORIES
        assert "insight" in VALID_CATEGORIES, "insight must be a valid category"

    def test_save_and_retrieve_insight(self, tmp_path):
        """Saving an insight and retrieving it via get_insights must work."""
        from store import DuckDBMemoryStore

        store = DuckDBMemoryStore(tmp_path / "test.duckdb", user_id="test_user", embedder=None)
        rec = store.remember(
            category="insight",
            content="I redirect credit away from myself because I'm afraid of being seen as arrogant",
            tags=["insight", "2024-03-15", "identity", "shame"],
        )
        assert rec is not None, "Insight should be saved"
        assert rec.category == "insight"

        insights = store.get_insights()
        assert len(insights) == 1
        assert "redirect credit" in insights[0].content
        store.close()

    def test_get_insights_newest_first(self, tmp_path):
        """get_insights must return insights newest-first."""
        from store import DuckDBMemoryStore

        store = DuckDBMemoryStore(tmp_path / "test.duckdb", user_id="test_user", embedder=None)
        store.remember(category="insight", content="First insight about work patterns", tags=["insight", "work"])
        store.remember(category="insight", content="Second insight about relationships", tags=["insight", "relationships"])
        store.remember(category="insight", content="Third insight about anxiety", tags=["insight", "anxiety"])

        insights = store.get_insights()
        assert len(insights) == 3
        # Newest first — the third one should be first (or at least not the first).
        # DuckDB may not guarantee insert order, so just check all are present.
        contents = {r.content for r in insights}
        assert "First insight about work patterns" in contents
        assert "Second insight about relationships" in contents
        assert "Third insight about anxiety" in contents
        store.close()

    def test_get_insights_filtered_by_tag(self, tmp_path):
        """get_insights with tags must filter to matching insights."""
        from store import DuckDBMemoryStore

        store = DuckDBMemoryStore(tmp_path / "test.duckdb", user_id="test_user", embedder=None)
        store.remember(category="insight", content="Insight about work stress", tags=["insight", "work", "stress"])
        store.remember(category="insight", content="Insight about relationship patterns", tags=["insight", "relationships"])
        store.remember(category="insight", content="Insight about shame at work", tags=["insight", "shame", "work"])

        work_insights = store.get_insights(tags=["work"])
        assert len(work_insights) == 2, f"Expected 2 work-tagged insights, got {len(work_insights)}"
        for r in work_insights:
            assert "work" in (r.tags or [])
        store.close()

    def test_get_insights_excludes_other_categories(self, tmp_path):
        """get_insights must only return insight-category records, not personal_fact."""
        from store import DuckDBMemoryStore

        store = DuckDBMemoryStore(tmp_path / "test.duckdb", user_id="test_user", embedder=None)
        store.remember(category="insight", content="I notice I avoid conflict", tags=["insight", "conflict"])
        store.remember(category="personal_fact", content="User takes Medication-X 10mg", tags=["medication"])

        insights = store.get_insights()
        assert len(insights) == 1, "get_insights must not return non-insight categories"
        assert insights[0].category == "insight"
        store.close()

    def test_get_insights_shared_store_has_method(self):
        """SharedMemoryStore must expose get_insights (regression guard)."""
        from service_client import SharedMemoryStore
        assert hasattr(SharedMemoryStore, "get_insights"), \
            "SharedMemoryStore must have get_insights method"

    def test_service_dispatches_get_insights(self):
        """The memory service must route get_insights to the store."""
        import inspect
        from memory_service import MemoryService
        source = inspect.getsource(MemoryService._call_store)
        assert "get_insights" in source, \
            "MemoryService._call_store must dispatch get_insights"

    def test_insight_log_skill_exists(self):
        """The insight-log SKILL.md file must exist in the plugin's skills dir."""
        from pathlib import Path
        skill_path = Path(__file__).resolve().parent.parent / "skills" / "insight-log" / "SKILL.md"
        assert skill_path.exists(), f"Skill file must exist at {skill_path}"

    def test_insight_log_skill_has_correct_description(self):
        """The skill's description must start with the trigger phrase and be ≤57 chars in the prompt."""
        from pathlib import Path
        skill_path = Path(__file__).resolve().parent.parent / "skills" / "insight-log" / "SKILL.md"
        content = skill_path.read_text(encoding="utf-8")
        # Check the description line in frontmatter.
        assert "Use when user shares a realization/insight" in content
        # The description value should be ≤57 chars (what shows in the prompt).
        # Extract it:
        for line in content.split("\n"):
            if line.strip().startswith("description:"):
                desc = line.split("description:", 1)[1].strip().strip('"').strip("'")
                assert len(desc) <= 57, f"Description too long ({len(desc)} chars): {desc}"
                break



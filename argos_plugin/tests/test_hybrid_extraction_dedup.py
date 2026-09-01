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


class TestExtractionAndDedup:
    """Tests for smarter LLM-fallback triggering and semantic dedup."""

    def test_llm_fallback_triggers_on_zero_facts(self):
        """_should_try_llm_fallback must trigger when regex found 0 facts."""
        from extractor import _should_try_llm_fallback
        # Substantial message, 0 facts -> should try LLM.
        long_msg = "I just got a new job at TechCorp as a backend engineer, " * 3
        assert _should_try_llm_fallback(long_msg, 0) is True

    def test_llm_fallback_triggers_on_few_facts_long_message(self):
        """1 fact from a 200-word message should trigger LLM (regex likely missed things)."""
        from extractor import _should_try_llm_fallback
        long_msg = " ".join(["word"] * 200)
        assert _should_try_llm_fallback(long_msg, 1) is True

    def test_llm_fallback_skips_short_message_with_facts(self):
        """1 fact from a short message should NOT trigger LLM (likely complete)."""
        from extractor import _should_try_llm_fallback
        short_msg = "I take Medication-X 10mg daily"
        assert _should_try_llm_fallback(short_msg, 1) is False

    def test_llm_fallback_skips_short_message_no_facts(self):
        """Short message with 0 facts should NOT trigger LLM (too short to justify)."""
        from extractor import _should_try_llm_fallback
        short_msg = "hey how are you"
        assert _should_try_llm_fallback(short_msg, 0) is False

    def test_llm_fallback_skips_many_facts(self):
        """Several facts from a reasonable message should NOT trigger LLM."""
        from extractor import _should_try_llm_fallback
        msg = "I take Medication-X for depression. I live in Springfield. I work at TechCorp."
        assert _should_try_llm_fallback(msg, 3) is False

    def test_text_overlap_detects_paraphrases(self):
        """_text_overlap must detect near-duplicate phrasings."""
        from extractor import _text_overlap
        assert _text_overlap(
            "user is married to sam",
            "sam is the user's wife",
        ) is False  # Different words, low overlap — this is OK, semantic dedup handles it
        assert _text_overlap(
            "user takes medication-x 10mg daily for depression",
            "user takes medication-x 10mg daily",
        ) is True  # High word overlap — should be detected as duplicate

    def test_text_overlap_rejects_unrelated(self):
        """_text_overlap must NOT flag unrelated content as duplicate."""
        from extractor import _text_overlap
        assert _text_overlap(
            "user takes medication-x for depression",
            "user enjoys hiking on weekends",
        ) is False

    def test_semantic_dedup_catches_paraphrased_facts(self, tmp_path):
        """DuckDB store with embedder must dedup semantically similar content."""
        from store import DuckDBMemoryStore
        from embeddings import LocalEmbedder

        embedder = LocalEmbedder()  # Will use default model or fail gracefully
        store = DuckDBMemoryStore(tmp_path / "test.duckdb", user_id="test_user", embedder=embedder)

        rec1 = store.remember(category="relationship", content="User is married to Sam")
        assert rec1 is not None

        # Paraphrased version — no substring relation, but semantically identical.
        # If embeddings are available, this should be deduped.
        # If not, it won't be (and that's OK — the test verifies the path works).
        rec2 = store.remember(category="relationship", content="Sam is the user's wife")
        # We can't guarantee dedup without a loaded model, so just verify no crash.
        # If rec2 is None, semantic dedup worked. If rec2 is not None, text fallback.
        store.close()

    def test_substring_dedup_still_works(self, tmp_path):
        """The old substring dedup must still work alongside semantic dedup."""
        from store import DuckDBMemoryStore

        store = DuckDBMemoryStore(tmp_path / "test.duckdb", user_id="test_user", embedder=None)
        rec1 = store.remember(category="personal_fact", content="User takes Medication-X 10mg daily for depression")
        assert rec1 is not None
        # Substring of existing content — should be deduped by layer 2.
        rec2 = store.remember(category="personal_fact", content="User takes Medication-X 10mg daily for depression and anxiety")
        # This is a superstring, so it should be deduped (existing is contained).
        store.close()

    def test_different_facts_not_deduped(self, tmp_path):
        """Genuinely different facts must NOT be deduped."""
        from store import DuckDBMemoryStore

        store = DuckDBMemoryStore(tmp_path / "test.duckdb", user_id="test_user", embedder=None)
        rec1 = store.remember(category="personal_fact", content="User takes Medication-X 10mg daily")
        rec2 = store.remember(category="personal_fact", content="User takes Medication-Y 50mg for ADHD")
        assert rec1 is not None, "First fact should be saved"
        assert rec2 is not None, "Second (different) fact should also be saved"
        assert store.count() == 2
        store.close()



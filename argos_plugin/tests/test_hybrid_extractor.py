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


class TestExtractor:
    def test_extracts_personal_facts(self):
        from extractor import extract_from_turn

        user_msg = (
            "I take FocusTool and CalmTool for my example condition. "
            "Sam is my wife. "
            "I'm working on tapering off ExampleMedication. "
            "I tend to redirect credit away from myself. "
            "I prefer direct communication."
        )
        facts = extract_from_turn(user_msg, "Assistant response", use_llm_fallback=False)
        assert len(facts) >= 3
        categories = {f["category"] for f in facts}
        assert "relationship" in categories

    def test_extracts_tech_facts(self):
        """Extractor should work for work/tech topics, not just personal."""
        from extractor import extract_from_turn

        tech_msg = (
            "I use Vim as my primary editor. "
            "I work at TechCorp as a backend engineer. "
            "I'm learning Rust. "
            "I switched from Docker Swarm to Kubernetes. "
            "I always test before deploying."
        )
        facts = extract_from_turn(tech_msg, "", use_llm_fallback=False)
        assert len(facts) >= 3, f"Expected >= 3 tech facts, got {len(facts)}"
        categories = {f["category"] for f in facts}
        assert "personal_fact" in categories

    def test_ignores_assistant_content(self):
        from extractor import extract_from_turn

        facts = extract_from_turn("", "I take FocusTool for example condition", use_llm_fallback=False)
        assert len(facts) == 0

    def test_ignores_short_text(self):
        from extractor import extract_from_turn

        facts = extract_from_turn("hi", "hello", use_llm_fallback=False)
        assert len(facts) == 0

    def test_ignores_transient_states(self):
        """Should not extract 'I am tired' or 'I am busy' as durable facts."""
        from extractor import extract_from_turn

        facts = extract_from_turn("I am tired and hungry right now.", "", use_llm_fallback=False)
        assert len(facts) == 0, f"Should not extract transient states, got {facts}"



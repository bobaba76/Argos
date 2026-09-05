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

    def test_insight_re_matches_just_modifier(self):
        """#170: INSIGHT_RE must match 'I just realized...' and 'I now
        noticed...' — the optional just/now modifier was missing."""
        from extractor import extract_from_turn
        facts = extract_from_turn(
            "I just realized I work better in the morning.",
            "", use_llm_fallback=False,
        )
        assert len(facts) >= 1, f"Expected insight extraction, got {facts}"
        assert any("realized" in f.get("content", "").lower() or "morning" in f.get("content", "").lower()
                   for f in facts), f"Expected 'realized/morning' insight, got {facts}"

    def test_assistant_directive_does_not_fire_on_first_person(self):
        """#171: 'I always use Vim' should be a habit, not an assistant
        directive. The first-person lookbehind prevents the directive
        regex from matching."""
        from extractor import extract_from_turn
        facts = extract_from_turn(
            "I always use Vim as my editor.",
            "", use_llm_fallback=False,
        )
        assert len(facts) >= 1, f"Expected extraction, got {facts}"
        # Should NOT be categorized as a directive.
        assert not any(f.get("category") == "directive" for f in facts), \
            f"'I always use Vim' should not be a directive, got {facts}"

    def test_llm_fallback_handles_plain_string_response(self):
        """#172: The LLM fallback must handle plain-string responses,
        not just OpenAI-style response objects."""
        from unittest.mock import patch
        from extractor import _extract_facts_llm
        json_response = '[{"category": "preference", "content": "I like tea"}]'
        long_content = "I like tea and coffee every morning and I also enjoy hiking on weekends"
        with patch("agent.auxiliary_client.call_llm", return_value=json_response):
            facts = _extract_facts_llm(long_content)
        assert len(facts) >= 1, f"Expected plain-string response parsed, got {facts}"
        assert any("tea" in f.get("content", "") for f in facts), \
            f"Expected 'tea' in extracted facts, got {facts}"

    def test_value_extractor_is_now_transition(self):
        """#173: 'is now'/'are now' must trigger value-supersession
        detection."""
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from value_extractor import _TRANSITION_VERBS
        assert _TRANSITION_VERBS.search("My salary is now $500"), \
            "'is now' should be a transition verb"
        assert _TRANSITION_VERBS.search("The costs are now higher"), \
            "'are now' should be a transition verb"
        assert not _TRANSITION_VERBS.search("My salary is $500"), \
            "'is' without 'now' should not be a transition"



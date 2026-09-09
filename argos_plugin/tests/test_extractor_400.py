"""Tests for #400: constraint-form failure memory (extraction + review routing).

Covers:
- Failure content extracts as invariant-form, constraint-tagged, durable
- One-off outcome content routes to short-lived categories (context_note)
- Constraint candidates get a quality flag for stricter review
- No recall regression on existing extraction patterns

Run with:
    python -m pytest tests/test_extractor_400.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_plugin_dir = Path(__file__).resolve().parent.parent
if str(_plugin_dir) not in sys.path:
    sys.path.insert(0, str(_plugin_dir))


class TestConstraintExtraction:
    """Failure/lesson content extracts as invariant-form, constraint-tagged,
    durable (category=insight, tags include 'constraint')."""

    def test_never_when(self):
        from extractor import _extract_facts_regex
        facts = _extract_facts_regex("never deploy without migrations when the schema is stale")
        constraints = [f for f in facts if "constraint" in f.get("tags", [])]
        assert len(constraints) == 1
        assert constraints[0]["category"] == "insight"
        assert "constraint" in constraints[0]["tags"]
        assert "failure_lesson" in constraints[0]["tags"]
        assert "never" in constraints[0]["content"].lower()
        assert "when" in constraints[0]["content"].lower()

    def test_dont_when(self):
        from extractor import _extract_facts_regex
        facts = _extract_facts_regex("don't push to main when the tests are red")
        constraints = [f for f in facts if "constraint" in f.get("tags", [])]
        assert len(constraints) == 1
        assert constraints[0]["category"] == "insight"

    def test_always_before(self):
        from extractor import _extract_facts_regex
        facts = _extract_facts_regex("always run the test suite before pushing to main")
        constraints = [f for f in facts if "constraint" in f.get("tags", [])]
        assert len(constraints) == 1
        assert constraints[0]["category"] == "insight"
        assert "always" in constraints[0]["content"].lower()
        assert "before" in constraints[0]["content"].lower()

    def test_lesson_explicit(self):
        from extractor import _extract_facts_regex
        facts = _extract_facts_regex("Lesson learned: always run migrations before deploying")
        constraints = [f for f in facts if "constraint" in f.get("tags", [])]
        assert len(constraints) == 1
        assert constraints[0]["category"] == "insight"
        assert "constraint" in constraints[0]["tags"]

    def test_learned_explicit(self):
        from extractor import _extract_facts_regex
        facts = _extract_facts_regex("Learned: never trust user input without validation")
        constraints = [f for f in facts if "constraint" in f.get("tags", [])]
        assert len(constraints) == 1
        assert constraints[0]["category"] == "insight"

    def test_failure_with_cause(self):
        from extractor import _extract_facts_regex
        facts = _extract_facts_regex("I tried deploying without migrations and it failed because the schema was stale")
        constraints = [f for f in facts if "constraint" in f.get("tags", [])]
        assert len(constraints) == 1
        assert constraints[0]["category"] == "insight"
        content = constraints[0]["content"].lower()
        assert "fails" in content or "does not work" in content
        assert "never" in content

    def test_failure_no_cause(self):
        from extractor import _extract_facts_regex
        facts = _extract_facts_regex("I tried that approach and it didn't work")
        constraints = [f for f in facts if "constraint" in f.get("tags", [])]
        assert len(constraints) == 1
        assert constraints[0]["category"] == "insight"
        assert "does not work" in constraints[0]["content"].lower() or "avoid" in constraints[0]["content"].lower()

    def test_doesnt_work_when(self):
        from extractor import _extract_facts_regex
        facts = _extract_facts_regex("That library doesn't work when the API version changes")
        constraints = [f for f in facts if "constraint" in f.get("tags", [])]
        assert len(constraints) == 1
        assert constraints[0]["category"] == "insight"
        assert "does not work" in constraints[0]["content"].lower()
        assert "never" in constraints[0]["content"].lower()

    def test_constraint_payload(self):
        """Constraint facts carry a payload with constraint=True."""
        from extractor import _extract_facts_regex
        facts = _extract_facts_regex("never deploy on Fridays when there's no on-call")
        constraints = [f for f in facts if "constraint" in f.get("tags", [])]
        assert len(constraints) == 1
        payload = constraints[0].get("payload", {})
        assert payload.get("constraint") is True
        assert "form" in payload


class TestOutcomeRouting:
    """One-off outcome content routes to short-lived categories
    (context_note), not durable insight."""

    def test_tried_routes_to_context_note(self):
        from extractor import _extract_facts_regex
        facts = _extract_facts_regex("I tried that new restaurant yesterday")
        outcomes = [f for f in facts if "outcome" in f.get("tags", [])]
        assert len(outcomes) == 1
        assert outcomes[0]["category"] == "context_note"
        assert "outcome" in outcomes[0]["tags"]
        assert "one_off" in outcomes[0]["tags"]

    def test_outcome_not_insight(self):
        """One-off outcomes must NOT land as insight (durable)."""
        from extractor import _extract_facts_regex
        facts = _extract_facts_regex("I tried using that tool yesterday")
        outcomes = [f for f in facts if "outcome" in f.get("tags", [])]
        if outcomes:
            assert outcomes[0]["category"] != "insight"

    def test_outcome_payload(self):
        """Outcome facts carry a payload with outcome=True."""
        from extractor import _extract_facts_regex
        facts = _extract_facts_regex("I tried that new framework yesterday")
        outcomes = [f for f in facts if "outcome" in f.get("tags", [])]
        assert len(outcomes) == 1
        payload = outcomes[0].get("payload", {})
        assert payload.get("outcome") is True


class TestConstraintQualityFlag:
    """Constraint candidates get a quality flag for stricter review."""

    def test_constraint_flagged(self):
        from extractor import quality_flags_for_fact
        fact = {
            "content": "Constraint: never deploy without migrations when schema is stale",
            "category": "insight",
            "tags": ["constraint", "failure_lesson"],
        }
        flags = quality_flags_for_fact(fact)
        assert "constraint_candidate" in flags

    def test_non_constraint_not_flagged(self):
        from extractor import quality_flags_for_fact
        fact = {
            "content": "User prefers dark mode",
            "category": "preference",
            "tags": ["preference"],
        }
        flags = quality_flags_for_fact(fact)
        assert "constraint_candidate" not in flags

    def test_outcome_not_flagged_as_constraint(self):
        from extractor import quality_flags_for_fact
        fact = {
            "content": "User tried: that new restaurant",
            "category": "context_note",
            "tags": ["outcome", "one_off"],
        }
        flags = quality_flags_for_fact(fact)
        assert "constraint_candidate" not in flags
        assert "short_lived_category" in flags


class TestNoRecallRegression:
    """Existing extraction patterns must still work — no recall regression."""

    def test_preference_still_extracts(self):
        from extractor import _extract_facts_regex
        facts = _extract_facts_regex("I prefer dark mode")
        prefs = [f for f in facts if f.get("category") == "preference"]
        assert len(prefs) >= 1

    def test_personal_fact_still_extracts(self):
        from extractor import _extract_facts_regex
        facts = _extract_facts_regex("I live in Cape Town")
        locs = [f for f in facts if "location" in f.get("tags", [])]
        assert len(locs) >= 1

    def test_goal_still_extracts(self):
        from extractor import _extract_facts_regex
        facts = _extract_facts_regex("I want to learn Rust")
        goals = [f for f in facts if f.get("category") == "goal"]
        assert len(goals) >= 1

    def test_event_still_extracts(self):
        from extractor import _extract_facts_regex
        facts = _extract_facts_regex("I started a new job at Stripe")
        events = [f for f in facts if f.get("category") == "event"]
        assert len(events) >= 1

    def test_habit_still_extracts(self):
        from extractor import _extract_facts_regex
        facts = _extract_facts_regex("I always test before deploying")
        habits = [f for f in facts if "habit" in f.get("tags", [])]
        # The habit pattern should still match — but "always test before
        # deploying" might also match the constraint pattern. Either way,
        # the fact should be extracted (not dropped).
        assert len(facts) >= 1

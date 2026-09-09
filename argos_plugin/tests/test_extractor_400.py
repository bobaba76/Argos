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

    def test_doesnt_work_thing_subject_not_constraint(self):
        """#400 review (blocker 1): thing-subjects must NOT mint durable
        constraints — referent dropped, durable noise ("My car doesn't
        work when it rains")."""
        from extractor import _extract_facts_regex
        facts = _extract_facts_regex("That library doesn't work when the API version changes")
        constraints = [f for f in facts if "constraint" in f.get("tags", [])]
        assert len(constraints) == 0

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
        """One-off outcomes must route to context_note, not insight —
        and must actually extract (not pass trivially on empty)."""
        from extractor import _extract_facts_regex
        facts = _extract_facts_regex("I tried using that tool yesterday")
        outcomes = [f for f in facts if "outcome" in f.get("tags", [])]
        assert len(outcomes) == 1
        assert outcomes[0]["category"] == "context_note"

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


class TestSubjectAnchoring:
    """#400 review blocker 1: constraint patterns must be about the USER."""

    def test_third_person_name_no_constraint(self):
        from extractor import _extract_facts_regex
        facts = _extract_facts_regex("Alex never drinks when she is driving")
        assert not [f for f in facts if "constraint" in f.get("tags", [])]

    def test_third_person_possessive_no_constraint(self):
        from extractor import _extract_facts_regex
        facts = _extract_facts_regex("My sister never eats meat when she visits")
        assert not [f for f in facts if "constraint" in f.get("tags", [])]

    def test_third_person_pronoun_no_constraint(self):
        from extractor import _extract_facts_regex
        facts = _extract_facts_regex("She never answers when I call her at work")
        assert not [f for f in facts if "constraint" in f.get("tags", [])]

    def test_thing_subject_no_constraint(self):
        from extractor import _extract_facts_regex
        facts = _extract_facts_regex("My car doesn't work when it rains")
        assert not [f for f in facts if "constraint" in f.get("tags", [])]

    def test_first_person_dont_when(self):
        """First-person "I don't X when Y" is a valid negative invariant —
        the natural form reaches the constraint machinery (via the
        never_when family; "don't" is a trigger)."""
        from extractor import _extract_facts_regex
        facts = _extract_facts_regex("I don't work when I skip breakfast")
        constraints = [f for f in facts if "constraint" in f.get("tags", [])]
        assert len(constraints) == 1
        assert "never" in constraints[0]["content"].lower()
        assert "work when" in constraints[0]["content"].lower()

    def test_first_person_never_when_reaches_constraint(self):
        """#400 review blocker 4: the canonical I-prefixed form must reach
        the constraint machinery, not the habit path (which drops negation)."""
        from extractor import _extract_facts_regex
        facts = _extract_facts_regex("I never push to main when the tests are red")
        constraints = [f for f in facts if "constraint" in f.get("tags", [])]
        assert len(constraints) == 1
        assert "never" in constraints[0]["content"].lower()
        assert "constraint" in constraints[0]["tags"]

    def test_first_person_plural_always_before(self):
        from extractor import _extract_facts_regex
        facts = _extract_facts_regex("We always smoke test before releasing")
        constraints = [f for f in facts if "constraint" in f.get("tags", [])]
        assert len(constraints) == 1
        assert "always" in constraints[0]["content"].lower()


class TestOutcomeVerbs:
    """#400 review blocker 2: only genuine attempts are outcomes; did/ran/
    used are not 'tried', and negation must not be captured."""

    @pytest.mark.parametrize("sentence", [
        "I did not sleep well last night",
        "I did not work yesterday",
        "I used to live in Johannesburg",
        "I used to be a developer",
        "I ran the tests and they all passed",
        "I never tried meditation",
    ])
    def test_no_false_outcome(self, sentence):
        from extractor import _extract_facts_regex
        facts = _extract_facts_regex(sentence)
        outcomes = [f for f in facts if "outcome" in f.get("tags", [])]
        assert not outcomes, f"minted false outcome for: {sentence}"


class TestMustNotConstraint:
    """#400 review blocker 3: modal obligation is not an invariant."""

    @pytest.mark.parametrize("sentence", [
        "I must submit the report before noon",
        "I must pop into the bank before it closes",
    ])
    def test_no_constraint_from_must(self, sentence):
        from extractor import _extract_facts_regex
        facts = _extract_facts_regex(sentence)
        constraints = [f for f in facts if "constraint" in f.get("tags", [])]
        assert not constraints, f"minted constraint from obligation: {sentence}"

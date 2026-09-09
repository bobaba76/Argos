"""Tests for #395: reject question/fragment candidates at extraction time.

Regression tests for the two real captures that motivated the fix:
- "User has: A 12GB CARD???" (question_or_request)
- "User has: found a REALLY good" (sentence_fragment via _DANGLING_END_RE)

Also includes a recall probe for the _DANGLING_END_RE: complete facts
ending in a word on the dangling list (e.g. "User likes dark mode
better") must NOT be falsely rejected as fragments.

Run with:
    python -m pytest tests/test_extractor_395.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_plugin_dir = Path(__file__).resolve().parent.parent
if str(_plugin_dir) not in sys.path:
    sys.path.insert(0, str(_plugin_dir))


class TestQualityFlags:
    """quality_flags_for_fact must flag questions and fragments correctly."""

    def test_question_flagged(self):
        from extractor import quality_flags_for_fact
        fact = {"content": "User has: A 12GB CARD???", "category": "personal_fact"}
        flags = quality_flags_for_fact(fact)
        assert "question_or_request" in flags

    def test_dangling_end_flagged_as_fragment(self):
        from extractor import quality_flags_for_fact
        fact = {"content": "User has: found a REALLY good", "category": "personal_fact"}
        flags = quality_flags_for_fact(fact)
        assert "sentence_fragment" in flags

    def test_fragment_start_flagged(self):
        from extractor import quality_flags_for_fact
        fact = {"content": "or maybe the config is wrong", "category": "context_note"}
        flags = quality_flags_for_fact(fact)
        assert "sentence_fragment" in flags

    def test_fragment_end_flagged(self):
        from extractor import quality_flags_for_fact
        fact = {"content": "User went to the store with", "category": "personal_fact"}
        flags = quality_flags_for_fact(fact)
        assert "sentence_fragment" in flags


class TestDanglingEndRecall:
    """Recall probe: _DANGLING_END_RE must NOT flag complete facts that
    happen to end in a word on the dangling list. These are legitimate
    facts, not truncated fragments.

    The reviewer flagged this as a data-loss risk: "User likes dark mode
    better" ends in 'better' and would be rejected as a fragment.

    The narrowed regex only matches when a determiner/intensifier precedes
    the dangling word (e.g. "found a REALLY good"), not when the word is a
    predicate adjective in a complete clause.
    """

    @pytest.mark.parametrize("content", [
        "User likes dark mode better",           # ends in 'better' — complete
        "User thinks the new design is nice",    # ends in 'nice' — complete
        "User prefers the old interface",        # ends in 'old' — complete
        "User's favorite coffee is really good", # ends in 'good' — complete
        "User says the performance is fast",     # ends in 'fast' — complete
        "User wants the bigger apartment",       # ends in 'bigger' — complete
        "User likes Python a lot",               # not a dangling word
        "User works at a really big company",    # 'company' is not dangling
    ])
    def test_complete_facts_not_flagged(self, content):
        from extractor import quality_flags_for_fact
        fact = {"content": content, "category": "personal_fact"}
        flags = quality_flags_for_fact(fact)
        assert "sentence_fragment" not in flags, (
            f"Complete fact falsely flagged as fragment: {content!r} "
            f"(flags: {flags})"
        )

    @pytest.mark.parametrize("content", [
        "User has: found a REALLY good",         # truncated — determiner + dangling
        "User has: found a really",              # truncated — determiner + dangling
        "User has: the very",                    # truncated — determiner + dangling
    ])
    def test_truncated_facts_flagged(self, content):
        from extractor import quality_flags_for_fact
        fact = {"content": content, "category": "personal_fact"}
        flags = quality_flags_for_fact(fact)
        assert "sentence_fragment" in flags, (
            f"Truncated fact not flagged: {content!r} (flags: {flags})"
        )


class TestExtractionRejection:
    """The extraction loop must DROP candidates flagged as
    question_or_request or sentence_fragment before they reach
    save_candidate. This is the core #395 fix.

    These tests mock _extract_facts_regex to return controlled candidates
    so we can verify the filtering logic in _extract_from_turn_impl
    without depending on which regex patterns happen to match.
    """

    def test_question_candidate_dropped(self, monkeypatch):
        """A fact with '?' in content is flagged question_or_request and
        dropped at extraction time."""
        from extractor import _extract_from_turn_impl
        # Mock the regex stage to return a question candidate.
        monkeypatch.setattr(
            "extractor._extract_facts_regex",
            lambda content: [{
                "content": "User has: A 12GB CARD???",
                "category": "personal_fact",
                "payload": {},
            }],
        )
        results = _extract_from_turn_impl(
            "irrelevant text",
            "",
            use_llm_fallback=False,
        )
        contents = [r.get("content", "") for r in results]
        assert not any("12GB" in c and "???" in c for c in contents), (
            f"Question candidate was not dropped: {contents}"
        )

    def test_fragment_candidate_dropped(self, monkeypatch):
        """A fact ending in a dangling qualifier is flagged
        sentence_fragment and dropped at extraction time."""
        from extractor import _extract_from_turn_impl
        monkeypatch.setattr(
            "extractor._extract_facts_regex",
            lambda content: [{
                "content": "User has: found a REALLY good",
                "category": "personal_fact",
                "payload": {},
            }],
        )
        results = _extract_from_turn_impl(
            "irrelevant text",
            "",
            use_llm_fallback=False,
        )
        contents = [r.get("content", "") for r in results]
        assert not any("REALLY good" in c for c in contents), (
            f"Fragment candidate was not dropped: {contents}"
        )

    def test_normal_fact_not_dropped(self, monkeypatch):
        """A normal complete fact must NOT be dropped."""
        from extractor import _extract_from_turn_impl
        monkeypatch.setattr(
            "extractor._extract_facts_regex",
            lambda content: [{
                "content": "User lives in Cape Town",
                "category": "personal_fact",
                "payload": {},
            }],
        )
        results = _extract_from_turn_impl(
            "irrelevant text",
            "",
            use_llm_fallback=False,
        )
        contents = [r.get("content", "") for r in results]
        assert any("Cape Town" in c for c in contents), (
            f"Normal fact was dropped: {contents}"
        )

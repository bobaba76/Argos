"""Golden regression-fixture suite for the extractor's regex stage (#132).

This is the regression baseline for ``_extract_facts_regex`` — the
deterministic, no-LLM Stage 1 of extraction. Every proposed extractor
change (newline handling #133, abbreviation handling #135, config refactor
#134, lexicon generalization #136) is gated by this suite: a behavior
change must flip a fixture deliberately and visibly, not silently.

Design notes
------------
- **Data-driven**: fixtures are a single table of ``(id, input, expected,
  known_limit)`` for reviewability — a reviewer reads the table, not a
  wall of test functions.
- **Hermetic**: runs with ``ARGOS_HERMETIC_TESTS=1`` (set by the gate
  script) so no real Hermes runtime / LLM is touched. The regex stage
  never calls the LLM regardless, but the hermetic flag keeps the import
  environment deterministic.
- **0 LLM calls**: ``_extract_facts_regex`` is Stage 1 only; it never
  imports ``agent.auxiliary_client``. This suite asserts that indirectly
  by never exercising the LLM path.
- **Captures CURRENT behavior** as the baseline, including known-imperfect
  cases. Those are marked ``known_limit`` with a short note so a future
  fix is a deliberate, visible flip rather than a silent regression.

Run with (Hermes venv python, offline):
    python -m pytest tests/test_extraction_golden.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_plugin_dir = Path(__file__).resolve().parent.parent
if str(_plugin_dir) not in sys.path:
    sys.path.insert(0, str(_plugin_dir))


# ---------------------------------------------------------------------------
# Fixture: a couple of learned role words so the "my X is Y" relationship
# path is exercised (mirrors what the provider does at initialize()).
# Snapshotted/restored so this never leaks into other test modules.
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True, scope="module")
def _learned_role_words():
    import extractor
    saved = set(extractor._extra_role_words)
    extractor.set_role_words({"doula", "housemate"})
    yield
    extractor.set_role_words(saved)


# ---------------------------------------------------------------------------
# Golden fixture table.
#
# Each entry:
#   id        — stable identifier used in the parametrized test id
#   input     — the user message fed to _extract_facts_regex
#   expected  — list of {category, content, tags, durability} asserted
#               exactly against the regex stage output (order-sensitive,
#               since sentence order is preserved)
#   known_limit — None, or a short note pinning a known-imperfect behavior
#                 that a future issue is expected to flip deliberately
# ---------------------------------------------------------------------------

GOLDEN: list[dict] = [
    # --- identity ----------------------------------------------------------
    {
        "id": "identity",
        "input": "I am a backend engineer. I'm 34 years old.",
        "expected": [
            {"category": "personal_fact", "content": "User is backend engineer",
             "tags": ["personal_fact", "identity"], "durability": "durable"},
            {"category": "personal_fact", "content": "User is 34 years old",
             "tags": ["personal_fact", "identity"], "durability": "durable"},
        ],
    },
    {
        "id": "identity_transient_rejected",
        "input": "I am tired and hungry right now. I'm ready.",
        "expected": [],
        "known_limit": None,
    },
    # --- possession / usage -----------------------------------------------
    {
        "id": "have_use",
        "input": "I use Vim for editing. I take FocusTool in the morning.",
        "expected": [
            {"category": "personal_fact", "content": "User uses/has: Vim for editing",
             "tags": ["personal_fact"], "durability": "durable"},
            {"category": "personal_fact", "content": "User uses/has: FocusTool in the morning",
             "tags": ["personal_fact"], "durability": "durable"},
        ],
    },
    {
        "id": "attribute",
        "input": "I have a dog named Rex. I own a vintage guitar.",
        "expected": [
            {"category": "personal_fact", "content": "User has: a dog named Rex",
             "tags": ["personal_fact"], "durability": "durable"},
            {"category": "personal_fact", "content": "User has: a vintage guitar",
             "tags": ["personal_fact"], "durability": "durable"},
        ],
    },
    {
        "id": "my_x_is_attribute",
        "input": "My favorite editor is Vim. My role is engineer.",
        "expected": [
            {"category": "personal_fact", "content": "User's favorite editor: Vim",
             "tags": ["personal_fact", "favorite editor"], "durability": "durable"},
            {"category": "personal_fact", "content": "User's role: engineer",
             "tags": ["personal_fact", "role"], "durability": "durable"},
        ],
    },
    # --- work / location ---------------------------------------------------
    {
        "id": "work",
        "input": "I work at Google. I'm working at a startup called Acme.",
        "expected": [
            {"category": "personal_fact", "content": "User works at: Google",
             "tags": ["personal_fact", "work"], "durability": "durable"},
            {"category": "personal_fact", "content": "User works at: a startup called Acme",
             "tags": ["personal_fact", "work"], "durability": "durable"},
        ],
    },
    {
        "id": "location",
        "input": "I live in Berlin. I'm from Tokyo.",
        "expected": [
            {"category": "personal_fact", "content": "User location: Berlin",
             "tags": ["personal_fact", "location"], "durability": "durable"},
            {"category": "personal_fact", "content": "User location: Tokyo",
             "tags": ["personal_fact", "location"], "durability": "durable"},
        ],
    },
    # --- preferences -------------------------------------------------------
    {
        "id": "preference",
        "input": "I prefer dark mode. I hate meetings before 10am.",
        "expected": [
            {"category": "preference", "content": "User prefers: dark mode",
             "tags": ["preference"], "durability": "durable"},
            {"category": "preference", "content": "User dislikes: meetings before 10am",
             "tags": ["preference"], "durability": "durable"},
        ],
    },
    {
        "id": "preference_negated",
        "input": "I don't like coffee. I can't stand loud noises.",
        "expected": [
            {"category": "preference", "content": "User dislikes: coffee",
             "tags": ["preference"], "durability": "durable"},
            {"category": "preference", "content": "User dislikes: loud noises",
             "tags": ["preference"], "durability": "durable"},
        ],
    },
    {
        "id": "assistant_directive",
        "input": "Always give me the short version. Never use code comments. Call me Mike.",
        "expected": [
            {"category": "preference", "content": "User directive: Always give me the short version",
             "tags": ["preference", "assistant_side"], "durability": "durable"},
            {"category": "preference", "content": "User directive: Never use code comments",
             "tags": ["preference", "assistant_side"], "durability": "durable"},
        ],
    },
    {
        "id": "habit",
        "input": "I always test before deploying. I never push to main.",
        "expected": [
            {"category": "preference", "content": "User habit: test before deploying",
             "tags": ["preference", "habit"], "durability": "durable"},
            {"category": "preference", "content": "User habit: push to main",
             "tags": ["preference", "habit"], "durability": "durable"},
        ],
    },
    # --- goals / insights --------------------------------------------------
    {
        "id": "goal",
        "input": "I'm working on a side project. I want to learn Rust.",
        "expected": [
            {"category": "goal", "content": "User goal: a side project",
             "tags": ["goal"], "durability": "temporary"},
            {"category": "goal", "content": "User goal: learn Rust",
             "tags": ["goal"], "durability": "temporary"},
        ],
    },
    {
        "id": "insight",
        "input": "I tend to overthink. I realized I need breaks.",
        "expected": [
            {"category": "insight", "content": "User self-observation: overthink",
             "tags": ["insight", "self_observation"], "durability": "durable"},
            {"category": "insight", "content": "User self-observation: I need breaks",
             "tags": ["insight", "self_observation"], "durability": "durable"},
        ],
    },
    # --- events / transitions / ongoing -----------------------------------
    {
        "id": "event",
        "input": "I started a new job. I quit smoking. I launched the app.",
        "expected": [
            {"category": "event", "content": "Life event: user started a new job",
             "tags": ["event"], "durability": "temporary"},
            {"category": "event", "content": "Life event: user quit smoking",
             "tags": ["event"], "durability": "temporary"},
            {"category": "event", "content": "Life event: user launched the app",
             "tags": ["event"], "durability": "temporary"},
        ],
    },
    {
        "id": "switch",
        "input": "I switched from Vim to Emacs. I moved from London to Lisbon.",
        "expected": [
            {"category": "event", "content": "User switched from Vim to Emacs",
             "tags": ["event", "transition"], "durability": "temporary"},
            {"category": "event", "content": "User switched from London to Lisbon",
             "tags": ["event", "transition"], "durability": "temporary"},
        ],
    },
    {
        "id": "ongoing",
        "input": "I've been feeling anxious. I've been using Docker.",
        "expected": [
            {"category": "context_note", "content": "User has been feeling anxious",
             "tags": ["context_note", "ongoing"], "durability": "temporary"},
            {"category": "context_note", "content": "User has been using Docker",
             "tags": ["context_note", "ongoing"], "durability": "temporary"},
        ],
    },
    # --- relationships -----------------------------------------------------
    {
        "id": "relationship",
        "input": "Alice is my wife. Bob is my manager.",
        "expected": [
            {"category": "relationship", "content": "Alice is the user's wife",
             "tags": ["relationship", "alice", "wife"], "durability": "durable"},
            {"category": "relationship", "content": "Bob is the user's manager",
             "tags": ["relationship", "bob", "manager"], "durability": "durable"},
        ],
    },
    {
        "id": "relationship_my_x_is",
        "input": "My doula is Carla. My housemate is Dave.",
        "expected": [
            {"category": "relationship", "content": "User's doula is Carla",
             "tags": ["relationship", "doula"], "durability": "durable"},
            {"category": "relationship", "content": "User's housemate is Dave",
             "tags": ["relationship", "housemate"], "durability": "durable"},
        ],
    },
    # --- rejection paths ---------------------------------------------------
    {
        "id": "assistant_speak_rejected",
        "input": "I'll search authoritative sources for that.",
        "expected": [],
        "known_limit": None,
    },
    {
        "id": "short_rejected",
        "input": "hi",
        "expected": [],
        "known_limit": None,
    },
    {
        "id": "empty",
        "input": "",
        "expected": [],
        "known_limit": None,
    },
    # --- newline-aware splitting (#133 — line boundaries preserved) --------
    {
        "id": "multiline_paste",
        "input": "I use Vim\nI prefer dark mode",
        "expected": [
            {"category": "preference", "content": "User prefers: dark mode",
             "tags": ["preference"], "durability": "durable"},
        ],
        "known_limit": None,
    },
    {
        "id": "multiline_blank_line_boundary",
        "input": "I like tea.\n\nI hate coffee.",
        "expected": [],
        "known_limit": None,
    },
    {
        "id": "multiline_list_known_limit",
        "input": "1. Use Vim\n2. Prefer dark mode",
        "expected": [],
        "known_limit": "list-number '.' still splits '1.' into its own "
                       "fragment; the bare 'Use Vim' / 'Prefer dark mode' "
                       "fragments lack a first-person subject so no pattern "
                       "matches. A future list-aware splitter would handle "
                       "numbered list items; out of #133's scope.",
    },
    # --- abbreviation-aware splitting (#135 — false-splits rejoined) ------
    {
        "id": "abbreviation_dr",
        "input": "I met Dr. Smith. He is nice.",
        "expected": [],
        "known_limit": None,
    },
    {
        "id": "abbreviation_mr",
        "input": "I met Mr. Jones. He is my boss.",
        "expected": [],
        "known_limit": None,
    },
    {
        "id": "work_abbreviation",
        "input": "I work at Inc. with my friend.",
        "expected": [],
        "known_limit": None,
    },
    {
        "id": "work_abbreviation_pty_ltd",
        "input": "I work at Pty. Ltd. with my team.",
        "expected": [],
        "known_limit": None,
    },
    {
        "id": "work_abbreviation_multitoken_kept",
        "input": "I work at Acme Inc. They are great.",
        "expected": [
            {"category": "personal_fact", "content": "User works at: Acme Inc",
             "tags": ["personal_fact", "work"], "durability": "durable"},
        ],
        "known_limit": None,
    },
    {
        "id": "abbreviation_e_g_known_limit",
        "input": "I like tools e.g. Vim and Emacs.",
        "expected": [
            {"category": "preference", "content": "User prefers: tools e",
             "tags": ["preference"], "durability": "durable"},
        ],
        "known_limit": "the merge rejoins the 'e.g.' false-split, but the "
                       "non-greedy _PREFERENCE_RE still terminates at the "
                       "abbreviation's period, capturing a truncated 'tools e'. "
                       "A regex-termination fix (abbreviation-aware terminator) "
                       "is a separate, larger change; documented as a known "
                       "low-frequency merge edge per #135.",
    },
    # --- non-English / SA-business context --------------------------------
    {
        "id": "non_english_sa",
        "input": "Ek werk by 'n maatskappy in Kaapstad. I prefer Afrikaans.",
        "expected": [
            {"category": "preference", "content": "User prefers: Afrikaans",
             "tags": ["preference"], "durability": "durable"},
        ],
        "known_limit": "Afrikaans 'Ek werk by ...' is not matched by the "
                       "English work pattern; only the English clause extracts. "
                       "An af.json pattern pack (#134) would close this gap.",
    },
]


def _project(facts: list[dict]) -> list[dict]:
    """Project regex-stage facts to the comparable subset."""
    return [
        {
            "category": f.get("category"),
            "content": f.get("content"),
            "tags": f.get("tags"),
            "durability": f.get("durability"),
        }
        for f in facts
    ]


@pytest.mark.parametrize(
    "case",
    GOLDEN,
    ids=[c["id"] for c in GOLDEN],
)
def test_regex_stage_golden(case):
    """Each golden fixture must match the regex stage output exactly."""
    from extractor import _extract_facts_regex

    facts = _extract_facts_regex(case["input"])
    projected = _project(facts)
    assert projected == case["expected"], (
        f"\n  input: {case['input']!r}\n  got:    {projected}\n  want:   {case['expected']}"
    )


def test_golden_corpus_covers_all_categories():
    """The golden corpus must exercise every extractor category at least
    once, so no category regresses unobserved."""
    from extractor import _extract_facts_regex

    categories_seen: set[str] = set()
    for case in GOLDEN:
        for fact in _extract_facts_regex(case["input"]):
            categories_seen.add(fact.get("category"))
    # Every category the regex stage can emit must be covered.
    required = {"personal_fact", "preference", "insight", "event",
                "relationship", "goal", "context_note"}
    missing = required - categories_seen
    assert not missing, f"Golden corpus missing categories: {missing}"


def test_known_limit_markers_are_documented():
    """Every KNOWN-LIMIT fixture must carry an explanatory note so the
    reason it is pinned is visible to a future reader."""
    for case in GOLDEN:
        if case.get("known_limit") is not None:
            assert isinstance(case["known_limit"], str) and len(case["known_limit"]) > 10, (
                f"Fixture {case['id']} has a known_limit marker but no real note"
            )


# ---------------------------------------------------------------------------
# #136 — extensible lexicons: a domain term added via config is recognized
# without editing extractor.py. These tests verify the set_*() functions
# and dynamic regex rebuild work end-to-end.
# ---------------------------------------------------------------------------

class TestExtensibleLexicons:
    """Verify that preference verbs, event verbs, and transient words are
    extensible via set_*() calls (#136), and that the regex is rebuilt
    with re.escape (so a verb with regex metacharacters is safe)."""

    def setup_method(self):
        import extractor
        self._saved_pref = set(extractor._extra_preference_verbs)
        self._saved_event = set(extractor._extra_event_verbs)
        self._saved_trans = set(extractor._extra_transient_words)
        self._saved_pref_re = extractor._PREFERENCE_RE
        self._saved_event_re = extractor._EVENT_RE

    def teardown_method(self):
        import extractor
        extractor.set_preference_verbs(self._saved_pref)
        extractor.set_event_verbs(self._saved_event)
        extractor.set_transient_words(self._saved_trans)
        extractor._PREFERENCE_RE = self._saved_pref_re
        extractor._EVENT_RE = self._saved_event_re

    def test_extra_preference_verb_recognized(self):
        """A domain-specific preference verb added via set_preference_verbs
        is recognized by _extract_facts_regex without editing extractor.py."""
        import extractor
        extractor.set_preference_verbs({"adore"})
        facts = extractor._extract_facts_regex("I adore Vim for editing.")
        prefs = [f for f in facts if f["category"] == "preference"]
        assert len(prefs) == 1
        assert "Vim for editing" in prefs[0]["content"]

    def test_extra_event_verb_recognized(self):
        """A domain-specific event verb added via set_event_verbs is
        recognized by _extract_facts_regex without editing extractor.py."""
        import extractor
        extractor.set_event_verbs({"deployed"})
        facts = extractor._extract_facts_regex("I deployed the new service.")
        events = [f for f in facts if f["category"] == "event"]
        assert len(events) == 1
        assert "deployed" in events[0]["content"]

    def test_extra_transient_word_rejects_identity(self):
        """A domain-specific transient word added via set_transient_words
        causes the identity gate to reject "I am <word>" as non-durable."""
        import extractor
        # "deployed" is not in the base transient set — "I am deployed in
        # the field." should produce a personal_fact before adding it.
        # (Sentence must be >= _MIN_LENGTH=15 chars to be processed.)
        facts_before = extractor._extract_facts_regex("I am deployed in the field.")
        assert any(f["category"] == "personal_fact" for f in facts_before)
        # Now add "deployed" as transient and verify it's rejected.
        extractor.set_transient_words({"deployed"})
        facts_after = extractor._extract_facts_regex("I am deployed in the field.")
        assert not any(f["category"] == "personal_fact" for f in facts_after)

    def test_regex_escape_on_extra_verbs(self):
        """Extra verbs with regex metacharacters are re.escape()'d, so they
        match literally and cannot cause ReDoS (#136 ReDoS audit)."""
        import extractor
        # "shipped+live" contains a regex metacharacter (+). It must match
        # literally, not as "shipped" followed by one-or-more "live".
        extractor.set_event_verbs({"shipped+live"})
        facts = extractor._extract_facts_regex("I shipped+live the app.")
        events = [f for f in facts if f["category"] == "event"]
        assert len(events) == 1
        assert "shipped+live" in events[0]["content"]
        # Verify the literal "+" did NOT act as a quantifier: "shippedlive"
        # (without the +) should NOT match.
        facts_no_plus = extractor._extract_facts_regex("I shippedlive the app.")
        assert not any(f["category"] == "event" for f in facts_no_plus)

    def test_base_verbs_still_work_after_extension(self):
        """Adding extra verbs does not remove the base verbs — both
        base and extra are in the combined alternation."""
        import extractor
        extractor.set_event_verbs({"deployed"})
        # Base verb "started" still works.
        facts = extractor._extract_facts_regex("I started a new job.")
        assert any(f["category"] == "event" and "started" in f["content"]
                   for f in facts)
        # Extra verb "deployed" also works.
        facts2 = extractor._extract_facts_regex("I deployed the service.")
        assert any(f["category"] == "event" and "deployed" in f["content"]
                   for f in facts2)

"""Tests for extraction-time dedupe hardening.

Covers:
1. Store layer: find_semantic_duplicate() — restated fact matches an active
   memory; distinct facts don't; threshold honored; superseded/quarantined
   memories excluded; fail-soft on missing/broken embedder.
2. Provider layer: the candidate-creation path (sync_turn) skips proposals
   already covered by an active memory, still emits genuinely new facts, and
   still emits candidates when the embedder errors.
3. Reviewer lane: bounded retry (1 retry with backoff) on the reviewer LLM
   call; fail-soft stays pending_user_confirmation.

Run with (Hermes venv python, offline):
    python -m pytest tests/test_extraction_dedupe.py -v
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

_plugin_dir = Path(__file__).resolve().parent.parent
if str(_plugin_dir) not in sys.path:
    sys.path.insert(0, str(_plugin_dir))


@pytest.fixture
def store(tmp_path):
    """A fresh DuckDBMemoryStore with the BGE embedder."""
    from store import DuckDBMemoryStore
    from embeddings import LocalEmbedder
    embedder = LocalEmbedder("BAAI/bge-small-en-v1.5")
    s = DuckDBMemoryStore(
        tmp_path / "test.duckdb", user_id="test_user", embedder=embedder,
    )
    yield s
    s.close()


# ---------------------------------------------------------------------------
# Store layer: find_semantic_duplicate
# ---------------------------------------------------------------------------

class TestFindSemanticDuplicate:
    def test_restated_fact_matches_active_memory(self, store):
        """A fact already in the cabinet, restated differently, is found."""
        rec = store.remember(
            category="personal_fact",
            content="User's full name is Shaun Davies",
            dedup=False,
        )
        assert rec is not None
        dup = store.find_semantic_duplicate(
            "My full name is Shaun Davies", min_similarity=0.88
        )
        assert dup is not None
        assert dup.memory_id == rec.memory_id
        assert dup.similarity >= 0.88

    def test_distinct_fact_returns_none(self, store):
        """A genuinely new fact does not match anything."""
        store.remember(
            category="personal_fact",
            content="User's full name is Shaun Davies",
            dedup=False,
        )
        assert store.find_semantic_duplicate(
            "User prefers concise technical answers", min_similarity=0.88
        ) is None

    def test_threshold_respected(self, store):
        """Raising the threshold above the observed cosine returns None."""
        store.remember(
            category="personal_fact",
            content="User's full name is Shaun Davies",
            dedup=False,
        )
        # 1.0 = exact-match only; a rephrasing must not clear it.
        assert store.find_semantic_duplicate(
            "My name is Shaun", min_similarity=1.0
        ) is None

    def test_superseded_memory_not_matched(self, store):
        """Only the active head (valid_to IS NULL) is a dedupe target."""
        v1 = store.remember(
            category="personal_fact",
            content="User's full name is Shaun Davies",
            dedup=False,
        )
        v2 = store.update_memory(
            v1.memory_id, content="User prefers concise technical answers"
        )
        assert v2 is not None
        # v1 is superseded (valid_to set) — restating it must not match.
        assert store.find_semantic_duplicate(
            "My full name is Shaun Davies", min_similarity=0.88
        ) is None
        # The active head still matches.
        assert store.find_semantic_duplicate(
            "User prefers concise technical answers", min_similarity=0.88
        ) is not None

    def test_quarantined_memory_not_matched(self, store):
        """Quarantined memories are not dedupe targets."""
        rec = store.remember(
            category="personal_fact",
            content="User's full name is Shaun Davies",
            dedup=False,
        )
        assert store.quarantine_memory(rec.memory_id, "test")
        assert store.find_semantic_duplicate(
            "My name is Shaun", min_similarity=0.88
        ) is None

    def test_no_embedder_fails_soft(self, tmp_path):
        """No embedder → None (dedupe never blocks capture)."""
        from store import DuckDBMemoryStore
        s = DuckDBMemoryStore(tmp_path / "noemb.duckdb", user_id="test_user", embedder=None)
        try:
            assert s.find_semantic_duplicate("My name is Shaun") is None
        finally:
            s.close()

    def test_embedder_error_fails_soft(self, tmp_path):
        """An embedder that raises → None (dedupe never blocks capture)."""
        from store import DuckDBMemoryStore

        class _BrokenEmbedder:
            def embed(self, text, *, is_query=False):
                raise RuntimeError("embedder down")

        s = DuckDBMemoryStore(
            tmp_path / "broken.duckdb", user_id="test_user", embedder=_BrokenEmbedder()
        )
        try:
            assert s.find_semantic_duplicate("My name is Shaun") is None
        finally:
            s.close()

    def test_empty_content_returns_none(self, store):
        assert store.find_semantic_duplicate("   ") is None


# ---------------------------------------------------------------------------
# Provider layer: candidate-creation path (sync_turn)
# ---------------------------------------------------------------------------

def _stub_hermes_runtime() -> None:
    """Stub the Hermes runtime modules the provider imports at module load."""
    if "agent" not in sys.modules:
        sys.modules["agent"] = types.ModuleType("agent")
    if "agent.memory_provider" not in sys.modules:
        _mp = types.ModuleType("agent.memory_provider")

        class MemoryProvider:
            pass

        _mp.MemoryProvider = MemoryProvider
        sys.modules["agent.memory_provider"] = _mp
    if "tools" not in sys.modules:
        sys.modules["tools"] = types.ModuleType("tools")
    if "tools.registry" not in sys.modules:
        _tr = types.ModuleType("tools.registry")

        def _tool_error(msg):
            return json.dumps({"error": str(msg)})

        _tr.tool_error = _tool_error
        sys.modules["tools.registry"] = _tr


def _make_provider(store):
    """Build an ArgosProvider wired to a real store, extraction on, review off."""
    _stub_hermes_runtime()
    try:
        import argos_plugin
    except ModuleNotFoundError:
        import argos as argos_plugin
    provider = argos_plugin.ArgosProvider()
    provider._store = store
    provider._graph = None
    provider._agent_context = "primary"
    provider._auto_extract = True
    provider._auto_extract_paused = False
    provider._llm_fallback = False
    provider._llm_model = ""
    provider._llm_provider = ""
    provider._extraction_shadow_diff = False
    provider._auto_review = False
    provider._extraction_dup_threshold = 0.88
    return provider


def _run_turn(provider, user_content: str) -> None:
    provider.sync_turn(user_content, "ok", session_id="sess-dedupe")
    provider._sync_queue.join()


class TestProviderExtractionDedupe:
    def test_duplicate_fact_skipped_no_candidate(self, tmp_path):
        """A fact already active, restated differently → NO new candidate."""
        from store import DuckDBMemoryStore
        from embeddings import LocalEmbedder

        store = DuckDBMemoryStore(
            tmp_path / "dedupe.duckdb", user_id="test_user",
            embedder=LocalEmbedder("BAAI/bge-small-en-v1.5"),
        )
        store.remember(
            category="preference",
            content="User prefers concise technical explanations",
            dedup=False,
        )
        provider = _make_provider(store)
        try:
            _run_turn(provider, "I prefer concise technical answers")
            pending = store.list_candidates(status="pending")
            assert pending == [], f"Duplicate proposal was not skipped: {pending}"
        finally:
            provider.shutdown()

    def test_new_fact_emits_candidate(self, tmp_path):
        """A genuinely new fact still produces a pending candidate."""
        from store import DuckDBMemoryStore
        from embeddings import LocalEmbedder

        store = DuckDBMemoryStore(
            tmp_path / "newfact.duckdb", user_id="test_user",
            embedder=LocalEmbedder("BAAI/bge-small-en-v1.5"),
        )
        store.remember(
            category="personal_fact",
            content="User's full name is Shaun Davies",
            dedup=False,
        )
        provider = _make_provider(store)
        try:
            _run_turn(provider, "I prefer concise technical answers")
            pending = store.list_candidates(status="pending")
            assert len(pending) == 1, f"Expected 1 candidate, got {pending}"
            assert "concise technical answers" in pending[0]["content"]
        finally:
            provider.shutdown()

    def test_embedder_error_still_emits_candidate(self, tmp_path):
        """Embedder failure must never block memory capture."""
        from store import DuckDBMemoryStore

        class _BrokenEmbedder:
            def embed(self, text, *, is_query=False):
                raise RuntimeError("embedder down")

        store = DuckDBMemoryStore(
            tmp_path / "broken.duckdb", user_id="test_user",
            embedder=_BrokenEmbedder(),
        )
        provider = _make_provider(store)
        try:
            _run_turn(provider, "I prefer concise technical answers")
            pending = store.list_candidates(status="pending")
            assert len(pending) == 1, f"Expected 1 candidate, got {pending}"
        finally:
            provider.shutdown()


# ---------------------------------------------------------------------------
# Reviewer lane: bounded retry
# ---------------------------------------------------------------------------

class _Msg:
    def __init__(self, content: str):
        self.content = content


class _Choice:
    def __init__(self, content: str):
        self.message = _Msg(content)


class _Resp:
    def __init__(self, content: str):
        self.choices = [_Choice(content)]


def _stub_auxiliary_client(monkeypatch, fake_call_llm) -> None:
    """Point reviewer's `from agent.auxiliary_client import call_llm` at a fake."""
    agent_mod = sys.modules.get("agent") or types.ModuleType("agent")
    if "agent" not in sys.modules:
        sys.modules["agent"] = agent_mod
    aux_mod = types.ModuleType("agent.auxiliary_client")
    aux_mod.call_llm = fake_call_llm
    monkeypatch.setitem(sys.modules, "agent.auxiliary_client", aux_mod)


_CANDIDATE = {
    "category": "preference",
    "content": "User prefers concise technical answers",
    "payload": {},
    "evidence_text": "I prefer concise technical answers.",
}


class TestReviewerRetry:
    def test_reviewer_retries_once_then_succeeds(self, monkeypatch):
        """A transient Connection error is retried once and the decision lands."""
        from reviewer import review_candidate_with_llm

        calls = {"n": 0}

        def fake_call_llm(**kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise ConnectionError("Connection error")
            return _Resp(json.dumps({
                "decision": "approve",
                "confidence": 0.95,
                "reason": "clear preference",
                "durability": "durable",
                "scope": "profile",
            }))

        _stub_auxiliary_client(monkeypatch, fake_call_llm)
        result = review_candidate_with_llm(_CANDIDATE)
        assert calls["n"] == 2, "Expected exactly one retry"
        assert result["decision"] == "approve"

    def test_reviewer_fails_soft_after_retries(self, monkeypatch):
        """Persistent failure → conservative pending_user_confirmation."""
        from reviewer import review_candidate_with_llm

        calls = {"n": 0}

        def fake_call_llm(**kwargs):
            calls["n"] += 1
            raise ConnectionError("Connection error")

        _stub_auxiliary_client(monkeypatch, fake_call_llm)
        result = review_candidate_with_llm(_CANDIDATE)
        assert calls["n"] == 2, "Expected two attempts then fail-soft"
        assert result["decision"] == "pending_user_confirmation"
        assert "Connection error" in result["reason"]


# ---------------------------------------------------------------------------
# #89 — ReDoS guard: adversarial input must not hang or crash extraction
# ---------------------------------------------------------------------------

class TestReDoSGuard:
    """The extractor's regex stage must handle adversarial input without
    catastrophic backtracking (#89)."""

    def test_repeated_trigger_words_no_punctuation_completes(self):
        """The exact trigger from the issue: 'always ' * 40 + trigger phrase."""
        from extractor import extract_from_turn, _reset_extraction_failure_stats

        _reset_extraction_failure_stats()
        # 200 repetitions of "always " followed by a trigger — no punctuation
        # so the naive splitter treats it as one long "sentence".
        adversarial = "always " * 200 + "give me the short version"
        # Must complete quickly (bounded quantifiers + length caps) and
        # return a list (possibly with the directive fact, possibly empty).
        facts = extract_from_turn(adversarial, "", use_llm_fallback=False)
        assert isinstance(facts, list)

    def test_huge_unpunctuated_input_completes(self):
        """A 50k-char unpunctuated wall of text must not hang the regex stage."""
        from extractor import extract_from_turn

        huge = "i use " * 10000  # ~60k chars, no sentence boundaries
        facts = extract_from_turn(huge, "", use_llm_fallback=False)
        assert isinstance(facts, list)

    def test_top_level_guard_catches_exception(self, monkeypatch):
        """An exception inside the extraction stages is caught by the
        top-level guard and returns [] (#85/#89)."""
        import extractor

        def _boom(user_content):
            raise RuntimeError("simulated regex stage failure")

        monkeypatch.setattr(extractor, "_extract_facts_regex", _boom)
        facts = extractor.extract_from_turn("I prefer concise answers", "", use_llm_fallback=False)
        assert facts == []

    def test_extraction_failure_counter_increments(self, monkeypatch):
        """The failure counter is surfaced for review (#89/#85)."""
        import extractor

        extractor._reset_extraction_failure_stats()
        before = extractor.get_extraction_failure_stats()["extraction_failures"]

        def _boom(user_content):
            raise RuntimeError("simulated failure")

        monkeypatch.setattr(extractor, "_extract_facts_regex", _boom)
        extractor.extract_from_turn("I prefer concise answers", "", use_llm_fallback=False)
        after = extractor.get_extraction_failure_stats()["extraction_failures"]
        assert after == before + 1


# ---------------------------------------------------------------------------
# #85 — egress import/gate failure in _extract_facts_llm fails soft
# ---------------------------------------------------------------------------

class TestExtractorEgressGuard:
    """The LLM fallback's egress import/gate must fail soft (return []),
    never propagate (#85)."""

    def test_egress_import_failure_returns_empty(self, monkeypatch):
        """A broken egress module does not crash the LLM fallback."""
        import sys
        import types
        import extractor

        # Poison the egress module so `from egress import gate` raises.
        _poison = types.ModuleType("egress")

        class _GateBoom:
            def __getattr__(self, name):
                raise ImportError("poisoned")

        _poison.__spec__ = None  # force import to use this stub
        monkeypatch.setitem(sys.modules, "egress", _poison)

        # Make _should_try_llm_fallback return True so the LLM path is reached
        monkeypatch.setattr(extractor, "_should_try_llm_fallback", lambda c, n: True)
        # The call must not raise — it returns [].
        facts = extractor.extract_from_turn(
            "I prefer concise technical answers and always want plain English explanations.",
            "", use_llm_fallback=True,
        )
        assert isinstance(facts, list)

    def test_egress_gate_raise_returns_empty(self, monkeypatch):
        """A raising egress gate (e.g. malformed config) fails soft (#85)."""
        import sys
        import types
        import extractor

        _fake = types.ModuleType("egress")

        def _raising_gate(kind, text="", cfg=None):
            raise RuntimeError("malformed config")

        _fake.gate = _raising_gate
        monkeypatch.setitem(sys.modules, "egress", _fake)

        monkeypatch.setattr(extractor, "_should_try_llm_fallback", lambda c, n: True)
        extractor._reset_extraction_failure_stats()
        # Direct call to _extract_facts_llm must not raise. Content must be
        # long enough to pass the _LLM_MIN_CONTENT_LENGTH gate (60 chars).
        result = extractor._extract_facts_llm(
            "I prefer concise technical answers and always want plain English explanations first."
        )
        assert result == []
        # The failure counter should have incremented.
        assert extractor.get_extraction_failure_stats()["extraction_failures"] >= 1


# ---------------------------------------------------------------------------
# #15 — sentence splitter unit tests (acceptance criterion)
# ---------------------------------------------------------------------------

class TestSentenceSplitter:
    """Unit tests for _split_sentences (#15 acceptance criterion).

    The splitter is intentionally simple (regex on [.!?] + whitespace) but
    must be covered by tests so future changes are gated. The ReDoS cap
    (#89: _MAX_SENTENCE_CHARS) is also verified here.
    """

    def test_basic_split_on_period(self):
        from extractor import _split_sentences
        result = _split_sentences("I like tea. I hate coffee.")
        assert result == ["I like tea.", "I hate coffee."]

    def test_split_on_exclamation_and_question(self):
        from extractor import _split_sentences
        result = _split_sentences("Are you sure? Yes! Okay.")
        assert result == ["Are you sure?", "Yes!", "Okay."]

    def test_no_punctuation_is_one_sentence(self):
        from extractor import _split_sentences
        result = _split_sentences("just a plain message with no ending")
        assert result == ["just a plain message with no ending"]

    def test_empty_string_returns_empty_list(self):
        from extractor import _split_sentences
        assert _split_sentences("") == []

    def test_whitespace_only_returns_empty_list(self):
        from extractor import _split_sentences
        assert _split_sentences("   \t\n  ") == []

    def test_newlines_collapsed_to_spaces(self):
        from extractor import _split_sentences
        result = _split_sentences("I like tea.\nI hate coffee.")
        assert result == ["I like tea.", "I hate coffee."]

    def test_multiple_spaces_collapsed(self):
        from extractor import _split_sentences
        result = _split_sentences("I like   tea.  I hate coffee.")
        assert result == ["I like tea.", "I hate coffee."]

    def test_long_unpunctuated_sentence_is_capped(self):
        """A 10k-char unpunctuated run is capped at _MAX_SENTENCE_CHARS (#89)."""
        from extractor import _split_sentences, _MAX_SENTENCE_CHARS
        long_text = "a" * 10_000
        result = _split_sentences(long_text)
        assert len(result) == 1
        assert len(result[0]) == _MAX_SENTENCE_CHARS

    def test_trailing_whitespace_stripped(self):
        from extractor import _split_sentences
        result = _split_sentences("I like tea.   I hate coffee.   ")
        assert result == ["I like tea.", "I hate coffee."]

    def test_abbreviation_not_split(self):
        """The simple splitter does NOT handle abbreviations (Mr., Dr.) —
        this test documents that behavior so a future improvement is a
        deliberate change, not a silent regression."""
        from extractor import _split_sentences
        result = _split_sentences("I met Dr. Smith. He is nice.")
        # The splitter splits after "Dr." — documented behavior.
        assert "I met Dr." in result


# ---------------------------------------------------------------------------
# #99.1 — action/intent gate: memory-control commands are not durable facts
# ---------------------------------------------------------------------------

class TestMemoryControlGate:
    """Imperative memory-management commands must not be extracted as facts (#99).

    "bin it", "discard that proposal", "forget this" are actions against the
    memory system, not durable preferences. The extractor must return [] for
    them so a discard command never becomes "User wants to bin (discard) memory."
    """

    def test_bin_it_is_not_extracted(self):
        from extractor import extract_from_turn, is_memory_control_command
        assert is_memory_control_command("bin it")
        facts = extract_from_turn("bin it", "", use_llm_fallback=False)
        assert facts == []

    def test_discard_that_proposal_is_not_extracted(self):
        from extractor import is_memory_control_command
        assert is_memory_control_command("discard that proposal")

    def test_forget_this_is_not_extracted(self):
        from extractor import is_memory_control_command
        assert is_memory_control_command("forget this")

    def test_delete_that_memory_is_not_extracted(self):
        from extractor import is_memory_control_command
        assert is_memory_control_command("delete that memory")

    def test_reject_that_candidate_is_not_extracted(self):
        from extractor import is_memory_control_command
        assert is_memory_control_command("reject that candidate")

    def test_drop_it_is_not_extracted(self):
        from extractor import is_memory_control_command
        assert is_memory_control_command("drop it")

    def test_legitimate_directive_is_not_blocked(self):
        """'always explain in plain English' is a legitimate assistant-side
        directive, NOT a memory-control command — it must be extractable."""
        from extractor import is_memory_control_command
        assert not is_memory_control_command("always explain in plain English")

    def test_legitimate_preference_is_not_blocked(self):
        """'I prefer concise answers' is a durable preference, not a command."""
        from extractor import is_memory_control_command
        assert not is_memory_control_command("I prefer concise answers")

    def test_habit_with_forget_is_not_blocked(self):
        """'I always forget my keys' is a durable habit, not a memory-control
        command — the 'I' subject makes it a statement, not an imperative."""
        from extractor import is_memory_control_command
        assert not is_memory_control_command("I always forget my keys")

    def test_long_message_is_not_blocked(self):
        """A long message is unlikely to be a pure memory-control command."""
        from extractor import is_memory_control_command
        long_msg = "bin it " + "and also " * 30
        assert not is_memory_control_command(long_msg)

    def test_bin_the_proposal_about_x_is_detected(self):
        """'bin the proposal about the weather' is a memory-control command
        targeting a specific proposal."""
        from extractor import is_memory_control_command
        assert is_memory_control_command("bin the proposal about the weather")

    def test_forget_what_i_just_said_is_detected(self):
        from extractor import is_memory_control_command
        assert is_memory_control_command("forget what I just said")

    def test_empty_string_is_not_a_command(self):
        from extractor import is_memory_control_command
        assert not is_memory_control_command("")

    def test_normal_sentence_not_a_command(self):
        from extractor import is_memory_control_command
        assert not is_memory_control_command("I like tea.")

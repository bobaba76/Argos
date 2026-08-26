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

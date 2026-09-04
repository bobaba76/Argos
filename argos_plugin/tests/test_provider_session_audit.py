"""Audit tests for provider_session.py (PS1-PS10, issue #229).

Covers:
- PS1: atomic config write + lock for _persist_learned_role_word
- PS2: consolidation_auto_apply gate for session-end consolidation
- PS3: LLM prompt word escaping
- PS4: get_sync_stats surfaces dropped turn count
- PS5: shutdown drains sync queue before sentinel
- PS6: per-turn tool call rate limit
- PS7: LLM classification cap per memory
- PS8: memory_search payload field filtering
- PS9: sync worker catches BaseException
- PS10: memory_id UUID validation

Run with (Hermes venv python, offline):
    python -m pytest tests/test_provider_session_audit.py -v
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

_plugin_dir = Path(__file__).resolve().parent.parent
for _path in (_plugin_dir.parent, _plugin_dir):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


# ---------------------------------------------------------------------------
# PS1 — atomic config write + lock
# ---------------------------------------------------------------------------

class TestPS1AtomicConfigWrite:
    def test_lock_exists(self):
        from provider_session import ProviderSessionMixin
        assert hasattr(ProviderSessionMixin, "_role_word_persist_lock")

    def test_persist_uses_tempfile_and_replace(self):
        """PS1: the _persist_learned_role_word source uses os.replace
        (atomic) rather than write_text (non-atomic). We check the source
        contains the atomic-write pattern."""
        import inspect
        from provider_session import ProviderSessionMixin
        src = inspect.getsource(ProviderSessionMixin._persist_learned_role_word)
        assert "os.replace" in src, "PS1: should use os.replace for atomic write"
        assert "tempfile" in src, "PS1: should use tempfile for atomic write"


# ---------------------------------------------------------------------------
# PS2 — consolidation_auto_apply gate
# ---------------------------------------------------------------------------

class TestPS2ConsolidationGate:
    def test_auto_apply_check_in_on_session_end(self):
        """PS2: on_session_end checks consolidation_auto_apply config."""
        import inspect
        from provider_session import ProviderSessionMixin
        src = inspect.getsource(ProviderSessionMixin.on_session_end)
        assert "consolidation_auto_apply" in src, (
            "PS2: on_session_end should check consolidation_auto_apply"
        )


# ---------------------------------------------------------------------------
# PS3 — LLM prompt word escaping
# ---------------------------------------------------------------------------

class TestPS3PromptEscaping:
    def test_llm_classify_uses_json_dumps(self):
        """PS3: _llm_classify_role_word escapes the word via json.dumps."""
        import inspect
        from provider_session import ProviderSessionMixin
        src = inspect.getsource(ProviderSessionMixin._llm_classify_role_word)
        assert "json.dumps(word)" in src or "json.dumps" in src, (
            "PS3: should escape word via json.dumps"
        )


# ---------------------------------------------------------------------------
# PS4 — get_sync_stats
# ---------------------------------------------------------------------------

class TestPS4SyncStats:
    def test_get_sync_stats_exists(self):
        from provider_session import ProviderSessionMixin
        assert hasattr(ProviderSessionMixin, "get_sync_stats")

    def test_get_sync_stats_returns_dropped_turns(self):
        """PS4: get_sync_stats returns a dict with dropped_turns key."""
        from provider_session import ProviderSessionMixin

        class FakeProvider:
            _sync_dropped_turns = 3
            get_sync_stats = ProviderSessionMixin.get_sync_stats

        p = FakeProvider()
        stats = p.get_sync_stats()
        assert "dropped_turns" in stats
        assert stats["dropped_turns"] == 3


# ---------------------------------------------------------------------------
# PS5 — shutdown drains queue
# ---------------------------------------------------------------------------

class TestPS5ShutdownDrains:
    def test_shutdown_calls_queue_join(self):
        """PS5: shutdown drains the sync queue before sending sentinel."""
        import inspect
        from provider_session import ProviderSessionMixin
        src = inspect.getsource(ProviderSessionMixin.shutdown)
        assert "join" in src, "PS5: shutdown should drain queue via join()"


# ---------------------------------------------------------------------------
# PS6 — per-turn tool call rate limit
# ---------------------------------------------------------------------------

class TestPS6ToolCallRateLimit:
    def test_rate_limit_constants_exist(self):
        from provider_session import ProviderSessionMixin
        assert hasattr(ProviderSessionMixin, "_TOOL_CALL_MAX_PER_TURN")
        assert ProviderSessionMixin._TOOL_CALL_MAX_PER_TURN > 0

    def test_rate_limit_enforced(self):
        """PS6: handle_tool_call rejects calls above the per-turn limit."""
        from provider_session import ProviderSessionMixin

        class FakeStore:
            pass

        class FakeProvider:
            _store = FakeStore()
            _tool_call_count = 0
            _TOOL_CALL_MAX_PER_TURN = 3
            _max_injected = 5
            _search_memories = lambda self, *a, **kw: []
            handle_tool_call = ProviderSessionMixin.handle_tool_call

        p = FakeProvider()
        # First 3 calls should not hit the rate limit error.
        for i in range(3):
            p._tool_call_count = i
            result = p.handle_tool_call("memory_search", {"query": "test"})
            assert "rate limit" not in result.lower(), (
                f"PS6: call {i+1} should not be rate-limited"
            )
        # 4th call should be rate-limited.
        p._tool_call_count = 4
        result = p.handle_tool_call("memory_search", {"query": "test"})
        assert "rate limit" in result.lower(), "PS6: 4th call should be rate-limited"


# ---------------------------------------------------------------------------
# PS7 — LLM classification cap
# ---------------------------------------------------------------------------

class TestPS7LLMClassificationCap:
    def test_cap_in_extract_role_aliases(self):
        """PS7: _extract_role_aliases has a cap on LLM classifications."""
        import inspect
        from provider_session import ProviderSessionMixin
        src = inspect.getsource(ProviderSessionMixin._extract_role_aliases)
        assert "MAX_LLM_CLASSIFICATIONS" in src or "max_llm_classifications" in src.lower(), (
            "PS7: should have a cap on LLM classifications per memory"
        )


# ---------------------------------------------------------------------------
# PS8 — memory_search payload filtering
# ---------------------------------------------------------------------------

class TestPS8PayloadFiltering:
    def test_payload_fields_filtered(self):
        """PS8: handle_tool_call memory_search filters payload fields."""
        import inspect
        from provider_session import ProviderSessionMixin
        src = inspect.getsource(ProviderSessionMixin.handle_tool_call)
        assert "_LLM_PAYLOAD_FIELDS" in src, (
            "PS8: should filter payload to LLM-needed fields only"
        )


# ---------------------------------------------------------------------------
# PS9 — sync worker catches BaseException
# ---------------------------------------------------------------------------

class TestPS9BaseException:
    def test_baseexception_caught_in_worker(self):
        """PS9: _sync_worker_loop catches BaseException (not just Exception)."""
        import inspect
        from provider_session import ProviderSessionMixin
        src = inspect.getsource(ProviderSessionMixin._sync_worker_loop)
        assert "BaseException" in src, (
            "PS9: sync worker should catch BaseException"
        )


# ---------------------------------------------------------------------------
# PS10 — memory_id UUID validation
# ---------------------------------------------------------------------------

class TestPS10MemoryIdValidation:
    def test_valid_memory_id_helper_exists(self):
        from provider_session import _valid_memory_id

    def test_valid_uuid_accepted(self):
        from provider_session import _valid_memory_id
        assert _valid_memory_id("12345678-1234-1234-1234-123456789abc")

    def test_invalid_string_rejected(self):
        from provider_session import _valid_memory_id
        assert not _valid_memory_id("not-a-uuid")
        assert not _valid_memory_id("")
        assert not _valid_memory_id("'; DROP TABLE memories; --")
        assert not _valid_memory_id("12345")  # too short

    def test_validation_in_handle_tool_call(self):
        """PS10: handle_tool_call validates memory_id for memory_update."""
        from provider_session import ProviderSessionMixin

        class FakeStore:
            pass

        class FakeProvider:
            _store = FakeStore()
            _tool_call_count = 0
            _TOOL_CALL_MAX_PER_TURN = 50
            _max_injected = 5
            handle_tool_call = ProviderSessionMixin.handle_tool_call

        p = FakeProvider()
        result = p.handle_tool_call("memory_update", {"memory_id": "not-a-uuid"})
        assert "Invalid memory_id" in result or "UUID" in result, (
            "PS10: invalid memory_id should be rejected"
        )

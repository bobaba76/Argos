"""Tests for provider_session.py audit fixes."""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

_plugin_dir = Path(__file__).resolve().parent.parent
if str(_plugin_dir) not in sys.path:
    sys.path.insert(0, str(_plugin_dir))


def _stub_hermes_runtime():
    """Stub the Hermes runtime so we can import the provider."""
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


# ---------------------------------------------------------------------------
# P1 — expiry auto-suggest is dead code (checks "approved" not "reviewed_approved")
# ---------------------------------------------------------------------------


def test_expiry_auto_suggest_fires_on_reviewed_approved(tmp_path):
    """P1: The expiry auto-suggest must fire when final_status is
    'reviewed_approved' (the actual status produced by the decision_map),
    not 'approved' (which the decision_map never produces)."""
    _stub_hermes_runtime()

    from store import DuckDBMemoryStore
    try:
        import argos_plugin
    except ModuleNotFoundError:
        import argos as argos_plugin

    store = DuckDBMemoryStore(tmp_path / "test.duckdb", user_id="test_user")

    # Save a candidate that will be auto-reviewed.
    # Use source="llm_extraction" so grounding=extracted (ceiling=approved),
    # which allows reviewed_approved to survive the grounding ceiling check.
    candidate = store.save_candidate(
        category="context_note",
        content="User is working on the Q3 report for 2 weeks",
        source="llm_extraction",
        confidence=0.9,
        scope="profile",
        evidence_text="I am working on the Q3 report for 2 weeks.",
    )
    assert candidate is not None

    # Construct a provider with auto-review + expiry auto-suggest enabled.
    provider = argos_plugin.ArgosProvider()
    provider._store = store
    provider._graph = None
    provider._auto_review = True
    provider._expiry_auto_suggest = True
    provider._expiry_enabled = True
    provider._expiry_ttl_days = {"context_note": 30}
    provider._expiry_default_days = 90
    provider._llm_model = ""
    provider._llm_provider = ""

    # Mock the LLM reviewer to return "approve" — which the decision_map
    # converts to "reviewed_approved".
    mock_review = {
        "decision": "approve",
        "confidence": 0.9,
        "reason": "clear temporary fact",
        "durability": "temporary",
        "scope": "session",
        "review_model": "memory_review",
    }

    # Patch review_candidate_with_llm to return our mock, and patch
    # suggest_expiry so we can capture whether it was called. Must patch
    # the argos_plugin.provider_session module (not the top-level
    # provider_session module) because ArgosProvider inherits from
    # argos_plugin.provider_session.ProviderSessionMixin.
    import argos_plugin.provider_session as ps_mod
    with patch.object(ps_mod, "review_candidate_with_llm",
                      return_value=mock_review):
        with patch.object(ps_mod, "suggest_expiry",
                          return_value="2026-12-31T23:59:59+00:00") as mock_suggest:
            provider._review_candidate(candidate)

    # suggest_expiry MUST have been called — the auto-suggest should fire
    # on reviewed_approved (the actual status from the decision_map).
    assert mock_suggest.called, (
        "suggest_expiry was not called — the expiry auto-suggest is dead "
        "code because it checks final_status == 'approved' but the "
        "decision_map produces 'reviewed_approved'"
    )

    store.close()


def test_expiry_auto_suggest_does_not_fire_on_reject(tmp_path):
    """Sanity: the expiry auto-suggest must NOT fire on reject."""
    _stub_hermes_runtime()

    from store import DuckDBMemoryStore
    try:
        import argos_plugin
    except ModuleNotFoundError:
        import argos as argos_plugin

    store = DuckDBMemoryStore(tmp_path / "test.duckdb", user_id="test_user")

    candidate = store.save_candidate(
        category="context_note",
        content="User is working on the Q3 report for 2 weeks",
        source="llm_extraction",
        confidence=0.9,
        scope="profile",
        evidence_text="I am working on the Q3 report for 2 weeks.",
    )
    assert candidate is not None

    provider = argos_plugin.ArgosProvider()
    provider._store = store
    provider._graph = None
    provider._auto_review = True
    provider._expiry_auto_suggest = True
    provider._expiry_enabled = True
    provider._expiry_ttl_days = {"context_note": 30}
    provider._expiry_default_days = 90
    provider._llm_model = ""
    provider._llm_provider = ""

    mock_review = {
        "decision": "reject",
        "confidence": 0.9,
        "reason": "not a durable fact",
        "durability": "temporary",
        "scope": "session",
        "review_model": "memory_review",
    }

    import argos_plugin.provider_session as ps_mod
    with patch.object(ps_mod, "review_candidate_with_llm",
                      return_value=mock_review):
        with patch.object(ps_mod, "suggest_expiry",
                          return_value="2026-12-31T23:59:59+00:00") as mock_suggest:
            provider._review_candidate(candidate)

    assert not mock_suggest.called, (
        "suggest_expiry should not fire on reject"
    )

    store.close()

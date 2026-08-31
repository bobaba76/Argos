"""Tests for deterministic reviewer policy gates."""
from __future__ import annotations

import json


class _Msg:
    def __init__(self, content: str):
        self.content = content


class _Choice:
    def __init__(self, content: str):
        self.message = _Msg(content)


class _Resp:
    def __init__(self, content: str):
        self.choices = [_Choice(content)]


def test_obvious_garbage_is_quarantined_without_llm():
    from reviewer import review_candidate_with_llm

    result = review_candidate_with_llm({
        "category": "goal",
        "content": "User goal: stop you",
        "payload": {},
        "evidence_text": "I was asking the assistant to stop.",
    })
    assert result["decision"] == "quarantined"
    assert result["review_model"] == "deterministic_gate"


def test_sensitive_candidate_requires_confirmation_when_reviewer_unavailable(monkeypatch):
    """A sensitive candidate routes to pending_user_confirmation when the
    LLM reviewer is unavailable — it must NOT be auto-approved."""
    import sys
    import types

    from reviewer import review_candidate_with_llm

    # Allow the egress gate to pass so we reach the LLM-client import.
    _fake_egress = types.ModuleType("egress")
    _fake_egress.gate = lambda kind, text="", cfg=None: True
    monkeypatch.setitem(sys.modules, "egress", _fake_egress)

    # Force the "no LLM client" state: poison agent.auxiliary_client so
    # `from agent.auxiliary_client import call_llm` raises ImportError
    # (the attribute does not exist on the stub module). This triggers
    # the reviewer's import-guard → reviewer_unavailable path.
    _poison_aux = types.ModuleType("agent.auxiliary_client")
    # Deliberately do NOT set call_llm — the missing attribute makes
    # `from ... import call_llm` raise ImportError.
    _agent_mod = sys.modules.get("agent") or types.ModuleType("agent")
    if "agent" not in sys.modules:
        sys.modules["agent"] = _agent_mod
    monkeypatch.setitem(sys.modules, "agent.auxiliary_client", _poison_aux)

    result = review_candidate_with_llm({
        "category": "personal_fact",
        "content": "User has a medical condition",
        "payload": {},
        "evidence_text": "I have a medical condition.",
    })
    assert result["decision"] == "pending_user_confirmation"
    assert result["review_model"] == "reviewer_unavailable"


def test_missing_evidence_never_auto_approves():
    from reviewer import review_candidate_with_llm

    result = review_candidate_with_llm({
        "category": "preference",
        "content": "User prefers direct answers",
        "payload": {},
        "evidence_text": "",
    })
    assert result["decision"] == "pending_user_confirmation"
    assert "evidence" in result["reason"].lower()


def test_confirmation_block_contains_data_and_safe_instructions():
    from confirmation import build_confirmation_block

    block = build_confirmation_block([{
        "candidate_id": "cand-123",
        "category": "personal_fact",
        "content": "User has a medical condition",
        "review_reason": "Sensitive proposal requires confirmation.",
    }])
    assert "cand-123" in block
    assert "User has a medical condition" in block
    assert "untrusted memory data" in block
    assert "Do not call memory_candidate_review" in block


# ---------------------------------------------------------------------------
# #86 — egress gate import/call failure fails closed to pending_user_confirmation
# ---------------------------------------------------------------------------

def test_reviewer_egress_import_failure_fails_closed(monkeypatch):
    """An egress import failure routes to pending_user_confirmation, not a crash (#86)."""
    import sys
    import types

    from reviewer import review_candidate_with_llm

    # Force the egress import to fail by poisoning sys.modules with a module
    # that raises on attribute access. The reviewer must catch this and
    # return pending_user_confirmation.
    _poison = types.ModuleType("egress")

    def _boom(*args, **kwargs):
        raise ImportError("egress poisoned for test")

    _poison.gate = _boom
    monkeypatch.setitem(sys.modules, "egress", _poison)

    result = review_candidate_with_llm({
        "category": "preference",
        "content": "User prefers concise answers",
        "payload": {},
        "evidence_text": "I prefer concise answers.",
    })
    assert result["decision"] == "pending_user_confirmation"
    assert result["review_model"] == "egress_gate_unavailable"


def test_reviewer_egress_gate_raise_fails_closed(monkeypatch):
    """A raising egress gate (e.g. malformed config) fails closed (#86)."""
    import sys
    import types

    from reviewer import review_candidate_with_llm

    _fake = types.ModuleType("egress")

    def _raising_gate(kind, text="", cfg=None):
        raise RuntimeError("malformed hybrid_memory.json")

    _fake.gate = _raising_gate
    monkeypatch.setitem(sys.modules, "egress", _fake)

    result = review_candidate_with_llm({
        "category": "preference",
        "content": "User prefers concise answers",
        "payload": {},
        "evidence_text": "I prefer concise answers.",
    })
    assert result["decision"] == "pending_user_confirmation"
    assert result["review_model"] == "egress_gate_unavailable"


# ---------------------------------------------------------------------------
# #99 — confirmation block: reviewer-failure outcomes are not "should this be saved?"
# ---------------------------------------------------------------------------

def test_confirmation_block_skips_reviewer_failure_candidates():
    """A candidate whose review_model is a reviewer failure (e.g.
    reviewer_unavailable) is NOT rendered as a confirmation prompt (#99)."""
    from confirmation import build_confirmation_block

    block = build_confirmation_block([{
        "candidate_id": "cand-fail",
        "category": "preference",
        "content": "User prefers concise answers",
        "review_reason": "The Hermes auxiliary LLM reviewer is unavailable.",
        "review_model": "reviewer_unavailable",
    }])
    assert block == ""


def test_confirmation_block_renders_genuine_confirmation():
    """A genuine sensitive confirmation (from memory_review) IS rendered (#99)."""
    from confirmation import build_confirmation_block

    block = build_confirmation_block([{
        "candidate_id": "cand-genuine",
        "category": "personal_fact",
        "content": "User has a medical condition",
        "review_reason": "Sensitive proposal requires user confirmation.",
        "review_model": "memory_review",
    }])
    assert "cand-genuine" in block
    assert "User has a medical condition" in block


def test_confirmation_block_skips_evidence_gate_candidates():
    """An evidence_gate candidate (no evidence) is not a confirmation prompt (#99)."""
    from confirmation import build_confirmation_block

    block = build_confirmation_block([{
        "candidate_id": "cand-noev",
        "category": "preference",
        "content": "User prefers direct answers",
        "review_reason": "No original user evidence is available.",
        "review_model": "evidence_gate",
    }])
    assert block == ""


# ---------------------------------------------------------------------------
# #99 — low-confidence reject/quarantine must not become user confirmation
# ---------------------------------------------------------------------------

def test_low_confidence_reject_stays_reject(monkeypatch):
    """A low-confidence reject is the safe default — it must NOT be converted
    to pending_user_confirmation (#99)."""
    import sys
    import types

    from reviewer import review_candidate_with_llm

    _fake_egress = types.ModuleType("egress")
    _fake_egress.gate = lambda kind, text="", cfg=None: True
    monkeypatch.setitem(sys.modules, "egress", _fake_egress)

    _aux = types.ModuleType("agent.auxiliary_client")
    _aux.call_llm = lambda **kw: _Resp(json.dumps({
        "decision": "reject",
        "confidence": 0.5,  # below 0.85 threshold
        "reason": "not a durable fact",
        "durability": "temporary",
        "scope": "session",
    }))
    agent_mod = sys.modules.get("agent") or types.ModuleType("agent")
    if "agent" not in sys.modules:
        sys.modules["agent"] = agent_mod
    monkeypatch.setitem(sys.modules, "agent.auxiliary_client", _aux)

    result = review_candidate_with_llm({
        "category": "context_note",
        "content": "User is tired",
        "payload": {},
        "evidence_text": "I am tired.",
    })
    assert result["decision"] == "reject", (
        f"Low-confidence reject must stay reject, got {result['decision']}"
    )


def test_low_confidence_quarantine_stays_quarantine(monkeypatch):
    """A low-confidence quarantine is the safe default — it must NOT be
    converted to pending_user_confirmation (#99)."""
    import sys
    import types

    from reviewer import review_candidate_with_llm

    _fake_egress = types.ModuleType("egress")
    _fake_egress.gate = lambda kind, text="", cfg=None: True
    monkeypatch.setitem(sys.modules, "egress", _fake_egress)

    _aux = types.ModuleType("agent.auxiliary_client")
    _aux.call_llm = lambda **kw: _Resp(json.dumps({
        "decision": "quarantine",
        "confidence": 0.5,  # below 0.85 threshold
        "reason": "malformed text",
        "durability": "temporary",
        "scope": "session",
    }))
    agent_mod = sys.modules.get("agent") or types.ModuleType("agent")
    if "agent" not in sys.modules:
        sys.modules["agent"] = agent_mod
    monkeypatch.setitem(sys.modules, "agent.auxiliary_client", _aux)

    result = review_candidate_with_llm({
        "category": "context_note",
        "content": "User is tired",
        "payload": {},
        "evidence_text": "I am tired.",
    })
    assert result["decision"] == "quarantine", (
        f"Low-confidence quarantine must stay quarantine, got {result['decision']}"
    )


def test_low_confidence_approve_becomes_confirmation(monkeypatch):
    """A low-confidence approve IS downgraded to pending_user_confirmation —
    this is the only decision where the threshold applies (#99)."""
    import sys
    import types

    from reviewer import review_candidate_with_llm

    _fake_egress = types.ModuleType("egress")
    _fake_egress.gate = lambda kind, text="", cfg=None: True
    monkeypatch.setitem(sys.modules, "egress", _fake_egress)

    _aux = types.ModuleType("agent.auxiliary_client")
    _aux.call_llm = lambda **kw: _Resp(json.dumps({
        "decision": "approve",
        "confidence": 0.5,  # below 0.85 threshold
        "reason": "seems like a fact",
        "durability": "durable",
        "scope": "profile",
    }))
    agent_mod = sys.modules.get("agent") or types.ModuleType("agent")
    if "agent" not in sys.modules:
        sys.modules["agent"] = agent_mod
    monkeypatch.setitem(sys.modules, "agent.auxiliary_client", _aux)

    result = review_candidate_with_llm({
        "category": "personal_fact",
        "content": "User likes tea",
        "payload": {},
        "evidence_text": "I like tea.",
    })
    assert result["decision"] == "pending_user_confirmation"

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


# ---------------------------------------------------------------------------
# #184 — _SENSITIVE_RE over-matches common words (internal, partner, job, name)
# ---------------------------------------------------------------------------

def test_sensitive_re_does_not_match_benign_internal():
    """#184: bare 'internal' in non-sensitive contexts should not flag."""
    from reviewer import is_sensitive_candidate
    assert not is_sensitive_candidate({
        "category": "personal_fact",
        "content": "I read about internal medicine for my studies",
    })
    assert not is_sensitive_candidate({
        "category": "personal_fact",
        "content": "The internal combustion engine is fascinating",
    })


def test_sensitive_re_matches_internal_memo():
    """#184: 'internal memo' / 'internal document' should still flag."""
    from reviewer import is_sensitive_candidate
    assert is_sensitive_candidate({
        "category": "personal_fact",
        "content": "I saw an internal memo about layoffs",
    })


def test_sensitive_re_does_not_match_benign_partner():
    """#184: 'project partner' / 'business partner' (non-romantic) should
    not flag via bare 'partner'. 'business partner' is explicitly matched
    but 'project partner' should not be."""
    from reviewer import is_sensitive_candidate
    assert not is_sensitive_candidate({
        "category": "personal_fact",
        "content": "My project partner reviewed the code",
    })


def test_sensitive_re_matches_my_partner():
    """#184: 'my partner' (romantic sense) should still flag."""
    from reviewer import is_sensitive_candidate
    assert is_sensitive_candidate({
        "category": "personal_fact",
        "content": "My partner is visiting this weekend",
    })


def test_sensitive_re_does_not_match_benign_job():
    """#184: 'job scheduling' / 'job queue' should not flag."""
    from reviewer import is_sensitive_candidate
    assert not is_sensitive_candidate({
        "category": "personal_fact",
        "content": "The job queue is full and job scheduling is slow",
    })


def test_sensitive_re_matches_job_title():
    """#184: 'job title' / 'job loss' should still flag."""
    from reviewer import is_sensitive_candidate
    assert is_sensitive_candidate({
        "category": "personal_fact",
        "content": "My job title is Senior Engineer",
    })


def test_sensitive_re_does_not_match_benign_name():
    """#184: 'name the file' should not flag as sensitive."""
    from reviewer import is_sensitive_candidate
    assert not is_sensitive_candidate({
        "category": "personal_fact",
        "content": "I need to name the file output.txt",
    })


def test_sensitive_re_matches_possessive_name():
    """#184: 'her name is' / 'my name is' should still flag."""
    from reviewer import is_sensitive_candidate
    assert is_sensitive_candidate({
        "category": "personal_fact",
        "content": "Her name is Alice and she lives nearby",
    })

# Reviewer audit R2 — suggest_expiry must not return a past fixed date
# ---------------------------------------------------------------------------

def test_suggest_expiry_rejects_past_fixed_date():
    """An explicit fixed date in the past must NOT be returned as the
    suggested expiry — a past expires_at would make the memory invisible
    immediately. Fall through to the category TTL fallback instead."""
    from datetime import datetime, timezone
    from reviewer import suggest_expiry

    # "until 15 Dec 2020" is in the past relative to any test run.
    result = suggest_expiry({
        "category": "context_note",
        "content": "Reminder: report due until 15 Dec 2020",
    })
    assert result is not None, "Should fall through to TTL fallback, not return None"
    # The fallback must be a future date, not 2020-12-15.
    parsed = datetime.fromisoformat(result)
    assert parsed > datetime.now(timezone.utc), (
        f"suggest_expiry returned a past date {result}; should fall through to TTL"
    )


def test_suggest_expiry_future_fixed_date_used_directly():
    """A future fixed date is used as-is (sanity check — R2 only rejects past)."""
    from datetime import datetime, timezone
    from reviewer import suggest_expiry

    result = suggest_expiry({
        "category": "context_note",
        "content": "Project deadline until 15 Dec 2099",
    })
    assert result is not None
    parsed = datetime.fromisoformat(result)
    assert parsed.year == 2099


# ---------------------------------------------------------------------------
# Reviewer audit R4 — _parse_json_response extracts JSON embedded in prose
# ---------------------------------------------------------------------------

def test_parse_json_response_extracts_json_after_prose():
    """If the LLM wraps its JSON in conversational prose
    ("Here is my review:\\n```json {...} ```\\nLet me know."), the parser
    must still extract the JSON object, not return None."""
    from reviewer import _parse_json_response

    body = 'Here is my review:\n```json\n{"decision": "approve", "confidence": 0.9}\n```\nLet me know.'
    result = _parse_json_response(_Resp(body))
    assert result is not None
    assert result["decision"] == "approve"


def test_parse_json_response_extracts_bare_json_in_prose():
    """JSON embedded in prose without any code fence must also be extracted."""
    from reviewer import _parse_json_response

    body = 'Sure! {"decision": "reject", "confidence": 0.8, "reason": "no"} Hope that helps.'
    result = _parse_json_response(_Resp(body))
    assert result is not None
    assert result["decision"] == "reject"


# ---------------------------------------------------------------------------
# Reviewer audit R5 — scan_inbound_text raising must fail closed, not crash
# ---------------------------------------------------------------------------

def test_reviewer_scan_inbound_text_raise_fails_closed(monkeypatch):
    """If scan_inbound_text raises (e.g. corrupted scanner state), the
    reviewer must route the external candidate to pending_user_confirmation,
    not propagate the exception."""
    import sys
    import types

    from reviewer import review_candidate_with_llm

    # Egress gate must pass so we reach the external-source gate.
    _fake_egress = types.ModuleType("egress")
    _fake_egress.gate = lambda kind, text="", cfg=None: True
    monkeypatch.setitem(sys.modules, "egress", _fake_egress)

    # Poison inbound_security so scan_inbound_text raises when called.
    _fake_inbound = types.ModuleType("inbound_security")

    def _boom(*args, **kwargs):
        raise RuntimeError("scanner corrupted")

    _fake_inbound.scan_inbound_text = _boom
    # The reviewer imports via `from .inbound_security import ...` when
    # __package__ is set (it is, under pytest). We need to inject the
    # module under the name the reviewer uses.
    import argos_plugin.reviewer as _rev_mod
    # Ensure the reviewer's __package__ path resolves to our fake.
    monkeypatch.setitem(sys.modules, "argos_plugin.inbound_security", _fake_inbound)
    monkeypatch.setitem(sys.modules, "inbound_security", _fake_inbound)

    result = review_candidate_with_llm({
        "category": "context_note",
        "content": "User mentioned a deadline",
        "payload": {"external_source": "email"},
        "evidence_text": "Reminder from my colleague about the deadline.",
    })
    assert result["decision"] == "pending_user_confirmation", (
        f"Scanner exception must fail closed, got {result['decision']}"
    )
    assert "inbound" in result["review_model"].lower() or "scan" in result["reason"].lower()

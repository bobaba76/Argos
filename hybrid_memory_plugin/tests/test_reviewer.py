"""Tests for deterministic reviewer policy gates."""
from __future__ import annotations


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


def test_sensitive_candidate_requires_confirmation_when_reviewer_unavailable():
    from reviewer import review_candidate_with_llm

    result = review_candidate_with_llm({
        "category": "personal_fact",
        "content": "Alex has a medical diagnosis",
        "payload": {},
        "evidence_text": "My name is Alex and I have a medical diagnosis.",
    })
    assert result["decision"] == "pending_user_confirmation"


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

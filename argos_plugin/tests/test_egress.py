"""Egress gate tests.

The egress gate refuses a plugin-owned LLM call when local_only is on, or
when a conversation-derived payload carries PII identifiers; the caller
then fails soft. Store-derived payloads (graph typing, distillation) are
governed by their own config flags plus local_only.
"""
from __future__ import annotations

import sys
from pathlib import Path

_plugin_dir = Path(__file__).resolve().parent.parent
for _path in (_plugin_dir.parent, _plugin_dir):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import pytest

from egress import (  # noqa: E402
    SENSITIVE_KINDS,
    SITES,
    contains_sensitive,
    gate,
    report,
    site_live,
)

ALL_KINDS = [site["kind"] for site in SITES]

PLAIN = "Alex prefers flat white coffee on Sundays."
SENSITIVE = (
    "Alex's number is 0831234567, email alex.p@example.co.za, "
    "ID 8601015012084, card 4912 3456 7890 1234."
)


def test_local_only_blocks_every_kind():
    cfg = {"local_only": "true"}
    for kind in ALL_KINDS:
        assert gate(kind, PLAIN, cfg) is False


def test_sensitive_identifier_blocks_conversation_kinds():
    cfg = {}
    for kind in SENSITIVE_KINDS:
        assert gate(kind, SENSITIVE, cfg) is False


def test_store_kinds_ignore_sensitive_identifier_gate():
    cfg = {}
    for kind in ("graph_typing", "distillation"):
        assert gate(kind, SENSITIVE, cfg) is True


def test_gate_allows_plain_text():
    cfg = {}
    assert gate("extractor", PLAIN, cfg) is True
    assert gate("reviewer", PLAIN, cfg) is True
    assert gate("query_expansion", PLAIN, cfg) is True


def test_gate_fails_closed_on_unknown_kind():
    """An unknown kind must be refused (fail-closed), not allowed."""
    cfg = {}
    assert gate("nonexistent_kind", PLAIN, cfg) is False
    assert gate("", PLAIN, cfg) is False


def test_contains_sensitive_labels_identifiers():
    assert contains_sensitive("mail a@b.co.za today") == "email address"
    assert contains_sensitive("phone 0831234567") == "South African phone number"
    assert contains_sensitive("id 8601015012084") == "13-digit ID number"
    assert contains_sensitive("no identifiers here") is None
    assert contains_sensitive("") is None


def test_site_live_reflects_config_and_local_only():
    dist = {"kind": "distillation", "gate": "distillation_enabled", "default": False}
    assert site_live(dist, {}) == "OFF"
    assert site_live(dist, {"distillation_enabled": "true"}) == "ON"
    assert site_live(dist, {"local_only": "true"}) == "blocked"


def test_report_mentions_all_sites_and_groups():
    out = report({"local_only": "true"})
    for kind in ALL_KINDS:
        assert kind in out
    for name, _kinds in __import__("egress").GROUPS:
        assert name in out
    assert "local_only: True" in out


def test_reviewer_downgrades_on_sensitive_evidence():
    """The reviewer refuses to send sensitive payloads; waits for the user."""
    from reviewer import review_candidate_with_llm

    candidate = {
        "category": "personal_fact",
        "content": "Alex's new contact detail",
        "payload": {},
        "evidence_text": "My email is alex.person@example.com and I just moved.",
    }
    res = review_candidate_with_llm(candidate)
    assert res["decision"] == "pending_user_confirmation"
    assert res["review_model"] == "egress_gate"


def test_reviewer_healthy_candidate_fails_soft_when_client_unavailable(monkeypatch):
    """Without local_only/sensitive content, the reviewer tries the LLM
    and fails soft with reviewer_unavailable when the client is down."""
    import sys
    import types

    from reviewer import review_candidate_with_llm

    fake_client = types.ModuleType("agent.auxiliary_client")

    def boom(*a, **k):
        raise RuntimeError("client unavailable")

    fake_client.call_llm = boom
    fake_agent = types.ModuleType("agent")
    fake_agent.auxiliary_client = fake_client
    monkeypatch.setitem(sys.modules, "agent", fake_agent)
    monkeypatch.setitem(sys.modules, "agent.auxiliary_client", fake_client)

    candidate = {
        "category": "preference",
        "content": "Alex prefers flat whites",
        "payload": {},
        "evidence_text": "I like flat white coffee.",
    }
    res = review_candidate_with_llm(candidate)
    # Fail-soft property: on client failure the candidate is never
    # auto-approved — it stays pending for user confirmation.
    assert res["decision"] == "pending_user_confirmation"
    assert "unavailable" in res.get("reason", "")
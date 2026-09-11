"""Egress gate tests.

The egress gate refuses a plugin-owned LLM call when local_only is on, or
when a conversation-derived payload carries PII identifiers; the caller
then fails soft. Store-derived payloads (graph typing, distillation,
rollup) run under store_derived_identifier_mode (#404): "redact" (the
default) masks identifiers to [redacted: ...] markers and proceeds;
"gate" refuses such calls; "off" sends unchanged.
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
    STORE_DERIVED_KINDS,
    contains_sensitive,
    gate,
    gate_payload,
    redact,
    report,
    site_live,
    store_derived_identifier_mode,
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
    # E3: the per-site flag is now enforced in gate(), so distillation
    # (default OFF) must be explicitly enabled here to isolate the test's
    # original intent: the sensitive-identifier REFUSAL gate does not
    # apply to store-derived kinds at the gate() layer.
    # #404: gate() remains the low-level primitive; the store-derived
    # identifier policy lives in gate_payload() (default: redact — masks,
    # does not refuse) which the store-derived call sites MUST use.
    cfg = {"distillation_enabled": "true", "llm_fallback": "true"}
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

# ---------------------------------------------------------------------------
# #404: store-derived identifier policy (redact | gate | off)
# ---------------------------------------------------------------------------

def test_redact_masks_identifiers_and_reports_labels():
    out, labels = redact(SENSITIVE)
    assert "0831234567" not in out
    assert "alex.p@example.co.za" not in out
    assert "8601015012084" not in out
    assert "[redacted: email address]" in out
    assert "South African phone number" in labels
    assert "email address" in labels
    assert redact("") == ("", [])
    assert redact("no identifiers here") == ("no identifiers here", [])


def test_gate_payload_redact_is_default():
    cfg = {
        "distillation_enabled": "true",
        "rollup_enabled": "true",
        "llm_fallback": "true",
    }
    allowed, send = gate_payload("graph_typing", SENSITIVE, cfg)
    assert allowed is True
    assert "alex.p@example.co.za" not in send
    assert "[redacted:" in send
    # Plain text passes through untouched.
    _, send_plain = gate_payload("graph_typing", PLAIN, cfg)
    assert send_plain == PLAIN
    # All three store-derived kinds are redacted by default.
    for kind in STORE_DERIVED_KINDS:
        allowed, send = gate_payload(kind, SENSITIVE, cfg)
        assert allowed is True, kind
        assert "alex.p@example.co.za" not in send, kind
        assert "[redacted:" in send, kind


def test_gate_payload_gate_mode_refuses_identifiers():
    cfg = {
        "store_derived_identifier_mode": "gate",
        "distillation_enabled": "true",
    }
    allowed, send = gate_payload("distillation", SENSITIVE, cfg)
    assert allowed is False
    assert send == SENSITIVE  # unchanged; caller must not send it
    allowed, _ = gate_payload("distillation", PLAIN, cfg)
    assert allowed is True


def test_gate_payload_off_mode_is_legacy():
    cfg = {
        "store_derived_identifier_mode": "off",
        "distillation_enabled": "true",
    }
    allowed, send = gate_payload("distillation", SENSITIVE, cfg)
    assert allowed is True
    assert send == SENSITIVE


def test_gate_payload_invalid_mode_falls_back_to_redact():
    cfg = {
        "store_derived_identifier_mode": "skip-clever-things",
        "llm_fallback": "true",
    }
    assert store_derived_identifier_mode(cfg) == "redact"
    allowed, send = gate_payload("graph_typing", SENSITIVE, cfg)
    assert allowed is True
    assert "alex.p@example.co.za" not in send


def test_gate_payload_conversation_kinds_keep_refusal_semantics():
    """Non-store kinds: gate_payload is exactly gate() — refusal, and the
    text is never silently rewritten."""
    allowed, send = gate_payload("extractor", SENSITIVE, {})
    assert allowed is False
    assert send == SENSITIVE
    allowed, send = gate_payload("extractor", PLAIN, {})
    assert allowed is True
    assert send == PLAIN


def test_gate_payload_local_only_and_site_flag_still_apply():
    assert (
        gate_payload("graph_typing", PLAIN, {"local_only": "true"})[0] is False
    )
    # Distillation defaults OFF in the config; the site flag still refuses.
    assert gate_payload("distillation", PLAIN, {})[0] is False
    cfg = {"distillation_enabled": "true"}
    assert gate_payload("distillation", PLAIN, cfg)[0] is True


def test_two_hop_store_regression_rollup_prompt_redacted():
    """#404 regression: an identifier refused at the conversation gate can
    still be STORED (regex extraction never sends it anywhere); the
    store-derived re-send must carry it masked, not raw."""
    cfg = {"rollup_enabled": "true"}
    stored = "\n".join(
        ["- Alex contact alex.p@example.co.za", "- phone 0831234567"]
    )
    allowed, send = gate_payload("memory_rollup", "Records:\n" + stored, cfg)
    assert allowed is True
    assert "alex.p@example.co.za" not in send
    assert "0831234567" not in send
    assert send.count("[redacted:") >= 2


def test_report_shows_store_derived_mode():
    out = report({"local_only": "false"})
    assert "store_derived_identifier_mode: redact" in out
    out = report({"store_derived_identifier_mode": "gate"})
    assert "store_derived_identifier_mode: gate" in out


def test_store_derived_call_sites_route_through_gate_payload():
    """#404 audit: the graph/distillation/rollup call sites must route
    through gate_payload so the policy reaches the wire; a bare gate()
    call for these kinds would silently bypass redaction."""
    import re as _re

    here = Path(__file__).resolve().parent.parent
    expectations = {
        "graph.py": "graph_typing",
        "distillation.py": "distillation",
        "rollup.py": "memory_rollup",
    }
    for fname, kind in expectations.items():
        src = (here / fname).read_text(encoding="utf-8")
        assert _re.search(
            "gate_payload\\(\\s*\"%s\"" % kind, src
        ), "%s must route %s through gate_payload (#404)" % (fname, kind)


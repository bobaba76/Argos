"""Conservative LLM review for pending memory proposals."""
from __future__ import annotations

import json
import re
from typing import Any, Dict

if __package__:
    from .extractor import hard_quality_flags, quality_flags_for_fact
else:
    from extractor import hard_quality_flags, quality_flags_for_fact

_SENSITIVE_RE = re.compile(
    r"\b(?:financial|salary|password|secret|private|personal|confidential|"
    r"sensitive|classified|internal|restricted|proprietary|relationship|"
    r"wife|husband|partner|girlfriend|boyfriend|age|birthday|location|lives\s+in|"
    r"works?\s+(?:at|for)|job|identity|name)\b",
    re.IGNORECASE,
)


def is_sensitive_candidate(candidate: Dict[str, Any]) -> bool:
    category = str(candidate.get("category", ""))
    content = str(candidate.get("content", ""))
    if category == "relationship":
        return True
    if category == "personal_fact" and _SENSITIVE_RE.search(content):
        return True
    if category in {"insight", "context_note"} and _SENSITIVE_RE.search(content):
        return True
    return False


def _parse_json_response(response: Any) -> dict | None:
    try:
        text = response.choices[0].message.content
    except (AttributeError, IndexError, KeyError, TypeError):
        return None
    if not text or not text.strip():
        return None
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def review_candidate_with_llm(candidate: Dict[str, Any], *, model: str = "", provider: str = "") -> Dict[str, Any]:
    """Review a candidate; never auto-approve sensitive or contextless data."""
    payload = candidate.get("payload") or {}
    flags = quality_flags_for_fact(candidate)
    hard_flags = hard_quality_flags(flags)
    if hard_flags:
        return {
            "decision": "quarantined",
            "confidence": 0.99,
            "reason": "Deterministic quality gate: " + ", ".join(hard_flags),
            "review_model": "deterministic_gate",
        }

    evidence = str(candidate.get("evidence_text") or "").strip()
    if not evidence:
        return {
            "decision": "pending_user_confirmation",
            "confidence": 0.0,
            "reason": "No original user evidence is available for this proposal.",
            "review_model": "evidence_gate",
        }

    try:
        from agent.auxiliary_client import call_llm
    except Exception:
        return {
            "decision": "pending_user_confirmation",
            "confidence": 0.0,
            "reason": "The Hermes auxiliary LLM reviewer is unavailable.",
            "review_model": "reviewer_unavailable",
        }

    system = """You are a conservative reviewer of a memory proposal for a personal AI assistant.
Review only what the USER explicitly states or unambiguously supports in the evidence.
Do not treat assistant text, questions, plans, speculation, roleplay, or conversation
scaffolding as durable fact. Do not invent missing context.

Return one JSON object with exactly:
- decision: approve, reject, quarantine, or pending_user_confirmation
- confidence: number from 0 to 1
- reason: concise explanation
- durability: permanent, durable, or temporary
- scope: profile, project, or session

Use pending_user_confirmation for sensitive topics, identity, ambiguous personal
claims, or anything where the evidence does not clearly support the proposed memory.
Use reject for obvious non-memory text. Use quarantine for malformed or suspicious text.
"""
    user = json.dumps(
        {
            "proposal": {
                "category": candidate.get("category"),
                "content": candidate.get("content"),
                "tags": candidate.get("tags", []),
                "source": candidate.get("source"),
                "confidence": candidate.get("confidence"),
                "scope": candidate.get("scope"),
                "quality_flags": flags,
            },
            "evidence": {
                "role": candidate.get("evidence_role", "user_turn"),
                "text": evidence,
                "session_id": candidate.get("session_id", ""),
            },
            "sensitive_candidate": is_sensitive_candidate(candidate),
            "payload_metadata": {
                key: value for key, value in payload.items()
                if key in {"source", "quality_flags", "fact_type", "legacy_store"}
            },
        },
        ensure_ascii=False,
    )
    try:
        response = call_llm(
            task="memory_review",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.0,
            max_tokens=500,
            timeout=15.0,
            model=model or None,
            provider=provider or None,
        )
        parsed = _parse_json_response(response)
    except Exception as exc:
        parsed = None
        failure = str(exc)
    else:
        failure = ""

    if not parsed:
        return {
            "decision": "pending_user_confirmation",
            "confidence": 0.0,
            "reason": "Reviewer returned no valid decision" + (f": {failure}" if failure else "."),
            "review_model": "memory_review",
        }

    decision = str(parsed.get("decision", "pending_user_confirmation")).lower().strip()
    if decision not in {"approve", "reject", "quarantine", "pending_user_confirmation"}:
        decision = "pending_user_confirmation"
    try:
        confidence = max(0.0, min(1.0, float(parsed.get("confidence", 0.0))))
    except (TypeError, ValueError):
        confidence = 0.0
    reason = str(parsed.get("reason", "No review reason supplied")).strip()[:1000]

    sensitive = is_sensitive_candidate(candidate)
    if sensitive and decision == "approve":
        decision = "pending_user_confirmation"
        reason = "Sensitive proposal requires user confirmation. " + reason
    if confidence < 0.85 and decision in {"approve", "reject", "quarantine"}:
        decision = "pending_user_confirmation"
        reason = "Reviewer confidence is below the automatic decision threshold. " + reason

    durability = str(parsed.get("durability", candidate.get("durability", "durable"))).lower()
    if durability not in {"permanent", "durable", "temporary"}:
        durability = candidate.get("durability", "durable")
    scope = str(parsed.get("scope", candidate.get("scope", "profile"))).lower()
    if scope not in {"profile", "project", "session"}:
        scope = candidate.get("scope", "profile")
    return {
        "decision": decision,
        "confidence": confidence,
        "reason": reason,
        "durability": durability,
        "scope": scope,
        "review_model": "memory_review",
    }

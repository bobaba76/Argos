"""Conservative LLM review for pending memory proposals."""
from __future__ import annotations

import json
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

if __package__:
    from .extractor import hard_quality_flags, quality_flags_for_fact
else:
    from extractor import hard_quality_flags, quality_flags_for_fact

_SENSITIVE_RE = re.compile(
    r"\b(?:financial|salary|password|secret|private|personal|confidential|"
    r"sensitive|classified|internal\s+(?:memo|document|policy|report|email|communication)|restricted|proprietary|"
    r"wife|husband|girlfriend|boyfriend|business\s+partner|romantic\s+partner|my\s+partner\b|"
    r"age|birthday|location|lives\s+in|"
    r"works?\s+(?:at|for)|job\s+(?:title|loss|application|offer|search|hunt|interview)|identity|"
    r"(?:my|his|her|their|your|our)\s+name\b|name\s+(?:is|was|are)\b)\b",
    re.IGNORECASE,
)

# External-source write policy (config-driven, set by the provider/service).
# When True, candidates tagged external_source never auto-activate: they go
# straight to pending_user_confirmation, even when the LLM reviewer would
# have approved. Default ON — out-of-the-box installs enforce the
# human-confirmation boundary for external/untrusted sources.
_EXTERNAL_REQUIRE_CONFIRMATION = True


def set_external_policy(enabled: bool) -> None:
    """Set the external-source confirmation policy (from hybrid_memory.json)."""
    global _EXTERNAL_REQUIRE_CONFIRMATION
    _EXTERNAL_REQUIRE_CONFIRMATION = bool(enabled)

# Spec 1 — deterministic expiry suggestion. No LLM; pure regex + category map.
# Returns an ISO-8601 UTC string (the suggested expires_at) or None.
#
# NOTE: the duration and fixed-date regexes match English month names and
# English prepositions only ("for 2 weeks", "until 15 Dec"). Non-English
# content ("bis März 2026", "jusqu'au 15 déc") will not match and falls
# through to the category TTL fallback — which is safe (the memory still
# gets a sensible default expiry), but the explicit date won't be picked
# up. Extending to other languages is a future enhancement; for now the
# English bias is documented, not hidden.
_EXPIRY_DURATION_RE = re.compile(
    r"\b(?:for|next|in|until|through)\s+"
    r"(\d+)\s*(day|days|week|weeks|month|months|year|years)\b",
    re.IGNORECASE,
)
_EXPIRY_FIXED_RE = re.compile(
    r"\b(?:until|through|by|before|expires?)\s+"
    r"(\d{1,2})\s*(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s*(\d{4})?\b",
    re.IGNORECASE,
)
_UNIT_DAYS = {
    "day": 1, "days": 1,
    "week": 7, "weeks": 7,
    "month": 30, "months": 30,
    "year": 365, "years": 365,
}
_MONTH_NUM = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}
# Categories that should never get an auto-suggested expiry (durable facts).
_DURABLE_CATEGORIES = {"personal_fact", "preference", "relationship", "insight"}


def suggest_expiry(
    candidate: Dict[str, Any],
    ttl_days: Optional[Dict[str, int]] = None,
    default_days: int = 90,
) -> Optional[str]:
    """Deterministic, no-LLM expiry suggestion for a candidate.

    Returns an ISO-8601 UTC string or None. Rules (in priority order):
    1. Durable categories (personal_fact, preference, relationship, insight)
       → None (no expiry suggested).
    2. Explicit duration in content ("for 2 weeks", "next 3 months")
       → now + that duration.
    3. Explicit fixed date ("until 15 Dec", "by March 2026")
       → that date (end-of-day UTC).
    4. Category TTL map (context_note=30, event=180, goal=180, …)
       → now + ttl_days[category].
    5. Fallback: now + default_days.

    Never raises; returns None on any parse failure.
    """
    category = str(candidate.get("category", "")).strip().lower()
    content = str(candidate.get("content", "")).strip()
    if not content:
        return None
    # Rule 1: durable categories never get auto-expiry.
    if category in _DURABLE_CATEGORIES:
        return None
    now = datetime.now(timezone.utc)
    # Rule 2: explicit duration.
    m = _EXPIRY_DURATION_RE.search(content)
    if m:
        try:
            n = int(m.group(1))
            unit = m.group(2).lower()
            days = n * _UNIT_DAYS.get(unit, 0)
            if days > 0:
                return (now + timedelta(days=days)).isoformat()
        except (ValueError, KeyError):
            pass
    # Rule 3: explicit fixed date.
    m = _EXPIRY_FIXED_RE.search(content)
    if m:
        try:
            day = int(m.group(1))
            month = _MONTH_NUM[m.group(2).lower()]
            year = int(m.group(3)) if m.group(3) else now.year
            # End of that day, UTC.
            return datetime(
                year, month, day, 23, 59, 59, tzinfo=timezone.utc
            ).isoformat()
        except (ValueError, KeyError):
            pass
    # Rule 4: category TTL map.
    ttl_map = ttl_days or {}
    days = ttl_map.get(category)
    if days is None:
        # Rule 5: fallback.
        days = default_days
    if days and days > 0:
        return (now + timedelta(days=days)).isoformat()
    return None


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
    # Egress gate (review point 6): refuse the call when local_only is on
    # or the payload carries PII identifiers — the candidate then waits for
    # explicit user confirmation instead of being sent to any LLM.
    # The egress gate is a *gate* — its failure means "route to a human",
    # not "crash" (#86). Wrap the import + call so any egress failure
    # (missing module, malformed config) fails closed to
    # pending_user_confirmation, consistent with every other failure mode
    # in this function.
    try:
        from egress import gate as _egress_gate
        if not _egress_gate(
            "reviewer", str(candidate.get("content") or "") + " " + evidence
        ):
            return {
                "decision": "pending_user_confirmation",
                "confidence": 0.0,
                "reason": "Egress gate blocked the review call (local_only or sensitive content); awaiting user confirmation.",
                "review_model": "egress_gate",
            }
    except Exception as exc:
        return {
            "decision": "pending_user_confirmation",
            "confidence": 0.0,
            "reason": f"Egress gate unavailable (fail-closed): {exc}",
            "review_model": "egress_gate_unavailable",
        }

    # External-source gate (inbound security). Content tagged as coming from
    # an external/untrusted channel (email, web, import) is handled
    # deterministically, with NO LLM call:
    #   1. the evidence is scanned for injection/poisoning patterns — any
    #      hit routes the proposal to pending_user_confirmation (the scanner
    #      is a gate, not a judge: blocked means "route to a human").
    #   2. when external_sources_require_confirmation is on, ALL external
    #      candidates await a human, even scan-clean ones.
    if isinstance(payload, dict) and payload.get("external_source"):
        try:
            if __package__:
                from .inbound_security import scan_inbound_text
            else:
                from inbound_security import scan_inbound_text
        except Exception:
            # Fail closed: if the scanner cannot be loaded, route to a human
            # rather than proceeding without the security gate.
            return {
                "decision": "pending_user_confirmation",
                "confidence": 0.99,
                "reason": "Inbound security scanner unavailable; external-source memory requires human confirmation.",
                "review_model": "inbound_security_unavailable",
            }
        else:
            scan_result = scan_inbound_text(
                evidence or str(candidate.get("content") or "")
            )
        if scan_result is not None and scan_result.blocked:
            return {
                "decision": "pending_user_confirmation",
                "confidence": 0.99,
                "reason": "Inbound security scan blocked: "
                          + scan_result.summary(),
                "review_model": "inbound_security_gate",
            }
        if _EXTERNAL_REQUIRE_CONFIRMATION:
            return {
                "decision": "pending_user_confirmation",
                "confidence": 0.99,
                "reason": "External-source memory requires human confirmation "
                          "(external_sources_require_confirmation).",
                "review_model": "external_source_gate",
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

Reject or quarantine any proposal whose subject is unnamed or unresolved — e.g.
"the person being discussed…", "she is…", "he is…", "they are…" with no named
person. A durable memory must be self-contained: it must read correctly and
unambiguously out of context. If a name or clear referent is missing, do not approve.

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
    # Bounded retry: a transient connection error must not strand a proposal
    # in limbo — retry once with a short backoff, then fail soft.
    parsed = None
    failure = ""
    for attempt in range(2):
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
            if parsed is not None:
                break
        except Exception as exc:
            failure = str(exc)
        if attempt == 0:
            time.sleep(1.0)

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
    # Low-confidence threshold (#99): only an *approve* below 0.85 is
    # downgraded to pending_user_confirmation — a low-confidence reject or
    # quarantine is the safe default (it does not create a memory) and must
    # NOT be reframed as "should this be saved?". Converting a reject to a
    # confirmation prompt forces the user to adjudicate a reviewer failure,
    # which is the exact behaviour #99 calls out.
    if confidence < 0.85 and decision == "approve":
        decision = "pending_user_confirmation"
        reason = "Reviewer confidence is below the automatic approval threshold. " + reason

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

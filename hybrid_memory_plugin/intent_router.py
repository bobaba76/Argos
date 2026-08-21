"""Argos intent router — P2A.

Decides which ANSWERER model this turn should use, based on lightweight
temporal / multi-hop query detection, and returns it as a ``model`` key that
the core ``pre_llm_call`` hook path honors (see agent/turn_context.py).

Design (v2 — class-based, confidence-scored):
  * Temporal and multi-hop are scored as *evidence classes*, not literal
    regex hits.  Verb groups (reporting/comparison), date/time NER-lite,
    weekday/month/year tables, relative-time adverbs, and entity-pair
    heuristics each contribute additive confidence.
  * A SINGLE strong multi-hop signal now routes (e.g. "what did she say about
    about the move") — the old two-marker rule let the whole class fall
    through to Flash.  Ordinary turns score ~0 and stay on the default
    model, preserving the cost win.
  * Thresholds are module constants, config-overridable
    (router_temporal_threshold / router_multihop_threshold).
  * Always returns an explicit model (smart for temporal/multi-hop, default
    otherwise) so the answerer self-corrects across turns.
  * Failure here is never allowed to break the turn — best-effort only.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# -- Default routing thresholds -------------------------------------------------
# Temporal evidence is the strongest measured signal (+26pp bucket), so its
# threshold is lower.  Multi-hop requires a little more confidence to avoid
# erasing the cost win with over-routing, but a single reporting verb + entity
# must clear it (that was the dead-class bug).
ROUTE_TEMPORAL_THRESHOLD = 0.38
# Multi-hop confidence above this routes to the smart model.
# NOTE: deliberately low. A single strong reporting/comparison verb PLUS a
# topic or entity is already genuine multi-hop ("what did she say about
# the move") — the old 2-marker gate (≥2 signals ≈ 0.70+) silently starved
# the whole class. Probe 2026-08-21: 6 multi-hop queries at 0.35, plain
# queries at 0.00 → 0.32 separates cleanly with zero precision loss.
ROUTE_MULTI_HOP_THRESHOLD = 0.32

# ---------------------------------------------------------------------------
# Temporal evidence classes
# ---------------------------------------------------------------------------

_MONTHS = {
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december", "jan", "feb", "mar", "apr",
    "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
}
_WEEKDAYS = {
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday",
    "sunday", "mon", "tue", "tues", "wed", "thu", "thur", "thurs", "fri",
    "sat", "sun",
}
_RELATIVE_TIME = {
    "yesterday", "today", "tomorrow", "ago", "earlier", "previously",
    "recently", "back then", "last week", "last month", "last year",
    "next week", "next month", "next year", "this week", "this month",
    "this year", "tonight", "last night", "midnight", "noon", "morning",
    "afternoon", "evening",
}
_DURATION_RE = re.compile(
    r"\b\d+\s*(day|days|week|weeks|fortnight|fortnights|month|months|"
    r"year|years|hour|hours|minute|minutes|decade|decades)\b", re.I,
)
_DATE_RE = re.compile(
    r"\b\d{1,2}(st|nd|rd|th)?\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|"
    r"oct|nov|dec|january|february|march|april|june|july|august|september|"
    r"october|november|december)[a-z]*\b", re.I,
)
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_WHEN_RE = re.compile(
    r"\b(how long|when\s+(?:did|was|were|is|are|does|do|will|has|have)|"
    r"what\s+(?:year|date|month|day)|how many (?:days|weeks|months|years)|"
    r"has it been|since|until|before|after|between\s+\d{4}\s+and\s+\d{4}|"
    r"how old)\b", re.I,
)
_PAST_TENSE_FACT_RE = re.compile(
    r"\b(what|when|how)\s+(did|was|were)\s+(i|we|you|the|my|our|"
    r"she|he|it|they)\b", re.I,
)
_ORDER_RE = re.compile(
    r"\bwhat\s+(?:came|happened|was)\s+(?:first|before|after)\b"
    r"|\b(?:what|which)\s+(?:one|event|thing)\s+(?:came|happened)\s+(?:first|before|after)\b"
    r"|\b(?:happened|occurred)\s+(?:before|after)\b",
    re.I,
)

# ---------------------------------------------------------------------------
# Multi-hop evidence classes
# ---------------------------------------------------------------------------

# Reporting verbs: one hit with an entity/topic nearby is a strong signal.
_REPORTING_VERBS = {
    "said", "says", "say", "told", "tell", "mentioned", "mentions",
    "discussed", "discuss", "agreed", "agree", "explained", "explain",
    "wrote", "write", "sent", "send", "asked", "ask", "recommended",
    "recommend", "suggested", "suggest", "claimed", "claim", "stated",
    "state", "replied", "reply", "messaged", "mentioned", "talked about",
    "talk about", "brought up", "raised",
}
_COMPARISON_VERBS = {
    "compare", "compared", "comparison", "versus", "vs", "different",
    "differ", "better than", "worse than", "how does", "how do", "which is",
    "which one", "vs.",
}
_CHAIN_TERMS = {
    "and then", "after that", "related to", "connected", "both",
    "in common", "link", "links", "involves", "involve", "ties to",
}
# Second-person reference to a prior assistant/user statement.
_YOU_SAID_RE = re.compile(
    r"\b(you\s+(said|told|wrote|mentioned|agreed|suggested|recommended|"
    r"advised|claimed|explained|sent))\b", re.I,
)
# Entity-pair: two capitalized names or "X (and|then) Y" where both look like
# people/places (proper-noun heuristic).  Avoids "A and B products" false hits
# by requiring at least one reporting/comparison/chain verb nearby.
_QUESTION_RE = re.compile(
    r"\b(?:what|when|where|who|which|why|how)\b"
    r"|\b(?:did|does|do|is|are|was|were|has|have|had)\s+\w+"
    r"|\?",
    re.IGNORECASE,
)

_ENTITY_PAIR_RE = re.compile(
    r"\b([A-Z][a-z]+)\s+(and|vs|versus|then|compared to)\s+([A-Z][a-z]+)\b"
)

_WORDS_RE = re.compile(r"[a-z0-9']+", re.I)


def _words(text: str):
    return _WORDS_RE.findall(text.lower())


def _has_any(text_lower: str, items) -> bool:
    return any(item in text_lower for item in items)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def temporal_score(query: str) -> float:
    """Additive temporal confidence in [0, 1].

    A single strong signal (year, dated month, "when did", relative-day)
    scores >= 0.5; anything above ROUTE_TEMPORAL_THRESHOLD routes.
    """
    q = " ".join(query.lower().split())
    if not q:
        return 0.0
    score = 0.0
    words = set(_words(q))
    # Month / weekday mentions (any tense-reference context).
    if words & _MONTHS:
        score += 0.50
    if _WEEKDAYS & words:
        score += 0.50
    # Chronological ordering ("what came first", "what happened before/after X").
    if _ORDER_RE.search(q):
        score += 0.50
    # Relative-time adverbs.
    if _has_any(q, _RELATIVE_TIME):
        score += 0.45
    # Explicit durations, dated formats, years.
    if _DURATION_RE.search(q):
        score += 0.45
    if _DATE_RE.search(q):
        score += 0.60
    if _YEAR_RE.search(q):
        score += 0.55
    # Time-interrogative structures.
    if _WHEN_RE.search(q):
        score += 0.50
    # Past-tense fact probes ("what did I…", "when was my…") — weak but real.
    if _PAST_TENSE_FACT_RE.search(q):
        score += 0.30
    return min(1.0, score)


def multi_hop_score(query: str) -> float:
    """Additive multi-hop confidence in [0, 1].

    Reporting verb + entity/topic reference is the classic dead class; a
    single reporting verb alone scores ~0.35 (below threshold), but with an
    entity (proper noun) or a chain/comparison term it clears 0.50.
    """
    q = " ".join(query.lower().split())
    if not q:
        return 0.0
    score = 0.0
    words = set(_words(q))
    reporting_hits = words & _REPORTING_VERBS
    comparison_hits = words & _COMPARISON_VERBS
    chain_hits = words & _CHAIN_TERMS

    if reporting_hits:
        score += 0.35 * min(len(reporting_hits), 2)
    if comparison_hits:
        score += 0.35 * min(len(comparison_hits), 2)
    if chain_hits:
        score += 0.30 * min(len(chain_hits), 2)
    if _YOU_SAID_RE.search(q):
        score += 0.35
    # Entity-pair / proper-noun adjacency — only pays when >=1 verb class hit
    # already present (prevents "X and Y products" false positives).
    if _ENTITY_PAIR_RE.search(q) and (reporting_hits or comparison_hits or chain_hits):
        score += 0.30
    return min(1.0, score)


def is_temporal_or_multihop(query: str) -> bool:
    """Return True when the query reads as temporal and/or multi-hop.

    Only genuine questions route (interrogative word, auxiliary-led
    question, or "?"); commands and statements ("tell me a joke",
    "schedule a meeting for friday") never pay the smart-model cost.
    """
    if not query or not query.strip():
        return False
    if not _QUESTION_RE.search(query):
        return False
    return (
        temporal_score(query) >= ROUTE_TEMPORAL_THRESHOLD
        or multi_hop_score(query) >= ROUTE_MULTI_HOP_THRESHOLD
    )


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("true", "1", "yes", "on")


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def route_answerer(config: Dict[str, Any], user_message: str) -> Optional[Dict[str, str]]:
    """Return ``{"model": ..., "provider": ...}`` (provider optional) or None.

    Called from ``_on_pre_llm_call``.  Returns the smart model for
    temporal/multi-hop queries and the default model otherwise (so turns
    self-correct back), or None when routing is disabled / not configured.
    """
    try:
        if not _as_bool(config.get("router_enabled"), default=False):
            return None
        smart = str(config.get("router_smart_model") or "").strip()
        default = str(config.get("router_default_model") or "").strip()
        if not smart or not default:
            logger.debug("router: enabled but missing smart/default model config")
            return None
        msg = user_message or ""
        if not _QUESTION_RE.search(msg):
            pick = default
            provider = str(config.get("router_default_provider") or "").strip()
            result: Dict[str, str] = {"model": pick}
            if provider:
                result["provider"] = provider
            return result
        temporal = temporal_score(msg)
        multi_hop = multi_hop_score(msg)
        t_thresh = _as_float(config.get("router_temporal_threshold"), ROUTE_TEMPORAL_THRESHOLD)
        m_thresh = _as_float(config.get("router_multihop_threshold"), ROUTE_MULTI_HOP_THRESHOLD)
        if temporal >= t_thresh or multi_hop >= m_thresh:
            pick = smart
            provider = str(config.get("router_smart_provider") or "").strip()
        else:
            pick = default
            provider = str(config.get("router_default_provider") or "").strip()
        result: Dict[str, str] = {"model": pick}
        if provider:
            result["provider"] = provider
        return result
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("router: failed to route: %s", exc)
        return None
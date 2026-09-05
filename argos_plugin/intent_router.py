"""Argos intent router — P2A.

Decides which ANSWERER model this turn should use, based on lightweight
temporal / multi-hop query detection, and returns it as a ``model`` key that
the core ``pre_llm_call`` hook path honors (see agent/turn_context.py).

Design (v3 — precision-fixed):
  * v2 over-routed: a SINGLE reporting verb (said/told/asked/... = 0.35) beat
    the 0.32 multi-hop threshold and a single relative-time weekday/month word
    (today/morning/friday = 0.45-0.50) beat the 0.38 temporal threshold, so
    ordinary chat was being switched to the smart model.  Probing 2026-08-21
    with realistic casual messages showed 5/10 question-shaped casual queries
    falsely routed to deepseek-v4-pro (openrouter) — burning real money.
  * v3: single weak signals never route.  Multi-hop requires a reporting verb
    AND a proper-noun entity (Alex/Devin/...) or an "about X" topic phrase.
    Temporal requires a genuine time anchor: an actual date/year, an
    order/interrogative structure ("when did", "how long", "between M and M"),
    or a past-tense fact probe combined with another signal.  Bare "friday",
    "today", "last night", "morning" in a casual question stay on Flash.
  * Thresholds are module constants, config-overridable
    (router_temporal_threshold / router_multihop_threshold).
  * Always returns an explicit model (smart for temporal/multi-hop, default
    otherwise) so the answerer self-corrects across turns.
  * Failure here is never allowed to break the turn — best-effort only.
    Swallowed routing failures are counted in ``ROUTING_FAILURES`` (see
    ``routing_failure_count``) so they stay visible to monitoring.

Scoring model: ``temporal_score`` / ``multi_hop_score`` are SUMS of
independent evidence weights, clamped to 1.0.  They are NOT probabilities —
a query hitting many signals can exceed 1.0 before the clamp, and a score of
0.50 does not mean "50% likely".  Only the ordering relative to the
thresholds is meaningful.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# -- Default routing thresholds -------------------------------------------------
# v3: raised from (0.38, 0.32).  A single weak signal must NOT route; the
# deliberate gaps below each single-signal weight let casual chat stay on the
# cheap default model.  Genuine queries clear by wide margins.
ROUTE_TEMPORAL_THRESHOLD = 0.50
ROUTE_MULTI_HOP_THRESHOLD = 0.50

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
# Relative-time adverbs: weak on their own (casual chat is full of them).
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
# Strong time-interrogative structures (these anchor a genuine temporal query).
# "what happened" added in v3 so "what happened last night" clears via the
# weak-relative-time + interrogative combination.
_WHEN_RE = re.compile(
    r"\b(how long|when\s+(?:did|was|were|is|are|does|do|will|has|have)|"
    r"what\s+(?:year|date|month|day)|how many (?:days|weeks|months|years)|"
    r"has it been|since|until|how old|what happened)\b"
    r"|between\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec|"
    r"january|february|march|april|june|july|august|september|october|"
    r"november|december)[a-z]*\s+and\s+(?:jan|feb|mar|apr|may|jun|jul|aug|"
    r"sep|sept|oct|nov|dec|january|february|march|april|june|july|august|"
    r"september|october|november|december)[a-z]*",
    re.I,
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

# Reporting verbs are DEMOTED in v3: a bare verb alone (0.25) must not route;
# it only pays when an entity or "about X" topic is present.
# Single-word verbs are matched against the query's word set; multi-word
# phrases are matched with a word-boundary regex (a set intersection can
# never hit a phrase containing a space).
_REPORTING_VERBS = {
    "said", "says", "say", "told", "tell", "mentioned", "mentions",
    "discussed", "discuss", "agreed", "agree", "explained", "explain",
    "wrote", "write", "sent", "send", "asked", "ask", "recommended",
    "recommend", "suggested", "suggest", "claimed", "claim", "stated",
    "state", "replied", "reply", "messaged", "raised",
}
_REPORTING_PHRASES = {"talked about", "talk about", "brought up"}
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
# A "?" only counts as question shape when it terminates the query; a quoted
# question inside a statement ("he said 'are you sure?' and left") does not.
_QUESTION_RE = re.compile(
    r"\b(?:what|when|where|who|which|why|how)\b"
    r"|\b(?:did|does|do|is|are|was|were|has|have|had)\s+\w+"
    r"|\?\s*$",
    re.IGNORECASE,
)

# Case-sensitive on purpose: searched against the ORIGINAL query (not the
# lowercased ``q``) because capitalisation is the only entity signal here.
_ENTITY_PAIR_RE = re.compile(
    r"\b([A-Z][a-z]+)\s+(and|vs|versus|then|compared to)\s+([A-Z][a-z]+)\b"
)

# "about X" / "regarding X" topic phrase — the strongest single multi-hop
# marker ("what did she say about the move").  The target word must be a real
# topic: adverbs/pronouns ("about earlier", "about that") are not topics.
_TOPIC_RE = re.compile(
    r"\b(?:about|regarding)\s+(?:(?:the|my|our|your|this|that|a|an)\s+)?"
    r"(?!(?:earlier|later|before|after|today|yesterday|tomorrow|tonight|now|"
    r"then|here|there|it|them|him|her|us|me|you|this|that|these|those|"
    r"what|when|where|why|how|something|someone|somehow|anything|anyone|"
    r"nothing)\b)[a-z0-9'-]{2,}",
    re.I,
)

_WORDS_RE = re.compile(r"[a-z0-9']+", re.I)


def _phrase_re(items) -> "re.Pattern[str]":
    """Word-boundary alternation over ``items`` (longest first)."""
    alts = "|".join(re.escape(i) for i in sorted(items, key=len, reverse=True))
    return re.compile(r"\b(?:" + alts + r")\b", re.I)


_RELATIVE_TIME_RE = _phrase_re(_RELATIVE_TIME)
_REPORTING_PHRASES_RE = _phrase_re(_REPORTING_PHRASES)

# English sentence-starters / function words that get capitalized at
# sentence start but are NOT proper nouns.  Excluded from entity detection.
_NON_ENTITY_CAPS = {
    "the", "this", "that", "these", "those", "there", "what", "when",
    "where", "why", "which", "who", "whose", "how", "did", "does", "do",
    "is", "are", "was", "were", "have", "has", "had", "can", "could",
    "would", "should", "will", "shall", "may", "might", "must", "i", "im",
    "ive", "id", "ill", "you", "your", "yours", "my", "mine", "our", "ours",
    "we", "they", "she", "he", "it", "its", "a", "an", "and", "but", "so",
    "or", "if", "of", "for", "to", "in", "on", "at", "not", "no", "yes",
    "ok", "okay", "hey", "hi", "hello", "then", "now", "also", "just",
    "well", "right", "wait", "sure", "yeah", "ya", "yep", "please", "thanks",
    "thank", "actually", "honestly", "literally", "basically", "really",
    "though", "though", "although", "because", "while", "after", "before",
    "during", "with", "without", "from", "by", "via", "etc", "e", "g",
    "vs",
    # Imperative sentence-starters (capitalised only by position).
    "tell", "remind", "remember", "show", "give", "let", "find", "check",
    "look", "help", "explain", "list", "summarize", "summarise", "compare",
}
# Months/weekdays are calendar words, not entities.
_NON_ENTITY_CAPS |= _MONTHS | _WEEKDAYS


def _words(text: str):
    return _WORDS_RE.findall(text.lower())


def _proper_nouns(query: str) -> int:
    """Count capitalized words that are plausibly named entities.

    Excludes a stopword set of common capitalized function words and
    imperative sentence-starters, so a sentence-initial "Alex" still counts
    while "What"/"Tell" do not.  Case-sensitive on the ORIGINAL query —
    "alex" lowercase is not an entity, "Alex" is.
    """
    tokens = re.findall(r"[A-Za-z][A-Za-z']*", query)
    count = 0
    for tok in tokens:
        low = tok.lower()
        if (
            len(tok) >= 3
            and tok[0].isupper()
            and tok[1:].islower()
            and low not in _NON_ENTITY_CAPS
        ):
            count += 1
    return count


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def temporal_score(query: str) -> float:
    """Additive temporal evidence weight, clamped to [0, 1].

    Not a probability: each matched signal adds its fixed weight and the raw
    sum may exceed 1.0 before clamping (see module docstring).

    v3: single weak calendar/time words (weeks, months, relative adverbs)
    are deliberately below the 0.50 threshold on their own.  Genuine anchors
    (explicit dates, years, time-interrogatives, order questions, month
    intervals) still clear by wide margins.
    """
    q = " ".join(query.lower().split())
    if not q:
        return 0.0
    score = 0.0
    words = set(_words(q))

    # Calendar words: weak alone, additive with each other.
    month_hits = words & _MONTHS
    if month_hits:
        score += 0.25 + 0.10 * min(len(month_hits) - 1, 2)
    if _WEEKDAYS & words:
        score += 0.20

    # Chronological ordering ("what came first", "happened before/after").
    if _ORDER_RE.search(q):
        score += 0.50
    # Relative-time adverbs: weak on purpose.
    if _RELATIVE_TIME_RE.search(q):
        score += 0.25
    # Explicit durations, dated formats, years.
    if _DURATION_RE.search(q):
        score += 0.45
    if _DATE_RE.search(q):
        score += 0.60
    if _YEAR_RE.search(q):
        score += 0.55
    # Time-interrogative structures (incl. month intervals, "what happened").
    if _WHEN_RE.search(q):
        score += 0.50
    # Past-tense fact probes ("what did I…", "when was my…") — weak but real.
    if _PAST_TENSE_FACT_RE.search(q):
        score += 0.30
    return min(1.0, score)


def multi_hop_score(query: str) -> float:
    """Additive multi-hop evidence weight, clamped to [0, 1].

    Not a probability: each matched signal adds its fixed weight and the raw
    sum may exceed 1.0 before clamping (see module docstring).

    v3: a bare reporting verb scores 0.25 — below the 0.45 threshold.  It
    routes only when paired with a proper-noun entity (+0.30), an "about X"
    topic phrase (+0.35), or a chain/comparison signal.  This kills the
    casual-chat false positives ("what did you suggest", "she said...")
    while keeping the designed class ("what did alex say about the move").
    """
    q = " ".join(query.lower().split())
    if not q:
        return 0.0
    score = 0.0
    words = set(_words(q))
    reporting_hits = (words & _REPORTING_VERBS) | set(
        m.lower() for m in _REPORTING_PHRASES_RE.findall(q)
    )
    comparison_hits = words & _COMPARISON_VERBS
    chain_hits = words & _CHAIN_TERMS

    if reporting_hits:
        score += 0.25 * min(len(reporting_hits), 2)
    if comparison_hits:
        score += 0.25 * min(len(comparison_hits), 2)
    if chain_hits:
        score += 0.20 * min(len(chain_hits), 2)
    if _YOU_SAID_RE.search(q):
        score += 0.30
    # Proper-noun entity — only pays when >=1 verb class hit already present.
    if (reporting_hits or comparison_hits or chain_hits) and _proper_nouns(query):
        score += 0.30
    # "about X / regarding X" topic phrase — the strongest multi-hop marker.
    if (reporting_hits or comparison_hits or chain_hits) and _TOPIC_RE.search(q):
        score += 0.35
    # Entity-pair ("X and Y") — only pays when >=1 verb class hit present
    # (prevents "X and Y products" false positives).  Searches the original
    # ``query`` (not ``q``) because the pattern relies on capitalisation.
    if _ENTITY_PAIR_RE.search(query) and (reporting_hits or comparison_hits or chain_hits):
        score += 0.30
    return min(1.0, score)


def is_temporal_or_multihop(query: str) -> bool:
    """Return True when the query reads as temporal and/or multi-hop.

    Only genuine questions route (interrogative word, auxiliary-led
    question, or "?"); commands and statements never pay the smart-model
    cost.
    """
    if not query or not query.strip():
        return False
    if not _QUESTION_RE.search(query):
        return False
    return (
        temporal_score(query) >= ROUTE_TEMPORAL_THRESHOLD
        or multi_hop_score(query) >= ROUTE_MULTI_HOP_THRESHOLD
    )


# ---------------------------------------------------------------------------
# Historical-time detection (#3: history at current time)
# ---------------------------------------------------------------------------

_HISTORY_PAST_RE = re.compile(
    r"\b(use[d]?\s+to|used\s+to\s+(?:live|work|stay|drive|play)|"
    r"previously|formerly|before\s+(?:the\s+)?(?:move|relocation)|"
    r"(?:old|previous|former)\s+(?:address|house|flat|apartment|job|office|number)|"
    r"where\s+(?:did|does)\s+\w+\s+(?:use|used)\s+to|"
    r"what\s+(?:did|was)\s+\w+\s+(?:using|driving|playing|watching|reading)\s+before)",
    re.IGNORECASE,
)
_HISTORY_TIME_RE = re.compile(
    r"\bin\s+the\s+past\b|\bback\s+then\b|\b(at|in)\s+that\s+time\b|"
    r"\bbefore\s+(?:that|this|the\s+update|the\s+change|we\s+moved)\b",
    re.IGNORECASE,
)


def is_historical_query(query: str) -> bool:
    """True when the query asks about a SUPERSEDED state of the world.

    Precision-first (v3 philosophy): a match requires BOTH an explicit
    past-state marker ("used to", "previously", "old address", "back
    then") AND question shape, so ordinary current-state questions never
    pay the widened-search cost.  This gates the include-closed-versions
    retrieval path in ``_search_memories``.

    Deliberately does NOT reuse ``_PAST_TENSE_FACT_RE`` from
    ``temporal_score``: that probe fires on any past-tense fact question
    ("what did I eat yesterday") whose answer is still a CURRENT fact, so
    sharing it would widen retrieval to closed versions for ordinary
    questions.  The two detectors overlap on "used to" style phrasing only
    because both happen to describe the past; their purposes differ.
    """
    if not query or not query.strip():
        return False
    if not _QUESTION_RE.search(query):
        return False
    return bool(_HISTORY_PAST_RE.search(query) or _HISTORY_TIME_RE.search(query))


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


# Config thresholds are clamped to this range: 0.0 would route every
# question, >1.0 could never be reached (scores are clamped to 1.0).
THRESHOLD_MIN = 0.1
THRESHOLD_MAX = 1.0


def _threshold(value: Any, default: float) -> float:
    parsed = _as_float(value, default)
    if parsed != parsed:  # NaN
        parsed = default
    return max(THRESHOLD_MIN, min(THRESHOLD_MAX, parsed))


# Count of swallowed exceptions in ``route_answerer`` (IR7): routing bugs
# fall back to the default model silently, so expose them for monitoring.
ROUTING_FAILURES = 0


def routing_failure_count() -> int:
    return ROUTING_FAILURES


def route_answerer(config: Dict[str, Any], user_message: str) -> Optional[Dict[str, str]]:
    """Return ``{"model": ..., "provider": ...}`` (provider optional) or None.

    Called from ``_on_pre_llm_call``.  Returns the smart model ONLY for
    genuine temporal/multi-hop queries above threshold; returns None for
    everything else (casual chat, statements, below-threshold questions).

    Regression fix 2026-08-23: older versions returned the DEFAULT model
    for every non-smart message ("so turns self-correct back").  The core
    honors any returned model via agent.switch_model(), so that made the
    pre_llm_call hook force-switch EVERY chat/session to
    router_default_model on every turn — stomping the user's explicitly
    selected per-session model in the desktop UI (observed:
    ox-alpha-free -> deepseek-v4-flash) and changing which model actually
    answered the turn.  The hook must never override a session's own
    model choice unless a genuine smart route fired.
    """
    try:
        if not _as_bool(config.get("router_enabled"), default=False):
            return None
        smart = str(config.get("router_smart_model") or "").strip()
        if not smart:
            logger.debug("router: enabled but missing smart model config")
            return None
        msg = user_message or ""
        if not _QUESTION_RE.search(msg):
            return None
        temporal = temporal_score(msg)
        multi_hop = multi_hop_score(msg)
        t_thresh = _threshold(config.get("router_temporal_threshold"), ROUTE_TEMPORAL_THRESHOLD)
        m_thresh = _threshold(config.get("router_multihop_threshold"), ROUTE_MULTI_HOP_THRESHOLD)
        if temporal >= t_thresh or multi_hop >= m_thresh:
            pick = smart
            provider = str(config.get("router_smart_provider") or "").strip()
        else:
            return None
        # #275 LP2: increment the router_calls counter.
        try:
            try:
                from .liveness import increment_counter
            except ImportError:
                from liveness import increment_counter
            increment_counter("router_calls")
        except Exception:
            pass
        result: Dict[str, str] = {"model": pick}
        if provider:
            result["provider"] = provider
        return result
    except Exception as exc:  # pragma: no cover - defensive
        global ROUTING_FAILURES
        ROUTING_FAILURES += 1
        logger.warning("router: failed to route (failure #%d): %s", ROUTING_FAILURES, exc)
        return None
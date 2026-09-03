"""Deterministic value extractor for write-time value-supersession.

Extracts ``(subject, numeric_value, unit)`` triples from memory content using
regex only — zero LLM calls.  Used by the store to detect when a new fact
carries a different numeric value for the same subject as an existing active
fact (stale-number detection).

Supported value patterns:
- Percentages: "89.8%", "82.2 percent"
- Currency: "R$ 449", "$1,200", "€500"
- Counts: "449 rows", "1,200 items"
- Ratios: "449/500", "27/30"
- Years: "2026", "born in 1990"
- Ages: "age 35", "35 years old"

The "subject" is the text surrounding the value — we extract a window of
~6 tokens on either side, which the store uses for token-overlap matching
against existing records.

Design: Argos issue #4.  Regex/deterministic only.  No LLM.  Proposal-first.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class ExtractedValue:
    """A single (subject, value, unit) triple extracted from text."""
    subject: str       # token window around the value
    value: str         # normalized value string (e.g. "89.8", "449", "2026")
    unit: str          # "percent", "currency:R$", "count", "ratio", "year", "age"
    raw: str           # the raw matched text (e.g. "89.8%", "R$ 449", "449/500")


# --- regex patterns --------------------------------------------------------

# Percentages: "89.8%", "82.2 percent", "90 per cent"
# Note: no \b after % — % is non-word so \b fails at end-of-string.
_PERCENT_RE = re.compile(
    r"(\d+\.?\d*)\s*(?:%|percent\b|per\s*cent\b)",
    re.IGNORECASE,
)

# Currency: "R$ 449", "$1,200.50", "€500", "£100", "¥1000"
# Captures the symbol + amount.  Currency symbols: $, R$, €, £, ¥, ₹
_CURRENCY_RE = re.compile(
    r"(R\$|US\$|\$|€|£|¥|₹)\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)",
)

# Ratios: "449/500", "27/30"
_RATIO_RE = re.compile(
    r"\b(\d+)/(\d+)\b",
)

# Counts: "449 rows", "1,200 items", "3 meetings"
# Must have a noun after the number to distinguish from bare numbers.
_COUNT_RE = re.compile(
    r"\b(\d{1,3}(?:,\d{3})*|\d+)\s+([a-z][a-z\s]{0,20}?)\b",
    re.IGNORECASE,
)

# Years: "2026", "in 1990" — 4-digit numbers in 1900-2100 range
_YEAR_RE = re.compile(
    r"\b(19\d{2}|20\d{2})\b",
)

# Ages: "age 35", "35 years old", "is 35"
_AGE_RE = re.compile(
    r"(?:age\s+|is\s+|aged\s+)(\d{1,3})\b|\b(\d{1,3})\s+years?\s+old\b",
    re.IGNORECASE,
)


def _token_window(text: str, start: int, end: int, tokens_each_side: int = 6) -> str:
    """Extract a token window around a match for subject matching.

    Takes ~tokens_each_side tokens on either side of the [start, end) span,
    lowercased, to form the subject phrase used for overlap matching.
    """
    # Split the full text into tokens (word-like chunks)
    before = text[:start]
    after = text[end:]
    before_tokens = re.findall(r"\w+", before)
    after_tokens = re.findall(r"\w+", after)
    window = (
        before_tokens[-tokens_each_side:]
        + re.findall(r"\w+", text[start:end])
        + after_tokens[:tokens_each_side]
    )
    return " ".join(t.lower() for t in window)


def extract_values(text: str) -> List[ExtractedValue]:
    """Extract all (subject, value, unit) triples from *text*.

    Returns a list of ExtractedValue, one per match.  Empty list if no
    numeric values are found.  The order is: percentages, currencies,
    ratios, counts, years, ages — most specific first to avoid
    double-counting (a percentage is also a count, but we want the
    percentage interpretation).
    """
    if not text or not text.strip():
        return []

    results: List[ExtractedValue] = []
    consumed_spans: List[tuple[int, int]] = []

    def _overlaps(start: int, end: int) -> bool:
        for cs, ce in consumed_spans:
            if start < ce and end > cs:
                return True
        return False

    def _add(start: int, end: int, value: str, unit: str, raw: str) -> None:
        if _overlaps(start, end):
            return
        subject = _token_window(text, start, end)
        results.append(ExtractedValue(
            subject=subject, value=value, unit=unit, raw=raw,
        ))
        consumed_spans.append((start, end))

    # 1. Percentages
    for m in _PERCENT_RE.finditer(text):
        val = m.group(1)
        _add(m.start(), m.end(), val, "percent", m.group(0))

    # 2. Currency
    for m in _CURRENCY_RE.finditer(text):
        symbol = m.group(1)
        amount = m.group(2).replace(",", "")
        _add(m.start(), m.end(), amount, f"currency:{symbol}", m.group(0))

    # 3. Ratios
    for m in _RATIO_RE.finditer(text):
        val = f"{m.group(1)}/{m.group(2)}"
        _add(m.start(), m.end(), val, "ratio", m.group(0))

    # 4. Ages (before counts — "35 years old" should be age, not count)
    for m in _AGE_RE.finditer(text):
        val = m.group(1) or m.group(2)
        if val:
            _add(m.start(), m.end(), val, "age", m.group(0))

    # 5. Counts (number + noun)
    for m in _COUNT_RE.finditer(text):
        num = m.group(1).replace(",", "")
        _add(m.start(), m.start() + len(m.group(1)), num, "count", m.group(1))

    # 6. Years (4-digit 1900-2100)
    for m in _YEAR_RE.finditer(text):
        _add(m.start(), m.end(), m.group(1), "year", m.group(0))

    return results


def subject_overlap(s1: str, s2: str, threshold: float = 0.5) -> bool:
    """Check if two subject phrases overlap enough to be "the same subject".

    Uses token-Jaccard similarity: |intersection| / |union| >= threshold.
    Default 0.5 means at least half the tokens are shared.
    """
    if not s1 or not s2:
        return False
    t1 = set(s1.lower().split())
    t2 = set(s2.lower().split())
    if not t1 or not t2:
        return False
    intersection = t1 & t2
    union = t1 | t2
    return len(intersection) / len(union) >= threshold


def values_conflict(
    new_values: List[ExtractedValue],
    old_values: List[ExtractedValue],
    subject_threshold: float = 0.5,
) -> Optional[tuple[ExtractedValue, ExtractedValue]]:
    """Check if any new value conflicts with an old value for the same subject.

    A conflict means: same subject (token-overlap >= threshold), same unit
    (normalised), AND different value.  Same subject + same value + same
    unit = idempotent (no conflict).  Different unit = incomparable
    quantities ("3.5 percent" vs "3.5 years") — never a supersession
    candidate.  Different subject = no conflict.

    Returns the first conflicting (new, old) pair, or None if no conflict.
    """
    for new_v in new_values:
        for old_v in old_values:
            if not subject_overlap(new_v.subject, old_v.subject, subject_threshold):
                continue
            # Units are normalised (stripped + lower-cased) so "Percent" /
            # "percent" / None don't masquerade as a unit change.
            new_unit = (new_v.unit or "").strip().lower()
            old_unit = (old_v.unit or "").strip().lower()
            if new_unit != old_unit:
                # Different dimensions are incomparable quantities — not a
                # supersession candidate (false-supersession class, #91).
                continue
            if new_v.value == old_v.value:
                continue  # same value + unit — idempotent
            # Same subject, same unit, different value — conflict
            return (new_v, old_v)
    return None


# --- transition-verb gate (#36) -------------------------------------------
# Only transition statements should close a standing fact. A plain restatement
# ("I use X") must not invalidate a legacy value — the user may hold several
# things at once. The transition gate checks whether the new content carries
# an explicit transition signal before treating a value conflict as a true
# supersession candidate.

# Transition verbs and phrases that signal a change of state.
# Matched case-insensitively on the full content text.
_TRANSITION_VERBS = re.compile(
    r"\b(?:"
    r"switched\s+to|switch\s+to|"
    r"changed\s+to|change\s+to|"
    r"moved\s+to|move\s+to|"
    r"replaced|replace(?:d)?\s+(?:with|by|to)?|"
    r"stopped\s+(?:using|doing|taking|going|working)|"
    r"started\s+(?:using|doing|taking|going|working)|"
    r"now\s+(?:use|uses|using|live|lives|living|work|works|working|drive|drives|take|takes|is|are|was|equals|equal)|"
    r"(?:is|are)\s+now\b|"
    r"now\s+is\b|"
    r"no\s+longer\s+(?:use|uses|using|live|lives|work|works|drive|drives|take|takes|is|are)|"
    r"used\s+to\s+(?:use|live|work|drive|take)|"
    r"upgraded\s+to|upgrade\s+to|"
    r"downgraded\s+to|downgrade\s+to|"
    r"switched\s+from|switch\s+from|"
    r"changed\s+from|change\s+from|"
    r"moved\s+from|move\s+from"
    r")\b",
    re.IGNORECASE,
)

# Negation of transition — "didn't switch", "hasn't changed", "still uses".
# If the content negates a transition, it's a corroboration, not a change.
_TRANSITION_NEGATION = re.compile(
    r"\b(?:"
    r"didn['\u2019]?t\s+(?:switch|change|move|replace|stop|start)|"
    r"hasn['\u2019]?t\s+(?:switched|changed|moved|replaced|stopped|started)|"
    r"haven['\u2019]?t\s+(?:switched|changed|moved|replaced|stopped|started)|"
    r"don['\u2019]?t\s+(?:switch|change|move|replace|stop|start)|"
    r"doesn['\u2019]?t\s+(?:switch|change|move|replace|stop|start)|"
    r"still\s+(?:use|uses|using|live|lives|work|works|drive|drives|take|takes|have|has)|"
    r"never\s+(?:switched|changed|moved|replaced|stopped|started)"
    r")\b",
    re.IGNORECASE,
)


def is_transition_statement(text: str) -> bool:
    """Check if *text* carries an explicit transition signal (#36).

    A transition statement is one that signals a change of state —
    "switched to", "changed to", "stopped using", "now uses", etc.
    A plain restatement ("I use 449 rows") is NOT a transition.

    Negated transitions ("didn't switch", "still uses") are treated as
    corroboration, not transition.

    Returns True if the text contains a non-negated transition verb.
    """
    if not text or not text.strip():
        return False
    # If the text negates a transition, it's not a transition.
    if _TRANSITION_NEGATION.search(text):
        return False
    return bool(_TRANSITION_VERBS.search(text))

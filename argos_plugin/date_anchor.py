"""Date-anchored re-rank helper for hybrid memory injection.

P2B2: when a temporal turn carries an explicit date expression ("10 days ago",
"last Tuesday", "on March 2nd", "Valentine's day"), resolve it to a concrete
date and re-sort the injected top-k by proximity to that date, so the model
reads the correct time window first.  Zero-LLM (regex + datetime), best-effort;
any failure returns None and the caller keeps relevance order.
"""

from __future__ import annotations

import datetime as _dt
import re

_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11,
    "december": 12,
}
_WEEKDAYS = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}
_WORD_NUMS = {
    "a": 1, "an": 1, "one": 1, "couple": 2, "few": 3, "two": 2, "three": 3,
    "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "fortnight": 14,
}
_UNIT_MULT = {"day": 1, "week": 7, "month": 30, "year": 365}

_RE_AGO = re.compile(
    r"\b(?:"
    r"(?P<n>\d+(?:\.\d+)?|[a-z]+(?:-[a-z]+)?|a|an|the)"  # number or word
    r"\s*[- ]?\s*)"
    r"(?P<unit>day|days|week|weeks|month|months|year|years)\s+ago\b",
    re.IGNORECASE,
)
_RE_LAST_WD = re.compile(r"\blast\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", re.IGNORECASE)
_RE_LAST_PERIOD = re.compile(r"\blast\s+(week|month|year)\b", re.IGNORECASE)
_RE_WD_AGO = re.compile(
    r"\bthe\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\s+"
    r"(\d+|[a-z]+(?:-[a-z]+)?|a|an)\s*(week|weeks|month|months)\s+ago\b",
    re.IGNORECASE,
)
_RE_ON_MONTH_DAY = re.compile(
    r"\bon\s+(" + "|".join(_MONTHS) + r")\s+(\d{1,2})(?:st|nd|rd|th)?\b", re.IGNORECASE
)
_RE_FIXED = re.compile(r"\bvalentine'?s\s+day\b", re.IGNORECASE)
_RE_YESTERDAY = re.compile(r"\b(the\s+day\s+before\s+yesterday|yesterday)\b", re.IGNORECASE)
_RE_PAST_WEEKEND = re.compile(r"\b(the\s+)?(past|previous|last)\s+weekend\b", re.IGNORECASE)


def _to_int(tok: str) -> int | None:
    tok = tok.strip().lower().replace("-", " ")
    if tok.isdigit():
        return int(tok)
    if tok in _WORD_NUMS:
        return _WORD_NUMS[tok]
    return None


def _year_adjust(month: int, day: int, today: _dt.date) -> _dt.date:
    try:
        d = _dt.date(today.year, month, day)
    except ValueError:
        return None
    if d > today:  # "on March 2nd" asked in May -> March of this year (past)
        d = _dt.date(today.year - 1, month, day)
    return d


def resolve_target_date(query: str, today: _dt.date | None = None):
    """Resolve a date expression in ``query`` to a concrete date (or None).

    Returns (target_date, expression_label).  ``today`` defaults to
    ``datetime.date.today()``.
    """
    if not query or not query.strip():
        return None, None
    today = today or _dt.date.today()
    q = query.strip().lower()

    m = _RE_ON_MONTH_DAY.search(q)
    if m:
        d = _year_adjust(_MONTHS[m.group(1)], int(m.group(2)), today)
        return (d, f"on {m.group(1)} {m.group(2)}") if d else (None, None)

    if _RE_FIXED.search(q):
        return (_year_adjust(2, 14, today), "valentine's day")

    m = _RE_YESTERDAY.search(q)
    if m:
        days = 2 if m.group(1).startswith("the day before") else 1
        return (today - _dt.timedelta(days=days), m.group(1))

    if _RE_PAST_WEEKEND.search(q):
        # most recent Sunday strictly before today
        back = (today.weekday() + 1) % 7
        return (today - _dt.timedelta(days=back), "past weekend")

    m = _RE_WD_AGO.search(q)
    if m:
        wd, num, unit = m.group(1), m.group(2), m.group(3)
        n = _to_int(num)
        if n:
            base = today - _dt.timedelta(days=n * _UNIT_MULT[unit.rstrip("s")])
            delta = (base.weekday() - _WEEKDAYS[wd.lower()]) % 7
            return (base - _dt.timedelta(days=delta), f"the {wd} {n} {unit} ago")

    m = _RE_AGO.search(q)
    if m:
        n = _to_int(m.group("n"))
        unit = m.group("unit").lower().rstrip("s")
        if n and unit in _UNIT_MULT and n <= 366 * 3:
            return (today - _dt.timedelta(days=n * _UNIT_MULT[unit]), f"{n} {unit}(s) ago")

    m = _RE_LAST_WD.search(q)
    if m:
        wd = _WEEKDAYS[m.group(1).lower()]
        delta = (today.weekday() - wd) % 7
        if delta == 0:
            delta = 7
        return (today - _dt.timedelta(days=delta), f"last {m.group(1)}")

    m = _RE_LAST_PERIOD.search(q)
    if m:
        unit = m.group(1).lower()
        return (today - _dt.timedelta(days=_UNIT_MULT[unit]), f"last {unit}")

    return None, None


def reorder_by_date(records, query: str, today: _dt.date | None = None):
    """Return (reordered_records, target, label).  Stable fallback to input.

    Undated records sink to the end; on any failure the input list is
    returned unchanged.
    """
    try:
        target, label = resolve_target_date(query, today)
        if not target:
            return records, None, None
        dated, undated = [], []
        for r in records:
            ts = (getattr(r, "created_at", None) or "")[:10]
            try:
                d = _dt.date.fromisoformat(ts)
            except (ValueError, TypeError):
                undated.append(r)
                continue
            dated.append((abs((d - target).days), r))
        dated.sort(key=lambda pair: (pair[0],))
        return [r for _, r in dated] + undated, target, label
    except Exception:
        return records, None, None
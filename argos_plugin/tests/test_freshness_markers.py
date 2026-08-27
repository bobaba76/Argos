"""Freshness-marker tests (Tier-2 anti-staleness, 2026-08-27).

A recalled memory whose CONTENT carries an explicit date anchor
("26/8", "PR #96224", "August 27, 2026", "last week") gets a compact
as-of marker built from the record's own update timestamp, so a stale
anchor is never read as current state. Append-only, zero-LLM, no
ranking/retrieval change.
"""

import sys
from pathlib import Path

_plugin_dir = Path(__file__).resolve().parent.parent
if str(_plugin_dir) not in sys.path:
    sys.path.insert(0, str(_plugin_dir))
if str(_plugin_dir.parent) not in sys.path:
    sys.path.insert(0, str(_plugin_dir.parent))

try:
    from argos_plugin import _freshness_marker_for, _DATE_ANCHOR_RE
except ImportError:
    from argos import _freshness_marker_for, _DATE_ANCHOR_RE


def _mark(content: str, as_of: str = "2026-08-26T14:00:00+00:00") -> str:
    return _freshness_marker_for(content, as_of)


def test_explicit_short_date():
    # '26/8' — the exact failure class from the PR-timeline mixup.
    assert _mark("handoff was on 26/8 and shipped") == "\u27e8as of 2026-08-26\u27e9"


def test_iso_date():
    assert _mark("commit dated 2026-08-27 landed") == "\u27e8as of 2026-08-26\u27e9"


def test_month_name_year():
    assert _mark("we shipped this on August 27, 2026") == "\u27e8as of 2026-08-26\u27e9"


def test_pr_reference():
    # Concrete anchor from the real failure: PR numbers.
    assert _mark("PR #96224 was a commit-author rewrite") == "\u27e8as of 2026-08-26\u27e9"


def test_relative_anchor():
    assert _mark("we shipped the fix last week") == "\u27e8as of 2026-08-26\u27e9"
    assert _mark("that was 3 days ago") == "\u27e8as of 2026-08-26\u27e9"
    assert _mark("this changed today") == "\u27e8as of 2026-08-26\u27e9"


def test_benign_content_no_marker():
    assert _mark("The user prefers one clean direct answer over lectures.") == ""


def test_anchor_without_asof_no_marker():
    assert _freshness_marker_for("shipped on 26/8", "") == ""
    assert _freshness_marker_for("", "2026-08-26") == ""


def test_marker_truncates_to_date():
    assert _mark("PR #5 merged", as_of="2026-08-26T23:59:59+00:00") == "\u27e8as of 2026-08-26\u27e9"


def test_common_words_not_anchors():
    # No false positives on ordinary prose.
    assert _mark("May I ask a question about the store?") == ""
    assert _mark("The March of events continued") == ""
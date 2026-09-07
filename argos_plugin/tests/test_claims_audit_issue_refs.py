"""Issue-ref canary for CLAIMS-AUDIT.md (#354).

Issue #74 closed when ``store.ingest_versioned()`` landed, but the audit kept
quoting it as an open gap ("filed as #74") with no resolution entry — the same
doc-lags-HEAD class as the test-count staleness (#325/#350). These checks pin
the audit's #74 text to the tree:

* the audit must carry a #74 resolution entry naming ``ingest_versioned``;
* the "proof-at-scale not banked" caveat must stay true — if an in-tree
  harness/adapter starts calling ``ingest_versioned``, the caveat must be
  revisited in the same pass.
"""
from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_AUDIT_DOC = _REPO_ROOT / "CLAIMS-AUDIT.md"
_STORE_WRITE = _REPO_ROOT / "argos_plugin" / "store_write.py"

# One §5 history entry: header bullet through the next blank line.
_RESOLUTION_RE = re.compile(
    r"^- \*\*\d{4}-\d{2}-\d{2}[^*\n]*#74 resolution[^*\n]*\*\*(?:.(?!\n\n))*",
    re.MULTILINE | re.DOTALL,
)
_NOT_BANKED_RE = re.compile(r"\*\*Proof-at-scale is NOT\s+yet banked\*\*")

# Directories that would hold an in-tree benchmark ingest path.
_HARNESS_DIRS = (
    _REPO_ROOT / "eval",
    _REPO_ROOT / "scripts",
    _REPO_ROOT / "argos_plugin" / "eval",
)


def _audit_text() -> str:
    return _AUDIT_DOC.read_text(encoding="utf-8")


def _in_tree_harness_callers() -> list[Path]:
    callers: list[Path] = []
    for root in _HARNESS_DIRS:
        if not root.is_dir():
            continue
        for path in root.rglob("*.py"):
            if "tests" in path.parts:
                continue
            if "ingest_versioned(" in path.read_text(encoding="utf-8", errors="replace"):
                callers.append(path)
    return callers


def test_ingest_versioned_exists_in_store():
    assert "def ingest_versioned(" in _STORE_WRITE.read_text(encoding="utf-8"), (
        "CLAIMS-AUDIT's #74 resolution entry cites store.ingest_versioned(); it is gone"
    )


def test_audit_has_74_resolution_entry():
    text = _audit_text()
    match = _RESOLUTION_RE.search(text)
    assert match, (
        "CLAIMS-AUDIT.md §5 has no #74 resolution entry (the #23/#24 pattern: the filing "
        "entry stays, a later entry records the resolution)"
    )
    entry = match.group(0)
    assert "is **CLOSED**" in entry
    assert "ingest_versioned" in entry


def test_audit_74_ref_in_section3_points_at_resolution():
    text = _audit_text()
    assert "(filed as #74)." not in text, (
        "CLAIMS-AUDIT.md still quotes '#74' as an open filing with no pointer to its resolution"
    )
    assert "filed as #74; #74 is now **closed**" in text


def test_not_banked_caveat_matches_tree():
    """The audit says no in-tree harness calls ingest_versioned. Keep that honest."""
    caveat_present = bool(_NOT_BANKED_RE.search(_audit_text()))
    callers = _in_tree_harness_callers()
    if callers:
        assert not caveat_present, (
            "An in-tree harness now calls ingest_versioned() "
            f"({', '.join(str(p.relative_to(_REPO_ROOT)) for p in callers)}); "
            "revisit the CLAIMS-AUDIT #74 'Proof-at-scale is NOT yet banked' caveat and bank "
            "the chained-run artifacts (or drop the caveat) in the same pass."
        )
    else:
        assert caveat_present, (
            "No in-tree harness calls ingest_versioned(), yet CLAIMS-AUDIT.md no longer "
            "carries the #74 'Proof-at-scale is NOT yet banked' caveat"
        )

"""Parity canary: the CLAIMS-AUDIT.md §2 "Test suite" row and the README
Verification "Test suite" bullet must track the suite.

The audit doc's premise is "nothing quoted without evidence". Issue #325
found the row ~2x stale (59 modules / 1,169 tests quoted vs 136 / 2,490 on
disk). This test parses the quoted counts (in both docs) and compares them against
``scripts/count_test_fns.py`` so the rows cannot silently rot again. A
tolerance is allowed so ordinary test additions don't fail the suite;
refresh the rows (and their changelog/Verification entries) when the drift exceeds it.
"""
from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_AUDIT_DOC = _REPO_ROOT / "CLAIMS-AUDIT.md"
_README_DOC = _REPO_ROOT / "README.md"
_COUNT_SCRIPT = _REPO_ROOT / "scripts" / "count_test_fns.py"

# Drift allowed before the audit row must be refreshed.
_TOLERANCE = 0.05

_ROW_RE = re.compile(
    r"^\|\s*Test suite\s*\|.*?\|\s*(?P<modules>\d+)\s+test modules\b.*?"
    r"\((?P<tests>[\d,]+)\s+`def test_`",
    re.MULTILINE,
)

_README_RE = re.compile(
    r"^- \*\*Test suite\*\* —\s*(?P<tests>[\d,]+)\s+test functions across\s+"
    r"(?P<modules>\d+)\s+test modules\s*\(as of",
    re.MULTILINE,
)


def _load_counter():
    spec = importlib.util.spec_from_file_location("count_test_fns", _COUNT_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _quoted_counts() -> dict[str, int]:
    text = _AUDIT_DOC.read_text(encoding="utf-8")
    match = _ROW_RE.search(text)
    assert match, "CLAIMS-AUDIT.md §2 'Test suite' row not found or not in the expected shape"
    return {
        "modules": int(match["modules"]),
        "tests": int(match["tests"].replace(",", "")),
    }


def _quoted_readme_counts() -> dict[str, int]:
    text = _README_DOC.read_text(encoding="utf-8")
    match = _README_RE.search(text)
    assert match, "README Verification 'Test suite' bullet not found or not in the expected shape"
    return {
        "modules": int(match["modules"]),
        "tests": int(match["tests"].replace(",", "")),
    }


def test_count_script_counts_this_suite():
    counts = _load_counter().count_suite()
    assert counts["modules"] >= 1
    assert counts["tests"] >= counts["modules"]


def test_count_script_tolerates_bom(tmp_path):
    counter = _load_counter()
    (tmp_path / "test_bom.py").write_text(
        "\ufeffdef test_a():\n    pass\n\nasync def test_b():\n    pass\n",
        encoding="utf-8",
    )
    (tmp_path / "test_plain.py").write_text(
        "def test_c():\n    pass\n\ndef helper():\n    pass\n", encoding="utf-8"
    )
    (tmp_path / "conftest.py").write_text("def test_ignored():\n    pass\n", encoding="utf-8")
    assert counter.count_suite(tmp_path) == {"modules": 2, "tests": 3}


@pytest.mark.parametrize("key", ["modules", "tests"])
def test_claims_audit_test_suite_row_within_tolerance(key):
    actual = _load_counter().count_suite()[key]
    quoted = _quoted_counts()[key]
    drift = abs(actual - quoted) / actual
    assert drift <= _TOLERANCE, (
        f"CLAIMS-AUDIT.md §2 'Test suite' row quotes {quoted} {key} but the suite has "
        f"{actual} ({drift:.1%} drift > {_TOLERANCE:.0%}). Refresh the row and add a "
        f"changelog entry; regenerate counts with `python scripts/count_test_fns.py`."
    )


@pytest.mark.parametrize("key", ["modules", "tests"])
def test_readme_test_suite_row_within_tolerance(key):
    actual = _load_counter().count_suite()[key]
    quoted = _quoted_readme_counts()[key]
    drift = abs(actual - quoted) / actual
    assert drift <= _TOLERANCE, (
        f"README Verification 'Test suite' bullet quotes {quoted} {key} but the suite has "
        f"{actual} ({drift:.1%} drift > {_TOLERANCE:.0%}). Sync the README bullet with the "
        f"CLAIMS-AUDIT §2 row (and its date) when refreshing."
    )

"""#292: Tier 1 — curated retrieval slice + CI wiring tests.

Covers:
- the tiny synthetic slice runs and reports delta vs the recorded
  baseline;
- a regression (delta below threshold) FAILS the check; improvements
  never do;
- non-trigger paths do NOT run it (path matcher + workflow declaration);
- the baseline is recorded per tier (a file with the numbers).

The slice runner is eval/ci_tier1_slice.py (reused, not rebuilt).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# The `eval` package resolves from the REPO ROOT (tests/conftest.py puts
# argos_plugin on sys.path, not the root) — guard on the root we insert.
_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from eval.ci_tier1_slice import (  # noqa: E402
    BASELINE_FILE, GOLD_SLICE, PATHS_FILE, load_gold, paths_match,
    run_slice, slice_verdict,
)


class TestTier1Slice:
    """A tiny synthetic slice runs, reports delta vs the baseline."""

    def test_slice_runs_all_probes(self):
        scores = run_slice()
        assert scores["probe_count"] == 12
        assert scores["ladder"] == [1, 3, 5]
        assert "recall@1" in scores["overall"]
        assert "mrr" in scores["overall"]

    def test_slice_matches_recorded_baseline(self):
        """Delta vs the recorded baseline: no regression."""
        baseline = json.loads(BASELINE_FILE.read_text(encoding="utf-8"))
        current = run_slice()
        ok, failures = slice_verdict(current, baseline)
        assert ok, f"Tier 1 regression vs baseline: {failures}"

    def test_baseline_recorded_with_numbers(self):
        """Baseline recorded per tier — a file with the numbers."""
        assert BASELINE_FILE.exists(), (
            "tier1 baseline missing — record with "
            "python argos_plugin/eval/ci_tier1_slice.py --record-baseline"
        )
        baseline = json.loads(BASELINE_FILE.read_text(encoding="utf-8"))
        assert baseline.get("probe_count") == 12
        overall = baseline.get("overall", {})
        assert "mrr" in overall
        assert baseline.get("ladder") == [1, 3, 5]

    def test_regression_below_threshold_fails(self):
        """A delta below threshold fails the check (regression = delta)."""
        baseline = {
            "ladder": [1, 3, 5],
            "overall": {"recall@1": 1.0, "recall@3": 1.0,
                        "recall@5": 1.0, "mrr": 1.0},
            "by_template": {},
        }
        regressed = {
            "ladder": [1, 3, 5],
            "overall": {"recall@1": 0.6, "recall@3": 0.75,
                        "recall@5": 0.83, "mrr": 0.7},
            "by_template": {},
        }
        ok, failures = slice_verdict(regressed, baseline)
        assert not ok, "a real regression must fail"
        assert failures

    def test_improvement_never_fails(self):
        """Only regressions fail — improvements never do (delta rule)."""
        baseline = {
            "ladder": [1, 3, 5],
            "overall": {"recall@1": 0.8, "recall@3": 0.9,
                        "recall@5": 0.9, "mrr": 0.8},
            "by_template": {},
        }
        improved = {
            "ladder": [1, 3, 5],
            "overall": {"recall@1": 1.0, "recall@3": 1.0,
                        "recall@5": 1.0, "mrr": 1.0},
            "by_template": {},
        }
        ok, failures = slice_verdict(improved, baseline)
        assert ok and failures == []


class TestPathTrigger:
    """Tier 1 triggers only on retrieval/embedding-touching paths."""

    def test_trigger_paths_match(self):
        assert paths_match(["argos_plugin/store_retrieval.py"])
        assert paths_match(["argos_plugin/embeddings.py"])
        assert paths_match(["argos_plugin/graph.py"])
        assert paths_match(["argos_plugin/eval/ci_slice_v1.jsonl"])

    def test_non_trigger_paths_do_not_run_it(self):
        assert not paths_match(["README.md"])
        assert not paths_match(["docs/site/index.md"])
        assert not paths_match(["argos_plugin/hermes_weather.py"])
        assert not paths_match([".github/workflows/ci.yml"])

    def test_trigger_list_contains_canonical_retrieval_paths(self):
        prefixes = [
            ln.strip()
            for ln in PATHS_FILE.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.strip().startswith("#")
        ]
        for required in (
            "argos_plugin/store_retrieval.py",
            "argos_plugin/provider_retrieval.py",
            "argos_plugin/store_core.py",
            "argos_plugin/embeddings.py",
            "argos_plugin/graph.py",
        ):
            assert any(p == required or p.startswith(required)
                       for p in prefixes), (
                f"{required} missing from the trigger list"
            )


class TestBaselineFile:
    """Baseline recorded per tier; the slice covers the tricky classes."""

    def test_baseline_recorded(self):
        assert BASELINE_FILE.exists(), (
            "tier1 baseline missing — record with "
            "python argos_plugin/eval/ci_tier1_slice.py --record-baseline"
        )
        baseline = json.loads(BASELINE_FILE.read_text(encoding="utf-8"))
        assert baseline.get("probe_count") == 12
        assert "mrr" in baseline.get("overall", {})

    def test_slice_gold_covers_tricky_classes(self):
        """The slice covers temporal, conflict, provenance, injection —
        and is 10-20 questions per the issue."""
        gold = load_gold()
        templates = {g.get("template") for g in gold}
        assert {"temporal", "conflict", "provenance",
                "injection"} <= templates
        assert 10 <= len(gold) <= 20

"""Tests for #21: verdict threshold agreement + weekly_recon coverage.

Covers:
- Shared verdict module: PASS/FAIL on synthetic cases
- Agreement: both run_gate.gate_verdict and verdict.gate_verdict agree
- eval_self_corpus._verdict delegates to shared thresholds
- Per-probe timeout in run_gate
- weekly_recon invocation (find_frozen_pair, branch logic)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_plugin_dir = Path(__file__).resolve().parent.parent
if str(_plugin_dir) not in sys.path:
    sys.path.insert(0, str(_plugin_dir))
_eval_dir = _plugin_dir / "eval"
if str(_eval_dir) not in sys.path:
    sys.path.insert(0, str(_eval_dir))

import verdict  # noqa: E402
import run_gate  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _scores(
    overall: dict | None = None,
    by_category: dict | None = None,
    ladder: list[int] | None = None,
) -> dict:
    """Build a synthetic scores dict for verdict testing."""
    l = ladder or [5, 20, 96]
    base_overall = {f"recall@{k}": 0.90 for k in l}
    base_overall["mrr"] = 0.60
    if overall:
        base_overall.update(overall)
    base_cats = by_category or {
        "personal_fact": {f"recall@{k}": 0.90 for k in l} | {"mrr": 0.60},
    }
    return {
        "ladder": l,
        "overall": base_overall,
        "by_category": base_cats,
    }


# ---------------------------------------------------------------------------
# Shared verdict module
# ---------------------------------------------------------------------------

class TestSharedVerdict:
    """The shared verdict module should produce correct PASS/FAIL."""

    def test_pass_identical(self):
        s = _scores()
        ok, failures = verdict.gate_verdict(s, s)
        assert ok and failures == []

    def test_pass_on_improvement(self):
        base = _scores()
        cur = _scores(overall={"recall@96": 0.95, "mrr": 0.70})
        ok, _ = verdict.gate_verdict(cur, base)
        assert ok, "improvements never fail the gate"

    def test_fail_category_recall_drop_over_1pp(self):
        base = _scores()
        cur = _scores(by_category={
            "personal_fact": {"recall@96": 0.88, "mrr": 0.60},
        })
        ok, failures = verdict.gate_verdict(cur, base)
        assert not ok
        assert any("personal_fact" in f for f in failures)

    def test_pass_category_recall_within_1pp(self):
        base = _scores()
        cur = _scores(by_category={
            "personal_fact": {"recall@96": 0.891, "mrr": 0.60},
        })
        ok, _ = verdict.gate_verdict(cur, base)
        assert ok, "0.9pp category drop is within tolerance"

    def test_fail_overall_recall_drop_over_0_5pp(self):
        base = _scores()
        cur = _scores(overall={"recall@96": 0.894, "mrr": 0.60})
        ok, failures = verdict.gate_verdict(cur, base)
        assert not ok
        assert any("overall" in f for f in failures)

    def test_fail_mrr_drop_over_0_01(self):
        base = _scores()
        cur = _scores(overall={"recall@96": 0.90, "mrr": 0.58})
        ok, failures = verdict.gate_verdict(cur, base)
        assert not ok
        assert any("MRR" in f for f in failures)

    def test_pass_mrr_within_0_01(self):
        base = _scores()
        cur = _scores(overall={"recall@96": 0.90, "mrr": 0.595})
        ok, _ = verdict.gate_verdict(cur, base)
        assert ok

    def test_verdict_string_pass(self):
        s = _scores()
        result = verdict.verdict_string(s, s)
        assert result.startswith("PASS")

    def test_verdict_string_fail(self):
        base = _scores()
        cur = _scores(overall={"recall@96": 0.80, "mrr": 0.50})
        result = verdict.verdict_string(cur, base)
        assert result.startswith("FAIL")


# ---------------------------------------------------------------------------
# Agreement: run_gate and verdict agree
# ---------------------------------------------------------------------------

class TestVerdictAgreement:
    """run_gate.gate_verdict and verdict.gate_verdict must agree."""

    def test_agree_on_pass(self):
        s = _scores()
        ok1, f1 = run_gate.gate_verdict(s, s)
        ok2, f2 = verdict.gate_verdict(s, s)
        assert ok1 == ok2
        assert f1 == f2

    def test_agree_on_category_fail(self):
        base = _scores()
        cur = _scores(by_category={
            "personal_fact": {"recall@96": 0.85, "mrr": 0.60},
        })
        ok1, f1 = run_gate.gate_verdict(cur, base)
        ok2, f2 = verdict.gate_verdict(cur, base)
        assert ok1 == ok2
        assert f1 == f2

    def test_agree_on_overall_recall_fail(self):
        base = _scores()
        cur = _scores(overall={"recall@96": 0.85, "mrr": 0.60})
        ok1, f1 = run_gate.gate_verdict(cur, base)
        ok2, f2 = verdict.gate_verdict(cur, base)
        assert ok1 == ok2
        assert f1 == f2

    def test_agree_on_mrr_fail(self):
        base = _scores()
        cur = _scores(overall={"recall@96": 0.90, "mrr": 0.50})
        ok1, f1 = run_gate.gate_verdict(cur, base)
        ok2, f2 = verdict.gate_verdict(cur, base)
        assert ok1 == ok2
        assert f1 == f2

    def test_agree_on_improvement(self):
        base = _scores()
        cur = _scores(overall={"recall@96": 0.99, "mrr": 0.80})
        ok1, f1 = run_gate.gate_verdict(cur, base)
        ok2, f2 = verdict.gate_verdict(cur, base)
        assert ok1 == ok2 and ok1  # both pass
        assert f1 == f2 == []

    def test_threshold_constants_match(self):
        """The re-exported constants must match the shared module."""
        assert run_gate.CATEGORY_RECALL_PP == verdict.CATEGORY_RECALL_PP
        assert run_gate.OVERALL_RECALL_PP == verdict.OVERALL_RECALL_PP
        assert run_gate.OVERALL_MRR == verdict.OVERALL_MRR


# ---------------------------------------------------------------------------
# eval_self_corpus._verdict delegates to shared module
# ---------------------------------------------------------------------------

class TestEvalSelfCorpusVerdict:
    """eval_self_corpus._verdict should delegate to the shared module."""

    def test_verdict_uses_shared_thresholds(self):
        import eval_self_corpus as esc
        # A 2pp drop in recall@max-k (96) should FAIL under the new
        # shared thresholds (overall recall > 0.5pp), but would have
        # PASSED under the old 3pp recall@20-only verdict.
        base = _scores()
        cur = _scores(overall={"recall@96": 0.88, "mrr": 0.60})
        result = esc._verdict(cur, base)
        assert result.startswith("FAIL"), (
            "2pp recall@96 drop must FAIL under shared thresholds "
            "(old 3pp recall@20-only verdict would have passed)"
        )

    def test_verdict_pass_on_clean(self):
        import eval_self_corpus as esc
        s = _scores()
        result = esc._verdict(s, s)
        assert result.startswith("PASS")


# ---------------------------------------------------------------------------
# Per-probe timeout
# ---------------------------------------------------------------------------

class TestProbeTimeout:
    """run_gate should support a per-probe timeout."""

    def test_timeout_zero_no_timeout(self):
        """timeout=0 should run normally without timeout."""
        class FakeStore:
            def search(self, query, limit, suppress_retrieval):
                return []
        gold_line = {"query": "test", "memory_id": "mem-1"}
        result = run_gate._score_probe_with_timeout(
            FakeStore(), gold_line, [5, 20, 96], timeout=0,
        )
        assert "per_window" in result
        assert "rank" in result

    def test_timeout_returns_degenerate_on_hang(self):
        """A hanging store.search should produce a degenerate miss."""
        import time
        class HangingStore:
            def search(self, query, limit, suppress_retrieval):
                time.sleep(10)  # hang
                return []
        gold_line = {"query": "test", "memory_id": "mem-1"}
        result = run_gate._score_probe_with_timeout(
            HangingStore(), gold_line, [5, 20, 96], timeout=0.5,
        )
        assert result["rank"] is None
        assert all(v is False for v in result["per_window"].values())
        assert result.get("_timeout") is True

    def test_normal_probe_completes_within_timeout(self):
        """A fast probe should complete normally within the timeout."""
        class FakeStore:
            def search(self, query, limit, suppress_retrieval):
                return []
        gold_line = {"query": "test", "memory_id": "mem-1"}
        result = run_gate._score_probe_with_timeout(
            FakeStore(), gold_line, [5, 20, 96], timeout=30,
        )
        assert "per_window" in result
        assert "rank" in result
        assert "_timeout" not in result


# ---------------------------------------------------------------------------
# weekly_recon coverage
# ---------------------------------------------------------------------------

class TestWeeklyRecon:
    """weekly_recon should find frozen pairs and report verdicts."""

    def test_find_frozen_pair_none(self, tmp_path):
        """find_frozen_pair should return None when no baseline exists."""
        import weekly_recon
        import snapshot_store
        # Create a snapshot dir with no gate_baseline.json.
        snap_dir = tmp_path / "snapshots"
        snap_dir.mkdir()
        snap_id = "test-snap-001"
        snap_path = snap_dir / snap_id
        snap_path.mkdir()
        # Write a manifest so list_snapshots finds it.
        (snap_path / "manifest.json").write_text(json.dumps({
            "snapshot_id": snap_id,
            "db_filename": "hybrid_memory.duckdb",
            "db_sha256": "abc123",
            "record_count": 100,
            "timestamp": "2026-01-01T00:00:00Z",
        }))
        result = weekly_recon.find_frozen_pair(snap_dir)
        assert result is None

    def test_find_frozen_pair_exists(self, tmp_path):
        """find_frozen_pair should return the snapshot with gate_baseline.json."""
        import weekly_recon
        snap_dir = tmp_path / "snapshots"
        snap_dir.mkdir()
        snap_id = "test-snap-002"
        snap_path = snap_dir / snap_id
        snap_path.mkdir()
        (snap_path / "manifest.json").write_text(json.dumps({
            "snapshot_id": snap_id,
            "db_filename": "hybrid_memory.duckdb",
            "db_sha256": "abc123",
            "record_count": 100,
            "timestamp": "2026-01-01T00:00:00Z",
        }))
        (snap_path / "gate_baseline.json").write_text(json.dumps({
            "snapshot_id": snap_id,
            "overall": {"recall@96": 0.90, "mrr": 0.60},
        }))
        result = weekly_recon.find_frozen_pair(snap_dir)
        assert result is not None
        assert result.name == snap_id

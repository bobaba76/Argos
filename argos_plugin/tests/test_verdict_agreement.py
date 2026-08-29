"""Cross-tool agreement tests for the self-corpus regression verdict (issue #21).

``run_gate.gate_verdict`` (the mandated sync gate) and
``eval_self_corpus._verdict`` (the standalone ``--baseline`` canary) must
apply the SAME thresholds to the SAME metrics, so a regression cannot pass
one gate and fail the other.

Both delegate to ``verdict.gate_verdict``; these tests pin that delegation
end-to-end: a synthetic regression FAILs both tools, a clean frozen pair
PASSes both, and the threshold constants are literally one source.
"""
from __future__ import annotations

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
import eval_self_corpus as esc  # noqa: E402


def _canonical(overall, by_category, ladder=(5, 20, 96)):
    """Build a canonical gate-scores dict (the run_gate.compute_scores shape)."""
    return {"ladder": list(ladder), "overall": dict(overall), "by_category": dict(by_category)}


def _esc_from_canonical(canonical):
    """Build an eval_self_corpus run_summary dict that canonicalizes back to
    ``canonical``. esc stores by_category as {recall@k: {cat: v}} (inverted)."""
    by_cat = {}
    for cat, metrics in canonical["by_category"].items():
        for metric, val in metrics.items():
            by_cat.setdefault(metric, {})[cat] = val
    return {
        "ladder": list(canonical["ladder"]),
        "overall": dict(canonical["overall"]),
        "by_category": by_cat,
    }


def _agree(canonical_current, canonical_baseline):
    """Return (run_gate_passes, esc_passes) for the same regression pair."""
    rg_ok, _ = run_gate.gate_verdict(canonical_current, canonical_baseline)
    esc_str = esc._verdict(
        _esc_from_canonical(canonical_current),
        _esc_from_canonical(canonical_baseline),
    )
    return rg_ok, esc_str.startswith("PASS")


# ---------------------------------------------------------------------------
# One threshold source
# ---------------------------------------------------------------------------

class TestOneThresholdSource:
    def test_constants_are_the_same_object(self):
        # The thresholds must be imported from verdict, not redefined.
        assert run_gate.CATEGORY_RECALL_PP is verdict.CATEGORY_RECALL_PP
        assert run_gate.OVERALL_RECALL_PP is verdict.OVERALL_RECALL_PP
        assert run_gate.OVERALL_MRR is verdict.OVERALL_MRR

    def test_run_gate_verdict_delegates_to_verdict(self):
        s = _canonical({"recall@96": 0.90, "mrr": 0.60},
                       {"personal_fact": {"recall@96": 0.90, "mrr": 0.60}})
        assert run_gate.gate_verdict(s, s) == verdict.gate_verdict(s, s)


# ---------------------------------------------------------------------------
# Agreement: same regression pair → same verdict from both tools
# ---------------------------------------------------------------------------

class TestAgreement:
    def _pair(self, cur_overall, cur_cats, base_overall, base_cats, ladder=(5, 20, 96)):
        base = _canonical(base_overall, base_cats, ladder)
        cur = _canonical(cur_overall, cur_cats, ladder)
        return cur, base

    def test_clean_pair_passes_both(self):
        cur, base = self._pair(
            {"recall@96": 0.90, "mrr": 0.60},
            {"personal_fact": {"recall@96": 0.90, "mrr": 0.60}},
            {"recall@96": 0.90, "mrr": 0.60},
            {"personal_fact": {"recall@96": 0.90, "mrr": 0.60}},
        )
        rg, e = _agree(cur, base)
        assert rg and e, "clean frozen pair must PASS both tools"

    def test_improvement_passes_both(self):
        cur, base = self._pair(
            {"recall@96": 0.95, "mrr": 0.70},
            {"personal_fact": {"recall@96": 0.95, "mrr": 0.70}},
            {"recall@96": 0.90, "mrr": 0.60},
            {"personal_fact": {"recall@96": 0.90, "mrr": 0.60}},
        )
        rg, e = _agree(cur, base)
        assert rg and e, "improvements never fail either gate"

    def test_overall_recall_regression_fails_both(self):
        cur, base = self._pair(
            {"recall@96": 0.85, "mrr": 0.60},
            {"personal_fact": {"recall@96": 0.90, "mrr": 0.60}},
            {"recall@96": 0.92, "mrr": 0.60},
            {"personal_fact": {"recall@96": 0.90, "mrr": 0.60}},
        )
        rg, e = _agree(cur, base)
        assert (not rg) and (not e), "5pp overall recall drop must FAIL both tools"

    def test_category_recall_regression_fails_both(self):
        cur, base = self._pair(
            {"recall@96": 0.92, "mrr": 0.60},
            {"personal_fact": {"recall@96": 0.80, "mrr": 0.60}},  # 10pp category drop
            {"recall@96": 0.92, "mrr": 0.60},
            {"personal_fact": {"recall@96": 0.90, "mrr": 0.60}},
        )
        rg, e = _agree(cur, base)
        assert (not rg) and (not e), "10pp category recall drop must FAIL both tools"

    def test_mrr_regression_fails_both(self):
        # Recall unchanged; MRR drops 0.05 (> 0.01) → FAIL both.
        cur, base = self._pair(
            {"recall@96": 0.90, "mrr": 0.55},
            {"personal_fact": {"recall@96": 0.90, "mrr": 0.55}},
            {"recall@96": 0.90, "mrr": 0.60},
            {"personal_fact": {"recall@96": 0.90, "mrr": 0.60}},
        )
        rg, e = _agree(cur, base)
        assert (not rg) and (not e), "MRR drop > 0.01 must FAIL both tools"

    def test_within_tolerance_passes_both(self):
        # 0.4pp overall recall drop, 0.005 MRR drop → within tolerance → PASS both.
        cur, base = self._pair(
            {"recall@96": 0.916, "mrr": 0.595},
            {"personal_fact": {"recall@96": 0.90, "mrr": 0.60}},
            {"recall@96": 0.920, "mrr": 0.600},
            {"personal_fact": {"recall@96": 0.90, "mrr": 0.60}},
        )
        rg, e = _agree(cur, base)
        assert rg and e, "within-tolerance drift must PASS both tools"

    def test_canonicalize_round_trips(self):
        canonical = _canonical(
            {"recall@96": 0.90, "recall@20": 0.80, "mrr": 0.60},
            {"personal_fact": {"recall@96": 0.90, "mrr": 0.60},
             "preference": {"recall@96": 0.88, "mrr": 0.55}},
        )
        esc_shape = _esc_from_canonical(canonical)
        re_canonical = verdict.canonicalize_esc_metrics(esc_shape)
        # overall + ladder + by_category recall@max-k must survive the round trip.
        assert re_canonical["ladder"] == canonical["ladder"]
        assert re_canonical["overall"]["recall@96"] == canonical["overall"]["recall@96"]
        assert re_canonical["overall"]["mrr"] == canonical["overall"]["mrr"]
        for cat in canonical["by_category"]:
            assert re_canonical["by_category"][cat]["recall@96"] == \
                canonical["by_category"][cat]["recall@96"]


# ---------------------------------------------------------------------------
# Per-probe timeout (run_gate)
# ---------------------------------------------------------------------------

class TestProbeTimeout:
    def test_timeout_records_miss(self, monkeypatch):
        """A probe that exceeds the timeout is recorded as a miss, not a stall."""
        import run_gate as rg

        def slow_score_probe(store, gold_line, ladder):
            import time
            time.sleep(2)
            return {"per_window": {str(k): True for k in ladder}, "rank": 1}

        monkeypatch.setattr(rg, "score_probe", slow_score_probe)
        store = object()  # never used by the patched score_probe
        gold_lines = [{"memory_id": "m1"}, {"memory_id": "m2"}]
        results = rg._run_probes_with_timeout(store, gold_lines, [5, 20], probe_timeout=0.1)
        assert len(results) == 2
        for r in results:
            assert r["rank"] is None
            assert all(v is False for v in r["per_window"].values())

    def test_no_timeout_when_fast(self, monkeypatch):
        import run_gate as rg

        def fast_probe(store, gold_line, ladder):
            return {"per_window": {str(k): True for k in ladder}, "rank": 1}

        monkeypatch.setattr(rg, "score_probe", fast_probe)
        results = rg._run_probes_with_timeout(
            object(), [{"memory_id": "m1"}, {"memory_id": "m2"}], [5, 20], probe_timeout=10,
        )
        assert [r["rank"] for r in results] == [1, 1]

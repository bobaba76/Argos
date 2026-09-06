"""#292 Tier 0 — fast deterministic retrieval smoke (every PR, seconds).

Thin wrapper over ``eval/ci_tier0_smoke.py`` (the canonical runner).
NO LLM, no embedder, no network: a tiny fixed synthetic store with
canonical assertions. Regression = delta vs the recorded baseline
(``eval/ci_tier0_baseline.json``) — a lost rank-1, a top-5 drop,
an empty retrieval, or nondeterministic ranking fails loudly.

Runs on EVERY PR (unconditionally included in the CI pytest step).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_plugin_dir = Path(__file__).resolve().parent.parent
if str(_plugin_dir) not in sys.path:
    sys.path.insert(0, str(_plugin_dir.parent))

from eval.ci_tier0_smoke import (  # noqa: E402
    GOLD_TIER0, SMOKE_SET, run_smoke, smoke_verdict,
)


@pytest.fixture(scope="module")
def current():
    """Run the smoke once per module (seconds; deterministic)."""
    return run_smoke()


def test_smoke_runs_green_deterministic_no_llm(current):
    """Tier 0 acceptance: the smoke runs green on the trigger paths —
    deterministic, no LLM (the store is built without an embedder)."""
    assert current["probe_count"] == len(SMOKE_SET)
    assert current["all_top1"] is True, "a canonical fact lost rank-1"
    assert current["all_top5"] is True
    assert current["no_empty"] is True, "empty injection on a standard query"
    assert current["deterministic"] is True, "ranking nondeterministic"


def test_smoke_matches_recorded_baseline(current):
    """Regression = delta vs the recorded baseline file."""
    baseline = json.loads(GOLD_TIER0.read_text(encoding="utf-8"))
    ok, failures = smoke_verdict(current, baseline)
    assert ok, f"Tier 0 regression vs baseline: {failures}"


def test_baseline_file_is_in_sync_with_smoke_set(current):
    """The committed baseline covers exactly the current smoke set."""
    baseline = json.loads(GOLD_TIER0.read_text(encoding="utf-8"))
    base_ids = {e["memory_id"] for e in baseline["expectations"]}
    cur_ids = {e["memory_id"] for e in current["expectations"]}
    assert base_ids == cur_ids, (
        "baseline file and smoke set drifted — re-record with "
        "python argos_plugin/eval/ci_tier0_smoke.py --record-baseline"
    )


def test_regression_detection_fails_on_lost_rank1():
    """A delta regression (rank-1 lost) must FAIL the verdict."""
    baseline = json.loads(GOLD_TIER0.read_text(encoding="utf-8"))
    current = json.loads(json.dumps(baseline))  # deep copy
    current["expectations"][0]["top1"] = False
    ok, failures = smoke_verdict(current, baseline)
    assert not ok
    assert any("rank-1" in f for f in failures)


def test_no_llm_and_no_network_needed(tmp_path):
    """The smoke must not need an embedder/LLM: run it against a store
    constructed with embedder=None and assert retrieval still works."""
    from eval.ci_tier0_smoke import build_smoke_store

    store = build_smoke_store(tmp_path)
    try:
        assert getattr(store, "embedder", None) is None, (
            "tier0 smoke store must be embedder-free"
        )
        rec = store.search("zephyr project deadline", limit=1,
                           suppress_retrieval=True)
        assert rec and rec[0].memory_id == "smoke-01"
    finally:
        store.close()

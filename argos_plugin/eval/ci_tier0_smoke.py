#!/usr/bin/env python3
"""ci_tier0_smoke.py — Tier 0 fast deterministic retrieval smoke (#292).

Runs in seconds on every PR. NO LLM, no embedder, no network, no long
ingest: a tiny fixed synthetic store (text-leg ranking) with canonical
assertions — every canonical fact retrieves at rank 1 for its direct
query, recall@5 is complete, a standard query never injects an empty
result set, and ranking is deterministic across runs.

Regression = delta vs the recorded baseline
(``eval/gold/ci_tier0_baseline.json``): the baseline records the
expected per-probe outcome (top1 / in_top5); the smoke fails if the
observed outcome is worse. Deterministic by construction.

Usage:
    python eval/ci_tier0_smoke.py                     # run + delta verdict
    python eval/ci_tier0_smoke.py --record-baseline   # (re)record baseline

Exit codes: 0 PASS, 1 regression, 2 error.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

_EVAL_DIR = Path(__file__).resolve().parent
_PLUGIN_ROOT = _EVAL_DIR.parent
for _p in (str(_PLUGIN_ROOT), str(_EVAL_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

GOLD_TIER0 = _EVAL_DIR / "ci_tier0_baseline.json"
LADDER = (1, 5)

# The canonical smoke set: (memory_id, category, content, direct query).
# Distinctive anchors; fixed order; no LLM, no embedder.
SMOKE_SET: List[Tuple[str, str, str, str]] = [
    ("smoke-01", "personal_fact",
     "Zephyr project deadline is Friday",
     "when is the zephyr project deadline"),
    ("smoke-02", "relationship",
     "Marisol coordinates the logistics team",
     "who coordinates the logistics team"),
    ("smoke-03", "preference",
     "Tavish prefers window seats on trains",
     "what seats does tavish prefer"),
    ("smoke-04", "event",
     "The Bellhaven launch is scheduled for the fourth of June",
     "when is the bellhaven launch"),
    ("smoke-05", "goal",
     "Odile aims to finish the lighthouse mural by May",
     "what does odile aim to finish"),
    ("smoke-06", "insight",
     "Corvin notices he writes best before noon",
     "when does corvin write best"),
]


def build_smoke_store(tmp_dir: Path):
    """Build the deterministic smoke store (no embedder, no LLM)."""
    from store import DuckDBMemoryStore

    store = DuckDBMemoryStore(tmp_dir / "tier0.duckdb",
                              user_id="tier0_smoke", embedder=None)
    for mid, category, content, _query in SMOKE_SET:
        store.remember(category=category, content=content, dedup=False)
        with store._state.lock:
            store.connection.execute(
                "UPDATE memory_records SET memory_id = ? WHERE content = ?",
                [mid, content],
            )
    return store


def query_of(memory_id: str) -> str:
    for mid, _cat, _content, query in SMOKE_SET:
        if mid == memory_id:
            return query
    raise KeyError(memory_id)


def _run_smoke_inner() -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Run the probes; return (per-probe outcomes, extra checks)."""
    from store import DuckDBMemoryStore  # noqa: F401  (import check)

    outcomes: List[Dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="ci_tier0_") as td:
        store = build_smoke_store(Path(td))
        try:
            for mid, _cat, _content, query in SMOKE_SET:
                results = store.search(query, limit=max(LADDER),
                                       suppress_retrieval=True)
                ids = [r.memory_id for r in results]
                outcomes.append({
                    "memory_id": mid,
                    "query": query,
                    "top1": bool(ids) and ids[0] == mid,
                    "in_top5": mid in ids[:5],
                    "non_empty": bool(ids),
                })
            # Determinism probe: same query twice → identical order.
            q = SMOKE_SET[0][3]
            ids1 = [r.memory_id for r in store.search(
                q, limit=max(LADDER), suppress_retrieval=True)]
            ids2 = [r.memory_id for r in store.search(
                q, limit=max(LADDER), suppress_retrieval=True)]
            deterministic = ids1 == ids2
        finally:
            store.close()
    return outcomes, {"deterministic": deterministic}


def run_smoke() -> Dict[str, Any]:
    outcomes, extra = _run_smoke_inner()
    return {
        "tier": 0,
        "ladder": list(LADDER),
        "probe_count": len(outcomes),
        "expectations": [
            {"memory_id": o["memory_id"], "top1": o["top1"],
             "in_top5": o["in_top5"]}
            for o in outcomes
        ],
        "all_top1": all(o["top1"] for o in outcomes),
        "all_top5": all(o["in_top5"] for o in outcomes),
        "no_empty": all(o["non_empty"] for o in outcomes),
        "deterministic": extra["deterministic"],
    }


def smoke_verdict(current: Dict[str, Any],
                  baseline: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Delta-only verdict vs the recorded baseline (fail on regression)."""
    failures: List[str] = []
    base_by_id = {e["memory_id"]: e for e in baseline.get("expectations", [])}
    for exp in current.get("expectations", []):
        base = base_by_id.get(exp["memory_id"])
        if base is None:
            failures.append(f"{exp['memory_id']}: missing from baseline")
            continue
        if base.get("top1") and not exp.get("top1"):
            failures.append(f"{exp['memory_id']}: lost rank-1")
        if base.get("in_top5") and not exp.get("in_top5"):
            failures.append(f"{exp['memory_id']}: dropped out of top-5")
    if baseline.get("no_empty") and not current.get("no_empty"):
        failures.append("empty retrieval appeared (gross breakage)")
    if baseline.get("deterministic") and not current.get("deterministic"):
        failures.append("ranking became nondeterministic")
    return (not failures, failures)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Tier 0 deterministic retrieval smoke (#292)",
    )
    parser.add_argument("--record-baseline", action="store_true",
                        help="(Re)record the baseline from the current run.")
    parser.add_argument("--baseline", default=None, type=Path,
                        help="Baseline file override (testing).")
    args = parser.parse_args()

    current = run_smoke()
    baseline_path = args.baseline or GOLD_TIER0

    if args.record_baseline or not baseline_path.exists():
        baseline_path.write_text(
            json.dumps(current, indent=2, sort_keys=True), encoding="utf-8",
        )
        print(f"baseline recorded: {baseline_path}")
        return 0

    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    ok, failures = smoke_verdict(current, baseline)
    print(json.dumps({
        "verdict": "PASS" if ok else "FAIL",
        "all_top1": current["all_top1"],
        "no_empty": current["no_empty"],
        "deterministic": current["deterministic"],
        "failures": failures,
    }, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

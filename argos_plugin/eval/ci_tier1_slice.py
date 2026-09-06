#!/usr/bin/env python3
"""ci_tier1_slice.py — Tier 1 curated retrieval slice for CI (#292).

A ~12-question curated slice covering the tricky retrieval classes
(temporal, conflict, provenance, injection/negation, entity, direct).
Deterministic and LLM-free: the slice seeds a fixed synthetic store (no
embedder — the text leg ranks), scores each probe, and aggregates
recall@k + MRR as a DELTA against the recorded baseline
(``eval/gold/ci_slice_baseline.json``). Regression = delta, not
absolute — the honest-benchmarks rule.

This is NOT the full LongMemEval gate (that's Tier 2, ``run_gate.py``
on the frozen snapshot, weekly). The slice runs in seconds-to-minutes,
needs no snapshot, no HF model, and no network.

Usage:
    python eval/ci_tier1_slice.py                     # score + delta verdict
    python eval/ci_tier1_slice.py --record-baseline   # (re)record baseline
    python eval/ci_tier1_slice.py --paths <changed files...>

Exit codes: 0 PASS/skip, 1 regression vs baseline, 2 error.
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

GOLD_SLICE = _EVAL_DIR / "ci_slice_v1.jsonl"
BASELINE_FILE = _EVAL_DIR / "ci_slice_baseline.json"
PATHS_FILE = _EVAL_DIR / "ci_retrieval_paths.txt"

# Small-store ladder (12 probes; the full gate uses 5,20,96 on 995).
LADDER = [1, 3, 5]

# Delta thresholds — SCALED for a 12-probe slice (1 probe = 8.3pp, so the
# full gate's 0.5pp overall threshold is meaningless here). Same
# philosophy as eval/verdict.py: regression = delta vs baseline, and
# improvements never fail.
SLICE_OVERALL_RECALL_PP = 15.0   # overall recall@max-k drop > 15pp (2 probes) fails
SLICE_MRR = 0.15                 # overall MRR drop > 0.15 fails


def load_gold(path: Path | None = None) -> List[Dict[str, Any]]:
    """Load the curated slice probes (gold line format)."""
    path = path or GOLD_SLICE
    out: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def build_slice_store(gold: List[Dict[str, Any]], tmp_dir: Path):
    """Build the deterministic slice store (no embedder, no LLM).

    One record per gold probe (memory_id rewritten to the gold's stable
    id) plus fixed distractors so ranking is non-trivial.
    """
    from store import DuckDBMemoryStore

    store = DuckDBMemoryStore(tmp_dir / "ci_slice.duckdb",
                              user_id="ci_slice", embedder=None)
    for g in gold:
        store.remember(category=g["category"], content=g["content"],
                       dedup=False)
        with store._state.lock:
            store.connection.execute(
                "UPDATE memory_records SET memory_id = ? WHERE content = ?",
                [g["memory_id"], g["content"]],
            )
    distractors = [
        "The quarterly parking permits renew on the second Friday of January.",
        "Milo keeps a spare projector cable in the second-floor cabinet.",
        "The shared spreadsheet password rotates every ninety days.",
        "A backup of the photo archive lives on the drive labeled ORANGE.",
    ]
    for d in distractors:
        store.remember(category="context_note", content=d, dedup=False)
    return store


def score_probe(store: Any, gold: Dict[str, Any], ladder: List[int]) -> Dict[str, Any]:
    """Rank the gold target for one query; per-window hits + rank."""
    max_k = max(ladder)
    results = store.search(gold["query"], limit=max_k,
                           suppress_retrieval=True)
    result_ids = [r.memory_id for r in results]
    rank = None
    for i, rid in enumerate(result_ids):
        if rid == gold["memory_id"]:
            rank = i + 1
            break
    return {
        "per_window": {str(k): (rank is not None and rank <= k)
                       for k in ladder},
        "rank": rank,
    }


def _mrr(rank: int | None) -> float:
    return 1.0 / rank if rank else 0.0


def _summarize(probes: List[Dict[str, Any]], ladder: List[int]) -> Dict[str, float]:
    n = len(probes)
    out: Dict[str, float] = {}
    for k in ladder:
        hits = sum(1 for p in probes if p["per_window"].get(str(k)))
        out[f"recall@{k}"] = round(hits / n, 4) if n else 0.0
    out["mrr"] = round(
        sum(1.0 / p["rank"] for p in probes if p["rank"]) / n, 4,
    ) if n else 0.0
    return out


def run_slice(gold_path: Path | None = None) -> Dict[str, Any]:
    """Run the slice; return scores in the run_gate shape (delta-ready)."""
    gold = load_gold(gold_path)
    with tempfile.TemporaryDirectory(prefix="ci_slice_") as td:
        store = build_slice_store(gold, Path(td))
        try:
            probe_results = [score_probe(store, g, LADDER) for g in gold]
        finally:
            store.close()
    by_template: Dict[str, List[Dict[str, Any]]] = {}
    for g, p in zip(gold, probe_results):
        by_template.setdefault(g.get("template") or "direct", []).append(p)
    return {
        "gold": GOLD_SLICE.name,
        "probe_count": len(probe_results),
        "ladder": list(LADDER),
        "overall": _summarize(probe_results, LADDER),
        "by_template": {
            t: _summarize(ps, LADDER) for t, ps in sorted(by_template.items())
        },
    }


def slice_verdict(
    current: Dict[str, Any], baseline: Dict[str, Any],
) -> Tuple[bool, List[str]]:
    """Delta-only verdict (regression = delta vs baseline, not absolute).

    Mirrors eval/verdict.py's philosophy with slice-scaled thresholds:
    only regressions fail — improvements never do.
    """
    ladder = current.get("ladder") or baseline.get("ladder") or LADDER
    max_k = max(ladder)
    rk = f"recall@{max_k}"
    failures: List[str] = []
    bo = baseline.get("overall", {})
    co = current.get("overall", {})
    drop_r = bo.get(rk, 0.0) - co.get(rk, 0.0)
    if drop_r * 100 > SLICE_OVERALL_RECALL_PP:
        failures.append(
            f"overall {rk}: {co.get(rk, 0.0)*100:.1f}% vs baseline "
            f"{bo.get(rk, 0.0)*100:.1f}% ({drop_r*100:+.1f}pp)"
        )
    drop_m = bo.get("mrr", 0.0) - co.get("mrr", 0.0)
    if drop_m > SLICE_MRR:
        failures.append(
            f"overall MRR: {co.get('mrr', 0.0):.4f} vs baseline "
            f"{bo.get('mrr', 0.0):.4f} ({drop_m:+.4f})"
        )
    return (not failures, failures)


def paths_match(changed_files: List[str]) -> bool:
    """True iff any changed file matches a retrieval-touching path.

    The trigger list is the checked-in ``ci_retrieval_paths.txt`` —
    the CI workflow and this function share one source of truth.
    """
    prefixes = [
        ln.strip() for ln in PATHS_FILE.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]
    for f in changed_files:
        for p in prefixes:
            if f == p or f.startswith(p):
                return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Tier 1 curated retrieval slice (#292)",
    )
    parser.add_argument("--record-baseline", action="store_true",
                        help="(Re)record the baseline from the current run.")
    parser.add_argument("--paths", nargs="*", default=None,
                        help="Changed-file paths: print the trigger decision.")
    parser.add_argument("--baseline", default=None, type=Path,
                        help="Baseline file override (testing).")
    args = parser.parse_args()

    if args.paths is not None:
        print(f"tier1 trigger: {paths_match(args.paths)}")
        return 0

    current = run_slice()
    baseline_path = args.baseline or BASELINE_FILE

    if args.record_baseline or not baseline_path.exists():
        baseline_path.write_text(
            json.dumps(current, indent=2, sort_keys=True), encoding="utf-8",
        )
        print(f"baseline recorded: {baseline_path}")
        return 0

    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    ok, failures = slice_verdict(current, baseline)
    print(json.dumps({
        "verdict": "PASS" if ok else "FAIL",
        "overall": current["overall"],
        "baseline_overall": baseline.get("overall", {}),
        "failures": failures,
    }, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

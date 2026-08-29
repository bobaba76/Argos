"""Shared verdict thresholds for the self-corpus gate (#21).

Single source of truth for PASS/FAIL thresholds. Both ``run_gate`` and
``eval_self_corpus --baseline`` delegate to these constants so the two
tools agree on synthetic PASS/FAIL cases.

Thresholds (per the regression-gate spec):
- No category recall@max-k drops > 1pp
- Overall recall@max-k drop <= 0.5pp
- Overall MRR drop <= 0.01

Only regressions fail the gate — improvements never do.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

# --- Threshold constants (the single source of truth) ---------------------

CATEGORY_RECALL_PP = 1.0    # no category recall@max-k drops > 1pp
OVERALL_RECALL_PP = 0.5     # overall recall@max-k drop <= 0.5pp
OVERALL_MRR = 0.01          # overall MRR drop <= 0.01


def gate_verdict(
    current: Dict[str, Any],
    baseline: Dict[str, Any],
) -> Tuple[bool, List[str]]:
    """Compare current scores vs baseline; return (pass, failure reasons).

    PASS if: no category recall@max-k drops > 1pp, overall recall@max-k
    drop <= 0.5pp, and overall MRR drop <= 0.01.  Only regressions fail
    the gate — improvements never do.

    This is the single verdict function used by both ``run_gate`` and
    ``eval_self_corpus --baseline``.
    """
    ladder = current.get("ladder") or baseline.get("ladder") or [5, 20, 96]
    max_k = max(ladder)
    rk = f"recall@{max_k}"
    failures: List[str] = []
    base_cats = baseline.get("by_category", {})
    cur_cats = current.get("by_category", {})
    for cat, bm in base_cats.items():
        cm = cur_cats.get(cat, {})
        drop = bm.get(rk, 0.0) - cm.get(rk, 0.0)
        if drop > CATEGORY_RECALL_PP / 100.0:
            failures.append(
                f"category {cat}: {rk} {cm.get(rk, 0.0)*100:.1f}% vs baseline "
                f"{bm.get(rk, 0.0)*100:.1f}% ({drop*100:+.1f}pp)"
            )
    bo = baseline.get("overall", {})
    co = current.get("overall", {})
    drop_r = bo.get(rk, 0.0) - co.get(rk, 0.0)
    if drop_r > OVERALL_RECALL_PP / 100.0:
        failures.append(
            f"overall {rk}: {co.get(rk, 0.0)*100:.1f}% vs baseline "
            f"{bo.get(rk, 0.0)*100:.1f}% ({drop_r*100:+.1f}pp)"
        )
    drop_m = bo.get("mrr", 0.0) - co.get("mrr", 0.0)
    if drop_m > OVERALL_MRR:
        failures.append(
            f"overall MRR: {co.get('mrr', 0.0):.4f} vs baseline "
            f"{bo.get('mrr', 0.0):.4f} ({drop_m:+.4f})"
        )
    return (not failures, failures)


def verdict_string(current: Dict[str, Any], baseline: Dict[str, Any]) -> str:
    """Return a human-readable verdict string (PASS/FAIL + details).

    Used by ``eval_self_corpus --baseline`` for its summary output.
    """
    ok, failures = gate_verdict(current, baseline)
    if ok:
        co = current.get("overall", {})
        bo = baseline.get("overall", {})
        ladder = current.get("ladder") or baseline.get("ladder") or [5, 20, 96]
        max_k = max(ladder)
        rk = f"recall@{max_k}"
        return (
            f"PASS {rk}={co.get(rk, 0.0)*100:.1f}% "
            f"(baseline {bo.get(rk, 0.0)*100:.1f}%)"
        )
    parts = ["FAIL"]
    for f in failures:
        parts.append(f"  - {f}")
    return "\n".join(parts)

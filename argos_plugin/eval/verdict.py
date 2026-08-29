"""verdict.py — single source of truth for the self-corpus regression verdict.

Both retrieval-eval tools (`run_gate.py` — the mandated sync gate, and
`eval_self_corpus.py --baseline` — the standalone 3pp canary) must apply the
SAME thresholds to the SAME metrics, so a regression cannot pass one gate and
fail the other (issue #21).

Canonical scores shape (produced by ``run_gate.compute_scores`` and by
``canonicalize_esc_metrics`` for the eval_self_corpus metrics):

    {
      "ladder": [5, 20, 96],
      "overall": {"recall@<k>": float, ..., "mrr": float},
      "by_category": {<cat>: {"recall@<k>": float, ..., "mrr": float}, ...},
    }

Verdict rule (regressions only — improvements never fail):

  PASS if: no category recall@<max-k> drops > CATEGORY_RECALL_PP,
           overall recall@<max-k> drop <= OVERALL_RECALL_PP,
           overall MRR drop <= OVERALL_MRR.
  Else FAIL.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

# Verdict thresholds — the one place these live. Both tools import from here.
CATEGORY_RECALL_PP = 1.0    # no category recall@max-k drops > 1pp
OVERALL_RECALL_PP = 0.5     # overall recall@max-k drop <= 0.5pp
OVERALL_MRR = 0.01          # overall MRR drop <= 0.01


def mrr(rank: Optional[int]) -> float:
    """Reciprocal rank: 1/rank, or 0.0 when there is no hit."""
    return 1.0 / rank if rank else 0.0


def _max_k(scores: Dict[str, Any], fallback: List[int] = (5, 20, 96)) -> int:
    ladder = scores.get("ladder") or list(fallback)
    return max(ladder) if ladder else max(fallback)


def gate_verdict(current: Dict[str, Any], baseline: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Compare current vs baseline canonical scores; return (pass, failure reasons).

    PASS if: no category recall@max-k drops > 1pp, overall recall@max-k
    drop <= 0.5pp, and overall MRR drop <= 0.01.  Only regressions fail
    the gate — improvements never do.
    """
    max_k = max(_max_k(current), _max_k(baseline))
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


def canonicalize_esc_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Convert an eval_self_corpus ``compute_metrics`` dict into the canonical
    gate-scores shape so ``gate_verdict`` can consume it.

    eval_self_corpus stores per-category recall as ``by_category[recall@k][cat]``
    (keyed by window, then category) and historically omitted MRR. The canonical
    shape is ``by_category[cat][recall@k]`` (keyed by category, then metric) with
    ``overall["mrr"]`` present. MRR is read from ``overall`` if present (added by
    compute_metrics); otherwise it defaults to 0.0 in both arms of the verdict,
    which makes the MRR check a no-op for legacy baselines that never carried it.
    """
    ladder = metrics.get("ladder") or [5, 20, 96]
    overall_in = metrics.get("overall", {}) or {}
    overall = {k: float(v) for k, v in overall_in.items()}
    overall.setdefault("mrr", 0.0)

    # Flip by_category[recall@k][cat] -> by_category[cat][recall@k].
    by_cat_in = metrics.get("by_category", {}) or {}
    by_category: Dict[str, Dict[str, float]] = {}
    for window, cat_map in by_cat_in.items():
        if not isinstance(cat_map, dict):
            continue
        for cat, val in cat_map.items():
            by_category.setdefault(cat, {})[window] = float(val)
    # by_category MRR is not checked by gate_verdict (only recall@max-k is),
    # so we do not synthesize per-category MRR here.
    return {"ladder": list(ladder), "overall": overall, "by_category": by_category}

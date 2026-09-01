"""Rejection-ledger quality monitor (#121).

A read-only quality monitor over existing rejection/quarantine data:
per-category decision rates over configurable windows, optional drift
thresholds, and export of rejected/quarantined rows as a labeled hard-case
eval set for regression gates.

Zero new data collection — the rows already exist in ``memory_candidates``
(reviewed decisions) and ``rejection_ledger`` (claim-slot rejections).
Zero LLM calls. Zero schema changes. Zero writes — the monitor is
guaranteed read-only: it only calls the store's ``query_*`` methods,
which never mutate.

Three entry points:
- :func:`decision_rate_report` — per-category decision counts + rates over
  a time window. The base signal.
- :func:`drift_check` — compares two adjacent windows and flags categories
  whose rejection rate moved beyond a configured threshold. Opt-in.
- :func:`export_hard_cases` — rejected/quarantined rows as labeled eval
  items (gold = the recorded rejection/quarantine reason).

Design notes:
- Bucketing: by category first (the issue's primary axis). The ``by``
  parameter on ``decision_rate_report`` adds optional sub-grouping by
  ``provenance_origin`` or ``source``.
- Output: JSON-serializable dicts — stdout/JSON, cron-able.
- Growth policy for hard-case export: caller passes ``limit`` (default
  500, capped at 10000 by the store query). No auto-sampling — the caller
  decides. The report includes a ``total_available`` count so the caller
  can decide whether to sample.
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Decision statuses that count as "rejected" for rate computation.
# ``rejected`` = explicit reviewer rejection.
# ``quarantined`` = blocked at ingestion (injection/security) — never
#   reached a reviewer, but is still a "did not activate" outcome.
# ``deduplicated`` = dropped by dedup, not a rejection — excluded.
REJECTED_STATUSES = frozenset({"rejected", "quarantined"})

# Decision statuses that count as "approved" for rate computation.
APPROVED_STATUSES = frozenset({"approved", "reviewed_approved"})

# All terminal (reviewed) statuses — pending is excluded.
TERMINAL_STATUSES = REJECTED_STATUSES | APPROVED_STATUSES | {"deduplicated"}


def _window_bounds(
    window: str,
    end: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """Compute (since, until) ISO timestamps for a named window.

    *window* is one of ``"daily"``, ``"weekly"``, ``"monthly"`` (or
    ``"all"`` for no time bound). *end* is an optional ISO cutoff
    (defaults to now).
    """
    if window == "all":
        return None, None
    if end:
        try:
            end_dt = datetime.fromisoformat(end)
        except (ValueError, TypeError):
            end_dt = datetime.now(timezone.utc)
    else:
        end_dt = datetime.now(timezone.utc)
    if end_dt.tzinfo is None:
        end_dt = end_dt.replace(tzinfo=timezone.utc)
    until = end_dt.isoformat()
    if window == "daily":
        since_dt = end_dt - timedelta(days=1)
    elif window == "weekly":
        since_dt = end_dt - timedelta(weeks=1)
    elif window == "monthly":
        since_dt = end_dt - timedelta(days=30)
    else:
        return None, None
    return since_dt.isoformat(), until


def _bucket_key(
    row: dict,
    by: str = "category",
) -> str:
    """Compute the bucket key for a decision row.

    *by* is ``"category"`` (default), ``"provenance_origin"``, or
    ``"source"``. Missing values fall back to ``"unknown"``.
    """
    if by == "category":
        return row.get("category") or "unknown"
    if by == "provenance_origin":
        return row.get("provenance_origin") or "unknown"
    if by == "source":
        return row.get("source") or "unknown"
    return row.get("category") or "unknown"


def decision_rate_report(
    store: Any,
    *,
    window: str = "weekly",
    end: Optional[str] = None,
    by: str = "category",
) -> Dict[str, Any]:
    """Per-bucket decision counts and rates over a time window.

    Args:
        store: a DuckDBMemoryStore (or compatible) with
            ``query_candidate_decisions``.
        window: ``"daily"``, ``"weekly"``, ``"monthly"``, or ``"all"``.
        end: optional ISO cutoff (defaults to now).
        by: bucketing axis — ``"category"``, ``"provenance_origin"``,
            or ``"source"``.

    Returns a JSON-serializable dict::

        {
          "window": "weekly",
          "since": "...", "until": "...",
          "bucketed_by": "category",
          "total_decisions": 42,
          "buckets": {
            "personal_fact": {
              "total": 20,
              "approved": 15, "rejected": 3, "quarantined": 1,
              "deduplicated": 1,
              "rejection_rate": 0.2,   # (rejected+quarantined)/total
              "approval_rate": 0.75
            },
            ...
          }
        }

    Read-only — never writes to the store.
    """
    since, until = _window_bounds(window, end)
    rows = store.query_candidate_decisions(since=since, until=until)
    buckets: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "total": 0, "approved": 0, "rejected": 0,
            "quarantined": 0, "deduplicated": 0,
        }
    )
    for row in rows:
        status = row.get("status") or "unknown"
        key = _bucket_key(row, by)
        b = buckets[key]
        b["total"] += 1
        if status in APPROVED_STATUSES:
            b["approved"] += 1
        elif status == "rejected":
            b["rejected"] += 1
        elif status == "quarantined":
            b["quarantined"] += 1
        elif status == "deduplicated":
            b["deduplicated"] += 1
    # Compute rates.
    result_buckets: Dict[str, Dict[str, Any]] = {}
    total_decisions = 0
    for key, b in buckets.items():
        total = b["total"]
        total_decisions += total
        rejected_count = b["rejected"] + b["quarantined"]
        b_out = dict(b)
        b_out["rejection_rate"] = (
            round(rejected_count / total, 4) if total else 0.0
        )
        b_out["approval_rate"] = (
            round(b["approved"] / total, 4) if total else 0.0
        )
        result_buckets[key] = b_out
    return {
        "window": window,
        "since": since,
        "until": until,
        "bucketed_by": by,
        "total_decisions": total_decisions,
        "buckets": result_buckets,
    }


def drift_check(
    store: Any,
    *,
    window: str = "weekly",
    end: Optional[str] = None,
    threshold: float = 0.15,
    by: str = "category",
) -> Dict[str, Any]:
    """Flag buckets whose rejection rate moved beyond *threshold* between
    the current window and the preceding window of the same length.

    Opt-in: the caller explicitly invokes this with a threshold. The
    default 0.15 (15 percentage points) is conservative.

    Args:
        store: a DuckDBMemoryStore (or compatible).
        window: ``"daily"``, ``"weekly"``, or ``"monthly"``.
        end: optional ISO cutoff for the END of the current window.
        threshold: absolute rate change (0.0–1.0) that triggers a flag.
        by: bucketing axis (same as ``decision_rate_report``).

    Returns::

        {
          "window": "weekly",
          "threshold": 0.15,
          "current_window": {"since": ..., "until": ...},
          "previous_window": {"since": ..., "until": ...},
          "flags": [
            {
              "bucket": "personal_fact",
              "current_rate": 0.35, "previous_rate": 0.10,
              "delta": 0.25, "direction": "up"
            },
            ...
          ],
          "all_buckets": [ ... ]   # every bucket with both-window data
        }

    Read-only.
    """
    since_cur, until_cur = _window_bounds(window, end)
    # Previous window: shift end back by one window length.
    if since_cur and until_cur:
        try:
            cur_end = datetime.fromisoformat(until_cur)
        except (ValueError, TypeError):
            cur_end = datetime.now(timezone.utc)
        if cur_end.tzinfo is None:
            cur_end = cur_end.replace(tzinfo=timezone.utc)
        if window == "daily":
            shift = timedelta(days=1)
        elif window == "weekly":
            shift = timedelta(weeks=1)
        elif window == "monthly":
            shift = timedelta(days=30)
        else:
            shift = timedelta(weeks=1)
        prev_end = (cur_end - shift).isoformat()
        since_prev, until_prev = _window_bounds(window, prev_end)
    else:
        # "all" window — drift check is meaningless without time bounds.
        return {
            "window": window,
            "threshold": threshold,
            "error": "drift_check requires a time-bounded window "
                     "(daily/weekly/monthly), not 'all'",
            "flags": [],
            "all_buckets": [],
        }
    cur_rows = store.query_candidate_decisions(since=since_cur, until=until_cur)
    prev_rows = store.query_candidate_decisions(since=since_prev, until=until_prev)

    def _rates(rows: List[dict]) -> Dict[str, Tuple[int, float]]:
        buckets: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"total": 0, "rejected": 0}
        )
        for row in rows:
            key = _bucket_key(row, by)
            b = buckets[key]
            b["total"] += 1
            status = row.get("status") or ""
            if status in REJECTED_STATUSES:
                b["rejected"] += 1
        out: Dict[str, Tuple[int, float]] = {}
        for key, b in buckets.items():
            rate = (b["rejected"] / b["total"]) if b["total"] else 0.0
            out[key] = (b["total"], round(rate, 4))
        return out

    cur_rates = _rates(cur_rows)
    prev_rates = _rates(prev_rows)
    flags: List[Dict[str, Any]] = []
    all_buckets: List[Dict[str, Any]] = []
    for key in sorted(set(cur_rates) | set(prev_rates)):
        cur_total, cur_rate = cur_rates.get(key, (0, 0.0))
        prev_total, prev_rate = prev_rates.get(key, (0, 0.0))
        delta = round(cur_rate - prev_rate, 4)
        entry = {
            "bucket": key,
            "current_rate": cur_rate,
            "previous_rate": prev_rate,
            "current_total": cur_total,
            "previous_total": prev_total,
            "delta": delta,
            "direction": "up" if delta > 0 else ("down" if delta < 0 else "flat"),
        }
        all_buckets.append(entry)
        if abs(delta) >= threshold and (cur_total > 0 or prev_total > 0):
            flags.append(entry)
    return {
        "window": window,
        "threshold": threshold,
        "current_window": {"since": since_cur, "until": until_cur},
        "previous_window": {"since": since_prev, "until": until_prev},
        "flags": flags,
        "all_buckets": all_buckets,
    }


def export_hard_cases(
    store: Any,
    *,
    statuses: Tuple[str, ...] = ("rejected", "quarantined"),
    limit: int = 500,
) -> Dict[str, Any]:
    """Export rejected/quarantined candidates as labeled hard-case eval
    items (#121).

    Each item's gold label is the recorded ``review_reason`` (for rejected)
    or ``quarantine_reason`` (for quarantined). The export is a
    JSON-serializable dict suitable for feeding into a regression gate
    whenever the extractor or reviewer changes.

    Args:
        store: a DuckDBMemoryStore (or compatible) with ``query_hard_cases``.
        statuses: which terminal statuses to include.
        limit: max rows (caller decides growth policy).

    Returns::

        {
          "total_exported": 42,
          "total_available": 42,
          "statuses": ["rejected", "quarantined"],
          "items": [
            {
              "candidate_id": "...",
              "category": "personal_fact",
              "content": "...",
              "status": "rejected",
              "label": "review_rejected",
              "label_field": "review_reason",
              "provenance_origin": "internal",
              "source": "llm_extraction",
              "created_at": "...", "reviewed_at": "..."
            },
            ...
          ]
        }

    Read-only — never writes to the store.
    """
    rows = store.query_hard_cases(statuses=statuses, limit=limit)
    items: List[Dict[str, Any]] = []
    for row in rows:
        status = row.get("status") or ""
        if status == "quarantined":
            label = row.get("quarantine_reason") or "quarantined"
            label_field = "quarantine_reason"
        else:
            label = row.get("review_reason") or "rejected"
            label_field = "review_reason"
        items.append({
            "candidate_id": row.get("candidate_id"),
            "category": row.get("category"),
            "content": row.get("content"),
            "status": status,
            "label": label,
            "label_field": label_field,
            "provenance_origin": row.get("provenance_origin"),
            "source": row.get("source"),
            "created_at": row.get("created_at"),
            "reviewed_at": row.get("reviewed_at"),
        })
    return {
        "total_exported": len(items),
        "total_available": len(items),
        "statuses": list(statuses),
        "items": items,
    }


def render_report_text(report: Dict[str, Any]) -> str:
    """Render a ``decision_rate_report`` result as human-readable text
    (stdout/cron-able)."""
    lines = [
        f"Rejection Quality Monitor — window={report['window']} "
        f"bucketed_by={report['bucketed_by']}",
        f"Range: {report.get('since') or 'begin'} → "
        f"{report.get('until') or 'now'}",
        f"Total decisions: {report['total_decisions']}",
        "",
    ]
    buckets = report.get("buckets") or {}
    if not buckets:
        lines.append("(no reviewed candidates in window)")
        return "\n".join(lines)
    lines.append(
        f"{'bucket':<24} {'total':>6} {'appr':>6} {'rej':>6} "
        f"{'quar':>6} {'dedup':>6} {'rej_rate':>9} {'appr_rate':>9}"
    )
    lines.append("-" * 80)
    for key in sorted(buckets):
        b = buckets[key]
        lines.append(
            f"{key:<24} {b['total']:>6} {b['approved']:>6} "
            f"{b['rejected']:>6} {b['quarantined']:>6} "
            f"{b['deduplicated']:>6} "
            f"{b['rejection_rate']:>9.1%} {b['approval_rate']:>9.1%}"
        )
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point: ``python -m rejection_quality_monitor [report|drift|export]``.

    Reads from the store at $HERMES_HOME/hybrid_memory.duckdb (or the path
    in argv). Zero LLM, zero writes.
    """
    import sys
    if argv is None:
        argv = sys.argv[1:]
    action = argv[0] if argv else "report"
    db_path = None
    window = "weekly"
    threshold = 0.15
    by = "category"
    limit = 500
    # Minimal arg parsing — no argparse dependency for a cron-able script.
    i = 1
    while i < len(argv):
        if argv[i] == "--db" and i + 1 < len(argv):
            db_path = argv[i + 1]
            i += 2
        elif argv[i] == "--window" and i + 1 < len(argv):
            window = argv[i + 1]
            i += 2
        elif argv[i] == "--threshold" and i + 1 < len(argv):
            threshold = float(argv[i + 1])
            i += 2
        elif argv[i] == "--by" and i + 1 < len(argv):
            by = argv[i + 1]
            i += 2
        elif argv[i] == "--limit" and i + 1 < len(argv):
            limit = int(argv[i + 1])
            i += 2
        else:
            i += 1
    if db_path is None:
        try:
            from pathlib import Path
            import os
            try:
                from hermes_constants import get_hermes_home
                home = Path(get_hermes_home())
            except Exception:
                home = Path(os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes")))
            db_path = str(home / "hybrid_memory.duckdb")
        except Exception as exc:
            print(f"Could not determine DB path: {exc}", file=sys.stderr)
            return 1
    try:
        if __package__:
            from .store import DuckDBMemoryStore
        else:
            from store import DuckDBMemoryStore
        store = DuckDBMemoryStore(db_path, user_id="monitor")
    except Exception as exc:
        print(f"Could not open store at {db_path}: {exc}", file=sys.stderr)
        return 1
    try:
        if action == "drift":
            result = drift_check(store, window=window, threshold=threshold, by=by)
        elif action == "export":
            result = export_hard_cases(store, limit=limit)
        else:
            result = decision_rate_report(store, window=window, by=by)
            print(render_report_text(result))
            return 0
        print(json.dumps(result, indent=2, default=str))
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())

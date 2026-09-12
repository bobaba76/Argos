#!/usr/bin/env python3
"""Aggregate the Argos LLM call trace (egress.py #454) into a cost ledger.

Reads the JSONL written by the egress trace (default: hermes home
llm_trace.jsonl) and prints, per kind/task: call count, prompt/completion/
cached tokens, and dated money at the reference rates. Also prints a
per-hour histogram so per-message attribution can be read off the timeline.

Usage:
    python benchmarks/trace_report.py [--path PATH] [--since YYYY-MM-DD] [--json]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

# Reference rates (OpenRouter, dated) — same table as cost_overhead.py.
RATES = {
    "deepseek/deepseek-v4-flash-0731": {"in": 0.065, "cache": 0.016, "out": 0.18},
}
RATES_DATE = "2026-09-11"


def default_path() -> str:
    home = Path.home() / "AppData/Local/hermes"
    if (home / "llm_trace.jsonl").is_file():
        return str(home / "llm_trace.jsonl")
    return str(home / "llm_trace.jsonl")


def load_rows(path: str, since: str | None):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if since and row.get("ts", "")[:10] < since:
                continue
            rows.append(row)
    return rows


def price(row: dict) -> float:
    model = row.get("model") or "deepseek/deepseek-v4-flash-0731"
    rates = RATES.get(model)
    if not rates:
        return 0.0
    pt = row.get("prompt_tokens") or 0
    ct = row.get("completion_tokens") or 0
    cached = row.get("cached_tokens") or 0
    fresh = max(pt - cached, 0)
    return (fresh * rates["in"] + cached * rates["cache"] + ct * rates["out"]) / 1_000_000


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--path", default=None)
    ap.add_argument("--since", default=None, help="YYYY-MM-DD; only rows on/after this date")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    args = ap.parse_args()

    path = args.path or default_path()
    if not Path(path).is_file():
        print(f"trace file not found: {path}", file=sys.stderr)
        print("enable with llm_trace_enabled=true in hybrid_memory.json and restart the service", file=sys.stderr)
        return 2

    rows = load_rows(path, args.since)
    if not rows:
        print("no trace rows", file=sys.stderr)
        return 1

    by_kind = defaultdict(lambda: {"calls": 0, "pt": 0, "ct": 0, "cached": 0, "ok": 0, "fail": 0})
    by_hour: Counter[str] = Counter()
    total = 0.0
    for row in rows:
        kind = row.get("kind", row.get("task", "unknown"))
        b = by_kind[kind]
        b["calls"] += 1
        b["pt"] += row.get("prompt_tokens") or 0
        b["ct"] += row.get("completion_tokens") or 0
        b["cached"] += row.get("cached_tokens") or 0
        b["ok" if row.get("ok", True) else "fail"] += 1
        total += price(row)
        by_hour[row.get("ts", "")[:13]] += 1

    if args.json:
        print(json.dumps({
            "rates_date": RATES_DATE,
            "rows": len(rows),
            "total_usd": round(total, 6),
            "by_kind": {k: {**v, "usd": round(sum(price(r) for r in rows if r.get("kind", r.get("task")) == k), 6)} for k, v in by_kind.items()},
            "by_hour": dict(sorted(by_hour.items())),
        }, indent=2))
        return 0

    print(f"trace: {path}  ({len(rows)} rows, rates {RATES_DATE})")
    print(f"{'kind':<20} {'calls':>6} {'ok':>5} {'fail':>5} {'prompt':>9} {'cached':>9} {'completion':>11} {'est $':>9}")
    print("-" * 80)
    for kind in sorted(by_kind):
        b = by_kind[kind]
        kind_cost = sum(price(r) for r in rows if r.get("kind", r.get("task")) == kind)
        print(f"{kind:<20} {b['calls']:>6} {b['ok']:>5} {b['fail']:>5} {b['pt']:>9} {b['cached']:>9} {b['ct']:>11} {kind_cost:>9.5f}")
    print("-" * 80)
    print(f"{'TOTAL':<20} {len(rows):>6} {sum(b['ok'] for b in by_kind.values()):>5} {sum(b['fail'] for b in by_kind.values()):>5} "
          f"{sum(b['pt'] for b in by_kind.values()):>9} {sum(b['cached'] for b in by_kind.values()):>9} "
          f"{sum(b['ct'] for b in by_kind.values()):>11} {total:>9.5f}")
    print()
    print("calls per hour (UTC):")
    for hour, n in sorted(by_hour.items()):
        print(f"  {hour}:00  {'#' * min(n, 60)} {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
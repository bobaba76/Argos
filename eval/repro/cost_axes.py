#!/usr/bin/env python3
"""Cost-axes counter for BENCHMARK_REPRODUCIBILITY.md section 9.

Recomputes the composed-context token/turn figures from a committed
phase-A cache. Deterministic: same cache -> same numbers.

Usage:
    python cost_axes.py <cache.phaseA.jsonl> [--k 96] [--cap 1500] [--floor 0.30]
"""
import argparse
import json
import statistics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cache")
    ap.add_argument("--k", type=int, default=96)
    ap.add_argument("--cap", type=int, default=1500)
    ap.add_argument("--floor", type=float, default=0.30)
    args = ap.parse_args()

    total = n = secs = 0
    counts = []
    with open(args.cache, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            recs = [r for r in rec.get("records", [])
                    if r.get("sim", 0.0) >= args.floor]
            recs.sort(key=lambda r: -r.get("sim", 0.0))
            recs = recs[:args.k]
            total += sum(min(len(r.get("content", "")), args.cap) for r in recs)
            counts.append(len(recs))
            secs += rec.get("secs", 0.0)
            n += 1

    print(f"cache           : {args.cache}")
    print(f"questions       : {n}")
    print(f"k / cap / floor : {args.k} / {args.cap} / {args.floor}")
    print(f"context         : {total/n:,.0f} chars/question = {total/n/4:,.0f} tok/turn (approx)")
    print(f"median items    : {statistics.median(counts):.1f}")
    print(f"retrieval       : {secs/n:.1f} s/question")


if __name__ == "__main__":
    main()

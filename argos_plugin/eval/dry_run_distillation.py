#!/usr/bin/env python
"""Dry-run the distillation pass against a copy of the live store.

Verifies:
- system_state table creates successfully on the live schema
- Distillation pass runs end-to-end (gating, clustering, LLM, proposals)
- Every payload.sources resolves to an existing memory_id
- Prints total calls and proposals for cost estimation

Usage:
    python dry_run_distillation.py --db <path>
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Set offline mode for HF models.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

# Ensure the plugin package is importable.
_plugin_dir = Path(__file__).resolve().parent.parent
if str(_plugin_dir) not in sys.path:
    sys.path.insert(0, str(_plugin_dir))

import argparse


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Dry-run distillation against a copy of the live store.",
    )
    parser.add_argument(
        "--db", required=True,
        help="Path to a COPY of hybrid_memory.duckdb (never the live DB).",
    )
    parser.add_argument("--min-new-records", type=int, default=5,
                        help="Override min_new_records for the dry-run (default 5).")
    parser.add_argument("--cooldown-hours", type=int, default=0,
                        help="Override cooldown_hours (default 0 = no cooldown).")
    parser.add_argument("--max-records", type=int, default=100,
                        help="Max records per run (default 100).")
    parser.add_argument("--max-calls", type=int, default=10,
                        help="Max LLM calls (default 10).")
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"Error: DB not found at {db_path}", file=sys.stderr)
        return 1

    from store import DuckDBMemoryStore
    from embeddings import LocalEmbedder
    from distillation import run_distillation, _STATE_KEY_LAST_RUN

    embedder = LocalEmbedder("BAAI/bge-small-en-v1.5")
    store = DuckDBMemoryStore(db_path, user_id="default_user", embedder=embedder)

    # Verify system_state table exists.
    try:
        rows = store.connection.execute("SELECT COUNT(*) FROM system_state").fetchone()
        print(f"system_state table: OK ({rows[0]} rows)")
    except Exception as e:
        print(f"system_state table: FAILED — {e}", file=sys.stderr)
        store.close()
        return 1

    # Check current state.
    last_run = store.get_state(_STATE_KEY_LAST_RUN)
    print(f"Current distillation_last_run: {last_run or '(never)'}")

    # Count eligible records.
    from distillation import _count_eligible_since
    eligible = _count_eligible_since(store, last_run)
    print(f"Eligible records since last_run: {eligible}")

    # Run distillation.
    print(f"\nRunning distillation (min_new={args.min_new_records}, "
          f"cooldown={args.cooldown_hours}h, max_records={args.max_records}, "
          f"max_calls={args.max_calls})...")

    report = run_distillation(
        store,
        min_new_records=args.min_new_records,
        cooldown_hours=args.cooldown_hours,
        max_records_per_run=args.max_records,
        max_calls=args.max_calls,
    )

    print(f"\n--- Distillation Report ---")
    print(json.dumps(report, indent=2, default=str))

    # Verify proposals if any were emitted.
    if report.get("ran"):
        candidates = store.list_candidates(status="pending")
        distillation_cands = [
            c for c in candidates if c.get("source") == "distillation"
        ]
        print(f"\n--- Proposal Verification ---")
        print(f"Total distillation candidates: {len(distillation_cands)}")
        verified = 0
        for c in distillation_cands:
            payload = c.get("payload")
            if isinstance(payload, str):
                payload = json.loads(payload)
            sources = payload.get("sources", [])
            for sid in sources:
                row = store.connection.execute(
                    "SELECT 1 FROM memory_records WHERE memory_id = ?",
                    [sid],
                ).fetchone()
                if row:
                    verified += 1
                else:
                    print(f"  WARNING: source {sid} does not resolve!")
        print(f"Source references verified: {verified}/{sum(len(json.loads(c['payload']).get('sources', [])) if isinstance(c.get('payload'), str) else len(c.get('payload', {}).get('sources', [])) for c in distillation_cands)}")

    store.close()
    print("\nDry-run complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

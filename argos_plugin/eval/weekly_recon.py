#!/usr/bin/env python3
"""weekly_recon.py — Monday 06:00 drift recon for the self-corpus gate.

Invoked by a no-agent Hermes cron job (Monday 06:00). Silent on PASS;
reports on drift/FAIL/error. Delivery: whatever the cron runner sends
(telegram:63941908).

Branch A (memory service DOWN): take a FRESH snapshot of the live store,
run the full gate on it vs the frozen baseline → the true live-store drift
sensor. Only fires in a maintenance window (service down / PC off
overnight → service dead at 06:00).

Branch B (memory service UP — the common case): bit-stability rerun of the
last frozen pair → catches environment drift (embedder cache, hardware,
deps) with zero risk to live data.

Every failure path prints a report ending with a `verdict:` line — never a
silent black hole.

Run with the Hermes venv python, `env -u PYTHONPATH`, `HF_HUB_OFFLINE=1`
(see ENGINEERING_NOTES -> self-corpus regression gate, gotchas).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

# Embedder loads MUST stay offline (a by-name HF HEAD-check hangs for
# minutes on this machine).  Set it before any plugin import touches HF.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

_PLUGIN_ROOT = Path(__file__).resolve().parents[1]
for p in (_PLUGIN_ROOT, Path(__file__).resolve().parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import run_gate  # noqa: E402
import snapshot_store  # noqa: E402

DEFAULT_LADDER = "5,20,96"


def _hermes_home() -> Path:
    home = os.environ.get("HERMES_HOME")
    if not home:
        raise SystemExit(
            "ERROR: HERMES_HOME is not set; pass --live-db/--endpoint explicitly.\n"
            "verdict: ERROR"
        )
    return Path(home)


def _report(msg: str) -> None:
    print(msg, flush=True)


def find_frozen_pair(snapshots_dir: Path) -> Optional[Path]:
    """Return the snapshot dir whose gate_baseline.json exists (the frozen pair)."""
    for m in snapshot_store.list_snapshots(snapshots_dir):
        d = snapshots_dir / m["snapshot_id"]
        if (d / "gate_baseline.json").exists():
            return d
    return None


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--live-db", required=True, help="Path to the live hybrid_memory.duckdb")
    parser.add_argument("--endpoint", required=True, help="Path to hybrid_memory_service.json")
    parser.add_argument("--snapshots-dir", required=True, help="eval/snapshots dir")
    parser.add_argument("--gold", required=True, help="Frozen gold JSONL (approved lines)")
    parser.add_argument("--ladder", default=DEFAULT_LADDER)
    parser.add_argument("--user-id", default="default_user")
    args = parser.parse_args(argv)

    snap_dir = Path(args.snapshots_dir)
    gold = Path(args.gold)
    ladder = [int(x.strip()) for x in args.ladder.split(",") if x.strip()]

    frozen = find_frozen_pair(snap_dir)
    if frozen is None:
        _report("ERROR: no frozen pair found (no snapshot has gate_baseline.json). "
                "Run the gate once manually to record the baseline.")
        _report("verdict: ERROR (no baseline)")
        return 2

    service_up = snapshot_store.service_running(Path(args.endpoint))

    if not service_up:
        # Branch A: fresh snapshot of the live store + full gate vs frozen baseline.
        try:
            manifest = snapshot_store.take_snapshot(
                Path(args.live_db), snap_dir, endpoint=Path(args.endpoint),
            )
        except Exception as exc:
            _report(f"ERROR: snapshot failed: {exc}")
            _report("verdict: ERROR")
            return 2
        newest = snap_dir / manifest["snapshot_id"]
        out_path = newest / "recon_scores.json"
        rc = run_gate.run_gate(
            newest, gold, out_path, compare_path=frozen / "gate_baseline.json",
            ladder=ladder, user_id=args.user_id,
        )
        _report(f"recon branch A: fresh snapshot {manifest['snapshot_id']} "
                f"({manifest['record_count']} records), gate rc={rc}")
        if rc == 0:
            _report("verdict: PASS")
            return 0
        if rc == 1:
            _report("DRIFT: fresh-snapshot gate FAILED vs frozen baseline — "
                    "live store drifted (details above). Blocks sync to live.")
            _report("verdict: FAIL")
            return 1
        _report("verdict: ERROR")
        return 2

    # Branch B: bit-stability rerun of the frozen pair (service up).
    out_path = frozen / "recon_bitstability.json"
    rc = run_gate.run_gate(
        frozen, gold, out_path, compare_path=frozen / "gate_baseline.json",
        ladder=ladder, user_id=args.user_id,
    )
    if rc == 0:
        # Silent on PASS.
        return 0
    if rc == 1:
        _report("DRIFT: frozen-pair rerun FAILED vs baseline — environment drift "
                "(embedder/cache/hardware). Inspect and re-baseline after fixing.")
        _report("verdict: FAIL")
        return 1
    _report(f"ERROR: gate error on frozen pair (rc={rc})")
    _report("verdict: ERROR")
    return 2


if __name__ == "__main__":
    sys.exit(main())
#!/usr/bin/env python3
"""ci_tier2_check.py — Tier 2 weekly full-gate preflight + drift report (#292).

The weekly job runs the REAL eval (run_gate.py over the frozen snapshot
+ gold_v1, ~91 min for 500-q — never per-PR). This script is the
scheduled job's preflight and drift reporter:

- ``--check`` (preflight / dry-report): verifies the runner-local
  artifacts exist and are coherent — snapshot manifest + db (sha256
  vs manifest), gold file (parseable JSONL, probe count), baseline
  scores (has overall recall/mrr). Exit 0 = the gate WOULD run; exit 2
  = artifacts missing/incoherent (the scheduled job fails loudly
  instead of silently skipping).
- ``--drift-history <dir>``: summarize recorded gate scores over time
  (recall@max-k / MRR per run) so drift is visible across weeks.

The gate itself (delta vs baseline, thresholds in eval/verdict.py) is
run by ``run_gate.py --compare <baseline>`` — reused, not rebuilt.

Usage (self-hosted runner, where the artifacts live):
    python eval/ci_tier2_check.py --check \
        --snapshot eval/snapshots/<id> \
        --gold eval/gold/gold_v1.jsonl \
        --baseline eval/snapshots/<id>/gate_baseline.json

    python eval/ci_tier2_check.py --drift-history eval/snapshots/<id>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def preflight(snapshot_dir: Path, gold_path: Path,
              baseline_path: Path) -> Tuple[int, Dict[str, Any]]:
    """Dry-report: verify the artifacts the weekly gate needs.

    Exit 0 = the gate would run; exit 2 = missing/incoherent artifact
    (named explicitly — never a silent skip).
    """
    report: Dict[str, Any] = {
        "check": "tier2-preflight",
        "snapshot": str(snapshot_dir),
        "gold": str(gold_path),
        "baseline": str(baseline_path),
        "problems": [],
    }
    problems: List[str] = report["problems"]

    # 1. Snapshot manifest + db + integrity.
    manifest_path = snapshot_dir / "manifest.json"
    if not manifest_path.exists():
        problems.append(f"snapshot manifest missing: {manifest_path}")
    else:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            db_name = str(manifest.get("db_filename",
                                       "hybrid_memory.duckdb"))
            db_path = snapshot_dir / db_name
            if not db_path.exists():
                problems.append(f"snapshot db missing: {db_path}")
            else:
                expected_sha = manifest.get("db_sha256")
                if expected_sha:
                    actual = _sha256_file(db_path)
                    if actual != expected_sha:
                        problems.append(
                            f"snapshot db sha mismatch (manifest "
                            f"{str(expected_sha)[:12]}… vs actual "
                            f"{actual[:12]}…)"
                        )
                report["snapshot_db"] = str(db_path)
                report["snapshot_db_bytes"] = db_path.stat().st_size
        except Exception as exc:
            problems.append(f"unreadable snapshot manifest: {exc}")

    # 2. Gold file: parseable JSONL with the required keys.
    if not Path(gold_path).exists():
        problems.append(f"gold file missing: {gold_path}")
    else:
        n_probes = 0
        try:
            for line in Path(gold_path).read_text(
                    encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                obj = json.loads(line)
                if "query" not in obj or "memory_id" not in obj:
                    problems.append(
                        f"gold line {n_probes + 1}: missing query/memory_id"
                    )
                    break
                n_probes += 1
            report["gold_probes"] = n_probes
        except Exception as exc:
            problems.append(f"gold file unparseable: {exc}")

    # 3. Baseline scores (the delta reference).
    if not baseline_path.exists():
        problems.append(f"gate baseline missing: {baseline_path}")
    else:
        try:
            baseline_scores = json.loads(
                baseline_path.read_text(encoding="utf-8"))
            if "mrr" not in (baseline_scores.get("overall") or {}):
                problems.append("baseline has no overall.mrr")
            else:
                report["baseline_overall"] = baseline_scores.get("overall")
        except Exception as exc:
            problems.append(f"baseline unreadable: {exc}")

    ok = not problems
    report["ok"] = ok
    return (0 if ok else 2), report


def drift_history(history_dir: Path) -> Dict[str, Any]:
    """Summarize recorded gate scores over time (drift tracking)."""
    entries: List[Dict[str, Any]] = []
    for p in sorted(history_dir.glob("gate_scores*.json")):
        try:
            scores = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        overall = scores.get("overall", {})
        ladder = scores.get("ladder") or [5, 20, 96]
        max_k = max(ladder)
        entry = {
            "file": p.name,
            "timestamp": scores.get("timestamp"),
            "probe_count": scores.get("probe_count"),
            f"recall@{max_k}": overall.get(f"recall@{max_k}"),
            "mrr": overall.get("mrr"),
        }
        if scores.get("gold_sha256"):
            entry["gold_sha256"] = scores["gold_sha256"][:12]
        entry["drift_vs_previous"] = None  # filled below
        entries.append(entry)
    # Delta vs the previous run (drift over time).
    for i in range(1, len(entries)):
        prev, cur = entries[i - 1], entries[i]
        key = next((k for k in cur if k.startswith("recall@")), None)
        if key and cur.get(key) is not None and prev.get(key) is not None:
            cur["drift_vs_previous"] = round(cur[key] - prev[key], 4)
    return {
        "runs": len(entries),
        "history": entries,
    }


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(
        description="Tier 2 weekly gate preflight + drift report (#292)",
    )
    parser.add_argument("--check", action="store_true",
                        help="Preflight: verify artifacts, dry-report.")
    parser.add_argument("--snapshot", default=None, type=Path)
    parser.add_argument("--gold", default=None, type=Path)
    parser.add_argument("--baseline", default=None, type=Path)
    parser.add_argument("--drift-history", default=None, type=Path,
                        help="Summarize gate_scores*.json drift in a dir.")
    args = parser.parse_args()

    if args.drift_history:
        print(json.dumps(drift_history(args.drift_history), indent=2))
        return 0

    if args.check:
        missing = [n for n in ("snapshot", "gold", "baseline")
                   if getattr(args, n) is None]
        if missing:
            print(f"ERROR: --check requires --{', --'.join(missing)}",
                  file=sys.stderr)
            return 2
        code, report = preflight(args.snapshot, args.gold, args.baseline)
        print(json.dumps(report, indent=2))
        return code

    print("nothing to do: pass --check or --drift-history")
    return 2


if __name__ == "__main__":
    sys.exit(main())

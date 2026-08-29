#!/usr/bin/env python3
"""run_gate.py — the self-corpus regression gate.

Runs the frozen (snapshot, gold set, seed 42, ladder 5,20,96) retrieval
measurement against a snapshot of the user's own store and emits a
scores JSON.  With ``--compare <baseline.json>`` it emits a PASS/FAIL
verdict:

  PASS if: no category recall@<max-k> drops > 1pp,
           overall recall@<max-k> drop <= 0.5pp,
           overall MRR drop <= 0.01.
  Else FAIL (exit code 1).

The first run on a frozen pair records the baseline (writes the file
passed to ``--compare``).  Any store/extractor/ranking change must pass
the gate before syncing to live.

Usage:
    python eval/run_gate.py --snapshot eval/snapshots/<id> --gold eval/gold/gold_v1.jsonl \
        --out gate_scores.json --compare eval/snapshots/<id>/gate_baseline.json

Run with the Hermes venv python, ``HF_HUB_OFFLINE=1``, and a clean
PYTHONPATH (terminal sessions inherit the Hermes env).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_PLUGIN_ROOT = Path(__file__).resolve().parents[1]
if str(_PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_ROOT))
_EVAL_DIR = Path(__file__).resolve().parent
if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))

import eval_self_corpus as esc  # noqa: E402
import build_gold  # noqa: E402
import verdict as _verdict_mod  # noqa: E402

DEFAULT_LADDER = "5,20,96"
DEFAULT_EMBEDDER = "BAAI/bge-small-en-v1.5"
DEFAULT_SEED = 42
MIN_APPROVED = 50
DEFAULT_OUT = "gate_scores.json"
DEFAULT_PROBE_TIMEOUT = 30  # seconds per probe; 0 = no timeout

# Verdict thresholds — delegated to the shared verdict module (#21).
# These constants are re-exported for backward compatibility; the single
# source of truth is eval/verdict.py.
CATEGORY_RECALL_PP = _verdict_mod.CATEGORY_RECALL_PP
OVERALL_RECALL_PP = _verdict_mod.OVERALL_RECALL_PP
OVERALL_MRR = _verdict_mod.OVERALL_MRR


def _mrr(rank: Optional[int]) -> float:
    return 1.0 / rank if rank else 0.0


def score_probe(
    store: Any,
    gold_line: Dict[str, Any],
    ladder: List[int],
) -> Dict[str, Any]:
    """Run retrieval for one gold query; return per-window hits + rank."""
    query = gold_line["query"]
    target_id = gold_line["memory_id"]
    max_k = max(ladder)
    results = store.search(query, limit=max_k, suppress_retrieval=True)
    result_ids = [r.memory_id for r in results]
    result_contents = {r.memory_id: r.content for r in results}
    chain_ids = esc._build_chain_set(store, target_id)
    per_window = {
        str(k): esc.is_hit(gold_line, result_ids[:k], result_contents, chain_ids)
        for k in ladder
    }
    rank: Optional[int] = None
    for i, rid in enumerate(result_ids):
        if rid == target_id or rid in chain_ids:
            rank = i + 1
            break
    return {"per_window": per_window, "rank": rank}


def _score_probe_with_timeout(
    store: Any,
    gold_line: Dict[str, Any],
    ladder: List[int],
    timeout: float,
) -> Dict[str, Any]:
    """Run score_probe with a per-probe timeout (#21).

    If the probe doesn't complete within *timeout* seconds, returns a
    degenerate result (all misses, no rank) so one hung embedder/query
    can't stall the gate indefinitely.
    """
    if timeout <= 0:
        return score_probe(store, gold_line, ladder)
    import threading
    result_box: List[Any] = []
    def _worker():
        try:
            result_box.append(score_probe(store, gold_line, ladder))
        except Exception as exc:
            result_box.append(exc)
    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join(timeout=timeout)
    if t.is_alive():
        # Timed out — return a degenerate miss.
        logger.warning(
            "Probe timed out after %.1fs (query=%r, memory_id=%s)",
            timeout, gold_line.get("query", "")[:60], gold_line.get("memory_id", "?"),
        )
        return {
            "per_window": {str(k): False for k in ladder},
            "rank": None,
            "_timeout": True,
        }
    if result_box and isinstance(result_box[0], Exception):
        raise result_box[0]
    if result_box:
        return result_box[0]
    # Thread died without producing a result.
    return {"per_window": {str(k): False for k in ladder}, "rank": None}


def _summarize(probe_results: List[Dict[str, Any]], ladder: List[int]) -> Dict[str, float]:
    n = len(probe_results)
    out: Dict[str, float] = {}
    for k in ladder:
        hits = sum(1 for p in probe_results if p["per_window"].get(str(k)))
        out[f"recall@{k}"] = round(hits / n, 4) if n else 0.0
    out["mrr"] = round(sum(_mrr(p["rank"]) for p in probe_results) / n, 4) if n else 0.0
    return out


def compute_scores(
    probe_results: List[Dict[str, Any]],
    ladder: List[int],
    gold_lines: List[Dict[str, Any]],
    snapshot_id: str,
    gold_sha: str,
    config_hash: str,
    seed: int,
) -> Dict[str, Any]:
    """Aggregate per-probe results into the gate scores JSON."""
    by_category: Dict[str, List[Dict[str, Any]]] = {}
    by_template: Dict[str, List[Dict[str, Any]]] = {}
    for p, g in zip(probe_results, gold_lines):
        by_category.setdefault(g.get("category") or "context_note", []).append(p)
        by_template.setdefault(g.get("template") or "direct", []).append(p)
    return {
        "snapshot_id": snapshot_id,
        "gold_sha256": gold_sha,
        "seed": seed,
        "ladder": list(ladder),
        "config_hash": config_hash,
        "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "probe_count": len(probe_results),
        "overall": _summarize(probe_results, ladder),
        "by_category": {
            c: _summarize(ps, ladder) for c, ps in sorted(by_category.items())
        },
        "by_template": {
            t: _summarize(ps, ladder) for t, ps in sorted(by_template.items())
        },
    }


def gate_verdict(current: Dict[str, Any], baseline: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Compare current scores vs baseline; return (pass, failure reasons).

    Delegates to the shared verdict module (#21) so run_gate and
    eval_self_corpus --baseline agree on all PASS/FAIL cases.
    """
    return _verdict_mod.gate_verdict(current, baseline)


def run_gate(
    snapshot_dir: Path,
    gold_path: Path,
    out_path: Path,
    compare_path: Optional[Path] = None,
    ladder: Optional[List[int]] = None,
    embedder_model: str = DEFAULT_EMBEDDER,
    user_id: str = "default_user",
    seed: int = DEFAULT_SEED,
    probe_timeout: float = DEFAULT_PROBE_TIMEOUT,
) -> int:
    """Run the gate; return 0 PASS / 1 FAIL / 2 error."""
    ladder = ladder or [5, 20, 96]
    snapshot_dir = Path(snapshot_dir)
    manifest_path = snapshot_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"ERROR: snapshot dir missing manifest.json: {snapshot_dir}", file=sys.stderr)
        return 2
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"ERROR: unreadable snapshot manifest: {exc}", file=sys.stderr)
        return 2
    db_path = snapshot_dir / str(manifest.get("db_filename", "hybrid_memory.duckdb"))
    if not db_path.exists():
        print(f"ERROR: snapshot db missing: {db_path}", file=sys.stderr)
        return 2

    # Snapshot integrity: the db must match the manifest sha256.
    import snapshot_store
    actual = snapshot_store._sha256_file(db_path)
    if actual != manifest.get("db_sha256"):
        print(
            f"ERROR: snapshot integrity check failed ({db_path.name} sha256 "
            f"{actual[:16]}… != manifest {str(manifest.get('db_sha256'))[:16]}…)",
            file=sys.stderr,
        )
        return 2

    gold_lines = [l for l in build_gold.load_gold(gold_path) if l.get("status") == "approved"]
    if len(gold_lines) < MIN_APPROVED:
        print(
            f"ERROR: only {len(gold_lines)} approved gold queries (need >= {MIN_APPROVED}). "
            "Review eval/gold/*.jsonl and freeze it first.",
            file=sys.stderr,
        )
        return 2
    gold_sha = build_gold.gold_sha256(gold_lines)

    # The store's init runs writes (CREATE/ALTER/backfill) — run against a
    # temp copy so the snapshot stays pristine.
    tmpdir = Path(tempfile.mkdtemp(prefix="gate_"))
    copy = tmpdir / db_path.name
    shutil.copy2(db_path, copy)
    wal = Path(str(db_path) + ".wal")
    if wal.exists():
        shutil.copy2(wal, Path(str(copy) + ".wal"))
    try:
        if embedder_model:
            from embeddings import LocalEmbedder, _resolve_embedding_model_path
            resolved = _resolve_embedding_model_path(embedder_model, hermes_home=None)
            embedder = LocalEmbedder(resolved)
        else:
            embedder = None
        from store import DuckDBMemoryStore
        store = DuckDBMemoryStore(copy, user_id=user_id, embedder=embedder)
        try:
            cfg_hash = esc._config_hash(db_path, embedder_model or "none")
            probe_results = [
                _score_probe_with_timeout(store, line, ladder, probe_timeout)
                for line in gold_lines
            ]
        finally:
            store.close()
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    scores = compute_scores(
        probe_results, ladder, gold_lines,
        snapshot_id=str(manifest.get("snapshot_id", "?")),
        gold_sha=gold_sha, config_hash=cfg_hash, seed=seed,
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(scores, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"=== Gate run ({scores['probe_count']} probes, snapshot {scores['snapshot_id']}) ===")
    print(f"gold_sha256: {gold_sha}")
    print(f"config_hash: {cfg_hash}")
    for k in ladder:
        print(f"  overall recall@{k}: {scores['overall'][f'recall@{k}']*100:.1f}%")
    print(f"  overall MRR: {scores['overall']['mrr']:.4f}")
    print(f"scores written to {out_path}")

    if compare_path is None:
        return 0
    baseline_path = Path(compare_path)
    if not baseline_path.exists():
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_path.write_text(json.dumps(scores, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n=== BASELINE RECORDED ({baseline_path}) ===")
        return 0
    try:
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"ERROR: unreadable baseline: {exc}", file=sys.stderr)
        return 2
    if baseline.get("gold_sha256") != gold_sha:
        print(f"WARNING: gold set sha256 changed since baseline ({baseline.get('gold_sha256')} -> {gold_sha}); verdict may be unreliable.")
    ok, failures = gate_verdict(scores, baseline)
    if ok:
        print("\n=== VERDICT: PASS ===")
        return 0
    print("\n=== VERDICT: FAIL (blocks sync to live) ===")
    for f in failures:
        print(f"  - {f}")
    return 1


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Self-corpus regression gate (fixed snapshot + gold set).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--snapshot", required=True, help="Snapshot dir containing hybrid_memory.duckdb + manifest.json.")
    parser.add_argument("--gold", required=True, help="Validated gold JSONL (approved lines only).")
    parser.add_argument("--out", default=DEFAULT_OUT, help=f"Scores JSON (default '{DEFAULT_OUT}').")
    parser.add_argument("--compare", default="", help="Baseline JSON path. Missing file = record baseline; existing = PASS/FAIL verdict.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"Fixed sampling seed (default {DEFAULT_SEED}).")
    parser.add_argument("--ladder", default=DEFAULT_LADDER, help=f"Recall windows (default '{DEFAULT_LADDER}').")
    parser.add_argument("--embedder-model", default=DEFAULT_EMBEDDER, help=f"Embedder (default '{DEFAULT_EMBEDDER}'); empty = text search only.")
    parser.add_argument("--threads", type=int, default=4, help="Accepted for CLI parity; keep modest (default 4).")
    parser.add_argument("--user-id", default="default_user")
    parser.add_argument("--probe-timeout", type=float, default=DEFAULT_PROBE_TIMEOUT,
                        help=f"Per-probe timeout in seconds (default {DEFAULT_PROBE_TIMEOUT}; 0 = no timeout).")
    args = parser.parse_args(argv)

    try:
        ladder = [int(x.strip()) for x in args.ladder.split(",") if x.strip()]
    except ValueError:
        ladder = [5, 20, 96]
    compare = Path(args.compare) if args.compare else None
    return run_gate(
        Path(args.snapshot), Path(args.gold), Path(args.out),
        compare_path=compare, ladder=ladder,
        embedder_model=args.embedder_model, user_id=args.user_id, seed=args.seed,
        probe_timeout=args.probe_timeout,
    )


if __name__ == "__main__":
    sys.exit(main())

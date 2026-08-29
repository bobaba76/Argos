#!/usr/bin/env python3
"""probe_rank1_loss.py — measure how often RRF drops a strong semantic rank-1 (#38).

The regression guard exists because RRF can bury an arm's clear rank-1
when the other arm ranks it poorly. Before trusting the guard, measure
the failure on a real store slice: for each query, run the vector and
text arms independently, compute the raw (unguarded) RRF fusion, and
count how often the vector arm's clear rank-1 (margin >= 1.5x its own
rank-2) is absent from the fused top-3 — then verify the guard rescues it.

Usage:
    python eval/probe_rank1_loss.py --db <store.duckdb> [--gold gold_v1.jsonl]
        [--embedder-model BAAI/bge-small-en-v1.5] [--limit 96] [--top-k 3]

Exit code 0 with a verdict line; run offline (HF_HUB_OFFLINE=1) with the
Hermes venv python like the gate.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

os.environ.setdefault("HF_HUB_OFFLINE", "1")

_PLUGIN_ROOT = Path(__file__).resolve().parents[2]
if str(_PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_ROOT))

DEFAULT_LADDER = "5,20,96"


def _load_gold_queries(gold_path: Path, max_queries: int) -> List[Dict[str, Any]]:
    """Approved gold lines, each with query + memory_id."""
    queries: List[Dict[str, Any]] = []
    for line in open(gold_path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if obj.get("status") == "approved" and obj.get("query"):
            queries.append(obj)
            if len(queries) >= max_queries:
                break
    return queries


def probe_rank1_loss(
    store,
    queries: List[Dict[str, Any]],
    limit: int = 96,
    top_k: int = 3,
) -> Dict[str, Any]:
    """Run the probe: count vector rank-1s lost by raw RRF, rescued by guard.

    Uses the store's real vector/text arms and the real _rrf_fuse guard.
    Reports both the guard's contract (clear-margin rank-1, ratio >= 1.5)
    and the underlying phenomenon (vector rank-1 lost regardless of
    margin), so the issue's "accept or adjust" decision has data.
    """
    from store import DuckDBMemoryStore

    n_clear_rank1 = 0
    n_lost_by_raw_rrf = 0
    n_rescued_by_guard = 0
    n_all_rank1 = 0
    n_all_rank1_lost = 0
    n_margin12_lost = 0
    examples: List[Dict[str, Any]] = []

    for gold in queries:
        query = gold["query"]
        try:
            embedder = getattr(store, "embedder", None)
            if embedder is None or not hasattr(embedder, "embed"):
                continue
            emb = embedder.embed(query)
            if not emb:
                continue
            vector_results = store._vector_search_raw(emb, limit, set())
            text_results = store._text_search_raw(query, limit, set())
        except Exception:
            continue
        if len(vector_results) < 2:
            continue
        s1 = float(getattr(vector_results[0], "similarity", 0.0) or 0.0)
        s2 = float(getattr(vector_results[1], "similarity", 0.0) or 0.0)

        # Raw RRF without the guard, then the guarded fusion — compare
        # where the rank-1 landed in each. Each call must get its own
        # copies: _rrf_fuse writes the fused score onto the arm records'
        # ``similarity``, so a second call on the same objects would read
        # fused scores instead of arm scores in the guard margin check.
        import copy
        raw_fused = DuckDBMemoryStore._rrf_fuse(
            copy.deepcopy(vector_results), copy.deepcopy(text_results),
            enable_rank1_guard=False,
        )
        guarded_fused = DuckDBMemoryStore._rrf_fuse(
            copy.deepcopy(vector_results), copy.deepcopy(text_results),
        )
        raw_top_k = [r.memory_id for r in raw_fused[:top_k]]
        guarded_top_k = [r.memory_id for r in guarded_fused[:top_k]]
        rid = vector_results[0].memory_id

        # Underlying phenomenon: vector rank-1 lost by raw RRF, margin-free.
        n_all_rank1 += 1
        if rid not in raw_top_k:
            n_all_rank1_lost += 1
            if s2 > 0 and s1 / s2 >= 1.2:
                n_margin12_lost += 1

        # Guard's contract: clear-margin rank-1 (ratio >= 1.5).
        clear = s1 > 0.0 and (s2 <= 0.0 or s1 >= s2 * 1.5)
        if not clear:
            continue
        n_clear_rank1 += 1
        if rid in raw_top_k:
            continue  # survived raw RRF — nothing for the guard to do
        n_lost_by_raw_rrf += 1
        # With the guard, the rank-1 is guaranteed in the top-k.
        if rid in guarded_top_k:
            n_rescued_by_guard += 1
        if len(examples) < 5:
            examples.append({
                "query": query[:120],
                "rank1_id": rid,
                "raw_top_k": raw_top_k,
            })

    loss_rate = (n_lost_by_raw_rrf / n_clear_rank1) if n_clear_rank1 else 0.0
    all_loss_rate = (n_all_rank1_lost / n_all_rank1) if n_all_rank1 else 0.0
    return {
        "queries_scanned": len(queries),
        "queries_with_clear_vector_rank1": n_clear_rank1,
        "lost_by_raw_rrf": n_lost_by_raw_rrf,
        "rescued_by_guard": n_rescued_by_guard,
        "loss_rate": round(loss_rate, 4),
        "all_rank1": n_all_rank1,
        "all_rank1_lost": n_all_rank1_lost,
        "all_rank1_loss_rate": round(all_loss_rate, 4),
        "margin12_lost": n_margin12_lost,
        "examples": examples,
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", required=True, help="Path to a hybrid_memory.duckdb store")
    parser.add_argument("--gold", required=True, help="Gold JSONL with approved queries")
    parser.add_argument("--embedder-model", default="BAAI/bge-small-en-v1.5",
                        help="Embedder; empty = text-only (probe needs vector arm)")
    parser.add_argument("--limit", type=int, default=96, help="Per-arm candidate limit")
    parser.add_argument("--top-k", type=int, default=3, help="Fused top-k window")
    parser.add_argument("--max-queries", type=int, default=200)
    args = parser.parse_args(argv)

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"ERROR: db not found: {db_path}", file=sys.stderr)
        return 2
    gold_path = Path(args.gold)
    if not gold_path.exists():
        print(f"ERROR: gold not found: {gold_path}", file=sys.stderr)
        return 2

    # Run against a temp copy so the store stays pristine (init runs writes).
    tmpdir = Path(tempfile.mkdtemp(prefix="probe_rank1_"))
    copy = tmpdir / db_path.name
    shutil.copy2(db_path, copy)
    try:
        embedder = None
        if args.embedder_model:
            from embeddings import LocalEmbedder, _resolve_embedding_model_path
            resolved = _resolve_embedding_model_path(args.embedder_model, hermes_home=None)
            embedder = LocalEmbedder(resolved)
        from store import DuckDBMemoryStore
        store = DuckDBMemoryStore(copy, user_id="default_user", embedder=embedder)
        try:
            queries = _load_gold_queries(gold_path, args.max_queries)
            if not queries:
                print("ERROR: no approved gold queries loaded", file=sys.stderr)
                return 2
            stats = probe_rank1_loss(store, queries, limit=args.limit, top_k=args.top_k)
        finally:
            store.close()
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    print("=== Rank-1 loss probe (#38) ===")
    print(f"queries scanned:          {stats['queries_scanned']}")
    print(f"clear vector rank-1:      {stats['queries_with_clear_vector_rank1']}")
    print(f"lost by raw RRF:          {stats['lost_by_raw_rrf']}")
    print(f"rescued by guard:         {stats['rescued_by_guard']}")
    print(f"loss rate (clear):        {stats['loss_rate']*100:.1f}%")
    print(f"all vector rank-1:        {stats['all_rank1']}")
    print(f"all rank-1 lost by RRF:   {stats['all_rank1_lost']} "
          f"({stats['all_rank1_loss_rate']*100:.1f}%)")
    print(f"  of which ratio>=1.2:    {stats['margin12_lost']}")
    if stats["examples"]:
        print("\nexamples (query | raw top-k):")
        for ex in stats["examples"]:
            print(f"  {ex['query'][:60]} | {ex['raw_top_k']}")
    print(f"\nverdict: clear-loss={stats['loss_rate']*100:.1f}% "
          f"({stats['lost_by_raw_rrf']}/{stats['queries_with_clear_vector_rank1']}), "
          f"all-rank1-loss={stats['all_rank1_loss_rate']*100:.1f}% "
          f"({stats['all_rank1_lost']}/{stats['all_rank1']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

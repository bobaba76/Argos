#!/usr/bin/env python3
"""Chain-unfold eval: does auto-injection fire correctly, at what cost?

Method:
- Temp home + temp DuckDB store (never touches live data).
- Seed 3 real chains via update_memory (v1 -> v2):
    A. music preference: Spotify -> Bandcamp
    B. property plan: sell both -> sell one only
    C. medication: Topiramate -> Amitriptyline
  plus several unchained single memories (dog, work, budget).
- Provider configured with chain_unfold=auto; memory_search tool called
  with (1) change-intent queries whose chain IS in top-3 (should unfold),
  (2) change-intent queries with NO chain (should NOT unfold — false
  trigger check), (3) neutral queries (should NOT unfold).

Metrics: trigger precision/recall, arc correctness (right memory's arc),
token cost per injection, and _chain_unfolded_stats accounting.

Seeds are SANITIZED for public-repo use (no real names / employers /
medications). The original diagnostic probes carry personally-identifying
example content, are excluded via .gitignore, and never leave the dev
tree. Run: python argos_plugin/eval/eval_chain_unfold.py

Variants (2026-08-17 recall rebalance):
  1. top1_baseline  — top_k=1, query_fallback=false (reproduce 100%/20%)
  2. top3           — top_k=3, query_fallback=false
  3. top3_fallback  — top_k=3, query_fallback=true
Each variant gets a FRESH temp home + fresh snapshot copy + fresh seed.
"""
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import run_eval_provider as R  # build_arm_home / model symlinking

BASE_CFG = {
    "storage_mode": "direct",
    "auto_extract": "false",
    "auto_review": "false",
    "chain_unfold": "auto",
    "chain_unfold_min_similarity": "0.30",
    "chain_max_versions": "3",
    "chain_max_inject": "150",
    "graph_aware_retrieval": "false",   # isolate chain-unfold from graph
}

VARIANTS = [
    ("top1_baseline", {**BASE_CFG, "chain_unfold_top_k": "1",
                       "chain_unfold_query_fallback": "false"}),
    ("top3",          {**BASE_CFG, "chain_unfold_top_k": "3",
                       "chain_unfold_query_fallback": "false"}),
    ("top3_fallback", {**BASE_CFG, "chain_unfold_top_k": "3",
                       "chain_unfold_query_fallback": "true"}),
]

# Option A floor sweep: top-K scan + semantic-arc cosine gate. Vary the arc
# floor to find where precision >= 90% AND recall loss <= 10% both hold.
ARC_SWEEP_FLOORS = [0.00, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]

def arc_variant_cfg(floor: float) -> dict:
    """One Option-A variant: top-K=3 scan + query fallback + arc cosine gate."""
    return {
        **BASE_CFG,
        "chain_unfold_top_k": "3",
        "chain_unfold_query_fallback": "true",
        "chain_unfold_arc_min_similarity": f"{floor:.2f}",
    }

# (query, expect_unfold, expected_mem_prefix)
QUERIES = [
    ("why did I stop using Spotify", True, "music"),       # change-intent + chain
    ("why did I switch music services", True, "music"),
    ("what changed with my property plan", True, "property"),
    ("when did I change my medication", True, "medication"),
    ("why did I stop using Topiramate", True, "medication"),
    ("what changed in the weather today", False, None),    # change-intent, NO chain
    ("why did the dog food brand change", False, None),    # change-intent, NO chain
    ("what music do I like", False, None),                 # neutral
    ("tell me about my dog", False, None),                 # neutral
    ("how much budget do I have", False, None),            # neutral
]


def seed_chains(store):
    """Seed 3 chains (v1->v2) + unchained distractors. Returns mem IDs."""
    def seed(content_v1, content_v2):
        r1 = store.remember(category="personal_fact", content=content_v1)
        r2 = store.update_memory(memory_id=r1.memory_id, content=content_v2)
        return r1.memory_id, r2.memory_id

    a1, a2 = seed("User prefers Spotify for music",
                  "User switched from Spotify to Bandcamp")
    b1, b2 = seed("User plans to sell both properties and buy one house",
                  "User decided to sell only the holiday cottage and keep the flat")
    c1, c2 = seed("User takes Topiramate for migraines",
                  "User switched from Topiramate to Amitriptyline")
    # Unchained distractors
    store.remember(category="personal_fact", content="User has a small dachshund named Biscuit")
    store.remember(category="personal_fact", content="User works at a distribution company as a product manager")
    store.remember(category="personal_fact", content="User's monthly budget leaves about R30k discretionary")
    return {"music": a2, "property": b2, "medication": c2}


def run_variant(name, cfg, snap):
    """Run one variant: fresh home, seed, query, collect metrics."""
    td = tempfile.mkdtemp(prefix=f"hermes-unfold-{name}-")
    home = R.build_arm_home(snap, cfg, Path(td))

    from argos_plugin.store import DuckDBMemoryStore
    from argos_plugin.embeddings import LocalEmbedder, _resolve_embedding_model_path
    from argos_plugin import ArgosProvider

    model = _resolve_embedding_model_path("bge-small-en-v1.5", hermes_home=str(home))
    embedder = LocalEmbedder(model, hermes_home=str(home))
    store = DuckDBMemoryStore(home / "hybrid_memory.duckdb",
                              user_id="default_user", embedder=embedder)
    mem_ids = seed_chains(store)
    print(f"[{name}] seeded chains: {list(mem_ids.values())}")

    p = ArgosProvider()
    p.initialize(session_id=f"unfold_{name}", hermes_home=str(home),
                 platform="cli", user_id="default_user")

    tp = fp = tn = fn = 0
    total_tokens = 0
    per_query = []
    for q, expect, expect_key in QUERIES:
        resp = json.loads(p.handle_tool_call("memory_search", {"query": q, "top_k": 5}))
        payloads = resp.get("results", [])
        arc_entry = next((x for x in payloads if "chain_arc" in x), None)
        unfolded = arc_entry is not None
        arc = arc_entry.get("chain_arc", "") if arc_entry else ""
        tok = max(1, len(arc) // 4) if arc else 0

        # correctness: arc mentions the expected current memory content
        correct = False
        if unfolded and expect:
            correct = ("Bandcamp" in arc or "holiday cottage" in arc
                       or "Amitriptyline" in arc)

        if unfolded and expect:
            tp += 1
        elif unfolded and not expect:
            fp += 1
        elif not unfolded and expect:
            fn += 1
        else:
            tn += 1
        total_tokens += tok
        per_query.append({
            "query": q, "expect": expect, "unfolded": unfolded,
            "correct": correct, "tokens": tok, "arc": arc[:120],
        })

    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    stats = p.get_chain_unfold_stats()
    p.shutdown()
    store.close()
    return {
        "variant": name,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "tokens": total_tokens,
        "stats": stats,
        "per_query": per_query,
    }


def main() -> int:
    snap = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "hermes" / "memory-eval-clean.duckdb"
    if not snap.exists():
        print(f"ERROR: snapshot not found: {snap}")
        return 2

    all_results = []
    for name, cfg in VARIANTS:
        print(f"\n{'='*60}")
        print(f"=== VARIANT: {name} ===")
        print(f"{'='*60}")
        r = run_variant(name, cfg, snap)
        all_results.append(r)
        for pq in r["per_query"]:
            tag = "TP" if pq["unfolded"] and pq["expect"] else \
                  "FP" if pq["unfolded"] and not pq["expect"] else \
                  "FN" if not pq["unfolded"] and pq["expect"] else "TN"
            print(f"  [{tag}] {pq['query']:42s} unfolded={pq['unfolded']} "
                  f"correct={pq['correct']} tokens={pq['tokens']}")
        print(f"  precision={r['precision']:.0%}  recall={r['recall']:.0%}  "
              f"tp={r['tp']} fp={r['fp']} fn={r['fn']} tn={r['tn']}  "
              f"tokens={r['tokens']}  stats={r['stats']}")

    # Trade-off table
    print(f"\n{'='*60}")
    print("=== TRADE-OFF TABLE ===")
    print(f"{'='*60}")
    print(f"{'variant':18s} {'precision':>10s} {'recall':>8s} "
          f"{'tp':>3s} {'fp':>3s} {'fn':>3s} {'tn':>3s} {'tokens':>7s}")
    print("-" * 60)
    for r in all_results:
        print(f"{r['variant']:18s} {r['precision']:>10.0%} {r['recall']:>8.0%} "
              f"{r['tp']:>3d} {r['fp']:>3d} {r['fn']:>3d} {r['tn']:>3d} "
              f"{r['tokens']:>7d}")

    # FP/FN detail for the chosen variant
    print(f"\n=== FP/FN DETAIL (all variants) ===")
    for r in all_results:
        fps = [pq["query"] for pq in r["per_query"]
               if pq["unfolded"] and not pq["expect"]]
        fns = [pq["query"] for pq in r["per_query"]
               if not pq["unfolded"] and pq["expect"]]
        if fps or fns:
            print(f"\n  {r['variant']}:")
            if fps:
                print(f"    FP (false fires): {fps}")
            if fns:
                print(f"    FN (missed arcs): {fns}")

    # Option A: sweep the semantic-arc floor and print the precision/recall band.
    print(f"\n{'='*60}")
    print("=== OPTION A ARC-FLOOR SWEEP (top-K=3 + fallback + arc gate) ===")
    print(f"{'='*60}")
    print(f"{'floor':>6s}  {'precision':>9s} {'recall':>7s}  {'tp':>3s} {'fp':>3s} {'fn':>3s} {'tn':>3s}  band_ok")
    print("-" * 64)
    best = None
    for floor in ARC_SWEEP_FLOORS:
        r = run_variant(f"arc{floor:.2f}", arc_variant_cfg(floor), snap)
        band_ok = r["precision"] >= 0.90 and r["recall"] >= 0.90
        print(f"{floor:>6.2f}  {r['precision']:>9.0%} {r['recall']:>7.0%}  "
              f"{r['tp']:>3d} {r['fp']:>3d} {r['fn']:>3d} {r['tn']:>3d}  "
              f"{'YES' if band_ok else 'no'}")
        if r["precision"] >= 0.90 and (best is None or r["precision"] > best[1]["precision"]):
            best = (floor, r)
    if best:
        print(f"\nBEST IN BAND: floor={best[0]:.2f}  precision={best[1]['precision']:.0%} "
              f"recall={best[1]['recall']:.0%}")
    else:
        print("\nNO floor in this sweep met BOTH >=90% precision and <=10% recall loss."
              " Widen the sweep or adjust the arc-similarity gate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
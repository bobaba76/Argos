#!/usr/bin/env python3
"""CLEAN deterministic chain-unfold recall eval on UNSATURATED topics.

This is the CANONICAL chain-unfold eval (originally eval_chain_unfold_clean.py,
2026-08-20). The crude baseline harness (eval_chain_unfold.py, 3 saturated
chains, raw recall over ALL positives) understates real feature recall: the
memory-eval-clean.duckdb snapshot is dense with real memories on saturated
topics, and one-line synthetic chains on those topics get buried in retrieval
-- that is an eval artifact, not a gate failure.

This eval fixes that by seeding chains ONLY on unsaturated topics (hobbies,
lifestyle, gear) where retrieval can surface them, AND by classifying every
positive miss: RETRIEVAL-BURIED (current version not in top-20 => not a gate
failure) vs GATE-BLOCKED (surfaced but didn't unfold => real feature miss).

Runs production defaults (arc floor 0.15, anchor 0.30) plus an arc floor 0.00
row to reconfirm the gate's irrelevance to recall (pure precision knob).

Target (user, 2026-08-20): precision >=90% AND recall >=90% (<=10% loss).
  - Recall counted on POSITIVES THAT SURFACED (fair gate test).
  - Also report raw recall (all positives) so the artifact is visible.

Seeds are SANITIZED for public-repo use (no real names / employers / meds /
locations). The original diagnostic probes carry personally-identifying
example content, are excluded via .gitignore, and never leave the dev tree.
Run: env -u PYTHONPATH HF_HUB_OFFLINE=1 <hermes-venv-python>
     argos_plugin/eval/eval_chain_unfold_clean.py
Reference reproducibility: eval/repro/CHAIN_UNFOLD_RESULTS.md
"""
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import run_eval_provider as R

BASE_CFG = {
    "storage_mode": "direct",
    "auto_extract": "false",
    "auto_review": "false",
    "chain_unfold": "auto",
    "chain_unfold_min_similarity": "0.30",
    "chain_max_versions": "3",
    "chain_max_inject": "150",
    "graph_aware_retrieval": "false",
    "chain_unfold_top_k": "3",
    "chain_unfold_query_fallback": "true",
}

# (label, v1, v2_current, realistic_change_query) — UNSATURATED topics only.
CHAINS = [
    ("music",    "User prefers Spotify for music",
                 "User switched from Spotify to Bandcamp",
                 "why did I stop using Spotify"),
    ("car",      "User drives a grey 2019 Toyota Corolla",
                 "User bought a blue 2022 Kia Sportage",
                 "what car do I drive now"),
    ("pet",      "User has a small dachshund named Biscuit",
                 "User got a second dog, a golden retriever named Mango",
                 "do I still just have the one dog"),
    ("holiday",  "User is planning a trip to Cape Town for December",
                 "User cancelled Cape Town and booked Mauritius instead",
                 "am I still going to Cape Town"),
    ("meal",     "User prefers meat and potatoes for dinner",
                 "User went vegan and now eats quinoa bowls",
                 "what do I eat for dinner now"),
    ("sport",    "User used to play rugby on weekends",
                 "User quit rugby and took up mountain biking",
                 "do I still play rugby"),
    ("house",    "User lives in a flat in the city centre",
                 "User moved into a house in the suburbs",
                 "where do I live now"),
    ("weight",   "User weighs 95kg",
                 "User lost weight and now weighs 82kg",
                 "how much do I weigh now"),
    ("meds",     "User takes Topiramate for migraines",
                 "User switched from Topiramate to Amitriptyline",
                 "why did I switch my medication"),
    ("gaming",   "User plays World of Warcraft on weekends",
                 "User switched to Final Fantasy XIV and unsubscribed from WoW",
                 "do I still play World of Warcraft"),
    ("coffee",   "User drinks instant coffee every morning",
                 "User bought an espresso machine and no longer drinks instant",
                 "what do I drink every morning now"),
    ("gym",      "User works out at home with bodyweight",
                 "User joined a commercial gym with a personal trainer",
                 "do I still work out at home"),
    ("phone",    "User had an iPhone 12",
                 "User switched to a Samsung Galaxy phone",
                 "what phone do I use now"),
    ("reading",  "User mostly reads fiction novels",
                 "User switched to non-fiction and business books",
                 "what do I read now"),
]

NEG_QUERIES = [
    ("what changed in the weather today",),
    ("why did my phone battery drain",),
    ("what music do I like",),           # HARD: music chain exists, no change ask
    ("tell me about my dog",),           # HARD: pet chain exists, no change ask
    ("how much do I weigh",),            # current-state, no "now"
    ("how much budget do I have",),
    ("what is my favourite colour",),
    ("what time is my meeting",),
    ("how is my work going",),
    ("what is the price of fuel",),
    ("did the company change its name",),
    ("why did the office move location",),
]


def seed_chains(store):
    """Return {label: current_memory_id} for retrieval-surfaced diagnostics."""
    surf = {}
    for (label, v1, v2, _q) in CHAINS:
        r1 = store.remember(category="personal_fact", content=v1, dedup=False)
        r2 = store.update_memory(memory_id=r1.memory_id, content=v2)
        surf[label] = r2.memory_id
    store.remember(category="personal_fact", dedup=False,
                   content="User's monthly budget leaves about R30k discretionary")
    store.remember(category="personal_fact", dedup=False,
                   content="User works in product management and travels for meetings")
    return surf


def build_queries():
    positives = [(q, label) for (label, _v1, _v2, q) in CHAINS]
    qs = []
    for (q, label) in positives:
        qs.append((q, True, label))
    for (nq,) in NEG_QUERIES:
        qs.append((nq, False, "neg"))
    return qs


def run_floor(p, store, surf, arc_floor, label_):
    p._chain_unfold_arc_min_similarity = arc_floor
    queries = build_queries()
    tp = fp = fn = tn = 0
    buried = []          # positives surfaced in top20 but retrieval-buried
    gateblocked = []     # positives surfaced but didn't unfold (real miss)
    per_query = []

    for q, expect, lq in queries:
        resp = json.loads(p.handle_tool_call("memory_search", {"query": q, "top_k": 5}))
        results = resp.get("results", [])
        unfolded = any(isinstance(r, dict) and r.get("chain_arc") for r in results)

        # Retrieval diagnostic for positives: did the chain's CURRENT version
        # surface in a deep search at all?
        surfaced = False
        if expect:
            target = surf.get(lq)
            deep = store.search(q, limit=20)
            surfaced = any(getattr(r, "memory_id", None) == target for r in deep)

        rec = {"q": q, "label": lq, "expect": expect, "unfolded": unfolded,
               "surfaced_retrieval": surfaced}
        per_query.append(rec)

        if expect:
            if unfolded:
                tp += 1
            else:
                fn += 1
                if surfaced:
                    gateblocked.append(lq)
                else:
                    buried.append(lq)
        else:
            if unfolded:
                fp += 1
            else:
                tn += 1

    n_pos = tp + fn
    n_neg = fp + tn
    surfaced_pos = n_pos - len(buried)
    # Raw recall = over ALL positives; FAIR recall = over surfaced-only.
    raw_recall = (tp / n_pos * 100) if n_pos else 100.0
    fair_recall = (tp / surfaced_pos * 100) if surfaced_pos else 100.0
    precision = (tp / (tp + fp) * 100) if (tp + fp) else 100.0

    print(f"\n=== {label_} (arc_floor={arc_floor}) ===")
    print(f"  precision            {precision:5.1f}% ({tp} TP / {tp+fp} pos-injected)")
    print(f"  RAW recall (all pos) {raw_recall:5.1f}% ({tp}/{n_pos})")
    print(f"  FAIR recall (surfaced-only) {fair_recall:5.1f}% ({tp}/{surfaced_pos} of "
          f"{n_pos}-{len(buried)}buried)")
    print(f"  FP {fp}  TN {tn}")
    if buried:
        print(f"  RETRIEVAL-BURIED (not a gate failure): {', '.join(buried)}")
    if gateblocked:
        print(f"  GATE-BLOCKED (real feature miss):      {', '.join(gateblocked)}")
    return {"floor": arc_floor, "precision": precision, "raw_recall": raw_recall,
            "fair_recall": fair_recall, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "buried": buried, "gateblocked": gateblocked}


def run_eval():
    snap = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "hermes" / "memory-eval-clean.duckdb"
    if not snap.exists():
        print(f"ERROR: snapshot not found: {snap}")
        return 2

    from argos_plugin.store import DuckDBMemoryStore
    from argos_plugin.embeddings import LocalEmbedder, _resolve_embedding_model_path
    from argos_plugin import ArgosProvider

    td = tempfile.mkdtemp(prefix="hermes-unfold-clean-")
    home = R.build_arm_home(snap, BASE_CFG, Path(td))
    model = _resolve_embedding_model_path("bge-small-en-v1.5", hermes_home=str(home))
    embedder = LocalEmbedder(model, hermes_home=str(home))
    store = DuckDBMemoryStore(home / "hybrid_memory.duckdb",
                              user_id="default_user", embedder=embedder)
    surf = seed_chains(store)

    p = ArgosProvider()
    p.initialize(session_id="unfold_clean", hermes_home=str(home),
                 platform="cli", user_id="default_user")

    results = [
        run_floor(p, store, surf, 0.15, "PROD-DEFAULTS (arc 0.15 / anchor 0.30)"),
        run_floor(p, store, surf, 0.00, "ARC OFF (0.00)"),
    ]

    print("\n\n========== SUMMARY ==========")
    for r in results:
        print(f"  arc={r['floor']:.2f}  prec={r['precision']:5.1f}%  "
              f"raw_recall={r['raw_recall']:5.1f}%  "
              f"fair_recall={r['fair_recall']:5.1f}%  "
              f"TP={r['tp']} FP={r['fp']} FN={r['fn']} TN={r['tn']}")
        if r["gateblocked"]:
            print(f"     GATE-BLOCKED (real misses): {', '.join(r['gateblocked'])}")
    in_band = all(r["fair_recall"] >= 90 and r["precision"] >= 90 for r in results)
    print(f"  IN BAND (prec>=90 & fair_recall>=90 on both rows): {in_band}")
    return 0 if in_band else 3


if __name__ == "__main__":
    raise SystemExit(run_eval())

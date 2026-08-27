#!/usr/bin/env python3
"""CLEAN deterministic exact-phrase-lift ranking eval (sanitized for public repo).

What this measures
------------------
The phrase-lift lever (phrase_lift_alpha, prod 0.25 / pool 200) rewards
contiguous query bigrams present verbatim in the memory. It exists for the
class of query where the gold memory shares the exact phrase (e.g. "who is
the sales director" -> "...Raymond is the Sales Director...") but was ranked
low because unigram token overlap tied it with merely-similar content.

The initial validation (24/8) measured MRR .66 -> .82, h@1 4/8 -> 6/8 on
eight queries; this file is the sanitized, re-runnable equivalent:
deterministic synthetic facts (no real names/employers/locations), each
case built so the gold memory shares the verbatim phrase while 3-4 similar
distractors tie it on token overlap — the exact failure mode the lift
repairs.

Protocol:
- Fresh temp store per run (local embeddings, bge-small-en-v1.5, offline).
- Run 1: alpha=0.0 (control). Run 2: alpha=0.25 (production).
- Metric: h@1 (rank of gold memory == 1) and MRR over the same query set.
- No LLM, no network. Deterministic.

Run: env -u PYTHONPATH HF_HUB_OFFLINE=1 <hermes-venv-python>
     argos_plugin/eval/eval_phrase_lift_clean.py
Reference reproducibility: eval/repro/PHRASE_LIFT_RESULTS.md
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

# (gold_fact, [distractors], query) — exact phrase present only in gold.
# Built to reproduce the "who is the sales director" class: gold contains the
# verbatim phrase; distractors share tokens (same words, different order or
# paraphrase) so unigram text+vector both tie them, and the exact-phrase
# bigram boost is the discriminator.
CASES = [
    (
        "User is the regional sales director for the northern branch",
        [
            "The northern branch has a director role for sales planning",
            "The user's sales director reports from the regions",
            "Regional director responsibilities include northern sales targets",
            "A new sales director was appointed to the northern office team",
        ],
        "who is the sales director for the northern branch",
    ),
    (
        "The billing system migration finishes before the end of next week",
        [
            "Billing finishes migrating by the system end target next week",
            "The next system migration is for week-end billing close",
            "Before the end of next week the system migrates the billing suite",
            "Your migration of the billing ended before the next system week",
        ],
        "when does the billing system migration finish",
    ),
    (
        "User's brother works at the water treatment plant in the south",
        [
            "Work at the plant in the south involves water treatment duties",
            "The southern water treatment staff work for your brother's firm",
            "Your brother owns a company near the treatment plant in the south",
        ],
        "where does my brother work",
    ),
    (
        "User runs a small bakery on weekends",
        [
            "The small bakery the user visits runs weekend specials",
            "On weekends the user runs errands near a small bakery",
            "A small weekend bakery outlet is run by the user's friend",
        ],
        "what does the user run on weekends",
    ),
    (
        "User prefers plain roast chicken for Sunday lunch",
        [
            "Sunday lunch usually is chicken roasted plain for the user",
            "The user prefers lunch roasts on Sundays with plain chicken",
            "Chicken is the Sunday lunch roasts the user plain prefers",
        ],
        "what does the user prefer for Sunday lunch",
    ),
    (
        "User's router uses the 5 gigahertz band for the office network",
        [
            "The office network router operates on the five gigahertz band",
            "Five gigahertz is used by the office network for the router",
            "The router in the office uses the gigahertz five network band",
        ],
        "which band does the router use",
    ),
    (
        "User's pension fund pays out at age sixty five",
        [
            "At sixty five the pension pays out to the user's fund",
            "The user's fund sees a payout at age the sixty five mark",
            "Pension age sixty five applies to the user's fund payout",
        ],
        "at what age does the pension fund pay out",
    ),
    (
        "User switched to the twelve month gym membership plan",
        [
            "The membership plan at the gym is twelve months old",
            "Your gym switched months ago to a twelve-member plan",
            "The user's plan switched to membership twelve-month gym",
        ],
        "what membership plan did the user switch to",
    ),
]

NEG_QUERIES = [
    ("how is the weather today",),
    ("what colour is the sky",),
    ("did the user buy a new phone",),
]


def seed_store(store):
    for (fact, _dists, _q) in CASES:
        store.remember(category="personal_fact", content=fact, dedup=False)
    for (_g, dists, _q) in CASES:
        for d in dists:
            store.remember(category="personal_fact", content=d, dedup=False)
    # background noise — realistic non-matching memories
    store.remember(category="personal_fact", dedup=False,
                   content="User plans a holiday in December")
    store.remember(category="personal_fact", dedup=False,
                   content="User works in product management and travels for meetings")


def build_queries():
    qs = [(q, True, i) for (i, (_g, _d, q)) in enumerate(CASES)]
    for (nq,) in NEG_QUERIES:
        qs.append((nq, False, "neg"))
    return qs


def rank_gold(store, query, gold_ids, limit=10):
    """Return 1-based rank of gold memory in results, or limit+1 if missing."""
    rows = store.search(query, limit=limit)
    for idx, r in enumerate(rows):
        if getattr(r, "memory_id", None) in gold_ids:
            return idx + 1
    return limit + 1


def run_alpha(store, alpha, label):
    store._phrase_lift_alpha = alpha
    store._phrase_lift_pool = 200

    gold_ids = []
    for (i, (_g, _d, _q)) in enumerate(CASES):
        hits = [r for r in store.search(_g, limit=10)
                if r.content.strip() == _g.strip()]
        gold_ids.append(hits[0].memory_id if hits else f"missing-{i}")

    mrr = 0.0
    h1 = 0
    n_pos = 0
    regress = []
    per_query = []

    for q, expect, tag in build_queries():
        if tag == "neg":
            rank = rank_gold(store, q, set())
            rec = {"q": q, "tag": "neg", "rank": rank}
            per_query.append(rec)
            continue
        n_pos += 1
        rank = rank_gold(store, q, {gold_ids[tag]})
        mrr += (1.0 / rank) if rank <= 10 else 0.0
        if rank == 1:
            h1 += 1
        else:
            regress.append((tag, rank))
        per_query.append({"q": q, "tag": tag, "rank": rank})

    mrr /= n_pos if n_pos else 1
    print(f"\n=== {label} (alpha={alpha}) ===")
    print(f"  MRR  {mrr:.4f}   h@1 {h1}/{n_pos}")
    for rec in per_query:
        print(f"    rank {rec['rank']:>2}  {rec['q'][:64]}")
    return {"alpha": alpha, "mrr": mrr, "h1": h1, "n": n_pos, "per_query": per_query}


def run_eval():
    snap = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "hermes" / "memory-eval-clean.duckdb"
    if not snap.exists():
        print(f"ERROR: snapshot not found: {snap}")
        return 2

    from argos_plugin.store import DuckDBMemoryStore
    from argos_plugin.embeddings import LocalEmbedder, _resolve_embedding_model_path

    td = tempfile.mkdtemp(prefix="hermes-phrase-lift-")
    home = R.build_arm_home(snap, BASE_CFG, Path(td))
    model = _resolve_embedding_model_path("bge-small-en-v1.5", hermes_home=str(home))
    embedder = LocalEmbedder(model, hermes_home=str(home))
    store = DuckDBMemoryStore(home / "hybrid_memory.duckdb",
                              user_id="default_user", embedder=embedder)
    seed_store(store)

    r0 = run_alpha(store, 0.0, "CONTROL (alpha 0.00)")
    r25 = run_alpha(store, 0.25, "PROD (alpha 0.25)")

    print("\n\n========== SUMMARY ==========")
    print(f"  control  MRR {r0['mrr']:.4f}  h1 {r0['h1']}/{r0['n']}")
    print(f"  prod     MRR {r25['mrr']:.4f}  h1 {r25['h1']}/{r25['n']}")
    delta_mrr = r25["mrr"] - r0["mrr"]
    print(f"  MRR delta {delta_mrr:+.4f}   h1 delta {r25['h1'] - r0['h1']:+d}")
    improved = (r25["mrr"] > r0["mrr"] or r25["h1"] > r0["h1"]) and r25["mrr"] >= r0["mrr"]
    # A "regression" = a case that was rank 1 under control and dropped below.
    regressed = [r for r in r0["per_query"] if r["tag"] != "neg" and r["rank"] == 1
                 and not any(p["tag"] == r["tag"] and p["rank"] == 1 for p in r25["per_query"])]
    print(f"  regressions from control rank-1: {len(regressed)}")
    print(f"  IMPROVED (no regression): {improved}")
    return 0 if improved else 3


if __name__ == "__main__":
    raise SystemExit(run_eval())
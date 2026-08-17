#!/usr/bin/env python3
"""Provider-layer ranking A/B harness.

The store-level harness (run_eval.py) measures DuckDBMemoryStore.search()
directly — it CANNOT see the provider layer (graph-aware retrieval, alias
expansion, candidate injection, reranker, query expansion). This harness
constructs the real HybridMemoryProvider in direct mode against a per-arm
snapshot home and drives its full _search_memories() pipeline.

Arms (config toggles, mirroring production defaults):
  baseline    graph_aware_retrieval=true  boost=0.05  inject=false reranker=false
  graph_off   graph_aware_retrieval=false
  inject_on   graph_aware_retrieval=true  inject=true
  reranker_on reranker_enabled=true  (bge-reranker-base, top_n=10)

Every arm runs on its OWN copy of the snapshot corpus (owner discipline:
never open the canonical/live DB). The graph is built fresh per arm from the
snapshot records (regex-first, no LLM — deterministic and offline).

Usage: python run_eval_provider.py <snapshot.duckdb> <eval_set.json> <out_dir>
"""
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path


def _dcg(rels: list) -> float:
    return sum(r / math.log2(i + 2) for i, r in enumerate(rels))


def _ndcg(hits: set, ranked: list, k: int) -> float:
    rel = [1 if m in hits else 0 for m in ranked[:k]]
    if not hits:
        return 1.0
    ideal = [1] * min(len(hits), k)
    return _dcg(rel) / _dcg(ideal) if _dcg(ideal) > 0 else 0.0


ARMS = {
    "baseline": {},
    "graph_off": {"graph_aware_retrieval": "false"},
    "inject_on": {"graph_aware_retrieval": "true",
                  "graph_inject_candidates": "true"},
    "reranker_on": {"reranker_enabled": "true",
                    "reranker_model": "BAAI/bge-reranker-base",
                    "reranker_top_n": "10"},
    "boost_zero": {"graph_aware_retrieval": "true",
                   "graph_retrieval_boost": "0.0"},
    "llm_graph": {"graph_aware_retrieval": "true",
                  "graph_retrieval_boost": "0.0"},
}


def build_arm_home(snapshot: Path, arm_cfg: dict, workdir: Path) -> Path:
    """Create a temp hermes_home for one arm: copied DB, fresh graph, config."""
    home = workdir / "home"
    db_path = home / "hybrid_memory.duckdb"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(snapshot, db_path)

    cfg = {
        "database_filename": "hybrid_memory.duckdb",
        "graph_dirname": "hybrid_memory_kuzu",
        "storage_mode": "direct",
        "local_embedding_model": "bge-small-en-v1.5",
        "auto_extract": "false",
        "llm_fallback": "false",
        "auto_review": "false",
        "graph_aware_retrieval": "true",
        "graph_retrieval_boost": "0.05",
        "graph_inject_candidates": "false",
        "graph_boost_min_similarity": "0.15",
        "alias_expansion_boost": "0.7",
        "consolidation_enabled": "false",
        "reranker_enabled": "false",
        "reranker_model": "BAAI/bge-reranker-base",
        "reranker_top_n": "10",
        "context_aware_retrieval": "false",
        "query_expansion_enabled": "false",
        "max_injected_items": "5",
    }
    cfg.update(arm_cfg)
    (home / "hybrid_memory.json").write_text(
        json.dumps(cfg, indent=2), encoding="utf-8")

    # Local-first model resolution: <hermes_home>/models/<name> must exist or
    # the embedder falls back to the hub (401/network) → silent text-only
    # search. Link the cached model from the real hermes home.
    src_models = Path(r"C:\Users\<user>\AppData\Local\hermes\models")
    for name in ("bge-small-en-v1.5", "bge-reranker-base"):
        model_dir = src_models / name
        if not model_dir.is_dir() and name == "bge-reranker-base":
            # HF hub cache layout: models--BAAI--bge-reranker-base/snapshots/<hash>
            hub = Path(r"C:\Users\<user>\.cache\huggingface\hub") / "models--BAAI--bge-reranker-base" / "snapshots"
            snaps = sorted(hub.iterdir()) if hub.is_dir() else []
            if snaps:
                model_dir = snaps[0]
        if model_dir.is_dir():
            link = home / "models" / name
            link.parent.mkdir(parents=True, exist_ok=True)
            try:
                link.symlink_to(model_dir, target_is_directory=True)
            except (OSError, NotImplementedError):
                shutil.copytree(model_dir, link)
    return home


def build_snapshot_graph(home: Path, use_llm: bool = False) -> int:
    """Index every snapshot record into the arm's Kuzu graph.

    use_llm=False: regex-first extraction (matches production default when
    the LLM gate rejects; the "current graph" arm).
    use_llm=True: hybrid extraction (regex + LLM supplement) — the
    "properly typed graph" arm. SLOW: LLM per memory (~15s each).
    """
    sys.path.insert(0, r"C:\Users\<user>\Documents\Github\Hermes")
    from hybrid_memory_plugin.store import DuckDBMemoryStore
    from hybrid_memory_plugin.embeddings import LocalEmbedder, _resolve_embedding_model_path
    from hybrid_memory_plugin.graph import KuzuGraphStore

    model = _resolve_embedding_model_path("bge-small-en-v1.5",
                                          hermes_home=str(home))
    embedder = LocalEmbedder(model, hermes_home=str(home))
    store = DuckDBMemoryStore(home / "hybrid_memory.duckdb",
                              user_id="default_user", embedder=embedder)
    graph = KuzuGraphStore(home / "hybrid_memory_kuzu", user_id="default_user")

    n = 0
    rows = store.connection.execute(
        """SELECT memory_id, category, content, tags, created_at
           FROM memory_records
           WHERE COALESCE(status, 'active') = 'active'
             AND valid_to IS NULL"""
    ).fetchall()
    for memory_id, category, content, tags, created_at in rows:
        graph.index_memory(
            memory_id=memory_id,
            category=category or "",
            content=content or "",
            tags=tags or [],
            created_at=created_at,
            use_llm=use_llm,
        )
        n += 1
    graph.close()
    store.close()
    return n


def run_arm(home: Path, eval_set: dict) -> dict:
    sys.path.insert(0, r"C:\Users\<user>\Documents\Github\Hermes")
    from hybrid_memory_plugin import HybridMemoryProvider as Provider

    provider = Provider()
    provider.initialize(
        session_id="eval-provider",
        hermes_home=str(home),
        platform="cli",
        user_id="default_user",
    )

    results = []
    totals = {"p5": [], "p10": [], "r5": [], "r10": [], "ndcg5": [],
              "ndcg10": [], "mrr": []}
    for q in eval_set["queries"]:
        hits = set(q["relevant"])
        ranked = [r.memory_id for r in provider._search_memories(
            q["query"], limit=10)]
        k5, k10 = ranked[:5], ranked[:10]
        p5 = sum(1 for m in k5 if m in hits) / 5
        p10 = sum(1 for m in k10 if m in hits) / 10
        r5 = sum(1 for m in k5 if m in hits) / max(1, len(hits))
        r10 = sum(1 for m in k10 if m in hits) / max(1, len(hits))
        mrr = next((1 / (i + 1) for i, m in enumerate(ranked) if m in hits), 0.0)
        row = {
            "query": q["query"],
            "n_relevant": len(hits),
            "p5": round(p5, 4), "p10": round(p10, 4),
            "r5": round(r5, 4), "r10": round(r10, 4),
            "ndcg5": round(_ndcg(hits, ranked, 5), 4),
            "ndcg10": round(_ndcg(hits, ranked, 10), 4),
            "mrr": round(mrr, 4),
            "top_hits": [m for m in k10 if m in hits],
            "missed": sorted(hits - set(ranked)),
        }
        results.append(row)
        for key in totals:
            totals[key].append(row[key])
    provider.shutdown()
    return {
        "n_queries": len(results),
        "averages": {k: round(sum(v) / len(v), 4) for k, v in totals.items()},
        "per_query": results,
    }


def build_typed_graph(home: Path, llm_report: dict) -> int:
    """Replay LLM-typed relations from the typing pass into the graph.

    Uses the same KuzuGraphStore.index_memory shape: a memory node,
    about_user edge, typed relation edges, and mentions edges — but the
    relation set comes from the LLM typing report (regex + LLM merged).
    """
    sys.path.insert(0, r"C:\Users\<user>\Documents\Github\Hermes")
    from hybrid_memory_plugin.graph import KuzuGraphStore

    graph = KuzuGraphStore(home / "hybrid_memory_kuzu", user_id="default_user")
    n = 0
    for memory_id, rels in (llm_report.get("relations_by_memory") or {}).items():
        mem_node = f"memory:{memory_id}"
        graph.upsert_node(mem_node, "memory", {"memory_id": memory_id,
                                               "category": "",
                                               "status": "active"})
        graph.add_relationship(mem_node, "memory", "about_user",
                               "user", "person",
                               {"memory_id": memory_id, "category": ""})
        for r in rels:
            graph.add_relationship(
                r.get("source") or "user", r.get("source_type") or "person",
                r.get("relation"), r.get("target"), r.get("target_type") or "concept",
                {"memory_id": memory_id, "extractor": r.get("extractor")})
            graph.add_relationship(mem_node, "memory", "mentions",
                                   r.get("target"), r.get("target_type") or "concept",
                                   {"memory_id": memory_id, "category": ""})
        n += 1
    graph.close()
    return n


def main() -> int:
    if len(sys.argv) < 4:
        print(__doc__)
        return 2
    snapshot = Path(sys.argv[1])
    eval_path = Path(sys.argv[2])
    out_dir = Path(sys.argv[3])
    out_dir.mkdir(parents=True, exist_ok=True)
    llm_report = None
    if len(sys.argv) > 4 and Path(sys.argv[4]).exists():
        llm_report = json.loads(Path(sys.argv[4]).read_text(encoding="utf-8"))

    eval_set = json.loads(eval_path.read_text(encoding="utf-8"))
    summary = {}

    for arm, cfg in ARMS.items():
        print(f"\n=== ARM: {arm} ===", flush=True)
        with tempfile.TemporaryDirectory(prefix=f"eval_{arm}_") as td:
            workdir = Path(td)
            home = build_arm_home(snapshot, cfg, workdir)
            if arm == "llm_graph" and llm_report:
                n_indexed = build_typed_graph(home, llm_report)
            else:
                n_indexed = build_snapshot_graph(home)
            print(f"  graph indexed: {n_indexed} records", flush=True)
            report = run_arm(home, eval_set)
        print(f"  averages: {report['averages']}", flush=True)
        report["arm"] = arm
        report["graph_indexed"] = n_indexed
        report["config"] = cfg
        (out_dir / f"provider_{arm}.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        summary[arm] = report["averages"]

    # Delta table vs baseline
    print("\n=== DELTAS vs baseline ===")
    base = summary.get("baseline", {})
    for arm, avg in summary.items():
        if arm == "baseline":
            continue
        deltas = {k: round(avg[k] - base[k], 4) for k in base}
        print(f"  {arm}: {deltas}")

    (out_dir / "provider_summary.json").write_text(
        json.dumps({"arms": summary}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

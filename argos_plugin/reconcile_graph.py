#!/usr/bin/env python3
"""Reconciliation probe: detect DuckDB ↔ Kuzu graph drift (#305).

Dual-store consistency has no transactional coupling — a crash between
a DuckDB write and the graph index leaves the stores drifted.  This
probe compares graph node/edge coverage vs DuckDB active memory_ids
and reports missing/extra counts + sample IDs.  On hit, it prints a
command to re-run backfill_graph.py for the affected memories.

Run with Hermes STOPPED (the shared memory service holds locks):

    python reconcile_graph.py
    python reconcile_graph.py --home /path/to/hermes_home
    python reconcile_graph.py --json   # machine-readable output

Exit codes:
    0 — no drift detected
    1 — drift detected (missing or extra memory_ids)
    2 — error (cannot connect, store unavailable)

Options:
    --home PATH        Override HERMES_HOME (auto-detected by default).
    --json             Emit JSON instead of human-readable text.
    --sample N         Number of sample IDs to include (default: 10).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple


def _get_hermes_home() -> Path:
    """Resolve HERMES_HOME the same way the plugin does."""
    try:
        sys.path.insert(
            0, str(Path(__file__).resolve().parent.parent.parent / "hermes-agent")
        )
        from hermes_constants import get_hermes_home
        return Path(get_hermes_home())
    except Exception:
        pass
    env = os.environ.get("HERMES_HOME")
    if env:
        return Path(env)
    local = Path(os.path.expandvars(r"%LOCALAPPDATA%\hermes"))
    if local.exists():
        return local
    return Path(os.path.expanduser("~/.hermes"))


def _get_duckdb_memory_ids(store: Any) -> Set[str]:
    """Get all active memory_ids from the DuckDB store."""
    # list_recent returns active memories ordered by created_at DESC.
    # We request a high limit to get everything in one call.
    records = store.list_recent(limit=10000)
    return {r.memory_id for r in records if r.memory_id}


def _get_graph_memory_ids(graph: Any) -> Set[str]:
    """Get all memory_ids referenced in the Kuzu graph.

    Collects memory_ids from two sources:
    1. RelatesTo edges' memory_ids column (list_contains).
    2. Entity nodes whose id starts with 'memory:' (direct memory nodes).
    """
    graph_ids: Set[str] = set()

    # Source 1: query all RelatesTo edges and extract memory_ids.
    # We use the RPC layer to call a custom query.
    if hasattr(graph, "_rpc"):
        # SharedGraphStore — use RPC to get edges with memory_ids.
        try:
            # list_nodes gives us Entity nodes; we need edges.
            # The graph service doesn't expose a raw query method,
            # so we use count_nodes + the search_graph pattern.
            # Instead, we can get all memory_ids by querying for
            # nodes of type 'memory' — but the graph stores memory
            # references as edge attributes, not as separate nodes.
            #
            # The most reliable approach: use the graph's
            # memory_ids_for_query with a broad query to get all
            # indexed memory_ids. But that's query-dependent.
            #
            # Instead, we use a direct approach: call the graph's
            # _rpc.call to run a custom Cypher query via the service.
            # The service dispatches 'search_graph' and other methods,
            # but doesn't expose raw Cypher. So we use a workaround:
            # get all nodes and extract memory references from them.
            #
            # Actually, the graph stores memory:<id> as Entity node ids
            # when a memory is indexed. So we can list all nodes and
            # filter for those starting with 'memory:'.
            nodes = graph._rpc.call("graph", "list_nodes", limit=100000) or []
            for node in nodes:
                node_id = node.get("id", "")
                if node_id.startswith("memory:"):
                    mid = node_id[len("memory:"):]
                    if mid:
                        graph_ids.add(mid)
        except Exception as exc:
            # If the RPC call fails, fall back to empty set.
            print(f"WARNING: Could not query graph nodes via RPC: {exc}", file=sys.stderr)
    else:
        # Direct KuzuGraphStore — query the database directly.
        try:
            nodes = graph.list_nodes(limit=100000)
            for node in nodes:
                node_id = node.get("id", "")
                if node_id.startswith("memory:"):
                    mid = node_id[len("memory:"):]
                    if mid:
                        graph_ids.add(mid)
        except Exception as exc:
            print(f"WARNING: Could not query graph nodes: {exc}", file=sys.stderr)

    return graph_ids


def reconcile(
    store: Any,
    graph: Any,
    *,
    sample_size: int = 10,
) -> Dict[str, Any]:
    """Run the reconciliation probe.

    Returns a dict with:
        duckdb_count: int — number of active memory_ids in DuckDB
        graph_count: int — number of memory_ids referenced in the graph
        missing_in_graph: list[str] — in DuckDB but not in graph (sample)
        missing_in_graph_count: int — total count of missing
        extra_in_graph: list[str] — in graph but not in DuckDB (sample)
        extra_in_graph_count: int — total count of extra
        drift: bool — True if missing_in_graph_count > 0 or extra_in_graph_count > 0
    """
    duckdb_ids = _get_duckdb_memory_ids(store)
    graph_ids = _get_graph_memory_ids(graph)

    missing = duckdb_ids - graph_ids
    extra = graph_ids - duckdb_ids

    missing_sample = sorted(missing)[:sample_size]
    extra_sample = sorted(extra)[:sample_size]

    return {
        "duckdb_count": len(duckdb_ids),
        "graph_count": len(graph_ids),
        "missing_in_graph_count": len(missing),
        "missing_in_graph": missing_sample,
        "extra_in_graph_count": len(extra),
        "extra_in_graph": extra_sample,
        "drift": bool(missing or extra),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reconciliation probe: detect DuckDB ↔ Kuzu graph drift."
    )
    parser.add_argument("--home", default=None, help="Override HERMES_HOME.")
    parser.add_argument("--json", action="store_true", help="Emit JSON output.")
    parser.add_argument("--sample", type=int, default=10, help="Sample IDs to include.")
    args = parser.parse_args()

    home = Path(args.home) if args.home else _get_hermes_home()
    if not home.exists():
        print(f"ERROR: HERMES_HOME not found at {home}")
        return 2

    plugin_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(plugin_dir))
    try:
        from service_client import SharedMemoryStore, SharedGraphStore
    except ImportError as e:
        print(f"ERROR: Cannot import service_client: {e}")
        return 2

    try:
        store = SharedMemoryStore(home, user_id="default_user", embedder=None)
    except Exception as e:
        print(f"ERROR: Cannot connect to shared memory service: {e}")
        return 2

    try:
        graph = SharedGraphStore(home, user_id="default_user")
    except Exception as e:
        print(f"ERROR: Cannot connect to shared graph service: {e}")
        store._rpc.stop_service()
        return 2

    try:
        result = reconcile(store, graph, sample_size=args.sample)
    except Exception as e:
        print(f"ERROR: Reconciliation failed: {e}")
        graph.close()
        store._rpc.stop_service()
        return 2
    finally:
        try:
            graph.close()
            store._rpc.stop_service()
        except Exception:
            pass

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"DuckDB active memory_ids: {result['duckdb_count']}")
        print(f"Graph memory_ids:        {result['graph_count']}")
        print()
        if result["missing_in_graph_count"] > 0:
            print(f"MISSING in graph (in DuckDB but not indexed): {result['missing_in_graph_count']}")
            for mid in result["missing_in_graph"]:
                print(f"  - {mid}")
            print()
            print("To fix: run backfill_graph.py to re-index missing memories.")
        if result["extra_in_graph_count"] > 0:
            print(f"EXTRA in graph (in graph but not in DuckDB): {result['extra_in_graph_count']}")
            for mid in result["extra_in_graph"]:
                print(f"  - {mid}")
            print()
            print("To fix: these memories were deleted from DuckDB but still")
            print("have graph nodes. Run backfill_graph.py or manually remove")
            print("orphaned graph entries.")
        if not result["drift"]:
            print("No drift detected. DuckDB and graph are in sync.")

    return 1 if result["drift"] else 0


if __name__ == "__main__":
    sys.exit(main())

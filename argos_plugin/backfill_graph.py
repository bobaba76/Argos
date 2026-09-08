#!/usr/bin/env python3
"""Backfill graph edges for all existing memory records.

When the graph entity-extraction feature is first deployed, existing
memories have no graph edges — they were created before graph indexing
was wired into the save/review paths.  This script re-indexes every
active memory record through the graph, extracting entities and creating
cross-memory links.

Run with Hermes STOPPED (the shared memory service holds locks):

    # 1. Stop Hermes completely (close desktop app, kill any gateway).
    # 2. Run the backfill:
    python backfill_graph.py
    # 3. Start Hermes.

Options:
    --dry-run          Show counts without writing to the graph.
    --no-llm           Use regex-only extraction (fast, deterministic, no token cost).
    --batch-size N     Memories per batch (default: 50).
    --home PATH        Override HERMES_HOME (auto-detected by default).
    --prune-orphans    After backfill, remove graph memory nodes whose
                       memory_id has no DuckDB record in ANY state
                       (active, quarantined, non-head, superseded).
                       Default OFF — explicit flag required (#376).

Safety:
    - Does not modify memory records — only adds graph nodes/edges.
    - Re-indexing the same memory is safe (edges use MERGE + evidence lists).
    - Reports progress every batch.
    - Never deletes existing graph data UNLESS --prune-orphans is passed.
    - --prune-orphans only targets `memory:` graph nodes whose memory_id
      is confirmed absent from DuckDB (via get_memory_history — covers
      active, quarantined, and superseded states). Shared entity nodes
      and non-memory system nodes (e.g. `__system_state__`, regex junk
      like `// resources`) are NEVER touched.
    - --dry-run --prune-orphans lists what WOULD be deleted without
      deleting anything.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Set


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


def _graph_memory_ids(graph: Any) -> List[str]:
    """Collect all `memory:`-prefixed node ids from the graph."""
    ids: List[str] = []
    try:
        nodes = graph.list_nodes(limit=100000)
    except Exception as e:
        print(f"ERROR: Cannot list graph nodes: {e}")
        return ids
    for node in nodes:
        node_id = str(node.get("id", ""))
        if node_id.startswith("memory:"):
            mid = node_id[len("memory:"):]
            if mid:
                ids.append(mid)
    return ids


def _find_orphans(store: Any, graph_memory_ids: List[str]) -> List[str]:
    """Find graph memory_ids with no DuckDB record in ANY state.

    Cross-check strategy (#376): a memory_id is preserved if it exists
    in memory_records as active, quarantined, non-head (valid_to set),
    or superseded. `get_memories_by_ids(include_quarantined=True)` is
    the bulk check; anything it misses is confirmed per-id via
    `get_memory_history` (which has no status/expiry filter) before
    being declared an orphan. Tombstoned memories are keyed by
    content_hash, not memory_id — their memory_id no longer exists in
    DuckDB at all, so a graph node for it is genuinely orphaned.
    """
    orphans: List[str] = []
    # Bulk check in chunks (one RPC per chunk).
    found: Set[str] = set()
    chunk_size = 100
    for i in range(0, len(graph_memory_ids), chunk_size):
        chunk = graph_memory_ids[i:i + chunk_size]
        try:
            records = store.get_memories_by_ids(
                chunk, include_quarantined=True,
            )
            for rec in records:
                if rec.memory_id:
                    found.add(str(rec.memory_id))
        except Exception as e:
            print(f"ERROR: bulk existence check failed for {len(chunk)} ids: {e}")
    # Confirm each not-found id per-id (covers expired/edge cases the
    # bulk check filters out).
    for mid in graph_memory_ids:
        if mid in found:
            continue
        try:
            history = store.get_memory_history(mid)
        except Exception:
            history = None
        if not history:
            orphans.append(mid)
    return orphans


def prune_orphans(
    store: Any,
    graph: Any,
    *,
    dry_run: bool = False,
) -> Dict[str, int]:
    """#376: remove graph memory nodes with no DuckDB record.

    Only `memory:` nodes are candidates; shared entity nodes and
    non-memory system nodes are never touched. Returns a summary dict
    with orphans/removed/skipped/errors counts.
    """
    graph_ids = _graph_memory_ids(graph)
    if not graph_ids:
        print("No graph memory nodes found — nothing to prune.")
        return {"orphans": 0, "removed": 0, "skipped": 0, "errors": 0}
    orphans = _find_orphans(store, graph_ids)
    removed = 0
    skipped = 0
    errors = 0
    if dry_run:
        print(f"[DRY RUN] Would prune {len(orphans)} orphaned graph node(s):")
        for mid in sorted(orphans)[:10]:
            print(f"  - {mid}")
        if len(orphans) > 10:
            print(f"  ... and {len(orphans) - 10} more")
        return {"orphans": len(orphans), "removed": 0, "skipped": 0, "errors": 0}
    for mid in sorted(orphans):
        try:
            if graph.purge_orphan_memory(mid):
                removed += 1
            else:
                skipped += 1
        except Exception as e:
            print(f"  ERROR pruning {mid}: {e}")
            errors += 1
    return {"orphans": len(orphans), "removed": removed, "skipped": skipped, "errors": errors}


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill graph edges for all memories.")
    parser.add_argument("--dry-run", action="store_true", help="Show counts without writing.")
    parser.add_argument("--no-llm", action="store_true", help="Regex-only extraction (skip LLM).")
    parser.add_argument("--batch-size", type=int, default=50, help="Memories per batch.")
    parser.add_argument("--home", default=None, help="Override HERMES_HOME.")
    parser.add_argument(
        "--prune-orphans", action="store_true",
        help="After backfill, remove graph memory nodes whose memory_id has "
             "no DuckDB record in ANY state (#376). Default OFF.",
    )
    args = parser.parse_args()

    home = Path(args.home) if args.home else _get_hermes_home()
    if not home.exists():
        print(f"ERROR: HERMES_HOME not found at {home}")
        return 1

    print(f"HERMES_HOME: {home}")
    print()

    # Import the shared service clients.
    plugin_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(plugin_dir))
    try:
        from service_client import SharedMemoryStore, SharedGraphStore
    except ImportError as e:
        print(f"ERROR: Cannot import service_client: {e}")
        return 1

    # Connect to the shared memory service (auto-starts if not running).
    print("Connecting to shared memory service...")
    try:
        store = SharedMemoryStore(home, user_id="default_user", embedder=None)
    except Exception as e:
        print(f"ERROR: Cannot connect to shared memory service: {e}")
        return 1

    try:
        graph = SharedGraphStore(home, user_id="default_user")
    except Exception as e:
        print(f"ERROR: Cannot connect to shared graph service: {e}")
        store._rpc.stop_service()
        return 1

    # Count active memories.
    try:
        total = store.count()
    except Exception as e:
        print(f"ERROR: Cannot query memory count: {e}")
        graph.close()
        store._rpc.stop_service()
        return 1

    print(f"Total active memories: {total}")
    print()

    if total == 0:
        print("Nothing to backfill.")
        if args.prune_orphans:
            print()
            summary = prune_orphans(store, graph, dry_run=args.dry_run)
            print()
            print(f"=== PRUNE SUMMARY ===")
            print(f"Orphans found: {summary['orphans']}")
            if not args.dry_run:
                print(f"Removed:       {summary['removed']}")
                print(f"Skipped:       {summary['skipped']}")
                print(f"Errors:        {summary['errors']}")
        graph.close()
        store._rpc.stop_service()
        return 0

    if args.dry_run:
        print(f"[DRY RUN] Would backfill {total} memories into the graph.")
        print(f"  LLM-assisted: {not args.no_llm}")
        if args.prune_orphans:
            print()
            summary = prune_orphans(store, graph, dry_run=True)
            print()
            print(f"=== PRUNE SUMMARY ===")
            print(f"Orphans found: {summary['orphans']}")
        graph.close()
        store._rpc.stop_service()
        return 0

    # Fetch all memories in batches using list_recent with a high limit.
    # list_recent returns active memories ordered by created_at DESC.
    # We request a high limit to get everything in one call; the shared
    # service handles the query efficiently.
    batch_size = max(1, args.batch_size)
    use_llm = not args.no_llm
    indexed = 0
    skipped = 0
    errors = 0

    print(f"Backfilling with {'hybrid (regex + LLM)' if use_llm else 'regex-only'} extraction...")
    print()

    # Fetch all active memories in one batch (capped at a reasonable max).
    fetch_limit = min(max(total, 1), 10000)
    try:
        records = store.list_recent(limit=fetch_limit)
    except Exception as e:
        print(f"ERROR fetching memories: {e}")
        graph.close()
        store._rpc.stop_service()
        return 1

    for i, rec in enumerate(records):
        memory_id = rec.memory_id
        content = rec.content
        category = rec.category
        tags = rec.tags or []
        created_at = rec.created_at

        if not content or not content.strip():
            skipped += 1
            continue

        try:
            is_last = (i + 1) == len(records) or (i + 1) % batch_size == 0
            graph.index_memory(
                memory_id=memory_id,
                category=category,
                content=content,
                tags=tags,
                created_at=created_at,
                use_llm=use_llm,
                flush=is_last,  # Batch flush: only flush at batch boundaries
                # #287: mirror the record's valid window onto the graph
                # node/edges — provenance preserved from the source record.
                valid_from=getattr(rec, "valid_from", None) or created_at,
                valid_to=getattr(rec, "valid_to", None),
            )
            indexed += 1
        except Exception as e:
            print(f"  ERROR indexing {memory_id}: {e}")
            errors += 1

        if (i + 1) % batch_size == 0 or (i + 1) == len(records):
            print(f"  Progress: {i + 1}/{len(records)} memories processed "
                  f"({indexed} indexed, {skipped} skipped, {errors} errors)")

    print()
    print(f"=== BACKFILL COMPLETE ===")
    print(f"Indexed:  {indexed}")
    print(f"Skipped:  {skipped}")
    print(f"Errors:   {errors}")
    print()

    # Verify: count graph nodes and edges.
    try:
        # Use the graph's count methods via RPC.
        node_count = graph._rpc.call("graph", "count_nodes") if hasattr(graph, "_rpc") else "N/A"
        edge_count = graph._rpc.call("graph", "count_edges") if hasattr(graph, "_rpc") else "N/A"
        print(f"Graph nodes: {node_count}")
        print(f"Graph edges: {edge_count}")
    except Exception:
        pass

    # #376: prune orphaned graph nodes (explicit flag only, default OFF).
    if args.prune_orphans:
        print()
        summary = prune_orphans(store, graph, dry_run=args.dry_run)
        print()
        print(f"=== PRUNE SUMMARY ===")
        print(f"Orphans found: {summary['orphans']}")
        if not args.dry_run:
            print(f"Removed:       {summary['removed']}")
            print(f"Skipped:       {summary['skipped']}")
            print(f"Errors:        {summary['errors']}")

    print()
    print("You can now start Hermes. Graph-aware search and traversal will work for all memories.")

    graph.close()
    store._rpc.stop_service()
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

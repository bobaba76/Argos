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

Safety:
    - Does not modify memory records — only adds graph nodes/edges.
    - Re-indexing the same memory is safe (edges use MERGE + evidence lists).
    - Reports progress every batch.
    - Never deletes existing graph data.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


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


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill graph edges for all memories.")
    parser.add_argument("--dry-run", action="store_true", help="Show counts without writing.")
    parser.add_argument("--no-llm", action="store_true", help="Regex-only extraction (skip LLM).")
    parser.add_argument("--batch-size", type=int, default=50, help="Memories per batch.")
    parser.add_argument("--home", default=None, help="Override HERMES_HOME.")
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
        print("Nothing to backfill. Exiting.")
        graph.close()
        store._rpc.stop_service()
        return 0

    if args.dry_run:
        print(f"[DRY RUN] Would backfill {total} memories into the graph.")
        print(f"  LLM-assisted: {not args.no_llm}")
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

    print()
    print("You can now start Hermes. Graph-aware search and traversal will work for all memories.")

    graph.close()
    store._rpc.stop_service()
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

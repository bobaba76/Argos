#!/usr/bin/env python3
"""Rebuild the Kùzu graph from DuckDB memory records.

This is the canonical "DuckDB is truth, Kùzu is disposable index" rebuild.
It clears all graph nodes and edges for the current user scope, then
re-indexes every active memory record through the graph extractor.

Run with Hermes STOPPED (the shared memory service holds locks):

    # 1. Stop Hermes completely (close desktop app, kill any gateway).
    # 2. Run the rebuild:
    python rebuild_graph.py
    # 3. Start Hermes.

Options:
    --dry-run          Show counts without changing anything.
    --no-llm           Use regex-only extraction (fast, deterministic, no token cost).
    --batch-size N     Memories per batch (default: 50).
    --home PATH        Override HERMES_HOME (auto-detected by default).
    --user-id ID       Override user scope (default: default_user).

Safety:
    - Does not modify memory records — only rebuilds graph nodes/edges.
    - Clears graph for the current user scope only; other users are untouched.
    - Reports progress every batch.
    - Verifies final node/edge counts.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _get_hermes_home() -> Path:
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
    parser = argparse.ArgumentParser(description="Rebuild the Kùzu graph from DuckDB memories.")
    parser.add_argument("--dry-run", action="store_true", help="Show counts without changing anything.")
    parser.add_argument("--no-llm", action="store_true", help="Regex-only extraction (skip LLM).")
    parser.add_argument("--batch-size", type=int, default=50, help="Memories per batch.")
    parser.add_argument("--home", default=None, help="Override HERMES_HOME.")
    parser.add_argument("--user-id", default="default_user", help="User scope to rebuild.")
    args = parser.parse_args()

    home = Path(args.home) if args.home else _get_hermes_home()
    if not home.exists():
        print(f"ERROR: HERMES_HOME not found at {home}")
        return 1

    print(f"HERMES_HOME: {home}")
    print(f"User scope:  {args.user_id}")
    print()

    plugin_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(plugin_dir))
    try:
        from service_client import SharedMemoryStore, SharedGraphStore
    except ImportError as e:
        print(f"ERROR: Cannot import service_client: {e}")
        return 1

    print("Connecting to shared memory service...")
    try:
        store = SharedMemoryStore(home, user_id=args.user_id, embedder=None)
    except Exception as e:
        print(f"ERROR: Cannot connect to shared memory service: {e}")
        return 1

    try:
        graph = SharedGraphStore(home, user_id=args.user_id)
    except Exception as e:
        print(f"ERROR: Cannot connect to shared graph service: {e}")
        store._rpc.stop_service()
        return 1

    # Count active memories and current graph state.
    try:
        total_memories = store.count()
        old_nodes = graph.count_nodes()
        old_edges = graph.count_edges()
    except Exception as e:
        print(f"ERROR: Cannot query counts: {e}")
        graph.close()
        store._rpc.stop_service()
        return 1

    print(f"Active memories: {total_memories}")
    print(f"Current graph:    {old_nodes} nodes, {old_edges} edges")
    print()

    if total_memories == 0:
        print("No memories to rebuild. Exiting.")
        graph.close()
        store._rpc.stop_service()
        return 0

    if args.dry_run:
        print(f"[DRY RUN] Would clear {old_nodes} nodes and {old_edges} edges,")
        print(f"          then re-index {total_memories} memories.")
        print(f"  LLM-assisted: {not args.no_llm}")
        graph.close()
        store._rpc.stop_service()
        return 0

    # Phase 1: Clear the graph for this user scope.
    print("Phase 1: Clearing existing graph...")
    try:
        remaining_nodes, remaining_edges = graph.clear_scope()
        print(f"  Cleared. Remaining: {remaining_nodes} nodes, {remaining_edges} edges")
    except Exception as e:
        print(f"ERROR clearing graph: {e}")
        graph.close()
        store._rpc.stop_service()
        return 1

    print()

    # Phase 2: Re-index all active memories.
    batch_size = max(1, args.batch_size)
    use_llm = not args.no_llm
    indexed = 0
    skipped = 0
    errors = 0

    print(f"Phase 2: Re-indexing {total_memories} memories "
          f"({'hybrid (regex + LLM)' if use_llm else 'regex-only'})...")
    print()

    fetch_limit = min(max(total_memories, 1), 10000)
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
            graph.index_memory(
                memory_id=memory_id,
                category=category,
                content=content,
                tags=tags,
                created_at=created_at,
                use_llm=use_llm,
            )
            indexed += 1
        except Exception as e:
            print(f"  ERROR indexing {memory_id}: {e}")
            errors += 1

        if (i + 1) % batch_size == 0 or (i + 1) == len(records):
            print(f"  Progress: {i + 1}/{len(records)} processed "
                  f"({indexed} indexed, {skipped} skipped, {errors} errors)")

    print()

    # Phase 3: Verify.
    try:
        new_nodes = graph.count_nodes()
        new_edges = graph.count_edges()
    except Exception:
        new_nodes, new_edges = -1, -1

    print("=== REBUILD COMPLETE ===")
    print(f"Indexed:  {indexed}")
    print(f"Skipped:  {skipped}")
    print(f"Errors:   {errors}")
    print()
    print(f"Graph before: {old_nodes} nodes, {old_edges} edges")
    print(f"Graph after:  {new_nodes} nodes, {new_edges} edges")
    print()
    if errors == 0:
        print("Success. You can now start Hermes.")
    else:
        print(f"Completed with {errors} errors. Review the output above.")

    graph.close()
    store._rpc.stop_service()
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

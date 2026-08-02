#!/usr/bin/env python3
"""Dump all memories from DuckDB + Kuzu graph for inspection.

Run with:
    python dump_memories.py
    python dump_memories.py --limit 50
    python dump_memories.py --json

This shows you the ACTUAL data in your memory system — not what the
agent claims to remember, but the raw records. Use it to spot:
  - Junk facts that shouldn't have been saved
  - Duplicate/near-duplicate entries
  - Missing relationships the graph should have linked
  - Tags or categories that are wrong

Recommended: run this every few days while the memory count is small.
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
        # Try importing from hermes_constants (works when run from hermes-agent dir)
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "hermes-agent"))
        from hermes_constants import get_hermes_home
        return Path(get_hermes_home())
    except Exception:
        pass
    # Fallback: check env var, then common locations.
    env = os.environ.get("HERMES_HOME")
    if env:
        return Path(env)
    # Windows: AppData/Local/hermes
    local = Path(os.path.expandvars(r"%LOCALAPPDATA%\hermes"))
    if local.exists():
        return local
    # Unix: ~/.hermes
    return Path(os.path.expanduser("~/.hermes"))


def dump_duckdb(home: Path, limit: int) -> None:
    """Dump memory records from DuckDB."""
    db_path = home / "hybrid_memory.duckdb"
    if not db_path.exists():
        print(f"  [DuckDB] Database not found at {db_path}")
        return

    try:
        import duckdb
    except ImportError:
        print("  [DuckDB] duckdb not installed — run: pip install duckdb")
        return

    try:
        conn = duckdb.connect(str(db_path), read_only=True)
    except Exception as e:
        # If read-only fails due to lock, try without read_only flag
        # (DuckDB sometimes allows this on Windows).
        msg = str(e).lower()
        if "being used by another process" in msg or "cannot access" in msg:
            print(f"  [DuckDB] Database is locked by Hermes (PID in error above).")
            print(f"  [DuckDB] Close Hermes and re-run, or use: --json with Hermes closed.")
            print(f"  [DuckDB] Error: {e}")
            return
        print(f"  [DuckDB] Cannot open database: {e}")
        return

    # Count
    try:
        count = conn.execute("SELECT COUNT(*) FROM memory_records").fetchone()[0]
    except Exception:
        print("  [DuckDB] Table 'memory_records' not found — database may be empty or uninitialized.")
        conn.close()
        return

    print(f"  [DuckDB] {db_path}")
    print(f"  [DuckDB] Total memories: {count}")
    if count == 0:
        conn.close()
        return

    print()
    print(f"  {'Created':<26} {'Category':<16} {'Content'}")
    print(f"  {'-' * 26} {'-' * 16} {'-' * 60}")

    rows = conn.execute(
        "SELECT created_at, category, content, tags, payload "
        "FROM memory_records ORDER BY created_at DESC LIMIT ?",
        [limit],
    ).fetchall()

    for created_at, category, content, tags, payload in rows:
        # Truncate content for display.
        display = content[:80] + "..." if len(content) > 80 else content
        print(f"  {str(created_at):<26} {str(category):<16} {display}")
        if tags:
            print(f"  {'':<26} {'':<16} tags: {list(tags) if not isinstance(tags, str) else tags}")

    # Category breakdown
    print()
    cats = conn.execute(
        "SELECT category, COUNT(*) FROM memory_records GROUP BY category ORDER BY COUNT(*) DESC"
    ).fetchall()
    print("  Category breakdown:")
    for cat, cnt in cats:
        print(f"    {cat:<20} {cnt}")

    # Check for potential duplicates (same content, different IDs)
    print()
    dups = conn.execute(
        "SELECT content, COUNT(*) as cnt FROM memory_records "
        "GROUP BY content HAVING cnt > 1 ORDER BY cnt DESC LIMIT 10"
    ).fetchall()
    if dups:
        print(f"  [WARNING] Potential duplicates ({len(dups)} content strings appear more than once):")
        for content, cnt in dups:
            display = content[:60] + "..." if len(content) > 60 else content
            print(f"    {cnt}x  {display}")
    else:
        print("  No duplicate content found.")

    conn.close()


def dump_kuzu(home: Path, limit: int) -> None:
    """Dump nodes and edges from Kuzu graph."""
    graph_path = home / "hybrid_memory_kuzu"
    if not graph_path.exists():
        print(f"  [Kuzu] Graph directory not found at {graph_path}")
        return

    try:
        import kuzu
    except ImportError:
        print("  [Kuzu] kuzu not installed — run: pip install kuzu")
        return

    try:
        db = kuzu.Database(str(graph_path))
        conn = kuzu.Connection(db)
    except Exception as e:
        msg = str(e).lower()
        if "lock" in msg or "could not set" in msg:
            print(f"  [Kuzu] Graph is locked by Hermes.")
            print(f"  [Kuzu] Close Hermes and re-run to dump the graph.")
            print(f"  [Kuzu] Error: {e}")
            return
        print(f"  [Kuzu] Cannot open graph: {e}")
        return

    # Count nodes
    try:
        result = conn.execute("MATCH (n:Entity) RETURN COUNT(*)")
        node_count = result.get_next()[0]
    except Exception as e:
        print(f"  [Kuzu] Cannot query nodes: {e}")
        return

    # Count edges
    try:
        result = conn.execute("MATCH ()-[r:RelatesTo]->() RETURN COUNT(*)")
        edge_count = result.get_next()[0]
    except Exception:
        edge_count = 0

    print(f"  [Kuzu] {graph_path}")
    print(f"  [Kuzu] Total nodes: {node_count}, edges: {edge_count}")
    if node_count == 0:
        return

    # Dump nodes
    print()
    print(f"  {'Node ID':<30} {'Type':<16} {'Scope'}")
    print(f"  {'-' * 30} {'-' * 16} {'-' * 20}")

    result = conn.execute(
        "MATCH (n:Entity) RETURN n.id, n.entity_type, n.user_scope "
        "ORDER BY n.entity_type, n.id LIMIT ?",
        [limit],
    )
    while result.has_next():
        row = result.get_next()
        node_id = str(row[0])[:28]
        node_type = str(row[1])[:16]
        scope = str(row[2] or "global")[:20]
        print(f"  {node_id:<30} {node_type:<16} {scope}")

    # Dump edges
    if edge_count > 0:
        print()
        print(f"  {'Source':<20} {'Relation':<25} {'Target'}")
        print(f"  {'-' * 20} {'-' * 25} {'-' * 20}")

        result = conn.execute(
            "MATCH (a:Entity)-[r:RelatesTo]->(b:Entity) "
            "RETURN a.id, r.relation_type, b.id LIMIT ?",
            [limit],
        )
        while result.has_next():
            row = result.get_next()
            src = str(row[0])[:18]
            rel = str(row[1])[:25]
            tgt = str(row[2])[:20]
            print(f"  {src:<20} {rel:<25} {tgt}")

    # Check for junk nodes
    print()
    junk_words = {
        "who", "what", "where", "when", "why", "how", "which",
        "the", "this", "that", "a", "an", "is", "are",
        "top", "best", "show", "give", "list", "i", "my", "me",
        "it", "they", "he", "she", "we", "you", "and", "or", "but",
    }
    result = conn.execute("MATCH (n:Entity) RETURN n.id")
    junk = []
    while result.has_next():
        nid = str(result.get_next()[0])
        if nid.lower() in junk_words or len(nid.strip()) <= 2:
            junk.append(nid)
    if junk:
        print(f"  [WARNING] Junk nodes found ({len(junk)}): {junk[:10]}")
    else:
        print("  No junk nodes found.")


def dump_json(home: Path, limit: int) -> None:
    """Dump everything as JSON."""
    output = {"duckdb": [], "kuzu": {"nodes": [], "edges": []}}

    db_path = home / "hybrid_memory.duckdb"
    if db_path.exists():
        try:
            import duckdb
            conn = duckdb.connect(str(db_path), read_only=True)
            rows = conn.execute(
                "SELECT memory_id, category, content, tags, payload, "
                "created_at, updated_at FROM memory_records "
                "ORDER BY created_at DESC LIMIT ?",
                [limit],
            ).fetchall()
            for row in rows:
                output["duckdb"].append({
                    "memory_id": row[0],
                    "category": row[1],
                    "content": row[2],
                    "tags": list(row[3]) if row[3] else [],
                    "payload": json.loads(row[4]) if row[4] else {},
                    "created_at": row[5],
                    "updated_at": row[6],
                })
            conn.close()
        except Exception as e:
            output["duckdb_error"] = str(e)

    graph_path = home / "hybrid_memory_kuzu"
    if graph_path.exists():
        try:
            import kuzu
            db = kuzu.Database(str(graph_path))
            conn = kuzu.Connection(db)
            result = conn.execute(
                "MATCH (n:Entity) RETURN n.id, n.entity_type, n.user_scope LIMIT ?",
                [limit],
            )
            while result.has_next():
                row = result.get_next()
                output["kuzu"]["nodes"].append({
                    "id": row[0], "type": row[1], "scope": row[2],
                })
            result = conn.execute(
                "MATCH (a:Entity)-[r:RelatesTo]->(b:Entity) "
                "RETURN a.id, r.relation_type, b.id LIMIT ?",
                [limit],
            )
            while result.has_next():
                row = result.get_next()
                output["kuzu"]["edges"].append({
                    "source": row[0], "relation": row[1], "target": row[2],
                })
        except Exception as e:
            output["kuzu_error"] = str(e)

    print(json.dumps(output, indent=2, default=str))


def main():
    parser = argparse.ArgumentParser(
        description="Dump memories from DuckDB + Kuzu graph for inspection."
    )
    parser.add_argument(
        "--limit", type=int, default=100,
        help="Max records to show per store (default: 100).",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output as JSON instead of formatted table.",
    )
    parser.add_argument(
        "--home", type=str, default=None,
        help="Override HERMES_HOME path.",
    )
    args = parser.parse_args()

    home = Path(args.home) if args.home else _get_hermes_home()
    print()
    print("=" * 70)
    print(f"  Hybrid Memory Dump — {home}")
    print("=" * 70)
    print()

    if args.json:
        dump_json(home, args.limit)
    else:
        print("--- DuckDB (Vector Store) ---")
        print()
        dump_duckdb(home, args.limit)
        print()
        print("--- Kuzu (Relationship Graph) ---")
        print()
        dump_kuzu(home, args.limit)

    print()
    print("=" * 70)
    print("  Dump complete.")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()

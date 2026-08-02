#!/usr/bin/env python3
"""Clean up junk memories from the hybrid memory database.

Run this with Hermes CLOSED (the database is locked while Hermes is running).

Usage:
    python cleanup_memories.py           # Dry run — shows what would be deleted
    python cleanup_memories.py --apply   # Actually deletes junk memories
"""
import argparse
import json
import sys
from pathlib import Path

# Resolve paths relative to this script.
PLUGIN_DIR = Path(__file__).resolve().parent
HERMES_HOME = PLUGIN_DIR.parent.parent  # plugins/ -> hermes/ -> parent

# Add plugin dir to path so we can import the modules.
sys.path.insert(0, str(PLUGIN_DIR))

from extractor import _is_junk
import duckdb


def main():
    parser = argparse.ArgumentParser(description="Clean up junk memories")
    parser.add_argument(
        "--apply", action="store_true",
        help="Actually delete memories (default: dry run)",
    )
    parser.add_argument(
        "--db", default=str(HERMES_HOME / "hybrid_memory.duckdb"),
        help="Path to the DuckDB database file",
    )
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"Database not found: {db_path}")
        sys.exit(1)

    try:
        conn = duckdb.connect(str(db_path))
    except Exception as e:
        if "being used by another process" in str(e).lower():
            print("ERROR: Database is locked. Close Hermes first.")
            sys.exit(1)
        raise

    rows = conn.execute(
        "SELECT memory_id, category, content, tags, payload, created_at "
        "FROM memory_records ORDER BY created_at DESC"
    ).fetchall()

    print(f"Total memories: {len(rows)}")
    print()

    # Identify junk.
    junk_ids = []
    seen_fingerprints = {}  # normalized[:60] -> (memory_id, content)

    for row in rows:
        mid, cat, content, tags, payload, created = row
        fact = {
            "category": cat,
            "content": content,
            "tags": list(tags) if tags else [],
            "payload": json.loads(payload) if payload else {},
        }

        # Check junk filter.
        if _is_junk(fact):
            junk_ids.append((mid, cat, content, "junk filter"))
            continue

        # Check for near-duplicates.
        normalized = content.lower().strip()[:60]
        if normalized in seen_fingerprints:
            existing_id, existing_content = seen_fingerprints[normalized]
            if len(content) > len(existing_content):
                junk_ids.append((existing_id, cat, existing_content, "duplicate (shorter)"))
                seen_fingerprints[normalized] = (mid, content)
            else:
                junk_ids.append((mid, cat, content, "duplicate (shorter)"))
        else:
            seen_fingerprints[normalized] = (mid, content)

    if not junk_ids:
        print("No junk memories found. Database is clean.")
        return

    print(f"Junk memories found: {len(junk_ids)}")
    print()
    print(f"{'ID':<20} {'Category':<16} {'Reason':<25} Content")
    print("-" * 120)
    for mid, cat, content, reason in junk_ids:
        display = content[:70] + ("..." if len(content) > 70 else "")
        print(f"{mid:<20} {cat:<16} {reason:<25} {display}")

    print()
    if not args.apply:
        print("DRY RUN — no memories were deleted.")
        print("Run with --apply to actually delete them.")
    else:
        for mid, _, _, _ in junk_ids:
            conn.execute("DELETE FROM memory_records WHERE memory_id = ?", [mid])
        conn.close()
        print(f"Deleted {len(junk_ids)} junk memories.")


if __name__ == "__main__":
    main()

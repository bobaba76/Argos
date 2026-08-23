#!/usr/bin/env python3
"""Clean up junk memories from the hybrid memory database.

Run this with Hermes CLOSED (the database is locked while Hermes is running).

Usage:
    python cleanup_memories.py               # Dry run — shows review candidates
    python cleanup_memories.py --quarantine # Hides candidates without deleting them
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

from extractor import hard_quality_flags, quality_flags_for_fact
import duckdb


def main():
    parser = argparse.ArgumentParser(description="Clean up junk memories")
    parser.add_argument(
        "--quarantine", action="store_true",
        help="Hide candidates from retrieval without deleting them",
    )
    parser.add_argument(
        "--apply", action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--db", default=str(HERMES_HOME / "hybrid_memory.duckdb"),
        help="Path to the DuckDB database file",
    )
    args = parser.parse_args()
    if args.apply:
        print("ERROR: --apply is disabled. Use --quarantine; records will not be deleted.")
        sys.exit(2)

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
        "FROM memory_records WHERE COALESCE(status, 'active') = 'active' "
        "ORDER BY created_at DESC"
    ).fetchall()

    print(f"Total memories: {len(rows)}")
    print()

    # Identify junk.
    junk_ids = []
    seen_fingerprints = {}  # (category, normalized[:60]) -> (memory_id, content)

    for row in rows:
        mid, cat, content, tags, payload, created = row
        fact = {
            "category": cat,
            "content": content,
            "tags": list(tags) if tags else [],
            "payload": json.loads(payload) if payload else {},
        }

        flags = hard_quality_flags(quality_flags_for_fact(fact))
        if flags:
            junk_ids.append((mid, cat, content, "; ".join(flags)))
            continue

        normalized = content.lower().strip()[:60]
        key = (cat, normalized)
        if key in seen_fingerprints:
            existing_id, existing_content = seen_fingerprints[key]
            if len(content) > len(existing_content):
                junk_ids.append((existing_id, cat, existing_content, "duplicate (shorter)"))
                seen_fingerprints[key] = (mid, content)
            else:
                junk_ids.append((mid, cat, content, "duplicate (shorter)"))
        else:
            seen_fingerprints[key] = (mid, content)

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
    if not args.quarantine:
        print("DRY RUN — no memories were changed.")
        print("Run with --quarantine to hide these records without deleting them.")
    else:
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        for mid, _, _, reason in junk_ids:
            conn.execute(
                """UPDATE memory_records
                   SET status = 'quarantined', quarantine_reason = ?,
                       quarantined_at = ?, updated_at = ?
                   WHERE memory_id = ?""",
                [reason, now, now, mid],
            )
        conn.close()
        print(f"Quarantined {len(junk_ids)} memories; no records were deleted.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Import legacy gateway memories into the primary proposal queue.

This is intentionally non-destructive: the gateway database is read-only and
records become pending candidates in the canonical primary store. Nothing is
deleted or made active automatically.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import duckdb

if __package__:
    from .store import DuckDBMemoryStore
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from store import DuckDBMemoryStore


def _json_object(value) -> dict:
    if isinstance(value, str):
        try:
            value = json.loads(value) if value else {}
        except json.JSONDecodeError:
            value = {}
    return value if isinstance(value, dict) else {}


def migrate(source_path: Path, target_path: Path) -> tuple[int, int]:
    source = duckdb.connect(str(source_path), read_only=True)
    target = DuckDBMemoryStore(target_path, user_id="default_user", embedder=None)
    imported = 0
    skipped = 0
    try:
        columns = [row[0] for row in source.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'memory_records' ORDER BY ordinal_position"
        ).fetchall()]
        rows = source.execute("SELECT * FROM memory_records").fetchall()
        for raw in rows:
            row = dict(zip(columns, raw))
            if str(row.get("status") or "active") != "active":
                skipped += 1
                continue
            payload = _json_object(row.get("payload"))
            payload.update({
                "legacy_store": "gateway",
                "legacy_memory_id": row.get("memory_id", ""),
                "original_source": row.get("source") or payload.get("source", "unknown"),
            })
            candidate = target.save_candidate(
                category=row.get("category") or "context_note",
                content=row.get("content") or "",
                tags=list(row.get("tags") or []),
                payload=payload,
                source="legacy_gateway",
                confidence=row.get("confidence") if row.get("confidence") is not None else 0.35,
                durability=row.get("durability") or "durable",
                scope=row.get("scope") or "profile",
                project_id=row.get("project_id"),
                session_id=f"legacy_gateway:{row.get('memory_id', '')}",
                dedup=True,
            )
            if candidate is None:
                skipped += 1
            else:
                imported += 1
    finally:
        source.close()
        target.close()
    return imported, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="Legacy gateway DuckDB file")
    parser.add_argument("--target", required=True, type=Path, help="Canonical primary DuckDB file")
    args = parser.parse_args()
    if not args.source.exists():
        raise SystemExit(f"Source database not found: {args.source}")
    if not args.target.exists():
        raise SystemExit(f"Target database not found: {args.target}")
    imported, skipped = migrate(args.source, args.target)
    print(f"Imported pending gateway proposals: {imported}")
    print(f"Skipped quarantined/duplicate/invalid gateway records: {skipped}")
    print("Source gateway database was not modified; no records were made active.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""CLI wrapper for memory_why_not (Spec 2).

Diagnose why a memory did not surface in retrieval.
Deterministic, free (no LLM), strictly read-only.

Usage:
    python why_not_cli.py --db <path> --query "what is my age" \\
        --expected-memory-id mem-abc123 [--top-k 20]

The script opens the store in read-only mode (or connects to the running
memory service if --service-url is given), runs explain_retrieval, and
prints a human-readable diagnostic report.

Exit codes:
    0 — diagnosis completed (regardless of whether the memory was found)
    1 — error (missing args, store not found, etc.)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Ensure the plugin package is importable when run as a standalone script.
_plugin_dir = Path(__file__).resolve().parent.parent
if str(_plugin_dir) not in sys.path:
    sys.path.insert(0, str(_plugin_dir))


def _open_store_readonly(db_path: str):
    """Open a DuckDBMemoryStore in read-only mode for diagnosis.

    This never writes to the store. If the DB is locked by the live
    memory service, DuckDB opens a read-only connection automatically.
    """
    import store as store_mod

    # We construct the store but NEVER call any write method.
    # explain_retrieval only reads.
    store = store_mod.DuckDBMemoryStore(db_path, user_id="default_user")
    # Force read-only by trying to open read-only first.
    try:
        import duckdb
        # Close the write connection and reopen read-only.
        store.close()
        conn = duckdb.connect(str(db_path), read_only=True)
        store.connection = conn
    except Exception:
        # If read-only fails, the store's own connection (which auto-falls
        # back to read-only on lock) is fine. Re-open if we closed it.
        if getattr(store, "connection", None) is None:
            store = store_mod.DuckDBMemoryStore(db_path, user_id="default_user")
    return store


def _format_report(explanation: dict) -> str:
    """Format the explanation dict as a human-readable report."""
    lines = []
    lines.append("=" * 60)
    lines.append("memory_why_not — retrieval diagnostic")
    lines.append("=" * 60)
    lines.append("")

    expected = explanation.get("expected")
    if expected is None:
        lines.append(f"  Expected memory: {explanation.get('expected_memory_id')}")
        lines.append("  STATUS: NOT FOUND")
        lines.append("")
        for r in explanation.get("reasons", []):
            lines.append(f"  • {r}")
        lines.append("")
        return "\n".join(lines)

    lines.append(f"  Expected memory: {expected.get('memory_id')}")
    lines.append(f"  Content: {expected.get('content', '')[:100]}")
    lines.append(f"  Category: {expected.get('category')}")
    lines.append(f"  Status: {expected.get('status')}")
    if expected.get("expires_at"):
        lines.append(f"  Expires at: {expected['expires_at']}")
    if expected.get("valid_to"):
        lines.append(f"  Superseded at: {expected['valid_to']}")
        lines.append(f"  Superseded by: {expected.get('superseded_by')}")
    lines.append("")

    found = explanation.get("found_in_results", False)
    rank = explanation.get("rank")
    if found:
        lines.append(f"  FOUND in top-{explanation.get('top_results', []) and len(explanation['top_results'])} results at rank #{rank}")
    else:
        lines.append(f"  NOT FOUND in top-{len(explanation.get('top_results', []))} results")
    lines.append("")

    lines.append("  Reasons:")
    for r in explanation.get("reasons", []):
        lines.append(f"    • {r}")
    lines.append("")

    diag = explanation.get("diagnostics", {})
    if diag:
        lines.append("  Diagnostics:")
        for k, v in sorted(diag.items()):
            lines.append(f"    {k}: {v}")
        lines.append("")

    top = explanation.get("top_results", [])
    if top:
        lines.append(f"  Top {len(top)} results:")
        for i, r in enumerate(top, 1):
            lines.append(
                f"    {i}. [{r.get('similarity', 0):.4f}] "
                f"{r.get('memory_id', '')[:12]}... "
                f"{r.get('content', '')[:60]}"
            )
        lines.append("")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Diagnose why a memory did not surface in retrieval (Spec 2).",
    )
    parser.add_argument(
        "--db", required=True,
        help="Path to hybrid_memory.duckdb (live store or a copy).",
    )
    parser.add_argument(
        "--query", required=True,
        help="The search query that failed to surface the expected memory.",
    )
    parser.add_argument(
        "--expected-memory-id", required=True,
        help="The memory_id you expected to see in results.",
    )
    parser.add_argument(
        "--top-k", type=int, default=20,
        help="How many top results to inspect (default 20).",
    )
    parser.add_argument(
        "--user-id", default="default_user",
        help="User scope (default 'default_user').",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output raw JSON instead of a formatted report.",
    )
    args = parser.parse_args()

    if not Path(args.db).exists():
        print(f"Error: database not found at {args.db}", file=sys.stderr)
        return 1

    store = _open_store_readonly(args.db)
    store.user_id = args.user_id

    try:
        explanation = store.explain_retrieval(
            args.query, args.expected_memory_id, top_k=args.top_k,
        )
    except Exception as exc:
        print(f"Error: explain_retrieval failed: {exc}", file=sys.stderr)
        return 1
    finally:
        store.close()

    if args.json:
        print(json.dumps(explanation, default=str, indent=2))
    else:
        print(_format_report(explanation))
    return 0


if __name__ == "__main__":
    sys.exit(main())

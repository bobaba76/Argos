#!/usr/bin/env python3
"""cluster_layout_families.py — Spec-09 (#112) pre-labelling clustering.

Clusters the pilot corpus by layout-family fingerprint and prints family
counts, so the gold set is built *stratified* by layout family — not a
post-hoc excuse. Per-family accuracy is reported alongside the aggregate
by ``eval_self_corpus.py`` (see ``by_layout_family`` in the run summary).

This script runs BEFORE gold-set construction:

    python eval/cluster_layout_families.py --db <snapshot>/hybrid_memory.duckdb
    # ... inspect the family counts; ensure the gold set covers each family ...

It also optionally labels families in the sidecar registry:

    python eval/cluster_layout_families.py --db <db> --label <fingerprint> "Acme invoice"

Deterministic, no LLM, no new deps.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_PLUGIN_ROOT = Path(__file__).resolve().parents[1]
if str(_PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_ROOT))


def cluster_families(db_path: Path) -> List[Dict[str, Any]]:
    """Cluster the catalog by layout_family and return family counts.

    Returns one row per distinct layout_family with doc_count and a sample
    canonical_path. Rows with NULL layout_family are reported separately
    as 'unfingerprinted'.
    """
    import duckdb
    conn = duckdb.connect(str(db_path), read_only=True)
    try:
        rows = conn.execute(
            """
            SELECT layout_family,
                   COUNT(*) AS doc_count,
                   MIN(canonical_path) AS sample_path,
                   MIN(doc_type) AS sample_doc_type
            FROM file_catalog
            WHERE status = 'active'
            GROUP BY layout_family
            ORDER BY doc_count DESC
            """
        ).fetchall()
    finally:
        conn.close()
    cols = ["layout_family", "doc_count", "sample_path", "sample_doc_type"]
    return [dict(zip(cols, r)) for r in rows]


def stratified_sample_plan(
    families: List[Dict[str, Any]],
    *,
    total: int = 200,
    min_per_family: int = 3,
) -> Dict[str, int]:
    """Compute a stratified per-family sample allocation.

    Proportional to family size, with a floor of ``min_per_family`` for
    small families (so rare layouts are represented, not drowned). Caps
    at ``total``. Families with NULL fingerprint are excluded from the
    stratified plan (they go in the 'none' bucket of the eval).
    """
    labelled = [f for f in families if f.get("layout_family")]
    total_docs = sum(f["doc_count"] for f in labelled)
    if total_docs == 0:
        return {}
    plan: Dict[str, int] = {}
    # First pass: assign the floor.
    remaining = total
    for f in labelled:
        alloc = min(min_per_family, remaining)
        plan[f["layout_family"]] = alloc
        remaining -= alloc
        if remaining <= 0:
            return plan
    # Second pass: proportional fill of the remainder.
    for f in labelled:
        if remaining <= 0:
            break
        proportional = max(0, int(total * f["doc_count"] / total_docs) - plan[f["layout_family"]])
        add = min(proportional, remaining)
        plan[f["layout_family"]] += add
        remaining -= add
    return plan


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Cluster the pilot corpus by layout-family fingerprint (Spec-09 #112).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--db", required=True, help="Path to hybrid_memory.duckdb (snapshot or live).")
    parser.add_argument("--total", type=int, default=200, help="Target gold-set size for the stratified plan (default 200).")
    parser.add_argument("--min-per-family", type=int, default=3, help="Floor per family in the stratified plan (default 3).")
    parser.add_argument("--label", nargs=2, metavar=("FINGERPRINT", "LABEL"), help="Label a family in the sidecar registry.")
    parser.add_argument("--registry", default="", help="Path to layout_families.json (default: alongside the db).")
    args = parser.parse_args(argv)

    db_path = Path(args.db).resolve()
    if not db_path.exists():
        print(f"ERROR: DB not found: {db_path}", file=sys.stderr)
        return 2

    families = cluster_families(db_path)

    # Separate fingerprinted from unfingerprinted.
    unfingerprinted = [f for f in families if not f.get("layout_family")]
    fingerprinted = [f for f in families if f.get("layout_family")]

    print(f"=== Layout-family clustering ({db_path.name}) ===", flush=True)
    print(f"Fingerprinted families: {len(fingerprinted)}", flush=True)
    print(f"Unfingerprinted docs: {unfingerprinted[0]['doc_count'] if unfingerprinted else 0}", flush=True)
    print(flush=True)
    print("Family counts (descending):", flush=True)
    for f in fingerprinted:
        fp = f["layout_family"]
        print(f"  {fp[:16]}  n={f['doc_count']:5d}  type={f.get('sample_doc_type') or '?':5s}  sample={f.get('sample_path') or ''}", flush=True)
    print(flush=True)

    plan = stratified_sample_plan(fingerprinted, total=args.total, min_per_family=args.min_per_family)
    print(f"Stratified gold-set plan (total={args.total}, min/family={args.min_per_family}):", flush=True)
    for fp, n in plan.items():
        print(f"  {fp[:16]}  -> {n} probes", flush=True)
    print(flush=True)
    print("Use this plan to build the gold set stratified by layout family.", flush=True)

    if args.label:
        fp, label = args.label
        try:
            from layout_family_registry import registry_path_for, label_family
            reg_path = args.registry or str(registry_path_for(db_path))
            doc_count = next((f["doc_count"] for f in fingerprinted if f["layout_family"] == fp), None)
            if label_family(reg_path, fp, label, doc_count=doc_count):
                print(f"Labelled {fp[:16]} -> '{label}' in {reg_path}", flush=True)
            else:
                print(f"ERROR: failed to label {fp}", file=sys.stderr)
                return 1
        except Exception as exc:
            print(f"ERROR: labelling failed: {exc}", file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

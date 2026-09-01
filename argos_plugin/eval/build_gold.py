#!/usr/bin/env python3
"""build_gold.py — build the reviewable gold set for the regression gate.

Samples memories from a SNAPSHOT db (never the live store), generates one
probe query per memory, and emits a reviewable JSONL the user validates
once before it freezes.  Freezing records the sha256 of the validated
set plus the snapshot it was built from in ``gold_manifest.json``.

Gold sets carry personal memory content — never commit them (the repo
gitignores ``eval/gold/*.jsonl``; only the manifest sha is committed).

Usage:
    python eval/build_gold.py --db <snapshot>/hybrid_memory.duckdb [--limit 200]
    # ... user reviews eval/gold/gold_v1.jsonl (status: approved/rejected) ...
    python eval/build_gold.py --db <snapshot>/hybrid_memory.duckdb --freeze
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_PLUGIN_ROOT = Path(__file__).resolve().parents[1]
if str(_PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_ROOT))
_EVAL_DIR = Path(__file__).resolve().parent
if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))

import eval_self_corpus as esc  # noqa: E402

DEFAULT_GOLD = Path(__file__).resolve().parent / "gold" / "gold_v1.jsonl"
DEFAULT_LIMIT = 200
MIN_LIMIT = 150
MAX_LIMIT = 1000
MIN_APPROVED = 50
_SHA_KEYS = ("memory_id", "category", "query", "template", "layout_family")


def gold_sha256(lines: List[Dict[str, Any]]) -> str:
    """Canonical sha256 over the approved gold lines (stable across runs)."""
    approved = sorted(
        (l for l in lines if l.get("status") == "approved"),
        key=lambda l: l.get("memory_id", ""),
    )
    blob = "\n".join(
        json.dumps(
            {k: l.get(k) for k in _SHA_KEYS}, sort_keys=True, ensure_ascii=False
        )
        for l in approved
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def load_gold(path: Path) -> List[Dict[str, Any]]:
    """Load gold JSONL lines (skips malformed/empty lines)."""
    lines: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and obj.get("memory_id") and obj.get("query"):
                lines.append(obj)
    return lines


def build_gold(
    db_path: Path,
    out_path: Path = DEFAULT_GOLD,
    limit: int = DEFAULT_LIMIT,
    seed: int = 42,
    user_id: str = "default_user",
    auto_approve: bool = False,
) -> List[Dict[str, Any]]:
    """Sample memories + one probe each; write the reviewable JSONL."""
    import duckdb
    conn = duckdb.connect(str(db_path), read_only=True)
    try:
        memories = esc._sample_memories(conn, user_id, limit, seed)
    finally:
        conn.close()
    if not memories:
        raise RuntimeError("No active memories found to sample.")
    assigned = esc.assign_templates(memories)
    lines: List[Dict[str, Any]] = []
    for mem, template in assigned:
        probe = esc.generate_probe(mem, template)
        if probe is None:
            continue
        lines.append({
            "memory_id": mem["memory_id"],
            "category": mem.get("category") or "context_note",
            "content": mem.get("content") or "",
            "query": probe["query"],
            "template": probe["template"],
            "layout_family": mem.get("layout_family"),
            "status": "approved" if auto_approve else "pending",
        })
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    return lines


def freeze_gold(out_path: Path, db_path: Path) -> Dict[str, Any]:
    """Validate the reviewed gold set and write gold_manifest.json."""
    lines = load_gold(out_path)
    pending = [l for l in lines if l.get("status") not in {"approved", "rejected"}]
    if pending:
        raise RuntimeError(
            f"{len(pending)} lines still pending review; freeze requires every "
            "line to be approved or rejected."
        )
    approved = [l for l in lines if l.get("status") == "approved"]
    if len(approved) < MIN_APPROVED:
        raise RuntimeError(
            f"Only {len(approved)} approved queries (need >= {MIN_APPROVED})."
        )
    snapshot_id = None
    snapshot_db_sha = None
    manifest_path = Path(db_path).parent / "manifest.json"
    if manifest_path.exists():
        try:
            m = json.loads(manifest_path.read_text(encoding="utf-8"))
            snapshot_id = m.get("snapshot_id")
            snapshot_db_sha = m.get("db_sha256")
        except Exception:
            pass
    gold_manifest = {
        "gold_file": str(Path(out_path).resolve()),
        "sha256": gold_sha256(lines),
        "snapshot_id": snapshot_id,
        "snapshot_db_sha256": snapshot_db_sha,
        "approved_count": len(approved),
        "rejected_count": len(lines) - len(approved),
        "created_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }
    manifest_path_out = Path(out_path).parent / "gold_manifest.json"
    manifest_path_out.write_text(
        json.dumps(gold_manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return gold_manifest


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build/freeze the reviewable gold set for the regression gate.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--db", required=True, help="Path to a SNAPSHOT hybrid_memory.duckdb (never the live store).")
    parser.add_argument("--out", default=str(DEFAULT_GOLD), help=f"Gold JSONL path (default '{DEFAULT_GOLD}'). Never commit.")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help=f"Memories sampled ({MIN_LIMIT}-{MAX_LIMIT}; default {DEFAULT_LIMIT}).")
    parser.add_argument("--seed", type=int, default=42, help="Reproducible sampling seed (default 42).")
    parser.add_argument("--user-id", default="default_user")
    parser.add_argument("--auto-approve", action="store_true", help="Mark every probe approved (MECHANICAL VERIFICATION ONLY — the real freeze requires human review).")
    parser.add_argument("--freeze", action="store_true", help="Validate the reviewed JSONL and write gold_manifest.json.")
    args = parser.parse_args(argv)

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"ERROR: DB not found: {db_path}", file=sys.stderr)
        return 2
    limit = max(MIN_LIMIT, min(args.limit, MAX_LIMIT))

    if args.freeze:
        manifest = freeze_gold(Path(args.out), db_path)
        print(f"FROZEN: {manifest['approved_count']} approved, {manifest['rejected_count']} rejected")
        print(f"  sha256: {manifest['sha256']}")
        print(f"  snapshot: {manifest['snapshot_id']}")
        return 0

    lines = build_gold(
        db_path, Path(args.out), limit=limit, seed=args.seed,
        user_id=args.user_id, auto_approve=args.auto_approve,
    )
    by_cat: Dict[str, int] = {}
    by_fam: Dict[str, int] = {}
    for l in lines:
        by_cat[l["category"]] = by_cat.get(l["category"], 0) + 1
        fam = l.get("layout_family") or "none"
        by_fam[fam] = by_fam.get(fam, 0) + 1
    print(f"Wrote {len(lines)} probes to {Path(args.out)}")
    print(f"  categories: {by_cat}")
    print(f"  layout_families: {by_fam}")
    print("  Review the JSONL (status: approved/rejected), then re-run with --freeze.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

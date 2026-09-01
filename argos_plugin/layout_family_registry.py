"""Spec-09 (#112): layout-family label registry (sidecar).

The catalog row stores the *fingerprint* (factual, per-document, immutable
for a given file). The *human label* ("Acme invoice template", "Payroll
summary — weekly") is a registry that maps fingerprint → label, and changes
as humans label or relabel a family. Mixing mutable labels into the catalog
row would couple them; a sidecar keeps the catalog purely factual.

This module is a tiny JSON-file registry: ``{fingerprint: {label, labelled_at,
labelled_by, doc_count_at_labelling}}``. It is loaded by the labelling
surface and by the eval stratification script (so per-family accuracy can
be reported with human-readable names).

Design:
- One JSON file per store (default: alongside the DuckDB, named
  ``layout_families.json``). Principals-only write; any reader can read.
- Append-only-ish: relabelling overwrites the label but keeps a
  ``label_history`` trail (audit).
- Never raises — missing/corrupt file returns an empty registry.
- No LLM, no deps.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_REGISTRY_NAME = "layout_families.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def registry_path_for(db_path: str | Path, name: str = DEFAULT_REGISTRY_NAME) -> Path:
    """Default registry path sits next to the DuckDB file."""
    return Path(db_path).parent / name


def load_registry(path: str | Path) -> Dict[str, Dict[str, Any]]:
    """Load the label registry. Returns {} on missing/corrupt file."""
    p = Path(path)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except Exception as exc:
        logger.warning("layout family registry corrupt at %s: %s", path, exc)
    return {}


def save_registry(path: str | Path, registry: Dict[str, Dict[str, Any]]) -> bool:
    """Persist the registry. Returns True on success."""
    p = Path(path)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(registry, indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        return True
    except Exception as exc:
        logger.warning("layout family registry save failed at %s: %s", path, exc)
        return False


def label_family(
    path: str | Path,
    fingerprint: str,
    label: str,
    *,
    labelled_by: str = "",
    doc_count: Optional[int] = None,
) -> bool:
    """Label a layout family (or relabel, keeping history).

    Returns True on success. Relabelling appends the old label to
    ``label_history`` (audit trail).
    """
    registry = load_registry(path)
    existing = registry.get(fingerprint, {})
    history: List[Dict[str, Any]] = list(existing.get("label_history", []))
    if existing.get("label") and existing["label"] != label:
        history.append({
            "label": existing["label"],
            "labelled_at": existing.get("labelled_at", ""),
            "labelled_by": existing.get("labelled_by", ""),
        })
    registry[fingerprint] = {
        "label": label,
        "labelled_at": _now(),
        "labelled_by": labelled_by,
        "doc_count_at_labelling": doc_count,
        "label_history": history,
    }
    return save_registry(path, registry)


def get_label(path: str | Path, fingerprint: str) -> Optional[str]:
    """Look up the human label for a fingerprint, or None if unlabelled."""
    return load_registry(path).get(fingerprint, {}).get("label")


def labelled_families(path: str | Path) -> Dict[str, str]:
    """Return {fingerprint: label} for all labelled families."""
    reg = load_registry(path)
    return {fp: entry["label"] for fp, entry in reg.items() if entry.get("label")}


def merge_catalog_counts(
    registry: Dict[str, Dict[str, Any]],
    family_counts: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Merge live catalog doc-counts into a loaded registry (in-memory).

    ``family_counts`` is the output of ``store.list_layout_families()``:
    ``[{layout_family, doc_count}, ...]``. Returns a copy of the registry
    with a ``doc_count`` field refreshed on each entry; unlabelled families
    from the catalog are included with an empty label (so the labelling
    surface can show "needs label" rows).
    """
    merged: Dict[str, Dict[str, Any]] = {}
    for fp, entry in registry.items():
        merged[fp] = dict(entry)
    for row in family_counts:
        fp = row.get("layout_family")
        if not fp:
            continue
        if fp not in merged:
            merged[fp] = {"label": "", "label_history": []}
        merged[fp]["doc_count"] = int(row.get("doc_count", 0))
    return merged

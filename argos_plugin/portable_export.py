"""#294: Portable export/import — anti-lock-in, data sovereignty.

Pure format module (no store dependency): versioned export payload,
deterministic JSONL serialization, human-readable Markdown digest, and
the import-side parser/validator. The store-side gather/apply lives in
``store_maintenance.export_portable`` / ``import_portable``.

Format (versioned from day one, per #288's model):

    Line 1  : header  {"export_format": "argos-portable",
                       "export_version": 1, "schema_version": <int>,
                       "scope": {...}, "counts": {...}}
    Line 2+ : row     {"type": "record"|"evidence"|"tombstone"|"receipt"
                               |"candidate"|"rejection"|"alias",
                       "data": {...}}

Determinism: rows are ordered by their primary key and every line is
serialized with ``sort_keys=True`` — the same store state produces
byte-identical JSONL. The export timestamp lives ONLY in the Markdown
digest (a human document), never in the JSONL, so byte-stability holds.

Completeness (anti-lock-in): the export carries ALL record fields
(provenance, version chains valid_from/valid_to/superseded_by,
tombstones, embedding metadata), the evidence rows, pending candidates,
the rejection ledger, entity aliases, deletion tombstones, and the
POPIA deletion receipts. Nothing user-owned is left behind. The graph
is NOT exported — it is a derived index rebuilt by backfill_graph.py /
rebuild_graph.py (documented in the store methods).
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

EXPORT_FORMAT = "argos-portable"
EXPORT_VERSION = 1

# Row types in the JSONL stream, with their identity keys (used for
# deterministic ordering and idempotent import).
ROW_TYPES = ("record", "evidence", "tombstone", "receipt",
             "candidate", "rejection", "alias")

# Required key per row type (validation on import).
_REQUIRED_KEYS: Dict[str, Tuple[str, ...]] = {
    "record": ("memory_id", "category", "content"),
    "evidence": ("memory_id",),
    "tombstone": ("content_hash", "category"),
    "receipt": ("receipt_id",),
    "candidate": ("candidate_id",),
    "rejection": ("subject", "predicate"),
    "alias": ("alias", "canonical_entity"),
}

# Identity key used to order each row type deterministically.
_ORDER_KEY: Dict[str, str] = {
    "record": "memory_id",
    "evidence": "memory_id",
    "tombstone": "content_hash",
    "receipt": "receipt_id",
    "candidate": "candidate_id",
    "rejection": "subject",
    "alias": "alias",
}


class PortableFormatError(ValueError):
    """Import refused: malformed or unsupported export format."""


def make_header(
    *,
    schema_version: int,
    scope: Dict[str, Any],
    counts: Dict[str, int],
) -> Dict[str, Any]:
    """Build the export header (deterministic — no timestamps)."""
    return {
        "export_format": EXPORT_FORMAT,
        "export_version": EXPORT_VERSION,
        "schema_version": int(schema_version),
        "scope": scope or {},
        "counts": counts or {},
    }


def serialize_export(header: Dict[str, Any], rows: List[Dict[str, Any]]) -> str:
    """Serialize an export to deterministic JSONL.

    Byte-stable: rows are ordered by type then identity key, and every
    line is JSON with sorted keys. No timestamps anywhere in the JSONL.
    """
    lines: List[str] = [json.dumps(header, sort_keys=True)]
    for row in rows:
        lines.append(json.dumps(row, sort_keys=True, default=str))
    return "\n".join(lines) + "\n"


def order_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deterministic row order: by declared type order, then identity key."""
    type_rank = {t: i for i, t in enumerate(ROW_TYPES)}
    def _key(row: Dict[str, Any]):
        rtype = str(row.get("type", ""))
        data = row.get("data") or {}
        ident = str(data.get(_ORDER_KEYS.get(rtype, "memory_id"), ""))
        return (type_rank.get(rtype, 99), ident)
    return sorted(rows, key=_key)


_ORDER_KEYS = {
    "record": "memory_id",
    "evidence": "memory_id",
    "tombstone": "content_hash",
    "receipt": "receipt_id",
    "candidate": "candidate_id",
    "rejection": "subject",
    "alias": "alias",
}


def render_markdown(
    header: Dict[str, Any],
    rows: List[Dict[str, Any]],
    *,
    generated_at: str = "",
    source: str = "",
) -> str:
    """Human-readable Markdown digest of an export.

    Covers EVERY record (id, category, timestamps, content, provenance)
    plus summary sections for evidence, tombstones, receipts,
    candidates, rejections and aliases. Deterministic ordering (same as
    the JSONL); the generation timestamp appears here only — the JSONL
    stays byte-stable.
    """
    counts = header.get("counts") or {}
    lines: List[str] = []
    lines.append("# Argos portable export")
    lines.append("")
    lines.append(f"- format: {header.get('export_format')}")
    lines.append(f"- export_version: {header.get('export_version')}")
    lines.append(f"- schema_version: {header.get('schema_version')}")
    if source:
        lines.append(f"- source: {source}")
    if generated_at:
        lines.append(f"- generated_at: {generated_at}")
    scope = header.get("scope") or {}
    if scope:
        lines.append(f"- scope: {json.dumps(scope, sort_keys=True)}")
    lines.append(f"- counts: {json.dumps(counts, sort_keys=True)}")
    lines.append("")

    records = [r["data"] for r in rows if r.get("type") == "record"]
    lines.append(f"## Records ({len(records)})")
    lines.append("")
    for rec in records:
        lines.append(f"### {rec.get('memory_id')} ({rec.get('category')})")
        lines.append(f"- created: {rec.get('created_at')}")
        lines.append(f"- updated: {rec.get('updated_at')}")
        lines.append(f"- status: {rec.get('status')}")
        vf = rec.get("valid_from")
        vt = rec.get("valid_to")
        lines.append(f"- valid window: [{vf or ''} .. {vt or 'current'}]")
        if rec.get("superseded_by"):
            lines.append(f"- superseded_by: {rec.get('superseded_by')}")
        lines.append(f"- provenance: origin={rec.get('provenance_origin')}"
                     f" grounding={rec.get('grounding')}"
                     f" source={rec.get('source')}"
                     f" source_doc_id={rec.get('source_doc_id')}")
        if rec.get("embedder_id"):
            lines.append(f"- embedding: dim={rec.get('embedding_dim')}"
                         f" embedder={rec.get('embedder_id')}"
                         f" embedded_at={rec.get('embedded_at')}")
        content = str(rec.get("content") or "").replace("\n", " ")
        lines.append(f"- content: {content}")
        lines.append("")

    def _section(title: str, rtype: str, fields: List[str]) -> None:
        items = [r["data"] for r in rows if r.get("type") == rtype]
        lines.append(f"## {title} ({len(items)})")
        lines.append("")
        for item in items:
            parts = [f"{f}: {item.get(f)}" for f in fields]
            lines.append(f"- {' | '.join(parts)}")
        lines.append("")

    _section("Evidence", "evidence",
             ["memory_id", "evidence_role", "source_session_id",
              "reviewer_decision"])
    _section("Deletion tombstones", "tombstone",
             ["content_hash", "category", "reason"])
    _section("Deletion receipts", "receipt",
             ["receipt_id", "subject", "memory_id", "requested_by",
              "outcome"])
    _section("Pending candidates", "candidate",
             ["candidate_id", "category", "status"])
    _section("Rejection ledger", "rejection",
             ["subject", "predicate", "reason"])
    _section("Entity aliases", "alias", ["alias", "canonical_entity"])
    return "\n".join(lines) + "\n"


def parse_import(text: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Parse + validate an exported JSONL document.

    Returns ``(header, rows, errors)``. Refuses unknown/newer format
    versions (fail loud). Every malformed line is collected with its
    line number — never silently skipped.
    """
    errors: List[Dict[str, Any]] = []
    if not isinstance(text, str) or not text.strip():
        raise PortableFormatError("import data is empty")
    lines = text.strip().splitlines()
    if not lines:
        raise PortableFormatError("import data has no lines")

    # 1. Header.
    try:
        header = json.loads(lines[0])
    except json.JSONDecodeError as exc:
        raise PortableFormatError(f"malformed header: {exc}") from exc
    if not isinstance(header, dict):
        raise PortableFormatError("header must be a JSON object")
    fmt = header.get("export_format")
    if fmt != EXPORT_FORMAT:
        raise PortableFormatError(
            f"unsupported export format {fmt!r} (expected {EXPORT_FORMAT!r})"
        )
    try:
        version = int(header.get("export_version"))
    except (TypeError, ValueError):
        raise PortableFormatError("header.export_version must be an integer")
    if version > EXPORT_VERSION:
        raise PortableFormatError(
            f"export version {version} is newer than this build supports "
            f"({EXPORT_VERSION}) — upgrade Argos to import it"
        )
    if version < 1:
        raise PortableFormatError(f"invalid export version {version}")

    # 2. Rows.
    rows: List[Dict[str, Any]] = []
    for lineno, line in enumerate(lines[1:], start=2):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append({"line": lineno, "errors": [f"malformed JSON: {exc}"]})
            continue
        if not isinstance(row, dict) or "type" not in row or "data" not in row:
            errors.append({
                "line": lineno,
                "errors": ["row must be an object with 'type' and 'data'"],
            })
            continue
        rtype = row["type"]
        if rtype not in ROW_TYPES:
            errors.append({
                "line": lineno,
                "errors": [f"unknown row type {rtype!r}"],
            })
            continue
        data = row["data"]
        if not isinstance(data, dict):
            errors.append({
                "line": lineno,
                "errors": ["row data must be an object"],
            })
            continue
        missing = [k for k in _REQUIRED_KEYS[rtype] if not data.get(k)]
        if missing:
            errors.append({
                "line": lineno,
                "errors": [
                    f"missing required field {k!r} for {rtype}" for k in missing
                ],
            })
            continue
        rows.append({"type": rtype, "data": data, "line": lineno})

    return header, rows, errors

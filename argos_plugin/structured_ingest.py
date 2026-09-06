"""#289: Structured ingestion — JSON/CSV → memory facts with provenance.

Parses structured tabular data (JSON arrays of objects, CSV with
headers), validates it against a field-mapping spec, and maps each row
to a memory fact. This module is PURE (no store dependency) so parsing
and validation are testable in isolation; the store-side apply/preview
logic lives in ``store_write.ingest_structured``.

Scope guard: structured numeric/tabular data ONLY — this is NOT another
document-extraction path (the watcher covers documents). No changes to
retrieval or ranking.

Design:
  - Mapping spec: which fields → which memory fact. A ``category``
    (must be a valid store category), a ``content_template`` with
    ``{field}`` placeholders, optional per-field type/required
    constraints, an optional ``key_field`` (stable row identity for
    supersession), and optional static/derived tags.
  - Validation is fail-loud with a PER-ROW report: malformed rows are
    collected with row numbers and reasons — never silently skipped.
  - Template rendering uses safe ``{field}`` substitution (regex-based,
    NOT ``str.format`` — a mapping template is untrusted input and
    ``str.format`` would allow attribute traversal like
    ``{a.__class__}``).
  - Provenance identity: ``mapping_id`` = sha256 of the canonical
    mapping spec (12 hex chars). Every mapped fact carries
    (source_name, row_number, mapping_id) so the store can stamp
    first-class provenance on everything it writes.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import re
from typing import Any, Dict, List, Tuple

try:
    from .store_common import VALID_CATEGORIES
except ImportError:
    from store_common import VALID_CATEGORIES


class IngestError(Exception):
    """Batch-level ingest failure (malformed input, bad spec).

    Fail-loud: the caller receives the reason and NOTHING is written.
    """


# {field} placeholder in content_template — safe subset (identifier chars
# only; no attribute access, no format specs, no indexing).
_TEMPLATE_REF_RE = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")

_SUPPORTED_TYPES = frozenset({"str", "int", "float", "bool"})


def mapping_id(mapping: Dict[str, Any]) -> str:
    """Stable identity for a mapping spec (sha256 of canonical JSON, 12 hex)."""
    canonical = json.dumps(mapping, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def validate_mapping(mapping: Dict[str, Any]) -> Dict[str, Any]:
    """Validate a field-mapping spec; return the normalized spec.

    Raises IngestError (fail-loud) on a broken spec — a bad spec must
    never produce a partially-mapped batch.

    Required keys:
      category: target memory category (must be in VALID_CATEGORIES).
      content_template: fact template with ``{field}`` placeholders.

    Optional keys:
      fields: {name: {"type": "str"|"int"|"float"|"bool",
                      "required": bool}} — per-field validation.
      key_field: stable row-identity field (enables row-level
                 supersession across re-ingests).
      tags: static tags applied to every row's fact.
      tags_fields: field names whose values are appended as tags.
      client_scope / doc_class: optional ACL metadata stamped on rows.
    """
    if not isinstance(mapping, dict):
        raise IngestError("mapping spec must be a dict")
    category = str(mapping.get("category", "")).strip()
    if not category:
        raise IngestError("mapping.category is required")
    if category not in VALID_CATEGORIES:
        raise IngestError(
            f"mapping.category {category!r} is not a valid category "
            f"(must be one of {sorted(VALID_CATEGORIES)})"
        )
    template = mapping.get("content_template")
    if not isinstance(template, str) or not template.strip():
        raise IngestError("mapping.content_template is required")
    refs = set(_TEMPLATE_REF_RE.findall(template))
    if not refs:
        raise IngestError(
            "mapping.content_template must reference at least one "
            "{field} placeholder"
        )
    fields = mapping.get("fields") or {}
    if not isinstance(fields, dict):
        raise IngestError("mapping.fields must be a dict")
    normalized_fields: Dict[str, Dict[str, Any]] = {}
    for name, spec in fields.items():
        if not isinstance(spec, dict):
            raise IngestError(f"mapping.fields[{name!r}] must be a dict")
        ftype = str(spec.get("type", "str")).strip().lower()
        if ftype not in _SUPPORTED_TYPES:
            raise IngestError(
                f"mapping.fields[{name!r}].type must be one of "
                f"{sorted(_SUPPORTED_TYPES)}"
            )
        normalized_fields[str(name)] = {
            "type": ftype,
            "required": bool(spec.get("required", False)),
        }
    # Every template ref must be a known field (or at least present in
    # rows — unknown refs fail at map time with a per-row error; here we
    # only reject refs that are neither declared nor plausibly row keys).
    key_field = mapping.get("key_field")
    if key_field is not None:
        key_field = str(key_field).strip()
        if not key_field:
            raise IngestError("mapping.key_field must not be empty")
    tags = mapping.get("tags") or []
    if not isinstance(tags, list) or not all(
        isinstance(t, str) for t in tags
    ):
        raise IngestError("mapping.tags must be a list of strings")
    tags_fields = mapping.get("tags_fields") or []
    if not isinstance(tags_fields, list) or not all(
        isinstance(t, str) for t in tags_fields
    ):
        raise IngestError("mapping.tags_fields must be a list of strings")
    client_scope = mapping.get("client_scope")
    doc_class = mapping.get("doc_class")
    namespace = mapping.get("namespace")
    return {
        "category": category,
        "content_template": template,
        "template_refs": sorted(refs),
        "fields": normalized_fields,
        "key_field": key_field,
        "tags": [str(t) for t in tags],
        "tags_fields": [str(t) for t in tags_fields],
        "client_scope": str(client_scope) if client_scope else None,
        "doc_class": str(doc_class) if doc_class else None,
        "namespace": str(namespace) if namespace else None,
    }


def parse_rows(data: Any, fmt: str) -> List[Dict[str, Any]]:
    """Parse raw JSON/CSV input into a list of row dicts.

    Fail-loud: malformed input raises IngestError with the reason.
    CSV requires a header row (column names become field names).
    """
    fmt = str(fmt or "").strip().lower()
    if fmt not in {"json", "csv"}:
        raise IngestError(f"unsupported ingest format {fmt!r} (json|csv)")
    if not isinstance(data, str) or not data.strip():
        raise IngestError("ingest data is empty")
    if fmt == "json":
        return _parse_json_rows(data)
    return _parse_csv_rows(data)


def _parse_json_rows(data: str) -> List[Dict[str, Any]]:
    try:
        parsed = json.loads(data)
    except json.JSONDecodeError as exc:
        raise IngestError(f"malformed JSON: {exc}") from exc
    if isinstance(parsed, dict):
        # Allow {"rows": [...]} wrappers; a single object is one row.
        if isinstance(parsed.get("rows"), list):
            parsed = parsed["rows"]
        else:
            parsed = [parsed]
    if not isinstance(parsed, list):
        raise IngestError("JSON ingest data must be a list of objects")
    rows: List[Dict[str, Any]] = []
    for i, item in enumerate(parsed):
        if not isinstance(item, dict):
            raise IngestError(
                f"row {i}: JSON rows must be objects, got "
                f"{type(item).__name__}"
            )
        rows.append(item)
    return rows


def _parse_csv_rows(data: str) -> List[Dict[str, Any]]:
    reader = csv.DictReader(io.StringIO(data))
    if reader.fieldnames is None:
        raise IngestError("CSV ingest data has no header row")
    fieldnames = [f.strip() for f in reader.fieldnames if f and f.strip()]
    if not fieldnames:
        raise IngestError("CSV ingest data has an empty header row")
    rows: List[Dict[str, Any]] = []
    for i, raw in enumerate(reader):
        # Normalize: strip header whitespace, drop the None restkey
        # (extra cells) but KEEP None values so required-field checks
        # can report them as missing per row.
        row: Dict[str, Any] = {}
        for k, v in raw.items():
            if k is None:
                # Extra cells beyond the header — ragged row.
                if v:
                    row["__extra__"] = True
                continue
            row[str(k).strip()] = v
        rows.append(row)
    return rows


def _coerce_value(value: Any, ftype: str) -> Any:
    """Coerce/validate a raw cell to the declared field type.

    Raises ValueError/TypeError on impossible coercions (reported
    per-row, never silently coerced to junk).
    """
    if value is None:
        return None
    if ftype == "str":
        return str(value)
    if ftype == "int":
        if isinstance(value, bool):
            raise ValueError("bool is not a valid int")
        if isinstance(value, int):
            return value
        return int(str(value).strip())
    if ftype == "float":
        if isinstance(value, bool):
            raise ValueError("bool is not a valid float")
        if isinstance(value, (int, float)):
            return float(value)
        return float(str(value).strip())
    if ftype == "bool":
        if isinstance(value, bool):
            return value
        v = str(value).strip().lower()
        if v in {"true", "1", "yes", "y"}:
            return True
        if v in {"false", "0", "no"}:
            return False
        raise ValueError(f"cannot parse {value!r} as bool")
    raise ValueError(f"unsupported field type {ftype!r}")


def render_template(template: str, row: Dict[str, Any]) -> str:
    """Safely render ``{field}`` placeholders from a row.

    Manual substitution — NOT ``str.format`` — so a mapping template can
    never trigger attribute traversal / format-spec attacks
    (``{a.__class__}``, ``{x:{y}}``). Unknown references raise KeyError.
    """
    def _sub(match: re.Match) -> str:
        key = match.group(1)
        if key not in row:
            raise KeyError(key)
        val = row[key]
        return "" if val is None else str(val)

    return _TEMPLATE_REF_RE.sub(_sub, template)


def map_rows(
    rows: List[Dict[str, Any]],
    mapping: Dict[str, Any],
    source_name: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Validate + map parsed rows to memory facts.

    Returns ``(facts, row_errors)``. Every failure carries its row
    number and reason — malformed rows are NEVER silently skipped.

    Each fact dict:
      row_number: 1-based source row number
      category, content, tags: memory fields
      key_value: row identity (when key_field is set)
      source_doc_id: stable row identity for doc-scoped dedupe
      fields: the validated/coerced row values (goes into payload)
    """
    spec = validate_mapping(mapping)
    category = spec["category"]
    template = spec["content_template"]
    fields_spec = spec["fields"]
    key_field = spec["key_field"]

    # Template refs must resolve against declared fields OR be present
    # in the rows. Declared-but-unreferenced fields may still be
    # required (existence validation).
    required_names = {
        name for name, fs in fields_spec.items() if fs["required"]
    } | set(spec["template_refs"])
    if key_field:
        required_names.add(key_field)

    facts: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    for i, raw_row in enumerate(rows):
        row_number = i + 1
        row_errors: List[str] = []
        row: Dict[str, Any] = {}
        for name in sorted(set(list(raw_row.keys()) + list(fields_spec.keys()))):
            if name == "__extra__":
                row_errors.append("row has more cells than the header")
                continue
            raw = raw_row.get(name)
            ftype = fields_spec.get(name, {}).get("type", "str")
            try:
                row[name] = _coerce_value(raw, ftype)
            except (ValueError, TypeError) as exc:
                row_errors.append(f"field {name!r}: {exc}")

        # Required-field check (missing or empty).
        for name in sorted(required_names):
            val = row.get(name)
            if val is None or (isinstance(val, str) and not val.strip()):
                row_errors.append(f"missing required field {name!r}")

        # Render the fact text (safe substitution).
        content = ""
        if not row_errors:
            try:
                content = render_template(template, row).strip()
            except KeyError as exc:
                row_errors.append(f"template references unknown field {exc}")
            except Exception as exc:
                row_errors.append(f"template render failed: {exc}")
            if not row_errors and not content:
                row_errors.append("mapped content is empty")

        if row_errors:
            errors.append({
                "row_number": row_number,
                "errors": row_errors,
            })
            continue

        tags = list(spec["tags"])
        for tf in spec["tags_fields"]:
            val = row.get(tf)
            if val is not None and str(val).strip():
                tags.append(str(val).strip())

        key_value = str(row.get(key_field)).strip() if key_field else None
        facts.append({
            "row_number": row_number,
            "category": category,
            "content": content,
            "tags": tags,
            "key_value": key_value,
            "source_doc_id": (
                f"{source_name}#{key_value}" if key_value else source_name
            ),
            "fields": row,
        })
    return facts, errors

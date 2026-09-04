"""Runtime validation for free-text config values (#234).

``config_schema.py`` is a pure declarative description of the UI fields; the
host renders it but does not validate the values it stores. Every consumer of
a free-text field that names a path or carries structured JSON must therefore
validate at load time. The helpers here are pure (no I/O, no plugin imports)
so they can be shared by the provider, the shared memory service, and the
CLI tools.

Convention: ``*_error`` helpers return a human-readable reason or ``None``;
``parse_*`` / ``safe_*`` helpers are fail-soft and never raise.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# -- Storage names (database_filename / graph_dirname) ----------------------

def storage_name_error(value: str) -> Optional[str]:
    """Why *value* is not an acceptable HERMES_HOME-relative storage name.

    A storage name must stay inside HERMES_HOME: relative, no ``..``
    components, no drive letter, no UNC prefix. Returns ``None`` when valid.
    """
    if not isinstance(value, str) or not value.strip():
        return "is empty"
    value = value.strip()
    if "\x00" in value:
        return "contains a NUL byte"
    if value.startswith("\\\\") or value.startswith("//"):
        return "is a UNC path (must be relative to HERMES_HOME)"
    if len(value) >= 2 and value[1] == ":":
        return "contains a drive letter (must be relative to HERMES_HOME)"
    p = Path(value)
    # Path.is_absolute() needs a drive letter on Windows, so also check for a
    # leading separator explicitly.
    if p.is_absolute() or value.startswith("/") or value.startswith("\\"):
        return "must be relative to HERMES_HOME, not absolute"
    if ".." in p.parts or ".." in value.replace("\\", "/").split("/"):
        return "contains '..' (path traversal not allowed)"
    return None


def safe_storage_name(value: Any, field: str, default: str) -> str:
    """Return *value* if it is a safe storage name, else *default* (logged)."""
    text = str(value) if value is not None else ""
    reason = storage_name_error(text)
    if reason is None:
        return text.strip()
    logger.warning(
        "config %s=%r %s; falling back to %r", field, text, reason, default,
    )
    return default


# -- JSON maps (entity_aliases / expiry_ttl_days) ---------------------------

def parse_string_map(raw: Any, field: str) -> Dict[str, str]:
    """Parse a JSON object with string keys and string values (fail-soft).

    Non-string values are dropped individually; anything that is not a JSON
    object (or not valid JSON) yields ``{}`` with a warning.
    """
    if raw is None:
        return {}
    if isinstance(raw, dict):
        data: Any = raw
    else:
        text = str(raw).strip()
        if not text:
            return {}
        try:
            data = json.loads(text)
        except (json.JSONDecodeError, TypeError) as exc:
            logger.warning("config %s is not valid JSON (%s); ignoring", field, exc)
            return {}
    if not isinstance(data, dict):
        logger.warning(
            "config %s must be a JSON object, got %s; ignoring",
            field, type(data).__name__,
        )
        return {}
    out: Dict[str, str] = {}
    for key, val in data.items():
        k = str(key).strip()
        if not k or not isinstance(val, str) or not val.strip():
            logger.warning("config %s: dropping entry %r (values must be non-empty strings)", field, key)
            continue
        out[k] = val.strip()
    return out


def parse_positive_int_map(
    raw: Any, field: str, default: Dict[str, int], *, maximum: int = 3650,
) -> Dict[str, int]:
    """Parse a JSON object of category -> positive integer days (fail-soft).

    Invalid JSON or a non-object falls back to *default*. Entries whose
    value is not a positive integer within ``1..maximum`` are dropped.
    """
    if raw is None:
        return dict(default)
    if isinstance(raw, dict):
        data: Any = raw
    else:
        text = str(raw).strip()
        if not text:
            return dict(default)
        try:
            data = json.loads(text)
        except (json.JSONDecodeError, TypeError) as exc:
            logger.warning("config %s is not valid JSON (%s); using default", field, exc)
            return dict(default)
    if not isinstance(data, dict):
        logger.warning(
            "config %s must be a JSON object, got %s; using default",
            field, type(data).__name__,
        )
        return dict(default)
    out: Dict[str, int] = {}
    for key, val in data.items():
        k = str(key).strip()
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            logger.warning("config %s: dropping %r (days must be a number)", field, key)
            continue
        days = int(val)
        if not k or days < 1 or days > maximum:
            logger.warning(
                "config %s: dropping %r=%r (days must be 1..%d)", field, key, val, maximum,
            )
            continue
        out[k] = days
    return out


# -- role_words --------------------------------------------------------------

def parse_role_words(raw: Any) -> List[str]:
    """Parse ``role_words`` into a list of non-empty lower-case words.

    Canonical format is a JSON array of strings (that is what the learned
    role-word persistence writes). A plain comma-separated string is still
    accepted for hand-edited configs. Anything else yields ``[]``.
    """
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        items: Any = list(raw)
    else:
        text = str(raw).strip()
        if not text:
            return []
        if text[0] in "[{":
            try:
                items = json.loads(text)
            except (json.JSONDecodeError, TypeError) as exc:
                logger.warning("config role_words is not a valid JSON array (%s); ignoring", exc)
                return []
            if not isinstance(items, list):
                logger.warning("config role_words JSON must be an array; ignoring")
                return []
        else:
            items = text.split(",")
    out: List[str] = []
    for item in items:
        if not isinstance(item, str):
            continue
        word = item.strip().lower()
        if word and word not in out:
            out.append(word)
    return out


# -- Cross-field: deployment_mode vs data_residency -------------------------

_CONSISTENT_RESIDENCY = {"cloud_pilot": "cloud", "local_sku": "local"}


def deployment_consistency_error(deployment_mode: Any, data_residency: Any) -> Optional[str]:
    """Why ``deployment_mode`` and ``data_residency`` disagree, or ``None``.

    ``cloud_pilot`` requires ``cloud`` residency and ``local_sku`` requires
    ``local``; unknown values of either field are reported too.
    """
    mode = str(deployment_mode or "").strip()
    residency = str(data_residency or "").strip()
    if mode not in _CONSISTENT_RESIDENCY:
        return f"deployment_mode={mode!r} is not one of {sorted(_CONSISTENT_RESIDENCY)}"
    if residency not in _CONSISTENT_RESIDENCY.values():
        return f"data_residency={residency!r} is not one of {sorted(set(_CONSISTENT_RESIDENCY.values()))}"
    expected = _CONSISTENT_RESIDENCY[mode]
    if residency != expected:
        return (
            f"deployment_mode={mode!r} requires data_residency={expected!r}, "
            f"got {residency!r}"
        )
    return None

"""
Location resolution for Hermes.

Provides a single ``get_location()`` helper that returns the user's configured
location string (e.g. `` "City-X"``, ``"Riverton, ZA"``) for ambient context
injection. Free-form by design — no geocoding, no coordinates — so the agent
gets a human-readable place name it can reason about ("you're in City-X,
your morning") without any external service dependency.

Resolution order:
  1. ``HERMES_LOCATION`` environment variable
  2. ``location`` key in ``~/.hermes/config.yaml``
  3. Falls back to an empty string (no location hint injected)

Mirrors ``hermes_time.py``'s resolution + caching pattern so the two stay
symmetric. Invalid/empty values are normalized to ``""`` — Hermes never
crashes due to a bad location string.

Set it with::

    hermes config set location  "City-X"
"""

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# Cached state — resolved once, reused on every call.
# Call reset_cache() to force re-resolution (e.g. after config changes).
_cached_location: Optional[str] = None
_cache_resolved: bool = False


def _resolve_location_name() -> str:
    """Read the configured location string (or empty string).

    This does file I/O when falling through to config.yaml, so callers
    should cache the result rather than calling on every ``get_location()``.
    """
    # 1. Environment variable (highest priority — set by Supervisor, gateway, etc.)
    loc_env = os.getenv("HERMES_LOCATION", "").strip()
    if loc_env:
        return loc_env

    # 2. config.yaml ``location`` key
    try:
        # Prefer the shared cached raw-config reader (mtime/size-keyed cache +
        # libyaml C loader) — a direct yaml.safe_load of a large config.yaml
        # costs ~100ms+ and this used to run inside a critical path.
        try:
            from hermes_cli.config import read_raw_config
            cfg = read_raw_config() or {}
        except Exception:
            import yaml
            from hermes_constants import get_config_path
            config_path = get_config_path()
            if config_path.exists():
                with open(config_path, encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
            else:
                cfg = {}
        if cfg:
            # Managed scope: an administrator can pin ``location`` too. Overlay
            # via the shared helper (fail-open) since this reads config.yaml directly.
            try:
                from hermes_cli import managed_scope
                cfg = managed_scope.apply_managed_overlay(cfg)
            except Exception:
                pass
            loc_cfg = cfg.get("location", "")
            if isinstance(loc_cfg, str) and loc_cfg.strip():
                return loc_cfg.strip()
    except Exception:
        pass

    return ""


def get_location() -> str:
    """Return the user's configured location string, or ``""`` when unset.

    Resolved once and cached. Call ``reset_cache()`` after config changes.
    """
    global _cached_location, _cache_resolved
    if not _cache_resolved:
        _cached_location = _resolve_location_name()
        _cache_resolved = True
    return _cached_location or ""


def reset_cache() -> None:
    """Clear the cached location so the next call re-resolves it.

    Call this after the configured location may have changed (e.g. after a
    config edit or ``HERMES_LOCATION`` update) to force ``get_location()`` to
    read the new value instead of the value cached at first use.
    """
    global _cached_location, _cache_resolved
    _cached_location = None
    _cache_resolved = False

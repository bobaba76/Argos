"""
Location resolution for Hermes.

Provides a single ``get_location()`` helper that returns the user's location
string (e.g. ``"Suburb-Y"``, ``"Riverton"``) for ambient context injection.

Resolution order (``location_mode: auto`` — the default):

  1. ``HERMES_LOCATION`` environment variable — explicit override, always
     wins (set by Supervisor, gateway, or the user to force-pin a place).
  2. Network detection, cached for ``location_cache_ttl_s`` (default 7200s
     = 2h — the user doesn't move often enough to warrant a shorter TTL):

     a. **Wi-Fi SSID map** — ``location_map`` in config.yaml maps a Wi-Fi
        SSID to a place name (``{MyHomeNetwork: Suburb-Y}``). Exact match,
        deterministic, offline, zero latency. Skipped when not on Wi-Fi
        (ethernet, disconnected, non-Windows).
     b. **Windows location platform** (Wi-Fi triangulation, ~10-50m
        accuracy — the same mechanism Google Maps uses; see
        ``_os_location``). Requires the ``winsdk`` package and Windows
        location consent. The resulting lat/lon is resolved to a place
        name via the nearest entry in ``known_places`` (offline, exact
        for home/work, default radius 5km) or, failing that, a
        Nominatim/OpenStreetMap reverse-geocode lookup.
     c. **IP geolocation fallback** — free ip-api.com lookup (no key),
        city-level accuracy (~10-20km), normalized through
        ``location_aliases`` in config.yaml (e.g. IP says "City-Z"
        but you want "Riverton" displayed while on that line).
  3. ``location`` key in ``~/.hermes/config.yaml`` — static fallback used
     when detection is unavailable (offline, API down, unknown network).

With ``location_mode: pinned``, detection is skipped entirely and the
``location`` config key (or env var) is used verbatim — the old behavior
for people who never want auto-detection.

Failed detection (no SSID match, API error) is cached for only 60s so a
transient offline blip or a fresh Wi-Fi connection is picked up quickly.

The heavy parts (``netsh`` subprocess, network call) never run more than
once per TTL — the per-turn hint builder re-resolves ``_resolve_location_name()``
fresh each turn (so ``hermes config set location`` still lands next turn)
but hits the TTL cache for the expensive detection.

Set config keys with ``hermes config set <key> <value>`` or edit
config.yaml directly. Weather follows automatically: hermes_weather
geocodes whatever this module resolves.
"""

import json
import logging
import os
import re
import subprocess
import time
import urllib.request
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Cached state — resolved once, reused on every call.
# Call reset_cache() to force re-resolution (e.g. after config changes).
_cached_location: Optional[str] = None
_cache_resolved: bool = False

# Detection cache: (value, resolved_at_monotonic). A successful detection
# lives for location_cache_ttl_s (config, default 7200s = 2h); a failed
# detection (None) lives for _DETECT_FAIL_TTL_S so transient offline states
# retry quickly instead of pinning the fallback for hours.
_detected_cache: Optional[Tuple[Optional[str], float]] = None
_DETECT_FAIL_TTL_S = 60

# IP geolocation timeout (seconds). Free tier: http only, no key, ~45
# req/min per IP — we do at most 1 per 2h.
_IP_GEO_TIMEOUT_S = 3
_IP_GEO_URL = "http://ip-api.com/json/"

_SSID_RE = re.compile(r"^\s*SSID\s*:\s*(.+?)\s*$")

# Windows location platform (winsdk) — Wi-Fi triangulation, the same
# mechanism Google Maps uses. Needs `pip install winsdk` + Windows
# location consent (Settings → Privacy → Location → allow desktop apps).
_OS_LOCATION_TIMEOUT_S = 10
_KNOWN_PLACES_RADIUS_KM = 5.0  # default; override via known_places_radius_km
_NOMINATIM_REVERSE_URL = "https://nominatim.openstreetmap.org/reverse"
_NOMINATIM_TIMEOUT_S = 5
# OSM usage policy requires an identifying User-Agent.
_USER_AGENT = "hermes-agent/0.20.0 (hermes location autodetection)"


def _load_raw_config() -> dict:
    """Read the user config dict (or {} on any failure). Cheap: the shared
    read_raw_config helper is mtime/size-keyed cached."""
    try:
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
            try:
                from hermes_cli import managed_scope
                cfg = managed_scope.apply_managed_overlay(cfg)
            except Exception:
                pass
        return cfg or {}
    except Exception:
        return {}


def _normalize(value) -> str:
    """Strip and normalize a candidate location; '' for falsy/whitespace."""
    if not isinstance(value, str):
        return ""
    return value.strip()


def _as_dict(value) -> dict:
    """Normalize a config value to a dict.

    Accepts a real dict, a JSON-string dict (what ``hermes config set``
    writes for complex values, e.g. ``known_places``), and empty/garbage
    → {}. Never raises.
    """
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return {}
        try:
            parsed = json.loads(s)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
    return {}


def _ssid_from_netsh() -> Optional[str]:
    """Return the current Wi-Fi SSID via netsh (Windows), or None.

    Returns None when not on Wi-Fi (ethernet / disconnected), when netsh is
    unavailable (non-Windows), or on any failure. Never raises."""
    try:
        kwargs = {}
        if hasattr(subprocess, "CREATE_NO_WINDOW"):
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        out = subprocess.run(
            ["netsh", "wlan", "show", "interfaces"],
            capture_output=True, text=True, timeout=5, **kwargs,
        ).stdout
        for line in out.splitlines():
            m = _SSID_RE.match(line)
            if m and m.group(1):
                return m.group(1)
    except Exception:
        pass
    return None


def _city_from_ip() -> Optional[str]:
    """Geolocate via ip-api.com (free, no key) → city name, or None.

    City-level accuracy only. Never raises — any failure (timeout, DNS,
    JSON error, offline) yields None and we fall back to config."""
    try:
        with urllib.request.urlopen(_IP_GEO_URL, timeout=_IP_GEO_TIMEOUT_S) as resp:
            data = json.loads(resp.read().decode("utf-8", "replace"))
        if data.get("status") != "success":
            return None
        city = _normalize(data.get("city"))
        return city or None
    except Exception:
        return None


def _os_location() -> Optional[Tuple[float, float]]:
    """Return (lat, lon) from the Windows location platform, or None.

    Uses ``winsdk`` (Windows.Devices.Geolocation) — the same platform that
    powers Windows Maps and the browser Geolocation API: Wi-Fi
    triangulation from the passively-scanned nearby access points,
    accurate to ~10-50m, even when not connected to any network. No GPS
    chip required.

    Never raises: lazy-imports winsdk (missing package → None), catches
    AccessDenied (location consent off → None) and timeouts. The first
    call can take several seconds (platform warm-up); callers should go
    through the TTL cache, not invoke this per turn.
    """
    try:
        import asyncio
        import winsdk.windows.devices.geolocation as wdg

        async def _get():
            locator = wdg.Geolocator()
            pos = await locator.get_geoposition_async()
            return (pos.coordinate.latitude, pos.coordinate.longitude)

        return asyncio.run(
            asyncio.wait_for(_get(), timeout=_OS_LOCATION_TIMEOUT_S)
        )
    except Exception:
        return None


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km between two coordinates."""
    import math
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _reverse_geocode(lat: float, lon: float, cfg: dict) -> Optional[str]:
    """Resolve (lat, lon) to a place name: nearest ``known_places`` entry
    (offline, deterministic) within ``known_places_radius_km``, else a
    Nominatim reverse-geocode (first display-name element). Never raises."""
    known = _as_dict(cfg.get("known_places"))
    if known:
        radius = _KNOWN_PLACES_RADIUS_KM
        try:
            radius = float(cfg.get("known_places_radius_km") or radius)
        except (TypeError, ValueError):
            pass
        best_name, best_d = None, radius
        for name, coords in known.items():
            try:
                plat, plon = coords
                d = _haversine_km(lat, lon, float(plat), float(plon))
                if d <= best_d:
                    best_name, best_d = str(name).strip() or None, d
            except (TypeError, ValueError):
                continue
        if best_name:
            return best_name
    try:
        import urllib.parse
        q = urllib.parse.urlencode({
            "lat": f"{lat:.6f}", "lon": f"{lon:.6f}",
            "format": "jsonv2", "accept-language": "en",
        })
        req = urllib.request.Request(
            f"{_NOMINATIM_REVERSE_URL}?{q}",
            headers={"User-Agent": _USER_AGENT},
        )
        with urllib.request.urlopen(req, timeout=_NOMINATIM_TIMEOUT_S) as resp:
            data = json.loads(resp.read().decode("utf-8", "replace"))
        # Prefer the structured address object over display_name: the raw
        # first display_name element is often street-level ("16th Road"),
        # not a place name. Suburb-first suits SA naming (Riverton and
        # Suburb-Y are suburbs of City-Z).
        addr = data.get("address")
        if isinstance(addr, dict):
            for key in ("suburb", "city", "town", "village", "municipality", "county"):
                name = _normalize(addr.get(key))
                if name:
                    return name
        display = _normalize(data.get("display_name"))
        if display:
            return display.split(",")[0].strip()
    except Exception:
        pass
    return None


def _detect_once(cfg: dict) -> Optional[str]:
    """One detection pass: SSID map → OS location → IP geolocation."""
    # 1. Wi-Fi SSID → location_map (exact, deterministic, offline)
    ssid = _ssid_from_netsh()
    if ssid:
        loc_map = _as_dict(cfg.get("location_map"))
        if loc_map:
            mapped = _normalize(loc_map.get(ssid))
            if mapped:
                return mapped
    # 2. Windows location platform (Wi-Fi triangulation) → place name.
    # _os_location/_reverse_geocode self-catch, but never trust a third-party
    # boundary: guard the call site too so detection always degrades.
    try:
        coords = _os_location()
    except Exception:
        coords = None
    if coords:
        try:
            place = _reverse_geocode(coords[0], coords[1], cfg)
        except Exception:
            place = None
        if place:
            return place
    # 3. IP geolocation → location_aliases normalization
    city = _city_from_ip()
    if city:
        aliases = _as_dict(cfg.get("location_aliases"))
        if aliases:
            alias = _normalize(aliases.get(city))
            if alias:
                return alias
        return city
    return None


def _detected_network_location(cfg: dict) -> Optional[str]:
    """TTL-cached network detection (success: location_cache_ttl_s; failure:
    60s fast-retry). Returns the detected place name or None."""
    global _detected_cache
    now = time.monotonic()
    ttl_s = 7200
    try:
        ttl_s = int(cfg.get("location_cache_ttl_s", 7200) or 7200)
        if ttl_s < 30:
            ttl_s = 30
    except (TypeError, ValueError):
        ttl_s = 7200
    if _detected_cache is not None:
        value, resolved_at = _detected_cache
        effective_ttl = _DETECT_FAIL_TTL_S if value is None else ttl_s
        if now - resolved_at < effective_ttl:
            return value
    value = _detect_once(cfg)
    _detected_cache = (value, now)
    return value


def _resolve_location_name() -> str:
    """Resolve the location string (or empty string).

    Env override → (pinned mode → config) → network detection → config
    fallback → ''. The expensive detection is TTL-cached inside
    ``_detected_network_location``, so calling this fresh every turn is cheap.
    """
    # 1. Environment variable (highest priority — explicit override)
    loc_env = os.getenv("HERMES_LOCATION", "").strip()
    if loc_env:
        return loc_env

    cfg = _load_raw_config()

    # 2. Pinned mode: manual config wins, no detection
    mode = _normalize(cfg.get("location_mode"))
    if mode == "pinned":
        return _normalize(cfg.get("location"))

    # 3. Network detection (SSID map → IP geolocation + alias)
    detected = _detected_network_location(cfg)
    if detected:
        return detected

    # 4. Static fallback: config.yaml ``location`` key
    return _normalize(cfg.get("location"))


def get_location() -> str:
    """Return the user's location string, or ``""`` when unset.

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
    global _cached_location, _cache_resolved, _detected_cache
    _cached_location = None
    _cache_resolved = False
    _detected_cache = None

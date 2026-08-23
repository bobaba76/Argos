"""
Weather lookup for Hermes.

Provides ``get_weather()`` which returns a short human-readable weather
string (e.g. ``"14°C, light rain"``) for the user's configured location.

Uses Open-Meteo (https://open-meteo.com/) — free, no API key, no signup.
Two endpoints:
  1. Geocoding: https://geocoding-api.open-meteo.com/v1/search?name=Example City
  2. Forecast:  https://api.open-meteo.com/v1/forecast?latitude=...&current=...

Caching strategy (keeps it off the time-to-first-token critical path):
  - Geocode results (city → lat/lon): cached forever in-process. City
    coordinates don't move.
  - Weather results: cached for ``WEATHER_CACHE_TTL_S`` (default 20 min).
    Most turns are cache hits — a dict lookup, zero latency, zero network.
    A cache miss (every ~20 min) is one HTTP call (~200-500ms), the same
    order as the memory prefetch that already runs in the prologue.
  - Network failure / timeout: returns ``""`` (no hint that turn). The
    injection path is unaffected — the agent simply gets no weather line.

Resolution order for the location to look up:
  1. OS coordinates (Windows location platform via
     ``hermes_location.get_location_coords()``) — exact lat/lon, never
     depends on a place-name string
  2. ``HERMES_LOCATION`` environment variable
  3. ``location`` key in ``~/.hermes/config.yaml``
  4. Empty → no weather (returns ``""``)

Set the location with::

    hermes config set location "Example City"

Disable with::

    hermes config set weather.enabled false
"""

import json
import logging
import os
import threading
import time
import urllib.parse
import urllib.request
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# How long to cache weather data before re-fetching. 20 minutes is a good
# balance — weather doesn't change minute-to-minute, and this means most
# turns in a conversation are zero-network cache hits.
WEATHER_CACHE_TTL_S: float = 20 * 60.0

# Timeout for the HTTP calls. Kept short so a slow/dead network doesn't
# block the prologue — we'd rather have no weather than a slow first token.
_HTTP_TIMEOUT_S: float = 4.0

# Geocode cache: {location_string: (lat, lon)} — never invalidated (coords
# don't move). Populated lazily on first lookup.
_geocode_cache: dict[str, Tuple[float, float]] = {}

# Weather cache: {(lat, lon): (timestamp, weather_string)} — invalidated
# after WEATHER_CACHE_TTL_S. Populated lazily.
_weather_cache: dict[Tuple[float, float], Tuple[float, str]] = {}
_cache_lock = threading.Lock()

# OS-coordinate provider (Windows location platform), imported lazily-safe:
# relative import inside the package, absolute import when this module is
# loaded standalone (tests), None when hermes_location is unavailable —
# get_weather() then falls back to name-based geocoding.
try:
    from .hermes_location import get_location_coords
except ImportError:
    try:
        from hermes_location import get_location_coords
    except Exception:
        get_location_coords = None

# WMO weather interpretation codes → short human-readable description.
# https://open-meteo.com/en/docs (scroll to "WMO Weather interpretation codes")
_WMO_CODES: dict[int, str] = {
    0: "clear",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "fog",
    51: "light drizzle",
    53: "drizzle",
    55: "heavy drizzle",
    56: "freezing drizzle",
    57: "freezing drizzle",
    61: "light rain",
    63: "rain",
    65: "heavy rain",
    66: "freezing rain",
    67: "freezing rain",
    71: "light snow",
    73: "snow",
    75: "heavy snow",
    77: "snow grains",
    80: "light showers",
    81: "showers",
    82: "heavy showers",
    85: "light snow showers",
    86: "heavy snow showers",
    95: "thunderstorm",
    96: "thunderstorm with hail",
    99: "thunderstorm with heavy hail",
}


def _resolve_location_name() -> str:
    """Resolve the location string (or empty string).

    Delegates to hermes_location so weather follows the same resolution as
    the Location hint (env override → SSID map → IP geolocation → config
    fallback). Falls back to the local read (env → config) if the module is
    unavailable — this module stays self-contained.
    """
    try:
        from .hermes_location import _resolve_location_name as _resolve
        return _resolve()
    except Exception:
        pass
    loc_env = os.getenv("HERMES_LOCATION", "").strip()
    if loc_env:
        return loc_env
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
            loc_cfg = cfg.get("location", "")
            if isinstance(loc_cfg, str) and loc_cfg.strip():
                return loc_cfg.strip()
    except Exception:
        pass
    return ""


def _is_weather_enabled() -> bool:
    """Check if weather injection is enabled (default: true)."""
    try:
        try:
            from hermes_cli.config import read_raw_config
            cfg = read_raw_config() or {}
        except Exception:
            cfg = {}
        if cfg:
            try:
                from hermes_cli import managed_scope
                cfg = managed_scope.apply_managed_overlay(cfg)
            except Exception:
                pass
            weather_cfg = cfg.get("weather", {})
            if isinstance(weather_cfg, dict):
                enabled = weather_cfg.get("enabled", True)
                if isinstance(enabled, bool):
                    return enabled
    except Exception:
        pass
    return True


def _http_get_json(url: str) -> Optional[dict]:
    """GET a URL and return parsed JSON, or None on any failure."""
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "HermesAgent/1.0"},
        )
        with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT_S) as resp:
            if resp.status != 200:
                return None
            data = resp.read()
            return json.loads(data.decode("utf-8"))
    except Exception:
        return None


def _geocode(location: str) -> Optional[Tuple[float, float]]:
    """Geocode a location name to (lat, lon) via Open-Meteo's geocoding API.

    Does NOT check the cache — the caller (``get_weather``) checks the
    geocode cache before calling this, so the mock-friendly boundary is at
    ``get_weather`` level. This function only does the network lookup.
    """
    url = (
        "https://geocoding-api.open-meteo.com/v1/search"
        f"?name={urllib.parse.quote(location)}&count=1&language=en&format=json"
    )
    data = _http_get_json(url)
    if not data or not isinstance(data, dict):
        return None
    results = data.get("results")
    if not results or not isinstance(results, list) or len(results) == 0:
        return None
    first = results[0]
    lat = first.get("latitude")
    lon = first.get("longitude")
    if not isinstance(lat, (int, float)) or not isinstance(lon, (int, float)):
        return None
    return (float(lat), float(lon))


def _fetch_weather(coords: Tuple[float, float]) -> Optional[str]:
    """Fetch current weather for coords via Open-Meteo, return a short string.

    Does NOT check the cache — the caller (``get_weather``) checks the
    weather cache before calling this, so the mock-friendly boundary is at
    ``get_weather`` level. This function only does the network lookup.

    e.g. ``"14°C, light rain"``
    """
    lat, lon = coords
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&current=temperature_2m,weather_code,wind_speed_10m"
        "&timezone=auto"
    )
    data = _http_get_json(url)
    if not data or not isinstance(data, dict):
        return None
    current = data.get("current")
    if not current or not isinstance(current, dict):
        return None

    temp = current.get("temperature_2m")
    code = current.get("weather_code")
    if not isinstance(temp, (int, float)):
        return None

    # Round temperature to nearest integer for a compact hint.
    temp_str = f"{round(temp)}°C"

    # Map WMO code to description; unknown codes get nothing.
    desc = ""
    if isinstance(code, int):
        desc = _WMO_CODES.get(code, "")

    if desc:
        return f"{temp_str}, {desc}"
    return temp_str


def _weather_for_coords(coords: Tuple[float, float]) -> str:
    """Weather string for exact coordinates, using the shared weather cache.

    Extracted so the OS-coords path and the name-geocode path share one
    cache/fetch boundary. Returns ``""`` on fetch failure.
    """
    now_ts = time.time()
    with _cache_lock:
        cached = _weather_cache.get(coords)
        if cached is not None:
            ts, weather = cached
            if now_ts - ts < WEATHER_CACHE_TTL_S:
                return weather

    weather = _fetch_weather(coords)
    if not weather:
        return ""
    with _cache_lock:
        _weather_cache[coords] = (time.time(), weather)
    return weather


def get_weather() -> str:
    """Return a short weather string for the configured location, or ``""``.

    Returns ``""`` when:
      - Weather is disabled (``weather.enabled: false`` in config.yaml)
      - No location is configured
      - The geocode or weather lookup fails (network, timeout, bad response)

    On success: ``"14°C, light rain"`` or just ``"14°C"`` if the weather code
    is unrecognized.

    Caching is at this level (not inside ``_geocode``/``_fetch_weather``) so
    that the mock boundary for tests is clean — mocks replace the network
    functions, and the cache check/populate happens here, outside the mock.
    """
    if not _is_weather_enabled():
        return ""

    # Prefer exact OS coordinates: weather by lat/lon never depends on a
    # place-name string, so an ungeocodable name (e.g. "City Ward
    # 97") can't kill the weather line.
    coords = get_location_coords() if get_location_coords is not None else None
    if coords:
        return _weather_for_coords(coords)

    location = _resolve_location_name()
    if not location:
        return ""

    # Geocode cache: city coords don't move, cache forever.
    coords = _geocode_cache.get(location)
    if coords is None:
        coords = _geocode(location)
        if coords is None:
            return ""
        _geocode_cache[location] = coords

    return _weather_for_coords(coords)


def reset_cache() -> None:
    """Clear all caches (geocode + weather). Call after config changes."""
    with _cache_lock:
        _geocode_cache.clear()
        _weather_cache.clear()

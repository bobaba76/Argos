"""
Tests for weather support (hermes_weather module + hint builder).

Moved from ambient_context/tests/test_weather.py when the ambient modules
were relocated into the plugin package.  The hint builders now live in
hybrid_memory_plugin.__init__ (not agent.turn_context) and ride the native
pre_llm_call plugin hook instead of a core source patch.

Covers:
  - get_weather() returns "" when no location is configured
  - get_weather() returns "" when disabled via config
  - get_weather() returns a weather string when location + network are available
  - Geocode results are cached forever
  - Weather results are cached with TTL
  - _build_weather_hint() renders the "Weather: ..." line and never crashes
  - Network failures return "" (no hint, no crash)

Network calls are mocked — no real HTTP requests in tests.
"""

import os
import time
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import sys

# Ensure the plugin directory is importable for sibling module imports
# and the package parent for package imports.
_plugin_dir = Path(__file__).resolve().parent.parent
if str(_plugin_dir) not in sys.path:
    sys.path.insert(0, str(_plugin_dir))
if str(_plugin_dir.parent) not in sys.path:
    sys.path.insert(0, str(_plugin_dir.parent))

import hermes_weather

# The hint builders live in the package __init__. In the bundle repo the
# package is hybrid_memory_plugin; when installed to HERMES_HOME it's
# hybrid_memory. Use the same try/except fallback as test_hybrid_memory.py.
try:
    from hybrid_memory_plugin import _build_weather_hint
    _pkg_name = "hybrid_memory_plugin"
except ImportError:
    from hybrid_memory import _build_weather_hint
    _pkg_name = "hybrid_memory"

# Package-relative module paths for patch targets — __init__.py imports
# via `from .hermes_weather import ...`, so we must patch the package
# module, not the standalone one.
_wx_module_path = _pkg_name + ".hermes_weather"


def _reset_weather_cache():
    """Reset all weather caches."""
    hermes_weather.reset_cache()
    os.environ.pop("HERMES_LOCATION", None)


# =========================================================================
# hermes_weather.get_weather() — core helper
# =========================================================================

class TestGetWeather:
    """Test the weather resolution helper."""

    @pytest.fixture(autouse=True)
    def _no_os_coords(self):
        """Default: no OS coordinates — legacy name-path tests exercise the
        geocode path (and never hit the real Windows location platform).
        Coords-path tests override with their own patch."""
        with patch("hermes_weather.get_location_coords", return_value=None):
            yield

    def setup_method(self):
        _reset_weather_cache()

    def teardown_method(self):
        _reset_weather_cache()

    def test_uses_os_coords_directly(self):
        """When OS coordinates are available, weather is fetched by lat/lon
        and the name path (resolve/geocode) is never consulted — so an
        ungeocodable name like 'BigCity Ward 97' can't kill weather."""
        with patch("hermes_weather.get_location_coords",
                   return_value=(40.0500, -74.4800)), \
             patch("hermes_weather._fetch_weather",
                   return_value="15°C, clear"), \
             patch("hermes_weather._resolve_location_name") as rln, \
             patch("hermes_weather._geocode") as geo:
            assert hermes_weather.get_weather() == "15°C, clear"
        rln.assert_not_called()
        geo.assert_not_called()

    def test_falls_back_to_name_when_no_coords(self):
        """Without OS coordinates (winsdk missing, consent off), the name
        geocode path still works."""
        os.environ["HERMES_LOCATION"] = "Northtown"
        with patch("hermes_weather.get_location_coords", return_value=None), \
             patch("hermes_weather._geocode",
                   return_value=(40.0000, -74.5000)), \
             patch("hermes_weather._fetch_weather",
                   return_value="14°C, light rain"):
            assert hermes_weather.get_weather() == "14°C, light rain"

    def test_coords_path_is_cached(self):
        """Weather fetched by OS coords is cached by (lat, lon) like the
        name path — repeated calls don't re-fetch."""
        fetch_count = {"n": 0}

        def _mock_fetch(coords):
            fetch_count["n"] += 1
            return "15°C, clear"

        with patch("hermes_weather.get_location_coords",
                   return_value=(40.0500, -74.4800)), \
             patch("hermes_weather._fetch_weather", side_effect=_mock_fetch):
            hermes_weather.get_weather()
            hermes_weather.get_weather()
        assert fetch_count["n"] == 1

    def test_empty_when_no_location(self):
        """With no location configured, get_weather() returns ""."""
        with patch("hermes_weather._resolve_location_name", return_value=""):
            assert hermes_weather.get_weather() == ""

    def test_empty_when_disabled(self):
        """When weather.enabled is false in config, get_weather() returns ""
        even if a location is set."""
        os.environ["HERMES_LOCATION"] = "Example City"
        with patch("hermes_weather._is_weather_enabled",
                   return_value=False):
            assert hermes_weather.get_weather() == ""

    def test_returns_weather_string_when_location_set(self):
        """With a location set and a successful network call, get_weather()
        returns a short string like '14°C, light rain'."""
        os.environ["HERMES_LOCATION"] = "Example City"
        # Mock the geocode + weather fetch to avoid real HTTP.
        with patch("hermes_weather._geocode",
                   return_value=(40.2000, -74.1000)), \
             patch("hermes_weather._fetch_weather",
                   return_value="14°C, light rain"):
            result = hermes_weather.get_weather()
        assert result == "14°C, light rain"

    def test_empty_when_geocode_fails(self):
        """If geocoding fails (unknown place, network down), return ""."""
        os.environ["HERMES_LOCATION"] = "NonexistentPlace12345"
        with patch("hermes_weather._geocode",
                   return_value=None):
            assert hermes_weather.get_weather() == ""

    def test_empty_when_weather_fetch_fails(self):
        """If the weather API call fails, return ""."""
        os.environ["HERMES_LOCATION"] = "Example City"
        with patch("hermes_weather._geocode",
                   return_value=(40.2000, -74.1000)), \
             patch("hermes_weather._fetch_weather",
                   return_value=None):
            assert hermes_weather.get_weather() == ""


class TestWeatherCaching:
    """Test the caching behavior."""

    @pytest.fixture(autouse=True)
    def _no_os_coords(self):
        with patch("hermes_weather.get_location_coords", return_value=None):
            yield

    def setup_method(self):
        _reset_weather_cache()

    def teardown_method(self):
        _reset_weather_cache()

    def test_geocode_cached_forever(self):
        """Geocode results are cached in-process — the second call doesn't
        hit the network."""
        os.environ["HERMES_LOCATION"] = "Example City"
        call_count = {"n": 0}

        def _mock_geocode(location):
            call_count["n"] += 1
            return (40.2000, -74.1000)

        with patch("hermes_weather._geocode",
                   side_effect=_mock_geocode), \
             patch("hermes_weather._fetch_weather",
                   return_value="14°C, clear"):
            hermes_weather.get_weather()
            hermes_weather.get_weather()
        # Geocode was only called once (cached on second call).
        assert call_count["n"] == 1

    def test_weather_cached_within_ttl(self):
        """Weather results are cached — within the TTL, no re-fetch."""
        os.environ["HERMES_LOCATION"] = "Example City"
        fetch_count = {"n": 0}

        def _mock_fetch(coords):
            fetch_count["n"] += 1
            return "14°C, clear"

        with patch("hermes_weather._geocode",
                   return_value=(40.2000, -74.1000)), \
             patch("hermes_weather._fetch_weather",
                   side_effect=_mock_fetch):
            hermes_weather.get_weather()
            hermes_weather.get_weather()
        # Weather was only fetched once (cached on second call within TTL).
        assert fetch_count["n"] == 1

    def test_weather_refetched_after_ttl(self):
        """After the TTL expires, weather is re-fetched."""
        os.environ["HERMES_LOCATION"] = "Example City"
        fetch_count = {"n": 0}

        def _mock_fetch(coords):
            fetch_count["n"] += 1
            return "14°C, clear"

        with patch("hermes_weather._geocode",
                   return_value=(40.2000, -74.1000)), \
             patch("hermes_weather._fetch_weather",
                   side_effect=_mock_fetch), \
             patch("hermes_weather.WEATHER_CACHE_TTL_S", 0.01):
            hermes_weather.get_weather()
            time.sleep(0.02)
            hermes_weather.get_weather()
        # Weather was fetched twice (TTL expired between calls).
        assert fetch_count["n"] == 2


# =========================================================================
# _build_weather_hint — the per-turn injection line (now in the plugin)
# =========================================================================

class TestBuildWeatherHint:
    """The per-turn ``Weather: ...`` line built by the plugin's pre_llm_call
    hook callback."""

    def setup_method(self):
        _reset_weather_cache()

    def teardown_method(self):
        _reset_weather_cache()

    def test_renders_when_weather_available(self):
        with patch(_wx_module_path + ".get_weather",
                   return_value="14°C, light rain"):
            assert _build_weather_hint() == "Weather: 14°C, light rain"

    def test_empty_when_no_weather(self):
        with patch(_wx_module_path + ".get_weather",
                   return_value=""):
            assert _build_weather_hint() == ""

    def test_never_crashes_on_exception(self):
        """If hermes_weather can't be imported for any reason, the hint is
        empty — the injection path is unaffected."""
        with patch(_wx_module_path + ".get_weather",
                   side_effect=RuntimeError("boom")):
            assert _build_weather_hint() == ""

    def test_temp_only_when_unknown_code(self):
        """When the WMO code is unrecognized, the hint is just the
        temperature (e.g. 'Weather: 14°C')."""
        with patch(_wx_module_path + ".get_weather",
                   return_value="14°C"):
            assert _build_weather_hint() == "Weather: 14°C"


# =========================================================================
# WMO weather code mapping
# =========================================================================

class TestWmoCodeMapping:
    """Verify the WMO weather code -> description mapping covers the common
    codes and produces sensible strings."""

    def test_clear_sky(self):
        assert hermes_weather._WMO_CODES[0] == "clear"

    def test_rain_codes(self):
        assert hermes_weather._WMO_CODES[61] == "light rain"
        assert hermes_weather._WMO_CODES[63] == "rain"
        assert hermes_weather._WMO_CODES[65] == "heavy rain"

    def test_thunderstorm(self):
        assert hermes_weather._WMO_CODES[95] == "thunderstorm"

    def test_snow(self):
        assert hermes_weather._WMO_CODES[71] == "light snow"
        assert hermes_weather._WMO_CODES[75] == "heavy snow"

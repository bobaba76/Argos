"""
Tests for location support (hermes_location module + hint builder).

Moved from ambient_context/tests/test_location.py when the ambient modules
were relocated into the plugin package.  The hint builders now live in
hybrid_memory_plugin.__init__ (not agent.turn_context) and ride the native
pre_llm_call plugin hook instead of a core source patch.

Covers:
  - HERMES_LOCATION env var wins over everything
  - config.yaml ``location`` key is the static fallback
  - location_mode: pinned skips detection
  - Network detection: SSID map → IP geolocation → alias normalization
  - Detection TTL cache (2h success / 60s failure fast-retry)
  - Empty/unset location returns "" (no hint injected)
  - Whitespace-only values are normalized to ""
  - reset_cache() forces re-resolution after config changes
  - _build_location_hint() renders the "Location: ..." line and never crashes
"""

import os
import time
import pytest
from unittest.mock import patch
from pathlib import Path
import sys

# Ensure the plugin directory is importable for sibling module imports
# and the package parent for package imports.
_plugin_dir = Path(__file__).resolve().parent.parent
if str(_plugin_dir) not in sys.path:
    sys.path.insert(0, str(_plugin_dir))
if str(_plugin_dir.parent) not in sys.path:
    sys.path.insert(0, str(_plugin_dir.parent))

import hermes_location

# The hint builders live in the package __init__. In the bundle repo the
# package is hybrid_memory_plugin; when installed to HERMES_HOME it's
# hybrid_memory. Use the same try/except fallback as test_hybrid_memory.py.
try:
    from hybrid_memory_plugin import _build_location_hint
    _pkg_name = "hybrid_memory_plugin"
except ImportError:
    from hybrid_memory import _build_location_hint
    _pkg_name = "hybrid_memory"

# The package-relative module that __init__.py imports from — patch targets
# must use this path, not the standalone module name.
_loc_module_path = _pkg_name + ".hermes_location"


def _reset_location_cache():
    """Reset the hermes_location module + detection caches."""
    hermes_location._cached_location = None
    hermes_location._cache_resolved = False
    hermes_location._detected_cache = None


# =========================================================================
# hermes_location.get_location() — core helper
# =========================================================================

class TestHermesLocationGet:
    """Test the location resolution helper."""

    def setup_method(self):
        _reset_location_cache()

    def teardown_method(self):
        _reset_location_cache()
        os.environ.pop("HERMES_LOCATION", None)

    def test_env_var_wins(self):
        """With HERMES_LOCATION set, get_location() returns it without
        reading config.yaml."""
        os.environ["HERMES_LOCATION"] = "Example City"
        assert hermes_location.get_location() == "Example City"

    def test_env_var_strips_whitespace(self):
        os.environ["HERMES_LOCATION"] = "  Another City  "
        assert hermes_location.get_location() == "Another City"

    def test_empty_env_falls_through(self):
        """An empty HERMES_LOCATION env var is treated as unset — get_location
        falls through to detection/config (or empty if all are unset)."""
        os.environ["HERMES_LOCATION"] = ""
        # Mock the resolver to return empty so the test is environment-independent.
        with patch("hermes_location._resolve_location_name", return_value=""):
            assert hermes_location.get_location() == ""

    def test_unset_returns_empty(self):
        """With no env var and no location available, get_location() returns ""."""
        with patch("hermes_location._resolve_location_name", return_value=""):
            assert hermes_location.get_location() == ""

    def test_free_form_string_preserved(self):
        """Location is free-form — commas, spaces, region codes all survive
        verbatim. No geocoding, no normalization beyond strip()."""
        os.environ["HERMES_LOCATION"] = "Another City, US"
        assert hermes_location.get_location() == "Another City, US"

    def test_whitespace_only_normalizes_to_empty(self):
        os.environ["HERMES_LOCATION"] = "   "
        with patch("hermes_location._resolve_location_name", return_value=""):
            assert hermes_location.get_location() == ""


class TestHermesLocationCache:
    """Test the caching behavior (mirrors hermes_time's cache tests)."""

    def setup_method(self):
        _reset_location_cache()

    def teardown_method(self):
        _reset_location_cache()
        os.environ.pop("HERMES_LOCATION", None)

    def test_cached_value_reused(self):
        """get_location() resolves once and caches — changing the env var
        after the first call does not affect subsequent calls until
        reset_cache()."""
        os.environ["HERMES_LOCATION"] = "Example City"
        assert hermes_location.get_location() == "Example City"
        os.environ["HERMES_LOCATION"] = "Another City"
        # Still cached — returns the old value.
        assert hermes_location.get_location() == "Example City"

    def test_reset_cache_forces_reresolution(self):
        os.environ["HERMES_LOCATION"] = "Example City"
        assert hermes_location.get_location() == "Example City"
        os.environ["HERMES_LOCATION"] = "Another City"
        hermes_location.reset_cache()
        assert hermes_location.get_location() == "Another City"


# =========================================================================
# Hybrid resolution: env → pinned → SSID map → IP geolocation → config
# =========================================================================

class TestHybridResolution:
    """The new detection chain. All heavy/mockable boundaries (netsh, IP
    lookup, config read) are patched so tests never touch the network."""

    def setup_method(self):
        _reset_location_cache()

    def teardown_method(self):
        _reset_location_cache()
        os.environ.pop("HERMES_LOCATION", None)

    def test_env_var_wins_over_detection(self):
        os.environ["HERMES_LOCATION"] = "Portside"
        with patch("hermes_location._load_raw_config",
                   return_value={"location_map": {"TestWiFi": "Northtown"}}), \
             patch("hermes_location._detected_network_location") as detect:
            assert hermes_location._resolve_location_name() == "Portside"
            detect.assert_not_called()

    def test_pinned_mode_skips_detection(self):
        with patch("hermes_location._load_raw_config",
                   return_value={"location": "Capital City", "location_mode": "pinned"}), \
             patch("hermes_location._detected_network_location") as detect:
            assert hermes_location._resolve_location_name() == "Capital City"
            detect.assert_not_called()

    def test_ssid_map_exact_match_wins(self):
        cfg = {"location_map": {"TestWiFi": "Northtown"}}
        with patch("hermes_location._load_raw_config", return_value=cfg), \
             patch("hermes_location._ssid_from_netsh", return_value="TestWiFi"), \
             patch("hermes_location._city_from_ip") as ip:
            assert hermes_location._resolve_location_name() == "Northtown"
            ip.assert_not_called()

    def test_unmapped_ssid_falls_through_to_ip(self):
        cfg = {"location_map": {"TestWiFi": "Northtown"}}
        with patch("hermes_location._load_raw_config", return_value=cfg), \
             patch("hermes_location._ssid_from_netsh", return_value="WorkWiFi"), \
             patch("hermes_location._os_location", return_value=None), \
             patch("hermes_location._city_from_ip", return_value="BigCity"):
            assert hermes_location._resolve_location_name() == "BigCity"

    def test_no_ssid_uses_ip_geolocation(self):
        with patch("hermes_location._load_raw_config", return_value={}), \
             patch("hermes_location._ssid_from_netsh", return_value=None), \
             patch("hermes_location._os_location", return_value=None), \
             patch("hermes_location._city_from_ip", return_value="Uptown"):
            assert hermes_location._resolve_location_name() == "Uptown"

    def test_os_location_precedes_ip(self):
        """Windows location platform (Wi-Fi triangulation) beats IP — the
        Google-Maps-grade tier. IP must not be consulted when OS location
        resolves."""
        with patch("hermes_location._load_raw_config", return_value={}), \
             patch("hermes_location._ssid_from_netsh", return_value=None), \
             patch("hermes_location._os_location",
                   return_value=(40.3100, -74.1900)), \
             patch("hermes_location._reverse_geocode", return_value="Southtown"), \
             patch("hermes_location._city_from_ip") as ip:
            assert hermes_location._resolve_location_name() == "Southtown"
            ip.assert_not_called()

    def test_os_location_unavailable_falls_to_ip(self):
        with patch("hermes_location._load_raw_config", return_value={}), \
             patch("hermes_location._ssid_from_netsh", return_value=None), \
             patch("hermes_location._os_location", return_value=None), \
             patch("hermes_location._city_from_ip", return_value="BigCity"):
            assert hermes_location._resolve_location_name() == "BigCity"

    def test_os_location_returns_none_on_error(self):
        """winsdk missing / AccessDenied / timeout → detection must still
        fall through to IP, never raise."""
        with patch("hermes_location._load_raw_config", return_value={}), \
             patch("hermes_location._ssid_from_netsh", return_value=None), \
             patch("hermes_location._os_location", side_effect=RuntimeError("boom")), \
             patch("hermes_location._city_from_ip", return_value="BigCity"):
            assert hermes_location._resolve_location_name() == "BigCity"

    def test_ip_city_alias_applied(self):
        cfg = {"location_aliases": {"BigCity": "Southtown"}}
        with patch("hermes_location._load_raw_config", return_value=cfg), \
             patch("hermes_location._ssid_from_netsh", return_value=None), \
             patch("hermes_location._os_location", return_value=None), \
             patch("hermes_location._city_from_ip", return_value="BigCity"):
            assert hermes_location._resolve_location_name() == "Southtown"

    def test_detection_unavailable_uses_config_fallback(self):
        cfg = {"location": "Northtown"}
        with patch("hermes_location._load_raw_config", return_value=cfg), \
             patch("hermes_location._ssid_from_netsh", return_value=None), \
             patch("hermes_location._os_location", return_value=None), \
             patch("hermes_location._city_from_ip", return_value=None):
            assert hermes_location._resolve_location_name() == "Northtown"

    def test_everything_unavailable_returns_empty(self):
        with patch("hermes_location._load_raw_config", return_value={}), \
             patch("hermes_location._ssid_from_netsh", return_value=None), \
             patch("hermes_location._os_location", return_value=None), \
             patch("hermes_location._city_from_ip", return_value=None):
            assert hermes_location._resolve_location_name() == ""


class TestReverseGeocode:
    """(lat, lon) → place name: nearest known_places entry within radius,
    else Nominatim address-parsing. Never raises, no network in tests."""

    def setup_method(self):
        _reset_location_cache()

    def teardown_method(self):
        _reset_location_cache()
        os.environ.pop("HERMES_LOCATION", None)

    def test_known_places_nearest_match(self):
        cfg = {"known_places": {"Northtown": [40.0000, -74.5000],
                                "Southtown": [40.3000, -74.2000]}}
        # Southtown centre-ish: ~2.4km from the Southtown entry, ~30km from Northtown.
        assert hermes_location._reverse_geocode(40.3100, -74.1900, cfg) == "Southtown"

    def test_known_places_as_json_string(self):
        """`hermes config set known_places '{"a": [...]}'` writes a JSON
        string — must still work."""
        cfg = {"known_places": '{"Northtown": [40.0000, -74.5000], "Southtown": [40.3000, -74.2000]}'}
        assert hermes_location._reverse_geocode(40.3100, -74.1900, cfg) == "Southtown"

    def test_known_places_outside_radius_uses_nominatim(self):
        cfg = {"known_places": {"Bayport": [35.0000, -80.0000]}}
        import json as _json
        payload = _json.dumps({"address": {"suburb": "Hillside"}})
        with patch("urllib.request.urlopen") as urlopen:
            urlopen.return_value.__enter__.return_value.read.return_value = payload.encode()
            assert hermes_location._reverse_geocode(40.3100, -74.1900, cfg) == "Hillside"

    def test_nominatim_address_suburb_beats_city(self):
        """Suburb-first parsing, not city-first."""
        import json as _json
        payload = _json.dumps({"address": {"suburb": "Southtown", "city": "BigCity"}})
        with patch("urllib.request.urlopen") as urlopen:
            urlopen.return_value.__enter__.return_value.read.return_value = payload.encode()
            assert hermes_location._reverse_geocode(40.3100, -74.1900, {}) == "Southtown"

    def test_ward_label_suburb_skipped_city_wins(self):
        """Ward/sector administrative labels (e.g. 'BigCity Ward 97')
        are not real place names and Open-Meteo can't geocode them — the
        city key must win instead."""
        import json as _json
        payload = _json.dumps({"address": {
            "suburb": "BigCity Ward 97",
            "city": "Northtown",
            "municipality": "City of BigCity Metropolitan Municipality",
        }})
        with patch("urllib.request.urlopen") as urlopen:
            urlopen.return_value.__enter__.return_value.read.return_value = payload.encode()
            assert hermes_location._reverse_geocode(40.050, -74.480, {}) == "Northtown"

    def test_default_radius_covers_big_suburbs(self):
        """Default known_places radius is 10km — a point ~7km from the
        Northtown centre matches offline (no Nominatim call)."""
        cfg = {"known_places": {"Northtown": [40.0000, -74.5000]}}
        with patch("urllib.request.urlopen") as urlopen:
            result = hermes_location._reverse_geocode(40.0500, -74.4800, cfg)
            urlopen.assert_not_called()
        assert result == "Northtown"

    def test_get_location_coords_exposes_os_position(self):
        with patch("hermes_location._os_location",
                   return_value=(40.0500, -74.4800)):
            assert hermes_location.get_location_coords() == (40.0500, -74.4800)

    def test_get_location_coords_never_raises(self):
        with patch("hermes_location._os_location",
                   side_effect=RuntimeError("platform unavailable")):
            assert hermes_location.get_location_coords() is None

    def test_nominatim_failure_returns_none(self):
        with patch("urllib.request.urlopen", side_effect=OSError("offline")):
            assert hermes_location._reverse_geocode(40.3100, -74.1900, {}) is None

    def test_as_dict_handles_garbage(self):
        assert hermes_location._as_dict(None) == {}
        assert hermes_location._as_dict(42) == {}
        assert hermes_location._as_dict("{not json") == {}
        assert hermes_location._as_dict('{"a": [1, 2]}') == {"a": [1, 2]}
        assert hermes_location._as_dict({"a": 1}) == {"a": 1}


class TestDetectionTTL:
    """The detection cache: success cached for location_cache_ttl_s (default
    2h), failures fast-retried after 60s."""

    def setup_method(self):
        _reset_location_cache()

    def teardown_method(self):
        _reset_location_cache()
        os.environ.pop("HERMES_LOCATION", None)

    def test_success_cached_until_ttl(self):
        calls = {"n": 0}

        def fake_detect(cfg):
            calls["n"] += 1
            return "Southtown"

        with patch("hermes_location._load_raw_config", return_value={}), \
             patch("hermes_location._detect_once", side_effect=fake_detect):
            assert hermes_location._resolve_location_name() == "Southtown"
            assert hermes_location._resolve_location_name() == "Southtown"
        assert calls["n"] == 1
        # Age the cache past the TTL → re-detect.
        hermes_location._detected_cache = ("Southtown", time.monotonic() - 100000)
        with patch("hermes_location._load_raw_config", return_value={}), \
             patch("hermes_location._detect_once", side_effect=fake_detect):
            assert hermes_location._resolve_location_name() == "Southtown"
        assert calls["n"] == 2

    def test_custom_ttl_from_config(self):
        calls = {"n": 0}

        def fake_detect(cfg):
            calls["n"] += 1
            return "Southtown"

        cfg = {"location_cache_ttl_s": 30}
        # Fresh cache entry → detected once.
        with patch("hermes_location._load_raw_config", return_value=cfg), \
             patch("hermes_location._detect_once", side_effect=fake_detect):
            assert hermes_location._resolve_location_name() == "Southtown"
        assert calls["n"] == 1

    def test_failure_retried_after_60s(self):
        calls = {"n": 0}

        def fake_detect(cfg):
            calls["n"] += 1
            return None

        with patch("hermes_location._load_raw_config", return_value={}), \
             patch("hermes_location._detect_once", side_effect=fake_detect):
            assert hermes_location._resolve_location_name() == ""
            assert hermes_location._resolve_location_name() == ""
        # Failure cached for 60s — two calls, one detection.
        assert calls["n"] == 1
        # Age past the failure TTL → re-detect.
        hermes_location._detected_cache = (None, time.monotonic() - 61)
        with patch("hermes_location._load_raw_config", return_value={}), \
             patch("hermes_location._detect_once", side_effect=fake_detect):
            assert hermes_location._resolve_location_name() == ""
        assert calls["n"] == 2

    def test_reset_cache_clears_detection(self):
        hermes_location._detected_cache = ("Southtown", time.monotonic())
        hermes_location.reset_cache()
        assert hermes_location._detected_cache is None


# =========================================================================
# _build_location_hint — the per-turn injection line (now in the plugin)
# =========================================================================

class TestBuildLocationHint:
    """The per-turn ``Location: ...`` line built by the plugin's pre_llm_call
    hook callback. Must use the configured location and never crash."""

    def setup_method(self):
        _reset_location_cache()

    def teardown_method(self):
        _reset_location_cache()
        os.environ.pop("HERMES_LOCATION", None)

    def test_uses_configured_location(self):
        os.environ["HERMES_LOCATION"] = "Example City"
        assert _build_location_hint() == "Location: Example City"

    def test_empty_when_unset(self):
        with patch(_loc_module_path + "._resolve_location_name", return_value=""):
            assert _build_location_hint() == ""

    def test_empty_when_whitespace_only(self):
        os.environ["HERMES_LOCATION"] = "   "
        with patch(_loc_module_path + "._resolve_location_name", return_value=""):
            assert _build_location_hint() == ""

    def test_never_crashes_on_missing_module(self):
        """If hermes_location can't be imported for any reason, the hint is
        empty — the injection path is unaffected."""
        with patch(_loc_module_path + "._resolve_location_name",
                   side_effect=RuntimeError("boom")):
            assert _build_location_hint() == ""

    def test_free_form_with_comma(self):
        os.environ["HERMES_LOCATION"] = "Another City, US"
        assert _build_location_hint() == "Location: Another City, US"

    def test_picks_up_mid_session_change(self):
        """The hint re-resolves each turn (bypassing the module cache) so a
        ``hermes config set location`` mid-session takes effect on the next
        turn — no /reset needed."""
        os.environ["HERMES_LOCATION"] = "Example City"
        assert _build_location_hint() == "Location: Example City"
        os.environ["HERMES_LOCATION"] = "Another City"
        assert _build_location_hint() == "Location: Another City"

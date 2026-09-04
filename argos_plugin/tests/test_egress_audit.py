"""Tests for the egress audit fixes (E1-E8, issue #219).

Covers:
- E1: cross-platform home path (no Windows-only AppData hardcode)
- E2: plugin directory is ``plugins/hybrid_memory`` not ``plugins/argos``
- E3: gate enforces per-site config flags (not just local_only + PII)
- E4: PII patterns catch spaced/dashed phones, obfuscated emails, SSN, etc.
- E5: role_word site uses ``None`` gate (always-on), not a descriptive string
- E6: load_config caches with mtime invalidation
- E7: known-kinds set is a module-level constant
- E8: contains_sensitive reports all matches (all_sensitive_labels)

Run with (Hermes venv python, offline):
    python -m pytest tests/test_egress_audit.py -v
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

_plugin_dir = Path(__file__).resolve().parent.parent
for _path in (_plugin_dir.parent, _plugin_dir):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import egress  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_cache():
    """Clear the egress config cache between tests (E6)."""
    egress._reset_config_cache()
    yield
    egress._reset_config_cache()


# ---------------------------------------------------------------------------
# E1 — cross-platform home path
# ---------------------------------------------------------------------------

class TestE1CrossPlatformHome:
    def test_hermes_home_env_var(self, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", "/tmp/hermes-test")
        # Bypass hermes_constants if it's importable so the env path is used.
        monkeypatch.setitem(sys.modules, "hermes_constants", None)
        home = egress._hermes_home()
        assert str(home) == str(Path("/tmp/hermes-test"))

    def test_hermes_home_fallback_not_windows_only(self, monkeypatch):
        """When hermes_constants is unavailable and HERMES_HOME is unset,
        the fallback is ``~/.hermes`` (cross-platform), NOT the Windows-only
        ``~/AppData/Local/hermes``."""
        monkeypatch.delenv("HERMES_HOME", raising=False)
        monkeypatch.setitem(sys.modules, "hermes_constants", None)
        home = egress._hermes_home()
        # Must NOT contain the Windows AppData path components.
        assert "AppData" not in str(home)
        assert str(home).endswith(".hermes")


# ---------------------------------------------------------------------------
# E2 — plugin directory name
# ---------------------------------------------------------------------------

class TestE2PluginDir:
    def test_load_config_checks_hybrid_memory_dir(self, monkeypatch, tmp_path):
        """load_config looks in ``plugins/hybrid_memory/``, not
        ``plugins/argos/``."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setitem(sys.modules, "hermes_constants", None)
        # Put a config in the correct directory.
        correct = tmp_path / "plugins" / "hybrid_memory" / "hybrid_memory.json"
        correct.parent.mkdir(parents=True)
        correct.write_text('{"local_only": "true"}', encoding="utf-8")
        egress._reset_config_cache()
        cfg = egress.load_config()
        assert cfg.get("local_only") == "true"

    def test_old_argos_dir_is_dead_code(self, monkeypatch, tmp_path):
        """A config in ``plugins/argos/`` is NOT read (wrong directory)."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setitem(sys.modules, "hermes_constants", None)
        wrong = tmp_path / "plugins" / "argos" / "hybrid_memory.json"
        wrong.parent.mkdir(parents=True)
        wrong.write_text('{"local_only": "true"}', encoding="utf-8")
        egress._reset_config_cache()
        cfg = egress.load_config()
        assert cfg.get("local_only") is None


# ---------------------------------------------------------------------------
# E3 — gate enforces per-site flags
# ---------------------------------------------------------------------------

class TestE3PerSiteFlagEnforcement:
    def test_gate_blocks_when_per_site_flag_off(self):
        """A site whose per-site flag is OFF is refused by gate, even with
        plain text and local_only off."""
        cfg = {"distillation_enabled": "false"}
        assert egress.gate("distillation", "plain text", cfg) is False

    def test_gate_allows_when_per_site_flag_on(self):
        cfg = {"distillation_enabled": "true"}
        assert egress.gate("distillation", "plain text", cfg) is True

    def test_gate_blocks_query_expansion_when_disabled(self):
        cfg = {"query_expansion_enabled": "false"}
        assert egress.gate("query_expansion", "plain text", cfg) is False

    def test_gate_allows_query_expansion_when_enabled(self):
        cfg = {"query_expansion_enabled": "true"}
        assert egress.gate("query_expansion", "plain text", cfg) is True

    def test_gate_consistent_with_site_live(self):
        """gate and site_live now agree: when site_live says OFF, gate
        refuses (the old inconsistency where the report showed OFF but the
        gate allowed the call is fixed)."""
        cfg = {"distillation_enabled": "false"}
        dist_site = next(s for s in egress.SITES if s["kind"] == "distillation")
        assert egress.site_live(dist_site, cfg) == "OFF"
        assert egress.gate("distillation", "plain text", cfg) is False


# ---------------------------------------------------------------------------
# E4 — PII pattern gaps
# ---------------------------------------------------------------------------

class TestE4PIIPatterns:
    def test_spaced_sa_phone_detected(self):
        assert egress.contains_sensitive("+27 82 123 4567") is not None

    def test_dashed_sa_phone_detected(self):
        assert egress.contains_sensitive("082-123-4567") is not None

    def test_plain_sa_phone_still_detected(self):
        assert egress.contains_sensitive("0831234567") == "South African phone number"

    def test_obfuscated_email_detected(self):
        assert egress.contains_sensitive("contact me at user at example dot com") is not None

    def test_us_ssn_detected(self):
        assert egress.contains_sensitive("my SSN is 123-45-6789") is not None

    def test_international_phone_detected(self):
        assert egress.contains_sensitive("call +1 555 123 4567") is not None

    def test_unix_ms_timestamp_not_flagged_as_id(self):
        """A 13-digit Unix millisecond timestamp (1700000000000) is NOT a
        SA ID number — the date-shaped pattern rejects it."""
        # 1700000000000 → 17 00 00 0000000. Month "00" is not 01-12, so the
        # date-validated ID pattern does not match.
        result = egress.contains_sensitive("timestamp 1700000000000 here")
        # It should not be labelled as a 13-digit ID number.
        assert result != "13-digit ID number"

    def test_sa_id_with_valid_date_still_detected(self):
        assert egress.contains_sensitive("id 8601015012084") == "13-digit ID number"


# ---------------------------------------------------------------------------
# E5 — role_word site gate is None (always-on)
# ---------------------------------------------------------------------------

class TestE5RoleWordGate:
    def test_role_word_gate_is_none(self):
        site = next(s for s in egress.SITES if s["kind"] == "role_word")
        assert site["gate"] is None

    def test_role_word_site_live_is_on(self):
        site = next(s for s in egress.SITES if s["kind"] == "role_word")
        assert egress.site_live(site, {}) == "ON"

    def test_role_word_gate_allows_when_not_local_only(self):
        assert egress.gate("role_word", "plain text", {}) is True

    def test_role_word_blocked_by_local_only(self):
        assert egress.gate("role_word", "plain text", {"local_only": "true"}) is False


# ---------------------------------------------------------------------------
# E6 — config caching with mtime invalidation
# ---------------------------------------------------------------------------

class TestE6ConfigCache:
    def test_reset_cache_exists(self):
        egress._reset_config_cache()
        assert egress._config_cache is None

    def test_config_is_cached(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setitem(sys.modules, "hermes_constants", None)
        cfg_path = tmp_path / "hybrid_memory.json"
        cfg_path.write_text('{"local_only": "true"}', encoding="utf-8")
        egress._reset_config_cache()
        cfg1 = egress.load_config()
        assert egress._config_cache is not None
        # Second call serves from cache (same object).
        cfg2 = egress.load_config()
        assert cfg1 is cfg2

    def test_cache_invalidated_on_mtime_change(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setitem(sys.modules, "hermes_constants", None)
        cfg_path = tmp_path / "hybrid_memory.json"
        cfg_path.write_text('{"local_only": "true"}', encoding="utf-8")
        egress._reset_config_cache()
        egress.load_config()
        # Change the file and bump mtime.
        time.sleep(0.05)
        cfg_path.write_text('{"local_only": "false"}', encoding="utf-8")
        os.utime(cfg_path, None)
        cfg = egress.load_config()
        assert cfg.get("local_only") == "false"


# ---------------------------------------------------------------------------
# E7 — known-kinds set is a module-level constant
# ---------------------------------------------------------------------------

class TestE7KnownKinds:
    def test_known_kinds_constant_exists(self):
        assert isinstance(egress._KNOWN_KINDS, set)
        assert egress._KNOWN_KINDS == {site["kind"] for site in egress.SITES}

    def test_sites_by_kind_exists(self):
        assert isinstance(egress._SITES_BY_KIND, dict)
        assert set(egress._SITES_BY_KIND) == egress._KNOWN_KINDS


# ---------------------------------------------------------------------------
# E8 — contains_sensitive reports all matches
# ---------------------------------------------------------------------------

class TestE8AllMatches:
    def test_all_sensitive_labels_returns_all(self):
        text = "email a@b.co.za and phone 0831234567"
        labels = egress.all_sensitive_labels(text)
        assert "email address" in labels
        assert "South African phone number" in labels

    def test_contains_sensitive_still_returns_first(self):
        """Backward compat: contains_sensitive returns the first label."""
        text = "email a@b.co.za and phone 0831234567"
        assert egress.contains_sensitive(text) is not None

    def test_all_sensitive_labels_empty_for_clean_text(self):
        assert egress.all_sensitive_labels("no identifiers here") == []

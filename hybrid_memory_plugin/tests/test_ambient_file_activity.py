"""
Tests for file activity support (hermes_file_activity module + hint builder).

Moved from ambient_context/tests/test_file_activity.py when the ambient
modules were relocated into the plugin package.  The hint builders now live
in hybrid_memory_plugin.__init__ (not agent.turn_context) and ride the native
pre_llm_call plugin hook instead of a core source patch.

Covers:
  - get_recent_files() returns "" when disabled
  - get_recent_files() returns "" when no directories to scan
  - get_recent_files() finds recently modified files
  - Excluded directories (.git, node_modules, venv, __pycache__) are skipped
  - Excluded file patterns (*.log, *.pyc, .DS_Store) are skipped
  - Old files beyond max_age_minutes are not reported
  - Caching: results cached within TTL, re-scanned after TTL
  - _build_file_activity_hint() renders the line and never crashes
  - Paths are shortened with ~ for home directory
"""

import os
import time
import pytest
from pathlib import Path
from unittest.mock import patch
import sys

# Ensure the plugin directory is importable for sibling module imports
# (hermes_file_activity, etc.) and the package parent for package imports.
_plugin_dir = Path(__file__).resolve().parent.parent
if str(_plugin_dir) not in sys.path:
    sys.path.insert(0, str(_plugin_dir))
if str(_plugin_dir.parent) not in sys.path:
    sys.path.insert(0, str(_plugin_dir.parent))

import hermes_file_activity

# The hint builders live in the package __init__. In the bundle repo the
# package is hybrid_memory_plugin; when installed to HERMES_HOME it's
# hybrid_memory. Use the same try/except fallback as test_hybrid_memory.py.
try:
    from hybrid_memory_plugin import _build_file_activity_hint
    _pkg_name = "hybrid_memory_plugin"
except ImportError:
    from hybrid_memory import _build_file_activity_hint
    _pkg_name = "hybrid_memory"

# Package-relative module path for patch targets — __init__.py imports
# via `from .hermes_file_activity import ...`, so we must patch the
# package module, not the standalone one.
_fa_module_path = _pkg_name + ".hermes_file_activity"


def _reset_file_activity_cache():
    """Reset the file activity cache."""
    hermes_file_activity.reset_cache()
    os.environ.pop("TERMINAL_CWD", None)


# =========================================================================
# hermes_file_activity.get_recent_files() — core helper
# =========================================================================

class TestGetRecentFiles:
    """Test the recent files resolution helper."""

    def setup_method(self):
        _reset_file_activity_cache()

    def teardown_method(self):
        _reset_file_activity_cache()

    def test_empty_when_disabled(self):
        """When file_activity.enabled is false, returns ""."""
        with patch("hermes_file_activity._resolve_config",
                   return_value={"enabled": False}):
            assert hermes_file_activity.get_recent_files() == ""

    def test_empty_when_no_dirs(self):
        """When no directories are configured/found, returns ""."""
        with patch("hermes_file_activity._resolve_config",
                   return_value={"enabled": True, "directories": []}), \
             patch("hermes_file_activity._resolve_scan_dirs",
                   return_value=[]):
            assert hermes_file_activity.get_recent_files() == ""

    def test_finds_recently_modified_files(self, tmp_path):
        """With a directory containing recently modified files, returns them."""
        # Create a recently modified file
        f = tmp_path / "foo.py"
        f.write_text("# test")
        os.utime(f, (time.time(), time.time()))

        os.environ["TERMINAL_CWD"] = str(tmp_path)
        with patch("hermes_file_activity._resolve_config",
                   return_value={
            "enabled": True, "directories": [], "max_files": 5, "max_age_minutes": 30,
        }):
            result = hermes_file_activity.get_recent_files()
        assert "foo.py" in result
        assert "min ago" in result or "just now" in result

    def test_excludes_git_dir(self, tmp_path):
        """Files inside .git/ are not reported."""
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        f = git_dir / "config"
        f.write_text("test")
        os.utime(f, (time.time(), time.time()))

        os.environ["TERMINAL_CWD"] = str(tmp_path)
        with patch("hermes_file_activity._resolve_config",
                   return_value={
            "enabled": True, "directories": [], "max_files": 5, "max_age_minutes": 30,
        }):
            result = hermes_file_activity.get_recent_files()
        assert ".git" not in result
        assert "config" not in result or ".git" not in result

    def test_excludes_node_modules(self, tmp_path):
        """Files inside node_modules/ are not reported."""
        nm = tmp_path / "node_modules"
        nm.mkdir()
        f = nm / "package.json"
        f.write_text("{}")
        os.utime(f, (time.time(), time.time()))

        os.environ["TERMINAL_CWD"] = str(tmp_path)
        with patch("hermes_file_activity._resolve_config",
                   return_value={
            "enabled": True, "directories": [], "max_files": 5, "max_age_minutes": 30,
        }):
            result = hermes_file_activity.get_recent_files()
        assert "node_modules" not in result

    def test_excludes_pyc_files(self, tmp_path):
        """Compiled Python files are not reported."""
        f = tmp_path / "module.pyc"
        f.write_bytes(b"\x00\x00\x00\x00")
        os.utime(f, (time.time(), time.time()))

        os.environ["TERMINAL_CWD"] = str(tmp_path)
        with patch("hermes_file_activity._resolve_config",
                   return_value={
            "enabled": True, "directories": [], "max_files": 5, "max_age_minutes": 30,
        }):
            result = hermes_file_activity.get_recent_files()
        assert ".pyc" not in result

    def test_excludes_log_files(self, tmp_path):
        """Log files are not reported."""
        f = tmp_path / "debug.log"
        f.write_text("log entry")
        os.utime(f, (time.time(), time.time()))

        os.environ["TERMINAL_CWD"] = str(tmp_path)
        with patch("hermes_file_activity._resolve_config",
                   return_value={
            "enabled": True, "directories": [], "max_files": 5, "max_age_minutes": 30,
        }):
            result = hermes_file_activity.get_recent_files()
        assert ".log" not in result

    def test_excludes_old_files(self, tmp_path):
        """Files older than max_age_minutes are not reported."""
        f = tmp_path / "old.py"
        f.write_text("# old")
        # Set mtime to 2 hours ago (beyond the 30 min default window)
        old_time = time.time() - (2 * 3600)
        os.utime(f, (old_time, old_time))

        os.environ["TERMINAL_CWD"] = str(tmp_path)
        with patch("hermes_file_activity._resolve_config",
                   return_value={
            "enabled": True, "directories": [], "max_files": 5, "max_age_minutes": 30,
        }):
            result = hermes_file_activity.get_recent_files()
        assert result == ""

    def test_max_files_limit(self, tmp_path):
        """Only the N most recent files are reported."""
        for i in range(10):
            f = tmp_path / f"file_{i}.py"
            f.write_text(f"# {i}")
            # Stagger mtimes so file_9 is most recent
            os.utime(f, (time.time() - (10 - i), time.time() - (10 - i)))

        os.environ["TERMINAL_CWD"] = str(tmp_path)
        with patch("hermes_file_activity._resolve_config",
                   return_value={
            "enabled": True, "directories": [], "max_files": 3, "max_age_minutes": 30,
        }):
            result = hermes_file_activity.get_recent_files()
        # Should contain at most 3 files (comma-separated)
        file_count = result.count(".py")
        assert file_count <= 3

    def test_shortens_home_path(self, tmp_path):
        """Paths under $HOME are shortened with ~."""
        f = tmp_path / "test.py"
        f.write_text("# test")
        os.utime(f, (time.time(), time.time()))

        # Patch home to be tmp_path so the path gets shortened
        with patch("hermes_file_activity._resolve_config",
                   return_value={
            "enabled": True, "directories": [str(tmp_path)], "max_files": 5, "max_age_minutes": 30,
        }), patch("os.path.expanduser", return_value=str(tmp_path)):
            result = hermes_file_activity.get_recent_files()
        assert "~/" in result


class TestFileActivityCaching:
    """Test the caching behavior."""

    def setup_method(self):
        _reset_file_activity_cache()

    def teardown_method(self):
        _reset_file_activity_cache()

    def test_cached_within_ttl(self, tmp_path):
        """Results are cached — within the TTL, no re-scan."""
        f = tmp_path / "foo.py"
        f.write_text("# test")
        os.utime(f, (time.time(), time.time()))

        os.environ["TERMINAL_CWD"] = str(tmp_path)
        with patch("hermes_file_activity._resolve_config",
                   return_value={
            "enabled": True, "directories": [], "max_files": 5, "max_age_minutes": 30,
        }):
            r1 = hermes_file_activity.get_recent_files()
            r2 = hermes_file_activity.get_recent_files()
        assert r1 == r2

    def test_refetched_after_ttl(self, tmp_path):
        """After the TTL expires, results are re-scanned."""
        f = tmp_path / "foo.py"
        f.write_text("# test")
        os.utime(f, (time.time(), time.time()))

        os.environ["TERMINAL_CWD"] = str(tmp_path)
        with patch("hermes_file_activity._resolve_config",
                   return_value={
            "enabled": True, "directories": [], "max_files": 5, "max_age_minutes": 30,
        }), patch("hermes_file_activity.FILE_ACTIVITY_CACHE_TTL_S", 0.01):
            r1 = hermes_file_activity.get_recent_files()
            time.sleep(0.02)
            r2 = hermes_file_activity.get_recent_files()
        # Both should find the file (re-scanned after TTL)
        assert "foo.py" in r1
        assert "foo.py" in r2


# =========================================================================
# _build_file_activity_hint — the per-turn injection line (now in the plugin)
# =========================================================================

class TestBuildFileActivityHint:
    """The per-turn ``Last edited: ...`` line built by the plugin's
    pre_llm_call hook callback."""

    def setup_method(self):
        _reset_file_activity_cache()

    def teardown_method(self):
        _reset_file_activity_cache()

    def test_renders_when_files_available(self):
        with patch(_fa_module_path + ".get_recent_files",
                   return_value="~/project/foo.py (4 min ago)"):
            assert _build_file_activity_hint() == "Last edited: ~/project/foo.py (4 min ago)"

    def test_empty_when_no_files(self):
        with patch(_fa_module_path + ".get_recent_files",
                   return_value=""):
            assert _build_file_activity_hint() == ""

    def test_never_crashes_on_exception(self):
        with patch(_fa_module_path + ".get_recent_files",
                   side_effect=RuntimeError("boom")):
            assert _build_file_activity_hint() == ""

    def test_multiple_files_formatted(self):
        with patch(_fa_module_path + ".get_recent_files",
                   return_value="~/project/foo.py (4 min ago), ~/project/bar.py (12 min ago)"):
            result = _build_file_activity_hint()
        assert result == "Last edited: ~/project/foo.py (4 min ago), ~/project/bar.py (12 min ago)"

"""
Recent file activity for Hermes.

Provides ``get_recent_files()`` which returns a short human-readable string
describing the most recently modified files in the user's working directory,
e.g. ``"~/project/foo.py (4 min ago), ~/project/bar.py (12 min ago)"``.

This is the "what was I just doing?" ambient context hint — when the user
messages from Telegram/Discord, the agent has no idea what's happening on
their machine. This hint bridges that gap so the agent can say "I see you
were just editing foo.py — want me to pick up where you left off?" without
the user having to explain.

Scanning strategy (keeps it off the time-to-first-token critical path):
  - Scans the configured working directory (``terminal.cwd`` / ``TERMINAL_CWD``
    env var) for recently modified files.
  - Cached for ``FILE_ACTIVITY_CACHE_TTL_S`` (default 5 min). Most turns are
    cache hits — a dict lookup, zero I/O.
  - A cache miss is a directory scan (~50-200ms for typical project dirs),
    the same order as the memory prefetch that already runs in the prologue.
  - Network failure / permissions error: returns ``""`` (no hint that turn).

Exclusions: skips ``.git``, ``node_modules``, ``venv``, ``__pycache__``, etc.
so the hint reflects actual user work, not tool/cache churn.

Disable with::

    hermes config set file_activity.enabled false
"""

import fnmatch
import logging
import os
import threading
import time
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# How long to cache scan results before re-scanning. 5 minutes is a good
# balance — "what were you doing" doesn't change second-to-second, and this
# means most turns in a conversation are zero-I/O cache hits.
FILE_ACTIVITY_CACHE_TTL_S: float = 5 * 60.0

# Only report files modified within this window. 30 min captures "what was
# I just doing" without dragging in stale context from earlier in the day.
_DEFAULT_MAX_AGE_MINUTES: int = 30

# Max files to report. Keep it short — this is a hint, not a file listing.
_DEFAULT_MAX_FILES: int = 5

# Max scan depth. 2 levels catches src/foo.py and src/module/bar.py without
# scanning deep into dependency trees.
_DEFAULT_MAX_DEPTH: int = 2

# Directories to skip — these are tool/cache/VCS churn, not user work.
# Mirrors agent/skill_utils.py EXCLUDED_SKILL_DIRS plus common build dirs.
_EXCLUDED_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        ".github",
        ".hub",
        ".archive",
        ".venv",
        "venv",
        "env",
        "node_modules",
        "site-packages",
        "__pycache__",
        ".tox",
        ".nox",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "build",
        "dist",
        ".eggs",
        ".idea",
        ".vscode",
        ".next",
        ".nuxt",
        ".output",
        ".turbo",
        "target",
        ".cargo",
        ".gradle",
        ".terraform",
    }
)

# File patterns to skip — logs, temp files, editor swap files.
_EXCLUDED_PATTERNS: tuple[str, ...] = (
    "*.log",
    "*.tmp",
    "*.cache",
    "*.swp",
    "*.swo",
    "*~",
    "*.pyc",
    "*.pyo",
    "*.class",
    "*.o",
    "*.so",
    "*.dll",
    "*.dylib",
    "*.exe",
    ".DS_Store",
    "Thumbs.db",
)

# Cache: (cache_key, timestamp, result_string)
_cache_lock = threading.Lock()
_cached: Optional[Tuple[str, float, str]] = None


def _resolve_config() -> dict:
    """Read the file_activity config section (or defaults)."""
    defaults = {
        "enabled": True,
        "directories": [],
        "max_files": _DEFAULT_MAX_FILES,
        "max_age_minutes": _DEFAULT_MAX_AGE_MINUTES,
    }
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
            fa_cfg = cfg.get("file_activity", {})
            if isinstance(fa_cfg, dict):
                for key in defaults:
                    if key in fa_cfg:
                        defaults[key] = fa_cfg[key]
    except Exception:
        pass
    return defaults


def _resolve_scan_dirs(config: dict) -> List[Path]:
    """Determine which directories to scan.

    Priority:
      1. ``file_activity.directories`` config (if non-empty)
      2. ``TERMINAL_CWD`` env var
      3. ``terminal.cwd`` config key
      4. Current working directory
    """
    dirs_cfg = config.get("directories", [])
    if dirs_cfg and isinstance(dirs_cfg, list):
        paths = []
        for d in dirs_cfg:
            if isinstance(d, str) and d.strip():
                p = Path(d).expanduser()
                if p.is_dir():
                    paths.append(p)
        if paths:
            return paths

    # Fall back to TERMINAL_CWD env var (set by gateway from terminal.cwd config)
    raw_cwd = os.environ.get("TERMINAL_CWD", "").strip()
    if raw_cwd:
        p = Path(raw_cwd).expanduser()
        if p.is_dir():
            return [p]

    # Last resort: current working directory
    try:
        cwd = Path(os.getcwd())
        if cwd.is_dir():
            return [cwd]
    except Exception:
        pass

    return []


def _is_excluded(path: Path) -> bool:
    """Check if a path should be excluded from the scan."""
    # Skip excluded directory components
    for part in path.parts:
        if part in _EXCLUDED_DIRS:
            return True
    # Skip excluded file patterns
    name = path.name
    for pattern in _EXCLUDED_PATTERNS:
        if fnmatch.fnmatch(name, pattern):
            return True
    # Skip dotfiles (but not dot-directories like .config which may have
    # legitimate config files — actually skip those too, they're not "work")
    if name.startswith(".") and name not in (".env", ".env.local"):
        return True
    return False


def _scan_dir(
    root: Path, max_depth: int, max_age_s: float, now_ts: float
) -> List[Tuple[float, Path]]:
    """Scan a directory for recently modified files.

    Returns a list of (mtime, path) tuples sorted by mtime descending.
    """
    results: List[Tuple[float, Path]] = []
    try:
        for entry in root.iterdir():
            if entry.is_dir():
                # Skip excluded directories at the top level
                if entry.name in _EXCLUDED_DIRS or entry.name.startswith("."):
                    continue
                if max_depth > 1:
                    results.extend(
                        _scan_dir(entry, max_depth - 1, max_age_s, now_ts)
                    )
            elif entry.is_file():
                if _is_excluded(entry):
                    continue
                try:
                    mtime = entry.stat().st_mtime
                except (OSError, PermissionError):
                    continue
                if now_ts - mtime <= max_age_s:
                    results.append((mtime, entry))
    except (OSError, PermissionError):
        pass
    return results


def _format_relative_time(seconds_ago: float) -> str:
    """Format seconds ago as a human-readable string."""
    mins = int(seconds_ago / 60)
    if mins < 1:
        return "just now"
    if mins == 1:
        return "1 min ago"
    if mins < 60:
        return f"{mins} min ago"
    hours = int(mins / 60)
    if hours == 1:
        return "1 hr ago"
    if hours < 24:
        return f"{hours} hr ago"
    days = int(hours / 24)
    if days == 1:
        return "1 day ago"
    return f"{days} days ago"


def _shorten_path(path: Path, home: Path) -> str:
    """Shorten a path for display — replace $HOME with ~."""
    try:
        rel = path.relative_to(home)
        return f"~/{rel}"
    except (ValueError, TypeError):
        return str(path)


def get_recent_files() -> str:
    """Return a short string describing recently modified files, or ``""``.

    Returns ``""`` when:
      - File activity is disabled (``file_activity.enabled: false``)
      - No directories to scan
      - No recent files found
      - The scan fails (permissions, etc.)

    On success: ``"~/project/foo.py (4 min ago), ~/project/bar.py (12 min ago)"``
    """
    global _cached

    config = _resolve_config()
    if not config.get("enabled", True):
        return ""

    scan_dirs = _resolve_scan_dirs(config)
    if not scan_dirs:
        return ""

    # Cache key: directories + config values. If the user changes directories
    # or config, the cache key changes and we re-scan.
    cache_key = str((tuple(str(d) for d in scan_dirs), config.get("max_files"), config.get("max_age_minutes")))

    now_ts = time.time()
    with _cache_lock:
        if _cached is not None:
            key, ts, result = _cached
            if key == cache_key and now_ts - ts < FILE_ACTIVITY_CACHE_TTL_S:
                return result

    max_files = config.get("max_files", _DEFAULT_MAX_FILES)
    max_age_minutes = config.get("max_age_minutes", _DEFAULT_MAX_AGE_MINUTES)
    max_age_s = float(max_age_minutes) * 60.0

    # Scan all directories, collect results, sort by mtime.
    all_results: List[Tuple[float, Path]] = []
    home = Path(os.path.expanduser("~"))
    for d in scan_dirs:
        all_results.extend(_scan_dir(d, _DEFAULT_MAX_DEPTH, max_age_s, now_ts))

    if not all_results:
        result = ""
    else:
        all_results.sort(key=lambda x: x[0], reverse=True)
        top = all_results[:max_files]
        parts = [
            f"{_shorten_path(p, home)} ({_format_relative_time(now_ts - m)})"
            for m, p in top
        ]
        result = ", ".join(parts)

    with _cache_lock:
        _cached = (cache_key, now_ts, result)
    return result


def reset_cache() -> None:
    """Clear the scan cache. Call after config changes."""
    global _cached
    with _cache_lock:
        _cached = None

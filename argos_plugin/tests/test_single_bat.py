"""Tests for scripts/single.bat - the one-shot Windows onboarding (issue #485).

The bat itself is Windows-only; the structural checks run on any platform,
and the execute-only tests are skipped off-win32.  The /check path must be
side-effect free, so a real run against a scratch home is safe on CI that
happens to run Windows.
"""
from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import tempfile

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
SINGLE_BAT = REPO_ROOT / "scripts" / "single.bat"


def _read_bat() -> str:
    return SINGLE_BAT.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "needle",
    [
        "@echo off",
        "bobaba76/Argos",
        "issue #485",
        "/check",
        "/auto",
        "/nui",
        "/home:",
        "/port:",
        "/src:",
        "SINGLE_HOME",
        "plugins\\hybrid_memory",
        "admin_console.py",
        "schtasks /create",
        "sc onlogon",
        "curl",
        "config set memory.provider hybrid_memory",
        "http://127.0.0.1",
        "Create your admin key",
        "exit /b 3",
        "exit /b 1",
        "exit /b 0",
    ],
)
def test_bat_markers_present(needle: str) -> None:
    """Every contract in the issue body is represented in the file."""
    assert needle in _read_bat()


def test_bat_has_crlf_line_endings() -> None:
    raw = SINGLE_BAT.read_bytes()
    # A CRLF/eol=crlf-normalized batch file has \\r\\n throughout.
    assert b"\r\n" in raw
    assert raw.count(b"\r\n") == raw.count(b"\n")


def test_bat_rem_lines_never_start_with_question_mark() -> None:
    """cmd.exe mis-parses 'rem /? ...' (help switch) - a gotcha recorded in
    the argos-plugin-deploy skill. Keep question marks out of REM lines."""
    for line in _read_bat().splitlines():
        stripped = line.lstrip()
        if stripped.startswith("rem "):
            assert not stripped[4:].lstrip().startswith("/?"), line


@pytest.mark.skipif(
    sys.platform != "win32", reason="single.bat only executes on Windows"
)
def test_check_mode_on_missing_home_is_clean(tmp_path: pathlib.Path) -> None:
    """/check with a non-existent home exits 3 and creates nothing."""
    missing = tmp_path / "no-such-hermes-home"
    result = subprocess.run(
        ["cmd.exe", "/c", str(SINGLE_BAT), "/check", f"/home:{missing}"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 3
    assert "Hermes home not found" in result.stdout


@pytest.mark.skipif(
    sys.platform != "win32", reason="single.bat only executes on Windows"
)
def test_check_mode_on_minimal_home_is_side_effect_free(tmp_path: pathlib.Path) -> None:
    """/check against a minimal fake home prints state and touches nothing."""
    home = tmp_path / "hermes"
    home.mkdir()
    (home / "config.yaml").write_text("provider: hybrid_memory\n", encoding="utf-8")
    plugins = home / "plugins" / "hybrid_memory"
    plugins.mkdir(parents=True)
    (plugins / "admin_console.py").write_text("print('hi')\n", encoding="utf-8")
    (home / "admin_console.log").write_text("", encoding="utf-8")

    result = subprocess.run(
        ["cmd.exe", "/c", str(SINGLE_BAT), "/check", f"/home:{home}"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0
    assert "CHECK OK" in result.stdout
    # The check mode must not have written anything.
    assert list(tmp_path.iterdir()) == [home]
    assert (plugins / "admin_console.py").read_text() == "print('hi')\n"
"""Tests for #308: deploy.py atomic swap / versioned rollback.

Tests the staging-then-atomic-rename deploy mode and the rollback mode.
Uses temp dirs as fake source/target — does NOT run deploy.py against
the real live install.

Run with (Hermes venv python, hermetic):
    ARGOS_HERMETIC_TESTS=1 python -m pytest tests/test_deploy_atomic_swap.py -v
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_plugin_dir = Path(__file__).resolve().parent.parent
_repo_root = _plugin_dir.parent
for _path in (_repo_root, _plugin_dir):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


def _load_deploy_module():
    """Load scripts/deploy.py as a module (it's not in a package)."""
    deploy_path = _repo_root / "scripts" / "deploy.py"
    spec = importlib.util.spec_from_file_location("deploy", deploy_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def fake_source(tmp_path):
    """Create a fake source plugin dir with a few .py files."""
    src = tmp_path / "fake_source"
    src.mkdir()
    (src / "plugin.yaml").write_text("name: test\nversion: '1.0'\n", encoding="utf-8")
    (src / "module_a.py").write_text("# module a\n", encoding="utf-8")
    (src / "module_b.py").write_text("# module b\n", encoding="utf-8")
    # Source also has extractor_patterns/ with en.json.
    ep = src / "extractor_patterns"
    ep.mkdir()
    (ep / "en.json").write_text('{"patterns": []}', encoding="utf-8")
    return src


@pytest.fixture
def fake_target(tmp_path, fake_source):
    """Create a fake target (live install) dir with old versions of files
    plus live-only artifacts that must be preserved."""
    tgt = tmp_path / "fake_target"
    tgt.mkdir()
    (tgt / "plugin.yaml").write_text("name: old\nversion: '0.9'\n", encoding="utf-8")
    (tgt / "module_a.py").write_text("# old module a\n", encoding="utf-8")
    # Live-only artifacts.
    (tgt / "skills").mkdir()  # insight-log
    (tgt / "skills" / "insight.md").write_text("insight log", encoding="utf-8")
    (tgt / "eval").mkdir()  # eval scripts
    (tgt / "state.db").write_text("state", encoding="utf-8")  # local state
    # Live extractor_patterns/ has en.json (from a prior deploy) PLUS
    # a live-only locale file (fr.json) not in the repo source.
    ep = tgt / "extractor_patterns"
    ep.mkdir()
    (ep / "en.json").write_text('{"patterns": ["old"]}', encoding="utf-8")
    (ep / "fr.json").write_text('{"patterns": ["fr"]}', encoding="utf-8")
    return tgt


@pytest.fixture
def deploy_mod():
    return _load_deploy_module()


class TestAtomicSwap:
    """#308: staging-then-atomic-rename deploy."""

    def test_atomic_swap_creates_versioned_live(self, fake_source, fake_target, tmp_path, deploy_mod, monkeypatch):
        """--atomic-swap copies source to a staged dir, verifies byte-parity,
        then atomically swaps the live dir reference."""
        monkeypatch.setattr(deploy_mod, "_is_service_running", lambda: False)
        state = tmp_path / "deploy_state.json"
        result = deploy_mod.atomic_swap_mode(
            fake_source, fake_target, state, restart=False
        )
        assert result == 0

        # The live dir should now contain the new files.
        assert (fake_target / "plugin.yaml").read_text(encoding="utf-8") == \
            "name: test\nversion: '1.0'\n"
        assert (fake_target / "module_a.py").read_text(encoding="utf-8") == \
            "# module a\n"
        assert (fake_target / "module_b.py").read_text(encoding="utf-8") == \
            "# module b\n"

        # The old live dir should be preserved as a backup.
        backups = deploy_mod._find_versioned_backups(fake_target)
        assert len(backups) >= 1
        backup = backups[0]
        assert (backup / "module_a.py").read_text(encoding="utf-8") == \
            "# old module a\n"

        # deploy_state.json should record the deployment.
        state_data = deploy_mod.load_state(state)
        deployments = state_data.get("deployments", [])
        assert len(deployments) == 1
        assert deployments[0]["mode"] == "atomic-swap"
        assert "backup_dir" in deployments[0]

    def test_atomic_swap_preserves_live_only_artifacts(self, fake_source, fake_target, tmp_path, deploy_mod, monkeypatch):
        """The new live dir MUST preserve live-only artifacts (skills/,
        eval/, state.db) — they are copied into the staged dir BEFORE
        the swap. The backup also retains them."""
        monkeypatch.setattr(deploy_mod, "_is_service_running", lambda: False)
        state = tmp_path / "deploy_state.json"
        result = deploy_mod.atomic_swap_mode(
            fake_source, fake_target, state, restart=False
        )
        assert result == 0

        # The NEW live dir should STILL have skills/ (preserved).
        assert (fake_target / "skills").exists()
        assert (fake_target / "skills" / "insight.md").read_text(encoding="utf-8") == \
            "insight log"
        # The NEW live dir should STILL have eval/ (preserved).
        assert (fake_target / "eval").exists()
        # The NEW live dir should STILL have state.db (preserved).
        assert (fake_target / "state.db").exists()

        # The backup should also have them (it's the old live dir).
        backups = deploy_mod._find_versioned_backups(fake_target)
        assert len(backups) >= 1
        backup = backups[0]
        assert (backup / "skills").exists()
        assert (backup / "eval").exists()
        assert (backup / "state.db").exists()

    def test_atomic_swap_preserves_live_only_extractor_patterns(self, fake_source, fake_target, tmp_path, deploy_mod, monkeypatch):
        """extractor_patterns/ is loaded at runtime by extractor.py:112.
        Live-only locale files (fr.json) not in the repo source MUST be
        preserved. Source files (en.json) are overwritten with the new
        version."""
        monkeypatch.setattr(deploy_mod, "_is_service_running", lambda: False)
        state = tmp_path / "deploy_state.json"
        result = deploy_mod.atomic_swap_mode(
            fake_source, fake_target, state, restart=False
        )
        assert result == 0

        # en.json should be the NEW version from source.
        assert (fake_target / "extractor_patterns" / "en.json").read_text(encoding="utf-8") == \
            '{"patterns": []}'
        # fr.json (live-only) should be preserved.
        assert (fake_target / "extractor_patterns" / "fr.json").read_text(encoding="utf-8") == \
            '{"patterns": ["fr"]}'

    def test_atomic_swap_byte_parity_verification(self, fake_source, fake_target, tmp_path, deploy_mod, monkeypatch):
        """Every copied file is byte-parity verified (sha256 match)."""
        monkeypatch.setattr(deploy_mod, "_is_service_running", lambda: False)
        state = tmp_path / "deploy_state.json"
        result = deploy_mod.atomic_swap_mode(
            fake_source, fake_target, state, restart=False
        )
        assert result == 0

        state_data = deploy_mod.load_state(state)
        deployment = state_data["deployments"][0]
        copied = deployment["copied"]
        # Each copied entry should have a sha256.
        for entry in copied:
            assert "sha256" in entry
            assert len(entry["sha256"]) == 64  # sha256 hex digest

    def test_atomic_swap_records_preserved_artifacts(self, fake_source, fake_target, tmp_path, deploy_mod, monkeypatch):
        """deploy_state.json records which live-only artifacts were preserved."""
        monkeypatch.setattr(deploy_mod, "_is_service_running", lambda: False)
        state = tmp_path / "deploy_state.json"
        result = deploy_mod.atomic_swap_mode(
            fake_source, fake_target, state, restart=False
        )
        assert result == 0

        state_data = deploy_mod.load_state(state)
        deployment = state_data["deployments"][0]
        assert "preserved" in deployment
        preserved = deployment["preserved"]
        # skills, eval, state.db should be in the preserved list.
        assert "skills" in preserved
        assert "eval" in preserved
        assert "state.db" in preserved


class TestAtomicSwapServiceCheck:
    """#308: --atomic-swap refuses to run while the memory service is alive."""

    def test_atomic_swap_refuses_while_service_running(self, fake_source, fake_target, tmp_path, deploy_mod, monkeypatch):
        """--atomic-swap returns error code 2 if memory_service.py is running."""
        # Mock _is_service_running to return True.
        monkeypatch.setattr(deploy_mod, "_is_service_running", lambda: True)
        state = tmp_path / "deploy_state.json"
        result = deploy_mod.atomic_swap_mode(
            fake_source, fake_target, state, restart=False
        )
        assert result == 2
        # The live dir should be untouched.
        assert (fake_target / "module_a.py").read_text(encoding="utf-8") == \
            "# old module a\n"

    def test_atomic_swap_proceeds_when_service_stopped(self, fake_source, fake_target, tmp_path, deploy_mod, monkeypatch):
        """--atomic-swap proceeds when the service is not running."""
        # Mock _is_service_running to return False.
        monkeypatch.setattr(deploy_mod, "_is_service_running", lambda: False)
        state = tmp_path / "deploy_state.json"
        result = deploy_mod.atomic_swap_mode(
            fake_source, fake_target, state, restart=False
        )
        assert result == 0


class TestRollback:
    """#308: versioned rollback."""

    def test_rollback_restores_previous_version(self, fake_source, fake_target, tmp_path, deploy_mod, monkeypatch):
        """--rollback renames current live → failed, restores the backup."""
        monkeypatch.setattr(deploy_mod, "_is_service_running", lambda: False)
        state = tmp_path / "deploy_state.json"

        # First, do an atomic swap to create a backup.
        result = deploy_mod.atomic_swap_mode(
            fake_source, fake_target, state, restart=False
        )
        assert result == 0

        # Verify the new content is live.
        assert (fake_target / "module_a.py").read_text(encoding="utf-8") == \
            "# module a\n"

        # Now roll back.
        result = deploy_mod.rollback_mode(fake_target, state)
        assert result == 0

        # The live dir should now have the OLD content.
        assert (fake_target / "module_a.py").read_text(encoding="utf-8") == \
            "# old module a\n"

        # The failed dir should have the new content.
        parent = fake_target.parent
        failed_dirs = [
            p for p in parent.iterdir()
            if p.is_dir() and p.name.startswith("fake_target.failed-")
        ]
        assert len(failed_dirs) >= 1
        failed = failed_dirs[0]
        assert (failed / "module_a.py").read_text(encoding="utf-8") == \
            "# module a\n"

    def test_rollback_without_state_uses_newest_backup(self, fake_source, fake_target, tmp_path, deploy_mod, monkeypatch):
        """--rollback falls back to scanning for backup dirs if no state entry."""
        monkeypatch.setattr(deploy_mod, "_is_service_running", lambda: False)
        state = tmp_path / "deploy_state.json"

        # Do an atomic swap (creates state + backup).
        result = deploy_mod.atomic_swap_mode(
            fake_source, fake_target, state, restart=False
        )
        assert result == 0

        # Delete the state file — rollback should still work via scanning.
        state.unlink()

        result = deploy_mod.rollback_mode(fake_target, state)
        assert result == 0
        assert (fake_target / "module_a.py").read_text(encoding="utf-8") == \
            "# old module a\n"

    def test_rollback_without_backup_returns_error(self, fake_target, tmp_path, deploy_mod):
        """--rollback returns error code 2 if no backup exists."""
        state = tmp_path / "deploy_state.json"
        result = deploy_mod.rollback_mode(fake_target, state)
        assert result == 2


class TestListVersions:
    """#308: list available versioned live dirs."""

    def test_list_versions_shows_backups(self, fake_source, fake_target, tmp_path, deploy_mod, monkeypatch):
        """--list-versions shows available backup dirs."""
        monkeypatch.setattr(deploy_mod, "_is_service_running", lambda: False)
        state = tmp_path / "deploy_state.json"

        # Do two atomic swaps to create two backups.
        result = deploy_mod.atomic_swap_mode(
            fake_source, fake_target, state, restart=False
        )
        assert result == 0

        # Modify source and swap again.
        (fake_source / "module_a.py").write_text("# module a v2\n", encoding="utf-8")
        result = deploy_mod.atomic_swap_mode(
            fake_source, fake_target, state, restart=False
        )
        assert result == 0

        result = deploy_mod.list_versions_mode(fake_target, state)
        assert result == 0

    def test_list_versions_no_backups(self, fake_target, tmp_path, deploy_mod):
        """--list-versions returns 0 with no backups."""
        state = tmp_path / "deploy_state.json"
        result = deploy_mod.list_versions_mode(fake_target, state)
        assert result == 0


class TestAtomicSwapRecovery:
    """#308: atomic swap handles errors gracefully."""

    def test_atomic_swap_missing_source_returns_error(self, tmp_path, deploy_mod, monkeypatch):
        """--atomic-swap returns 2 if source doesn't exist."""
        monkeypatch.setattr(deploy_mod, "_is_service_running", lambda: False)
        source = tmp_path / "nonexistent"
        target = tmp_path / "target"
        target.mkdir()
        state = tmp_path / "deploy_state.json"
        result = deploy_mod.atomic_swap_mode(source, target, state, restart=False)
        assert result == 2

    def test_atomic_swap_missing_target_returns_error(self, tmp_path, deploy_mod, monkeypatch):
        """--atomic-swap returns 2 if target doesn't exist."""
        monkeypatch.setattr(deploy_mod, "_is_service_running", lambda: False)
        source = tmp_path / "source"
        source.mkdir()
        (source / "test.py").write_text("# test\n", encoding="utf-8")
        target = tmp_path / "nonexistent"
        state = tmp_path / "deploy_state.json"
        result = deploy_mod.atomic_swap_mode(source, target, state, restart=False)
        assert result == 2

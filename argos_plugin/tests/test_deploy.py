"""Tests for scripts/deploy.py (issue #7): repo → live plugin sync.

Covers:
- Drift detection: CHANGED/NEW/REMOVED/UNCHANGED classification, exit codes
- Copy round-trip: byte-identical, .bak-<ts> backup before overwrite
- deploy_state.json records HEAD + copied files
- --prune removes in-scope strays but never protected artifacts
- Dev utilities and test files never sync
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_plugin_dir = Path(__file__).resolve().parent.parent
if str(_plugin_dir) not in sys.path:
    sys.path.insert(0, str(_plugin_dir))
_scripts_dir = _plugin_dir.parent / "scripts"
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))

import deploy  # noqa: E402


def _write(path: Path, content: str = "x = 1\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _live_plugin_dir() -> Path:
    """The deployed plugin copy (#360 canary target)."""
    import os as _os

    return (
        Path(_os.environ.get("LOCALAPPDATA", ""))
        / "hermes"
        / "plugins"
        / "hybrid_memory"
    )


class TestDiffFiles:
    """diff_files should classify drift correctly."""

    def test_clean_pair(self, tmp_path):
        src, tgt = tmp_path / "src", tmp_path / "tgt"
        _write(src / "a.py", "same\n")
        _write(tgt / "a.py", "same\n")
        diff = deploy.diff_files(src, tgt)
        assert diff["unchanged"] == ["a.py"]
        assert diff["changed"] == [] and diff["new"] == [] and diff["removed"] == []

    def test_changed_new_removed(self, tmp_path):
        src, tgt = tmp_path / "src", tmp_path / "tgt"
        _write(src / "a.py", "new version\n")
        _write(tgt / "a.py", "old version\n")
        _write(src / "b.py")
        _write(tgt / "c.py")
        diff = deploy.diff_files(src, tgt)
        assert diff["changed"] == ["a.py"]
        assert diff["new"] == ["b.py"]
        assert diff["removed"] == ["c.py"]

    def test_dev_utilities_never_in_scope(self, tmp_path):
        """Dev utilities in source are excluded; legacy ones in target are
        excluded too (never reported as drift, never pruned)."""
        src, tgt = tmp_path / "src", tmp_path / "tgt"
        _write(src / "cleanup_memories.py")
        _write(src / "backfill_graph.py")
        _write(tgt / "cleanup_memories.py")
        _write(tgt / "dump_memories.py")
        _write(tgt / "test_argos.py")
        diff = deploy.diff_files(src, tgt)
        assert diff["new"] == [] and diff["removed"] == []

    def test_plugin_yaml_in_scope(self, tmp_path):
        src, tgt = tmp_path / "src", tmp_path / "tgt"
        _write(src / "plugin.yaml", "name: hybrid_memory\n")
        _write(tgt / "plugin.yaml", "name: hybrid_memory\n")
        diff = deploy.diff_files(src, tgt)
        assert diff["unchanged"] == ["plugin.yaml"]


class TestCopyMode:
    """Copy mode should verify byte-identity and record state."""

    def test_copy_roundtrip_byte_identical(self, tmp_path):
        src, tgt = tmp_path / "src", tmp_path / "tgt"
        tgt.mkdir()
        _write(src / "a.py", "content A\n")
        _write(src / "b.py", "content B\n")
        state = tmp_path / "state.json"
        rc = deploy.copy_mode(src, tgt, state, prune=False, restart=False)
        assert rc == 0
        assert deploy.sha256(src / "a.py") == deploy.sha256(tgt / "a.py")
        assert deploy.sha256(src / "b.py") == deploy.sha256(tgt / "b.py")

    def test_backup_created_before_overwrite(self, tmp_path):
        src, tgt = tmp_path / "src", tmp_path / "tgt"
        tgt.mkdir()
        _write(src / "a.py", "new\n")
        _write(tgt / "a.py", "old\n")
        state = tmp_path / "state.json"
        rc = deploy.copy_mode(src, tgt, state, prune=False, restart=False)
        assert rc == 0
        baks = list(tgt.glob("a.py.bak-*"))
        assert len(baks) == 1
        assert baks[0].read_text(encoding="utf-8") == "old\n"
        assert (tgt / "a.py").read_text(encoding="utf-8") == "new\n"

    def test_state_records_head_and_copies(self, tmp_path):
        src, tgt = tmp_path / "src", tmp_path / "tgt"
        tgt.mkdir()
        _write(src / "a.py")
        state = tmp_path / "state.json"
        deploy.copy_mode(src, tgt, state, prune=False, restart=False)
        data = json.loads(state.read_text(encoding="utf-8"))
        assert len(data["deployments"]) == 1
        entry = data["deployments"][-1]
        assert entry["head"] == deploy.repo_head()
        assert entry["copied"] == [{"file": "a.py", "sha256": deploy.sha256(src / "a.py")}]

    def test_prune_removes_stray_not_protected(self, tmp_path):
        src, tgt = tmp_path / "src", tmp_path / "tgt"
        tgt.mkdir()
        _write(src / "a.py")
        _write(tgt / "a.py")
        _write(tgt / "stray.py")
        _write(tgt / "skills")  # protected artifact name
        _write(tgt / "hybrid_memory.duckdb")
        _write(tgt / "old.py.bak-20260101")
        state = tmp_path / "state.json"
        rc = deploy.copy_mode(src, tgt, state, prune=True, restart=False)
        assert rc == 0
        assert not (tgt / "stray.py").exists(), "in-scope stray must be pruned"
        assert (tgt / "skills").exists(), "protected artifact must survive"
        assert (tgt / "hybrid_memory.duckdb").exists(), "db must survive"
        assert (tgt / "old.py.bak-20260101").exists(), "backup artifact must survive"

    def test_no_prune_by_default(self, tmp_path):
        src, tgt = tmp_path / "src", tmp_path / "tgt"
        tgt.mkdir()
        _write(src / "a.py")
        _write(tgt / "a.py")
        _write(tgt / "stray.py")
        state = tmp_path / "state.json"
        deploy.copy_mode(src, tgt, state, prune=False, restart=False)
        assert (tgt / "stray.py").exists(), "without --prune nothing is deleted"


class TestCheckMode:
    """check_mode should return 0 clean / 1 drift / 2 error."""

    def test_clean_exit_zero(self, tmp_path):
        src, tgt = tmp_path / "src", tmp_path / "tgt"
        tgt.mkdir()
        _write(src / "a.py", "same\n")
        _write(tgt / "a.py", "same\n")
        assert deploy.check_mode(src, tgt, tmp_path / "state.json") == 0

    def test_drift_exit_one(self, tmp_path):
        src, tgt = tmp_path / "src", tmp_path / "tgt"
        tgt.mkdir()
        _write(src / "a.py", "new\n")
        _write(tgt / "a.py", "old\n")
        assert deploy.check_mode(src, tgt, tmp_path / "state.json") == 1

    def test_missing_target_exit_two(self, tmp_path):
        src = tmp_path / "src"
        _write(src / "a.py")
        assert deploy.check_mode(src, tmp_path / "missing", tmp_path / "s.json") == 2


class TestVersionStamp360:
    """#360: version stamp + repo-vs-live module inventory parity."""

    def test_version_consistent_across_files(self):
        import re as _re

        vfile = (_plugin_dir / "version.py").read_text(encoding="utf-8")
        m = _re.search(r'__version__\s*=\s*"([^"]+)"', vfile)
        assert m, "version.py must define __version__"
        ver = m.group(1)
        assert _re.match(r"^\d+\.\d+\.\d+$", ver), ver
        yaml_text = (_plugin_dir / "plugin.yaml").read_text(encoding="utf-8")
        assert "version: %s" % ver in yaml_text
        init_text = (_plugin_dir / "__init__.py").read_text(encoding="utf-8")
        assert "__version__" in init_text

    def test_deploy_reads_plugin_version(self, tmp_path):
        d = tmp_path / "src"
        _write(d / "version.py", '__version__ = "9.9.9"\n')
        assert deploy._plugin_version(d) == "9.9.9"
        assert deploy._plugin_version(tmp_path / "missing") == "unknown"
        (d / "version.py").write_text("no version literal", encoding="utf-8")
        assert deploy._plugin_version(d) == "unknown"

    def test_check_reports_version_and_inventory(self, tmp_path, capsys):
        src, tgt = tmp_path / "src", tmp_path / "tgt"
        _write(src / "a.py")
        _write(src / "version.py", '__version__ = "1.2.3"\n')
        _write(src / "plugin.yaml", "name: x\nversion: 1.2.3\n")
        tgt.mkdir()
        rc = deploy.check_mode(src, tgt, tmp_path / "state.json")
        out = capsys.readouterr().out
        assert "plugin version: 1.2.3 (repo) / unknown (live)" in out
        assert "module inventory: repo 3, live 0" in out
        assert "missing in live: a.py" in out
        assert rc == 1  # drift present
        rc = deploy.copy_mode(
            src, tgt, tmp_path / "state.json", prune=False, restart=False
        )
        assert rc == 0
        rc = deploy.check_mode(src, tgt, tmp_path / "state.json")
        out = capsys.readouterr().out
        assert "module inventory: repo 3, live 3" in out
        assert "plugin version: 1.2.3 (repo) / 1.2.3 (live)" in out
        assert rc == 0

    def test_state_entry_records_plugin_version(self, tmp_path):
        import json as _json

        src, tgt = tmp_path / "src", tmp_path / "tgt"
        _write(src / "a.py")
        _write(src / "version.py", '__version__ = "4.5.6"\n')
        tgt.mkdir()
        state = tmp_path / "state.json"
        rc = deploy.copy_mode(src, tgt, state, prune=False, restart=False)
        assert rc == 0
        data = _json.loads(state.read_text(encoding="utf-8"))
        entry = data["deployments"][-1]
        assert entry.get("plugin_version") == "4.5.6"

    @pytest.mark.skipif(
        not _live_plugin_dir().is_dir(), reason="no live install on this machine"
    )
    def test_canary_repo_modules_present_in_live(self):
        """#360 acceptance: fails when a repo module is missing from the
        live install (enforced on dev machines; CI has no live install
        and skips)."""
        src_mods = {
            deploy._rel_key(p, _plugin_dir)
            for p in deploy.source_files(_plugin_dir)
        }
        live_mods = {
            deploy._rel_key(p, _live_plugin_dir())
            for p in deploy.source_files(_live_plugin_dir())
        }
        missing = sorted(src_mods - live_mods)
        assert not missing, (
            "repo modules missing from the live install (run "
            "scripts/deploy.py): %s" % missing
        )

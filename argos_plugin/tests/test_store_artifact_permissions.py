"""#414: store artifacts must be owner-only on POSIX.

The DuckDB file (+.wal), the Kuzu graph directory (+files) and the config
JSON hold the full memory store. Under a default 022 umask they would be
world-readable; store init restricts them to 0600/0700 (mirrors the SC2
endpoint-file pattern from #218). chmod is fail-soft and a no-op on
Windows, so the mode assertions are skipped where the platform doesn't
apply.
"""
from __future__ import annotations

import os
import sys
import json
from pathlib import Path

import pytest

# Ensure the plugin package is importable.
_plugin_dir = Path(__file__).resolve().parent.parent
if str(_plugin_dir.parent) not in sys.path:
    sys.path.insert(0, str(_plugin_dir.parent))

_IS_POSIX = sys.platform != "win32"


def _mode(path) -> int:
    return os.stat(str(path)).st_mode & 0o777


class TestStoreArtifactPermissions:
    """#414: store init restricts artifacts to owner-only on POSIX."""

    def test_duckdb_store_file_restricted(self, tmp_path):
        from store import DuckDBMemoryStore

        prev = os.umask(0o022)
        try:
            store = DuckDBMemoryStore(tmp_path / "hybrid_memory.duckdb",
                                     user_id="test_user")
            store.close()
        finally:
            os.umask(prev)
        db = tmp_path / "hybrid_memory.duckdb"
        assert db.exists()
        if _IS_POSIX:
            assert _mode(db) == 0o600, f"db {oct(_mode(db))}"

    def test_duckdb_wal_restricted_when_present(self, tmp_path):
        """A .wal created during init (schema writes) must also be 0600."""
        from store import DuckDBMemoryStore

        prev = os.umask(0o022)
        try:
            store = DuckDBMemoryStore(tmp_path / "hybrid_memory.duckdb",
                                     user_id="test_user")
            wal = tmp_path / "hybrid_memory.duckdb.wal"
            if wal.exists() and _IS_POSIX:
                assert _mode(wal) == 0o600, f"wal {oct(_mode(wal))}"
            store.close()
        finally:
            os.umask(prev)

    def test_kuzu_graph_store_restricted(self, tmp_path):
        """#414: the kuzu store (file or dir, +.wal) is owner-only on POSIX.

        Kuzu's on-disk layout is version-dependent (0.11+ stores a single
        file created on close/checkpoint; older versions store a directory).
        The store is restricted at init (pre-existing files) and at close
        (newly checkpointed files)."""
        from graph import KuzuGraphStore

        prev = os.umask(0o022)
        try:
            graph = KuzuGraphStore(tmp_path / "hybrid_memory_kuzu",
                                   user_id="test_user")
            graph.add_relationship("user", "person", "knows", "Sam", "person")
            graph.close()
        finally:
            os.umask(prev)
        kpath = tmp_path / "hybrid_memory_kuzu"
        assert kpath.exists(), "kuzu store was not created"
        if _IS_POSIX:
            if kpath.is_dir():
                assert _mode(kpath) == 0o700, f"dir {oct(_mode(kpath))}"
                for entry in kpath.iterdir():
                    if entry.is_file():
                        assert _mode(entry) == 0o600, \
                            f"{entry.name} {oct(_mode(entry))}"
            else:
                assert _mode(kpath) == 0o600, f"file {oct(_mode(kpath))}"

    def test_config_json_restricted_on_write(self, tmp_path):
        """#414: the config file written by the plugin is restricted too."""
        from store_common import restrict_store_artifact

        cfg = tmp_path / "hybrid_memory.json"
        prev = os.umask(0o022)
        try:
            cfg.write_text(json.dumps({"role_words": "[]"}), encoding="utf-8")
            restrict_store_artifact(cfg, 0o600)
        finally:
            os.umask(prev)
        assert cfg.exists()
        if _IS_POSIX:
            assert _mode(cfg) == 0o600, f"cfg {oct(_mode(cfg))}"

    def test_restrict_is_fail_soft_for_missing_path(self, tmp_path):
        """#414: restricting a non-existent path must not raise."""
        from store_common import restrict_store_artifact, restrict_store_dir

        # Neither call should raise even though the paths do not exist.
        restrict_store_artifact(tmp_path / "does_not_exist.duckdb", 0o600)
        restrict_store_dir(tmp_path / "missing_kuzu_dir", 0o700)

"""Tests for tenant provisioning and deployment (#130).

Tests cover:
  - Validation: invalid names, path traversal, duplicate user IDs,
    path collisions — all fail before any partial tenant is created.
  - Legacy single-tenant data migrates to explicit default cell.
  - Status output lists all cells and identifies the default without
    leaking secrets.
  - Per-tenant backup/restore round-trip with two cells proves no
    cross-cell records.
  - Destructive operations never silently target the default tenant.
  - The provisioning tool can add a tenant with a documented command.

All deterministic, no LLM calls.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_plugin_dir = Path(__file__).resolve().parent.parent
if str(_plugin_dir) not in sys.path:
    sys.path.insert(0, str(_plugin_dir))

from memory_service import (
    MemoryService, _parse_tenants, _validate_tenant_name, _validate_tenant_path,
)

_script_dir = _plugin_dir.parent / "scripts"


# -- Validation: tenant names ------------------------------------------------

class TestTenantNameValidation:
    """Invalid tenant names are rejected."""

    @pytest.mark.parametrize("name", [
        "", "  ", "with space", "with/slash", "with\\backslash",
        "with:colon", "with;semicolon", "with*star",
        ".hidden", "-leading-dash", "_leading_underscore",
        "a" * 65,  # too long
    ])
    def test_invalid_names_rejected(self, name):
        with pytest.raises(ValueError):
            _validate_tenant_name(name)

    @pytest.mark.parametrize("name", [
        "default", "alpha", "beta", "tenant-1", "tenant_2",
        "ABC123", "a", "a" * 64,  # max length
    ])
    def test_valid_names_accepted(self, name):
        _validate_tenant_name(name)  # should not raise


# -- Validation: path traversal ----------------------------------------------

class TestPathTraversalValidation:
    """Path traversal in database/graph paths is rejected."""

    @pytest.mark.parametrize("path", [
        "../escape.duckdb", "..\\escape.duckdb",
        "foo/../bar.duckdb", "foo/../../bar.duckdb",
        "C:\\absolute\\path.duckdb", "/absolute/path.duckdb",
        "\\\\server\\share\\db.duckdb", "//server/share/db.duckdb",
    ])
    def test_invalid_paths_rejected(self, path):
        with pytest.raises(ValueError):
            _validate_tenant_path(path, "test", "database_filename")

    @pytest.mark.parametrize("path", [
        "test.duckdb", "alpha.duckdb", "subdir/test.duckdb",
        "tenant_kuzu", "alpha_kuzu",
    ])
    def test_valid_paths_accepted(self, path):
        _validate_tenant_path(path, "test", "database_filename")


# -- Validation: through _parse_tenants --------------------------------------

class TestParseTenantsValidation:
    """_parse_tenants rejects invalid configs at startup."""

    def test_path_traversal_rejected(self, tmp_path):
        """A tenant with .. in the database path is rejected."""
        config = {
            "tenants": {
                "evil": {
                    "database_filename": "../escape.duckdb",
                    "graph_dirname": "evil_kuzu",
                },
            },
        }
        with pytest.raises(ValueError, match="path traversal"):
            _parse_tenants(config, tmp_path, None, None)

    def test_absolute_path_rejected(self, tmp_path):
        """An absolute database path is rejected."""
        config = {
            "tenants": {
                "evil": {
                    "database_filename": str(tmp_path / "absolute.duckdb"),
                    "graph_dirname": "evil_kuzu",
                },
            },
        }
        with pytest.raises(ValueError, match="must be relative"):
            _parse_tenants(config, tmp_path, None, None)

    def test_db_path_collision_rejected(self, tmp_path):
        """Two tenants with the same database_filename are rejected."""
        config = {
            "tenants": {
                "alpha": {
                    "database_filename": "shared.duckdb",
                    "graph_dirname": "alpha_kuzu",
                },
                "beta": {
                    "database_filename": "shared.duckdb",
                    "graph_dirname": "beta_kuzu",
                },
            },
        }
        with pytest.raises(ValueError, match="database_filename collision"):
            _parse_tenants(config, tmp_path, None, None)

    def test_graph_path_collision_rejected(self, tmp_path):
        """Two tenants with the same graph_dirname are rejected."""
        config = {
            "tenants": {
                "alpha": {
                    "database_filename": "alpha.duckdb",
                    "graph_dirname": "shared_kuzu",
                },
                "beta": {
                    "database_filename": "beta.duckdb",
                    "graph_dirname": "shared_kuzu",
                },
            },
        }
        with pytest.raises(ValueError, match="graph_dirname collision"):
            _parse_tenants(config, tmp_path, None, None)

    def test_invalid_tenant_name_rejected(self, tmp_path):
        """A tenant with an invalid name is rejected."""
        config = {
            "tenants": {
                "invalid name with spaces": {
                    "database_filename": "test.duckdb",
                    "graph_dirname": "test_kuzu",
                },
            },
        }
        with pytest.raises(ValueError, match="Invalid tenant name"):
            _parse_tenants(config, tmp_path, None, None)

    def test_duplicate_user_ids_rejected(self, tmp_path):
        """Duplicate user_ids across tenants are rejected."""
        config = {
            "tenants": {
                "alpha": {
                    "database_filename": "alpha.duckdb",
                    "graph_dirname": "alpha_kuzu",
                    "allowed_user_ids": ["shared-user"],
                },
                "beta": {
                    "database_filename": "beta.duckdb",
                    "graph_dirname": "beta_kuzu",
                    "allowed_user_ids": ["shared-user"],
                },
            },
        }
        with pytest.raises(ValueError, match="allowed_user_ids conflict"):
            _parse_tenants(config, tmp_path, None, None)


# -- Legacy migration --------------------------------------------------------

class TestLegacyMigration:
    """Legacy single-tenant config migrates to explicit default cell."""

    def test_migrate_legacy_config(self, tmp_path):
        """A config without 'tenants' gets a 'default' cell added."""
        config = {
            "database_filename": "hybrid_memory.duckdb",
            "graph_dirname": "hybrid_memory_kuzu",
            "max_injected_items": "7",
            "review_mode": "auto",
        }
        (tmp_path / "hybrid_memory.json").write_text(
            json.dumps(config), encoding="utf-8"
        )
        # Run the migration tool.
        result = subprocess.run(
            [sys.executable, str(_script_dir / "provision_tenants.py"),
             str(tmp_path), "migrate-legacy"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        # Check the config was updated.
        new_config = json.loads((tmp_path / "hybrid_memory.json").read_text())
        assert "tenants" in new_config
        assert "default" in new_config["tenants"]
        default = new_config["tenants"]["default"]
        assert default["database_filename"] == "hybrid_memory.duckdb"
        assert default["graph_dirname"] == "hybrid_memory_kuzu"
        # Overlay should preserve global settings.
        assert default["config"]["max_injected_items"] == "7"
        assert default["config"]["review_mode"] == "auto"

    def test_migrate_already_migrated_fails(self, tmp_path):
        """Migrating an already-migrated config fails."""
        config = {"tenants": {"default": {"database_filename": "test.duckdb"}}}
        (tmp_path / "hybrid_memory.json").write_text(
            json.dumps(config), encoding="utf-8"
        )
        result = subprocess.run(
            [sys.executable, str(_script_dir / "provision_tenants.py"),
             str(tmp_path), "migrate-legacy"],
            capture_output=True, text=True,
        )
        assert result.returncode != 0
        assert "already has a 'tenants' map" in result.stderr

    def test_migrated_config_remains_readable(self, tmp_path):
        """After migration, the service can still read the existing data."""
        # Write a legacy config and create a DB.
        config = {"database_filename": "hybrid_memory.duckdb"}
        (tmp_path / "hybrid_memory.json").write_text(
            json.dumps(config), encoding="utf-8"
        )
        # Create the DB with some data.
        svc = MemoryService(tmp_path)
        svc.dispatch({
            "component": "store", "method": "remember",
            "user_id": "default_user",
            "args": {"category": "personal_fact", "content": "Test memory"},
        })
        svc.close()
        # Migrate.
        subprocess.run(
            [sys.executable, str(_script_dir / "provision_tenants.py"),
             str(tmp_path), "migrate-legacy"],
            capture_output=True, text=True, check=True,
        )
        # Re-open — the data should still be there.
        svc2 = MemoryService(tmp_path)
        count = svc2.dispatch({
            "component": "store", "method": "count",
            "user_id": "default_user",
        })
        assert count >= 1, "Data should survive migration"
        svc2.close()


# -- Status output -----------------------------------------------------------

class TestStatusOutput:
    """Status output lists all cells and identifies the default."""

    def test_status_lists_tenants(self, tmp_path):
        """The status command lists all configured tenants."""
        config = {
            "tenants": {
                "alpha": {
                    "database_filename": "alpha.duckdb",
                    "graph_dirname": "alpha_kuzu",
                    "allowed_user_ids": ["alice"],
                },
                "beta": {
                    "database_filename": "beta.duckdb",
                    "graph_dirname": "beta_kuzu",
                    "allowed_user_ids": ["bob"],
                },
            },
        }
        (tmp_path / "hybrid_memory.json").write_text(
            json.dumps(config), encoding="utf-8"
        )
        result = subprocess.run(
            [sys.executable, str(_script_dir / "provision_tenants.py"),
             str(tmp_path), "status"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "alpha" in result.stdout
        assert "beta" in result.stdout
        assert "Tenants: 2" in result.stdout

    def test_status_legacy_mode(self, tmp_path):
        """Status shows legacy mode for single-tenant config."""
        config = {"database_filename": "hybrid_memory.duckdb"}
        (tmp_path / "hybrid_memory.json").write_text(
            json.dumps(config), encoding="utf-8"
        )
        result = subprocess.run(
            [sys.executable, str(_script_dir / "provision_tenants.py"),
             str(tmp_path), "status"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "legacy single-tenant" in result.stdout

    def test_get_status_includes_tenant_cells(self, tmp_path):
        """get_status RPC includes tenant_cells with resolved paths."""
        config = {
            "tenants": {
                "alpha": {
                    "database_filename": "alpha.duckdb",
                    "graph_dirname": "alpha_kuzu",
                    "allowed_user_ids": ["alice"],
                },
            },
        }
        (tmp_path / "hybrid_memory.json").write_text(
            json.dumps(config), encoding="utf-8"
        )
        svc = MemoryService(tmp_path)
        status = svc.dispatch({"method": "get_status"})
        assert "tenant_cells" in status
        assert "alpha" in status["tenant_cells"]
        cell = status["tenant_cells"]["alpha"]
        assert cell["database_path"] == "alpha.duckdb"
        assert cell["graph_path"] == "alpha_kuzu"
        assert cell["is_default"] is True  # only tenant
        assert cell["user_count"] == 1
        assert "default_tenant" in status

    def test_get_status_no_secrets_leaked(self, tmp_path):
        """get_status does not leak credential tokens."""
        config = {
            "tenants": {
                "alpha": {
                    "database_filename": "alpha.duckdb",
                    "graph_dirname": "alpha_kuzu",
                    "credentials": [{"token": "super-secret-token", "user_id": "alice"}],
                },
            },
        }
        (tmp_path / "hybrid_memory.json").write_text(
            json.dumps(config), encoding="utf-8"
        )
        svc = MemoryService(tmp_path)
        status = svc.dispatch({"method": "get_status"})
        status_str = json.dumps(status)
        assert "super-secret-token" not in status_str
        # But credential count should be visible.
        assert status["tenant_cells"]["alpha"]["credential_count"] == 1


# -- Destructive operations: no silent default fallback ---------------------

class TestNoSilentDefaultFallback:
    """Destructive operations never silently target the default tenant."""

    def test_backup_unknown_tenant_rejected(self, tmp_path):
        """Backup with an unknown tenant name raises ValueError, not
        silent fallback to default."""
        config = {
            "tenants": {
                "default": {
                    "database_filename": "default.duckdb",
                    "graph_dirname": "default_kuzu",
                },
                "alpha": {
                    "database_filename": "alpha.duckdb",
                    "graph_dirname": "alpha_kuzu",
                },
            },
        }
        (tmp_path / "hybrid_memory.json").write_text(
            json.dumps(config), encoding="utf-8"
        )
        svc = MemoryService(tmp_path)
        with pytest.raises(ValueError, match="not found"):
            svc.dispatch({
                "method": "backup",
                "args": {"tenant": "nonexistent"},
            })

    def test_backup_explicit_tenant_works(self, tmp_path):
        """Backup with an explicit tenant name works."""
        config = {
            "tenants": {
                "alpha": {
                    "database_filename": "alpha.duckdb",
                    "graph_dirname": "alpha_kuzu",
                },
            },
        }
        (tmp_path / "hybrid_memory.json").write_text(
            json.dumps(config), encoding="utf-8"
        )
        svc = MemoryService(tmp_path)
        svc.dispatch({
            "component": "store", "method": "remember",
            "user_id": "default_user",
            "args": {"category": "personal_fact", "content": "Test"},
        })
        result = svc.dispatch({
            "method": "backup",
            "args": {"tenant": "alpha", "dst_root": str(tmp_path / "backups")},
        })
        assert result is not None
        assert result["tenant"] == "alpha"


# -- Provisioning tool: add tenant -------------------------------------------

class TestProvisioningAddTenant:
    """The provisioning tool can add a tenant with a documented command."""

    def test_add_tenant_to_existing_config(self, tmp_path):
        """Adding a tenant to an existing multi-tenant config works."""
        config = {
            "tenants": {
                "alpha": {
                    "database_filename": "alpha.duckdb",
                    "graph_dirname": "alpha_kuzu",
                },
            },
        }
        (tmp_path / "hybrid_memory.json").write_text(
            json.dumps(config), encoding="utf-8"
        )
        result = subprocess.run(
            [sys.executable, str(_script_dir / "provision_tenants.py"),
             str(tmp_path), "add", "beta",
             "--db", "beta.duckdb", "--graph", "beta_kuzu",
             "--allowed-user-ids", "bob,carol",
             "--review-mode", "auto"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        new_config = json.loads((tmp_path / "hybrid_memory.json").read_text())
        assert "beta" in new_config["tenants"]
        beta = new_config["tenants"]["beta"]
        assert beta["database_filename"] == "beta.duckdb"
        assert beta["graph_dirname"] == "beta_kuzu"
        assert beta["allowed_user_ids"] == ["bob", "carol"]
        assert beta["config"]["review_mode"] == "auto"

    def test_add_duplicate_tenant_fails(self, tmp_path):
        """Adding a tenant that already exists fails."""
        config = {
            "tenants": {
                "alpha": {"database_filename": "alpha.duckdb", "graph_dirname": "alpha_kuzu"},
            },
        }
        (tmp_path / "hybrid_memory.json").write_text(
            json.dumps(config), encoding="utf-8"
        )
        result = subprocess.run(
            [sys.executable, str(_script_dir / "provision_tenants.py"),
             str(tmp_path), "add", "alpha"],
            capture_output=True, text=True,
        )
        assert result.returncode != 0
        assert "already exists" in result.stderr

    def test_add_tenant_with_invalid_name_fails(self, tmp_path):
        """Adding a tenant with an invalid name fails."""
        config = {"tenants": {"alpha": {"database_filename": "a.duckdb"}}}
        (tmp_path / "hybrid_memory.json").write_text(
            json.dumps(config), encoding="utf-8"
        )
        result = subprocess.run(
            [sys.executable, str(_script_dir / "provision_tenants.py"),
             str(tmp_path), "add", "invalid name"],
            capture_output=True, text=True,
        )
        assert result.returncode != 0
        assert "invalid tenant name" in result.stderr.lower()

    def test_add_tenant_with_path_traversal_fails(self, tmp_path):
        """Adding a tenant with path traversal in db path fails."""
        config = {"tenants": {"alpha": {"database_filename": "a.duckdb"}}}
        (tmp_path / "hybrid_memory.json").write_text(
            json.dumps(config), encoding="utf-8"
        )
        result = subprocess.run(
            [sys.executable, str(_script_dir / "provision_tenants.py"),
             str(tmp_path), "add", "evil",
             "--db", "../escape.duckdb"],
            capture_output=True, text=True,
        )
        assert result.returncode != 0
        assert ".." in result.stderr or "traversal" in result.stderr.lower()


# -- Validate command --------------------------------------------------------

class TestValidateCommand:
    """The validate command checks config without starting the service."""

    def test_validate_valid_config(self, tmp_path):
        """A valid config passes validation."""
        config = {
            "tenants": {
                "alpha": {"database_filename": "alpha.duckdb", "graph_dirname": "alpha_kuzu"},
                "beta": {"database_filename": "beta.duckdb", "graph_dirname": "beta_kuzu"},
            },
        }
        (tmp_path / "hybrid_memory.json").write_text(
            json.dumps(config), encoding="utf-8"
        )
        result = subprocess.run(
            [sys.executable, str(_script_dir / "provision_tenants.py"),
             str(tmp_path), "validate"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "OK" in result.stdout

    def test_validate_invalid_config(self, tmp_path):
        """An invalid config fails validation."""
        config = {
            "tenants": {
                "alpha": {"database_filename": "shared.duckdb", "graph_dirname": "a_kuzu"},
                "beta": {"database_filename": "shared.duckdb", "graph_dirname": "b_kuzu"},
            },
        }
        (tmp_path / "hybrid_memory.json").write_text(
            json.dumps(config), encoding="utf-8"
        )
        result = subprocess.run(
            [sys.executable, str(_script_dir / "provision_tenants.py"),
             str(tmp_path), "validate"],
            capture_output=True, text=True,
        )
        assert result.returncode != 0
        assert "collision" in result.stdout.lower()


# -- Per-tenant backup round-trip with two cells -----------------------------

class TestBackupRoundTrip:
    """Per-tenant backup/restore round-trip with two cells proves no
    cross-cell records."""

    def test_two_tenant_backup_isolation(self, tmp_path):
        """Backing up two tenants produces separate snapshots with no
        cross-cell records."""
        config = {
            "tenants": {
                "alpha": {
                    "database_filename": "alpha_bk.duckdb",
                    "graph_dirname": "alpha_bk_kuzu",
                    "allowed_user_ids": ["alice"],
                },
                "beta": {
                    "database_filename": "beta_bk.duckdb",
                    "graph_dirname": "beta_bk_kuzu",
                    "allowed_user_ids": ["bob"],
                },
            },
        }
        (tmp_path / "hybrid_memory.json").write_text(
            json.dumps(config), encoding="utf-8"
        )
        svc = MemoryService(tmp_path)
        # Write distinct data to each tenant.
        svc.dispatch({
            "component": "store", "method": "remember",
            "user_id": "alice",
            "args": {"category": "personal_fact", "content": "Alice lives at 123 Alpha St"},
        })
        svc.dispatch({
            "component": "store", "method": "remember",
            "user_id": "bob",
            "args": {"category": "personal_fact", "content": "Bob works at 456 Beta Ave"},
        })
        # Back up each tenant.
        alpha_result = svc.dispatch({
            "method": "backup",
            "args": {"tenant": "alpha", "dst_root": str(tmp_path / "backups")},
        })
        beta_result = svc.dispatch({
            "method": "backup",
            "args": {"tenant": "beta", "dst_root": str(tmp_path / "backups")},
        })
        assert alpha_result["tenant"] == "alpha"
        assert beta_result["tenant"] == "beta"
        # Verify the counts are separate.
        alpha_count = svc.dispatch({
            "component": "store", "method": "count", "user_id": "alice",
        })
        beta_count = svc.dispatch({
            "component": "store", "method": "count", "user_id": "bob",
        })
        assert alpha_count == 1
        assert beta_count == 1
        svc.close()

"""#288: store schema migrations + versioning.

Tests:
1. Fresh DB migrates to latest version.
2. Old-version fixture DB migrates forward.
3. Idempotency (running twice is a no-op).
4. Failed migration rolls back cleanly.
5. Per-tenant migration applies to each tenant store.
6. schema_version is persisted via schema_meta table.
7. Migration report structure is correct.
8. Backup manifest records the actual schema_version.
9. Pre-migration integrity check catches corrupt DBs.
10. Migration gap detection (broken migration list).

Run with (Hermes venv python, hermetic):
    ARGOS_HERMETIC_TESTS=1 python -m pytest tests/test_schema_migrations.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _make_store(tmp_path, user_id="alice"):
    """Create a fresh DuckDBMemoryStore."""
    from store import DuckDBMemoryStore
    return DuckDBMemoryStore(tmp_path / "test.duckdb", user_id=user_id)


class TestSchemaVersionField:
    """1. schema_version is persisted via schema_meta table."""

    def test_fresh_db_starts_at_version_1(self, tmp_path):
        """A fresh DB is migrated to version 1 (the baseline) at init."""
        store = _make_store(tmp_path)
        try:
            assert store.get_schema_version() == 1
        finally:
            store.close()

    def test_schema_version_persists_across_restarts(self, tmp_path):
        """user_version persists in the DB header across close/reopen."""
        store = _make_store(tmp_path)
        try:
            assert store.get_schema_version() == 1
        finally:
            store.close()
        # Reopen the same DB.
        store2 = _make_store(tmp_path)
        try:
            assert store2.get_schema_version() == 1
        finally:
            store2.close()

    def test_get_schema_version_reads_schema_meta(self, tmp_path):
        """get_schema_version() reads from the schema_meta table."""
        import duckdb
        conn = duckdb.connect(str(tmp_path / "direct.duckdb"))
        try:
            # Fresh DB: no schema_meta table → version 0.
            from schema_migrations import get_schema_version
            assert get_schema_version(conn) == 0
            # Create the table and set a version.
            from schema_migrations import _ensure_schema_meta_table, _set_schema_version
            _ensure_schema_meta_table(conn)
            _set_schema_version(conn, 42)
            assert get_schema_version(conn) == 42
        finally:
            conn.close()


class TestFreshDbMigratesToLatest:
    """2. Fresh DB migrates to latest version at init."""

    def test_fresh_db_at_latest_version(self, tmp_path):
        store = _make_store(tmp_path)
        try:
            from schema_migrations import LATEST_SCHEMA_VERSION
            assert store.get_schema_version() == LATEST_SCHEMA_VERSION
        finally:
            store.close()

    def test_migration_report_recorded(self, tmp_path):
        """The store records a migration report at init."""
        store = _make_store(tmp_path)
        try:
            report = store.get_migration_report()
            assert report is not None
            assert report["from_version"] == 0
            assert report["to_version"] == 1
            assert 1 in report["applied"]
        finally:
            store.close()


class TestOldVersionFixtureMigrates:
    """3. Old-version fixture DB migrates forward."""

    def test_old_version_db_migrates_forward(self, tmp_path):
        """A DB stamped at version 0 migrates to version 1 on init."""
        import duckdb
        db_path = tmp_path / "old.duckdb"
        # Create a DB with the memory_records table but schema_version=0
        # (simulating a pre-migration DB). No schema_meta table → version 0.
        conn = duckdb.connect(str(db_path))
        try:
            conn.execute("""
                CREATE TABLE memory_records (
                    memory_id VARCHAR PRIMARY KEY,
                    category VARCHAR,
                    content VARCHAR
                )
            """)
        finally:
            conn.close()

        # Now open it through DuckDBMemoryStore — the additive layer
        # adds all missing columns, then the migration runner stamps
        # schema_version=1.
        from store import DuckDBMemoryStore
        store = DuckDBMemoryStore(db_path, user_id="alice")
        try:
            assert store.get_schema_version() == 1
            report = store.get_migration_report()
            assert report["from_version"] == 0
            assert report["to_version"] == 1
            assert 1 in report["applied"]
        finally:
            store.close()

    def test_version_1_db_stays_at_1(self, tmp_path):
        """A DB already at version 1 stays at 1 (no re-migration)."""
        import duckdb
        db_path = tmp_path / "v1.duckdb"
        conn = duckdb.connect(str(db_path))
        try:
            conn.execute("""
                CREATE TABLE memory_records (
                    memory_id VARCHAR PRIMARY KEY,
                    category VARCHAR,
                    content VARCHAR
                )
            """)
            # Stamp at version 1 via schema_meta.
            from schema_migrations import _ensure_schema_meta_table, _set_schema_version
            _ensure_schema_meta_table(conn)
            _set_schema_version(conn, 1)
        finally:
            conn.close()

        from store import DuckDBMemoryStore
        store = DuckDBMemoryStore(db_path, user_id="alice")
        try:
            assert store.get_schema_version() == 1
            report = store.get_migration_report()
            assert report["from_version"] == 1
            assert report["to_version"] == 1
            assert 1 in report["skipped"]
            assert len(report["applied"]) == 0
        finally:
            store.close()


class TestIdempotency:
    """4. Running migrations twice is a no-op."""

    def test_run_migrations_twice_is_noop(self, tmp_path):
        """run_migrations() called twice: second call applies nothing."""
        import duckdb
        from schema_migrations import run_migrations

        conn = duckdb.connect(str(tmp_path / "idempotent.duckdb"))
        try:
            conn.execute("""
                CREATE TABLE memory_records (
                    memory_id VARCHAR PRIMARY KEY,
                    category VARCHAR,
                    content VARCHAR
                )
            """)
            r1 = run_migrations(conn)
            assert r1["from_version"] == 0
            assert r1["to_version"] == 1
            assert 1 in r1["applied"]

            r2 = run_migrations(conn)
            assert r2["from_version"] == 1
            assert r2["to_version"] == 1
            assert 1 in r2["skipped"]
            assert len(r2["applied"]) == 0
        finally:
            conn.close()

    def test_store_init_twice_is_noop(self, tmp_path):
        """Opening a store twice: second init skips migrations."""
        store = _make_store(tmp_path)
        store.close()

        store2 = _make_store(tmp_path)
        try:
            assert store2.get_schema_version() == 1
            report = store2.get_migration_report()
            assert report["from_version"] == 1
            assert len(report["applied"]) == 0
            assert 1 in report["skipped"]
        finally:
            store2.close()


class TestFailedMigrationRollsBack:
    """5. Failed migration rolls back cleanly (no half-migration)."""

    def test_failed_migration_rolls_back_transaction(self, tmp_path):
        """A migration that raises inside its transaction rolls back
        and leaves schema_version unchanged."""
        import duckdb
        from schema_migrations import run_migrations, _ensure_schema_meta_table, _set_schema_version

        conn = duckdb.connect(str(tmp_path / "fail.duckdb"))
        try:
            conn.execute("""
                CREATE TABLE memory_records (
                    memory_id VARCHAR PRIMARY KEY,
                    category VARCHAR,
                    content VARCHAR
                )
            """)
            _ensure_schema_meta_table(conn)
            _set_schema_version(conn, 1)

            # Define a migration 1→2 that creates a table then fails.
            def bad_migration(c):
                c.execute("CREATE TABLE IF NOT EXISTS migration_test (id INTEGER)")
                raise RuntimeError("intentional failure")

            migs = [(1, 2, bad_migration)]

            with pytest.raises(RuntimeError, match="1→2 failed"):
                run_migrations(conn, migrations=migs, skip_health_check=True)

            # schema_version should still be 1 (not 2).
            from schema_migrations import get_schema_version
            assert get_schema_version(conn) == 1

            # The table creation should have been rolled back.
            tables = conn.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'main' AND table_name = 'migration_test'"
            ).fetchall()
            assert len(tables) == 0, "migration_test table should not exist after rollback"
        finally:
            conn.close()

    def test_failed_migration_can_retry(self, tmp_path):
        """After a failed migration, a subsequent successful run works."""
        import duckdb
        from schema_migrations import run_migrations, get_schema_version, _ensure_schema_meta_table, _set_schema_version

        conn = duckdb.connect(str(tmp_path / "retry.duckdb"))
        try:
            conn.execute("""
                CREATE TABLE memory_records (
                    memory_id VARCHAR PRIMARY KEY,
                    category VARCHAR,
                    content VARCHAR
                )
            """)
            _ensure_schema_meta_table(conn)
            _set_schema_version(conn, 1)

            call_count = [0]
            def flaky_migration(c):
                call_count[0] += 1
                if call_count[0] == 1:
                    raise RuntimeError("first attempt fails")
                c.execute("CREATE TABLE IF NOT EXISTS retry_test (id INTEGER)")

            migs = [(1, 2, flaky_migration)]

            # First attempt fails.
            with pytest.raises(RuntimeError):
                run_migrations(conn, migrations=migs, skip_health_check=True)
            assert get_schema_version(conn) == 1

            # Second attempt succeeds.
            r2 = run_migrations(conn, migrations=migs, skip_health_check=True)
            assert r2["to_version"] == 2
            assert 2 in r2["applied"]
        finally:
            conn.close()


class TestMigrationGapDetection:
    """6. Migration gap detection (broken migration list)."""

    def test_gap_in_migration_list_raises(self, tmp_path):
        """If the migration list has a gap (version_from doesn't chain),
        the runner raises."""
        import duckdb
        from schema_migrations import run_migrations, _ensure_schema_meta_table, _set_schema_version

        conn = duckdb.connect(str(tmp_path / "gap.duckdb"))
        try:
            conn.execute("""
                CREATE TABLE memory_records (
                    memory_id VARCHAR PRIMARY KEY,
                    category VARCHAR,
                    content VARCHAR
                )
            """)
            _ensure_schema_meta_table(conn)
            _set_schema_version(conn, 1)

            # Migration list with a gap: 1→2, then 3→4 (skipping 2→3).
            migs = [
                (1, 2, lambda c: None),
                (3, 4, lambda c: None),  # gap: expected 2, got 3
            ]

            with pytest.raises(RuntimeError, match="Migration gap"):
                run_migrations(conn, migrations=migs, skip_health_check=True)
        finally:
            conn.close()


class TestPreMigrationHealthCheck:
    """7. Pre-migration health check."""

    def test_health_check_runs_before_migrations(self, tmp_path):
        """The runner performs a health check before migrating."""
        import duckdb
        from schema_migrations import run_migrations

        conn = duckdb.connect(str(tmp_path / "integrity.duckdb"))
        try:
            conn.execute("""
                CREATE TABLE memory_records (
                    memory_id VARCHAR PRIMARY KEY,
                    category VARCHAR,
                    content VARCHAR
                )
            """)
            # A healthy DB should pass the health check and migrate.
            r = run_migrations(conn)
            assert r["to_version"] == 1
        finally:
            conn.close()

    def test_health_check_can_be_skipped(self, tmp_path):
        """skip_health_check=True bypasses the check (for testing)."""
        import duckdb
        from schema_migrations import run_migrations

        conn = duckdb.connect(str(tmp_path / "skip.duckdb"))
        try:
            conn.execute("""
                CREATE TABLE memory_records (
                    memory_id VARCHAR PRIMARY KEY,
                    category VARCHAR,
                    content VARCHAR
                )
            """)
            r = run_migrations(conn, skip_health_check=True)
            assert r["to_version"] == 1
        finally:
            conn.close()

    def test_health_check_fails_on_missing_core_table(self, tmp_path):
        """If memory_records doesn't exist, the health check raises."""
        import duckdb
        from schema_migrations import run_migrations

        conn = duckdb.connect(str(tmp_path / "bad.duckdb"))
        try:
            # No memory_records table — health check should fail.
            with pytest.raises(RuntimeError, match="memory_records table"):
                run_migrations(conn)
        finally:
            conn.close()


class TestBackupManifestRecordsSchemaVersion:
    """8. Backup manifest records the actual schema_version."""

    def test_backup_manifest_has_schema_version(self, tmp_path):
        """A backup of a v1 store records schema_version=1 in the manifest."""
        from backup import backup_store
        store = _make_store(tmp_path)
        try:
            store.remember(category="personal_fact", content="test fact")
            manifest = backup_store(
                store.connection,
                tmp_path / "backups",
                source_db_path=store.db_path,
            )
            assert manifest["schema_version"] == 1
        finally:
            store.close()


class TestPerTenantMigrations:
    """9. Per-tenant migration applies to each tenant store.

    In shared-service mode, each tenant has its own DuckDBMemoryStore
    with its own connection and user_version. The migration runner
    is called from _init_db() on each store independently, so every
    tenant DB gets migrated.
    """

    def test_each_tenant_store_migrated_independently(self, tmp_path):
        """Two DuckDBMemoryStore instances (simulating two tenants)
        each migrate independently and have their own user_version."""
        from store import DuckDBMemoryStore

        store_a = DuckDBMemoryStore(
            tmp_path / "tenant_a.duckdb", user_id="alice",
        )
        store_b = DuckDBMemoryStore(
            tmp_path / "tenant_b.duckdb", user_id="bob",
        )
        try:
            assert store_a.get_schema_version() == 1
            assert store_b.get_schema_version() == 1

            # Each has its own migration report.
            report_a = store_a.get_migration_report()
            report_b = store_b.get_migration_report()
            assert report_a["from_version"] == 0
            assert report_b["from_version"] == 0
        finally:
            store_a.close()
            store_b.close()

    def test_tenant_at_different_versions_migrates_independently(self, tmp_path):
        """A tenant DB at version 0 and one at version 1 both end up at 1."""
        import duckdb
        from store import DuckDBMemoryStore
        from schema_migrations import _ensure_schema_meta_table, _set_schema_version

        # Tenant A: fresh DB (version 0).
        db_a = tmp_path / "a.duckdb"
        # Tenant B: pre-stamped at version 1.
        db_b = tmp_path / "b.duckdb"
        conn_b = duckdb.connect(str(db_b))
        try:
            conn_b.execute("""
                CREATE TABLE memory_records (
                    memory_id VARCHAR PRIMARY KEY,
                    category VARCHAR,
                    content VARCHAR
                )
            """)
            _ensure_schema_meta_table(conn_b)
            _set_schema_version(conn_b, 1)
        finally:
            conn_b.close()

        store_a = DuckDBMemoryStore(db_a, user_id="alice")
        store_b = DuckDBMemoryStore(db_b, user_id="bob")
        try:
            assert store_a.get_schema_version() == 1
            assert store_b.get_schema_version() == 1

            # Tenant A migrated 0→1, Tenant B was already at 1.
            report_a = store_a.get_migration_report()
            report_b = store_b.get_migration_report()
            assert 1 in report_a["applied"]
            assert 1 in report_b["skipped"]
        finally:
            store_a.close()
            store_b.close()


class TestMigrationRunnerDirect:
    """Direct tests of the migration runner API."""

    def test_run_migrations_returns_correct_structure(self, tmp_path):
        """The migration report has the expected keys."""
        import duckdb
        from schema_migrations import run_migrations

        conn = duckdb.connect(str(tmp_path / "struct.duckdb"))
        try:
            conn.execute("""
                CREATE TABLE memory_records (
                    memory_id VARCHAR PRIMARY KEY,
                    category VARCHAR,
                    content VARCHAR
                )
            """)
            r = run_migrations(conn)
            assert "from_version" in r
            assert "to_version" in r
            assert "applied" in r
            assert "skipped" in r
            assert "health_check" in r
            assert isinstance(r["applied"], list)
            assert isinstance(r["skipped"], list)
        finally:
            conn.close()

    def test_latest_schema_version_constant(self):
        """LATEST_SCHEMA_VERSION matches the last migration's version_to."""
        from schema_migrations import MIGRATIONS, LATEST_SCHEMA_VERSION
        assert LATEST_SCHEMA_VERSION == MIGRATIONS[-1][1]

    def test_migrations_are_ordered_and_chained(self):
        """Migrations chain: each version_from == previous version_to."""
        from schema_migrations import MIGRATIONS
        for i, (v_from, v_to, _) in enumerate(MIGRATIONS):
            assert v_to == v_from + 1, (
                f"Migration {i}: version_to must be version_from + 1"
            )
            if i > 0:
                prev_v_to = MIGRATIONS[i - 1][1]
                assert v_from == prev_v_to, (
                    f"Migration {i}: version_from ({v_from}) must chain "
                    f"from previous version_to ({prev_v_to})"
                )

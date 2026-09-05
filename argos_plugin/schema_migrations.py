"""#288: Store schema migrations + versioning.

Ordered, idempotent, transactional migration runner for the DuckDB
memory store. Runs AFTER the additive layer (ALTER TABLE ADD COLUMN IF
NOT EXISTS) at store init time.

Design:
  - schema_version is persisted in a ``schema_meta`` table (single row,
    key='schema_version', value=integer). This is portable across
    DuckDB versions and doesn't depend on SQLite-specific PRAGMAs
    (DuckDB does not support PRAGMA user_version).
  - Each migration is (version_from, version_to, apply_fn). The runner
    applies migrations in order, skipping already-applied versions.
  - Each migration runs inside a transaction (BEGIN/COMMIT). A failure
    rolls back and raises loudly — the store is NOT left half-migrated.
  - Before running migrations, the schema_meta table is created if it
    doesn't exist (zero-migration). On failure, the pre-migration state
    is preserved (the transaction rolled back; schema_version is only
    bumped after a successful COMMIT).
  - Per-tenant: the runner is called on each tenant store's connection
    independently (the shared service creates one DuckDBMemoryStore per
    tenant cell, each with its own connection and schema_meta row).

Version history:
  0 → 1: baseline. The additive layer already brought every existing
         DB to the v1 schema (all columns, indexes, backfills). This
         migration is a no-op that just stamps schema_version=1 so the
         runner knows the baseline is established. Future migrations
         (#285 dimension-generic vectors, etc.) will be 1 → 2, 2 → 3, …

Scope guard:
  - Does NOT change the additive layer's behavior (it keeps running
    first, as always).
  - Does NOT touch ranking/relevance/injection.
  - The runner is ready for data-transform migrations (not just
    additive) — each migration fn receives the connection and can run
    arbitrary SQL inside its transaction.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, List, Tuple

logger = logging.getLogger(__name__)

# -- Migration type ---------------------------------------------------------
# A migration is (version_from, version_to, apply_fn).
# apply_fn(conn) runs arbitrary SQL inside a transaction managed by the
# runner. It must be idempotent within its own version range (the runner
# guards against double-application via schema_version, but the fn
# itself should not assume it's the only thing that ever touched the
# schema).
Migration = Tuple[int, int, Callable[[Any], None]]

# -- Ordered migration list -------------------------------------------------
# Append new migrations here. Each must bump version_to by exactly 1
# and chain from the previous version_to.

def _migration_0_to_1(conn) -> None:
    """Baseline stamp: v1 = the additive layer has already run.

    The additive layer (ALTER TABLE ADD COLUMN IF NOT EXISTS, CREATE
    TABLE IF NOT EXISTS, backfill UPDATEs) in store_core._init_db()
    already brings every DB to the v1 schema. This migration is a
    deliberate no-op that exists only so the runner can stamp
    schema_version=1 and know the baseline is established.

    Future migrations (1→2, 2→3, …) will do real work here.
    """
    # No schema changes — the additive layer already did them.
    pass


MIGRATIONS: List[Migration] = [
    (0, 1, _migration_0_to_1),
]

# The latest schema version = the last migration's version_to.
LATEST_SCHEMA_VERSION = MIGRATIONS[-1][1] if MIGRATIONS else 0

# -- schema_meta table ------------------------------------------------------
# Stores the persisted schema version. Created by the runner before
# reading/writing. Single row: key='schema_version', value=<int>.
_SCHEMA_META_DDL = """
    CREATE TABLE IF NOT EXISTS schema_meta (
        key   VARCHAR PRIMARY KEY,
        value VARCHAR
    )
"""


def _ensure_schema_meta_table(conn) -> None:
    """Create the schema_meta table if it doesn't exist."""
    conn.execute(_SCHEMA_META_DDL)


def get_schema_version(conn) -> int:
    """Read the persisted schema version from the ``schema_meta`` table.

    Returns 0 for a fresh DB (no schema_meta table or no row). The
    value persists across restarts in the DB.
    """
    try:
        # Check if schema_meta exists before querying — avoids errors
        # on fresh DBs where the table hasn't been created yet.
        row = conn.execute(
            "SELECT value FROM schema_meta WHERE key = 'schema_version'"
        ).fetchone()
        return int(row[0]) if row and row[0] is not None else 0
    except Exception:
        # Table doesn't exist yet — fresh DB, version 0.
        return 0


def _set_schema_version(conn, version: int) -> None:
    """Persist the schema version in the ``schema_meta`` table (upsert).

    The schema_meta table must already exist (call
    ``_ensure_schema_meta_table`` first).
    """
    conn.execute(
        """INSERT INTO schema_meta (key, value) VALUES ('schema_version', ?)
           ON CONFLICT(key) DO UPDATE SET value = excluded.value""",
        [str(int(version))],
    )


def _verify_db_health(conn) -> None:
    """Quick health check before running migrations.

    Verifies that the core ``memory_records`` table exists and is
    queryable. If the DB is corrupt or the core schema is missing, we
    must NOT run migrations (they'd fail confusingly or make things
    worse). Raises RuntimeError on failure.
    """
    try:
        rows = conn.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'main' AND table_name = 'memory_records'"
        ).fetchall()
        if not rows:
            raise RuntimeError(
                "Pre-migration health check failed: memory_records table "
                "does not exist. The additive layer should have created it."
            )
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(
            f"Pre-migration health check errored: {exc}"
        ) from exc


def run_migrations(
    conn,
    *,
    migrations: List[Migration] | None = None,
    skip_health_check: bool = False,
) -> dict:
    """Run ordered migrations on *conn*.

    Called from ``store_core._init_db()`` AFTER the additive layer.
    Also called per-tenant in shared-service mode (each tenant store
    has its own connection + schema_meta row).

    Idempotent: if schema_version >= a migration's version_to, that
    migration is skipped. Running twice is a no-op.

    Transactional: each migration runs inside BEGIN/COMMIT. A failure
    rolls back and raises — the store is NOT left half-migrated.
    schema_version is only bumped AFTER a successful COMMIT, so a
    crashed migration will retry cleanly on next init.

    Args:
        conn: open DuckDB connection (the store's own connection).
        migrations: override the migration list (for testing).
        skip_health_check: skip the pre-migration health check
            (for testing with fixture DBs that may not have the full
            schema).

    Returns:
        dict with: from_version, to_version, applied (list of
        version_to ints that were run), skipped (list of version_to
        ints that were already applied), health_check (bool).
    """
    migs = migrations if migrations is not None else MIGRATIONS

    # Ensure the schema_meta table exists before reading from it.
    _ensure_schema_meta_table(conn)

    from_version = get_schema_version(conn)
    applied: List[int] = []
    skipped: List[int] = []
    health_check_ok = True

    # Pre-migration health check.
    if not skip_health_check:
        try:
            _verify_db_health(conn)
        except RuntimeError as exc:
            logger.error("Pre-migration health check failed: %s", exc)
            # Fail loudly — do NOT run migrations on a suspect DB.
            raise

    logger.info(
        "Schema migration: starting from version %d, target %d",
        from_version, LATEST_SCHEMA_VERSION,
    )

    current = from_version
    for v_from, v_to, apply_fn in migs:
        if v_to <= current:
            skipped.append(v_to)
            continue
        if v_from != current:
            raise RuntimeError(
                f"Migration gap: expected version_from={current} "
                f"but migration says version_from={v_from} "
                f"(version_to={v_to}). The migration list is broken "
                f"or a migration was inserted out of order."
            )
        logger.info("Schema migration: applying %d → %d", v_from, v_to)
        # Each migration runs inside a transaction. A failure rolls
        # back and raises — the store is NOT left half-migrated.
        conn.execute("BEGIN TRANSACTION")
        try:
            apply_fn(conn)
            # Bump schema_version INSIDE the transaction so a crash
            # between the schema work and COMMIT doesn't lose the
            # version stamp. If COMMIT fails, the schema changes AND
            # the version bump are both rolled back — the next init
            # will re-run the migration (idempotent fn).
            _set_schema_version(conn, v_to)
            conn.execute("COMMIT")
            applied.append(v_to)
            current = v_to
            logger.info("Schema migration: %d → %d complete", v_from, v_to)
        except Exception as exc:
            try:
                conn.execute("ROLLBACK")
            except Exception:
                pass
            logger.error(
                "Schema migration FAILED at %d → %d: %s. "
                "Transaction rolled back. schema_version remains %d. "
                "The store is NOT half-migrated — fix the migration "
                "and restart.",
                v_from, v_to, exc, current,
            )
            raise RuntimeError(
                f"Schema migration {v_from}→{v_to} failed: {exc}"
            ) from exc

    to_version = get_schema_version(conn)
    logger.info(
        "Schema migration: complete. %d → %d (applied=%s, skipped=%s)",
        from_version, to_version, applied, skipped,
    )

    # Post-migration verification: assert schema_version matches.
    if to_version != LATEST_SCHEMA_VERSION and migrations is None:
        logger.warning(
            "Schema migration: schema_version=%d but LATEST_SCHEMA_VERSION=%d "
            "(migration list may be incomplete or a migration was skipped)",
            to_version, LATEST_SCHEMA_VERSION,
        )

    return {
        "from_version": from_version,
        "to_version": to_version,
        "applied": applied,
        "skipped": skipped,
        "health_check": health_check_ok,
    }

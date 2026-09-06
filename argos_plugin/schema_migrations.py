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

import json
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
    """
    pass


def _migration_1_to_2(conn) -> None:
    """#286: dimension-generic vector provenance.

    The additive layer already added ``embedding_dim``, ``embedder_id``,
    and ``embedded_at`` columns via ALTER TABLE ADD COLUMN IF NOT EXISTS.
    This migration backfills ``embedding_dim`` for existing rows that
    have embeddings but no dim stamp (legacy rows from before #286).

    The backfill computes ``len(embedding)`` for each row with a non-
    NULL embedding and NULL embedding_dim. This is a data transform
    (not just a schema stamp) — it reads every embedded row and writes
    its dimension. Idempotent: rows with a non-NULL embedding_dim are
    skipped (WHERE embedding_dim IS NULL).

    embedder_id and embedded_at remain NULL for legacy rows (we don't
    know which model produced them). The re-embed orchestration stamps
    them on future re-embeds.

    Defensive: if the ``embedding_dim`` column doesn't exist yet (e.g.
    when run_migrations is called directly on a minimal test DB without
    the additive layer), the migration is a no-op. The additive layer
    in store_core._init_db() adds the column before the runner executes.
    """
    # Check if embedding_dim and embedding columns exist. If not, skip
    # (the additive layer will add them on a real store init).
    try:
        cols = conn.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema = 'main' AND table_name = 'memory_records' "
            "AND column_name IN ('embedding_dim', 'embedding')"
        ).fetchall()
        col_names = {r[0] for r in cols}
        if "embedding_dim" not in col_names or "embedding" not in col_names:
            # Columns don't exist — nothing to backfill. The additive
            # layer will add them on a real store init.
            return
    except Exception:
        return

    # Backfill embedding_dim for existing rows with embeddings.
    # DuckDB's len() returns the length of a list column.
    conn.execute("""
        UPDATE memory_records
        SET embedding_dim = len(embedding)
        WHERE embedding IS NOT NULL
          AND embedding_dim IS NULL
    """)


def _migration_2_to_3(conn) -> None:
    """#293: POPIA deletion receipts — append-only erase evidence.

    Creates the ``deletion_receipts`` table: an append-only log of what
    was deleted, when, by whom, and at whose request. This is the
    "provable" part of the erase-request workflow:

    - One row per erased record: receipt_id (PK), request_id (the erase
      batch), subject (who the erase was for), memory_id, content_hash
      (verifiable proof of WHAT was deleted without retaining the
      content), category, user_scope, requested_by, reason, outcome,
      created_at.
    - APPEND-ONLY: the codebase only ever INSERTs into this table —
      there is no UPDATE/DELETE path, and the erase flow itself cannot
      touch it (it matches memory_records, not receipts). Receipts
      survive the deletion they prove.
    - Content is NOT retained in the receipt (POPIA minimality): the
      hash + memory_id + category prove the deletion; the tombstone
      table independently blocks re-feeding the content.

    Idempotent (CREATE TABLE IF NOT EXISTS) and transactional via the
    runner.
    """
    conn.execute("""
        CREATE TABLE IF NOT EXISTS deletion_receipts (
            receipt_id   VARCHAR PRIMARY KEY,
            request_id   VARCHAR,
            subject      VARCHAR,
            memory_id    VARCHAR,
            content_hash VARCHAR,
            category     VARCHAR,
            user_scope   VARCHAR,
            requested_by VARCHAR,
            reason       VARCHAR DEFAULT 'erase_request',
            outcome      VARCHAR,
            details      VARCHAR,
            created_at   VARCHAR
        )
    """)


MIGRATIONS: List[Migration] = [
    (0, 1, _migration_0_to_1),
    (1, 2, _migration_1_to_2),
    (2, 3, _migration_2_to_3),
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


# ===========================================================================
# Graph migrations (Kuzu) — #287
# ===========================================================================
# The Kuzu graph has its own ordered migration list, mirroring the DuckDB
# runner's model (#288): ordered, idempotent, transactional, fail-loud.
# The graph version persists in a ``GraphMeta`` node table inside the Kuzu
# DB itself (key='graph_schema_version') — the DuckDB schema_meta table is
# not visible from the Kuzu connection.
#
# Kuzu-specific constraint: a FAILED statement inside an explicit
# transaction kills the transaction ("No active transaction for COMMIT").
# Migration functions must therefore pre-check preconditions (e.g. column
# existence via ``CALL table_info``) and only execute statements that are
# known to succeed — never rely on catching a failed statement inside the
# transaction.
#
# The runner is invoked from KuzuGraphStore._init_db() after the base DDL
# and the G4 memory_ids migration.

# Timestamp columns added to Entity nodes and RelatesTo edges (#287).
# ISO-8601 strings — lexicographic order == chronological order, matching
# the record layer's valid_from/valid_to VARCHAR comparison semantics.
_GRAPH_NODE_TEMPORAL_COLUMNS = ("created_at", "valid_from", "valid_to")
_GRAPH_EDGE_TEMPORAL_COLUMNS = ("created_at", "valid_from", "valid_to")

_GRAPH_META_DDL = (
    "CREATE NODE TABLE GraphMeta("
    "key STRING, value STRING, PRIMARY KEY(key))"
)


def _ensure_graph_meta_table(conn) -> None:
    """Create the GraphMeta node table if it doesn't exist (auto-commit)."""
    try:
        conn.execute(_GRAPH_META_DDL)
    except RuntimeError as exc:
        if not _is_kuzu_already_exists_error(exc):
            raise


def _is_kuzu_already_exists_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "already exists" in msg or "catalog exception" in msg


def _kuzu_table_columns(conn, table_name: str) -> set:
    """Return the set of column names for a Kuzu table."""
    cols = set()
    result = conn.execute(
        f"CALL table_info('{table_name}') RETURN *"
    )
    while result.has_next():
        row = result.get_next()
        # table_info row: [cid, name, type, default_val, primary_key]
        if len(row) > 1 and row[1]:
            cols.add(str(row[1]))
    return cols


def get_graph_schema_version(conn) -> int:
    """Read the persisted graph schema version from the GraphMeta table.

    Returns 0 for a fresh graph (no GraphMeta table or no row).
    """
    try:
        result = conn.execute(
            "MATCH (m:GraphMeta {key: 'graph_schema_version'}) "
            "RETURN m.value"
        )
        if result.has_next():
            return int(result.get_next()[0])
        return 0
    except Exception:
        # GraphMeta table doesn't exist yet — fresh graph, version 0.
        return 0


def _set_graph_schema_version(conn, version: int) -> None:
    """Persist the graph schema version in the GraphMeta table (upsert).

    Must be called inside the migration's transaction (or auto-commit —
    both fine; MERGE is a single statement).
    """
    conn.execute(
        """
        MERGE (m:GraphMeta {key: 'graph_schema_version'})
        ON MATCH SET m.value = $value
        ON CREATE SET m.value = $value
        """,
        parameters={"value": str(int(version))},
    )


def _graph_migration_0_to_1(conn) -> None:
    """#287: temporal-aware graph — timestamp columns on nodes and edges.

    Adds ``created_at`` / ``valid_from`` / ``valid_to`` (ISO-8601 STRING)
    to the Entity node table and the RelatesTo edge table, then backfills
    from each row's own attributes JSON provenance:

    - ``created_at`` / ``valid_from`` ← attributes.observed_at (edges,
      stamped by index_memory since #138) or attributes.created_at
      (memory nodes). Rows with no temporal provenance in their JSON
      stay NULL — provenance is never invented.
    - ``valid_to`` stays NULL (open) — the graph cannot know a record's
      closure on its own; the authoritative window backfill runs through
      the re-index path (rebuild_graph.py / backfill_graph.py), which
      reads valid_from/valid_to from the DuckDB source records.

    Idempotent: pre-checks column existence via table_info and only
    ALTERs missing columns, so a second run executes nothing. All DDL +
    backfill + version stamp run inside ONE transaction — a failure
    rolls back cleanly (no half-migrated graph).

    Kuzu constraint: a failed statement kills the transaction, so the
    column checks happen FIRST (reads are safe inside a txn) and only
    missing columns are ALTERed.
    """
    # Pre-check (safe reads) which columns are missing.
    node_cols = _kuzu_table_columns(conn, "Entity")
    edge_cols = _kuzu_table_columns(conn, "RelatesTo")
    node_alters = [
        f"ALTER TABLE Entity ADD {col} STRING"
        for col in _GRAPH_NODE_TEMPORAL_COLUMNS
        if col not in node_cols
    ]
    edge_alters = [
        f"ALTER TABLE RelatesTo ADD {col} STRING"
        for col in _GRAPH_EDGE_TEMPORAL_COLUMNS
        if col not in edge_cols
    ]

    conn.execute("BEGIN TRANSACTION")
    try:
        for stmt in node_alters:
            conn.execute(stmt)
        for stmt in edge_alters:
            conn.execute(stmt)

        # Backfill created_at/valid_from from the row's own attributes
        # JSON provenance (Python-side parse — Kuzu has no JSON functions
        # at personal-store scale this scan is cheap, mirroring the G4
        # backfill pattern).
        _backfill_graph_timestamps(conn)

        conn.execute("COMMIT")
    except Exception:
        try:
            conn.execute("ROLLBACK")
        except Exception:
            pass
        raise


def _backfill_graph_timestamps(conn) -> int:
    """Stamp created_at/valid_from on nodes/edges from attributes JSON.

    Provenance-preserving: only stamps from temporal evidence already in
    the row's own attributes JSON (observed_at on edges since #138,
    created_at on memory nodes). Never invents timestamps. Rows with no
    provenance stay NULL until the re-index path backfills them from the
    DuckDB source records.

    Returns the number of rows stamped.
    """
    stamped = 0

    # Edges: observed_at lives in the attributes JSON (stamped by
    # index_memory since #138).
    result = conn.execute(
        """MATCH (a:Entity)-[r:RelatesTo]->(b:Entity)
           WHERE r.created_at IS NULL AND r.valid_from IS NULL
           RETURN a.id, r.relation_type, b.id, r.attributes"""
    )
    rows = []
    while result.has_next():
        rows.append(result.get_next())
    for src, rel, dst, raw_attrs in rows:
        try:
            attrs = json.loads(raw_attrs) if raw_attrs else {}
        except Exception:
            attrs = {}
        observed = attrs.get("observed_at") or attrs.get("created_at")
        if not observed:
            continue
        conn.execute(
            """MATCH (a:Entity {id: $src})-[r:RelatesTo {relation_type: $rel}]->(b:Entity {id: $dst})
               SET r.created_at = $ts, r.valid_from = $ts""",
            parameters={"src": src, "rel": rel, "dst": dst, "ts": str(observed)},
        )
        stamped += 1

    # Nodes: memory nodes carry created_at in their attributes JSON.
    result = conn.execute(
        """MATCH (n:Entity)
           WHERE n.created_at IS NULL AND n.entity_type = 'memory'
           RETURN n.id, n.attributes"""
    )
    rows = []
    while result.has_next():
        rows.append(result.get_next())
    for nid, raw_attrs in rows:
        try:
            attrs = json.loads(raw_attrs) if raw_attrs else {}
        except Exception:
            attrs = {}
        created = attrs.get("created_at")
        if not created:
            continue
        conn.execute(
            "MATCH (n:Entity {id: $id}) SET n.created_at = $ts, n.valid_from = $ts",
            parameters={"id": nid, "ts": str(created)},
        )
        stamped += 1

    if stamped:
        logger.info(
            "Graph temporal backfill: stamped timestamps on %d row(s) "
            "from attributes provenance",
            stamped,
        )
    return stamped


GRAPH_MIGRATIONS: List[Migration] = [
    (0, 1, _graph_migration_0_to_1),
]

# The latest graph schema version = the last graph migration's version_to.
LATEST_GRAPH_SCHEMA_VERSION = (
    GRAPH_MIGRATIONS[-1][1] if GRAPH_MIGRATIONS else 0
)


def run_graph_migrations(
    conn,
    *,
    migrations: List[Migration] | None = None,
) -> dict:
    """Run ordered graph (Kuzu) migrations on *conn*.

    Mirrors the DuckDB ``run_migrations`` model (#288): ordered,
    idempotent, transactional, fail-loud. Called from
    ``KuzuGraphStore._init_db()`` after the base DDL.

    Each migration runs inside BEGIN/COMMIT (managed jointly by the
    runner and the migration fn — see the Kuzu constraint note above).
    A failure rolls back and raises — the graph is NOT left
    half-migrated. The version stamp is written AFTER the migration fn
    commits (a separate implicit transaction — Kuzu DDL pre-checks must
    run inside the migration fn's txn, so the runner cannot own the
    txn boundary).

    Returns a report dict: from_version, to_version, applied, skipped.
    """
    migs = migrations if migrations is not None else GRAPH_MIGRATIONS

    # Ensure the GraphMeta table exists before reading from it
    # (auto-commit DDL — safe outside a transaction).
    _ensure_graph_meta_table(conn)

    from_version = get_graph_schema_version(conn)
    applied: List[int] = []
    skipped: List[int] = []

    logger.info(
        "Graph schema migration: starting from version %d, target %d",
        from_version, LATEST_GRAPH_SCHEMA_VERSION,
    )

    current = from_version
    for v_from, v_to, apply_fn in migs:
        if v_to <= current:
            skipped.append(v_to)
            continue
        if v_from != current:
            raise RuntimeError(
                f"Graph migration gap: expected version_from={current} "
                f"but migration says version_from={v_from} "
                f"(version_to={v_to}). The migration list is broken "
                f"or a migration was inserted out of order."
            )
        logger.info("Graph schema migration: applying %d → %d", v_from, v_to)
        try:
            # The migration fn manages its own transaction (BEGIN/
            # COMMIT/ROLLBACK) because Kuzu pre-checks must happen
            # inside the txn but failed statements kill it — see the
            # module comment. The fn raises on failure after rolling
            # back, so the graph is never half-migrated.
            apply_fn(conn)
            _set_graph_schema_version(conn, v_to)
            applied.append(v_to)
            current = v_to
            logger.info("Graph schema migration: %d → %d complete", v_from, v_to)
        except Exception as exc:
            logger.error(
                "Graph schema migration FAILED at %d → %d: %s. "
                "Transaction rolled back. graph_schema_version remains %d. "
                "The graph is NOT half-migrated — fix the migration "
                "and restart.",
                v_from, v_to, exc, current,
            )
            raise RuntimeError(
                f"Graph schema migration {v_from}→{v_to} failed: {exc}"
            ) from exc

    to_version = get_graph_schema_version(conn)
    logger.info(
        "Graph schema migration: complete. %d → %d (applied=%s, skipped=%s)",
        from_version, to_version, applied, skipped,
    )

    return {
        "from_version": from_version,
        "to_version": to_version,
        "applied": applied,
        "skipped": skipped,
    }

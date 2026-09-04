"""Regression: legacy stores (pre-batch-F) must receive the ``tier`` column
via the ALTER migration map on init.

The first batch-F release (P5.1, #6 — memory lifecycle) added ``tier`` to the
CREATE TABLE base schema only. Pre-existing stores (created before batch-F)
never got the column, so every retrieval binder-errored on
``COALESCE(tier, 'active') = 'active'``. Fresh-DB tests could not catch this
because they always start from the current base schema.

Guard: simulate a legacy store (drop the column), reopen it, and assert
retrieval runs clean and respects the tier filter.
"""
from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import pytest

_plugin_dir = Path(__file__).resolve().parent.parent
if str(_plugin_dir) not in sys.path:
    sys.path.insert(0, str(_plugin_dir))

from store import DuckDBMemoryStore  # noqa: E402


def _legacy_store_without_tier(db_path: Path) -> None:
    """Create a store then strip the tier column, emulating a pre-batch-F DB.

    The column is dropped with data intact so the migration must ADD COLUMN
    IF NOT EXISTS on reopen — exactly the live-store condition that failed.
    Dependent indexes are dropped first (DuckDB refuses to alter a table with
    dependent entries); _init_db recreates them IF NOT EXISTS on reopen.
    """
    store = DuckDBMemoryStore(db_path, embedder=None)
    try:
        store.remember(content="The tax deadline is 30 September", category="finance")
    finally:
        store.close()
    conn = duckdb.connect(str(db_path))
    try:
        rows = conn.execute(
            "SELECT index_name FROM duckdb_indexes() WHERE table_name = 'memory_records'"
        ).fetchall()
        for (idx,) in rows:
            conn.execute(f"DROP INDEX IF EXISTS \"{idx}\"")
        conn.execute("ALTER TABLE memory_records DROP COLUMN tier")
    finally:
        conn.close()


def test_legacy_store_without_tier_column_migrates_on_init(tmp_path):
    db = tmp_path / "legacy.duckdb"
    _legacy_store_without_tier(db)

    # Pre-reopen sanity: the column really is gone (else the test is vacuous).
    conn = duckdb.connect(str(db))
    try:
        cols = [r[0] for r in conn.execute("DESCRIBE memory_records").fetchall()]
        assert "tier" not in cols, "test setup failed: tier should be absent"
    finally:
        conn.close()

    # Reopen — _init_db must ALTER-add tier back.
    store = DuckDBMemoryStore(db, embedder=None)
    try:
        results = store.search("tax deadline", limit=5)
        assert results, "retrieval should return the remembered memory"
        assert any("tax deadline" in r.content for r in results)
        # The migrated column defaults to 'active', so the record is visible
        # under the default (non-archived) tier filter.
        hit = next(r for r in results if "tax deadline" in r.content)
        assert getattr(hit, "tier", "active") == "active"
        # And the column now exists for lifecycle ops.
        with store._state.lock:
            row = store.connection.execute(
                "SELECT tier FROM memory_records LIMIT 1"
            ).fetchone()
        assert row[0] == "active"
    finally:
        store.close()
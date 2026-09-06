"""#294: portable export/import (anti-lock-in, data sovereignty).

Tests:
1. Export completeness: every record field, evidence row, version
   chain, tombstone present in the JSONL; Markdown digest covers all
   records.
2. Format versioning: export carries format+version; import refuses
   unknown/newer versions; deterministic byte-stable output.
3. Re-import equivalence: a fresh store reproduces the original
   (ALL fields incl. provenance + tombstones, not just content).
4. Idempotency: replaying the same import twice is a no-op.
5. Scope: exporting tenant A never includes tenant B's records;
   import stamps the target cell's user_scope (no cross-tenant leak).
6. Receipts/tombstones round-trip (POPIA): deleted records remain
   deleted; deletion receipts survive.
7. Validation: malformed/incomplete export → per-row failure report,
   no partial silent write.

Run with (Hermes venv python, hermetic):
    ARGOS_HERMETIC_TESTS=1 python -m pytest tests/test_portable_export.py -v
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

_plugin_dir = Path(__file__).resolve().parent.parent
if str(_plugin_dir) not in sys.path:
    sys.path.insert(0, str(_plugin_dir))

NOW = datetime.now(timezone.utc)

# Tables compared for full-store equivalence.
_EQUIVALENCE_TABLES = (
    "memory_records",
    "memory_evidence",
    "deletion_tombstones",
    "deletion_receipts",
    "entity_aliases",
    "rejection_ledger",
)


@pytest.fixture
def store_a(tmp_path):
    from store import DuckDBMemoryStore
    s = DuckDBMemoryStore(tmp_path / "a.duckdb", user_id="alice")
    yield s
    s.close()


@pytest.fixture
def store_b(tmp_path):
    from store import DuckDBMemoryStore
    s = DuckDBMemoryStore(tmp_path / "b.duckdb", user_id="alice")
    yield s
    s.close()


def _dump_tables(store) -> dict:
    """Dump the user-scoped rows of every equivalence table, ordered."""
    out = {}
    with store._state.lock:
        for table in _EQUIVALENCE_TABLES:
            cur = store.connection.execute(
                f"SELECT * FROM {table} WHERE (user_scope IS NULL OR "
                f"user_scope = ?) ORDER BY 1",
                [store.user_id],
            )
            cols = [d[0] for d in cur.description]
            out[table] = [dict(zip(cols, r)) for r in cur.fetchall()]
    return out


def _seed_rich_store(store):
    """Seed records with provenance, a version chain, evidence, an
    alias, and an erase (tombstone + receipt)."""
    # Records WITH evidence (candidate → review → evidence row).
    cand = store.save_candidate(
        category="personal_fact",
        content="Alice works at Acme",
        source="structured_ingest",
        evidence_text="structured_ingest: seed.json row 1 (mapping abc)",
        evidence_role="structured_ingest",
        provenance_origin="external",
        grounding="extracted",
    )
    review = store.review_candidate(
        candidate_id=cand["candidate_id"],
        decision="approved",
        review_source="tool",
    )
    head_id = review["memory"]["memory_id"]
    # Version chain: supersede the head.
    store.update_memory(head_id, content="Alice works at Acme Corp")
    # A plain record + an alias.
    store.remember(category="preference", content="Alice prefers tea")
    store.add_alias("my wife", "Sam")
    # Erase one record → tombstone + receipt.
    store.erase_subject("prefers tea", mode="apply", confirm=True)
    return head_id


class TestExportCompleteness:
    """1. Export carries every field; digest covers all records."""

    def test_export_contains_all_row_types(self, store_a):
        _seed_rich_store(store_a)
        exp = store_a.export_portable()
        types = {r["type"] for r in exp["rows"]}
        assert "record" in types
        assert "evidence" in types
        assert "tombstone" in types
        assert "receipt" in types
        assert "alias" in types

    def test_record_fields_preserved_in_jsonl(self, store_a):
        _seed_rich_store(store_a)
        exp = store_a.export_portable()
        records = [r["data"] for r in exp["rows"] if r["type"] == "record"]
        assert records
        for rec in records:
            # Provenance fields.
            assert "provenance_origin" in rec
            assert "grounding" in rec
            assert "source" in rec
            assert "source_doc_id" in rec
            # Version-chain fields.
            assert "valid_from" in rec
            assert "valid_to" in rec
            assert "superseded_by" in rec
            # Embedding-provenance fields (#286).
            assert "embedding_dim" in rec
            assert "embedder_id" in rec
            assert "embedded_at" in rec
            # ACL fields.
            assert "user_scope" in rec
            assert "namespace" in rec
            assert "client_scope" in rec

    def test_version_chain_present(self, store_a):
        head_id = _seed_rich_store(store_a)
        exp = store_a.export_portable()
        records = {r["data"]["memory_id"]: r["data"]
                   for r in exp["rows"] if r["type"] == "record"}
        # The old version is in the export with valid_to set and
        # superseded_by pointing at the new head.
        old = records[head_id]
        assert old["valid_to"] is not None
        assert old["superseded_by"] is not None
        new_id = old["superseded_by"]
        assert new_id in records

    def test_evidence_rows_present(self, store_a):
        _seed_rich_store(store_a)
        exp = store_a.export_portable()
        evidence = [r["data"] for r in exp["rows"] if r["type"] == "evidence"]
        assert evidence, "evidence rows missing from export"
        ev = evidence[0]
        assert ev.get("evidence_text")
        assert "evidence_role" in ev
        assert "reviewer_decision" in ev

    def test_markdown_digest_covers_all_records(self, store_a):
        _seed_rich_store(store_a)
        exp = store_a.export_portable()
        records = [r["data"] for r in exp["rows"] if r["type"] == "record"]
        assert records
        for rec in records:
            assert rec["memory_id"] in exp["markdown"], (
                f"digest missing record {rec['memory_id']}"
            )
        assert "## Records" in exp["markdown"]
        assert "## Deletion receipts" in exp["markdown"]
        assert "## Deletion tombstones" in exp["markdown"]

    def test_header_carries_format_version_and_counts(self, store_a):
        _seed_rich_store(store_a)
        exp = store_a.export_portable()
        header = exp["header"]
        assert header["export_format"] == "argos-portable"
        assert header["export_version"] == 1
        assert isinstance(header["schema_version"], int)
        assert header["schema_version"] >= 3
        assert header["counts"].get("record", 0) >= 1
        assert header["scope"].get("user_scope") == "alice"


class TestFormatVersioning:
    """2. Versioned format; refuse unknown/newer; byte-stable."""

    def test_export_deterministic_byte_stable(self, store_a):
        _seed_rich_store(store_a)
        exp1 = store_a.export_portable()
        exp2 = store_a.export_portable()
        assert exp1["jsonl"] == exp2["jsonl"], "export not byte-stable"
        assert exp1["markdown"] == exp2["markdown"]

    def test_import_refuses_unknown_format(self, store_b):
        bad = json.dumps({"export_format": "other-tool", "export_version": 1})
        with pytest.raises(Exception, match="unsupported export format"):
            store_b.import_portable(bad, mode="preview")

    def test_import_refuses_newer_version(self, store_b):
        newer = json.dumps({
            "export_format": "argos-portable", "export_version": 99,
        })
        with pytest.raises(Exception, match="newer"):
            store_b.import_portable(newer, mode="preview")

    def test_import_accepts_current_version(self, store_a, store_b):
        _seed_rich_store(store_a)
        exp = store_a.export_portable()
        rep = store_b.import_portable(exp["jsonl"], mode="preview")
        assert rep["export_version"] == 1
        assert rep["error_rows"] == 0


class TestReimportEquivalence:
    """3. A fresh store reproduces the original — ALL fields."""

    def test_full_equivalence_after_import(self, store_a, store_b):
        _seed_rich_store(store_a)
        exp = store_a.export_portable()
        rep = store_b.import_portable(exp["jsonl"], mode="apply", confirm=True)
        assert rep["wrote"] is True
        assert rep["error_rows"] == 0
        da, db = _dump_tables(store_a), _dump_tables(store_b)
        for table in _EQUIVALENCE_TABLES:
            assert da[table] == db[table], (
                f"table {table} differs after round-trip:\n"
                f"A={da[table]}\nB={db[table]}"
            )

    def test_provenance_and_version_chain_survive(self, store_a, store_b):
        head_id = _seed_rich_store(store_a)
        exp = store_a.export_portable()
        store_b.import_portable(exp["jsonl"], mode="apply", confirm=True)
        # The old version's chain fields are identical in both stores.
        with store_a._state.lock:
            a = store_a.connection.execute(
                "SELECT valid_from, valid_to, superseded_by, "
                "provenance_origin, grounding FROM memory_records "
                "WHERE memory_id = ?", [head_id],
            ).fetchone()
        with store_b._state.lock:
            b = store_b.connection.execute(
                "SELECT valid_from, valid_to, superseded_by, "
                "provenance_origin, grounding FROM memory_records "
                "WHERE memory_id = ?", [head_id],
            ).fetchone()
        assert a == b
        assert b[2] is not None  # superseded_by preserved


class TestIdempotency:
    """4. Replaying the same import twice is a no-op."""

    def test_replay_is_noop(self, store_a, store_b):
        _seed_rich_store(store_a)
        exp = store_a.export_portable()
        r1 = store_b.import_portable(exp["jsonl"], mode="apply", confirm=True)
        assert r1["wrote"] is True
        before = _dump_tables(store_b)
        r2 = store_b.import_portable(exp["jsonl"], mode="apply", confirm=True)
        after = _dump_tables(store_b)
        assert before == after, "replay changed the store"
        assert r2["restored"] == 0
        assert r2["unchanged"] == r1["restored"] + r1["unchanged"]

    def test_preview_writes_nothing(self, store_a, store_b):
        _seed_rich_store(store_a)
        exp = store_a.export_portable()
        rep = store_b.import_portable(exp["jsonl"], mode="preview")
        assert rep["wrote"] is False
        da, db = _dump_tables(store_a), _dump_tables(store_b)
        assert db["memory_records"] == []
        assert db["deletion_receipts"] == []
        # And the preview reports what WOULD be restored.
        assert rep["restored"] == 0
        would = [r for r in rep["rows"] if r["outcome"] == "would_restore"]
        assert would, "preview reported nothing to restore"


class TestScopeIsolation:
    """5. Export/import is tenant-scoped — no cross-tenant leakage."""

    def test_export_scoped_per_tenant(self, store_a):
        store_a.remember(category="personal_fact",
                         content="Alice private fact xyz")
        store_a.set_user_scope("bob")
        store_a.remember(category="personal_fact",
                         content="Bob secret fact qrs")
        store_a.set_user_scope("alice")
        exp = store_a.export_portable()
        contents = [r["data"].get("content") for r in exp["rows"]
                    if r["type"] == "record"]
        assert any("Alice private" in (c or "") for c in contents)
        assert not any("Bob secret" in (c or "") for c in contents), (
            "cross-tenant leak in export"
        )

    def test_import_stamps_target_scope(self, store_a, store_b):
        store_a.remember(category="personal_fact",
                         content="Alice portable fact")
        exp = store_a.export_portable()
        # Import into bob's cell — rows are stamped bob, not alice.
        store_b.set_user_scope("bob")
        rep = store_b.import_portable(exp["jsonl"], mode="apply", confirm=True)
        assert rep["wrote"] is True
        with store_b._state.lock:
            scopes = store_b.connection.execute(
                "SELECT DISTINCT user_scope FROM memory_records"
            ).fetchall()
        assert all(s[0] == "bob" for s in scopes), (
            "import wrote rows outside the target tenant scope"
        )
        # Alice's cell in store_b is untouched.
        store_b.set_user_scope("alice")
        assert store_b.list_recent(limit=100) == []


class TestReceiptsTombstonesRoundTrip:
    """6. POPIA: deleted stays deleted; receipts survive the round-trip."""

    def test_erased_record_stays_deleted_after_import(self, store_a, store_b):
        store_a.remember(category="personal_fact",
                         content="Erasable fact about Zed")
        rep = store_a.erase_subject("Zed", mode="apply", confirm=True)
        assert rep["erased_count"] == 1
        exp = store_a.export_portable()
        # The export contains the tombstone + receipt but NOT the record.
        types = [r["type"] for r in exp["rows"]]
        assert "tombstone" in types and "receipt" in types
        assert not any(
            r["type"] == "record" and "Zed" in (r["data"].get("content") or "")
            for r in exp["rows"]
        )
        # Import into a fresh store: the tombstone + receipt are
        # restored; the record is not resurrected.
        store_b.import_portable(exp["jsonl"], mode="apply", confirm=True)
        with store_b._state.lock:
            n = store_b.connection.execute(
                "SELECT COUNT(*) FROM memory_records "
                "WHERE content LIKE '%Zed%'"
            ).fetchone()[0]
            receipts = store_b.connection.execute(
                "SELECT COUNT(*) FROM deletion_receipts"
            ).fetchone()[0]
        assert n == 0, "erased record resurrected by import"
        assert receipts == 1, "deletion receipt did not survive"

    def test_tombstone_blocks_reimport_of_erased_content(self, store_a, store_b):
        """A record erased AFTER an older export must not be resurrected
        by replaying that export into the erased store."""
        store_a.remember(category="personal_fact",
                         content="Replay-resurrect target Wex")
        exp = store_a.export_portable()
        # Erase AFTER the export.
        store_a.erase_subject("Wex", mode="apply", confirm=True)
        # Re-import the OLD export into the SAME (erased) store — the
        # tombstone must block the record's resurrection.
        rep = store_a.import_portable(exp["jsonl"], mode="apply", confirm=True)
        blocked = [r for r in rep["rows"]
                   if r["type"] == "record" and r["outcome"] == "tombstone_blocked"]
        assert blocked, "tombstone did not block the re-import"
        with store_a._state.lock:
            n = store_a.connection.execute(
                "SELECT COUNT(*) FROM memory_records WHERE content LIKE '%Wex%'"
            ).fetchone()[0]
        assert n == 0, "erased record resurrected by replaying an old export"


class TestImportValidation:
    """7. Malformed/incomplete export → per-row report, no partial write."""

    def test_malformed_lines_reported_per_row(self, store_b):
        header = json.dumps({
            "export_format": "argos-portable", "export_version": 1,
            "schema_version": 3, "scope": {}, "counts": {},
        })
        data = "\n".join([
            header,
            "{not json",
            json.dumps({"type": "record", "data": {"memory_id": "m1"}}),  # no content
            json.dumps({"type": "unknown_type", "data": {}}),
            json.dumps({"type": "record", "data": {
                "memory_id": "m-ok", "category": "personal_fact",
                "content": "valid row",
            }}),
        ])
        rep = store_b.import_portable(data, mode="preview")
        assert rep["error_rows"] == 3
        assert rep["valid_rows"] == 1
        lines = {e["line"] for e in rep["errors"]}
        assert lines == {2, 3, 4}

    def test_apply_with_validation_errors_writes_nothing(self, store_b):
        header = json.dumps({
            "export_format": "argos-portable", "export_version": 1,
            "schema_version": 3, "scope": {}, "counts": {},
        })
        data = "\n".join([
            header,
            json.dumps({"type": "record", "data": {
                "memory_id": "m-ok", "category": "personal_fact",
                "content": "valid row",
            }}),
            "{broken",
        ])
        with pytest.raises(ValueError, match="validation failed"):
            store_b.import_portable(data, mode="apply", confirm=True)
        # All-or-nothing: the valid row must NOT have been written.
        with store_b._state.lock:
            n = store_b.connection.execute(
                "SELECT COUNT(*) FROM memory_records"
            ).fetchone()[0]
        assert n == 0

    def test_apply_requires_strict_confirm(self, store_a, store_b):
        _seed_rich_store(store_a)
        exp = store_a.export_portable()
        # String "false" must NOT pass (bool("false") is True).
        with pytest.raises(ValueError, match="confirm=True"):
            store_b.import_portable(exp["jsonl"], mode="apply",
                                    confirm="false")
        with pytest.raises(ValueError, match="confirm=True"):
            store_b.import_portable(exp["jsonl"], mode="apply")
        # Nothing written.
        assert _dump_tables(store_b)["memory_records"] == []

    def test_empty_import_fails_loud(self, store_b):
        with pytest.raises(Exception, match="empty"):
            store_b.import_portable("", mode="preview")

"""#293: POPIA retention policy + erase-request workflow (provable deletion).

Tests:
1. Retention: per-class policy applied deterministically given a clock
   (records across classes; exact expiry match; nothing extra expires).
2. Scheduled enforcement: runs on the existing session-end hook
   pattern; no records expired early; idempotent re-run.
3. Erase-request: subject-scoped delete; dry-run reports what WOULD be
   deleted and writes nothing; apply requires explicit confirm (strict
   — the "false" string must NOT pass, mirroring #289).
4. Receipts: appended during the deletion, survive it (queryable
   after), NOT vulnerable to the erase flow themselves (append-only).
5. ACL/cells: erase under one tenant does not touch another; records
   under legal-hold are blocked with a clear error.
6. Tombstone consistency: erased records leave a tombstone; the
   receipt is verifiable end-to-end (record gone + tombstone present).
7. Facade wiring: preview/apply/confirm gate, identity server-derived.

Run with (Hermes venv python, hermetic):
    ARGOS_HERMETIC_TESTS=1 python -m pytest tests/test_popia_retention.py -v
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

_plugin_dir = Path(__file__).resolve().parent.parent
if str(_plugin_dir) not in sys.path:
    sys.path.insert(0, str(_plugin_dir))

# Deterministic clock.
NOW = datetime(2026, 9, 6, 12, 0, 0, tzinfo=__import__("datetime").timezone.utc)


@pytest.fixture
def store(tmp_path):
    """A fresh DuckDBMemoryStore with NO embedder (hermetic, fast)."""
    from store import DuckDBMemoryStore
    s = DuckDBMemoryStore(tmp_path / "popia.duckdb", user_id="alice")
    yield s
    s.close()


def _seed(store, memory_id: str, category: str, content: str,
          created: datetime) -> None:
    """Insert a record with a deterministic memory_id + created_at."""
    store.remember(category=category, content=content, dedup=False,
                   created_at=created.isoformat())
    with store._state.lock:
        store.connection.execute(
            "UPDATE memory_records SET memory_id = ? WHERE content = ?",
            [memory_id, content],
        )


# Categories WITHOUT a default TTL stamp from remember() — retention
# tests use these so pre-existing TTL behavior doesn't interfere.
NO_TTL_CATEGORIES = ("personal_fact", "preference", "insight",
                     "relationship", "goal")


class TestRetentionDeterministic:
    """1. Per-class retention applied deterministically given a clock."""

    def test_expires_exactly_past_retention(self, store):
        """Records past their class retention expire; nothing extra.

        Uses personal_fact (no default TTL from remember()) so the
        retention stamping is the only expires_at writer here."""
        old_fact = NOW - timedelta(days=1100)   # past 730d retention
        new_fact = NOW - timedelta(days=100)    # within retention
        _seed(store, "m-old", "personal_fact", "old fact about Berlin", old_fact)
        _seed(store, "m-new", "personal_fact", "new fact about Berlin", new_fact)
        _seed(store, "m-note", "context_note", "some chat note",
              NOW - timedelta(days=3000))  # no policy for this class

        report = store.enforce_retention_policies(
            {"personal_fact": 730}, now=NOW,
        )
        assert report["expired_ids"] == ["m-old"]
        assert report["held_ids"] == []
        assert report["per_class"] == {"personal_fact": {"expired": 1, "held": 0}}

        # The expired record's expires_at == created_at + 730 days
        # (the retention deadline).
        with store._state.lock:
            row = store.connection.execute(
                "SELECT expires_at FROM memory_records "
                "WHERE memory_id = 'm-old'"
            ).fetchone()
        expected = (old_fact + timedelta(days=730)).isoformat()
        assert row[0] == expected

        # Untouched records: m-new has no expires_at at all; m-note's
        # expires_at (remember()'s default TTL) is NOT the retention
        # deadline.
        with store._state.lock:
            rows = store.connection.execute(
                "SELECT memory_id, expires_at FROM memory_records "
                "WHERE memory_id IN ('m-new', 'm-note')"
            ).fetchall()
        by_id = {r[0]: r[1] for r in rows}
        assert by_id["m-new"] is None
        assert by_id["m-note"] != expected

    def test_boundary_exact_match_expires(self, store):
        """A record created EXACTLY retention-days ago expires (deadline
        <= now)."""
        boundary = NOW - timedelta(days=730)
        _seed(store, "m-edge", "personal_fact", "edge fact", boundary)
        report = store.enforce_retention_policies(
            {"personal_fact": 730}, now=NOW,
        )
        assert report["expired_ids"] == ["m-edge"]

    def test_one_day_before_boundary_not_expired(self, store):
        """A record one day inside its retention window is untouched."""
        almost = NOW - timedelta(days=729)
        _seed(store, "m-almost", "personal_fact", "almost fact", almost)
        report = store.enforce_retention_policies(
            {"personal_fact": 730}, now=NOW,
        )
        assert report["expired_ids"] == []

    def test_dry_run_reports_without_writing(self, store):
        _seed(store, "m-old", "personal_fact", "old fact",
              NOW - timedelta(days=1100))
        report = store.enforce_retention_policies(
            {"personal_fact": 730}, now=NOW, dry_run=True,
        )
        assert report["dry_run"] is True
        assert report["expired_ids"] == ["m-old"]
        # Nothing written.
        with store._state.lock:
            row = store.connection.execute(
                "SELECT expires_at FROM memory_records WHERE memory_id = 'm-old'"
            ).fetchone()
        assert row[0] is None

    def test_idempotent_rerun(self, store):
        _seed(store, "m-old", "personal_fact", "old fact",
              NOW - timedelta(days=1100))
        r1 = store.enforce_retention_policies({"personal_fact": 730}, now=NOW)
        assert r1["expired_count"] == 1
        r2 = store.enforce_retention_policies({"personal_fact": 730}, now=NOW)
        assert r2["expired_count"] == 0, "re-run must not re-expire"

    def test_no_early_expiry(self, store):
        """Records NOT past retention are never expired."""
        _seed(store, "m-a", "personal_fact", "fact a",
              NOW - timedelta(days=100))
        _seed(store, "m-b", "event", "event b",
              NOW - timedelta(days=5000))  # old, but no policy for event
        report = store.enforce_retention_policies(
            {"personal_fact": 730}, now=NOW,
        )
        assert report["expired_count"] == 0

    def test_legal_hold_blocks_expiry(self, store):
        _seed(store, "m-held", "personal_fact", "held fact",
              NOW - timedelta(days=1100))
        report = store.enforce_retention_policies(
            {"personal_fact": 730}, now=NOW,
            legal_hold_check=lambda rec: rec["memory_id"] == "m-held",
        )
        assert report["expired_count"] == 0
        assert report["held_ids"] == ["m-held"]
        # Not expired.
        with store._state.lock:
            row = store.connection.execute(
                "SELECT expires_at FROM memory_records WHERE memory_id = 'm-held'"
            ).fetchone()
        assert row[0] is None

    def test_broken_legal_hold_check_fails_closed(self, store):
        """A raising hold check must NOT expire the record (fail-closed)."""
        _seed(store, "m-x", "personal_fact", "x fact",
              NOW - timedelta(days=1100))
        def broken(rec):
            raise RuntimeError("hold registry unavailable")
        report = store.enforce_retention_policies(
            {"personal_fact": 730}, now=NOW, legal_hold_check=broken,
        )
        assert report["expired_count"] == 0
        assert report["held_count"] == 1

    def test_tenant_scoped(self, store):
        _seed(store, "m-alice", "personal_fact", "alice fact",
              NOW - timedelta(days=1100))
        store.set_user_scope("bob")
        report = store.enforce_retention_policies({"personal_fact": 730}, now=NOW)
        assert report["expired_count"] == 0, "must not expire other tenants"
        store.set_user_scope("alice")
        report = store.enforce_retention_policies({"personal_fact": 730}, now=NOW)
        assert report["expired_ids"] == ["m-alice"]


class TestEraseRequestWorkflow:
    """Erase: subject-scoped, dry-run first, strict confirm gate."""

    def _seed_bob(self, store):
        _seed(store, "m-b1", "personal_fact",
              "Bob's phone number is 555-0100", NOW - timedelta(days=9))
        _seed(store, "m-b2", "context_note",
              "Bob mentioned his phone 555-0100 yesterday",
              NOW - timedelta(days=4))
        _seed(store, "m-c1", "personal_fact",
              "Carol works at Initech", NOW - timedelta(days=4))

    def test_preview_writes_nothing(self, store):
        self._seed_bob(store)
        rep = store.erase_subject("Bob", mode="preview")
        assert rep["matched_count"] == 2
        assert rep["wrote"] is False
        assert all(r["outcome"] == "would_erase" for r in rep["records"])
        with store._state.lock:
            n = store.connection.execute(
                "SELECT COUNT(*) FROM memory_records WHERE content LIKE '%Bob%'"
            ).fetchone()[0]
        assert n == 2, "preview must not delete"

    def test_apply_requires_strict_confirm(self, store):
        self._seed_bob(store)
        # String "false" must NOT pass (bool("false") is True).
        with pytest.raises(ValueError, match="confirm=True"):
            store.erase_subject("Bob", mode="apply", confirm="false")
        with pytest.raises(ValueError, match="confirm=True"):
            store.erase_subject("Bob", mode="apply", confirm=1)
        with pytest.raises(ValueError, match="confirm=True"):
            store.erase_subject("Bob", mode="apply")
        # Nothing deleted.
        with store._state.lock:
            n = store.connection.execute(
                "SELECT COUNT(*) FROM memory_records WHERE content LIKE '%Bob%'"
            ).fetchone()[0]
        assert n == 2

    def test_apply_erases_and_receipts(self, store):
        self._seed_bob(store)
        rep = store.erase_subject(
            "Bob", mode="apply", confirm=True, requested_by="ops-admin",
        )
        assert rep["erased_count"] == 2
        assert rep["wrote"] is True
        assert rep["request_id"]
        # Records gone — all versions.
        with store._state.lock:
            n = store.connection.execute(
                "SELECT COUNT(*) FROM memory_records WHERE content LIKE '%Bob%'"
            ).fetchone()[0]
        assert n == 0
        # Every erased record has a receipt with the request id.
        for r in rep["records"]:
            assert r["receipt_id"]
        receipts = store.list_deletion_receipts(
            request_id=rep["request_id"],
        )
        assert len(receipts) == 2
        assert {r["requested_by"] for r in receipts} == {"ops-admin"}
        assert {r["subject"] for r in receipts} == {"Bob"}

    def test_erase_covers_all_versions_and_states(self, store):
        """Erasure covers current AND historical (superseded) versions."""
        _seed(store, "m-v1", "personal_fact",
              "Bob drives a red car", NOW - timedelta(days=30))
        # Supersede it: m-v1 becomes historical (valid_to set).
        rec = store.get_memories_by_ids(["m-v1"])[0]
        store.update_memory("m-v1", content="Bob drives a blue car")
        with store._state.lock:
            old = store.connection.execute(
                "SELECT valid_to FROM memory_records WHERE memory_id = 'm-v1'"
            ).fetchone()
        assert old[0] is not None  # superseded
        rep = store.erase_subject("Bob", mode="apply", confirm=True)
        # Both the current AND the historical version are erased.
        assert rep["erased_count"] == 2
        with store._state.lock:
            n = store.connection.execute(
                "SELECT COUNT(*) FROM memory_records WHERE content LIKE '%Bob%'"
            ).fetchone()[0]
        assert n == 0

    def test_tombstone_prevents_refeed(self, store):
        self._seed_bob(store)
        rep = store.erase_subject("Bob", mode="apply", confirm=True)
        # The erased content is tombstoned — remember() refuses it.
        check = store.tombstone_check(
            "Bob's phone number is 555-0100", "personal_fact",
        )
        assert check is not None, "erased content must be tombstoned"

    def test_scope_filters_narrow_the_subject(self, store):
        _seed(store, "m-s1", "context_note", "Bob at the gym",
              NOW - timedelta(days=1))
        _seed(store, "m-s2", "personal_fact", "Bob joined the gym",
              NOW - timedelta(days=1))
        rep = store.erase_subject(
            "Bob", mode="apply", confirm=True, categories=["context_note"],
        )
        assert rep["erased_count"] == 1
        with store._state.lock:
            n = store.connection.execute(
                "SELECT COUNT(*) FROM memory_records WHERE memory_id = 'm-s2'"
            ).fetchone()[0]
        assert n == 1, "category filter must narrow the erase"


class TestDeletionReceipts:
    """Receipts: appended during deletion, survive it, append-only."""

    def test_receipts_survive_the_deletion(self, store):
        _seed(store, "m-r1", "personal_fact", "Eve works at Umbrella",
              NOW - timedelta(days=2))
        rep = store.erase_subject("Eve", mode="apply", confirm=True)
        receipt_id = rep["records"][0]["receipt_id"]
        # The record is gone…
        with store._state.lock:
            n = store.connection.execute(
                "SELECT COUNT(*) FROM memory_records WHERE memory_id = 'm-r1'"
            ).fetchone()[0]
        assert n == 0
        # …but the receipt is queryable afterwards.
        receipts = store.list_deletion_receipts(memory_id="m-r1")
        assert len(receipts) == 1
        assert receipts[0]["receipt_id"] == receipt_id
        assert receipts[0]["subject"] == "Eve"
        assert receipts[0]["content_hash"], "receipt carries the content hash"

    def test_receipts_not_deletable_via_erase_flow(self, store):
        """The erase flow matches memory_records — receipts are not
        records and cannot be erased through it (append-only)."""
        _seed(store, "m-r2", "personal_fact", "Frank likes hiking",
              NOW - timedelta(days=2))
        rep = store.erase_subject("Frank", mode="apply", confirm=True)
        assert rep["erased_count"] == 1
        # Erase "again" — nothing matches (record gone), receipts intact.
        rep2 = store.erase_subject("Frank", mode="apply", confirm=True)
        assert rep2["matched_count"] == 0
        receipts = store.list_deletion_receipts(subject="Frank")
        assert len(receipts) == 1, "receipt must survive re-erase attempts"

    def test_verify_erase_receipt_end_to_end(self, store):
        """POPIA posture: erase provable end-to-end."""
        _seed(store, "m-v", "personal_fact", "Zara speaks French",
              NOW - timedelta(days=1))
        rep = store.erase_subject("Zara", mode="apply", confirm=True)
        receipt_id = rep["records"][0]["receipt_id"]
        v = store.verify_erase_receipt(receipt_id)
        assert v["valid"] is True
        assert v["record_gone"] is True
        assert v["tombstoned"] is True
        # A bogus receipt does not verify.
        v2 = store.verify_erase_receipt("rcpt-nonexistent")
        assert v2["valid"] is False

    def test_receipt_scoped_per_user(self, store):
        _seed(store, "m-t1", "personal_fact", "Yara works at Acme",
              NOW - timedelta(days=1))
        rep = store.erase_subject("Yara", mode="apply", confirm=True)
        # Another tenant cannot see alice's receipts.
        store.set_user_scope("bob")
        assert store.list_deletion_receipts(subject="Yara") == []
        store.set_user_scope("alice")
        assert len(store.list_deletion_receipts(subject="Yara")) == 1


class TestTenantAndLegalHold:
    """ACL/cells: erase is tenant-scoped; legal hold blocks with a clear error."""

    def test_erase_scoped_per_tenant(self, store):
        _seed(store, "m-alice", "personal_fact", "Kyle works at Acme",
              NOW - timedelta(days=1))
        # Bob's erase matches nothing of alice's.
        store.set_user_scope("bob")
        rep = store.erase_subject("Kyle", mode="apply", confirm=True)
        assert rep["matched_count"] == 0
        assert rep["erased_count"] == 0
        # Alice's record untouched.
        store.set_user_scope("alice")
        with store._state.lock:
            n = store.connection.execute(
                "SELECT COUNT(*) FROM memory_records WHERE memory_id = 'm-alice'"
            ).fetchone()[0]
        assert n == 1

    def test_legal_hold_blocks_erase_with_clear_error(self, store):
        _seed(store, "m-hold", "personal_fact", "Lena works at Globex",
              NOW - timedelta(days=1))
        rep = store.erase_subject(
            "Lena", mode="apply", confirm=True,
            legal_hold_check=lambda rec: True,
        )
        assert rep["erased_count"] == 0
        assert rep["blocked_count"] == 1
        entry = rep["records"][0]
        assert entry["outcome"] == "blocked_legal_hold"
        assert "legal hold" in entry["reason"]
        # The record survives.
        with store._state.lock:
            n = store.connection.execute(
                "SELECT COUNT(*) FROM memory_records WHERE memory_id = 'm-hold'"
            ).fetchone()[0]
        assert n == 1

    def test_legal_hold_absent_by_default(self, store):
        """Without a legal-hold registry, the blocker is absent."""
        _seed(store, "m-free", "personal_fact", "Mike works at Initech",
              NOW - timedelta(days=1))
        rep = store.erase_subject("Mike", mode="apply", confirm=True)
        assert rep["erased_count"] == 1
        assert rep["blocked_count"] == 0


class TestGraphConsistencyOnErase:
    """Erased records leave no dangling graph rows (mirror removal)."""

    def test_graph_mirror_removes_erased_memory(self, store, tmp_path):
        from graph import KuzuGraphStore
        graph = KuzuGraphStore(tmp_path / "g", user_id="alice")
        try:
            _seed(store, "m-g1", "personal_fact",
                  "Nina works at Umbrella with Paul", NOW - timedelta(days=1))
            graph.index_memory(
                memory_id="m-g1",
                category="personal_fact",
                content="Nina works at Umbrella with Paul",
                created_at=(NOW - timedelta(days=1)).isoformat(),
                use_llm=False,
            )
            # Graph has evidence for the memory.
            assert graph.count_edges() > 0
            rep = store.erase_subject(
                "Nina", mode="apply", confirm=True, graph=graph,
            )
            assert rep["erased_count"] == 1
            # No dangling graph rows for the erased memory.
            with graph._shared_conn_lock:
                result = graph.conn.execute(
                    """MATCH (a:Entity)-[r:RelatesTo]->(b:Entity)
                       WHERE list_contains(r.memory_ids, $mid)
                       RETURN COUNT(*)""",
                    parameters={"mid": "m-g1"},
                )
                assert result.get_next()[0] == 0
        finally:
            graph.close()


class TestFacadeEraseOperation:
    """Facade wiring: preview/apply, strict confirm, identity server-set."""

    @pytest.fixture
    def facade(self, tmp_path):
        from store import DuckDBMemoryStore
        from api_facade import ArgosAPIFacade
        s = DuckDBMemoryStore(tmp_path / "facade.duckdb", user_id="alice")
        f = ArgosAPIFacade(s)
        yield f, s
        s.close()

    def _ctx(self):
        from api_facade import AuthContext, READ_OPERATIONS, PROPOSAL_OPERATIONS
        return AuthContext(
            principal="ops-agent",
            tenant="default",
            user_id="alice",
            transport="test",
            allowed_operations=set(READ_OPERATIONS) | set(PROPOSAL_OPERATIONS),
            can_propose=True,
        )

    def test_facade_preview_then_apply(self, facade):
        f, store = facade
        _seed(store, "m-k1", "personal_fact", "Kyle works at Acme",
              NOW - timedelta(days=1))
        ctx = self._ctx()
        params = {"subject": "Kyle", "mode": "preview"}
        result = f.execute(ctx, "erase_request", params)
        assert result["matched_count"] == 1
        assert result["wrote"] is False
        params["mode"] = "apply"
        params["confirm"] = True
        result = f.execute(ctx, "erase_request", params)
        assert result["erased_count"] == 1
        assert result["wrote"] is True

    def test_facade_string_confirm_rejected(self, facade):
        f, store = facade
        _seed(store, "m-k2", "personal_fact", "Kyle plays chess",
              NOW - timedelta(days=1))
        ctx = self._ctx()
        with pytest.raises(Exception):
            f.execute(ctx, "erase_request", {
                "subject": "Kyle", "mode": "apply", "confirm": "false",
            })
        # Nothing deleted.
        with store._state.lock:
            n = store.connection.execute(
                "SELECT COUNT(*) FROM memory_records WHERE memory_id = 'm-k2'"
            ).fetchone()[0]
        assert n == 1

    def test_facade_rejects_identity_claims(self, facade):
        """D4: requested_by/user_scope are server-set."""
        f, _store = facade
        ctx = self._ctx()
        with pytest.raises(Exception):
            f.execute(ctx, "erase_request", {
                "subject": "Kyle", "mode": "preview",
                "requested_by": "someone-else",
            })

    def test_facade_records_requested_by_principal(self, facade):
        f, store = facade
        _seed(store, "m-k3", "personal_fact", "Kyle drives a sedan",
              NOW - timedelta(days=1))
        ctx = self._ctx()
        result = f.execute(ctx, "erase_request", {
            "subject": "Kyle", "mode": "apply", "confirm": True,
        })
        assert result["erased_count"] == 1
        receipts = store.list_deletion_receipts(
            request_id=result["request_id"],
        )
        assert receipts
        assert receipts[0]["requested_by"] == "ops-agent"


class TestSchemaMigrationReceipts:
    """The deletion_receipts table ships via the #288 migration runner."""

    def test_schema_version_is_3(self, store):
        assert store.get_schema_version() == 3

    def test_receipts_table_exists_and_append_only_shape(self, store):
        with store._state.lock:
            cols = store.connection.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'deletion_receipts'"
            ).fetchall()
        names = {r[0] for r in cols}
        assert {
            "receipt_id", "request_id", "subject", "memory_id",
            "content_hash", "category", "user_scope", "requested_by",
            "reason", "outcome", "details", "created_at",
        } <= names

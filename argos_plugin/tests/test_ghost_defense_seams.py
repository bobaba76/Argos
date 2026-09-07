"""#353: ghost-defense seam audit — every write path consults BOTH gates.

The tombstone/rejection gates ("a re-fed fact neither lands nor reaches
the reviewer") were enforced on ``remember()`` and ``save_candidate()``
only. Two other seams could resurrect a ghost:

1. ``import_portable`` checked tombstones but NOT the rejection ledger —
   replaying an export taken BEFORE a rejection decision restored the
   rejected claim (or a paraphrase of it) as active memory.
2. ``update_memory`` (and ``ingest_versioned``'s supersede branch) ran
   the inbound scan but neither gate — a restatement of a live record
   landed a new value on a rejected slot / tombstoned content.

Run with (Hermes venv python, hermetic):
    ARGOS_HERMETIC_TESTS=1 python -m pytest tests/test_ghost_defense_seams.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_plugin_dir = Path(__file__).resolve().parent.parent
if str(_plugin_dir) not in sys.path:
    sys.path.insert(0, str(_plugin_dir))

from store import DuckDBMemoryStore  # noqa: E402


@pytest.fixture
def store(tmp_path):
    s = DuckDBMemoryStore(tmp_path / "a.duckdb", user_id="alice", embedder=None)
    yield s
    s.close()


@pytest.fixture
def store_b(tmp_path):
    s = DuckDBMemoryStore(tmp_path / "b.duckdb", user_id="alice", embedder=None)
    yield s
    s.close()


def _reject_slot(store, category: str, payload: dict) -> None:
    with store._state.lock:
        store.record_rejection(category, payload, reason="review_rejected")


def _count_like(store, needle: str) -> int:
    with store._state.lock:
        return store.connection.execute(
            "SELECT COUNT(*) FROM memory_records WHERE content LIKE ?",
            [f"%{needle}%"],
        ).fetchone()[0]


class TestImportRejectionGate:
    """import_portable must consult the TARGET store's rejection ledger."""

    def test_replay_of_pre_rejection_export_is_rejection_blocked(self, store):
        rec = store.remember(
            category="personal_fact", content="Alice is forty years old",
            payload={"attribute": "age"},
        )
        assert rec is not None
        keep = store.remember(
            category="personal_fact", content="Alice lives in Lisbon",
            payload={"attribute": "location"},
        )
        assert keep is not None
        exp = store.export_portable()

        # Reject the slot AFTER the export. Remove the row WITHOUT a
        # tombstone (raw SQL — delete_memory would tombstone the exact
        # text and mask the rejection gate) so only the ledger stands
        # between the replay and resurrection.
        with store._state.lock:
            store.connection.execute(
                "DELETE FROM memory_records WHERE memory_id = ?", [rec.memory_id],
            )
        _reject_slot(store, "personal_fact", {"attribute": "age"})
        assert _count_like(store, "forty") == 0

        preview = store.import_portable(exp["jsonl"], mode="preview")
        outcomes = {r["identity"]: r["outcome"]
                    for r in preview["rows"] if r["type"] == "record"}
        assert outcomes[rec.memory_id] == "rejection_blocked"
        assert outcomes[keep.memory_id] == "unchanged"
        assert preview["rejection_blocked"] == 1
        assert preview["wrote"] is False

        rep = store.import_portable(exp["jsonl"], mode="apply", confirm=True)
        assert rep["rejection_blocked"] == 1
        assert rep["tombstone_blocked"] == 0
        assert _count_like(store, "forty") == 0, \
            "rejected claim resurrected by replaying an old export"
        assert _count_like(store, "Lisbon") == 1

    def test_paraphrase_of_rejected_slot_is_blocked_on_import(self, store, store_b):
        # Source store holds a paraphrase of a claim the target rejected.
        rec = store.remember(
            category="personal_fact", content="Alice turned 40 last spring",
            payload={"attribute": "age"},
        )
        assert rec is not None
        exp = store.export_portable()

        _reject_slot(store_b, "personal_fact", {"attribute": "age"})
        rep = store_b.import_portable(exp["jsonl"], mode="apply", confirm=True)
        blocked = [r for r in rep["rows"]
                   if r["type"] == "record" and r["outcome"] == "rejection_blocked"]
        assert [b["identity"] for b in blocked] == [rec.memory_id]
        assert _count_like(store_b, "turned 40") == 0

    def test_unrelated_slot_still_imports(self, store, store_b):
        rec = store.remember(
            category="personal_fact", content="Alice lives in Lisbon",
            payload={"attribute": "location"},
        )
        assert rec is not None
        exp = store.export_portable()
        _reject_slot(store_b, "personal_fact", {"attribute": "age"})
        rep = store_b.import_portable(exp["jsonl"], mode="apply", confirm=True)
        assert rep["rejection_blocked"] == 0
        assert rep["restored"] >= 1
        assert _count_like(store_b, "Lisbon") == 1

    def test_purged_rejection_allows_import(self, store, store_b):
        rec = store.remember(
            category="personal_fact", content="Alice is forty years old",
            payload={"attribute": "age"},
        )
        assert rec is not None
        exp = store.export_portable()
        _reject_slot(store_b, "personal_fact", {"attribute": "age"})
        assert store_b.purge_rejection("personal_fact", {"attribute": "age"})
        rep = store_b.import_portable(exp["jsonl"], mode="apply", confirm=True)
        assert rep["rejection_blocked"] == 0
        assert _count_like(store_b, "forty") == 1


class TestUpdateGhostGate:
    """update_memory / ingest_versioned supersede must not re-land ghosts."""

    def test_update_onto_rejected_slot_is_refused(self, store):
        rec = store.remember(
            category="personal_fact", content="Alice is forty years old",
            payload={"attribute": "age"},
        )
        assert rec is not None
        _reject_slot(store, "personal_fact", {"attribute": "age"})

        with pytest.raises(ValueError, match="rejected"):
            store.update_memory(rec.memory_id, content="Alice is forty-one years old")

        # No new version was minted; the head is untouched.
        head = store._fetch_records(
            "SELECT * FROM memory_records WHERE memory_id = ?", [rec.memory_id],
        )[0]
        assert head.valid_to is None and head.superseded_by is None
        assert _count_like(store, "forty-one") == 0

    def test_update_payload_into_rejected_slot_is_refused(self, store):
        rec = store.remember(
            category="personal_fact", content="Alice has a fact",
            payload={"attribute": "location"},
        )
        assert rec is not None
        _reject_slot(store, "personal_fact", {"attribute": "age"})
        with pytest.raises(ValueError, match="rejected"):
            store.update_memory(
                rec.memory_id, content="Alice is forty",
                payload_updates={"attribute": "age"},
            )

    def test_update_onto_tombstoned_content_is_refused(self, store):
        ghost = store.remember(category="personal_fact",
                               content="Alice once lived in Porto")
        assert ghost is not None
        store.erase_subject("Porto", mode="apply", confirm=True)
        assert store.tombstone_check("Alice once lived in Porto", "personal_fact")

        live = store.remember(category="personal_fact",
                              content="Alice lives in Lisbon")
        assert live is not None
        with pytest.raises(ValueError, match="tombstoned"):
            store.update_memory(live.memory_id, content="Alice once lived in Porto")
        assert _count_like(store, "Porto") == 0

    def test_update_on_unrelated_slot_still_works(self, store):
        rec = store.remember(
            category="personal_fact", content="Alice lives in Lisbon",
            payload={"attribute": "location"},
        )
        assert rec is not None
        _reject_slot(store, "personal_fact", {"attribute": "age"})
        new = store.update_memory(rec.memory_id, content="Alice lives in Porto")
        assert new is not None and new.memory_id != rec.memory_id

    def test_ingest_versioned_supersede_reports_blocked(self, store):
        rec = store.remember(
            category="personal_fact",
            content="Alice is forty years old and lives in Lisbon",
            payload={"attribute": "age"},
        )
        assert rec is not None
        _reject_slot(store, "personal_fact", {"attribute": "age"})
        # Substring restatement (>=80% overlap) routes to the supersede
        # branch without an embedder.
        new_head, outcome = store.ingest_versioned(
            "personal_fact",
            "Alice is forty years old and lives in Lisbon!",
            payload={"attribute": "age"},
        )
        assert outcome == "blocked"
        assert new_head is None
        head = store._fetch_records(
            "SELECT * FROM memory_records WHERE memory_id = ?", [rec.memory_id],
        )[0]
        assert head.valid_to is None

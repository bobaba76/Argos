"""Tests for the created_at override in store.remember() (issue #8).

The MemConflict adapter stamps created_at with the ingest wall clock while
sessions carry in-world dates. This makes version-chain/supersession logic
structurally blind — it cannot see that "dating Helen" predates "married
Helen". The fix adds a created_at parameter to remember() so the adapter
can backdate memories to their in-world date.

These tests verify:
1. created_at override sets the created_at column (not the wall clock).
2. valid_from follows created_at (a memory is valid from its in-world date).
3. updated_at stays at the wall clock (the row was physically written now).
4. The as_of temporal filter respects the in-world date.
5. Version chains form in in-world order, not ingest order.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

_plugin_dir = Path(__file__).resolve().parent.parent
if str(_plugin_dir) not in sys.path:
    sys.path.insert(0, str(_plugin_dir))


@pytest.fixture
def store(tmp_path):
    from store import DuckDBMemoryStore
    s = DuckDBMemoryStore(tmp_path / "test.duckdb", user_id="default_user")
    yield s
    s.close()


class TestCreatedAtOverride:
    def test_default_created_at_is_wall_clock(self, store):
        """Without created_at, the wall clock is used (existing behavior)."""
        before = datetime.now(timezone.utc)
        rec = store.remember(category="personal_fact", content="User lives in Springfield",
                             dedup=False)
        after = datetime.now(timezone.utc)
        assert rec is not None
        created = datetime.fromisoformat(rec.created_at)
        assert before - timedelta(seconds=2) <= created <= after + timedelta(seconds=2)

    def test_created_at_override_backdates(self, store):
        """created_at override sets the column to the in-world date."""
        in_world = "2022-01-03T10:00:00+00:00"
        rec = store.remember(category="personal_fact", content="User lives in Bayport",
                             dedup=False, created_at=in_world)
        assert rec is not None
        assert rec.created_at == in_world

    def test_valid_from_follows_created_at(self, store):
        """valid_from = in-world creation time, not ingest time (issue #8)."""
        in_world = "2022-01-03T10:00:00+00:00"
        rec = store.remember(category="personal_fact", content="User works at TechCorp",
                             dedup=False, created_at=in_world)
        assert rec is not None
        assert rec.valid_from == in_world

    def test_updated_at_stays_wall_clock(self, store):
        """updated_at is always the wall clock (the row was physically written now)."""
        in_world = "2022-01-03T10:00:00+00:00"
        before = datetime.now(timezone.utc)
        rec = store.remember(category="personal_fact", content="User drives a Toyota",
                             dedup=False, created_at=in_world)
        after = datetime.now(timezone.utc)
        assert rec is not None
        updated = datetime.fromisoformat(rec.updated_at)
        assert before - timedelta(seconds=2) <= updated <= after + timedelta(seconds=2)

    def test_as_of_filter_respects_in_world_date(self, store):
        """as_of temporal filter uses the in-world date, not ingest time."""
        # Ingest a memory dated 2022-01-03.
        store.remember(category="personal_fact", content="User lives in Springfield",
                       dedup=False, created_at="2022-01-03T10:00:00+00:00")
        # Query as_of 2022-01-01 (before the memory's in-world date) → not visible.
        results_before = store.search("where does the user live", limit=10,
                                      as_of="2022-01-01T00:00:00+00:00")
        assert not any("Springfield" in r.content for r in results_before)
        # Query as_of 2022-01-04 (after) → visible.
        results_after = store.search("where does the user live", limit=10,
                                     as_of="2022-01-04T00:00:00+00:00")
        assert any("Springfield" in r.content for r in results_after)


class TestVersionChainOrder:
    def test_in_world_order_not_ingest_order(self, store):
        """Two memories about the same subject, ingested in reverse in-world
        order, must have valid_from reflecting in-world order — so supersession
        and chain-unfold see the correct temporal sequence.

        Simulates the #8 scenario: "married Helen" (2022-06-01) is ingested
        BEFORE "dating Helen" (2022-01-01), but the in-world dates must
        determine the version chain, not the ingest order.
        """
        # Ingest "married Helen" first (later in-world date).
        rec_married = store.remember(
            category="relationship", content="User is married to Helen",
            dedup=False, created_at="2022-06-01T10:00:00+00:00",
        )
        # Ingest "dating Helen" second (earlier in-world date).
        rec_dating = store.remember(
            category="relationship", content="User is dating Helen",
            dedup=False, created_at="2022-01-01T10:00:00+00:00",
        )
        assert rec_married is not None and rec_dating is not None
        # valid_from reflects in-world order, not ingest order.
        assert rec_dating.valid_from < rec_married.valid_from
        # created_at also reflects in-world order.
        assert rec_dating.created_at < rec_married.created_at

    def test_chronological_recall_uses_in_world_date(self, store):
        """The --chrono lever in the adapter sorts by created_at; with the
        fix, it sorts by in-world date, so the answerer sees old→new version
        structure. Verify the store's created_at is the in-world date."""
        store.remember(category="personal_fact", content="User lives in Springfield",
                       dedup=False, created_at="2022-01-01T00:00:00+00:00")
        store.remember(category="personal_fact", content="User lives in Bayport",
                       dedup=False, created_at="2022-06-01T00:00:00+00:00")
        results = store.search("where does the user live", limit=10)
        # Sort by created_at (what --chrono does).
        chrono = sorted(results, key=lambda r: r.created_at)
        # Springfield (Jan) comes before Bayport (Jun) in in-world order.
        assert "Springfield" in chrono[0].content
        assert "Bayport" in chrono[1].content

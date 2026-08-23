"""Tests for Spec 1 — TTL expiry tiers / best-before dates.

Covers:
- remember(expires_at=...) explicit override
- remember(durability="temporary") auto-TTL from configurable map
- update_memory(expires_at=None) clears expiry (revive)
- update_memory(expires_at=...) sets new expiry
- search(include_expired=True) returns expired memories
- search default excludes expired
- consolidate() reports expired/expiring_soon counts
- reviewer.suggest_expiry() deterministic rules
- Config: expiry_enabled=False preserves old behavior exactly

Run with:
    python -m pytest tests/test_ttl_expiry.py -v
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

_plugin_dir = Path(__file__).resolve().parent.parent
if str(_plugin_dir) not in sys.path:
    sys.path.insert(0, str(_plugin_dir))


# ---------------------------------------------------------------------------
# remember() — explicit expires_at
# ---------------------------------------------------------------------------

class TestRememberExplicitExpiry:
    def test_explicit_expires_at_string(self, tmp_path):
        from store import DuckDBMemoryStore
        store = DuckDBMemoryStore(tmp_path / "test.duckdb", user_id="test_user")
        rec = store.remember(
            category="context_note",
            content="User is on leave for 2 weeks",
            expires_at="2026-12-31T23:59:59+00:00",
        )
        assert rec is not None
        assert rec.expires_at == "2026-12-31T23:59:59+00:00"
        store.close()

    def test_explicit_none_clears_expiry(self, tmp_path):
        """expires_at=None means no expiry (skip auto-TTL)."""
        from store import DuckDBMemoryStore
        store = DuckDBMemoryStore(tmp_path / "test.duckdb", user_id="test_user")
        # context_note is temporary by default → would get auto-TTL.
        # But expires_at=None explicitly says "no expiry".
        rec = store.remember(
            category="context_note",
            content="User is working on a project",
            expires_at=None,
        )
        assert rec is not None
        assert rec.expires_at is None
        store.close()

    def test_default_no_expires_at_uses_auto_ttl(self, tmp_path):
        """Without expires_at, temporary categories get auto-TTL."""
        from store import DuckDBMemoryStore
        store = DuckDBMemoryStore(tmp_path / "test.duckdb", user_id="test_user")
        rec = store.remember(
            category="context_note",
            content="User has a meeting today",
        )
        assert rec is not None
        # context_note is temporary → gets 30-day TTL by default.
        assert rec.expires_at is not None
        store.close()

    def test_durable_category_no_auto_ttl(self, tmp_path):
        """personal_fact is durable → no auto-TTL."""
        from store import DuckDBMemoryStore
        store = DuckDBMemoryStore(tmp_path / "test.duckdb", user_id="test_user")
        rec = store.remember(
            category="personal_fact",
            content="User is 38 years old",
        )
        assert rec is not None
        assert rec.expires_at is None
        store.close()


# ---------------------------------------------------------------------------
# remember() — configurable TTL map
# ---------------------------------------------------------------------------

class TestConfigurableTTL:
    def test_expiry_enabled_uses_config_ttl_map(self, tmp_path):
        from store import DuckDBMemoryStore
        store = DuckDBMemoryStore(tmp_path / "test.duckdb", user_id="test_user")
        store.expiry_enabled = True
        store.ttl_days = {"context_note": 7, "event": 30, "goal": 90}
        store.expiry_default_days = 14
        rec = store.remember(
            category="context_note",
            content="User is traveling this week",
        )
        assert rec is not None
        assert rec.expires_at is not None
        # Should be ~7 days from now.
        expiry = datetime.fromisoformat(rec.expires_at)
        delta = (expiry - datetime.now(timezone.utc)).total_seconds() / 86400
        assert 6.9 < delta < 7.1
        store.close()

    def test_expiry_disabled_uses_hardcoded_map(self, tmp_path):
        from store import DuckDBMemoryStore
        store = DuckDBMemoryStore(tmp_path / "test.duckdb", user_id="test_user")
        store.expiry_enabled = False
        rec = store.remember(
            category="context_note",
            content="User has a temporary task",
        )
        assert rec is not None
        assert rec.expires_at is not None
        # Hardcoded: context_note = 30 days.
        expiry = datetime.fromisoformat(rec.expires_at)
        delta = (expiry - datetime.now(timezone.utc)).total_seconds() / 86400
        assert 29.9 < delta < 30.1
        store.close()

    def test_ttl_fallback_default_days(self, tmp_path):
        """Category not in TTL map → use expiry_default_days."""
        from store import DuckDBMemoryStore
        store = DuckDBMemoryStore(tmp_path / "test.duckdb", user_id="test_user")
        store.expiry_enabled = True
        store.ttl_days = {"context_note": 7}  # no 'event' entry
        store.expiry_default_days = 45
        rec = store.remember(
            category="event",
            content="User attended a conference",
        )
        assert rec is not None
        assert rec.expires_at is not None
        expiry = datetime.fromisoformat(rec.expires_at)
        delta = (expiry - datetime.now(timezone.utc)).total_seconds() / 86400
        assert 44.9 < delta < 45.1
        store.close()


# ---------------------------------------------------------------------------
# update_memory() — expires_at semantics
# ---------------------------------------------------------------------------

class TestUpdateMemoryExpiry:
    def test_update_clears_expiry_with_none(self, tmp_path):
        """update_memory(expires_at=None) revives an expired memory."""
        from store import DuckDBMemoryStore
        store = DuckDBMemoryStore(tmp_path / "test.duckdb", user_id="test_user")
        rec = store.remember(
            category="context_note",
            content="User has a promo code",
            expires_at="2026-01-01T00:00:00+00:00",  # already expired
        )
        assert rec.expires_at == "2026-01-01T00:00:00+00:00"
        # Revive: clear expiry.
        updated = store.update_memory(rec.memory_id, expires_at=None)
        assert updated is not None
        assert updated.expires_at is None
        # Now it should appear in search.
        results = store.search("promo code", limit=10, suppress_retrieval=True)
        assert updated.memory_id in {r.memory_id for r in results}
        store.close()

    def test_update_sets_new_expiry(self, tmp_path):
        from store import DuckDBMemoryStore
        store = DuckDBMemoryStore(tmp_path / "test.duckdb", user_id="test_user")
        rec = store.remember(
            category="personal_fact",
            content="User lives in Springfield",
        )
        assert rec.expires_at is None
        updated = store.update_memory(
            rec.memory_id,
            expires_at="2026-12-31T23:59:59+00:00",
        )
        assert updated is not None
        assert updated.expires_at == "2026-12-31T23:59:59+00:00"
        store.close()

    def test_update_carries_forward_expiry_when_omitted(self, tmp_path):
        from store import DuckDBMemoryStore
        store = DuckDBMemoryStore(tmp_path / "test.duckdb", user_id="test_user")
        rec = store.remember(
            category="context_note",
            content="User is on a short project",
            expires_at="2026-06-01T00:00:00+00:00",
        )
        # Update content without specifying expires_at → carry forward.
        updated = store.update_memory(rec.memory_id, content="User is on a longer project")
        assert updated is not None
        assert updated.expires_at == "2026-06-01T00:00:00+00:00"
        store.close()


# ---------------------------------------------------------------------------
# search() — include_expired
# ---------------------------------------------------------------------------

class TestSearchIncludeExpired:
    def test_default_excludes_expired(self, tmp_path):
        from store import DuckDBMemoryStore
        store = DuckDBMemoryStore(tmp_path / "test.duckdb", user_id="test_user")
        past = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
        rec = store.remember(
            category="context_note",
            content="User has a unique expired promo code XYZ789",
            expires_at=past,
        )
        results = store.search("unique expired promo code XYZ789", limit=10, suppress_retrieval=True)
        assert rec.memory_id not in {r.memory_id for r in results}
        store.close()

    def test_include_expired_returns_expired(self, tmp_path):
        from store import DuckDBMemoryStore
        store = DuckDBMemoryStore(tmp_path / "test.duckdb", user_id="test_user")
        past = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
        rec = store.remember(
            category="context_note",
            content="User has a unique expired promo code ABC123",
            expires_at=past,
        )
        results = store.search(
            "unique expired promo code ABC123", limit=10,
            suppress_retrieval=True, include_expired=True,
        )
        ids = {r.memory_id for r in results}
        assert rec.memory_id in ids, "include_expired=True must return expired memories"
        store.close()


# ---------------------------------------------------------------------------
# consolidate() — expiry reporting
# ---------------------------------------------------------------------------

class TestConsolidateExpiryReport:
    def test_consolidate_reports_expired_count(self, tmp_path):
        from store import DuckDBMemoryStore
        store = DuckDBMemoryStore(tmp_path / "test.duckdb", user_id="test_user")
        past = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
        store.remember(
            category="context_note",
            content="Expired memory one",
            expires_at=past,
        )
        store.remember(
            category="context_note",
            content="Expired memory two",
            expires_at=past,
        )
        store.remember(
            category="personal_fact",
            content="Active durable memory",
        )
        report = store.consolidate(dry_run=True, max_actions=0)
        assert report["expired_count"] == 2
        assert report["expired_revivable_count"] == 2
        store.close()

    def test_consolidate_reports_expiring_soon(self, tmp_path):
        from store import DuckDBMemoryStore
        store = DuckDBMemoryStore(tmp_path / "test.duckdb", user_id="test_user")
        soon = (datetime.now(timezone.utc) + timedelta(days=3)).isoformat()
        store.remember(
            category="context_note",
            content="Memory expiring soon",
            expires_at=soon,
        )
        report = store.consolidate(dry_run=True, max_actions=0)
        assert report["expiring_soon_count"] == 1
        store.close()


# ---------------------------------------------------------------------------
# reviewer.suggest_expiry() — deterministic rules
# ---------------------------------------------------------------------------

class TestSuggestExpiry:
    def test_durable_category_returns_none(self):
        from reviewer import suggest_expiry
        cand = {"category": "personal_fact", "content": "User is 38 years old"}
        assert suggest_expiry(cand) is None

    def test_preference_returns_none(self):
        from reviewer import suggest_expiry
        cand = {"category": "preference", "content": "User likes dark mode"}
        assert suggest_expiry(cand) is None

    def test_duration_in_content(self):
        from reviewer import suggest_expiry
        cand = {"category": "context_note", "content": "User is on leave for 2 weeks"}
        result = suggest_expiry(cand)
        assert result is not None
        expiry = datetime.fromisoformat(result)
        delta = (expiry - datetime.now(timezone.utc)).total_seconds() / 86400
        assert 13.9 < delta < 14.1  # 2 weeks = 14 days

    def test_fixed_date_in_content(self):
        from reviewer import suggest_expiry
        cand = {"category": "event", "content": "User has a deadline until 15 Dec 2026"}
        result = suggest_expiry(cand)
        assert result is not None
        expiry = datetime.fromisoformat(result)
        assert expiry.year == 2026
        assert expiry.month == 12
        assert expiry.day == 15

    def test_category_ttl_fallback(self):
        from reviewer import suggest_expiry
        cand = {"category": "context_note", "content": "User is working on something"}
        result = suggest_expiry(cand, ttl_days={"context_note": 30})
        assert result is not None
        expiry = datetime.fromisoformat(result)
        delta = (expiry - datetime.now(timezone.utc)).total_seconds() / 86400
        assert 29.9 < delta < 30.1

    def test_default_days_fallback(self):
        from reviewer import suggest_expiry
        cand = {"category": "event", "content": "User attended a meetup"}
        # No 'event' in ttl_days → use default_days.
        result = suggest_expiry(cand, ttl_days={}, default_days=60)
        assert result is not None
        expiry = datetime.fromisoformat(result)
        delta = (expiry - datetime.now(timezone.utc)).total_seconds() / 86400
        assert 59.9 < delta < 60.1

    def test_no_llm_pure_deterministic(self):
        """suggest_expiry must not import or call any LLM client."""
        from reviewer import suggest_expiry
        # Block agent.auxiliary_client to prove no LLM dependency.
        import sys
        saved = sys.modules.get("agent.auxiliary_client")
        sys.modules["agent.auxiliary_client"] = None
        try:
            cand = {"category": "context_note", "content": "User is busy for 3 days"}
            result = suggest_expiry(cand)
            assert result is not None
        finally:
            if saved is not None:
                sys.modules["agent.auxiliary_client"] = saved
            else:
                sys.modules.pop("agent.auxiliary_client", None)


# ---------------------------------------------------------------------------
# Cross-cutting: expiry_enabled=False preserves old behavior
# ---------------------------------------------------------------------------

class TestExpiryDisabledPreservesBehavior:
    def test_no_expiry_attrs_no_crash(self, tmp_path):
        """A fresh store (expiry_enabled=False) must behave exactly as before."""
        from store import DuckDBMemoryStore
        store = DuckDBMemoryStore(tmp_path / "test.duckdb", user_id="test_user")
        # Default: expiry_enabled=False, hardcoded TTL map.
        assert store.expiry_enabled is False
        rec = store.remember(
            category="context_note",
            content="Temporary context note",
        )
        assert rec is not None
        # Gets the hardcoded 30-day TTL.
        assert rec.expires_at is not None
        expiry = datetime.fromisoformat(rec.expires_at)
        delta = (expiry - datetime.now(timezone.utc)).total_seconds() / 86400
        assert 29.9 < delta < 30.1
        store.close()

    def test_search_still_excludes_expired_when_disabled(self, tmp_path):
        from store import DuckDBMemoryStore
        store = DuckDBMemoryStore(tmp_path / "test.duckdb", user_id="test_user")
        past = (datetime.now(timezone.utc) - timedelta(days=5)).isoformat()
        rec = store.remember(
            category="context_note",
            content="Unique expired test marker ZZZ999",
            expires_at=past,
        )
        results = store.search("ZZZ999", limit=10, suppress_retrieval=True)
        assert rec.memory_id not in {r.memory_id for r in results}
        store.close()

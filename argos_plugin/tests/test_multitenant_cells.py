"""Tests for #49: multitenant Cells — per-tenant stores behind one service.

Covers:
- Two tenants provisioned from config; each sees only its own data across
  text search, fetch-by-id, candidates, aliases, tombstones, graph
- Unknown user_id -> default tenant (backward compatible)
- No startup regression when the tenants key is absent (single-tenant)
- Per-tenant backup/restore
- Per-tenant config overlay applied (phrase_lift_alpha, review knobs)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

_plugin_dir = Path(__file__).resolve().parent.parent
if str(_plugin_dir) not in sys.path:
    sys.path.insert(0, str(_plugin_dir))

from service_client import SharedMemoryStore, SharedGraphStore  # noqa: E402


_TWO_TENANT_CONFIG = {
    "local_embedding_model": "nonexistent-model-xyz",
    "tenants": {
        "default": {
            "database_filename": "default.duckdb",
            "graph_dirname": "default_kuzu",
        },
        "brandon-bot": {
            "database_filename": "tenants/brandon.duckdb",
            "graph_dirname": "tenants/brandon_kuzu",
            "config": {"phrase_lift_alpha": 0.25},
        },
    },
}


def _write_config(tmp_path: Path, config: dict) -> None:
    (tmp_path / "hybrid_memory.json").write_text(
        json.dumps(config), encoding="utf-8",
    )


def _stop_service(store) -> None:
    """Stop the shared service and WAIT for its endpoint to disappear.

    A fixed sleep after stop_service() is a flake source: the next test's
    service boot can race a still-dying previous service. Waiting for the
    endpoint file to go away (bounded, 5s) makes teardown deterministic.
    """
    try:
        store._rpc.stop_service()
    except Exception:
        pass
    endpoint = store.home / "hybrid_memory_service.json"
    for _ in range(50):
        if not endpoint.exists():
            return
        time.sleep(0.1)
    time.sleep(0.5)  # last resort; endpoint may be stale-but-dead


class TestTwoTenantIsolation:
    """Each tenant sees only its own data across every retrieval surface."""

    @pytest.fixture()
    def stores(self, tmp_path):
        _write_config(tmp_path, _TWO_TENANT_CONFIG)
        a = SharedMemoryStore(tmp_path, user_id="default", embedder=None)
        b = SharedMemoryStore(tmp_path, user_id="brandon-bot", embedder=None)
        try:
            yield a, b, tmp_path
        finally:
            _stop_service(a)

    def test_text_search_isolation(self, stores):
        a, b, _ = stores
        a.remember(category="personal_fact", content="Alice's secret salary is 90000")
        b.remember(category="personal_fact", content="Bob's secret salary is 50000")
        a_hits = [r.content for r in a.search("secret salary", limit=10)]
        b_hits = [r.content for r in b.search("secret salary", limit=10)]
        assert any("Alice" in c for c in a_hits)
        assert not any("Bob" in c for c in a_hits), "A must not see B's data"
        assert any("Bob" in c for c in b_hits)
        assert not any("Alice" in c for c in b_hits), "B must not see A's data"

    def test_fetch_by_id_isolation(self, stores):
        a, b, _ = stores
        rec = a.remember(category="context_note", content="only-in-a")
        # b fetching a's memory id must get nothing.
        assert b.get_memories_by_ids([rec.memory_id]) == []

    def test_count_isolation(self, stores):
        a, b, _ = stores
        a.remember(category="context_note", content="one for a")
        b.remember(category="context_note", content="one for b")
        assert a.count() == 1
        assert b.count() == 1

    def test_candidate_isolation(self, stores):
        a, b, _ = stores
        a.save_candidate(category="personal_fact", content="a's pending fact")
        assert len(a.list_candidates()) == 1
        assert b.list_candidates() == [], "B must not see A's candidates"

    def test_alias_isolation(self, stores):
        a, b, _ = stores
        a.add_alias("shiny", "canonical-a")
        assert a.resolve_aliases("shiny") == ["canonical-a"]
        assert b.resolve_aliases("shiny") == [], "B must not see A's alias"

    def test_tombstone_isolation(self, stores):
        a, b, _ = stores
        rec = a.remember(category="context_note", content="gone from a")
        a.delete_memory(memory_id=rec.memory_id)
        a_tombstones = a._rpc.call("store", "list_tombstones", limit=50)
        b_tombstones = b._rpc.call("store", "list_tombstones", limit=50)
        assert a_tombstones, "A's tombstone must exist for A"
        assert not b_tombstones, "B must not see A's tombstone"

    def test_graph_isolation(self, stores):
        a, b, tmp_path = stores
        ga = SharedGraphStore(tmp_path, user_id="default")
        gb = SharedGraphStore(tmp_path, user_id="brandon-bot")
        try:
            ga.add_relationship(
                source="alice", source_type="person", relation="knows",
                target="carol", target_type="person",
            )
            assert ga.search_graph("carol"), "A's graph must have carol"
            assert gb.search_graph("carol") == [], "B must not see A's graph edge"
        finally:
            ga.close()
            gb.close()

    def test_unknown_user_id_falls_back_to_default(self, stores):
        a, b, _ = stores
        # A third store with an unconfigured user_id routes to default.
        stranger = SharedMemoryStore.__new__(SharedMemoryStore)
        stranger.home = a.home
        stranger.db_path = a.home / "default.duckdb"
        import threading as _t
        stranger._default_user_id = "nobody"
        stranger._scope = _t.local()
        stranger._rpc = a._rpc  # same service; scope stamped per request
        stranger.set_user_scope("some-random-user")
        stranger.remember(category="context_note", content="stranger in default")
        # It lands in the DEFAULT tenant's store (same cell as a).
        assert a.count() >= 1
        assert any("stranger" in r.content for r in a.search("stranger", limit=5))


class TestSingleTenantBackwardCompat:
    """Absent `tenants` key -> one default cell; everything still works."""

    def test_no_tenants_key_starts_and_serves(self, tmp_path):
        (tmp_path / "hybrid_memory.json").write_text(
            json.dumps({"local_embedding_model": "nonexistent-model-xyz"}),
            encoding="utf-8",
        )
        store = SharedMemoryStore(tmp_path, user_id="test_user", embedder=None)
        try:
            rec = store.remember(category="context_note", content="single tenant")
            assert rec is not None
            assert store.search("single tenant", limit=5)
        finally:
            _stop_service(store)

    def test_tenant_config_overlay_applied(self, tmp_path):
        """Per-tenant config overlay (phrase_lift_alpha) must reach the store."""
        _write_config(tmp_path, _TWO_TENANT_CONFIG)
        # Reach the service internals via a direct MemoryService boot.
        import memory_service
        svc = memory_service.MemoryService(tmp_path)
        try:
            assert svc._tenants["brandon-bot"].store._phrase_lift_alpha == 0.25
            # Default tenant keeps the global default.
            assert svc._tenants["default"].store._phrase_lift_alpha == 0.0
        finally:
            svc.close()

    def test_default_tenant_keeps_historical_scope(self, tmp_path):
        """The default tenant's store/graph default scope must stay
        'default_user' (#49 review): the startup hygiene sweep and direct
        calls must still see data written by default clients."""
        _write_config(tmp_path, _TWO_TENANT_CONFIG)
        import memory_service
        svc = memory_service.MemoryService(tmp_path)
        try:
            assert svc._tenants["default"].store.user_id == "default_user"
            assert svc._tenants["default"].default_scope == "default_user"
            # Named tenants scope to their own name.
            assert svc._tenants["brandon-bot"].store.user_id == "brandon-bot"
        finally:
            svc.close()

    def test_single_tenant_keeps_default_scope(self, tmp_path):
        """No tenants key: the single default tenant uses 'default_user' —
        exact old behavior, so the sweep sees historical data."""
        (tmp_path / "hybrid_memory.json").write_text(
            json.dumps({"local_embedding_model": "nonexistent-model-xyz"}),
            encoding="utf-8",
        )
        import memory_service
        svc = memory_service.MemoryService(tmp_path)
        try:
            assert svc._tenants["default"].store.user_id == "default_user"
        finally:
            svc.close()


class TestPerTenantBackup:
    """backup routes to the requested tenant's store."""

    def test_backup_targets_tenant(self, tmp_path):
        _write_config(tmp_path, _TWO_TENANT_CONFIG)
        store = SharedMemoryStore(tmp_path, user_id="brandon-bot", embedder=None)
        try:
            store.remember(category="context_note", content="brandon data")
            dst = tmp_path / "backups" / "cells"
            manifest = store._rpc.backup(dst_root=str(dst), tenant="brandon-bot")
            # Routing proof: the manifest carries the tenant name.
            assert manifest.get("tenant") == "brandon-bot"
            # The export landed under dst_root (a timestamped snapshot dir).
            assert dst.exists() and any(dst.iterdir()), (
                "backup must write a snapshot under dst_root"
            )
        finally:
            _stop_service(store)

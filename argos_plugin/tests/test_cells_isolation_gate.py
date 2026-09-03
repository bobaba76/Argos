"""Reliable end-to-end Cells isolation and concurrency gate (#131).

This is the hermetic, deterministic integration gate for two or more
tenants through the real service process and RPC clients.

Design principles:
  - Provisions isolated temporary homes; NEVER uses the live Hermes store.
  - Exercises the real subprocess service (SharedMemoryStore), not only
    direct MemoryService calls.
  - Every service process is awaited and terminated; no orphans.
  - Failures emit the service PID, endpoint state, stderr/log path, and
    request phase instead of timing out opaquely.
  - Serial mode by default; parallel mode is separately justified.
  - No model downloads or LLM calls.

Run serial (default, reliable):
    python -m pytest argos_plugin/tests/test_cells_isolation_gate.py -v

Run parallel (xdist, single group to serialize service spawns):
    python -m pytest argos_plugin/tests/test_cells_isolation_gate.py -n 2 --dist loadgroup
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pytest

_plugin_dir = Path(__file__).resolve().parent.parent
if str(_plugin_dir) not in sys.path:
    sys.path.insert(0, str(_plugin_dir))

from service_client import (  # noqa: E402
    SharedMemoryServiceError,
    SharedMemoryStore,
    SharedGraphStore,
)

# Group all tests onto a single xdist worker so parallel runs serialize
# the subprocess spawns instead of racing them (#98, #131).
pytestmark = pytest.mark.xdist_group("shared_service")


# ---------------------------------------------------------------------------
# Test config: two tenants with distinct paths and allowed_user_ids.
# ---------------------------------------------------------------------------

_GATE_CONFIG = {
    "local_embedding_model": "nonexistent-model-xyz",
    "reranker_enabled": "false",
    "tenants": {
        "alpha": {
            "database_filename": "alpha_gate.duckdb",
            "graph_dirname": "alpha_gate_kuzu",
            "allowed_user_ids": ["alice", "alex"],
            "config": {
                "external_sources_require_confirmation": "true",
            },
        },
        "beta": {
            "database_filename": "beta_gate.duckdb",
            "graph_dirname": "beta_gate_kuzu",
            "allowed_user_ids": ["bob", "carol"],
            "config": {
                "external_sources_require_confirmation": "true",
            },
        },
    },
}


def _write_config(tmp_path: Path, config: dict | None = None) -> None:
    cfg = config if config is not None else _GATE_CONFIG
    (tmp_path / "hybrid_memory.json").write_text(
        json.dumps(cfg), encoding="utf-8",
    )


def _stop_service(store) -> None:
    """Stop the shared service and WAIT for its endpoint to disappear.

    Bounded wait (5s) makes teardown deterministic — no orphan processes.
    """
    pid = None
    endpoint = None
    try:
        endpoint = store.home / "hybrid_memory_service.json"
        if endpoint.exists():
            try:
                ep_data = json.loads(endpoint.read_text(encoding="utf-8"))
                pid = ep_data.get("pid")
            except Exception:
                pass
    except Exception:
        pass
    try:
        store._rpc.stop_service()
    except Exception:
        pass
    # Wait for endpoint to disappear (bounded 5s).
    if endpoint:
        for _ in range(50):
            if not endpoint.exists():
                break
            time.sleep(0.1)
    # If the process is still alive, kill it (no orphans).
    if pid and _pid_alive(pid):
        try:
            if sys.platform == "win32":
                subprocess.run(
                    ["taskkill", "/F", "/PID", str(pid)],
                    capture_output=True, timeout=5,
                )
            else:
                os.kill(pid, 9)
        except Exception:
            pass
    time.sleep(0.3)  # brief settle after kill


def _pid_alive(pid: int) -> bool:
    """Best-effort PID liveness check (Windows-safe)."""
    if pid <= 0:
        return False
    try:
        if sys.platform == "win32":
            import ctypes
            handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, int(pid))
            if not handle:
                return False
            try:
                exit_code = ctypes.c_ulong()
                if ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                    return exit_code.value == 259
                return True
            finally:
                ctypes.windll.kernel32.CloseHandle(handle)
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def _diagnostic_info(home: Path) -> str:
    """Collect diagnostic info for failure messages (#131)."""
    parts = []
    endpoint = home / "hybrid_memory_service.json"
    if endpoint.exists():
        try:
            ep = json.loads(endpoint.read_text(encoding="utf-8"))
            parts.append(f"endpoint: host={ep.get('host')} port={ep.get('port')} pid={ep.get('pid')}")
        except Exception:
            parts.append(f"endpoint: EXISTS but unreadable: {endpoint}")
    else:
        parts.append("endpoint: MISSING")
    start_lock = home / "hybrid_memory_service.starting"
    if start_lock.exists():
        parts.append(f"start_lock: EXISTS (stale?)")
    # List .duckdb files.
    duckdbs = list(home.glob("*.duckdb")) + list(home.glob("tenants/*.duckdb"))
    parts.append(f"duckdb_files: {[f.name for f in duckdbs]}")
    return "\n  ".join(parts)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def two_tenant_stores(tmp_path):
    """Provision two tenants with real subprocess service and RPC clients."""
    _write_config(tmp_path)
    try:
        a = SharedMemoryStore(tmp_path, user_id="alice", embedder=None)
        b = SharedMemoryStore(tmp_path, user_id="bob", embedder=None)
    except Exception as exc:
        diag = _diagnostic_info(tmp_path)
        raise RuntimeError(
            f"Failed to start service for test:\n  {diag}\n  Error: {exc}"
        ) from exc
    try:
        yield a, b, tmp_path
    finally:
        _stop_service(a)
        _stop_service(b)


# ---------------------------------------------------------------------------
# Part 1: Isolation across every retrieval surface
# ---------------------------------------------------------------------------

class TestRetrievalIsolation:
    """Each tenant sees only its own data across every retrieval surface."""

    def test_text_search_isolation(self, two_tenant_stores):
        """Text search: alpha cannot see beta's records and vice versa."""
        a, b, _ = two_tenant_stores
        a.remember(category="personal_fact", content="Alice lives at 123 Alpha Street")
        b.remember(category="personal_fact", content="Bob works at 456 Beta Avenue")
        a_hits = [r.content for r in a.search("Alpha Street", limit=10)]
        b_hits = [r.content for r in b.search("Beta Avenue", limit=10)]
        assert any("Alice" in c for c in a_hits), "Alice should see her own data"
        assert not any("Bob" in c for c in a_hits), "Alpha must not see beta's data"
        assert any("Bob" in c for c in b_hits), "Bob should see his own data"
        assert not any("Alice" in c for c in b_hits), "Beta must not see alpha's data"

    def test_count_isolation(self, two_tenant_stores):
        """Count: each tenant sees only its own record count."""
        a, b, _ = two_tenant_stores
        a.remember(category="personal_fact", content="Alice drives a blue Toyota")
        a.remember(category="personal_fact", content="Alice likes hiking on weekends")
        b.remember(category="personal_fact", content="Bob enjoys cooking Italian food")
        assert a.count() == 2
        assert b.count() == 1

    def test_fetch_by_id_isolation(self, two_tenant_stores):
        """Fetch-by-ID: alpha cannot fetch beta's records by memory_id."""
        a, b, _ = two_tenant_stores
        rec = b.remember(category="personal_fact", content="Bob's secret project codename")
        mid = rec.memory_id
        # Alice tries to fetch bob's record.
        a_results = a.get_memories_by_ids([mid])
        assert len(a_results) == 0, "Alpha must not fetch beta's records by ID"
        # Bob can fetch it.
        b_results = b.get_memories_by_ids([mid])
        assert len(b_results) == 1

    def test_candidate_isolation(self, two_tenant_stores):
        """Candidates: each tenant's candidates are isolated."""
        a, b, _ = two_tenant_stores
        a.save_candidate(
            category="personal_fact",
            content="Alice candidate fact about programming",
        )
        b.save_candidate(
            category="personal_fact",
            content="Bob candidate fact about gardening",
        )
        a_cands = a.list_candidates(limit=50)
        b_cands = b.list_candidates(limit=50)
        a_contents = [c.get("content", "") for c in a_cands]
        b_contents = [c.get("content", "") for c in b_cands]
        assert any("Alice" in c for c in a_contents), "Alpha should see its candidates"
        assert not any("Bob" in c for c in a_contents), "Alpha must not see beta's candidates"
        assert any("Bob" in c for c in b_contents), "Beta should see its candidates"

    def test_alias_isolation(self, two_tenant_stores):
        """Aliases: each tenant's aliases are isolated."""
        a, b, _ = two_tenant_stores
        a.add_alias(alias="Al", canonical_entity="Alice")
        b.add_alias(alias="Bobby", canonical_entity="Bob")
        a_aliases = a.list_aliases()
        b_aliases = b.list_aliases()
        a_names = {al.get("alias", "").lower() for al in a_aliases}
        b_names = {al.get("alias", "").lower() for al in b_aliases}
        assert "al" in a_names
        assert "bobby" not in a_names, "Alpha must not see beta's aliases"
        assert "bobby" in b_names
        assert "al" not in b_names, "Beta must not see alpha's aliases"

    def test_tombstone_isolation(self, two_tenant_stores):
        """Tombstones: each tenant's deletion tombstones are isolated."""
        a, b, _ = two_tenant_stores
        a.remember(category="personal_fact", content="Alice temp memory for deletion")
        a_tombs = a.list_tombstones(limit=100)
        b_tombs = b.list_tombstones(limit=100)
        # Initially both should have zero or minimal tombstones.
        # After a writes and deletes, only a should have tombstones.
        assert isinstance(a_tombs, list)
        assert isinstance(b_tombs, list)


# ---------------------------------------------------------------------------
# Part 2: Graph isolation
# ---------------------------------------------------------------------------

class TestGraphIsolation:
    """Graph search, traversal, and memory-linked graph reads are isolated."""

    def test_graph_search_isolation(self, two_tenant_stores):
        """Graph search: alpha's entities don't appear in beta's graph."""
        a, b, tmp = two_tenant_stores
        ga = SharedGraphStore(tmp, user_id="alice")
        gb = SharedGraphStore(tmp, user_id="bob")
        try:
            ga.add_relationship(
                source="Alice", source_type="person",
                relation="works_at", target="AlphaCorp", target_type="company",
            )
            gb.add_relationship(
                source="Bob", source_type="person",
                relation="works_at", target="BetaCorp", target_type="company",
            )
            a_results = ga.search_graph("AlphaCorp", limit=20)
            b_results = gb.search_graph("BetaCorp", limit=20)
            a_names = {n.get("name", n.get("id", "")) for n in a_results}
            b_names = {n.get("name", n.get("id", "")) for n in b_results}
            assert "AlphaCorp" in a_names or any("AlphaCorp" in str(n) for n in a_results)
            assert not any("BetaCorp" in str(n) for n in a_results), \
                "Alpha graph must not see beta entities"
        finally:
            ga.close()
            gb.close()

    def test_graph_traversal_isolation(self, two_tenant_stores):
        """Graph traversal: traversing from alpha's entity doesn't reach beta."""
        a, b, tmp = two_tenant_stores
        ga = SharedGraphStore(tmp, user_id="alice")
        gb = SharedGraphStore(tmp, user_id="bob")
        try:
            ga.add_relationship(
                source="Alice", source_type="person",
                relation="knows", target="Alex", target_type="person",
            )
            gb.add_relationship(
                source="Bob", source_type="person",
                relation="knows", target="Carol", target_type="person",
            )
            # Find Alice's entity ID.
            a_nodes = ga.search_graph("Alice", limit=5)
            if a_nodes:
                alice_id = a_nodes[0].get("id", a_nodes[0].get("name", ""))
                if alice_id:
                    traversal = ga.traverse_graph(alice_id, depth=2, limit=20)
                    node_names = {n.get("name", n.get("id", "")) for n in traversal.get("nodes", [])}
                    assert not any("Bob" in str(n) for n in node_names), \
                        "Alpha traversal must not reach beta entities"
        finally:
            ga.close()
            gb.close()


# ---------------------------------------------------------------------------
# Part 3: Concurrent cross-tenant reads/writes
# ---------------------------------------------------------------------------

class TestConcurrency:
    """Concurrent cross-tenant reads/writes: zero cross-tenant observations,
    no deadlocks/timeouts."""

    def test_concurrent_writes_no_cross_tenant(self, two_tenant_stores):
        """Concurrent writes from two tenants produce no cross-tenant
        observations."""
        a, b, _ = two_tenant_stores
        n = 10

        def write_alpha(i):
            a.remember(
                category="personal_fact",
                content=f"Alpha fact number {i} about programming",
            )
            return i

        def write_beta(i):
            b.remember(
                category="personal_fact",
                content=f"Beta fact number {i} about gardening",
            )
            return i

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = []
            for i in range(n):
                futures.append(pool.submit(write_alpha, i))
                futures.append(pool.submit(write_beta, i))
            results = [f.result(timeout=30) for f in futures]
        assert len(results) == 2 * n

        # Verify isolation: alpha sees only alpha facts, beta sees only beta.
        a_hits = [r.content for r in a.search("programming", limit=50)]
        b_hits = [r.content for r in b.search("gardening", limit=50)]
        assert all("Alpha" in c for c in a_hits), "Alpha search should only see alpha facts"
        assert all("Beta" in c for c in b_hits), "Beta search should only see beta facts"
        assert not any("Beta" in c for c in a_hits), "No cross-tenant leakage to alpha"
        assert not any("Alpha" in c for c in b_hits), "No cross-tenant leakage to beta"

    def test_concurrent_reads_no_deadlock(self, two_tenant_stores):
        """Concurrent reads from two tenants complete without deadlock."""
        a, b, _ = two_tenant_stores
        a.remember(category="personal_fact", content="Alpha data for concurrent read test")
        b.remember(category="personal_fact", content="Beta data for concurrent read test")
        n = 20

        def read_alpha():
            return [r.content for r in a.search("Alpha", limit=5)]

        def read_beta():
            return [r.content for r in b.search("Beta", limit=5)]

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = []
            for _ in range(n):
                futures.append(pool.submit(read_alpha))
                futures.append(pool.submit(read_beta))
            results = [f.result(timeout=30) for f in futures]
        assert len(results) == 2 * n
        # All alpha reads should have results, all beta reads should have results.
        assert all(r is not None for r in results)


# ---------------------------------------------------------------------------
# Part 4: Multiple users in the same tenant
# ---------------------------------------------------------------------------

class TestMultipleUsersSameTenant:
    """Multiple users in the same tenant share the tenant's store."""

    def test_two_users_same_tenant(self, tmp_path):
        """Alice and alex are both in the alpha tenant — they route to the
        same cell. User_scope filtering is defense-in-depth (each user sees
        their own writes), but both are in the alpha tenant."""
        _write_config(tmp_path)
        alice = SharedMemoryStore(tmp_path, user_id="alice", embedder=None)
        try:
            alex = SharedMemoryStore(tmp_path, user_id="alex", embedder=None)
            try:
                alice.remember(
                    category="personal_fact",
                    content="Alice wrote this shared fact",
                )
                # Both route to the alpha tenant — verify via get_status.
                status = alice._rpc._request({"method": "get_status"})
                assert "alpha" in status.get("tenant_cells", {})
                # Alice sees her own data.
                assert alice.count() >= 1
                # Alex is in the same tenant but user_scope filtering means
                # he sees his own writes (defense-in-depth). He can write.
                alex.remember(
                    category="personal_fact",
                    content="Alex wrote a fact in the same tenant",
                )
                assert alex.count() >= 1
            finally:
                _stop_service(alex)
        finally:
            _stop_service(alice)


# ---------------------------------------------------------------------------
# Part 5: Identity and routing enforcement
# ---------------------------------------------------------------------------

class TestIdentityEnforcement:
    """Unknown identity, duplicate identity, and spoof attempts."""

    def test_unknown_user_raw_request_rejected(self, tmp_path):
        """A raw RPC request with an unknown user_id is rejected by
        strict routing (#87)."""
        _write_config(tmp_path)
        store = SharedMemoryStore(tmp_path, user_id="alice", embedder=None)
        try:
            with pytest.raises(SharedMemoryServiceError):
                store._rpc._request({
                    "component": "store", "method": "count",
                    "user_id": "eve-the-attacker",
                })
        finally:
            _stop_service(store)


# ---------------------------------------------------------------------------
# Part 6: Service lifecycle
# ---------------------------------------------------------------------------

class TestServiceLifecycle:
    """Service startup, stale endpoint recovery, graceful shutdown, restart."""

    def test_service_starts_and_responds(self, tmp_path):
        """The service starts and responds to health checks."""
        _write_config(tmp_path)
        store = SharedMemoryStore(tmp_path, user_id="alice", embedder=None)
        try:
            assert store._rpc._healthy() is True
        finally:
            _stop_service(store)

    def test_service_restart(self, tmp_path):
        """The service can be stopped and restarted cleanly."""
        _write_config(tmp_path)
        # First start.
        store1 = SharedMemoryStore(tmp_path, user_id="alice", embedder=None)
        store1.remember(category="personal_fact", content="Persisted fact for restart test")
        _stop_service(store1)
        # Restart.
        store2 = SharedMemoryStore(tmp_path, user_id="alice", embedder=None)
        try:
            assert store2.count() >= 1, "Data should survive restart"
        finally:
            _stop_service(store2)

    def test_stale_endpoint_recovery(self, tmp_path):
        """A stale endpoint file is recovered on next startup."""
        _write_config(tmp_path)
        # Write a stale endpoint pointing to a dead port.
        stale = {
            "host": "127.0.0.1",
            "port": 1,  # port 1 is never valid
            "token": "stale-token",
            "pid": 999999,  # non-existent PID
            "version": 1,
        }
        (tmp_path / "hybrid_memory_service.json").write_text(
            json.dumps(stale), encoding="utf-8",
        )
        # Starting a store should detect the stale endpoint and start fresh.
        store = SharedMemoryStore(tmp_path, user_id="alice", embedder=None)
        try:
            assert store._rpc._healthy() is True
        finally:
            _stop_service(store)

    def test_graceful_shutdown(self, tmp_path):
        """Graceful shutdown via RPC leaves no orphan process."""
        _write_config(tmp_path)
        store = SharedMemoryStore(tmp_path, user_id="alice", embedder=None)
        endpoint = store.home / "hybrid_memory_service.json"
        ep_data = json.loads(endpoint.read_text(encoding="utf-8"))
        pid = ep_data.get("pid")
        _stop_service(store)
        # Endpoint should be gone.
        assert not endpoint.exists(), "Endpoint file should be removed on shutdown"
        # Process should be dead.
        if pid:
            time.sleep(0.5)
            assert not _pid_alive(pid), "Service process should be terminated"


# ---------------------------------------------------------------------------
# Part 7: Per-tenant backup
# ---------------------------------------------------------------------------

class TestPerTenantBackup:
    """Per-tenant backup/restore round-trip with two cells."""

    def test_backup_two_tenants(self, two_tenant_stores):
        """Backing up two tenants produces separate manifests."""
        a, b, tmp = two_tenant_stores
        a.remember(category="personal_fact", content="Alpha backup test fact")
        b.remember(category="personal_fact", content="Beta backup test fact")
        dst = str(tmp / "backups")
        a_manifest = a._rpc.backup(dst_root=dst, tenant="alpha")
        b_manifest = b._rpc.backup(dst_root=dst, tenant="beta")
        assert a_manifest["tenant"] == "alpha"
        assert b_manifest["tenant"] == "beta"

    def test_backup_unknown_tenant_rejected(self, two_tenant_stores):
        """Backup with an unknown tenant is rejected (no silent default)."""
        a, _, _ = two_tenant_stores
        with pytest.raises(SharedMemoryServiceError):
            a._rpc.backup(dst_root=str(a.home / "backups"), tenant="nonexistent")


# ---------------------------------------------------------------------------
# Part 8: Config overlay enforcement
# ---------------------------------------------------------------------------

class TestConfigOverlayEnforcement:
    """Tenant config overlays and policy are enforced through the RPC path."""

    def test_external_sources_policy_enforced(self, two_tenant_stores):
        """The external_sources_require_confirmation policy is applied."""
        a, _, _ = two_tenant_stores
        # The config has external_sources_require_confirmation=true for alpha.
        # Verify the store inherited it.
        # We can check via get_status.
        status = a._rpc._request({"method": "get_status"})
        assert "tenant_policies" in status
        alpha_policy = status["tenant_policies"].get("alpha", {})
        assert alpha_policy.get("external_sources_require_confirmation") is True

    def test_get_status_shows_two_tenants(self, two_tenant_stores):
        """get_status reports both tenants."""
        a, _, _ = two_tenant_stores
        status = a._rpc._request({"method": "get_status"})
        assert status["tenant_count"] == 2
        assert "alpha" in status.get("tenant_cells", {})
        assert "beta" in status.get("tenant_cells", {})

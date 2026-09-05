"""Tests for #312: write_access_audit + export_access_audit RPC proxy.

Proves the SharedMemoryStore RPC proxy gap is closed:
(1) SharedMemoryStore.write_access_audit proxies over RPC to the shared
    service, which writes a durable row to the access_audit table.
(2) SharedMemoryStore.export_access_audit proxies over RPC and returns
    the audit rows.
(3) A denial written via the facade (#300) through a SharedMemoryStore
    appears in export_access_audit output — the full end-to-end path.
(4) Rows survive a service restart (close + reopen on the same DB).

Live-mode tests: spawn a real shared memory service subprocess.

Run with (Hermes venv python, hermetic):
    ARGOS_HERMETIC_TESTS=1 python -m pytest tests/test_rpc_access_audit.py -v
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

_plugin_dir = Path(__file__).resolve().parent.parent
for _path in (_plugin_dir.parent, _plugin_dir):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_plugin_dir))

# Group with other shared-service tests so xdist serializes the spawns.
pytestmark = pytest.mark.xdist_group("shared_service")


def _make_store(tmp_path, user_id="test_user"):
    """Create a SharedMemoryStore with a disposable home dir."""
    from service_client import SharedMemoryStore
    (tmp_path / "hybrid_memory.json").write_text(
        json.dumps({"local_embedding_model": "nonexistent-model-xyz"}),
        encoding="utf-8",
    )
    return SharedMemoryStore(tmp_path, user_id=user_id, embedder=None)


class TestWriteAccessAuditRpcProxy:
    """(1) write_access_audit proxies over RPC to the durable table."""

    def test_write_access_audit_proxies_to_service(self, tmp_path):
        """A write_access_audit call through SharedMemoryStore creates a
        durable row in the access_audit table on the shared service."""
        store = _make_store(tmp_path)
        try:
            # Write an audit row via the RPC proxy.
            store.write_access_audit(
                user_id="test_user",
                query_text="search:secret query",
                granted_count=5,
                denied_count=2,
                denied_scopes="project_x",
                excluded=True,
                tenant="default",
            )

            # Export and verify the row is there.
            exported = store.export_access_audit(format="jsonl")
            assert exported, "export_access_audit returned empty"
            rows = [json.loads(line) for line in exported.strip().splitlines() if line]
            assert len(rows) >= 1

            # Find our row (there may be rows from search/remember too).
            our_rows = [r for r in rows if r.get("denied_count") == 2]
            assert len(our_rows) >= 1, "denial row not found in export"
            row = our_rows[0]
            assert row["user_id"] == "test_user"
            assert row["granted_count"] == 5
            assert row["excluded"] is True
            # query_text is hashed (SHA-256, 16 chars) — raw text must NOT appear.
            assert row["query_text"] != "search:secret query"
            assert len(row["query_text"]) == 16
        finally:
            try:
                store._rpc.stop_service()
            finally:
                time.sleep(0.5)

    def test_write_access_audit_signature_matches_store(self):
        """SharedMemoryStore.write_access_audit has the same kwargs as
        DuckDBMemoryStore.write_access_audit (store_retrieval.py)."""
        from service_client import SharedMemoryStore
        import inspect
        sig = inspect.signature(SharedMemoryStore.write_access_audit)
        params = set(sig.parameters.keys()) - {"self"}
        expected = {"user_id", "query_text", "granted_count", "denied_count",
                    "denied_scopes", "excluded", "tenant"}
        assert params == expected, f"Signature mismatch: {params} vs {expected}"


class TestExportAccessAuditRpcProxy:
    """(2) export_access_audit proxies over RPC and returns rows."""

    def test_export_returns_jsonl(self, tmp_path):
        """export_access_audit returns JSONL format by default."""
        store = _make_store(tmp_path)
        try:
            store.write_access_audit(
                user_id="test_user",
                query_text="test query",
                granted_count=1,
                denied_count=0,
            )
            exported = store.export_access_audit()
            assert exported
            # Each line must be valid JSON.
            for line in exported.strip().splitlines():
                if line:
                    row = json.loads(line)
                    assert "audit_id" in row
                    assert "ts" in row
                    assert "user_id" in row
        finally:
            try:
                store._rpc.stop_service()
            finally:
                time.sleep(0.5)

    def test_export_returns_csv(self, tmp_path):
        """export_access_audit supports CSV format."""
        store = _make_store(tmp_path)
        try:
            store.write_access_audit(
                user_id="test_user",
                query_text="csv test",
                granted_count=1,
                denied_count=0,
            )
            exported = store.export_access_audit(format="csv")
            assert exported
            # CSV should have a header row.
            lines = exported.strip().splitlines()
            assert len(lines) >= 2  # header + at least one data row
            assert "audit_id" in lines[0]
        finally:
            try:
                store._rpc.stop_service()
            finally:
                time.sleep(0.5)


class TestFacadeDenialThroughSharedStore:
    """(3) A facade denial through a SharedMemoryStore appears in the
    export_access_audit output — the full end-to-end path from #300."""

    def test_facade_denial_appears_in_export(self, tmp_path):
        """A forbidden-op attempt through the facade, backed by a
        SharedMemoryStore, creates a durable access_audit row that
        appears in export_access_audit."""
        from api_facade import ArgosAPIFacade, AuthContext, APIError, READ_OPERATIONS

        store = _make_store(tmp_path)
        try:
            facade = ArgosAPIFacade(store)
            ctx = AuthContext(
                principal="test-principal",
                tenant="default",
                user_id="test_user",
                transport="rest",
                allowed_operations=set(READ_OPERATIONS),
            )
            # Trigger a forbidden-operation denial.
            with pytest.raises(APIError):
                facade.execute(ctx, "shutdown", {})

            # The denial must appear in the durable audit export.
            exported = store.export_access_audit()
            assert exported, "export_access_audit returned empty"
            rows = [json.loads(line) for line in exported.strip().splitlines() if line]
            denial_rows = [r for r in rows if r.get("excluded") is True]
            assert len(denial_rows) >= 1, "facade denial not found in durable audit"
            row = denial_rows[0]
            assert row["denied_count"] == 1
            assert row["granted_count"] == 0
            assert "forbidden_operation" in (row.get("denied_scopes") or "")
        finally:
            try:
                store._rpc.stop_service()
            finally:
                time.sleep(0.5)


class TestAuditSurvivesRestart:
    """(4) Audit rows survive a service restart (close + reopen)."""

    def test_denial_survives_service_restart(self, tmp_path):
        """A denial written before a service restart is still present
        after reopening the SharedMemoryStore on the same DB."""
        from api_facade import ArgosAPIFacade, AuthContext, APIError, READ_OPERATIONS

        # Phase 1: write a denial via the facade.
        store1 = _make_store(tmp_path)
        try:
            facade = ArgosAPIFacade(store1)
            ctx = AuthContext(
                principal="test-principal",
                tenant="default",
                user_id="test_user",
                transport="rest",
                allowed_operations=set(READ_OPERATIONS),
            )
            with pytest.raises(APIError):
                facade.execute(ctx, "shutdown", {})
        finally:
            try:
                store1._rpc.stop_service()
            finally:
                time.sleep(1.0)

        # Phase 2: reopen on the same DB — the denial must still be there.
        store2 = _make_store(tmp_path)
        try:
            exported = store2.export_access_audit()
            assert exported, "export_access_audit returned empty after restart"
            rows = [json.loads(line) for line in exported.strip().splitlines() if line]
            denial_rows = [r for r in rows if r.get("excluded") is True]
            assert len(denial_rows) >= 1, "denial row lost after service restart"
            assert any(
                "forbidden_operation" in (r.get("denied_scopes") or "")
                for r in denial_rows
            )
        finally:
            try:
                store2._rpc.stop_service()
            finally:
                time.sleep(0.5)

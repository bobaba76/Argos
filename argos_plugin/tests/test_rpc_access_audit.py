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


# -- #200 PR 2/3 fix: collection write gating at the raw-RPC seam -------------

class TestCollectionWriteRpcGating:
    """#200 PR 2/3 fix (second fix): raw RPC collection writes are denied
    at the service dispatch. The gate authority is the _confirmed flag in
    the RPC request ENVELOPE (set by _SharedRPC.call_gated), NOT a client-
    supplied confirm in args. A raw RPC caller cannot forge the envelope
    flag by passing confirm=True in args — _sanitize_args strips confirm
    (it's in _FORBIDDEN_CLIENT_ARGS), and the service only trusts
    _confirmed from the envelope.

    Tests:
      - raw RPC via call() without _confirmed → denied + audit
      - raw RPC via call() WITH forged confirm=True in args → DENIED
      - **forged _confirmed=true in envelope without HMAC → DENIED** (the real spoof-closing test)
      - forged _confirmed=true WITH forged _gate_hmac → DENIED (HMAC verification)
      - forged client user_id/tenant → stripped, service-resolved identity in audit
      - facade-mediated write → succeeds + audit granted row
      - CAS conflict → audit denial row, no write
      - reads stay un-gated
      - _GATED_COLLECTION_WRITE_METHODS is exactly the 4 writes
    """

    def test_raw_rpc_create_denied(self, tmp_path):
        """Raw RPC create_collection via call() (no _confirmed envelope)
        → PermissionError + audit denial row."""
        store = _make_store(tmp_path)
        try:
            from service_client import SharedMemoryServiceError
            with pytest.raises((SharedMemoryServiceError, PermissionError)):
                store._rpc.call("store", "create_collection", name="Raw RPC")
            exported = store.export_access_audit()
            rows = [json.loads(line) for line in exported.strip().splitlines() if line]
            denial_rows = [
                r for r in rows
                if r.get("excluded") is True
                and "not_confirmed" in (r.get("denied_scopes") or "")
            ]
            assert len(denial_rows) >= 1, "raw RPC denial not audited"
        finally:
            try:
                store._rpc.stop_service()
            finally:
                time.sleep(0.5)

    def test_forged_confirmed_in_envelope_denied(self, tmp_path):
        """THE REAL SPOOF-CLOSING TEST: a raw RPC caller forges
        _confirmed=true in the envelope WITHOUT a valid HMAC → DENIED.

        This proves the gate authority is the server-verified HMAC, not
        the client-asserted _confirmed boolean. A raw caller with the
        endpoint token can set _confirmed=true in the JSON, but without
        the boot-time gate_secret they cannot compute the correct HMAC.
        The service strips _confirmed when the HMAC doesn't match.
        """
        store = _make_store(tmp_path)
        try:
            from service_client import SharedMemoryServiceError
            # Forge _confirmed=true in the envelope with NO HMAC.
            with pytest.raises((SharedMemoryServiceError, PermissionError)):
                store._rpc._request_once(
                    {
                        "component": "store",
                        "method": "create_collection",
                        "args": {"name": "Forged Envelope"},
                        "_confirmed": True,
                    },
                    timeout=10.0,
                )
            # Audit denial row written.
            exported = store.export_access_audit()
            rows = [json.loads(line) for line in exported.strip().splitlines() if line]
            denial_rows = [
                r for r in rows
                if r.get("excluded") is True
                and "not_confirmed" in (r.get("denied_scopes") or "")
            ]
            assert len(denial_rows) >= 1, "forged _confirmed denial not audited"
        finally:
            try:
                store._rpc.stop_service()
            finally:
                time.sleep(0.5)

    def test_forged_confirmed_with_forged_hmac_denied(self, tmp_path):
        """Forged _confirmed=true WITH a forged _gate_hmac (wrong secret)
        → DENIED. The HMAC verification catches a caller who tries to
        guess the gate_secret."""
        store = _make_store(tmp_path)
        try:
            from service_client import SharedMemoryServiceError
            import hashlib as _hashlib
            import hmac as _hmac
            # Compute an HMAC with the WRONG secret.
            fake_secret = "wrong-secret-not-the-real-one"
            body = {
                "component": "store",
                "method": "create_collection",
                "args": {"name": "Forged HMAC"},
                "_confirmed": True,
                "v": 1,
                "user_id": "test_user",
            }
            fake_hmac = _hmac.new(
                fake_secret.encode("utf-8"),
                json.dumps(body, sort_keys=True).encode("utf-8"),
                _hashlib.sha256,
            ).hexdigest()
            with pytest.raises((SharedMemoryServiceError, PermissionError)):
                store._rpc._request_once(
                    {
                        "component": "store",
                        "method": "create_collection",
                        "args": {"name": "Forged HMAC"},
                        "_confirmed": True,
                        "_gate_hmac": fake_hmac,
                    },
                    timeout=10.0,
                )
            exported = store.export_access_audit()
            rows = [json.loads(line) for line in exported.strip().splitlines() if line]
            denial_rows = [
                r for r in rows
                if r.get("excluded") is True
                and "not_confirmed" in (r.get("denied_scopes") or "")
            ]
            assert len(denial_rows) >= 1, "forged HMAC denial not audited"
        finally:
            try:
                store._rpc.stop_service()
            finally:
                time.sleep(0.5)

    def test_forged_confirm_in_args_still_denied(self, tmp_path):
        """THE SPOOF-CLOSING TEST: raw RPC with forged confirm=True in args
        → DENIED. The gate authority is _confirmed in the ENVELOPE, not
        confirm in args. _sanitize_args strips confirm (in
        _FORBIDDEN_CLIENT_ARGS), so it never reaches the gate logic."""
        store = _make_store(tmp_path)
        try:
            from service_client import SharedMemoryServiceError
            # Forge confirm=True in args — must be DENIED.
            with pytest.raises((SharedMemoryServiceError, PermissionError)):
                store._rpc.call(
                    "store", "create_collection",
                    name="Forged Confirm", confirm=True,
                )
            # Audit denial row written with not_confirmed reason.
            exported = store.export_access_audit()
            rows = [json.loads(line) for line in exported.strip().splitlines() if line]
            denial_rows = [
                r for r in rows
                if r.get("excluded") is True
                and "not_confirmed" in (r.get("denied_scopes") or "")
            ]
            assert len(denial_rows) >= 1, "forged confirm denial not audited"
        finally:
            try:
                store._rpc.stop_service()
            finally:
                time.sleep(0.5)

    def test_forged_confirm_on_add_item_denied(self, tmp_path):
        """Forged confirm=True on add_collection_item → DENIED."""
        store = _make_store(tmp_path)
        try:
            from service_client import SharedMemoryServiceError
            # Create a collection via the gated path first.
            store.create_collection(name="Valid")
            col = store.list_collections()[0]
            # Forged confirm=True in args → denied.
            with pytest.raises((SharedMemoryServiceError, PermissionError)):
                store._rpc.call(
                    "store", "add_collection_item",
                    collection_id=col["collection_id"],
                    fields={"title": "forged"},
                    confirm=True,
                )
        finally:
            try:
                store._rpc.stop_service()
            finally:
                time.sleep(0.5)

    def test_forged_confirm_on_update_denied(self, tmp_path):
        """Forged confirm=True on update_collection_item → DENIED."""
        store = _make_store(tmp_path)
        try:
            from service_client import SharedMemoryServiceError
            store.create_collection(name="Valid")
            col = store.list_collections()[0]
            store.add_collection_item(
                collection_id=col["collection_id"], fields={"title": "item"},
            )
            items = store.list_collection_items(collection_id=col["collection_id"])
            item_id = items[0]["item_id"]
            with pytest.raises((SharedMemoryServiceError, PermissionError)):
                store._rpc.call(
                    "store", "update_collection_item",
                    item_id=item_id, status="done", confirm=True,
                )
        finally:
            try:
                store._rpc.stop_service()
            finally:
                time.sleep(0.5)

    def test_forged_confirm_on_remove_denied(self, tmp_path):
        """Forged confirm=True on remove_collection_item → DENIED."""
        store = _make_store(tmp_path)
        try:
            from service_client import SharedMemoryServiceError
            store.create_collection(name="Valid")
            col = store.list_collections()[0]
            store.add_collection_item(
                collection_id=col["collection_id"], fields={"title": "item"},
            )
            items = store.list_collection_items(collection_id=col["collection_id"])
            item_id = items[0]["item_id"]
            with pytest.raises((SharedMemoryServiceError, PermissionError)):
                store._rpc.call(
                    "store", "remove_collection_item",
                    item_id=item_id, confirm=True,
                )
        finally:
            try:
                store._rpc.stop_service()
            finally:
                time.sleep(0.5)

    def test_forged_client_identity_stripped(self, tmp_path):
        """Raw RPC with forged user_id/tenant in args → stripped by
        _sanitize_args; service-resolved identity used in audit."""
        store = _make_store(tmp_path, user_id="real_user")
        try:
            from service_client import SharedMemoryServiceError
            # Forge user_id and tenant in the args — _sanitize_args strips them.
            with pytest.raises((SharedMemoryServiceError, PermissionError)):
                store._rpc.call(
                    "store", "create_collection",
                    name="Forged", user_id="attacker", tenant="evil",
                    confirm=True,
                )
            # The denial audit row must use the service-resolved user_id
            # (real_user), not the forged "attacker".
            exported = store.export_access_audit()
            rows = [json.loads(line) for line in exported.strip().splitlines() if line]
            denial_rows = [
                r for r in rows
                if r.get("excluded") is True
                and "not_confirmed" in (r.get("denied_scopes") or "")
            ]
            assert len(denial_rows) >= 1
            assert denial_rows[0]["user_id"] == "real_user"
            assert denial_rows[0]["user_id"] != "attacker"
        finally:
            try:
                store._rpc.stop_service()
            finally:
                time.sleep(0.5)

    def test_facade_collection_write_succeeds_with_audit(self, tmp_path):
        """Facade-mediated collection write → succeeds + audit granted row.
        The facade uses call_gated() which sets _confirmed in the envelope."""
        from api_facade import (
            ArgosAPIFacade, AuthContext,
            COLLECTION_READ_OPERATIONS, COLLECTION_WRITE_OPERATIONS,
            READ_OPERATIONS, WRITE_OPERATIONS,
        )
        store = _make_store(tmp_path)
        try:
            facade = ArgosAPIFacade(store)
            ctx = AuthContext(
                principal="test-principal",
                tenant="default",
                user_id="test_user",
                transport="loopback",
                allowed_operations=(
                    READ_OPERATIONS | WRITE_OPERATIONS
                    | COLLECTION_READ_OPERATIONS | COLLECTION_WRITE_OPERATIONS
                ),
                is_loopback=True,
            )
            # Facade-mediated create → succeeds (call_gated sets _confirmed).
            r = facade.execute(ctx, "collection_create", {"name": "Via Facade"})
            assert r["status"] == "created"
            # Audit granted row written.
            exported = store.export_access_audit()
            rows = [json.loads(line) for line in exported.strip().splitlines() if line]
            granted = [r for r in rows if r.get("granted_count", 0) > 0 and not r.get("excluded")]
            assert len(granted) >= 1, "facade collection write not audited as granted"
        finally:
            try:
                store._rpc.stop_service()
            finally:
                time.sleep(0.5)

    def test_cas_conflict_audited_as_denial(self, tmp_path):
        """CAS conflict on update_collection_item → audit denial row, no write."""
        store = _make_store(tmp_path)
        try:
            from service_client import SharedMemoryServiceError
            # Create collection + item via the gated proxy path.
            store.create_collection(name="CAS Test")
            col = store.list_collections()[0]
            store.add_collection_item(
                collection_id=col["collection_id"], fields={"title": "item"},
            )
            items = store.list_collection_items(collection_id=col["collection_id"])
            item_id = items[0]["item_id"]
            # CAS conflict: wrong expected_version (via gated proxy).
            with pytest.raises((SharedMemoryServiceError, Exception)):
                store.update_collection_item(
                    item_id=item_id,
                    status="done",
                    expected_version="stale-version",
                )
            # Audit denial row for the CAS conflict.
            exported = store.export_access_audit()
            rows = [json.loads(line) for line in exported.strip().splitlines() if line]
            cas_denials = [
                r for r in rows
                if r.get("excluded") is True
                and "cas_conflict" in (r.get("denied_scopes") or "")
            ]
            assert len(cas_denials) >= 1, "CAS conflict denial not audited"
            # The item status must NOT have changed.
            items_after = store.list_collection_items(collection_id=col["collection_id"])
            assert items_after[0]["status"] == "open"
        finally:
            try:
                store._rpc.stop_service()
            finally:
                time.sleep(0.5)

    def test_reads_not_gated(self, tmp_path):
        """Read operations (list/count/get) are NOT gated — they work
        without _confirmed. The gated set is exactly the 4 WRITE methods."""
        store = _make_store(tmp_path)
        try:
            # Create a collection via the gated proxy path first.
            store.create_collection(name="Read Test")
            col = store.list_collections()[0]
            store.add_collection_item(
                collection_id=col["collection_id"], fields={"title": "item"},
            )
            # Reads via call() (no _confirmed) → all succeed (no gating).
            cols = store.list_collections()
            assert len(cols) >= 1
            items = store.list_collection_items(collection_id=col["collection_id"])
            assert len(items) >= 1
            count = store.count_collection_items(collection_id=col["collection_id"])
            assert count >= 1
            got = store.get_collection(collection_id=col["collection_id"])
            assert got is not None
        finally:
            try:
                store._rpc.stop_service()
            finally:
                time.sleep(0.5)

    def test_gated_set_is_exactly_4_writes(self):
        """The _GATED_COLLECTION_WRITE_METHODS set contains exactly the 4
        collection write methods — no reads, no extras."""
        import sys
        _plugin_dir = Path(__file__).resolve().parent.parent
        if str(_plugin_dir) not in sys.path:
            sys.path.insert(0, str(_plugin_dir))
        from memory_service import _GATED_COLLECTION_WRITE_METHODS
        assert _GATED_COLLECTION_WRITE_METHODS == frozenset({
            "create_collection",
            "add_collection_item",
            "update_collection_item",
            "remove_collection_item",
        })

    def test_confirm_in_forbidden_client_args(self):
        """confirm is in _FORBIDDEN_CLIENT_ARGS — _sanitize_args strips it."""
        import sys
        _plugin_dir = Path(__file__).resolve().parent.parent
        if str(_plugin_dir) not in sys.path:
            sys.path.insert(0, str(_plugin_dir))
        from memory_service import _FORBIDDEN_CLIENT_ARGS, _sanitize_args
        assert "confirm" in _FORBIDDEN_CLIENT_ARGS
        # _sanitize_args strips it.
        sanitized = _sanitize_args({"name": "test", "confirm": True})
        assert "confirm" not in sanitized
        assert sanitized == {"name": "test"}

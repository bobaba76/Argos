"""Spec-13 (#393 slice S3): unreviewed-class resolution tests.

Covers:
  - store review_memory: promote clears the marker (rank penalty removed);
    dismiss quarantines + fingerprints the claim slot in the rejection
    ledger (re-assertion blocked only when the slot is identifiable - the
    reassertion_blocked flag tells which); head/scope guards; None for
    unknown ids
  - resolution invariant: only tool/manual review may resolve - anything
    else is refused loudly with a memory_review_refused mutation event
  - mutation events: memory_promoted / memory_dismissed / refused
  - search behavior: penalized before promote, top-ranked after
  - RPC client + service dispatch: server-derived review class (ungated
    callers are forced to auto_review; gated claims are honored)
  - facade: op registration, validation, model-principal denial, scoping
  - MCP surface: tool mapping + strict schema + idempotency membership
  - REST surface: /v1/memories/{id}/decision route (auth, idempotency,
    validation, replay dedup)

S2 lesson applied: exercise the FULL seam (store -> service -> client ->
facade -> transports), not a single layer.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

_plugin_dir = Path(__file__).resolve().parent.parent
for _path in (_plugin_dir.parent, _plugin_dir):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


def _make_store(tmp_path):
    from argos.store import DuckDBMemoryStore

    return DuckDBMemoryStore(tmp_path / "test.duckdb", user_id="test_user")


UNREVIEWED_TEXT = "Michael keeps a spare garage door remote in the kitchen drawer"
SLOT_TEXT = "Michael's harbour project gate code follows the marina pattern"


def _seed_unreviewed(store, **kw):
    return store.remember(
        category=kw.pop("category", "context_note"),
        content=kw.pop("content", UNREVIEWED_TEXT),
        source="explicit",
        trust_class="unreviewed",
        **kw,
    )


# ===========================================================================
# Store: promote
# ===========================================================================

class TestStorePromote:
    def test_promote_clears_marker(self, tmp_path):
        store = _make_store(tmp_path)
        try:
            mem = _seed_unreviewed(store)
            assert mem.trust_class == "unreviewed"
            out = store.review_memory(
                memory_id=mem.memory_id,
                decision="promote",
                reason="verified by user",
                review_source="manual",
                reviewer="test_user",
            )
            assert out is not None
            assert out["changed"] is True
            assert out["decision"] == "promoted"
            assert out["previous_trust_class"] == "unreviewed"
            refetched = store.get_memories_by_ids([mem.memory_id])[0]
            assert refetched.trust_class is None
        finally:
            store.close()

    def test_promote_second_time_is_noop(self, tmp_path):
        store = _make_store(tmp_path)
        try:
            mem = _seed_unreviewed(store)
            store.review_memory(memory_id=mem.memory_id, decision="promote")
            out = store.review_memory(memory_id=mem.memory_id, decision="promote")
            assert out is not None
            assert out["changed"] is False
        finally:
            store.close()

    def test_promote_unknown_id_returns_none(self, tmp_path):
        store = _make_store(tmp_path)
        try:
            assert store.review_memory(
                memory_id="mem-does-not-exist", decision="promote"
            ) is None
        finally:
            store.close()

    def test_promote_non_head_version_returns_none(self, tmp_path):
        store = _make_store(tmp_path)
        try:
            mem = _seed_unreviewed(store)
            store._mark_superseded(mem.memory_id, reason="test", superseded_by=None)
            assert store.review_memory(
                memory_id=mem.memory_id, decision="promote"
            ) is None
        finally:
            store.close()

    def test_promote_emits_event(self, tmp_path):
        store = _make_store(tmp_path)
        try:
            mem = _seed_unreviewed(store)
            store.review_memory(
                memory_id=mem.memory_id,
                decision="promote",
                reason="vouched",
                review_source="manual",
                reviewer="reviewer-1",
            )
            events = store.list_mutation_events(event_type="memory_promoted")
            assert len(events) == 1
            ev = events[0]
            assert ev["entity_key"] == mem.memory_id
            refs = ev.get("refs") or {}
            assert refs.get("reviewer") == "reviewer-1"
            assert refs.get("review_source") == "manual"
            assert refs.get("previous_trust_class") == "unreviewed"
        finally:
            store.close()

    def test_promote_default_source_is_manual(self, tmp_path):
        store = _make_store(tmp_path)
        try:
            mem = _seed_unreviewed(store)
            out = store.review_memory(memory_id=mem.memory_id, decision="promote")
            assert out["changed"] is True
        finally:
            store.close()


# ===========================================================================
# Store: dismiss
# ===========================================================================

class TestStoreDismiss:
    def test_dismiss_quarantines_record(self, tmp_path):
        store = _make_store(tmp_path)
        try:
            mem = _seed_unreviewed(store)
            out = store.review_memory(
                memory_id=mem.memory_id,
                decision="dismiss",
                reason="hallucinated",
                review_source="manual",
            )
            assert out is not None
            assert out["changed"] is True
            assert out["decision"] == "dismissed"
            assert out["previous_status"] == "active"
            # By-id fetch helpers exclude quarantined rows by design, so
            # read the raw row to verify the quarantine landed.
            row = store.connection.execute(
                """SELECT status, quarantine_reason FROM memory_records
                   WHERE memory_id = ?""",
                [mem.memory_id],
            ).fetchone()
            assert row is not None
            assert row[0] == "quarantined"
            assert "hallucinated" in (row[1] or "")
        finally:
            store.close()

    def test_dismiss_slotless_reports_not_blocked(self, tmp_path):
        # rejection_key is intentionally empty for slot-less records - the
        # summary must say so instead of implying a block that cannot exist.
        store = _make_store(tmp_path)
        try:
            mem = _seed_unreviewed(store)
            out = store.review_memory(
                memory_id=mem.memory_id, decision="dismiss", reason="not real"
            )
            assert out["reassertion_blocked"] is False
        finally:
            store.close()

    def test_dismiss_with_slot_blocks_reassertion(self, tmp_path):
        store = _make_store(tmp_path)
        try:
            mem = _seed_unreviewed(
                store,
                content=SLOT_TEXT,
                payload={"attribute": "gate_code"},
            )
            out = store.review_memory(
                memory_id=mem.memory_id,
                decision="dismiss",
                reason="wrong value",
                review_source="manual",
            )
            assert out["reassertion_blocked"] is True
            # The ledger row exists for the claim slot.
            rj = store.rejection_check(
                "context_note", {"attribute": "gate_code"}
            )
            assert rj is not None
            assert "wrong value" in (rj.get("reason") or "")
            # Re-asserting the same slot is blocked at save time.
            again = store.remember(
                category="context_note",
                content=SLOT_TEXT,
                source="explicit",
                payload={"attribute": "gate_code"},
            )
            assert again is None
        finally:
            store.close()

    def test_dismiss_emits_event(self, tmp_path):
        store = _make_store(tmp_path)
        try:
            mem = _seed_unreviewed(store)
            store.review_memory(
                memory_id=mem.memory_id, decision="dismiss", reason="noise"
            )
            events = store.list_mutation_events(event_type="memory_dismissed")
            assert len(events) == 1
            assert events[0]["entity_key"] == mem.memory_id
        finally:
            store.close()

    def test_dismiss_unknown_id_returns_none(self, tmp_path):
        store = _make_store(tmp_path)
        try:
            assert store.review_memory(
                memory_id="mem-nope", decision="dismiss"
            ) is None
        finally:
            store.close()

    def test_dismissed_record_hidden_from_search(self, tmp_path):
        store = _make_store(tmp_path)
        try:
            mem = _seed_unreviewed(store)
            store.review_memory(memory_id=mem.memory_id, decision="dismiss")
            hits = store.search("garage door remote", limit=10)
            assert all(h.memory_id != mem.memory_id for h in hits)
        finally:
            store.close()


# ===========================================================================
# Resolution invariant + input validation
# ===========================================================================

class TestResolutionInvariant:
    def test_auto_review_refused_with_event(self, tmp_path):
        store = _make_store(tmp_path)
        try:
            mem = _seed_unreviewed(store)
            with pytest.raises(ValueError):
                store.review_memory(
                    memory_id=mem.memory_id,
                    decision="promote",
                    review_source="auto_review",
                )
            events = store.list_mutation_events(event_type="memory_review_refused")
            assert len(events) >= 1
            # The refusal must not have changed the record.
            refetched = store.get_memories_by_ids([mem.memory_id])[0]
            assert refetched.trust_class == "unreviewed"
        finally:
            store.close()

    def test_system_source_refused(self, tmp_path):
        store = _make_store(tmp_path)
        try:
            mem = _seed_unreviewed(store)
            with pytest.raises(ValueError):
                store.review_memory(
                    memory_id=mem.memory_id,
                    decision="dismiss",
                    review_source="system",
                )
        finally:
            store.close()

    def test_invalid_decision_raises(self, tmp_path):
        store = _make_store(tmp_path)
        try:
            mem = _seed_unreviewed(store)
            with pytest.raises(ValueError):
                store.review_memory(memory_id=mem.memory_id, decision="banana")
        finally:
            store.close()


# ===========================================================================
# Search behavior after resolution
# ===========================================================================

class TestSearchAfterResolution:
    def _seed_rank(self, store):
        mem = store.remember(
            category="context_note",
            content="falcon falcon falcon falcon falcon sighting",
            source="explicit",
            trust_class="unreviewed",
        )
        for word in ("alpha", "bravo", "charlie", "delta"):
            store.remember(
                category="context_note",
                content=f"falcon report {word}",
                source="explicit",
            )
        return mem

    def test_penalty_removed_after_promote(self, tmp_path):
        store = _make_store(tmp_path)
        try:
            mem = self._seed_rank(store)
            hits = store.search("falcon", limit=5)
            idx_before = next(
                i for i, h in enumerate(hits) if h.memory_id == mem.memory_id
            )
            assert idx_before > 0  # penalized below clean rows
            store.review_memory(memory_id=mem.memory_id, decision="promote")
            hits2 = store.search("falcon", limit=5)
            idx_after = next(
                i for i, h in enumerate(hits2) if h.memory_id == mem.memory_id
            )
            # Strongest lexical match, no marker -> top slot.
            assert idx_after == 0
            assert idx_after < idx_before
        finally:
            store.close()


# ===========================================================================
# Service dispatch (_call_store): server-derived review class
# ===========================================================================

class _RecordingStore:
    def __init__(self) -> None:
        self.calls: List[tuple] = []
        self.user_id = "boot_user"

    def set_user_scope(self, user_id: str) -> None:
        self.user_id = user_id

    def review_memory(self, **kwargs):
        self.calls.append(("review_memory", dict(kwargs)))
        return {
            "memory_id": kwargs.get("memory_id"),
            "decision": "promoted",
            "changed": True,
        }


class TestServiceDispatch:
    def _svc(self):
        from memory_service import MemoryService

        return MemoryService.__new__(MemoryService)

    def test_ungated_call_forced_auto_review(self):
        store = _RecordingStore()
        self._svc()._call_store(
            "review_memory",
            {"memory_id": "m1", "decision": "promote", "review_source": "tool"},
            "u9",
            store,
            confirmed=False,
        )
        call = [c for c in store.calls if c[0] == "review_memory"][0][1]
        assert call["review_source"] == "auto_review"

    def test_gated_call_honors_claimed_source(self):
        store = _RecordingStore()
        self._svc()._call_store(
            "review_memory",
            {"memory_id": "m1", "decision": "promote", "review_source": "tool"},
            "u9",
            store,
            confirmed=True,
        )
        call = [c for c in store.calls if c[0] == "review_memory"][0][1]
        assert call["review_source"] == "tool"

    def test_gated_call_without_claim_defaults_auto(self):
        store = _RecordingStore()
        self._svc()._call_store(
            "review_memory",
            {"memory_id": "m1", "decision": "promote"},
            "u9",
            store,
            confirmed=True,
        )
        call = [c for c in store.calls if c[0] == "review_memory"][0][1]
        assert call["review_source"] == "auto_review"

    def test_scope_applied_before_call(self):
        store = _RecordingStore()
        self._svc()._call_store(
            "review_memory",
            {"memory_id": "m1", "decision": "promote"},
            "u9",
            store,
        )
        assert store.user_id == "u9"


# ===========================================================================
# Facade: registration, validation, denial, scoping
# ===========================================================================

class TestFacadeOp:
    def _ctx(self, user_id: str = "test_user", **kw):
        from api_facade import AuthContext, PROPOSAL_OPERATIONS

        return AuthContext(
            principal=kw.pop("principal", "test-principal"),
            tenant="default",
            user_id=user_id,
            transport=kw.pop("transport", "rest"),
            allowed_operations=kw.pop(
                "allowed_operations", set(PROPOSAL_OPERATIONS)
            ),
            **kw,
        )

    def test_op_registered(self):
        from api_facade import PROPOSAL_OPERATIONS

        assert "review_memory" in PROPOSAL_OPERATIONS

    def test_validate_promote_and_dismiss(self):
        from api_facade import _validate_review_memory_params

        for decision in ("promote", "dismiss", "  PROMOTE  "):
            cleaned = _validate_review_memory_params(
                {"memory_id": "mem-1", "decision": decision, "reason": "ok"}
            )
            assert cleaned["decision"] in ("promote", "dismiss")
            assert cleaned["memory_id"] == "mem-1"

    def test_validate_rejects_junk(self):
        from api_facade import APIError, _validate_review_memory_params

        with pytest.raises(APIError):
            _validate_review_memory_params(
                {"memory_id": "mem-1", "decision": "banana"}
            )
        with pytest.raises(APIError):
            _validate_review_memory_params({"decision": "promote"})
        with pytest.raises(APIError):
            _validate_review_memory_params(
                {"memory_id": "x" * 300, "decision": "promote"}
            )

    def test_happy_promote_through_facade(self, tmp_path):
        from api_facade import ArgosAPIFacade
        from access_scoping import ACLConfig

        store = _make_store(tmp_path)
        try:
            mem = _seed_unreviewed(store)
            facade = ArgosAPIFacade(store, acl=ACLConfig(), api_mode=False)
            result = facade.execute(
                self._ctx(),
                "review_memory",
                {"memory_id": mem.memory_id, "decision": "promote", "reason": "ok"},
            )
            assert result["changed"] is True
            assert result["reviewer"] == "test-principal"
            refetched = store.get_memories_by_ids([mem.memory_id])[0]
            assert refetched.trust_class is None
        finally:
            store.close()

    def test_not_found_for_unknown_id(self, tmp_path):
        from api_facade import APIError, ArgosAPIFacade
        from access_scoping import ACLConfig

        store = _make_store(tmp_path)
        try:
            facade = ArgosAPIFacade(store, acl=ACLConfig(), api_mode=False)
            with pytest.raises(APIError) as excinfo:
                facade.execute(
                    self._ctx(),
                    "review_memory",
                    {"memory_id": "mem-nope", "decision": "promote"},
                )
            assert excinfo.value.code == "not_found"
        finally:
            store.close()

    def test_model_principal_denied(self, tmp_path):
        from api_facade import APIError, ArgosAPIFacade
        from access_scoping import ACLConfig

        store = _make_store(tmp_path)
        try:
            mem = _seed_unreviewed(store)
            facade = ArgosAPIFacade(store, acl=ACLConfig(), api_mode=False)
            ctx = self._ctx(principal_type="model")
            with pytest.raises(APIError) as excinfo:
                facade.execute(
                    ctx,
                    "review_memory",
                    {"memory_id": mem.memory_id, "decision": "promote"},
                )
            assert excinfo.value.code == "forbidden"
            # Nothing changed.
            refetched = store.get_memories_by_ids([mem.memory_id])[0]
            assert refetched.trust_class == "unreviewed"
        finally:
            store.close()

    def test_store_without_support_denied(self):
        from api_facade import APIError, ArgosAPIFacade
        from access_scoping import ACLConfig

        class _BareStore:
            user_id = "boot_user"

            def set_user_scope(self, user_id):
                self.user_id = user_id

        facade = ArgosAPIFacade(_BareStore(), acl=ACLConfig(), api_mode=False)
        with pytest.raises(APIError) as excinfo:
            facade.execute(
                self._ctx(),
                "review_memory",
                {"memory_id": "mem-1", "decision": "promote"},
            )
        assert excinfo.value.code == "method_not_allowed"

    def test_scope_restored_on_failure(self):
        from api_facade import APIError, ArgosAPIFacade
        from access_scoping import ACLConfig

        class _BoomStore:
            def __init__(self):
                self.user_id = "boot_user"

            def set_user_scope(self, user_id):
                self.user_id = user_id

            def review_memory(self, **kwargs):
                raise RuntimeError("boom")

        store = _BoomStore()
        facade = ArgosAPIFacade(store, acl=ACLConfig(), api_mode=False)
        with pytest.raises(APIError) as excinfo:
            facade.execute(
                self._ctx(),
                "review_memory",
                {"memory_id": "mem-1", "decision": "promote"},
            )
        assert excinfo.value.code == "internal_error"
        assert store.user_id == "boot_user"

    def test_denied_without_allowed_op(self):
        from api_facade import APIError, ArgosAPIFacade, AuthContext
        from access_scoping import ACLConfig

        store = None
        from access_scoping import ACLConfig as _A

        facade = ArgosAPIFacade(
            type("S", (), {"user_id": "u", "set_user_scope": lambda self, x: None})(),
            acl=_A(),
            api_mode=False,
        )
        ctx = AuthContext(
            principal="test-principal",
            tenant="default",
            user_id="test_user",
            transport="rest",
            allowed_operations={"search"},
        )
        with pytest.raises(APIError):
            facade.execute(
                ctx,
                "review_memory",
                {"memory_id": "mem-1", "decision": "promote"},
            )


# ===========================================================================
# MCP surface
# ===========================================================================

class TestMcpSurface:
    def test_tool_mapping(self):
        from mcp_server import TOOL_TO_OPERATION

        assert TOOL_TO_OPERATION.get("memory_review") == "review_memory"

    def test_idempotency_membership(self):
        from mcp_server import TOOLS_WITH_IDEMPOTENCY_KEY

        assert "memory_review" in TOOLS_WITH_IDEMPOTENCY_KEY

    def test_tool_definition_strict_schema(self):
        from mcp_server import TOOL_DEFINITIONS

        entries = [d for d in TOOL_DEFINITIONS if d["name"] == "memory_review"]
        assert len(entries) == 1
        schema = entries[0]["inputSchema"]
        assert schema["type"] == "object"
        assert schema["additionalProperties"] is False
        assert set(schema["required"]) == {
            "memory_id", "decision", "idempotency_key",
        }
        assert schema["properties"]["decision"]["enum"] == [
            "promote", "dismiss",
        ]


# ===========================================================================
# REST surface
# ===========================================================================

class _RestStubStore:
    def __init__(self) -> None:
        self.user_id = "boot_user"
        self.review_calls: List[Dict[str, Any]] = []

    def set_user_scope(self, user_id: str) -> None:
        self.user_id = user_id

    def review_memory(self, **kwargs) -> Dict[str, Any]:
        self.review_calls.append(dict(kwargs))
        return {
            "memory_id": kwargs.get("memory_id"),
            "decision": "promoted",
            "changed": True,
            "previous_trust_class": "unreviewed",
            "previous_status": None,
            "reassertion_blocked": None,
        }


def _make_client(store=None, token="test-rest-token"):
    from api_facade import ArgosAPIFacade
    from access_scoping import ACLConfig
    from rest_server import create_app
    from fastapi.testclient import TestClient

    store = store or _RestStubStore()
    facade = ArgosAPIFacade(store, acl=ACLConfig(), api_mode=False)
    app = create_app(facade, auth_token=token)
    return TestClient(app), store


def _auth_headers(token: str = "test-rest-token", idem: str | None = None):
    headers = {"Authorization": f"Bearer {token}"}
    if idem:
        headers["Idempotency-Key"] = idem
    return headers


class TestRestSurface:
    def test_route_promote_returns_summary(self, monkeypatch):
        # Class B is human-only; the REST transport defaults to
        # principal_type="model" (fail-closed). A human-driven UI must
        # explicitly declare itself.
        monkeypatch.setenv("ARGOS_API_PRINCIPAL_TYPE", "human")
        client, store = _make_client()
        r = client.post(
            "/v1/memories/mem-1/decision",
            headers=_auth_headers(idem="k-promote-1"),
            json={"decision": "promote", "reason": "ok"},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["memory_id"] == "mem-1"
        assert body["changed"] is True
        assert body["reviewer"] == "local"
        assert store.review_calls
        assert store.review_calls[-1]["review_source"] == "tool"

    def test_route_requires_auth(self):
        client, _ = _make_client()
        r = client.post(
            "/v1/memories/mem-1/decision",
            headers={"Idempotency-Key": "k1"},
            json={"decision": "promote"},
        )
        assert r.status_code == 401

    def test_route_requires_idempotency_key(self):
        client, _ = _make_client()
        r = client.post(
            "/v1/memories/mem-1/decision",
            headers=_auth_headers(),
            json={"decision": "promote"},
        )
        assert r.status_code == 400

    def test_model_principal_denied_by_default(self, monkeypatch):
        # No ARGOS_API_PRINCIPAL_TYPE -> "model" (fail-closed): class B
        # routes must refuse. This is the self-vouch spoof guard.
        monkeypatch.delenv("ARGOS_API_PRINCIPAL_TYPE", raising=False)
        client, store = _make_client()
        r = client.post(
            "/v1/memories/mem-1/decision",
            headers=_auth_headers(idem="k-model"),
            json={"decision": "promote"},
        )
        assert r.status_code == 403
        assert not store.review_calls

    def test_invalid_decision_rejected(self):
        client, _ = _make_client()
        r = client.post(
            "/v1/memories/mem-1/decision",
            headers=_auth_headers(idem="k-bad"),
            json={"decision": "banana"},
        )
        assert r.status_code == 422

    def test_idempotency_replay_dedups(self, monkeypatch):
        monkeypatch.setenv("ARGOS_API_PRINCIPAL_TYPE", "human")
        client, store = _make_client()
        payload = {"decision": "promote", "reason": "ok"}
        r1 = client.post(
            "/v1/memories/mem-1/decision",
            headers=_auth_headers(idem="k-replay"),
            json=payload,
        )
        r2 = client.post(
            "/v1/memories/mem-1/decision",
            headers=_auth_headers(idem="k-replay"),
            json=payload,
        )
        assert r1.status_code == 200
        assert r2.status_code == 200
        assert len(store.review_calls) == 1
        assert r1.json() == r2.json()

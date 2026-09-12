"""#386 (Spec-12): external parity — MCP/REST vs the direct store.

The verified contract: a transport client and the in-process path run the
SAME store-layer retrieval pipeline for the same query on the same
store/user — identical ordered IDs, no silent text-only fallback.
Divergence here = a transport-side wiring gap (file it; do not tune the
test).

Also pins the #386 ingest wiring: preview writes nothing on any
transport; apply materializes ACTIVE records through the candidate path;
apply is class-C posture (loopback only — the facade denies non-loopback,
fail-closed); idempotency key is required on MCP and REST.

Hermetic: in-process DuckDBMemoryStore + DeterministicEmbedder, no
service, no network, no LLM calls.
"""
from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest
from fastapi.testclient import TestClient

os.environ["ARGOS_HERMETIC_TESTS"] = "1"

_plugin_dir = Path(__file__).resolve().parent.parent
if str(_plugin_dir) not in sys.path:
    sys.path.insert(0, str(_plugin_dir))

from api_facade import (  # noqa: E402
    APIError,
    ArgosAPIFacade,
    AuthContext,
    READ_OPERATIONS,
    PROPOSAL_OPERATIONS,
    WRITE_OPERATIONS,
)

from access_scoping import ACLConfig  # noqa: E402
from mcp_server import MCPServer  # noqa: E402
from rest_server import create_app  # noqa: E402
from store import DuckDBMemoryStore  # noqa: E402

from conftest import DeterministicEmbedder  # noqa: E402

USER = "test_user"
TOKEN = "test-rest-token"

SEED_FACTS = [
    ("Michael drives a silver Hilux and tracks fuel costs every month.", "personal_fact"),
    ("The mileapp logs GPS trips automatically for the tax logbook.", "context_note"),
    ("Simone wants to monetise the trip logging app eventually.", "context_note"),
    ("Creditors reconciliation is the first paying consulting prospect.", "context_note"),
    ("The office vape rule says the device stays in the car.", "personal_fact"),
    ("Argos retrieval blends embedding search with a reranker stage.", "context_note"),
]

# The ingest fixture: two valid rows against one mapping.
INGEST_CSV = "name,employer,role\nAlice,Acme Corp,engineer\nBob,Globex,manager"
INGEST_MAPPING = {
    "category": "context_note",
    "content_template": "{name} works at {employer} as a {role}",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seed_store(tmp_path: Path) -> DuckDBMemoryStore:
    store = DuckDBMemoryStore(
        tmp_path / "test.duckdb", user_id=USER,
        embedder=DeterministicEmbedder(),
    )
    for content, category in SEED_FACTS:
        store.remember(content=content, category=category)
    return store


def _make_facade(store) -> ArgosAPIFacade:
    return ArgosAPIFacade(store, acl=ACLConfig(), api_mode=False)


def _make_mcp(store, *, is_loopback: bool = True, allowed_ops=None):
    facade = _make_facade(store)
    auth = AuthContext(
        principal="test-principal", tenant="default", user_id=USER,
        transport="mcp-stdio",
        allowed_operations=(
            allowed_ops
            if allowed_ops is not None
            else READ_OPERATIONS | PROPOSAL_OPERATIONS | WRITE_OPERATIONS
        ),
        can_propose=True, can_feedback=True,
        principal_type="human", is_loopback=is_loopback,
    )
    stdin = io.StringIO("")
    stdout = io.StringIO()
    stderr = io.StringIO()
    server = MCPServer(facade, auth, stdin=stdin, stdout=stdout, stderr=stderr)
    server._initialized = True
    return server, stdout, stderr


def _mcp_call(server, stdout, tool_name: str, arguments: Dict[str, Any]):
    line = json.dumps({
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": tool_name, "arguments": arguments},
    })
    server._handle_line(line)
    stdout.seek(0)
    msgs = [json.loads(l) for l in stdout if l.strip()]
    return msgs[0] if msgs else {}


def _mcp_result_payload(msg: Dict[str, Any]) -> Dict[str, Any]:
    """Extract the facade result dict from a tools/call response."""
    result = msg.get("result", {})
    payload = result.get("structuredContent")
    if payload is None:
        content = result.get("content") or []
        if content:
            payload = json.loads(content[0]["text"])
    return payload or {}


def _make_rest_client(store, monkeypatch, *, is_loopback: bool = True):
    facade = _make_facade(store)
    monkeypatch.setenv("ARGOS_API_PRINCIPAL_TYPE", "human")
    monkeypatch.setenv("ARGOS_API_USER_ID", USER)
    monkeypatch.setenv("ARGOS_API_CAN_PROPOSE", "1")
    monkeypatch.setenv("ARGOS_API_CAN_FEEDBACK", "1")
    if is_loopback:
        monkeypatch.setenv("ARGOS_API_CAN_WRITE", "1")
        monkeypatch.delenv("ARGOS_API_NO_LOOPBACK", raising=False)
    else:
        monkeypatch.delenv("ARGOS_API_CAN_WRITE", raising=False)
        monkeypatch.setenv("ARGOS_API_NO_LOOPBACK", "1")
    app = create_app(facade, auth_token=TOKEN)
    return TestClient(app)


def _auth_headers() -> Dict[str, str]:
    return {"Authorization": f"Bearer {TOKEN}"}


def _ids(records: List[Any]) -> List[str]:
    return [getattr(r, "memory_id", None) for r in records]


def _payload_ids(payload: Dict[str, Any]) -> List[str]:
    items = payload.get("results", payload)
    return [it.get("memory_id") for it in items]


# ===========================================================================
# Retrieval parity: MCP / REST vs the direct store
# ===========================================================================

class TestRetrievalParity:
    """Same store+query → identical ordered IDs through every surface."""

    def test_mcp_search_matches_direct_store_order(self, tmp_path):
        store = _seed_store(tmp_path)
        query = "GPS trip tracking for the tax logbook"

        direct = store.search(query, limit=4)
        assert len(direct) >= 3, "seed produced too few results"

        server, stdout, _ = _make_mcp(store)
        msg = _mcp_call(server, stdout, "memory_search",
                        {"query": query, "limit": 4})
        payload = _mcp_result_payload(msg)

        assert _payload_ids(payload) == _ids(direct), (
            "MCP search diverged from the direct store path — a transport "
            "wiring gap, not a test-tuning problem (#386)"
        )
        # The vector leg fired: results carry a positive blended score.
        top = payload.get("results", payload)[0]
        sim = top.get("similarity")
        assert sim is not None and sim > 0, (
            "MCP results carry no similarity — retrieval degraded to a "
            "score-less fallback"
        )

    def test_mcp_search_trust_class_filter_parity(self, tmp_path):
        store = _seed_store(tmp_path)
        query = "creditors reconciliation prospect"

        direct = store.search(query, limit=5, trust_class="clean")
        server, stdout, _ = _make_mcp(store)
        msg = _mcp_call(server, stdout, "memory_search",
                        {"query": query, "limit": 5, "trust_class": "clean"})
        payload = _mcp_result_payload(msg)
        assert _payload_ids(payload) == _ids(direct)

    def test_rest_search_matches_direct_store_order(self, tmp_path, monkeypatch):
        store = _seed_store(tmp_path)
        query = "vape rule office car"

        direct = store.search(query, limit=4)
        assert len(direct) >= 3

        client = _make_rest_client(store, monkeypatch)
        r = client.post(
            "/v1/memory/search",
            headers=_auth_headers(),
            json={"query": query, "limit": 4},
        )
        assert r.status_code == 200, r.text
        assert _payload_ids(r.json()) == _ids(direct), (
            "REST search diverged from the direct store path (#386)"
        )


# ===========================================================================
# Ingest wiring: preview / apply / gates
# ===========================================================================

class TestIngestTransportGates:
    """Facade-level gate: apply is class C (loopback only); preview is not."""

    def test_apply_denied_on_non_loopback(self, tmp_path):
        store = _seed_store(tmp_path)
        facade = _make_facade(store)
        ctx = AuthContext(
            principal="external", tenant="default", user_id=USER,
            transport="rest", allowed_operations=READ_OPERATIONS | PROPOSAL_OPERATIONS,
            can_propose=True, can_feedback=False,
            principal_type="model", is_loopback=False,
        )
        with pytest.raises(APIError) as exc:
            facade.execute(ctx, "ingest", {
                "data": INGEST_CSV, "fmt": "csv",
                "source_name": "gates", "mapping": INGEST_MAPPING,
                "mode": "apply", "confirm": True,
            })
        assert exc.value.code == "forbidden"

    def test_preview_allowed_on_non_loopback_writes_nothing(self, tmp_path):
        store = _seed_store(tmp_path)
        before = store.count()
        facade = _make_facade(store)
        ctx = AuthContext(
            principal="external", tenant="default", user_id=USER,
            transport="rest", allowed_operations=READ_OPERATIONS | PROPOSAL_OPERATIONS,
            can_propose=True, can_feedback=False,
            principal_type="model", is_loopback=False,
        )
        report = facade.execute(ctx, "ingest", {
            "data": INGEST_CSV, "fmt": "csv",
            "source_name": "gates", "mapping": INGEST_MAPPING,
            "mode": "preview",
        })
        assert report["wrote"] is False
        assert report["valid_rows"] == 2
        assert store.count() == before


class TestIngestMcp:
    """memory_ingest over the MCP transport."""

    def test_preview_writes_nothing(self, tmp_path):
        store = _seed_store(tmp_path)
        before = store.count()
        server, stdout, _ = _make_mcp(store)
        msg = _mcp_call(server, stdout, "memory_ingest", {
            "data": INGEST_CSV, "fmt": "csv",
            "source_name": "mcp-fixture", "mapping": INGEST_MAPPING,
            "mode": "preview", "idempotency_key": "ing-mcp-1",
        })
        report = _mcp_result_payload(msg)
        assert report["wrote"] is False
        assert report["valid_rows"] == 2
        assert store.count() == before

    def test_apply_materializes_active_records(self, tmp_path):
        store = _seed_store(tmp_path)
        before = store.count()
        server, stdout, _ = _make_mcp(store)
        msg = _mcp_call(server, stdout, "memory_ingest", {
            "data": INGEST_CSV, "fmt": "csv",
            "source_name": "mcp-fixture", "mapping": INGEST_MAPPING,
            "mode": "apply", "confirm": True,
            "idempotency_key": "ing-mcp-2",
        })
        report = _mcp_result_payload(msg)
        assert report["wrote"] is True
        assert report["inserted"] == 2
        assert store.count() == before + 2

        # ACTIVE immediately: the rows are searchable right away.
        hits = store.search("Alice works at Acme Corp", limit=3)
        assert any("Alice" in getattr(h, "content", "") for h in hits), (
            "ingested row did not materialize as active memory"
        )

    def test_apply_requires_literal_confirm(self, tmp_path):
        store = _seed_store(tmp_path)
        server, stdout, _ = _make_mcp(store)
        msg = _mcp_call(server, stdout, "memory_ingest", {
            "data": INGEST_CSV, "fmt": "csv",
            "source_name": "mcp-fixture", "mapping": INGEST_MAPPING,
            "mode": "apply", "confirm": False,
            "idempotency_key": "ing-mcp-3",
        })
        # Schema passes; the facade validator rejects apply without confirm.
        assert msg.get("error") is not None or msg.get("result", {}).get("isError") is True


class TestIngestRest:
    """POST /v1/ingest over the REST transport."""

    def test_preview_then_apply_loopback(self, tmp_path, monkeypatch):
        store = _seed_store(tmp_path)
        before = store.count()
        client = _make_rest_client(store, monkeypatch, is_loopback=True)

        body_preview = {
            "data": INGEST_CSV, "fmt": "csv",
            "source_name": "rest-fixture", "mapping": INGEST_MAPPING,
            "mode": "preview",
        }
        r = client.post("/v1/ingest", headers={
            **_auth_headers(), "Idempotency-Key": "ing-rest-1",
        }, json=body_preview)
        assert r.status_code == 200, r.text
        assert r.json()["wrote"] is False
        assert store.count() == before

        body_apply = dict(body_preview, mode="apply", confirm=True)
        r2 = client.post("/v1/ingest", headers={
            **_auth_headers(), "Idempotency-Key": "ing-rest-2",
        }, json=body_apply)
        assert r2.status_code == 200, r2.text
        assert r2.json()["wrote"] is True
        assert store.count() == before + 2

    def test_missing_idempotency_key_rejected(self, tmp_path, monkeypatch):
        store = _seed_store(tmp_path)
        client = _make_rest_client(store, monkeypatch)
        r = client.post("/v1/ingest", headers=_auth_headers(), json={
            "data": INGEST_CSV, "fmt": "csv",
            "source_name": "rest-fixture", "mapping": INGEST_MAPPING,
        })
        assert r.status_code == 400

    def test_apply_denied_on_non_loopback(self, tmp_path, monkeypatch):
        store = _seed_store(tmp_path)
        client = _make_rest_client(store, monkeypatch, is_loopback=False)
        r = client.post("/v1/ingest", headers={
            **_auth_headers(), "Idempotency-Key": "ing-rest-3",
        }, json={
            "data": INGEST_CSV, "fmt": "csv",
            "source_name": "rest-fixture", "mapping": INGEST_MAPPING,
            "mode": "apply", "confirm": True,
        })
        assert r.status_code == 403

    def test_provenance_fields_forbidden(self, tmp_path, monkeypatch):
        store = _seed_store(tmp_path)
        client = _make_rest_client(store, monkeypatch)
        r = client.post("/v1/ingest", headers={
            **_auth_headers(), "Idempotency-Key": "ing-rest-4",
        }, json={
            "data": INGEST_CSV, "fmt": "csv",
            "source_name": "rest-fixture", "mapping": INGEST_MAPPING,
            "provenance_origin": "spoofed",
        })
        # extra="forbid" at the model — 422 from the transport.
        assert r.status_code == 422

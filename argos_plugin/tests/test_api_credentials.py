"""#387 (Spec-12): credential-backed identity for MCP/REST transports.

Contract under test:
- api_credential.json parsing (legacy single-token + versioned
  credentials list), hash-at-rest, duplicate/invalid entry rejection
  (fail closed, never silently drop a broken credential).
- Token resolution: match, unknown, expired; revocation = entry removal.
- build_context: granular classes; class C write ops require loopback;
  env vars can only narrow (never widen); read-only intersects down.
- RESTAuth: legacy transport token -> env context (unchanged); valid
  credential -> per-principal context; 401 invalid/expired; 500
  malformed config; live re-read after revocation; home=None is
  legacy-only.
- create_app e2e: scoped capabilities per credential over TestClient.
- MCP _load_auth_context: credential branch; fail-closed SystemExit on
  unknown/expired; legacy env path unchanged.
- Facade integration: a model credential cannot review (no model
  self-approval); a human credential can (class B); 'propose' does not
  imply erase/review; the facade's loopback write gate still applies.

Hermetic: no service, no network, no LLM calls.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

os.environ["ARGOS_HERMETIC_TESTS"] = "1"

_plugin_dir = Path(__file__).resolve().parent.parent
if str(_plugin_dir) not in sys.path:
    sys.path.insert(0, str(_plugin_dir))

from api_credentials import (  # noqa: E402
    CLASS_TO_OPERATIONS,
    Credential,
    CredentialFileError,
    build_context,
    credential_file_path,
    drop_legacy_token,
    mint_token,
    parse_credentials_file,
    resolve_by_name,
    resolve_by_token,
    revoke_credential,
    sha256_hex,
    write_credential,
)
from api_facade import (  # noqa: E402
    APIError,
    ArgosAPIFacade,
    AuthContext,
    WRITE_OPERATIONS,
)
from access_scoping import ACLConfig  # noqa: E402
from rest_server import RESTAuth, create_app  # noqa: E402
import mcp_server  # noqa: E402


# -- Helpers ------------------------------------------------------------------

_ENV_VARS = (
    "ARGOS_API_CREDENTIAL",
    "ARGOS_API_CREDENTIAL_FILE",
    "ARGOS_API_PRINCIPAL",
    "ARGOS_API_USER_ID",
    "ARGOS_API_TENANT",
    "ARGOS_API_PRINCIPAL_TYPE",
    "ARGOS_API_NO_LOOPBACK",
    "ARGOS_API_READ_ONLY",
    "ARGOS_API_CAN_PROPOSE",
    "ARGOS_API_CAN_FEEDBACK",
    "ARGOS_API_CAN_WRITE",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in _ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def _write_raw(home: Path, payload: Dict[str, Any]) -> Path:
    path = credential_file_path(home)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _entry(name: str = "probe", token: str = "tok-1", **over) -> Dict[str, Any]:
    entry = {
        "name": name,
        "token_sha256": sha256_hex(token),
        "user_id": "u1",
        "allowed_classes": ["read"],
    }
    entry.update(over)
    return entry


def _mk_cred(classes, principal_type="model", user_id="u1", name="ctx-maker"):
    return Credential(
        name=name,
        token_sha256=sha256_hex("ctx-token"),
        user_id=user_id,
        principal_type=principal_type,
        allowed_classes=frozenset(classes),
    )


def _past_iso(hours: int = 1) -> str:
    return (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()


# =============================================================================
# Parsing
# =============================================================================


class TestParse:
    def test_missing_file_is_legacy(self, tmp_path):
        token, creds = parse_credentials_file(tmp_path / "api_credential.json")
        assert token is None
        assert creds == []

    def test_legacy_single_token_file(self, tmp_path):
        path = _write_raw(tmp_path, {"token": "legacy-tok"})
        token, creds = parse_credentials_file(path)
        assert token == "legacy-tok"
        assert creds == []

    def test_versioned_file_with_creds(self, tmp_path):
        path = _write_raw(tmp_path, {
            "token": "legacy-tok",
            "version": 1,
            "credentials": [_entry()],
        })
        token, creds = parse_credentials_file(path)
        assert token == "legacy-tok"
        assert len(creds) == 1
        assert creds[0].name == "probe"
        assert creds[0].token_sha256 == sha256_hex("tok-1")

    def test_plaintext_token_hashed_at_load(self, tmp_path):
        path = _write_raw(tmp_path, {"credentials": [
            {"name": "p", "token": "plain-tok", "allowed_classes": ["read"]},
        ]})
        _, creds = parse_credentials_file(path)
        assert creds[0].token_sha256 == sha256_hex("plain-tok")

    def test_both_token_forms_rejected(self, tmp_path):
        path = _write_raw(tmp_path, {"credentials": [
            {"name": "p", "token": "x", "token_sha256": sha256_hex("x")},
        ]})
        with pytest.raises(CredentialFileError):
            parse_credentials_file(path)

    def test_duplicate_name_rejected(self, tmp_path):
        path = _write_raw(tmp_path, {"credentials": [_entry(), _entry(token="tok-2")]})
        with pytest.raises(CredentialFileError, match="duplicate credential name"):
            parse_credentials_file(path)

    def test_duplicate_token_rejected(self, tmp_path):
        path = _write_raw(tmp_path, {"credentials": [_entry(), _entry(name="other")]})
        with pytest.raises(CredentialFileError, match="collides"):
            parse_credentials_file(path)

    def test_unknown_class_rejected(self, tmp_path):
        path = _write_raw(tmp_path, {"credentials": [
            _entry(allowed_classes=["read", "superuser"]),
        ]})
        with pytest.raises(CredentialFileError, match="unknown class"):
            parse_credentials_file(path)

    def test_invalid_principal_type_rejected(self, tmp_path):
        path = _write_raw(tmp_path, {"credentials": [_entry(principal_type="robot")]})
        with pytest.raises(CredentialFileError, match="principal_type"):
            parse_credentials_file(path)

    def test_invalid_expiry_rejected(self, tmp_path):
        path = _write_raw(tmp_path, {"credentials": [_entry(expires_at="not-a-date")]})
        with pytest.raises(CredentialFileError, match="expires_at"):
            parse_credentials_file(path)

    def test_malformed_json_rejected(self, tmp_path):
        path = credential_file_path(tmp_path)
        path.write_text("{broken", encoding="utf-8")
        with pytest.raises(CredentialFileError):
            parse_credentials_file(path)

    def test_expiry_parsed_and_is_expired(self, tmp_path):
        path = _write_raw(tmp_path, {"credentials": [_entry(expires_at=_past_iso())]})
        _, creds = parse_credentials_file(path)
        assert creds[0].is_expired() is True

        future = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()
        path = _write_raw(tmp_path, {"credentials": [_entry(expires_at=future)]})
        _, creds = parse_credentials_file(path)
        assert creds[0].is_expired() is False


# =============================================================================
# Minting / write_credential
# =============================================================================


class TestMinting:
    def test_mint_token_format(self):
        t1, t2 = mint_token(), mint_token()
        assert t1.startswith("argos_") and t2.startswith("argos_")
        assert t1 != t2 and len(t1) > 30

    def test_hash_at_rest(self, tmp_path):
        token, cred = write_credential(
            credential_file_path(tmp_path), name="a", user_id="ua",
            allowed_classes=["read"],
        )
        raw = credential_file_path(tmp_path).read_text(encoding="utf-8")
        assert token not in raw
        assert cred.token_sha256 in raw
        assert sha256_hex(token) == cred.token_sha256

    def test_preserves_legacy_key_and_existing(self, tmp_path):
        path = _write_raw(tmp_path, {"token": "legacy-tok"})
        write_credential(path, name="a", user_id="ua", allowed_classes=["read"])
        write_credential(path, name="b", user_id="ub", allowed_classes=["propose"])
        token, creds = parse_credentials_file(path)
        assert token == "legacy-tok"
        assert [c.name for c in creds] == ["a", "b"]

    def test_duplicate_name_refused_and_file_untouched(self, tmp_path):
        path = credential_file_path(tmp_path)
        write_credential(path, name="a", user_id="ua", allowed_classes=["read"])
        before = path.read_text(encoding="utf-8")
        with pytest.raises(CredentialFileError, match="duplicate"):
            write_credential(path, name="a", user_id="ua", allowed_classes=["read"])
        assert path.read_text(encoding="utf-8") == before

    def test_bad_class_refused_before_write(self, tmp_path):
        path = credential_file_path(tmp_path)
        with pytest.raises(CredentialFileError, match="unknown class"):
            write_credential(path, name="a", user_id="ua", allowed_classes=["nope"])
        assert not path.exists()

    def test_expiry_recorded(self, tmp_path):
        future = datetime.now(timezone.utc) + timedelta(days=30)
        _token, cred = write_credential(
            credential_file_path(tmp_path), name="a", user_id="ua",
            allowed_classes=["read"], expires_at=future,
        )
        assert cred.expires_at is not None
        assert cred.is_expired() is False


# =============================================================================
# Resolution
# =============================================================================


class TestResolve:
    def _load(self, tmp_path, **entry_over):
        path = _write_raw(tmp_path, {"credentials": [_entry(**entry_over)]})
        _, creds = parse_credentials_file(path)
        return creds

    def test_by_token_match(self, tmp_path):
        creds = self._load(tmp_path)
        cred, expired = resolve_by_token(creds, "tok-1")
        assert cred is not None and cred.name == "probe" and expired is False

    def test_by_token_unknown(self, tmp_path):
        creds = self._load(tmp_path)
        cred, expired = resolve_by_token(creds, "wrong")
        assert cred is None and expired is False

    def test_by_token_expired_flag(self, tmp_path):
        creds = self._load(tmp_path, expires_at=_past_iso())
        cred, expired = resolve_by_token(creds, "tok-1")
        assert cred is None and expired is True

    def test_by_name(self, tmp_path):
        creds = self._load(tmp_path)
        assert resolve_by_name(creds, "probe").name == "probe"
        assert resolve_by_name(creds, "nope") is None


# =============================================================================
# Context building
# =============================================================================


class TestBuildContext:
    def test_class_map_is_granular(self):
        assert "erase_request" in CLASS_TO_OPERATIONS["erase"]
        assert "erase_request" not in CLASS_TO_OPERATIONS["propose"]
        assert "ingest" not in CLASS_TO_OPERATIONS["propose"]
        assert "memory_propose" in CLASS_TO_OPERATIONS["propose"]
        assert "review_candidate" in CLASS_TO_OPERATIONS["review"]
        assert "review_memory" in CLASS_TO_OPERATIONS["review"]
        assert "export" in CLASS_TO_OPERATIONS["read"]
        assert "browse" in CLASS_TO_OPERATIONS["read"]
        assert "memory_save" in CLASS_TO_OPERATIONS["write"]

    def test_read_class(self):
        ctx = build_context(_mk_cred(["read"]), transport="rest", is_loopback=True)
        assert "search" in ctx.allowed_operations
        assert "export" in ctx.allowed_operations
        assert "memory_save" not in ctx.allowed_operations
        assert "review_memory" not in ctx.allowed_operations
        assert ctx.can_propose is False

    def test_propose_does_not_imply_erase_or_review(self):
        ctx = build_context(_mk_cred(["read", "propose"]), transport="rest", is_loopback=True)
        assert "memory_propose" in ctx.allowed_operations
        assert "erase_request" not in ctx.allowed_operations
        assert "review_memory" not in ctx.allowed_operations
        assert "ingest" not in ctx.allowed_operations
        assert ctx.can_propose is True

    def test_write_requires_loopback(self):
        loop = build_context(_mk_cred(["write"]), transport="rest", is_loopback=True)
        assert "memory_save" in loop.allowed_operations
        non_loop = build_context(_mk_cred(["write"]), transport="rest", is_loopback=False)
        assert "memory_save" not in non_loop.allowed_operations
        assert non_loop.is_loopback is False

    def test_collection_write_requires_loopback(self):
        loop = build_context(_mk_cred(["collection_write"]), transport="rest", is_loopback=True)
        assert "collection_create" in loop.allowed_operations
        non_loop = build_context(_mk_cred(["collection_write"]), transport="rest", is_loopback=False)
        assert "collection_create" not in non_loop.allowed_operations

    def test_env_cannot_widen_principal_type(self):
        # cred human + env model -> model (narrowed)
        ctx = build_context(
            _mk_cred(["review"], principal_type="human"),
            transport="rest", is_loopback=True, env_principal_type="model",
        )
        assert ctx.principal_type == "model"
        # cred model + env human -> still model (no widening)
        ctx2 = build_context(
            _mk_cred(["review"], principal_type="model"),
            transport="rest", is_loopback=True, env_principal_type="human",
        )
        assert ctx2.principal_type == "model"

    def test_credential_identity_wins_over_env_scope(self):
        ctx = build_context(
            _mk_cred(["read"], user_id="credential-user", name="named"),
            transport="rest", is_loopback=True,
        )
        assert ctx.user_id == "credential-user"
        assert ctx.principal == "named"

    def test_read_only_intersects_down(self):
        ctx = build_context(
            _mk_cred(["read", "write"]),
            transport="rest", is_loopback=True, is_read_only=True,
        )
        assert "search" in ctx.allowed_operations
        assert "memory_save" not in ctx.allowed_operations

    def test_read_only_does_not_add_ops(self):
        ctx = build_context(
            _mk_cred(["propose"]),
            transport="rest", is_loopback=True, is_read_only=True,
        )
        assert ctx.allowed_operations == set()


# =============================================================================
# REST transport
# =============================================================================


class TestRESTAuth:
    def _auth(self, tmp_path, legacy="legacy-tok"):
        home = tmp_path / "home"
        home.mkdir(exist_ok=True)
        return RESTAuth(legacy, home=home), home

    def test_legacy_token_gets_env_context(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ARGOS_API_PRINCIPAL", "legacy-principal")
        monkeypatch.setenv("ARGOS_API_USER_ID", "legacy-user")
        auth, _ = self._auth(tmp_path)
        ctx = auth("Bearer legacy-tok")
        assert ctx.principal == "legacy-principal"
        assert ctx.user_id == "legacy-user"
        assert ctx.transport == "rest"

    def test_credential_token_gets_credential_context(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ARGOS_API_PRINCIPAL", "env-principal")
        monkeypatch.setenv("ARGOS_API_USER_ID", "env-user")
        auth, home = self._auth(tmp_path)
        token, _ = write_credential(
            credential_file_path(home), name="simone-laptop", user_id="simone",
            allowed_classes=["read", "propose"],
        )
        ctx = auth(f"Bearer {token}")
        assert ctx.principal == "simone-laptop"
        assert ctx.user_id == "simone"  # credential wins over env
        assert "memory_propose" in ctx.allowed_operations
        assert "memory_save" not in ctx.allowed_operations  # no write class

    def test_invalid_token_401(self, tmp_path):
        auth, _ = self._auth(tmp_path)
        with pytest.raises(HTTPException) as exc:
            auth("Bearer wrong-token")
        assert exc.value.status_code == 401
        assert exc.value.detail["error"]["message"] == "Invalid credentials."

    def test_expired_credential_401(self, tmp_path):
        auth, home = self._auth(tmp_path)
        _write_raw(home, {"credentials": [_entry(token="exp-tok", expires_at=_past_iso())]})
        with pytest.raises(HTTPException) as exc:
            auth("Bearer exp-tok")
        assert exc.value.status_code == 401
        assert exc.value.detail["error"]["message"] == "Credential expired."

    def test_revocation_applies_next_request(self, tmp_path):
        auth, home = self._auth(tmp_path)
        token, _ = write_credential(
            credential_file_path(home), name="rev", user_id="u1",
            allowed_classes=["read"],
        )
        ctx = auth(f"Bearer {token}")  # works
        assert ctx.principal == "rev"
        # Revoke: rewrite the file without the entry.
        _write_raw(home, {"credentials": []})
        with pytest.raises(HTTPException) as exc:
            auth(f"Bearer {token}")
        assert exc.value.status_code == 401

    def test_file_added_after_start_is_picked_up(self, tmp_path):
        auth, home = self._auth(tmp_path)
        with pytest.raises(HTTPException) as exc:
            auth("Bearer later-tok")
        assert exc.value.status_code == 401
        _write_raw(home, {"credentials": [_entry(token="later-tok")]})
        ctx = auth("Bearer later-tok")
        assert ctx.principal == "probe"

    def test_malformed_file_fails_closed_500(self, tmp_path):
        auth, home = self._auth(tmp_path)
        credential_file_path(home).write_text("{broken", encoding="utf-8")
        with pytest.raises(HTTPException) as exc:
            auth("Bearer anything")
        assert exc.value.status_code == 500
        assert exc.value.detail["error"]["code"] == "invalid_credential_config"

    def test_missing_and_bad_headers_401(self, tmp_path):
        auth, _ = self._auth(tmp_path)
        with pytest.raises(HTTPException) as exc:
            auth("")
        assert exc.value.status_code == 401
        with pytest.raises(HTTPException) as exc2:
            auth("Basic abc")
        assert exc2.value.status_code == 401

    def test_home_none_rejects_credentials(self):
        auth = RESTAuth("legacy-tok", home=None)
        with pytest.raises(HTTPException) as exc:
            auth("Bearer some-token")
        assert exc.value.status_code == 401
        # Legacy still works.
        ctx = auth("Bearer legacy-tok")
        assert ctx.transport == "rest"


class TestRESTApp:
    def _app(self, tmp_path):
        from test_rest_server import StubStore

        facade = ArgosAPIFacade(StubStore(), acl=ACLConfig(), api_mode=False)
        home = tmp_path / "home"
        home.mkdir(exist_ok=True)
        app = create_app(facade, auth_token="legacy-tok", home=home)
        return app, home

    def test_scoped_capabilities_per_credential(self, tmp_path):
        app, home = self._app(tmp_path)
        reader_tok, _ = write_credential(
            credential_file_path(home), name="reader", user_id="r1",
            allowed_classes=["read"],
        )
        writer_tok, _ = write_credential(
            credential_file_path(home), name="writer", user_id="w1",
            allowed_classes=["read", "write"],
        )
        client = TestClient(app)

        r = client.get("/v1/capabilities",
                       headers={"Authorization": f"Bearer {reader_tok}"})
        assert r.status_code == 200
        body = r.json()
        assert body["principal"] == "reader"
        assert "search" in body["operations"]
        assert "memory_save" not in body["operations"]

        r2 = client.get("/v1/capabilities",
                        headers={"Authorization": f"Bearer {writer_tok}"})
        assert r2.status_code == 200
        assert r2.json()["principal"] == "writer"
        assert "memory_save" in r2.json()["operations"]  # loopback default

    def test_invalid_token_401_end_to_end(self, tmp_path):
        app, _ = self._app(tmp_path)
        client = TestClient(app)
        r = client.get("/v1/capabilities",
                       headers={"Authorization": "Bearer nope"})
        assert r.status_code == 401

    def test_revoked_token_401_end_to_end(self, tmp_path):
        app, home = self._app(tmp_path)
        token, _ = write_credential(
            credential_file_path(home), name="rev", user_id="u1",
            allowed_classes=["read"],
        )
        client = TestClient(app)
        assert client.get("/v1/capabilities",
                          headers={"Authorization": f"Bearer {token}"}).status_code == 200
        _write_raw(home, {"credentials": []})
        assert client.get("/v1/capabilities",
                          headers={"Authorization": f"Bearer {token}"}).status_code == 401


# =============================================================================
# MCP transport
# =============================================================================


class TestMCPContext:
    def _home(self, tmp_path):
        home = tmp_path / "home"
        home.mkdir(exist_ok=True)
        return home

    def test_credential_context(self, tmp_path, monkeypatch):
        home = self._home(tmp_path)
        write_credential(
            credential_file_path(home), name="mcp-agent", user_id="agent-user",
            principal_type="model", allowed_classes=["read", "propose", "ingest"],
        )
        monkeypatch.setenv("ARGOS_API_CREDENTIAL", "mcp-agent")
        monkeypatch.setenv("ARGOS_API_USER_ID", "env-user")  # ignored
        monkeypatch.setenv("ARGOS_API_PRINCIPAL_TYPE", "human")  # cannot widen
        ctx = mcp_server._load_auth_context(home)
        assert ctx.principal == "mcp-agent"
        assert ctx.user_id == "agent-user"
        assert ctx.principal_type == "model"
        assert ctx.transport == "mcp-stdio"
        assert ctx.is_loopback is True
        assert "memory_propose" in ctx.allowed_operations
        assert "ingest" in ctx.allowed_operations
        assert "erase_request" not in ctx.allowed_operations

    def test_unknown_credential_refuses_start(self, tmp_path, monkeypatch):
        home = self._home(tmp_path)
        _write_raw(home, {"credentials": [_entry()]})
        monkeypatch.setenv("ARGOS_API_CREDENTIAL", "ghost")
        with pytest.raises(SystemExit):
            mcp_server._load_auth_context(home)

    def test_expired_credential_refuses_start(self, tmp_path, monkeypatch):
        home = self._home(tmp_path)
        _write_raw(home, {"credentials": [
            _entry(name="old", expires_at=_past_iso()),
        ]})
        monkeypatch.setenv("ARGOS_API_CREDENTIAL", "old")
        with pytest.raises(SystemExit):
            mcp_server._load_auth_context(home)

    def test_missing_file_refuses_start(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ARGOS_API_CREDENTIAL", "ghost")
        with pytest.raises(SystemExit):
            mcp_server._load_auth_context(tmp_path / "home")

    def test_invalid_file_refuses_start(self, tmp_path, monkeypatch):
        home = self._home(tmp_path)
        credential_file_path(home).write_text("{broken", encoding="utf-8")
        monkeypatch.setenv("ARGOS_API_CREDENTIAL", "x")
        with pytest.raises(SystemExit):
            mcp_server._load_auth_context(home)

    def test_file_override(self, tmp_path, monkeypatch):
        alt = tmp_path / "alt-cred.json"
        alt.write_text(json.dumps({"credentials": [
            _entry(name="alt-agent", token="alt-tok"),
        ]}), encoding="utf-8")
        monkeypatch.setenv("ARGOS_API_CREDENTIAL", "alt-agent")
        monkeypatch.setenv("ARGOS_API_CREDENTIAL_FILE", str(alt))
        ctx = mcp_server._load_auth_context(tmp_path / "nohome")
        assert ctx.principal == "alt-agent"

    def test_legacy_env_path_unchanged(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ARGOS_API_PRINCIPAL", "legacy-user")
        monkeypatch.setenv("ARGOS_API_USER_ID", "legacy-uid")
        ctx = mcp_server._load_auth_context(tmp_path / "nohome")
        assert ctx.principal == "legacy-user"
        assert ctx.user_id == "legacy-uid"
        assert ctx.transport == "mcp-stdio"

    def test_loopback_flag_strips_write_class(self, tmp_path, monkeypatch):
        home = self._home(tmp_path)
        write_credential(
            credential_file_path(home), name="w", user_id="u",
            allowed_classes=["read", "write"],
        )
        monkeypatch.setenv("ARGOS_API_CREDENTIAL", "w")
        monkeypatch.setenv("ARGOS_API_NO_LOOPBACK", "1")
        ctx = mcp_server._load_auth_context(home)
        assert "memory_save" not in ctx.allowed_operations


# =============================================================================
# Facade integration (class gates under credential contexts)
# =============================================================================


class TestFacadeIntegration:
    def _facade_store(self, tmp_path):
        from store import DuckDBMemoryStore

        store = DuckDBMemoryStore(tmp_path / "t.duckdb", user_id="test_user")
        facade = ArgosAPIFacade(store, acl=ACLConfig(), api_mode=False)
        return facade, store

    def _seed_unreviewed(self, store):
        return store.remember(
            category="context_note",
            content="Michael keeps a spare garage door remote in the kitchen drawer",
            source="explicit",
            trust_class="unreviewed",
        )

    def test_model_credential_cannot_review(self, tmp_path):
        facade, store = self._facade_store(tmp_path)
        mem = self._seed_unreviewed(store)
        ctx = build_context(
            _mk_cred(["read", "review"], principal_type="model", user_id="test_user"),
            transport="rest", is_loopback=True,
        )
        with pytest.raises(APIError) as exc:
            facade.execute(ctx, "review_memory", {
                "memory_id": mem.memory_id, "decision": "promote", "reason": "ok",
            })
        assert exc.value.code == "forbidden"

    def test_human_credential_can_review(self, tmp_path):
        facade, store = self._facade_store(tmp_path)
        mem = self._seed_unreviewed(store)
        ctx = build_context(
            _mk_cred(["read", "review"], principal_type="human", user_id="test_user"),
            transport="rest", is_loopback=True,
        )
        result = facade.execute(ctx, "review_memory", {
            "memory_id": mem.memory_id, "decision": "promote", "reason": "ok",
        })
        assert result.get("changed") is True

    def test_review_class_not_implied_by_propose(self, tmp_path):
        facade, store = self._facade_store(tmp_path)
        mem = self._seed_unreviewed(store)
        ctx = build_context(
            _mk_cred(["read", "propose"], principal_type="human", user_id="test_user"),
            transport="rest", is_loopback=True,
        )
        with pytest.raises(APIError) as exc:
            facade.execute(ctx, "review_memory", {
                "memory_id": mem.memory_id, "decision": "promote", "reason": "ok",
            })
        assert exc.value.code == "forbidden"

    def test_propose_class_cannot_write(self, tmp_path):
        facade, _ = self._facade_store(tmp_path)
        ctx = build_context(
            _mk_cred(["read", "propose"], user_id="test_user"),
            transport="rest", is_loopback=True,
        )
        with pytest.raises(APIError) as exc:
            facade.execute(ctx, "memory_save", {
                "content": "x", "category": "context_note",
            })
        assert exc.value.code == "forbidden"

    def test_nonloopback_write_class_denied(self, tmp_path):
        facade, _ = self._facade_store(tmp_path)
        ctx = build_context(
            _mk_cred(["read", "write"], user_id="test_user"),
            transport="rest", is_loopback=False,
        )
        with pytest.raises(APIError) as exc:
            facade.execute(ctx, "memory_save", {
                "content": "x", "category": "context_note",
            })
        assert exc.value.code == "forbidden"
        assert "not authorized" in exc.value.message

    def test_facade_loopback_gate_still_enforced(self, tmp_path):
        """Belt-and-suspenders: even if a ctx somehow carries write ops,
        the facade's own loopback gate denies class C off-loopback."""
        facade, _ = self._facade_store(tmp_path)
        ctx = AuthContext(
            principal="forged", tenant="default", user_id="test_user",
            transport="rest", allowed_operations=set(WRITE_OPERATIONS),
            principal_type="model", is_loopback=False,
        )
        with pytest.raises(APIError) as exc:
            facade.execute(ctx, "memory_save", {
                "content": "x", "category": "context_note",
            })
        assert exc.value.code == "forbidden"
        assert "loopback" in exc.value.message


# -- Revocation (#484 Phase 2) ------------------------------------------------

class TestRevocation:
    """revoke_credential / drop_legacy_token — entry removal, atomic."""

    def test_revoke_removes_only_that_entry(self, tmp_path: Path):
        p = credential_file_path(tmp_path)
        write_credential(p, name="a", principal_type="human",
                         allowed_classes=["read"], user_id="default_user")
        write_credential(p, name="b", principal_type="human",
                         allowed_classes=["read"], user_id="default_user")
        assert revoke_credential(p, "a") is True
        _, creds = parse_credentials_file(p)
        assert [c.name for c in creds] == ["b"]

    def test_revoke_missing_entry_is_noop(self, tmp_path: Path):
        p = credential_file_path(tmp_path)
        write_credential(p, name="a", principal_type="human",
                         allowed_classes=["read"], user_id="default_user")
        assert revoke_credential(p, "nope") is False
        _, creds = parse_credentials_file(p)
        assert [c.name for c in creds] == ["a"]

    def test_revoke_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(CredentialFileError):
            revoke_credential(tmp_path / "nope.json", "a")

    def test_revoke_malformed_file_raises(self, tmp_path: Path):
        p = tmp_path / "api_credential.json"
        p.write_text("{broken", encoding="utf-8")
        with pytest.raises(CredentialFileError):
            revoke_credential(p, "a")

    def test_drop_legacy_preserves_credentials(self, tmp_path: Path):
        p = credential_file_path(tmp_path)
        write_credential(p, name="a", principal_type="human",
                         allowed_classes=["read"], user_id="default_user")
        data = json.loads(p.read_text(encoding="utf-8"))
        data["token"] = "legacy-secret"
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")
        assert drop_legacy_token(p) is True
        legacy, creds = parse_credentials_file(p)
        assert legacy is None
        assert [c.name for c in creds] == ["a"]

    def test_drop_legacy_without_token_is_noop(self, tmp_path: Path):
        p = credential_file_path(tmp_path)
        write_credential(p, name="a", principal_type="human",
                         allowed_classes=["read"], user_id="default_user")
        assert drop_legacy_token(p) is False

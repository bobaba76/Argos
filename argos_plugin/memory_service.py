"""Single-owner local memory service for Hermes hybrid memory.

The service owns the canonical DuckDB and Kùzu files. Hermes desktop, CLI, and
remote gateway plugin instances communicate with it over loopback TCP so no
client opens the database files directly.
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import signal
import socket
import socketserver
import sys
import threading
import time
from pathlib import Path
from typing import Any

if __package__:
    from .embeddings import LocalEmbedder
    from .graph import KuzuGraphStore
    from .store import DuckDBMemoryStore
    from .access_scoping import ACLConfig, filter_records_by_access
    from .config_validation import storage_name_error
    from .config_model import MemoryConfig
    from .store_protocol import _PROTOCOL_VERSION
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from embeddings import LocalEmbedder
    from graph import KuzuGraphStore
    from store import DuckDBMemoryStore
    from access_scoping import ACLConfig, filter_records_by_access
    from config_validation import storage_name_error
    from config_model import MemoryConfig
    from store_protocol import _PROTOCOL_VERSION

logger = logging.getLogger("argos.service")
_ENDPOINT_NAME = "hybrid_memory_service.json"
_MAX_REQUEST_BYTES = 4 * 1024 * 1024
# MS6: response size limit — prevents unbounded memory from large responses.
_MAX_RESPONSE_BYTES = 16 * 1024 * 1024  # 16 MB
# MS8: single-instance guard probe read cap.
_PROBE_MAX_BYTES = 65536


class ServiceUnavailableError(RuntimeError):
    """A requested subsystem is not available in this service process."""


# #333: exception types whose message is written for the caller (auth
# denials, validation errors, unknown methods) and may cross the wire.
# Anything else (DuckDB/Kùzu errors, OSError, unexpected failures) is
# reported with a generic message; the detail stays in the server log.
_CLIENT_SAFE_EXCEPTIONS: tuple[type[BaseException], ...] = (
    PermissionError,
    ValueError,
    LookupError,
    TypeError,
    ServiceUnavailableError,
)
_INTERNAL_ERROR_CLASS = "InternalError"
_INTERNAL_ERROR_MESSAGE = "internal service error"
_MAX_ERROR_MESSAGE_CHARS = 512
# Windows drive paths, UNC paths, and POSIX/home paths with >= 2 segments.
_PATH_RE = re.compile(
    r"(?<![\w.])(?:"
    r"[A-Za-z]:[\\/][^\s'\"<>|,;()]*"
    r"|\\\\[^\s'\"<>|,;()]+"
    r"|~?/(?:[^\s'\"<>|,;()/]+/)+[^\s'\"<>|,;()]*"
    r")"
)
_SQL_RE = re.compile(
    r"\b(?:SELECT|INSERT|UPDATE|DELETE|CREATE|ALTER|DROP|FROM|WHERE|JOIN|"
    r"MATCH|MERGE|RETURN)\b"
)


def _redact_error_message(message: str) -> str:
    """Strip filesystem paths and SQL/Cypher fragments from a message."""
    if _SQL_RE.search(message):
        return _INTERNAL_ERROR_MESSAGE
    message = _PATH_RE.sub("<path>", message)
    if len(message) > _MAX_ERROR_MESSAGE_CHARS:
        message = message[:_MAX_ERROR_MESSAGE_CHARS] + "..."
    return message


def _classify_error(exc: BaseException) -> tuple[str, str]:
    """Map an exception to the (error, error_class) sent to the client."""
    if isinstance(exc, _CLIENT_SAFE_EXCEPTIONS):
        return _redact_error_message(str(exc)), type(exc).__name__
    return _INTERNAL_ERROR_MESSAGE, _INTERNAL_ERROR_CLASS


# MS1/MS9: server-set fields that clients may NOT inject via **args.
# These are stripped from client-supplied args before passing to the store
# or graph. The facade (api_facade.py) already strips these — the RPC
# service must too, since it's a separate trust boundary.
_FORBIDDEN_CLIENT_ARGS = frozenset({
    "provenance_origin",
    "grounding",
    "status",
    "source",
    "user_scope",
    "confidence",
    "review_mode",  # policy is server-derived (#127)
    "approval_mode",  # Spec-13 (#393 S1): write policy is server-derived —
                      # a client-supplied value could flip a tenant's
                      # approval policy; the service applies the tenant
                      # policy value in _call_store.
    "requested_by",  # #293: erase-receipt attribution is server-derived —
                     # a client-supplied requested_by would forge the
                     # audit trail of a provable deletion.
    # #200 Spec-10: facade_delete_memory identity args — the service
    # resolves user_id from its own context (same as erase_subject #293).
    # A client-chosen user_id would let any local process with the
    # endpoint token delete another user's memories.
    "user_id",
    "tenant",
    # #200 PR-2 fix: confirm is NOT a client-supplied gate authority.
    # A raw RPC caller can pass confirm=True and bypass the gate. The
    # real gate authority is the _confirmed flag in the RPC request
    # ENVELOPE (set by _SharedRPC._call_gated, NOT from client args).
    # Stripping confirm here ensures a client-supplied confirm never
    # reaches the gate logic.
    "confirm",
})

# #423: review privilege classes a GATED (HMAC-confirmed) RPC call may
# claim. The service honors these only on the gated channel; ungated
# review calls are server-derived to "auto_review" regardless of what
# the args claim, so a raw RPC caller with the endpoint token cannot
# obtain the user-confirmed class (tool/manual) at the storage boundary.
_GATED_REVIEW_SOURCES = frozenset({"tool", "manual", "auto_review", "system"})

# MS2: destructive/admin methods forbidden on the RPC boundary (same as
# the facade's FORBIDDEN_OPERATIONS). Any local process with the endpoint
# token should NOT be able to call these.
#
# DELIBERATE EXCEPTION (#293): ``erase_subject`` is NOT in this set.
# It is destructive, but it is the SANCTIONED POPIA path — the RPC
# handler enforces the same controls as the facade: strict confirm
# (only literal True), preview-first, per-record report, tenant scoping
# from the service-resolved identity, legal-hold hook, and an
# append-only deletion receipt per erased record. Blocking it here
# would break the RPC-backed REST facade (rest_server constructs the
# facade over SharedMemoryStore). The raw destructive ops above stay
# forbidden — erasure must go through the gated workflow, not ad-hoc
# deletes. Mirrors the run_compaction precedent (#281).
_FORBIDDEN_STORE_METHODS = frozenset({
    "delete_memory",
    "quarantine_memory",
    "restore_memory",
    "cleanup_junk",
    "consolidate",
    "purge_tombstone",
    "mark_superseded",
    "set_state",  # MS7: system state corruption risk
})

_FORBIDDEN_GRAPH_METHODS = frozenset({
    "clear_scope",
})

# #200 Spec-10 PR 2/3 fix: Collection WRITE methods that are gated at the
# raw-RPC seam. These are NOT in _FORBIDDEN_STORE_METHODS (the facade
# legitimately calls them), but a raw RPC caller with the endpoint token
# must NOT be able to bypass the facade's loopback gate + audit. The
# dispatch handler requires confirm=True (literal bool — same precedent
# as facade_delete_memory and erase_subject #293) and writes an audit
# row on every call (success + denial). Reads (list_collections,
# list_collection_items, count_collection_items, get_collection) are
# NOT gated — they're read-only and scope-filtered.
_GATED_COLLECTION_WRITE_METHODS = frozenset({
    "create_collection",
    "add_collection_item",
    "update_collection_item",
    "remove_collection_item",
})


def _sanitize_args(args: dict) -> dict:
    """MS1/MS9: strip server-set fields from client-supplied args.

    The RPC service is a trust boundary — clients can send arbitrary keys
    in ``args``. Fields like ``provenance_origin``, ``grounding``,
    ``status``, ``source``, ``user_scope``, ``confidence``, and
    ``review_mode`` are server-set and must not be injectable.
    """
    if not isinstance(args, dict):
        return {}
    return {k: v for k, v in args.items() if k not in _FORBIDDEN_CLIENT_ARGS}


def _load_config(home: Path) -> dict:
    path = home / "hybrid_memory.json"
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            return {}
    except Exception as exc:
        # Log + return defaults — a malformed config must not be silently
        # ignored (matches provider_core._load_config's behaviour).
        logger.warning("malformed config %s: %s", path, exc)
        return {}
    # #244: validate known keys via MemoryConfig.  Extra keys (tenants,
    # backup, review_mode) are allowed here because memory_service has
    # tenant-specific config that is not part of the global model.
    known = set(MemoryConfig.model_fields.keys())
    clean = {k: v for k, v in value.items() if k in known}
    try:
        MemoryConfig.model_validate(clean)
    except Exception as exc:
        logger.warning("config validation error in %s: %s", path, exc)
    return value


def endpoint_path(home: str | Path) -> Path:
    return Path(home) / _ENDPOINT_NAME


def _record_to_dict(record: Any) -> dict | None:
    if record is None:
        return None
    return record.to_dict() if hasattr(record, "to_dict") else record


# -- Per-tenant policy (#127) ------------------------------------------------

def _cfg_bool(val: Any, default: bool = False) -> bool:
    if isinstance(val, bool):
        return val
    return str(val).strip().lower() in ("true", "1", "yes", "on") if val is not None else default


def _cfg_int(val: Any, default: int) -> int:
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


class TenantPolicy:
    """Per-tenant policy overlay (#127).

    Captures all policy fields that must be tenant-scoped:
    - review_mode: "confirm" (default, human approval required) or "auto"
    - approval_mode: "auto" (default, writes materialize immediately with
      a server-classified trust tier) or "human" (v1 queue behavior);
      Spec-13 #393
    - max_injected_items: injection cap per turn
    - inject_content_char_cap: per-memory char cap in injection
    - external_sources_require_confirmation: external-origin auto-activate gate
    - local_only: no plugin-owned auxiliary LLM calls

    Policy is resolved from the tenant's merged config at startup and
    is immutable for the lifetime of a request. A client-supplied
    argument cannot change policy state.

    Any policy not explicitly overridden by the tenant config inherits
    from the global configuration (already merged into the tenant's
    config dict by _parse_tenants).
    """

    __slots__ = (
        "review_mode", "approval_mode", "max_injected_items",
        "inject_content_char_cap",
        "external_sources_require_confirmation", "local_only",
    )

    def __init__(self, config: dict) -> None:
        self.review_mode: str = str(
            config.get("review_mode", "confirm")
        ).strip().lower()
        if self.review_mode not in ("confirm", "auto"):
            self.review_mode = "confirm"  # fail closed
        # Spec-13 (#393 S1): write policy. "auto" (default): saves
        # materialize immediately, medium/high-risk signals stamped
        # unreviewed (never auto-promoted). "human": v1 queue behavior.
        # Invalid values fail closed to "human".
        self.approval_mode: str = str(
            config.get("approval_mode", "auto")
        ).strip().lower()
        if self.approval_mode not in ("auto", "human"):
            self.approval_mode = "human"  # fail closed
        self.max_injected_items: int = max(
            0, min(_cfg_int(config.get("max_injected_items"), 5), 50)
        )
        self.inject_content_char_cap: int = max(
            100, min(_cfg_int(config.get("inject_content_char_cap"), 800), 5000)
        )
        self.external_sources_require_confirmation: bool = _cfg_bool(
            config.get("external_sources_require_confirmation"), True
        )
        self.local_only: bool = _cfg_bool(
            config.get("local_only"), False
        )

    def to_dict(self) -> dict:
        return {
            "review_mode": self.review_mode,
            "approval_mode": self.approval_mode,
            "max_injected_items": self.max_injected_items,
            "inject_content_char_cap": self.inject_content_char_cap,
            "external_sources_require_confirmation": self.external_sources_require_confirmation,
            "local_only": self.local_only,
        }


class _Tenant:
    """One isolated cell: own store, own graph, own locks (#49).

    Isolation is filesystem-level by construction — each tenant has its
    own ``.duckdb`` file and graph dir. ``user_scope`` filtering stays in
    the store as defense-in-depth. Each tenant also carries its own locks,
    so one tenant's long consolidate/backup never blocks another tenant's
    calls.
    """

    def __init__(
        self,
        name: str,
        config: dict,
        home: Path,
        embedder,
        reranker,
        default_scope: str | None = None,
    ) -> None:
        self.name = name
        self.config = config
        db_name = str(config.get("database_filename", "hybrid_memory.duckdb"))
        graph_name = str(config.get("graph_dirname", "hybrid_memory_kuzu"))
        # Default scope for direct (non-RPC) calls like the startup hygiene
        # sweep. The DEFAULT tenant keeps the historical "default_user" so
        # the sweep still sees entities written by default clients — a
        # scope change here silently disabled the sweep (#49 review).
        self.default_scope = default_scope or "default_user"
        # Per-tenant policy overlay (#127): captures review_mode, injection
        # caps, local_only, external_sources_require_confirmation. Resolved
        # from the tenant's merged config (global + overlay). Immutable for
        # the lifetime of a request — a client-supplied argument cannot
        # change policy state.
        self.policy = TenantPolicy(config)
        # The tenant name is the store's default scope; every request
        # re-scopes per user_id anyway (defense in depth).
        # BK3: recover from a crashed restore before opening the store —
        # if a previous restore crashed between swap steps, the live DB
        # may be missing and the old DB is at .pre-restore.duckdb.
        try:
            if __package__:
                from .backup import recover_from_failed_restore
            else:
                from backup import recover_from_failed_restore
            if recover_from_failed_restore(home / db_name):
                logger.warning(
                    "Tenant %r: recovered live DB from a failed restore.",
                    name,
                )
        except Exception as exc:
            logger.warning(
                "Tenant %r: startup restore-recovery check failed: %s",
                name, exc,
            )
        self.store = DuckDBMemoryStore(
            home / db_name, user_id=self.default_scope,
            embedder=embedder, reranker=reranker,
        )
        try:
            self.store._reranker_top_n = max(
                5, min(int(config.get("reranker_top_n", 20)), 100)
            )
        except (TypeError, ValueError):
            self.store._reranker_top_n = 20
        try:
            self.store._phrase_lift_alpha = max(
                0.0, min(float(config.get("phrase_lift_alpha", 0.0)), 1.0)
            )
        except (TypeError, ValueError):
            self.store._phrase_lift_alpha = 0.0
        try:
            self.store._phrase_lift_pool = max(
                0, min(int(config.get("phrase_lift_pool", 200)), 1000)
            )
        except (TypeError, ValueError):
            self.store._phrase_lift_pool = 200
        # External-source write policy (per-tenant overlay, #127).
        # Now resolved through TenantPolicy and applied to the store.
        self.store.external_sources_require_confirmation = (
            self.policy.external_sources_require_confirmation
        )
        try:
            self.graph = KuzuGraphStore(home / graph_name, user_id=self.default_scope)
        except Exception as exc:
            logger.warning(
                "Kùzu unavailable for tenant %r: %s", name, exc,
            )
            self.graph = None
        # Per-tenant ACL (#128): load ACLConfig from the tenant's config
        # overlay. The acl key may contain roles, user_roles, deny_lists,
        # and enforcement_on. Missing ACL config = open store (backward
        # compatible). When enforcement is on, unassigned users fail
        # closed (deny-all).
        acl_data = config.get("acl") if isinstance(config.get("acl"), dict) else {}
        self.acl = ACLConfig.from_dict(acl_data)
        # Attach the ACL to the store and graph so the existing
        # _acl_config checks in provider_retrieval.py and graph.py
        # enforce it. This is the production wiring that was missing.
        self.store._acl_config = self.acl
        if self.graph is not None:
            self.graph._acl_config = self.acl
        self.store_lock = threading.RLock()
        self.graph_lock = threading.RLock()

    def close(self) -> None:
        with self.store_lock:
            try:
                self.store.close()
            finally:
                if self.graph:
                    with self.graph_lock:
                        self.graph.close()


# -- #130: Tenant config validation -----------------------------------------

import re as _re

_TENANT_NAME_RE = _re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$")


def _validate_tenant_name(name: str) -> None:
    """Validate a tenant name: alphanumeric + _-, 1-64 chars, no path chars."""
    if not isinstance(name, str) or not name:
        raise ValueError(f"Invalid tenant name: {name!r} (must be non-empty)")
    if not _TENANT_NAME_RE.match(name):
        raise ValueError(
            f"Invalid tenant name: {name!r} (must be alphanumeric, _, or -, "
            f"1-64 chars, start with alphanumeric)"
        )


def _validate_tenant_path(path: str, tenant_name: str, field: str) -> None:
    """Validate a tenant database/graph path: relative, no traversal."""
    reason = storage_name_error(path)
    if reason is not None:
        raise ValueError(f"Tenant {tenant_name!r}: {field}={path!r} {reason}")


def _parse_tenants(
    config: dict, home: Path, embedder, reranker,
) -> tuple[dict, dict, bool, dict, bool]:
    """Build the tenant registry from ``hybrid_memory.json`` (#49).

    With a ``tenants`` map, each entry is a cell: its own database + graph
    paths (relative to home) and a nested ``config`` overlay applied on top
    of the global config. Without ``tenants`` (the current single-tenant
    shape), the global config IS the ``default`` tenant — fully backward
    compatible.

    Returns ``(tenants, user_tenant_map, strict_routing, credential_map, credential_mode)``:

    - ``tenants``: name -> _Tenant
    - ``user_tenant_map``: user_id -> tenant name, built from optional
      ``allowed_user_ids`` lists on each tenant entry (#87). A user_id may
      appear in at most one tenant's allowlist; duplicates are a config
      error.
    - ``strict_routing``: True if any tenant declared ``allowed_user_ids``.
      When True, dispatch rejects user_ids not in the map (defense-in-depth
      against tenant spoofing by any local process that holds the endpoint
      token). When False, the legacy fallback-to-default behavior is
      preserved (backward compat) with a startup warning.
    - ``credential_map``: token_hash -> (tenant_name, user_id), built from
      optional per-tenant ``credentials`` lists (#129). The server derives
      identity from the credential, not from a client-supplied user_id.
    - ``credential_mode``: True if any tenant declared credentials. When
      True, the service is in multi-user/hosted mode — credentials are
      required and client-supplied user_id is ignored/validated. When
      False, trusted-local mode (legacy single endpoint token).
    """
    tenants_map = config.get("tenants")
    if not isinstance(tenants_map, dict) or not tenants_map:
        tenants_map = {"default": config}
    tenants: dict = {}
    user_tenant_map: dict = {}
    strict_routing = False
    credential_map: dict = {}  # token_hash -> (tenant_name, user_id)
    credential_mode = False
    # #130: Track database/graph paths to detect collisions across tenants.
    seen_db_paths: dict = {}  # db_filename -> tenant_name
    seen_graph_paths: dict = {}  # graph_dirname -> tenant_name
    for name, entry in tenants_map.items():
        entry = entry if isinstance(entry, dict) else {}
        # #130: Validate tenant name — must be a safe identifier.
        _validate_tenant_name(name)
        overlay = entry.get("config") if isinstance(entry.get("config"), dict) else {}
        merged = dict(config)
        merged.update(overlay)
        # Cell paths come from the tenant entry itself.
        merged["database_filename"] = entry.get(
            "database_filename", config.get("database_filename", "hybrid_memory.duckdb")
        )
        merged["graph_dirname"] = entry.get(
            "graph_dirname", config.get("graph_dirname", "hybrid_memory_kuzu")
        )
        # #130: Validate paths — no path traversal, no cross-tenant
        # collisions. Paths must be relative to home (no absolute, no ..).
        _validate_tenant_path(
            str(merged["database_filename"]), name, "database_filename",
        )
        _validate_tenant_path(
            str(merged["graph_dirname"]), name, "graph_dirname",
        )
        if merged["database_filename"] in seen_db_paths:
            raise ValueError(
                f"database_filename collision: {merged['database_filename']!r} "
                f"used by both tenant {seen_db_paths[merged['database_filename']]!r} "
                f"and {name!r}"
            )
        seen_db_paths[merged["database_filename"]] = name
        if merged["graph_dirname"] in seen_graph_paths:
            raise ValueError(
                f"graph_dirname collision: {merged['graph_dirname']!r} "
                f"used by both tenant {seen_graph_paths[merged['graph_dirname']]!r} "
                f"and {name!r}"
            )
        seen_graph_paths[merged["graph_dirname"]] = name
        # Default tenant keeps the historical "default_user" scope (#49
        # review): the startup hygiene sweep and any direct store/graph
        # calls must see the data default clients write.
        default_scope = "default_user" if name == "default" else name
        tenants[name] = _Tenant(
            name, merged, home, embedder, reranker, default_scope=default_scope,
        )
        # Optional per-tenant user_id allowlist (#87).
        allowed = entry.get("allowed_user_ids")
        if isinstance(allowed, list):
            for uid in allowed:
                uid_s = str(uid)
                if uid_s in user_tenant_map and user_tenant_map[uid_s] != name:
                    raise ValueError(
                        f"allowed_user_ids conflict: {uid_s!r} appears in both "
                        f"tenant {user_tenant_map[uid_s]!r} and {name!r}"
                    )
                user_tenant_map[uid_s] = name
            strict_routing = True
        # Optional per-tenant credentials (#129): each credential binds a
        # token to a user_id within this tenant. The server derives
        # identity from the credential — a client-supplied user_id is
        # validated against the credential's user_id, not trusted.
        creds = entry.get("credentials")
        if isinstance(creds, list):
            for cred in creds:
                if not isinstance(cred, dict):
                    continue
                token = str(cred.get("token", ""))
                cred_user = str(cred.get("user_id", ""))
                if not token or not cred_user:
                    continue
                # Hash the token so we never store raw tokens in memory
                # longer than needed. The lookup uses hmac.compare_digest
                # against the hash.
                token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
                if token_hash in credential_map:
                    existing_tenant, existing_user = credential_map[token_hash]
                    if existing_tenant != name or existing_user != cred_user:
                        raise ValueError(
                            f"credential conflict: token appears in both "
                            f"tenant {existing_tenant!r} (user {existing_user!r}) "
                            f"and tenant {name!r} (user {cred_user!r})"
                        )
                credential_map[token_hash] = (name, cred_user)
                credential_mode = True
    return tenants, user_tenant_map, strict_routing, credential_map, credential_mode


class MemoryService:
    def __init__(self, home: Path) -> None:
        self.home = home
        config = _load_config(home)
        model_name = str(
            config.get(
                "local_embedding_model",
                "BAAI/bge-small-en-v1.5",
            )
        )
        self.embedder = LocalEmbedder(model_name)
        # Reranker (lazy — loads on first rerank call)
        reranker_enabled = str(config.get("reranker_enabled", "true")).lower() in (
            "true", "1", "yes"
        )
        reranker = None
        if reranker_enabled:
            try:
                if __package__:
                    from .embeddings import CrossEncoderReranker
                else:
                    from embeddings import CrossEncoderReranker
                reranker_model = str(
                    config.get("reranker_model", "BAAI/bge-reranker-base")
                )
                reranker = CrossEncoderReranker(reranker_model)
            except Exception as exc:
                logger.warning("Reranker unavailable in shared service: %s", exc)
        # Tenant registry (#49): per-tenant stores/graphs behind one service.
        # _parse_tenants also returns the user_id→tenant allowlist,
        # strict-routing flag (#87), credential map, and credential mode
        # (#129).
        (
            self._tenants,
            self._user_tenant_map,
            self._strict_routing,
            self._credential_map,
            self._credential_mode,
        ) = _parse_tenants(config, home, self.embedder, reranker)
        if self._credential_mode:
            logger.info(
                "Credential mode enabled (#129): %d credential(s) across "
                "%d tenant(s). Server derives identity from credentials; "
                "client-supplied user_id is validated, not trusted.",
                len(self._credential_map), len(self._tenants),
            )
        elif self._strict_routing:
            logger.info(
                "Strict tenant routing enabled (#87): %d user_id(s) allowlisted "
                "across %d tenant(s). Trusted-local mode — any local process "
                "with the endpoint token can spoof any allowlisted user_id.",
                len(self._user_tenant_map), len(self._tenants),
            )
        elif len(self._tenants) > 1:
            logger.warning(
                "Multi-tenant config without allowed_user_ids (#87): any local "
                "process with the endpoint token can spoof any user_id and access "
                "any tenant. Add 'allowed_user_ids' to each tenant entry to "
                "enable strict routing, or add 'credentials' for multi-user mode."
            )
        # Backward-compat aliases: the DEFAULT tenant's handles. Anything
        # that referenced self.store/self.graph before still works.
        self._default_tenant = "default" if "default" in self._tenants else next(
            iter(self._tenants)
        )
        default_tenant = self._tenants[self._default_tenant]
        self.store = default_tenant.store
        self.graph = default_tenant.graph
        self._lock_wait_total_s = 0.0
        self._lock_wait_count = 0
        self.server = None

    def _resolve_tenant(self, user_id: str) -> _Tenant:
        """Map a user_id to its tenant cell (#49, #87).

        Strict mode (#87 — any tenant has ``allowed_user_ids``):
        - user_id in the allowlist → route to the mapped tenant
        - user_id NOT in the allowlist → raise PermissionError

        Legacy mode (no ``allowed_user_ids`` anywhere):
        - Exact match on tenant name → that tenant
        - Unknown user_id → default tenant (backward compatible)
        """
        if self._strict_routing:
            tenant_name = self._user_tenant_map.get(user_id)
            if tenant_name is None:
                raise PermissionError(
                    f"user_id {user_id!r} is not allowlisted on this service "
                    f"(#87 strict routing)"
                )
            return self._tenants[tenant_name]
        if user_id in self._tenants:
            return self._tenants[user_id]
        return self._tenants[self._default_tenant]

    def _call_store(self, method: str, args: dict, user_id: str, store, policy: "TenantPolicy | None" = None, tenant: "_Tenant | None" = None, confirmed: bool = False) -> Any:
        store.set_user_scope(user_id)
        # #347: stamp the actor server-side per-request. set_user_scope
        # resets the actor to user_id/"human"; re-stamp with the
        # server-derived identity here so mutation_events records the
        # real actor. In credential mode (#129), the identity came from
        # a credential — force "model" since the service resolved it,
        # not a human-driven envelope (#341/#344). In trusted-local
        # mode, the caller is the local user — "human" is the right
        # default (the facade overrides with ctx.principal_type when
        # available, but the RPC path doesn't carry that — this stamp
        # is the authoritative service-side actor).
        _actor_type = "model" if getattr(self, "_credential_mode", False) else "human"
        if hasattr(store, "set_actor_context"):
            store.set_actor_context(user_id, _actor_type)
        if method == "search":
            records = store.search(
                args.get("query", ""),
                limit=int(args.get("limit", 5)),
                exclude_categories=args.get("exclude_categories"),
                category_filter=args.get("category_filter"),
                project_id=args.get("project_id"),
                namespace=args.get("namespace"),
                client_scope=args.get("client_scope"),
                as_of=args.get("as_of"),
                suppress_retrieval=bool(args.get("suppress_retrieval", False)),
                                include_expired=bool(args.get("include_expired", False)),
                                include_closed=bool(args.get("include_closed", False)),
                                include_archived=bool(args.get("include_archived", False)),
                                trust_class=args.get("trust_class"),
                            )
            # #128: ACL enforcement — filter results by the user's ACL
            # mask. Denied content is hidden (not merely removed after
            # answer generation). Write an audit row with granted/denied
            # counts.
            acl = getattr(store, "_acl_config", None)
            if acl is not None and not acl.is_open_store:
                visible, denied_count = filter_records_by_access(records, acl, user_id)
                # Write audit row.
                store.write_access_audit(
                    user_id=user_id,
                    query_text=args.get("query", ""),
                    granted_count=len(visible),
                    denied_count=denied_count,
                    tenant=tenant.name if tenant else "default",
                    excluded=denied_count > 0,
                )
                records = visible
            return [_record_to_dict(record) for record in records]
        if method == "unreviewed_stats":
            # #393 S2: live unreviewed-class backlog (count + oldest
            # age). Scope = server-resolved caller identity.
            return store.unreviewed_stats(user_scope=user_id)
        if method == "remember":
            # MS1: strip server-set fields from client args.
            rec = store.remember(**_sanitize_args(args))
            # #200 Spec-10: post-write graph hook — index the new memory
            # in the graph (mirror what provider_session does on the
            # native path). Graph failures must not fail the write.
            if rec is not None and tenant is not None and tenant.graph is not None:
                try:
                    tenant.graph.index_memory(
                        memory_id=rec.memory_id,
                        category=rec.category,
                        content=rec.content,
                        tags=rec.tags or [],
                        created_at=rec.created_at,
                        use_llm=False,
                        valid_from=getattr(rec, "valid_from", None) or rec.created_at,
                        valid_to=getattr(rec, "valid_to", None),
                    )
                except Exception as exc:
                    logger.debug(
                        "Post-write graph index failed for %s: %s",
                        rec.memory_id, exc,
                    )
            return _record_to_dict(rec)
        if method == "update_memory":
            # MS1: strip server-set fields from client args.
            old_memory_id = args.get("memory_id", "")
            rec = store.update_memory(**_sanitize_args(args))
            # #200 Spec-10: post-write graph hook — update_memory creates
            # a new version (new memory_id); remove the old id from the
            # graph and index the new version (mirror provider_session).
            if rec is not None and tenant is not None and tenant.graph is not None:
                try:
                    if old_memory_id and old_memory_id != rec.memory_id:
                        tenant.graph.remove_memory(old_memory_id)
                except Exception as exc:
                    logger.debug(
                        "Graph remove_memory failed for updated %s: %s",
                        old_memory_id, exc,
                    )
                try:
                    tenant.graph.index_memory(
                        memory_id=rec.memory_id,
                        category=rec.category,
                        content=rec.content,
                        tags=rec.tags or [],
                        created_at=rec.created_at,
                        use_llm=False,
                        valid_from=getattr(rec, "valid_from", None) or rec.created_at,
                        valid_to=getattr(rec, "valid_to", None),
                    )
                except Exception as exc:
                    logger.debug(
                        "Post-write graph index failed for updated %s: %s",
                        rec.memory_id, exc,
                    )
            return _record_to_dict(rec)
        if method == "get_memories_by_ids":
            records = store.get_memories_by_ids(
                args.get("memory_ids", []),
                include_quarantined=bool(args.get("include_quarantined", False)),
            )
            # #128: ACL enforcement on fetch-by-ID.
            acl = getattr(store, "_acl_config", None)
            if acl is not None and not acl.is_open_store:
                visible, denied_count = filter_records_by_access(records, acl, user_id)
                if denied_count > 0:
                    store.write_access_audit(
                        user_id=user_id,
                        query_text=f"fetch:{','.join(args.get('memory_ids', [])[:5])}",
                        granted_count=len(visible),
                        denied_count=denied_count,
                        tenant=tenant.name if tenant else "default",
                        excluded=True,
                    )
                records = visible
            return [_record_to_dict(record) for record in records]
        if method == "save_candidate":
            # MS1: strip server-set fields from client args.
            _args = _sanitize_args(args)
            # Spec-13 (#393 S1): approval_mode is SERVER-derived - a
            # client-supplied value was stripped above; the tenant policy
            # value is applied here (same trust model as review_mode,
            # #127). Defaults to "auto" when no policy is in context:
            # the service IS the deployment boundary.
            _args["approval_mode"] = (
                policy.approval_mode if policy is not None else "auto"
            )
            return store.save_candidate(**_args)
        if method == "save_api_candidate":
            # #422: external-API proposal path. Provenance is server-set
            # inside the store method (source="api",
            # provenance_origin="external", grounding) — mirroring the
            # ingest_structured branch below, the wire args carry no
            # provenance params, so _sanitize_args cannot strip the
            # stamping and a raw RPC caller cannot claim it.
            # pre_scan_blocked can only downgrade (speculative/0.0); the
            # store's own inbound scan (external=True) remains the
            # authoritative boundary.
            _args = _sanitize_args(args)
            _args["approval_mode"] = (
                policy.approval_mode if policy is not None else "auto"
            )
            return store.save_api_candidate(**_args)
        if method == "ingest_structured":
            # #289: structured ingestion (JSON/CSV → memory with
            # provenance). Provenance is server-set inside
            # ingest_structured (source="structured_ingest",
            # provenance_origin="external") — the client cannot claim it
            # because the method's signature has no provenance params.
            # user_scope comes from the service-resolved identity (the
            # dispatch already ran store.set_user_scope(user_id)), not
            # from client args. client_scope/doc_class are narrowed by
            # the facade when present; over raw RPC they are accepted
            # as-is (trusted-local boundary, same as save_candidate).
            args = _sanitize_args(args)
            # Strict bool (#289 fix): bool("false") is True — only the
            # literal boolean True passes the human-in-loop gate.
            confirm = args.get("confirm", False) is True
            return store.ingest_structured(
                data=str(args.get("data", "")),
                fmt=str(args.get("fmt", "")),
                mapping=args.get("mapping") or {},
                source_name=str(args.get("source_name", "ingest")),
                mode=str(args.get("mode", "preview")),
                confirm=confirm,
                client_scope=args.get("client_scope"),
                doc_class=args.get("doc_class"),
                project_id=args.get("project_id"),
            )
        if method == "erase_subject":
            # #293: POPIA erase-request workflow (provable deletion).
            # Identity is server-derived: user_scope comes from the
            # service-resolved identity (dispatch already ran
            # store.set_user_scope(user_id)); requested_by is ALWAYS the
            # service-resolved caller — the client cannot claim it
            # ("requested_by" is in _FORBIDDEN_CLIENT_ARGS, stripped
            # above), so the receipt's audit trail cannot be forged over
            # RPC. STRICT confirm (bool("false") is True — only literal
            # True passes). The confirm gate authority is the _confirmed
            # envelope flag (set by call_gated), NOT a client-supplied
            # confirm (which is stripped by _sanitize_args). Same fix as
            # facade_delete_memory #200 PR-2.
            args = _sanitize_args(args)
            confirm = confirmed or (args.get("confirm", False) is True)
            return store.erase_subject(
                subject=str(args.get("subject", "")),
                mode=str(args.get("mode", "preview")),
                confirm=confirm,
                categories=args.get("categories"),
                client_scope=args.get("client_scope"),
                doc_class=args.get("doc_class"),
                namespace=args.get("namespace"),
                requested_by=user_id,
            )
        if method == "list_deletion_receipts":
            args = _sanitize_args(args)
            try:
                limit = max(1, int(args.get("limit", 100)))
            except (TypeError, ValueError):
                limit = 100
            return store.list_deletion_receipts(
                request_id=args.get("request_id"),
                subject=args.get("subject"),
                memory_id=args.get("memory_id"),
                limit=limit,
            )
        if method == "verify_erase_receipt":
            return store.verify_erase_receipt(
                receipt_id=str(args.get("receipt_id", "")),
            )
        if method == "export_portable":
            # #294: portable export (anti-lock-in). Read-only, scoped to
            # the service-resolved user (dispatch already ran
            # store.set_user_scope(user_id)) — a tenant can only export
            # its own data.
            args = _sanitize_args(args)
            return store.export_portable(
                categories=args.get("categories"),
                namespace=args.get("namespace"),
                client_scope=args.get("client_scope"),
                doc_class=args.get("doc_class"),
            )
        if method == "import_portable":
            # #294: portable re-import. STRICT confirm (bool("false") is
            # True — only literal True passes); rows are stamped with the
            # service-resolved user_scope inside the store method.
            args = _sanitize_args(args)
            confirm = args.get("confirm", False) is True
            return store.import_portable(
                data=str(args.get("data", "")),
                mode=str(args.get("mode", "preview")),
                confirm=confirm,
            )
        if method == "find_semantic_duplicate":
            return _record_to_dict(
                store.find_semantic_duplicate(
                    content=args.get("content", ""),
                    min_similarity=float(args.get("min_similarity", 0.88)),
                )
            )
        if method == "list_candidates":
            return store.list_candidates(**args)
        if method == "review_candidate":
            # #127: Per-tenant review_mode enforcement. If the tenant's
            # policy is "confirm" (the default), auto-review approvals are
            # downgraded to pending_user_confirmation — a human must
            # approve. This prevents a permissive tenant's auto-reviewer
            # from activating memories in a restrictive tenant.
            # Policy state is immutable — a client-supplied review_mode
            # in the args is stripped and ignored.
            args = dict(args)  # don't mutate the caller's dict
            args.pop("review_mode", None)  # policy is server-derived, not client
            # #423: the review privilege class is SERVER-DERIVED. A raw
            # RPC caller with the endpoint token must not be able to
            # claim the user-confirmed class (review_source tool/manual)
            # — it selects the grounding lift and bypasses the
            # external-origin downgrade at the storage boundary. Gated
            # calls (HMAC-verified via _SharedRPC.call_gated — the same
            # channel as confirm, #200 PR-2) may set the class
            # explicitly; ungated calls are auto_review-class regardless
            # of what the args claim (omitting the field no longer
            # defaults to "manual").
            _claimed_source = str(args.pop("review_source", "") or "")
            if confirmed and _claimed_source in _GATED_REVIEW_SOURCES:
                args["review_source"] = _claimed_source
            else:
                args["review_source"] = "auto_review"
            if policy is not None and policy.review_mode == "confirm":
                review_source = args["review_source"]
                decision = str(args.get("decision", ""))
                if review_source == "auto_review" and decision in {"approved", "reviewed_approved"}:
                    args["decision"] = "pending_user_confirmation"
                    args["reason"] = (
                        "Tenant policy (review_mode=confirm): auto-review "
                        "cannot activate memories without human confirmation. "
                        + str(args.get("reason", ""))
                    ).strip()
            return store.review_candidate(**args)
        if method == "review_memory":
            # Spec-13 S3: resolution ops (promote/dismiss) are human-made
            # decisions. The review class is SERVER-DERIVED (#423 pattern,
            # mirrors review_candidate): gated calls (HMAC-verified via
            # call_gated) may claim tool/manual; ungated raw RPC callers
            # are forced to auto_review, where the storage resolution
            # invariant refuses them loudly (no clock-as-evidence, no
            # model self-vouch; resolution only via explicit human action).
            args = dict(args)  # don't mutate the caller's dict
            _claimed_source = str(args.pop("review_source", "") or "")
            if confirmed and _claimed_source in _GATED_REVIEW_SOURCES:
                args["review_source"] = _claimed_source
            else:
                args["review_source"] = "auto_review"
            return store.review_memory(**args)
        if method == "find_supersede_candidates":
            return store.find_supersede_candidates(
                candidate_id=args.get("candidate_id", ""),
                limit=int(args.get("limit", 3)),
            )
        if method == "quarantine_memory":
            return store.quarantine_memory(**args)
        if method == "restore_memory":
            return store.restore_memory(**args)
        if method == "record_feedback":
            return store.record_feedback(**args)
        if method == "mark_superseded":
            # Live-admin supersession (e.g. retroactive chains after a new
            # benchmark supersedes an old one).  Sets valid_to; the read side
            # excludes the record automatically.
            return store._mark_superseded(
                memory_id=args.get("memory_id", ""),
                reason=args.get("reason", ""),
                superseded_by=args.get("superseded_by"),
            )
        if method == "delete_memory":
            return store.delete_memory(**args)
        # #200 Spec-10: sanctioned facade-only delete path. The facade
        # gates this with ctx.is_loopback + server-derived identity before
        # calling. delete_memory is in _FORBIDDEN_STORE_METHODS for raw
        # RPC; facade_delete_memory is NOT — but it enforces the SAME
        # controls server-side (same pattern as erase_subject #293 and
        # run_compaction #281): strict confirm (literal True only),
        # strict CAS (expected_version required, compare+delete in one
        # store call), identity stripped + service-resolved, and an audit
        # row for every call (denied included). A raw RPC caller with the
        # endpoint token cannot skip these gates.
        if method == "facade_delete_memory":
            args = _sanitize_args(args)
            memory_id = str(args.get("memory_id", ""))
            expected_version = args.get("expected_version")
            tenant_name = tenant.name if tenant else "default"
            # Gate 1: _confirmed envelope flag (NOT client-supplied confirm).
            # #200 PR-2 fix: confirm is now in _FORBIDDEN_CLIENT_ARGS and
            # stripped by _sanitize_args. The gate authority is the
            # _confirmed flag in the RPC envelope (set by call_gated),
            # which a raw RPC caller cannot forge. Same fix as collection
            # writes — closes the client-spoofable confirm=True gap.
            if not confirmed:
                store.write_access_audit(
                    user_id=user_id,
                    query_text=f"facade_delete:{memory_id}",
                    granted_count=0,
                    denied_count=1,
                    denied_scopes="not_confirmed",
                    tenant=tenant_name,
                    excluded=True,
                )
                raise PermissionError(
                    "facade_delete_memory requires a facade-confirmed "
                    "request (_confirmed in the RPC envelope). Raw RPC "
                    "deletes are not permitted — use the facade's "
                    "memory_delete operation."
                )
            # Gate 2: strict CAS — expected_version is REQUIRED. The
            # caller must send the last-seen memory_id (each update mints
            # a new id). Compare+delete in one store call: fetch the
            # record, verify it's active and its memory_id matches
            # expected_version, then delete. This is a single logical
            # transaction under the tenant store lock (already held).
            if not expected_version:
                store.write_access_audit(
                    user_id=user_id,
                    query_text=f"facade_delete:{memory_id}",
                    granted_count=0,
                    denied_count=1,
                    denied_scopes="cas_required",
                    tenant=tenant_name,
                    excluded=True,
                )
                raise PermissionError(
                    "facade_delete_memory requires expected_version "
                    "(CAS). Send the last-seen memory_id as "
                    "expected_version."
                )
            # CAS compare: verify the record exists, is active, and its
            # memory_id matches expected_version. TOCTOU note: this is
            # check-then-act under the tenant store lock, so it's safe
            # in today's single-owner loopback context. A future
            # multi-writer deployment should make this a single atomic
            # store-level call (compare_and_delete).
            records = store.get_memories_by_ids([memory_id])
            if not records:
                store.write_access_audit(
                    user_id=user_id,
                    query_text=f"facade_delete:{memory_id}",
                    granted_count=0,
                    denied_count=1,
                    denied_scopes="not_found",
                    tenant=tenant_name,
                    excluded=True,
                )
                raise ValueError(f"Memory not found: {memory_id}")
            rec = records[0]
            rec_status = getattr(rec, "status", None)
            rec_mid = getattr(rec, "memory_id", None)
            if rec_status != "active" or rec_mid != str(expected_version):
                store.write_access_audit(
                    user_id=user_id,
                    query_text=f"facade_delete:{memory_id}",
                    granted_count=0,
                    denied_count=1,
                    denied_scopes="cas_conflict",
                    tenant=tenant_name,
                    excluded=True,
                )
                raise ValueError(
                    f"CAS conflict: expected_version={expected_version} "
                    f"does not match the active record (memory_id={rec_mid}, "
                    f"status={rec_status})."
                )
            # All gates passed — delete.
            result = store.delete_memory(memory_id=memory_id)
            # Audit the successful deletion.
            store.write_access_audit(
                user_id=user_id,
                query_text=f"facade_delete:{memory_id}",
                granted_count=1,
                denied_count=0,
                denied_scopes="",
                tenant=tenant_name,
                excluded=False,
            )
            # Post-write graph hook: remove the deleted memory from the
            # graph (mirror what provider_session does on the native path).
            if tenant is not None and tenant.graph is not None:
                try:
                    tenant.graph.remove_memory(memory_id)
                except Exception as exc:
                    logger.debug(
                        "Graph remove_memory failed for deleted %s: %s",
                        memory_id, exc,
                    )
            return result
        # -- #200 Spec-10 PR 2/3: Collections dispatch -----------------------
        # All collection methods delegate to the store's StoreCollectionsMixin.
        # Identity is server-resolved (dispatch already ran
        # store.set_user_scope(user_id)); client-supplied user_scope/tenant
        # are stripped by _sanitize_args (in _FORBIDDEN_CLIENT_ARGS).
        if method == "list_collections":
            return store.list_collections(
                status=args.get("status"),
                limit=int(args.get("limit", 200)),
            )
        if method == "list_collection_items":
            return store.list_collection_items(
                collection_id=args.get("collection_id", ""),
                status=args.get("status"),
                include_archived=bool(args.get("include_archived", False)),
                limit=int(args.get("limit", 0)),
            )
        if method == "count_collection_items":
            return store.count_collection_items(
                collection_id=args.get("collection_id", ""),
                status=args.get("status"),
            )
        if method == "get_collection":
            return store.get_collection(args.get("collection_id", ""))
        # -- #200 Spec-10 PR 2/3 fix: gated collection writes -------------
        # These 4 write methods require _confirmed=True from the RPC
        # request ENVELOPE (set by _SharedRPC.call_gated), NOT a client-
        # supplied confirm in args. A raw RPC caller using call() cannot
        # set _confirmed — it's in the envelope, not in args. And
        # _sanitize_args strips 'confirm' from args (it's in
        # _FORBIDDEN_CLIENT_ARGS), so even a forged confirm=True in args
        # never reaches this gate. Every call (success + denial) writes
        # an audit row with the service-resolved user_id. Same precedent
        # as facade_delete_memory and erase_subject #293.
        if method in _GATED_COLLECTION_WRITE_METHODS:
            args = _sanitize_args(args)
            tenant_name = tenant.name if tenant else "default"
            if not confirmed:
                store.write_access_audit(
                    user_id=user_id,
                    query_text=f"collection_write:{method}",
                    granted_count=0,
                    denied_count=1,
                    denied_scopes="not_confirmed",
                    tenant=tenant_name,
                    excluded=True,
                )
                raise PermissionError(
                    f"{method} requires a facade-confirmed request "
                    f"(_confirmed in the RPC envelope). Raw RPC "
                    f"collection writes are not permitted — use the "
                    f"facade (class C loopback)."
                )
            # Gate passed — dispatch to the store method.
            try:
                if method == "create_collection":
                    result = store.create_collection(
                        name=args.get("name", ""),
                        template=args.get("template"),
                        schema=args.get("schema"),
                        tenant=tenant_name,
                    )
                elif method == "add_collection_item":
                    result = store.add_collection_item(
                        collection_id=args.get("collection_id", ""),
                        fields=args.get("fields", {}),
                        status=args.get("status", "open"),
                        tenant=tenant_name,
                    )
                elif method == "update_collection_item":
                    result = store.update_collection_item(
                        item_id=args.get("item_id", ""),
                        fields=args.get("fields"),
                        status=args.get("status"),
                        expected_version=args.get("expected_version"),
                    )
                elif method == "remove_collection_item":
                    result = store.remove_collection_item(
                        item_id=args.get("item_id", ""),
                        expected_version=args.get("expected_version"),
                    )
                else:
                    raise ValueError(f"Unhandled gated collection write: {method}")
            except ValueError as exc:
                # CAS conflict / not found / invalid — audit the denial.
                msg = str(exc).lower()
                if "cas conflict" in msg:
                    denied_reason = "cas_conflict"
                elif "not found" in msg:
                    denied_reason = "not_found"
                else:
                    denied_reason = "invalid_input"
                store.write_access_audit(
                    user_id=user_id,
                    query_text=f"collection_write:{method}",
                    granted_count=0,
                    denied_count=1,
                    denied_scopes=denied_reason,
                    tenant=tenant_name,
                    excluded=True,
                )
                raise
            # Success — audit the granted write.
            store.write_access_audit(
                user_id=user_id,
                query_text=f"collection_write:{method}",
                granted_count=1,
                denied_count=0,
                denied_scopes="",
                tenant=tenant_name,
                excluded=False,
            )
            return result
        # -- deletion tombstones (read-only visibility + escape hatch) ---------
        if method == "list_tombstones":
            return store.list_tombstones(
                limit=int(args.get("limit", 200)),
            )
        if method == "purge_tombstone":
            return store.purge_tombstone(
                content=args.get("content", ""),
                category=args.get("category", ""),
            )
        if method == "cleanup_junk":
            return store.cleanup_junk(**args)
        if method == "consolidate":
            return store.consolidate(**args)
        if method == "count":
            return store.count()
        if method == "record_retrieval":
            store.record_retrieval(args.get("memory_ids", []))
            return True
        if method == "get_evidence":
            return store.get_evidence(args.get("memory_id", ""))
        if method == "get_evidence_batch":
            return store.get_evidence_batch(args.get("memory_ids", []) or [])
        if method == "get_memory_history":
            return [
                _record_to_dict(record)
                for record in store.get_memory_history(
                    args.get("memory_id", ""),
                    max_versions=args.get("max_versions"),
                )
            ]
        if method == "get_chain_membership":
            return store.get_chain_membership(args.get("memory_ids", []) or [])
        if method == "provenance":
            return store.provenance(args.get("memory_id", ""))
        if method == "explain_retrieval":
            # Read-only diagnostic (Spec 2): why-not for a memory.
            # No writes, no retrieval side-effects — safe on the RPC
            # boundary. user_scope is enforced server-side by the store.
            return store.explain_retrieval(
                query=str(args.get("query", "")),
                expected_memory_id=str(args.get("expected_memory_id", "")),
                top_k=int(args.get("top_k", 20)),
                project_id=args.get("project_id"),
            )
        if method == "backfill_evidence":
            return store.backfill_evidence(
                retention=args.get("retention", "full")
            )
        if method == "get_scale_metrics":
            return store.get_scale_metrics()
        if method == "set_scale_thresholds":
            store.set_scale_thresholds(
                args.get("warn_latency_ms", 300.0),
                args.get("warn_records", 5000),
            )
            return True
        if method == "list_recent":
            return [
                _record_to_dict(record)
                for record in store.list_recent(
                    limit=int(args.get("limit", 100)),
                )
            ]
        if method == "list_memories":
            return [
                _record_to_dict(record)
                for record in store.list_memories(
                    category=args.get("category"),
                    limit=int(args.get("limit", 100)),
                )
            ]
        if method == "get_insights":
            return [
                _record_to_dict(record)
                for record in store.get_insights(
                    tags=args.get("tags"),
                    since=args.get("since"),
                    limit=int(args.get("limit", 50)),
                )
            ]
        if method == "add_alias":
            store.add_alias(
                alias=args.get("alias", ""),
                canonical_entity=args.get("canonical_entity", ""),
            )
            return True
        if method == "remove_alias":
            return store.remove_alias(
                alias=args.get("alias", ""),
                canonical_entity=args.get("canonical_entity"),
            )
        if method == "resolve_aliases":
            return store.resolve_aliases(args.get("text", ""))
        if method == "list_aliases":
            return store.list_aliases()
        if method == "aliases_for_canonical":
            return store.aliases_for_canonical(args.get("canonical_entity", ""))
        # #128: Audit export — restricted to wheel/principals only.
        if method == "export_access_audit":
            acl = getattr(store, "_acl_config", None)
            if acl is not None and not acl.is_open_store:
                # Only wheel users (principals) may export the audit log.
                role = acl.role_for(user_id)
                if role is None:
                    raise PermissionError(
                        "Audit export is restricted to authorized principals. "
                        "Unassigned users may not export."
                    )
                role_def = acl.roles.get(role, {})
                if not role_def.get("wheel", False):
                    raise PermissionError(
                        "Audit export is restricted to principals (wheel role). "
                        f"Role {role!r} is not authorized."
                    )
            return store.export_access_audit(
                limit=int(args.get("limit", 10000)),
                format=args.get("format", "jsonl"),
                tenant_name=tenant.name if tenant else None,
            )
        # #347: mutation_events read surface — same wheel/principals gate
        # as export_access_audit. No rotation; full history is exportable.
        if method == "export_mutation_events":
            acl = getattr(store, "_acl_config", None)
            if acl is not None and not acl.is_open_store:
                role = acl.role_for(user_id)
                if role is None:
                    raise PermissionError(
                        "Mutation event export is restricted to authorized "
                        "principals. Unassigned users may not export."
                    )
                role_def = acl.roles.get(role, {})
                if not role_def.get("wheel", False):
                    raise PermissionError(
                        "Mutation event export is restricted to principals "
                        f"(wheel role). Role {role!r} is not authorized."
                    )
            return store.export_mutation_events(
                limit=int(args.get("limit", 10000)),
                offset=int(args.get("offset", 0)),
                event_type=args.get("event_type"),
                format=args.get("format", "jsonl"),
            )
        if method == "list_mutation_events":
            return store.list_mutation_events(
                limit=int(args.get("limit", 10000)),
                offset=int(args.get("offset", 0)),
                event_type=args.get("event_type"),
            )
        # #347: set_actor_context — retained for facade/local-store
        # compatibility. The per-request stamp at the top of _call_store
        # is the authoritative service-side actor; this dispatch arm is
        # a no-op for the RPC path (the stamp above already ran) but
        # keeps the proxy method working for local stores that call it
        # directly.
        if method == "set_actor_context":
            _actor_type = args.get("actor_type", "human")
            if getattr(self, "_credential_mode", False):
                _actor_type = "model"
            store.set_actor_context(user_id, _actor_type)
            return True
        # -- system state KV + distillation data access (P4.2) -----------------
        if method == "get_state":
            return store.get_state(args.get("key", ""))
        if method == "set_state":
            store.set_state(args.get("key", ""), args.get("value", ""))
            return True
        if method == "advance_distillation_state":
            # #392: narrow server-side op for distillation run-state
            # advancement. Replaces the broad set_sanctioned_state alias.
            # Only writes distillation_last_run + distillation_last_count
            # (both in _STATE_KEY_ALLOWLIST). Follows the run_compaction
            # precedent (#281): server-side op, not in _FORBIDDEN.
            records_processed = int(args.get("records_processed", 0))
            from datetime import datetime, timezone
            now = datetime.now(timezone.utc).isoformat()
            store.set_state("distillation_last_run", now)
            store.set_state("distillation_last_count", str(records_processed))
            return True
        if method == "advance_rollup_state":
            # #427: narrow server-side op for rollup run-state advancement.
            # The proxy cannot call set_state (MS7), so rollup_last_run
            # never advanced in shared-service mode and the cooldown never
            # engaged (rollup fired at every session boundary). Writes
            # rollup_last_run + rollup_last_count only (both in
            # _STATE_KEY_ALLOWLIST). Same shape as advance_distillation_state.
            records_processed = int(args.get("records_processed", 0))
            from datetime import datetime, timezone
            now = datetime.now(timezone.utc).isoformat()
            store.set_state("rollup_last_run", now)
            store.set_state("rollup_last_count", str(records_processed))
            return True
        if method == "advance_retention_state":
            # #427 sibling: POPIA retention run-state advancement.
            # retention_last_run is in _STATE_KEY_ALLOWLIST (#293); the
            # provider's direct set_state was swallowed on the proxy path.
            from datetime import datetime, timezone
            store.set_state(
                "retention_last_run", datetime.now(timezone.utc).isoformat(),
            )
            return True
        if method == "mark_system_internal_sweep_done":
            # #392: narrow server-side op for the lexicon sweep run-once
            # guard. Only writes system_internal_sweep_done (in
            # _STATE_KEY_ALLOWLIST). Follows the run_compaction precedent.
            store.set_state("system_internal_sweep_done", "1")
            return True
        if method == "mark_system_internal_sweep_dry_run_done":
            # #392: narrow server-side op for the lexicon sweep dry-run
            # marker. Only writes system_internal_sweep_dry_run_done.
            store.set_state("system_internal_sweep_dry_run_done", "1")
            return True
        if method == "claim_stale_review_pass":
            # #425: cross-process single-flight for the stale-review sweep.
            # One pass per interval across ALL providers sharing the
            # service (previously 43 daemon starts/day raced the same
            # backlog: ~130 passes/h, concurrent passes 71 ms apart).
            # The store does an atomic compare-and-set on
            # stale_sweep_last_pass; fail-open on bookkeeping errors.
            return bool(store.claim_stale_review_pass(
                float(args.get("interval_s", 0) or 0),
            ))
        if method == "note_stale_review_outcome":
            # #425: sweep re-review bookkeeping (attempts + next eligible
            # review time) persisted on the candidate payload.
            store.note_stale_review_outcome(
                str(args.get("candidate_id", "")),
                attempts=int(args.get("attempts", 0) or 0),
                next_review_at=args.get("next_review_at"),
            )
            return True
        if method == "count_eligible_since":
            return store.count_eligible_since(
                args.get("since"),
                exclude_system_internal=bool(args.get("exclude_system_internal", False)),
            )
        if method == "load_eligible_records":
            return [
                _record_to_dict(record)
                for record in store.load_eligible_records(
                    args.get("since"), int(args.get("limit", 100)),
                    exclude_system_internal=bool(args.get("exclude_system_internal", False)),
                )
            ]
        if method == "load_high_signal_records":
            return [
                _record_to_dict(record)
                for record in store.load_high_signal_records(
                    int(args.get("limit", 20)),
                )
            ]
        if method == "load_rollup_candidates":
            return [
                _record_to_dict(record)
                for record in store.load_rollup_candidates(
                    int(args.get("limit", 100)),
                )
            ]
        if method == "count_rollup_candidates_since":
            return store.count_rollup_candidates_since(args.get("since"))
        # #312: write_access_audit — proxy the durable audit write over RPC
        # so SharedMemoryStore (and the facade via #300) can route denials
        # to the durable access_audit table. The store-side method hashes
        # query_text (SHA-256, 16 chars) — no raw query text is persisted.
        if method == "write_access_audit":
            store.write_access_audit(
                user_id=user_id,
                query_text=args.get("query_text", ""),
                granted_count=int(args.get("granted_count", 0)),
                denied_count=int(args.get("denied_count", 0)),
                denied_scopes=args.get("denied_scopes"),
                excluded=bool(args.get("excluded", False)),
                tenant=args.get("tenant"),
            )
            return None
        # #281: run_compaction — execute the schedule-aware compaction
        # pass SERVER-SIDE so it has direct access to the DuckDB store
        # (consolidate, set_state, _fetch_records, connection). The
        # client-side SharedMemoryStore proxy cannot reach these because
        # they are in _FORBIDDEN_STORE_METHODS (MS2/MS7 security model).
        # This narrow RPC method is NOT in the forbidden set — it runs
        # the compaction module's run_compaction() helper inside the
        # service process, returning the report dict.
        if method == "run_compaction":
            try:
                from compaction import run_compaction as _run_compaction
            except ImportError:
                from argos_plugin.compaction import run_compaction as _run_compaction
            return _run_compaction(
                store,
                interval_days=int(args.get("interval_days", 7)),
                aggressiveness=float(args.get("aggressiveness", 1.0)),
                dry_run=bool(args.get("dry_run", False)),
            )
        if method == "apply_system_internal_sweep":
            # #392: run the lexicon sweep SERVER-SIDE so it has direct
            # access to the DuckDB store (load_eligible_records, UPDATE
            # memory_records). The client-side SharedMemoryStore proxy
            # cannot reach store._state.lock / store.connection. Follows
            # the run_compaction precedent (#281).
            try:
                from system_internal_lexicon import sweep_system_internal as _sweep
            except ImportError:
                from argos_plugin.system_internal_lexicon import sweep_system_internal as _sweep
            return _sweep(
                store,
                dry_run=bool(args.get("dry_run", True)),
            )
        raise ValueError(f"Unsupported store method: {method}")

    def _call_graph(self, method: str, args: dict, user_id: str, graph, store=None) -> Any:
        if graph is None:
            raise ServiceUnavailableError("Relationship graph is unavailable")
        graph.set_user_scope(user_id)
        if method == "search_graph":
            return graph.search_graph(
                args.get("term", ""),
                limit=int(args.get("limit", 100)),
                as_of=args.get("as_of"),
            )
        if method == "memory_ids_for_query":
            return graph.memory_ids_for_query(
                args.get("query", ""),
                limit=int(args.get("limit", 100)),
            )
        if method == "query_graph":
            return graph.query_graph(
                args.get("entity_id", ""),
                as_of=args.get("as_of"),
            )
        if method == "traverse_graph":
            return graph.traverse_graph(
                args.get("entity_id", ""),
                depth=args.get("depth", 2),
                limit=args.get("limit", 100),
                as_of=args.get("as_of"),
            )
        if method == "count_nodes":
            return graph.count_nodes()
        if method == "count_edges":
            return graph.count_edges()
        if method == "list_nodes":
            return graph.list_nodes(
                node_type=args.get("node_type"),
                limit=int(args.get("limit", 100)),
            )
        if method == "add_relationship":
            # MS9: strip server-set fields from client args.
            # NOTE: _FORBIDDEN_CLIENT_ARGS includes "source" (provenance
            # field for memory records), but add_relationship uses "source"
            # as the source ENTITY ID — a completely different concept.
            # Use a graph-specific sanitize that preserves source/target.
            _graph_forbidden = _FORBIDDEN_CLIENT_ARGS - {"source"}
            clean = {k: v for k, v in args.items() if k not in _graph_forbidden}
            return graph.add_relationship(**clean)
        if method == "index_memory":
            # MS9: strip server-set fields from client args.
            args = _sanitize_args(args)
            # #287 (PR #323 review): derive the temporal windows
            # SERVER-SIDE from the source record — a client must not be
            # able to claim arbitrary valid_from/valid_to and forge graph
            # provenance. The record layer is the source of truth for
            # valid windows. When the record does not exist in the store,
            # client-supplied windows are stripped (fail-safe to NULL —
            # permissive as-of inclusion, no invented provenance).
            mid = args.get("memory_id")
            if mid is not None:
                args.pop("valid_from", None)
                args.pop("valid_to", None)
                if store is not None:
                    try:
                        recs = store.get_memories_by_ids(
                            [str(mid)], include_quarantined=True,
                        )
                        if recs:
                            args["valid_from"] = recs[0].valid_from
                            args["valid_to"] = recs[0].valid_to
                    except Exception as exc:
                        logger.debug(
                            "index_memory window derivation failed for %s: %s",
                            mid, exc,
                        )
            return graph.index_memory(**args)
        if method == "remove_memory":
            # MS9: strip server-set fields from client args.
            return graph.remove_memory(**_sanitize_args(args))
        if method == "purge_orphan_memory":
            # #376: hard-delete a memory node that is truly absent from
            # DuckDB (caller-verified). Graph-only, irreversible.
            return graph.purge_orphan_memory(**_sanitize_args(args))
        if method == "quarantine_junk_entities":
            return graph.quarantine_junk_entities()
        if method == "clear_scope":
            return list(graph.clear_scope())
        raise ValueError(f"Unsupported graph method: {method}")

    def dispatch(self, request: dict) -> Any:
        if request.get("method") == "health":
            return {
                "status": "ok",
                "pid": os.getpid(),
                "lock_wait_total_s": round(self._lock_wait_total_s, 4),
                "lock_wait_count": self._lock_wait_count,
            }
        if request.get("method") == "get_status":
            # MS3: in credential mode, require a valid credential.
            # MS5: restrict status to the caller's tenant only.
            caller_tenant = None
            if self._credential_mode:
                credential = str(request.get("credential") or "")
                if not credential:
                    raise PermissionError(
                        "Credential required for get_status in multi-user mode"
                    )
                cred_hash = hashlib.sha256(credential.encode("utf-8")).hexdigest()
                matched = None
                for stored_hash, (cred_tenant, cred_user) in self._credential_map.items():
                    if hmac.compare_digest(cred_hash, stored_hash):
                        matched = (cred_tenant, cred_user)
                        break
                if matched is None:
                    raise PermissionError(
                        "Authentication failed: invalid or revoked credential"
                    )
                caller_tenant = matched[0]
            # #147: return a config fingerprint + runtime info so clients
            # can detect staleness (service running old config/code vs disk).
            # #127: include per-tenant policy summaries.
            config = _load_config(self.home)
            config_str = json.dumps(config, sort_keys=True)
            config_hash = hashlib.sha256(config_str.encode("utf-8")).hexdigest()[:16]
            # MS5: in credential mode, restrict to the caller's tenant.
            visible_tenants = (
                {caller_tenant: self._tenants[caller_tenant]}
                if caller_tenant and caller_tenant in self._tenants
                else self._tenants
            )
            tenant_policies = {
                name: t.policy.to_dict()
                for name, t in visible_tenants.items()
            }
            tenant_acl_status = {
                name: {
                    "enforcement_on": t.acl.enforcement_on,
                    "is_open_store": t.acl.is_open_store,
                    "role_count": len(t.acl.roles),
                    "user_count": len(t.acl.user_roles),
                }
                for name, t in visible_tenants.items()
            }
            # #130: Per-tenant cell info — resolved paths, identity mode.
            # No secrets or personal data leaked.
            tenant_cells = {
                name: {
                    "is_default": name == self._default_tenant,
                    "database_path": str(t.config.get("database_filename", "")),
                    "graph_path": str(t.config.get("graph_dirname", "")),
                    "user_count": len([
                        u for u, tn in self._user_tenant_map.items()
                        if tn == name
                    ]),
                    "credential_count": sum(
                        1 for _h, (tn, _u) in self._credential_map.items()
                        if tn == name
                    ),
                    "acl_enforcement": t.acl.enforcement_on,
                    "review_mode": t.policy.review_mode,
                    "approval_mode": t.policy.approval_mode,
                }
                for name, t in visible_tenants.items()
            }
            # #393 S2: live unreviewed backlog per visible tenant
            # (global view — this is the admin/status surface).
            tenant_unreviewed = {}
            for name, t in visible_tenants.items():
                try:
                    _t_store = getattr(t, "store", None)
                    tenant_unreviewed[name] = (
                        _t_store.unreviewed_stats(user_scope=None)
                        if _t_store is not None
                        else None
                    )
                except Exception:
                    tenant_unreviewed[name] = None
            # #330: surface fail-loud subsystem health from the
            # subprocess's liveness singleton. In shared_service mode the
            # audit writes/purges and graph WAL flush execute INSIDE this
            # process, so the gateway's own liveness singleton is always
            # empty — the health signal must come from here.
            try:
                try:
                    from liveness import get_health
                except ImportError:
                    from .liveness import get_health
                _health = get_health()
                subsystem_health = _health.snapshot()
                degraded_subsystems = _health.degraded()
            except Exception:
                subsystem_health = {}
                degraded_subsystems = []
            # #329: cached in-service drift-watch result (visible tenants
            # only, mirroring the tenant_policies scoping above).
            try:
                try:
                    from drift_watch import last_result as _drift_last
                except ImportError:
                    from .drift_watch import last_result as _drift_last
                drift_all = _drift_last() or {}
            except Exception:
                drift_all = {}
            visible_drift = {
                name: res
                for name, res in (drift_all.get("tenants") or {}).items()
                if name in visible_tenants
            }
            return {
                "status": "ok",
                "pid": os.getpid(),
                "config_hash": config_hash,
                "reranker_enabled": str(config.get("reranker_enabled", "true")).lower() in (
                    "true", "1", "yes"
                ),
                "tenant_count": len(self._tenants),
                "strict_routing": self._strict_routing,
                "auth_mode": "multi-user" if self._credential_mode else "trusted-local",
                "credential_count": len(self._credential_map),
                "tenant_policies": tenant_policies,
                "tenant_acl_status": tenant_acl_status,
                "tenant_cells": tenant_cells,
                "unreviewed": tenant_unreviewed,
                "default_tenant": self._default_tenant,
                "lock_wait_total_s": round(self._lock_wait_total_s, 4),
                "lock_wait_count": self._lock_wait_count,
                "subsystem_health": subsystem_health,
                "degraded_subsystems": degraded_subsystems,
                "graph_drift": {
                    "checked_at": drift_all.get("checked_at"),
                    "drift": bool(drift_all.get("drift", False)),
                    "tenants": visible_drift,
                },
            }
        if request.get("method") == "stats":
            return {
                "lock_wait_total_s": round(self._lock_wait_total_s, 4),
                "lock_wait_count": self._lock_wait_count,
            }
        if request.get("method") == "shutdown":
            # MS3: in credential mode, require a valid credential.
            # shutdown is an admin operation — any client with the shared
            # service token should not be able to shut down a multi-user
            # service.
            if self._credential_mode:
                credential = str(request.get("credential") or "")
                if not credential:
                    raise PermissionError(
                        "Credential required for shutdown in multi-user mode"
                    )
                cred_hash = hashlib.sha256(credential.encode("utf-8")).hexdigest()
                matched = None
                for stored_hash, (cred_tenant, cred_user) in self._credential_map.items():
                    if hmac.compare_digest(cred_hash, stored_hash):
                        matched = (cred_tenant, cred_user)
                        break
                if matched is None:
                    raise PermissionError(
                        "Authentication failed: invalid or revoked credential"
                    )
            if self.server is not None:
                threading.Thread(target=self.server.shutdown, daemon=True).start()
            return {"status": "shutting_down"}
        if request.get("method") == "backup":
            # Service-coordinated backup: the service is the sole DB writer,
            # so CHECKPOINT + EXPORT is safe and cross-platform.  No
            # component/method split — this is a top-level service command.
            # #129: In credential mode, the backup is scoped to the
            # credential's tenant — a credential for tenant A cannot
            # back up tenant B.
            backup_args = dict(request.get("args") or {})
            if self._credential_mode:
                credential = str(request.get("credential") or "")
                if not credential:
                    raise PermissionError(
                        "Credential required for backup in multi-user mode"
                    )
                cred_hash = hashlib.sha256(credential.encode("utf-8")).hexdigest()
                matched = None
                for stored_hash, (cred_tenant, cred_user) in self._credential_map.items():
                    if hmac.compare_digest(cred_hash, stored_hash):
                        matched = (cred_tenant, cred_user)
                        break
                if matched is None:
                    raise PermissionError(
                        "Authentication failed: invalid or revoked credential"
                    )
                cred_tenant, _cred_user = matched
                # Force the backup to the credential's tenant.
                backup_args["tenant"] = cred_tenant
            return self._backup(backup_args)

        component = request.get("component")
        method = request.get("method")
        args = request.get("args") or {}
        # #129: Identity resolution. In credential mode (multi-user/hosted),
        # the server derives identity from the credential token — a client-
        # supplied user_id is validated against the credential's bound
        # user_id, not trusted. In trusted-local mode (legacy), the client
        # supplies user_id as before.
        credential = str(request.get("credential") or "")
        if self._credential_mode:
            if not credential:
                raise PermissionError(
                    "Credential required: service is in multi-user mode. "
                    "Provide a 'credential' field with a valid tenant credential token."
                )
            cred_hash = hashlib.sha256(credential.encode("utf-8")).hexdigest()
            # Constant-time lookup: compare against all stored hashes.
            matched = None
            for stored_hash, (cred_tenant, cred_user) in self._credential_map.items():
                if hmac.compare_digest(cred_hash, stored_hash):
                    matched = (cred_tenant, cred_user)
                    break
            if matched is None:
                # Stale or revoked credential — reject without revealing
                # whether the token existed.
                logger.warning(
                    "Authentication failed: invalid or revoked credential "
                    "(no secrets logged)"
                )
                raise PermissionError(
                    "Authentication failed: invalid or revoked credential"
                )
            cred_tenant, cred_user = matched
            # The server-derived user_id is authoritative. If the client
            # also supplied a user_id, it must match — otherwise it's a
            # spoofing attempt.
            client_user_id = str(request.get("user_id") or "")
            if client_user_id and client_user_id != cred_user:
                logger.warning(
                    "Identity mismatch: credential bound to user %r but "
                    "request claims user %r (tenant %r, no secrets logged)",
                    cred_user, client_user_id, cred_tenant,
                )
                raise PermissionError(
                    "Identity mismatch: credential does not match request user_id"
                )
            user_id = cred_user
            # In credential mode, the tenant is derived from the credential,
            # not from the user_tenant_map. This prevents a credential for
            # tenant A from accessing tenant B.
            tenant = self._tenants.get(cred_tenant)
            if tenant is None:
                raise PermissionError(
                    f"Credential references unknown tenant {cred_tenant!r}"
                )
        else:
            # Trusted-local mode (legacy): client supplies user_id.
            user_id = str(request.get("user_id") or "default_user")
            # Tenant routing (#49): user_id -> tenant cell. In strict mode
            # (#87), unknown user_ids are rejected instead of falling back
            # to default — prevents tenant spoofing by any local process
            # that holds the endpoint token.
            tenant = self._resolve_tenant(user_id)
        if component not in {"store", "graph"} or not isinstance(method, str):
            raise ValueError("Invalid service request")
        # MS2/MS7: operation allowlist — forbid destructive/admin methods
        # on the RPC boundary (same as the facade's FORBIDDEN_OPERATIONS).
        # Any local process with the endpoint token should NOT be able to
        # delete memories, purge tombstones, corrupt system state, etc.
        if component == "store" and method in _FORBIDDEN_STORE_METHODS:
            raise PermissionError(
                f"Method {method!r} is not available on the RPC boundary "
                f"(destructive/admin operation). Use the facade or CLI."
            )
        if component == "graph" and method in _FORBIDDEN_GRAPH_METHODS:
            raise PermissionError(
                f"Method {method!r} is not available on the RPC boundary "
                f"(destructive graph operation)."
            )
        t0 = time.monotonic()
        # Per-tenant locks (#20 + #49): store and graph calls run
        # concurrently, and one tenant's long operation never blocks
        # another tenant. Health/stats are lock-free (handled above).
        lock = tenant.store_lock if component == "store" else tenant.graph_lock
        with lock:
            self._lock_wait_total_s += time.monotonic() - t0
            self._lock_wait_count += 1
            if component == "store":
                # #200 PR-2 fix: pass the _confirmed envelope flag to
                # _call_store. This is the gate authority for collection
                # writes — it comes from the RPC envelope (set by
                # _SharedRPC.call_gated), NOT from client-supplied args.
                confirmed = request.get("_confirmed", False) is True
                return self._call_store(method, args, user_id, tenant.store, tenant.policy, tenant, confirmed)
            return self._call_graph(method, args, user_id, tenant.graph, tenant.store)

    def _backup(self, args: dict) -> Any:
        """Service-coordinated backup via EXPORT DATABASE (FORMAT PARQUET).

        Per-tenant (#49): pass ``tenant`` in args to back up a specific
        cell. #130: The tenant must be explicitly specified — no silent
        fallback to default for administrative operations. If the tenant
        is not found, raise ValueError (do not silently use default).

        BK8: only the CHECKPOINT + EXPORT phase holds the store lock; the
        verify + manifest + prune phase runs outside it so a large backup
        doesn't block the tenant's reads/writes for its whole duration.

        MS10: the backup config (dst_root, retention) is reloaded from
        disk on every call via ``_load_config``. This means if the config
        file changes between startup and backup, the backup could use
        different settings than the service was started with. This is an
        accepted v1 behavior — it allows runtime config changes without
        a service restart. For strict consistency, cache the config at
        startup (future work).
        """
        if __package__:
            from .backup import _export_for_backup, _finalize_backup, list_snapshots
        else:
            from backup import _export_for_backup, _finalize_backup, list_snapshots
        tenant_name = str(args.get("tenant") or self._default_tenant)
        # #130: No silent fallback to default for unknown tenants.
        tenant = self._tenants.get(tenant_name)
        if tenant is None:
            raise ValueError(
                f"Backup target tenant {tenant_name!r} not found. "
                f"Available: {list(self._tenants.keys())}"
            )
        # Resolve dst_root from config or default to <home>/backups/memory.
        config = _load_config(self.home)
        backup_cfg = config.get("backup", {}) if isinstance(config.get("backup"), dict) else {}
        dst_root = str(backup_cfg.get("dst_root", str(self.home / "backups" / "memory")))
        retention = int(backup_cfg.get("retention_snapshots", 6))
        # If args override (CLI can pass these), use them.
        dst_root = str(args.get("dst_root", dst_root))
        retention = int(args.get("retention_snapshots", retention))
        if args.get("list"):
            return {"snapshots": list_snapshots(dst_root)}
        # BK8: EXPORT under the store lock (needs the service's exclusive
        # DB connection); verify + manifest + prune outside it.
        with tenant.store_lock:
            snap, tables, counts, duckdb_version, schema_version = _export_for_backup(
                tenant.store.connection,
                dst_root,
                source_db_path=tenant.store.db_path,
            )
        manifest = _finalize_backup(
            snap, tables, counts, duckdb_version,
            dst_root=dst_root,
            retention_snapshots=retention,
            source_db_path=tenant.store.db_path,
            schema_version=schema_version,
        )
        manifest["tenant"] = tenant_name
        return manifest

    def close(self) -> None:
        for tenant in self._tenants.values():
            try:
                tenant.close()
            except Exception as exc:
                logger.warning("Error closing tenant %r: %s", tenant.name, exc)


class _ThreadingTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True
    # In-flight request counter (#20): lets shutdown drain handlers instead
    # of the OS killing them mid-operation (e.g. mid-backup) when the main
    # thread exits. daemon_threads stays True so a hung client never blocks
    # process exit forever — the drain is bounded.
    in_flight = 0
    in_flight_lock = threading.Lock()


class _RequestHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        server = self.server
        with server.in_flight_lock:
            server.in_flight += 1
        try:
            raw = self.rfile.readline(_MAX_REQUEST_BYTES + 1)
            if len(raw) > _MAX_REQUEST_BYTES:
                self._write({"ok": False, "error": "request too large",
                             "error_class": "RequestTooLarge"})
                return
            try:
                request = json.loads(raw.decode("utf-8"))
                if not isinstance(request, dict):
                    raise ValueError("request must be an object")
                # #246: wire protocol version check. Reject mismatched or
                # missing versions with a structured VersionMismatch error
                # (NOT a bare disconnect) so the client can self-heal.
                client_v = request.get("v")
                if client_v != _PROTOCOL_VERSION:
                    self._write({
                        "ok": False,
                        "error": {
                            "class": "VersionMismatch",
                            "supported": [_PROTOCOL_VERSION],
                            "received": client_v,
                        },
                        "error_class": "VersionMismatch",
                    })
                    return
                token = str(request.pop("token", ""))
                if not hmac.compare_digest(token, server.auth_token):
                    raise PermissionError("invalid service token")
                # #200 PR-2 fix: verify the gate HMAC before honoring
                # _confirmed. The facade's call_gated() computes an
                # HMAC-SHA256 over the request (minus _gate_hmac and
                # token) using the boot-time gate_secret. A raw RPC
                # caller with the endpoint token but without the
                # gate_secret cannot forge this HMAC. Without this
                # verification, _confirmed is just another client-
                # asserted boolean — the same spoof one layer up.
                gate_hmac = request.pop("_gate_hmac", "")
                client_confirmed = request.get("_confirmed", False) is True
                if client_confirmed:
                    # _confirmed is only honored if backed by a valid
                    # HMAC. Compute the expected HMAC over the request
                    # dict (token and _gate_hmac already removed).
                    expected_hmac = hmac.new(
                        server.gate_secret.encode("utf-8"),
                        json.dumps(request, sort_keys=True).encode("utf-8"),
                        hashlib.sha256,
                    ).hexdigest()
                    if not isinstance(gate_hmac, str) or not hmac.compare_digest(
                        gate_hmac, expected_hmac
                    ):
                        # The HMAC doesn't match — _confirmed is forged.
                        # Strip it so the gate denies.
                        request["_confirmed"] = False
                else:
                    # No _confirmed claimed — ensure it's False.
                    request["_confirmed"] = False
                result = server.memory_service.dispatch(request)
                self._write({"ok": True, "result": result})
            except Exception as exc:
                # MS4/#333: never send a traceback or raw str(exc) — both
                # can carry file paths, SQL queries, or secrets. Only
                # _CLIENT_SAFE_EXCEPTIONS surface a (redacted) message;
                # everything else gets a generic envelope. The full
                # detail is logged server-side (exc_info=True).
                logger.warning("Memory service request failed: %s", exc, exc_info=True)
                error, error_class = _classify_error(exc)
                self._write({
                    "ok": False,
                    "error": error,
                    "error_class": error_class,
                })
        finally:
            with server.in_flight_lock:
                server.in_flight -= 1

    def _write(self, value: dict) -> None:
        data = (json.dumps(value, ensure_ascii=False) + "\n").encode("utf-8")
        # MS6: response size limit — prevent unbounded memory from large
        # responses (e.g. list_recent with a high limit). If the serialized
        # response exceeds _MAX_RESPONSE_BYTES, return an error instead.
        if len(data) > _MAX_RESPONSE_BYTES:
            error_data = (
                json.dumps({
                    "ok": False,
                    "error": f"Response exceeds max size {_MAX_RESPONSE_BYTES} bytes",
                    "error_class": "ResponseTooLarge",
                }) + "\n"
            ).encode("utf-8")
            self.wfile.write(error_data)
            self.wfile.flush()
            return
        self.wfile.write(data)
        self.wfile.flush()


def _write_endpoint(path: Path, port: int, token: str, gate_secret: str) -> None:
    try:
        spawner_pid = int(os.environ.get("HERMES_SERVICE_SPAWNER_PID") or 0) or None
    except ValueError:
        spawner_pid = None
    payload = {
        "host": "127.0.0.1",
        "port": port,
        "token": token,
        "gate_secret": gate_secret,
        "pid": os.getpid(),
        "version": 1,
        # 2026-09-10: the process that spawned this service (set by
        # service_client._ensure_service). stop_service() only stops a
        # service this process started; intentional stops pass force=True.
        "spawner_pid": spawner_pid,
    }
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(payload), encoding="utf-8")
    # SC2: restrict the endpoint file to owner-only on POSIX. The file
    # contains the auth token — if world-readable, another user could
    # read it and impersonate the client. On Windows, the default ACL
    # already restricts access to the user's profile, so chmod is a
    # no-op (Windows ignores POSIX permission bits).
    try:
        os.chmod(str(temp), 0o600)
    except OSError:
        pass
    os.replace(temp, path)
    # Also restrict the final file (os.replace may preserve temp perms,
    # but be explicit in case the target already existed with looser perms).
    try:
        os.chmod(str(path), 0o600)
    except OSError:
        pass


def serve(home: Path, port: int = 0) -> None:
    """Run the shared memory service until shut down (RPC shutdown or signal).

    Shutdown reality on Windows (#20): CPython accepts signal.signal(SIGTERM)
    but SIGTERM is NEVER delivered — os.kill() maps to TerminateProcess, a
    hard kill. The graceful path on Windows is the RPC ``shutdown`` method
    (client ``stop_service()``), which calls server.shutdown() and drains
    in-flight handlers. A hard kill leaves the endpoint file stale; the
    single-instance guard probes the port on next boot and steals it, and
    the pid-matched cleanup below removes it on graceful paths. A CTRL_BREAK
    handler is registered when available (console Ctrl+Break still works).
    """
    home.mkdir(parents=True, exist_ok=True)
    endpoint = endpoint_path(home)

    # #143: Single-instance guard BEFORE DB init. On Windows, a venv launcher
    # stub spawns the base interpreter as the real service child, then the
    # launcher process continues executing this script. Without this early
    # guard, the launcher opens the DB, hits the file lock (child already
    # owns it), and crashes with exit 1. By probing for a healthy service
    # before constructing MemoryService (which opens the DB), the launcher
    # exits 0 cleanly if the child is already running.
    try:
        existing = json.loads(endpoint.read_text(encoding="utf-8"))
        if existing.get("host") and existing.get("port") and existing.get("token"):
            probe_sock = socket.create_connection(
                (str(existing["host"]), int(existing["port"])), timeout=2.0
            )
            probe_sock.sendall((json.dumps({"v": _PROTOCOL_VERSION, "token": str(existing["token"]), "method": "health"}) + "\n").encode("utf-8"))
            _resp = b""
            while b"\n" not in _resp:
                chunk = probe_sock.recv(65536)
                if not chunk:
                    break
                _resp += chunk
                # MS8: cap the probe response size to prevent memory
                # exhaustion from a rogue endpoint.
                if len(_resp) > _PROBE_MAX_BYTES:
                    break
            probe_sock.close()
            if _resp and json.loads(_resp.splitlines()[0]).get("ok"):
                logger.info("Shared memory service already running; exiting 0.")
                return
    except Exception:
        pass  # no/stale endpoint -> we are the one true instance

    service = MemoryService(home)
    token = secrets.token_urlsafe(32)
    # #200 PR-2 fix: gate_secret is a separate boot-time secret used by
    # the facade to HMAC-sign gated requests (collection writes,
    # facade_delete_memory). The service verifies the HMAC before
    # honoring _confirmed — a raw RPC caller with the endpoint token
    # but without the gate_secret cannot forge the HMAC. This closes
    # the client-spoofable _confirmed gap.
    gate_secret = secrets.token_urlsafe(32)
    server = _ThreadingTCPServer(("127.0.0.1", port), _RequestHandler)
    server.auth_token = token
    server.gate_secret = gate_secret
    server.memory_service = service
    service.server = server

    _write_endpoint(endpoint, int(server.server_address[1]), token, gate_secret)

    # Opportunistic one-time graph hygiene at startup: quarantine junk/leak
    # entity nodes so noise fades from graph-aware recall. Runs off the hot
    # path so it never delays first RPC. Safe to re-run (MERGE + reversible).
    # Sweeps every tenant's graph (#49).
    def _sweep() -> None:
        for tenant in service._tenants.values():
            try:
                if tenant.graph is not None:
                    tenant.graph.quarantine_junk_entities()
            except Exception:
                pass  # graph hygiene is non-fatal; do not block service boot
    threading.Thread(target=_sweep, daemon=True).start()

    # #329: periodic DuckDB-to-Kuzu drift watch (+ idempotent auto-heal).
    # Interval/auto-heal come from config (0 min disables). First cycle
    # is delayed inside the watch so boot work wins; non-fatal either way.
    try:
        try:
            from drift_watch import start_drift_watch_thread as _start_drift
        except ImportError:
            from .drift_watch import start_drift_watch_thread as _start_drift
        _start_drift(service)
    except Exception:
        pass  # drift watch is non-fatal; never block service boot

    def _stop(_signum, _frame) -> None:
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, _stop)
    if hasattr(signal, "SIGINT"):
        signal.signal(signal.SIGINT, _stop)
    # Windows: SIGTERM is never delivered (os.kill -> TerminateProcess), but
    # console Ctrl+Break IS delivered as SIGBREAK — register it so an
    # interactive Ctrl+Break gets the graceful path too.
    if hasattr(signal, "SIGBREAK"):
        try:
            signal.signal(signal.SIGBREAK, _stop)
        except (ValueError, OSError):
            pass
    logger.info("Shared memory service listening on 127.0.0.1:%s", server.server_address[1])
    try:
        server.serve_forever(poll_interval=0.2)
    finally:
        # Drain in-flight handlers (bounded) so shutdown doesn't kill a
        # mid-backup thread (#20). daemon_threads=True still caps the wait.
        for _ in range(50):  # up to ~5s
            with server.in_flight_lock:
                if server.in_flight == 0:
                    break
            time.sleep(0.1)
        server.server_close()
        service.close()
        try:
            current = json.loads(endpoint.read_text(encoding="utf-8"))
            if current.get("pid") == os.getpid():
                endpoint.unlink(missing_ok=True)
        except Exception:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Hermes shared memory service")
    parser.add_argument("--home", required=True, type=Path)
    parser.add_argument("--port", type=int, default=0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    serve(args.home, args.port)


if __name__ == "__main__":
    main()

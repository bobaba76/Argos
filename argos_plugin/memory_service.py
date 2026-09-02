"""Single-owner local memory service for Hermes hybrid memory.

The service owns the canonical DuckDB and Kùzu files. Hermes desktop, CLI, and
remote gateway plugin instances communicate with it over loopback TCP so no
client opens the database files directly.
"""
from __future__ import annotations

import argparse
import hmac
import json
import logging
import os
import secrets
import signal
import socket
import socketserver
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any

if __package__:
    from .embeddings import LocalEmbedder
    from .graph import KuzuGraphStore
    from .store import DuckDBMemoryStore
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from embeddings import LocalEmbedder
    from graph import KuzuGraphStore
    from store import DuckDBMemoryStore

logger = logging.getLogger("argos.service")
_ENDPOINT_NAME = "hybrid_memory_service.json"
_MAX_REQUEST_BYTES = 4 * 1024 * 1024


def _load_config(home: Path) -> dict:
    path = home / "hybrid_memory.json"
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except Exception as exc:
        # Log + return defaults — a malformed config must not be silently
        # ignored (matches provider_core._load_config's behaviour).
        logger.warning("malformed config %s: %s", path, exc)
        return {}


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
        "review_mode", "max_injected_items", "inject_content_char_cap",
        "external_sources_require_confirmation", "local_only",
    )

    def __init__(self, config: dict) -> None:
        self.review_mode: str = str(
            config.get("review_mode", "confirm")
        ).strip().lower()
        if self.review_mode not in ("confirm", "auto"):
            self.review_mode = "confirm"  # fail closed
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


def _parse_tenants(
    config: dict, home: Path, embedder, reranker,
) -> tuple[dict, dict, bool]:
    """Build the tenant registry from ``hybrid_memory.json`` (#49).

    With a ``tenants`` map, each entry is a cell: its own database + graph
    paths (relative to home) and a nested ``config`` overlay applied on top
    of the global config. Without ``tenants`` (the current single-tenant
    shape), the global config IS the ``default`` tenant — fully backward
    compatible.

    Returns ``(tenants, user_tenant_map, strict_routing)``:

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
    """
    tenants_map = config.get("tenants")
    if not isinstance(tenants_map, dict) or not tenants_map:
        tenants_map = {"default": config}
    tenants: dict = {}
    user_tenant_map: dict = {}
    strict_routing = False
    for name, entry in tenants_map.items():
        entry = entry if isinstance(entry, dict) else {}
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
    return tenants, user_tenant_map, strict_routing


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
        # _parse_tenants also returns the user_id→tenant allowlist and
        # strict-routing flag (#87).
        (
            self._tenants,
            self._user_tenant_map,
            self._strict_routing,
        ) = _parse_tenants(config, home, self.embedder, reranker)
        if self._strict_routing:
            logger.info(
                "Strict tenant routing enabled (#87): %d user_id(s) allowlisted "
                "across %d tenant(s)",
                len(self._user_tenant_map), len(self._tenants),
            )
        elif len(self._tenants) > 1:
            logger.warning(
                "Multi-tenant config without allowed_user_ids (#87): any local "
                "process with the endpoint token can spoof any user_id and access "
                "any tenant. Add 'allowed_user_ids' to each tenant entry to "
                "enable strict routing."
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

    def _call_store(self, method: str, args: dict, user_id: str, store, policy: "TenantPolicy | None" = None) -> Any:
        store.set_user_scope(user_id)
        if method == "search":
            return [
                _record_to_dict(record)
                for record in store.search(
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
                                    )
            ]
        if method == "remember":
            return _record_to_dict(store.remember(**args))
        if method == "update_memory":
            return _record_to_dict(store.update_memory(**args))
        if method == "get_memories_by_ids":
            return [
                _record_to_dict(record)
                for record in store.get_memories_by_ids(
                    args.get("memory_ids", []),
                    include_quarantined=bool(args.get("include_quarantined", False)),
                )
            ]
        if method == "save_candidate":
            return store.save_candidate(**args)
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
            if policy is not None and policy.review_mode == "confirm":
                review_source = str(args.get("review_source", "manual"))
                decision = str(args.get("decision", ""))
                if review_source == "auto_review" and decision in {"approved", "reviewed_approved"}:
                    args["decision"] = "pending_user_confirmation"
                    args["reason"] = (
                        "Tenant policy (review_mode=confirm): auto-review "
                        "cannot activate memories without human confirmation. "
                        + str(args.get("reason", ""))
                    ).strip()
            return store.review_candidate(**args)
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
        # -- system state KV + distillation data access (P4.2) -----------------
        if method == "get_state":
            return store.get_state(args.get("key", ""))
        if method == "set_state":
            store.set_state(args.get("key", ""), args.get("value", ""))
            return True
        if method == "count_eligible_since":
            return store.count_eligible_since(args.get("since"))
        if method == "load_eligible_records":
            return [
                _record_to_dict(record)
                for record in store.load_eligible_records(
                    args.get("since"), int(args.get("limit", 100)),
                )
            ]
        if method == "load_high_signal_records":
            return [
                _record_to_dict(record)
                for record in store.load_high_signal_records(
                    int(args.get("limit", 20)),
                )
            ]
        raise ValueError(f"Unsupported store method: {method}")

    def _call_graph(self, method: str, args: dict, user_id: str, graph) -> Any:
        if graph is None:
            raise RuntimeError("Relationship graph is unavailable")
        graph.set_user_scope(user_id)
        if method == "search_graph":
            return graph.search_graph(
                args.get("term", ""),
                limit=int(args.get("limit", 100)),
            )
        if method == "memory_ids_for_query":
            return graph.memory_ids_for_query(
                args.get("query", ""),
                limit=int(args.get("limit", 100)),
            )
        if method == "query_graph":
            return graph.query_graph(args.get("entity_id", ""))
        if method == "traverse_graph":
            return graph.traverse_graph(
                args.get("entity_id", ""),
                depth=args.get("depth", 2),
                limit=args.get("limit", 100),
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
            return graph.add_relationship(**args)
        if method == "index_memory":
            return graph.index_memory(**args)
        if method == "remove_memory":
            return graph.remove_memory(**args)
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
            # #147: return a config fingerprint + runtime info so clients
            # can detect staleness (service running old config/code vs disk).
            # #127: include per-tenant policy summaries.
            import hashlib
            config = _load_config(self.home)
            config_str = json.dumps(config, sort_keys=True)
            config_hash = hashlib.sha256(config_str.encode("utf-8")).hexdigest()[:16]
            tenant_policies = {
                name: t.policy.to_dict()
                for name, t in self._tenants.items()
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
                "tenant_policies": tenant_policies,
                "lock_wait_total_s": round(self._lock_wait_total_s, 4),
                "lock_wait_count": self._lock_wait_count,
            }
        if request.get("method") == "stats":
            return {
                "lock_wait_total_s": round(self._lock_wait_total_s, 4),
                "lock_wait_count": self._lock_wait_count,
            }
        if request.get("method") == "shutdown":
            if self.server is not None:
                threading.Thread(target=self.server.shutdown, daemon=True).start()
            return {"status": "shutting_down"}
        if request.get("method") == "backup":
            # Service-coordinated backup: the service is the sole DB writer,
            # so CHECKPOINT + EXPORT is safe and cross-platform.  No
            # component/method split — this is a top-level service command.
            return self._backup(request.get("args") or {})

        component = request.get("component")
        method = request.get("method")
        args = request.get("args") or {}
        user_id = str(request.get("user_id") or "default_user")
        if component not in {"store", "graph"} or not isinstance(method, str):
            raise ValueError("Invalid service request")
        # Tenant routing (#49): user_id -> tenant cell. In strict mode
        # (#87), unknown user_ids are rejected instead of falling back to
        # default — prevents tenant spoofing by any local process that
        # holds the endpoint token.
        tenant = self._resolve_tenant(user_id)
        t0 = time.monotonic()
        # Per-tenant locks (#20 + #49): store and graph calls run
        # concurrently, and one tenant's long operation never blocks
        # another tenant. Health/stats are lock-free (handled above).
        lock = tenant.store_lock if component == "store" else tenant.graph_lock
        with lock:
            self._lock_wait_total_s += time.monotonic() - t0
            self._lock_wait_count += 1
            if component == "store":
                return self._call_store(method, args, user_id, tenant.store, tenant.policy)
            return self._call_graph(method, args, user_id, tenant.graph)

    def _backup(self, args: dict) -> Any:
        """Service-coordinated backup via EXPORT DATABASE (FORMAT PARQUET).

        Per-tenant (#49): pass ``tenant`` in args to back up a specific
        cell; the default is the default tenant's store. Whole-file EXPORT
        is cell-scoped by construction.
        """
        if __package__:
            from .backup import backup_store, list_snapshots
        else:
            from backup import backup_store, list_snapshots
        tenant_name = str(args.get("tenant") or self._default_tenant)
        tenant = self._tenants.get(tenant_name) or self._tenants[self._default_tenant]
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
        with tenant.store_lock:
            manifest = backup_store(
                tenant.store.connection,
                dst_root,
                retention_snapshots=retention,
                source_db_path=tenant.store.db_path,
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
                token = str(request.pop("token", ""))
                if not hmac.compare_digest(token, server.auth_token):
                    raise PermissionError("invalid service token")
                result = server.memory_service.dispatch(request)
                self._write({"ok": True, "result": result})
            except Exception as exc:
                # Error envelope (#20): carry the error class + a short
                # traceback so clients can distinguish failure kinds; log
                # the full traceback server-side (diagnostics are silent
                # today — the client only ever sees str(exc)).
                logger.warning("Memory service request failed: %s", exc, exc_info=True)
                tb = traceback.format_exc(limit=6)
                self._write({
                    "ok": False,
                    "error": str(exc),
                    "error_class": type(exc).__name__,
                    "traceback": tb[-1200:],
                })
        finally:
            with server.in_flight_lock:
                server.in_flight -= 1

    def _write(self, value: dict) -> None:
        data = (json.dumps(value, ensure_ascii=False) + "\n").encode("utf-8")
        self.wfile.write(data)
        self.wfile.flush()


def _write_endpoint(path: Path, port: int, token: str) -> None:
    payload = {
        "host": "127.0.0.1",
        "port": port,
        "token": token,
        "pid": os.getpid(),
        "version": 1,
    }
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(temp, path)


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
            probe_sock.sendall((json.dumps({"token": str(existing["token"]), "method": "health"}) + "\n").encode("utf-8"))
            _resp = b""
            while b"\n" not in _resp:
                chunk = probe_sock.recv(65536)
                if not chunk:
                    break
                _resp += chunk
            probe_sock.close()
            if _resp and json.loads(_resp.splitlines()[0]).get("ok"):
                logger.info("Shared memory service already running; exiting 0.")
                return
    except Exception:
        pass  # no/stale endpoint -> we are the one true instance

    service = MemoryService(home)
    token = secrets.token_urlsafe(32)
    server = _ThreadingTCPServer(("127.0.0.1", port), _RequestHandler)
    server.auth_token = token
    server.memory_service = service
    service.server = server

    _write_endpoint(endpoint, int(server.server_address[1]), token)

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

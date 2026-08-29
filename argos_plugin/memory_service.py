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
    except Exception:
        return {}


def endpoint_path(home: str | Path) -> Path:
    return Path(home) / _ENDPOINT_NAME


def _record_to_dict(record: Any) -> dict | None:
    if record is None:
        return None
    return record.to_dict() if hasattr(record, "to_dict") else record


class MemoryService:
    def __init__(self, home: Path) -> None:
        self.home = home
        config = _load_config(home)
        db_name = str(config.get("database_filename", "hybrid_memory.duckdb"))
        graph_name = str(config.get("graph_dirname", "hybrid_memory_kuzu"))
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
        self.store = DuckDBMemoryStore(
            home / db_name, user_id="default_user", embedder=self.embedder,
            reranker=reranker,
        )
        try:
            self.store._reranker_top_n = max(
                5, min(int(config.get("reranker_top_n", 20)), 100)
            )
        except (TypeError, ValueError):
            self.store._reranker_top_n = 20
        # Exact-phrase lift (parity with provider config)
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
        # External-source write policy (parity with the provider config).
        self.store.external_sources_require_confirmation = str(
            config.get("external_sources_require_confirmation", "true")
        ).lower() in ("true", "1", "yes")
        try:
            self.graph = KuzuGraphStore(home / graph_name, user_id="default_user")
        except Exception as exc:
            logger.warning("Kùzu unavailable in shared memory service: %s", exc)
            self.graph = None
        self.store_lock = threading.RLock()  # serializes store writes (#20)
        self.graph_lock = threading.RLock()  # separate graph lock: store and
        # graph calls no longer serialize behind ONE global lock, so a long
        # graph traversal no longer queues every store search behind it.
        self._lock_wait_total_s = 0.0
        self._lock_wait_count = 0
        self.server = None

    def _call_store(self, method: str, args: dict, user_id: str) -> Any:
        self.store.set_user_scope(user_id)
        if method == "search":
            return [
                _record_to_dict(record)
                for record in self.store.search(
                    args.get("query", ""),
                    limit=int(args.get("limit", 5)),
                    exclude_categories=args.get("exclude_categories"),
                    category_filter=args.get("category_filter"),
                    project_id=args.get("project_id"),
                    as_of=args.get("as_of"),
                    suppress_retrieval=bool(args.get("suppress_retrieval", False)),
                                        include_expired=bool(args.get("include_expired", False)),
                                        include_closed=bool(args.get("include_closed", False)),
                                    )
            ]
        if method == "remember":
            return _record_to_dict(self.store.remember(**args))
        if method == "update_memory":
            return _record_to_dict(self.store.update_memory(**args))
        if method == "get_memories_by_ids":
            return [
                _record_to_dict(record)
                for record in self.store.get_memories_by_ids(
                    args.get("memory_ids", []),
                    include_quarantined=bool(args.get("include_quarantined", False)),
                )
            ]
        if method == "save_candidate":
            return self.store.save_candidate(**args)
        if method == "find_semantic_duplicate":
            return _record_to_dict(
                self.store.find_semantic_duplicate(
                    content=args.get("content", ""),
                    min_similarity=float(args.get("min_similarity", 0.88)),
                )
            )
        if method == "list_candidates":
            return self.store.list_candidates(**args)
        if method == "review_candidate":
            return self.store.review_candidate(**args)
        if method == "find_supersede_candidates":
            return self.store.find_supersede_candidates(
                candidate_id=args.get("candidate_id", ""),
                limit=int(args.get("limit", 3)),
            )
        if method == "quarantine_memory":
            return self.store.quarantine_memory(**args)
        if method == "restore_memory":
            return self.store.restore_memory(**args)
        if method == "record_feedback":
            return self.store.record_feedback(**args)
        if method == "mark_superseded":
            # Live-admin supersession (e.g. retroactive chains after a new
            # benchmark supersedes an old one).  Sets valid_to; the read side
            # excludes the record automatically.
            return self.store._mark_superseded(
                memory_id=args.get("memory_id", ""),
                reason=args.get("reason", ""),
                superseded_by=args.get("superseded_by"),
            )
        if method == "delete_memory":
            return self.store.delete_memory(**args)
        # -- deletion tombstones (read-only visibility + escape hatch) ---------
        if method == "list_tombstones":
            return self.store.list_tombstones(
                limit=int(args.get("limit", 200)),
            )
        if method == "purge_tombstone":
            return self.store.purge_tombstone(
                content=args.get("content", ""),
                category=args.get("category", ""),
            )
        if method == "cleanup_junk":
            return self.store.cleanup_junk(**args)
        if method == "consolidate":
            return self.store.consolidate(**args)
        if method == "count":
            return self.store.count()
        if method == "record_retrieval":
            self.store.record_retrieval(args.get("memory_ids", []))
            return True
        if method == "get_evidence":
            return self.store.get_evidence(args.get("memory_id", ""))
        if method == "get_evidence_batch":
            return self.store.get_evidence_batch(args.get("memory_ids", []) or [])
        if method == "get_memory_history":
            return [
                _record_to_dict(record)
                for record in self.store.get_memory_history(
                    args.get("memory_id", ""),
                    max_versions=args.get("max_versions"),
                )
            ]
        if method == "get_chain_membership":
            return self.store.get_chain_membership(args.get("memory_ids", []) or [])
        if method == "backfill_evidence":
            return self.store.backfill_evidence(
                retention=args.get("retention", "full")
            )
        if method == "get_scale_metrics":
            return self.store.get_scale_metrics()
        if method == "set_scale_thresholds":
            self.store.set_scale_thresholds(
                args.get("warn_latency_ms", 300.0),
                args.get("warn_records", 5000),
            )
            return True
        if method == "list_recent":
            return [
                _record_to_dict(record)
                for record in self.store.list_recent(
                    limit=int(args.get("limit", 100)),
                )
            ]
        if method == "get_insights":
            return [
                _record_to_dict(record)
                for record in self.store.get_insights(
                    tags=args.get("tags"),
                    since=args.get("since"),
                    limit=int(args.get("limit", 50)),
                )
            ]
        if method == "add_alias":
            self.store.add_alias(
                alias=args.get("alias", ""),
                canonical_entity=args.get("canonical_entity", ""),
            )
            return True
        if method == "remove_alias":
            return self.store.remove_alias(
                alias=args.get("alias", ""),
                canonical_entity=args.get("canonical_entity"),
            )
        if method == "resolve_aliases":
            return self.store.resolve_aliases(args.get("text", ""))
        if method == "list_aliases":
            return self.store.list_aliases()
        if method == "aliases_for_canonical":
            return self.store.aliases_for_canonical(args.get("canonical_entity", ""))
        # -- system state KV + distillation data access (P4.2) -----------------
        if method == "get_state":
            return self.store.get_state(args.get("key", ""))
        if method == "set_state":
            self.store.set_state(args.get("key", ""), args.get("value", ""))
            return True
        if method == "count_eligible_since":
            return self.store.count_eligible_since(args.get("since"))
        if method == "load_eligible_records":
            return [
                _record_to_dict(record)
                for record in self.store.load_eligible_records(
                    args.get("since"), int(args.get("limit", 100)),
                )
            ]
        if method == "load_high_signal_records":
            return [
                _record_to_dict(record)
                for record in self.store.load_high_signal_records(
                    int(args.get("limit", 20)),
                )
            ]
        raise ValueError(f"Unsupported store method: {method}")

    def _call_graph(self, method: str, args: dict, user_id: str) -> Any:
        if self.graph is None:
            raise RuntimeError("Relationship graph is unavailable")
        self.graph.set_user_scope(user_id)
        if method == "search_graph":
            return self.graph.search_graph(
                args.get("term", ""),
                limit=int(args.get("limit", 100)),
            )
        if method == "memory_ids_for_query":
            return self.graph.memory_ids_for_query(
                args.get("query", ""),
                limit=int(args.get("limit", 100)),
            )
        if method == "query_graph":
            return self.graph.query_graph(args.get("entity_id", ""))
        if method == "traverse_graph":
            return self.graph.traverse_graph(
                args.get("entity_id", ""),
                depth=args.get("depth", 2),
                limit=args.get("limit", 100),
            )
        if method == "count_nodes":
            return self.graph.count_nodes()
        if method == "count_edges":
            return self.graph.count_edges()
        if method == "list_nodes":
            return self.graph.list_nodes(
                node_type=args.get("node_type"),
                limit=int(args.get("limit", 100)),
            )
        if method == "add_relationship":
            return self.graph.add_relationship(**args)
        if method == "index_memory":
            return self.graph.index_memory(**args)
        if method == "remove_memory":
            return self.graph.remove_memory(**args)
        if method == "quarantine_junk_entities":
            return self.graph.quarantine_junk_entities()
        if method == "clear_scope":
            return list(self.graph.clear_scope())
        raise ValueError(f"Unsupported graph method: {method}")

    def dispatch(self, request: dict) -> Any:
        if request.get("method") == "health":
            return {
                "status": "ok",
                "pid": os.getpid(),
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
        t0 = time.monotonic()
        # Per-store locks (#20): store and graph calls run concurrently;
        # health/stats are lock-free (handled above) so a long backup no
        # longer blocks the health check.
        lock = self.store_lock if component == "store" else self.graph_lock
        with lock:
            self._lock_wait_total_s += time.monotonic() - t0
            self._lock_wait_count += 1
            if component == "store":
                return self._call_store(method, args, user_id)
            return self._call_graph(method, args, user_id)

    def _backup(self, args: dict) -> Any:
        """Service-coordinated backup via EXPORT DATABASE (FORMAT PARQUET)."""
        if __package__:
            from .backup import backup_store, list_snapshots
        else:
            from backup import backup_store, list_snapshots
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
        with self.store_lock:
            manifest = backup_store(
                self.store.connection,
                dst_root,
                retention_snapshots=retention,
                source_db_path=self.store.db_path,
            )
        return manifest

    def close(self) -> None:
        with self.store_lock:
            try:
                self.store.close()
            finally:
                if self.graph:
                    with self.graph_lock:
                        self.graph.close()


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
    service = MemoryService(home)
    token = secrets.token_urlsafe(32)
    server = _ThreadingTCPServer(("127.0.0.1", port), _RequestHandler)
    server.auth_token = token
    server.memory_service = service
    service.server = server
    endpoint = endpoint_path(home)

    # Single-instance guard: if another live service already owns the store,
    # exit quietly instead of fighting it for the DuckDB file lock (which
    # would wedge every client). Covers stolen-start-lock races.
    try:
        existing = json.loads(endpoint.read_text(encoding="utf-8"))
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
            logger.warning("Shared memory service already running; exiting.")
            server.server_close()
            service.close()
            return
    except Exception:
        pass  # no/stale endpoint -> we are the one true instance

    _write_endpoint(endpoint, int(server.server_address[1]), token)

    # Opportunistic one-time graph hygiene at startup: quarantine junk/leak
    # entity nodes so noise fades from graph-aware recall. Runs off the hot
    # path so it never delays first RPC. Safe to re-run (MERGE + reversible).
    if service.graph is not None:
        def _sweep() -> None:
            try:
                service.graph.quarantine_junk_entities()
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

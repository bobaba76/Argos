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
import socketserver
import sys
import threading
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

logger = logging.getLogger("hybrid_memory.service")
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
        self.store = DuckDBMemoryStore(
            home / db_name, user_id="default_user", embedder=self.embedder
        )
        try:
            self.graph = KuzuGraphStore(home / graph_name, user_id="default_user")
        except Exception as exc:
            logger.warning("Kùzu unavailable in shared memory service: %s", exc)
            self.graph = None
        self.lock = threading.RLock()
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
        if method == "list_candidates":
            return self.store.list_candidates(**args)
        if method == "review_candidate":
            return self.store.review_candidate(**args)
        if method == "quarantine_memory":
            return self.store.quarantine_memory(**args)
        if method == "restore_memory":
            return self.store.restore_memory(**args)
        if method == "record_feedback":
            return self.store.record_feedback(**args)
        if method == "delete_memory":
            return self.store.delete_memory(**args)
        if method == "cleanup_junk":
            return self.store.cleanup_junk()
        if method == "consolidate":
            return self.store.consolidate(**args)
        if method == "count":
            return self.store.count()
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
        raise ValueError(f"Unsupported graph method: {method}")

    def dispatch(self, request: dict) -> Any:
        if request.get("method") == "health":
            return {"status": "ok", "pid": os.getpid()}
        if request.get("method") == "shutdown":
            if self.server is not None:
                threading.Thread(target=self.server.shutdown, daemon=True).start()
            return {"status": "shutting_down"}

        component = request.get("component")
        method = request.get("method")
        args = request.get("args") or {}
        user_id = str(request.get("user_id") or "default_user")
        if component not in {"store", "graph"} or not isinstance(method, str):
            raise ValueError("Invalid service request")
        with self.lock:
            if component == "store":
                return self._call_store(method, args, user_id)
            return self._call_graph(method, args, user_id)

    def close(self) -> None:
        with self.lock:
            try:
                self.store.close()
            finally:
                if self.graph:
                    self.graph.close()


class _ThreadingTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True


class _RequestHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        raw = self.rfile.readline(_MAX_REQUEST_BYTES + 1)
        if len(raw) > _MAX_REQUEST_BYTES:
            self._write({"ok": False, "error": "request too large"})
            return
        try:
            request = json.loads(raw.decode("utf-8"))
            if not isinstance(request, dict):
                raise ValueError("request must be an object")
            token = str(request.pop("token", ""))
            if not hmac.compare_digest(token, self.server.auth_token):
                raise PermissionError("invalid service token")
            result = self.server.memory_service.dispatch(request)
            self._write({"ok": True, "result": result})
        except Exception as exc:
            logger.debug("Memory service request failed: %s", exc)
            self._write({"ok": False, "error": str(exc)})

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
    home.mkdir(parents=True, exist_ok=True)
    service = MemoryService(home)
    token = secrets.token_urlsafe(32)
    server = _ThreadingTCPServer(("127.0.0.1", port), _RequestHandler)
    server.auth_token = token
    server.memory_service = service
    service.server = server
    endpoint = endpoint_path(home)
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
    logger.info("Shared memory service listening on 127.0.0.1:%s", server.server_address[1])
    try:
        server.serve_forever(poll_interval=0.2)
    finally:
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

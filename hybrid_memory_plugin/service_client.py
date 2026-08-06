"""Client-side adapters for the local shared hybrid-memory service."""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

if __package__:
    from .memory_service import endpoint_path
    from .store import MemoryRecord
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from memory_service import endpoint_path
    from store import MemoryRecord

_START_LOCK_NAME = "hybrid_memory_service.starting"
_DEFAULT_TIMEOUT = 30.0
_START_TIMEOUT = 30.0


class SharedMemoryServiceError(RuntimeError):
    pass


def _read_endpoint(home: Path) -> dict | None:
    path = endpoint_path(home)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, dict) and value.get("host") and value.get("port") and value.get("token"):
            return value
    except (OSError, ValueError, TypeError):
        pass
    return None


def _record_from_dict(value: dict | None) -> MemoryRecord | None:
    if not value:
        return None
    return MemoryRecord(
        memory_id=value.get("memory_id", ""),
        category=value.get("category", "context_note"),
        content=value.get("content", ""),
        tags=value.get("tags", []),
        payload=value.get("payload", {}),
        created_at=value.get("created_at"),
        updated_at=value.get("updated_at"),
        expires_at=value.get("expires_at"),
        similarity=float(value.get("similarity", 0.0) or 0.0),
        status=value.get("status", "active"),
        source=value.get("source", "explicit"),
        confidence=value.get("confidence"),
        durability=value.get("durability", "durable"),
        scope=value.get("scope", "profile"),
        project_id=value.get("project_id"),
        retrieval_count=value.get("retrieval_count", 0),
        last_retrieved_at=value.get("last_retrieved_at"),
        helpful_count=value.get("helpful_count", 0),
        dismissed_count=value.get("dismissed_count", 0),
        quarantine_reason=value.get("quarantine_reason"),
        quarantined_at=value.get("quarantined_at"),
    )


class _SharedRPC:
    def __init__(self, home: str | Path, user_id: str = "default_user") -> None:
        self.home = Path(home)
        self.user_id = user_id or "default_user"
        self._ensure_service()

    def _request(self, request: dict, timeout: float = _DEFAULT_TIMEOUT) -> Any:
        endpoint = _read_endpoint(self.home)
        if endpoint is None:
            raise SharedMemoryServiceError("shared memory service endpoint is unavailable")
        request = dict(request)
        request["token"] = endpoint["token"]
        request.setdefault("user_id", self.user_id)
        try:
            with socket.create_connection(
                (str(endpoint["host"]), int(endpoint["port"])), timeout=timeout
            ) as connection:
                connection.settimeout(timeout)
                connection.sendall((json.dumps(request, ensure_ascii=False) + "\n").encode("utf-8"))
                chunks = []
                while True:
                    chunk = connection.recv(65536)
                    if not chunk:
                        break
                    chunks.append(chunk)
                    if b"\n" in chunk:
                        break
            response = json.loads(b"".join(chunks).splitlines()[0].decode("utf-8"))
        except (OSError, ValueError, IndexError) as exc:
            raise SharedMemoryServiceError(f"shared memory service request failed: {exc}") from exc
        if not response.get("ok"):
            raise SharedMemoryServiceError(str(response.get("error", "service error")))
        return response.get("result")

    def _healthy(self) -> bool:
        try:
            result = self._request({"method": "health"}, timeout=1.0)
            return isinstance(result, dict) and result.get("status") == "ok"
        except SharedMemoryServiceError:
            return False

    def _ensure_service(self) -> None:
        if self._healthy():
            return
        self.home.mkdir(parents=True, exist_ok=True)
        lock_path = self.home / _START_LOCK_NAME
        owner = False
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            owner = True
        except FileExistsError:
            pass
        if owner:
            try:
                script = Path(__file__).with_name("memory_service.py")
                creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
                subprocess.Popen(
                    [sys.executable, str(script), "--home", str(self.home)],
                    cwd=str(script.parent),
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=creationflags,
                )
            except Exception:
                try:
                    lock_path.unlink(missing_ok=True)
                except OSError:
                    pass
                raise
        deadline = time.monotonic() + _START_TIMEOUT
        try:
            while time.monotonic() < deadline:
                if self._healthy():
                    return
                time.sleep(0.2)
            raise SharedMemoryServiceError("shared memory service did not become healthy")
        finally:
            if owner:
                try:
                    lock_path.unlink(missing_ok=True)
                except OSError:
                    pass

    def call(self, component: str, method: str, **args: Any) -> Any:
        return self._request({"component": component, "method": method, "args": args})

    def stop_service(self) -> Any:
        return self._request({"method": "shutdown"})


class SharedMemoryStore:
    """DuckDBMemoryStore-compatible client backed by the shared service."""

    def __init__(self, home: str | Path, user_id: str = "default_user", embedder=None) -> None:
        self.home = Path(home)
        self.db_path = self.home / "hybrid_memory.duckdb"
        self.user_id = user_id or "default_user"
        self._rpc = _SharedRPC(self.home, self.user_id)

    def set_user_scope(self, user_id: str | None) -> None:
        self.user_id = user_id or "default_user"
        self._rpc.user_id = self.user_id

    def search(
        self,
        query: str,
        limit: int = 5,
        exclude_categories: List[str] | None = None,
        category_filter: str | None = None,
    ) -> List[MemoryRecord]:
        values = self._rpc.call(
            "store", "search", query=query, limit=limit,
            exclude_categories=exclude_categories,
            category_filter=category_filter,
        )
        return [_record_from_dict(value) for value in (values or [])]

    def remember(self, **kwargs: Any) -> MemoryRecord | None:
        return _record_from_dict(self._rpc.call("store", "remember", **kwargs))

    def save_candidate(self, **kwargs: Any) -> dict | None:
        return self._rpc.call("store", "save_candidate", **kwargs)

    def list_candidates(self, **kwargs: Any) -> List[dict]:
        return self._rpc.call("store", "list_candidates", **kwargs) or []

    def review_candidate(self, **kwargs: Any) -> dict | None:
        value = self._rpc.call("store", "review_candidate", **kwargs)
        if value and value.get("memory"):
            value["memory"] = _record_from_dict(value["memory"]).to_dict()
        return value

    def quarantine_memory(self, **kwargs: Any) -> bool:
        return bool(self._rpc.call("store", "quarantine_memory", **kwargs))

    def restore_memory(self, **kwargs: Any) -> bool:
        return bool(self._rpc.call("store", "restore_memory", **kwargs))

    def record_feedback(self, **kwargs: Any) -> bool:
        return bool(self._rpc.call("store", "record_feedback", **kwargs))

    def delete_memory(self, **kwargs: Any) -> bool:
        return bool(self._rpc.call("store", "delete_memory", **kwargs))

    def cleanup_junk(self) -> int:
        return int(self._rpc.call("store", "cleanup_junk") or 0)

    def count(self) -> int:
        return int(self._rpc.call("store", "count") or 0)

    def close(self) -> None:
        return None


class SharedGraphStore:
    """Kùzu graph client backed by the shared service."""

    def __init__(self, home: str | Path, user_id: str = "default_user") -> None:
        self.home = Path(home)
        self.user_id = user_id or "default_user"
        self._rpc = _SharedRPC(self.home, self.user_id)

    def set_user_scope(self, user_id: str | None) -> None:
        self.user_id = user_id or "default_user"
        self._rpc.user_id = self.user_id

    def search_graph(self, term: str) -> List[dict]:
        return self._rpc.call("graph", "search_graph", term=term) or []

    def query_graph(self, entity_id: str) -> List[dict]:
        return self._rpc.call("graph", "query_graph", entity_id=entity_id) or []

    def add_relationship(self, **kwargs: Any) -> None:
        self._rpc.call("graph", "add_relationship", **kwargs)

    def quarantine_junk_entities(self) -> int:
        return int(self._rpc.call("graph", "quarantine_junk_entities") or 0)

    def close(self) -> None:
        return None

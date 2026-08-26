"""Client-side adapters for the local shared argos service."""
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
_START_LOCK_STALE_SECS = 90.0  # > _START_TIMEOUT; a healthy spawner unlinks well before this
_DEFAULT_TIMEOUT = 30.0
_START_TIMEOUT = 30.0


def _pid_alive(pid: int) -> bool:
    """Best-effort PID liveness check. Windows-safe (no os.kill signal 0)."""
    if pid <= 0:
        return False
    try:
        if sys.platform == "win32":
            import ctypes

            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            STILL_ACTIVE = 259
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.OpenProcess(
                PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid)
            )
            if not handle:
                return False
            try:
                exit_code = ctypes.c_ulong()
                if kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                    return exit_code.value == STILL_ACTIVE
                return True  # query failed -> assume alive
            finally:
                kernel32.CloseHandle(handle)
        import errno

        os.kill(pid, 0)
        return True
    except OSError as exc:
        return exc.errno != errno.ESRCH
    except Exception:
        return True  # cannot tell -> assume alive (conservative)


def _start_lock_is_stale(lock_path: Path) -> bool:
    """True when the lock holder is provably gone (dead PID) or ancient."""
    try:
        info = json.loads(lock_path.read_text(encoding="utf-8").strip() or "{}")
        pid = int(info.get("pid", 0))
        ts = float(info.get("ts", 0.0))
    except (OSError, ValueError, TypeError):
        try:
            ts = lock_path.stat().st_mtime  # legacy/garbage lock: fall back to age
        except OSError:
            return False
        pid = 0
    if pid and _pid_alive(pid):
        return False
    return (time.time() - ts) > _START_LOCK_STALE_SECS


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
        raw_similarity=float(value.get("raw_similarity", 0.0) or 0.0),
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
        valid_from=value.get("valid_from"),
        valid_to=value.get("valid_to"),
        superseded_by=value.get("superseded_by"),
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
            payload = json.dumps({"pid": os.getpid(), "ts": time.time()}).encode("utf-8")
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            try:
                os.write(fd, payload)
            finally:
                os.close(fd)
            owner = True
        except FileExistsError:
            # A crashed spawner can leave the lock behind forever. Steal it when
            # its holder is provably dead (or it predates any healthy start).
            if _start_lock_is_stale(lock_path):
                try:
                    lock_path.unlink(missing_ok=True)
                except OSError:
                    pass
                try:
                    payload = json.dumps({"pid": os.getpid(), "ts": time.time()}).encode("utf-8")
                    fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                    try:
                        os.write(fd, payload)
                    finally:
                        os.close(fd)
                    owner = True
                except FileExistsError:
                    pass  # another stealer won; just wait for health below
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
            project_id: str | None = None,
            as_of: str | None = None,
            suppress_retrieval: bool = False,
            include_expired: bool = False,
            include_closed: bool = False,
        ) -> List[MemoryRecord]:
            values = self._rpc.call(
                "store", "search", query=query, limit=limit,
                exclude_categories=exclude_categories,
                category_filter=category_filter,
                project_id=project_id,
                as_of=as_of,
                suppress_retrieval=suppress_retrieval,
                include_expired=include_expired,
                include_closed=include_closed,
            )
            return [_record_from_dict(value) for value in (values or [])]

    def get_memories_by_ids(
        self,
        memory_ids: List[str],
        *,
        include_quarantined: bool = False,
    ) -> List[MemoryRecord]:
        values = self._rpc.call(
            "store", "get_memories_by_ids",
            memory_ids=memory_ids,
            include_quarantined=include_quarantined,
        ) or []
        return [_record_from_dict(value) for value in values]

    def remember(self, **kwargs: Any) -> MemoryRecord | None:
        return _record_from_dict(self._rpc.call("store", "remember", **kwargs))

    def record_retrieval(self, memory_ids: List[str]) -> None:
        """Explicitly credit retrieval for the final injected memory list."""
        self._rpc.call("store", "record_retrieval", memory_ids=list(memory_ids or []))

    def get_evidence(self, memory_id: str) -> dict | None:
        """Return the provenance record for a memory, or None."""
        return self._rpc.call("store", "get_evidence", memory_id=memory_id)

    def get_evidence_batch(self, memory_ids: List[str]) -> dict:
        """Return provenance rows for a batch of memory IDs (one round trip)."""
        return self._rpc.call(
            "store", "get_evidence_batch", memory_ids=list(memory_ids or [])
        ) or {}

    def get_memory_history(
        self, memory_id: str, *, max_versions: int | None = None,
    ) -> List[MemoryRecord]:
        """Return the full version chain for a memory (oldest first).

        Keyword-only signature matching the facade convention. *max_versions*
        truncates to the most recent N versions (head always retained).
        """
        values = self._rpc.call(
            "store", "get_memory_history",
            memory_id=memory_id, max_versions=max_versions,
        ) or []
        return [_record_from_dict(value) for value in values]

    def get_chain_membership(self, memory_ids: List[str]) -> dict:
        """Batched chain-membership annotation for search-result IDs."""
        return self._rpc.call(
            "store", "get_chain_membership", memory_ids=list(memory_ids or [])
        ) or {}

    def backfill_evidence(self, retention: str = "full") -> int:
        """Backfill memory_evidence from approved candidates (pre-Wave-2 memories)."""
        return int(
            self._rpc.call("store", "backfill_evidence", retention=retention) or 0
        )

    def get_scale_metrics(self) -> dict:
        """Return scale-trigger metrics (query latency, record count)."""
        return self._rpc.call("store", "get_scale_metrics") or {}

    def set_scale_thresholds(self, warn_latency_ms: float, warn_records: int) -> bool:
        """Configure scale-trigger thresholds on the shared store."""
        return bool(
            self._rpc.call(
                "store", "set_scale_thresholds",
                warn_latency_ms=warn_latency_ms, warn_records=warn_records,
            )
        )

    def set_retriever(self, retriever: Any) -> None:
        """Retrieval engines are only swappable on the direct store."""
        raise NotImplementedError(
            "set_retriever is only supported on DuckDBMemoryStore"
        )

    def update_memory(self, **kwargs: Any) -> MemoryRecord | None:
        return _record_from_dict(self._rpc.call("store", "update_memory", **kwargs))

    def save_candidate(self, **kwargs: Any) -> dict | None:
        return self._rpc.call("store", "save_candidate", **kwargs)

    def find_semantic_duplicate(
        self, content: str, min_similarity: float = 0.88,
    ) -> MemoryRecord | None:
        """Return the closest active memory if it semantically covers *content*."""
        return _record_from_dict(
            self._rpc.call(
                "store", "find_semantic_duplicate",
                content=content, min_similarity=min_similarity,
            )
        )

    # -- system state KV + distillation data access (P4.2) --------------------
    # These forward to the service so distillation can run against the proxy
    # without reaching into _lock / connection / _fetch_records.

    def get_state(self, key: str) -> str | None:
        return self._rpc.call("store", "get_state", key=key)

    def set_state(self, key: str, value: str) -> None:
        self._rpc.call("store", "set_state", key=key, value=value)

    def count_eligible_since(self, since: str | None) -> int:
        return int(self._rpc.call("store", "count_eligible_since", since=since) or 0)

    def load_eligible_records(
        self, since: str | None, limit: int,
    ) -> list:
        values = self._rpc.call(
            "store", "load_eligible_records", since=since, limit=limit,
        ) or []
        return [_record_from_dict(value) for value in values]

    def load_high_signal_records(self, limit: int = 20) -> list:
        values = self._rpc.call(
            "store", "load_high_signal_records", limit=limit,
        ) or []
        return [_record_from_dict(value) for value in values]

    def list_candidates(self, **kwargs: Any) -> List[dict]:
        return self._rpc.call("store", "list_candidates", **kwargs) or []

    def review_candidate(self, **kwargs: Any) -> dict | None:
        value = self._rpc.call("store", "review_candidate", **kwargs)
        if value and value.get("memory"):
            value["memory"] = _record_from_dict(value["memory"]).to_dict()
        return value

    def find_supersede_candidates(
        self, candidate_id: str, limit: int = 3,
    ) -> List[dict]:
        """Surface current memories similar to a candidate (supersede options)."""
        return self._rpc.call(
            "store", "find_supersede_candidates",
            candidate_id=candidate_id, limit=limit,
        ) or []

    def quarantine_memory(self, **kwargs: Any) -> bool:
        return bool(self._rpc.call("store", "quarantine_memory", **kwargs))

    def restore_memory(self, **kwargs: Any) -> bool:
        return bool(self._rpc.call("store", "restore_memory", **kwargs))

    def record_feedback(self, **kwargs: Any) -> bool:
        return bool(self._rpc.call("store", "record_feedback", **kwargs))

    def delete_memory(self, **kwargs: Any) -> bool | dict:
        """Chain-aware delete. Returns False if not found, else a result dict
        with an 'action' key (deleted | quarantined | promoted)."""
        value = self._rpc.call("store", "delete_memory", **kwargs)
        if value is False or value is None:
            return False
        return value

    def list_tombstones(self, limit: int = 200) -> List[Dict[str, Any]]:
        """Read-only census of deletion tombstones (hash+metadata, newest first)."""
        return list(self._rpc.call("store", "list_tombstones", limit=limit) or [])

    def purge_tombstone(self, content: str, category: str) -> bool:
        """Escape hatch: lift a deletion tombstone so the fact may be re-fed."""
        return bool(
            self._rpc.call(
                "store",
                "purge_tombstone",
                content=content,
                category=category,
            )
        )

    def cleanup_junk(self, return_ids: bool = False) -> int | dict:
        value = self._rpc.call("store", "cleanup_junk", return_ids=return_ids) or 0
        return value if return_ids else int(value)

    def consolidate(self, **kwargs: Any) -> dict:
        return self._rpc.call("store", "consolidate", **kwargs) or {}

    def count(self) -> int:
        return int(self._rpc.call("store", "count") or 0)

    def list_recent(self, limit: int = 100) -> List[MemoryRecord]:
        return [
            _record_from_dict(value)
            for value in (self._rpc.call("store", "list_recent", limit=limit) or [])
        ]

    def get_insights(
        self,
        tags: List[str] | None = None,
        since: str | None = None,
        limit: int = 50,
    ) -> List[MemoryRecord]:
        return [
            _record_from_dict(value)
            for value in (self._rpc.call(
                "store", "get_insights", tags=tags, since=since, limit=limit,
            ) or [])
        ]

    def close(self) -> None:
        return None

    # -- entity aliases -------------------------------------------------------

    def add_alias(self, alias: str, canonical_entity: str) -> None:
        self._rpc.call(
            "store", "add_alias",
            alias=alias, canonical_entity=canonical_entity,
        )

    def remove_alias(self, alias: str, canonical_entity: str | None = None) -> bool:
        return bool(self._rpc.call(
            "store", "remove_alias",
            alias=alias, canonical_entity=canonical_entity,
        ) or False)

    def resolve_aliases(self, text: str) -> List[str]:
        return self._rpc.call("store", "resolve_aliases", text=text) or []

    def list_aliases(self) -> List[Dict[str, str]]:
        return self._rpc.call("store", "list_aliases") or []

    def aliases_for_canonical(self, canonical_entity: str) -> List[str]:
        return self._rpc.call(
            "store", "aliases_for_canonical",
            canonical_entity=canonical_entity,
        ) or []


class SharedGraphStore:
    """Kùzu graph client backed by the shared service."""

    def __init__(self, home: str | Path, user_id: str = "default_user") -> None:
        self.home = Path(home)
        self.user_id = user_id or "default_user"
        self._rpc = _SharedRPC(self.home, self.user_id)

    def set_user_scope(self, user_id: str | None) -> None:
        self.user_id = user_id or "default_user"
        self._rpc.user_id = self.user_id

    def search_graph(self, term: str, limit: int = 100) -> List[dict]:
        return self._rpc.call("graph", "search_graph", term=term, limit=limit) or []

    def memory_ids_for_query(self, query: str, limit: int = 100) -> List[str]:
        return self._rpc.call(
            "graph", "memory_ids_for_query", query=query, limit=limit,
        ) or []

    def query_graph(self, entity_id: str) -> List[dict]:
        return self._rpc.call("graph", "query_graph", entity_id=entity_id) or []

    def traverse_graph(
        self,
        entity_id: str,
        depth: int = 2,
        limit: int = 100,
    ) -> Dict[str, Any]:
        return self._rpc.call(
            "graph", "traverse_graph",
            entity_id=entity_id, depth=depth, limit=limit,
        ) or {"entity_id": entity_id, "depth": depth, "nodes": [], "edges": []}

    def count_nodes(self) -> int:
        return int(self._rpc.call("graph", "count_nodes") or 0)

    def count_edges(self) -> int:
        return int(self._rpc.call("graph", "count_edges") or 0)

    def list_nodes(self, node_type: str | None = None, limit: int = 100) -> List[dict]:
        return self._rpc.call(
            "graph", "list_nodes", node_type=node_type, limit=limit,
        ) or []

    def add_relationship(self, **kwargs: Any) -> None:
        self._rpc.call("graph", "add_relationship", **kwargs)

    def index_memory(self, **kwargs: Any) -> int:
        return int(self._rpc.call("graph", "index_memory", **kwargs) or 0)

    def remove_memory(self, memory_id: str) -> bool:
        return bool(self._rpc.call("graph", "remove_memory", memory_id=memory_id))

    def quarantine_junk_entities(self) -> int:
        return int(self._rpc.call("graph", "quarantine_junk_entities") or 0)

    def purge_junk_entities(self) -> int:
        """Alias for quarantine_junk_entities — matches KuzuGraphStore's method name."""
        return self.quarantine_junk_entities()

    def clear_scope(self) -> tuple[int, int]:
        result = self._rpc.call("graph", "clear_scope") or [0, 0]
        return (int(result[0]), int(result[1]))

    def close(self) -> None:
        return None

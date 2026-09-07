"""Client-side adapters for the local shared argos service."""
from __future__ import annotations

import json
import logging
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List

if __package__:
    from .memory_service import endpoint_path
    from .store import MemoryRecord
    from .store_protocol import _PROTOCOL_VERSION
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from memory_service import endpoint_path
    from store import MemoryRecord
    from store_protocol import _PROTOCOL_VERSION

_START_LOCK_NAME = "hybrid_memory_service.starting"
_START_LOCK_STALE_SECS = 90.0  # > _START_TIMEOUT; a healthy spawner unlinks well before this
_DEFAULT_TIMEOUT = 30.0
_START_TIMEOUT = 30.0
# Retry-once policy (#20): only ConnectionRefusedError at connect time is
# retried — the request never reached the server, so retrying is safe for
# ANY method (no duplicate-write risk). Timeouts are NOT retried (the
# server may have processed the request).
_RETRY_BACKOFF_S = 0.5
# SC1: cap total response bytes to prevent unbounded memory consumption
# from a malicious or buggy server. 64MB is generous for any legitimate
# response (search results, candidates, graph traversals) while preventing
# a runaway server from exhausting client memory.
_MAX_RESPONSE_BYTES = 64 * 1024 * 1024

logger = logging.getLogger(__name__)


def _pid_alive(pid: int) -> bool:
    """Best-effort PID liveness check. Windows-safe (no os.kill signal 0).

    SC3: On Windows, ``GetExitCodeProcess`` returns ``STILL_ACTIVE = 259``
    for a running process. If a process exits with exit code 259, this
    function incorrectly reports it as alive. This is a very rare edge
    case (a process would have to exit with code 259). The 90-second
    stale-lock timeout (``_START_LOCK_STALE_SECS``) is the fallback:
    even if ``_pid_alive`` incorrectly reports alive, the lock will be
    considered stale after 90 seconds regardless.
    """
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
    """True when the lock holder is provably gone (dead PID) or ancient.

    SC4: If ``_pid_alive`` returns False for a process that's actually
    alive but not responding to ``OpenProcess`` (Windows) or
    ``os.kill(pid, 0)`` (POSIX), the lock is considered stale and a
    second service may be started. This is safe because the single-
    instance guard in ``memory_service.serve`` probes the port and exits
    cleanly if a service is already running — the duplicate does no harm.
    """
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
    """RPC failure carrying the server-reported error class (#20).

    The server now returns ``error_class`` + a short traceback in the
    error envelope; this exception surfaces the class so clients can
    distinguish failure kinds (e.g. ValueError vs OSError) instead of
    pattern-matching on message text.
    """

    def __init__(self, message: str, error_class: str | None = None) -> None:
        super().__init__(message)
        self.error_class = error_class or "SharedMemoryServiceError"
        self.received_version: int | None = None  # #246: set on VersionMismatch


def _read_endpoint(home: Path) -> dict | None:
    path = endpoint_path(home)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, dict) and value.get("host") and value.get("port") and value.get("token"):
            return value
    except (OSError, ValueError, TypeError):
        pass
    return None


def _compute_gate_hmac(gate_secret: str, request: dict) -> str:
    """#200 PR-2 fix: compute HMAC-SHA256 over the request (minus
    _gate_hmac and token) using the boot-time gate_secret. The service
    verifies this before honoring _confirmed — a raw RPC caller without
    the gate_secret cannot forge it."""
    import hmac as _hmac
    import hashlib as _hashlib
    body = {k: v for k, v in request.items() if k not in ("_gate_hmac", "token")}
    return _hmac.new(
        gate_secret.encode("utf-8"),
        json.dumps(body, sort_keys=True).encode("utf-8"),
        _hashlib.sha256,
    ).hexdigest()


def _record_from_dict(value: dict | None) -> MemoryRecord | None:
    if not value:
        return None
    return MemoryRecord(
        memory_id=value.get("memory_id", ""),
        category=value.get("category", "context_note"),
        content=value.get("content", ""),
        # SC5: use `or []` / `or {}` to avoid sharing the same mutable
        # default object across calls when the key is missing or falsy.
        tags=value.get("tags") or [],
        payload=value.get("payload") or {},
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
        namespace=value.get("namespace", "conversation"),
        client_scope=value.get("client_scope"),
        doc_class=value.get("doc_class"),
        source_doc_id=value.get("source_doc_id"),
        source_loc=value.get("source_loc"),
        extraction_method=value.get("extraction_method"),
        extracted_at=value.get("extracted_at"),
        verified_state=value.get("verified_state", "current"),
        verified_at=value.get("verified_at"),
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
        self._default_user_id = user_id or "default_user"
        # user_id is thread-local (#20): set_user_scope from one thread
        # must not change the scope stamped on another thread's requests.
        self._scope = threading.local()
        self._ensure_service()

    @property
    def user_id(self) -> str:
        return getattr(self._scope, "user_id", self._default_user_id)

    @user_id.setter
    def user_id(self, value: str) -> None:
        self._scope.user_id = value or "default_user"

    def _request(self, request: dict, timeout: float = _DEFAULT_TIMEOUT) -> Any:
        """Send one request; retry once on connection-refused (#20) or
        VersionMismatch (#246).

        ConnectionRefusedError at connect time means the request never
        reached the server (e.g. the service is mid-restart), so a retry
        is safe for ANY method — no duplicate-write risk. The endpoint
        file is re-read on retry because a restart may have bound a new
        port. Timeouts and other OSErrors are NOT retried: the request
        may have been processed.

        #246: VersionMismatch means the running service is stale (old
        protocol). Self-heal: stop the stale service, respawn via
        _ensure_service, retry ONCE. Turns silent drift into self-healing.
        """
        attempt = 0
        while True:
            attempt += 1
            try:
                return self._request_once(request, timeout)
            except ConnectionRefusedError as exc:
                if attempt >= 2:
                    raise SharedMemoryServiceError(
                        f"shared memory service refused connection: {exc}",
                        error_class="ConnectionRefusedError",
                    ) from exc
                time.sleep(_RETRY_BACKOFF_S)
            except SharedMemoryServiceError as exc:
                if exc.error_class == "VersionMismatch" and attempt < 2:
                    # #246: stale service — kill + respawn + retry once.
                    logger.warning(
                        "Protocol version mismatch from service (got %s); "
                        "respawning stale service and retrying",
                        getattr(exc, "received_version", "?"),
                    )
                    self._kill_stale_service()
                    self._ensure_service()
                    continue
                raise

    def _request_once(self, request: dict, timeout: float) -> Any:
        endpoint = _read_endpoint(self.home)
        if endpoint is None:
            raise SharedMemoryServiceError("shared memory service endpoint is unavailable")
        request = dict(request)
        request["v"] = _PROTOCOL_VERSION
        request["token"] = endpoint["token"]
        request.setdefault("user_id", self.user_id)
        # #200 PR-2 fix: if this is a gated call, compute the HMAC over
        # the final request (all envelope fields added) using the
        # boot-time gate_secret. The service verifies this before
        # honoring _confirmed. The _needs_gate_hmac flag is stripped
        # from the request before sending (it's a client-side marker).
        if request.pop("_needs_gate_hmac", False):
            gate_secret = endpoint.get("gate_secret", "")
            if gate_secret:
                request["_gate_hmac"] = _compute_gate_hmac(gate_secret, request)
        try:
            with socket.create_connection(
                (str(endpoint["host"]), int(endpoint["port"])), timeout=timeout
            ) as connection:
                connection.settimeout(timeout)
                connection.sendall((json.dumps(request, ensure_ascii=False) + "\n").encode("utf-8"))
                chunks = []
                total_read = 0
                while True:
                    chunk = connection.recv(65536)
                    if not chunk:
                        break
                    total_read += len(chunk)
                    # SC1: cap total response bytes to prevent unbounded
                    # memory consumption from a malicious or buggy server.
                    if total_read > _MAX_RESPONSE_BYTES:
                        raise SharedMemoryServiceError(
                            f"shared memory service response exceeded {_MAX_RESPONSE_BYTES} bytes",
                            error_class="ResponseTooLarge",
                        )
                    chunks.append(chunk)
                    if b"\n" in chunk:
                        break
            response = json.loads(b"".join(chunks).splitlines()[0].decode("utf-8"))
        except ConnectionRefusedError:
            # Propagate so the retry loop in _request can handle it (#20):
            # the request never reached the server, so retrying is safe.
            raise
        except (OSError, ValueError, IndexError) as exc:
            raise SharedMemoryServiceError(
                f"shared memory service request failed: {exc}",
                error_class=type(exc).__name__,
            ) from exc
        if not response.get("ok"):
            # Error envelope (#20): surface the server's error class.
            err = SharedMemoryServiceError(
                str(response.get("error", "service error")),
                error_class=str(response.get("error_class") or "ServiceError"),
            )
            # #246: attach the received version for VersionMismatch so the
            # retry loop can log it.
            err_info = response.get("error")
            if isinstance(err_info, dict) and err_info.get("class") == "VersionMismatch":
                err.received_version = err_info.get("received")
            raise err
        return response.get("result")

    def _healthy(self) -> bool:
        try:
            result = self._request({"method": "health"}, timeout=1.0)
            return isinstance(result, dict) and result.get("status") == "ok"
        except SharedMemoryServiceError:
            return False

    def _kill_stale_service(self) -> None:
        """#246: stop the running service so _ensure_service can respawn it.

        Best-effort: sends a shutdown request via the current endpoint.
        If that fails (service already dead, endpoint stale), just
        unlink the endpoint file so _ensure_service starts fresh.
        """
        try:
            self._request_once({"method": "shutdown"}, timeout=3.0)
        except Exception:
            pass  # service may already be dead — that's fine
        # Remove the stale endpoint so _ensure_service doesn't try to
        # connect to a dead port.
        try:
            ep = endpoint_path(self.home)
            ep.unlink(missing_ok=True)
        except Exception:
            pass

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

    def call_gated(self, component: str, method: str, **args: Any) -> Any:
        """Like call(), but marks the request as facade-confirmed with
        an HMAC proof the server verifies.

        #200 PR-2 fix: the _confirmed flag goes in the request ENVELOPE,
        NOT in args. But _confirmed alone is still client-asserted — so
        call_gated also sets _needs_gate_hmac=True, which _request_once
        uses to compute an HMAC-SHA256 over the final request (after all
        envelope fields like v, token, user_id are added) using the
        boot-time gate_secret. The service verifies the HMAC before
        honoring _confirmed. A raw RPC caller with the endpoint token
        but without the gate_secret cannot forge the HMAC.
        """
        request = {"component": component, "method": method, "args": args}
        request["_confirmed"] = True
        request["_needs_gate_hmac"] = True
        return self._request(request)

    def stop_service(self) -> Any:
        return self._request({"method": "shutdown"})

    def backup(self, dst_root: str | None = None, retention: int | None = None,
               tenant: str | None = None) -> Any:
        """Trigger a service-coordinated backup. Returns the manifest dict.

        ``tenant`` (#49) selects the cell to back up; None = default tenant.
        """
        args: Dict[str, Any] = {}
        if dst_root is not None:
            args["dst_root"] = dst_root
        if retention is not None:
            args["retention_snapshots"] = retention
        if tenant is not None:
            args["tenant"] = tenant
        return self._request({"method": "backup", "args": args})

    def list_backups(self, dst_root: str | None = None) -> Any:
        """List available snapshots in the backup destination."""
        args: Dict[str, Any] = {"list": True}
        if dst_root is not None:
            args["dst_root"] = dst_root
        return self._request({"method": "backup", "args": args})


class SharedMemoryStore:
    """DuckDBMemoryStore-compatible client backed by the shared service."""

    def __init__(self, home: str | Path, user_id: str = "default_user", embedder=None) -> None:
        self.home = Path(home)
        self.db_path = self.home / "hybrid_memory.duckdb"
        self._default_user_id = user_id or "default_user"
        # Thread-local scope (#20): see set_user_scope.
        self._scope = threading.local()
        self._rpc = _SharedRPC(self.home, self._default_user_id)

    @property
    def user_id(self) -> str:
        return getattr(self._scope, "user_id", self._default_user_id)

    @user_id.setter
    def user_id(self, value: str | None) -> None:
        self._scope.user_id = value or "default_user"

    def set_user_scope(self, user_id: str | None) -> None:
        """Set the scope for the CURRENT thread only (#20).

        Previously this mutated shared instance state, so concurrent calls
        from threads with different scopes raced: thread A's
        set_user_scope("alice") could be observed by thread B's next
        request. Now each thread carries its own scope; the server still
        re-scopes the store per request under its lock (defense in depth).
        """
        self.user_id = user_id
        self._rpc.user_id = user_id

    def search(
            self,
            query: str,
            limit: int = 5,
            exclude_categories: List[str] | None = None,
            category_filter: str | None = None,
            project_id: str | None = None,
            namespace: str | None = None,
            client_scope: str | None = None,
            as_of: str | None = None,
            suppress_retrieval: bool = False,
            include_expired: bool = False,
            include_closed: bool = False,
            include_archived: bool = False,
        ) -> List[MemoryRecord]:
            values = self._rpc.call(
                "store", "search", query=query, limit=limit,
                exclude_categories=exclude_categories,
                category_filter=category_filter,
                project_id=project_id,
                namespace=namespace,
                client_scope=client_scope,
                as_of=as_of,
                suppress_retrieval=suppress_retrieval,
                include_expired=include_expired,
                include_closed=include_closed,
                include_archived=include_archived,
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

    def provenance(self, memory_id: str) -> dict:
        """#280: Assemble the full provenance view for a memory (RPC proxy)."""
        return self._rpc.call("store", "provenance", memory_id=memory_id) or {}

    def explain_retrieval(
        self,
        query: str,
        expected_memory_id: str,
        *,
        top_k: int = 20,
        project_id: str | None = None,
    ) -> dict:
        """#280: diagnose why a memory did NOT surface in retrieval (RPC proxy).

        Read-only, deterministic, zero-LLM. The diagnostic pass runs in
        the memory service process (no writes, retrieval suppressed,
        no reranker side-effects).
        """
        return self._rpc.call(
            "store", "explain_retrieval",
            query=query,
            expected_memory_id=expected_memory_id,
            top_k=top_k,
            project_id=project_id,
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

    def ingest_structured(self, **kwargs: Any) -> dict:
        """#289: structured ingestion (JSON/CSV → memory with provenance)."""
        result = self._rpc.call("store", "ingest_structured", **kwargs)
        return result or {
            "mode": kwargs.get("mode", "preview"), "wrote": False,
            "rows": [], "errors": [],
        }

    def erase_subject(self, **kwargs: Any) -> dict:
        """#293: POPIA erase-request workflow (provable deletion)."""
        result = self._rpc.call("store", "erase_subject", **kwargs)
        return result or {
            "mode": kwargs.get("mode", "preview"), "wrote": False,
            "records": [],
        }

    def list_deletion_receipts(self, **kwargs: Any) -> list:
        """#293: query the append-only deletion-receipt log."""
        return self._rpc.call(
            "store", "list_deletion_receipts", **kwargs,
        ) or []

    def verify_erase_receipt(self, receipt_id: str) -> dict:
        """#293: verify a deletion receipt against the live store."""
        return self._rpc.call(
            "store", "verify_erase_receipt", receipt_id=receipt_id,
        ) or {"valid": False, "reason": "receipt not found"}

    def export_portable(self, **kwargs: Any) -> dict:
        """#294: portable export (anti-lock-in, data sovereignty)."""
        result = self._rpc.call("store", "export_portable", **kwargs)
        return result or {"header": {}, "rows": [], "jsonl": "", "markdown": ""}

    def import_portable(self, **kwargs: Any) -> dict:
        """#294: portable re-import (idempotent, tombstone-aware)."""
        result = self._rpc.call("store", "import_portable", **kwargs)
        return result or {
            "mode": kwargs.get("mode", "preview"), "wrote": False,
            "rows": [], "errors": [],
        }

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

    def load_rollup_candidates(self, limit: int) -> list:
        """Proxy for store.load_rollup_candidates (RU2)."""
        values = self._rpc.call(
            "store", "load_rollup_candidates", limit=limit,
        ) or []
        return [_record_from_dict(value) for value in values]

    def count_rollup_candidates_since(self, since: str | None) -> int:
        """Proxy for store.count_rollup_candidates_since (RU3 novelty gate)."""
        return int(self._rpc.call(
            "store", "count_rollup_candidates_since", since=since,
        ) or 0)

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

    def mark_superseded(
        self, memory_id: str, reason: str = "",
        superseded_by: str | None = None,
    ) -> bool:
        """Live admin: supersede a memory (sets valid_to; read side excludes it).

        Route for one-off/retroactive chains — the candidate-review path uses
        the same store method internally.
        """
        return bool(
            self._rpc.call(
                "store", "mark_superseded",
                memory_id=memory_id, reason=reason, superseded_by=superseded_by,
            )
        )

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

    def facade_delete_memory(self, **kwargs: Any) -> bool | dict:
        """#200 Spec-10: sanctioned facade-only delete path.

        delete_memory is in _FORBIDDEN_STORE_METHODS on the RPC boundary
        (no raw RPC passthrough). The facade gates this with ctx.is_loopback
        + server-derived identity before calling this method. The service
        handler enforces the SAME controls server-side: strict
        _confirmed (envelope flag, not client-supplied confirm), strict
        CAS (expected_version required, compare+delete atomically under
        the tenant lock), identity stripped + service-resolved, and an
        audit row for every call (denied included). A raw RPC caller
        with the endpoint token cannot skip these gates.

        #200 PR-2 fix: uses call_gated() to set _confirmed in the RPC
        envelope — same mechanism as collection writes. A client-supplied
        confirm in args is stripped by _sanitize_args and NOT trusted.
        """
        kwargs.pop("confirm", None)  # strip — gate authority is in the envelope
        value = self._rpc.call_gated("store", "facade_delete_memory", **kwargs)
        if value is False or value is None:
            return False
        return value

    # -- #200 Spec-10 PR 2/3: Collections proxies ---------------------------
    # Read proxies use call() (un-gated). Write proxies use call_gated()
    # which sets _confirmed in the RPC envelope — the service checks
    # this envelope flag (NOT a client-supplied confirm in args) as the
    # gate authority for collection writes.

    def list_collections(self, **kwargs: Any) -> List[Dict[str, Any]]:
        return list(self._rpc.call("store", "list_collections", **kwargs) or [])

    def list_collection_items(self, **kwargs: Any) -> List[Dict[str, Any]]:
        return list(self._rpc.call("store", "list_collection_items", **kwargs) or [])

    def count_collection_items(self, **kwargs: Any) -> int:
        return int(self._rpc.call("store", "count_collection_items", **kwargs) or 0)

    def get_collection(self, **kwargs: Any) -> Dict[str, Any] | None:
        return self._rpc.call("store", "get_collection", **kwargs)

    def create_collection(self, **kwargs: Any) -> Dict[str, Any]:
        kwargs.pop("confirm", None)  # strip — gate authority is in the envelope
        return self._rpc.call_gated("store", "create_collection", **kwargs)

    def add_collection_item(self, **kwargs: Any) -> Dict[str, Any]:
        kwargs.pop("confirm", None)
        return self._rpc.call_gated("store", "add_collection_item", **kwargs)

    def update_collection_item(self, **kwargs: Any) -> Dict[str, Any]:
        kwargs.pop("confirm", None)
        return self._rpc.call_gated("store", "update_collection_item", **kwargs)

    def remove_collection_item(self, **kwargs: Any) -> Dict[str, Any]:
        kwargs.pop("confirm", None)
        return self._rpc.call_gated("store", "remove_collection_item", **kwargs)

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

    def run_compaction(
        self,
        *,
        interval_days: int = 7,
        aggressiveness: float = 1.0,
        dry_run: bool = False,
    ) -> dict:
        """#281: run schedule-aware compaction SERVER-SIDE via RPC.

        The compaction pass needs direct access to consolidate(),
        set_state, _fetch_records, and connection — all of which are
        in _FORBIDDEN_STORE_METHODS or private on the proxy. This
        narrow RPC method executes run_compaction() inside the service
        process and returns the report dict.
        """
        return self._rpc.call(
            "store", "run_compaction",
            interval_days=interval_days,
            aggressiveness=aggressiveness,
            dry_run=dry_run,
        ) or {}

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
        """SC6: No-op for the shared service client.

        ``_SharedRPC`` doesn't hold persistent connections — each request
        opens and closes a socket. There are no resources to release.
        """
        return None

    # -- access audit (#312) --------------------------------------------------

    def write_access_audit(
        self,
        *,
        user_id: str,
        query_text: str,
        granted_count: int,
        denied_count: int,
        denied_scopes: str | None = None,
        excluded: bool = False,
        tenant: str | None = None,
    ) -> None:
        """#312: Proxy for store.write_access_audit over RPC.

        Appends a row to the durable access_audit table on the shared
        service. Used by the facade (#300) to route denials to the
        durable sink when the store handle is a SharedMemoryStore.

        The store-side method hashes query_text (SHA-256, 16 chars) —
        no raw query text crosses the RPC boundary in the persisted row.
        """
        self._rpc.call(
            "store", "write_access_audit",
            user_id=user_id,
            query_text=query_text,
            granted_count=int(granted_count),
            denied_count=int(denied_count),
            denied_scopes=denied_scopes,
            excluded=bool(excluded),
            tenant=tenant,
        )

    def export_access_audit(
        self,
        *,
        limit: int = 10000,
        format: str = "jsonl",
    ) -> str:
        """#312: Proxy for store.export_access_audit over RPC.

        Returns the access audit log as JSONL or CSV. The service-side
        dispatch (#128) restricts export to wheel/principals only when
        an ACL config is active.
        """
        return self._rpc.call(
            "store", "export_access_audit",
            limit=limit, format=format,
        ) or ""

    # -- #347: mutation_events read surface + actor context -------------------

    def export_mutation_events(
        self,
        *,
        limit: int = 10000,
        format: str = "jsonl",
    ) -> str:
        """#347: Proxy for store.export_mutation_events over RPC.

        Returns the mutation event log as JSONL or CSV. Same wheel/
        principals gate as export_access_audit. No rotation — the full
        history is exportable.
        """
        return self._rpc.call(
            "store", "export_mutation_events",
            limit=limit, format=format,
        ) or ""

    def list_mutation_events(
        self,
        *,
        limit: int = 10000,
        offset: int = 0,
        event_type: str | None = None,
    ) -> list:
        """#347: Proxy for store.list_mutation_events over RPC.

        Returns a paginated, scope-filtered list of mutation events.
        """
        return self._rpc.call(
            "store", "list_mutation_events",
            limit=limit, offset=offset, event_type=event_type,
        ) or []

    def set_actor_context(self, actor_type: str = "human") -> None:
        """#347: Proxy for store.set_actor_context over RPC.

        The actor identity is server-derived from the resolved user_id
        (#341/#344) — the client cannot set it. Only actor_type is sent;
        the service overrides it to "model" in credential mode.

        Note: the per-request stamp in _call_store is authoritative; this
        method is retained for facade/local-store compatibility but is a
        no-op on the RPC path (the stamp already ran at _call_store entry).
        """
        self._rpc.call(
            "store", "set_actor_context",
            actor_type=actor_type,
        )

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
        self._default_user_id = user_id or "default_user"
        self._scope = threading.local()  # thread-local scope (#20)
        self._rpc = _SharedRPC(self.home, self._default_user_id)

    @property
    def user_id(self) -> str:
        return getattr(self._scope, "user_id", self._default_user_id)

    @user_id.setter
    def user_id(self, value: str | None) -> None:
        self._scope.user_id = value or "default_user"

    def set_user_scope(self, user_id: str | None) -> None:
        """Set the scope for the CURRENT thread only (#20, see SharedMemoryStore)."""
        self.user_id = user_id
        self._rpc.user_id = user_id

    def search_graph(
        self, term: str, limit: int = 100, as_of: str | None = None,
    ) -> List[dict]:
        return self._rpc.call(
            "graph", "search_graph", term=term, limit=limit, as_of=as_of,
        ) or []

    def memory_ids_for_query(self, query: str, limit: int = 100) -> List[str]:
        return self._rpc.call(
            "graph", "memory_ids_for_query", query=query, limit=limit,
        ) or []

    def query_graph(self, entity_id: str, as_of: str | None = None) -> List[dict]:
        return self._rpc.call(
            "graph", "query_graph", entity_id=entity_id, as_of=as_of,
        ) or []

    def traverse_graph(
        self,
        entity_id: str,
        depth: int = 2,
        limit: int = 100,
        as_of: str | None = None,
    ) -> Dict[str, Any]:
        return self._rpc.call(
            "graph", "traverse_graph",
            entity_id=entity_id, depth=depth, limit=limit, as_of=as_of,
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
        """SC6: No-op — see SharedMemoryStore.close."""
        return None

"""In-service DuckDB-to-Kuzu drift watch (#329).

Wire the #305 reconciliation probe into the running memory service so
drift between DuckDB (source of truth) and the Kuzu graph (derived data,
see backup.py) is detected without a human running
``scripts/reconcile_graph.py`` — and healed the same way
``backfill_graph.py`` heals it: re-index the missing memories through
``graph.index_memory`` (idempotent MERGE + evidence lists; source
records are never modified).

Detection is literally ``reconcile_graph.reconcile`` (same semantics as
the CLI: missing / extra / detection-errors; detection errors are never
counted as drift and never healed).

Surface: ``memory_service`` ``get_status`` includes a cached
``graph_drift`` block per visible tenant; the loop itself starts from
``memory_service`` at boot (interval from
``graph_drift_check_interval_min``, 0 disables; auto-heal from
``graph_drift_auto_heal``, default true).
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

try:  # pragma: no cover - import-context dependent
    from reconcile_graph import reconcile
except ImportError:
    from .reconcile_graph import reconcile

try:  # pragma: no cover - import-context dependent
    from liveness import increment_counter
except ImportError:
    from .liveness import increment_counter

logger = logging.getLogger(__name__)

_DEFAULT_INTERVAL_MIN = 360  # 6h
_MAX_INTERVAL_MIN = 10080  # 7d

# Bounds the work (and graph-typing calls) one cycle can trigger; the
# next cycle picks up the remainder (MERGE-safe, so overlap is harmless).
HEAL_LIMIT_PER_CYCLE = 200

_stop = threading.Event()
_started = threading.Event()
_last: Dict[str, Any] = {}


def _read_config(home: Any) -> Dict[str, Any]:
    """Minimal read of the install config for the watch's own knobs.

    Deliberately independent of memory_service's loader: the watch must
    not fail (or import-cycle) because of unrelated config plumbing.
    """
    try:
        path = Path(str(home)) / "hybrid_memory.json"
        return json.loads(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def _cfg_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _read_interval_min(config: Dict[str, Any]) -> int:
    """Interval in minutes; 0 disables the watch; junk falls back to 6h."""
    raw = config.get("graph_drift_check_interval_min", _DEFAULT_INTERVAL_MIN)
    try:
        val = int(float(raw))
    except (TypeError, ValueError):
        return _DEFAULT_INTERVAL_MIN
    if val < 0:
        return _DEFAULT_INTERVAL_MIN
    return min(val, _MAX_INTERVAL_MIN)


def heal_missing(
    store: Any,
    graph: Any,
    missing_ids: Any,
    *,
    limit: int = HEAL_LIMIT_PER_CYCLE,
) -> Tuple[int, int]:
    """Re-index memories whose graph node is missing. Returns (healed, failed).

    Fail-soft per memory: one bad record never blocks the rest. Unknown
    ids (record no longer active) are skipped, not failures.
    """
    ids = [str(mid) for mid in missing_ids][:limit]
    if not ids:
        return 0, 0
    by_id: Dict[str, Any] = {}
    try:
        for rec in store.list_recent(limit=10000):
            mid = getattr(rec, "memory_id", None)
            if mid:
                by_id[str(mid)] = rec
    except Exception as exc:
        logger.debug("drift heal: could not load records: %s", exc)
        return 0, len(ids)
    healed = failed = 0
    for mid in ids:
        rec = by_id.get(mid)
        if rec is None:
            continue
        try:
            graph.index_memory(
                memory_id=mid,
                category=str(getattr(rec, "category", "") or "context_note"),
                content=str(getattr(rec, "content", "") or ""),
                tags=list(getattr(rec, "tags", None) or []),
                created_at=getattr(rec, "created_at", None),
            )
            healed += 1
        except Exception as exc:
            failed += 1
            logger.debug("drift heal: index_memory failed for %s: %s", mid, exc)
    return healed, failed


def run_drift_cycle(
    store: Any,
    graph: Any,
    *,
    auto_heal: bool = True,
    sample_size: int = 10,
    heal_limit: int = HEAL_LIMIT_PER_CYCLE,
) -> Dict[str, Any]:
    """One check (+ optional heal) for one store/graph pair."""
    result = reconcile(store, graph, sample_size=sample_size, full_lists=True)
    result["checked_at"] = datetime.now(timezone.utc).isoformat()
    if result.get("drift"):
        increment_counter("graph_drift_detected")
        logger.warning(
            "graph drift: %d missing in graph, %d extra (%d detection errors)",
            result.get("missing_in_graph_count", 0),
            result.get("extra_in_graph_count", 0),
            result.get("detection_errors", 0),
        )
        if auto_heal and result.get("missing_in_graph_count", 0) > 0:
            healed, failed = heal_missing(
                store, graph, result.get("missing_in_graph_all") or [],
                limit=heal_limit,
            )
            result["healed"] = healed
            result["heal_failed"] = failed
            if healed:
                increment_counter("graph_drift_healed", healed)
                logger.info(
                    "graph drift heal: re-indexed %d/%d missing memories",
                    healed, result["missing_in_graph_count"],
                )
    return result


def run_drift_cycle_for_service(
    service: Any, *, auto_heal: Optional[bool] = None
) -> Dict[str, Any]:
    """Check every tenant's cell; cache the summary for the status surface."""
    if auto_heal is None:
        config = _read_config(getattr(service, "home", ""))
        auto_heal = _cfg_bool(config.get("graph_drift_auto_heal"), True)
    per_tenant: Dict[str, Any] = {}
    overall_drift = False
    for name, tenant in (getattr(service, "_tenants", {}) or {}).items():
        store = getattr(tenant, "store", None)
        graph = getattr(tenant, "graph", None)
        if store is None or graph is None:
            continue
        try:
            res = run_drift_cycle(store, graph, auto_heal=auto_heal)
        except Exception as exc:
            logger.warning("graph drift check failed for tenant %s: %s", name, exc)
            res = {"error": str(exc), "drift": False}
        per_tenant[str(name)] = res
        overall_drift = overall_drift or bool(res.get("drift"))
    summary = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "drift": overall_drift,
        "tenants": per_tenant,
    }
    _last.clear()
    _last.update(summary)
    return summary


def last_result() -> Dict[str, Any]:
    """Cached latest summary (empty dict until the first cycle completes)."""
    return dict(_last)


def start_drift_watch_thread(service: Any, *, first_delay_s: float = 120.0):
    """Start the daemon loop once per process. Returns the thread, or None.

    None when disabled (interval 0) or already started. The first cycle
    is delayed so boot sweeps and the cold-start reranker load win.
    """
    config = _read_config(getattr(service, "home", ""))
    interval_min = _read_interval_min(config)
    if interval_min <= 0:
        logger.info(
            "graph drift watch disabled (graph_drift_check_interval_min=0)"
        )
        return None
    if _started.is_set():
        return None
    _started.set()

    def _loop() -> None:
        if _stop.wait(first_delay_s):
            return
        while True:
            try:
                run_drift_cycle_for_service(service)
            except Exception:
                logger.debug("drift watch cycle error", exc_info=True)
            if _stop.wait(interval_min * 60):
                return

    thread = threading.Thread(
        target=_loop, name="graph-drift-watch", daemon=True,
    )
    thread.start()
    logger.info("graph drift watch started (every %d min)", interval_min)
    return thread

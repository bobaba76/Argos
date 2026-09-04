"""Shared state object for the store mixin family (#249-slice).

The four store mixins (store_core / store_write / store_retrieval /
store_maintenance) share ~12 ``self._*`` attributes that were previously
scattered across the files — initialised in one, read/written in another,
with no single place documenting the contract.  This dataclass is that
place: every field has a docstring, and the four mixins access shared
state through ``self._state.<attr>`` instead of bare ``self._<attr>``.

This is a pure state-relocation: no behaviour change, no method moves,
no composition refactor.  The 150+ audit tests are the safety net.
"""
from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Deque, Dict, List, Optional, Tuple


@dataclass
class StoreMixinState:
    """Cross-mixin shared state for ``DuckDBMemoryStore``.

    Created once in ``StoreCoreMixin.__init__`` and accessed by all four
    mixins via ``self._state``.  Each field below was previously a bare
    ``self._<name>`` attribute set in one mixin and read in another.
    """

    # -- concurrency ---------------------------------------------------------
    lock: RLock = field(default_factory=RLock)
    """Re-entrant lock guarding all store mutations and reads.
    Set in ``store_core.__init__``; used as ``with self._state.lock:`` in
    all four mixins (~70 acquire sites)."""

    # -- scale-trigger metrics -----------------------------------------------
    scale_warn_latency_ms: float = 300.0
    """p95 latency threshold (ms) above which a scale warning fires.
    Set in ``store_core.__init__``; updated by ``store_retrieval.set_scale_thresholds``."""

    scale_warn_records: int = 5000
    """Record-count threshold above which a scale warning fires.
    Set in ``store_core.__init__``; updated by ``store_retrieval.set_scale_thresholds``."""

    scale_window: int = 50
    """Rolling window size for the latency deque.
    Set in ``store_core.__init__``; read only there to size the deque."""

    scale_latencies: Deque[float] = field(default_factory=lambda: deque(maxlen=50))
    """Rolling p95 latency window (milliseconds).
    Initialised in ``store_core.__init__``; appended and read in
    ``store_retrieval._record_scale_metric`` and ``get_scale_metrics``."""

    scale_queries: int = 0
    """Total retrieval queries measured.
    Incremented in ``store_retrieval._record_scale_metric``; read in
    ``get_scale_metrics``."""

    scale_warnings_fired: int = 0
    """Number of scale warnings emitted.
    Incremented in ``store_retrieval._record_scale_metric``; read in
    ``get_scale_metrics``."""

    scale_last_count_check: int = 0
    """``scale_queries`` value at the last record-count check.
    Updated in ``store_retrieval._record_scale_metric`` to avoid
    counting records on every query (checked every 25 queries)."""

    scale_record_count: Optional[int] = None
    """Cached total record count (refreshed every 25 queries).
    Updated in ``store_retrieval._record_scale_metric``; read in
    ``get_scale_metrics`` and the warning threshold check."""

    # -- alias cache ---------------------------------------------------------
    alias_cache: Optional[Dict[str, List[Tuple[str, str]]]] = None
    """Cached alias→canonical mapping (avoids full-table scan per query).
    Invalidated (set to ``None``) on ``add_alias`` / ``remove_alias`` /
    ``set_user_scope`` in ``store_core`` and ``store_maintenance``;
    populated on first ``resolve_aliases`` call in ``store_maintenance``."""

    # -- read-only fallback --------------------------------------------------
    read_only: bool = False
    """True when DuckDB was opened read-only (locked by another process).
    Set in ``store_core._connect``; read by ``is_read_only`` and to gate
    writes in ``store_core``."""

    # -- retrieval engine ----------------------------------------------------
    retriever: Any = None
    """Active ``DuckDBRetriever`` instance (lazy-initialised).
    Set in ``store_retrieval._get_retriever`` / ``set_retriever``;
    read in ``store_retrieval``."""

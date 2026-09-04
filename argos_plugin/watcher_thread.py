"""Watcher runtime thread (W6, issue #213): wires the spec-07 watcher
into the provider lifecycle.

The watcher was a library only — ``scan_pass`` and ``extract_facts_from_doc``
were called only from tests. This module provides the daemon thread that
runs the catalog pass on a periodic timer, populates ``file_catalog``,
tombstones deleted files, and routes hot docs through the extraction
pass using the config-specified ``extraction_llm_model`` / ``extraction_llm_provider``.

Config-gated: no watcher config = zero behaviour change. The thread is
only started when ``watcher_enabled`` is true and ``watcher_scan_roots``
is non-empty.

W3/W4 (folded in):
- CSV extraction streams via ``csv.reader`` (bounded by ``_MAX_CSV_TEXT_CHARS``).
- The extraction text returned to callers is bounded — the thread
  persists a truncated excerpt in the catalog, not the full document text.

Started by the provider after initialization (mirrors the
``StaleReviewSweepThread`` pattern). Runs every ``watcher_interval_min``
minutes. Fail-soft: any exception in a scan pass is logged and the
thread continues to the next interval. The thread never blocks the
RPC hot path.
"""
from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# W4: cap the excerpt persisted in the catalog to bound memory. The full
# extraction text is never held indefinitely by the caller — only a
# bounded excerpt is kept for audit/debugging.
_MAX_EXCERPT_CHARS = 4096


def _build_catalog_index(store: Any) -> Dict[str, Dict[str, Any]]:
    """Load the current catalog state from the store into the dict format
    expected by ``watcher.scan_pass``.

    Returns ``{file_id: {paths: [...], status: ...}}``.
    """
    catalog: Dict[str, Dict[str, Any]] = {}
    try:
        entries = store.list_catalog(status="active", limit=10000)
    except Exception as exc:
        logger.debug("watcher: list_catalog failed: %s", exc)
        return catalog
    for entry in entries:
        file_id = entry.get("file_id")
        if not file_id:
            continue
        canonical_path = entry.get("canonical_path", "")
        catalog[file_id] = {
            "paths": [canonical_path] if canonical_path else [],
            "status": entry.get("status", "active"),
        }
    return catalog


def run_watcher_pass(
    store: Any,
    roots: List[str],
    *,
    extraction_llm_model: str = "",
    extraction_llm_provider: str = "",
) -> Dict[str, int]:
    """Run one watcher pass: scan → catalog → tombstone → extract.

    Args:
        store: the memory store (DuckDBMemoryStore or SharedMemoryStore).
        roots: directories to scan.
        extraction_llm_model: model for the extraction LLM call.
        extraction_llm_provider: provider for the extraction LLM call.

    Returns:
        Dict with counts: ``{"new": N, "changed": N, "moved": N,
        "deleted": N, "unchanged": N, "extracted": N}``.

    Never raises — a failed pass is a no-op, not a crash.
    """
    counts = {"new": 0, "changed": 0, "moved": 0, "deleted": 0,
              "unchanged": 0, "extracted": 0}
    try:
        try:
            from .watcher import scan_pass, extract_facts_from_doc
        except ImportError:
            from watcher import scan_pass, extract_facts_from_doc
    except ImportError:
        logger.debug("watcher not importable — pass is a no-op")
        return counts

    catalog = _build_catalog_index(store)
    try:
        result = scan_pass(roots, catalog)
    except Exception as exc:
        logger.debug("watcher: scan_pass failed: %s", exc)
        return counts

    # Catalog pass: upsert new/changed/moved, tombstone deleted.
    for bucket in ("new", "changed", "moved"):
        for info in result.get(bucket, []):
            try:
                store.upsert_catalog_entry(
                    file_id=info["file_id"],
                    canonical_path=info["path"],
                    size=info["size"],
                    mtime=info["mtime"],
                    doc_type=info["doc_type"],
                    layout_family=info.get("layout_family"),
                )
                counts[bucket] += 1
            except Exception as exc:
                logger.debug("watcher: upsert failed for %s: %s",
                             info.get("path", "?"), exc)

    counts["unchanged"] = len(result.get("unchanged", []))

    for info in result.get("deleted", []):
        try:
            store.tombstone_catalog_entry(file_id=info["file_id"])
            counts["deleted"] += 1
        except Exception as exc:
            logger.debug("watcher: tombstone failed for %s: %s",
                         info.get("file_id", "?"), exc)

    # Extraction pass: hot docs only (new + changed). The extract_hash
    # gate in the store ensures unchanged files are not re-extracted.
    for info in result.get("new", []) + result.get("changed", []):
        try:
            facts, extract_hash, method, text = extract_facts_from_doc(
                info["path"],
                info["doc_type"],
                extraction_llm_model=extraction_llm_model,
                extraction_llm_provider=extraction_llm_provider,
            )
            if facts:
                # W4: persist a bounded excerpt, not the full text.
                excerpt = text[:_MAX_EXCERPT_CHARS] if text else ""
                # Write facts as candidates via the normal store path.
                for fact in facts:
                    if isinstance(fact, dict):
                        content = str(fact.get("content", "")).strip()
                        if content:
                            store.save_candidate(
                                content=content,
                                category=fact.get("category", "insight"),
                                source="watcher",
                                confidence=float(fact.get("confidence", 0.7)),
                                evidence_text=excerpt,
                                source_loc=info["path"],
                            )
                counts["extracted"] += 1
        except Exception as exc:
            logger.debug("watcher: extraction failed for %s: %s",
                         info.get("path", "?"), exc)

    if any(counts.values()):
        logger.info("watcher pass: %s", dict(counts))
    return counts


class WatcherThread:
    """Daemon thread that runs the watcher pass on a periodic timer.

    Started by the provider after initialization (config-gated). Runs
    every ``interval_min`` minutes. Fail-soft: any exception in a pass
    is logged and the thread continues to the next interval.

    The thread exits when ``stop()`` is called or the provider shuts
    down. It never blocks the RPC hot path.
    """

    def __init__(
        self,
        store: Any,
        *,
        scan_roots: List[str],
        interval_min: int = 30,
        extraction_llm_model: str = "",
        extraction_llm_provider: str = "",
    ):
        self._store = store
        self._scan_roots = scan_roots
        self._interval_s = max(60, interval_min * 60)  # at least 1 min
        self._extraction_llm_model = extraction_llm_model
        self._extraction_llm_provider = extraction_llm_provider
        self._thread: Optional[threading.Thread] = None
        self._stopped = threading.Event()

    def start(self) -> None:
        """Start the watcher thread (daemon, won't block process exit)."""
        if self._thread and self._thread.is_alive():
            return
        self._stopped.clear()
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="watcher-pass",
        )
        self._thread.start()

    def stop(self) -> None:
        """Signal the watcher thread to stop."""
        self._stopped.set()

    def _loop(self) -> None:
        """Main loop: sleep, scan, repeat."""
        # Startup catch-up: run one pass immediately on boot to populate
        # the catalog from a cold start, then settle into the interval.
        while not self._stopped.is_set():
            try:
                run_watcher_pass(
                    self._store,
                    self._scan_roots,
                    extraction_llm_model=self._extraction_llm_model,
                    extraction_llm_provider=self._extraction_llm_provider,
                )
            except Exception as exc:
                logger.debug("watcher pass failed: %s", exc)
            # Wait for the interval, but wake up early if stopped.
            self._stopped.wait(self._interval_s)

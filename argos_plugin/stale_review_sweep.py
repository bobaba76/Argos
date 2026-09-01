"""Stale-pending review sweep (#10): periodically re-review memory proposals
stranded in 'pending' after a failed/rate-limited reviewer call.

Consumes the four ``stale_review_*`` config keys that were previously
parsed but never read:
  - stale_review_sweep_enabled (bool, default true)
  - stale_review_interval_min (int, default 15)
  - stale_review_min_age_min (int, default 30)
  - stale_review_max_batch (int, default 25)

The sweep rides on the existing review_pending.py engine — it calls
``review_candidate_with_llm`` for each stale candidate and records the
outcome via ``store.review_candidate``. No auto-promotion: the decision
map is identical to the manual CLI; low-risk approvals become
``reviewed_approved`` and still require explicit promotion.

Runs as a daemon thread on the provider side (piggybacks on the existing
thread pattern). Fail-soft on LLM error — a failed sweep is a no-op,
not a crash. The sweep never blocks the RPC hot path.
"""
from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Decision map — identical to review_pending.py (no auto-promotion).
_DECISION_MAP = {
    "approve": "reviewed_approved",
    "reject": "rejected",
    "quarantine": "quarantined",
    "pending_user_confirmation": "pending_user_confirmation",
}


def _parse_iso(ts: str | None) -> Optional[datetime]:
    """Parse an ISO timestamp string, returning None on failure."""
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def _is_stale(candidate: dict, min_age_min: int, now: datetime) -> bool:
    """Check if a pending candidate is older than min_age_min minutes."""
    created = _parse_iso(candidate.get("created_at"))
    if created is None:
        # Can't parse the timestamp — err on the side of reviewing it.
        return True
    # Handle naive datetimes (no timezone) by assuming UTC.
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    age_min = (now - created).total_seconds() / 60.0
    return age_min >= min_age_min


def run_stale_review_sweep(
    store: Any,
    *,
    min_age_min: int = 30,
    max_batch: int = 25,
    llm_model: str = "",
    llm_provider: str = "",
) -> Dict[str, int]:
    """Run one sweep pass: re-review stale pending candidates.

    Args:
        store: the memory store (DuckDBMemoryStore or SharedMemoryStore).
        min_age_min: only re-review candidates older than this many minutes.
        max_batch: maximum candidates to re-review per sweep.
        llm_model: model for the LLM review call.
        llm_provider: provider for the LLM review call.

    Returns:
        Dict mapping outcome status to count (e.g.
        {"reviewed_approved": 3, "rejected": 1}).

    Never raises — a failed sweep is a no-op, not a crash. The sweep
    preserves the no-auto-promotion invariant: low-risk approvals become
    ``reviewed_approved`` and still require explicit promotion.
    """
    counts: Dict[str, int] = {}
    try:
        from reviewer import review_candidate_with_llm
    except ImportError:
        try:
            from .reviewer import review_candidate_with_llm
        except ImportError:
            logger.debug("reviewer not importable — sweep is a no-op")
            return counts

    # Fetch more than max_batch so we can filter by age and still fill
    # the batch. Cap at 500 to bound the query cost.
    fetch_limit = min(max_batch * 4, 500)
    try:
        candidates = store.list_candidates(status="pending", limit=fetch_limit)
    except Exception as exc:
        logger.debug("sweep: list_candidates failed: %s", exc)
        return counts

    now = datetime.now(timezone.utc)
    reviewed = 0
    for candidate in candidates:
        if reviewed >= max_batch:
            break
        if not _is_stale(candidate, min_age_min, now):
            continue  # fresh — may still be mid-review
        try:
            result = review_candidate_with_llm(
                candidate, model=llm_model, provider=llm_provider,
            )
            decision = result.get("decision", "pending_user_confirmation")
            status = _DECISION_MAP.get(decision, "pending_user_confirmation")
            store.review_candidate(
                candidate_id=candidate["candidate_id"],
                decision=status,
                reason=result.get("reason", ""),
                review_confidence=result.get("confidence"),
                review_model=result.get("review_model", "memory_review"),
                durability=result.get("durability"),
                scope=result.get("scope"),
            )
            counts[status] = counts.get(status, 0) + 1
            reviewed += 1
        except Exception as exc:
            logger.debug(
                "sweep: review failed for %s: %s",
                candidate.get("candidate_id", "?"), exc,
            )
            # Fail-soft: skip this candidate, continue the sweep.
            continue

    if reviewed:
        logger.info(
            "stale-review sweep: re-reviewed %d candidates (%s)",
            reviewed, dict(counts),
        )
    return counts


class StaleReviewSweepThread:
    """Daemon thread that runs the stale-review sweep on a periodic timer.

    Started by the provider after initialization. Runs every
    ``interval_min`` minutes. Fail-soft: any exception in a sweep pass
    is logged and the thread continues to the next interval.

    The thread exits when ``stop()`` is called or the provider shuts
    down. It never blocks the RPC hot path — the sweep runs in a
    background thread.
    """

    def __init__(
        self,
        store: Any,
        *,
        interval_min: int = 15,
        min_age_min: int = 30,
        max_batch: int = 25,
        llm_model: str = "",
        llm_provider: str = "",
    ):
        self._store = store
        self._interval_s = max(60, interval_min * 60)  # at least 1 min
        self._min_age_min = min_age_min
        self._max_batch = max_batch
        self._llm_model = llm_model
        self._llm_provider = llm_provider
        self._thread: Optional[threading.Thread] = None
        self._stopped = threading.Event()

    def start(self) -> None:
        """Start the sweep thread (daemon, won't block process exit)."""
        if self._thread and self._thread.is_alive():
            return
        self._stopped.clear()
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="stale-review-sweep",
        )
        self._thread.start()

    def stop(self) -> None:
        """Signal the sweep thread to stop."""
        self._stopped.set()

    def _loop(self) -> None:
        """Main loop: sleep, sweep, repeat."""
        # Startup catch-up: run one sweep immediately on boot to clear
        # any backlog from a long downtime, then settle into the interval.
        while not self._stopped.is_set():
            try:
                run_stale_review_sweep(
                    self._store,
                    min_age_min=self._min_age_min,
                    max_batch=self._max_batch,
                    llm_model=self._llm_model,
                    llm_provider=self._llm_provider,
                )
            except Exception as exc:
                logger.debug("stale-review sweep pass failed: %s", exc)
            # Wait for the interval, but wake up early if stopped.
            self._stopped.wait(self._interval_s)

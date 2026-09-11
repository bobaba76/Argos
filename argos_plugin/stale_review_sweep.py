"""Stale-pending review sweep (#10): periodically re-review memory proposals
stranded in 'pending' after a failed/rate-limited reviewer call, plus
never-confirmed candidates stranded in 'reviewed_approved' or
'pending_user_confirmation' (which the auto-reviewer can leave unconfirmed
indefinitely).

Consumes the four ``stale_review_*`` config keys that were previously
parsed but never read:
  - stale_review_sweep_enabled (bool, default true)
  - stale_review_interval_min (int, default 15)
  - stale_review_min_age_min (int, default 30)
  - stale_review_max_batch (int, default 25)

The sweep rides on the existing review_pending.py engine — it calls
``review_candidate_with_llm`` for each stale candidate and records the
outcome via ``store.review_candidate``. The decision map is identical to
the manual CLI: the sweep never reaches the user-confirmed class — a
low-risk approval becomes ``reviewed_approved``, which under the
materialize-await-lift semantic (#429) materializes (or keeps) the record
at its capped grounding tier; explicit user confirmation is still
required to LIFT the tier. It never promotes a candidate to ``approved``.

Coordination (#425): the daemon is started once per provider, so the store's
``claim_stale_review_pass`` provides cross-process single-flight — at most
one pass per interval across desktop sessions, gateway, and cron (previously
~43 daemon starts/day raced the same backlog). A per-candidate no-progress
cooldown (``sweep_attempts`` / ``sweep_next_review_at`` on the payload;
24h doubling to a 7-day cap) keeps a never-finalized backlog from being
re-reviewed on every pass.

Runs as a daemon thread on the provider side (piggybacks on the existing
thread pattern). Fail-soft on LLM error — a failed sweep is a no-op,
not a crash. The sweep never blocks the RPC hot path.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Decision map — identical to review_pending.py (no auto-promotion of the
# user-confirmed class; reviewed_approved materializes at the capped tier
# per #429 and still awaits explicit user confirmation to LIFT).
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


def _candidate_payload(candidate: dict) -> dict:
    """#425: candidate payload as a dict (rows may carry JSON str or dict)."""
    payload = candidate.get("payload")
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except (TypeError, ValueError):
            return {}
    return dict(payload) if isinstance(payload, dict) else {}


def _sweep_backoff(attempts: int) -> timedelta:
    """#425: cooldown after *attempts* consecutive no-progress re-reviews.

    24h after the first, doubling to a 7-day cap — a never-finalized
    backlog is re-examined on a decaying schedule instead of every pass.
    """
    hours = min(24 * (2 ** max(0, attempts - 1)), 24 * 7)
    return timedelta(hours=hours)


def _eligible_for_sweep(candidate: dict, now: datetime) -> bool:
    """#425: False while the candidate is inside its no-progress cooldown."""
    raw = _candidate_payload(candidate).get("sweep_next_review_at")
    if not raw:
        return True
    nxt = _parse_iso(raw)
    if nxt is None:
        return True
    if nxt.tzinfo is None:
        nxt = nxt.replace(tzinfo=timezone.utc)
    return now >= nxt


def run_stale_review_sweep(
    store: Any,
    *,
    min_age_min: int = 30,
    max_batch: int = 25,
    llm_model: str = "",
    llm_provider: str = "",
) -> Dict[str, int]:
    """Run one sweep pass: re-review stale never-finalized candidates.

    Args:
        store: the memory store (DuckDBMemoryStore or SharedMemoryStore).
        min_age_min: only re-review candidates older than this many minutes.
        max_batch: maximum candidates to re-review per sweep.
        llm_model: model for the LLM review call.
        llm_provider: provider for the LLM review call.

    The sweep covers the three never-finalized candidate states —
    'pending', 'reviewed_approved', and 'pending_user_confirmation' —
    deduped across statuses, so a strand in any of them is eventually
    re-examined (fixes the atlas Q4 gap: auto-approved-never-confirmed
    candidates sat forever).

    Returns:
        Dict mapping outcome status to count (e.g.
        {"reviewed_approved": 3, "rejected": 1}).

    Never raises — a failed sweep is a no-op, not a crash. The sweep
    preserves the no-auto-promotion invariant for the user-confirmed
    class: low-risk approvals become ``reviewed_approved``, which
    materializes at the capped grounding tier (#429, materialize-await-
    lift) and still awaits explicit user confirmation to LIFT the tier.
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
        candidates = []
        seen = set()
        for status in ("pending", "reviewed_approved", "pending_user_confirmation"):
            for candidate in store.list_candidates(status=status, limit=fetch_limit):
                if candidate.get("candidate_id") not in seen:
                    candidates.append(candidate)
                    seen.add(candidate.get("candidate_id"))
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
        if not _eligible_for_sweep(candidate, now):
            continue  # #425: inside the no-progress cooldown
        try:
            result = review_candidate_with_llm(
                candidate, model=llm_model, provider=llm_provider,
            )
            decision = result.get("decision", "pending_user_confirmation")
            status = _DECISION_MAP.get(decision, "pending_user_confirmation")
            outcome = store.review_candidate(
                candidate_id=candidate["candidate_id"],
                decision=status,
                reason=result.get("reason", ""),
                review_confidence=result.get("confidence"),
                review_model=result.get("review_model", "memory_review"),
                durability=result.get("durability"),
                scope=result.get("scope"),
                # #423: unattended engine — the unprivileged auto_review
                # class must be explicit so the storage boundary never
                # mistakes a sweep decision for a human confirmation.
                review_source="auto_review",
            )
            # #425: no-progress bookkeeping — escalate the cooldown when a
            # re-review leaves the candidate in the same never-finalized
            # state, so a stale backlog is not re-chewed every pass.
            try:
                note = getattr(store, "note_stale_review_outcome", None)
                if note is not None:
                    prior_attempts = int(
                        _candidate_payload(candidate).get("sweep_attempts") or 0
                    )
                    # The store may downgrade the decision (grounding
                    # ceiling) — compare against the FINAL status so a
                    # silently downgraded candidate still accrues the
                    # no-progress cooldown instead of being re-chewed.
                    final_status = status
                    if isinstance(outcome, dict):
                        _cand_row = outcome.get("candidate")
                        if isinstance(_cand_row, dict) and _cand_row.get("status"):
                            final_status = str(_cand_row["status"])
                    if final_status == str(candidate.get("status") or ""):
                        attempts = prior_attempts + 1
                        next_at = (now + _sweep_backoff(attempts)).isoformat()
                    else:
                        attempts = 0
                        next_at = None
                    note(
                        candidate["candidate_id"],
                        attempts=attempts, next_review_at=next_at,
                    )
            except Exception as exc:
                logger.debug("sweep: bookkeeping failed: %s", exc)
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
        """Main loop: claim, sweep, wait, repeat."""
        # Startup catch-up: run one sweep immediately on boot to clear
        # any backlog from a long downtime, then settle into the interval.
        while not self._stopped.is_set():
            try:
                # #425: cross-process single-flight. Every provider
                # (desktop, gateway, cron) starts its own daemon; the
                # store claim collapses them to at most one pass per
                # interval ACROSS processes. Claim window is a hair under
                # the interval so a daemon firing at the boundary can
                # still claim.
                claim = getattr(self._store, "claim_stale_review_pass", None)
                if claim is not None and not claim(self._interval_s * 0.95):
                    self._stopped.wait(self._interval_s)
                    continue
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

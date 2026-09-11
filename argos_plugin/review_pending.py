#!/usr/bin/env python3
"""Review pending memory proposals through the shared service.

This is safe to run with Hermes stopped or running because the service owns the
canonical database. It never promotes a proposal to the user-confirmed class
automatically; low-risk approvals become ``reviewed_approved``, which
materializes at the capped grounding tier (#429, materialize-await-lift) and
still awaits explicit user confirmation to LIFT the tier.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__:
    from .reviewer import review_candidate_with_llm
    from .reviewer import set_external_policy
    from .service_client import SharedMemoryStore
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from reviewer import review_candidate_with_llm
    from reviewer import set_external_policy
    from service_client import SharedMemoryStore


_DECISION_MAP = {
    "approve": "reviewed_approved",
    "reject": "rejected",
    "quarantine": "quarantined",
    "pending_user_confirmation": "pending_user_confirmation",
}


def _sync_policy_from_config(home: Path) -> None:
    """Mirror the hybrid_memory.json external-source policy into the reviewer."""
    enabled = True
    cfg_path = home / "hybrid_memory.json"
    try:
        if cfg_path.exists():
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            enabled = str(cfg.get("external_sources_require_confirmation", "true")).lower() in (
                "true", "1", "yes"
            )
    except Exception:
        pass
    set_external_policy(enabled)


def review_pending(home: Path, limit: int) -> dict[str, int]:
    _sync_policy_from_config(home)
    store = SharedMemoryStore(home, user_id="default_user", embedder=None)
    counts: dict[str, int] = {}
    try:
        candidates = store.list_candidates(status="pending", limit=limit)
        for candidate in candidates:
            result = review_candidate_with_llm(candidate)
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
                # #423: unattended engine — the unprivileged auto_review
                # class must be explicit so the storage boundary never
                # mistakes a CLI review for a human confirmation.
                review_source="auto_review",
            )
            counts[status] = counts.get(status, 0) + 1
    finally:
        store.close()
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home", type=Path, required=True, help="Hermes home directory")
    parser.add_argument("--limit", type=int, default=25, help="Maximum proposals to review")
    args = parser.parse_args()
    counts = review_pending(args.home, max(1, min(args.limit, 500)))
    print("Reviewed proposals:", sum(counts.values()))
    for status, count in sorted(counts.items()):
        print(f"  {status}: {count}")
    print("No proposal was promoted automatically.")


if __name__ == "__main__":
    main()

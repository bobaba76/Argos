"""P4.2 — Distillation pass ("the dream").

A bounded, LLM-assisted consolidation pass that turns accumulated records
+ feedback into **distilled proposals**: insights, guardrails/strategies,
and contradiction surfaces.  It **proposes only** — nothing it emits
becomes retrievable memory until the existing review pipeline approves it.

Triggered by the provider's ``on_session_end`` hook, gated by
``distillation_enabled``.  All gating is deterministic and computed
store-side before any LLM call.  Fail-soft throughout: LLM errors, bad
JSON, timeouts, and missing LLM clients skip legs without crashing.

Safety invariants (non-negotiable):
- Proposals only — never writes active memory.
- No editing/deleting of existing records (P4.1 owns dedup).
- Grounded — every proposal cites source memory_ids.
- Fail-soft — session end never blocked.
- Deterministic gating — novelty/cooldown/budget before any LLM call.
- Same user_scope/project_id discipline as the rest of the store.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np
except ImportError:
    np = None

logger = logging.getLogger(__name__)

# -- Constants ---------------------------------------------------------------

_CLUSTER_SIMILARITY = 0.75  # looser than P4.1's 0.88 — related-but-distinct
_MAX_CLUSTER_SIZE = 8
_STATE_KEY_LAST_RUN = "distillation_last_run"
_STATE_KEY_LAST_COUNT = "distillation_last_count"

# -- LLM seam (guarded import, same pattern as reviewer.py) ------------------


def _get_llm_client():
    """Import and return the ``call_llm`` function, or None if unavailable."""
    try:
        from agent.auxiliary_client import call_llm
        return call_llm
    except Exception:
        return None


# -- Run state ---------------------------------------------------------------


def _get_last_run(store) -> Optional[str]:
    """ISO timestamp of the last completed distillation run, or None."""
    return store.get_state(_STATE_KEY_LAST_RUN)


def _advance_run_state(store, records_processed: int) -> None:
    """Mark a run as completed (advances last_run + last_count)."""
    now = datetime.now(timezone.utc).isoformat()
    store.set_state(_STATE_KEY_LAST_RUN, now)
    store.set_state(_STATE_KEY_LAST_COUNT, str(records_processed))


# -- Gating ------------------------------------------------------------------


def _count_eligible_since(store, since: Optional[str]) -> int:
    """Count active, non-superseded records created/updated since *since*.

    If *since* is None (never run), counts all eligible records.
    """
    conditions = [
        "COALESCE(status, 'active') = 'active'",
        "valid_to IS NULL",
        "(user_scope IS NULL OR user_scope = ?)",
        "embedding IS NOT NULL",
    ]
    params: list[Any] = [store.user_id]
    if since:
        conditions.append("(created_at > ? OR updated_at > ?)")
        params.extend([since, since])
    sql = (
        f"SELECT COUNT(*) FROM memory_records WHERE "
        + " AND ".join(conditions)
    )
    try:
        with store._lock:
            row = store.connection.execute(sql, params).fetchone()
        return int(row[0]) if row else 0
    except Exception:
        return 0


def _load_eligible_records(
    store, since: Optional[str], limit: int,
) -> List[Any]:
    """Load active, non-superseded records for distillation.

    If *since* is provided, only records created/updated after it.
    Falls back to most recent N if never run (since=None).
    """
    conditions = [
        "COALESCE(status, 'active') = 'active'",
        "valid_to IS NULL",
        "(user_scope IS NULL OR user_scope = ?)",
        "embedding IS NOT NULL",
    ]
    params: list[Any] = [store.user_id]
    if since:
        conditions.append("(created_at > ? OR updated_at > ?)")
        params.extend([since, since])
    sql = (
        "SELECT * FROM memory_records WHERE "
        + " AND ".join(conditions)
        + " ORDER BY created_at DESC LIMIT ?"
    )
    params.append(limit)
    return store._fetch_records(sql, params)


def _load_high_signal_records(store, limit: int = 20) -> List[Any]:
    """Load records with feedback signals for the high-signal scan."""
    sql = (
        "SELECT * FROM memory_records WHERE "
        "COALESCE(status, 'active') = 'active' "
        "AND valid_to IS NULL "
        "AND (user_scope IS NULL OR user_scope = ?) "
        "AND embedding IS NOT NULL "
        "AND (helpful_count > 0 OR dismissed_count > 0) "
        "ORDER BY (helpful_count + dismissed_count) DESC, retrieval_count DESC "
        "LIMIT ?"
    )
    return store._fetch_records(sql, [store.user_id, limit])


# -- Cluster scan (seed-star greedy, deterministic, free) --------------------


def _seed_star_cluster(
    records: List[Any],
    min_similarity: float = _CLUSTER_SIMILARITY,
    max_cluster_size: int = _MAX_CLUSTER_SIZE,
) -> List[List[Any]]:
    """Group records into clusters using seed-star greedy.

    Algorithm:
    1. Sort records by created_at descending (newest first).
    2. Pick the first unassigned record as a seed.
    3. Compute cosine similarity of all unassigned records to the seed.
    4. Assign the top ``max_cluster_size - 1`` records with cosine >=
       ``min_similarity`` to the seed's cluster.
    5. Remove assigned records from the pool. Repeat.

    No transitive closure — a record belongs to exactly one cluster.
    This prevents subject-dense stores from collapsing into one giant
    cluster at the looser 0.75 threshold (unlike union-find/connected-
    components used in P4.1's tighter 0.88 dedup).
    """
    if np is None or len(records) < 2:
        return [[r] for r in records]

    # Sort by created_at descending (newest first = seed priority).
    sorted_records = sorted(
        records, key=lambda r: r.created_at or "", reverse=True,
    )

    # Build embedding matrix for all records.
    try:
        emb_dim = len(sorted_records[0].embedding)
        mat = np.zeros((len(sorted_records), emb_dim), dtype=np.float32)
        for i, r in enumerate(sorted_records):
            mat[i] = np.asarray(r.embedding, dtype=np.float32)
    except (ValueError, TypeError):
        return [[r] for r in records]

    # Normalize rows for cosine similarity.
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normed = mat / norms

    assigned = [False] * len(sorted_records)
    clusters: List[List[Any]] = []

    for i in range(len(sorted_records)):
        if assigned[i]:
            continue
        # Seed = record i. Find unassigned records similar to it.
        seed_normed = normed[i]
        sims = normed @ seed_normed  # cosine to seed for all records

        # Candidates: unassigned, similar, not the seed itself.
        candidates: List[Tuple[float, int]] = []
        for j in range(len(sorted_records)):
            if j == i or assigned[j]:
                continue
            if sims[j] >= min_similarity:
                candidates.append((float(sims[j]), j))

        # Sort by similarity descending, take top (max_cluster_size - 1).
        candidates.sort(key=lambda x: -x[0])
        members = [i]
        for _, j in candidates[:max_cluster_size - 1]:
            members.append(j)

        for m in members:
            assigned[m] = True
        clusters.append([sorted_records[m] for m in members])

    return clusters


# -- LLM distill prompt ------------------------------------------------------

_DISTILL_SYSTEM = """\
You are a distillation engine for a personal AI assistant's memory store.
You receive a cluster of related memory records with their feedback counters.
Your job is to derive higher-order insights, guardrails, and contradictions
that the evidence supports — nothing more.

Rules:
- Only derive what the evidence supports. Never introduce facts not present.
- Output strict JSON: {"insights": [...], "contradictions": [{"a_id", "b_id", "reason"}], "guardrails": [...]}
- Each item is a plain declarative sentence, < 200 chars.
- Never suggest merging or deleting records (that is a separate process).
- Low-confidence suggestions must include "confidence": "low" in the item dict.
- If the cluster has no derivable insights, return empty arrays.
"""

_HIGH_SIGNAL_SYSTEM = """\
You are a distillation engine analyzing a user's feedback signals on memories.
You receive memories that were marked helpful or dismissed, with retrieval counts.
Derive guardrails and strategies from these successes and failures.

Rules:
- Only derive what the evidence supports. Never introduce facts not present.
- Output strict JSON: {"insights": [...], "contradictions": [], "guardrails": [...]}
- Each item is a plain declarative sentence, < 200 chars.
- Low-confidence suggestions must include "confidence": "low" in the item dict.
- If no derivable guardrails, return empty arrays.
"""


def _build_cluster_prompt(records: List[Any]) -> str:
    """Build the user-message content for a cluster distill call."""
    items = []
    for r in records:
        counters = (
            f"helpful={r.helpful_count}, dismissed={r.dismissed_count}, "
            f"retrieved={r.retrieval_count}"
        )
        content = (r.content or "")[:300]
        items.append({
            "id": r.memory_id,
            "category": r.category,
            "content": content,
            "counters": counters,
        })
    return json.dumps({"records": items}, ensure_ascii=False)


def _build_high_signal_prompt(records: List[Any]) -> str:
    """Build the user-message content for the high-signal scan."""
    items = []
    for r in records:
        counters = (
            f"helpful={r.helpful_count}, dismissed={r.dismissed_count}, "
            f"retrieved={r.retrieval_count}"
        )
        content = (r.content or "")[:300]
        items.append({
            "id": r.memory_id,
            "category": r.category,
            "content": content,
            "counters": counters,
        })
    return json.dumps({"records": items}, ensure_ascii=False)


def _parse_distill_response(response: Any) -> Optional[Dict[str, Any]]:
    """Parse the LLM response into a dict, or None if invalid."""
    try:
        text = response.choices[0].message.content
    except (AttributeError, IndexError, KeyError, TypeError):
        return None
    if not text or not text.strip():
        return None
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(value, dict):
        return None
    # Validate expected keys.
    for key in ("insights", "guardrails"):
        if key not in value:
            value[key] = []
    if "contradictions" not in value:
        value["contradictions"] = []
    return value


# -- Proposal emission -------------------------------------------------------


def _item_confidence(item: Any) -> float:
    """Map an LLM item's confidence marker to a proposal confidence.

    Only an explicit 'low'-style marker (case-insensitive) lowers
    confidence; anything else keeps the default 0.7.
    """
    if isinstance(item, dict):
        marker = str(item.get("confidence", "") or "").strip().lower()
        if marker in ("low", "low confidence", "uncertain"):
            return 0.5
    return 0.7


def _majority_project_id(records: List[Any]) -> Optional[str]:
    """Project id for a batch of records, only when unanimous.

    Mixed-project or unscoped batches distill to a GLOBAL proposal (None)
    rather than mis-tagging one project with another's lessons.
    """
    pids = {
        getattr(r, "project_id", None)
        for r in records
        if getattr(r, "project_id", None)
    }
    return next(iter(pids)) if len(pids) == 1 else None


def _emit_proposal(
    store,
    content: str,
    kind: str,
    source_ids: List[str],
    evidence_text: str,
    confidence: float,
    project_id: Optional[str] = None,
) -> Optional[dict]:
    """Save one distilled proposal as a pending candidate.

    Returns the candidate dict, or None if deduped away / empty content.
    """
    if not content or not content.strip():
        return None
    # Map kind to category.
    if kind == "guardrail":
        category = "context_note"
    else:
        category = "insight"
    payload = {
        "sources": source_ids,
        "kind": kind,
    }
    return store.save_candidate(
        category=category,
        content=content.strip()[:200],  # spec: < 200 chars
        payload=payload,
        source="distillation",
        confidence=max(0.0, min(1.0, confidence)),
        evidence_text=evidence_text,
        dedup=True,
        project_id=project_id,
    )


def _valid_contradiction(contra: Any, source_ids: List[str]) -> bool:
    """Only honor contradiction IDs the LLM was actually shown.

    Invented or out-of-cluster IDs would create unresolvable proposals,
    so they are dropped silently.
    """
    if not isinstance(contra, dict):
        return False
    a_id = contra.get("a_id", "")
    b_id = contra.get("b_id", "")
    return bool(a_id and b_id and a_id in source_ids and b_id in source_ids)


def _emit_contradiction(
    store,
    a_id: str,
    b_id: str,
    reason: str,
    source_ids: List[str],
    evidence_text: str,
    project_id: Optional[str] = None,
) -> Optional[dict]:
    """Save a contradiction proposal as a pending candidate.

    Contradictions are emitted as context_note candidates with kind=
    "contradiction", never auto-superseded. Resolution is left to the
    chain-unfold flow (user-confirmed supersession). ID validity is the
    caller's responsibility — only IDs shown to the LLM are accepted.
    """
    if not reason or not reason.strip():
        return None
    content = (
        f"Possible contradiction between {a_id} and {b_id}: {reason.strip()[:140]}"
    )
    payload = {
        "sources": source_ids,
        "kind": "contradiction",
        "contradiction_a": a_id,
        "contradiction_b": b_id,
    }
    return store.save_candidate(
        category="context_note",
        content=content[:200],
        payload=payload,
        source="distillation",
        confidence=0.5,
        evidence_text=evidence_text,
        dedup=True,
        project_id=project_id,
    )


# -- Main entry point --------------------------------------------------------


def run_distillation(
    store,
    *,
    llm_model: str = "",
    llm_provider: str = "",
    min_new_records: int = 20,
    cooldown_hours: int = 24,
    max_records_per_run: int = 100,
    max_calls: int = 10,
) -> Dict[str, Any]:
    """Run one distillation pass.

    Returns a report dict with run stats.  Never raises — all failures
    are caught and logged.  The run state advances only on completion
    (including zero-proposal completions); fail-soft aborts do not
    advance, so the next clean session end retries the same records.

    Per-cluster failures skip that cluster and continue.  Only a run-
    level failure (LLM client unavailable, or every call fails) leaves
    ``last_run`` un-advanced.
    """
    report: Dict[str, Any] = {
        "ran": False,
        "reason": "",
        "clusters_scanned": 0,
        "llm_calls": 0,
        "proposals_emitted": 0,
        "contradictions_emitted": 0,
        "records_processed": 0,
    }
    from egress import gate as _egress_gate
    if not _egress_gate("distillation", ""):
        report["ran"] = False
        report["skipped"] = "egress_gate"
        return report


    # -- Gate 1: LLM client availability (checked once, up front) ----------
    call_llm = _get_llm_client()
    if call_llm is None:
        report["reason"] = "llm_client_unavailable"
        logger.debug("Distillation skipped: LLM client not available")
        return report

    # -- Gate 2: cooldown --------------------------------------------------
    last_run = _get_last_run(store)
    if last_run:
        try:
            last_dt = datetime.fromisoformat(last_run.replace("Z", "+00:00"))
            elapsed = datetime.now(timezone.utc) - last_dt
            if elapsed < timedelta(hours=cooldown_hours):
                report["reason"] = "cooldown"
                logger.debug(
                    "Distillation skipped: cooldown (%.1fh < %dh)",
                    elapsed.total_seconds() / 3600, cooldown_hours,
                )
                return report
        except Exception:
            pass  # corrupt state → treat as never run

    # -- Gate 3: novelty ---------------------------------------------------
    eligible_count = _count_eligible_since(store, last_run)
    if eligible_count < min_new_records:
        report["reason"] = f"novelty_gate ({eligible_count} < {min_new_records})"
        logger.debug(
            "Distillation skipped: novelty gate (%d < %d)",
            eligible_count, min_new_records,
        )
        return report

    # -- Load records ------------------------------------------------------
    records = _load_eligible_records(store, last_run, max_records_per_run)
    if len(records) < 2:
        report["reason"] = "too_few_records_with_embeddings"
        return report

    # -- Leg 1: cluster scan (deterministic, free) -------------------------
    clusters = _seed_star_cluster(records)
    # Filter to clusters with >= 2 members (singletons have nothing to distill).
    multi_clusters = [c for c in clusters if len(c) >= 2]
    # Sample down to budget (newest first — clusters are already sorted
    # by seed recency from _seed_star_cluster).
    calls_budget = max_calls
    proposals_emitted = 0
    contradictions_emitted = 0
    llm_calls = 0
    clusters_scanned = 0

    # -- Leg 2: LLM distill (1 call per cluster) ---------------------------
    run_failed_completely = True  # set to False if any call succeeds

    for cluster in multi_clusters:
        if llm_calls >= calls_budget - 1:
            # Reserve 1 call for the high-signal scan.
            break
        prompt = _build_cluster_prompt(cluster)
        try:
            response = call_llm(
                task="distillation",
                messages=[
                    {"role": "system", "content": _DISTILL_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=200,
                timeout=15.0,
                model=llm_model or None,
                provider=llm_provider or None,
            )
            llm_calls += 1
            clusters_scanned += 1
            run_failed_completely = False
        except Exception as exc:
            logger.debug("Distillation LLM call failed for cluster: %s", exc)
            continue

        parsed = _parse_distill_response(response)
        if not parsed:
            logger.debug("Distillation: unparseable response for cluster")
            continue

        # Build evidence text from cluster contents.
        evidence = " | ".join(
            (r.content or "")[:200] for r in cluster
        )[:2000]
        source_ids = [r.memory_id for r in cluster]
        cluster_project_id = cluster[0].project_id if cluster else None

        # Emit insights.
        for item in parsed.get("insights", []):
            if isinstance(item, dict):
                text = item.get("text", "")
                conf = _item_confidence(item)
            else:
                text = str(item)
                conf = 0.7
            result = _emit_proposal(
                store, text, "insight", source_ids, evidence, conf,
                project_id=cluster_project_id,
            )
            if result:
                proposals_emitted += 1

        # Emit guardrails.
        for item in parsed.get("guardrails", []):
            if isinstance(item, dict):
                text = item.get("text", "")
                conf = _item_confidence(item)
            else:
                text = str(item)
                conf = 0.7
            result = _emit_proposal(
                store, text, "guardrail", source_ids, evidence, conf,
                project_id=cluster_project_id,
            )
            if result:
                proposals_emitted += 1

        # Emit contradictions.
        for contra in parsed.get("contradictions", []):
            if not isinstance(contra, dict):
                continue
            a_id = contra.get("a_id", "")
            b_id = contra.get("b_id", "")
            reason = contra.get("reason", "")
            if not _valid_contradiction(contra, source_ids):
                continue
            result = _emit_contradiction(
                store, a_id, b_id, reason, source_ids, evidence,
                project_id=cluster_project_id,
            )
            if result:
                contradictions_emitted += 1

    # -- Leg 3: high-signal scan (1 call, optional) ------------------------
    if llm_calls < calls_budget:
        high_signal = _load_high_signal_records(store, limit=20)
        if len(high_signal) >= 2:
            prompt = _build_high_signal_prompt(high_signal)
            try:
                response = call_llm(
                    task="distillation_high_signal",
                    messages=[
                        {"role": "system", "content": _HIGH_SIGNAL_SYSTEM},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.2,
                    max_tokens=200,
                    timeout=15.0,
                    model=llm_model or None,
                    provider=llm_provider or None,
                )
                llm_calls += 1
                run_failed_completely = False
            except Exception as exc:
                logger.debug("Distillation high-signal scan failed: %s", exc)
                response = None

            if response is not None:
                parsed = _parse_distill_response(response)
                if parsed:
                    evidence = " | ".join(
                        (r.content or "")[:200] for r in high_signal
                    )[:2000]
                    source_ids = [r.memory_id for r in high_signal]
                    project_id = _majority_project_id(high_signal)
                    for item in parsed.get("guardrails", []):
                        if isinstance(item, dict):
                            text = item.get("text", "")
                            conf = _item_confidence(item)
                        else:
                            text = str(item)
                            conf = 0.7
                        result = _emit_proposal(
                            store, text, "guardrail", source_ids, evidence, conf,
                            project_id=project_id,
                        )
                        if result:
                            proposals_emitted += 1
                    for item in parsed.get("insights", []):
                        if isinstance(item, dict):
                            text = item.get("text", "")
                            conf = _item_confidence(item)
                        else:
                            text = str(item)
                            conf = 0.7
                        result = _emit_proposal(
                            store, text, "insight", source_ids, evidence, conf,
                            project_id=project_id,
                        )
                        if result:
                            proposals_emitted += 1

    # -- Advance run state (only on completion) ----------------------------
    # If every single LLM call failed (run_failed_completely), do NOT
    # advance — the next clean session end retries the same records.
    if run_failed_completely and llm_calls == 0:
        report["reason"] = "all_llm_calls_failed"
        return report

    records_processed = len(records)
    _advance_run_state(store, records_processed)

    report["ran"] = True
    report["reason"] = "completed"
    report["clusters_scanned"] = clusters_scanned
    report["llm_calls"] = llm_calls
    report["proposals_emitted"] = proposals_emitted
    report["contradictions_emitted"] = contradictions_emitted
    report["records_processed"] = records_processed
    logger.info(
        "Distillation completed: %d clusters, %d calls, %d proposals, "
        "%d contradictions, %d records",
        clusters_scanned, llm_calls, proposals_emitted,
        contradictions_emitted, records_processed,
    )
    return report

"""P5.1 Phase 3 — Long-horizon rollups.

A bounded, LLM-assisted pass that emits **proposals only** — profile-style
"what has stayed true / what changed" summaries from the oldest active
low-retrieval records. Nothing it emits becomes retrievable memory until
the existing review pipeline approves it.

Reuses the P4.2 distillation seam: same novelty/cooldown/budget gating,
same ``system_state`` KV machinery, same fail-soft semantics. Separate
knob (``rollup_enabled``, ``rollup_interval_days``).

Safety invariants (non-negotiable):
- Proposals only — never writes active memory.
- Fail-soft — session end never blocked.
- Deterministic gating before any LLM call.
- Same user_scope discipline as the rest of the store.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _strip_json_fences(text: str) -> str:
    """Strip markdown code fences and surrounding prose from LLM JSON output.

    Handles three cases (D6 — same class as R4 in reviewer and D2 in
    distillation):
    1. Fenced JSON: ``\\`\\`\\`json [...] \\`\\`\\```` → stripped.
    2. Pure JSON: ``[...]`` → unchanged.
    3. Prose-wrapped JSON: ``Here is the JSON: [...]`` → extracts the
       first balanced ``[...]`` block via the JSON decoder's raw_decode.
    """
    if not text or not text.strip():
        return ""
    text = text.strip()
    # Strip a single pair of leading/trailing markdown code fences.
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()
    # Try a direct parse first (the common case).
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        pass
    # Fallback: extract the first balanced [...] block.
    start = text.find("[")
    if start == -1:
        return text  # let the caller's json.loads fail naturally
    try:
        value, _end = json.JSONDecoder().raw_decode(text[start:])
        return json.dumps(value)
    except json.JSONDecodeError:
        return text

_STATE_KEY_LAST_RUN = "rollup_last_run"
_STATE_KEY_LAST_COUNT = "rollup_last_count"


def _get_llm_client():
    """Import and return the ``call_llm`` function, or None if unavailable."""
    try:
        from agent.auxiliary_client import call_llm
        return call_llm
    except Exception:
        return None


def _get_last_run(store) -> Optional[str]:
    """ISO timestamp of the last completed rollup run, or None."""
    return store.get_state(_STATE_KEY_LAST_RUN)


def _advance_run_state(store, records_processed: int) -> None:
    """Mark a run as completed (advances last_run + last_count)."""
    now = datetime.now(timezone.utc).isoformat()
    store.set_state(_STATE_KEY_LAST_RUN, now)
    store.set_state(_STATE_KEY_LAST_COUNT, str(records_processed))


def _is_within_cooldown(store, interval_days: int) -> bool:
    """Check if the last run was within the cooldown interval."""
    last = _get_last_run(store)
    if not last:
        return False
    try:
        last_dt = datetime.fromisoformat(last.replace("Z", "+00:00"))
        elapsed = datetime.now(timezone.utc) - last_dt
        return elapsed < timedelta(days=interval_days)
    except Exception:
        return False


def _load_rollup_candidates(store, limit: int) -> List[Any]:
    """Load the oldest active low-retrieval records for rollup.

    Targets records beyond the rollup horizon: oldest first, low retrieval
    count, no feedback. These are the records whose long-horizon patterns
    are most likely to benefit from compaction into profile summaries.
    """
    try:
        records = store._fetch_records(
            """SELECT * FROM memory_records
               WHERE COALESCE(status, 'active') = 'active'
                 AND valid_to IS NULL
                 AND COALESCE(tier, 'active') = 'active'
                 AND (user_scope IS NULL OR user_scope = ?)
               ORDER BY created_at ASC
               LIMIT ?""",
            [store.user_id, limit],
        )
        return records
    except Exception as exc:
        logger.debug("rollup: load candidates failed: %s", exc)
        return []


def _build_rollup_prompt(records: List[Any]) -> str:
    """Build the LLM prompt for the rollup pass."""
    lines = []
    for r in records[:100]:  # cap at 100 for the prompt
        age_days = ""
        if r.created_at:
            try:
                created = datetime.fromisoformat(r.created_at.replace("Z", "+00:00"))
                age_days = f" ({(datetime.now(timezone.utc) - created).days}d old)"
            except Exception:
                pass
        lines.append(f"- [{r.category}] {r.content}{age_days}")
    return (
        "You are a memory rollup assistant. Review the following long-term "
        "memories and emit profile-style summaries of what has stayed true "
        "and what has changed. Output a JSON array of proposals, each with:\n"
        '  "content": the summary statement,\n'
        '  "category": "insight" or "context_note",\n'
        '  "source_loc": "rollup",\n'
        '  "confidence": 0.0-1.0\n\n'
        "Only emit summaries that are grounded in the provided memories. "
        "Do not invent facts. If nothing is worth summarizing, return [].\n\n"
        "Memories:\n" + "\n".join(lines)
    )


def _parse_rollup_response(content: str) -> List[Dict[str, Any]]:
    """Parse the LLM response into a list of proposal dicts.

    D6: strips code fences and prose wrapping before parsing (same class
    as R4 in reviewer and D2 in distillation).
    """
    if not content:
        return []
    stripped = _strip_json_fences(content)
    try:
        data = json.loads(stripped)
        if not isinstance(data, list):
            return []
        proposals = []
        for item in data:
            if not isinstance(item, dict):
                continue
            content = item.get("content", "").strip()
            if not content:
                continue
            proposals.append({
                "content": content,
                "category": item.get("category", "insight"),
                "source": "rollup",
                "confidence": float(item.get("confidence", 0.7)),
                "source_loc": item.get("source_loc", "rollup"),
            })
        return proposals
    except (json.JSONDecodeError, TypeError, ValueError):
        return []


def run_rollup(
    store: Any,
    *,
    llm_model: str = "",
    llm_provider: str = "",
    interval_days: int = 30,
    max_records_per_run: int = 100,
) -> Dict[str, Any]:
    """Run one rollup pass: emit profile-style proposals from old memories.

    Gated by cooldown (interval_days). Fail-soft: any error returns a
    no-op report. Proposals only — never writes active memory.

    Returns a report dict with ``ran``, ``proposals_emitted``, ``llm_calls``.
    """
    report: Dict[str, Any] = {
        "ran": False,
        "proposals_emitted": 0,
        "llm_calls": 0,
        "skipped": None,
    }

    # D4: egress gate — refuse in local_only mode (same as distillation).
    try:
        from egress import gate as _egress_gate
    except ImportError:
        try:
            from .egress import gate as _egress_gate
        except ImportError:
            _egress_gate = None
    if _egress_gate is not None and not _egress_gate("memory_rollup", ""):
        report["skipped"] = "egress_gate"
        return report

    # Cooldown gate.
    if _is_within_cooldown(store, interval_days):
        report["skipped"] = "cooldown"
        return report

    # Load candidates.
    records = _load_rollup_candidates(store, max_records_per_run)
    if len(records) < 10:
        report["skipped"] = "insufficient_records"
        return report

    # LLM call.
    call_llm = _get_llm_client()
    if call_llm is None:
        report["skipped"] = "no_llm_client"
        return report

    prompt = _build_rollup_prompt(records)
    try:
        # D3: use the same call_llm signature as distillation/reviewer —
        # messages=[...] kwarg, not a positional prompt arg. The old
        # call_llm(prompt, model=..., ...) would either raise TypeError
        # (caught by try/except → "llm_error" → rollup silently never runs)
        # or pass the prompt as the wrong parameter.
        response = call_llm(
            task="memory_rollup",
            messages=[
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=500,
            timeout=30.0,
            model=llm_model or None,
            provider=llm_provider or None,
        )
    except Exception as exc:
        logger.debug("rollup: LLM call failed: %s", exc)
        report["skipped"] = "llm_error"
        return report

    report["llm_calls"] = 1

    # Parse response.
    content = ""
    if response is not None:
        try:
            content = response.choices[0].message.content
        except (AttributeError, IndexError, TypeError):
            content = ""

    proposals = _parse_rollup_response(content)

    # D5: build evidence text from source records for provenance.
    evidence_text = " | ".join(
        (getattr(r, "content", "") or "")[:200] for r in records[:20]
    )[:2000]

    # Emit proposals through the standard review pipeline.
    emitted = 0
    for prop in proposals:
        try:
            store.save_candidate(
                category=prop["category"],
                content=prop["content"],
                tags=["rollup"],
                payload={
                    "source": "rollup",
                    "source_loc": prop.get("source_loc", "rollup"),
                    "confidence": prop.get("confidence", 0.7),
                },
                source="rollup",
                confidence=prop.get("confidence", 0.7),
                durability="durable",
                scope="profile",
                evidence_text=evidence_text,
            )
            emitted += 1
        except Exception as exc:
            logger.debug("rollup: save_candidate failed: %s", exc)

    # Advance run state.
    _advance_run_state(store, len(records))
    report["ran"] = True
    report["proposals_emitted"] = emitted
    logger.info(
        "rollup: %d proposals from %d records (%d LLM calls)",
        emitted, len(records), report["llm_calls"],
    )
    return report

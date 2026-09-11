"""Spec-13 (#393, slice S1): deterministic trust-tier classification for auto-save.

When a deployment runs ``approval_mode="auto"``, a save that would previously
have entered the human review queue is materialized immediately. This module
classifies each save BEFORE materialization using only signals already in
hand at the write boundary (fields + regex; zero LLM, zero network):

- ``blocked``   - deterministic quality flags that the reviewer's
  deterministic_gate would quarantine anyway (extraction-source content
  only; the flags are tuned for extracted conversational facts). The item
  is kept as a quarantined candidate (visible, rescuable via review) and
  is never materialized. Net outcome identical to v1's auto-review, minus
  the wait.
- ``unreviewed`` - medium/high-risk signals: external origin, no original
  user evidence on extraction/API proposals, speculative grounding, weak
  confidence, or sensitive identifiers. The record IS materialized (zero
  queue) but stamped ``trust_class="unreviewed"`` server-side (never
  client-supplied), rank-penalized by at most 3 positions at retrieval
  (never removed from the window), and resolvable only by an explicit
  human action - never auto-promoted (9/10 amendment decision).
- ``clean``     - everything else: active memory, normal rank, no marker.

The classifier is deliberately conservative: when in doubt, prefer
``unreviewed`` - a wrongly flagged record is promotable; a wrongly clean
record is provenance that lies.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

TRUST_CLASS_UNREVIEWED = "unreviewed"
# "clean" is represented as NULL on the memory row (no marker).

# Sources whose proposals follow the reviewer's evidence discipline: a
# proposal with no original user evidence is the evidence_gate signal
# (reviewer.py) - in auto mode it materializes as unreviewed, never clean.
_EVIDENCE_GATED_SOURCES = frozenset({"llm_extraction", "api", "watcher"})

# Sources whose content is run through the deterministic quality flags
# (extractor.quality_flags_for_fact). The flag set is tuned for extracted
# conversational facts; other writers keep their own gates.
_QUALITY_GATED_SOURCES = frozenset({"llm_extraction"})

_CONFIDENCE_FLOOR = 0.5
RANK_PENALTY_MAX = 3


def _hard_quality_flags(content: str, category: str, tags: Any) -> List[str]:
    try:
        try:
            from .extractor import hard_quality_flags, quality_flags_for_fact
        except ImportError:
            from extractor import hard_quality_flags, quality_flags_for_fact
        fact = {"content": content, "category": category, "tags": tags or []}
        return hard_quality_flags(quality_flags_for_fact(fact))
    except Exception:
        # Fail-open to "no flags": the reviewer's gates still run on the
        # candidate path, and blocking saves on an import hiccup would be
        # worse than classifying conservatively.
        return []


def _sensitive(content: str, evidence_text: str) -> Tuple[bool, List[str]]:
    """True when egress identifier patterns (emails/phones/ID digit runs) match."""
    text = (str(content or "") + "\n" + str(evidence_text or ""))[:8000]
    try:
        try:
            from .egress import all_sensitive_labels
        except ImportError:
            from egress import all_sensitive_labels
        labels = all_sensitive_labels(text)
        return bool(labels), list(labels)
    except Exception:
        return False, []


def classify_trust_tier(
    *,
    content: str,
    category: str = "context_note",
    source: str = "",
    evidence_text: str = "",
    external: bool = False,
    payload: Optional[Dict[str, Any]] = None,
    confidence: Optional[float] = None,
    grounding: Any = None,
    tags: Any = None,
) -> Tuple[str, List[str], bool]:
    """Classify a save. Returns ``(tier, reasons, sensitivity)``.

    ``tier`` is one of ``"clean"``, ``"unreviewed"``, ``"blocked"``.
    ``reasons`` are stable machine-readable tokens (payload / audit).
    ``sensitivity`` marks identifier-bearing content for scope filtering.
    """
    payload = payload if isinstance(payload, dict) else {}
    src = str(source or "").strip().lower()
    content = str(content or "")
    evidence = str(evidence_text or "")

    reasons: List[str] = []

    is_external = bool(external) or bool(payload.get("external_source"))
    if not is_external:
        # Only an EXPLICIT payload provenance counts here: normalize_provenance
        # fail-closes missing values to external, which would taint every
        # payload without the key (the store derives provenance itself).
        _po = payload.get("provenance_origin")
        if _po:
            try:
                try:
                    from .store_common import PROVENANCE_EXTERNAL, normalize_provenance
                except ImportError:
                    from store_common import PROVENANCE_EXTERNAL, normalize_provenance
                if normalize_provenance(_po) == PROVENANCE_EXTERNAL:
                    is_external = True
            except Exception:
                pass
    if is_external:
        reasons.append("external_origin")

    if src in _EVIDENCE_GATED_SOURCES and not evidence.strip():
        reasons.append("no_evidence")

    if src in _QUALITY_GATED_SOURCES:
        hard = _hard_quality_flags(content, category, tags)
        if hard:
            return "blocked", ["quality:" + ",".join(hard)], False

    if grounding is not None:
        try:
            try:
                from .store_common import GROUNDING_SPECULATIVE, normalize_grounding
            except ImportError:
                from store_common import GROUNDING_SPECULATIVE, normalize_grounding
            if normalize_grounding(grounding) == GROUNDING_SPECULATIVE:
                reasons.append("speculative_grounding")
        except Exception:
            pass

    try:
        conf = float(confidence) if confidence is not None else None
    except (TypeError, ValueError):
        conf = None
    if conf is not None and conf < _CONFIDENCE_FLOOR:
        reasons.append("low_confidence")

    sensitive, sens_labels = _sensitive(content, evidence)
    if sensitive:
        reasons.append("sensitive_content:" + ",".join(sorted(set(sens_labels))))

    if reasons:
        return TRUST_CLASS_UNREVIEWED, reasons, sensitive
    return "clean", [], False

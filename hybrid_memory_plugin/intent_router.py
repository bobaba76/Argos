"""Argos intent router — P2A.

Decides which ANSWERER model this turn should use, based on lightweight
temporal / multi-hop query detection, and returns it as a ``model`` key that
the core ``pre_llm_call`` hook path honors (see agent/turn_context.py).

Design:
  * Always returns an explicit model (smart for temporal/multi-hop, default
    otherwise), so the answerer self-corrects across turns: a temporal turn
    that was routed to the smart model is switched BACK to the default model
    on the next ordinary turn.
  * The core only calls ``switch_model`` when the requested model differs from
    the current one, so non-routed turns are a cheap no-op (no client churn).
  * Failure here is never allowed to break the turn — routing is best-effort
    and every branch is wrapped.
  * Default ``router_enabled=false`` so behavior is unchanged until a user
    opts in by picking models in the UI.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# -- Lightweight temporal / multi-hop intent detection -----------------------
# These are deliberately broad: over-routing costs a little extra (a strong
# model answers an ordinary turn), under-routing loses the whole point of the
# router (a date-math question gets Flash and we forfeit the +26pp win).  When
# in doubt, route to the smart model.

_TEMPORAL_PATTERNS = (
    # Relative time / date-math triggers — the measured +26pp bucket.
    re.compile(r"\b(how long ago|how long has it been|how many days)\b", re.I),
    re.compile(r"\b(when (did|was|were|is|are|was it|did i))\b", re.I),
    re.compile(r"\b(what (year|date|month|day))\b", re.I),
    re.compile(r"\b(in \d{4}|on \d{1,2}(st|nd|rd|th)? [a-z]+ \d{4})\b", re.I),
    re.compile(r"\b(before|after|since|until|by) (then|that|last|this|next)\b", re.I),
    re.compile(r"\b(last (week|month|year|friday|monday|tuesday|wednesday|thursday))+", re.I),
    re.compile(r"\b(a (week|month|year|fortnight) (ago|before))\b", re.I),
    re.compile(r"\b(how old)\b", re.I),
    re.compile(r"\b(between \d{4} and \d{4})\b", re.I),
    re.compile(r"\b(what came (first|before)|which happened (first|before|more recently))\b", re.I),
    re.compile(r"\b(before x|before \d{4})\b", re.I),
)

# Mild multi-hop markers: query references info supplied earlier ("you said",
# "you told me") or chains two entities/relations.
_MULTI_HOP_PATTERNS = (
    re.compile(r"\b(you (said|told me|mentioned|wrote|sent))\b", re.I),
    re.compile(r"\b(what did .* (say|say about) )\b", re.I),
    re.compile(r"\b(who .* (tell|say|said))\b", re.I),
    re.compile(r"\b(about .* and .*)\b", re.I),
    re.compile(r"\b(remember|recall|bring back)\b", re.I),
    re.compile(r"\b(then .* (said|did|went))|(and then)\b", re.I),
    re.compile(r"\b(how .* (compare|compare to|different))\b", re.I),
)

# Combine only if a single token budget is exceeded or explicit multi-hop
# markers fire.  We keep combining level light on purpose: too-aggressive
# multi-hop routing would push most real conversation into the smart model and
# erase the cost win.  Temporal detection is the strong signal; multi-hop is a
# supporting signal evaluated independently.
def is_temporal_or_multihop(query: str) -> bool:
    """Return True when the query reads as temporal and/or multi-hop."""
    if not query or not query.strip():
        return False
    q = " ".join(query.split())
    for rx in _TEMPORAL_PATTERNS:
        if rx.search(q):
            return True
    multi_hits = sum(1 for rx in _MULTI_HOP_PATTERNS if rx.search(q))
    return multi_hits >= 2  # multi-hop needs two markers to be confident


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("true", "1", "yes", "on")


def route_answerer(config: Dict[str, Any], user_message: str) -> Optional[Dict[str, str]]:
    """Return ``{"model": ..., "provider": ...}`` (provider optional) or None.

    Called from ``_on_pre_llm_call``.  Returns the smart model for
    temporal/multi-hop queries and the default model otherwise (so turns
    self-correct back), or None when routing is disabled / not configured.
    """
    try:
        if not _as_bool(config.get("router_enabled"), default=False):
            return None
        smart = str(config.get("router_smart_model") or "").strip()
        default = str(config.get("router_default_model") or "").strip()
        if not smart or not default:
            logger.debug("router: enabled but missing smart/default model config")
            return None
        if is_temporal_or_multihop(user_message or ""):
            pick = smart
            provider = str(config.get("router_smart_provider") or "").strip()
        else:
            pick = default
            provider = str(config.get("router_default_provider") or "").strip()
        result: Dict[str, str] = {"model": pick}
        if provider:
            result["provider"] = provider
        return result
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("router: failed to route: %s", exc)
        return None

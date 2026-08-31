"""Ambient context: per-turn hints, pre_llm_call hook and /ilog handlers.

Extracted verbatim from __init__.py during the god-file split (behavior-
neutral: no renames, no fixes). Module-level helpers only — no mixin.
"""
from __future__ import annotations

import logging
import os
import traceback
from pathlib import Path

try:
    from .provider_core import _active_user_id, _load_config
except ImportError:  # provider_ambient.py imported as a top-level module
    from provider_core import _active_user_id, _load_config

logger = logging.getLogger(__name__)


# Cached import of route_answerer (issue #29: was re-importing every turn).
_route_answerer_fn = None

# Idempotency guard for hook registration.  ``PluginManager.register_hook``
# APPENDS to a list without dedup, and the desktop process re-runs plugin
# ``register()`` for every chat session opened since app start — so Argos
# accumulated one pre_llm_call callback per session, injecting N byte-identical
# ambient blocks per turn (observed 2026-08-23: 7 copies of time/location/
# weather on a single message; memory provider stayed ×1 because provider
# registration is keyed/replace while hooks are append-only).
_HOOKS_REGISTERED: set = set()

def _build_timestamp_hint() -> str:
    """Render the per-turn local-time line, e.g. ``Current time: Friday 2026-07-31 19:55 SAST``.

    Uses the native ``hermes_time.now()`` so the user's configured IANA timezone
    wins (``HERMES_TIMEZONE`` env → ``config.yaml`` ``timezone`` → server-local).
    Returns ``""`` on any failure so the injection path is unaffected — the
    agent simply gets no time hint that turn. Kept to one short line (~10-15
    tokens) for ambient awareness without per-turn overhead.
    """
    try:
        from hermes_time import now as _hermes_now
        _now = _hermes_now()
        return "Current time: " + _now.strftime("%A %Y-%m-%d %H:%M %Z").strip()
    except Exception:
        return ""


def _build_location_hint() -> str:
    """Render the per-turn location line, e.g. ``Location: City-X``.

    Resolves the location fresh each turn via ``hermes_location._resolve_location_name()``
    (bypassing the module-level cache) so that a mid-session
    ``hermes config set location "City-X"`` is picked up on the very next
    turn — even in a long-lived CLI session.

    Returns ``""`` when unset or on any failure. Kept to one short line
    (~5-10 tokens) for ambient awareness.
    """
    try:
        from .hermes_location import _resolve_location_name as _resolve
        _loc = _resolve().strip()
        return f"Location: {_loc}" if _loc else ""
    except Exception:
        return ""


def _build_weather_hint() -> str:
    """Render the per-turn weather line, e.g. ``Weather: 14°C, light rain``.

    Uses ``hermes_weather.get_weather()`` which geocodes the configured
    location and fetches current weather from Open-Meteo (free, no API key).
    Cached for ~20 minutes so most turns are zero-network cache hits.

    Returns ``""`` when disabled, no location, or the network call fails.
    """
    try:
        from .hermes_weather import get_weather as _get_weather
        _w = _get_weather().strip()
        return f"Weather: {_w}" if _w else ""
    except Exception:
        return ""


def _build_file_activity_hint() -> str:
    """Render the per-turn file activity line, e.g. ``Last edited: ~/project/foo.py (4 min ago)``.

    Uses ``hermes_file_activity.get_recent_files()`` which scans the
    configured working directory for recently modified files. Cached for ~5
    minutes so most turns are zero-I/O cache hits.

    Returns ``""`` when disabled, no recent files, or the scan fails.
    """
    try:
        from .hermes_file_activity import get_recent_files as _get_recent
        _r = _get_recent().strip()
        return f"Last edited: {_r}" if _r else ""
    except Exception:
        return ""


def _build_coder_directive(user_message: str = "") -> str:
    """Per-turn coder-MCP usage directive (conditional).

    The coder MCP tools are registered but deferred behind Hermes'
    ``tool_search`` bridge — the model only reaches for them when told to.
    This returns a short instruction to prefer ``mcp__coder__*`` for code
    questions whenever the turn looks code-adjacent (indexed repo names,
    code terms, or an explicit coder mention).  Returns ``""`` otherwise,
    so non-coding chats cost nothing.

    Keep the repo-name list in sync as new repos get indexed.
    """
    text = (user_message or "").lower()
    repo_signals = (
        "salesdash", "miser", "codebrain", "argos", "argos",
        "hermes-agent", "longmemeval", "documents\\github", "documents/github",
        "github\\", "github/", "coder mcp", "coder",
    )
    code_signals = (
        "traceback", "refactor", "compile", "exception", "semantic search",
        "find_symbols", "callers", "callees", "impact_analysis", "reindex",
        "symbol", "kuzu", "duckdb", "lancedb", "embedding",
    )
    if not (any(s in text for s in repo_signals) or any(s in text for s in code_signals)):
        return ""
    return (
        "Tooling note: for code questions about indexed repos (SalesDash, Coder, "
        "Stock, Miser, Hermes, hermes-agent) prefer the coder MCP tools — "
        "mcp__coder__semantic_code_search, mcp__coder__find_symbols, "
        "mcp__coder__get_callers_and_callees, mcp__coder__impact_analysis, "
        "mcp__coder__unified_context — loaded via tool_search when not active, "
        "before falling back to grep/search_files. Coder only covers repos it "
        "has already indexed."
    )


def _on_pre_llm_call(**kwargs) -> dict:
    """``pre_llm_call`` hook — build ambient context hints synchronously.

    Called by Hermes on every turn before the LLM call.  Returns a dict with
    a ``context`` key whose value is injected into the API copy of the user
    message (via the native ``plugin_user_context`` path).  Each hint is
    built independently so a failure in one (e.g. weather network timeout)
    never suppresses the others.

    This replaces the former core-source patch to ``turn_context.py`` /
    ``conversation_loop.py`` that added 4 parameters to ``compose_user_api_content``
    and 4 fields to ``TurnContext``.  The native hook delivers the same
    per-turn, byte-stable injection with zero core changes.
    """
    parts: list[str] = []
    for builder in (
        _build_timestamp_hint,
        _build_location_hint,
        _build_weather_hint,
        _build_file_activity_hint,
    ):
        try:
            line = builder()
            if line:
                parts.append(line)
        except Exception:
            pass  # never let one hint break the others
    try:
        coder_line = _build_coder_directive(kwargs.get("user_message", ""))
        if coder_line:
            parts.append(coder_line)
    except Exception:
        pass  # directive failure must never suppress ambient hints
    if not parts:
        result: dict = {}
    else:
        result = {"context": "\n".join(parts)}

    # P2A intent routing: optionally return {"model", "provider"} for the
        # answerer.  The core pre_llm_call path (agent/turn_context.py) honors a
        # "model" key and calls switch_model() when it differs from the current
        # answerer.  Since 2026-08-23 route_answerer returns a pick ONLY for
        # genuine smart routes (never the default model), so the hook can never
        # stomp a session's own model choice; with router_subcall_enabled smart
        # routes don't switch at all (hint-only).
    #
    # When router_subcall_enabled is set, genuine temporal/multi-hop queries
    # do NOT switch the whole (expensive ~124k-token) turn to the smart model.
    # Instead Argos makes ONE cheap smart-model sub-call on a TRIMMED context
    # (question + a handful of dated memories) and injects the short answer
    # back as a hint — staying on the cheap Flash answerer throughout.
    try:
        # Cache the import (issue #29: was re-importing every turn).
        global _route_answerer_fn
        if _route_answerer_fn is None:
            from .intent_router import route_answerer as _ra
            _route_answerer_fn = _ra
        _cfg = _load_config()
        _q = (kwargs.get("user_message") or "").strip()
        _route = _route_answerer_fn(_cfg, _q)
        if _route:
            _smart = str(_cfg.get("router_smart_model") or "").strip()
            _subcall_on = _as_flag(cfg_value=_cfg.get("router_subcall_enabled"))
            _is_smart = bool(_smart) and _route.get("model") == _smart
            if _is_smart and _subcall_on:
                _hint = _build_temporal_hint(_q)
                if _hint:
                    parts.append(
                        "[Temporal fact hint (smart model, trimmed context)]\n" + _hint
                    )
                    result["context"] = "\n\n".join(parts)
                # deliberately do NOT set result["model"] — stay on cheap Flash;
                # the hint carries the temporal answer.
            else:
                result["model"] = _route["model"]
                if _route.get("provider"):
                    result["provider"] = _route["provider"]
    except Exception:
        pass  # routing failures must never break ambient context
    return result


def _as_flag(cfg_value) -> bool:
    """Small bool parse for plugin config strings ('true'/'1'/'yes'/'on')."""
    if cfg_value is None:
        return False
    if isinstance(cfg_value, bool):
        return cfg_value
    return str(cfg_value).strip().lower() in ("true", "1", "yes", "on")


def _build_temporal_hint(question: str) -> str:
    """One trimmed smart-model sub-call; returns a short answer or "".

    Retrieves a handful of dated memories for the question and asks the
    smart model to stitch the temporal answer on that small context.
    Fail-soft: any error returns "" (the cheap answerer just answers).
    """
    try:
        from .temporal_subcall import temporal_answer, format_evidence

        hint = ""
        store = _get_insight_store()
        if store is not None:
            recs = []
            try:
                if hasattr(store, "search"):
                    recs = store.search(question, limit=8) or []
            except Exception:
                recs = []
            try:
                store.close()
            except Exception:
                pass
            if recs:
                evidence = format_evidence(recs)
                if evidence.strip():
                    hint = temporal_answer(question, evidence)
        return (hint or "").strip()
    except Exception as exc:  # pragma: no cover - defensive
        logging.getLogger("argos").warning("temporal hint build failed: %s", exc)
        return ""




def _get_insight_store():
    """Get the active memory store for insight queries.

    Returns the SharedMemoryStore if the service is running, otherwise
    falls back to a direct DuckDBMemoryStore. Returns None on failure.
    """
    try:
        from service_client import SharedMemoryStore
        try:
            from hermes_constants import get_hermes_home
            home = Path(get_hermes_home())
        except Exception:
            home = Path(os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes")))
        return SharedMemoryStore(home, user_id=_active_user_id)
    except Exception:
        return None


def _handle_ilog_command(raw_args: str) -> str:
    """Handle /ilog — list saved personal insights, newest first.

    Usage:
        /ilog              — all insights, newest first (up to 20)
        /ilog work         — insights tagged 'work'
        /ilog ex shame     — insights tagged 'ex' OR 'shame'

    Note: /ilog is used instead of /insights to avoid conflicting with
    the built-in usage-analytics /insights command.
    """
    store = _get_insight_store()
    if store is None:
        return "Memory store is not available. Make sure Hermes is running."

    tags = [t.strip() for t in raw_args.split() if t.strip()] if raw_args else None
    try:
        insights = store.get_insights(tags=tags, limit=20)
    except Exception as e:
        return f"Could not retrieve insights: {e}"
    finally:
        store.close()

    if not insights:
        if tags:
            return f"No insights found with tags: {', '.join(tags)}"
        return "No insights saved yet. Share a realization in chat and it'll be captured automatically."

    lines = [f"**Insight Log** ({len(insights)} entries, newest first)\n"]
    for i, rec in enumerate(insights, 1):
        date = (rec.created_at or "")[:10]
        tag_str = ", ".join(rec.tags) if rec.tags else ""
        content_preview = rec.content[:120]
        if len(rec.content) > 120:
            content_preview += "..."
        lines.append(f"{i}. **[{date}]** {content_preview}")
        if tag_str:
            lines.append(f"   _tags: {tag_str}_")
        lines.append("")
    return "\n".join(lines)


def _handle_neg_command(raw_args: str) -> str:
    """Handle /neg — store an explicit negative (exclusion) claim.

    Negative memories answer "is X ...?" questions with a grounded no:
    they are injected with a [negative] label whenever the topic matches,
    so the model treats them as ground-truth exclusions instead of
    guessing from half-matching positive memories.

    Usage:
        /neg Alex does not drink coffee
    """
    claim = (raw_args or "").strip()
    if not claim:
        return (
            "Usage: /neg <claim> — e.g. /neg Alex does not drink coffee. "
            "Stored as category 'negative' and injected with a [negative] label."
        )
    store = _get_insight_store()
    if store is None:
        return "Memory store is not available. Make sure Hermes is running."
    try:
        rec = store.remember(category="negative", content=claim)
    except Exception as e:
        return f"Could not save negative memory: {e}"
    finally:
        store.close()
    if rec is None:
        return f"Saved (deduplicated or pending): /neg {claim}"
    mid = getattr(rec, "memory_id", "") or ""
    mid_s = f" [id: {mid}]" if mid else ""
    return f"Saved as [negative] memory{mid_s}: {claim}"


def _handle_revisit_command(raw_args: str) -> str:
    """Handle /revisit — surface a random older insight.

    Picks a random insight from at least 3 days ago (if available),
    falling back to any insight. Presents it conversationally for
    re-engagement.
    """
    import random
    from datetime import datetime, timezone, timedelta

    store = _get_insight_store()
    if store is None:
        return "Memory store is not available. Make sure Hermes is running."

    # Try to get older insights first (at least 3 days old).
    cutoff = (datetime.now(timezone.utc) - timedelta(days=3)).isoformat()
    try:
        older = store.get_insights(since=None, limit=50)
        recent = store.get_insights(since=cutoff, limit=50)
        # Prefer insights NOT in the recent set.
        recent_ids = {r.memory_id for r in recent}
        candidates = [r for r in older if r.memory_id not in recent_ids]
        if not candidates:
            candidates = older
    except Exception as e:
        return f"Could not retrieve insights: {e}"
    finally:
        store.close()

    if not candidates:
        return "No insights to revisit yet. Share a realization in chat and it'll be captured automatically."

    picked = random.choice(candidates)
    date = (picked.created_at or "")[:10]
    tag_str = ", ".join(picked.tags) if picked.tags else "none"

    return (
        f"**Revisiting an insight from {date}:**\n\n"
        f"> {picked.content}\n\n"
        f"_tags: {tag_str}_\n\n"
        f"Does this still resonate? Has anything shifted since then?"
    )


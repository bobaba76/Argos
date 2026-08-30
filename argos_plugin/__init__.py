"""Argos memory provider plugin — DuckDB + Kuzu + local embeddings.

Three-tier local memory for anything the user discusses — personal life,
work, tech, hobbies, relationships, goals. Fully offline storage
(no cloud, no API keys). Vector search via DuckDB's list_cosine_similarity
with text-search fallback. Relationship graph via Kuzu. Embeddings via
sentence-transformers with graceful degradation. Auto-extraction uses
generic syntactic patterns + optional LLM fallback for higher recall.

Configuration lives in $HERMES_HOME/hybrid_memory.json. The databases are
created at $HERMES_HOME/hybrid_memory.duckdb and $HERMES_HOME/hybrid_memory_kuzu/.

Categories:
  personal_fact  — stable things about the user (age, location, job, tools, traits)
  preference     — how the user likes things (tools, communication style, habits)
  insight        — self-observations, realizations, patterns noticed
  event          — notable events with date context (job changes, milestones)
  relationship   — people in the user's life and dynamics
  goal           — things the user is working toward
  context_note   — situational context that helps future conversations
"""
from __future__ import annotations

import difflib
import json
import re
import logging
import os
import queue
import threading
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error

from .embeddings import LocalEmbedder, _resolve_embedding_model_path
from .store import DuckDBMemoryStore, VALID_CATEGORIES
from .graph import KuzuGraphStore
from .confirmation import build_confirmation_block
from .extractor import extract_from_turn
from .routing import resolve_storage_names
from .service_client import SharedGraphStore, SharedMemoryStore
from .reviewer import review_candidate_with_llm, set_external_policy, suggest_expiry
from .query_expander import QueryExpander
from .distillation import run_distillation

logger = logging.getLogger(__name__)

# --- provider_core split (god-file refactor, behavior-neutral) ----------
# Module-level names that moved to provider_core.py are re-exported here
# so the full original surface survives (later mixins and tests import
# them from the package root).
try:
    from .provider_core import (  # noqa: F401
        _AUTO_EXTRACT_PAUSE_MARKER,
    _DEFAULT_INJECT_CONTENT_CHAR_CAP,
    _DEFAULT_INJECTION_MIN_SCORE,
    _DEFAULT_MAX_INJECTED,
    _DEFAULT_MODEL,
    _DATE_ANCHOR_RE,
    _INJECTION_FALLBACK_COUNT,
    _INJECT_CONTENT_CHAR_CAP,
    _MEMORY_FENCE_NOTE,
    _PREFETCH_WAIT_SECS,
    _TRIVIAL_QUERY_PATTERNS,
    _active_user_id,
    _config_cache,
    _config_cache_mtime,
    _config_cache_path,
    _flag,
    _freshness_marker_for,
    _load_config,
    _load_config_cached,
    _neutralize_markup,
    _store_config_cache,
    )
except ImportError:  # package imported as a top-level module
    from provider_core import (  # noqa: F401
        _AUTO_EXTRACT_PAUSE_MARKER,
    _DEFAULT_INJECT_CONTENT_CHAR_CAP,
    _DEFAULT_INJECTION_MIN_SCORE,
    _DEFAULT_MAX_INJECTED,
    _DEFAULT_MODEL,
    _DATE_ANCHOR_RE,
    _INJECTION_FALLBACK_COUNT,
    _INJECT_CONTENT_CHAR_CAP,
    _MEMORY_FENCE_NOTE,
    _PREFETCH_WAIT_SECS,
    _TRIVIAL_QUERY_PATTERNS,
    _active_user_id,
    _config_cache,
    _config_cache_mtime,
    _config_cache_path,
    _flag,
    _freshness_marker_for,
    _load_config,
    _load_config_cached,
    _neutralize_markup,
    _store_config_cache,
    )

try:
    from .provider_core import ProviderCoreMixin
except ImportError:  # package imported as a top-level module
    from provider_core import ProviderCoreMixin

# --- provider_session split (god-file refactor, behavior-neutral) -------
# Tool-schema dicts moved to provider_session.py (their only consumers live
# there); re-exported so ``from argos import SEARCH_SCHEMA`` keeps working.
try:
    from .provider_session import (  # noqa: F401
        CANDIDATE_LIST_SCHEMA,
    CANDIDATE_REVIEW_SCHEMA,
    CHAIN_SCHEMA,
    DELETE_SCHEMA,
    FEEDBACK_SCHEMA,
    FETCH_FULL_SCHEMA,
    GRAPH_QUERY_SCHEMA,
    GRAPH_SEARCH_SCHEMA,
    MAINTENANCE_SCHEMA,
    PURGE_TOMBSTONE_SCHEMA,
    RESTORE_SCHEMA,
    SAVE_SCHEMA,
    SEARCH_SCHEMA,
    TOMBSTONES_SCHEMA,
    UPDATE_SCHEMA,
    WHY_NOT_SCHEMA,
    )
except ImportError:  # package imported as a top-level module
    from provider_session import (  # noqa: F401
        CANDIDATE_LIST_SCHEMA,
    CANDIDATE_REVIEW_SCHEMA,
    CHAIN_SCHEMA,
    DELETE_SCHEMA,
    FEEDBACK_SCHEMA,
    FETCH_FULL_SCHEMA,
    GRAPH_QUERY_SCHEMA,
    GRAPH_SEARCH_SCHEMA,
    MAINTENANCE_SCHEMA,
    PURGE_TOMBSTONE_SCHEMA,
    RESTORE_SCHEMA,
    SAVE_SCHEMA,
    SEARCH_SCHEMA,
    TOMBSTONES_SCHEMA,
    UPDATE_SCHEMA,
    WHY_NOT_SCHEMA,
    )

try:
    from .provider_retrieval import ProviderRetrievalMixin
    from .provider_session import ProviderSessionMixin
except ImportError:  # package imported as a top-level module
    from provider_retrieval import ProviderRetrievalMixin
    from provider_session import ProviderSessionMixin







# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# MemoryProvider implementation
# ---------------------------------------------------------------------------

# Cached import of route_answerer (issue #29: was re-importing every turn).
_route_answerer_fn = None


class ArgosProvider(ProviderCoreMixin, ProviderRetrievalMixin, ProviderSessionMixin, MemoryProvider):
    """Three-tier local memory: DuckDB (vector) + Kuzu (graph) + local embeddings."""




# ---------------------------------------------------------------------------
# Ambient context — per-turn time/location/weather/file-activity hints
#
# These ride the native ``pre_llm_call`` plugin hook (not the async prefetch
# path) so they are built synchronously and injected unconditionally on every
# turn — including slow turns where prefetch times out and returns "".  The
# hook returns ``{"context": "..."}`` which Hermes appends to the API copy of
# the user message via the native ``plugin_user_context`` injection path,
# keeping the cached system prompt byte-stable.  No core source patch needed.
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------

def register(ctx) -> None:
    """Register Argos as a memory provider plugin.

    Also registers the insight-log skill, /ilog + /revisit slash commands,
    and a ``pre_llm_call`` hook for ambient context (time/location/weather/
    file-activity) — if the plugin context supports them.
    """
    try:
        ctx.register_memory_provider(ArgosProvider())
        logging.getLogger("argos").info(
            "argos: register() succeeded, provider registered"
        )
    except Exception as _e:
        logging.getLogger("argos").warning(
            "argos: register() failed: %s\n%s", _e, traceback.format_exc()
        )
        raise

    # Register the pre_llm_call hook for ambient context (time/location/
    # weather/file-activity).  Rides the native plugin_user_context injection
    # path — no core source patch needed.
    # Guarded: skip if THIS PROCESS already registered it (desktop reloads the
    # plugin per session; hooks are append-only, so an unguarded register
    # duplicates every ambient line per session count).
    try:
        if hasattr(ctx, "register_hook") and "pre_llm_call" not in _HOOKS_REGISTERED:
            ctx.register_hook("pre_llm_call", _on_pre_llm_call)
            _HOOKS_REGISTERED.add("pre_llm_call")
            logging.getLogger("argos").info(
                "argos: registered pre_llm_call hook for ambient context"
            )
    except Exception as _e:
        logging.getLogger("argos").debug(
            "argos: pre_llm_call hook registration skipped: %s", _e
        )

    # Register the insight-log skill (if the context supports skill registration).
    try:
        skill_md = Path(__file__).parent / "skills" / "insight-log" / "SKILL.md"
        if skill_md.exists() and hasattr(ctx, "register_skill"):
            ctx.register_skill("insight-log", skill_md)
            logging.getLogger("argos").info(
                "argos: registered insight-log skill"
            )
    except Exception as _e:
        logging.getLogger("argos").debug(
            "argos: skill registration skipped: %s", _e
        )

    # Register /ilog and /revisit slash commands (if supported).
    # Note: /ilog is used instead of /insights to avoid conflicting
    # with the built-in usage-analytics /insights command.
    try:
        if hasattr(ctx, "register_command"):
            ctx.register_command(
                "ilog",
                _handle_ilog_command,
                description="List saved personal insights (newest first)",
                args_hint="[tag]",
            )
            ctx.register_command(
                "revisit",
                _handle_revisit_command,
                description="Surface a random older insight for re-engagement",
            )
            ctx.register_command(
                "neg",
                _handle_neg_command,
                description="Remember an explicit negative/exclusion claim (category 'negative', injected as [negative])",
                args_hint="<claim>",
            )
            logging.getLogger("argos").info(
                "argos: registered /ilog, /revisit and /neg commands"
            )
    except Exception as _e:
        logging.getLogger("argos").debug(
            "argos: command registration skipped: %s", _e
        )


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

# Legacy name retained for eval harnesses importing the pre-rename symbol.
HybridMemoryProvider = ArgosProvider



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

# --- provider_ambient split (god-file refactor, behavior-neutral) -------
# Module-level helpers that moved to provider_ambient.py, re-exported so
# the full original surface survives.
try:
    from .provider_ambient import (  # noqa: F401
        _HOOKS_REGISTERED,
    _as_flag,
    _build_coder_directive,
    _build_file_activity_hint,
    _build_location_hint,
    _build_temporal_hint,
    _build_timestamp_hint,
    _build_weather_hint,
    _get_insight_store,
    _handle_ilog_command,
    _handle_neg_command,
    _handle_revisit_command,
    _on_pre_llm_call,
    _route_answerer_fn,
    )
except ImportError:  # package imported as a top-level module
    from provider_ambient import (  # noqa: F401
        _HOOKS_REGISTERED,
    _as_flag,
    _build_coder_directive,
    _build_file_activity_hint,
    _build_location_hint,
    _build_temporal_hint,
    _build_timestamp_hint,
    _build_weather_hint,
    _get_insight_store,
    _handle_ilog_command,
    _handle_neg_command,
    _handle_revisit_command,
    _on_pre_llm_call,
    _route_answerer_fn,
    )


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




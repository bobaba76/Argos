"""Hybrid Memory provider plugin — DuckDB + Kuzu + local embeddings.

Three-tier local memory for anything the user discusses — personal life,
work, tech, hobbies, health, relationships, goals. Fully offline storage
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
from .reviewer import review_candidate_with_llm
from .query_expander import QueryExpander

logger = logging.getLogger(__name__)

_PREFETCH_WAIT_SECS = 3.0
_DEFAULT_MAX_INJECTED = 20
# Per-memory content cap in the auto-injection block. Without this, a few
# long memories (3000+ chars) can blow the token budget at N=20. 200 chars
# preserves the key fact while keeping the injection block compact.
_INJECT_CONTENT_CHAR_CAP = 200
_DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
_AUTO_EXTRACT_PAUSE_MARKER = "hybrid_memory.auto_extract.paused"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _load_config(hermes_home: str | None = None) -> dict:
    """Load config from $HERMES_HOME/hybrid_memory.json.

    Uses the explicit hermes_home when provided (from initialize() kwargs),
    falling back to get_hermes_home() only when not provided. This ensures
    the config file and databases resolve to the same directory.
    """
    config = {
        "database_filename": "hybrid_memory.duckdb",
        "graph_dirname": "hybrid_memory_kuzu",
        "storage_mode": "shared_service",
        "max_injected_items": str(_DEFAULT_MAX_INJECTED),
        "local_embedding_model": _DEFAULT_MODEL,
        "auto_extract": "true",
        "llm_fallback": "true",
        "extraction_shadow_diff": "false",
        "auto_review": "true",
        "graph_aware_retrieval": "true",
        "graph_retrieval_boost": "0.0",
        "graph_inject_candidates": "false",
        "graph_boost_min_similarity": "0.15",
        "alias_expansion_boost": "0.7",
        "graph_traversal_enabled": "true",
        "graph_traversal_depth": "2",
        "graph_traversal_boost": "0.60",
        "chain_unfold": "auto",
        "chain_unfold_min_similarity": "0.30",
        "chain_max_versions": "3",
        "chain_max_inject": "150",
        "chain_unfold_top_k": "3",
        "chain_unfold_query_fallback": "false",
        "consolidation_enabled": "false",
        "consolidation_min_age_days": "30",
        "consolidation_max_actions": "25",
        "reranker_enabled": "false",
        "reranker_model": "BAAI/bge-reranker-base",
        "reranker_top_n": "10",
        "context_aware_retrieval": "true",
        "context_window_size": "3",
        "context_max_chars": "500",
        "query_expansion_enabled": "true",
        "query_expansion_similarity_floor": "0.3",
        "llm_model": "",
        "llm_provider": "",
        "entity_aliases": "",
    }
    if hermes_home:
        home = Path(hermes_home)
    else:
        try:
            from hermes_constants import get_hermes_home
            home = get_hermes_home()
        except Exception:
            home = Path(os.path.expanduser("~/.hermes"))

    config_path = home / "hybrid_memory.json"
    if config_path.exists():
        try:
            file_cfg = json.loads(config_path.read_text(encoding="utf-8"))
            config.update({k: v for k, v in file_cfg.items()
                           if v is not None and v != ""})
        except Exception:
            pass
    return config


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

SEARCH_SCHEMA = {
    "name": "memory_search",
    "description": (
        "Search the user's memory by meaning. Returns memories ranked "
        "by relevance — facts, preferences, insights, relationships, "
        "goals, and events from past conversations. Use this before answering "
        "anything that depends on what you know about the user from past "
        "conversations. For multi-part questions, search several times with "
        "different wording."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search for."},
            "top_k": {"type": "integer", "description": "Max results (default: 8, max: 50)."},
            "category": {
                "type": "string",
                "description": "Filter to a specific category (optional). "
                               "One of: personal_fact, preference, insight, event, "
                               "relationship, goal, context_note.",
            },
            "project_id": {
                "type": "string",
                "description": "Restrict results to this project's memories plus global memories (optional).",
            },
        },
        "required": ["query"],
    },
}

SAVE_SCHEMA = {
    "name": "memory_save",
    "description": (
        "Store a durable fact about the user. Call this the moment the user states "
        "a lasting preference, personal detail, tool choice, relationship fact, "
        "insight, goal, or any information worth recalling in future conversations. "
        "Do not store transient chit-chat. The content should be a clear, "
        "self-contained statement. For non-trivial reasoning (medical, attributetic, "
        "decision-making), include the REASONING that led to the conclusion — "
        "what evidence supports it, what was ruled out, and the uncertainty level. "
        "Long content (200-800 chars) is fine and encouraged for complex topics."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "The fact to store (a clear, self-contained statement)."},
            "category": {
                "type": "string",
                "description": "Memory category. One of: personal_fact, preference, "
                               "insight, event, relationship, goal, context_note.",
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional tags for this memory.",
            },
        },
        "required": ["content", "category"],
    },
}

UPDATE_SCHEMA = {
    "name": "memory_update",
    "description": (
        "Update an existing memory by its ID (take the ID from a memory_search "
        "result). Use when a stored fact has changed or was incorrect."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "memory_id": {"type": "string", "description": "Memory ID to update."},
            "content": {"type": "string", "description": "New content text (optional)."},
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "New tags (optional, replaces existing).",
            },
        },
        "required": ["memory_id"],
    },
}

DELETE_SCHEMA = {
    "name": "memory_delete",
    "description": (
        "Delete a memory by its ID (take the ID from a memory_search result). "
        "Use when a stored fact is obsolete or the user asks you to forget it."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "memory_id": {"type": "string", "description": "Memory ID to delete."},
        },
        "required": ["memory_id"],
    },
}

GRAPH_SEARCH_SCHEMA = {
    "name": "memory_graph_search",
    "description": (
        "Search the relationship graph for connections between people, tools, "
        "concepts, and entities in the user's life. Use this to find how things "
        "relate (e.g., 'who is Entity-B' or 'what tools does the user use'). "
        "Returns edges showing source -> relation -> target."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "term": {"type": "string", "description": "Entity name to search for (e.g., 'Entity-B', 'FocusTool')."},
        },
        "required": ["term"],
    },
}

GRAPH_QUERY_SCHEMA = {
    "name": "memory_graph_query",
    "description": (
        "Traverse the memory graph around an entity and return connected nodes, "
        "relationships, and supporting memories. Use this after memory_graph_search "
        "when you need context such as what an entity is connected to or which "
        "saved memories mention it."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "entity_id": {"type": "string", "description": "Entity name or ID to start from (e.g., 'Entity-B', 'memory:<id>')."},
            "depth": {"type": "integer", "description": "Traversal depth from 1 to 4 (default: 2)."},
            "limit": {"type": "integer", "description": "Maximum nodes/edges to return (default: 100)."},
        },
        "required": ["entity_id"],
    },
}

CANDIDATE_LIST_SCHEMA = {
    "name": "memory_candidate_list",
    "description": (
        "List pending memory proposals created by automatic extraction. These are "
        "not active memories until reviewed."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "status": {
                "type": "string",
                "description": "pending, reviewed_approved, pending_user_confirmation, approved, rejected, or quarantined (default: pending).",
            },
            "limit": {"type": "integer", "description": "Maximum proposals to return (default: 20)."},
        },
    },
}

CANDIDATE_REVIEW_SCHEMA = {
    "name": "memory_candidate_review",
    "description": (
        "Review a pending memory proposal. Approve only facts that are correct, "
        "durable, about the user, and scoped appropriately. When a candidate "
        "replaces or contradicts an existing current fact (e.g. a new employer, "
        "address, or changed opinion), pass supersedes_memory_id to chain the "
        "new memory behind the old one — preserving the change history."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "candidate_id": {"type": "string", "description": "Candidate ID from memory_candidate_list."},
            "decision": {
                "type": "string",
                "description": "approved, rejected, or quarantined.",
            },
            "reason": {"type": "string", "description": "Short review explanation (optional)."},
            "supersedes_memory_id": {
                "type": "string",
                "description": "Existing current memory_id this candidate replaces (optional). Chains the new memory behind the old one, preserving history. Use when the candidate is a replacement/contradiction of an existing fact.",
            },
        },
        "required": ["candidate_id", "decision"],
    },
}

RESTORE_SCHEMA = {
    "name": "memory_restore",
    "description": "Restore a quarantined memory to active retrieval after review.",
    "parameters": {
        "type": "object",
        "properties": {
            "memory_id": {"type": "string", "description": "Memory ID to restore."},
        },
        "required": ["memory_id"],
    },
}

FEEDBACK_SCHEMA = {
    "name": "memory_feedback",
    "description": "Mark an active memory helpful, dismissed, or incorrect.",
    "parameters": {
        "type": "object",
        "properties": {
            "memory_id": {"type": "string", "description": "Memory ID from a search result."},
            "feedback": {
                "type": "string",
                "description": "helpful, dismissed, or incorrect.",
            },
        },
        "required": ["memory_id", "feedback"],
    },
}

MAINTENANCE_SCHEMA = {
    "name": "memory_maintenance",
    "description": (
        "Preview or apply conservative, reversible memory maintenance. It can "
        "quarantine expired or stale temporary memories and lower-quality duplicates. "
        "Dry-run is the default; it never permanently deletes records."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "dry_run": {"type": "boolean", "description": "Preview only (default true)."},
            "max_actions": {"type": "integer", "description": "Maximum records to quarantine (default 25)."},
            "min_age_days": {"type": "integer", "description": "Age threshold for stale temporary memories (default 30)."},
        },
    },
}

CHAIN_SCHEMA = {
    "name": "memory_chain",
    "description": (
        "Show how a memory evolved over time — the version chain behind a "
        "fact (e.g. why an opinion, job, tool, or preference changed). Pass "
        "a memory_id from a memory_search result whose chain annotation said "
        "it has history. Modes: arc (one line per version, default), versions "
        "(full records), diff (what changed at each step)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "memory_id": {"type": "string", "description": "Any memory_id from the chain (head, middle, or tail)."},
            "mode": {
                "type": "string",
                "description": "arc (compact, default), versions (full), or diff (per-step deltas).",
            },
            "max_versions": {"type": "integer", "description": "Most recent versions to return (default 5, max 10)."},
        },
        "required": ["memory_id"],
    },
}


# ---------------------------------------------------------------------------
# MemoryProvider implementation
# ---------------------------------------------------------------------------

class HybridMemoryProvider(MemoryProvider):
    """Three-tier local memory: DuckDB (vector) + Kuzu (graph) + local embeddings."""

    def __init__(self) -> None:
        self._config: dict = {}
        self._hermes_home: str = ""
        self._embedder: Optional[LocalEmbedder] = None
        self._store: Optional[DuckDBMemoryStore] = None
        self._graph: Optional[KuzuGraphStore] = None
        self._user_id: str = "default_user"
        self._agent_context: str = "primary"
        self._platform: str = "cli"
        self._max_injected: int = _DEFAULT_MAX_INJECTED
        self._auto_extract: bool = True
        self._llm_fallback: bool = True
        self._extraction_shadow_diff: bool = False
        self._auto_review: bool = True
        self._graph_aware_retrieval: bool = True
        self._graph_retrieval_boost: float = 0.05
        self._graph_inject_candidates: bool = False
        self._graph_boost_min_similarity: float = 0.15
        self._alias_expansion_boost: float = 0.7
        self._graph_traversal_enabled: bool = False
        self._graph_traversal_depth: int = 2
        self._graph_traversal_boost: float = 0.0
        self._consolidation_enabled: bool = False
        self._consolidation_min_age_days: int = 30
        self._consolidation_max_actions: int = 25
        self._reranker_enabled: bool = False
        self._reranker_model: str = "BAAI/bge-reranker-base"
        self._reranker_top_n: int = 10
        self._reranker = None
        self._auto_extract_paused: bool = False
        # Chain-unfold (Hy-Memory headline feature): auto-inject a compact
        # version arc when the query signals change-intent AND a top result
        # has a chain. Ships OFF for the first wave — measure, then flip to
        # "auto". Accounting is separate from retrieval counters (see
        # _chain_unfolded_stats).
        self._chain_unfold: str = "off"  # "off" | "auto" | "always"
        self._chain_unfold_min_similarity: float = 0.30
        self._chain_max_versions: int = 3
        self._chain_max_inject: int = 150  # soft token cap per chain
        # Recall rebalance (2026-08-17): scan top-K results for a chain
        # anchor instead of top-1 only, with an optional query-side
        # fallback that searches deeper when no top-K result has a chain.
        # The 0.30 per-candidate floor is the precision guard.
        self._chain_unfold_top_k: int = 3
        self._chain_unfold_query_fallback: bool = False
        self._chain_unfolded_stats: Dict[str, int] = {
            "count": 0, "tokens_injected": 0,
        }
        self._initialized: bool = False
        self._current_project_id: str = ""
        # Context-aware retrieval: rolling window of recent messages.
        # Used to enrich queries that contain pronouns/references with
        # conversation context so the embedder can resolve "that", "he",
        # "the thing" etc.
        self._context_aware_retrieval: bool = True
        self._context_window_size: int = 3  # last N user messages
        self._context_max_chars: int = 500  # cap total context length
        self._recent_user_messages: list[str] = []
        self._context_lock = threading.Lock()
        # Query expansion (lazy/conditional, cached, fail-soft)
        self._query_expansion_enabled: bool = True
        self._query_expansion_similarity_floor: float = 0.3
        self._query_expander: Optional[QueryExpander] = None
        # LLM model/provider for auxiliary tasks (extraction, review, expansion)
        # Empty string = use the auxiliary client's default model
        self._llm_model: str = ""
        self._llm_provider: str = ""
        # Prefetch state.
        self._prefetch_thread: Optional[threading.Thread] = None
        self._prefetch_query: str = ""
        self._prefetch_result: str = ""
        self._prefetch_done: bool = False
        self._prefetch_lock = threading.Lock()
        # Sync state: bounded queue + persistent worker.
        self._sync_queue: "queue.Queue[Optional[tuple[str, str, str]]]" = queue.Queue(maxsize=3)
        self._sync_dropped_turns = 0
        self._sync_thread: Optional[threading.Thread] = None
        self._sync_lock = threading.Lock()
        self._sync_worker_started = False

    @property
    def name(self) -> str:
        return "hybrid_memory"

    # -- availability --------------------------------------------------------

    def is_available(self) -> bool:
        """True if duckdb is importable. Kuzu and embeddings are optional."""
        try:
            import duckdb  # noqa: F401
            return True
        except ImportError:
            return False

    # -- config schema (for `hermes memory setup`) ---------------------------

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return [
            {
                "key": "local_embedding_model",
                "description": "Sentence-transformers model name for local embeddings",
                "default": _DEFAULT_MODEL,
                "required": False,
            },
            {
                "key": "max_injected_items",
                "description": "Max memories to auto-inject per turn",
                "default": str(_DEFAULT_MAX_INJECTED),
                "required": False,
            },
            {
                "key": "auto_extract",
                "description": "Enable automatic candidate extraction after each turn; proposals require review before becoming active memory",
                "default": "true",
                "choices": ["true", "false"],
                "required": False,
            },
            {
                "key": "llm_fallback",
                "description": "Use LLM to extract facts when regex patterns miss them (adds latency + token cost per turn)",
                "default": "true",
                "choices": ["true", "false"],
                "required": False,
            },
            {
                "key": "auto_review",
                "description": "Automatically review new memory proposals with the LLM",
                "default": "true",
                "choices": ["true", "false"],
                "required": False,
            },
            {
                "key": "graph_aware_retrieval",
                "description": "Boost memory results supported by graph entities",
                "default": "true",
                "choices": ["true", "false"],
                "required": False,
            },
            {
                "key": "graph_retrieval_boost",
                "description": "Maximum graph-supported retrieval boost",
                "default": "0.05",
                "required": False,
            },
            {
                "key": "graph_inject_candidates",
                "description": "Inject graph-only memories not found by semantic search",
                "default": "false",
                "choices": ["true", "false"],
                "required": False,
            },
            {
                "key": "graph_boost_min_similarity",
                "description": "Minimum semantic similarity for a record to receive graph boost",
                "default": "0.15",
                "required": False,
            },
            {
                "key": "graph_traversal_enabled",
                "description": "Enable traversal-based retrieval: walk typed relations from query seed entities (hop-weighted BFS) and inject graph-only candidates under the similarity floor",
                "default": "false",
                "choices": ["true", "false"],
                "required": False,
            },
            {
                "key": "graph_traversal_depth",
                "description": "Max hops for traversal-based retrieval (1-4)",
                "default": "2",
                "required": False,
            },
            {
                "key": "alias_expansion_boost",
                "description": "Minimum similarity floor for alias-expanded candidates (identity mappings like 'my role'→'Entity-A'). Alias expansion is definitive, not fuzzy — the candidate IS about the query entity, so its similarity is floored to this value when the raw embedding similarity is lower.",
                "default": "0.7",
                "required": False,
            },
            {
                "key": "consolidation_enabled",
                "description": "Allow automatic reversible memory maintenance at session end",
                "default": "false",
                "choices": ["true", "false"],
                "required": False,
            },
        ]

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        config_path = Path(hermes_home) / "hybrid_memory.json"
        existing = {}
        if config_path.exists():
            try:
                existing = json.loads(config_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        existing.update(values)
        config_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")

    # -- initialization ------------------------------------------------------

    def initialize(self, session_id: str, **kwargs) -> None:
        self._hermes_home = kwargs.get("hermes_home", "")
        self._platform = kwargs.get("platform", "cli")
        self._agent_context = kwargs.get("agent_context", "primary")
        self._user_id = kwargs.get("user_id") or "default_user"
        self._current_project_id = str(kwargs.get("project_id") or "").strip()

        self._config = _load_config(self._hermes_home)
        home = Path(self._hermes_home) if self._hermes_home else Path(os.path.expanduser("~/.hermes"))

        db_filename = self._config.get("database_filename", "hybrid_memory.duckdb")
        graph_dirname = self._config.get("graph_dirname", "hybrid_memory_kuzu")

        storage_mode = str(self._config.get("storage_mode", "shared_service")).lower()
        use_shared_service = storage_mode not in {"local", "direct"}

        # Local fallback routing. The shared service path below always owns the
        # canonical primary files, regardless of the caller's platform.
        db_filename, graph_dirname = resolve_storage_names(
            self._platform, db_filename, graph_dirname
        )

        model_name = self._config.get("local_embedding_model", _DEFAULT_MODEL)

        try:
            self._max_injected = int(self._config.get("max_injected_items", _DEFAULT_MAX_INJECTED))
        except (ValueError, TypeError):
            self._max_injected = _DEFAULT_MAX_INJECTED

        auto = self._config.get("auto_extract", "true")
        self._auto_extract = (
            auto.lower() in ("true", "1", "yes") if isinstance(auto, str) else bool(auto)
        )

        llm_fb = self._config.get("llm_fallback", "true")
        self._llm_fallback = (
            llm_fb.lower() in ("true", "1", "yes") if isinstance(llm_fb, str) else bool(llm_fb)
        )
        shadow_diff = self._config.get("extraction_shadow_diff", "false")
        self._extraction_shadow_diff = (
            shadow_diff.lower() in ("true", "1", "yes")
            if isinstance(shadow_diff, str) else bool(shadow_diff)
        )
        auto_review = self._config.get("auto_review", "true")
        self._auto_review = (
            auto_review.lower() in ("true", "1", "yes")
            if isinstance(auto_review, str) else bool(auto_review)
        )
        graph_aware = self._config.get("graph_aware_retrieval", "true")
        self._graph_aware_retrieval = (
            graph_aware.lower() in ("true", "1", "yes")
            if isinstance(graph_aware, str) else bool(graph_aware)
        )
        try:
            self._graph_retrieval_boost = max(
                0.0, min(float(self._config.get("graph_retrieval_boost", 0.05)), 0.5)
            )
        except (TypeError, ValueError):
            self._graph_retrieval_boost = 0.05
        graph_inject = self._config.get("graph_inject_candidates", "false")
        self._graph_inject_candidates = (
            graph_inject.lower() in ("true", "1", "yes")
            if isinstance(graph_inject, str) else bool(graph_inject)
        )
        try:
            self._graph_boost_min_similarity = max(
                0.0, min(float(self._config.get("graph_boost_min_similarity", 0.15)), 1.0)
            )
        except (TypeError, ValueError):
            self._graph_boost_min_similarity = 0.15
        graph_trav = self._config.get("graph_traversal_enabled", "false")
        self._graph_traversal_enabled = (
            graph_trav.lower() in ("true", "1", "yes")
            if isinstance(graph_trav, str) else bool(graph_trav)
        )
        try:
            self._graph_traversal_depth = max(
                1, min(int(self._config.get("graph_traversal_depth", 2)), 4)
            )
        except (TypeError, ValueError):
            self._graph_traversal_depth = 2
        try:
            self._graph_traversal_boost = max(
                0.0, min(float(self._config.get("graph_traversal_boost", 0.0)), 1.0)
            )
        except (TypeError, ValueError):
            self._graph_traversal_boost = 0.0
        try:
            self._alias_expansion_boost = max(
                0.0, min(float(self._config.get("alias_expansion_boost", 0.7)), 1.0)
            )
        except (TypeError, ValueError):
            self._alias_expansion_boost = 0.7
        consolidation_enabled = self._config.get("consolidation_enabled", "false")
        self._consolidation_enabled = (
            consolidation_enabled.lower() in ("true", "1", "yes")
            if isinstance(consolidation_enabled, str) else bool(consolidation_enabled)
        )
        try:
            self._consolidation_min_age_days = max(
                1, int(self._config.get("consolidation_min_age_days", 30))
            )
        except (TypeError, ValueError):
            self._consolidation_min_age_days = 30
        try:
            self._consolidation_max_actions = max(
                1, min(int(self._config.get("consolidation_max_actions", 25)), 500)
            )
        except (TypeError, ValueError):
            self._consolidation_max_actions = 25
        # Reranker config
        reranker_enabled = self._config.get("reranker_enabled", "true")
        self._reranker_enabled = (
            reranker_enabled.lower() in ("true", "1", "yes")
            if isinstance(reranker_enabled, str) else bool(reranker_enabled)
        )
        self._reranker_model = str(
            self._config.get("reranker_model", "BAAI/bge-reranker-base")
        )
        try:
            self._reranker_top_n = max(
                5, min(int(self._config.get("reranker_top_n", 20)), 100)
            )
        except (TypeError, ValueError):
            self._reranker_top_n = 20
        # Context-aware retrieval config
        ctx_aware = self._config.get("context_aware_retrieval", "true")
        self._context_aware_retrieval = (
            ctx_aware.lower() in ("true", "1", "yes")
            if isinstance(ctx_aware, str) else bool(ctx_aware)
        )
        try:
            self._context_window_size = max(
                1, min(int(self._config.get("context_window_size", 3)), 10)
            )
        except (TypeError, ValueError):
            self._context_window_size = 3
        try:
            self._context_max_chars = max(
                100, min(int(self._config.get("context_max_chars", 500)), 2000)
            )
        except (TypeError, ValueError):
            self._context_max_chars = 500
        # Query expansion config
        qe_enabled = self._config.get("query_expansion_enabled", "true")
        self._query_expansion_enabled = (
            qe_enabled.lower() in ("true", "1", "yes")
            if isinstance(qe_enabled, str) else bool(qe_enabled)
        )
        try:
            self._query_expansion_similarity_floor = float(
                self._config.get("query_expansion_similarity_floor", "0.3")
            )
        except (TypeError, ValueError):
            self._query_expansion_similarity_floor = 0.3
        # LLM model/provider config (empty = use auxiliary client default)
        self._llm_model = str(self._config.get("llm_model", "")).strip()
        self._llm_provider = str(self._config.get("llm_provider", "")).strip()
        if self._query_expansion_enabled:
            self._query_expander = QueryExpander(
                similarity_floor=self._query_expansion_similarity_floor,
                model=self._llm_model,
                provider=self._llm_provider,
            )
        pause_marker = home / _AUTO_EXTRACT_PAUSE_MARKER
        env_pause = os.environ.get("HERMES_HYBRID_MEMORY_PAUSE_AUTO_EXTRACT", "")
        self._auto_extract_paused = pause_marker.exists() or env_pause.lower() in {
            "1", "true", "yes", "on"
        }

        # ---- Scale trigger instrumentation -------------------------------
        # The triage rule: build cheap/irreversible seams now, gate expensive
        # engines (ANN/BM25, fact families) behind MEASURED triggers. The
        # store owns the latency window and record-count sampling; the
        # provider exposes them via get_scale_metrics().
        self._scale_warn_latency_ms = float(
            self._config.get("scale_warn_latency_ms", 300.0)
        )
        self._scale_warn_records = int(
            self._config.get("scale_warn_records", 5000)
        )
        if self._store is not None:
            try:
                self._store.set_scale_thresholds(
                    self._scale_warn_latency_ms, self._scale_warn_records
                )
            except Exception:
                pass

        # Entity aliases: load from config (JSON mapping) into the store.
        aliases_json = str(self._config.get("entity_aliases", "")).strip()
        if aliases_json and self._store:
            try:
                alias_map = json.loads(aliases_json)
                if isinstance(alias_map, dict):
                    for alias, canonical in alias_map.items():
                        if isinstance(canonical, str):
                            self._store.add_alias(alias, canonical)
                    logger.info("Loaded %d entity aliases from config", len(alias_map))
            except (json.JSONDecodeError, Exception) as exc:
                logger.warning("Failed to parse entity_aliases config: %s", exc)

        # Embedder (lazy — model loads on first embed call).
        resolved_model = _resolve_embedding_model_path(model_name, home)
        self._embedder = LocalEmbedder(resolved_model, hermes_home=home)
        self._evidence_retention = str(
            self._config.get("evidence_retention", "full")
        ).lower()
        # Chain-unfold config (ships off; flip to "auto" after eval).
        self._chain_unfold = str(
            self._config.get("chain_unfold", "off")
        ).lower()
        if self._chain_unfold not in {"off", "auto", "always"}:
            self._chain_unfold = "off"
        try:
            self._chain_unfold_min_similarity = max(
                0.0, min(float(self._config.get("chain_unfold_min_similarity", 0.30)), 1.0)
            )
        except (TypeError, ValueError):
            self._chain_unfold_min_similarity = 0.30
        try:
            self._chain_max_versions = max(
                1, min(int(self._config.get("chain_max_versions", 3)), 10)
            )
        except (TypeError, ValueError):
            self._chain_max_versions = 3
        try:
            self._chain_max_inject = max(
                1, int(self._config.get("chain_max_inject", 150))
            )
        except (TypeError, ValueError):
            self._chain_max_inject = 150
        try:
            self._chain_unfold_top_k = max(
                1, min(int(self._config.get("chain_unfold_top_k", 3)), 20)
            )
        except (TypeError, ValueError):
            self._chain_unfold_top_k = 3
        self._chain_unfold_query_fallback = str(
            self._config.get("chain_unfold_query_fallback", "false")
        ).lower() in {"1", "true", "yes", "on"}

        # Reranker (lazy — model loads on first rerank call).
        if self._reranker_enabled:
            try:
                from .embeddings import CrossEncoderReranker
            except ImportError:
                from embeddings import CrossEncoderReranker
            self._reranker = CrossEncoderReranker(
                self._reranker_model, hermes_home=home
            )

        if use_shared_service:
            # One local service owns the canonical DuckDB/Kùzu files. The
            # provider process never opens those files directly in this mode.
            # The reranker runs inside the service process (passed via RPC
            # config), not here.
            self._store = SharedMemoryStore(
                home, user_id=self._user_id, embedder=self._embedder
            )
            try:
                self._graph = SharedGraphStore(home, user_id=self._user_id)
            except Exception as e:
                logger.warning("Shared Kùzu graph unavailable, continuing without it: %s", e)
                self._graph = None
        else:
            db_path = home / db_filename
            self._store = DuckDBMemoryStore(
                db_path, user_id=self._user_id, embedder=self._embedder,
                reranker=self._reranker,
            )
            self._store._reranker_top_n = self._reranker_top_n
            try:
                graph_path = home / graph_dirname
                self._graph = KuzuGraphStore(graph_path, user_id=self._user_id)
            except Exception as e:
                logger.warning("Kuzu graph unavailable, continuing without it: %s", e)
                self._graph = None

        self._initialized = True
        logger.info(
            "HybridMemory initialized: %d memories, graph=%s, embeddings=%s, "
            "auto_extract=%s, auto_review=%s, paused=%s, storage=%s, proposals=on",
            self._store.count(),
            "on" if self._graph else "off",
            "pending" if not self._embedder.is_available else "on",
            self._auto_extract,
            self._auto_review,
            self._auto_extract_paused,
            "shared_service" if use_shared_service else "direct",
        )

    # -- system prompt -------------------------------------------------------

    def system_prompt_block(self) -> str:
        # STATIC text only — must be byte-stable for prompt caching.
        # Dynamic state (memory count, embedding status) is NOT included here
        # because it changes between turns and would invalidate the cached prefix.
        # The prefetch() method injects dynamic recall context separately.
        graph_status = "available" if self._graph else "unavailable"
        return (
            "# Hybrid Memory (Local)\n"
            f"Active. Relationship graph: {graph_status}.\n"
            "You have persistent memory of this user from past conversations — "
            "any topic: personal life, work, tech, health, hobbies, relationships. "
            "Relevant memories are auto-injected before each turn. For deeper or "
            "multi-hop lookups, call memory_search with different wording.\n"
            "Categories: personal_fact (stable facts), preference (how they like things), "
            "insight (self-observations, realizations), event (life events, milestones), "
            "relationship (people in their life), goal (what they're working toward), "
            "context_note (situational context).\n"
            "When the user states a durable fact, preference, or insight — about "
            "ANY topic — call memory_save immediately — don't wait to be asked. "
            "Automatic extraction creates pending proposals, not active memories. "
            "Review them with memory_candidate_list and memory_candidate_review; "
            "never approve a proposal merely because another model produced it.\n"
            "\n"
            "## Save reasoning, not just conclusions\n"
            "When you work through a non-trivial topic with the user — technical reasoning, "
            "analytical reasoning, trade-off analysis, decision approaches, important "
            "decisions — save the REASONING CHAIN, not just the final conclusion. "
            "A bare fact like 'Fact-A might be Fact-B' is far less useful than the full "
            "reasoning: what evidence supports it, what was considered and ruled out, "
            "what the uncertainty level is, and what would confirm or deny it. "
            "Use the content field to store a self-contained reasoning summary "
            "(200-800 chars is fine — the system handles long content). "
            "This ensures future sessions can reconstruct WHY a conclusion was reached, "
            "not just WHAT it was.\n"
            "\n"
            "## Quality over quantity\n"
            "Don't save trivial facts the agent could infer from context ('user uses "
            "a keyboard', 'user is typing'). Don't save fragments of your own output. "
            "Don't save the same fact in slightly different wording. One rich, "
            "well-reasoned memory is worth ten shallow flashcards.\n"
            "Use memory_graph_search to find relationships between people, tools, "
            "and concepts in the user's life."
        )

    # -- retrieval ------------------------------------------------------------

    # -- context-aware retrieval ---------------------------------------------

    # Patterns that indicate a query depends on conversation context to
    # resolve references. If the query matches any of these AND we have
    # recent messages, we prepend the context to the query before search.
    _REFERENTIAL_PATTERNS = [
        r"\bthat\b", r"\bthis\b", r"\bit\b", r"\bthe thing\b",
        r"\bwhat about\b", r"\btell me more\b", r"\bhe\b", r"\bshe\b",
        r"\bhim\b", r"\bher\b", r"\bthem\b", r"\bthey\b",
        r"\bthe one\b", r"\bthe last\b", r"\bthe other\b",
        r"\bremember (when|that|the)\b",
    ]

    @classmethod
    def _is_referential_query(cls, query: str) -> bool:
        """Check if a query contains pronouns/references that need context."""
        import re
        query_lower = query.lower().strip()
        # Short queries with referential language are the strongest signal.
        # Long queries usually have enough keywords on their own.
        if len(query_lower) > 300:
            return False
        for pattern in cls._REFERENTIAL_PATTERNS:
            if re.search(pattern, query_lower):
                return True
        return False

    def _enrich_query_with_context(self, query: str) -> str:
        """Prepend recent conversation context to a referential query.

        This resolves pronouns like "that", "he", "the thing" by giving
        the embedder the surrounding conversation as context. The context
        is prepended (not appended) so the embedder sees it first.

        Returns the original query unchanged if:
        - context-aware retrieval is disabled
        - the query doesn't contain referential language
        - there are no recent messages
        """
        if not self._context_aware_retrieval:
            return query
        if not self._is_referential_query(query):
            return query
        with self._context_lock:
            recent = list(self._recent_user_messages)
        if not recent:
            return query
        # Build context string from recent messages, capped to max_chars.
        # We use the last N user messages (most recent last).
        context_parts: list[str] = []
        total_chars = 0
        for msg in reversed(recent):  # most recent first
            if total_chars + len(msg) > self._context_max_chars:
                break
            context_parts.insert(0, msg)
            total_chars += len(msg)
        if not context_parts:
            return query
        context = " ".join(context_parts)
        # Prepend context, then the query. The embedder will see both.
        return f"{context} {query}"

    def _record_user_message(self, message: str) -> None:
        """Add a user message to the rolling context window."""
        if not message or not message.strip():
            return
        with self._context_lock:
            self._recent_user_messages.append(message.strip())
            # Keep only the last N messages.
            while len(self._recent_user_messages) > self._context_window_size:
                self._recent_user_messages.pop(0)

    def _expand_and_merge(
        self,
        query: str,
        project_id: str | None,
        category_filter: str | None,
        candidate_limit: int,
        original_results: List[Any],
    ) -> List[Any]:
        """Expand a weak query into sub-queries and merge results via RRF.

        Fail-soft: if expansion produces no sub-queries or all sub-query
        searches fail, return the original results unchanged.
        """
        if not self._query_expander or not self._store:
            return original_results

        try:
            sub_queries = self._query_expander.expand(query)
        except Exception as exc:
            logger.debug("Query expansion failed: %s", exc)
            return original_results

        if not sub_queries:
            return original_results

        logger.debug("Query expansion: '%s' → %d sub-queries", query[:50], len(sub_queries))

        # Search each sub-query and merge via Reciprocal Rank Fusion
        # with the original results.
        all_results: dict[str, Any] = {}
        for r in original_results:
            all_results[r.memory_id] = r

        # RRF: original results get rank-based scores
        rrf_k = 20  # lowered from 60 to sharpen relevance discrimination
        rrf_scores: dict[str, float] = {}
        for rank, r in enumerate(original_results):
            rrf_scores[r.memory_id] = 1.0 / (rrf_k + rank + 1)

        # Search each sub-query
        for sq in sub_queries:
            try:
                sq_results = self._store.search(
                    sq,
                    limit=candidate_limit,
                    category_filter=category_filter,
                    project_id=project_id or None,
                    suppress_retrieval=True,
                )
            except Exception as exc:
                logger.debug("Sub-query search failed for '%s': %s", sq[:30], exc)
                continue

            for rank, r in enumerate(sq_results):
                if r.memory_id not in all_results:
                    all_results[r.memory_id] = r
                rrf_scores[r.memory_id] = rrf_scores.get(r.memory_id, 0.0) + 1.0 / (rrf_k + rank + 1)

        # Sort by RRF score
        merged = sorted(
            all_results.values(),
            key=lambda r: rrf_scores.get(r.memory_id, 0.0),
            reverse=True,
        )

        # Update similarity to RRF score (normalized to 0-1)
        max_score = max(rrf_scores.values()) if rrf_scores else 1.0
        for r in merged:
            r.similarity = rrf_scores.get(r.memory_id, 0.0) / max_score if max_score > 0 else 0.0

        return merged

    def _record_injected(self, records: List[Any]) -> None:
        """Record retrieval only for the final injected list (not pool filler).

        The store's search(suppress_retrieval=True) skips retrieval accounting
        for candidate-pool searches; the provider re-records here so only the
        memories actually injected into the conversation gain popularity credit.
        """
        try:
            if records and self._store is not None and hasattr(self._store, "record_retrieval"):
                self._store.record_retrieval([r.memory_id for r in records])
        except Exception as exc:
            logger.debug("Could not record injected retrieval: %s", exc)

    # -- chain-unfold (ships off; scaffolding for the Hy-Memory headline) -----

    # Change-intent patterns: queries that ask about HOW or WHY a fact changed.
    # When chain_unfold="auto", a top result with a chain + one of these
    # triggers a compact arc injection (budget-controlled).
    _CHANGE_INTENT_PATTERNS = (
        r"why did (i|you) (stop|start|switch|leave|change|quit|drop|abandon)",
        r"used to\b",
        r"what changed\b",
        r"before vs now\b",
        r"why.*no longer\b",
        r"when did (i|you) (change|switch|start|stop|leave|move)",
        r"how come (i|you) (don't|no longer|stopped|switched)",
        r"what did i (use to|used to) (think|believe|use|like|prefer)",
    )

    def _change_intent_match(self, query: str) -> bool:
        """True if the query signals change-intent (arc-relevant)."""
        import re
        q = query.lower()
        return any(re.search(p, q) for p in self._CHANGE_INTENT_PATTERNS)

    def _build_chain_arc(self, versions: List[Any]) -> str:
        """Compact one-line-per-version arc text (token-cheap)."""
        lines = []
        for i, v in enumerate(versions, 1):
            if getattr(v, "status", None) == "quarantined":
                lines.append(f"v{i} [quarantined]")
                continue
            content = v.content
            if len(content) > 120:
                content = content[:117] + "..."
            marker = " (current)" if v.valid_to is None else ""
            lines.append(f"v{i}{marker}: {content}")
        return "\n".join(lines)

    def _find_chain_anchor(self, results: List[Any], top_k: int) -> str | None:
        """Scan the top-K search results for the first one with a chain at
        >= the similarity floor. Returns the memory_id of the chain head, or
        None. The per-candidate floor is the precision guard — a chain only
        unfolds when the hit is genuinely about the query.
        """
        candidates = results[:top_k]
        if not candidates or self._store is None:
            return None
        try:
            membership = self._store.get_chain_membership(
                [r.memory_id for r in candidates]
            )
        except Exception:
            return None
        for r in candidates:
            raw = getattr(r, "raw_similarity", None)
            if raw is None:
                raw = getattr(r, "similarity", 0.0) or 0.0
            if raw < self._chain_unfold_min_similarity:
                continue
            info = membership.get(r.memory_id)
            if info and info.get("has_history"):
                return r.memory_id
        return None

    def _query_side_chain_lookup(self, query: str) -> str | None:
        """Fallback: search deeper for a chain matching the query.

        When change-intent matched but no top-K result has a chain, probe
        the store for a chain whose content is semantically close to the
        query (same 0.30 cosine floor). Uses suppress_retrieval=True so the
        deep search does NOT inflate retrieval counters. This is the
        "latest version exists but the semantic query didn't rank it in
        top-K" case.
        """
        if self._store is None:
            return None
        try:
            deep = self._store.search(
                query, limit=20, suppress_retrieval=True,
            )
        except Exception:
            return None
        if not deep:
            return None
        return self._find_chain_anchor(deep, len(deep))

    def _maybe_unfold_chain(self, query: str, results: List[Any]) -> str | None:
        """Chain-unfold trigger (budget-controlled, separate accounting).

        Returns a compact arc string to inject when chain_unfold is enabled,
        the query signals change-intent, a TOP-K result has a chain at
        sufficient similarity, and the arc cost is within budget. Returns
        None otherwise. Chain versions pulled by the walk do NOT touch
        retrieval counters — only the separate _chain_unfolded_stats
        counter is updated.

        Gate (measured 2026-08-13, eval_chain_unfold.py): the original
        top-3-any-chain gate fired on unrelated queries (weather query ->
        Chain A arc) and injected wrong arcs (Query X -> Chain Y arc).
        Tightened to: TOP-1 result only, raw_similarity >= 0.30 floor
        (same convention as query-expansion's floor), so a chain only
        unfolds when the top hit is genuinely about the query.

        Recall rebalance (2026-08-17): the top-1 gate was recall-starved
        (eval 100% precision / 20% recall — 4/5 misses were
        retrieval-driven: a real memory outranked the synthetic chain
        seed). Widened to scan TOP-K results (default K=3) for a chain
        anchor at >= 0.30, with an optional query-side fallback that
        searches deeper when no top-K result has a chain. The 0.30
        per-candidate floor is the precision guard — it targets exactly
        the measured failure class (chain ranked 2-4 behind a stronger
        real memory) without re-opening the false-trigger hole.
        """
        if self._chain_unfold == "off" or not results or self._store is None:
            return None
        if self._chain_unfold == "auto" and not self._change_intent_match(query):
            return None
        # Scan top-K results for a chain anchor at >= similarity floor.
        target_id = self._find_chain_anchor(results, self._chain_unfold_top_k)
        # Query-side fallback: if no anchor in top-K, search deeper for a
        # chain whose content is semantically close to the query. Catches
        # the "chain exists but didn't rank in top-K" case without
        # lowering the per-candidate similarity floor.
        if target_id is None and self._chain_unfold_query_fallback:
            target_id = self._query_side_chain_lookup(query)
        if target_id is None:
            return None
        try:
            versions = self._store.get_memory_history(
                target_id, max_versions=self._chain_max_versions,
            )
        except Exception:
            return None
        if len(versions) < 2:
            return None
        arc = self._build_chain_arc(versions)
        # Rough token estimate: ~4 chars/token.
        token_cost = max(1, len(arc) // 4)
        if token_cost > self._chain_max_inject:
            return None
        self._chain_unfolded_stats["count"] += 1
        self._chain_unfolded_stats["tokens_injected"] += token_cost
        return arc

    def get_chain_unfold_stats(self) -> Dict[str, int]:
        """Return chain-unfold accounting (count + tokens injected)."""
        return dict(self._chain_unfolded_stats)

    def get_scale_metrics(self) -> Dict[str, Any]:
        """Return current scale-trigger state (delegates to the store).

        The store owns the latency window and record-count sampling — it is
        the layer that actually executes search (both in-process and via the
        shared service), so its numbers are the ones the scaling roadmap's
        measured triggers gate on.
        """
        try:
            return dict(self._store.get_scale_metrics())
        except Exception:
            return {"error": "scale metrics unavailable"}

    def _search_memories(
        self,
        query: str,
        limit: int,
        category_filter: str | None = None,
        project_id: str | None = None,
    ) -> List[Any]:
        """Run hybrid search and apply a bounded graph-supported boost.

        When *project_id* is provided, memories from other projects are
        excluded. When None, the provider's current project scope is used.
        """
        if self._store is None:
            return []
        # Enrich the query with conversation context if it contains
        # pronouns/references that need resolution.
        effective_query = self._enrich_query_with_context(query)
        effective_project = project_id if project_id is not None else self._current_project_id
        candidate_limit = min(512, max(limit, limit * 4))
        results = self._store.search(
            effective_query,
            limit=candidate_limit,
            category_filter=category_filter,
            project_id=effective_project or None,
            suppress_retrieval=True,
        )

        # Query expansion: if the top hit's RAW similarity (pre-importance)
        # is below the similarity floor, ask the LLM to rewrite the query
        # into sub-queries and re-search.
        # This is lazy (only fires on weak results), cached, and fail-soft
        # (returns original results on any LLM failure).
        #
        # IMPORTANT: we gate on raw_similarity, NOT the final similarity.
        # The final similarity includes importance boosts (recency, retrieval
        # frequency) that contaminate the retrieval-strength signal. A memory
        # can score 1.5 on the adjusted scale but only 0.2 on raw retrieval
        # strength — that's the signal the gate needs.
        top_raw_sim = getattr(results[0], "raw_similarity", None) if results else 0.0
        if top_raw_sim is None:
            # Fallback for stub records without raw_similarity: use the
            # final similarity. This is the contaminated score but it's
            # the best we have for non-MemoryRecord results.
            top_raw_sim = results[0].similarity if results else 0.0
        if (
            self._query_expander
            and self._query_expander.enabled
            and results
            and self._query_expander.should_expand(query, top_raw_sim)
        ):
            results = self._expand_and_merge(
                query, effective_project, category_filter,
                candidate_limit, results,
            )
        elif (
            self._query_expander
            and self._query_expander.enabled
            and not results
            and self._query_expander.should_expand(query, 0.0)
        ):
            # No results at all — try expansion with floor=0
            results = self._expand_and_merge(
                query, effective_project, category_filter,
                candidate_limit, results,
            )

        if not self._graph or not self._graph_aware_retrieval:
            final_results = results[:limit]
            self._record_injected(final_results)
            return final_results
        try:
            # Entity alias resolution: expand the query with canonical
            # entity names for any aliases found in the query text.
            # Example: "tell me about my role" → also search for "Entity-A"
            alias_expansions: list[str] = []
            if hasattr(self._store, "resolve_aliases"):
                canonicals = self._store.resolve_aliases(effective_query)
                if canonicals:
                    alias_expansions = canonicals
                    logger.debug(
                        "Alias expansion: '%s' → %s",
                        effective_query[:50], alias_expansions,
                    )

            # Canonical→alias expansion: when the query mentions a canonical
            # entity name, also search for its aliases in the graph.
            # Example: "tell me about Entity-A" → also search for "my role"
            # so memories that say "my role" without naming Entity-A are found.
            alias_terms: list[str] = []
            if hasattr(self._store, "aliases_for_canonical"):
                for canonical in alias_expansions:
                    try:
                        aliases = self._store.aliases_for_canonical(canonical)
                        alias_terms.extend(aliases)
                    except Exception:
                        pass
                # Also check if the query itself contains a canonical name
                # that has aliases (even if no alias→canonical match fired)
                if not alias_terms:
                    for alias_map in (self._store.list_aliases() if hasattr(self._store, "list_aliases") else []):
                        canonical = alias_map.get("canonical_entity", "")
                        if canonical and canonical.lower() in effective_query.lower():
                            try:
                                aliases = self._store.aliases_for_canonical(canonical)
                                alias_terms.extend(aliases)
                            except Exception:
                                pass
                if alias_terms:
                    logger.debug(
                        "Canonical→alias expansion: '%s' → %s",
                        effective_query[:50], alias_terms,
                    )

            graph_ids = self._graph.memory_ids_for_query(
                effective_query, limit=max(10, candidate_limit)
            )
            # Traversal-based candidates: walk TYPED relations from seed
            # entities (hop-weighted BFS). These are graph-only candidates
            # eligible for injection under the same similarity floor as
            # alias-expanded IDs. Disabled unless graph_traversal_enabled
            # (config) — measured A/B gate.
            traversal_ids: list[str] = []
            if self._graph_traversal_enabled:
                try:
                    traversal_ids = self._graph.traversal_memory_ids(
                        effective_query, depth=self._graph_traversal_depth,
                        limit=max(10, candidate_limit),
                    )
                    logger.debug("TRAVERSAL-DBG: %d ids for %r", len(traversal_ids), effective_query[:40])
                    if traversal_ids:
                        seen = set(graph_ids)
                        for tid in traversal_ids:
                            if tid not in seen:
                                graph_ids.append(tid)
                                seen.add(tid)
                except Exception:
                    traversal_ids = []
            # Also query the graph for each canonical entity from aliases
            for canonical in alias_expansions:
                try:
                    extra_ids = self._graph.memory_ids_for_query(
                        canonical, limit=max(10, candidate_limit)
                    )
                    # Merge, preserving order (dedup)
                    seen = set(graph_ids)
                    for eid in extra_ids:
                        if eid not in seen:
                            graph_ids.append(eid)
                            seen.add(eid)
                except Exception:
                    pass
            # Also query the graph for each alias term (canonical→alias).
            # Track which IDs came from alias expansion specifically — these
            # are the only graph-only candidates eligible for injection, so
            # we don't re-introduce the noise regression that made
            # graph_inject_candidates=false necessary in the first place.
            alias_expanded_ids: list[str] = []
            for alias_term in alias_terms:
                try:
                    extra_ids = self._graph.memory_ids_for_query(
                        alias_term, limit=max(10, candidate_limit)
                    )
                    seen = set(graph_ids)
                    for eid in extra_ids:
                        if eid not in seen:
                            graph_ids.append(eid)
                            seen.add(eid)
                            alias_expanded_ids.append(eid)
                except Exception:
                    pass
            if graph_ids:
                existing = {record.memory_id for record in results}
                # Graph-only candidate injection. Two guards prevent noise:
                # 1. graph_inject_candidates must be true (the global gate),
                #    OR the candidate came from alias expansion specifically
                #    (the Ticket 1 path: "Entity-A" → "my role" → graph IDs).
                # 2. The candidate's semantic similarity to the query must
                #    clear graph_boost_min_similarity — a precision guard
                #    that stops unrelated graph neighbors from being injected.
                #    Records from get_memories_by_ids have no similarity
                #    computed, so we compute it here via the store's embedder.
                injectable_ids = set(alias_expanded_ids) if alias_expanded_ids else set()
                if self._graph_traversal_enabled:
                    injectable_ids.update(traversal_ids)
                    logger.debug("TRAVERSAL-DBG: injectable=%d (traversal=%d, alias=%d)",
                                 len(injectable_ids), len(traversal_ids), len(alias_expanded_ids))
                if self._graph_inject_candidates:
                    injectable_ids.update(graph_ids)
                if injectable_ids:
                    # Compute query embedding once for similarity scoring.
                    query_emb: List[float] = []
                    embedder = getattr(self._store, "embedder", None)
                    if embedder and hasattr(embedder, "embed"):
                        try:
                            query_emb = embedder.embed(effective_query, is_query=True)
                        except Exception:
                            query_emb = []
                    graph_records = self._store.get_memories_by_ids(
                        list(injectable_ids)
                    )
                    for record in graph_records:
                        if record.memory_id in existing:
                            continue
                        # Compute cosine similarity if we have embeddings;
                        # otherwise fall back to the record's existing
                        # similarity (set by get_memories_by_ids or a
                        # prior search path).
                        sim = 0.0
                        if query_emb and getattr(record, "embedding", None):
                            try:
                                import math
                                dot = sum(a * b for a, b in zip(query_emb, record.embedding))
                                norm_q = math.sqrt(sum(a * a for a in query_emb))
                                norm_r = math.sqrt(sum(b * b for b in record.embedding))
                                if norm_q > 0 and norm_r > 0:
                                    sim = dot / (norm_q * norm_r)
                            except Exception:
                                sim = 0.0
                        elif hasattr(record, "similarity"):
                            sim = record.similarity
                        record.similarity = sim
                        record.raw_similarity = sim
                        if sim >= self._graph_boost_min_similarity:
                            results.append(record)
                            if str(record.memory_id).startswith("mem-4d83cf06"):
                                logger.debug("TRAVERSAL-DBG: INDWE injected sim=%.3f", sim)
                        elif str(record.memory_id).startswith("mem-4d83cf06"):
                            logger.debug("TRAVERSAL-DBG: INDWE rejected sim=%.3f < %.2f",
                                         sim, self._graph_boost_min_similarity)
                graph_rank = {memory_id: rank for rank, memory_id in enumerate(graph_ids)}
                graph_count = max(len(graph_ids), 1)
                alias_id_set = set(alias_expanded_ids)
                traversal_id_set = set(traversal_ids)
                for record in results:
                    # Alias-expanded candidates: alias expansion is a
                    # definitive identity mapping (e.g. "my role" =
                    # "Entity-A"), not a fuzzy graph neighbor.  The raw
                    # embedding similarity is low only because the memory
                    # text doesn't contain the query word — but semantically
                    # it IS about the query entity.  Floor the similarity
                    # so high-similarity candidates are unaffected and low-
                    # similarity ones are lifted above the cutoff.
                    if record.memory_id in alias_id_set:
                        record.similarity = max(
                            record.similarity, self._alias_expansion_boost
                        )
                        continue
                    # Traversal candidates: memory reached by walking TYPED
                    # relations from a query seed entity. Evidence is
                    # relational (e.g. Indwe broker <- uses <- user with car
                    # finance) — semantically meaningful even if the surface
                    # text doesn't overlap. Floor lifts it above the cutoff
                    # without a full identity claim (alias-level).
                    if (self._graph_traversal_enabled
                            and record.memory_id in traversal_id_set):
                        record.similarity = max(
                            record.similarity, self._graph_traversal_boost
                        )
                        continue
                    rank = graph_rank.get(record.memory_id)
                    if rank is None:
                        continue
                    if record.similarity < self._graph_boost_min_similarity:
                        continue
                    decay = 1.0 - (rank / graph_count)
                    record.similarity += self._graph_retrieval_boost * max(0.0, decay)
                results.sort(key=lambda record: record.similarity, reverse=True)
        except Exception as exc:
            logger.debug("Graph-aware retrieval failed: %s", exc)
        final_results = results[:limit]
        self._record_injected(final_results)
        return final_results

    # -- prefetch (auto-inject context before each turn) ---------------------

    def on_turn_start(self, turn_number: int, message: str, **kwargs) -> None:
        self._record_user_message(message)
        self._start_prefetch(message)

    def _start_prefetch(self, query: str) -> None:
        if not query or self._store is None:
            return
        store = self._store
        max_items = self._max_injected

        with self._prefetch_lock:
            if self._prefetch_query == query:
                if self._prefetch_done or (self._prefetch_thread and self._prefetch_thread.is_alive()):
                    return
            self._prefetch_query = query
            self._prefetch_result = ""
            self._prefetch_done = False

        def _run() -> None:
            sections = []
            body = ""
            try:
                confirmation_candidates = store.list_candidates(
                    status="pending_user_confirmation", limit=1
                )
                confirmation = build_confirmation_block(confirmation_candidates)
                if confirmation:
                    sections.append(confirmation)

                results = self._search_memories(query, limit=max_items)
                if results:
                    lines = []
                    for r in results:
                        cat = r.category
                        content = r.content
                        if len(content) > _INJECT_CONTENT_CHAR_CAP:
                            content = content[:_INJECT_CONTENT_CHAR_CAP].rsplit(" ", 1)[0] + "..."
                        sim = f" (score: {r.similarity:.2f})" if r.similarity > 0 else ""
                        date = (r.created_at or "")[:10]
                        date_s = f"[{date}] " if date else ""
                        lines.append(f"- {date_s}[{cat}] {content}{sim}")
                    sections.append("## Recalled Memories\n" + "\n".join(lines))
                body = "\n\n".join(sections)
            except Exception as e:
                logger.debug("Prefetch failed: %s", e)
            with self._prefetch_lock:
                if self._prefetch_query == query:
                    self._prefetch_result = body
                    self._prefetch_done = True

        t = threading.Thread(target=_run, daemon=True, name="hybrid-prefetch")
        with self._prefetch_lock:
            self._prefetch_thread = t
        t.start()

    def _consume_prefetch_result(self, query: str) -> str | None:
        with self._prefetch_lock:
            if self._prefetch_query != query or not self._prefetch_done:
                return None
            result = self._prefetch_result
            self._prefetch_result = ""
            self._prefetch_done = False
            return result

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        cached = self._consume_prefetch_result(query)
        if cached is not None:
            return cached
        self._start_prefetch(query)
        with self._prefetch_lock:
            thread = self._prefetch_thread if self._prefetch_query == query else None
        if thread:
            thread.join(timeout=_PREFETCH_WAIT_SECS)
        cached = self._consume_prefetch_result(query)
        if cached is not None:
            return cached
        return ""

    def _review_candidate(self, candidate: Dict[str, Any]) -> None:
        """Run the conservative automatic reviewer for one new proposal."""
        try:
            review = review_candidate_with_llm(
                candidate, model=self._llm_model, provider=self._llm_provider,
            )
            decision = review.get("decision", "pending_user_confirmation")
            decision_map = {
                "approve": "approved",
                "reject": "rejected",
                "quarantine": "quarantined",
                "pending_user_confirmation": "pending_user_confirmation",
            }
            final_status = decision_map.get(decision, "pending_user_confirmation")
            result = self._store.review_candidate(
                evidence_retention=getattr(self, "_evidence_retention", "full"),
                candidate_id=candidate["candidate_id"],
                decision=final_status,
                reason=review.get("reason", ""),
                review_confidence=review.get("confidence"),
                review_model=review.get("review_model", "memory_review"),
                durability=review.get("durability"),
                scope=review.get("scope"),
            )
            # Index the promoted memory in the graph. This closes the gap
            # where auto-approved candidates never reached the graph.
            if result and result.get("memory"):
                mem = result["memory"]
                self._index_memory_graph(
                    mem.get("memory_id", ""),
                    mem.get("category", "context_note"),
                    mem.get("content", ""),
                    mem.get("tags", []),
                    mem.get("created_at"),
                )
        except Exception as exc:
            logger.warning("Automatic memory proposal review failed: %s", exc)

    # -- sync_turn (auto-extract after each turn) ----------------------------

    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        # Skip writes for non-primary contexts (cron, subagent, flush).
        if self._agent_context != "primary":
            return
        if not self._auto_extract or self._auto_extract_paused:
            return
        if self._store is None:
            return

        # Enqueue work for the persistent worker. If the queue is full,
        # drop the oldest pending item to make room for the latest turn.
        item = (user_content, assistant_content, session_id)
        try:
            self._sync_queue.put_nowait(item)
        except queue.Full:
            self._sync_dropped_turns += 1
            try:
                dropped = self._sync_queue.get_nowait()
                self._sync_queue.task_done()
                dropped_preview = (dropped[0] or "")[:80]
                logger.warning(
                    "Sync queue full; dropping oldest pending turn to make room "
                    "(preview: %r). Extraction/review is bounded to prevent "
                    "unbounded backlog under sustained load.",
                    dropped_preview,
                )
            except queue.Empty:
                pass
            try:
                self._sync_queue.put_nowait(item)
            except queue.Full:
                logger.warning(
                    "Sync queue still full after drop; skipping extraction for "
                    "this turn (preview: %r).",
                    (user_content or "")[:80],
                )
                return
        self._ensure_sync_worker()

    def _ensure_sync_worker(self) -> None:
        """Start the persistent sync worker if it isn't running."""
        with self._sync_lock:
            if self._sync_worker_started and self._sync_thread and self._sync_thread.is_alive():
                return
            self._sync_worker_started = True
            self._sync_thread = threading.Thread(
                target=self._sync_worker_loop, daemon=True, name="hybrid-sync"
            )
            self._sync_thread.start()

    def _sync_worker_loop(self) -> None:
        """Process extraction/review items from the bounded queue."""
        while True:
            try:
                item = self._sync_queue.get(timeout=10.0)
            except queue.Empty:
                # Idle for 10s; check if we should exit.
                with self._sync_lock:
                    if self._sync_queue.empty():
                        self._sync_worker_started = False
                        return
                continue
            if item is None:
                # Sentinel to stop the worker.
                self._sync_queue.task_done()
                with self._sync_lock:
                    self._sync_worker_started = False
                return
            user_content, assistant_content, session_id = item
            try:
                facts = extract_from_turn(
                    user_content, assistant_content,
                    use_llm_fallback=self._llm_fallback,
                    llm_model=self._llm_model,
                    llm_provider=self._llm_provider,
                    shadow_diff=self._extraction_shadow_diff,
                )
                proposed = 0
                for fact in facts:
                    payload = dict(fact.get("payload") or {})
                    candidate = self._store.save_candidate(
                        category=fact["category"],
                        content=fact["content"],
                        tags=fact.get("tags", []),
                        payload=payload,
                        source=fact.get("source", payload.get("source", "regex_extraction")),
                        confidence=fact.get("confidence", 0.45),
                        durability=fact.get("durability", "durable"),
                        scope=fact.get("scope", "profile"),
                        project_id=fact.get("project_id"),
                        session_id=session_id,
                        evidence_text=user_content,
                        evidence_role="user_turn",
                        dedup=True,
                    )
                    if candidate:
                        proposed += 1
                        if self._auto_review:
                            self._review_candidate(candidate)
                if proposed:
                    logger.info(
                        "Created %d pending memory proposals; no automatic active writes",
                        proposed,
                    )
            except Exception as e:
                logger.warning("Sync turn proposal failed: %s", e)
            finally:
                self._sync_queue.task_done()

    # -- session end ---------------------------------------------------------

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        # Purge junk graph entities at session end (cheap maintenance).
        if self._graph:
            try:
                deleted = self._graph.purge_junk_entities()
                if deleted:
                    logger.debug("Purged %d junk graph entities at session end", deleted)
            except Exception as e:
                logger.debug("Junk purge failed: %s", e)
        # Quarantine questionable memories at session end; never delete silently.
        if self._store:
            try:
                cleanup_result = self._store.cleanup_junk(return_ids=True)
                quarantined = int(cleanup_result.get("count", 0)) if isinstance(cleanup_result, dict) else int(cleanup_result)
                if quarantined:
                    logger.info("Quarantined %d questionable memories at session end", quarantined)
                    for memory_id in cleanup_result.get("memory_ids", []) if isinstance(cleanup_result, dict) else []:
                        try:
                            self._graph.remove_memory(memory_id)
                        except Exception:
                            pass
            except Exception as e:
                logger.debug("Memory quarantine failed: %s", e)
            if self._consolidation_enabled:
                try:
                    report = self._store.consolidate(
                        dry_run=False,
                        max_actions=self._consolidation_max_actions,
                        min_age_days=self._consolidation_min_age_days,
                    )
                    if report.get("quarantined_count"):
                        logger.info(
                            "Consolidation quarantined %d memories",
                            report["quarantined_count"],
                        )
                        for memory_id in report.get("quarantined_ids", []):
                            if not self._graph:
                                break
                            try:
                                self._graph.remove_memory(memory_id)
                            except Exception:
                                pass
                except Exception as e:
                    logger.debug("Consolidation failed: %s", e)

    # -- session switch ------------------------------------------------------

    def on_session_switch(
        self,
        new_session_id: str,
        *,
        parent_session_id: str = "",
        reset: bool = False,
        rewound: bool = False,
        **kwargs,
    ) -> None:
        project_id = str(kwargs.get("project_id") or "").strip()
        if project_id != self._current_project_id:
            self._current_project_id = project_id
        if reset:
            with self._prefetch_lock:
                self._prefetch_query = ""
                self._prefetch_result = ""
                self._prefetch_done = False

    # -- tools ---------------------------------------------------------------

    def _index_memory_graph(
        self,
        memory_id: str,
        category: str,
        content: str,
        tags: List[str] | None = None,
        created_at: str | None = None,
    ) -> None:
        """Index any saved memory in the graph, regardless of category.

        Uses regex-first, LLM-supplemented extraction. The LLM path only
        fires when regex finds few relations and content is substantial,
        so short memories don't pay the token cost.

        Also extracts role→canonical-name aliases at index time: when a
        memory contains "my role is Entity-A" or "Role is Entity-A", writes
        add_alias("my role", "Entity-A") so that searching for "Entity-A"
        also finds memories that say "my role" without naming Entity-A.
        """
        graph = getattr(self, "_graph", None)
        if not graph or not memory_id or not content:
            return
        try:
            graph.index_memory(
                memory_id=memory_id,
                category=category,
                content=content,
                tags=tags or [],
                created_at=created_at,
                use_llm=self._llm_fallback,
            )
        except Exception as exc:
            # Graph indexing is an enrichment path; a graph failure must not
            # make a successful memory write fail.
            logger.debug("Graph indexing failed for memory %s: %s", memory_id, exc)

        # Index-time alias expansion: extract role→canonical-name mappings
        # from the same content and write aliases so both directions work:
        #   alias→canonical (query "my role" → matches Entity-A) [already works]
        #   canonical→alias (query "Entity-A" → matches "my role") [NEW]
        self._extract_role_aliases(content, tags or [])

    def _extract_role_aliases(
        self, content: str, tags: List[str] | None = None
    ) -> None:
        """Extract role→canonical-person aliases from memory content.

        When a memory contains a pattern like "my role is Entity-A" or
        "Role is Entity-A", writes add_alias("my role", "Entity-A") so that:
        - searching "my role" resolves to Entity-A (alias→canonical)
        - searching "Entity-A" also finds memories mentioning "my role"
          (canonical→alias, via aliases_for_canonical at query time)

        Guards against over-minting:
        - Only fires on has_<role> relations with a person target
        - The canonical name must start with a capital letter (a real name,
          not a verb/adjective — same guard as the bare_role pattern)
        - Idempotent: INSERT OR REPLACE in add_alias
        """
        store = getattr(self, "_store", None)
        if not store or not content:
            return
        try:
            from graph import extract_graph_relations, _GRAPH_RELATIONSHIP_WORDS
            relations = extract_graph_relations(content, "context_note", tags)
            for rel in relations:
                relation = rel.get("relation", "")
                if not relation.startswith("has_"):
                    continue
                if rel.get("target_type") != "person":
                    continue
                role = relation[4:]  # e.g. "role", "contact", "doc"
                canonical = rel.get("target", "")
                # Guard: canonical must be a real name (capital letter)
                if not canonical or not canonical[0].isupper():
                    continue
                # Skip if the "name" is actually a role word capitalized
                # (e.g. "Role" — not a person name)
                if canonical.lower() in _GRAPH_RELATIONSHIP_WORDS:
                    continue
                alias = f"my {role}"
                store.add_alias(alias, canonical)
                logger.debug(
                    "Index-time alias: %r -> %r (from: %s)",
                    alias, canonical, content[:60],
                )
        except Exception as exc:
            logger.debug("Role-alias extraction failed: %s", exc)

    def _try_graph_relationship(self, content: str) -> None:
        """Attempt to extract a relationship from content text and add to graph.

        Retained for compatibility with older callers; new memory writes use
        _index_memory_graph() so every category participates in the graph.
        """
        if not self._graph:
            return
        try:
            from .extractor import _RELATIONSHIP_RE
            m = _RELATIONSHIP_RE.search(content)
            if m:
                name = m.group(1).strip()
                relation = m.group(2).strip().lower()
                if name.lower() not in ("this", "that", "it", "there", "here"):
                    self._graph.add_relationship(
                        source="user", source_type="person",
                        relation=f"has_{relation}",
                        target=name, target_type="person",
                    )
                    return
            # Fallback: "X is the user's Y" pattern (from auto-extractor output).
            import re
            m2 = re.search(r"(\w+)\s+is\s+the\s+user'?s?\s+(\w+)", content, re.IGNORECASE)
            if m2:
                name = m2.group(1).strip()
                relation = m2.group(2).strip().lower()
                self._graph.add_relationship(
                    source="user", source_type="person",
                    relation=f"has_{relation}",
                    target=name, target_type="person",
                )
        except Exception:
            pass

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        schemas = [
            SEARCH_SCHEMA,
            SAVE_SCHEMA,
            UPDATE_SCHEMA,
            DELETE_SCHEMA,
            CANDIDATE_LIST_SCHEMA,
            CANDIDATE_REVIEW_SCHEMA,
            RESTORE_SCHEMA,
            FEEDBACK_SCHEMA,
            MAINTENANCE_SCHEMA,
            CHAIN_SCHEMA,
        ]
        # Always include graph tool schemas so they are registered in the
        # MemoryManager routing table at add_provider() time — before
        # initialize() connects the Kùzu graph store. If the graph is not
        # available at call time, handle_tool_call returns a clear error.
        schemas.extend([GRAPH_SEARCH_SCHEMA, GRAPH_QUERY_SCHEMA])
        return schemas

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        if self._store is None:
            return tool_error("Memory store not initialized")

        if tool_name == "memory_search":
            query = args.get("query", "")
            if not query:
                return tool_error("Missing required parameter: query")
            try:
                top_k = max(1, min(int(args.get("top_k", self._max_injected)), 50))
            except (ValueError, TypeError):
                top_k = self._max_injected
            category = args.get("category")
            if category and category not in VALID_CATEGORIES:
                return tool_error(f"Invalid category. Valid: {', '.join(sorted(VALID_CATEGORIES))}")
            project_id = args.get("project_id")
            results = self._search_memories(
                query, limit=top_k, category_filter=category,
                project_id=project_id,
            )
            # Chain-presence annotation: a light, batched marker on each hit
            # so the agent KNOWS a fact has a history (trigger to call
            # memory_chain when the conversation is about change). Does NOT
            # unfold chains — annotation only. Fail-soft: any store error
            # leaves results intact with no chain field.
            result_payloads = [r.to_dict() for r in results]
            if results and hasattr(self._store, "get_chain_membership"):
                try:
                    membership = self._store.get_chain_membership(
                        [r.memory_id for r in results]
                    )
                    for payload in result_payloads:
                        mid = payload.get("memory_id", "")
                        if mid in membership:
                            payload["chain"] = membership[mid]
                except Exception as exc:
                    logger.debug("Chain membership annotation failed: %s", exc)
            # Chain-unfold: when enabled (off|auto|always), inject a compact
            # arc for a top result with history on change-intent queries.
            # Fail-soft: any error leaves the response unchanged. The arc
            # rides in a separate "chain_arc" field so the agent can use it
            # without conflating it with search results.
            try:
                arc = self._maybe_unfold_chain(query, results)
                if arc:
                    result_payloads.append({"chain_arc": arc})
            except Exception as exc:
                logger.debug("Chain unfold failed: %s", exc)
            return json.dumps({
                "query": query,
                "count": len(results),
                "results": result_payloads,
            })

        elif tool_name == "memory_save":
            content = args.get("content", "")
            category = args.get("category", "context_note")
            if not content:
                return tool_error("Missing required parameter: content")
            if category not in VALID_CATEGORIES:
                return tool_error(f"Invalid category. Valid: {', '.join(sorted(VALID_CATEGORIES))}")
            tags = args.get("tags", [])
            rec = self._store.remember(category=category, content=content, tags=tags, dedup=True)
            if rec is None:
                return json.dumps({"status": "deduplicated", "message": "Similar memory already exists"})
            # Index every memory category; entity links are additive and
            # graph failures do not affect the successful memory write.
            self._index_memory_graph(
                rec.memory_id,
                rec.category,
                rec.content,
                rec.tags,
                rec.created_at,
            )
            return json.dumps({"status": "saved", "memory_id": rec.memory_id})

        elif tool_name == "memory_update":
            memory_id = args.get("memory_id", "")
            if not memory_id:
                return tool_error("Missing required parameter: memory_id")
            content = args.get("content")
            tags = args.get("tags")
            # memory_id must be passed as a keyword: SharedMemoryStore.update_memory
            # is keyword-only (def update_memory(self, **kwargs)) over the shared
            # service path. Passing it positionally raises TypeError on the live
            # memory_update tool path (the direct DuckDBMemoryStore path accepted
            # positional args, so store-level tests missed this).
            rec = self._store.update_memory(memory_id=memory_id, content=content, tags=tags)
            if rec is None:
                return tool_error(f"Memory not found: {memory_id}")
            if getattr(self, "_graph", None):
                # update_memory creates a new version: rec.memory_id is the
                # NEW ID, but the graph was indexed against the OLD memory_id.
                # Remove the old ID from the graph (it's now superseded and
                # resolves to 0 records), then index the new version.
                try:
                    self._graph.remove_memory(memory_id)
                except Exception as exc:
                    logger.debug("Graph evidence cleanup failed for %s: %s", memory_id, exc)
                self._index_memory_graph(
                    rec.memory_id,
                    rec.category,
                    rec.content,
                    rec.tags,
                    rec.created_at,
                )
            return json.dumps({"status": "updated", "memory_id": rec.memory_id})

        elif tool_name == "memory_delete":
            memory_id = args.get("memory_id", "")
            if not memory_id:
                return tool_error("Missing required parameter: memory_id")
            result = self._store.delete_memory(memory_id=memory_id)
            if not result:
                return tool_error(f"Memory not found: {memory_id}")
            if getattr(self, "_graph", None):
                try:
                    self._graph.remove_memory(memory_id)
                except Exception as exc:
                    logger.debug("Graph evidence cleanup failed for %s: %s", memory_id, exc)
                # Head deletion promotes the predecessor to current —
                # re-index it so the graph points at the live version.
                promoted = None
                if isinstance(result, dict):
                    promoted = result.get("promoted_memory_id")
                if promoted:
                    restored = self._store.get_memories_by_ids([promoted])
                    if restored:
                        self._index_memory_graph(
                            restored[0].memory_id,
                            restored[0].category,
                            restored[0].content,
                            restored[0].tags,
                            restored[0].created_at,
                        )
            action = result.get("action", "deleted") if isinstance(result, dict) else "deleted"
            return json.dumps({"status": action, "memory_id": memory_id, "result": result})

        elif tool_name == "memory_candidate_list":
            status = args.get("status", "pending")
            if status not in {
                "pending", "reviewed_approved", "pending_user_confirmation",
                "approved", "rejected", "quarantined", "deduplicated",
            }:
                return tool_error("Invalid candidate status")
            try:
                limit = max(1, min(int(args.get("limit", 20)), 100))
            except (ValueError, TypeError):
                limit = 20
            candidates = self._store.list_candidates(status=status, limit=limit)
            return json.dumps({"status": status, "count": len(candidates), "candidates": candidates})

        elif tool_name == "memory_candidate_review":
            candidate_id = args.get("candidate_id", "")
            decision = args.get("decision", "")
            if not candidate_id or not decision:
                return tool_error("Missing required parameter: candidate_id or decision")
            supersedes_memory_id = args.get("supersedes_memory_id")
            try:
                result = self._store.review_candidate(
                    evidence_retention=getattr(self, "_evidence_retention", "full"),
                    candidate_id=candidate_id,
                    decision=decision,
                    reason=args.get("reason", ""),
                    supersedes_memory_id=supersedes_memory_id,
                )
            except ValueError as exc:
                return tool_error(str(exc))
            if result is None:
                return tool_error(f"Candidate not found: {candidate_id}")
            memory = result.get("memory")
            if memory and getattr(self, "_graph", None):
                self._index_memory_graph(
                    memory.get("memory_id", ""),
                    memory.get("category", "context_note"),
                    memory.get("content", ""),
                    memory.get("tags", []),
                    memory.get("created_at"),
                )
                # approve-with-supersede: the old (now-superseded) memory_id
                # must be removed from the graph (it resolves to 0 records
                # post-supersession), mirroring the memory_update graph path.
                if supersedes_memory_id and result.get("superseded"):
                    try:
                        self._graph.remove_memory(supersedes_memory_id)
                    except Exception as exc:
                        logger.debug(
                            "Graph cleanup for superseded %s failed: %s",
                            supersedes_memory_id, exc,
                        )
            return json.dumps(result)

        elif tool_name == "memory_restore":
            memory_id = args.get("memory_id", "")
            if not memory_id:
                return tool_error("Missing required parameter: memory_id")
            if not self._store.restore_memory(memory_id):
                return tool_error(f"Memory not found: {memory_id}")
            if self._graph:
                restored = self._store.get_memories_by_ids([memory_id])
                if restored:
                    self._index_memory_graph(
                        restored[0].memory_id,
                        restored[0].category,
                        restored[0].content,
                        restored[0].tags,
                        restored[0].created_at,
                    )
            return json.dumps({"status": "restored", "memory_id": memory_id})

        elif tool_name == "memory_feedback":
            memory_id = args.get("memory_id", "")
            feedback = args.get("feedback", "")
            if not memory_id or not feedback:
                return tool_error("Missing required parameter: memory_id or feedback")
            try:
                recorded = self._store.record_feedback(memory_id, feedback)
            except ValueError as exc:
                return tool_error(str(exc))
            if not recorded:
                return tool_error(f"Memory not found: {memory_id}")
            if feedback == "incorrect" and self._graph:
                try:
                    self._graph.remove_memory(memory_id)
                except Exception as exc:
                    logger.debug("Graph cleanup failed for incorrect memory %s: %s", memory_id, exc)
            return json.dumps({"status": "recorded", "memory_id": memory_id, "feedback": feedback})

        elif tool_name == "memory_maintenance":
            try:
                max_actions = max(1, min(int(args.get("max_actions", self._consolidation_max_actions)), 500))
            except (TypeError, ValueError):
                max_actions = self._consolidation_max_actions
            try:
                min_age_days = max(1, int(args.get("min_age_days", self._consolidation_min_age_days)))
            except (TypeError, ValueError):
                min_age_days = self._consolidation_min_age_days
            dry_run = args.get("dry_run", True)
            if isinstance(dry_run, str):
                dry_run = dry_run.lower() not in {"false", "0", "no"}
            report = self._store.consolidate(
                dry_run=bool(dry_run),
                max_actions=max_actions,
                min_age_days=min_age_days,
            )
            if not dry_run and self._graph:
                for memory_id in report.get("quarantined_ids", []):
                    try:
                        self._graph.remove_memory(memory_id)
                    except Exception as exc:
                        logger.debug("Graph cleanup failed for %s: %s", memory_id, exc)
            return json.dumps(report)

        elif tool_name == "memory_chain":
            memory_id = args.get("memory_id", "")
            if not memory_id:
                return tool_error("Missing required parameter: memory_id")
            mode = args.get("mode", "arc")
            if mode not in {"arc", "versions", "diff"}:
                return tool_error("Invalid mode. Valid: arc, versions, diff")
            try:
                max_versions = max(1, min(int(args.get("max_versions", 5)), 10))
            except (TypeError, ValueError):
                max_versions = 5
            versions = self._store.get_memory_history(
                memory_id, max_versions=max_versions,
            )
            if not versions:
                return tool_error(f"No memory chain found for: {memory_id}")
            # Batched evidence join (retention-aware: hash→digest, none→absent).
            try:
                evidence_map = self._store.get_evidence_batch(
                    [v.memory_id for v in versions]
                )
            except Exception:
                evidence_map = {}
            if mode == "versions":
                return json.dumps({
                    "memory_id": memory_id,
                    "mode": "versions",
                    "count": len(versions),
                    "versions": [
                        {**v.to_dict(), "evidence": evidence_map.get(v.memory_id)}
                        for v in versions
                    ],
                })
            if mode == "diff":
                steps = []
                for i in range(len(versions)):
                    cur = versions[i]
                    step = {
                        "version": i + 1,
                        "memory_id": cur.memory_id,
                        "valid_from": cur.valid_from,
                        "valid_to": cur.valid_to,
                        "category": cur.category,
                        "content": cur.content,
                        "tags": cur.tags,
                        "evidence": evidence_map.get(cur.memory_id),
                    }
                    if i > 0:
                        prev = versions[i - 1]
                        content_delta = list(difflib.unified_diff(
                            prev.content.splitlines(keepends=True) or [prev.content],
                            cur.content.splitlines(keepends=True) or [cur.content],
                            fromfile=f"v{i}", tofile=f"v{i+1}", lineterm="",
                        ))
                        changes = {"content_diff": content_delta}
                        if prev.category != cur.category:
                            changes["category"] = f"{prev.category} → {cur.category}"
                        if set(prev.tags) != set(cur.tags):
                            changes["tags"] = {
                                "added": sorted(set(cur.tags) - set(prev.tags)),
                                "removed": sorted(set(prev.tags) - set(cur.tags)),
                            }
                        step["changes_from_previous"] = changes
                    steps.append(step)
                return json.dumps({
                    "memory_id": memory_id, "mode": "diff",
                    "count": len(versions), "steps": steps,
                })
            # arc mode (default): compact one-line-per-version rendering.
            # Quarantined versions are marked as a gap (never break the walk).
            arc_lines = []
            for i, v in enumerate(versions, 1):
                if v.status == "quarantined":
                    arc_lines.append(f"v{i} [{v.valid_from or '?'}] [quarantined]")
                    continue
                content = v.content
                if len(content) > 120:
                    content = content[:117] + "..."
                marker = " (current)" if v.valid_to is None else ""
                arc_lines.append(
                    f"v{i} [{v.valid_from or '?'}]{marker} ({v.category}): {content}"
                )
            return json.dumps({
                "memory_id": memory_id, "mode": "arc",
                "count": len(versions),
                "arc": "\n".join(arc_lines),
                "versions": [
                    {"memory_id": v.memory_id, "valid_from": v.valid_from,
                     "valid_to": v.valid_to, "category": v.category,
                     "content": v.content,
                     "evidence": evidence_map.get(v.memory_id)}
                    for v in versions
                ],
            })

        elif tool_name == "memory_graph_search":
            if self._graph is None:
                return tool_error("Relationship graph is not available")
            term = args.get("term", "")
            if not term:
                return tool_error("Missing required parameter: term")
            edges = self._graph.search_graph(term)
            return json.dumps({"term": term, "count": len(edges), "edges": edges})

        elif tool_name == "memory_graph_query":
            if self._graph is None:
                return tool_error("Relationship graph is not available")
            entity_id = args.get("entity_id", "")
            if not entity_id:
                return tool_error("Missing required parameter: entity_id")
            try:
                depth = max(1, min(int(args.get("depth", 2)), 4))
                limit = max(1, min(int(args.get("limit", 100)), 250))
            except (TypeError, ValueError):
                depth, limit = 2, 100
            result = self._graph.traverse_graph(entity_id, depth=depth, limit=limit)
            return json.dumps(result)

        return tool_error(f"Unknown tool: {tool_name}")

    # -- backup paths --------------------------------------------------------

    def backup_paths(self) -> List[str]:
        """Return extra on-disk paths OUTSIDE HERMES_HOME for `hermes backup`.

        All our databases live inside HERMES_HOME, so `hermes backup` already
        captures them. Return empty list — no external paths to declare.
        """
        return []

    # -- shutdown ------------------------------------------------------------

    def shutdown(self) -> None:
        # Signal the sync worker to stop and wait for it.
        try:
            self._sync_queue.put_nowait(None)
        except queue.Full:
            pass  # worker will exit on idle timeout
        for t in (self._prefetch_thread, self._sync_thread):
            if t and t.is_alive():
                t.join(timeout=5.0)
        if self._store:
            try:
                self._store.close()
            except Exception:
                pass
        if self._graph:
            try:
                self._graph.close()
            except Exception:
                pass
        self._initialized = False


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
        "salesdash", "miser", "codebrain", "hybrid-memory", "hybrid_memory",
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
        return {}
    return {"context": "\n".join(parts)}


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------

def register(ctx) -> None:
    """Register HybridMemory as a memory provider plugin.

    Also registers the insight-log skill, /ilog + /revisit slash commands,
    and a ``pre_llm_call`` hook for ambient context (time/location/weather/
    file-activity) — if the plugin context supports them.
    """
    try:
        ctx.register_memory_provider(HybridMemoryProvider())
        logging.getLogger("hybrid_memory").info(
            "hybrid_memory: register() succeeded, provider registered"
        )
    except Exception as _e:
        logging.getLogger("hybrid_memory").warning(
            "hybrid_memory: register() failed: %s\n%s", _e, traceback.format_exc()
        )
        raise

    # Register the pre_llm_call hook for ambient context (time/location/
    # weather/file-activity).  Rides the native plugin_user_context injection
    # path — no core source patch needed.
    try:
        if hasattr(ctx, "register_hook"):
            ctx.register_hook("pre_llm_call", _on_pre_llm_call)
            logging.getLogger("hybrid_memory").info(
                "hybrid_memory: registered pre_llm_call hook for ambient context"
            )
    except Exception as _e:
        logging.getLogger("hybrid_memory").debug(
            "hybrid_memory: pre_llm_call hook registration skipped: %s", _e
        )

    # Register the insight-log skill (if the context supports skill registration).
    try:
        skill_md = Path(__file__).parent / "skills" / "insight-log" / "SKILL.md"
        if skill_md.exists() and hasattr(ctx, "register_skill"):
            ctx.register_skill("insight-log", skill_md)
            logging.getLogger("hybrid_memory").info(
                "hybrid_memory: registered insight-log skill"
            )
    except Exception as _e:
        logging.getLogger("hybrid_memory").debug(
            "hybrid_memory: skill registration skipped: %s", _e
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
            logging.getLogger("hybrid_memory").info(
                "hybrid_memory: registered /ilog and /revisit commands"
            )
    except Exception as _e:
        logging.getLogger("hybrid_memory").debug(
            "hybrid_memory: command registration skipped: %s", _e
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
        return SharedMemoryStore(home, user_id="default_user")
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

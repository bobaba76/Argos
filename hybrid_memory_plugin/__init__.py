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

import json
import logging
import os
import threading
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error

from .embeddings import LocalEmbedder
from .store import DuckDBMemoryStore, VALID_CATEGORIES
from .graph import KuzuGraphStore
from .confirmation import build_confirmation_block
from .extractor import extract_from_turn
from .routing import resolve_storage_names
from .service_client import SharedGraphStore, SharedMemoryStore
from .reviewer import review_candidate_with_llm

logger = logging.getLogger(__name__)

_PREFETCH_WAIT_SECS = 3.0
_DEFAULT_MAX_INJECTED = 8
_DEFAULT_MODEL = "sentence-transformers/bge-small-en-v1.5"
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
        "auto_review": "true",
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
        "relate (e.g., 'who is Pat' or 'what tools does the user use'). "
        "Returns edges showing source -> relation -> target."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "term": {"type": "string", "description": "Entity name to search for (e.g., 'Pat', 'FocusTool')."},
        },
        "required": ["term"],
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
        "durable, about the user, and scoped appropriately."
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
        self._auto_review: bool = True
        self._auto_extract_paused: bool = False
        self._initialized: bool = False
        # Prefetch state.
        self._prefetch_thread: Optional[threading.Thread] = None
        self._prefetch_query: str = ""
        self._prefetch_result: str = ""
        self._prefetch_done: bool = False
        self._prefetch_lock = threading.Lock()
        # Sync state.
        self._sync_thread: Optional[threading.Thread] = None
        self._sync_lock = threading.Lock()

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
        auto_review = self._config.get("auto_review", "true")
        self._auto_review = (
            auto_review.lower() in ("true", "1", "yes")
            if isinstance(auto_review, str) else bool(auto_review)
        )
        pause_marker = home / _AUTO_EXTRACT_PAUSE_MARKER
        env_pause = os.environ.get("HERMES_HYBRID_MEMORY_PAUSE_AUTO_EXTRACT", "")
        self._auto_extract_paused = pause_marker.exists() or env_pause.lower() in {
            "1", "true", "yes", "on"
        }

        # Embedder (lazy — model loads on first embed call).
        self._embedder = LocalEmbedder(model_name)

        if use_shared_service:
            # One local service owns the canonical DuckDB/Kùzu files. The
            # provider process never opens those files directly in this mode.
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
                db_path, user_id=self._user_id, embedder=self._embedder
            )
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

    # -- prefetch (auto-inject context before each turn) ---------------------

    def on_turn_start(self, turn_number: int, message: str, **kwargs) -> None:
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
            try:
                confirmation_candidates = store.list_candidates(
                    status="pending_user_confirmation", limit=1
                )
                confirmation = build_confirmation_block(confirmation_candidates)
                if confirmation:
                    sections.append(confirmation)

                results = store.search(query, limit=max_items)
                if results:
                    lines = []
                    for r in results:
                        cat = r.category
                        content = r.content
                        sim = f" (score: {r.similarity:.2f})" if r.similarity > 0 else ""
                        lines.append(f"- [{cat}] {content}{sim}")
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
            review = review_candidate_with_llm(candidate)
            decision = review.get("decision", "pending_user_confirmation")
            decision_map = {
                "approve": "reviewed_approved",
                "reject": "rejected",
                "quarantine": "quarantined",
                "pending_user_confirmation": "pending_user_confirmation",
            }
            final_status = decision_map.get(decision, "pending_user_confirmation")
            self._store.review_candidate(
                candidate_id=candidate["candidate_id"],
                decision=final_status,
                reason=review.get("reason", ""),
                review_confidence=review.get("confidence"),
                review_model=review.get("review_model", "memory_review"),
                durability=review.get("durability"),
                scope=review.get("scope"),
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

        def _sync() -> None:
            try:
                facts = extract_from_turn(
                    user_content, assistant_content,
                    use_llm_fallback=self._llm_fallback,
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

        with self._sync_lock:
            if self._sync_thread and self._sync_thread.is_alive():
                self._sync_thread.join(timeout=5.0)
            if self._sync_thread and self._sync_thread.is_alive():
                return
            self._sync_thread = threading.Thread(target=_sync, daemon=True, name="hybrid-sync")
            self._sync_thread.start()

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
                quarantined = self._store.cleanup_junk()
                if quarantined:
                    logger.info("Quarantined %d questionable memories at session end", quarantined)
            except Exception as e:
                logger.debug("Memory quarantine failed: %s", e)

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
        if reset:
            with self._prefetch_lock:
                self._prefetch_query = ""
                self._prefetch_result = ""
                self._prefetch_done = False

    # -- tools ---------------------------------------------------------------

    def _try_graph_relationship(self, content: str) -> None:
        """Attempt to extract a relationship from content text and add to graph.

        Tries the extractor's relationship regex first, then falls back to
        a simple 'X is the user's Y' pattern.
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
        ]
        if self._graph:
            schemas.append(GRAPH_SEARCH_SCHEMA)
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
            results = self._store.search(query, limit=top_k, category_filter=category)
            return json.dumps({
                "query": query,
                "count": len(results),
                "results": [r.to_dict() for r in results],
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
            # Populate graph for relationship memories.
            if category == "relationship" and self._graph:
                self._try_graph_relationship(content)
            return json.dumps({"status": "saved", "memory_id": rec.memory_id})

        elif tool_name == "memory_update":
            memory_id = args.get("memory_id", "")
            if not memory_id:
                return tool_error("Missing required parameter: memory_id")
            content = args.get("content")
            tags = args.get("tags")
            rec = self._store.update_memory(memory_id, content=content, tags=tags)
            if rec is None:
                return tool_error(f"Memory not found: {memory_id}")
            return json.dumps({"status": "updated", "memory_id": rec.memory_id})

        elif tool_name == "memory_delete":
            memory_id = args.get("memory_id", "")
            if not memory_id:
                return tool_error("Missing required parameter: memory_id")
            deleted = self._store.delete_memory(memory_id=memory_id)
            if not deleted:
                return tool_error(f"Memory not found: {memory_id}")
            return json.dumps({"status": "deleted", "memory_id": memory_id})

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
            try:
                result = self._store.review_candidate(
                    candidate_id=candidate_id,
                    decision=decision,
                    reason=args.get("reason", ""),
                )
            except ValueError as exc:
                return tool_error(str(exc))
            if result is None:
                return tool_error(f"Candidate not found: {candidate_id}")
            memory = result.get("memory")
            if memory and memory.get("category") == "relationship" and self._graph:
                self._try_graph_relationship(memory.get("content", ""))
            return json.dumps(result)

        elif tool_name == "memory_restore":
            memory_id = args.get("memory_id", "")
            if not memory_id:
                return tool_error("Missing required parameter: memory_id")
            if not self._store.restore_memory(memory_id):
                return tool_error(f"Memory not found: {memory_id}")
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
            return json.dumps({"status": "recorded", "memory_id": memory_id, "feedback": feedback})

        elif tool_name == "memory_graph_search":
            if self._graph is None:
                return tool_error("Relationship graph is not available")
            term = args.get("term", "")
            if not term:
                return tool_error("Missing required parameter: term")
            edges = self._graph.search_graph(term)
            return json.dumps({"term": term, "count": len(edges), "edges": edges})

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
# Plugin registration
# ---------------------------------------------------------------------------

def register(ctx) -> None:
    """Register HybridMemory as a memory provider plugin."""
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

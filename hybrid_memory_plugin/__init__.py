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
from .extractor import extract_from_turn

logger = logging.getLogger(__name__)

_PREFETCH_WAIT_SECS = 3.0
_DEFAULT_MAX_INJECTED = 8
_DEFAULT_MODEL = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"


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
        "max_injected_items": str(_DEFAULT_MAX_INJECTED),
        "local_embedding_model": _DEFAULT_MODEL,
        "auto_extract": "true",
        "llm_fallback": "true",
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
        "relate (e.g., 'who is Sam' or 'what tools does the user use'). "
        "Returns edges showing source -> relation -> target."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "term": {"type": "string", "description": "Entity name to search for (e.g., 'Sam', 'Item-E')."},
        },
        "required": ["term"],
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
                "description": "Enable automatic fact extraction after each turn",
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

        # Gateway processes (telegram, discord, slack, etc.) run in a separate
        # Python process from the desktop/CLI. DuckDB on Windows only allows
        # one writer per file, so two processes opening the same .duckdb file
        # causes a lock conflict and the second process falls back to read-only
        # (or fails entirely). To avoid this, gateway processes use a separate
        # database file with a "_gateway" suffix. The trade-off: memories saved
        # from Telegram won't be visible from the desktop and vice versa. This
        # is the simplest fix — a shared service layer would be the proper
        # solution but requires more infrastructure.
        if self._platform and self._platform != "cli":
            if not db_filename.endswith("_gateway.duckdb"):
                db_filename = db_filename.replace(".duckdb", "_gateway.duckdb")
            if not graph_dirname.endswith("_gateway"):
                graph_dirname = graph_dirname + "_gateway"

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

        # Embedder (lazy — model loads on first embed call).
        self._embedder = LocalEmbedder(model_name)

        # DuckDB store.
        db_path = home / db_filename
        self._store = DuckDBMemoryStore(db_path, user_id=self._user_id, embedder=self._embedder)

        # Kuzu graph (optional — degrade gracefully if kuzu unavailable).
        try:
            graph_path = home / graph_dirname
            self._graph = KuzuGraphStore(graph_path, user_id=self._user_id)
        except Exception as e:
            logger.warning("Kuzu graph unavailable, continuing without it: %s", e)
            self._graph = None

        self._initialized = True
        logger.info(
            "HybridMemory initialized: %d memories, graph=%s, embeddings=%s",
            self._store.count(),
            "on" if self._graph else "off",
            "pending" if not self._embedder.is_available else "on",
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
            "ANY topic — call memory_save immediately — don't wait to be asked.\n"
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
            body = ""
            try:
                results = store.search(query, limit=max_items)
                if results:
                    lines = []
                    for r in results:
                        cat = r.category
                        content = r.content
                        sim = f" (score: {r.similarity:.2f})" if r.similarity > 0 else ""
                        lines.append(f"- [{cat}] {content}{sim}")
                    body = "## Recalled Memories\n" + "\n".join(lines)
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
        if not self._auto_extract:
            return
        if self._store is None:
            return

        def _sync() -> None:
            try:
                facts = extract_from_turn(
                    user_content, assistant_content,
                    use_llm_fallback=self._llm_fallback,
                )
                saved = 0
                for fact in facts:
                    rec = self._store.remember(
                        category=fact["category"],
                        content=fact["content"],
                        tags=fact.get("tags", []),
                        payload=fact.get("payload", {}),
                        dedup=True,
                    )
                    if rec:
                        saved += 1
                        # Also add to graph if it's a relationship.
                        if fact["category"] == "relationship" and self._graph:
                            payload = fact.get("payload", {})
                            name = payload.get("name", "")
                            relation = payload.get("relation", "")
                            if name:
                                self._graph.add_relationship(
                                    source="user",
                                    source_type="person",
                                    relation=f"has_{relation}",
                                    target=name,
                                    target_type="person",
                                )
                if saved:
                    logger.debug("Auto-extracted %d facts from turn", saved)
            except Exception as e:
                logger.warning("Sync turn extraction failed: %s", e)

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
        # Clean up junk memories (fragments, duplicates, trivial facts).
        if self._store:
            try:
                deleted = self._store.cleanup_junk()
                if deleted:
                    logger.info("Cleaned up %d junk memories at session end", deleted)
            except Exception as e:
                logger.debug("Memory cleanup failed: %s", e)

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
        schemas = [SEARCH_SCHEMA, SAVE_SCHEMA, UPDATE_SCHEMA, DELETE_SCHEMA]
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
            deleted = self._store.delete_memory(memory_id)
            if not deleted:
                return tool_error(f"Memory not found: {memory_id}")
            return json.dumps({"status": "deleted", "memory_id": memory_id})

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

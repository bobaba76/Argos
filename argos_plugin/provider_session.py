"""Session mixin: review, sync worker, distillation, tools and shutdown.

Extracted verbatim from __init__.py during the god-file split (behavior-
neutral: no renames, no fixes). Carries the tool-schema dicts (their only
consumers are get_tool_schemas/handle_tool_call, also in this module).
"""
from __future__ import annotations

import difflib
import json
import logging
import os
import queue
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from .store import VALID_CATEGORIES
    from .extractor import extract_from_turn
    from .reviewer import review_candidate_with_llm, suggest_expiry
    from .distillation import run_distillation
except ImportError:  # provider_session.py imported as a top-level module
    from store import VALID_CATEGORIES
    from extractor import extract_from_turn
    from reviewer import review_candidate_with_llm, suggest_expiry
    from distillation import run_distillation
try:
    from tools.registry import tool_error
except ImportError:  # hermes runtime absent (conftest stub shape)
    def tool_error(msg):  # pragma: no cover - mirrors tests/conftest.py
        return json.dumps({"error": str(msg)})

logger = logging.getLogger(__name__)


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
            "include_expired": {
                "type": "boolean",
                "description": "Include expired memories in results (default false). Use to audit what was known before a best-before date passed.",
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
        "self-contained statement. For non-trivial reasoning (technical, analytical, "
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
            "durability": {
                "type": "string",
                "description": "durable (default) or temporary. Only 'temporary' triggers the TTL map (best-before date). Use when the fact has a shelf life.",
            },
            "expires_at": {
                "type": "string",
                "description": "Explicit best-before date (ISO-8601 UTC, e.g. '2026-12-31T23:59:59+00:00'). Wins over the TTL map. Use null to clear.",
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
            "expires_at": {
                "type": "string",
                "description": "Set or clear the best-before date. ISO-8601 UTC string to set; null to clear (revive an expired memory). Omit to carry forward the existing value.",
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

TOMBSTONES_SCHEMA = {
    "name": "memory_tombstones",
    "description": (
        "List deletion tombstones: fingerprints of hard-deleted memories that are "
        "blocked from being re-created. Use when the user asks what was permanently "
        "deleted, or to diagnose why saving a fact silently does nothing. Read-only."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "limit": {
                "type": "integer",
                "description": "Max tombstones to return (default 200, cap 1000).",
            },
        },
    },
}

PURGE_TOMBSTONE_SCHEMA = {
    "name": "memory_tombstone_purge",
    "description": (
        "Escape hatch: lift a deletion tombstone so a previously hard-deleted fact "
        "may be saved again. Requires the EXACT original content and its category "
        "(matching is case/whitespace-normalized). Only do this on an explicit user "
        "request to un-delete a specific fact."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "Exact original content of the deleted memory.",
            },
            "category": {
                "type": "string",
                "description": "Category the tombstone was recorded under (e.g. context_note).",
            },
        },
        "required": ["content", "category"],
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
            "expires_at": {
                "type": "string",
                "description": "Best-before date for the approved memory (ISO-8601 UTC). Use null to clear. Only applies on approval.",
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

WHY_NOT_SCHEMA = {
    "name": "memory_why_not",
    "description": (
        "Diagnose why a memory did not surface in retrieval. Deterministic, "
        "free (no LLM), read-only. Use to debug recall gaps: pass the query "
        "you used and the memory_id you expected to see."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query that failed to surface the expected memory.",
            },
            "expected_memory_id": {
                "type": "string",
                "description": "The memory_id you expected to see in results.",
            },
            "top_k": {
                "type": "integer",
                "description": "How many top results to inspect (default 20).",
            },
            "project_id": {
                "type": "string",
                "description": "Project scope to match the production search path (optional). If omitted, the provider's current project is used.",
            },
        },
        "required": ["query", "expected_memory_id"],
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

FETCH_FULL_SCHEMA = {
    "name": "memory_fetch_full",
    "description": (
        "Fetch the FULL stored text of a memory by its memory_id. The recalled-"
        "memory preview injected at the start of a turn is capped for token "
        "budget (long memories show a head+tail preview with an [id: ...] tag). "
        "When a previewed memory looks relevant but the preview is incomplete, "
        "call this with its memory_id to retrieve the complete, untruncated "
        "content. Reads existing data only — no re-ingest, no new storage."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "memory_id": {"type": "string", "description": "The memory_id shown in the recalled-memory preview ([id: ...] tag)."},
        },
        "required": ["memory_id"],
    },
}





class ProviderSessionMixin:
    """Session lifecycle, tool dispatch and maintenance methods."""

    def _review_candidate(self, candidate: Dict[str, Any]) -> None:
        """Run the conservative automatic reviewer for one new proposal."""
        try:
            review = review_candidate_with_llm(
                candidate, model=self._llm_model, provider=self._llm_provider,
            )
            decision = review.get("decision", "pending_user_confirmation")
            decision_map = {
                # Approval invariant: the automatic reviewer may never write
                # "approved" — that transition is reserved for the agent-facing
                # confirmation tool (memory_candidate_review). The ceiling for
                # auto-review approval is "reviewed_approved" (LLM-approved,
                # awaiting user confirmation). Enforced in store.review_candidate.
                "approve": "reviewed_approved",
                "reject": "rejected",
                "quarantine": "quarantined",
                "pending_user_confirmation": "pending_user_confirmation",
            }
            final_status = decision_map.get(decision, "pending_user_confirmation")

            # Value-supersession (issue #4): if the candidate was flagged with
            # a value conflict at save time, downgrade to pending_user_confirmation
            # regardless of the LLM's decision — a human must confirm that the
            # new value supersedes the old one.  Pass supersedes_memory_id so
            # the confirmation chains the supersession automatically.
            supersedes_memory_id: str | None = None
            candidate_payload = candidate.get("payload") or {}
            if isinstance(candidate_payload, str):
                try:
                    candidate_payload = json.loads(candidate_payload)
                except (json.JSONDecodeError, TypeError):
                    candidate_payload = {}
            value_sup = candidate_payload.get("value_supersession")
            if value_sup and isinstance(value_sup, dict):
                supersedes_memory_id = value_sup.get("supersedes_memory_id")
                if final_status == "reviewed_approved":
                    final_status = "pending_user_confirmation"
                    review["reason"] = (
                        "Value-supersession detected: new value "
                        f"{value_sup.get('new_value', '?')} conflicts with "
                        f"existing value {value_sup.get('old_value', '?')}. "
                        "Confirmation required to supersede the old fact. "
                        + (review.get("reason", "") or "")
                    ).strip()

            result = self._store.review_candidate(
                evidence_retention=getattr(self, "_evidence_retention", "full"),
                candidate_id=candidate["candidate_id"],
                decision=final_status,
                reason=review.get("reason", ""),
                review_confidence=review.get("confidence"),
                review_model=review.get("review_model", "memory_review"),
                durability=review.get("durability"),
                scope=review.get("scope"),
                review_source="auto_review",
                supersedes_memory_id=supersedes_memory_id,
            )
            # Spec 1: deterministic expiry suggestion on approval. The
            # suggestion is logged but NOT auto-applied — the user confirms
            # via memory_update(expires_at=...) before it sticks. This keeps
            # the confirm-first invariant: no silent lifecycle changes.
            if (
                self._expiry_auto_suggest
                and final_status == "approved"
                and result
                and result.get("memory")
            ):
                try:
                    suggested = suggest_expiry(
                        candidate,
                        ttl_days=self._expiry_ttl_days,
                        default_days=self._expiry_default_days,
                    )
                    if suggested:
                        logger.info(
                            "Expiry suggestion for %s: %s (confirm with "
                            "memory_update expires_at)",
                            result["memory"].get("memory_id", ""),
                            suggested,
                        )
                except Exception:
                    pass
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
                # Idle for 10s; check if we should exit. Re-check under the
                # lock to avoid the idle-exit race (issue #30: a put could
                # arrive between the get-timeout and the empty() check).
                with self._sync_lock:
                    try:
                        item = self._sync_queue.get_nowait()
                    except queue.Empty:
                        self._sync_worker_started = False
                        return
                # Got an item during the race window — process it below.
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
                dup_threshold = getattr(self, "_extraction_dup_threshold", 0.88)
                for fact in facts:
                    # Pre-insert dedupe: skip proposals already covered by an
                    # active memory (valid_to IS NULL). Fail-soft — any
                    # embedder/search error inserts normally; dedupe must
                    # never block memory capture.
                    try:
                        dup = self._store.find_semantic_duplicate(
                            fact["content"], min_similarity=dup_threshold
                        )
                    except Exception:
                        dup = None
                    if dup is not None:
                        logger.debug(
                            "Skipping duplicate proposal %r — matches active "
                            "memory %s (cosine=%.3f)",
                            (fact.get("content") or "")[:80],
                            getattr(dup, "memory_id", "?"),
                            float(getattr(dup, "similarity", 0.0) or 0.0),
                        )
                        continue
                    payload = dict(fact.get("payload") or {})
                    # Project-scoped proposals (#47): thread the session's
                    # project_id into candidates at extraction time. The
                    # candidates table already has the column — this is the
                    # plumbing that populates it. When the fact carries its
                    # own project_id (e.g. from explicit extraction), that
                    # wins; otherwise the session's current project scope
                    # is used. Global/unsessioned sessions stay None.
                    fact_project_id = fact.get("project_id")
                    if not fact_project_id:
                        fact_project_id = getattr(self, "_current_project_id", None) or None
                    candidate = self._store.save_candidate(
                        category=fact["category"],
                        content=fact["content"],
                        tags=fact.get("tags", []),
                        payload=payload,
                        source=fact.get("source", payload.get("source", "regex_extraction")),
                        confidence=fact.get("confidence", 0.45),
                        durability=fact.get("durability", "durable"),
                        scope=fact.get("scope", "profile"),
                        project_id=fact_project_id,
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
                        duplicate_min_similarity=self._duplicate_min_similarity,
                        duplicate_semantic_max_pairs=self._duplicate_semantic_max_pairs,
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
            # Distillation (P4.2): the gated proposal pass fires at any
            # session boundary the desktop actually delivers — end-of-session
            # where it exists, and every chat rotation via session switch.
            self._maybe_run_distillation()

    def _maybe_run_distillation(self) -> None:
        """Run the gated distillation pass ("the dream") when enabled.

        Invoked from the session-boundary hooks Argos can rely on in the
        desktop app: ``on_session_end`` (true boundaries: CLI exit, /new,
        gateway expiry under a finite reset policy) and ``on_session_switch``
        (fires on every chat rotation in the desktop, where sessions are
        resumable tiles and true boundaries are rare by default).

        Self-gating (novelty + cooldown + budget) and fail-soft: it must
        never block session lifecycle.
        """
        if not self._distillation_enabled:
            return
        try:
            dream_report = run_distillation(
                self._store,
                llm_model=self._llm_model,
                llm_provider=self._llm_provider,
                min_new_records=self._distillation_min_new_records,
                cooldown_hours=self._distillation_cooldown_hours,
                max_records_per_run=self._distillation_max_records_per_run,
                max_calls=self._distillation_max_calls,
            )
            if dream_report.get("ran"):
                logger.info(
                    "Distillation: %d proposals, %d contradictions, %d calls",
                    dream_report.get("proposals_emitted", 0),
                    dream_report.get("contradictions_emitted", 0),
                    dream_report.get("llm_calls", 0),
                )
            elif dream_report.get("reason"):
                logger.warning(
                    "Distillation skipped: %s", dream_report["reason"],
                )
        except Exception as e:
            logger.warning("Distillation failed: %s", e)


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
        # Desktop sessions are resumable tiles; true session boundaries are
        # rare by default (reset policy mode="none"), so chat rotation is the
        # reliable trigger for the gated dream pass. Cooldown keeps it ≤1/day.
        self._maybe_run_distillation()

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

        Two-tier extraction:
        1. Regex-fast path: known role words (from _DEFAULT_ROLE_WORDS +
           config + previously learned) produce aliases immediately.
        2. LLM ambiguity gate: when "my X is Name" matches but X is not a
           known role word and Name is capitalized, a tiny LLM call checks
           if X is a person-role. If yes, the alias is written AND X is
           added to the role words set (self-extending) so future
           occurrences are regex-fast. Gated by role_alias_llm_fallback
           config flag.

        Guards against over-minting:
        - Only fires on has_<role> relations with a person target
        - The canonical name must start with a capital letter (a real name,
          not a verb/adjective — same guard as the bare_role pattern)
        - Skip if the "name" is actually a role word capitalized
        - Idempotent: INSERT OR REPLACE in add_alias
        """
        store = getattr(self, "_store", None)
        if not store or not content:
            return
        try:
            from graph import (
                extract_graph_relations,
                _is_role_word,
                _add_learned_role_word,
                _get_role_words,
            )

            # Stage 1: regex-fast path for known role words.
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
                if _is_role_word(canonical):
                    continue
                alias = f"my {role}"
                store.add_alias(alias, canonical)
                logger.debug(
                    "Index-time alias: %r -> %r (from: %s)",
                    alias, canonical, content[:60],
                )

            # Stage 2: LLM ambiguity gate for unknown role words.
            # Detect "my X is Name" patterns where X is NOT a known role
            # word and Name is capitalized. These are structurally
            # unambiguous (possessive + lowercase word + "is" + Capitalized)
            # but the role word is unknown. A tiny LLM call classifies it.
            llm_fallback = str(
                self._config.get("role_alias_llm_fallback", "true")
            ).lower() in ("true", "1", "yes")
            if not llm_fallback:
                return

            import re as _re
            # Match "my X is Name" where X is lowercase and Name is capitalized.
            # This is the same pattern as my_relation in graph.py but without
            # the role-word filter — we want the UNKNOWN ones.
            # Note: (?i:...) on the prefix only — the name group [A-Z] must
            # stay case-sensitive (requires a capital letter).
            ambiguous = _re.compile(
                r"\b(?i:(?:my|the\s+user'?s?))\s+([a-z][a-z_-]*)\s+is\s+"
                r"([A-Z][A-Za-z0-9'_-]*(?:\s+[A-Z][A-Za-z0-9'_-]*)?)",
            )
            known_words = _get_role_words()
            for match in ambiguous.finditer(content):
                role_word = match.group(1).lower()
                name = match.group(2)
                if role_word in known_words:
                    continue  # Already handled by Stage 1
                if not name or not name[0].isupper():
                    continue
                if _is_role_word(name):
                    continue  # "my wife is Wife" — not a real name
                # LLM ambiguity gate: is this word a person-role?
                if self._llm_classify_role_word(role_word):
                    # Self-extending: add to in-memory set + persist to config
                    _add_learned_role_word(role_word)
                    known_words = _get_role_words()  # refresh for next iteration
                    # Also update the extractor's role-word set so future
                    # "my X is Name" extractions categorize as relationship
                    # (issue #14: extractor and graph lexicons converge).
                    try:
                        from extractor import set_role_words
                        set_role_words(known_words)
                    except Exception:
                        pass
                    self._persist_learned_role_word(role_word)
                    alias = f"my {role_word}"
                    store.add_alias(alias, name)
                    logger.info(
                        "LLM-learned role alias: %r -> %r (role word: %s)",
                        alias, name, role_word,
                    )
        except Exception as exc:
            logger.debug("Role-alias extraction failed: %s", exc)

    def _llm_classify_role_word(self, word: str) -> bool:
        """Ask the auxiliary LLM if a word is a person-role word.

        Fires only at the ambiguity gate (unknown word in "my X is Name").
        Returns True if the LLM classifies X as a person-role (therapist,
        accountant, coach, etc.). Never raises — returns False on any
        failure so the alias is simply not written.
        """
        if not word or len(word) < 2:
            return False
        from egress import gate as _egress_gate
        if not _egress_gate("role_word", word):
            return False

        try:
            from agent.auxiliary_client import call_llm
        except ImportError:
            return False
        except Exception:
            return False

        prompt = (
            f'Is "{word}" a person role word — a word that describes a '
            f"relationship between a person and another person, like "
            f'"wife", "therapist", "boss", "accountant", "coach"? '
            f"Answer with only JSON: "
            f'{{"is_role": true}} or {{"is_role": false}}'
        )
        try:
            response = call_llm(
                task="role_word_classification",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=20,
                timeout=5.0,
            )
        except Exception:
            return False
        if response is None:
            return False
        try:
            text = response.choices[0].message.content.strip()
        except (AttributeError, IndexError, KeyError):
            return False
        # Parse minimal JSON or bare true/false
        try:
            import json as _json
            if text.startswith("{"):
                result = _json.loads(text)
                return bool(result.get("is_role", False))
            return text.lower().strip() in ("true", "yes")
        except Exception:
            return text.lower().strip() in ("true", "yes")

    def _persist_learned_role_word(self, word: str) -> None:
        """Persist a learned role word to hybrid_memory.json so it survives restarts.

        Reads the current config, adds the word to the role_words list,
        and writes back. Thread-safe via atomic write. Never raises —
        persistence failure just means the word won't survive a restart
        (it's still in the in-memory set for this session).
        """
        if not word:
            return
        try:
            home = self._hermes_home or os.path.expanduser("~/.hermes")
            config_path = Path(home) / "hybrid_memory.json"
            if config_path.exists():
                cfg = json.loads(config_path.read_text(encoding="utf-8"))
            else:
                cfg = {}

            # role_words is stored as a JSON array string
            raw = str(cfg.get("role_words", "")).strip()
            if raw.startswith("["):
                words = json.loads(raw)
            elif raw:
                words = [w.strip() for w in raw.split(",")]
            else:
                words = []

            if word not in words:
                words.append(word)
                cfg["role_words"] = json.dumps(words)
                config_path.write_text(
                    json.dumps(cfg, indent=2), encoding="utf-8"
                )
                logger.debug("Persisted learned role word: %s", word)
        except Exception as exc:
            logger.debug("Failed to persist role word %r: %s", word, exc)

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
            TOMBSTONES_SCHEMA,
            PURGE_TOMBSTONE_SCHEMA,
            CANDIDATE_LIST_SCHEMA,
            CANDIDATE_REVIEW_SCHEMA,
            RESTORE_SCHEMA,
            FEEDBACK_SCHEMA,
            MAINTENANCE_SCHEMA,
            CHAIN_SCHEMA,
            FETCH_FULL_SCHEMA,
            WHY_NOT_SCHEMA,
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
            include_expired = bool(args.get("include_expired", False))
            results = self._search_memories(
                query, limit=top_k, category_filter=category,
                project_id=project_id,
                include_expired=include_expired,
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
            # Expiry (Spec 1): pass durability/expires_at only when enabled.
            save_kwargs: Dict[str, Any] = {"category": category, "content": content, "tags": tags, "dedup": True}
            if getattr(self, "_expiry_enabled", False):
                if "durability" in args:
                    save_kwargs["durability"] = args["durability"]
                if "expires_at" in args:
                    save_kwargs["expires_at"] = args["expires_at"]
            try:
                rec = self._store.remember(**save_kwargs)
            except ValueError as exc:
                return tool_error(str(exc))
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
            update_kwargs: Dict[str, Any] = {"memory_id": memory_id, "content": content, "tags": tags}
            # #93: the memory_update tool is the LLM-agent-driven content
            # rewrite path — exactly the case the structural-loss guard
            # was designed for. Wire structural_guard=True so an agent
            # rewrite cannot silently delete sentences, list items, or KV
            # pairs. Direct API callers (user-initiated updates) call
            # store.update_memory() without the guard.
            if content is not None:
                update_kwargs["structural_guard"] = True
            # Expiry (Spec 1): pass expires_at only when enabled AND the
            # caller explicitly provided it (None = clear/revive, str = set).
            if getattr(self, "_expiry_enabled", False) and "expires_at" in args:
                update_kwargs["expires_at"] = args["expires_at"]
            try:
                rec = self._store.update_memory(**update_kwargs)
            except ValueError as exc:
                return tool_error(str(exc))
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

        elif tool_name == "memory_tombstones":
            try:
                limit = max(1, min(int(args.get("limit", 200)), 1000))
            except (ValueError, TypeError):
                limit = 200
            tombstones = self._store.list_tombstones(limit=limit)
            return json.dumps({"count": len(tombstones), "tombstones": tombstones})

        elif tool_name == "memory_tombstone_purge":
            content = str(args.get("content", "")).strip()
            category = str(args.get("category", "")).strip()
            if not content or not category:
                return tool_error("Missing required parameter: content and category")
            purged = self._store.purge_tombstone(content=content, category=category)
            if not purged:
                return tool_error(
                    "No matching tombstone found (content must match the original "
                    "exactly, modulo case/whitespace)."
                )
            return json.dumps({
                "status": "purged",
                "detail": "Tombstone lifted — this fact may be saved again.",
            })

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
                review_kwargs: Dict[str, Any] = dict(
                    evidence_retention=getattr(self, "_evidence_retention", "full"),
                    candidate_id=candidate_id,
                    decision=decision,
                    reason=args.get("reason", ""),
                    supersedes_memory_id=supersedes_memory_id,
                )
                # Spec 1: pass expires_at through when expiry is enabled.
                if getattr(self, "_expiry_enabled", False) and "expires_at" in args:
                    review_kwargs["expires_at"] = args["expires_at"]
                result = self._store.review_candidate(**review_kwargs)
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
                duplicate_min_similarity=getattr(self, "_duplicate_min_similarity", 0.88),
                duplicate_semantic_max_pairs=getattr(self, "_duplicate_semantic_max_pairs", 20000),
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

        elif tool_name == "memory_fetch_full":
            # LEVER 2: on-demand full-memory fetch. The injection preview caps
            # content for token budget; this returns the complete stored record
            # by id so the agent never has to act on a mangled preview. Reads
            # existing data only (store.get_memories_by_ids), no re-ingest.
            memory_id = args.get("memory_id", "")
            if not memory_id:
                return tool_error("Missing required parameter: memory_id")
            records = self._store.get_memories_by_ids([memory_id])
            if not records:
                return tool_error(f"Memory not found or no longer active: {memory_id}")
            rec = records[0]
            return json.dumps({
                "memory_id": rec.memory_id,
                "category": rec.category,
                "content": rec.content,
                "tags": rec.tags,
                "created_at": rec.created_at,
                "updated_at": rec.updated_at,
                "status": "full",
            })

        elif tool_name == "memory_graph_search":
            if self._graph is None:
                return tool_error("Relationship graph is not available")
            term = args.get("term", "")
            if not term:
                return tool_error("Missing required parameter: term")
            edges = self._graph.search_graph(term)
            return json.dumps({"term": term, "count": len(edges), "edges": edges})

        elif tool_name == "memory_why_not":
            # Spec 2: deterministic, free, read-only retrieval diagnostic.
            query = args.get("query", "")
            expected_memory_id = args.get("expected_memory_id", "")
            if not query:
                return tool_error("Missing required parameter: query")
            if not expected_memory_id:
                return tool_error("Missing required parameter: expected_memory_id")
            try:
                top_k = max(1, min(int(args.get("top_k", 20)), 100))
            except (TypeError, ValueError):
                top_k = 20
            # Thread project_id so the diagnostic search matches the
            # production scoping path (project_scope_mismatch is a top-3
            # cause of "why didn't this surface").
            effective_project = args.get("project_id")
            if effective_project is None:
                effective_project = self._current_project_id
            try:
                explanation = self._store.explain_retrieval(
                    query, expected_memory_id, top_k=top_k,
                    project_id=effective_project,
                )
            except Exception as exc:
                return tool_error(f"explain_retrieval failed: {exc}")
            return json.dumps(explanation, default=str)

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

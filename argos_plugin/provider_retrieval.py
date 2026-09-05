"""Retrieval mixin: system prompt block, enrichment, chain-unfold, prefetch.

Extracted verbatim from __init__.py during the god-file split (behavior-
neutral: no renames, no fixes). Consts are imported from provider_core.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List

try:
    from .store_common import MemoryRecord as _MemoryRecord
except ImportError:  # provider_retrieval.py imported as a top-level module
    from store_common import MemoryRecord as _MemoryRecord

try:
    from .confirmation import choose_confirmation_block
except ImportError:  # provider_retrieval.py imported as a top-level module
    from confirmation import choose_confirmation_block

try:
    from .provider_core import (
        _DEFAULT_INJECT_CONTENT_CHAR_CAP,
        _INJECTION_FALLBACK_COUNT,
        _MEMORY_FENCE_NOTE,
        _PREFETCH_WAIT_SECS,
        _TRIVIAL_QUERY_PATTERNS,
        _freshness_marker_for,
        _neutralize_markup,
    )
except ImportError:  # provider_retrieval.py imported as a top-level module
    from provider_core import (
        _DEFAULT_INJECT_CONTENT_CHAR_CAP,
        _INJECTION_FALLBACK_COUNT,
        _MEMORY_FENCE_NOTE,
        _PREFETCH_WAIT_SECS,
        _TRIVIAL_QUERY_PATTERNS,
        _freshness_marker_for,
        _neutralize_markup,
    )
try:
    from .store_retrieval import StoreRetrievalMixin as _StoreRetrievalMixin
except ImportError:  # provider_retrieval.py imported as a top-level module
    from store_retrieval import StoreRetrievalMixin as _StoreRetrievalMixin

try:
    from .value_extractor import extract_values, values_conflict
except ImportError:  # provider_retrieval.py imported as a top-level module
    from value_extractor import extract_values, values_conflict

# #248: tuning constants consolidated in tuning.py
try:
    from .tuning import (
        ALIAS_CACHE_TTL_SECONDS as _TUNING_ALIAS_CACHE_TTL,
        GRAPH_CIRCUIT_BREAKER_THRESHOLD as _TUNING_CB_THRESHOLD,
        GRAPH_CIRCUIT_BREAKER_COOLDOWN as _TUNING_CB_COOLDOWN,
    )
except ImportError:  # provider_retrieval.py imported as a top-level module
    from tuning import (
        ALIAS_CACHE_TTL_SECONDS as _TUNING_ALIAS_CACHE_TTL,
        GRAPH_CIRCUIT_BREAKER_THRESHOLD as _TUNING_CB_THRESHOLD,
        GRAPH_CIRCUIT_BREAKER_COOLDOWN as _TUNING_CB_COOLDOWN,
    )

logger = logging.getLogger(__name__)

# -- #247: system prompt template (byte-stable for prompt caching) -----------
import functools
import os as _os

@functools.lru_cache(maxsize=1)
def _load_system_prompt() -> str:
    """Load the system prompt template once from the template file.

    #247: the template is a plain text file with a single {graph_status}
    placeholder. Loaded once and cached via lru_cache. The text must stay
    byte-stable for prompt caching — do not modify the template file.
    """
    _here = _os.path.dirname(_os.path.abspath(__file__))
    _path = _os.path.join(_here, "system_prompt_template.txt")
    with open(_path, "r", encoding="utf-8") as f:
        return f.read()


# -- Read-side conflict surfacing (config-gated, default OFF) ----------------
# When the injected set contains two ACTIVE records that conflict on the same
# subject, append an explicit conflict note to the context so the answerer
# surfaces the disagreement instead of smoothing two eras into one answer.
# Two triggers, both LLM-free and deterministic:
#   1. value conflict  -> values_conflict() (same subject + same unit + diff value)
#   2. discontinuation -> the records share a significant token AND at least one
#      record marks the rule/feature as stopped/removed/scoped (the poster's
#      "workaround documented an incident" class).
_CONFLICT_MARKERS = (
    "stopped", "ended", "ending", "discontinu", "retired", "removed",
    "scrapped", "closed", "reverted", "cancelled", "cancelled", "no longer",
    "no current", "disbanded", "scoped to", "limited to", "only in",
    "was retired", "was scrapped", "was removed", "was discontinued",
    "was reverted", "program ended", "period removed",
)
_CONFLICT_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "at", "for",
    "with", "is", "are", "was", "were", "we", "they", "it", "our", "your",
    "this", "that", "now", "get", "gets", "use", "used", "do", "does", "by",
}
_CONFLICT_SUBJECT_MIN_TOKENS = 1   # shared significant tokens required
_CONFLICT_MAX_NOTES = 2            # cap annotations to bound injection bloat


def _conflict_significant_tokens(text: str) -> set:
    toks = {t for t in re.findall(r"[a-z0-9]+", (text or "").lower())
            if len(t) >= 4 and t not in _CONFLICT_STOPWORDS}
    # also keep 3-char numeric-ish tokens ("v3", "10gb") — drop bare short words
    toks |= {t for t in re.findall(r"[a-z0-9]+", (text or "").lower())
             if len(t) >= 3 and any(ch.isdigit() for ch in t)}
    return toks


def _has_discontinuation_marker(text: str) -> bool:
    t = (text or "").lower()
    return any(m in t for m in _CONFLICT_MARKERS)


def _conflict_shared_subject(a: str, b: str) -> bool:
    """True if two records share a significant token (same subject)."""
    sa, sb = _conflict_significant_tokens(a), _conflict_significant_tokens(b)
    return bool(sa & sb)


class ProviderRetrievalMixin:
    """Injection, enrichment, chain-unfold and prefetch methods."""

    def system_prompt_block(self) -> str:
        # STATIC text only — must be byte-stable for prompt caching.
        # Dynamic state (memory count, embedding status) is NOT included here
        # because it changes between turns and would invalidate the cached prefix.
        # The prefetch() method injects dynamic recall context separately.
        # #247: the prompt text is loaded from system_prompt_template.txt to
        # keep it byte-stable and separable from code changes.
        graph_status = "available" if self._graph else "unavailable"
        return _load_system_prompt().format(graph_status=graph_status)

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
        # PR5: sanitize each context message with _neutralize_markup so
        # stored conversation content cannot inject markup or
        # instruction-like text into the embedder query.
        context_parts: list[str] = []
        total_chars = 0
        for msg in reversed(recent):  # most recent first
            # PR5: neutralize markup before prepending to the query.
            safe_msg = self._neutralize_context_message(msg)
            if total_chars + len(safe_msg) > self._context_max_chars:
                break
            context_parts.insert(0, safe_msg)
            total_chars += len(safe_msg)
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

    def _neutralize_context_message(self, msg: str) -> str:
        """PR5: neutralize markup/injection in a context message before
        prepending it to the embedder query.

        Strips XML/HTML-like tags and collapses repeated whitespace so a
        stored message containing ``<system>ignore previous instructions``
        cannot skew the embedder query. This is not a full injection guard
        (the embedder is not an LLM), but it prevents markup from
        dominating the query embedding.
        """
        if not msg:
            return ""
        # Strip XML/HTML-like tags.
        clean = re.sub(r"<[^>]+>", " ", msg)
        # Collapse repeated whitespace.
        clean = re.sub(r"\s+", " ", clean).strip()
        return clean

    # PR3: alias list cache with TTL. ``list_aliases()`` is O(n) per turn
    # on a store with many aliases; the cache avoids re-scanning on every
    # retrieval. Invalidated after _ALIAS_CACHE_TTL_SECONDS or on store
    # changes (add_alias etc. can call _invalidate_alias_cache).
    _alias_cache: list = []
    _alias_cache_time: float = 0.0
    # #248: tuning constants from tuning.py
    _ALIAS_CACHE_TTL_SECONDS = _TUNING_ALIAS_CACHE_TTL
    # PR10: circuit breaker for graph-aware retrieval. After N consecutive
    # failures, graph boosting is disabled for a cooldown period so a
    # persistent bug doesn't waste CPU and silently degrade every turn.
    _GRAPH_CIRCUIT_BREAKER_THRESHOLD = _TUNING_CB_THRESHOLD
    _GRAPH_CIRCUIT_BREAKER_COOLDOWN = _TUNING_CB_COOLDOWN
    _graph_retrieval_failures: int = 0
    _graph_retrieval_disabled_until: float = 0.0

    def _get_cached_alias_list(self) -> list:
        """PR3: return the alias list, cached with a TTL."""
        import time as _time
        if not hasattr(self._store, "list_aliases"):
            return []
        now = _time.monotonic()
        if self._alias_cache and (now - self._alias_cache_time) < self._ALIAS_CACHE_TTL_SECONDS:
            return self._alias_cache
        try:
            self._alias_cache = list(self._store.list_aliases())
            self._alias_cache_time = now
        except Exception:
            self._alias_cache = []
            self._alias_cache_time = now
        return self._alias_cache

    def _invalidate_alias_cache(self) -> None:
        """PR3: call when aliases change (add/remove) to force a re-scan."""
        self._alias_cache = []
        self._alias_cache_time = 0.0

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

        # RRF: original results get rank-based scores.
        # k is linked to the store's _RRF_K so tuning the fusion constant
        # can never silently desync the expansion merge (audit B5).
        rrf_k = _StoreRetrievalMixin._RRF_K
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
        # Current-state contrast probes: "where do I live NOW", "what car do
        # I drive NOW", "do I STILL ..." — imply a past->present change and
        # are the phrasing real users actually use. Added 2026-08-20 after
        # the scaled eval showed the 8 explicit regexes rejected 90% of
        # real change queries (recall 10%).
        r"\b(what|where|which|who)\b[^?]*(now|currently|these days)\b",
        r"\bhow (much|many|old|tall|long)\b[^?]*(now|currently|these days)\b",
        r"\bdo i still\b",
        r"\bam i still\b",
        r"\bstill (live|drive|work|take|use|play|eat|have|go|plan)\b",
    )

    def _change_intent_match(self, query: str) -> bool:
        """True if the query signals change-intent (arc-relevant)."""
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

        Gate (measured 2026-08-13, early chain-unfold eval): the original
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
        # Option A semantic-arc check: the chain's CURRENT version content
        # must be semantically close enough to the query (cosine >= floor)
        # before we inject. This is the precision guard that replaces the
        # recall-starving top-1 rule — it filters false triggers while still
        # scanning top-K/fallback for the actual chain. Cheap: one seek + one
        # dot against already-loaded embedder.
        if not self._arc_clears_similarity_floor(query, versions):
            return None
        # Rough token estimate: ~4 chars/token.
        token_cost = max(1, len(arc) // 4)
        if token_cost > self._chain_max_inject:
            return None
        self._chain_unfolded_stats["count"] += 1
        self._chain_unfolded_stats["tokens_injected"] += token_cost
        # #275 LP2: increment the chain_unfold_calls counter.
        try:
            try:
                from .liveness import increment_counter
            except ImportError:
                from liveness import increment_counter
            increment_counter("chain_unfold_calls")
        except Exception:
            pass
        return arc

    def _arc_clears_similarity_floor(self, query: str, versions: List[Any]) -> bool:
        """Cosine(query, current-version content) >= arc floor (Option A).

        PR7: fail-closed on error — a missed arc is better than an
        irrelevant one injected into context. The previous fail-open
        behavior could inject chain arcs even when the semantic check
        crashed, defeating the precision guard.
        """
        try:
            current = next((v for v in versions if getattr(v, "valid_to", None) is None), None)
            if current is None:
                current = versions[-1]
            content = getattr(current, "content", "") or ""
            if not content.strip() or self._embedder is None:
                # PR7: fail-closed — no embedder/content means we can't
                # verify semantic relevance, so don't inject.
                return False
            qe = self._embedder.embed(query, is_query=True)
            ce = self._embedder.embed(content)
            if not qe or not ce or len(qe) != len(ce):
                # PR7: fail-closed on embedding failure.
                return False
            denom = (sum(a * a for a in qe) ** 0.5) * (sum(b * b for b in ce) ** 0.5)
            if denom <= 0:
                return False
            cos = sum(a * b for a, b in zip(qe, ce)) / denom
            return cos >= self._chain_unfold_arc_min_similarity
        except Exception:
            # PR7: fail-closed — never let the guard crash inference into
            # injecting an irrelevant arc.
            return False

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
        include_expired: bool = False,
        include_closed: bool = False,
    ) -> List[Any]:
        """Run hybrid search and apply a bounded graph-supported boost.

        When *project_id* is provided, memories from other projects are
        excluded. When None, the provider's current project scope is used.

        When *include_expired* is True, expired memories are included in
        results (for auditing).
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
            include_expired=include_expired,
            include_closed=include_closed,
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
        # PR10: circuit breaker — skip graph boosting if recently disabled
        # by consecutive failures.
        import time as _time
        if self._graph_retrieval_disabled_until > 0 and _time.monotonic() < self._graph_retrieval_disabled_until:
            logger.debug(
                "Graph-aware retrieval disabled (PR10 circuit breaker, "
                "%.0fs remaining)",
                self._graph_retrieval_disabled_until - _time.monotonic(),
            )
            final_results = results[:limit]
            self._record_injected(final_results)
            return final_results
        # Reset failure counter if we get past the breaker (graph is healthy).
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
                # PR3: cache the alias list with a TTL so we don't scan all
                # aliases (O(n) substring checks) on every turn.
                if not alias_terms:
                    alias_maps = self._get_cached_alias_list()
                    for alias_map in alias_maps:
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
                    logger.debug("traversal: %d candidate ids for %r", len(traversal_ids), effective_query[:40])
                    # #275 LP2: increment the graph_injections counter
                    # when traversal produces candidates.
                    if traversal_ids:
                        try:
                            try:
                                from .liveness import increment_counter
                            except ImportError:
                                from liveness import increment_counter
                            increment_counter("graph_injections")
                        except Exception:
                            pass
                    if traversal_ids:
                        seen = set(graph_ids)
                        for tid in traversal_ids:
                            if tid not in seen:
                                graph_ids.append(tid)
                                seen.add(tid)
                except Exception:
                    traversal_ids = []
            # PPR-based candidates (issue #37): Personalized PageRank
            # diffusion from query-grounded seed entities. Replaces
            # traversal with diffusion — surfaces multi-hop associations
            # without a fixed depth cutoff. Disabled unless
            # graph_ppr_enabled (config) — eval-first A/B gate.
            ppr_ids: list[str] = []
            if self._graph_ppr_enabled:
                try:
                    ppr_ids = self._graph.ppr_memory_ids(
                        effective_query,
                        limit=max(10, candidate_limit),
                        damping=self._graph_ppr_damping,
                    )
                    logger.debug("ppr: %d candidate ids for %r", len(ppr_ids), effective_query[:40])
                    if ppr_ids:
                        seen = set(graph_ids)
                        for pid in ppr_ids:
                            if pid not in seen:
                                graph_ids.append(pid)
                                seen.add(pid)
                except Exception:
                    ppr_ids = []
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
                if self._graph_ppr_enabled:
                    injectable_ids.update(ppr_ids)
                    logger.debug("graph injectable ids: %d (traversal=%d, ppr=%d, alias=%d)",
                                 len(injectable_ids), len(traversal_ids),
                                 len(ppr_ids), len(alias_expanded_ids))
                elif self._graph_traversal_enabled:
                    logger.debug("graph injectable ids: %d (traversal=%d, alias=%d)",
                                 len(injectable_ids), len(traversal_ids), len(alias_expanded_ids))
                if self._graph_inject_candidates:
                    injectable_ids.update(graph_ids)
                # #81: pre-compute the boost-eligible ID sets so the inclusion
                # gate below can exempt them — the boost floor is applied
                # after the gate, so without exemption below-gate candidates
                # are dropped before the floor can lift them.
                alias_id_set_pre = set(alias_expanded_ids)
                traversal_id_set_pre = set(traversal_ids)
                ppr_id_set_pre = set(ppr_ids)
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
                                if len(query_emb) != len(record.embedding):
                                    # Dimension mismatch (e.g. leftover rows
                                    # from a previous embedding model): zip()
                                    # would silently truncate and produce a
                                    # wrong-but-plausible cosine (audit B2).
                                    sim = 0.0
                                else:
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
                        # PR8: raw_similarity is always the pre-boost value;
                        # similarity is post-boost+clamp (see below). This
                        # invariant is relied on by downstream gates that
                        # check raw_similarity to decide if a result was
                        # graph-boosted or organic.
                        record.raw_similarity = sim
                        # #81: alias/traversal/PPR candidates are exempt from
                        # the inclusion gate — their boost floor is applied
                        # below, which lifts below-gate raw similarity above
                        # the cutoff. Without exemption, the boost never runs.
                        is_boosted_candidate = (
                            record.memory_id in alias_id_set_pre
                            or (self._graph_traversal_enabled and record.memory_id in traversal_id_set_pre)
                            or (self._graph_ppr_enabled and record.memory_id in ppr_id_set_pre)
                        )
                        if is_boosted_candidate or sim >= self._graph_boost_min_similarity:
                            results.append(record)
                graph_rank = {memory_id: rank for rank, memory_id in enumerate(graph_ids)}
                graph_count = max(len(graph_ids), 1)
                alias_id_set = set(alias_expanded_ids)
                traversal_id_set = set(traversal_ids)
                ppr_id_set = set(ppr_ids)
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
                    # PPR candidates (issue #37): memory reached by PageRank
                    # diffusion from query-grounded seed entities. Same
                    # relational-evidence rationale as traversal — floor
                    # lifts it above the cutoff.
                    if (self._graph_ppr_enabled
                            and record.memory_id in ppr_id_set):
                        record.similarity = max(
                            record.similarity, self._graph_ppr_boost
                        )
                        continue
                    rank = graph_rank.get(record.memory_id)
                    if rank is None:
                        continue
                    if record.similarity < self._graph_boost_min_similarity:
                        continue
                    decay = 1.0 - (rank / graph_count)
                    record.similarity += self._graph_retrieval_boost * max(0.0, decay)
                # #142: clamp all results to [0, 1] — graph boost is additive
                # and can push a high-similarity record above 1.0.
                for record in results:
                    if record.similarity > 1.0:
                        record.similarity = 1.0
                results.sort(key=lambda record: record.similarity, reverse=True)
        except (NameError, AttributeError, ImportError) as exc:
            # #84: programming errors must NOT be swallowed.
            logger.error("Graph-aware retrieval programming error: %s", exc)
            raise
        except Exception as exc:
            # #84: expected fail-soft conditions degrade to unboosted results.
            self._graph_retrieval_failures = getattr(self, "_graph_retrieval_failures", 0) + 1
            # PR10: circuit breaker — after N consecutive graph failures,
            # disable graph-aware retrieval for a cooldown period and log
            # at ERROR level so a persistent bug isn't silently ignored.
            _fail_count = self._graph_retrieval_failures
            if _fail_count >= self._GRAPH_CIRCUIT_BREAKER_THRESHOLD:
                import time as _time
                self._graph_retrieval_disabled_until = (
                    _time.monotonic() + self._GRAPH_CIRCUIT_BREAKER_COOLDOWN
                )
                logger.error(
                    "Graph-aware retrieval failed %d times — disabling for "
                    "%.0fs cooldown (PR10 circuit breaker): %s",
                    _fail_count, self._GRAPH_CIRCUIT_BREAKER_COOLDOWN, exc,
                )
            else:
                logger.warning(
                    "Graph-aware retrieval failed (fail-soft, count=%d): %s",
                    _fail_count, exc,
                )
        final_results = results[:limit]
        if getattr(self, "_conflict_surfacing_enabled", False):
            try:
                notes = self._conflict_annotations(final_results)
                if notes:
                    final_results = notes + final_results
            except Exception as exc:  # noqa: BLE001 — surfacing must never break retrieval
                logger.warning("Conflict surfacing failed (fail-soft): %s", exc)
        self._record_injected(final_results)
        return final_results

    def _conflict_annotations(
        self, records: List[Any], max_notes: int = _CONFLICT_MAX_NOTES,
    ) -> List[Any]:
        """Return conflict-note records for unlinked conflicting pairs in *records*.

        Triggers (LLM-free, deterministic):
          1. ``values_conflict`` — same subject, same unit, different value
             (the 27/8 unlinked stale-number class).
          2. shared significant token + discontinuation/scoping marker on either
             side (the poster's "rule stopped / workaround reverted" class).

        A note is only emitted for pairs where BOTH members are in the injected
        set, so it never pulls new records into context. The note states both
        facts with dates and the resolution rule ("later replacement = current;
        discontinued/scoped = no current policy") so the answerer surfaces the
        disagreement instead of smoothing two eras into one answer.
        """
        notes: List[Any] = []
        flagged = set()
        now = datetime.now(timezone.utc).isoformat()
        n = len(records)
        # PR4: pre-compute value extractions once per record so the O(n²)
        # pair scan doesn't re-extract on every comparison.
        pre_values = [extract_values(r.content or "") for r in records]
        for i in range(n):
            if len(notes) >= max_notes:
                break
            for j in range(i + 1, n):
                if len(notes) >= max_notes:
                    break
                ri, rj = records[i], records[j]
                a, b = ri.content or "", rj.content or ""
                if not a or not b or a == b:
                    continue
                if (ri.valid_to is not None and ri.valid_to) or (
                        rj.valid_to is not None and rj.valid_to):
                    continue  # only active-vs-active unlinked pairs
                reason = None
                # PR4: use pre-computed values instead of re-extracting.
                if values_conflict(
                    pre_values[i], pre_values[j], subject_threshold=0.2,
                ):
                    reason = "differing values"
                elif (
                    _conflict_shared_subject(a, b)
                    and (_has_discontinuation_marker(a) or _has_discontinuation_marker(b))
                ):
                    reason = "one record says it was discontinued, removed, or is scoped elsewhere"
                if not reason:
                    continue
                key = tuple(sorted((ri.memory_id, rj.memory_id)))
                if key in flagged:
                    continue
                flagged.add(key)
                if (ri.created_at or "") <= (rj.created_at or ""):
                    older, newer = ri, rj
                else:
                    older, newer = rj, ri
                da = (older.created_at or "")[:10]
                db = (newer.created_at or "")[:10]
                sa = (older.content or "")[:70].replace("\n", " ")
                sb = (newer.content or "")[:70].replace("\n", " ")
                note = (
                    f"CONFLICT NOTE: two retrieved records disagree ({reason}): "
                    f"\"{sa}\" ({da}) vs \"{sb}\" ({db}). Neither is recorded as "
                    "superseding the other. If the later record states a replacement "
                    "value or rule, treat it as current. If it says the rule/feature "
                    "was discontinued, removed, or is scoped elsewhere, there is NO "
                    "current policy — say so rather than presenting the older "
                    "statement as current."
                )[:360]
                notes.append(_MemoryRecord(
                    memory_id="conflictnote-"
                              + hashlib.md5((ri.memory_id + rj.memory_id).encode()).hexdigest()[:8],
                    category="system_note",
                    content=note,
                    created_at=now,
                    similarity=max(
                        getattr(ri, "similarity", 0.0) or 0.0,
                        getattr(rj, "similarity", 0.0) or 0.0,
                    ),
                ))
        return notes

    # -- confirmation surfacing ledger ----------------------------------
    # Persisted in the store's system_state KV so a candidate is surfaced
    # at most once across provider restarts (the #99 failure mode was
    # re-asking the same candidate every turn, forever).
    # PR2: a lock guards the read-modify-write so two concurrent prefetch
    # threads can't clobber each other's addition.
    _confirmation_ledger_lock = threading.Lock()
    # PR1: cap the ledger so it doesn't grow unbounded over months/years.
    # IDs for candidates no longer pending are pruned on each write; the
    # cap is a safety net for stores where pending status isn't queryable.
    _CONFIRMATION_LEDGER_MAX = 1000

    def _surfaced_confirmation_ids(self) -> set:
        try:
            raw = (
                self._store.get_state("surfaced_confirmation_ids")
                if self._store
                else None
            )
            if raw:
                _ids = json.loads(raw)
                if isinstance(_ids, list):
                    return {str(x) for x in _ids}
        except Exception:
            pass
        return set()

    def _mark_surfaced_confirmation(self, candidate_id: str | None) -> None:
        if not candidate_id or self._store is None:
            return
        # PR2: lock the read-modify-write so concurrent prefetch threads
        # don't clobber each other's addition.
        with self._confirmation_ledger_lock:
            try:
                _ids = self._surfaced_confirmation_ids()
                _ids.add(str(candidate_id))
                # PR1: prune the ledger — remove IDs for candidates that are
                # no longer in pending_user_confirmation status, and cap the
                # total size as a safety net.
                _ids = self._prune_confirmation_ledger(_ids)
                self._store.set_state(
                    "surfaced_confirmation_ids", json.dumps(sorted(_ids))
                )
            except Exception:
                pass

    def _prune_confirmation_ledger(self, ids: set) -> set:
        """PR1: prune the confirmation ledger.

        Removes IDs for candidates that are no longer in
        ``pending_user_confirmation`` status (they've been confirmed or
        rejected, so re-surfacing is harmless but wasteful). Falls back to
        a size cap if the store doesn't support status queries.
        """
        if not ids or self._store is None:
            return ids
        try:
            # Try to prune by status — keep only IDs still pending.
            pending_ids: set = set()
            for cid in ids:
                try:
                    rec = self._store.get_by_id(str(cid))
                    if rec and getattr(rec, "status", None) == "pending_user_confirmation":
                        pending_ids.add(str(cid))
                except Exception:
                    # Can't check this ID — keep it (safe default).
                    pending_ids.add(str(cid))
            if pending_ids:
                return pending_ids
            # No pending IDs remain — return empty (all confirmed/rejected).
            if len(ids) > 0:
                return set()
        except Exception:
            pass
        # Fallback: cap by size (keep most recent — sorted = deterministic).
        if len(ids) > self._CONFIRMATION_LEDGER_MAX:
            return set(sorted(ids)[-self._CONFIRMATION_LEDGER_MAX:])
        return ids

    # -- prefetch (auto-inject context before each turn) ---------------------

    def on_turn_start(self, turn_number: int, message: str, **kwargs) -> None:
        # PS6: reset the per-turn tool call counter.
        self._tool_call_count = 0
        self._record_user_message(message)
        self._start_prefetch(message)

    def _start_prefetch(self, query: str) -> None:
        if not query or self._store is None:
            return
        # Trivial-query gate (opt-in): a greeting or "just a test" has no
        # information need; retrieving on it burns injection tokens on
        # near-random top-k items. Short-circuit BEFORE any search work.
        if getattr(self, "_skip_retrieval_on_trivial", False):
            try:
                _q = query.lower().strip()
                if len(_q.split()) <= 6 and any(
                    re.fullmatch(p, _q) for p in _TRIVIAL_QUERY_PATTERNS
                ):
                    logger.debug("Prefetch skipped (trivial-query gate)")
                    return
            except Exception:
                pass  # gate failure must never break prefetch
        max_items = self._max_injected

        with self._prefetch_lock:
            if self._prefetch_query == query:
                if self._prefetch_done or (self._prefetch_thread and self._prefetch_thread.is_alive()):
                    return
            self._prefetch_query = query
            self._prefetch_result = ""
            self._prefetch_done = False
            # PR6: set a cancel event on the old thread so it can exit
            # early instead of wasting CPU on a superseded query.
            old_cancel = getattr(self, "_prefetch_cancel_event", None)
            if old_cancel is not None:
                old_cancel.set()
            self._prefetch_cancel_event = threading.Event()

        def _run() -> None:
            sections = []
            body = ""
            try:
                # Guarded confirmation surfacing (#99 rework, 3/9): surface
                # ONE pending user-confirmation per turn — genuine needs
                # only (reviewer failures are never "should this be saved?"
                # prompts), never re-surface a candidate already shown
                # (ledger persisted in system_state, survives restarts),
                # and only on non-trivial turns (the trivial gate above
                # returns first). The candidate stays pending until the
                # user acts through memory_candidate_review — this block
                # only prompts the model to ask. Fail-soft: any store
                # hiccup skips surfacing, never retrieval.
                if getattr(self, "_confirmation_surfacing", False):
                    try:
                        _cands = self._store.list_candidates(
                            status="pending_user_confirmation", limit=10
                        )
                        _block, _surfaced_id = choose_confirmation_block(
                            _cands, self._surfaced_confirmation_ids()
                        )
                        if _block:
                            sections.append(_block)
                            self._mark_surfaced_confirmation(_surfaced_id)
                            logger.info(
                                "Surfaced confirmation candidate %s (guarded surfacing)",
                                _surfaced_id,
                            )
                    except Exception as _exc:
                        logger.debug(
                            "Confirmation surfacing skipped (fail-soft): %s", _exc
                        )

                # History-at-current-time (#3): on historical queries
                # ("where did I use to live"), widen retrieval to closed
                # versions so superseded facts are visible again.
                try:
                    try:
                        from .intent_router import is_historical_query
                    except ImportError:
                        from intent_router import is_historical_query
                    _include_closed = (
                        getattr(self, "_history_at_current_time", False)
                        and bool(query) and is_historical_query(query)
                    )
                except Exception:
                    _include_closed = False
                results = self._search_memories(
                    query, limit=max_items,
                    include_closed=_include_closed,
                )
                _floor = getattr(self, "_injection_min_score", 0.0)
                if results and _floor > 0:
                    _kept = [
                        r for r in results
                        if float(getattr(r, "similarity", 0.0) or 0.0) >= _floor
                    ]
                    if not _kept:
                        # Never-blind fallback: the floor suppressed every
                        # candidate. A turn whose evidence all sits below the
                        # floor still deserves its best (weak) evidence rather
                        # than silence — inject a few unfiltered top results.
                        logger.info(
                            "Injection floor %.2f suppressed all %d candidates; "
                            "falling back to unfiltered top-%d",
                            _floor, len(results), _INJECTION_FALLBACK_COUNT,
                        )
                        _kept = list(results[:_INJECTION_FALLBACK_COUNT])
                    results = _kept
                if results and getattr(self, "_chronological_injection", False):
                    # P2B: on temporal/multi-hop turns, re-sort the top-k by
                    # creation timestamp (oldest first) so the model reads a
                    # timeline in order instead of relevance-scrambled order.
                    # Relevance order is preserved for ordinary turns.
                    try:
                        from .intent_router import is_temporal_or_multihop
                        if is_temporal_or_multihop(query):
                            def _ts_key(r):
                                # created_at is an ISO-8601 string; lexicographic
                                # order is only chronological when offsets are
                                # uniform. Normalize to UTC epoch first so
                                # mixed "Z"/"+14:00"/"-05:00" rows sort
                                # correctly; unparseable rows keep raw ordering.
                                ts = getattr(r, "created_at", None) or ""
                                try:
                                    from datetime import datetime as _dt
                                    return _dt.fromisoformat(
                                        ts.replace("Z", "+00:00")
                                    ).timestamp()
                                except (ValueError, TypeError, AttributeError):
                                    return ts
                            results = sorted(results, key=_ts_key)
                    except Exception:
                        pass  # P2B is best-effort; never break injection
                if results and getattr(self, "_date_anchor_rerank", False):
                    # P2B2: date-anchored re-rank — when the temporal turn
                    # carries an explicit date expression ("10 days ago",
                    # "last Tuesday", "on March 2nd"), re-sort the top-k by
                    # proximity to the resolved target date so the model
                    # reads the right time window first. Zero-LLM; best-effort.
                    try:
                        from .intent_router import is_temporal_or_multihop
                        if is_temporal_or_multihop(query):
                            from .date_anchor import reorder_by_date
                            results, _t, _l = reorder_by_date(results, query)
                    except Exception:
                        pass  # P2B2 is best-effort; never break injection
                if results:
                    # Spec-06 (#69): access scoping — filter retrieval results
                    # by the user's ACL mask before ranking/partition. Hidden
                    # deny: excluded content never appears. The audit row is
                    # written by the store's search wrapper; this is the
                    # defence-in-depth re-validation on the injected set.
                    # AS2: fail-closed on exception — if the filter crashes,
                    # return empty results (not all results). Silently
                    # passing all results is the wrong default for a security
                    # filter.
                    # AS7: write an audit row when the prefetch filter denies
                    # content that the store wrapper didn't catch.
                    try:
                        acl = getattr(self, "_acl_config", None)
                        if acl is not None and not acl.is_open_store:
                            try:
                                from .access_scoping import (
                                    filter_records_by_access,
                                )
                            except ImportError:
                                from access_scoping import (
                                    filter_records_by_access,
                                )
                            results, _denied = filter_records_by_access(
                                results, acl, self._user_id,
                            )
                            # AS7: audit denied content from the prefetch
                            # defence-in-depth path.
                            if _denied > 0:
                                try:
                                    self._store.write_access_audit(
                                        user_id=self._user_id,
                                        query_text="(prefetch_acl_filter)",
                                        granted_count=len(results),
                                        denied_count=_denied,
                                    )
                                except Exception:
                                    logger.warning(
                                        "AS7: failed to write prefetch ACL "
                                        "audit row (denied=%d)", _denied,
                                    )
                    except Exception:
                        # AS2: fail-closed — a security filter crash must
                        # not pass all results unfiltered.
                        logger.warning(
                            "AS2: prefetch ACL filter crashed — failing "
                            "closed (no results injected this turn)"
                        )
                        results = []
                    # Spec-05 (#67): presence-aware namespace partition.
                    # Split the injection budget between conversation-sourced
                    # and document-sourced memories so a flood of doc-facts
                    # cannot crowd out conversational memories (and vice
                    # versa). Floors only bite when both namespaces are
                    # populated — a single-namespace store gets all slots.
                    # v1 floors are explicitly untuned (24/24, inverted
                    # 12/40 for client-scoped queries).
                    try:
                        try:
                            from .namespace_partition import (
                                partition_by_namespace,
                            )
                        except ImportError:
                            from namespace_partition import (
                                partition_by_namespace,
                            )
                        results = partition_by_namespace(
                            results,
                            cap=max_items,
                            client_scoped=bool(getattr(self, "_client_scope", None)),
                        )
                    except Exception:
                        pass  # partition must never break injection
                    lines = []
                    for r in results:
                        cat = r.category
                        content = r.content
                        _cap = getattr(self, "_inject_cap", _DEFAULT_INJECT_CONTENT_CHAR_CAP)
                        if len(content) > _cap:
                            content = content[:_cap].rsplit(" ", 1)[0] + "..."
                        sim = f" (score: {r.similarity:.2f})" if r.similarity > 0 else ""
                        date = (r.created_at or "")[:10]
                        date_s = f"[{date}] " if date else ""
                        # Expose memory_id so the agent can call memory_fetch_full
                        # when a capped preview looks relevant but incomplete.
                        mid = getattr(r, "memory_id", "") or ""
                        id_s = f" [id: {mid}]" if mid else ""
                        # Closed version = superseded fact surfaced by a
                        # historical query; label it so the model treats it
                        # as past state, not current truth.
                        hist_s = (
                            " (previously)" if getattr(r, "valid_to", None) else ""
                        )
                        # Freshness marker (Tier-2 anti-staleness): when the
                        # content carries an explicit date anchor, append a
                        # compact as-of marker from the record's own update
                        # time so a stale anchor is never read as current.
                        fr_s = ""
                        if getattr(self, "_freshness_markers", False):
                            try:
                                _asof = getattr(r, "updated_at", None) or (r.created_at or "")
                                fr_s = _freshness_marker_for(content, _asof or "")
                            except Exception:
                                fr_s = ""
                        lines.append(f"- {date_s}[{cat}] {content}{fr_s}{sim}{id_s}{hist_s}")
                    # Memory injection fence (#34): wrap the recalled block
                    # in a reference-data note so stored instructions cannot
                    # be read as system guidance. Neutralize < > in content
                    # so stored markup cannot be interpreted as prompt tags.
                    fenced_lines = [_neutralize_markup(ln) for ln in lines]
                    sections.append(
                        "## Recalled Memories\n"
                        + f"[{_MEMORY_FENCE_NOTE}]\n"
                        + "\n".join(fenced_lines)
                    )
                body = "\n\n".join(sections)
            except Exception as e:
                logger.debug("Prefetch failed: %s", e)
            # PR6: check the cancel event before writing — if a newer
            # query superseded this one, skip the write (the guard on
            # _prefetch_query already prevents overwriting, but this
            # avoids the unnecessary lock acquisition).
            _cancel = getattr(self, "_prefetch_cancel_event", None)
            if _cancel is not None and _cancel.is_set():
                return
            with self._prefetch_lock:
                if self._prefetch_query == query:
                    self._prefetch_result = body
                    self._prefetch_done = True

        t = threading.Thread(target=_run, daemon=True, name="hybrid-prefetch")
        with self._prefetch_lock:
            # Join the previous prefetch thread before overwriting the handle
            # (issue #30: was replaced without join, leaving a stale thread
            # that could complete and write state for an old query).
            old = self._prefetch_thread
            self._prefetch_thread = t
        if old and old.is_alive() and old is not t:
            old.join(timeout=0.1)  # brief — don't block the new turn
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
        """PR9: on cache miss, start the prefetch and wait briefly for the
        result. The wait adds latency to the first turn after restart, but
        returning empty would mean the first turn has no context at all.
        The wait is bounded by ``_PREFETCH_WAIT_SECS``; if the search is
        slow, the caller gets an empty string and the prefetched result
        is available on the next call (the thread continues in the
        background). This is the documented trade-off — accepted.
        """
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

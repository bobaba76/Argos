"""Query expansion for weak retrieval results.

When the bi-encoder search returns weak results (top hit below a similarity
floor), this module calls the host's LLM to rewrite the query into sub-queries
that can be searched independently. This addresses the "multi-topic keyword
query" failure mode where the embedding can't bridge unrelated keywords to
the right memories.

Design constraints (non-negotiable):
- **Lazy/conditional**: Only fires when top hit is below a similarity floor.
  Typical searches never touch the LLM.
- **Cached**: Cache key is the raw query string, so repeated natural phrasing
  is a hash hit.
- **Fail-soft**: If the LLM call errors or times out, fall back to the
  original query results with no user-visible change.

The expansion produces 1-3 sub-queries that are searched independently,
then results are merged via Reciprocal Rank Fusion.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# LLM call timeout in seconds. Short because this is in the search path.
_LLM_TIMEOUT = 10.0

# Cache TTL in seconds (1 hour). Prevents re-calling the LLM for the same query.
_CACHE_TTL = 3600

# Maximum number of sub-queries to generate.
_MAX_SUBQUERIES = 3

# Minimum query length to bother expanding.
_MIN_QUERY_LENGTH = 10

# Similarity floor: if the top hit is above this, don't expand.
# This is the "weak results" trigger.
_DEFAULT_SIMILARITY_FLOOR = 0.3

_LLM_SYSTEM_PROMPT = """You are a query expansion assistant for a personal memory system.

The user's memory store contains short, fragmented notes about their life,
work, topics, relationships, and projects. The search uses embedding
similarity, so queries need to match the vocabulary of the stored memories.

Your job: rewrite the user's query into 1-3 alternative search queries that
might match different memories relevant to the original question. Each
sub-query should target a single topic or concept from the original query.

Rules:
- Keep each sub-query short (5-15 words)
- Use concrete nouns and entity names, not pronouns
- Split multi-topic queries into single-topic sub-queries
- Don't add information that wasn't in the original query
- Don't repeat the original query verbatim

Return a JSON array of strings. Example:
Query: "product list work anxiety boss meeting"
Output: ["product list workflow Excel", "work anxiety boss email", "meeting stress sales team"]

Query: "Hermes configuration tools model OpenRouter MCP"
Output: ["Hermes config setup gateway", "OpenRouter model configuration", "MCP server tools"]

If the query is already specific and single-topic, return an empty array: []
"""


class QueryExpander:
    """Lazy, cached, fail-soft query expansion via the host's LLM.

    The expander is designed to be called only when the bi-encoder search
    returns weak results (top hit below a similarity floor). It uses the
    host's auxiliary LLM client (agent.auxiliary_client.call_llm) to
    rewrite the query into sub-queries.
    """

    def __init__(
        self,
        *,
        similarity_floor: float = _DEFAULT_SIMILARITY_FLOOR,
        max_subqueries: int = _MAX_SUBQUERIES,
        cache_ttl: int = _CACHE_TTL,
        timeout: float = _LLM_TIMEOUT,
    ) -> None:
        self._similarity_floor = similarity_floor
        self._max_subqueries = max_subqueries
        self._cache_ttl = cache_ttl
        self._timeout = timeout
        # Cache: {query_hash: (timestamp, sub_queries)}
        self._cache: Dict[str, Tuple[float, List[str]]] = {}
        self._cache_lock = threading.Lock()
        self._enabled = True

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value

    @property
    def similarity_floor(self) -> float:
        return self._similarity_floor

    def should_expand(self, query: str, top_similarity: float) -> bool:
        """Check whether query expansion should fire.

        Returns True only if:
        - expansion is enabled
        - the query is long enough
        - the top hit similarity is below the floor (weak results)
        """
        if not self._enabled:
            return False
        if not query or len(query.strip()) < _MIN_QUERY_LENGTH:
            return False
        if top_similarity >= self._similarity_floor:
            return False
        return True

    def expand(self, query: str) -> List[str]:
        """Expand a query into sub-queries.

        Returns a list of 0-3 sub-queries. Returns empty list if:
        - expansion is disabled
        - the LLM call fails (fail-soft)
        - the LLM returns no sub-queries
        - the result is cached and expired

        Never raises — all errors are caught and logged.
        """
        if not self._enabled or not query:
            return []

        # Check cache first
        cache_key = self._cache_hash(query)
        cached = self._get_cached(cache_key)
        if cached is not None:
            logger.debug("Query expansion cache hit for: %s", query[:50])
            return cached

        # Call LLM
        sub_queries = self._call_llm(query)
        if sub_queries:
            self._set_cached(cache_key, sub_queries)
        return sub_queries

    def _cache_hash(self, query: str) -> str:
        """Generate a cache key from the raw query string."""
        return hashlib.sha256(query.strip().lower().encode("utf-8")).hexdigest()[:16]

    def _get_cached(self, key: str) -> Optional[List[str]]:
        """Get a cached result if it exists and hasn't expired."""
        with self._cache_lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            timestamp, sub_queries = entry
            if time.time() - timestamp > self._cache_ttl:
                # Expired
                del self._cache[key]
                return None
            return list(sub_queries)

    def _set_cached(self, key: str, sub_queries: List[str]) -> None:
        """Store a result in the cache."""
        with self._cache_lock:
            self._cache[key] = (time.time(), list(sub_queries))
            # Evict old entries if cache grows too large (>1000 entries)
            if len(self._cache) > 1000:
                now = time.time()
                expired = [
                    k for k, (ts, _) in self._cache.items()
                    if now - ts > self._cache_ttl
                ]
                for k in expired:
                    del self._cache[k]

    def _call_llm(self, query: str) -> List[str]:
        """Call the host's LLM to expand the query.

        Uses agent.auxiliary_client.call_llm if available.
        Returns empty list on any failure (fail-soft).
        """
        try:
            from agent.auxiliary_client import call_llm
        except ImportError:
            logger.debug("Query expansion unavailable: agent.auxiliary_client not importable")
            return []
        except Exception as e:
            logger.debug("Query expansion unavailable: %s", e)
            return []

        messages = [
            {"role": "system", "content": _LLM_SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ]

        try:
            response = call_llm(
                task="query_expansion",
                messages=messages,
                temperature=0.0,
                max_tokens=200,
                timeout=self._timeout,
            )
        except Exception as exc:
            logger.debug("Query expansion LLM call failed: %s", exc)
            return []

        if response is None:
            return []

        # Extract text from the response (handles both string and
        # ChatCompletion object responses).
        text = response
        if hasattr(response, "choices"):
            try:
                text = response.choices[0].message.content
            except (IndexError, AttributeError, TypeError):
                return []
        if not isinstance(text, str):
            return []

        # Parse the response as a JSON array of strings
        return self._parse_response(text)

    def _parse_response(self, response: str) -> List[str]:
        """Parse the LLM response into a list of sub-queries."""
        # Try to extract a JSON array from the response
        text = response.strip()
        # Find the JSON array in the response. Use greedy match to get
        # the full array (non-greedy would stop at the first ]).
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if not match:
            return []
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return []

        if not isinstance(parsed, list):
            return []

        # Filter and clean sub-queries
        sub_queries: List[str] = []
        for item in parsed:
            if isinstance(item, str) and item.strip():
                cleaned = item.strip()
                if len(cleaned) >= 3 and len(cleaned) <= 100:
                    sub_queries.append(cleaned)
            if len(sub_queries) >= self._max_subqueries:
                break

        return sub_queries

    def clear_cache(self) -> None:
        """Clear the expansion cache."""
        with self._cache_lock:
            self._cache.clear()

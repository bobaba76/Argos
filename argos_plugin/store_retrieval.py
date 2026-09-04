"""Retrieval mixin: search, rank fusion, scale metrics and hybrid arms.

Extracted from store.py during the god-file split. Accessor edits vs the
original bytes: staticmethod bodies that referenced ``DuckDBMemoryStore``
directly now reference ``StoreRetrievalMixin`` (same resolution via MRO,
avoids the circular import). Everything else is verbatim.
"""
from __future__ import annotations

import json
import logging
import math
import time
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, List

try:
    from .store_common import (
        GROUNDING_OBSERVED,
        MemoryRecord,
        PROVENANCE_INTERNAL,
        _TEXT_STOPWORDS,
        _tokenize,
    )
except ImportError:  # store_retrieval.py imported as a top-level module
    from store_common import (
        GROUNDING_OBSERVED,
        MemoryRecord,
        PROVENANCE_INTERNAL,
        _TEXT_STOPWORDS,
        _tokenize,
    )
try:
    from .retriever import DuckDBRetriever
except ImportError:  # store_retrieval.py imported as a top-level module
    from retriever import DuckDBRetriever

# #248: tuning constants consolidated in tuning.py
try:
    from .tuning import BM25_K1, BM25_B, DEDUP_SIMILARITY_THRESHOLD, MAX_EMBEDDING_DIM
except ImportError:  # store_retrieval.py imported as a top-level module
    from tuning import BM25_K1, BM25_B, DEDUP_SIMILARITY_THRESHOLD, MAX_EMBEDDING_DIM

logger = logging.getLogger(__name__)


class StoreRetrievalMixin:
    """Search, fusion, scale-metric and hybrid-retrieval methods."""

    def _row_to_record(self, row: dict, similarity: float = 0.0) -> MemoryRecord:
        return MemoryRecord(
            memory_id=row.get("memory_id", ""),
            category=row.get("category", "context_note"),
            content=row.get("content", ""),
            tags=row.get("tags") if row.get("tags") is not None else [],
            payload=json.loads(row.get("payload")) if row.get("payload") else {},
            created_at=row.get("created_at"),
            updated_at=row.get("updated_at"),
            expires_at=row.get("expires_at"),
            embedding=row.get("embedding"),
            similarity=similarity,
            status=row.get("status", "active"),
            source=row.get("source", "explicit"),
            confidence=row.get("confidence"),
            durability=row.get("durability", "durable"),
            scope=row.get("scope", "profile"),
            project_id=row.get("project_id"),
            user_scope=row.get("user_scope"),
            namespace=row.get("namespace", "conversation"),
            client_scope=row.get("client_scope"),
            doc_class=row.get("doc_class"),
            source_doc_id=row.get("source_doc_id"),
            source_loc=row.get("source_loc"),
            extraction_method=row.get("extraction_method"),
            extracted_at=row.get("extracted_at"),
            verified_state=row.get("verified_state", "current"),
            verified_at=row.get("verified_at"),
            retrieval_count=row.get("retrieval_count", 0),
            last_retrieved_at=row.get("last_retrieved_at"),
            helpful_count=row.get("helpful_count", 0),
            dismissed_count=row.get("dismissed_count", 0),
            quarantine_reason=row.get("quarantine_reason"),
            quarantined_at=row.get("quarantined_at"),
            valid_from=row.get("valid_from"),
            valid_to=row.get("valid_to"),
            superseded_by=row.get("superseded_by"),
            provenance_origin=row.get("provenance_origin", PROVENANCE_INTERNAL),
            grounding=row.get("grounding", GROUNDING_OBSERVED),
            tier=row.get("tier", "active"),
        )

    def _fetch_records(
        self, query: str, params: list | None = None, sim_col: str = ""
    ) -> List[MemoryRecord]:
        with self._lock:
            assert self.connection is not None
            result = self.connection.execute(query, params or [])
            columns = [desc[0] for desc in result.description]
            records: List[MemoryRecord] = []
            for row in result.fetchall():
                row_dict = dict(zip(columns, row))
                sim = row_dict.pop(sim_col, 0.0) if sim_col else 0.0
                records.append(self._row_to_record(row_dict, float(sim) if sim else 0.0))
            return records

    # -- vector search --------------------------------------------------------

    @staticmethod
    def _is_vector_search_unavailable(exc: Exception) -> bool:
        msg = str(exc).lower()
        return (
            "list_cosine_similarity" in msg
            or "catalog error" in msg
            or "binder error" in msg
            or "no function matches" in msg
            or "type with name double[]" in msg
            or "vss" in msg
        )

    def _text_search_raw(
        self, query: str, limit: int, excluded: set[str],
        category_filter: str | None = None,
        project_id: str | None = None,
        namespace: str | None = None,
        client_scope: str | None = None,
        as_of: str | None = None,
        include_expired: bool = False,
        include_closed: bool = False,
        include_archived: bool = False,
    ) -> List[MemoryRecord]:
        """BM25-lite text search. Returns filtered records ranked by BM25.

        Does NOT record retrieval — the caller is responsible for that.
        Sets ``similarity`` to a max-normalized BM25 score so downstream
        fusion has a comparable signal.
        When *project_id* is provided, memories from other projects are
        excluded; global memories (project_id IS NULL) remain visible.

        Spec-05 (#67): *namespace* filters by source namespace
        ('conversation'/'document'); None = no filter (backward compatible).
        *client_scope* filters to a client's rows OR global (NULL) rows;
        None = no filter.

        When *include_expired* is True, the expiry filter is omitted (expired
        memories are returned, ranked normally).

        P5.1 (#6): when *include_archived* is False (default), archived
        records (tier='archived') are excluded from the injection pool.
        """
        tokens = [
            t for t in _tokenize(query)
            if len(t) > 2 and t not in _TEXT_STOPWORDS
        ][:8]
        if not tokens:
            return []
        patterns = [f"%{t}%" for t in tokens]
        conditions = " OR ".join(["content ILIKE ?" for _ in patterns])
        project_clause = ""
        namespace_clause = ""
        client_scope_clause = ""
        expiry_ref = as_of if as_of else self._now()
        if include_expired:
            expiry_clause = ""
            expiry_params: list = []
        else:
            expiry_clause = "AND (expires_at IS NULL OR expires_at > ?) "
            expiry_params = [expiry_ref]
        params: list = [self.user_id, *expiry_params]
        if project_id:
            project_clause = " AND (project_id IS NULL OR project_id = ?)"
            params.append(project_id)
        if namespace:
            namespace_clause = " AND namespace = ?"
            params.append(namespace)
        if client_scope:
            # NULL = global: a client-scoped query still sees global rows.
            client_scope_clause = " AND (client_scope IS NULL OR client_scope = ?)"
            params.append(client_scope)
        # Temporal filter: default to current (valid_to IS NULL),
        # or as_of (valid_from <= as_of AND (valid_to IS NULL OR valid_to > as_of)).
        # include_closed widens to closed versions too — used by the
        # historical-query path so "where did I use to live" can see
        # superseded facts. as_of takes precedence when both are set.
        if as_of:
            temporal_clause = (
                "AND valid_from <= ? "
                "AND (valid_to IS NULL OR valid_to > ?) "
            )
            temporal_params = [as_of, as_of]
        elif include_closed:
            temporal_clause = " "
            temporal_params = []
        else:
            temporal_clause = "AND valid_to IS NULL "
            temporal_params = []

        # Push category_filter and excluded into SQL so the LIMIT 2000
        # candidate pool is pre-filtered (issue #27: Python-side filtering
        # after the fetch starves narrow queries with wrong-category rows).
        category_clause = ""
        category_params: list = []
        if category_filter:
            category_clause = " AND category = ?"
            category_params.append(category_filter)
        if excluded:
            excluded_placeholders = ", ".join(["?" for _ in excluded])
            category_clause += f" AND LOWER(category) NOT IN ({excluded_placeholders})"
            category_params.extend(e.lower() for e in excluded)
        # P5.1 (#6): exclude archived records from the injection pool
        # unless include_archived is explicitly requested.
        tier_clause = "" if include_archived else " AND COALESCE(tier, 'active') = 'active'"
        sql = (
            "SELECT * FROM memory_records "
            "WHERE COALESCE(status, 'active') = 'active' "
            f"{temporal_clause} "
            "AND (user_scope IS NULL OR user_scope = ?) "
            f"{expiry_clause}"
            f"{project_clause}"
            f"{namespace_clause}"
            f"{client_scope_clause}"
            f"{category_clause}"
            f"{tier_clause} AND ("
            f"{conditions}) LIMIT 500"
        )
        results = self._fetch_records(
            sql, [*temporal_params, *params, *category_params, *patterns]
        )
        out: List[MemoryRecord] = []
        for r in results:
            if not self._matches_scope(r.payload):
                continue
            if not include_expired and self._is_expired(r.expires_at, at=expiry_ref):
                continue
            # Spec-07 (#71) D7: stale/unverified facts never presented as
            # current. invalidated facts are always excluded.
            vs = getattr(r, "verified_state", "current") or "current"
            if vs in ("stale", "invalidated"):
                continue
            out.append(r)
        # BM25-lite ranking over the candidate pool: per-token document
        # frequency -> idf, term frequency -> saturation, length normalization.
        # Pure-Python so no extension/network dependency; O(tokens x docs).
        # Token-based counting (issue #26): the prior code used
        # content_lower.count(t) which counts substring occurrences — "cat"
        # matched "caterpillar" and "concatenate". Tokenizing each doc with
        # the shared _tokenize() and using Counter gives exact word-boundary
        # matches. doc_len is now token count (not character count), which is
        # the standard BM25 length normalization.
        if out:
            doc_tokens = [Counter(_tokenize(r.content or "")) for r in out]
            doc_lens = [sum(c.values()) or 1 for c in doc_tokens]
            avg_len = max(1.0, sum(doc_lens) / len(doc_lens))
            n_docs = len(out)
            # df: number of docs containing each query token as an exact token.
            dfs = {t: sum(1 for c in doc_tokens if t in c) for t in tokens}
            for r, tokens_counter, dlen in zip(out, doc_tokens, doc_lens):
                score = 0.0
                for t in tokens:
                    tf = tokens_counter.get(t, 0)
                    if not tf:
                        continue
                    idf = math.log(1.0 + (n_docs - dfs[t] + 0.5) / (dfs[t] + 0.5))
                    score += idf * (BM25_K1 * tf / (tf + BM25_B * (dlen / avg_len)))
                r.similarity = score
            top = max((r.similarity for r in out), default=0.0)
            if top > 0:
                for r in out:
                    r.similarity /= top
        # Sort by text-match score descending.
        out.sort(key=lambda r: r.similarity, reverse=True)
        return out

    def _vector_search_raw(
        self, emb: List[float], limit: int, excluded: set[str],
        category_filter: str | None = None,
        project_id: str | None = None,
        namespace: str | None = None,
        client_scope: str | None = None,
        as_of: str | None = None,
        include_expired: bool = False,
        include_closed: bool = False,
        include_archived: bool = False,
    ) -> List[MemoryRecord]:
        """Vector similarity search. Returns filtered records ranked by cosine.

        Does NOT record retrieval — the caller is responsible for that.
        Raises on vector search errors so the caller can fall back.
        When *project_id* is provided, memories from other projects are
        excluded; global memories (project_id IS NULL) remain visible.

        Spec-05 (#67): *namespace* / *client_scope* filter the same way as
        _text_search_raw. None = no filter (backward compatible).

        When *include_expired* is True, the expiry filter is omitted (expired
        memories are returned, ranked normally).
        """
        project_clause = ""
        namespace_clause = ""
        client_scope_clause = ""
        # String-cast the query vector as a fixed-size array constant.
        # A Python-list parameter binds through an interpreted per-row path
        # (~1ms/row — measured ~1.2s at 1k rows); the string-cast form is
        # materialized once by the planner and scanned natively (~7ms at 1k,
        # ~14ms at 5k).  Exact — identical ranking, no approximation.  The
        # dimension is derived from the actual vector, so a model swap with a
        # different embedding size keeps working.
        # SR13: validate embedding dimension before SQL interpolation.
        if not emb or len(emb) > self._MAX_EMBEDDING_DIM:
            logger.warning(
                "embedder returned invalid dim %d — vector search skipped",
                len(emb) if emb else 0,
            )
            return []
        vec_text = "[" + ",".join(repr(float(x)) for x in emb) + "]"
        params: List[Any] = [vec_text]
        # Temporal filter: default to current (valid_to IS NULL),
        # or as_of (valid_from <= as_of AND (valid_to IS NULL OR valid_to > as_of)).
        # include_closed widens to closed versions — historical-query path.
        if as_of:
            temporal_clause = (
                "AND valid_from <= ? "
                "AND (valid_to IS NULL OR valid_to > ?) "
            )
            params.extend([as_of, as_of])
        elif include_closed:
            temporal_clause = " "
        else:
            temporal_clause = "AND valid_to IS NULL "
        expiry_ref = as_of if as_of else self._now()
        params.append(self.user_id)
        if include_expired:
            expiry_clause = ""
        else:
            expiry_clause = "  AND (expires_at IS NULL OR expires_at > ?) "
            params.append(expiry_ref)
        if project_id:
            project_clause = " AND (project_id IS NULL OR project_id = ?)"
            params.append(project_id)
        if namespace:
            namespace_clause = " AND namespace = ?"
            params.append(namespace)
        if client_scope:
            client_scope_clause = (
                " AND (client_scope IS NULL OR client_scope = ?)"
            )
            params.append(client_scope)
        # Push category_filter and excluded into SQL (issue #27).
        category_clause = ""
        if category_filter:
            category_clause = " AND category = ?"
            params.append(category_filter)
        if excluded:
            excluded_placeholders = ", ".join(["?" for _ in excluded])
            category_clause += f" AND LOWER(category) NOT IN ({excluded_placeholders})"
            params.extend(e.lower() for e in excluded)
        params.append(limit * 4)
        # P5.1 (#6): exclude archived records from the injection pool
        # unless include_archived is explicitly requested.
        tier_clause = "" if include_archived else " AND COALESCE(tier, 'active') = 'active'"
        sql = (
            f"SELECT *, list_cosine_similarity(embedding, CAST(? AS DOUBLE[{len(emb)}])) AS sim "
            "FROM memory_records "
            "WHERE COALESCE(status, 'active') = 'active' "
            f"  {temporal_clause} "
            "  AND embedding IS NOT NULL "
            "  AND (user_scope IS NULL OR user_scope = ?) "
            f"{expiry_clause}"
            f"{project_clause}"
            f"{namespace_clause}"
            f"{client_scope_clause}"
            f"{category_clause}"
            f"{tier_clause} "
            "ORDER BY sim DESC "
            "LIMIT ?"
        )
        results = self._fetch_records(sql, params, sim_col="sim")
        out: List[MemoryRecord] = []
        for r in results:
            if not self._matches_scope(r.payload):
                continue
            if not include_expired and self._is_expired(r.expires_at, at=expiry_ref):
                continue
            # Spec-07 (#71) D7: stale/unverified facts never presented as
            # current. invalidated facts are always excluded.
            vs = getattr(r, "verified_state", "current") or "current"
            if vs in ("stale", "invalidated"):
                continue
            out.append(r)
        return out

    def find_semantic_duplicate(
        self,
        content: str,
        min_similarity: float = 0.88,
    ) -> MemoryRecord | None:
        """Return the closest ACTIVE memory if it semantically covers *content*.

        Extraction-time dedupe: embed the proposed fact and run a top-1
        vector search against current memories (valid_to IS NULL, active,
        non-expired, same user scope). Returns the top record (with
        ``similarity`` set) when cosine >= *min_similarity*, else None.

        Fail-soft: any embedder or search error returns None so dedupe can
        never block memory capture. No LLM calls; local embedder only.
        """
        if not content or not content.strip():
            return None
        embedder = getattr(self, "embedder", None)
        if embedder is None or not hasattr(embedder, "embed"):
            return None
        try:
            emb = embedder.embed(content)
        except Exception:
            return None
        if not emb:
            return None
        try:
            hits = self._vector_search_raw(emb, limit=1, excluded=set())
        except Exception:
            return None
        if not hits:
            return None
        top = hits[0]
        sim = float(getattr(top, "similarity", 0.0) or 0.0)
        if sim >= min_similarity:
            return top
        return None

    # -- ranking: RRF + feedback + recency -----------------------------------

    _PHRASE_STOPWORDS = frozenset(
        ("the", "a", "an", "is", "are", "was", "were", "am", "be", "been",
         "i", "me", "my", "you", "your", "we", "our", "us", "this", "that",
         "these", "those", "it", "its", "of", "to", "in", "for", "and", "or",
         "but", "on", "with", "at", "by", "from", "as", "do", "does", "did",
         "what", "who", "how", "where", "when", "why", "which", "about", "so",
         "am", "very", "have", "has", "had", "would", "will", "can", "could")
    )

    _RRF_K = 20  # Lowered from 60 to sharpen relevance discrimination.
    # With k=60, rank 1 → 0.0164 and rank 10 → 0.0143 (spread ~0.002).
    # With k=20, rank 1 → 0.0476 and rank 10 → 0.0323 (spread ~0.015).
    # The wider spread lets relevance survive the importance adjustment.

    # Rank-1 survival guard (#38): if a single arm ranks an item #1 by a
    # clear margin, the fused top-k must still contain it. RRF can bury a
    # strong semantic rank-1 that lacks support in the other arm.
    # "Clear margin" = rank-1 score >= _RANK1_MARGIN_RATIO x rank-2 score.
    _RANK1_GUARD_TOP_K = 3
    _RANK1_MARGIN_RATIO = 1.5

    @classmethod
    def _rrf_fuse(
        cls,
        vector_results: List[MemoryRecord],
        text_results: List[MemoryRecord],
        *,
        enable_rank1_guard: bool = True,
    ) -> List[MemoryRecord]:
        """Fuse vector and text rankings via Reciprocal Rank Fusion.

        Includes the rank-1 survival guard (#38): an arm rank-1 with a
        clear margin over its own rank-2 must appear in the fused top-k
        even when the other arm ranks it poorly. ``enable_rank1_guard``
        is the fusion-policy knob — eval/probe_rank1_loss.py turns it
        off to measure the raw failure the guard fixes.
        """
        guard_ids = cls._rank1_guard_ids(vector_results, text_results)

        scores: Dict[str, float] = {}
        records_by_id: Dict[str, MemoryRecord] = {}

        for rank, record in enumerate(vector_results):
            rrf = 1.0 / (cls._RRF_K + rank + 1)
            scores[record.memory_id] = scores.get(record.memory_id, 0.0) + rrf
            records_by_id[record.memory_id] = record

        for rank, record in enumerate(text_results):
            rrf = 1.0 / (cls._RRF_K + rank + 1)
            scores[record.memory_id] = scores.get(record.memory_id, 0.0) + rrf
            # Prefer the vector record if it exists (has cosine sim);
            # otherwise use the text record.
            if record.memory_id not in records_by_id:
                records_by_id[record.memory_id] = record

        # Normalize: max possible is 2/(k+1) (rank 1 in both lists).
        max_rrf = 2.0 / (cls._RRF_K + 1)
        for mid, score in scores.items():
            records_by_id[mid].similarity = score / max_rrf  # 0.0 - 1.0

        fused = list(records_by_id.values())
        fused.sort(key=lambda r: r.similarity, reverse=True)
        if enable_rank1_guard:
            fused = cls._ensure_rank1_guards(fused, guard_ids)
        return fused

    @staticmethod
    def _rank1_guard_ids(
        vector_results: List[MemoryRecord],
        text_results: List[MemoryRecord],
    ) -> List[str]:
        """Memory ids that are a clear rank-1 in their arm (#38).

        An arm's rank-1 is "clear" when its score is at least
        _RANK1_MARGIN_RATIO x the arm's rank-2 score (and rank-2 has a
        positive score). Must be called BEFORE fusion overwrites
        ``similarity`` with the fused score — callers that fuse the same
        arm records twice (e.g. the probe) must pass fresh copies to the
        second call.
        """
        ids: List[str] = []
        for arm in (vector_results, text_results):
            if len(arm) < 2:
                continue
            s1 = float(getattr(arm[0], "similarity", 0.0) or 0.0)
            s2 = float(getattr(arm[1], "similarity", 0.0) or 0.0)
            if s1 > 0.0 and (s2 <= 0.0 or s1 >= s2 * StoreRetrievalMixin._RANK1_MARGIN_RATIO):
                ids.append(arm[0].memory_id)
        return ids

    @classmethod
    def _ensure_rank1_guards(
        cls,
        fused: List[MemoryRecord],
        guard_ids: List[str],
    ) -> List[MemoryRecord]:
        """Pull clear arm rank-1s that RRF dropped below top-k back in (#38).

        No-op when every guard is already in the fused top-k — the common
        case where the arms agree. When a guard is missing (RRF buried it),
        the guards move to the front in fused order: the arm that ranked
        them #1 by a clear margin wins the tie.
        """
        if not guard_ids or not fused:
            return fused
        k = cls._RANK1_GUARD_TOP_K
        guard_set = set(guard_ids)
        inside_ids = {r.memory_id for r in fused[:k]}
        if all(g in inside_ids for g in guard_set):
            return fused
        head = [r for r in fused if r.memory_id in guard_set]
        tail = [r for r in fused if r.memory_id not in guard_set]
        return head + tail

    @staticmethod
    def _parse_timestamp(ts: str | None) -> datetime | None:
        """Parse an ISO-8601 timestamp to a timezone-aware UTC datetime.

        Handles naive timestamps (no Z/offset) by assuming UTC — the
        shared normalization boundary for #28 finding 2 and #33 finding 3.
        Returns None if the timestamp is missing or unparseable, and logs
        a warning so silent zero-boost regressions are visible.
        """
        if not ts:
            return None
        try:
            parsed = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            # Assume UTC for naive timestamps (no tzinfo).
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed
        except (ValueError, TypeError, AttributeError) as exc:
            logger.warning("Unparseable timestamp %r: %s", ts, exc)
            return None

    @staticmethod
    def _recency_boost(created_at: str | None) -> float:
        """Exponential decay: +0.10 today, ~0.037 at 90 days, ~0.014 at 180.

        Returns 0.0 if the timestamp is missing or unparseable.
        """
        if not created_at:
            return 0.0
        created = StoreRetrievalMixin._parse_timestamp(created_at)
        if created is None:
            return 0.0
        days_old = max(0, (datetime.now(timezone.utc) - created).days)
        return 0.10 * math.exp(-days_old / 90.0)

    # Importance scoring weights — configurable via the store's attributes.
    # These are applied as additive adjustments to similarity (0-1 scale).
    _IMPORTANCE_HELPFUL_WEIGHT = 0.05       # per helpful vote
    _IMPORTANCE_DISMISSED_WEIGHT = -0.10    # per dismissed vote (before decay)
    _IMPORTANCE_CONFIDENCE_WEIGHT = 0.08    # * (confidence - 0.5)
    _IMPORTANCE_RETRIEVAL_WEIGHT = 0.02     # per retrieval (capped at 20)
    _IMPORTANCE_RETRIEVAL_CAP = 20
    _IMPORTANCE_AGE_DECAY_PER_DAY = 0.0005  # slow age penalty
    _IMPORTANCE_AGE_DECAY_CAP_DAYS = 730    # cap at 2 years
    _IMPORTANCE_DORMANCY_DECAY_PER_DAY = 0.001  # decay for not being retrieved
    _IMPORTANCE_DORMANCY_CAP_DAYS = 365     # cap at 1 year
    # Dismissal forgiveness: dismissals age out over this many days so a
    # single bad dismiss (frustration click, misattribution) doesn't
    # permanently sink a memory. The effective dismissed penalty is:
    #   -0.10 * dismissed_count * max(0, 1 - days_since / FORGIVENESS_DAYS)
    _IMPORTANCE_DISMISSAL_FORGIVENESS_DAYS = 180  # 6 months

    @classmethod
    def _importance_adjustment(cls, r: MemoryRecord) -> float:
        """Compute the importance-based similarity adjustment for a record.

        This is a transparent, additive scoring formula that combines:
          - Feedback signals (helpful/dismissed votes, with dismissal forgiveness)
          - Extraction confidence
          - Retrieval frequency (capped — popular memories get a boost)
          - Recency boost (exponential decay from creation date)
          - Age penalty (slow linear decay — old memories slowly fade)
          - Dormancy penalty (memories not retrieved recently slowly fade)

        Dismissal forgiveness: dismissals age out over
        _IMPORTANCE_DISMISSAL_FORGIVENESS_DAYS so a single bad dismiss
        (frustration click, misattribution) doesn't permanently sink a
        memory. Uses updated_at as a proxy for when the dismissal happened.
        """
        adj = 0.0
        # Feedback: helpful votes are permanent positive signal
        adj += cls._IMPORTANCE_HELPFUL_WEIGHT * r.helpful_count
        # Dismissed votes decay over time (forgiveness factor)
        if r.dismissed_count > 0:
            dismissal_factor = 1.0  # full penalty if no timestamp available
            if r.updated_at:
                updated = cls._parse_timestamp(r.updated_at)
                if updated is not None:
                    days_since = max(0, (datetime.now(timezone.utc) - updated).days)
                    dismissal_factor = max(
                        0.0,
                        1.0 - days_since / cls._IMPORTANCE_DISMISSAL_FORGIVENESS_DAYS,
                    )
            adj += cls._IMPORTANCE_DISMISSED_WEIGHT * r.dismissed_count * dismissal_factor
        # Confidence: reward high-confidence extractions
        if r.confidence is not None:
            adj += cls._IMPORTANCE_CONFIDENCE_WEIGHT * (r.confidence - 0.5)
        # Retrieval frequency: frequently-retrieved memories are important
        # (capped to prevent rich-get-richer spiral)
        adj += cls._IMPORTANCE_RETRIEVAL_WEIGHT * min(r.retrieval_count, cls._IMPORTANCE_RETRIEVAL_CAP)
        # Recency boost (from creation date)
        adj += cls._recency_boost(r.created_at)
        # Age penalty: slow linear decay
        if r.created_at:
            created = cls._parse_timestamp(r.created_at)
            if created is not None:
                age_days = max(0, (datetime.now(timezone.utc) - created).days)
                adj -= cls._IMPORTANCE_AGE_DECAY_PER_DAY * min(age_days, cls._IMPORTANCE_AGE_DECAY_CAP_DAYS)
        # Dormancy penalty: memories not retrieved recently slowly fade
        if r.last_retrieved_at:
            last_ret = cls._parse_timestamp(r.last_retrieved_at)
            if last_ret is not None:
                dormant_days = max(0, (datetime.now(timezone.utc) - last_ret).days)
                adj -= cls._IMPORTANCE_DORMANCY_DECAY_PER_DAY * min(dormant_days, cls._IMPORTANCE_DORMANCY_CAP_DAYS)
        return adj

    # Clamp the importance adjustment so it can't erase a relevance gap.
    # With k=20 RRF, the relevance spread between rank 1 and rank 10 is
    # ~0.015. An unclamped adjustment of +0.50 (from retrieval_count alone)
    # swamps that entirely. We split the adjustment into:
    #   - base signals (recency, age, dormancy) that are similar for nearby
    #     memories and don't discriminate — these get a tight clamp
    #   - feedback signals (helpful, dismissed, confidence, retrieval) that
    #     DO discriminate between equally-relevant memories — these get a
    #     wider clamp so a helpful vote still breaks ties
    _IMPORTANCE_BASE_CLAMP = 0.03   # recency/age/dormancy: tight clamp
    _IMPORTANCE_FEEDBACK_CLAMP = 0.08  # helpful/dismissed/confidence/retrieval

    @classmethod
    def _importance_adjustment_split(cls, r: MemoryRecord) -> tuple[float, float]:
        """Compute importance adjustment split into (base, feedback) components.

        base: recency + age + dormancy (similar for nearby memories)
        feedback: helpful + dismissed + confidence + retrieval (discriminates)
        """
        base = cls._recency_boost(r.created_at)
        # Age penalty
        if r.created_at:
            created = cls._parse_timestamp(r.created_at)
            if created is not None:
                age_days = max(0, (datetime.now(timezone.utc) - created).days)
                base -= cls._IMPORTANCE_AGE_DECAY_PER_DAY * min(age_days, cls._IMPORTANCE_AGE_DECAY_CAP_DAYS)
        # Dormancy penalty
        if r.last_retrieved_at:
            last_ret = cls._parse_timestamp(r.last_retrieved_at)
            if last_ret is not None:
                dormant_days = max(0, (datetime.now(timezone.utc) - last_ret).days)
                base -= cls._IMPORTANCE_DORMANCY_DECAY_PER_DAY * min(dormant_days, cls._IMPORTANCE_DORMANCY_CAP_DAYS)

        feedback = cls._IMPORTANCE_HELPFUL_WEIGHT * r.helpful_count
        if r.dismissed_count > 0:
            dismissal_factor = 1.0
            if r.updated_at:
                updated = cls._parse_timestamp(r.updated_at)
                if updated is not None:
                    days_since = max(0, (datetime.now(timezone.utc) - updated).days)
                    dismissal_factor = max(0.0, 1.0 - days_since / cls._IMPORTANCE_DISMISSAL_FORGIVENESS_DAYS)
            feedback += cls._IMPORTANCE_DISMISSED_WEIGHT * r.dismissed_count * dismissal_factor
        if r.confidence is not None:
            feedback += cls._IMPORTANCE_CONFIDENCE_WEIGHT * (r.confidence - 0.5)
        feedback += cls._IMPORTANCE_RETRIEVAL_WEIGHT * min(r.retrieval_count, cls._IMPORTANCE_RETRIEVAL_CAP)

        return base, feedback

    @classmethod
    def _apply_feedback_and_recency(cls, records: List[MemoryRecord]) -> None:
        """Adjust ``similarity`` in-place using importance scoring.

        The adjustment is split into base signals (recency/age/dormancy)
        and feedback signals (helpful/dismissed/confidence/retrieval).
        Each is clamped separately so:
        - base signals can't swamp relevance (tight ±0.03 clamp)
        - feedback signals can still break ties between equally-relevant
          memories (wider ±0.08 clamp)
        Without this split-clamp, a +0.40 retrieval boost (identical for
        all memories in the eval corpus) would swamp the ~0.015 RRF gap.
        """
        for r in records:
            base, feedback = cls._importance_adjustment_split(r)
            base = max(-cls._IMPORTANCE_BASE_CLAMP, min(cls._IMPORTANCE_BASE_CLAMP, base))
            feedback = max(-cls._IMPORTANCE_FEEDBACK_CLAMP, min(cls._IMPORTANCE_FEEDBACK_CLAMP, feedback))
            r.similarity = max(0.0, r.similarity + base + feedback)
        records.sort(key=lambda r: r.similarity, reverse=True)

    # -- P2C: recency/supersede-aware rank demotion (flag-gated, default OFF) --
    # Formally-superseded versions are already filtered out at retrieval time
    # (valid_to IS NULL), so this only covers the rarer case of two UNLINKED flat
    # memories that state the same fact (old address + new address) with separate
    # memory_ids. When a near-duplicate pair surfaces in the same result set, the
    # newer one should outrank the older. Gated by _P2C_ENABLED (default False) so
    # this never changes behaviour out of the box. Cheap: token-overlap only, no
    # LLM, no embeddings. Bounded: an older memory sinks at most a few ranks.
    _P2C_ENABLED = False
    _P2C_MIN_OVERLAP = 0.60        # token-Jaccard above this = "same fact"
    _P2C_MAX_SINK = 3              # bounded: never leapfrog more than this many ranks
    _P2C_SINK_EPSILON = 0.005      # newer must clear older by this tiny margin

    @staticmethod
    def _p2c_overlap(a: str, b: str) -> float:
        """Token Jaccard similarity between two content strings (tokenized).

        Uses the shared _tokenize() (regex word-boundary, punctuation-
        aware) so overlap is consistent with the rest of the pipeline —
        "Cape Town." and "Cape Town" are the same token set here, not
        different ones (audit B3).
        """
        if not a or not b:
            return 0.0
        sa = set(_tokenize(a))
        sb = set(_tokenize(b))
        if not sa or not sb:
            return 0.0
        inter = len(sa & sb)
        union = len(sa | sb)
        return inter / union if union else 0.0

    @staticmethod
    def _p2c_ts(content: str) -> float:
        """Parse an ISO created_at to an epoch float, or -inf if unparseable."""
        if not content:
            return float("-inf")
        parsed = StoreRetrievalMixin._parse_timestamp(content)
        if parsed is None:
            return float("-inf")
        return parsed.timestamp()

    @classmethod
    def _apply_p2c(cls, records: List[MemoryRecord]) -> None:
        """If enabled, promote the newer member of each near-duplicate pair above the older.

        SR11: O(n²) over the result set — gated by ``_P2C_ENABLED = False``
        (default off). If ever enabled, the result-size guard below caps
        the comparison window to avoid quadratic blowup on large result sets.
        """
        if not cls._P2C_ENABLED:
            return
        # SR11: cap the comparison window to avoid O(n²) blowup.
        if len(records) > 50:
            return
        for i in range(len(records)):
            for j in range(i + 1, len(records)):
                ri, rj = records[i], records[j]
                if cls._p2c_overlap(ri.content or "", rj.content or "") < cls._P2C_MIN_OVERLAP:
                    continue
                ti, tj = cls._p2c_ts(ri.created_at), cls._p2c_ts(rj.created_at)
                if ti == tj:
                    continue
                # Identical facts at two timestamps; determine older/newer
                # WITHOUT mutating the loop counters i/j (issue #28 finding 1).
                # Reassigning i/j mid-iteration corrupts the scan: the inner
                # loop continues from the swapped j, revisiting pairs in
                # reversed order, and the bounded-sink check evaluates
                # against swapped indices.
                if ti > tj:
                    older_idx, newer_idx = j, i  # rj is older, ri is newer
                else:
                    older_idx, newer_idx = i, j  # ri is older, rj is newer
                # Only demote when the older currently ranks higher AND the pair
                # is within the bounded sink window (no big leapfrogs).
                if older_idx < newer_idx and (newer_idx - older_idx) <= cls._P2C_MAX_SINK:
                    older, newer = records[older_idx], records[newer_idx]
                    # Guarantee the newer outranks the older by a small epsilon.
                    target = min(1.0, older.similarity + cls._P2C_SINK_EPSILON)
                    if newer.similarity < target:
                        newer.similarity = target
        records.sort(key=lambda r: r.similarity, reverse=True)

    def _record_retrieval(self, records: List[MemoryRecord]) -> None:
        """Record only memories that were actually returned to the caller."""
        if not records:
            return
        now = self._now()
        ids = [record.memory_id for record in records]
        try:
            with self._lock:
                assert self.connection is not None
                # Single batched UPDATE (was a per-ID loop: 96 round-trips ->
                # 1). Measured 2026-08-22: the loop cost 0.5-1.3s on a 96-row
                # search at ~1k records; one IN-list statement is ~1 round-trip.
                placeholders = ", ".join("?" for _ in ids)
                self.connection.execute(
                    f"""UPDATE memory_records
                        SET retrieval_count = COALESCE(retrieval_count, 0) + 1,
                            last_retrieved_at = ?
                        WHERE memory_id IN ({placeholders})
                          AND COALESCE(status, 'active') = 'active'
                          AND valid_to IS NULL
                          AND (user_scope IS NULL OR user_scope = ?)""",
                    [now, *ids, self.user_id],
                )
            for record in records:
                record.retrieval_count += 1
                record.last_retrieved_at = now
        except Exception as exc:
            # A read-only fallback connection must still be able to search.
            logger.debug("Could not record memory retrieval: %s", exc)

    def record_retrieval(self, memory_ids: List[str]) -> None:
        """Explicitly credit retrieval for the final injected memory list.

        The provider truncates a suppress_retrieval search to the memories it
        actually injects, then re-records them here — so pool filler and
        internal sub-queries never gain popularity credit.
        """
        if not memory_ids:
            return
        ids = [str(memory_id) for memory_id in memory_ids if str(memory_id).strip()]
        if not ids:
            return
        placeholders = ", ".join("?" for _ in ids)
        records = self._fetch_records(
            f"SELECT * FROM memory_records WHERE memory_id IN ({placeholders})"
            " AND (user_scope IS NULL OR user_scope = ?)"
            " AND COALESCE(status, 'active') = 'active' AND valid_to IS NULL",
            [*ids, self.user_id],
        )
        self._record_retrieval(records)

    def get_memories_by_ids(
        self,
        memory_ids: List[str],
        *,
        include_quarantined: bool = False,
    ) -> List[MemoryRecord]:
        """Fetch visible, in-scope memories by ID for graph-aware retrieval."""
        ids = [str(memory_id) for memory_id in memory_ids if str(memory_id).strip()]
        if not ids:
            return []
        placeholders = ", ".join("?" for _ in ids)
        status_clause = "" if include_quarantined else " AND COALESCE(status, 'active') = 'active' AND valid_to IS NULL"
        records = self._fetch_records(
            f"SELECT * FROM memory_records WHERE memory_id IN ({placeholders})"
            " AND (user_scope IS NULL OR user_scope = ?)"
            f"{status_clause}",
            [*ids, self.user_id],
        )
        by_id = {record.memory_id: record for record in records}
        return [by_id[memory_id] for memory_id in ids if memory_id in by_id
                and self._matches_scope(by_id[memory_id].payload)
                and not self._is_expired(by_id[memory_id].expires_at)]

    def search(
        self,
        query: str,
        *args: Any,
        **kwargs: Any,
    ) -> List[MemoryRecord]:
        """Run retrieval through the configured retriever.
        The default ``DuckDBRetriever`` wraps the existing scan-based hybrid
        search (vector + ILIKE + RRF fusion).  Alternative engines (ANN,
        BM25, graph-candidate) can be plugged in via ``set_retriever``
        without touching the provider.  Behavior is identical to the
        pre-seam implementation.
        """
        _t0 = time.perf_counter()
        try:
            retriever = getattr(self, "_retriever", None)
            if retriever is None:
                retriever = self._retriever = DuckDBRetriever(self)
            return retriever.search(query, *args, **kwargs)
        finally:
            self._record_scale_metric(time.perf_counter() - _t0)

    def _record_scale_metric(self, elapsed_s: float) -> None:
        """Track query latency and corpus size; warn when thresholds cross.

        The measured trigger for expensive-engine gating (ANN/BM25, fact
        families): a rolling p95 latency window and the active record count.
        Warnings fire once per crossing and are actionable — they name the
        threshold that was exceeded, not a vague "slow".
        """
        try:
            self._scale_latencies.append(elapsed_s * 1000.0)
            self._scale_queries += 1
            # Sample the record count every 25 queries (COUNT(*) on every
            # query would add overhead we're trying to measure).
            if self._scale_queries - self._scale_last_count_check >= 25:
                self._scale_last_count_check = self._scale_queries
                try:
                    self._scale_record_count = int(self.count())
                except Exception:
                    pass
            n = len(self._scale_latencies)
            if n < 5:
                return  # too few samples to be meaningful
            avg_ms = sum(self._scale_latencies) / n
            p95_ms = sorted(self._scale_latencies)[int(n * 0.95) - 1]
            over_latency = p95_ms > self._scale_warn_latency_ms
            over_count = (self._scale_record_count or 0) > self._scale_warn_records
            if over_latency or over_count:
                self._scale_warnings_fired += 1
                logger.warning(
                    "ARGOS_SCALE: p95=%.0fms avg=%.0fms (warn>%.0fms) "
                    "records=%s (warn>%s) — if this persists, enable ANN/BM25 "
                    "indexing or fact-family consolidation per the scaling "
                    "roadmap (trigger: %s)",
                    p95_ms, avg_ms, self._scale_warn_latency_ms,
                    self._scale_record_count, self._scale_warn_records,
                    "latency" if over_latency else "corpus-size",
                )
        except Exception:
            pass  # metrics must never break retrieval

    def set_scale_thresholds(self, warn_latency_ms: float, warn_records: int) -> None:
        """Configure the scale-trigger thresholds (from the provider config)."""
        self._scale_warn_latency_ms = float(warn_latency_ms)
        self._scale_warn_records = int(warn_records)

    def get_scale_metrics(self) -> Dict[str, Any]:
        """Return current scale-trigger state (for dashboards/handoffs)."""
        n = len(self._scale_latencies)
        p95 = sorted(self._scale_latencies)[int(n * 0.95) - 1] if n >= 5 else 0.0
        return {
            "queries_measured": self._scale_queries,
            "window": n,
            "avg_latency_ms": round(sum(self._scale_latencies) / n, 1) if n else 0.0,
            "p95_latency_ms": round(p95, 1),
            "max_latency_ms": round(max(self._scale_latencies), 1) if n else 0.0,
            "record_count": self._scale_record_count,
            "warnings_fired": self._scale_warnings_fired,
            "warn_latency_ms": self._scale_warn_latency_ms,
            "warn_records": self._scale_warn_records,
        }

    def set_retriever(self, retriever: Any) -> None:
        """Swap the retrieval engine (advanced; must match the protocol)."""
        self._retriever = retriever

    def _hybrid_search(
        self,
        query: str,
        limit: int = 5,
        exclude_categories: List[str] | None = None,
        category_filter: str | None = None,
        project_id: str | None = None,
        namespace: str | None = None,
        client_scope: str | None = None,
        as_of: str | None = None,
        suppress_retrieval: bool = False,
        include_expired: bool = False,
        include_closed: bool = False,
        include_archived: bool = False,
    ) -> List[MemoryRecord]:
        """Hybrid search: RRF-fused vector + text, with optional cross-encoder
        re-ranking, feedback, and recency.

        When embeddings are available, runs vector and text search in
        parallel and fuses results via Reciprocal Rank Fusion.  When
        embeddings are unavailable, falls back to text-only search.  If
        a cross-encoder reranker is available, the top candidates are
        re-scored with full bidirectional attention before final ranking.
        In all cases, the final ranking is adjusted by feedback signals
        (helpful/dismissed), confidence, and recency.

        When *project_id* is provided, memories from other projects are
        excluded; global memories (project_id IS NULL) remain visible.

        When *suppress_retrieval* is True, _record_retrieval is skipped —
        eval/diagnostic runs won't inflate retrieval_count on the memories
        they search. This prevents eval self-pollution where repeated eval
        runs pump the retrieval_count of eval-relevant memories to the cap,
        flattening the retrieval signal as a discriminator.

        When *include_expired* is True, expired memories are included in
        results (ranked normally) — for auditing "what did I know then".
        """
        # Validate as_of as ISO-8601 (#28 finding 3): a malformed value
        # silently behaves as a garbage cutoff in SQL temporal comparisons.
        # Normalize to the canonical aware-UTC form (#80) so downstream SQL
        # lexicographic comparisons are consistent across mixed ISO formats
        # (Z vs +00:00, date-only vs full timestamp).
        if as_of:
            parsed_as_of = self._parse_timestamp(as_of)
            if parsed_as_of is None:
                logger.warning("Invalid as_of timestamp %r — ignoring temporal filter", as_of)
                as_of = None
            else:
                as_of = parsed_as_of.isoformat()

        excluded = {c.lower() for c in (exclude_categories or [])}
        emb: List[float] = []
        if self.embedder and hasattr(self.embedder, "embed"):
            try:
                emb = self.embedder.embed(query, is_query=True)
            except Exception as exc:
                # Query-embed failure must NOT take down the whole search
                # (issue #45): a broken embedder (None model, load failure,
                # .lower() crash) used to propagate here and empty every
                # retrieval silently, because fail-soft callers swallowed
                # the exception and saw "no relevant memories". Fall back
                # to text-only retrieval — degraded but honest — and log
                # a warning so the failure is visible.
                logger.warning(
                    "Query embedding failed — falling back to text-only "
                    "search: %s", exc,
                )
                emb = []

        # Retrieve more candidates than requested so the reranker has a
        # larger pool to work with. If no reranker, just use limit.
        # Phrase-lift needs a wider pool so exact-phrase matches sitting just
        # outside the window can be pulled in.
        reranker_top_n = getattr(self, "_reranker_top_n", 20)
        phrase_pool = getattr(self, "_phrase_lift_pool", 0)
        if phrase_pool:
            pool_size = max(limit, reranker_top_n, phrase_pool)
        else:
            pool_size = max(limit, reranker_top_n) if self.reranker else limit

        # Gather candidate results from both paths.
        vector_results: List[MemoryRecord] = []
        text_results: List[MemoryRecord] = self._text_search_raw(
            query, pool_size, excluded, category_filter,
            project_id=project_id, namespace=namespace,
            client_scope=client_scope, as_of=as_of,
            include_expired=include_expired, include_closed=include_closed,
            include_archived=include_archived,
        )

        if emb:
            try:
                vector_results = self._vector_search_raw(
                    emb, pool_size, excluded, category_filter,
                    project_id=project_id, namespace=namespace,
                    client_scope=client_scope, as_of=as_of,
                    include_expired=include_expired,
                    include_closed=include_closed,
                    include_archived=include_archived,
                )
            except Exception as exc:
                if not self._is_vector_search_unavailable(exc):
                    logger.warning("Vector search error: %s", exc)
                vector_results = []

        # Fuse or select.
        if vector_results and text_results:
            fused = self._rrf_fuse(vector_results, text_results)
        elif vector_results:
            fused = vector_results
        elif text_results:
            fused = text_results
        else:
            return []

        # Preserve raw similarity (PRE-reranker, pre-importance) for gates
        # that need pure retrieval strength (e.g. the query-expansion
        # trigger). Saved before the cross-encoder blend so a reranker lift
        # cannot mask a weak bi-encoder retrieval (audit B1).
        for r in fused:
            r.raw_similarity = r.similarity

        # Cross-encoder re-ranking: re-score the top N candidates with
        # full bidirectional attention. The cross-encoder score is blended
        # with the existing RRF similarity (not replacing it) so we keep
        # the bi-encoder's signal while adding the cross-encoder's nuance.
        # Blend: 20% cross-encoder + 80% original similarity, after
        # normalizing the cross-encoder scores to 0-1. This is intentionally
        # conservative — the cross-encoder acts as a gentle tie-breaker
        # rather than overriding the bi-encoder's ranking.
        if self.reranker and len(fused) > 1:
            rerank_pool = fused[:reranker_top_n]
            documents = [r.content for r in rerank_pool]
            scores = self.reranker.score(query, documents)
            if scores and len(scores) == len(rerank_pool):
                min_s, max_s = min(scores), max(scores)
                range_s = max_s - min_s
                for i, record in enumerate(rerank_pool):
                    if range_s > 0:
                        ce_norm = (scores[i] - min_s) / range_s
                    else:
                        ce_norm = 0.5
                    record.similarity = 0.8 * record.similarity + 0.2 * ce_norm
                rerank_pool.sort(key=lambda r: r.similarity, reverse=True)
                fused = rerank_pool + fused[reranker_top_n:]

        # Preserve raw similarity (pre-importance) for gates that need
        # pure retrieval strength (e.g. query expansion trigger).
        # NOTE: the authoritative raw capture happens BEFORE the reranker
        # block (see above); this comment block is intentionally empty.
        # Phrase-lift and importance below mutate similarity only.

        # Exact-phrase lift (optional, default off): reward contiguous
        # query bigrams present verbatim in the memory, which the unigram-
        # only text search never scores. Fixes the class of query where the
        # gold memory shares the exact phrase (e.g. "who is the sales
        # director" -> "...Raymond is the Sales Director...") but was ranked
        # low because token overlap tied it with merely-similar content.
        _alpha = getattr(self, "_phrase_lift_alpha", 0.0)
        if _alpha and _alpha > 0.0:
            qwords = [w for w in _tokenize(query) if w not in self._PHRASE_STOPWORDS]
            qbigrams = [(t0, t1) for t0, t1 in zip(qwords, qwords[1:])]
            for r in fused:
                if not qbigrams or not r.content:
                    continue
                c = r.content.lower()
                ngram_win = sum(1 for a, b in qbigrams if f"{a} {b}" in c)
                if ngram_win:
                    r.similarity += _alpha * (ngram_win / len(qbigrams))
            fused.sort(key=lambda r: r.similarity, reverse=True)

        # Apply feedback weighting and recency boost, then truncate.
        self._apply_feedback_and_recency(fused)
        self._apply_p2c(fused)  # P2C: demote older of a near-duplicate pair (flag-gated)
        final = fused[:limit]
        # #142: clamp final similarity to [0, 1] — additive stages (phrase-lift,
        # importance, graph boost) can push a high base similarity above 1.0.
        # The clamp is applied AFTER sorting so feedback signals still break
        # ties (the unclamped value determines rank order; the clamped value
        # is what the caller sees).
        for r in final:
            if r.similarity > 1.0:
                r.similarity = 1.0
            elif r.similarity < 0.0:
                r.similarity = 0.0
        if not suppress_retrieval:
            self._record_retrieval(final)
        return final

    # -- write operations -----------------------------------------------------

    # #248: tuning constants consolidated in tuning.py. Class attrs alias
    # the module-level constants for backward compatibility with any code
    # that references them via the class.
    _DEDUP_SIMILARITY_THRESHOLD = DEDUP_SIMILARITY_THRESHOLD
    _MAX_EMBEDDING_DIM = MAX_EMBEDDING_DIM

    def _content_exists(
        self, content: str, category: str,
        *,
        namespace: str | None = None,
        client_scope: str | None = None,
        doc_class: str | None = None,
        source_doc_id: str | None = None,
    ) -> str | None:
        """Check if a very similar content already exists (dedup).

        Returns a reason string (``"exact"``, ``"substring"``,
        ``"semantic"``) if a duplicate is found, or ``None`` if the content
        is new. The reason lets the caller surface *why* a dedup drop
        happened instead of a silent None (#82).

        Thin wrapper over :meth:`_find_current_similar` (which returns the
        matching ``memory_id`` alongside the reason) — kept for the
        existing dedup call sites that only need the reason.

        Doc-identity scoping (#115): when *namespace* is set and not
        ``"conversation"``, the three dedup layers are scoped by
        (namespace, client_scope, doc_class, source_doc_id) so distinct
        documents never collapse on content proximity alone.  When
        *namespace* is ``"conversation"`` or None, dedup remains global
        (the conversational paraphrase-dedup feature is preserved).
        """
        memory_id, reason = self._find_current_similar(
            content, category,
            namespace=namespace,
            client_scope=client_scope,
            doc_class=doc_class,
            source_doc_id=source_doc_id,
        )
        return reason

    @staticmethod
    def _doc_identity_scope_clause(
        namespace: str | None,
        client_scope: str | None,
        doc_class: str | None,
        source_doc_id: str | None,
    ) -> tuple[str, list]:
        """Build a SQL fragment + params that narrows dedup to the same
        document identity (#115).

        Returns ``("", [])`` when the caller is in the conversation tier
        (namespace is None or ``"conversation"``) — global dedup is
        preserved for paraphrase collapse, which is a feature there.

        For the doc tier, each field uses ``IS NOT DISTINCT FROM`` so NULL
        matches NULL (a doc-tier fact with no client_scope dedups against
        other same-namespace facts that also have no client_scope).
        """
        if not namespace or namespace == "conversation":
            return "", []
        clauses = []
        params: list = []
        for col, val in (
            ("namespace", namespace),
            ("client_scope", client_scope),
            ("doc_class", doc_class),
            ("source_doc_id", source_doc_id),
        ):
            clauses.append(f"{col} IS NOT DISTINCT FROM ?")
            params.append(val)
        return " AND " + " AND ".join(clauses), params

    def _find_current_similar(
        self, content: str, category: str,
        *,
        namespace: str | None = None,
        client_scope: str | None = None,
        doc_class: str | None = None,
        source_doc_id: str | None = None,
    ) -> tuple[str | None, str | None]:
        """Find a current record restating *content*; return ``(memory_id, reason)``.

        Same three-layer detection as :meth:`_content_exists` (exact,
        substring, semantic) but also returns the ``memory_id`` of the
        matched current record so callers that need to act on the match
        — e.g. :meth:`ingest_versioned` (#74) routing a restatement
        through ``update_memory`` — don't have to re-scan.

        Returns ``(None, None)`` when the content is new.

        Doc-identity scoping (#115): when *namespace* is set and not
        ``"conversation"``, all three layers filter by the caller's
        (namespace, client_scope, doc_class, source_doc_id) so facts from
        distinct documents survive even when their content is identical or
        semantically near-identical.  Conversation-tier dedup (namespace
        ``"conversation"`` or None) remains global.

        Three-layer check:
        1. Exact match (case-sensitive, same category).
        2. Substring containment (case-insensitive, for >20 char strings)
           gated by an overlap ratio ≥ 0.8 — the shorter string must cover
           ≥80% of the longer string so distinct facts that share a long
           common prefix ("...in 2021" vs "...in 2022") are not silently
           dropped (#82).
        3. Semantic similarity (cosine similarity on embeddings, when an
           embedder is available).  Catches paraphrased duplicates like
           "User is married to Sam" vs "Sam is the user's partner".
        """
        scope_sql, scope_params = self._doc_identity_scope_clause(
            namespace, client_scope, doc_class, source_doc_id,
        )
        with self._lock:
            assert self.connection is not None
            # Layer 1: exact match (only against current versions).
            result = self.connection.execute(
                f"""SELECT memory_id FROM memory_records
                  WHERE content = ? AND category = ?
                    AND valid_to IS NULL
                    AND (user_scope IS NULL OR user_scope = ?)
                    {scope_sql}
                  LIMIT 1""",
                [content, category, self.user_id, *scope_params],
            ).fetchone()
            if result:
                return result[0], "exact"
            # Layer 2: substring containment (case-insensitive, current only).
            # ORDER BY created_at DESC so the scan window is recency-ordered,
            # not arbitrary storage order (#82).
            # SR10: reduced from 500 to 200 — bounded O(n) substring
            # comparisons on every remember() call. 200 is sufficient for
            # recency-ordered dedup; the exact-match Layer 1 catches most.
            result = self.connection.execute(
                f"""SELECT memory_id, content FROM memory_records
                  WHERE category = ?
                    AND valid_to IS NULL
                    AND (user_scope IS NULL OR user_scope = ?)
                    {scope_sql}
                  ORDER BY created_at DESC
                  LIMIT 200""",
                [category, self.user_id, *scope_params],
            ).fetchall()
            content_lower = content.lower().strip()
            for existing_id, existing in result:
                existing_lower = existing.lower().strip()
                if content_lower == existing_lower:
                    return existing_id, "substring"
                # Skip very short strings to avoid false positives.
                if len(content_lower) > 20 and len(existing_lower) > 20:
                    if content_lower in existing_lower or existing_lower in content_lower:
                        # #82: gate behind an overlap ratio so a long shared
                        # prefix does not declare distinct facts as duplicates.
                        shorter = min(len(content_lower), len(existing_lower))
                        longer = max(len(content_lower), len(existing_lower))
                        if longer > 0 and shorter / longer >= 0.8:
                            return existing_id, "substring"
                        # Below the overlap gate — log at debug so the
                        # near-miss is traceable but the fact is saved.
                        logger.debug(
                            "Substring dedup overlap gate blocked drop: "
                            "%d/%d (%.1f%%) for %r vs %r",
                            shorter, longer, 100 * shorter / longer,
                            content_lower[:50], existing_lower[:50],
                        )
            # Layer 3: semantic similarity via embeddings.
            if self.embedder and hasattr(self.embedder, "embed"):
                emb = self.embedder.embed(content)
                if emb:
                    # SR13: validate embedding dimension before SQL interpolation.
                    if len(emb) > self._MAX_EMBEDDING_DIM:
                        logger.warning(
                            "embedder returned invalid dim %d — semantic dedup skipped",
                            len(emb),
                        )
                        return None, None
                    try:
                        # Same string-cast constant trick as _vector_search_raw
                        # (Python-list params bind per-row; ~1s at 1k rows).
                        vec_text = "[" + ",".join(repr(float(x)) for x in emb) + "]"
                        result = self.connection.execute(
                            f"""SELECT memory_id FROM memory_records
                              WHERE category = ? AND embedding IS NOT NULL
                                AND valid_to IS NULL
                                AND (user_scope IS NULL OR user_scope = ?)
                                {scope_sql}
                                AND list_cosine_similarity(embedding, CAST(? AS DOUBLE[{len(emb)}])) > ?
                              ORDER BY list_cosine_similarity(embedding, CAST(? AS DOUBLE[{len(emb)}])) DESC
                               LIMIT 1""",
                            [category, self.user_id, *scope_params,
                             vec_text, self._DEDUP_SIMILARITY_THRESHOLD, vec_text],
                        ).fetchone()
                        if result:
                            return result[0], "semantic"
                    except Exception as exc:
                        if not self._is_vector_search_unavailable(exc):
                            logger.debug("Semantic dedup check failed: %s", exc)
            return None, None

    # -- Spec-06 (#69): access audit log -------------------------------------

    def write_access_audit(
        self,
        *,
        user_id: str,
        query_text: str,
        granted_count: int,
        denied_count: int,
        denied_scopes: str | None = None,
        excluded: bool = False,
        tenant: str | None = None,
    ) -> None:
        """Append a row to the access_audit table.

        Called on every query (granted_count > 0) and every deny
        (denied_count > 0, excluded=True). The audit log is
        practice-internal — principals-only read.

        SC3: query_text is hashed (SHA-256, 16 chars) for privacy,
        consistent with the facade's _hash_query. Raw queries are not
        persisted in the durable store.
        """
        import hashlib
        import uuid
        # SC3: hash query_text instead of storing raw text.
        _qt = query_text or ""
        if _qt:
            _qt = hashlib.sha256(_qt.encode("utf-8")).hexdigest()[:16]
        try:
            with self._lock:
                assert self.connection is not None
                self.connection.execute(
                    """INSERT INTO access_audit
                       (audit_id, ts, tenant, user_id, query_text,
                        granted_count, denied_count, denied_scopes, excluded)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    [
                        str(uuid.uuid4()),
                        self._now(),
                        tenant or "default",
                        user_id,
                        _qt,
                        int(granted_count),
                        int(denied_count),
                        denied_scopes,
                        bool(excluded),
                    ],
                )
        except Exception as exc:
            logger.warning("access_audit write failed: %s", exc)

    def export_access_audit(
        self,
        *,
        limit: int = 10000,
        format: str = "jsonl",
    ) -> str:
        """Export the access audit log as JSONL or CSV.

        Principals-only read; the caller is responsible for access
        control on the export itself.

        SR1: filtered to the caller's tenant/user_id — no cross-tenant
        audit leak.
        """
        try:
            with self._lock:
                assert self.connection is not None
                result = self.connection.execute(
                    """SELECT audit_id, ts, tenant, user_id, query_text,
                              granted_count, denied_count, denied_scopes, excluded
                       FROM access_audit
                       WHERE user_id = ? OR tenant = ?
                       ORDER BY ts DESC
                       LIMIT ?""",
                    [self.user_id, self.user_id,
                     max(1, min(int(limit), 100000))],
                )
                columns = [desc[0] for desc in result.description]
                rows = result.fetchall()
        except Exception as exc:
            logger.warning("access_audit export failed: %s", exc)
            return ""
        # SR12: hash query_text in the export to avoid leaking sensitive
        # user queries. The hash is sufficient for audit correlation
        # (matching repeated queries) without exposing the raw text.
        qt_idx = columns.index("query_text") if "query_text" in columns else -1
        if qt_idx >= 0:
            import hashlib as _hl
            hashed_rows = []
            for row in rows:
                row_list = list(row)
                if row_list[qt_idx]:
                    row_list[qt_idx] = _hl.sha256(
                        str(row_list[qt_idx]).encode("utf-8")
                    ).hexdigest()[:16]
                else:
                    row_list[qt_idx] = ""
                hashed_rows.append(tuple(row_list))
            rows = hashed_rows
        if format == "csv":
            import csv
            import io
            buf = io.StringIO()
            writer = csv.writer(buf)
            writer.writerow(columns)
            for row in rows:
                writer.writerow(row)
            return buf.getvalue()
        # Default: JSONL
        import io
        buf = io.StringIO()
        for row in rows:
            row_dict = dict(zip(columns, row))
            buf.write(json.dumps(row_dict) + "\n")
        return buf.getvalue()

    # -- Spec-07 (#71): file catalog + watcher -------------------------------

    def upsert_catalog_entry(
        self,
        *,
        file_id: str,
        canonical_path: str,
        size: int,
        mtime: str,
        doc_type: str,
        client_scope: str | None = None,
        doc_class: str | None = None,
        one_line_description: str | None = None,
        description_method: str = "heuristic",
        extract_hash: str | None = None,
        pinned: bool = False,
        layout_family: str | None = None,
    ) -> None:
        """Insert or update a file_catalog row. Called by the watcher's
        scan pass for new and changed files."""
        now = self._now()
        try:
            with self._lock:
                assert self.connection is not None
                self.connection.execute(
                    """INSERT INTO file_catalog
                       (file_id, canonical_path, size, mtime,
                        first_seen, last_seen, status,
                        client_scope, doc_class, doc_type,
                        one_line_description, description_method,
                        extract_hash, extracted_at, last_touch,
                        touch_count, pinned, hot_flags, hot_reason,
                        layout_family)
                       VALUES (?, ?, ?, ?, ?, ?, 'active',
                               ?, ?, ?, ?, ?, ?, NULL, NULL, 0, ?, NULL, NULL,
                               ?)
                       ON CONFLICT (file_id) DO UPDATE SET
                        canonical_path = excluded.canonical_path,
                        size = excluded.size,
                        mtime = excluded.mtime,
                        last_seen = excluded.last_seen,
                        client_scope = excluded.client_scope,
                        doc_class = excluded.doc_class,
                        doc_type = excluded.doc_type,
                        one_line_description = excluded.one_line_description,
                        description_method = excluded.description_method,
                        extract_hash = excluded.extract_hash,
                        pinned = excluded.pinned,
                        layout_family = excluded.layout_family""",
                    [
                        file_id, canonical_path, size, mtime,
                        now, now,
                        client_scope, doc_class, doc_type,
                        one_line_description, description_method,
                        extract_hash, pinned,
                        layout_family,
                    ],
                )
        except Exception as exc:
            logger.warning("catalog upsert failed for %s: %s", file_id, exc)

    def add_file_alias(self, *, file_id: str, path: str) -> None:
        """Record an alias path for a known file_id (move/rename detection)."""
        now = self._now()
        try:
            with self._lock:
                assert self.connection is not None
                self.connection.execute(
                    """INSERT OR IGNORE INTO file_aliases
                       (file_id, path, first_seen) VALUES (?, ?, ?)""",
                    [file_id, path, now],
                )
        except Exception as exc:
            logger.warning("alias insert failed: %s", exc)

    def tombstone_catalog_entry(self, *, file_id: str) -> None:
        """Mark a catalog entry as tombstoned (file deleted). Facts sourced
        from this file become invalidated (D4 lifecycle).

        SR6: the memory_records invalidation is guarded by user_scope —
        only the caller's own facts are invalidated. The catalog entry
        itself is shared (no user_scope on file_catalog).
        """
        now = self._now()
        try:
            with self._lock:
                assert self.connection is not None
                self.connection.execute(
                    "UPDATE file_catalog SET status = 'tombstoned', last_seen = ? "
                    "WHERE file_id = ?",
                    [now, file_id],
                )
                # Invalidate facts sourced from this document.
                # SR6: guard by user_scope to prevent cross-tenant modification.
                self.connection.execute(
                    "UPDATE memory_records SET verified_state = 'invalidated' "
                    "WHERE source_doc_id = ? AND verified_state != 'invalidated' "
                    "AND (user_scope IS NULL OR user_scope = ?)",
                    [file_id, self.user_id],
                )
        except Exception as exc:
            logger.warning("tombstone failed for %s: %s", file_id, exc)

    def get_catalog_entry(
        self, file_id: str, *, client_scope: str | None = None,
    ) -> dict | None:
        """Fetch a catalog entry by file_id.

        SR7: *client_scope* is an optional explicit filter (matching the
        ``list_catalog`` convention). When None (default), no scope
        filter is applied — the caller is responsible for access control.
        When set, only entries with matching or NULL client_scope are
        returned.
        """
        scope_clause = ""
        params: list = [file_id]
        if client_scope is not None:
            scope_clause = " AND (client_scope IS NULL OR client_scope = ?)"
            params.append(client_scope)
        try:
            with self._lock:
                assert self.connection is not None
                result = self.connection.execute(
                    f"SELECT * FROM file_catalog WHERE file_id = ?{scope_clause}",
                    params,
                )
                columns = [desc[0] for desc in result.description]
                row = result.fetchone()
                if row:
                    return dict(zip(columns, row))
        except Exception as exc:
            logger.warning("catalog get failed: %s", exc)
        return None

    def get_catalog_by_path(
        self, path: str, *, client_scope: str | None = None,
    ) -> dict | None:
        """Fetch a catalog entry by canonical_path.

        SR7: *client_scope* is an optional explicit filter (matching the
        ``list_catalog`` convention). When None (default), no scope
        filter is applied. When set, only entries with matching or NULL
        client_scope are returned.
        """
        scope_clause = ""
        params: list = [path]
        if client_scope is not None:
            scope_clause = " AND (client_scope IS NULL OR client_scope = ?)"
            params.append(client_scope)
        try:
            with self._lock:
                assert self.connection is not None
                result = self.connection.execute(
                    f"SELECT * FROM file_catalog WHERE canonical_path = ?{scope_clause}",
                    params,
                )
                columns = [desc[0] for desc in result.description]
                row = result.fetchone()
                if row:
                    return dict(zip(columns, row))
        except Exception as exc:
            logger.warning("catalog get by path failed: %s", exc)
        return None

    def list_catalog(
        self,
        *,
        status: str = "active",
        client_scope: str | None = None,
        limit: int = 100,
    ) -> List[dict]:
        """List catalog entries with optional filters."""
        conditions = ["status = ?"]
        params: list = [status]
        if client_scope:
            conditions.append("(client_scope IS NULL OR client_scope = ?)")
            params.append(client_scope)
        where = " AND ".join(conditions)
        params.append(max(1, min(int(limit), 10000)))
        try:
            with self._lock:
                assert self.connection is not None
                result = self.connection.execute(
                    f"SELECT * FROM file_catalog WHERE {where} "
                    "ORDER BY last_seen DESC LIMIT ?",
                    params,
                )
                columns = [desc[0] for desc in result.description]
                return [dict(zip(columns, row)) for row in result.fetchall()]
        except Exception as exc:
            logger.warning("catalog list failed: %s", exc)
            return []

    def list_catalog_by_layout_family(
        self,
        layout_family: str,
        *,
        status: str = "active",
        limit: int = 100,
    ) -> List[dict]:
        """List catalog entries sharing a layout-family fingerprint.

        Spec-09 (#112): form-level retrieval. Used by the known-family
        short-circuit check and the labelling surface.
        """
        params: list = [layout_family, status]
        params.append(max(1, min(int(limit), 10000)))
        try:
            with self._lock:
                assert self.connection is not None
                result = self.connection.execute(
                    "SELECT * FROM file_catalog "
                    "WHERE layout_family = ? AND status = ? "
                    "ORDER BY last_seen DESC LIMIT ?",
                    params,
                )
                columns = [desc[0] for desc in result.description]
                return [dict(zip(columns, row)) for row in result.fetchall()]
        except Exception as exc:
            logger.warning("catalog list by layout_family failed: %s", exc)
            return []

    # -- rejection quality monitor (#121, read-only) -------------------------

    def query_candidate_decisions(
        self,
        *,
        since: str | None = None,
        until: str | None = None,
    ) -> List[dict]:
        """Read-only: return reviewed candidate rows for the quality monitor.

        Returns one dict per reviewed candidate with: category, status,
        reviewed_at, review_reason, provenance_origin, source, created_at.
        Unreviewed (pending) rows are excluded — they have no decision yet.

        #121: this is the raw data source for ``decision_rate_report`` and
        ``drift_check``. Read-only — never writes to or mutates the
        candidates table or the ledger.
        """
        conditions = ["status NOT IN ('pending')",
                      "(user_scope IS NULL OR user_scope = ?)"]
        params: list = [self.user_id]
        if since:
            conditions.append("(reviewed_at IS NULL OR reviewed_at >= ?)")
            params.append(since)
        if until:
            conditions.append("(reviewed_at IS NULL OR reviewed_at <= ?)")
            params.append(until)
        where = " AND ".join(conditions)
        try:
            with self._lock:
                assert self.connection is not None
                result = self.connection.execute(
                    f"""SELECT category, status, reviewed_at, review_reason,
                              provenance_origin, source, created_at
                       FROM memory_candidates
                       WHERE {where}
                       ORDER BY reviewed_at ASC""",
                    params,
                )
                columns = [desc[0] for desc in result.description]
                return [dict(zip(columns, row)) for row in result.fetchall()]
        except Exception as exc:
            logger.warning("candidate decision query failed: %s", exc)
            return []

    def query_rejection_ledger(
        self,
        *,
        since: str | None = None,
        until: str | None = None,
    ) -> List[dict]:
        """Read-only: return rejection_ledger rows for the quality monitor.

        Returns one dict per ledger entry with: subject, predicate,
        user_scope, reason, created_at.

        #121: complements ``query_candidate_decisions`` — the ledger captures
        the claim-slot identity of rejections (subject/predicate/scope) while
        the candidates table captures the review decision. Read-only.
        """
        conditions: list[str] = ["(user_scope IS NULL OR user_scope = ?)"]
        params: list = [self.user_id]
        if since:
            conditions.append("created_at >= ?")
            params.append(since)
        if until:
            conditions.append("created_at <= ?")
            params.append(until)
        where = " WHERE " + " AND ".join(conditions)
        try:
            with self._lock:
                assert self.connection is not None
                result = self.connection.execute(
                    f"""SELECT subject, predicate, user_scope, reason, created_at
                       FROM rejection_ledger{where}
                       ORDER BY created_at ASC""",
                    params,
                )
                columns = [desc[0] for desc in result.description]
                return [dict(zip(columns, row)) for row in result.fetchall()]
        except Exception as exc:
            logger.warning("rejection ledger query failed: %s", exc)
            return []

    def query_hard_cases(
        self,
        *,
        statuses: tuple[str, ...] = ("rejected", "quarantined"),
        limit: int = 500,
    ) -> List[dict]:
        """Read-only: return rejected/quarantined candidates as labeled
        hard-case eval items (#121).

        Each row carries: candidate_id, category, content, status,
        review_reason, quarantine_reason, provenance_origin, source,
        created_at, reviewed_at. The ``review_reason`` / ``quarantine_reason``
        is the label (gold = the recorded rejection/quarantine reason).

        Read-only — never writes to or mutates the candidates table.
        """
        if not statuses:
            return []
        placeholders = ", ".join("?" for _ in statuses)
        cap = max(1, min(int(limit), 10000))
        try:
            with self._lock:
                assert self.connection is not None
                result = self.connection.execute(
                    f"""SELECT candidate_id, category, content, status,
                              review_reason, quarantine_reason,
                              provenance_origin, source, created_at, reviewed_at
                       FROM memory_candidates
                       WHERE status IN ({placeholders})
                         AND (user_scope IS NULL OR user_scope = ?)
                       ORDER BY reviewed_at DESC
                       LIMIT ?""",
                    [*statuses, self.user_id, cap],
                )
                columns = [desc[0] for desc in result.description]
                return [dict(zip(columns, row)) for row in result.fetchall()]
        except Exception as exc:
            logger.warning("hard-case query failed: %s", exc)
            return []


    def list_layout_families(
        self,
        *,
        status: str = "active",
        limit: int = 10000,
    ) -> List[Dict[str, Any]]:
        """Aggregate layout-family counts in the catalog.

        Returns one row per distinct layout_family with its document count.
        Rows with NULL layout_family are excluded (not yet fingerprinted).
        Used by the labelling surface and the eval stratification script.
        """
        try:
            with self._lock:
                assert self.connection is not None
                result = self.connection.execute(
                    "SELECT layout_family, COUNT(*) AS doc_count "
                    "FROM file_catalog "
                    "WHERE layout_family IS NOT NULL AND status = ? "
                    "GROUP BY layout_family "
                    "ORDER BY doc_count DESC LIMIT ?",
                    [status, max(1, min(int(limit), 100000))],
                )
                columns = [desc[0] for desc in result.description]
                return [dict(zip(columns, row)) for row in result.fetchall()]
        except Exception as exc:
            logger.warning("list_layout_families failed: %s", exc)
            return []

    def stale_facts_for_doc(self, file_id: str) -> int:
        """Mark facts from a document as stale (version bump → old facts
        stale, new facts current). Returns the number of stale-marked rows.

        SR4: guarded by user_scope — only the caller's own facts are
        marked stale.
        """
        try:
            with self._lock:
                assert self.connection is not None
                result = self.connection.execute(
                    "UPDATE memory_records SET verified_state = 'stale' "
                    "WHERE source_doc_id = ? AND verified_state = 'current' "
                    "AND (user_scope IS NULL OR user_scope = ?)",
                    [file_id, self.user_id],
                )
                return int(result.fetchone()[0]) if result else 0
        except Exception as exc:
            logger.warning("stale marking failed for %s: %s", file_id, exc)
            return 0

    def verify_fact(self, memory_id: str) -> bool:
        """Principal verify action: flip unverified → current.

        SR5: guarded by user_scope — only the caller's own facts can be
        verified.
        """
        try:
            with self._lock:
                assert self.connection is not None
                self.connection.execute(
                    "UPDATE memory_records SET verified_state = 'current', "
                    "verified_at = ? WHERE memory_id = ? "
                    "AND verified_state = 'unverified' "
                    "AND (user_scope IS NULL OR user_scope = ?)",
                    [self._now(), memory_id, self.user_id],
                )
                return True
        except Exception as exc:
            logger.warning("verify failed for %s: %s", memory_id, exc)
            return False

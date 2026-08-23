"""Retrieval seam — Wave 2 (2026-08-12).

The seam is the deliverable, not the engines.  ``DuckDBRetriever`` wraps the
existing scan-based hybrid search (vector + ILIKE + RRF fusion) with zero
behavior change.  Alternative engines (ANN, BM25, graph-candidate retrieval)
can be dropped in behind the same protocol later without touching the
provider or the store's public API.

Usage:
    store.set_retriever(MyAnnRetriever(store))   # advanced, pluggable
"""
from __future__ import annotations

from typing import Any, List, Optional, Protocol


class Retriever(Protocol):
    """Contract for retrieval engines behind ``store.search``.

    Implementations must honor scope/status/validity/expiry semantics and
    the ``suppress_retrieval`` accounting flag, or explicitly document
    where they do not.
    """

    def search(
        self,
        query: str,
        limit: int,
        category_filter: Optional[str] = None,
        project_id: Optional[str] = None,
        suppress_retrieval: bool = False,
        **kwargs: Any,
    ) -> List[Any]:
        """Return up to *limit* MemoryRecords ranked for *query*."""
        ...


class DuckDBRetriever:
    """Default engine: the existing DuckDB scan-based hybrid search.

    Pure delegation — intentionally no logic here.  The actual algorithm
    lives in ``DuckDBMemoryStore._hybrid_search`` so the seam adds zero
    behavior change today.
    """

    def __init__(self, store: Any) -> None:
        self._store = store

    def search(
        self,
        query: str,
        limit: int,
        category_filter: Optional[str] = None,
        project_id: Optional[str] = None,
        suppress_retrieval: bool = False,
        **kwargs: Any,
    ) -> List[Any]:
        return self._store._hybrid_search(
            query,
            limit=limit,
            category_filter=category_filter,
            project_id=project_id,
            suppress_retrieval=suppress_retrieval,
            **kwargs,
        )

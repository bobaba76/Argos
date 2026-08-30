"""DuckDB storage layer for hybrid memory.

Stores memory records with category, content, tags, JSON payload, timestamps,
and a DOUBLE[] embedding column for vector search via ``list_cosine_similarity``.
Falls back to ILIKE text search when embeddings are unavailable.

Categories (general-purpose — any topic):
  personal_fact  — stable things about the user (age, location, job, tools, diagnoses)
  preference     — how the user likes things (tools, communication style, habits)
  insight        — self-observations, realizations, patterns noticed
  event          — notable events with date context (job changes, milestones)
  relationship   — people in the user's life and dynamics
  goal           — things the user is working toward
  context_note   — situational context that helps future conversations
"""
from __future__ import annotations

import hashlib
import json
import re
import logging
import math
import threading

try:
    from .retriever import DuckDBRetriever
except ImportError:  # store.py imported as a top-level module (tests)
    from retriever import DuckDBRetriever

try:
    from .value_extractor import extract_values, values_conflict, is_transition_statement
except ImportError:  # store.py imported as a top-level module (tests)
    from value_extractor import extract_values, values_conflict, is_transition_statement
try:
    from .structural_loss import structural_loss_guard, is_append_only, LossReport
except ImportError:  # store.py imported as a top-level module (tests)
    from structural_loss import structural_loss_guard, is_append_only, LossReport
import time
import uuid
from collections import Counter, deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import duckdb

logger = logging.getLogger(__name__)

# --- store_common split (god-file refactor, behavior-neutral) ------------
# Every module-level name that moved to store_common.py is re-exported here
# so ``from store import <name>`` keeps working for the full original surface
# (tests, eval scripts and the benchmark clone import these from ``store``).
try:
    from .store_common import (  # noqa: F401
        _CONTROL_CHARS_RE,
    _DEFAULT_TTL_DAYS,
    _FORMAT_SPACES_RE,
    _GROUNDING_CEILING,
    _GROUNDING_ORDER,
    _HIDDEN_CHARS_RE,
    _INJECTION_PATTERNS,
    _NOT_PROVIDED,
    _TOKEN_RE,
    _TEXT_STOPWORDS,
    _TRUST_CLASS_ORDER,
    _VALID_GROUNDING,
    _VALID_PROVENANCE,
    _ci,
    _tokenize,
    GROUNDING_EXTRACTED,
    GROUNDING_INFERRED,
    GROUNDING_OBSERVED,
    GROUNDING_SPECULATIVE,
    MemoryRecord,
    PROVENANCE_EXTERNAL,
    PROVENANCE_INTERNAL,
    VALID_CATEGORIES,
    default_grounding_for_write,
    grounding_allows_status,
    grounding_rank,
    normalize_grounding,
    normalize_provenance,
        np,
    rejection_key,
    sanitize_content,
    trust_class_rank,
    )
except ImportError:  # store.py imported as a top-level module (tests)
    from store_common import (  # noqa: F401
        _CONTROL_CHARS_RE,
    _DEFAULT_TTL_DAYS,
    _FORMAT_SPACES_RE,
    _GROUNDING_CEILING,
    _GROUNDING_ORDER,
    _HIDDEN_CHARS_RE,
    _INJECTION_PATTERNS,
    _NOT_PROVIDED,
    _TOKEN_RE,
    _TEXT_STOPWORDS,
    _TRUST_CLASS_ORDER,
    _VALID_GROUNDING,
    _VALID_PROVENANCE,
    _ci,
    _tokenize,
    GROUNDING_EXTRACTED,
    GROUNDING_INFERRED,
    GROUNDING_OBSERVED,
    GROUNDING_SPECULATIVE,
    MemoryRecord,
    PROVENANCE_EXTERNAL,
    PROVENANCE_INTERNAL,
    VALID_CATEGORIES,
    default_grounding_for_write,
    grounding_allows_status,
    grounding_rank,
    normalize_grounding,
    normalize_provenance,
        np,
    rejection_key,
    sanitize_content,
    trust_class_rank,
    )

try:
    from .store_core import StoreCoreMixin
except ImportError:  # store.py imported as a top-level module (tests)
    from store_core import StoreCoreMixin
try:
    from .store_retrieval import StoreRetrievalMixin
except ImportError:  # store.py imported as a top-level module (tests)
    from store_retrieval import StoreRetrievalMixin
try:
    from .store_maintenance import StoreMaintenanceMixin
except ImportError:  # store.py imported as a top-level module (tests)
    from store_maintenance import StoreMaintenanceMixin

try:
    from .store_write import StoreWriteMixin
except ImportError:  # store.py imported as a top-level module (tests)
    from store_write import StoreWriteMixin




class DuckDBMemoryStore(StoreCoreMixin, StoreWriteMixin, StoreRetrievalMixin, StoreMaintenanceMixin):
    """DuckDB-backed memory store with vector + text search."""







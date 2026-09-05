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

Import mode (#304): this module MUST be imported as part of the
``argos_plugin`` (or ``argos`` alias) package — e.g.
``from argos_plugin.store import DuckDBMemoryStore`` or
``from store import DuckDBMemoryStore`` (the test conftest registers
``store`` as a top-level alias for ``argos.store``). Package-relative
imports (``from .store_common import ...``) are the only import mode;
the dual-branch try/except ImportError fallback was removed because it
duplicated the re-export list in two branches, creating a maintenance
hazard where any new name had to be added in both places.
"""
from __future__ import annotations

import hashlib
import json
import re
import logging
import math
import threading

from .retriever import DuckDBRetriever
from .value_extractor import extract_values, values_conflict, is_transition_statement
from .structural_loss import structural_loss_guard, is_append_only, LossReport

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
# #304: single import list — package-relative only. The test conftest
# registers top-level aliases (store, store_common, etc.) pointing at
# argos.store, argos.store_common, etc. so both import modes resolve
# to the same module object.
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

from .store_core import StoreCoreMixin
from .store_retrieval import StoreRetrievalMixin
from .store_maintenance import StoreMaintenanceMixin
from .store_write import StoreWriteMixin




class DuckDBMemoryStore(StoreCoreMixin, StoreWriteMixin, StoreRetrievalMixin, StoreMaintenanceMixin):
    """DuckDB-backed memory store with vector + text search."""

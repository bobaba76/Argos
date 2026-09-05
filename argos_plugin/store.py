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

Import mode (#304): supports BOTH package mode (``from argos_plugin.store
import DuckDBMemoryStore``) and script/test mode (``from store import
DuckDBMemoryStore``). The branch is on ``__package__`` — the same pattern
used by memory_service.py:25-41 and service_client.py:15-23. There is ONE
re-export name list (no duplication): the module is selected by the
``__package__`` branch, then names are pulled from it via a single
``globals()`` update.
"""
from __future__ import annotations

import hashlib
import json
import re
import logging
import math
import threading

import importlib
import sys
from pathlib import Path

# #304: branch on __package__ — same pattern as memory_service.py:25-41.
# Package mode: launched as part of argos_plugin (e.g. via __init__.py).
# Script/test mode: launched as a top-level script (memory_service.py
# subprocess with cwd=plugin_dir, __package__ None) or imported by tests
# via ``from store import ...``.
if __package__:
    _sc = importlib.import_module(".store_common", __package__)
    _retriever_mod = importlib.import_module(".retriever", __package__)
    _value_ext_mod = importlib.import_module(".value_extractor", __package__)
    _struct_loss_mod = importlib.import_module(".structural_loss", __package__)
    _core_mod = importlib.import_module(".store_core", __package__)
    _retrieval_mod = importlib.import_module(".store_retrieval", __package__)
    _maintenance_mod = importlib.import_module(".store_maintenance", __package__)
    _write_mod = importlib.import_module(".store_write", __package__)
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    _sc = importlib.import_module("store_common")
    _retriever_mod = importlib.import_module("retriever")
    _value_ext_mod = importlib.import_module("value_extractor")
    _struct_loss_mod = importlib.import_module("structural_loss")
    _core_mod = importlib.import_module("store_core")
    _retrieval_mod = importlib.import_module("store_retrieval")
    _maintenance_mod = importlib.import_module("store_maintenance")
    _write_mod = importlib.import_module("store_write")

DuckDBRetriever = _retriever_mod.DuckDBRetriever
extract_values = _value_ext_mod.extract_values
values_conflict = _value_ext_mod.values_conflict
is_transition_statement = _value_ext_mod.is_transition_statement
structural_loss_guard = _struct_loss_mod.structural_loss_guard
is_append_only = _struct_loss_mod.is_append_only
LossReport = _struct_loss_mod.LossReport

import time
import uuid
from collections import Counter, deque
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import duckdb

logger = logging.getLogger(__name__)

# --- store_common split (god-file refactor, behavior-neutral) ------------
# Every module-level name that moved to store_common.py is re-exported here
# so ``from store import <name>`` keeps working for the full original surface
# (tests, eval scripts and the benchmark clone import these from ``store``).
# #304: ONE name list — no duplication. The module (_sc) was selected by the
# __package__ branch above; names are pulled from it here.
_STORE_COMMON_NAMES = (
    "_CONTROL_CHARS_RE",
    "_DEFAULT_TTL_DAYS",
    "_FORMAT_SPACES_RE",
    "_GROUNDING_CEILING",
    "_GROUNDING_ORDER",
    "_HIDDEN_CHARS_RE",
    "_INJECTION_PATTERNS",
    "_NOT_PROVIDED",
    "_TOKEN_RE",
    "_TEXT_STOPWORDS",
    "_TRUST_CLASS_ORDER",
    "_VALID_GROUNDING",
    "_VALID_PROVENANCE",
    "_ci",
    "_tokenize",
    "GROUNDING_EXTRACTED",
    "GROUNDING_INFERRED",
    "GROUNDING_OBSERVED",
    "GROUNDING_SPECULATIVE",
    "MemoryRecord",
    "PROVENANCE_EXTERNAL",
    "PROVENANCE_INTERNAL",
    "VALID_CATEGORIES",
    "default_grounding_for_write",
    "grounding_allows_status",
    "grounding_rank",
    "normalize_grounding",
    "normalize_provenance",
    "np",
    "rejection_key",
    "sanitize_content",
    "trust_class_rank",
)
for _name in _STORE_COMMON_NAMES:
    globals()[_name] = getattr(_sc, _name)

StoreCoreMixin = _core_mod.StoreCoreMixin
StoreRetrievalMixin = _retrieval_mod.StoreRetrievalMixin
StoreMaintenanceMixin = _maintenance_mod.StoreMaintenanceMixin
StoreWriteMixin = _write_mod.StoreWriteMixin




class DuckDBMemoryStore(StoreCoreMixin, StoreWriteMixin, StoreRetrievalMixin, StoreMaintenanceMixin):
    """DuckDB-backed memory store with vector + text search."""

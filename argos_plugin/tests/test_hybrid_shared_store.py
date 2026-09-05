"""Pytest tests for the argos plugin.

Run with:
    python -m pytest tests/test_argos.py -v

"""
from __future__ import annotations

import sys
import tempfile
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pytest

# Ensure the plugin package is importable.
_plugin_dir = Path(__file__).resolve().parent.parent
if str(_plugin_dir.parent) not in sys.path:
    sys.path.insert(0, str(_plugin_dir.parent))


@pytest.fixture
def tmp_path_factory_dir(tmp_path):
    """Provide a clean temp directory for each test."""
    return tmp_path


class TestSharedStoreSurface:
    """Verify SharedMemoryStore and SharedGraphStore have method parity with
    their direct counterparts.  The shared service routes RPC calls to the
    underlying DuckDBMemoryStore/KuzuGraphStore, so every method the provider
    calls on the store/graph must exist on the shared client too.
    """

    def test_shared_memory_store_has_update_memory(self):
        """SharedMemoryStore must expose update_memory (regression: was missing)."""
        from service_client import SharedMemoryStore
        assert hasattr(SharedMemoryStore, "update_memory"), \
            "SharedMemoryStore must have update_memory method"

    def test_shared_graph_store_has_purge_junk_entities(self):
        """SharedGraphStore must expose purge_junk_entities (regression: was missing)."""
        from service_client import SharedGraphStore
        assert hasattr(SharedGraphStore, "purge_junk_entities"), \
            "SharedGraphStore must have purge_junk_entities method"

    def test_store_method_parity(self):
        """Every public method on DuckDBMemoryStore that the provider calls
        must also exist on SharedMemoryStore."""
        from store import DuckDBMemoryStore
        from service_client import SharedMemoryStore
        # Methods the provider calls on self._store (from __init__.py).
        required = {
            "search", "get_memories_by_ids", "remember", "update_memory",
            "consolidate", "save_candidate", "list_candidates", "review_candidate", "quarantine_memory",
            "restore_memory", "record_feedback", "delete_memory",
            "cleanup_junk", "count", "get_insights", "close", "set_user_scope",
        }
        for method in required:
            assert hasattr(SharedMemoryStore, method), \
                f"SharedMemoryStore missing method: {method}"

    def test_record_from_dict_preserves_temporal_fields(self):
        """_record_from_dict must include valid_from/valid_to/superseded_by
        and raw_similarity — without them, the shared-service client path
        (used by desktop/gateway) drops temporal validity data even though
        the DB and to_dict() serialization include it.

        Regression: these fields were missing from _record_from_dict,
        making temporal validity invisible through the shared-service path.
        """
        from service_client import _record_from_dict

        record_dict = {
            "memory_id": "mem-test",
            "category": "personal_fact",
            "content": "User pays R15000 rent",
            "tags": [],
            "payload": {},
            "created_at": "2026-08-01T00:00:00",
            "updated_at": "2026-08-01T00:00:00",
            "similarity": 0.95,
            "raw_similarity": 0.88,
            "valid_from": "2026-08-01T00:00:00",
            "valid_to": "2026-08-05T00:00:00",
            "superseded_by": "mem-newer",
        }

        record = _record_from_dict(record_dict)
        assert record is not None
        assert record.valid_from == "2026-08-01T00:00:00"
        assert record.valid_to == "2026-08-05T00:00:00"
        assert record.superseded_by == "mem-newer"
        assert record.raw_similarity == 0.88

    def test_record_from_dict_defaults_temporal_fields_to_none(self):
        """When the dict doesn't have temporal fields (e.g. older service),
        _record_from_dict should default them to None, not crash."""
        from service_client import _record_from_dict

        record_dict = {
            "memory_id": "mem-old",
            "category": "personal_fact",
            "content": "User has a cat",
            "tags": [],
            "payload": {},
        }

        record = _record_from_dict(record_dict)
        assert record is not None
        assert record.valid_from is None
        assert record.valid_to is None
        assert record.superseded_by is None
        assert record.raw_similarity == 0.0

    def test_shared_store_search_accepts_as_of(self):
        """SharedMemoryStore.search must accept and pass through the as_of
        parameter for historical queries."""
        import inspect
        from service_client import SharedMemoryStore

        sig = inspect.signature(SharedMemoryStore.search)
        assert "as_of" in sig.parameters, \
            "SharedMemoryStore.search must accept as_of parameter"

    def test_shared_store_has_all_alias_methods(self):
        """SharedMemoryStore must expose all alias methods for parity with
        DuckDBMemoryStore. Regression: list_aliases was missing, causing
        AttributeError on the shared-service path."""
        from service_client import SharedMemoryStore
        for method in ("add_alias", "remove_alias", "resolve_aliases",
                       "list_aliases", "aliases_for_canonical"):
            assert hasattr(SharedMemoryStore, method), \
                f"SharedMemoryStore missing alias method: {method}"

    def test_graph_method_parity(self):
        """Every public method on KuzuGraphStore that the provider calls
        must also exist on SharedGraphStore."""
        from service_client import SharedGraphStore
        # Methods the provider calls on self._graph (from __init__.py).
        required = {
            "search_graph", "memory_ids_for_query", "query_graph", "traverse_graph",
            "add_relationship", "index_memory", "remove_memory", "purge_junk_entities",
            "close", "set_user_scope",
        }
        for method in required:
            assert hasattr(SharedGraphStore, method), \
                f"SharedGraphStore missing method: {method}"

    def test_service_dispatches_update_memory(self):
        """The memory service must route update_memory to the store."""
        import inspect
        # We check the source rather than starting the service — the dispatch
        # is a simple if-chain in _call_store.
        from memory_service import MemoryService
        source = inspect.getsource(MemoryService._call_store)
        assert "update_memory" in source, \
            "MemoryService._call_store must dispatch update_memory"



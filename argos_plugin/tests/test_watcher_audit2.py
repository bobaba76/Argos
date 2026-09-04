"""Audit tests for watcher WT1-WT3 (issue #264).

Covers LLM-confidence normalization, stop() responsiveness, and
egress fail-closed behavior.

Run with (Hermes venv python, offline):
    python -m pytest tests/test_watcher_audit2.py -v
"""
from __future__ import annotations

import inspect
import sys
import threading
from pathlib import Path

import pytest

_plugin_dir = Path(__file__).resolve().parent.parent
if str(_plugin_dir) not in sys.path:
    sys.path.insert(0, str(_plugin_dir))


# ---------------------------------------------------------------------------
# WT1 -- per-fact confidence normalization
# ---------------------------------------------------------------------------

class TestWT1ConfidenceNormalization:
    def test_confidence_clamp_in_source(self):
        """WT1: run_watcher_pass normalizes confidence per-fact."""
        from watcher_thread import run_watcher_pass
        src = inspect.getsource(run_watcher_pass)
        # Should have the clamp + fallback.
        assert "max(0.0, min(1.0" in src
        assert "0.5" in src  # fallback value

    def test_malformed_confidence_does_not_kill_doc(self):
        """WT1: a non-numeric confidence on one fact doesn't drop the
        entire document's facts. The per-fact try/except isolates the
        failure and uses 0.5 as fallback."""
        from watcher_thread import run_watcher_pass

        saved = []

        class FakeStore:
            def save_candidate(self, **kwargs):
                saved.append(kwargs)

        def fake_extract(path, doc_type, **kwargs):
            facts = [
                {"content": "fact with string confidence",
                 "category": "insight", "confidence": "high"},
                {"content": "fact with float confidence",
                 "category": "insight", "confidence": 0.9},
                {"content": "fact with missing confidence",
                 "category": "insight"},
            ]
            return facts, "hash123", "llm", "some text"

        # Patch the real watcher module (already imported by the test runner).
        import watcher as w_mod
        orig_extract = w_mod.extract_facts_from_doc
        orig_scan = w_mod.scan_pass
        w_mod.extract_facts_from_doc = fake_extract
        w_mod.scan_pass = lambda roots, catalog: {"new": [{"path": "/x", "doc_type": "txt", "file_id": "1", "size": 100, "mtime": 0}], "changed": [], "moved": [], "deleted": [], "unchanged": []}
        try:
            counts = run_watcher_pass(FakeStore(), ["/tmp"])
        finally:
            w_mod.extract_facts_from_doc = orig_extract
            w_mod.scan_pass = orig_scan

        # All 3 facts should be saved (malformed confidence -> 0.5 fallback).
        assert len(saved) == 3
        # The string-confidence fact should get 0.5 fallback.
        confs = {s["content"]: s["confidence"] for s in saved}
        assert confs["fact with string confidence"] == 0.5
        assert confs["fact with float confidence"] == 0.9
        # Missing confidence defaults to 0.7 (clamped).
        assert confs["fact with missing confidence"] == 0.7


# ---------------------------------------------------------------------------
# WT2 -- stop() responsiveness
# ---------------------------------------------------------------------------

class TestWT2StopResponsiveness:
    def test_stop_event_param_exists(self):
        """WT2: run_watcher_pass accepts a stop_event parameter."""
        from watcher_thread import run_watcher_pass
        sig = inspect.signature(run_watcher_pass)
        assert "stop_event" in sig.parameters

    def test_pass_interrupts_on_stop_event(self):
        """WT2: when stop_event is set, the pass aborts between documents."""
        from watcher_thread import run_watcher_pass

        saved = []

        class FakeStore:
            def save_candidate(self, **kwargs):
                saved.append(kwargs)

        # Create 5 hot docs; set stop_event after the 2nd doc.
        stop_event = threading.Event()
        docs = [{"path": f"/doc{i}", "doc_type": "txt", "file_id": str(i),
                 "size": 100, "mtime": 0} for i in range(5)]

        call_count = [0]

        def fake_extract(path, doc_type, **kwargs):
            call_count[0] += 1
            if call_count[0] >= 2:
                stop_event.set()
            return [{"content": f"fact from {path}", "category": "insight"}], "h", "llm", "text"

        import watcher as w_mod
        orig_extract = w_mod.extract_facts_from_doc
        orig_scan = w_mod.scan_pass
        w_mod.extract_facts_from_doc = fake_extract
        w_mod.scan_pass = lambda roots, catalog: {"new": docs, "changed": [], "moved": [], "deleted": [], "unchanged": []}
        try:
            counts = run_watcher_pass(FakeStore(), ["/tmp"], stop_event=stop_event)
        finally:
            w_mod.extract_facts_from_doc = orig_extract
            w_mod.scan_pass = orig_scan

        # Should not have processed all 5 docs (stopped after ~2-3).
        assert call_count[0] < 5

    def test_max_docs_per_pass_cap(self):
        """WT2: no more than _MAX_DOCS_PER_PASS docs are processed per pass."""
        from watcher_thread import run_watcher_pass

        class FakeStore:
            def save_candidate(self, **kwargs):
                pass

        docs = [{"path": f"/doc{i}", "doc_type": "txt", "file_id": str(i),
                 "size": 100, "mtime": 0} for i in range(50)]

        call_count = [0]

        def fake_extract(path, doc_type, **kwargs):
            call_count[0] += 1
            return [{"content": "fact", "category": "insight"}], "h", "llm", "text"

        import watcher as w_mod
        orig_extract = w_mod.extract_facts_from_doc
        orig_scan = w_mod.scan_pass
        w_mod.extract_facts_from_doc = fake_extract
        w_mod.scan_pass = lambda roots, catalog: {"new": docs, "changed": [], "moved": [], "deleted": [], "unchanged": []}
        try:
            counts = run_watcher_pass(FakeStore(), ["/tmp"])
        finally:
            w_mod.extract_facts_from_doc = orig_extract
            w_mod.scan_pass = orig_scan

        # Should be capped at 20 (not all 50).
        assert call_count[0] <= 20

    def test_stop_event_aborts_before_scan(self):
        """WT2: if stop_event is already set when the pass starts, the
        scan is skipped entirely (no catalog upserts, no extraction)."""
        from watcher_thread import run_watcher_pass

        upserts = []

        class FakeStore:
            def upsert_catalog_entry(self, **kwargs):
                upserts.append(kwargs)
            def tombstone_catalog_entry(self, **kwargs):
                pass
            def save_candidate(self, **kwargs):
                pass

        stop_event = threading.Event()
        stop_event.set()  # already stopped before pass starts

        import watcher as w_mod
        orig_scan = w_mod.scan_pass
        scan_called = [False]
        def fake_scan(roots, catalog):
            scan_called[0] = True
            return {"new": [], "changed": [], "moved": [], "deleted": [], "unchanged": []}
        w_mod.scan_pass = fake_scan
        try:
            counts = run_watcher_pass(FakeStore(), ["/tmp"], stop_event=stop_event)
        finally:
            w_mod.scan_pass = orig_scan

        # Scan should not have been called.
        assert scan_called[0] is False
        # No upserts should have happened.
        assert len(upserts) == 0

    def test_stop_event_aborts_during_catalog_upserts(self):
        """WT2: stop_event set during catalog upserts aborts the loop."""
        from watcher_thread import run_watcher_pass

        upserts = []

        class FakeStore:
            def upsert_catalog_entry(self, **kwargs):
                upserts.append(kwargs)
            def tombstone_catalog_entry(self, **kwargs):
                pass
            def save_candidate(self, **kwargs):
                pass

        stop_event = threading.Event()
        docs = [{"path": f"/d{i}", "doc_type": "txt", "file_id": str(i),
                 "size": 100, "mtime": 0} for i in range(10)]

        import watcher as w_mod
        orig_scan = w_mod.scan_pass
        orig_extract = w_mod.extract_facts_from_doc
        def fake_scan(roots, catalog):
            return {"new": docs, "changed": [], "moved": [], "deleted": [], "unchanged": []}
        def fake_extract(path, doc_type, **kwargs):
            return [], "h", "llm", "text"
        w_mod.scan_pass = fake_scan
        w_mod.extract_facts_from_doc = fake_extract

        # Set stop_event after 3 upserts.
        original_upsert = FakeStore.upsert_catalog_entry
        def counting_upsert(self, **kwargs):
            upserts.append(kwargs)
            if len(upserts) >= 3:
                stop_event.set()
        FakeStore.upsert_catalog_entry = counting_upsert

        try:
            counts = run_watcher_pass(FakeStore(), ["/tmp"], stop_event=stop_event)
        finally:
            w_mod.scan_pass = orig_scan
            w_mod.extract_facts_from_doc = orig_extract
            FakeStore.upsert_catalog_entry = original_upsert

        # Should not have processed all 10 upserts.
        assert len(upserts) < 10

    def test_max_catalog_updates_cap(self):
        """WT2: no more than _MAX_CATALOG_UPDATES_PER_PASS (500) catalog
        upserts+tombstones per pass."""
        from watcher_thread import run_watcher_pass

        upserts = []

        class FakeStore:
            def upsert_catalog_entry(self, **kwargs):
                upserts.append(kwargs)
            def tombstone_catalog_entry(self, **kwargs):
                pass
            def save_candidate(self, **kwargs):
                pass

        # 600 new docs + 600 deleted — should cap at 500 total.
        docs = [{"path": f"/d{i}", "doc_type": "txt", "file_id": str(i),
                 "size": 100, "mtime": 0} for i in range(600)]
        deleted = [{"file_id": f"del-{i}"} for i in range(600)]

        import watcher as w_mod
        orig_scan = w_mod.scan_pass
        orig_extract = w_mod.extract_facts_from_doc
        def fake_scan(roots, catalog):
            return {"new": docs, "changed": [], "moved": [], "deleted": deleted, "unchanged": []}
        def fake_extract(path, doc_type, **kwargs):
            return [], "h", "llm", "text"
        w_mod.scan_pass = fake_scan
        w_mod.extract_facts_from_doc = fake_extract
        try:
            counts = run_watcher_pass(FakeStore(), ["/tmp"])
        finally:
            w_mod.scan_pass = orig_scan
            w_mod.extract_facts_from_doc = orig_extract

        # Total upserts should be capped at 500.
        assert len(upserts) <= 500


# ---------------------------------------------------------------------------
# WT3 -- egress fail-closed
# ---------------------------------------------------------------------------

class TestWT3EgressFailClosed:
    def test_fail_closed_in_source(self):
        """WT3: extract_doc_facts_llm fails closed on egress gate error."""
        from watcher import extract_doc_facts_llm
        src = inspect.getsource(extract_doc_facts_llm)
        # Should have logger.error (not just pass).
        assert "logger.error" in src
        # Should return [] on egress failure (fail-closed).
        assert "return []" in src
        # Should NOT have the old "fail soft" comment.
        assert "fail soft" not in src

    def test_egress_failure_returns_empty(self, monkeypatch):
        """WT3: when egress gate raises, extraction returns [] (fail-closed)."""
        import watcher

        def broken_gate(site, text):
            raise RuntimeError("config load failed")

        # Monkeypatch the egress module.
        import types
        fake_egress = types.ModuleType("egress")
        fake_egress.gate = broken_gate
        monkeypatch.setitem(sys.modules, "egress", fake_egress)

        result = watcher.extract_doc_facts_llm("A" * 100)
        assert result == []

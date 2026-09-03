#!/usr/bin/env python3
"""Tests for read-side conflict surfacing (config-gated, default OFF).

Covers the two triggers and the fail-soft boundary:
  1. value conflict  -> same subject + same unit + different value
  2. discontinuation -> shared significant token + a stopped/removed/scoped marker
"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from provider_retrieval import (
    _conflict_shared_subject,
    _has_discontinuation_marker,
    _conflict_significant_tokens,
)
from provider_retrieval import ProviderRetrievalMixin
from store_common import MemoryRecord


def _rec(mid, content, created, sim=0.9):
    return MemoryRecord(
        memory_id=mid, category="context_note", content=content,
        created_at=created, similarity=sim,
    )


class ConflictSurfacingHelpers(unittest.TestCase):
    def test_significant_tokens(self):
        self.assertIn("contractor", _conflict_significant_tokens("the contractor rate"))
        self.assertNotIn("the", _conflict_significant_tokens("the contractor rate"))
        self.assertIn("10gb", _conflict_significant_tokens("beta users get 10GB"))

    def test_shared_subject(self):
        self.assertTrue(_conflict_shared_subject(
            "Beta users get 10GB of storage.", "The beta program ended."))
        self.assertTrue(_conflict_shared_subject(
            "We use the nightly build pipeline for releases.",
            "The nightly build pipeline was retired."))
        self.assertFalse(_conflict_shared_subject(
            "Support hours are 8 to 5.", "The weather is sunny today."))

    def test_discontinuation_marker(self):
        self.assertTrue(_has_discontinuation_marker("The beta program ended."))
        self.assertTrue(_has_discontinuation_marker("FTP uploads were discontinued."))
        self.assertTrue(_has_discontinuation_marker("The cache workaround was reverted."))
        self.assertFalse(_has_discontinuation_marker(
            "The contractor day rate is R1,350."))


class ConflictSurfacingAnnotation(unittest.TestCase):
    def _make(self, records, enabled=True):
        class DummyProvider(ProviderRetrievalMixin):
            def __init__(self, recs):
                self._records = recs
                self._conflict_surfacing_enabled = enabled

            def _record_injected(self, records):
                pass

            @property
            def _store(self):
                return None

        return DummyProvider(records)

    def test_value_conflict_emits_note(self):
        recs = [
            _rec("a", "The early payment discount is 2% for invoices paid within 10 days.",
                 "2026-06-10T15:00:00"),
            _rec("b", "The early payment discount is 1.5%.", "2026-08-20T12:00:00"),
        ]
        prov = self._make(recs)
        notes = prov._conflict_annotations(recs)
        self.assertEqual(len(notes), 1)
        self.assertIn("CONFLICT NOTE", notes[0].content)
        self.assertIn("differing values", notes[0].content)

    def test_discontinuation_emits_note(self):
        recs = [
            _rec("a", "Beta users get 10GB of storage.", "2026-06-05T12:00:00"),
            _rec("b", "The beta program ended.", "2026-08-15T09:20:00"),
        ]
        prov = self._make(recs)
        notes = prov._conflict_annotations(recs)
        self.assertEqual(len(notes), 1)
        self.assertIn("discontinued", notes[0].content)

    def test_workaround_reverted_emits_note(self):
        recs = [
            _rec("a", "During the outage we disabled the cache to keep the site up.",
                 "2026-07-03T13:00:00"),
            _rec("b", "The cache workaround was reverted after the incident.",
                 "2026-08-21T16:30:00"),
        ]
        prov = self._make(recs)
        notes = prov._conflict_annotations(recs)
        self.assertEqual(len(notes), 1)
        self.assertIn("reverted", notes[0].content)

    def test_benign_pair_no_note(self):
        recs = [
            _rec("a", "The company was founded in 2011.", "2026-08-01T10:00:00"),
            _rec("b", "The product ships weekly.", "2026-08-03T09:30:00"),
        ]
        prov = self._make(recs)
        self.assertEqual(prov._conflict_annotations(recs), [])

    def test_corroboration_no_note(self):
        recs = [
            _rec("a", "The approval threshold is R50,000 for purchases.", "2026-07-05T12:00:00"),
            _rec("b", "The approval threshold is R50,000.", "2026-08-24T09:15:00"),
        ]
        prov = self._make(recs)
        self.assertEqual(prov._conflict_annotations(recs), [])

    def test_disabled_no_note(self):
        recs = [
            _rec("a", "Our model scores 82.2 on the retrieval benchmark.", "2026-06-05T09:00:00"),
            _rec("b", "The model's retrieval benchmark score is 89.8.", "2026-08-18T11:30:00"),
        ]
        prov = self._make(recs, enabled=False)
        self.assertEqual(prov._conflict_annotations(recs), [])

    def test_note_capped(self):
        recs = [
            _rec(f"a{i}", f"Rule {i} says 100 units.", f"2026-06-0{i+1}T09:00:00")
            for i in range(1, 6)
        ] + [
            _rec(f"b{i}", f"Rule {i} was discontinued.", f"2026-08-0{i+1}T09:00:00")
            for i in range(1, 6)
        ]
        prov = self._make(recs)
        notes = prov._conflict_annotations(recs)
        self.assertLessEqual(len(notes), 2)

    def test_dedup_same_pair(self):
        recs = [
            _rec("a", "Beta users get 10GB of storage.", "2026-06-05T12:00:00"),
            _rec("b", "The beta program ended.", "2026-08-15T09:20:00"),
        ]
        prov = self._make(recs)
        self.assertEqual(len(prov._conflict_annotations(recs)), 1)


if __name__ == "__main__":
    unittest.main()
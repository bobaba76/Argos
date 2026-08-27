"""Tests for write-time value-supersession (stale-number detection), issue #4.

Covers:
- Value extractor: percentages, currencies, counts, ratios, years, ages
- Subject overlap matching (token-Jaccard)
- Conflict detection: same subject + different value
- Idempotent: same subject + same value → no conflict
- Different subject + same value → no conflict
- No numeric value → no conflict
- Store-level: _find_conflicting_active_value finds the stale record
- Store-level: _mark_superseded sets valid_to
- Store-level: save_candidate records the conflict in payload
- E2E: candidate with value conflict → review → supersede → old excluded from retrieval
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make the plugin importable
_plugin_dir = Path(__file__).resolve().parent.parent
for _path in (_plugin_dir.parent, _plugin_dir):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


# -- value_extractor unit tests --------------------------------------------

class TestExtractValues:
    def test_percentage(self):
        from argos.value_extractor import extract_values
        vals = extract_values("LongMemEval score is 89.8%")
        assert len(vals) >= 1
        pct = [v for v in vals if v.unit == "percent"]
        assert len(pct) == 1
        assert pct[0].value == "89.8"

    def test_percentage_word(self):
        from argos.value_extractor import extract_values
        vals = extract_values("accuracy of 82.2 percent")
        pct = [v for v in vals if v.unit == "percent"]
        assert len(pct) == 1
        assert pct[0].value == "82.2"

    def test_currency_real(self):
        from argos.value_extractor import extract_values
        vals = extract_values("budget is R$ 449")
        cur = [v for v in vals if v.unit == "currency:R$"]
        assert len(cur) == 1
        assert cur[0].value == "449"

    def test_currency_dollar(self):
        from argos.value_extractor import extract_values
        vals = extract_values("cost $1,200.50")
        cur = [v for v in vals if v.unit.startswith("currency:")]
        assert len(cur) == 1
        assert cur[0].value == "1200.50"

    def test_ratio(self):
        from argos.value_extractor import extract_values
        vals = extract_values("scored 449/500 on the test")
        ratios = [v for v in vals if v.unit == "ratio"]
        assert len(ratios) == 1
        assert ratios[0].value == "449/500"

    def test_count(self):
        from argos.value_extractor import extract_values
        vals = extract_values("processed 1200 rows")
        counts = [v for v in vals if v.unit == "count"]
        assert len(counts) == 1
        assert counts[0].value == "1200"

    def test_year(self):
        from argos.value_extractor import extract_values
        vals = extract_values("born in 1990")
        years = [v for v in vals if v.unit == "year"]
        assert len(years) == 1
        assert years[0].value == "1990"

    def test_age(self):
        from argos.value_extractor import extract_values
        vals = extract_values("user is 35 years old")
        ages = [v for v in vals if v.unit == "age"]
        assert len(ages) == 1
        assert ages[0].value == "35"

    def test_no_values(self):
        from argos.value_extractor import extract_values
        vals = extract_values("the user likes pizza")
        assert vals == []

    def test_empty_string(self):
        from argos.value_extractor import extract_values
        assert extract_values("") == []
        assert extract_values("   ") == []

    def test_multiple_values(self):
        from argos.value_extractor import extract_values
        vals = extract_values("score 89.8% on 1200 rows in 2026")
        units = {v.unit for v in vals}
        assert "percent" in units
        assert "count" in units

    def test_subject_window(self):
        from argos.value_extractor import extract_values
        vals = extract_values("LongMemEval benchmark score is 89.8% accuracy")
        pct = [v for v in vals if v.unit == "percent"][0]
        # Subject should contain surrounding tokens
        assert "longmemeval" in pct.subject
        assert "benchmark" in pct.subject
        assert "accuracy" in pct.subject


class TestSubjectOverlap:
    def test_identical_subjects(self):
        from argos.value_extractor import subject_overlap
        assert subject_overlap("longmemeval benchmark score", "longmemeval benchmark score")

    def test_high_overlap(self):
        from argos.value_extractor import subject_overlap
        assert subject_overlap(
            "longmemeval benchmark score accuracy",
            "longmemeval benchmark score result",
        )

    def test_low_overlap(self):
        from argos.value_extractor import subject_overlap
        assert not subject_overlap(
            "longmemeval benchmark score",
            "pizza recipe ingredients",
        )

    def test_empty(self):
        from argos.value_extractor import subject_overlap
        assert not subject_overlap("", "something")
        assert not subject_overlap("something", "")


class TestValuesConflict:
    def test_same_subject_diff_value_conflict(self):
        from argos.value_extractor import ExtractedValue, values_conflict
        new = [ExtractedValue("longmemeval benchmark score", "89.8", "percent", "89.8%")]
        old = [ExtractedValue("longmemeval benchmark score", "82.2", "percent", "82.2%")]
        conflict = values_conflict(new, old)
        assert conflict is not None
        new_v, old_v = conflict
        assert new_v.value == "89.8"
        assert old_v.value == "82.2"

    def test_same_subject_same_value_idempotent(self):
        from argos.value_extractor import ExtractedValue, values_conflict
        new = [ExtractedValue("longmemeval benchmark score", "89.8", "percent", "89.8%")]
        old = [ExtractedValue("longmemeval benchmark score", "89.8", "percent", "89.8%")]
        assert values_conflict(new, old) is None

    def test_diff_subject_same_value_no_conflict(self):
        from argos.value_extractor import ExtractedValue, values_conflict
        new = [ExtractedValue("longmemeval benchmark score", "89.8", "percent", "89.8%")]
        old = [ExtractedValue("pizza recipe accuracy", "89.8", "percent", "89.8%")]
        assert values_conflict(new, old) is None

    def test_no_values_no_conflict(self):
        from argos.value_extractor import values_conflict
        assert values_conflict([], []) is None

    def test_multiple_values_one_conflicts(self):
        from argos.value_extractor import ExtractedValue, values_conflict
        new = [
            ExtractedValue("longmemeval score", "89.8", "percent", "89.8%"),
            ExtractedValue("budget total", "449", "currency:R$", "R$ 449"),
        ]
        old = [
            ExtractedValue("longmemeval score", "82.2", "percent", "82.2%"),
            ExtractedValue("budget total", "449", "currency:R$", "R$ 449"),
        ]
        conflict = values_conflict(new, old)
        assert conflict is not None
        # The first conflict is the percentage one
        assert conflict[0].value == "89.8"
        assert conflict[1].value == "82.2"


# -- store-level tests ------------------------------------------------------

class TestStoreSupersession:
    """Store-level tests for _find_conflicting_active_value and _mark_superseded."""

    def _make_store(self, tmp_path):
        from argos.store import DuckDBMemoryStore
        store = DuckDBMemoryStore(tmp_path / "test.duckdb", user_id="test_user")
        return store

    def test_find_conflicting_active_value_detects_stale(self, tmp_path):
        """New fact with different percentage for same subject → conflict found."""
        store = self._make_store(tmp_path)
        try:
            # Old fact: 82.2%
            store.remember(
                category="context_note",
                content="LongMemEval benchmark score is 82.2%",
                dedup=False,
            )
            # New fact: 89.8% — same subject, different value
            conflict = store._find_conflicting_active_value(
                "LongMemEval benchmark score is 89.8%",
                "context_note",
            )
            assert conflict is not None
            old_id, old_content, new_val, old_val = conflict
            assert old_val == "82.2"
            assert new_val == "89.8"
            assert "82.2" in old_content
        finally:
            store.close()

    def test_find_conflicting_no_conflict_same_value(self, tmp_path):
        """Same subject + same value → no conflict (idempotent)."""
        store = self._make_store(tmp_path)
        try:
            store.remember(
                category="context_note",
                content="LongMemEval benchmark score is 89.8%",
                dedup=False,
            )
            conflict = store._find_conflicting_active_value(
                "LongMemEval benchmark score is 89.8%",
                "context_note",
            )
            assert conflict is None
        finally:
            store.close()

    def test_find_conflicting_no_conflict_diff_subject(self, tmp_path):
        """Different subject + same value → no conflict."""
        store = self._make_store(tmp_path)
        try:
            store.remember(
                category="context_note",
                content="Pizza recipe accuracy is 89.8%",
                dedup=False,
            )
            conflict = store._find_conflicting_active_value(
                "LongMemEval benchmark score is 89.8%",
                "context_note",
            )
            assert conflict is None
        finally:
            store.close()

    def test_find_conflicting_no_value_no_conflict(self, tmp_path):
        """No numeric value in new content → no conflict."""
        store = self._make_store(tmp_path)
        try:
            store.remember(
                category="context_note",
                content="LongMemEval benchmark score is 89.8%",
                dedup=False,
            )
            conflict = store._find_conflicting_active_value(
                "The user likes pizza",
                "context_note",
            )
            assert conflict is None
        finally:
            store.close()

    def test_find_conflicting_excludes_superseded(self, tmp_path):
        """Old records with valid_to set are excluded from conflict search."""
        store = self._make_store(tmp_path)
        try:
            old = store.remember(
                category="context_note",
                content="LongMemEval benchmark score is 82.2%",
                dedup=False,
            )
            # Supersede the old record
            store._mark_superseded(old.memory_id, "test")
            # Now a new fact with 89.8% should NOT conflict (old is not active)
            conflict = store._find_conflicting_active_value(
                "LongMemEval benchmark score is 89.8%",
                "context_note",
            )
            assert conflict is None
        finally:
            store.close()

    def test_mark_superseded_sets_valid_to(self, tmp_path):
        """_mark_superseded sets valid_to and excludes from retrieval."""
        store = self._make_store(tmp_path)
        try:
            rec = store.remember(
                category="context_note",
                content="Score is 82.2%",
                dedup=False,
            )
            assert rec is not None
            assert rec.valid_to is None
            ok = store._mark_superseded(rec.memory_id, "test_supersession")
            assert ok
            # Verify valid_to is set
            fetched = store._fetch_records(
                "SELECT * FROM memory_records WHERE memory_id = ?",
                [rec.memory_id],
            )
            assert fetched and fetched[0].valid_to is not None
        finally:
            store.close()

    def test_mark_superseded_nonexistent_returns_false(self, tmp_path):
        store = self._make_store(tmp_path)
        try:
            assert not store._mark_superseded("mem-nonexistent", "test")
        finally:
            store.close()


class TestSaveCandidateWithSupersession:
    """save_candidate records value conflicts in the candidate payload."""

    def _make_store(self, tmp_path):
        from argos.store import DuckDBMemoryStore
        store = DuckDBMemoryStore(tmp_path / "test.duckdb", user_id="test_user")
        return store

    def test_candidate_records_supersession_conflict(self, tmp_path):
        """Candidate with a conflicting value records the conflict in payload."""
        store = self._make_store(tmp_path)
        try:
            # Existing active fact with old value
            store.remember(
                category="context_note",
                content="LongMemEval benchmark score is 82.2%",
                dedup=False,
            )
            # New candidate with new value
            candidate = store.save_candidate(
                category="context_note",
                content="LongMemEval benchmark score is 89.8%",
                source="llm_extraction",
            )
            assert candidate is not None
            payload = candidate.get("payload", {})
            if isinstance(payload, str):
                import json
                payload = json.loads(payload)
            assert "value_supersession" in payload
            vs = payload["value_supersession"]
            assert vs["new_value"] == "89.8"
            assert vs["old_value"] == "82.2"
            assert vs["supersedes_memory_id"]
        finally:
            store.close()

    def test_candidate_no_conflict_no_supersession(self, tmp_path):
        """Candidate with no value conflict has no value_supersession in payload."""
        store = self._make_store(tmp_path)
        try:
            candidate = store.save_candidate(
                category="context_note",
                content="The user likes pizza",
                source="llm_extraction",
            )
            assert candidate is not None
            payload = candidate.get("payload", {})
            if isinstance(payload, str):
                import json
                payload = json.loads(payload)
            assert "value_supersession" not in payload
        finally:
            store.close()


class TestEndToEndSupersession:
    """E2E: candidate with value conflict → review → old excluded from retrieval."""

    def _make_store(self, tmp_path):
        from argos.store import DuckDBMemoryStore
        store = DuckDBMemoryStore(tmp_path / "test.duckdb", user_id="test_user")
        return store

    def test_supersession_excludes_old_from_retrieval(self, tmp_path):
        """After supersession, the old record is excluded from retrieval."""
        store = self._make_store(tmp_path)
        try:
            # Old fact
            old = store.remember(
                category="context_note",
                content="LongMemEval benchmark score is 82.2%",
                dedup=False,
            )
            # New fact (directly, simulating post-approval)
            new = store.remember(
                category="context_note",
                content="LongMemEval benchmark score is 89.8%",
                dedup=False,
            )
            # Supersede the old with the new
            ok = store._mark_superseded(old.memory_id, "value_supersession", new.memory_id)
            assert ok
            # Search for LongMemEval — should only return the new record
            results = store.search("LongMemEval benchmark score", limit=10)
            contents = [r.content for r in results]
            assert any("89.8" in c for c in contents)
            assert not any("82.2" in c for c in contents), \
                "Old superseded record should not appear in retrieval"
        finally:
            store.close()

    def test_supersession_via_review_candidate(self, tmp_path):
        """Full flow: save_candidate → review with supersedes_memory_id → old excluded."""
        store = self._make_store(tmp_path)
        try:
            # Old fact
            old = store.remember(
                category="context_note",
                content="LongMemEval benchmark score is 82.2%",
                dedup=False,
            )
            # New candidate with conflicting value
            candidate = store.save_candidate(
                category="context_note",
                content="LongMemEval benchmark score is 89.8%",
                source="llm_extraction",
            )
            assert candidate is not None
            # Extract the supersession info from the payload
            payload = candidate.get("payload", {})
            if isinstance(payload, str):
                import json
                payload = json.loads(payload)
            vs = payload.get("value_supersession")
            assert vs is not None
            supersedes_id = vs["supersedes_memory_id"]
            # Approve the candidate with supersession
            result = store.review_candidate(
                candidate_id=candidate["candidate_id"],
                decision="approved",
                reason="user confirmed new value",
                supersedes_memory_id=supersedes_id,
                review_source="tool",
            )
            assert result is not None
            assert result["candidate"]["status"] == "approved"
            assert result["memory"] is not None
            # Old record should be superseded
            fetched = store._fetch_records(
                "SELECT * FROM memory_records WHERE memory_id = ?",
                [old.memory_id],
            )
            assert fetched and fetched[0].valid_to is not None
            new_mem_id = result["memory"]["memory_id"]
            assert fetched[0].superseded_by == new_mem_id
            # Search should only return the new record
            results = store.search("LongMemEval benchmark score", limit=10)
            contents = [r.content for r in results]
            assert not any("82.2" in c for c in contents), \
                "Old superseded record should not appear in retrieval"
        finally:
            store.close()

"""#289: structured ingestion API (JSON/CSV → memory with provenance).

Tests:
1. JSON ingest end-to-end with a mapping spec (fields → facts),
   provenance row present per row.
2. CSV ingest (headers, quoting, missing values) end-to-end.
3. No unprovenanced writes: every insert has an evidence row;
   provenance (source file + row + mapping) queryable via the
   existing evidence path.
4. Dedupe: re-ingesting the same file is a no-op (no duplicates).
5. Supersession: re-ingest with changed value supersedes; the valid_to
   window closes the old record.
6. Dry-run: reports what WOULD write, writes NOTHING; confirm gate
   required for the actual write.
7. Validation: malformed input → per-row failure report, no partial
   silent skip.
8. Tenant scope: ingest under one user_scope is not visible to
   another; ACL (client_scope) enforced at the facade.
9. Facade operation wiring: preview/apply/confirm gate, provenance
   claims rejected (D4).

Run with (Hermes venv python, hermetic):
    ARGOS_HERMETIC_TESTS=1 python -m pytest tests/test_structured_ingest.py -v
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_plugin_dir = Path(__file__).resolve().parent.parent
if str(_plugin_dir) not in sys.path:
    sys.path.insert(0, str(_plugin_dir))


@pytest.fixture
def store(tmp_path):
    """A fresh DuckDBMemoryStore with NO embedder (hermetic, fast)."""
    from store import DuckDBMemoryStore
    s = DuckDBMemoryStore(tmp_path / "ingest.duckdb", user_id="alice")
    yield s
    s.close()


EMPLOYEE_MAPPING = {
    "category": "personal_fact",
    "content_template": "{name} works at {employer} as {role}",
    "fields": {
        "employee_id": {"type": "str", "required": True},
        "name": {"type": "str", "required": True},
        "employer": {"type": "str", "required": True},
        "role": {"type": "str", "required": True},
    },
    "key_field": "employee_id",
}

EMPLOYEE_JSON = json.dumps([
    {"employee_id": "E1", "name": "Alice", "employer": "Acme", "role": "engineer"},
    {"employee_id": "E2", "name": "Bob", "employer": "Globex", "role": "designer"},
])

EMPLOYEE_CSV = (
    "employee_id,name,employer,role\n"
    'E3,"Carol Smith, Jr",Initech,manager\n'
    "E4,Dave,Umbrella,analyst\n"
)


class TestJsonIngestEndToEnd:
    """1. JSON ingest with a mapping spec; provenance per row."""

    def test_json_ingest_maps_fields_to_facts(self, store):
        report = store.ingest_structured(
            EMPLOYEE_JSON, "json", EMPLOYEE_MAPPING, "employees.json",
            mode="apply", confirm=True,
        )
        assert report["wrote"] is True
        assert report["inserted"] == 2
        assert report["error_rows"] == 0
        contents = {r["content"] for r in report["rows"]}
        assert "Alice works at Acme as engineer" in contents
        assert "Bob works at Globex as designer" in contents
        for r in report["rows"]:
            assert r["memory_id"]
            assert r["candidate_id"]

    def test_provenance_row_present_per_row(self, store):
        report = store.ingest_structured(
            EMPLOYEE_JSON, "json", EMPLOYEE_MAPPING, "employees.json",
            mode="apply", confirm=True,
        )
        for r in report["rows"]:
            ev = store.get_evidence(r["memory_id"])
            assert ev is not None, f"no evidence row for {r['memory_id']}"
            text = ev.get("evidence_text") or ""
            assert "employees.json" in text
            assert f"row {r['row_number']}" in text
            assert report["mapping_id"] in text
            assert ev.get("evidence_role") == "structured_ingest"
            assert ev.get("reviewer_decision") == "approved"

    def test_records_carry_ingest_metadata(self, store):
        report = store.ingest_structured(
            EMPLOYEE_JSON, "json", EMPLOYEE_MAPPING, "employees.json",
            mode="apply", confirm=True,
        )
        for r in report["rows"]:
            rec = store.get_memories_by_ids([r["memory_id"]])[0]
            assert rec.namespace == "ingest:employees.json"
            assert rec.source_doc_id.startswith("employees.json#")
            assert rec.provenance_origin == "external"


class TestCsvIngestEndToEnd:
    """2. CSV ingest (headers, quoting, missing values)."""

    def test_csv_ingest_with_quoting(self, store):
        report = store.ingest_structured(
            EMPLOYEE_CSV, "csv", EMPLOYEE_MAPPING, "employees.csv",
            mode="apply", confirm=True,
        )
        assert report["inserted"] == 2
        contents = {r["content"] for r in report["rows"]}
        # Quoted field containing a comma parses as one value.
        assert "Carol Smith, Jr works at Initech as manager" in contents
        assert "Dave works at Umbrella as analyst" in contents

    def test_csv_missing_required_value_fails_that_row(self, store):
        csv_data = (
            "employee_id,name,employer,role\n"
            "E5,Eve,Acme,\n"          # missing role
            "E6,Frank,Globex,ops\n"
        )
        with pytest.raises(ValueError, match="validation failed"):
            store.ingest_structured(
                csv_data, "csv", EMPLOYEE_MAPPING, "partial.csv",
                mode="apply", confirm=True,
            )
        # Nothing written from the failed batch.
        recs = store.list_recent(limit=100)
        assert all("Eve" not in r.content for r in recs)
        assert all("Frank" not in r.content for r in recs)

    def test_csv_type_coercion(self, store):
        mapping = {
            "category": "personal_fact",
            "content_template": "{name} is {age} years old",
            "fields": {
                "person_key": {"type": "str", "required": True},
                "name": {"type": "str", "required": True},
                "age": {"type": "int", "required": True},
            },
            "key_field": "person_key",
        }
        csv_data = "person_key,name,age\nP1,Grace,36\n"
        report = store.ingest_structured(
            csv_data, "csv", mapping, "ages.csv", mode="apply", confirm=True,
        )
        assert report["inserted"] == 1
        assert report["rows"][0]["content"] == "Grace is 36 years old"


class TestNoUnprovenancedWrites:
    """3. Every insert has evidence; provenance queryable."""

    def test_every_insert_has_evidence_row(self, store):
        report = store.ingest_structured(
            EMPLOYEE_JSON, "json", EMPLOYEE_MAPPING, "employees.json",
            mode="apply", confirm=True,
        )
        memory_ids = [r["memory_id"] for r in report["rows"]]
        batch = store.get_evidence_batch(memory_ids)
        for mid in memory_ids:
            ev = batch.get(mid)
            assert ev is not None, f"memory {mid} has no evidence row"
            assert "structured_ingest" in (ev.get("evidence_text") or "")

    def test_provenance_queryable_via_provenance_path(self, store):
        report = store.ingest_structured(
            EMPLOYEE_JSON, "json", EMPLOYEE_MAPPING, "employees.json",
            mode="apply", confirm=True,
        )
        mid = report["rows"][0]["memory_id"]
        prov = store.provenance(mid)
        assert prov.get("memory_id") == mid
        evidence = prov.get("evidence") or {}
        assert "employees.json" in (evidence.get("evidence_text") or "")

    def test_candidate_row_records_provenance(self, store):
        """The candidate (approval-ledger) row carries the ingest payload."""
        report = store.ingest_structured(
            EMPLOYEE_JSON, "json", EMPLOYEE_MAPPING, "employees.json",
            mode="apply", confirm=True,
        )
        cand = store.list_candidates(
            candidate_id=report["rows"][0]["candidate_id"], limit=1,
        )[0]
        payload = cand.get("payload") or {}
        ingest = payload.get("ingest") or {}
        assert ingest.get("source_file") == "employees.json"
        assert ingest.get("row_number") == 1
        assert ingest.get("mapping_id") == report["mapping_id"]
        assert cand.get("provenance_origin") == "external"


class TestDedupeOnIngest:
    """4. Re-ingesting the same file is a no-op."""

    def test_reingest_same_file_no_duplicates(self, store):
        r1 = store.ingest_structured(
            EMPLOYEE_JSON, "json", EMPLOYEE_MAPPING, "employees.json",
            mode="apply", confirm=True,
        )
        assert r1["inserted"] == 2
        r2 = store.ingest_structured(
            EMPLOYEE_JSON, "json", EMPLOYEE_MAPPING, "employees.json",
            mode="apply", confirm=True,
        )
        assert r2["inserted"] == 0
        assert r2["duplicates"] == 2
        # Still exactly 2 current records.
        recs = store.list_recent(limit=100)
        assert len(recs) == 2

    def test_in_batch_duplicate_rows_deduped(self, store):
        data = json.dumps([
            {"employee_id": "E9", "name": "Zed", "employer": "Acme", "role": "ops"},
            {"employee_id": "E9", "name": "Zed", "employer": "Acme", "role": "ops"},
        ])
        report = store.ingest_structured(
            data, "json", EMPLOYEE_MAPPING, "dupes.json",
            mode="apply", confirm=True,
        )
        assert report["inserted"] == 1
        assert report["duplicates"] == 1

    def test_keyed_same_content_different_keys_both_stored(self, store):
        """Regression (#289): in-batch dedupe keyed on CONTENT dropped
        distinct keyed rows. Two rows with identical rendered content but
        different key values are distinct documents — both must store."""
        data = json.dumps([
            {"employee_id": "E1", "name": "Alice", "employer": "Acme", "role": "engineer"},
            {"employee_id": "E2", "name": "Alice", "employer": "Acme", "role": "engineer"},
        ])
        report = store.ingest_structured(
            data, "json", EMPLOYEE_MAPPING, "twins.json",
            mode="apply", confirm=True,
        )
        assert report["inserted"] == 2
        assert report["duplicates"] == 0
        recs = store.list_recent(limit=100)
        assert len(recs) == 2
        doc_ids = {r.source_doc_id for r in recs}
        assert doc_ids == {"twins.json#E1", "twins.json#E2"}

    def test_keyless_in_batch_content_dedupe(self, store):
        """Keyless mode keeps content-based in-batch dedupe (all rows
        share one source_doc_id, so content IS the identity)."""
        mapping = {
            "category": "personal_fact",
            "content_template": "{name} works at {employer}",
            "fields": {
                "name": {"type": "str", "required": True},
                "employer": {"type": "str", "required": True},
            },
        }
        data = json.dumps([
            {"name": "Zed", "employer": "Acme"},
            {"name": "Zed", "employer": "Acme"},
        ])
        report = store.ingest_structured(
            data, "json", mapping, "keyless.json",
            mode="apply", confirm=True,
        )
        assert report["inserted"] == 1
        assert report["duplicates"] == 1


class TestKeyedLookupScopeAware:
    """Regression (#289): the keyed existing-record lookup must honor
    client_scope/doc_class — same contract as the keyless path. Without
    this, a keyed ingest under one client_scope deduped against (or with
    changed content SUPERSEDED) another scope's records."""

    def test_keyed_changed_content_other_scope_inserts_not_supersedes(
        self, store,
    ):
        r1 = store.ingest_structured(
            EMPLOYEE_JSON, "json", EMPLOYEE_MAPPING, "employees.json",
            mode="apply", confirm=True, client_scope="client-a",
        )
        assert r1["inserted"] == 2
        changed = json.dumps([
            {"employee_id": "E1", "name": "Alice", "employer": "Globex",
             "role": "engineer"},
            {"employee_id": "E2", "name": "Bob", "employer": "Globex",
             "role": "designer"},
        ])
        r2 = store.ingest_structured(
            changed, "json", EMPLOYEE_MAPPING, "employees.json",
            mode="apply", confirm=True, client_scope="client-b",
        )
        # client-b rows are NEW documents — never supersede client-a's.
        assert r2["inserted"] == 2
        assert r2["superseded"] == 0
        assert r2["duplicates"] == 0
        # client-a records are untouched and still current.
        with store._state.lock:
            rows = store.connection.execute(
                "SELECT client_scope, valid_to FROM memory_records "
                "WHERE namespace = 'ingest:employees.json'"
            ).fetchall()
        by_scope = {}
        for scope, valid_to in rows:
            by_scope.setdefault(scope, []).append(valid_to)
        assert len(by_scope.get("client-a", [])) == 2
        assert all(v is None for v in by_scope["client-a"])
        assert len(by_scope.get("client-b", [])) == 2
        assert all(v is None for v in by_scope["client-b"])

    def test_keyed_identical_content_other_scope_inserts(self, store):
        """Identical content under a different client_scope is NOT a
        duplicate — the keyed lookup is scope-narrowed."""
        store.ingest_structured(
            EMPLOYEE_JSON, "json", EMPLOYEE_MAPPING, "employees.json",
            mode="apply", confirm=True, client_scope="client-a",
        )
        r2 = store.ingest_structured(
            EMPLOYEE_JSON, "json", EMPLOYEE_MAPPING, "employees.json",
            mode="apply", confirm=True, client_scope="client-b",
        )
        assert r2["inserted"] == 2
        assert r2["duplicates"] == 0


class TestSupersessionOnIngest:
    """5. Re-ingest with changed value supersedes; valid_to closes old."""

    def test_changed_value_supersedes(self, store):
        r1 = store.ingest_structured(
            EMPLOYEE_JSON, "json", EMPLOYEE_MAPPING, "employees.json",
            mode="apply", confirm=True,
        )
        old_id = r1["rows"][0]["memory_id"]  # Alice @ Acme
        changed = json.dumps([
            {"employee_id": "E1", "name": "Alice", "employer": "Globex",
             "role": "engineer"},
            {"employee_id": "E2", "name": "Bob", "employer": "Globex",
             "role": "designer"},
        ])
        r2 = store.ingest_structured(
            changed, "json", EMPLOYEE_MAPPING, "employees.json",
            mode="apply", confirm=True,
        )
        assert r2["superseded"] == 1
        assert r2["duplicates"] == 1
        # The old record's valid window closed and points at the new head.
        with store._state.lock:
            row = store.connection.execute(
                "SELECT valid_to, superseded_by FROM memory_records "
                "WHERE memory_id = ?",
                [old_id],
            ).fetchone()
        assert row[0] is not None, "old record's valid_to did not close"
        new_id = r2["rows"][0]["memory_id"]
        assert row[1] == new_id
        # The new head is current.
        current = store.list_recent(limit=100)
        assert any(rec.memory_id == new_id for rec in current)

    def test_superseded_value_reingest_blocked_by_tombstone(self, store):
        """Re-asserting the superseded OLD value is blocked (tombstone)."""
        r1 = store.ingest_structured(
            EMPLOYEE_JSON, "json", EMPLOYEE_MAPPING, "employees.json",
            mode="apply", confirm=True,
        )
        old_id = r1["rows"][0]["memory_id"]
        changed = EMPLOYEE_JSON.replace('"employer": "Acme"', '"employer": "Globex"')
        r2 = store.ingest_structured(
            changed, "json", EMPLOYEE_MAPPING, "employees.json",
            mode="apply", confirm=True,
        )
        assert r2["superseded"] == 1
        # Re-ingest the ORIGINAL file (old value) — the old row's content
        # is tombstoned by the supersession, so the row is blocked, not
        # silently re-activated.
        r3 = store.ingest_structured(
            EMPLOYEE_JSON, "json", EMPLOYEE_MAPPING, "employees.json",
            mode="apply", confirm=True,
        )
        assert r3["inserted"] == 0
        blocked_rows = [x for x in r3["rows"] if x["outcome"] == "blocked"]
        assert any(
            x["content"] == "Alice works at Acme as engineer"
            for x in blocked_rows
        )

    def test_wrote_false_when_every_row_blocked(self, store):
        """Regression (#289): "wrote" must reflect ACTUAL applied rows —
        an apply batch where every row ended duplicate/blocked wrote
        nothing and must not claim otherwise."""
        store.ingest_structured(
            EMPLOYEE_JSON, "json", EMPLOYEE_MAPPING, "employees.json",
            mode="apply", confirm=True,
        )
        changed = EMPLOYEE_JSON.replace('"employer": "Acme"', '"employer": "Globex"')
        r2 = store.ingest_structured(
            changed, "json", EMPLOYEE_MAPPING, "employees.json",
            mode="apply", confirm=True,
        )
        assert r2["wrote"] is True  # the supersession DID write
        # Re-ingest the ORIGINAL values: the superseded row is tombstone-
        # blocked, the unchanged row is a duplicate — nothing applied.
        r3 = store.ingest_structured(
            EMPLOYEE_JSON, "json", EMPLOYEE_MAPPING, "employees.json",
            mode="apply", confirm=True,
        )
        assert r3["inserted"] == 0
        assert r3["superseded"] == 0
        assert r3["blocked"] >= 1
        assert r3["wrote"] is False


class TestDryRunPreview:
    """6. Preview reports what WOULD write; writes NOTHING; confirm gate."""

    def test_preview_writes_nothing(self, store):
        report = store.ingest_structured(
            EMPLOYEE_JSON, "json", EMPLOYEE_MAPPING, "employees.json",
            mode="preview",
        )
        assert report["wrote"] is False
        assert report["total_rows"] == 2
        outcomes = {r["outcome"] for r in report["rows"]}
        assert outcomes == {"would_insert"}
        # No records, no candidates.
        with store._state.lock:
            n_records = store.connection.execute(
                "SELECT COUNT(*) FROM memory_records"
            ).fetchone()[0]
            n_candidates = store.connection.execute(
                "SELECT COUNT(*) FROM memory_candidates"
            ).fetchone()[0]
        assert n_records == 0
        assert n_candidates == 0

    def test_preview_reports_dedupe_collisions(self, store):
        store.ingest_structured(
            EMPLOYEE_JSON, "json", EMPLOYEE_MAPPING, "employees.json",
            mode="apply", confirm=True,
        )
        report = store.ingest_structured(
            EMPLOYEE_JSON, "json", EMPLOYEE_MAPPING, "employees.json",
            mode="preview",
        )
        outcomes = {r["outcome"] for r in report["rows"]}
        assert outcomes == {"duplicate"}
        for r in report["rows"]:
            assert r["existing_memory_id"]

    def test_preview_reports_would_supersede(self, store):
        store.ingest_structured(
            EMPLOYEE_JSON, "json", EMPLOYEE_MAPPING, "employees.json",
            mode="apply", confirm=True,
        )
        changed = EMPLOYEE_JSON.replace('"employer": "Acme"', '"employer": "Globex"')
        report = store.ingest_structured(
            changed, "json", EMPLOYEE_MAPPING, "employees.json",
            mode="preview",
        )
        by_row = {r["row_number"]: r for r in report["rows"]}
        assert by_row[1]["outcome"] == "would_supersede"
        assert by_row[2]["outcome"] == "duplicate"
        assert report["wrote"] is False

    def test_apply_without_confirm_raises(self, store):
        with pytest.raises(ValueError, match="confirm=True"):
            store.ingest_structured(
                EMPLOYEE_JSON, "json", EMPLOYEE_MAPPING, "employees.json",
                mode="apply",
            )
        # Nothing written.
        recs = store.list_recent(limit=100)
        assert len(recs) == 0

    def test_invalid_mode_raises(self, store):
        with pytest.raises(ValueError, match="mode"):
            store.ingest_structured(
                EMPLOYEE_JSON, "json", EMPLOYEE_MAPPING, "employees.json",
                mode="yolo",
            )


class TestValidationFailLoud:
    """7. Malformed input → per-row failure report, no partial silent skip."""

    def test_malformed_json_fails_loud(self, store):
        report = store.ingest_structured(
            "{not json", "json", EMPLOYEE_MAPPING, "bad.json",
            mode="preview",
        )
        assert report["wrote"] is False
        assert report["total_rows"] == 0
        assert report["errors"]
        assert "malformed JSON" in report["errors"][0]["errors"][0]

    def test_missing_required_field_reported_per_row(self, store):
        data = json.dumps([
            {"employee_id": "E5", "name": "Eve", "employer": "Acme"},  # no role
            {"employee_id": "E6", "name": "Frank", "employer": "Acme", "role": "ops"},
        ])
        report = store.ingest_structured(
            data, "json", EMPLOYEE_MAPPING, "partial.json",
            mode="preview",
        )
        assert report["error_rows"] == 1
        assert report["valid_rows"] == 1
        err = report["errors"][0]
        assert err["row_number"] == 1
        assert any("role" in e for e in err["errors"])

    def test_type_mismatch_reported_per_row(self, store):
        mapping = {
            "category": "personal_fact",
            "content_template": "{name} is {age}",
            "fields": {
                "p_key": {"type": "str", "required": True},
                "name": {"type": "str", "required": True},
                "age": {"type": "int", "required": True},
            },
            "key_field": "p_key",
        }
        data = json.dumps([
            {"p_key": "P1", "name": "Grace", "age": "not-a-number"},
        ])
        report = store.ingest_structured(
            data, "json", mapping, "badtype.json", mode="preview",
        )
        assert report["error_rows"] == 1
        assert any("age" in e for e in report["errors"][0]["errors"])

    def test_apply_with_validation_errors_writes_nothing(self, store):
        data = json.dumps([
            {"employee_id": "E5", "name": "Eve", "employer": "Acme"},  # bad
            {"employee_id": "E6", "name": "Frank", "employer": "Acme", "role": "ops"},
        ])
        with pytest.raises(ValueError, match="validation failed"):
            store.ingest_structured(
                data, "json", EMPLOYEE_MAPPING, "partial.json",
                mode="apply", confirm=True,
            )
        # All-or-nothing: the valid row must NOT have been written.
        recs = store.list_recent(limit=100)
        assert all("Frank" not in r.content for r in recs)

    def test_apply_validation_failure_attaches_report(self, store):
        """Regression (#289): the apply-mode validation failure carries
        the full report (per-row errors included) on exc.report, so the
        caller does not have to re-run preview to see what failed."""
        data = json.dumps([
            {"employee_id": "E5", "name": "Eve", "employer": "Acme"},  # bad
            {"employee_id": "E6", "name": "Frank", "employer": "Acme", "role": "ops"},
        ])
        with pytest.raises(ValueError) as excinfo:
            store.ingest_structured(
                data, "json", EMPLOYEE_MAPPING, "partial.json",
                mode="apply", confirm=True,
            )
        report = getattr(excinfo.value, "report", None)
        assert report is not None, "exception carries no report"
        assert report["wrote"] is False
        assert report["error_rows"] == 1
        assert report["errors"][0]["row_number"] == 1
        assert any("role" in e for e in report["errors"][0]["errors"])

    def test_bad_mapping_spec_fails_loud(self, store):
        bad_mapping = {"category": "personal_fact", "content_template": "{nope}"}
        report = store.ingest_structured(
            EMPLOYEE_JSON, "json", bad_mapping, "badmap.json",
            mode="preview",
        )
        # Unknown template field → per-row errors, nothing mapped.
        assert report["valid_rows"] == 0
        assert report["error_rows"] == 2


class TestTenantScope:
    """8. Ingest under one user_scope is not visible to another."""

    def test_ingest_scoped_to_user(self, store):
        r = store.ingest_structured(
            EMPLOYEE_JSON, "json", EMPLOYEE_MAPPING, "employees.json",
            mode="apply", confirm=True,
        )
        assert r["inserted"] == 2
        # Alice sees her records.
        store.set_user_scope("alice")
        assert len(store.list_recent(limit=100)) == 2
        # Bob sees nothing.
        store.set_user_scope("bob")
        assert len(store.list_recent(limit=100)) == 0
        # Evidence is also scope-isolated.
        mid = r["rows"][0]["memory_id"]
        assert store.get_evidence(mid) is None

    def test_same_file_different_users_no_cross_talk(self, store):
        store.ingest_structured(
            EMPLOYEE_JSON, "json", EMPLOYEE_MAPPING, "employees.json",
            mode="apply", confirm=True,
        )
        store.set_user_scope("bob")
        r2 = store.ingest_structured(
            EMPLOYEE_JSON, "json", EMPLOYEE_MAPPING, "employees.json",
            mode="apply", confirm=True,
        )
        # Bob's ingest inserts his OWN copies (user_scope isolation) —
        # he does not dedupe against alice's rows.
        assert r2["inserted"] == 2
        store.set_user_scope("alice")
        assert len(store.list_recent(limit=100)) == 2
        store.set_user_scope("bob")
        assert len(store.list_recent(limit=100)) == 2


class TestFacadeIngestOperation:
    """9. Facade wiring: preview/apply/confirm gate, D4 provenance claims."""

    @pytest.fixture
    def facade(self, tmp_path):
        from store import DuckDBMemoryStore
        from api_facade import ArgosAPIFacade
        s = DuckDBMemoryStore(tmp_path / "facade.duckdb", user_id="alice")
        f = ArgosAPIFacade(s)
        yield f, s
        s.close()

    def _ctx(self):
        from api_facade import AuthContext, READ_OPERATIONS, PROPOSAL_OPERATIONS
        return AuthContext(
            principal="test-client",
            tenant="default",
            user_id="alice",
            transport="test",
            allowed_operations=set(READ_OPERATIONS) | set(PROPOSAL_OPERATIONS),
            can_propose=True,
        )

    def test_facade_preview_and_apply(self, facade):
        f, _store = facade
        ctx = self._ctx()
        params = {
            "data": EMPLOYEE_JSON,
            "fmt": "json",
            "mapping": EMPLOYEE_MAPPING,
            "source_name": "employees.json",
            "mode": "preview",
        }
        result = f.execute(ctx, "ingest", params)
        assert result["wrote"] is False
        assert result["total_rows"] == 2

        params["mode"] = "apply"
        params["confirm"] = True
        result = f.execute(ctx, "ingest", params)
        assert result["wrote"] is True
        assert result["inserted"] == 2

    def test_facade_apply_without_confirm_rejected(self, facade):
        f, _store = facade
        ctx = self._ctx()
        with pytest.raises(Exception):
            f.execute(ctx, "ingest", {
                "data": EMPLOYEE_JSON,
                "fmt": "json",
                "mapping": EMPLOYEE_MAPPING,
                "source_name": "employees.json",
                "mode": "apply",
            })

    def test_facade_string_confirm_rejected(self, facade):
        """Regression (#289): bool("false") is True — a client sending
        the STRING "false" must not pass the human-in-loop gate. Only
        the literal boolean True confirms."""
        f, store = facade
        ctx = self._ctx()
        with pytest.raises(Exception):
            f.execute(ctx, "ingest", {
                "data": EMPLOYEE_JSON,
                "fmt": "json",
                "mapping": EMPLOYEE_MAPPING,
                "source_name": "employees.json",
                "mode": "apply",
                "confirm": "false",
            })
        # Nothing written.
        assert len(store.list_recent(limit=100)) == 0

    def test_facade_ingest_size_limit_counts_bytes(self, facade):
        """Regression (#289): the size limit must count UTF-8 BYTES, not
        characters — multibyte content can exceed the wire limit while
        passing a character count."""
        f, _store = facade
        ctx = self._ctx()
        # ~200k chars but ~400k UTF-8 bytes (> 256 KiB limit).
        data = json.dumps([{"name": "é" * 200_000}])
        with pytest.raises(Exception) as excinfo:
            f.execute(ctx, "ingest", {
                "data": data,
                "fmt": "json",
                "mapping": EMPLOYEE_MAPPING,
                "source_name": "big.json",
                "mode": "preview",
            })
        assert getattr(excinfo.value, "code", "") == "request_too_large"

    def test_facade_rejects_provenance_claims(self, facade):
        """D4: the caller may not claim server-set provenance fields."""
        f, _store = facade
        ctx = self._ctx()
        with pytest.raises(Exception):
            f.execute(ctx, "ingest", {
                "data": EMPLOYEE_JSON,
                "fmt": "json",
                "mapping": EMPLOYEE_MAPPING,
                "source_name": "employees.json",
                "mode": "preview",
                "provenance_origin": "internal",
            })

    def test_facade_client_scope_narrowing(self, facade):
        """ACL: client_scope may only narrow the credential clearance."""
        f, _store = facade
        from api_facade import AuthContext, READ_OPERATIONS, PROPOSAL_OPERATIONS
        ctx = AuthContext(
            principal="test-client",
            tenant="default",
            user_id="alice",
            transport="test",
            allowed_operations=set(READ_OPERATIONS) | set(PROPOSAL_OPERATIONS),
            can_propose=True,
            max_client_scope="client-a",
        )
        result = f.execute(ctx, "ingest", {
            "data": EMPLOYEE_JSON,
            "fmt": "json",
            "mapping": EMPLOYEE_MAPPING,
            "source_name": "employees.json",
            "mode": "apply",
            "confirm": True,
        })
        assert result["inserted"] == 2
        # Rows stamped with the caller's clearance.
        mid = result["rows"][0]["memory_id"]
        rec = _store.get_memories_by_ids([mid])[0]
        assert rec.client_scope == "client-a"

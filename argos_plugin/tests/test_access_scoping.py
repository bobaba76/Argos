"""Spec-06 (#69): access scoping — per-user, per-client ACL inside a tenant.

Tests (cheapest falsifying first — deterministic, no LLM calls):
1. Mask evaluation: allow-only, wheel, deny-beats-allow, deny-beats-wheel,
   NULL-scope rows.
2. Retrieval filter: document-facts outside the mask never returned
   (records + candidates).
3. Graph guard: shared-entity expansion drops cross-scope neighbours.
4. Audit: every query and every deny writes a row; export works;
   single-user pilot mode still records identity.
5. Full suite green — no ACL config = today's behaviour (backward
   compatible; absence of a config file opens the store exactly as now).
"""
import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

# Ensure argos_plugin is importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from access_scoping import (
    ACLConfig,
    PRACTICE_INTERNAL,
    filter_graph_neighbours,
    filter_records_by_access,
)


# ---------------------------------------------------------------------------
# 1. Mask evaluation unit tests
# ---------------------------------------------------------------------------


class TestMaskEvaluation:
    """D1: role-based allow masks + hidden deny list (deny > allow > wheel)."""

    def test_open_store_allows_everything(self):
        """No ACL config = today's open-store behaviour."""
        config = ACLConfig()
        assert config.is_open_store
        assert config.can_see("anyone", client_scope="acme", doc_class=None)
        assert config.can_see("anyone", client_scope=None, doc_class=None)
        assert config.can_see("anyone", client_scope="acme", doc_class=PRACTICE_INTERNAL)

    def test_wheel_user_sees_all_scopes(self):
        """Principals/partners default to wheel (all client scopes)."""
        config = ACLConfig(
            roles={"principal": {"wheel": True}},
            user_roles={"alice": "principal"},
            enforcement_on=True,
        )
        assert not config.is_open_store
        assert config.allow_mask("alice") is None  # wheel = all scopes
        assert config.can_see("alice", client_scope="acme", doc_class=None)
        assert config.can_see("alice", client_scope="beta", doc_class=None)
        assert config.can_see("alice", client_scope=None, doc_class=None)

    def test_allow_only_user_sees_assigned_scopes(self):
        """Staff roles are assigned specific scopes."""
        config = ACLConfig(
            roles={"staff": {"client_scopes": ["acme"], "wheel": False}},
            user_roles={"bob": "staff"},
            enforcement_on=True,
        )
        mask = config.allow_mask("bob")
        assert mask == {"acme"}
        assert config.can_see("bob", client_scope="acme", doc_class=None)
        assert not config.can_see("bob", client_scope="beta", doc_class=None)

    def test_deny_beats_allow(self):
        """Deny list wins over allow mask."""
        config = ACLConfig(
            roles={"staff": {"client_scopes": ["acme", "beta"], "wheel": False}},
            user_roles={"bob": "staff"},
            deny_lists={"bob": [{"client_scope": "acme"}]},
            enforcement_on=True,
        )
        assert not config.can_see("bob", client_scope="acme", doc_class=None)
        assert config.can_see("bob", client_scope="beta", doc_class=None)

    def test_deny_beats_wheel(self):
        """Deny list wins over wheel (principals)."""
        config = ACLConfig(
            roles={"principal": {"wheel": True}},
            user_roles={"alice": "principal"},
            deny_lists={"alice": [{"client_scope": "hostile"}]},
            enforcement_on=True,
        )
        assert not config.can_see("alice", client_scope="hostile", doc_class=None)
        assert config.can_see("alice", client_scope="acme", doc_class=None)

    def test_null_scope_rows_visible_to_all(self):
        """NULL client_scope = global; visible inside any client query."""
        config = ACLConfig(
            roles={"staff": {"client_scopes": ["acme"], "wheel": False}},
            user_roles={"bob": "staff"},
            enforcement_on=True,
        )
        assert config.can_see("bob", client_scope=None, doc_class=None)

    def test_practice_internal_principals_only(self):
        """practice-internal doc_class = principals-only; staff denied."""
        config = ACLConfig(
            roles={
                "principal": {"wheel": True},
                "staff": {"client_scopes": ["acme"], "wheel": False},
            },
            user_roles={"alice": "principal", "bob": "staff"},
            enforcement_on=True,
        )
        assert config.can_see("alice", client_scope=None, doc_class=PRACTICE_INTERNAL)
        assert not config.can_see("bob", client_scope="acme", doc_class=PRACTICE_INTERNAL)

    def test_role_level_deny_list(self):
        """Deny lists can be at the role level, not just user level."""
        config = ACLConfig(
            roles={"staff": {"client_scopes": ["acme", "beta"], "wheel": False}},
            user_roles={"bob": "staff"},
            deny_lists={"staff": [{"doc_class": PRACTICE_INTERNAL}]},
            enforcement_on=True,
        )
        assert not config.can_see("bob", client_scope="acme", doc_class=PRACTICE_INTERNAL)
        assert config.can_see("bob", client_scope="acme", doc_class=None)

    def test_unassigned_user_under_enforcement_denied_all(self):
        """A user with no role under enforcement fails closed (deny-all)."""
        config = ACLConfig(
            roles={"staff": {"client_scopes": ["acme"], "wheel": False}},
            user_roles={"bob": "staff"},
            enforcement_on=True,
        )
        mask = config.allow_mask("charlie")  # no role assigned
        assert mask == set()
        assert not config.can_see("charlie", client_scope="acme", doc_class=None)

    def test_from_file_missing_returns_open_store(self, tmp_path):
        """Missing config file = open store."""
        config = ACLConfig.from_file(tmp_path / "nonexistent.json")
        assert config.is_open_store

    def test_from_file_valid(self, tmp_path):
        """Valid config file loads correctly."""
        (tmp_path / "acl.json").write_text(json.dumps({
            "roles": {"staff": {"client_scopes": ["acme"], "wheel": False}},
            "user_roles": {"bob": "staff"},
            "enforcement_on": True,
        }))
        config = ACLConfig.from_file(tmp_path / "acl.json")
        assert not config.is_open_store
        assert config.can_see("bob", client_scope="acme", doc_class=None)
        assert not config.can_see("bob", client_scope="beta", doc_class=None)

    def test_from_file_corrupt_returns_open_store(self, tmp_path):
        """Corrupt config file = open store (fail safe)."""
        (tmp_path / "acl.json").write_text("not valid json{{{")
        config = ACLConfig.from_file(tmp_path / "acl.json")
        assert config.is_open_store


# ---------------------------------------------------------------------------
# 2. Retrieval filter
# ---------------------------------------------------------------------------


def _record(memory_id, client_scope=None, doc_class=None):
    """Build a simple record-like object for filter tests."""
    return SimpleNamespace(
        memory_id=memory_id,
        client_scope=client_scope,
        doc_class=doc_class,
    )


class TestRetrievalFilter:
    """D3: retrieval filter — out-of-mask document facts never returned."""

    def test_open_store_returns_all(self):
        """No ACL config = all records pass through."""
        records = [
            _record("m1", client_scope="acme"),
            _record("m2", client_scope="beta"),
            _record("m3", client_scope=None, doc_class=PRACTICE_INTERNAL),
        ]
        config = ACLConfig()
        visible, denied = filter_records_by_access(records, config, "bob")
        assert len(visible) == 3
        assert denied == 0

    def test_staff_sees_only_assigned_scopes(self):
        """Staff with acme scope sees acme + global, not beta."""
        records = [
            _record("m1", client_scope="acme"),
            _record("m2", client_scope="beta"),
            _record("m3", client_scope=None),  # global
        ]
        config = ACLConfig(
            roles={"staff": {"client_scopes": ["acme"], "wheel": False}},
            user_roles={"bob": "staff"},
            enforcement_on=True,
        )
        visible, denied = filter_records_by_access(records, config, "bob")
        visible_ids = {r.memory_id for r in visible}
        assert visible_ids == {"m1", "m3"}
        assert denied == 1

    def test_deny_hides_record(self):
        """Deny list hides a record even if in the allow mask."""
        records = [
            _record("m1", client_scope="acme"),
            _record("m2", client_scope="acme"),
        ]
        config = ACLConfig(
            roles={"staff": {"client_scopes": ["acme"], "wheel": False}},
            user_roles={"bob": "staff"},
            deny_lists={"bob": [{"client_scope": "acme"}]},
            enforcement_on=True,
        )
        visible, denied = filter_records_by_access(records, config, "bob")
        assert len(visible) == 0
        assert denied == 2

    def test_practice_internal_hidden_from_staff(self):
        """practice-internal records hidden from non-wheel users."""
        records = [
            _record("m1", client_scope="acme", doc_class=PRACTICE_INTERNAL),
            _record("m2", client_scope="acme", doc_class=None),
        ]
        config = ACLConfig(
            roles={"staff": {"client_scopes": ["acme"], "wheel": False}},
            user_roles={"bob": "staff"},
            enforcement_on=True,
        )
        visible, denied = filter_records_by_access(records, config, "bob")
        visible_ids = {r.memory_id for r in visible}
        assert visible_ids == {"m2"}
        assert denied == 1

    def test_wheel_sees_practice_internal(self):
        """Wheel (principal) sees practice-internal records."""
        records = [
            _record("m1", client_scope=None, doc_class=PRACTICE_INTERNAL),
        ]
        config = ACLConfig(
            roles={"principal": {"wheel": True}},
            user_roles={"alice": "principal"},
            enforcement_on=True,
        )
        visible, denied = filter_records_by_access(records, config, "alice")
        assert len(visible) == 1
        assert denied == 0


# ---------------------------------------------------------------------------
# 3. Graph guard
# ---------------------------------------------------------------------------


def _node(node_id, client_scope=None):
    """Build a graph node dict with client_scope in attributes."""
    return {
        "id": node_id,
        "entity_type": "person",
        "attributes": {"client_scope": client_scope} if client_scope else {},
    }


def _edge(source, target, relation="knows"):
    return {"source": source, "target": target, "relation": relation}


class TestGraphGuard:
    """D3: graph traversal post-filter — cross-scope neighbours dropped."""

    def test_open_store_no_filter(self):
        """No ACL config = all nodes/edges pass through."""
        nodes = [_node("a", "acme"), _node("b", "beta")]
        edges = [_edge("a", "b")]
        config = ACLConfig()
        filtered_nodes, filtered_edges, dropped = filter_graph_neighbours(
            nodes, edges, config, "bob",
        )
        assert len(filtered_nodes) == 2
        assert len(filtered_edges) == 1
        assert dropped == 0

    def test_cross_scope_neighbours_dropped(self):
        """A shared director between acme and beta: beta node dropped for
        a user with only acme in their mask."""
        nodes = [_node("director", "acme"), _node("beta_fact", "beta")]
        edges = [_edge("director", "beta_fact")]
        config = ACLConfig(
            roles={"staff": {"client_scopes": ["acme"], "wheel": False}},
            user_roles={"bob": "staff"},
            enforcement_on=True,
        )
        filtered_nodes, filtered_edges, dropped = filter_graph_neighbours(
            nodes, edges, config, "bob",
        )
        node_ids = {n["id"] for n in filtered_nodes}
        assert node_ids == {"director"}
        assert len(filtered_edges) == 0  # edge to beta_fact dropped
        assert dropped == 1

    def test_global_nodes_visible(self):
        """NULL client_scope nodes (global) are visible to all."""
        nodes = [_node("global_entity", None), _node("acme_entity", "acme")]
        edges = [_edge("global_entity", "acme_entity")]
        config = ACLConfig(
            roles={"staff": {"client_scopes": ["acme"], "wheel": False}},
            user_roles={"bob": "staff"},
            enforcement_on=True,
        )
        filtered_nodes, filtered_edges, dropped = filter_graph_neighbours(
            nodes, edges, config, "bob",
        )
        assert len(filtered_nodes) == 2
        assert len(filtered_edges) == 1
        assert dropped == 0

    def test_wheel_sees_all_nodes(self):
        """Wheel (principal) sees all nodes, no filtering."""
        nodes = [_node("a", "acme"), _node("b", "beta")]
        edges = [_edge("a", "b")]
        config = ACLConfig(
            roles={"principal": {"wheel": True}},
            user_roles={"alice": "principal"},
            enforcement_on=True,
        )
        filtered_nodes, filtered_edges, dropped = filter_graph_neighbours(
            nodes, edges, config, "alice",
        )
        assert len(filtered_nodes) == 2
        assert len(filtered_edges) == 1
        assert dropped == 0


# ---------------------------------------------------------------------------
# 4. Audit log
# ---------------------------------------------------------------------------


class TestAccessAudit:
    """D4: append-only audit log — every query and every deny writes a row."""

    @pytest.fixture
    def store(self, tmp_path):
        """Build a DuckDBMemoryStore with the access_audit table."""
        from store import DuckDBMemoryStore
        return DuckDBMemoryStore(tmp_path / "test.duckdb", user_id="alice")

    def test_audit_table_exists(self, store):
        """The access_audit table is created on init."""
        result = store.connection.execute(
            "SELECT count(*) FROM information_schema.tables "
            "WHERE table_name = 'access_audit'"
        ).fetchone()
        assert result[0] == 1

    def test_write_audit_row(self, store):
        """write_access_audit inserts a row."""
        store.write_access_audit(
            user_id="alice",
            query_text="what is acme's VAT number?",
            granted_count=3,
            denied_count=1,
            denied_scopes="beta",
            excluded=True,
        )
        result = store.connection.execute(
            "SELECT user_id, query_text, granted_count, denied_count, excluded "
            "FROM access_audit"
        ).fetchone()
        assert result[0] == "alice"
        assert result[1] == "what is acme's VAT number?"
        assert result[2] == 3
        assert result[3] == 1
        assert result[4] is True

    def test_export_jsonl(self, store):
        """Export produces JSONL with the audit rows."""
        store.write_access_audit(
            user_id="alice", query_text="query 1",
            granted_count=2, denied_count=0,
        )
        store.write_access_audit(
            user_id="bob", query_text="query 2",
            granted_count=0, denied_count=3, excluded=True,
        )
        export = store.export_access_audit(format="jsonl")
        lines = [json.loads(line) for line in export.strip().split("\n") if line]
        assert len(lines) == 2
        assert lines[0]["user_id"] == "bob"  # DESC by ts
        assert lines[1]["user_id"] == "alice"

    def test_export_csv(self, store):
        """Export produces CSV with a header row."""
        store.write_access_audit(
            user_id="alice", query_text="query 1",
            granted_count=1, denied_count=0,
        )
        export = store.export_access_audit(format="csv")
        lines = export.strip().split("\n")
        assert "user_id" in lines[0]  # header
        assert len(lines) == 2  # header + 1 data row

    def test_single_user_pilot_records_identity(self, store):
        """Pilot mode (single-user) still records identity in the audit."""
        store.write_access_audit(
            user_id="alice", query_text="solo query",
            granted_count=5, denied_count=0,
        )
        result = store.connection.execute(
            "SELECT user_id FROM access_audit"
        ).fetchone()
        assert result[0] == "alice"


# ---------------------------------------------------------------------------
# 5. Store-level: doc_class column + backward compatibility
# ---------------------------------------------------------------------------


class TestStoreDocClass:
    """D2: doc_class column on memory_records + backward compatibility."""

    @pytest.fixture
    def store(self, tmp_path):
        from store import DuckDBMemoryStore
        return DuckDBMemoryStore(tmp_path / "test.duckdb", user_id="alice")

    def test_remember_accepts_doc_class(self, store):
        """remember() accepts and persists doc_class."""
        record = store.remember(
            category="personal_fact",
            content="Acme invoice #1234",
            namespace="document",
            client_scope="acme",
            doc_class="invoice",
        )
        assert record is not None
        assert record.doc_class == "invoice"

    def test_remember_defaults_doc_class_none(self, store):
        """remember() without doc_class defaults to None (legacy behaviour)."""
        record = store.remember(
            category="personal_fact",
            content="User likes coffee",
        )
        assert record is not None
        assert record.doc_class is None

    def test_doc_class_round_trips_through_search(self, store):
        """doc_class survives a save + search round-trip."""
        store.remember(
            category="personal_fact",
            content="Acme VAT number is 456",
            namespace="document",
            client_scope="acme",
            doc_class="tax",
        )
        results = store.search("VAT", limit=10)
        assert any(r.doc_class == "tax" for r in results)

    def test_doc_class_carries_through_update(self, store):
        """doc_class carries through the update_memory version chain."""
        record = store.remember(
            category="personal_fact",
            content="Acme VAT is 100",
            namespace="document",
            client_scope="acme",
            doc_class="tax",
        )
        updated = store.update_memory(
            record.memory_id, content="Acme VAT is 200",
        )
        assert updated is not None
        assert updated.doc_class == "tax"

    def test_candidate_carries_doc_class(self, store):
        """save_candidate accepts and list_candidates returns doc_class."""
        store.save_candidate(
            category="personal_fact",
            content="Acme revenue is $1M",
            namespace="document",
            client_scope="acme",
            doc_class="financial",
        )
        candidates = store.list_candidates(limit=10)
        assert any(c.get("doc_class") == "financial" for c in candidates)

    def test_no_acl_config_zero_behaviour_change(self, store):
        """Without an ACL config, search returns everything (backward compat)."""
        store.remember(
            category="personal_fact",
            content="Public fact about the weather",
            client_scope="acme",
            doc_class="general",
        )
        # No ACL config set on the store — open store behaviour.
        results = store.search("weather", limit=10)
        assert len(results) == 1
        assert results[0].content == "Public fact about the weather"


# ---------------------------------------------------------------------------
# AS1: Graph filter uses can_see() (deny lists + doc_class)
# ---------------------------------------------------------------------------

class TestGraphFilterCanSee:
    """AS1: filter_graph_neighbours must enforce deny lists and
    practice-internal doc_class, not just the allow mask."""

    def test_deny_list_enforced_in_graph(self):
        """A user with a deny entry on 'acme' must not see acme-scoped
        graph nodes, even though 'acme' is in their allow mask."""
        config = ACLConfig(
            roles={"staff": {"client_scopes": ["acme"]}},
            user_roles={"bob": "staff"},
            deny_lists={"bob": [{"client_scope": "acme"}]},
            enforcement_on=True,
        )
        nodes = [{"id": "n1", "attributes": {"client_scope": "acme"}}]
        edges = []
        filtered, _, dropped = filter_graph_neighbours(nodes, edges, config, "bob")
        assert len(filtered) == 0
        assert dropped == 1

    def test_practice_internal_hidden_from_staff_in_graph(self):
        """A staff user must not see practice-internal graph nodes."""
        config = ACLConfig(
            roles={"staff": {"client_scopes": ["acme"]}},
            user_roles={"bob": "staff"},
            enforcement_on=True,
        )
        nodes = [
            {"id": "n1", "attributes": {"client_scope": "acme", "doc_class": "invoice"}},
            {"id": "n2", "attributes": {"client_scope": "acme", "doc_class": PRACTICE_INTERNAL}},
        ]
        edges = []
        filtered, _, dropped = filter_graph_neighbours(nodes, edges, config, "bob")
        ids = {n["id"] for n in filtered}
        assert "n1" in ids
        assert "n2" not in ids
        assert dropped == 1

    def test_wheel_sees_practice_internal_in_graph(self):
        """A wheel (principal) user sees practice-internal graph nodes."""
        config = ACLConfig(
            roles={"principal": {"wheel": True}},
            user_roles={"alice": "principal"},
            enforcement_on=True,
        )
        nodes = [
            {"id": "n1", "attributes": {"client_scope": "acme", "doc_class": PRACTICE_INTERNAL}},
        ]
        edges = []
        filtered, _, dropped = filter_graph_neighbours(nodes, edges, config, "alice")
        assert len(filtered) == 1
        assert dropped == 0

    def test_deny_list_respected_over_wheel_in_graph(self):
        """A wheel user with a deny entry still can't see denied nodes."""
        config = ACLConfig(
            roles={"principal": {"wheel": True}},
            user_roles={"alice": "principal"},
            deny_lists={"alice": [{"client_scope": "beta"}]},
            enforcement_on=True,
        )
        nodes = [
            {"id": "n1", "attributes": {"client_scope": "acme"}},
            {"id": "n2", "attributes": {"client_scope": "beta"}},
        ]
        edges = []
        filtered, _, dropped = filter_graph_neighbours(nodes, edges, config, "alice")
        ids = {n["id"] for n in filtered}
        assert "n1" in ids
        assert "n2" not in ids
        assert dropped == 1


# ---------------------------------------------------------------------------
# AS3: client_scopes string-instead-of-list validation
# ---------------------------------------------------------------------------

class TestClientScopesValidation:
    """AS3: a string client_scopes must be wrapped in a list, not
    silently corrupted into single-character set entries."""

    def test_string_client_scopes_wrapped(self):
        """A string 'acme' should be wrapped to ['acme'], not set('acme')."""
        config = ACLConfig.from_dict({
            "roles": {"staff": {"client_scopes": "acme"}},
            "user_roles": {"bob": "staff"},
            "enforcement_on": True,
        })
        mask = config.allow_mask("bob")
        assert "acme" in mask
        assert "a" not in mask
        assert "c" not in mask
        assert config.can_see("bob", client_scope="acme", doc_class=None)

    def test_list_client_scopes_unchanged(self):
        """A proper list should pass through unchanged."""
        config = ACLConfig.from_dict({
            "roles": {"staff": {"client_scopes": ["acme", "beta"]}},
            "user_roles": {"bob": "staff"},
            "enforcement_on": True,
        })
        mask = config.allow_mask("bob")
        assert mask == {"acme", "beta"}

    def test_non_string_elements_rejected(self):
        """client_scopes with non-string elements should be parse_error."""
        config = ACLConfig.from_dict({
            "roles": {"staff": {"client_scopes": [1, 2, 3]}},
            "user_roles": {"bob": "staff"},
            "enforcement_on": True,
        })
        assert config.parse_error is True


# ---------------------------------------------------------------------------
# AS4: Dangling user_roles warning
# ---------------------------------------------------------------------------

class TestDanglingUserRoles:
    """AS4: a user assigned to a non-existent role should produce a
    warning log (the user is correctly deny-all, but silently)."""

    def test_dangling_role_warned(self, caplog):
        """from_dict should log a warning for dangling role references."""
        import logging
        with caplog.at_level(logging.WARNING, logger="argos_plugin.access_scoping"):
            ACLConfig.from_dict({
                "roles": {"staff": {"client_scopes": ["acme"]}},
                "user_roles": {"bob": "nonexistent_role"},
                "enforcement_on": True,
            })
        assert any("nonexistent_role" in r.message for r in caplog.records)

    def test_valid_role_no_warning(self, caplog):
        """Valid role assignments should not produce warnings."""
        import logging
        with caplog.at_level(logging.WARNING, logger="argos_plugin.access_scoping"):
            ACLConfig.from_dict({
                "roles": {"staff": {"client_scopes": ["acme"]}},
                "user_roles": {"bob": "staff"},
                "enforcement_on": True,
            })
        assert not any("does not exist" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# AS5: Empty deny entry warning
# ---------------------------------------------------------------------------

class TestEmptyDenyEntry:
    """AS5: a deny entry with no fields (wildcard deny) should produce
    a warning log."""

    def test_empty_deny_entry_warned(self, caplog):
        """from_dict should warn when a deny entry has no fields."""
        import logging
        with caplog.at_level(logging.WARNING, logger="argos_plugin.access_scoping"):
            ACLConfig.from_dict({
                "roles": {"staff": {"client_scopes": ["acme"]}},
                "user_roles": {"bob": "staff"},
                "deny_lists": {"bob": [{}]},
                "enforcement_on": True,
            })
        assert any("denies ALL" in r.message for r in caplog.records)

    def test_specific_deny_entry_no_warning(self, caplog):
        """A deny entry with fields should not produce the wildcard warning."""
        import logging
        with caplog.at_level(logging.WARNING, logger="argos_plugin.access_scoping"):
            ACLConfig.from_dict({
                "roles": {"staff": {"client_scopes": ["acme"]}},
                "user_roles": {"bob": "staff"},
                "deny_lists": {"bob": [{"client_scope": "acme"}]},
                "enforcement_on": True,
            })
        assert not any("denies ALL" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# AS8: Nodes without id field
# ---------------------------------------------------------------------------

class TestNodesWithoutId:
    """AS8: nodes without an 'id' field should be skipped (not treated
    as sharing the empty-string ID)."""

    def test_node_without_id_dropped(self):
        """A node without an 'id' field should be dropped, not visible."""
        config = ACLConfig(
            roles={"staff": {"client_scopes": ["acme"]}},
            user_roles={"bob": "staff"},
            enforcement_on=True,
        )
        nodes = [
            {"id": "n1", "attributes": {"client_scope": "acme"}},
            {"attributes": {"client_scope": "acme"}},  # no id
        ]
        edges = []
        filtered, _, dropped = filter_graph_neighbours(nodes, edges, config, "bob")
        ids = [n.get("id") for n in filtered]
        assert "n1" in ids
        assert None not in ids
        assert dropped == 1

    def test_all_nodes_without_id_dropped(self):
        """All nodes without ids should be dropped (not all visible via '')."""
        config = ACLConfig(
            roles={"staff": {"client_scopes": ["acme"]}},
            user_roles={"bob": "staff"},
            enforcement_on=True,
        )
        nodes = [
            {"attributes": {"client_scope": "acme"}},
            {"attributes": {"client_scope": "acme"}},
        ]
        edges = []
        filtered, _, dropped = filter_graph_neighbours(nodes, edges, config, "bob")
        assert len(filtered) == 0
        assert dropped == 2

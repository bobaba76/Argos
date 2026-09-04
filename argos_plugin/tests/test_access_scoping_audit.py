"""Audit regression guard for access_scoping (issue #211, AS1-AS8).

Note: the AS1-AS8 fixes were already applied in commit a01b7e2 (an ancestor
of HEAD) and are covered by ``test_access_scoping.py``. This file is an
explicit acceptance-criteria mapping — a fast, hermetic regression guard
that ties each AS finding to a focused test so a future regression is
caught here.

AS2 and AS7 are implemented in ``provider_retrieval.py`` (outside this
issue's file scope); this file tests the access_scoping.py layer and
documents the cross-file fixes.

Run with (Hermes venv python, offline):
    python -m pytest tests/test_access_scoping_audit.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_plugin_dir = Path(__file__).resolve().parent.parent
for _path in (_plugin_dir.parent, _plugin_dir):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from access_scoping import (  # noqa: E402
    ACLConfig,
    PRACTICE_INTERNAL,
    filter_graph_neighbours,
    filter_records_by_access,
)


# ---------------------------------------------------------------------------
# AS1 — graph filter uses can_see() (deny lists + doc_class enforced)
# ---------------------------------------------------------------------------

class TestAS1GraphFilterBypass:
    def test_deny_list_enforced_in_graph_filter(self):
        """A staff user with a deny entry on 'acme' cannot see acme-scoped
        nodes through the graph (the old filter only checked the allow
        mask, not deny lists)."""
        config = ACLConfig(
            roles={"staff": {"client_scopes": ["acme"]}},
            user_roles={"bob": "staff"},
            deny_lists={"bob": [{"client_scope": "acme"}]},
            enforcement_on=True,
        )
        nodes = [{"id": "n1", "attributes": {"client_scope": "acme"}}]
        edges: list = []
        filtered, _, dropped = filter_graph_neighbours(nodes, edges, config, "bob")
        assert len(filtered) == 0
        assert dropped == 1

    def test_practice_internal_blocked_in_graph_filter(self):
        """A staff user cannot see practice-internal nodes through the
        graph (can_see blocks practice-internal for non-wheel)."""
        config = ACLConfig(
            roles={"staff": {"client_scopes": ["acme"]}},
            user_roles={"bob": "staff"},
            enforcement_on=True,
        )
        nodes = [{"id": "n1", "attributes": {"doc_class": PRACTICE_INTERNAL}}]
        edges: list = []
        filtered, _, _ = filter_graph_neighbours(nodes, edges, config, "bob")
        assert len(filtered) == 0

    def test_wheel_user_deny_list_still_enforced(self):
        """A wheel (principals) user with a deny entry is still blocked
        by the deny list in the graph filter."""
        config = ACLConfig(
            roles={"principal": {"wheel": True}},
            user_roles={"alice": "principal"},
            deny_lists={"alice": [{"client_scope": "beta"}]},
            enforcement_on=True,
        )
        nodes = [{"id": "n1", "attributes": {"client_scope": "beta"}}]
        edges: list = []
        filtered, _, _ = filter_graph_neighbours(nodes, edges, config, "alice")
        assert len(filtered) == 0


# ---------------------------------------------------------------------------
# AS3 — client_scopes string-instead-of-list validation
# ---------------------------------------------------------------------------

class TestAS3ClientScopesValidation:
    def test_string_client_scopes_wrapped_in_list(self):
        """A string 'acme' instead of ['acme'] is wrapped in a list by
        from_dict, not corrupted into {'a','c','m','e'}."""
        config = ACLConfig.from_dict({
            "roles": {"staff": {"client_scopes": "acme"}},
            "user_roles": {"bob": "staff"},
            "enforcement_on": True,
        })
        mask = config.allow_mask("bob")
        assert mask == {"acme"}
        assert config.can_see("bob", client_scope="acme", doc_class=None) is True

    def test_non_list_non_string_client_scopes_rejected(self):
        """A non-list, non-string client_scopes is a parse error."""
        config = ACLConfig.from_dict({
            "roles": {"staff": {"client_scopes": 123}},
            "enforcement_on": True,
        })
        assert config.parse_error is True

    def test_list_of_non_strings_rejected(self):
        config = ACLConfig.from_dict({
            "roles": {"staff": {"client_scopes": ["acme", 123]}},
            "enforcement_on": True,
        })
        assert config.parse_error is True


# ---------------------------------------------------------------------------
# AS4 — dangling user_roles references warned
# ---------------------------------------------------------------------------

class TestAS4DanglingRoles:
    def test_dangling_role_warning_logged(self, caplog):
        """A user assigned to a non-existent role logs a warning."""
        import logging
        with caplog.at_level(logging.WARNING, logger="access_scoping"):
            ACLConfig.from_dict({
                "roles": {"staff": {"client_scopes": ["acme"]}},
                "user_roles": {"bob": "nonexistent_role"},
                "enforcement_on": True,
            })
        assert any("nonexistent_role" in r.getMessage() for r in caplog.records)

    def test_dangling_role_user_is_deny_all(self):
        """A user with a non-existent role gets deny-all (empty mask)."""
        config = ACLConfig(
            roles={"staff": {"client_scopes": ["acme"]}},
            user_roles={"bob": "nonexistent"},
            enforcement_on=True,
        )
        assert config.allow_mask("bob") == set()


# ---------------------------------------------------------------------------
# AS5 — empty deny entry warning
# ---------------------------------------------------------------------------

class TestAS5EmptyDenyEntry:
    def test_empty_deny_entry_warned(self, caplog):
        """A deny entry with no fields (wildcard deny) logs a warning."""
        import logging
        with caplog.at_level(logging.WARNING, logger="access_scoping"):
            ACLConfig.from_dict({
                "roles": {"staff": {"client_scopes": ["acme"]}},
                "user_roles": {"bob": "staff"},
                "deny_lists": {"bob": [{}]},
                "enforcement_on": True,
            })
        assert any("denies ALL" in r.getMessage() for r in caplog.records)

    def test_empty_deny_entry_denies_everything(self):
        """An empty deny entry {} matches all records (wildcard)."""
        config = ACLConfig(
            roles={"staff": {"client_scopes": ["acme"]}},
            user_roles={"bob": "staff"},
            deny_lists={"bob": [{}]},
            enforcement_on=True,
        )
        assert config.is_denied("bob", client_scope="acme", doc_class=None) is True
        assert config.is_denied("bob", client_scope="beta", doc_class=None) is True


# ---------------------------------------------------------------------------
# AS6 — can_see docstring matches code order
# ---------------------------------------------------------------------------

class TestAS6DocstringOrder:
    def test_docstring_lists_correct_precedence(self):
        """The can_see docstring lists: 1. Deny, 2. Practice-internal,
        3. Allow mask, 4. Default (matching the actual code order)."""
        doc = ACLConfig.can_see.__doc__ or ""
        assert "1. Deny" in doc or "1. Deny list" in doc
        assert "2. Practice-internal" in doc or "2. Practice" in doc
        assert "3. Allow mask" in doc or "3. Allow" in doc


# ---------------------------------------------------------------------------
# AS8 — nodes without id are skipped
# ---------------------------------------------------------------------------

class TestAS8NodeIdless:
    def test_node_without_id_is_dropped(self):
        """A node without an 'id' field is dropped (not given id='')."""
        config = ACLConfig(
            roles={"staff": {"client_scopes": ["acme"]}},
            user_roles={"bob": "staff"},
            enforcement_on=True,
        )
        nodes = [
            {"id": "n1", "attributes": {"client_scope": "acme"}},
            {"attributes": {"client_scope": "acme"}},  # no id
        ]
        edges: list = []
        filtered, _, dropped = filter_graph_neighbours(nodes, edges, config, "bob")
        # The id-less node is dropped, the id'd one is kept.
        assert len(filtered) == 1
        assert filtered[0]["id"] == "n1"
        assert dropped == 1

    def test_idless_nodes_dont_share_empty_id(self):
        """Multiple id-less nodes don't all become visible if one would be."""
        config = ACLConfig(
            roles={"staff": {"client_scopes": ["acme"]}},
            user_roles={"bob": "staff"},
            enforcement_on=True,
        )
        nodes = [
            {"attributes": {"client_scope": "beta"}},  # no id, wrong scope
            {"attributes": {"client_scope": "acme"}},  # no id, right scope
        ]
        edges: list = []
        filtered, _, _ = filter_graph_neighbours(nodes, edges, config, "bob")
        # Both are dropped (no id) — the right-scope one does NOT make the
        # wrong-scope one visible via shared "".
        assert len(filtered) == 0


# ---------------------------------------------------------------------------
# AS2/AS7 — cross-file fixes (documented, tested at the access_scoping layer)
# ---------------------------------------------------------------------------

class TestAS2AS7CrossFile:
    def test_filter_records_by_access_is_fail_closed_by_design(self):
        """filter_records_by_access applies can_see() to every record —
        the AS2 fail-closed fix in provider_retrieval.py wraps this in a
        try/except that clears results on crash. This test verifies the
        filter itself correctly drops denied records."""
        config = ACLConfig(
            roles={"staff": {"client_scopes": ["acme"]}},
            user_roles={"bob": "staff"},
            enforcement_on=True,
        )

        class FakeRecord:
            def __init__(self, cs):
                self.client_scope = cs
                self.doc_class = None

        records = [FakeRecord("acme"), FakeRecord("beta")]
        visible, denied = filter_records_by_access(records, config, "bob")
        assert len(visible) == 1
        assert denied == 1

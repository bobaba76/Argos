"""Tests for the shared WHERE-clause builder (#245).

T1 (parity): the builder composes the expected SQL fragments and params
in the exact canonical order for a representative matrix of filter combos.
T2 (structure guard): store_retrieval.py contains no f-string interpolation
of a column-name-style identifier into SQL.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

_plugin_dir = Path(__file__).resolve().parent.parent
if str(_plugin_dir) not in sys.path:
    sys.path.insert(0, str(_plugin_dir))

from store_retrieval import _build_memory_where


class TestBuilderParity:
    """T1: the builder produces the expected SQL + params for each combo."""

    def test_no_filters(self):
        """Bare: status + temporal(current) only."""
        sql, params = _build_memory_where()
        assert "COALESCE(status, 'active') = 'active'" in sql
        assert "AND valid_to IS NULL" in sql
        assert params == []

    def test_user_scope_only(self):
        sql, params = _build_memory_where(user_scope="alice", now="2026-01-01")
        assert "AND (user_scope IS NULL OR user_scope = ?)" in sql
        assert params == ["2026-01-01", "alice"]

    def test_project_only(self):
        sql, params = _build_memory_where(project_id="proj1", now="2026-01-01")
        assert "AND (project_id IS NULL OR project_id = ?)" in sql
        assert params == ["2026-01-01", "proj1"]

    def test_namespace_only(self):
        sql, params = _build_memory_where(namespace="document", now="2026-01-01")
        assert "AND namespace = ?" in sql
        assert params == ["2026-01-01", "document"]

    def test_client_scope_only(self):
        sql, params = _build_memory_where(client_scope="client_a", now="2026-01-01")
        assert "AND (client_scope IS NULL OR client_scope = ?)" in sql
        assert params == ["2026-01-01", "client_a"]

    def test_category_only(self):
        sql, params = _build_memory_where(category="fact", now="2026-01-01")
        assert "AND category = ?" in sql
        assert params == ["2026-01-01", "fact"]

    def test_excluded_only(self):
        sql, params = _build_memory_where(
            excluded={"junk", "spam"}, now="2026-01-01",
        )
        assert "AND LOWER(category) NOT IN (?, ?)" in sql
        assert params == ["2026-01-01", "junk", "spam"]

    def test_tier_only(self):
        sql, params = _build_memory_where(tier="active", now="2026-01-01")
        assert "AND COALESCE(tier, 'active') = 'active'" in sql
        assert params == ["2026-01-01"]

    def test_as_of_temporal(self):
        sql, params = _build_memory_where(as_of="2026-06-01", now="2026-01-01")
        assert "AND valid_from <= ? AND (valid_to IS NULL OR valid_to > ?)" in sql
        # as_of takes precedence for expiry_ref: temporal(2) + expiry(1)
        assert params == ["2026-06-01", "2026-06-01", "2026-06-01"]

    def test_include_closed(self):
        sql, params = _build_memory_where(
            include_closed=True, now="2026-01-01",
        )
        # No temporal gate, but expiry still present
        assert "valid_to IS NULL" not in sql
        assert "valid_from" not in sql
        assert "AND (expires_at IS NULL OR expires_at > ?)" in sql
        assert params == ["2026-01-01"]

    def test_include_expired(self):
        sql, params = _build_memory_where(include_expired=True, now="2026-01-01")
        assert "expires_at" not in sql
        assert params == []

    def test_extra_sql(self):
        sql, params = _build_memory_where(
            extra_sql="AND embedding IS NOT NULL", now="2026-01-01",
        )
        assert "AND embedding IS NOT NULL" in sql
        assert params == ["2026-01-01"]

    def test_all_filters_together(self):
        """Full matrix: every filter active, check canonical param order."""
        sql, params = _build_memory_where(
            user_scope="alice",
            project_id="proj1",
            namespace="document",
            client_scope="client_a",
            category="fact",
            excluded={"junk", "spam"},
            tier="active",
            as_of="2026-06-01",
            extra_sql="AND embedding IS NOT NULL",
        )
        # Canonical param order: temporal(as_of x2), expiry(as_of),
        # user_scope, project, namespace, client_scope, category, excluded...
        assert params == [
            "2026-06-01", "2026-06-01",  # temporal (as_of)
            "2026-06-01",                 # expiry (as_of takes precedence)
            "alice",                      # user_scope
            "proj1",                      # project_id
            "document",                   # namespace
            "client_a",                   # client_scope
            "fact",                       # category
            "junk", "spam",               # excluded (lowered)
        ]
        # Verify clause order in SQL
        pos_status = sql.index("COALESCE(status")
        pos_temporal = sql.index("valid_from")
        pos_expiry = sql.index("expires_at")
        pos_scope = sql.index("user_scope")
        pos_project = sql.index("project_id")
        pos_namespace = sql.index("namespace")
        pos_client = sql.index("client_scope")
        pos_category = sql.index("category")
        pos_tier = sql.index("tier")
        pos_extra = sql.index("embedding")
        assert pos_status < pos_temporal < pos_expiry < pos_scope
        assert pos_scope < pos_project < pos_namespace < pos_client
        assert pos_client < pos_category < pos_tier < pos_extra

    def test_expiry_ref_uses_now_when_no_as_of(self):
        """When as_of is None, expiry uses *now*."""
        sql, params = _build_memory_where(now="2026-03-15")
        assert params == ["2026-03-15"]

    def test_no_expiry_when_no_now_and_no_as_of(self):
        """When neither now nor as_of is provided, no expiry clause."""
        sql, params = _build_memory_where(user_scope="alice")
        assert "expires_at" not in sql
        assert params == ["alice"]


class TestStructureGuard:
    """T2: no f-string interpolation of column-name identifiers into SQL."""

    def test_no_column_name_interpolation(self):
        """Assert that store_retrieval.py does not interpolate bare column
        names (like project_id, namespace) via f-strings into SQL.

        The builder uses fixed SQL tokens only. This grep-based guard
        catches regressions where someone adds ``f"...{col_name}..."``
        inside a WHERE/SELECT string.
        """
        src = (_plugin_dir / "store_retrieval.py").read_text(encoding="utf-8")
        # Find all f-string expressions that look like column identifiers
        # being interpolated into SQL. We look for patterns like:
        #   f"...{some_var}..."  where the surrounding text contains SQL
        #   keywords (WHERE, SELECT, AND, OR) and the var name looks like
        #   a column name (snake_case identifier).
        #
        # The builder itself uses f-strings for placeholder generation
        # (e.g. f"AND LOWER(category) NOT IN ({placeholders})") — that's
        # fine because {placeholders} expands to ?, ?, ... not a column
        # name. We specifically check for interpolation of identifiers
        # that match known column names.
        column_names = [
            "project_id", "namespace", "client_scope", "category",
            "user_scope", "valid_from", "valid_to", "expires_at",
            "status", "tier", "content", "memory_id", "embedding",
        ]
        # Strip the builder function itself (it's the safe path).
        builder_start = src.index("def _build_memory_where")
        builder_end = src.index("\n\nclass StoreRetrievalMixin")
        non_builder_src = src[:builder_start] + src[builder_end:]
        # Also strip comments and docstrings.
        cleaned = re.sub(r'""".*?"""', '', non_builder_src, flags=re.DOTALL)
        cleaned = re.sub(r"'''.*?'''", '', cleaned, flags=re.DOTALL)
        cleaned = re.sub(r'#.*$', '', cleaned, flags=re.MULTILINE)
        # Check for f-string interpolation of column names into SQL.
        # Pattern: f"...SQL...{col_name}..." where col_name is a known column.
        for col in column_names:
            # Look for f-string interpolation of the column name itself
            # (not as a string literal, not as a parameter).
            # E.g. f"WHERE {project_id} = ?" is bad; f"WHERE project_id = ?" is fine.
            pattern = rf'f"[^"]*\{{\s*{col}\s*\}}[^"]*"'
            matches = re.findall(pattern, cleaned)
            assert not matches, (
                f"store_retrieval.py: f-string interpolation of column name "
                f"'{col}' found in SQL context ({len(matches)} matches). "
                f"Use fixed SQL tokens in the builder instead."
            )

    def test_builder_uses_fixed_tokens(self):
        """The builder function uses only fixed SQL string tokens for
        column names — no interpolation."""
        src = (_plugin_dir / "store_retrieval.py").read_text(encoding="utf-8")
        builder_start = src.index("def _build_memory_where")
        builder_end = src.index("\n\nclass StoreRetrievalMixin")
        builder_src = src[builder_start:builder_end]
        # The only f-string in the builder should be the excluded-placeholders
        # one (which generates ?, ?, ... — not a column name).
        f_strings = re.findall(r'f"[^"]*"', builder_src)
        for fs in f_strings:
            # The placeholders f-string is the only allowed one.
            assert "placeholders" in fs or "LOWER(category)" in fs, (
                f"Builder contains unexpected f-string: {fs}. "
                f"Only placeholder generation is allowed."
            )

    def test_both_search_methods_call_builder(self):
        """Both _text_search_raw and _vector_search_raw call
        _build_memory_where — no inline clause construction."""
        src = (_plugin_dir / "store_retrieval.py").read_text(encoding="utf-8")
        # Extract _text_search_raw method body.
        text_start = src.index("def _text_search_raw")
        text_end = src.index("\n    def _vector_search_raw")
        text_body = src[text_start:text_end]
        assert "_build_memory_where" in text_body, (
            "_text_search_raw must call _build_memory_where"
        )
        # Extract _vector_search_raw method body.
        vec_start = src.index("def _vector_search_raw")
        vec_end = src.index("\n    def find_semantic_duplicate")
        vec_body = src[vec_start:vec_end]
        assert "_build_memory_where" in vec_body, (
            "_vector_search_raw must call _build_memory_where"
        )
        # No inline clause variables should remain in either method.
        for var in ("temporal_clause", "expiry_clause", "project_clause",
                     "namespace_clause", "client_scope_clause",
                     "category_clause", "tier_clause"):
            assert var not in text_body, (
                f"_text_search_raw still has inline '{var}' — use the builder"
            )
            assert var not in vec_body, (
                f"_vector_search_raw still has inline '{var}' — use the builder"
            )

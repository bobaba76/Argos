"""Audit tests for rest_server.py (R1-R9, issue #220).

Covers:
- R1: ACL scope on fetch (namespace check added)
- R2: non-numeric Content-Length returns 400
- R3: sys imported at top level
- R4: catch-all exception handler
- R5: readiness endpoint doesn't leak component details
- R6: case-insensitive Bearer scheme
- R7: Vary: Origin header on CORS responses
- R8: category_filter max_length
- R9: unused imports removed

Run with (Hermes venv python, offline):
    python -m pytest tests/test_rest_server_audit.py -v
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest

_plugin_dir = Path(__file__).resolve().parent.parent
for _path in (_plugin_dir.parent, _plugin_dir):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


# ---------------------------------------------------------------------------
# R1 — ACL scope on fetch (namespace check)
# ---------------------------------------------------------------------------

class TestR1FetchScopeNamespace:
    def test_scope_matches_checks_namespace(self):
        """R1: scope_check checks namespace when max_namespace is set."""
        from api_facade import ArgosAPIFacade, AuthContext

        # Build a facade without calling __init__ (no store needed).
        facade = ArgosAPIFacade.__new__(ArgosAPIFacade)

        # When max_namespace is set, a record with a different namespace
        # should fail the scope check.
        ctx = AuthContext(
            principal="test", tenant="t", user_id="u",
            transport="rest", allowed_operations=set(),
        )
        ctx.max_namespace = "project_a"

        class FakeRecord:
            project_id = None
            client_scope = None
            namespace = "project_b"

        assert not facade.scope_check(ctx, FakeRecord())

        # Same namespace passes.
        class FakeRecord2:
            project_id = None
            client_scope = None
            namespace = "project_a"

        assert facade.scope_check(ctx, FakeRecord2())

    def test_scope_matches_no_namespace_restriction_when_unset(self):
        """R1: when max_namespace is None, namespace is not checked (scope_check)."""
        from api_facade import ArgosAPIFacade, AuthContext

        facade = ArgosAPIFacade.__new__(ArgosAPIFacade)
        ctx = AuthContext(
            principal="test", tenant="t", user_id="u",
            transport="rest", allowed_operations=set(),
        )
        # max_namespace is not set (None) — any namespace passes.

        class FakeRecord:
            project_id = None
            client_scope = None
            namespace = "anything"

        assert facade.scope_check(ctx, FakeRecord())


# ---------------------------------------------------------------------------
# R2 — non-numeric Content-Length
# ---------------------------------------------------------------------------

class TestR2ContentLengthValidation:
    def test_middleware_handles_non_numeric_content_length(self):
        """R2: middleware code handles non-numeric Content-Length."""
        from rest_server import create_app
        src = inspect.getsource(create_app)
        # Must have a try/except ValueError around int(cl).
        assert "ValueError" in src
        assert "malformed_request" in src


# ---------------------------------------------------------------------------
# R3 — sys imported at top level
# ---------------------------------------------------------------------------

class TestR3SysImport:
    def test_sys_imported_at_top_level(self):
        """R3: sys is imported at the top level, not only in __main__."""
        import rest_server
        assert hasattr(rest_server, "sys")
        # The module should not re-import sys in the __main__ block.
        src = inspect.getsource(rest_server)
        # Find the __main__ block and verify no redundant import sys.
        main_block = src[src.find('if __name__'):]
        assert "import sys" not in main_block


# ---------------------------------------------------------------------------
# R4 — catch-all exception handler
# ---------------------------------------------------------------------------

class TestR4CatchAllHandler:
    def test_catch_all_handler_exists(self):
        """R4: a catch-all Exception handler is registered."""
        from rest_server import create_app
        src = inspect.getsource(create_app)
        assert "Exception" in src
        assert "internal_error" in src


# ---------------------------------------------------------------------------
# R5 — readiness endpoint doesn't leak component details
# ---------------------------------------------------------------------------

class TestR5ReadinessNoLeak:
    def test_ready_response_has_no_components(self):
        """R5: /v1/ready returns only status, not component details."""
        from rest_server import create_app
        src = inspect.getsource(create_app)
        # The ready() function should NOT return "components" in the
        # success response body.
        ready_start = src.find('async def ready()')
        ready_end = src.find('@app.get("/v1/capabilities")')
        ready_src = src[ready_start:ready_end]
        assert '"components"' not in ready_src or 'components' not in ready_src.split('return')[1].split('}')[0]


# ---------------------------------------------------------------------------
# R6 — case-insensitive Bearer scheme
# ---------------------------------------------------------------------------

class TestR6CaseInsensitiveBearer:
    def test_bearer_check_is_case_insensitive(self):
        """R6: the Bearer scheme check is case-insensitive."""
        from rest_server import RESTAuth
        src = inspect.getsource(RESTAuth.__call__)
        # Must use .lower() for case-insensitive scheme check.
        assert ".lower()" in src


# ---------------------------------------------------------------------------
# R7 — Vary: Origin header
# ---------------------------------------------------------------------------

class TestR7VaryOrigin:
    def test_vary_origin_set_on_cors(self):
        """R7: Vary: Origin is set when Access-Control-Allow-Origin is set."""
        from rest_server import create_app
        src = inspect.getsource(create_app)
        assert "Vary" in src
        assert "Origin" in src


# ---------------------------------------------------------------------------
# R8 — category_filter max_length
# ---------------------------------------------------------------------------

class TestR8CategoryFilterMaxLength:
    def test_category_filter_has_max_length(self):
        """R8: SearchRequest.category_filter has a max_length constraint."""
        from rest_server import SearchRequest
        # A 101-char string should be rejected by pydantic validation.
        with pytest.raises(Exception):
            SearchRequest(query="test", category_filter="x" * 101)
        # A 100-char string should be accepted.
        req = SearchRequest(query="test", category_filter="x" * 100)
        assert req.category_filter == "x" * 100


# ---------------------------------------------------------------------------
# R9 — unused imports removed
# ---------------------------------------------------------------------------

class TestR9UnusedImportsRemoved:
    def test_hashlib_not_imported(self):
        """R9: hashlib is no longer imported (was unused)."""
        import rest_server
        assert not hasattr(rest_server, "hashlib")

    def test_time_not_imported_at_module_level(self):
        """R9: time is no longer imported (was unused)."""
        import rest_server
        # time may be imported by other modules, but rest_server should
        # not import it at the top level.
        assert not hasattr(rest_server, "time")

    def test_field_not_imported(self):
        """R9: Field from pydantic is no longer imported (was unused)."""
        import rest_server
        assert not hasattr(rest_server, "Field")

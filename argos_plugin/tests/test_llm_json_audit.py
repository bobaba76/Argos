"""Audit tests for shared LLM JSON parsing (R4/D2/D6, issue #204).

Verifies that the shared ``llm_json`` module handles all three cases
consistently:
1. Pure JSON
2. Code-fenced JSON (```json ... ```)
3. Prose-wrapped JSON ("Here is the JSON: {...} Done.")

Also verifies that reviewer.py, distillation.py, and rollup.py all
delegate to the shared utility instead of duplicating the logic.

Run with (Hermes venv python, offline):
    python -m pytest tests/test_llm_json_audit.py -v
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
# Shared utility — parse_llm_json_object
# ---------------------------------------------------------------------------

class TestParseJsonObject:
    def test_pure_json(self):
        from llm_json import parse_llm_json_object
        result = parse_llm_json_object('{"decision": "approve"}')
        assert result == {"decision": "approve"}

    def test_fenced_json(self):
        from llm_json import parse_llm_json_object
        result = parse_llm_json_object('```json\n{"decision": "approve"}\n```')
        assert result == {"decision": "approve"}

    def test_fenced_json_no_language(self):
        from llm_json import parse_llm_json_object
        result = parse_llm_json_object('```\n{"decision": "approve"}\n```')
        assert result == {"decision": "approve"}

    def test_prose_wrapped_json(self):
        from llm_json import parse_llm_json_object
        result = parse_llm_json_object('Here is my review:\n{"decision": "approve"}\nLet me know.')
        assert result == {"decision": "approve"}

    def test_empty_string(self):
        from llm_json import parse_llm_json_object
        assert parse_llm_json_object("") is None
        assert parse_llm_json_object("   ") is None

    def test_non_dict_returns_none(self):
        from llm_json import parse_llm_json_object
        assert parse_llm_json_object("[1, 2, 3]") is None
        assert parse_llm_json_object('"hello"') is None
        assert parse_llm_json_object("42") is None

    def test_invalid_json_returns_none(self):
        from llm_json import parse_llm_json_object
        assert parse_llm_json_object("not json at all") is None
        assert parse_llm_json_object("{broken") is None

    def test_prose_with_no_brace_returns_none(self):
        from llm_json import parse_llm_json_object
        assert parse_llm_json_object("Here is some text with no JSON") is None


# ---------------------------------------------------------------------------
# Shared utility — parse_llm_json_array
# ---------------------------------------------------------------------------

class TestParseJsonArray:
    def test_pure_json_array(self):
        from llm_json import parse_llm_json_array
        result = parse_llm_json_array('[{"content": "a"}, {"content": "b"}]')
        assert result == [{"content": "a"}, {"content": "b"}]

    def test_fenced_json_array(self):
        from llm_json import parse_llm_json_array
        result = parse_llm_json_array('```json\n[{"content": "a"}]\n```')
        assert result == [{"content": "a"}]

    def test_prose_wrapped_json_array(self):
        from llm_json import parse_llm_json_array
        result = parse_llm_json_array('Here are the proposals:\n[{"content": "a"}]\nDone.')
        assert result == [{"content": "a"}]

    def test_empty_string(self):
        from llm_json import parse_llm_json_array
        assert parse_llm_json_array("") is None
        assert parse_llm_json_array("   ") is None

    def test_non_list_returns_none(self):
        from llm_json import parse_llm_json_array
        assert parse_llm_json_array('{"key": "value"}') is None

    def test_invalid_json_returns_none(self):
        from llm_json import parse_llm_json_array
        assert parse_llm_json_array("not json") is None


# ---------------------------------------------------------------------------
# Shared utility — parse_llm_response (response object extraction)
# ---------------------------------------------------------------------------

class TestParseLlmResponse:
    def test_extracts_from_response_object(self):
        from llm_json import parse_llm_response

        class FakeResponse:
            choices = [type("C", (), {"message": type("M", (), {"content": '{"key": "val"}'})()})()]

        result = parse_llm_response(FakeResponse())
        assert result == {"key": "val"}

    def test_returns_none_on_bad_response(self):
        from llm_json import parse_llm_response
        assert parse_llm_response(None) is None
        assert parse_llm_response({}) is None


# ---------------------------------------------------------------------------
# Integration — reviewer uses shared utility
# ---------------------------------------------------------------------------

class TestReviewerUsesSharedUtility:
    def test_reviewer_delegates_to_llm_json(self):
        """R4: reviewer._parse_json_response uses the shared llm_json module."""
        from reviewer import _parse_json_response
        src = inspect.getsource(_parse_json_response)
        assert "llm_json" in src
        assert "parse_llm_response" in src


# ---------------------------------------------------------------------------
# Integration — distillation uses shared utility
# ---------------------------------------------------------------------------

class TestDistillationUsesSharedUtility:
    def test_distillation_delegates_to_llm_json(self):
        """D2: distillation._parse_distill_response uses the shared llm_json module."""
        from distillation import _parse_distill_response
        src = inspect.getsource(_parse_distill_response)
        assert "llm_json" in src
        assert "parse_llm_response" in src


# ---------------------------------------------------------------------------
# Integration — rollup uses shared utility
# ---------------------------------------------------------------------------

class TestRollupUsesSharedUtility:
    def test_rollup_delegates_to_llm_json(self):
        """D6: rollup._parse_rollup_response uses the shared llm_json module."""
        from rollup import _parse_rollup_response
        src = inspect.getsource(_parse_rollup_response)
        assert "llm_json" in src
        assert "parse_llm_json_array" in src

"""Config schema audit regression tests (#234, CS1-CS9).

Covers the pure helpers in ``config_validation.py`` and the declarative
``config_schema.py`` field kinds/descriptions. Deterministic, no DuckDB, no
LLM calls.

``config_schema.py`` imports ``plugins.memory.config_schema`` from the Hermes
host runtime. When that package is not importable (fresh clone), a minimal
dataclass stand-in with the same field names is installed so the schema can
be loaded and inspected.
"""
from __future__ import annotations

import dataclasses
import importlib
import importlib.util
import sys
import types
from pathlib import Path

import pytest

_plugin_dir = Path(__file__).resolve().parent.parent
if str(_plugin_dir) not in sys.path:
    sys.path.insert(0, str(_plugin_dir))

from config_validation import (  # noqa: E402
    deployment_consistency_error,
    parse_positive_int_map,
    parse_role_words,
    parse_string_map,
    safe_storage_name,
    storage_name_error,
)


# -- CS1: storage names ------------------------------------------------------

class TestStorageNameValidation:
    @pytest.mark.parametrize("value", [
        "hybrid_memory.duckdb",
        "hybrid_memory_kuzu",
        "cells/alpha.duckdb",
        "tenant-1_gateway.duckdb",
    ])
    def test_safe_names_accepted(self, value):
        assert storage_name_error(value) is None
        assert safe_storage_name(value, "database_filename", "x") == value

    @pytest.mark.parametrize("value,reason", [
        ("../../sensitive.db", "path traversal"),
        ("a/../b.duckdb", "path traversal"),
        ("a\\..\\b.duckdb", "path traversal"),
        ("/etc/passwd", "absolute"),
        ("\\Windows\\x.duckdb", "absolute"),
        ("C:\\Users\\x.duckdb", "drive letter"),
        ("\\\\nas\\share\\x.duckdb", "UNC"),
        ("//nas/share/x.duckdb", "UNC"),
        ("", "empty"),
        ("   ", "empty"),
        ("x\x00y.duckdb", "NUL"),
    ])
    def test_unsafe_names_rejected(self, value, reason):
        error = storage_name_error(value)
        assert error is not None
        assert reason in error

    def test_safe_storage_name_falls_back_to_default(self, caplog):
        with caplog.at_level("WARNING"):
            out = safe_storage_name("../../escape.duckdb", "database_filename", "hybrid_memory.duckdb")
        assert out == "hybrid_memory.duckdb"
        assert "database_filename" in caplog.text
        assert "path traversal" in caplog.text

    def test_safe_storage_name_handles_non_strings(self):
        assert safe_storage_name(None, "graph_dirname", "hybrid_memory_kuzu") == "hybrid_memory_kuzu"
        assert safe_storage_name(123, "graph_dirname", "hybrid_memory_kuzu") == "123"

    def test_memory_service_validator_delegates(self):
        """The tenant path validator and the provider share one rule set."""
        pytest.importorskip("duckdb")
        from memory_service import _validate_tenant_path
        with pytest.raises(ValueError, match="path traversal"):
            _validate_tenant_path("../escape.duckdb", "t", "database_filename")
        with pytest.raises(ValueError, match="must be relative"):
            _validate_tenant_path("/abs/escape.duckdb", "t", "database_filename")
        _validate_tenant_path("ok.duckdb", "t", "database_filename")


# -- CS4: entity_aliases -----------------------------------------------------

class TestEntityAliasesParsing:
    def test_valid_map(self):
        raw = '{"my role": "Entity-A", "the project": " Project-X "}'
        assert parse_string_map(raw, "entity_aliases") == {
            "my role": "Entity-A", "the project": "Project-X",
        }

    def test_empty_and_none(self):
        assert parse_string_map("", "entity_aliases") == {}
        assert parse_string_map(None, "entity_aliases") == {}

    def test_invalid_json_ignored(self, caplog):
        with caplog.at_level("WARNING"):
            assert parse_string_map("{not json", "entity_aliases") == {}
        assert "entity_aliases" in caplog.text

    def test_non_object_ignored(self):
        assert parse_string_map('["a", "b"]', "entity_aliases") == {}
        assert parse_string_map('"str"', "entity_aliases") == {}

    def test_non_string_values_dropped(self):
        raw = '{"ok": "Entity", "num": 5, "nested": {"a": 1}, "blank": "  ", "": "x"}'
        assert parse_string_map(raw, "entity_aliases") == {"ok": "Entity"}

    def test_dict_input_passthrough(self):
        assert parse_string_map({"a": "B"}, "entity_aliases") == {"a": "B"}


# -- CS5: expiry_ttl_days ----------------------------------------------------

_TTL_DEFAULT = {"context_note": 30, "event": 180, "goal": 180}


class TestExpiryTtlParsing:
    def test_default_string_parses(self):
        raw = '{"context_note":30,"event":180,"goal":180}'
        assert parse_positive_int_map(raw, "expiry_ttl_days", _TTL_DEFAULT) == _TTL_DEFAULT

    def test_invalid_json_uses_default(self, caplog):
        with caplog.at_level("WARNING"):
            out = parse_positive_int_map("{oops", "expiry_ttl_days", _TTL_DEFAULT)
        assert out == _TTL_DEFAULT
        assert out is not _TTL_DEFAULT
        assert "expiry_ttl_days" in caplog.text

    def test_non_object_uses_default(self):
        assert parse_positive_int_map("[1,2]", "expiry_ttl_days", _TTL_DEFAULT) == _TTL_DEFAULT
        assert parse_positive_int_map("", "expiry_ttl_days", _TTL_DEFAULT) == _TTL_DEFAULT
        assert parse_positive_int_map(None, "expiry_ttl_days", _TTL_DEFAULT) == _TTL_DEFAULT

    def test_negative_zero_and_non_numeric_dropped(self):
        raw = '{"event": -5, "goal": 0, "note": "30", "flag": true, "ok": 7.9, "huge": 99999}'
        assert parse_positive_int_map(raw, "expiry_ttl_days", _TTL_DEFAULT) == {"ok": 7}

    def test_dict_input(self):
        assert parse_positive_int_map({"x": 3}, "expiry_ttl_days", _TTL_DEFAULT) == {"x": 3}


# -- CS8: role_words ---------------------------------------------------------

class TestRoleWordsParsing:
    def test_json_array_canonical(self):
        assert parse_role_words('["Wife", "therapist", " boss "]') == ["wife", "therapist", "boss"]

    def test_comma_separated_accepted(self):
        assert parse_role_words("wife, therapist,boss") == ["wife", "therapist", "boss"]

    def test_dedupes_and_drops_blanks(self):
        assert parse_role_words('["wife", "WIFE", "", 3, null]') == ["wife"]
        assert parse_role_words("a,,b, ,a") == ["a", "b"]

    def test_bad_json_and_non_array(self):
        assert parse_role_words("[not json") == []
        assert parse_role_words('{"a": 1}') == []

    def test_empty(self):
        assert parse_role_words("") == []
        assert parse_role_words(None) == []
        assert parse_role_words([]) == []

    def test_list_input(self):
        assert parse_role_words(["Boss", "wife"]) == ["boss", "wife"]


# -- CS6: deployment_mode vs data_residency ----------------------------------

class TestDeploymentConsistency:
    @pytest.mark.parametrize("mode,residency", [
        ("cloud_pilot", "cloud"),
        ("local_sku", "local"),
    ])
    def test_consistent(self, mode, residency):
        assert deployment_consistency_error(mode, residency) is None

    @pytest.mark.parametrize("mode,residency", [
        ("cloud_pilot", "local"),
        ("local_sku", "cloud"),
    ])
    def test_inconsistent(self, mode, residency):
        error = deployment_consistency_error(mode, residency)
        assert error is not None
        assert mode in error and residency in error

    def test_unknown_values(self):
        assert "deployment_mode" in deployment_consistency_error("hybrid", "cloud")
        assert "data_residency" in deployment_consistency_error("cloud_pilot", "mars")
        assert deployment_consistency_error(None, None) is not None


# -- Schema (CS3, CS7, CS9) --------------------------------------------------

def _install_schema_stub() -> None:
    """Stand-in for ``plugins.memory.config_schema`` when Hermes is absent."""
    @dataclasses.dataclass(frozen=True)
    class ProviderFieldOption:
        value: str
        label: str

    @dataclasses.dataclass(frozen=True)
    class ProviderField:
        key: str
        label: str
        kind: str
        default: str = ""
        description: str = ""
        info: str = ""
        inline: bool = False
        group: str = ""
        options: tuple = ()

    @dataclasses.dataclass(frozen=True)
    class ProviderConfigSchema:
        name: str
        label: str
        storage: str
        fields: tuple = ()

    mod = types.ModuleType("plugins.memory.config_schema")
    mod.ProviderConfigSchema = ProviderConfigSchema
    mod.ProviderField = ProviderField
    mod.ProviderFieldOption = ProviderFieldOption
    for name in ("KIND_TEXT", "KIND_SELECT", "KIND_SECRET", "KIND_BOOL", "KIND_NUMBER"):
        setattr(mod, name, name.lower().replace("kind_", ""))
    mod.STORAGE_FLAT_JSON = "flat_json"
    plugins = types.ModuleType("plugins")
    plugins.__path__ = []  # type: ignore[attr-defined]
    memory = types.ModuleType("plugins.memory")
    memory.__path__ = []  # type: ignore[attr-defined]
    memory.config_schema = mod
    plugins.memory = memory
    sys.modules["plugins"] = plugins
    sys.modules["plugins.memory"] = memory
    sys.modules["plugins.memory.config_schema"] = mod


@pytest.fixture(scope="module")
def schema():
    stubbed = importlib.util.find_spec("plugins") is None
    if stubbed:
        _install_schema_stub()
    try:
        sys.modules.pop("config_schema", None)
        mod = importlib.import_module("config_schema")
        yield mod
    finally:
        sys.modules.pop("config_schema", None)
        if stubbed:
            for name in ("plugins.memory.config_schema", "plugins.memory", "plugins"):
                sys.modules.pop(name, None)


_INTEGER_TEXT_FIELDS = (
    "duplicate_semantic_max_pairs",
    "reranker_top_n",
    "context_window_size",
    "context_max_chars",
    "chain_max_versions",
    "chain_max_inject",
    "chain_unfold_top_k",
)


class TestConfigSchema:
    def test_integer_fields_are_numbers(self, schema):
        """CS3: integer-valued fields use KIND_NUMBER so the UI can enforce numeric input."""
        fields = {f.key: f for f in schema.CONFIG_SCHEMA.fields}
        for key in _INTEGER_TEXT_FIELDS:
            assert fields[key].kind == schema.KIND_NUMBER, key
            int(fields[key].default)  # default still parses as an integer

    def test_local_only_description_lists_rollup(self, schema):
        """CS9: rollup is egress-gated, so the description must say so."""
        fields = {f.key: f for f in schema.CONFIG_SCHEMA.fields}
        assert "rollup" in fields["local_only"].description

    def test_role_words_documents_canonical_format(self, schema):
        fields = {f.key: f for f in schema.CONFIG_SCHEMA.fields}
        assert "JSON array" in fields["role_words"].description

    def test_storage_names_document_path_rules(self, schema):
        fields = {f.key: f for f in schema.CONFIG_SCHEMA.fields}
        for key in ("database_filename", "graph_dirname"):
            assert "relative to HERMES_HOME" in fields[key].info, key

    def test_field_indentation_is_uniform(self):
        """CS7: every ProviderField( opener sits at the 8-space field level."""
        src = (_plugin_dir / "config_schema.py").read_text(encoding="utf-8").splitlines()
        openers = [line for line in src if line.strip() == "ProviderField("]
        assert openers, "no ProviderField openers found"
        assert all(line == "        ProviderField(" for line in openers), [
            line for line in openers if line != "        ProviderField("
        ]


# -- CS9: rollup is egress-gated ---------------------------------------------

def test_rollup_is_an_egress_site():
    from egress import SITES
    kinds = {site["kind"] for site in SITES}
    assert "memory_rollup" in kinds

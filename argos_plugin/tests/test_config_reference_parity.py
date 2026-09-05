"""Schema/reference parity test for #310.

Asserts that every key in config_schema.py and config_model.py appears
in CONFIG_REFERENCE.md, and every key in CONFIG_REFERENCE.md exists in
the schema or model (or is marked JSON-only). This prevents the
reference and schema from drifting silently.

Acceptance: CONFIG_REFERENCE covers 100% of the config surface; no
undocumented key; schema/reference parity test passes.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

_plugin_dir = Path(__file__).resolve().parent.parent
_repo_root = _plugin_dir.parent
if str(_plugin_dir) not in sys.path:
    sys.path.insert(0, str(_plugin_dir))


class TestConfigReferenceParity:
    """#310: every config key appears in CONFIG_REFERENCE.md and vice versa."""

    @staticmethod
    def _schema_keys() -> set:
        schema_path = _plugin_dir / "config_schema.py"
        text = schema_path.read_text(encoding="utf-8")
        return set(re.findall(r'key="(\w+)"', text))

    @staticmethod
    def _model_keys() -> set:
        from config_model import MemoryConfig
        return set(MemoryConfig.model_fields.keys())

    @staticmethod
    def _reference_keys() -> set:
        ref_path = _repo_root / "CONFIG_REFERENCE.md"
        text = ref_path.read_text(encoding="utf-8")
        # Keys are in backticks in the first column of table rows.
        # Filter to only those that look like config keys (snake_case,
        # not default values like "true", "10", etc.).
        raw = set(re.findall(r'\| `(\w+)`', text))
        # Filter out known non-key values that appear in backticks.
        non_keys = {
            "true", "false", "auto", "off", "always", "full", "minimal",
            "shared_service", "direct", "cloud", "local", "cloud_pilot",
            "local_sku", "hybrid_memory.duckdb", "hybrid_memory_kuzu",
        }
        # Also filter out pure numbers.
        return {k for k in raw if k not in non_keys and not k.isdigit()}

    def test_every_schema_key_in_reference(self):
        """Every key declared in config_schema.py appears in CONFIG_REFERENCE.md."""
        schema_keys = self._schema_keys()
        ref_keys = self._reference_keys()
        missing = schema_keys - ref_keys
        assert not missing, (
            f"config_schema.py keys missing from CONFIG_REFERENCE.md: {sorted(missing)}. "
            f"Add each key to CONFIG_REFERENCE.md with a description."
        )

    def test_every_model_key_in_reference(self):
        """Every key in MemoryConfig appears in CONFIG_REFERENCE.md."""
        model_keys = self._model_keys()
        ref_keys = self._reference_keys()
        missing = model_keys - ref_keys
        assert not missing, (
            f"MemoryConfig keys missing from CONFIG_REFERENCE.md: {sorted(missing)}. "
            f"Add each key to CONFIG_REFERENCE.md (mark as *(JSON only)* if not in the UI schema)."
        )

    def test_every_reference_key_in_config(self):
        """Every key in CONFIG_REFERENCE.md exists in the schema or model."""
        ref_keys = self._reference_keys()
        all_config = self._schema_keys() | self._model_keys()
        extra = ref_keys - all_config
        assert not extra, (
            f"CONFIG_REFERENCE.md keys not in schema or model: {sorted(extra)}. "
            f"Remove stale entries or add the key to config_schema.py / config_model.py."
        )

    def test_json_only_keys_are_model_only(self):
        """Keys marked *(JSON only)* in the reference should be model-only
        (not in the UI schema). This catches stale JSON-only markers."""
        ref_path = _repo_root / "CONFIG_REFERENCE.md"
        text = ref_path.read_text(encoding="utf-8")
        # Find keys marked JSON only: `key_name` *(JSON only)*
        json_only = set(re.findall(r'`(\w+)`\s*\*\(JSON only\)\*', text))
        schema_keys = self._schema_keys()
        model_keys = self._model_keys()
        # JSON-only keys should be in the model but NOT in the schema.
        in_schema = json_only & schema_keys
        assert not in_schema, (
            f"Keys marked *(JSON only)* but actually in config_schema.py: "
            f"{sorted(in_schema)}. Remove the *(JSON only)* marker — they're UI-exposed."
        )
        # They should be in the model.
        not_in_model = json_only - model_keys
        assert not not_in_model, (
            f"Keys marked *(JSON only)* but not in MemoryConfig: "
            f"{sorted(not_in_model)}. Add them to config_model.py or remove from reference."
        )

    def test_reference_covers_full_config_surface(self):
        """The reference covers 100% of the config surface (schema + model)."""
        all_config = self._schema_keys() | self._model_keys()
        ref_keys = self._reference_keys()
        coverage = len(all_config & ref_keys) / len(all_config)
        assert coverage == 1.0, (
            f"CONFIG_REFERENCE.md covers {coverage:.1%} of the config surface. "
            f"Missing: {sorted(all_config - ref_keys)}"
        )

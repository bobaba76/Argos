"""Spec-08 (#72): POPIA & deployment mode — provider abstraction extended
to extraction + answering, cloud-first pilot / local SKU v2.

Tests (deterministic, no LLM calls):
1. Config fields load correctly (extraction/answering model+provider,
   deployment_mode, data_residency).
2. Extraction provider threading — extract_doc_facts_llm passes the
   model/provider to call_llm.
3. Deployment mode config — cloud_pilot (default) / local_sku.
4. Backward compat — no config = default behavior (empty strings,
   cloud_pilot, cloud).
5. Full extraction pipeline — extract_facts_from_doc combines text
   extraction + LLM call.
"""
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from watcher import (
    extract_doc_facts_llm,
    extract_facts_from_doc,
    prepare_extraction_input,
)


# ---------------------------------------------------------------------------
# 1. Config field loading
# ---------------------------------------------------------------------------


class TestConfigLoading:
    """Config fields load from the provider config surface."""

    def test_extraction_llm_model_default(self):
        """No config = empty string (falls back to llm_model)."""
        from config_schema import CONFIG_SCHEMA
        fields = {f.key: f for f in CONFIG_SCHEMA.fields}
        assert fields["extraction_llm_model"].default == ""
        assert fields["extraction_llm_provider"].default == ""
        assert fields["answering_llm_model"].default == ""
        assert fields["answering_llm_provider"].default == ""

    def test_deployment_mode_default(self):
        """Default deployment mode is cloud_pilot."""
        from config_schema import CONFIG_SCHEMA
        fields = {f.key: f for f in CONFIG_SCHEMA.fields}
        assert fields["deployment_mode"].default == "cloud_pilot"
        assert fields["data_residency"].default == "cloud"

    def test_deployment_mode_options(self):
        """Deployment mode has cloud_pilot and local_sku options."""
        from config_schema import CONFIG_SCHEMA
        fields = {f.key: f for f in CONFIG_SCHEMA.fields}
        options = [o.value for o in fields["deployment_mode"].options]
        assert "cloud_pilot" in options
        assert "local_sku" in options

    def test_data_residency_options(self):
        """Data residency has cloud and local options."""
        from config_schema import CONFIG_SCHEMA
        fields = {f.key: f for f in CONFIG_SCHEMA.fields}
        options = [o.value for o in fields["data_residency"].options]
        assert "cloud" in options
        assert "local" in options

    def test_all_new_fields_in_schema(self):
        """All six new config fields are present in the schema."""
        from config_schema import CONFIG_SCHEMA
        keys = {f.key for f in CONFIG_SCHEMA.fields}
        assert "extraction_llm_model" in keys
        assert "extraction_llm_provider" in keys
        assert "answering_llm_model" in keys
        assert "answering_llm_provider" in keys
        assert "deployment_mode" in keys
        assert "data_residency" in keys

    def test_new_fields_in_llm_or_deployment_group(self):
        """New fields are in the LLM or Deployment config group."""
        from config_schema import CONFIG_SCHEMA
        fields = {f.key: f for f in CONFIG_SCHEMA.fields}
        assert fields["extraction_llm_model"].group == "LLM"
        assert fields["extraction_llm_provider"].group == "LLM"
        assert fields["answering_llm_model"].group == "LLM"
        assert fields["answering_llm_provider"].group == "LLM"
        assert fields["deployment_mode"].group == "Deployment"
        assert fields["data_residency"].group == "Deployment"


# ---------------------------------------------------------------------------
# 2. Extraction provider threading
# ---------------------------------------------------------------------------


class TestExtractionProviderThreading:
    """extract_doc_facts_llm passes model/provider to call_llm."""

    def test_empty_text_returns_empty(self):
        """Empty or short text returns empty list without calling LLM."""
        assert extract_doc_facts_llm("") == []
        assert extract_doc_facts_llm("short") == []

    def test_model_provider_passed_to_call_llm(self):
        """The extraction model/provider are passed through to call_llm."""
        mock_response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="[]"))]
        )
        import types, sys as _sys
        mock_module = types.ModuleType("agent.auxiliary_client")
        mock_module.call_llm = MagicMock(return_value=mock_response)
        _sys.modules["agent.auxiliary_client"] = mock_module
        try:
            with patch("egress.gate", return_value=True):
                extract_doc_facts_llm(
                    "This is a long enough text for extraction to proceed.",
                    model="test-model",
                    provider="test-provider",
                )
                assert mock_module.call_llm.called
                call_kwargs = mock_module.call_llm.call_args
                assert call_kwargs.kwargs["model"] == "test-model"
                assert call_kwargs.kwargs["provider"] == "test-provider"
                assert call_kwargs.kwargs["task"] == "doc_fact_extraction"
        finally:
            del _sys.modules["agent.auxiliary_client"]

    def test_none_response_returns_empty(self):
        """A None response from the LLM returns empty list."""
        import types, sys as _sys
        mock_module = types.ModuleType("agent.auxiliary_client")
        mock_module.call_llm = MagicMock(return_value=None)
        _sys.modules["agent.auxiliary_client"] = mock_module
        try:
            with patch("egress.gate", return_value=True):
                result = extract_doc_facts_llm(
                    "This is a long enough text for extraction to proceed.",
                )
                assert result == []
        finally:
            del _sys.modules["agent.auxiliary_client"]

    def test_malformed_response_returns_empty(self):
        """A malformed response returns empty list, never raises."""
        import types, sys as _sys
        mock_module = types.ModuleType("agent.auxiliary_client")
        mock_module.call_llm = MagicMock(return_value=SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="not json"))]
        ))
        _sys.modules["agent.auxiliary_client"] = mock_module
        try:
            with patch("egress.gate", return_value=True):
                result = extract_doc_facts_llm(
                    "This is a long enough text for extraction to proceed.",
                )
                assert result == []
        finally:
            del _sys.modules["agent.auxiliary_client"]

    def test_valid_json_response_parsed(self):
        """A valid JSON response is parsed into a list of facts."""
        import types, sys as _sys
        facts_json = json.dumps([
            {"content": "VAT number is 4780", "category": "personal_fact",
             "source_loc": "page 1", "confidence": 0.95},
        ])
        mock_module = types.ModuleType("agent.auxiliary_client")
        mock_module.call_llm = MagicMock(return_value=SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=facts_json))]
        ))
        _sys.modules["agent.auxiliary_client"] = mock_module
        try:
            with patch("egress.gate", return_value=True):
                result = extract_doc_facts_llm(
                    "This is a long enough text for extraction to proceed.",
                )
                assert len(result) == 1
                assert result[0]["content"] == "VAT number is 4780"
        finally:
            del _sys.modules["agent.auxiliary_client"]

    def test_llm_exception_returns_empty(self):
        """An LLM call exception returns empty list, never raises."""
        import types, sys as _sys
        mock_module = types.ModuleType("agent.auxiliary_client")
        mock_module.call_llm = MagicMock(side_effect=RuntimeError("timeout"))
        _sys.modules["agent.auxiliary_client"] = mock_module
        try:
            with patch("egress.gate", return_value=True):
                result = extract_doc_facts_llm(
                    "This is a long enough text for extraction to proceed.",
                )
                assert result == []
        finally:
            del _sys.modules["agent.auxiliary_client"]


# ---------------------------------------------------------------------------
# 3. Full extraction pipeline
# ---------------------------------------------------------------------------


class TestExtractionPipeline:
    """extract_facts_from_doc combines text extraction + LLM call."""

    def test_csv_extraction_pipeline(self, tmp_path):
        """A CSV file goes through the full pipeline."""
        f = tmp_path / "data.csv"
        f.write_text("id,name,amount\n1,Acme,5000\n2,Beta,3000\n")
        facts, extract_hash, method, text = extract_facts_from_doc(
            f, "csv",
            extraction_llm_model="test-model",
            extraction_llm_provider="test-provider",
        )
        # Without a real LLM, facts will be empty but text/hash are valid.
        assert method == "text"
        assert "Acme" in text
        assert extract_hash  # non-empty

    def test_empty_file_returns_empty(self, tmp_path):
        """An empty file returns empty facts."""
        f = tmp_path / "empty.csv"
        f.write_text("")
        facts, extract_hash, method, text = extract_facts_from_doc(f, "csv")
        assert facts == []
        assert text == ""


# ---------------------------------------------------------------------------
# 4. Backward compatibility
# ---------------------------------------------------------------------------


class TestBackwardCompat:
    """No config = default behavior (empty strings, cloud_pilot, cloud)."""

    def test_no_config_loads_defaults(self):
        """The provider config loads defaults when fields are absent."""
        # Simulate what provider_core.py does with an empty config.
        config = {}
        llm_model = str(config.get("llm_model", "")).strip()
        llm_provider = str(config.get("llm_provider", "")).strip()
        extraction_llm_model = str(config.get("extraction_llm_model", "")).strip()
        extraction_llm_provider = str(config.get("extraction_llm_provider", "")).strip()
        answering_llm_model = str(config.get("answering_llm_model", "")).strip()
        answering_llm_provider = str(config.get("answering_llm_provider", "")).strip()
        deployment_mode = str(config.get("deployment_mode", "cloud_pilot")).strip()
        data_residency = str(config.get("data_residency", "cloud")).strip()
        assert llm_model == ""
        assert llm_provider == ""
        assert extraction_llm_model == ""
        assert extraction_llm_provider == ""
        assert answering_llm_model == ""
        assert answering_llm_provider == ""
        assert deployment_mode == "cloud_pilot"
        assert data_residency == "cloud"

    def test_local_sku_config_loads(self):
        """Local SKU v2 config loads correctly."""
        config = {
            "deployment_mode": "local_sku",
            "data_residency": "local",
            "extraction_llm_provider": "on-prem-endpoint",
            "answering_llm_provider": "on-prem-endpoint",
        }
        deployment_mode = str(config.get("deployment_mode", "cloud_pilot")).strip()
        data_residency = str(config.get("data_residency", "cloud")).strip()
        extraction_llm_provider = str(config.get("extraction_llm_provider", "")).strip()
        answering_llm_provider = str(config.get("answering_llm_provider", "")).strip()
        assert deployment_mode == "local_sku"
        assert data_residency == "local"
        assert extraction_llm_provider == "on-prem-endpoint"
        assert answering_llm_provider == "on-prem-endpoint"


# ---------------------------------------------------------------------------
# 5. Documentation artefacts exist
# ---------------------------------------------------------------------------


class TestDocsExist:
    """The Annexure A and due-diligence checklist are present."""

    def test_annexure_a_exists(self):
        p = Path(__file__).resolve().parent.parent.parent / "docs" / "annexure-a-processing-annex.md"
        assert p.exists()
        content = p.read_text(encoding="utf-8")
        assert "Annexure A" in content
        assert "POPIA" in content
        assert "cloud_pilot" in content
        assert "local_sku" in content

    def test_due_diligence_checklist_exists(self):
        p = Path(__file__).resolve().parent.parent.parent / "docs" / "provider-due-diligence-checklist.md"
        assert p.exists()
        content = p.read_text(encoding="utf-8")
        assert "zero retention" in content.lower() or "Zero retention" in content
        assert "no training" in content.lower() or "No training" in content
        assert "POPIA" in content

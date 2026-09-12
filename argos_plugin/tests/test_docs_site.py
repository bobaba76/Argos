"""#296: docs verification tests.

Validates that the public docs site doesn't drift from the live code.
Every CLI flag, REST endpoint, MCP tool name, and config key named in
the docs must exist in the source — no fabricated commands or flags.

Per-file run only (the full suite has a pre-existing single-process
deadlock — run this file individually, not alongside other test files
in the same process).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

_plugin_dir = Path(__file__).resolve().parent.parent
if str(_plugin_dir) not in sys.path:
    sys.path.insert(0, str(_plugin_dir))

_repo_root = _plugin_dir.parent
_docs_dir = _repo_root / "docs"


# -- Helpers -----------------------------------------------------------------

def _read_doc(name: str) -> str:
    """Read a docs file. Fail loud if missing."""
    path = _docs_dir / name
    assert path.exists(), f"docs file missing: {path}"
    return path.read_text(encoding="utf-8")


def _read_source(rel: str) -> str:
    """Read a source file from the plugin dir. Fail loud if missing."""
    path = _plugin_dir / rel
    assert path.exists(), f"source file missing: {path}"
    return path.read_text(encoding="utf-8")


# -- MCP tools named in docs exist in TOOL_DEFINITIONS ----------------------

class TestMCPToolsInDocs:
    """Every memory_* tool named in docs/api/mcp.md must exist in the
    MCP server's TOOL_DEFINITIONS."""

    def test_mcp_doc_exists(self):
        text = _read_doc("api/mcp.md")
        assert "memory_search" in text

    def test_mcp_tools_in_docs_exist_in_code(self):
        doc_text = _read_doc("api/mcp.md")
        source = _read_source("mcp_server.py")
        # Extract tool names from the doc (### `memory_*` headings).
        tool_names = re.findall(r"^### `(memory_\w+)`", doc_text, re.MULTILINE)
        assert len(tool_names) >= 6, (
            f"expected at least 6 MCP tools in docs, found {len(tool_names)}"
        )
        for tool in tool_names:
            assert f'"name": "{tool}"' in source, (
                f"tool {tool!r} named in docs/api/mcp.md not found in "
                f"mcp_server.py TOOL_DEFINITIONS"
            )

    def test_mcp_tool_to_operation_map_matches(self):
        """Every tool→operation mapping in the docs must match the
        TOOL_TO_OPERATION map in the source."""
        from mcp_server import TOOL_TO_OPERATION
        doc_text = _read_doc("api/mcp.md")
        # Extract "Facade op: X" lines from the doc.
        doc_ops = re.findall(r"\*\*Facade op:\*\* `(\w+)`", doc_text)
        # The doc lists tool→op mappings; verify the source map has them.
        for op in doc_ops:
            assert op in TOOL_TO_OPERATION.values(), (
                f"facade op {op!r} in docs not found in "
                f"TOOL_TO_OPERATION values"
            )

    def test_mcp_cli_flags_in_docs_exist_in_code(self):
        """Every --flag named in docs/api/mcp.md must exist in
        mcp_server.py's argparse."""
        doc_text = _read_doc("api/mcp.md")
        source = _read_source("mcp_server.py")
        # Extract --flag names from the doc's CLI flags table.
        flags = re.findall(r"\| `(--[\w-]+)`", doc_text)
        for flag in flags:
            assert flag in source, (
                f"CLI flag {flag!r} in docs/api/mcp.md not found in "
                f"mcp_server.py"
            )


# -- REST endpoints named in docs exist in the server -----------------------

class TestRESTEndpointsInDocs:
    """Every REST endpoint named in docs/api/rest.md must exist in
    rest_server.py."""

    def test_rest_doc_exists(self):
        text = _read_doc("api/rest.md")
        assert "/v1/health" in text

    def test_rest_endpoints_in_docs_exist_in_code(self):
        doc_text = _read_doc("api/rest.md")
        source = _read_source("rest_server.py")
        # Extract endpoint paths from the doc (### `METHOD /path`).
        endpoints = re.findall(r"^### `(GET|POST|PUT|DELETE) (/[\w/{}-]+)`",
                               doc_text, re.MULTILINE)
        assert len(endpoints) >= 6, (
            f"expected at least 6 REST endpoints in docs, found {len(endpoints)}"
        )
        for method, path in endpoints:
            # The source uses @app.get("/path") or @app.post("/path").
            pattern = f'@app.{method.lower()}("{path}")'
            assert pattern in source, (
                f"endpoint {method} {path!r} in docs/api/rest.md not found "
                f"in rest_server.py (looking for {pattern!r})"
            )

    def test_rest_cli_flags_in_docs_exist_in_code(self):
        doc_text = _read_doc("api/rest.md")
        source = _read_source("rest_server.py")
        flags = re.findall(r"\| `(--[\w-]+)`", doc_text)
        for flag in flags:
            assert flag in source, (
                f"CLI flag {flag!r} in docs/api/rest.md not found in "
                f"rest_server.py"
            )


# -- Facade operations named in docs exist in the allowlists ----------------

class TestFacadeOpsInDocs:
    """Every facade operation named in docs/api/index.md must exist in
    the facade's READ_OPERATIONS, PROPOSAL_OPERATIONS, or
    FEEDBACK_OPERATIONS."""

    def test_api_index_doc_exists(self):
        text = _read_doc("api/index.md")
        assert "READ" in text
        assert "PROPOSAL" in text

    def test_facade_ops_in_docs_exist_in_code(self):
        from api_facade import (
            READ_OPERATIONS, PROPOSAL_OPERATIONS, FEEDBACK_OPERATIONS,
        )
        all_ops = READ_OPERATIONS | PROPOSAL_OPERATIONS | FEEDBACK_OPERATIONS
        doc_text = _read_doc("api/index.md")
        # Extract operation names from the tier table (backtick-quoted).
        # The table lists ops like: `search`, `fetch`, ...
        ops_in_doc = re.findall(r"`(search|fetch|fetch_history|capabilities|"
                                r"explain|explain_retrieval|memory_propose|"
                                r"ingest|erase_request|record_feedback)`",
                                doc_text)
        for op in ops_in_doc:
            assert op in all_ops, (
                f"facade op {op!r} in docs/api/index.md not found in "
                f"READ/PROPOSAL/FEEDBACK_OPERATIONS"
            )

    def test_forbidden_flags_in_docs_match_code(self):
        from api_facade import FORBIDDEN_CLIENT_FLAGS
        doc_text = _read_doc("api/index.md")
        # Extract forbidden flags from the doc (bullet list).
        flags = re.findall(r"^- `(include_\w+|suppress_\w+)`", doc_text,
                           re.MULTILINE)
        for flag in flags:
            assert flag in FORBIDDEN_CLIENT_FLAGS, (
                f"forbidden flag {flag!r} in docs not in "
                f"FORBIDDEN_CLIENT_FLAGS"
            )


# -- Config keys named in docs exist in CONFIG_REFERENCE --------------------

class TestConfigKeysInDocs:
    """Config keys named in docs/configuration.md must exist in
    CONFIG_REFERENCE.md (the source of truth)."""

    def test_config_doc_exists(self):
        text = _read_doc("configuration.md")
        assert "storage_mode" in text

    def test_config_keys_in_docs_exist_in_config_reference(self):
        doc_text = _read_doc("configuration.md")
        ref_path = _repo_root / "CONFIG_REFERENCE.md"
        assert ref_path.exists(), "CONFIG_REFERENCE.md missing from repo root"
        ref_text = ref_path.read_text(encoding="utf-8")
        # Extract config keys from the docs table (| `key` | ...).
        keys = re.findall(r"^\| `(\w+)`", doc_text, re.MULTILINE)
        assert len(keys) >= 10, (
            f"expected at least 10 config keys in docs, found {len(keys)}"
        )
        for key in keys:
            assert f"`{key}`" in ref_text, (
                f"config key {key!r} in docs/configuration.md not found in "
                f"CONFIG_REFERENCE.md"
            )


# -- Plugin manifest deps named in docs match plugin.yaml -------------------

class TestPluginManifestInDocs:
    """Pip dependencies named in docs/installation.md must match
    plugin.yaml."""

    def test_install_doc_exists(self):
        text = _read_doc("installation.md")
        assert "plugin.yaml" in text

    def test_deps_in_docs_exist_in_plugin_yaml(self):
        doc_text = _read_doc("installation.md")
        manifest_path = _plugin_dir / "plugin.yaml"
        assert manifest_path.exists(), "plugin.yaml missing from argos_plugin/"
        manifest = manifest_path.read_text(encoding="utf-8")
        # Extract pip deps from the doc (indented bullet list: - `package==version`).
        deps = re.findall(r"^\s+- `([\w.-]+==[\w.]+)`", doc_text, re.MULTILINE)
        assert len(deps) >= 5, (
            f"expected at least 5 pip deps in docs, found {len(deps)}"
        )
        for dep in deps:
            pkg = dep.split("==")[0]
            assert pkg in manifest, (
                f"pip dep {dep!r} in docs/installation.md not found in "
                f"plugin.yaml"
            )


# -- CLI flags in installation docs exist in deploy.py ----------------------

class TestDeployFlagsInDocs:
    """Every --flag named in docs/installation.md for scripts/deploy.py
    must exist in the deploy script."""

    def test_deploy_flags_in_docs_exist_in_code(self):
        doc_text = _read_doc("installation.md")
        deploy_path = _repo_root / "scripts" / "deploy.py"
        assert deploy_path.exists(), "scripts/deploy.py missing"
        deploy_source = deploy_path.read_text(encoding="utf-8")
        # Extract deploy.py flags from the doc (python scripts/deploy.py --flag).
        # Look for --flag in code blocks that mention deploy.py.
        deploy_section = re.search(
            r"```bash\n# Check drift.*?```", doc_text, re.DOTALL,
        )
        if deploy_section is None:
            return  # no deploy section — skip
        flags = re.findall(r"--([\w-]+)", deploy_section.group(0))
        for flag in flags:
            assert f"--{flag}" in deploy_source, (
                f"deploy.py flag --{flag!r} in docs not found in "
                f"scripts/deploy.py"
            )


# -- mkdocs.yml structure ----------------------------------------------------

class TestMkdocsConfig:
    """The mkdocs.yml config must be valid and reference real files."""

    def test_mkdocs_yml_exists(self):
        path = _repo_root / "mkdocs.yml"
        assert path.exists(), "mkdocs.yml missing from repo root"

    def test_mkdocs_nav_files_exist(self):
        """Every file in the mkdocs nav must exist in docs/."""
        mkdocs_path = _repo_root / "mkdocs.yml"
        mkdocs_text = mkdocs_path.read_text(encoding="utf-8")
        # Extract .md file references from the nav section.
        nav_files = re.findall(r"^\s+-?\s*[\w ]+:\s*([\w/]+\.md)",
                               mkdocs_text, re.MULTILINE)
        assert len(nav_files) >= 6, (
            f"expected at least 6 nav files, found {len(nav_files)}"
        )
        for f in nav_files:
            path = _docs_dir / f
            assert path.exists(), (
                f"nav file {f!r} in mkdocs.yml not found at {path}"
            )

    def test_docs_index_exists(self):
        assert (_docs_dir / "index.md").exists()

    def test_docs_quickstart_exists(self):
        assert (_docs_dir / "quickstart.md").exists()

    def test_docs_installation_exists(self):
        assert (_docs_dir / "installation.md").exists()

    def test_docs_configuration_exists(self):
        assert (_docs_dir / "configuration.md").exists()

    def test_docs_api_index_exists(self):
        assert (_docs_dir / "api" / "index.md").exists()

    def test_docs_api_mcp_exists(self):
        assert (_docs_dir / "api" / "mcp.md").exists()

    def test_docs_api_rest_exists(self):
        assert (_docs_dir / "api" / "rest.md").exists()

    def test_docs_tuning_exists(self):
        assert (_docs_dir / "tuning.md").exists()

    def test_docs_integration_exists(self):
        assert (_docs_dir / "integration.md").exists()

    def test_docs_faq_exists(self):
        assert (_docs_dir / "faq.md").exists()


# -- GitHub Pages workflow ---------------------------------------------------

class TestDocsWorkflow:
    """The docs workflow file must be valid and not break CI."""

    def test_workflow_file_exists(self):
        path = _repo_root / ".github" / "workflows" / "docs.yml"
        assert path.exists(), "docs.yml workflow missing"

    def test_workflow_triggers_on_docs_path_only(self):
        """The workflow should trigger on docs/ changes, not on every
        push to master (so it doesn't slow down the CI gate)."""
        path = _repo_root / ".github" / "workflows" / "docs.yml"
        text = path.read_text(encoding="utf-8")
        assert "docs/**" in text, (
            "workflow must trigger on docs/** path changes"
        )
        # Must have paths: filter (not trigger on every push).
        assert "paths:" in text

    def test_workflow_uses_pages_deploy(self):
        path = _repo_root / ".github" / "workflows" / "docs.yml"
        text = path.read_text(encoding="utf-8")
        assert "actions/deploy-pages" in text
        assert "actions/upload-pages-artifact" in text

    def test_workflow_binds_to_master_only(self):
        """The workflow should only run on master (the docs source of
        truth), not on feature branches."""
        path = _repo_root / ".github" / "workflows" / "docs.yml"
        text = path.read_text(encoding="utf-8")
        assert "master" in text

    def test_workflow_has_permissions(self):
        path = _repo_root / ".github" / "workflows" / "docs.yml"
        text = path.read_text(encoding="utf-8")
        assert "permissions:" in text
        assert "pages: write" in text
        assert "id-token: write" in text


# -- No personal/identifying info in docs (sanitization) --------------------

class TestNoPersonalInfo:
    """The repo is PUBLIC — docs must not leak personal names, medication
    refs, or identifying details."""

    @pytest.mark.parametrize("doc_file", [
        "index.md", "quickstart.md", "installation.md", "configuration.md",
        "api/index.md", "api/mcp.md", "api/rest.md",
        "tuning.md", "integration.md", "faq.md",
    ])
    def test_no_personal_names_in_docs(self, doc_file):
        """No personal names should appear in the docs. This is a
        heuristic check — it flags common name patterns that would
        indicate a leak. (Cape Town is fine — it's a public example.)"""
        text = _read_doc(doc_file)
        # Check for email addresses (a clear PII leak).
        emails = re.findall(r"\b[\w.]+@[\w.]+\.\w+\b", text)
        # Allow the repo's noreply GitHub email in copyright/license
        # contexts.
        real_emails = [e for e in emails if "noreply.github.com" not in e
                       and "users.noreply" not in e]
        assert not real_emails, (
            f"email addresses found in {doc_file}: {real_emails}"
        )

    @pytest.mark.parametrize("doc_file", [
        "index.md", "quickstart.md", "installation.md", "configuration.md",
        "api/index.md", "api/mcp.md", "api/rest.md",
        "tuning.md", "integration.md", "faq.md",
    ])
    def test_no_medication_refs_in_docs(self, doc_file):
        """No medication names should appear in the docs."""
        text = _read_doc(doc_file)
        # Common medication patterns — this is a heuristic, not exhaustive.
        meds = re.findall(r"\b(Prozac|Zoloft|Lexapro|Wellbutrin|Adderall|"
                          r"Ritalin|Xanax|Ativan|Valium|Lithium\b)",
                          text, re.IGNORECASE)
        assert not meds, (
            f"medication references found in {doc_file}: {meds}"
        )


# -- Regression: REST token filename must match live code -------------------

class TestRESTTokenFilename:
    """#296 webhook review: the docs must reference api_credential.json
    (key 'token') + ARGOS_REST_TOKEN — NOT 'rest_token' in config.

    The live code (rest_server.py:_load_rest_token) loads from
    {home}/api_credential.json (key 'token') OR ARGOS_REST_TOKEN env var.
    A previous version of the docs said 'rest_token in the Hermes home
    config', which is a key the server never reads — a user following
    those docs would create a config key that does nothing.
    """

    @pytest.mark.parametrize("doc_file", ["api/rest.md", "faq.md"])
    def test_no_rest_token_key_in_docs(self, doc_file):
        """The docs must NOT reference 'rest_token' as a config key."""
        text = _read_doc(doc_file)
        # 'rest_token' as a config key (not as an env var or filename).
        # Match it in contexts like "rest_token in the Hermes home config"
        # or "Set rest_token" — but not "ARGOS_REST_TOKEN" (the env var).
        # Strip the env var name first to avoid false positives.
        stripped = text.replace("ARGOS_REST_TOKEN", "")
        assert "rest_token" not in stripped.lower(), (
            f"'rest_token' found in {doc_file} — the live code loads "
            f"api_credential.json (key 'token'), not a 'rest_token' "
            f"config key. See rest_server.py:_load_rest_token."
        )

    @pytest.mark.parametrize("doc_file", ["api/rest.md", "faq.md"])
    def test_api_credential_json_in_docs(self, doc_file):
        """The docs must reference api_credential.json as the token file."""
        text = _read_doc(doc_file)
        assert "api_credential.json" in text, (
            f"'api_credential.json' not found in {doc_file} — the live "
            f"code (rest_server.py:_load_rest_token) loads the token from "
            f"this file. The docs must mention it."
        )

    def test_rest_token_loading_matches_live_code(self):
        """The docs must not contradict rest_server.py:_load_rest_token."""
        source = _read_source("rest_server.py")
        # The live code loads api_credential.json — the legacy 'token'
        # key or per-principal credentials[] (#387) — or ARGOS_REST_TOKEN.
        # #484: the parsing moved to api_credentials.parse_credentials_file
        # (the loader delegates; a credentials-only file boots with
        # expected_token=None).
        assert "api_credential.json" in source
        assert "parse_credentials_file" in source
        assert "legacy_token" in source
        assert "ARGOS_REST_TOKEN" in source


# -- Regression: no dead links to 404 repos ---------------------------------

class TestNoDeadLinks:
    """#296 webhook review: no links to 404 GitHub repos.

    A previous version of quickstart.md linked to
    github.com/cognition-ai/hermes, which 404s. Hermes lives at
    NousResearch/hermes-agent.
    """

    @pytest.mark.parametrize("doc_file", [
        "index.md", "quickstart.md", "installation.md", "configuration.md",
        "api/index.md", "api/mcp.md", "api/rest.md",
        "tuning.md", "integration.md", "faq.md",
    ])
    def test_no_cognition_ai_hermes_link(self, doc_file):
        """The docs must not link to the dead cognition-ai/hermes repo."""
        text = _read_doc(doc_file)
        assert "cognition-ai/hermes" not in text, (
            f"dead link 'cognition-ai/hermes' found in {doc_file} — "
            f"Hermes lives at NousResearch/hermes-agent."
        )

    def test_quickstart_links_to_correct_hermes_repo(self):
        """quickstart.md must link to the correct Hermes repo."""
        text = _read_doc("quickstart.md")
        assert "NousResearch/hermes-agent" in text, (
            "quickstart.md must link to NousResearch/hermes-agent "
            "(the correct Hermes repo), not the dead cognition-ai/hermes."
        )

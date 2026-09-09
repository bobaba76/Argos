"""Pytest conftest — make the plugin importable as 'argos' without a live
Hermes runtime, and keep the import state hermetic across tests.

Why the stubs (issue #51): ``argos_plugin/__init__.py`` imports
``agent.memory_provider`` / ``tools.registry`` at module level — those
packages only exist inside a running Hermes installation.  The suite's
LLM-path tests historically stubbed them by inserting synthetic modules
into ``sys.modules`` with bare assignments that were never undone.  In a
single-process run a leaked stub (a plain ``ModuleType`` with no
``__path__``) can wedge a lazy import in the import machinery — observed
as a DuckDB execute path spinning forever in ``find_spec``/``_path_stat``,
which is what forced per-file runs as the only gate.

This conftest centralizes the hermetic stand-ins (they are installed once,
at conftest scope, before anything imports the plugin) and additionally
snapshots/restores the stub keys around every test, so a leak can never
cross a test boundary even if a future test mutates ``sys.modules``.

Why there is deliberately NO ``tests/__init__.py``: pytest derives the
conftest's module name by walking up while ``__init__.py`` files exist,
so with one present the conftest is imported as
``argos_plugin.tests.conftest`` — which imports the ``argos_plugin``
package first, and on a fresh clone (no Hermes runtime) that package
import fails before this file ever executes.  The absence of
``tests/__init__.py`` makes the conftest load flat, so the stubs below
are in place before any plugin import happens.
"""
from __future__ import annotations

import importlib
import importlib.util
import importlib.machinery
import json
import os
import sys
import types
from pathlib import Path

import pytest

_plugin_dir = Path(__file__).resolve().parent.parent

# Ensure the plugin dir is on sys.path so its modules are importable.
if str(_plugin_dir) not in sys.path:
    sys.path.insert(0, str(_plugin_dir))

# Hermetic override (#105 follow-up): the venvs that resolve the real
# Hermes runtime (an editable install, e.g. hermes-agent/venv via
# __editable__.hermes_agent-*.pth) would otherwise let unmocked LLM-path
# tests make real calls. Set ARGOS_HERMETIC_TESTS=1 to force the stubs
# even when the real runtime is importable — the deterministic, offline
# behavior the suite assumes. The gate script (run_tests_visible.ps1)
# sets it; deployed-plugin runs that want the real runtime omit it.
_FORCE_HERMETIC = os.environ.get("ARGOS_HERMETIC_TESTS", "").strip().lower() in {"1", "true", "yes"}


# ---------------------------------------------------------------------------
# Hermes-runtime stand-ins (conftest scope, issue #51)
# ---------------------------------------------------------------------------

def _safe_find_spec(name: str):
    """importlib.util.find_spec wrapper that returns None instead of
    raising ModuleNotFoundError when a parent package is absent (Linux CI
    raises where Windows returns None)."""
    try:
        return importlib.util.find_spec(name)
    except ModuleNotFoundError:
        return None


def _install_hermes_stubs_if_missing() -> None:
    """Install synthetic ``agent``/``tools`` modules when the real Hermes
    runtime is not importable, so ``argos_plugin`` (and the ``argos``
    alias below) import cleanly on a fresh clone.

    When the real packages ARE importable (deployed plugin, or a venv
    that resolves the hermes-agent runtime), this is a no-op and the real
    modules are used — UNLESS ``ARGOS_HERMETIC_TESTS=1``, which evicts any
    real ``agent``/``tools`` modules from ``sys.modules`` and installs the
    stubs unconditionally so the suite is deterministic and offline
    everywhere.  The stub shape mirrors what the suite's tests have
    always assumed: ``agent.memory_provider.MemoryProvider`` and
    ``tools.registry.tool_error`` exist; ``agent.auxiliary_client`` does
    not (the plugin guards that lazy import and degrades to deterministic
    no-LLM paths, which is exactly what the hermetic tests want).
    """
    if _FORCE_HERMETIC:
        for _name in list(sys.modules):
            if _name == "agent" or _name == "tools" or _name.startswith("agent.") or _name.startswith("tools."):
                del sys.modules[_name]
    if _FORCE_HERMETIC or _safe_find_spec("agent") is None:
        _mp = types.ModuleType("agent.memory_provider")

        class MemoryProvider:  # minimal stand-in
            pass

        _mp.MemoryProvider = MemoryProvider
        _agent = types.ModuleType("agent")
        _agent.__path__ = []  # mark as package so submodule imports work
        _agent.memory_provider = _mp
        sys.modules.setdefault("agent", _agent)
        sys.modules.setdefault("agent.memory_provider", _mp)
    if _FORCE_HERMETIC or _safe_find_spec("tools") is None:
        _tr = types.ModuleType("tools.registry")
        _tr.tool_error = lambda msg: json.dumps({"error": str(msg)})
        _tools = types.ModuleType("tools")
        _tools.__path__ = []  # mark as package so submodule imports work
        _tools.registry = _tr
        sys.modules.setdefault("tools", _tools)
        sys.modules.setdefault("tools.registry", _tr)

    # agent.auxiliary_client: the extractor's LLM fallback imports
    # ``from agent.auxiliary_client import call_llm`` lazily and degrades to
    # a no-LLM path when it is absent.  Tests that exercise the LLM path
    # patch ``agent.auxiliary_client.call_llm``, which requires the module
    # to exist in sys.modules.  Install a stub whose ``call_llm`` returns
    # None so the hermetic no-LLM behaviour is preserved for tests that do
    # not patch it, while patchable for tests that do.
    if _FORCE_HERMETIC or _safe_find_spec("agent.auxiliary_client") is None:
        _aux = types.ModuleType("agent.auxiliary_client")
        _aux.call_llm = lambda **kwargs: None
        _agent_mod = sys.modules.get("agent")
        if _agent_mod is not None:
            _agent_mod.auxiliary_client = _aux
        sys.modules.setdefault("agent.auxiliary_client", _aux)

    # plugins.memory.config_schema: ``argos_plugin/config_schema.py`` imports
    # from ``plugins.memory.config_schema`` at module level.  That package
    # only exists inside a running Hermes installation.  Install a hermetic
    # stub providing the same dataclass API so the plugin imports cleanly on
    # a fresh clone.  The real module is used when available (deployed
    # plugin) unless ARGOS_HERMETIC_TESTS=1 forces the stub.
    if _FORCE_HERMETIC or _safe_find_spec("plugins.memory.config_schema") is None:
        import dataclasses as _dc

        KIND_TEXT = "text"
        KIND_SELECT = "select"
        KIND_SECRET = "secret"
        KIND_BOOL = "bool"
        KIND_NUMBER = "number"
        STORAGE_FLAT_JSON = "flat_json"

        @_dc.dataclass
        class ProviderFieldOption:  # type: ignore[no-redef]
            value: str
            label: str = ""
            description: str = ""

        @_dc.dataclass
        class ProviderField:  # type: ignore[no-redef]
            key: str
            label: str = ""
            kind: str = KIND_TEXT
            default: str = ""
            description: str = ""
            placeholder: str = ""
            options: tuple = ()
            env_key: object = None
            aliases: tuple = ()
            env_fallbacks: tuple = ()
            inline: bool = False
            group: str = ""
            info: str = ""
            scope: str = "host"

        @_dc.dataclass
        class ProviderConfigSchema:  # type: ignore[no-redef]
            name: str
            label: str = ""
            storage: str = STORAGE_FLAT_JSON
            docs_url: str = ""
            config_file: object = None
            fields: tuple = ()

        _pcs = types.ModuleType("plugins.memory.config_schema")
        _pcs.ProviderConfigSchema = ProviderConfigSchema
        _pcs.ProviderField = ProviderField
        _pcs.ProviderFieldOption = ProviderFieldOption
        _pcs.KIND_TEXT = KIND_TEXT
        _pcs.KIND_SELECT = KIND_SELECT
        _pcs.KIND_SECRET = KIND_SECRET
        _pcs.KIND_BOOL = KIND_BOOL
        _pcs.KIND_NUMBER = KIND_NUMBER
        _pcs.STORAGE_FLAT_JSON = STORAGE_FLAT_JSON
        _plugins = sys.modules.get("plugins")
        if _plugins is None:
            _plugins = types.ModuleType("plugins")
            _plugins.__path__ = []  # type: ignore[attr-defined]
            _plugins.__spec__ = importlib.machinery.ModuleSpec("plugins", None)  # type: ignore[attr-defined]
            sys.modules.setdefault("plugins", _plugins)
        _plugins_mem = sys.modules.get("plugins.memory")
        if _plugins_mem is None:
            _plugins_mem = types.ModuleType("plugins.memory")
            _plugins_mem.__path__ = []  # type: ignore[attr-defined]
            _plugins_mem.__spec__ = importlib.machinery.ModuleSpec("plugins.memory", None)  # type: ignore[attr-defined]
            sys.modules.setdefault("plugins.memory", _plugins_mem)
        _plugins_mem.config_schema = _pcs
        sys.modules.setdefault("plugins.memory.config_schema", _pcs)


_install_hermes_stubs_if_missing()


# ---------------------------------------------------------------------------
# Import-state hygiene (issue #51)
# ---------------------------------------------------------------------------

# Hermes-runtime module keys that the suite may stub in sys.modules.
# Exact keys, plus every submodule under the agent./tools. packages
# (agent.auxiliary_client, agent.memory_provider, tools.registry, ...).
_STUB_KEYS_EXACT = frozenset(
    {"agent", "tools", "plugins", "service_client", "inbound_security", "argos.inbound_security"}
)
_STUB_KEYS_PREFIX = ("agent.", "tools.", "plugins.")


def _is_stub_key(name: str) -> bool:
    return name in _STUB_KEYS_EXACT or name.startswith(_STUB_KEYS_PREFIX)


def _stub_keys() -> list[str]:
    return [name for name in sys.modules if _is_stub_key(name)]


@pytest.fixture(autouse=True)
def _restore_import_state_after_test():
    """Keep sys.modules stub pollution from leaking across tests.

    Snapshots the stub keys (and ``sys.meta_path``) before each test and
    restores them afterwards. The conftest-scope stubs are part of the
    snapshot's baseline and survive untouched; only what a test body
    adds, replaces or removes is reverted.
    """
    saved = {name: sys.modules[name] for name in _stub_keys()}
    saved_meta_path = list(sys.meta_path)
    yield
    # Restore every key that was in the snapshot (including ones a test body
    # may have deleted from sys.modules) and remove any new stub keys a test
    # body added.
    for name in set(saved) | set(_stub_keys()):
        if name in saved:
            sys.modules[name] = saved[name]
        else:
            sys.modules.pop(name, None)
    if list(sys.meta_path) != saved_meta_path:
        sys.meta_path[:] = saved_meta_path


# ---------------------------------------------------------------------------
# 'argos' package alias
# ---------------------------------------------------------------------------

def _register_argos_alias() -> None:
    """Register the plugin directory as the 'argos' package.

    This lets ``from argos.service_client import ...`` resolve
    to ``argos_plugin/service_client.py`` without renaming the
    directory or installing the package.
    """
    if "argos" in sys.modules:
        return  # already registered (e.g. deployed plugin)
    # Check if a real 'argos' package is importable first.
    try:
        importlib.import_module("argos")
        return  # real package exists, don't shadow it
    except ImportError:
        pass
    # Create a synthetic package alias pointing at the plugin dir.
    spec = importlib.util.spec_from_file_location(
        "argos",
        str(_plugin_dir / "__init__.py"),
        submodule_search_locations=[str(_plugin_dir)],
    )
    if spec is None:
        return
    module = importlib.util.module_from_spec(spec)
    sys.modules["argos"] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        # If the full __init__ fails (e.g. optional deps missing),
        # register a bare namespace package so submodule imports
        # (service_client, store, etc.) still work.
        sys.modules["argos"] = types.ModuleType("argos")
        sys.modules["argos"].__path__ = [str(_plugin_dir)]


_register_argos_alias()


# ---------------------------------------------------------------------------
# #304: top-level module aliases for store modules
# ---------------------------------------------------------------------------
# store.py uses package-relative imports only (from .store_common import ...).
# Tests import these modules as top-level names (from store import ...).
# We eagerly import argos.store (which triggers argos.store_common etc.)
# and register top-level aliases so both import modes resolve to the same
# module object. This eliminates the dual-branch try/except ImportError
# fallback pattern in store.py.
_STORE_MODULES = (
    "store", "store_common", "store_core", "store_retrieval",
    "store_write", "store_maintenance", "store_state",
    "retriever", "value_extractor", "structural_loss",
)
for _name in _STORE_MODULES:
    _full = f"argos.{_name}"
    try:
        if _full not in sys.modules:
            importlib.import_module(_full)
    except ImportError:
        pass  # module not available — leave it unaliased
    if _full in sys.modules and _name not in sys.modules:
        sys.modules[_name] = sys.modules[_full]


# ---------------------------------------------------------------------------
# Deterministic embedder for hermetic, model-free tests (issues #90, #98)
# ---------------------------------------------------------------------------

class DeterministicEmbedder:
    """Hashing-trick embedder with no external model dependency.

    Real ``LocalEmbedder`` tests load a ~130MB sentence-transformers model
    and share the HF cache across concurrent pytest processes — the root
    cause of the ``test_alias_expansion_injects_with_similarity_gate``
    flake (#90) and a major contributor to the 15–20 minute suite wall
    time (#98). This stand-in produces a stable vector space via the
    signing hashing trick: each lowercased alphanumeric token is hashed
    to a dimension and adds +1/-1, then the vector is L2-normalized.

    Cosine similarity therefore reflects token overlap, which is exactly
    what retrieval/ranking tests need to assert ordering and gate
    behaviour without touching the real model. It is hermetic (no cache,
    no network, no torch), deterministic, and runs in microseconds.

    Duck-typed to match ``LocalEmbedder``: ``embed``, ``embed_batch``,
    ``is_available``, ``dimension``.
    """

    def __init__(self, dim: int = 128) -> None:
        self._dim = dim

    def _vec(self, text: str) -> list[float]:
        import hashlib
        import math
        v = [0.0] * self._dim
        for tok in _tokenize(text):
            h = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
            idx = int.from_bytes(h[:4], "little") % self._dim
            sign = 1.0 if (h[4] & 1) == 0 else -1.0
            v[idx] += sign
        norm = math.sqrt(sum(x * x for x in v))
        if norm > 0:
            v = [x / norm for x in v]
        return v

    def embed(self, text: str, *, is_query: bool = False) -> list[float]:
        if not text or not text.strip():
            return []
        return self._vec(text)

    def embed_batch(self, texts, *, is_query: bool = False) -> list[list[float]]:
        return [self.embed(t, is_query=is_query) for t in texts]

    @property
    def is_available(self) -> bool:
        return True

    @property
    def dimension(self) -> int:
        return self._dim


def _tokenize(text: str) -> list[str]:
    import re
    return [t for t in re.findall(r"[a-z0-9]+", (text or "").lower()) if t]


@pytest.fixture
def deterministic_embedder() -> "DeterministicEmbedder":
    """A fresh hermetic embedder for retrieval/ranking tests (issues #90, #98)."""
    return DeterministicEmbedder()


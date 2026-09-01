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
import json
import sys
import types
from pathlib import Path

import pytest

_plugin_dir = Path(__file__).resolve().parent.parent

# Ensure the plugin dir is on sys.path so its modules are importable.
if str(_plugin_dir) not in sys.path:
    sys.path.insert(0, str(_plugin_dir))


# ---------------------------------------------------------------------------
# Hermes-runtime stand-ins (conftest scope, issue #51)
# ---------------------------------------------------------------------------

def _install_hermes_stubs_if_missing() -> None:
    """Install synthetic ``agent``/``tools`` modules when the real Hermes
    runtime is not importable, so ``argos_plugin`` (and the ``argos``
    alias below) import cleanly on a fresh clone.

    When the real packages ARE importable (deployed plugin, or a venv
    that resolves the hermes-agent runtime), this is a no-op and the real
    modules are used.  The stub shape mirrors what the suite's tests have
    always assumed: ``agent.memory_provider.MemoryProvider`` and
    ``tools.registry.tool_error`` exist; ``agent.auxiliary_client`` does
    not (the plugin guards that lazy import and degrades to deterministic
    no-LLM paths, which is exactly what the hermetic tests want).
    """
    if importlib.util.find_spec("agent") is None:
        _mp = types.ModuleType("agent.memory_provider")

        class MemoryProvider:  # minimal stand-in
            pass

        _mp.MemoryProvider = MemoryProvider
        _agent = types.ModuleType("agent")
        _agent.memory_provider = _mp
        sys.modules.setdefault("agent", _agent)
        sys.modules.setdefault("agent.memory_provider", _mp)
    if importlib.util.find_spec("tools") is None:
        _tr = types.ModuleType("tools.registry")
        _tr.tool_error = lambda msg: json.dumps({"error": str(msg)})
        _tools = types.ModuleType("tools")
        _tools.registry = _tr
        sys.modules.setdefault("tools", _tools)
        sys.modules.setdefault("tools.registry", _tr)


_install_hermes_stubs_if_missing()


# ---------------------------------------------------------------------------
# Import-state hygiene (issue #51)
# ---------------------------------------------------------------------------

# Hermes-runtime module keys that the suite may stub in sys.modules.
# Exact keys, plus every submodule under the agent./tools. packages
# (agent.auxiliary_client, agent.memory_provider, tools.registry, ...).
_STUB_KEYS_EXACT = frozenset(
    {"agent", "tools", "service_client", "inbound_security", "argos.inbound_security"}
)
_STUB_KEYS_PREFIX = ("agent.", "tools.")


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
    for name in _stub_keys():
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


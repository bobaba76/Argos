"""Never-blind fallback for injection_min_score (shipped 2026-08-24).

Measured motivation (_probe_min_score_fn.py over the fullbank caches):
at floor=0.30, 7/500 LongMemEval questions had ALL their retrievable
evidence below the floor — the turn would inject nothing at all. The
fallback injects the unfiltered top-N when the floor suppresses every
candidate, so a weak-evidence turn still gets its best evidence instead
of silence. Random drops at the same rate kill ~3.4x more evidence, so
the floor itself stays; this only removes total-suppression outcomes.
"""
import json
import sys
import types


def _stub_agent_modules():
    """Same hermetic stubbing the main suite uses (no live hermes import)."""
    if "agent" not in sys.modules:
        sys.modules["agent"] = types.ModuleType("agent")
    if "agent.memory_provider" not in sys.modules:
        _mp = types.ModuleType("agent.memory_provider")

        class MemoryProvider:  # minimal stand-in
            pass

        _mp.MemoryProvider = MemoryProvider
        sys.modules["agent.memory_provider"] = _mp
    if "tools" not in sys.modules:
        sys.modules["tools"] = types.ModuleType("tools")
    if "tools.registry" not in sys.modules:
        _tr = types.ModuleType("tools.registry")
        _tr.tool_error = lambda msg: json.dumps({"error": str(msg)})
        sys.modules["tools.registry"] = _tr


_stub_agent_modules()

from store import MemoryRecord  # noqa: E402

try:
    import argos_plugin as _hmp  # noqa: E402
except ModuleNotFoundError:
    import argos as _hmp  # noqa: E402


class _FakeStore:
    """Just enough store for the prefetch path: no pending candidates."""

    def list_candidates(self, status=None, limit=None):
        return []


def _make_provider(records, min_score):
    provider = _hmp.ArgosProvider()
    provider._store = _FakeStore()
    provider._search_memories = lambda query, limit=96, **kw: list(records)
    provider._injection_min_score = min_score
    provider._max_injected = 96
    return provider


def _rec(memory_id, sim):
    return MemoryRecord(
        memory_id=memory_id,
        category="personal_fact",
        content=f"weak evidence item {memory_id}",
        similarity=sim,
    )


class TestInjectionFloorFallback:
    def test_all_below_floor_falls_back_to_top_n(self):
        """Every candidate below floor -> inject unfiltered top-8, not silence."""
        records = [_rec(i, 0.10 + i * 0.01) for i in range(12)]  # sims .10-.21
        provider = _make_provider(records, min_score=0.30)
        body = provider.prefetch("unladen swallow speed question")
        assert "## Recalled Memories" in body, "fallback must inject SOMETHING"
        # top-8 by rank survive; ranks 9-12 stay cut
        for i in range(8):
            assert f"weak evidence item {i}" in body
        for i in range(8, 12):
            assert f"weak evidence item {i}" not in body

    def test_partial_below_floor_filters_normally(self):
        """Some candidates clear the floor -> normal filtering, no fallback."""
        records = [
            _rec("strong", 0.85),
            _rec("mid", 0.28),
            _rec("low", 0.20),
        ]
        provider = _make_provider(records, min_score=0.30)
        body = provider.prefetch("unladen swallow speed question")
        assert "weak evidence item strong" in body
        assert "weak evidence item mid" not in body
        assert "weak evidence item low" not in body

    def test_floor_zero_is_untouched_passthrough(self):
        """Floor disabled -> identical behaviour to pre-fallback code."""
        records = [_rec(i, 0.05 + i * 0.01) for i in range(3)]
        provider = _make_provider(records, min_score=0.0)
        body = provider.prefetch("unladen swallow speed question")
        for i in range(3):
            assert f"weak evidence item {i}" in body

    def test_fallback_caps_at_eight(self):
        """Fallback slice never exceeds _INJECTION_FALLBACK_COUNT."""
        records = [_rec(i, 0.05) for i in range(30)]
        provider = _make_provider(records, min_score=0.30)
        body = provider.prefetch("unladen swallow speed question")
        assert sum(f"weak evidence item {i}" in body for i in range(30)) == \
            _hmp._INJECTION_FALLBACK_COUNT

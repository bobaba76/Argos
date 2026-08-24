"""Negative-memory kind (#4, shipped 2026-08-24).

Grounded no: a 'negative' memory ranks whenever the query topic matches,
so "is X ...?" questions get a definitive exclusion from the store
instead of a model guess. Explicit write path only (/neg command) — no
auto-extraction, keeping the noise floor at zero.
"""
import sys
import types

try:
    import argos_plugin
except ModuleNotFoundError:
    import argos as argos_plugin

from store import DuckDBMemoryStore, VALID_CATEGORIES


class _FakeSharedStore:
    def __init__(self):
        self.calls = []

    def remember(self, **kwargs):
        self.calls.append(kwargs)
        return types.SimpleNamespace(memory_id="neg-1")

    def close(self):
        pass


def _stub_service_client(fake):
    mod = types.ModuleType("service_client")
    mod.SharedMemoryStore = lambda *a, **k: fake
    sys.modules["service_client"] = mod


def test_category_registered():
    assert "negative" in VALID_CATEGORIES


def test_remember_round_trip(tmp_path):
    store = DuckDBMemoryStore(tmp_path / "neg.duckdb", user_id="test_user")
    rec = store.remember(
        category="negative", content="Alex does not drink coffee")
    assert rec is not None and rec.category == "negative"

    hits = store.search("does Alex drink coffee?", limit=10)
    assert any(r.content == "Alex does not drink coffee" for r in hits)


def test_neg_command_usage_message_when_empty():
    _stub_service_client(_FakeSharedStore())
    # re-import so the lazy service_client import sees the stub
    out = argos_plugin._handle_neg_command("")
    assert "Usage: /neg" in out


def test_neg_command_stores_claim():
    fake = _FakeSharedStore()
    _stub_service_client(fake)
    out = argos_plugin._handle_neg_command("Alex does not drink coffee")
    assert "Saved as [negative] memory" in out
    assert "[id: neg-1]" in out
    assert fake.calls == [
        {"category": "negative", "content": "Alex does not drink coffee"}]


def test_prefetch_labels_negative(tmp_path):
    store = DuckDBMemoryStore(tmp_path / "neg.duckdb", user_id="test_user")
    store.remember(category="negative", content="Alex does not drink coffee")
    provider = argos_plugin.ArgosProvider()
    provider._store = store
    provider._injection_min_score = 0.0

    provider._start_prefetch("does Alex drink coffee?")
    import time as _t
    for _ in range(200):
        if getattr(provider, "_prefetch_done", False):
            break
        _t.sleep(0.05)
    body = str(getattr(provider, "_prefetch_result", ""))
    assert "[negative]" in body
    assert "Alex does not drink coffee" in body
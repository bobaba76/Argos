"""Temporal Layers 2/3 — chronological injection + date-anchored re-rank.

P2B  (chronological_injection): on temporal/multi-hop turns, re-sort the
      injected top-k by created_at oldest-first so the model reads a
      timeline in order. Ordinary turns keep relevance order.
P2B2 (date_anchor_rerank): on temporal turns with an explicit date
      expression, re-sort by proximity to the resolved target date.
      Undated records sink to the end; ties stay stable.

Key invariants under test:
  * temporal query -> oldest first
  * ordinary query -> relevance order untouched
  * mixed-offset created_at values ("Z" / "+14:00" / "-05:00") are
    normalized to UTC before sorting (raw lexicographic mis-orders them)
  * classifier failure -> injection unchanged (best-effort semantics)
  * flag parsing: string/bool truthy, defaults OFF
"""
import re
import sys
import time as _time
from datetime import date
from pathlib import Path

import pytest

_plugin_dir = Path(__file__).resolve().parent.parent
if str(_plugin_dir) not in sys.path:
    sys.path.insert(0, str(_plugin_dir))

from date_anchor import resolve_target_date, reorder_by_date  # noqa: E402
from intent_router import is_temporal_or_multihop  # noqa: E402


def _provider(tmp_path, chrono=False, anchor=False):
    from store import DuckDBMemoryStore
    from graph import KuzuGraphStore
    try:
        import argos_plugin
    except ModuleNotFoundError:
        import argos as argos_plugin

    store = DuckDBMemoryStore(tmp_path / "temporal.duckdb", user_id="test_user")
    graph = KuzuGraphStore(tmp_path / "temporal_kuzu", user_id="test_user")

    provider = argos_plugin.ArgosProvider()
    provider._store = store
    provider._graph = graph
    provider._config = {
        "chronological_injection": str(chrono).lower(),
        "date_anchor_rerank": str(anchor).lower(),
    }
    provider._chronological_injection = chrono
    provider._date_anchor_rerank = anchor
    provider._injection_min_score = 0.0
    return provider, store


def _seed(store, contents):
    mids = []
    for c in contents:
        rec = store.remember("personal_fact", c)
        assert rec is not None, f"remember returned None for {c!r}"
        mids.append(rec.memory_id)
    return mids


def _backdate(store, memory_id, ts):
    store.connection.execute(
        "UPDATE memory_records SET created_at = ? WHERE memory_id = ?",
        [ts, memory_id],
    )


def _run_prefetch(provider, query, timeout_s=10.0):
    provider._start_prefetch(query)
    deadline = _time.monotonic() + timeout_s
    while _time.monotonic() < deadline:
        if getattr(provider, "_prefetch_done", False):
            break
        _time.sleep(0.05)
    text = str(getattr(provider, "_prefetch_result", ""))
    assert text, "prefetch returned empty injection"
    return text


_LINE_RE = re.compile(r"^- \[(\d{4}-\d{2}-\d{2})\] ?\[[a-z_]+\]\s*(.*)$")


def _injected(line_text):
    """Return [(content, date_prefix), ...] in injected order."""
    out = []
    for line in line_text.splitlines():
        m = _LINE_RE.match(line.strip())
        if m:
            out.append((m.group(2), m.group(1)))
    return out


def _idx(text, needle):
    i = text.find(needle)
    assert i >= 0, f"{needle!r} not in injected text:\n{text}"
    return i


# ---------------------------------------------------------------------------
# Flag parsing
# ---------------------------------------------------------------------------

class TestFlagParse:
    @staticmethod
    def _mod():
        try:
            import argos_plugin
            return argos_plugin
        except ModuleNotFoundError:
            import argos
            return argos

    def test_defaults_off(self):
        m = self._mod()
        assert m._flag({}, "chronological_injection") is False
        assert m._flag({}, "date_anchor_rerank") is False

    def test_string_true_parses(self):
        m = self._mod()
        assert m._flag({"chronological_injection": "true"},
                       "chronological_injection") is True
        assert m._flag({"date_anchor_rerank": "1"}, "date_anchor_rerank") is True
        assert m._flag({"date_anchor_rerank": "yes"}, "date_anchor_rerank") is True

    def test_bool_true_parses(self):
        m = self._mod()
        assert m._flag({"chronological_injection": True},
                       "chronological_injection") is True
        assert m._flag({"chronological_injection": False},
                       "chronological_injection") is False

    def test_initialize_wires_both_flags(self):
        # Guard: initialize() must read both keys from the config object so
        # values in the live config actually reach the provider attributes.
        # #244: _flag() is replaced by MemoryConfig bool coercion.
        import inspect
        m = self._mod()
        src = inspect.getsource(m.ArgosProvider.initialize)
        assert 'cfg.chronological_injection' in src
        assert 'cfg.date_anchor_rerank' in src


# ---------------------------------------------------------------------------
# Chronological injection (P2B)
# ---------------------------------------------------------------------------

class TestChronoReorder:
    def test_temporal_query_oldest_first(self, tmp_path):
        provider, store = _provider(tmp_path, chrono=True)
        # Backdate so relevance order CANNOT be chronological by accident.
        ids = _seed(store, [
            "bought a new rake for the garden",
            "planted tomatoes in the garden",
            "built a wooden fence around the garden",
        ])
        _backdate(store, ids[0], "2026-06-03T09:00:00+00:00")
        _backdate(store, ids[1], "2026-07-15T09:00:00+00:00")
        _backdate(store, ids[2], "2026-08-20T09:00:00+00:00")

        text = _run_prefetch(
            provider,
            "What happened first in the garden, the rake or the tomatoes?",
        )
        assert _idx(text, "bought a new rake") < _idx(text, "planted tomatoes") < _idx(
            text, "built a wooden fence"
        )

    def test_ordinary_query_keeps_relevance_order(self, tmp_path, monkeypatch):
        try:
            import argos_plugin
        except ModuleNotFoundError:
            import argos as argos_plugin
        import argos_plugin.intent_router as _ir  # ensure submodule is loaded
        provider, store = _provider(tmp_path, chrono=True)

        ids = _seed(store, [
            "the red car is parked outside",
            "my favorite color is red",
            "the car needs a new battery",
            "red paint covers the garage door",
        ])
        for i, ts in enumerate(["2026-08-01T09:00:00+00:00",
                                "2026-08-10T09:00:00+00:00",
                                "2026-08-15T09:00:00+00:00",
                                "2026-08-20T09:00:00+00:00"]):
            _backdate(store, ids[i], ts)

        query = "What color is my car?"
        # Force the ordinary-turn decision even though the harness may not
        # classify this query: the point is the GATE, not the classifier.
        monkeypatch.setattr(_ir, "is_temporal_or_multihop", lambda q: False)

        text = _run_prefetch(provider, query)
        injected = _injected(text)
        dates = [d for _, d in injected]
        # Injection intact...
        assert len(injected) >= 3, f"injection thinned: {injected}"
        # ...and NOT re-sorted oldest-first (relevance/recency order kept).
        # Natural order here is recency-descending; a chronological sort
        # would flip it to ascending — that must NOT happen on ordinary turns.
        assert dates != sorted(dates), f"chronological reorder on ordinary turn: {dates}"

    def test_mixed_offset_timestamps_normalize_to_utc(self, tmp_path):
        # Same-day, offset-wrapped rows: raw lexicographic sort mis-orders
        # these (23:00+14:00 == 09:00 UTC must sort BEFORE 10:00Z).
        provider, store = _provider(tmp_path, chrono=True)
        ids = _seed(store, [
            "planted a lemon tree in the garden",
            "planted a fig tree in the garden",
            "planted an olive tree in the garden",
        ])
        _backdate(store, ids[0], "2026-06-30T23:00:00+14:00")  # 09:00 UTC
        _backdate(store, ids[1], "2026-06-30T10:00:00Z")       # 10:00 UTC
        _backdate(store, ids[2], "2026-07-01T08:00:00-05:00")  # 13:00 UTC

        text = _run_prefetch(
            provider, "What happened first in the garden, the lemon or the fig tree?")
        assert _idx(text, "lemon tree") < _idx(text, "fig tree") < _idx(text, "olive tree")

    def test_best_effort_classifier_failure_keeps_injection(self, tmp_path, monkeypatch):
        try:
            import argos_plugin
        except ModuleNotFoundError:
            import argos as argos_plugin
        import argos_plugin.intent_router as _ir  # ensure submodule is loaded
        provider, store = _provider(tmp_path, chrono=True)

        ids = _seed(store, [
            "the garden has a pond",
            "the garden has a greenhouse",
            "the garden has a compost bin",
        ])
        _backdate(store, ids[0], "2026-08-20T09:00:00+00:00")
        _backdate(store, ids[1], "2026-08-10T09:00:00+00:00")
        _backdate(store, ids[2], "2026-08-01T09:00:00+00:00")

        def _boom(q):
            raise RuntimeError("classifier exploded")

        monkeypatch.setattr(_ir, "is_temporal_or_multihop", _boom)
        query = "What happened first in the garden?"

        text = _run_prefetch(provider, query)
        injected = [c for c, _ in _injected(text)]
        dates = [d for _, d in _injected(text)]
        assert len(injected) == 3, f"injection blanked: {injected}"
        # Best-effort: a classifier crash must leave injection untouched —
        # same rows, and NO chronological reorder applied.
        assert dates != sorted(dates), f"chrono sort ran despite classifier failure: {dates}"


# ---------------------------------------------------------------------------
# Date-anchor resolution + reorder (P2B2)
# ---------------------------------------------------------------------------

class TestResolveTargetDate:
    def test_numeric_ago(self):
        d, label = resolve_target_date("What did I buy 10 days ago?",
                                       today=date(2026, 8, 26))
        assert d == date(2026, 8, 16) and label

    def test_word_ago(self):
        d, _ = resolve_target_date("What happened three weeks ago?",
                                   today=date(2026, 8, 26))
        assert d == date(2026, 8, 5)

    def test_last_weekday(self):
        d, label = resolve_target_date("Who did I go with last saturday?",
                                       today=date(2026, 8, 26))
        assert d == date(2026, 8, 22) and label

    def test_last_weekday_today_is_that_day(self):
        # "last saturday" asked ON a saturday must mean the previous one.
        d, _ = resolve_target_date("What happened last saturday?",
                                   today=date(2026, 8, 22))
        assert d == date(2026, 8, 15)

    def test_on_month_day(self):
        d, label = resolve_target_date("What did I do on July 15th?",
                                       today=date(2026, 8, 26))
        assert d == date(2026, 7, 15) and label

    def test_on_month_day_future_adjusts_to_last_year(self):
        d, _ = resolve_target_date("What did I do on March 2nd?",
                                   today=date(2026, 8, 26))
        assert d == date(2026, 3, 2)

    def test_fixed_holiday(self):
        d, _ = resolve_target_date("What was the airline on valentine's day?",
                                   today=date(2026, 8, 26))
        assert d == date(2026, 2, 14)

    def test_day_before_yesterday(self):
        d, _ = resolve_target_date("What did I do the day before yesterday?",
                                   today=date(2026, 8, 26))
        assert d == date(2026, 8, 24)

    def test_past_weekend(self):
        d, _ = resolve_target_date("Which bike did I fix the past weekend?",
                                   today=date(2026, 8, 26))
        assert d == date(2026, 8, 23)

    def test_no_expression_returns_none(self):
        assert resolve_target_date("What is your favorite color?",
                                   today=date(2026, 8, 26)) == (None, None)

    def test_empty_and_none_safe(self):
        assert resolve_target_date("", today=date(2026, 8, 26)) == (None, None)
        assert resolve_target_date(None, today=date(2026, 8, 26)) == (None, None)


class TestReorderByDate:
    def _rec(self, created_at):
        from types import SimpleNamespace
        return SimpleNamespace(created_at=created_at)

    def test_proximity_order_with_undated_last(self):
        recs = [
            self._rec("2026-08-20T09:00:00+00:00"),   # 36d from Jul 15
            self._rec("2026-07-15T09:00:00+00:00"),   # 0d
            self._rec("2026-06-01T09:00:00+00:00"),   # 44d
            self._rec(None),                          # undated -> end
            self._rec("garbage-timestamp"),           # undated -> end
        ]
        out, target, label = reorder_by_date(
            recs, "What did I do on July 15th?", today=date(2026, 8, 26))
        assert target == date(2026, 7, 15)
        assert [r.created_at for r in out] == [
            "2026-07-15T09:00:00+00:00",
            "2026-08-20T09:00:00+00:00",
            "2026-06-01T09:00:00+00:00",
            None,
            "garbage-timestamp",
        ]

    def test_stable_tie_keeps_input_order(self):
        recs = [
            self._rec("2026-07-20T09:00:00+00:00"),  # 5d
            self._rec("2026-07-10T09:00:00+00:00"),  # 5d — same distance
        ]
        out, _, _ = reorder_by_date(
            recs, "What did I do on July 15th?", today=date(2026, 8, 26))
        assert [r.created_at for r in out] == [r.created_at for r in recs]

    def test_no_date_expression_unchanged(self):
        recs = [self._rec("2026-07-15T09:00:00+00:00"),
                self._rec("2026-06-01T09:00:00+00:00")]
        out, target, label = reorder_by_date(recs, "What is your favorite color?")
        assert out == recs and target is None and label is None

    def test_empty_query_unchanged(self):
        recs = [self._rec("2026-07-15T09:00:00+00:00")]
        assert reorder_by_date(recs, "")[0] == recs

    def test_gate_question_shape_only(self):
        # The reorder must only fire on question-shaped temporal turns.
        assert is_temporal_or_multihop("What did I do on July 15th in the garden?") is True
        assert is_temporal_or_multihop("I went to the shop on July 15th") is False


class TestDateAnchorE2E:
    def test_nearest_date_injected_first(self, tmp_path):
        provider, store = _provider(tmp_path, anchor=True)
        ids = _seed(store, [
            "planted the roses in the garden",   # Jun 1
            "built the trellis in the garden",   # Jul 15  <- target
            "harvested the pumpkins in the garden",  # Aug 20
        ])
        _backdate(store, ids[0], "2026-06-01T09:00:00+00:00")
        _backdate(store, ids[1], "2026-07-15T09:00:00+00:00")
        _backdate(store, ids[2], "2026-08-20T09:00:00+00:00")

        text = _run_prefetch(provider, "What did I do in the garden on July 15th?")
        assert _idx(text, "built the trellis") < _idx(text, "planted the roses")
        assert _idx(text, "built the trellis") < _idx(text, "harvested the pumpkins")
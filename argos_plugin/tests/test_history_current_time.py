"""History-at-current-time (#3, shipped 2026-08-24).

The matrix proved: "Where did Alex use to live?" returns NOTHING at
current time because the valid_to IS NULL filter hides closed versions
entirely — even though as_of= bi-temporal queries work. The fix widens
retrieval to closed versions on historical queries and labels them
"(previously)" at injection so the model reads them as past state.
"""
import pytest


def _provider(tmp_path):
    from store import DuckDBMemoryStore
    from graph import KuzuGraphStore
    try:
        import argos_plugin
    except ModuleNotFoundError:
        import argos as argos_plugin

    store = DuckDBMemoryStore(
        tmp_path / "hist.duckdb", user_id="test_user")
    graph = KuzuGraphStore(
        tmp_path / "hist_kuzu", user_id="test_user")

    provider = argos_plugin.ArgosProvider()
    provider._store = store
    provider._graph = graph
    provider._config = {"history_at_current_time": True}
    provider._history_at_current_time = True
    provider._injection_min_score = 0.0
    return provider, store


# ---------------------------------------------------------------------------
# Detector unit tests (precision-first: question shape AND past marker)
# ---------------------------------------------------------------------------

class TestDetector:
    def _det(self):
        from intent_router import is_historical_query
        return is_historical_query

    def test_used_to_question(self):
        assert self._det()("Where did I use to live?") is True

    def test_did_i_use(self):
        assert self._det()("Did I use to work at a bank?") is True

    def test_previously_job(self):
        assert self._det()("What was my previous job title?") is True

    def test_old_address(self):
        assert self._det()("What was my old address?") is True

    def test_back_then(self):
        assert self._det()("How did we do things back then?") is True

    def test_current_state_not_flagged(self):
        assert self._det()("Where do I live?") is False

    def test_plain_fact_statement_not_flagged(self):
        # Statement-form mention must NOT widen search (v1 tradeoff)
        assert self._det()("I used to drive a Golf.") is False

    def test_future_tense_not_flagged(self):
        assert self._det()("Where will I live next year?") is False

    def test_empty_and_none_safe(self):
        assert self._det()("") is False
        assert self._det()(None) is False


# ---------------------------------------------------------------------------
# Store plumbing: include_closed widens both legs without touching defaults
# ---------------------------------------------------------------------------

class TestIncludeClosed:
    def test_current_search_hides_closed_version(self, tmp_path):
        _, store = _provider(tmp_path)
        rec = store.remember("personal_fact",
                             "Alex lives in Johannesburg")
        store.update_memory(rec.memory_id,
                            content="Alex lives in Centurion")
        hits = store.search("where does Alex live", limit=10,
                            include_closed=False)
        contents = [h.content for h in hits]
        assert "Alex lives in Centurion" in contents
        assert all("Johannesburg" not in c for c in contents)

    def test_widened_search_returns_both_versions(self, tmp_path):
        _, store = _provider(tmp_path)
        rec = store.remember("personal_fact",
                             "Alex lives in Johannesburg")
        store.update_memory(rec.memory_id,
                            content="Alex lives in Centurion")
        hits = store.search("where does Alex live", limit=10,
                            include_closed=True)
        contents = [h.content for h in hits]
        assert any("Johannesburg" in c for c in contents), \
            f"closed version missing: {contents}"
        assert any("Centurion" in c for c in contents), \
            f"current version missing: {contents}"

    def test_widening_does_not_break_relevance(self, tmp_path):
        # Widening must NOT turn relevance filtering off: an unrelated
        # query returns nothing even with include_closed=True.
        _, store = _provider(tmp_path)
        rec = store.remember("personal_fact",
                             "Alex drives a Golf")
        store.update_memory(rec.memory_id, content="Alex drives a Polo")
        hits = store.search("capital city of France", limit=10,
                            include_closed=True)
        assert hits == []

    def test_as_of_still_takes_precedence(self, tmp_path):
        _, store = _provider(tmp_path)
        rec = store.remember("personal_fact",
                             "Alex lives in Johannesburg")
        new = store.update_memory(rec.memory_id,
                                  content="Alex lives in Centurion")
        assert new is not None and new.valid_from > rec.valid_from
        # as_of BETWEEN the two versions must see ONLY the closed one,
        # proving as_of wins over include_closed.
        from datetime import timedelta, datetime
        v0 = datetime.fromisoformat(rec.valid_from)
        v1 = datetime.fromisoformat(new.valid_from)
        mid = ((v0 + (v1 - v0) / 2)).isoformat()
        if isinstance(mid, str):
            mid_s = mid
        else:
            if mid.tzinfo is None:
                mid = mid.replace(tzinfo=rec.valid_from.tzinfo)
            mid_s = mid.isoformat()
        hits = store.search("where does Alex live", limit=10,
                            as_of=mid_s, include_closed=True)
        contents = [h.content for h in hits]
        assert "Alex lives in Johannesburg" in contents, contents
        assert "Alex lives in Centurion" not in contents


# ---------------------------------------------------------------------------
# Provider wiring: detection gate + labelling at injection
# ---------------------------------------------------------------------------

class TestProviderWiring:
    def test_flag_defaults_on(self):
        try:
            import argos_plugin
        except ModuleNotFoundError:
            import argos as argos_plugin
        p = argos_plugin.ArgosProvider()
        p._config = {}
        if hasattr(p, "_apply_config_flags"):
            p._apply_config_flags()
        assert getattr(p, "_history_at_current_time", True) is True

    def test_historical_query_surfaces_closed_version(self, tmp_path):
        # End-to-end: the detection gate fires inside _start_prefetch,
        # widening to closed versions without any explicit kwarg, and the
        # injection formatter labels the closed row "(previously)".
        provider, store = _provider(tmp_path)
        rec = store.remember("personal_fact",
                             "Alex lives in Johannesburg")
        store.update_memory(rec.memory_id,
                            content="Alex lives in Centurion")
        provider._start_prefetch("Where did Alex use to live?")
        import time as _time
        for _ in range(200):  # up to ~10s, then fail loudly below
            if getattr(provider, "_prefetch_done", False):
                break
            _time.sleep(0.05)
        text = str(getattr(provider, "_prefetch_result", ""))
        assert "Johannesburg" in text, \
            f"closed version missing from injected text: {text[:400]}"
        assert "(previously)" in text, \
            f"injected text did not label closed version: {text[:400]}"

    def test_normal_query_stays_current_only(self, tmp_path):
        provider, store = _provider(tmp_path)
        rec = store.remember("personal_fact",
                             "Alex lives in Johannesburg")
        store.update_memory(rec.memory_id,
                            content="Alex lives in Centurion")
        provider._start_prefetch("Where does Alex live?")
        import time as _time
        for _ in range(200):
            if getattr(provider, "_prefetch_done", False):
                break
            _time.sleep(0.05)
        text = str(getattr(provider, "_prefetch_result", ""))
        assert "Johannesburg" not in text, \
            f"closed version leaked into normal query: {text[:400]}"
        assert "Centurion" in text

    def test_labelling_present_in_injection_formatter(self):
        # Guard: closed versions must be visibly marked at injection so
        # the model reads them as past state, never current truth.
        # The formatter lives in provider_retrieval.py since the god-file
        # split (stage 7); inspect that module, not the package root.
        import inspect
        try:
            import argos_plugin
        except ModuleNotFoundError:
            import argos as argos_plugin
        srcc = inspect.getsource(argos_plugin.provider_retrieval)
        assert chr(40) + 'previously' + chr(41) in srcc

"""Intent router regression tests (v3 — casual chat must stay on Flash).

Run standalone:
    python tests/test_intent_router.py

Or as part of the suite:
    python tests/run_tests.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_plugin_dir = Path(__file__).resolve().parent.parent
if str(_plugin_dir) not in sys.path:
    sys.path.insert(0, str(_plugin_dir))

import intent_router  # noqa: E402
from intent_router import (  # noqa: E402
    _QUESTION_RE,
    _TOPIC_RE,
    _proper_nouns,
    is_historical_query,
    route_answerer,
    routing_failure_count,
    temporal_score,
    multi_hop_score,
    is_temporal_or_multihop,
)

_CFG = {
    "router_enabled": "true",
    "router_smart_model": "deepseek/deepseek-v4-pro-0813",
    "router_smart_provider": "openrouter",
    "router_default_model": "deepseek-v4-flash",
    "router_default_provider": "opencode-go",
}


def _model(msg: str) -> str:
    out = route_answerer(_CFG, msg) or {}
    return out.get("model", "")


# Casual question-shaped messages that MUST stay on Flash (v2 false positives).
_CASUAL_FLASH = [
    "hey did you see the game last night",
    "so i told devin to fix it",
    "she said shes coming over later",
    "what did you suggest for the brackets",
    "im tired, rough morning",
    "friday works for me",
    "remember what i asked you about earlier",
    "how are the kids today",
    "ok so chatgpt said the opposite",
    "what do you think about the new layout",
    "is the car still making that noise",
    "can we grab dinner tomorrow",
    "did you get my message about the brackets",
    "how was your day",
    "what do you think about the new layout",
]

# Genuine temporal queries that MUST route to the smart model.
_GENUINE_TEMPORAL = [
    "when did we last change the fuel budget",
    "how long ago did we buy the car",
    "did the price change between june and july",
    "what year did we move into this house",
    "what happened last night",
    "how many days did the install take",
    "when was the last service",
    "what came first, the quote or the order",
]

# Genuine multi-hop queries that MUST route to the smart model.
_GENUINE_MULTIHOP = [
    "what did alex say about the move",
    "what did she say about the move",
    "what did devin say about the deploy",
]

# Statements (no question structure) must stay Flash even with strong words.
_STATEMENTS_FLASH = [
    "you said the supplier would cover the brackets",
    "she said shes coming over later",
    "ok so chatgpt said the opposite",
    "im tired, rough morning",
    "friday works for me",
]


def test_casual_stays_flash() -> None:
    for msg in _CASUAL_FLASH:
        got = _model(msg)
        assert "pro" not in got, f"casual message routed to pro: {msg!r} -> {got}"


def test_genuine_temporal_routes() -> None:
    for msg in _GENUINE_TEMPORAL:
        got = _model(msg)
        assert "pro" in got, f"temporal query stayed flash: {msg!r} -> {got}"


def test_genuine_multihop_routes() -> None:
    for msg in _GENUINE_MULTIHOP:
        got = _model(msg)
        assert "pro" in got, f"multi-hop query stayed flash: {msg!r} -> {got}"


def test_statements_stay_flash() -> None:
    for msg in _STATEMENTS_FLASH:
        got = _model(msg)
        assert "pro" not in got, f"statement routed to pro: {msg!r} -> {got}"


def test_disabled_returns_none() -> None:
    cfg = dict(_CFG, router_enabled="false")
    assert route_answerer(cfg, "when did we last change the fuel budget") is None


def test_scores_clean_gaps() -> None:
    """Single weak signals sit below thresholds; genuine combos clear them."""
    for msg in _CASUAL_FLASH:
        assert temporal_score(msg) < 0.50, f"casual temporal too high: {msg!r}"
        assert multi_hop_score(msg) < 0.50, f"casual multi-hop too high: {msg!r}"
    for msg in _GENUINE_TEMPORAL:
        assert temporal_score(msg) >= 0.50, f"temporal score too low: {msg!r}"
    for msg in _GENUINE_MULTIHOP:
        assert multi_hop_score(msg) >= 0.50, f"multi-hop score too low: {msg!r}"


def test_is_temporal_or_multihop_gate() -> None:
    assert is_temporal_or_multihop("when did we last change the fuel budget")
    assert is_temporal_or_multihop("what did alex say about the move")
    assert not is_temporal_or_multihop("hey did you see the game last night")
    assert not is_temporal_or_multihop("friday works for me")


# --- Issue #237 audit regressions (IR1-IR10) --------------------------------


def test_ir1_relative_time_word_boundary() -> None:
    """"ago" must not fire inside "cargo"; "may" inside "maybe" is a word check."""
    assert temporal_score("is the cargo here") == 0.0
    assert temporal_score("did the magnago arrive") == 0.0
    assert temporal_score("maybe we should go") == 0.0
    # The real adverb still counts as a weak signal.
    assert temporal_score("two days ago") > temporal_score("two days")
    assert temporal_score("what happened back then") >= 0.50


def test_ir2_multiword_reporting_phrases_count() -> None:
    """Multi-word reporting phrases now register as reporting hits."""
    assert multi_hop_score("what did Alex talk about regarding the move") >= 0.50
    assert multi_hop_score("what did she bring up about the deploy") < 0.50  # not a phrase
    assert multi_hop_score("what did Devin brought up about the deploy") >= 0.50
    # Substring fragments of a phrase do not match.
    assert multi_hop_score("what about the brought uppercase text") == 0.0
    # A bare phrase alone is still below threshold (v3 single-signal rule).
    assert multi_hop_score("what did they talk about") < 0.50


def test_ir3_first_word_entity_counted() -> None:
    assert _proper_nouns("Alex and Bob compared to Charlie") == 3
    assert _proper_nouns("Alex said what about the move") == 1
    # Sentence-start function words / imperatives are still not entities.
    assert _proper_nouns("What did she say about the move") == 0
    assert _proper_nouns("Tell me what you suggested") == 0
    assert _proper_nouns("Remind me what you suggested") == 0
    assert multi_hop_score("Alex and Bob compared to Charlie?") >= 0.50
    # Imperative openers must not create a fake entity bonus.
    assert multi_hop_score("Remind me what she suggested") < 0.50
    assert multi_hop_score("Show me what she said") < 0.50


def test_ir4_scores_clamped_and_documented() -> None:
    heavy = "when did we move in 2019, how long ago, 3 years before june and july"
    assert temporal_score(heavy) == 1.0
    assert "NOT probabilities" in intent_router.__doc__


def test_ir5_question_mark_only_terminal() -> None:
    assert not _QUESTION_RE.search("he told me 'sure?' and left")
    assert _QUESTION_RE.search("the car ready?")
    assert _QUESTION_RE.search("the car ready?  ")
    assert not is_temporal_or_multihop("Alex told Bob 'ready?' about the move and left")
    assert is_temporal_or_multihop("Alex told Bob about the move?")


def test_ir6_topic_lookahead_pronouns() -> None:
    for filler in ("something", "someone", "somehow", "anything", "anyone", "nothing"):
        assert not _TOPIC_RE.search(f"what did she say about {filler}"), filler
    assert multi_hop_score("what did she say about something") < 0.50
    # Lookahead is now word-bounded: "italy" is a topic, "it" is not.
    assert _TOPIC_RE.search("what did she say about italy")
    assert not _TOPIC_RE.search("what did she say about it")
    assert multi_hop_score("what did she say about the move") >= 0.50


def test_ir7_routing_failures_counted() -> None:
    before = routing_failure_count()
    orig = intent_router.temporal_score

    def _boom(_q):
        raise RuntimeError("boom")

    intent_router.temporal_score = _boom
    try:
        assert route_answerer(_CFG, "when did we last change the fuel budget") is None
    finally:
        intent_router.temporal_score = orig
    assert routing_failure_count() == before + 1


def test_ir8_historical_not_widened_by_past_tense_probe() -> None:
    """Past-tense fact probes are temporal, not historical (superseded state)."""
    assert temporal_score("what did I eat yesterday") > 0
    assert not is_historical_query("what did I eat yesterday")
    assert is_historical_query("what did I use to drive")


def test_ir9_entity_pair_case_sensitive() -> None:
    assert multi_hop_score("did Alex and Bob agree?") >= 0.50
    assert multi_hop_score("did alex and bob agree?") < 0.50


def test_ir10_thresholds_clamped() -> None:
    # 0.0 would route every question; clamped to 0.1 so a zero-score question stays put.
    cfg = dict(_CFG, router_temporal_threshold="0.0", router_multihop_threshold="0.0")
    assert route_answerer(cfg, "how was your day") is None
    # 2.0 can never be reached; clamped to 1.0 so a saturated query still routes.
    cfg = dict(_CFG, router_temporal_threshold="2.0", router_multihop_threshold="2.0")
    heavy = "when did we move in 2019, how long ago, 3 years before june and july"
    assert (route_answerer(cfg, heavy) or {}).get("model") == _CFG["router_smart_model"]
    # Garbage / NaN fall back to defaults.
    cfg = dict(_CFG, router_temporal_threshold="nan", router_multihop_threshold="abc")
    assert "pro" in _model("when did we last change the fuel budget")
    assert "pro" not in (route_answerer(cfg, "how was your day") or {}).get("model", "")


if __name__ == "__main__":
    failed = 0
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        try:
            fn()
            print(f"  [PASS] {name}")
        except AssertionError as e:
            print(f"  [FAIL] {name} - {e}")
            failed += 1
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
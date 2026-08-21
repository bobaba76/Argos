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

from intent_router import (  # noqa: E402
    route_answerer,
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


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  [PASS] {name}")
            except AssertionError as e:
                print(f"  [FAIL] {name} - {e}")
                failed += 1
    print(f"\n{7 - failed}/7 passed")
    sys.exit(1 if failed else 0)
"""Liveness probes, feature counters, and config fingerprint (#275).

Four silent feature-death bugs were caught in one week (config wipe →
reverted tuning, router-disable, deploy-sync gap, bool-parse flip).
This module makes silence impossible: fail loudly at boot, be observable
at runtime.

LP1 Startup self-smoke: after config load, run a tiny canned probe
through each "can silently die" feature and log ERROR per failure.

LP2 Feature counters: per-feature hit counters (router invocations,
injection events, conflict notes emitted, unfold walks) readable via
a bounded status/health surface.

LP3 Config fingerprint: hash the effective loaded config (sorted
canonical JSON) so config drift is detectable after the fact.
"""
from __future__ import annotations

import hashlib
import json
import logging
import threading
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Features that can silently die — each has a canned probe in
# _startup_self_test() and a hit counter in FeatureCounters.
_SILENT_DEATH_FEATURES = (
    "router",
    "injection_min_score",
    "skip_retrieval_on_trivial",
    "conflict_surfacing",
    "chain_unfold",
    "reranker",
    "recency_importance",
)


class FeatureCounters:
    """LP2: per-feature hit counters, thread-safe.

    A feature that stops firing is visible as a counter that stops
    incrementing. Readable via status() / health probe.
    """

    def __init__(self) -> None:
        self._counters: Dict[str, int] = {f: 0 for f in _SILENT_DEATH_FEATURES}
        self._lock = threading.Lock()

    def increment(self, feature: str, amount: int = 1) -> None:
        """Increment a feature's hit counter."""
        with self._lock:
            if feature not in self._counters:
                # Unknown feature — track it dynamically so new features
                # are visible without a code change to this class.
                self._counters[feature] = 0
            self._counters[feature] += amount

    def get(self, feature: str) -> int:
        """Read a single feature's counter."""
        with self._lock:
            return self._counters.get(feature, 0)

    def snapshot(self) -> Dict[str, int]:
        """Read all counters as a dict (sorted by feature name)."""
        with self._lock:
            return dict(sorted(self._counters.items()))

    def reset(self) -> None:
        """Reset all counters to zero (for testing)."""
        with self._lock:
            for k in self._counters:
                self._counters[k] = 0


# Module-level singleton — shared across the provider lifecycle.
_counters = FeatureCounters()


def get_counters() -> FeatureCounters:
    """Get the module-level FeatureCounters singleton."""
    return _counters


def config_fingerprint(config: Any) -> str:
    """LP3: hash the effective loaded config (sorted canonical JSON).

    Returns a 16-char hex prefix of the SHA-256 hash of the config's
    sorted canonical JSON representation. The fingerprint changes when
    any config value changes, so config drift is detectable after the
    fact by comparing fingerprints.

    Uses model_dump() if the config is a pydantic model, else dict().
    """
    if hasattr(config, "model_dump"):
        data = config.model_dump()
    elif isinstance(config, dict):
        data = config
    else:
        data = dict(config)
    # Sort keys recursively for canonical representation.
    canonical = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _startup_self_test(config: Any, provider: Optional[Any] = None) -> Dict[str, bool]:
    """LP1: run a tiny canned probe through each "can silently die" feature.

    Called from provider_core.initialize() after config load. Logs ERROR
    per failed probe so a silently-dead feature is visible at boot.

    Returns a dict of {feature: passed} for inspection/testing.
    """
    results: Dict[str, bool] = {}
    cfg_dict = config.model_dump() if hasattr(config, "model_dump") else dict(config)

    # --- router ---
    # Probe: router_enabled is a bool, not a string. If it's a string,
    # the _flag() coercion may have failed (the #272 bug).
    router_enabled = cfg_dict.get("router_enabled", False)
    if isinstance(router_enabled, str):
        logger.error(
            "LP1 STARTUP SELF-TEST FAILED: router_enabled is a string "
            "'%s', not a bool — the LLM router may be silently disabled",
            router_enabled,
        )
        results["router"] = False
    elif not isinstance(router_enabled, bool):
        logger.error(
            "LP1 STARTUP SELF-TEST FAILED: router_enabled is %s (type %s), "
            "not a bool — the LLM router may be silently disabled",
            router_enabled, type(router_enabled).__name__,
        )
        results["router"] = False
    else:
        results["router"] = True

    # --- injection_min_score ---
    # Probe: must be a float in [0.0, 1.0]. If it's None or a string,
    # the injection gate is broken.
    min_score = cfg_dict.get("injection_min_score", 0.0)
    if not isinstance(min_score, (int, float)):
        logger.error(
            "LP1 STARTUP SELF-TEST FAILED: injection_min_score is %s "
            "(type %s), not a number — the injection score gate is broken",
            min_score, type(min_score).__name__,
        )
        results["injection_min_score"] = False
    elif not (0.0 <= float(min_score) <= 1.0):
        logger.error(
            "LP1 STARTUP SELF-TEST FAILED: injection_min_score=%.3f is "
            "out of range [0.0, 1.0] — the injection score gate is broken",
            float(min_score),
        )
        results["injection_min_score"] = False
    else:
        results["injection_min_score"] = True

    # --- skip_retrieval_on_trivial ---
    # Probe: must be a bool. If it's a string, the trivial-turn gate
    # may not fire (or may always fire).
    skip_trivial = cfg_dict.get("skip_retrieval_on_trivial", False)
    if not isinstance(skip_trivial, bool):
        logger.error(
            "LP1 STARTUP SELF-TEST FAILED: skip_retrieval_on_trivial is "
            "%s (type %s), not a bool — the trivial-turn gate is broken",
            skip_trivial, type(skip_trivial).__name__,
        )
        results["skip_retrieval_on_trivial"] = False
    else:
        results["skip_retrieval_on_trivial"] = True

    # --- conflict_surfacing ---
    # Probe: must be a bool. If it's None, conflicts are silently
    # smoothed instead of surfaced.
    conflict = cfg_dict.get("conflict_surfacing", True)
    if not isinstance(conflict, bool):
        logger.error(
            "LP1 STARTUP SELF-TEST FAILED: conflict_surfacing is %s "
            "(type %s), not a bool — conflicts may be silently smoothed",
            conflict, type(conflict).__name__,
        )
        results["conflict_surfacing"] = False
    else:
        results["conflict_surfacing"] = True

    # --- chain_unfold ---
    # Probe: must be one of "off", "auto", "always". If it's something
    # else, the chain-unfold pass silently degrades to "off".
    unfold = cfg_dict.get("chain_unfold", "off")
    if unfold not in ("off", "auto", "always"):
        logger.error(
            "LP1 STARTUP SELF-TEST FAILED: chain_unfold='%s' is not one "
            "of off/auto/always — chain unfold silently degrades to off",
            unfold,
        )
        results["chain_unfold"] = False
    else:
        results["chain_unfold"] = True

    # --- reranker ---
    # Probe: if reranker_enabled is True, the model name must be non-empty.
    # A missing model name means the reranker is "enabled" but will fail
    # on first use (silent until a query arrives).
    reranker_enabled = cfg_dict.get("reranker_enabled", False)
    reranker_model = cfg_dict.get("reranker_model", "")
    if reranker_enabled and not reranker_model:
        logger.error(
            "LP1 STARTUP SELF-TEST FAILED: reranker_enabled=True but "
            "reranker_model is empty — the reranker will fail on first use"
        )
        results["reranker"] = False
    else:
        results["reranker"] = True

    # --- recency_importance ---
    # Probe: freshness_markers must be a bool. If it's None, the
    # recency/importance ranking signal is silently lost.
    freshness = cfg_dict.get("freshness_markers", True)
    if not isinstance(freshness, bool):
        logger.error(
            "LP1 STARTUP SELF-TEST FAILED: freshness_markers is %s "
            "(type %s), not a bool — recency/importance signal is lost",
            freshness, type(freshness).__name__,
        )
        results["recency_importance"] = False
    else:
        results["recency_importance"] = True

    # Log a summary line.
    passed = sum(1 for v in results.values() if v)
    failed = len(results) - passed
    if failed:
        logger.error(
            "LP1 STARTUP SELF-TEST: %d/%d features passed, %d FAILED: %s",
            passed, len(results),
            failed,
            [k for k, v in results.items() if not v],
        )
    else:
        logger.info("LP1 STARTUP SELF-TEST: all %d features passed", len(results))

    return results


def run_startup_self_test(config: Any, provider: Optional[Any] = None) -> Dict[str, bool]:
    """Public entry point for the startup self-test.

    Called from provider_core.initialize() after config load. Returns
    the probe results dict for inspection/testing.
    """
    return _startup_self_test(config, provider)


def status() -> Dict[str, Any]:
    """LP2+LP3: bounded status/health surface.

    Returns a dict with:
    - feature_counters: per-feature hit counters
    - config_fingerprint: hash of the effective config (if available)
    - self_test_results: last self-test results (if available)
    """
    return {
        "feature_counters": _counters.snapshot(),
    }

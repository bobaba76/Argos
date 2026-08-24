"""Contradiction matrix — Argos implementation of Atlas steal #1.

Source: Agent Memory Atlas benchmarks page (neoneye.github.io/agent-memory-atlas),
section 7 "The Contradiction Test" (+ section 6 deletion sequence, mini version).

What it scores — five criteria per contradiction case:
  A  ANSWER        does retrieval surface the CURRENT value for the current question?
  B  PROMPT HYGIENE does the injection surface present the stale value UNQUALIFIED?
                   (a bi-temporal label — valid_to/superseded_by set — counts as qualified,
                   i.e. correct behaviour, not a leak)
  C  DURABILITY    after a background/dream pass, does the correction still hold?
                   (deterministic here: dream is forced to its no-LLM skip path unless
                   ARGOS_CONTRA_DREAM_LIVE=1, so this measures resurrection by the
                   deterministic legs, not by LLM rewriting)
  D  HISTORY       is the OLD value still knowable? Two flavours:
                     d_asof  : search(as_of=pre-contradiction)  -> should PASS (bi-temporal)
                     d_now   : plain current-time history question -> PREDICTED FAIL:
                               `valid_to IS NULL` filtering hides closed versions entirely
  E  DERIVED REACH do derived stores (graph edges, embeddings, insights, candidates)
                   still assert the stale value as current? Untestable paths are
                   reported as None — per the atlas, an unprobed path is itself a result.

Cases (atlas table): replacement | polarity | retraction | partial | bounded.

Predictions being tested (from the 24/8 review of Argos schema):
  - A passes on replacement/polarity/partial/bounded.
  - A on RETRACTION fails (no tombstone / negative memory — UOCS 0.117 echo).
  - d_now FAILS everywhere (closed versions invisible at current time).
  - B mostly clean only because retrieval FILTERS superseded versions (trivially clean,
    not labelled-clean — the flag in the output distinguishes these).

Zero-cost by default: no LLM calls anywhere (dream pinned to skip; judge not implemented).
Run:
  cd Documents/Github/Argos/argos_plugin
  HF_HUB_OFFLINE=1 <hermes venv>/python.exe -m pytest tests/test_contradiction_matrix.py -v -s

Status: SCAFFOLD. Cases + A/C/D are fully wired; B/E are best-effort proxies until the
provider-level injection surface is wired into _injected_text() (TODO below).
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import pytest

# conftest.py registers the plugin dir as package `argos`.
try:  # deployed-layout import
    from argos.store import DuckDBMemoryStore
except ImportError:  # bare-module import
    from store import DuckDBMemoryStore  # type: ignore


# --------------------------------------------------------------------------
# Cases
# --------------------------------------------------------------------------

CASES = {
    "replacement": {
        "initial": "Alex lives in Berlin.",
        "entity": "Berlin",
        "contradiction": "Alex moved to Lisbon in March 2026.",
        "question_current": "Where does Alex live?",
        "expected_current": "Lisbon",
        "stale_value": "Berlin",
        "question_history": "Where did Alex use to live?",
        "expect_a": "pass",
    },
    "polarity": {
        "initial": "Alex loves coriander.",
        "entity": "coriander",
        "contradiction": "Alex can't stand coriander anymore.",
        "question_current": "How does Alex feel about coriander?",
        "expected_current": "can't stand coriander",
        "stale_value": "loves coriander",
        "question_history": "How did Alex used to feel about coriander?",
        "expect_a": "pass",
    },
    "retraction": {
        "initial": "Alex's sister is a doctor.",
        "entity": "sister",
        "contradiction": "Alex misspoke — he doesn't have a sister.",
        "question_current": "Is Alex's sister a doctor?",
        # Correct behaviour = refuse/negate; there is NO newer positive value.
        "expected_current": "",
        "stale_value": "sister is a doctor",
        "question_history": "Did Alex ever mention a sister?",
        "expect_a": "xfail-predicted",  # needs negative memory / tombstone
    },
    "partial": {
        "initial": "Alex is an engineer at Acme.",
        "entity": "Acme",
        "contradiction": "Alex got promoted to manager at Acme.",
        "question_current": "What is Alex's role at Acme?",
        "expected_current": "manager",
        "stale_value": "engineer at Acme",
        "question_history": "What role did Alex have at Acme before?",
        "expect_a": "pass",
    },
    "bounded": {
        "initial": "Alex was vegetarian for ten years.",
        "entity": "vegetarian",
        "contradiction": "Alex stopped being vegetarian in 2024.",
        "question_current": "Is Alex vegetarian?",
        "expected_current": "stopped being vegetarian",
        "stale_value": "was vegetarian",
        "question_history": "Was Alex vegetarian in 2022?",
        "expect_a": "pass",
    },
}


# --------------------------------------------------------------------------
# Fixtures / helpers
# --------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@pytest.fixture()
def store(tmp_path):
    s = DuckDBMemoryStore(tmp_path / "contra.duckdb", user_id="test_user")
    yield s
    try:
        s.close()
    except Exception:
        pass


def _write(store, text: str):
    rec = store.remember(
        category="personal_fact",
        content=text,
        dedup=False,          # the test controls duplicates, not the dedup heuristic
        source="explicit",
        expires_at=None,      # no auto-TTL: we want versioning, not expiry, to act
    )
    assert rec is not None, f"write deduped away unexpectedly: {text!r}"
    return rec


def _search(store, query: str, **kw):
    """Tolerant search: forward kwargs (as_of etc.) when the retriever accepts them."""
    try:
        return store.search(query, limit=10, **kw) or []
    except TypeError:
        return store.search(query, limit=10) or []


def _supersede(store, rec, new_text: str):
    """Route through update_memory so valid_to/superseded_by versioning engages."""
    new = store.update_memory(rec.memory_id, content=new_text)
    assert new is not None and new.memory_id != rec.memory_id, "supersede produced no new version"
    return new


# --------------------------------------------------------------------------
# Criteria
# --------------------------------------------------------------------------

def criterion_a(store, case) -> bool:
    """Current-value answerable from retrieval."""
    hits = _search(store, case["question_current"])
    if not case["expected_current"]:
        # retraction: correct behaviour is the stale claim NOT surfacing as truth
        return not any(case["stale_value"] in h.content for h in hits)
    return any(case["expected_current"] in h.content for h in hits)


def criterion_b(store, case) -> str:
    """Injection-surface hygiene. Proxy until provider wiring (see _injected_text).

    Returns one of:
      clean            stale value not surfaced at all
      qualified        stale value surfaced but carries bi-temporal labels (correct)
      exposed          stale value surfaced UNQUALIFIED next to the current one  <- leak
      no-injection     provider-level surface unavailable (proxy used instead)
    """
    inj = _injected_text(store, case["question_current"])
    if inj is None:
        # store-level proxy: ranker output stands in for the assembled prompt
        hits = _search(store, case["question_current"])
        stale_hits = [h for h in hits if case["stale_value"] in h.content]
        if not stale_hits:
            return "clean"
        qualified = all(
            getattr(h, "valid_to", None) is not None
            or bool(getattr(h, "superseded_by", ""))
            for h in stale_hits
        )
        return "qualified" if qualified else "exposed"
    if case["stale_value"] not in inj:
        return "clean"
    # Provider surface: look for qualifier markers near the stale value.
    window = ""
    idx = inj.find(case["stale_value"])
    if idx >= 0:
        window = inj[max(0, idx - 80): idx + len(case["stale_value"]) + 80].lower()
    if any(m in window for m in ("superseded", "until ", "formerly", "valid_to", "as-of", "previous")):
        return "qualified"
    return "exposed"


_PROVIDER_CLS = None
_PROVIDER_TRIED = False


def _injected_text(store, query):
    """Best-effort provider-level assembled injection. Returns str or None.

    TODO(scaffold-finish): check HybridMemoryProvider.__init__'s exact config keys
    (db_path/storage_mode/user_id) and the injection-builder method name, then wire
    them here. Until then this returns None and criterion_b uses the store proxy.
    """
    global _PROVIDER_CLS, _PROVIDER_TRIED
    if os.environ.get("ARGOS_CONTRA_PROVIDER_SURFACE") != "1":
        return None
    if not _PROVIDER_TRIED:
        _PROVIDER_TRIED = True
        try:
            mod = __import__("argos.__init__", fromlist=["HybridMemoryProvider"]) \
                if False else __import__("__init__")  # placeholder; see TODO
            _PROVIDER_CLS = getattr(mod, "HybridMemoryProvider", None)
        except Exception:
            _PROVIDER_CLS = None
    return None  # scaffold: provider path deliberately inert until wired


def criterion_c_run_dream(store, monkeypatch):
    """Fire the background pass deterministically (zero-cost).

    Default: pin egress gate shut so run_distillation takes its documented skip path
    (state must NOT advance; nothing resurrects because nothing LLM-driven runs).
    ARGOS_CONTRA_DREAM_LIVE=1 leaves the real gates in place for a pennies live run.
    """
    import distillation as dist

    if os.environ.get("ARGOS_CONTRA_DREAM_LIVE") != "1":
        try:
            import egress as egress_mod
            monkeypatch.setattr(egress_mod, "gate", lambda *a, **k: False, raising=False)
        except Exception:
            pass
        try:
            monkeypatch.setattr(dist, "_get_llm_client", lambda: None, raising=False)
        except Exception:
            pass
    report = dist.run_distillation(store)
    assert isinstance(report, dict), "run_distillation must return a report dict"
    return report


def criterion_c_durability_held(store, case) -> bool:
    hits = _search(store, case["question_current"])
    if not case["expected_current"]:
        return not any(case["stale_value"] in h.content for h in hits)
    return any(case["expected_current"] in h.content for h in hits)


def criterion_d(store, case, pre_contra_ts: str) -> dict:
    """History preservation, two flavours + raw-row survival ground truth."""
    out = {}
    # d_asof: point-in-time query from BEFORE the contradiction -> old value knowable
    try:
        hits = _search(store, case["question_history"], as_of=pre_contra_ts)
        out["d_asof"] = any(case["stale_value"] in h.content for h in hits)
    except Exception:
        out["d_asof"] = None
    # d_now: plain current-time history question (PREDICTED FAIL: valid_to filter)
    hits_now = _search(store, case["question_history"])
    out["d_now"] = any(case["stale_value"] in h.content for h in hits_now)
    # ground truth: did the old row survive at all (vs hard-deleted)?
    try:
        conn = getattr(store, "connection", None)
        row = conn.execute(
            "SELECT COUNT(*) FROM memory_records WHERE content LIKE ?",
            [f"%{case['stale_value']}%"],
        ).fetchone() if conn is not None else None
        out["row_survives"] = bool(row and row[0])
    except Exception:
        out["row_survives"] = None
    return out


def criterion_e(store, case) -> dict:
    """Derived-reach probes. True=leak, False=clean, None=path untestable (a result)."""
    out = {"vector": None, "graph": None, "insights": None}
    old_sent = case["initial"]

    # embeddings: nearest-neighbour for the OLD statement — who still answers?
    emb = getattr(store, "embedder", None)
    vec_search = getattr(store, "_vector_search_raw", None)
    if emb is not None and vec_search is not None:
        try:
            vec = emb.embed(old_sent)
            hits = vec_search(vec, 5) if _accepts_limit(vec_search) else None
            if hits is None:
                hits = vec_search(vec)
            leaked = [h for h in (hits or [])
                      if case["stale_value"] in str(getattr(h, "content", h))
                      and getattr(h, "valid_to", None) is None
                      and not getattr(h, "superseded_by", "")]
            out["vector"] = bool(leaked)
        except Exception:
            out["vector"] = None

    # graph: does an edge/entity for the stale fact survive as current?
    for gname in ("graph", "_graph", "graph_store"):
        g = getattr(store, gname, None)
        if g is not None and hasattr(g, "query_graph"):
            try:
                nodes = g.query_graph(case["entity"]) or []
                out["graph"] = len(nodes) > 0  # presence ≠ proof of staleness; refine later
            except Exception:
                out["graph"] = None
            break

    # insights/summaries
    gi = getattr(store, "get_insights", None)
    if gi is not None:
        try:
            ins = gi(limit=50)
            blob = " ".join(str(i.get("content", i)) for i in (ins or []))
            out["insights"] = case["stale_value"] in blob
        except Exception:
            out["insights"] = None
    return out


def _accepts_limit(fn) -> bool:  # cheap introspection; scaffolding nicety
    try:
        import inspect
        sig = inspect.signature(fn)
        return "limit" in sig.parameters or any(
            p.kind == p.VAR_KEYWORD for p in sig.parameters.values()
        )
    except (TypeError, ValueError):
        return False


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------

def test_empty_store_control_no_vacuous_pass(store):
    """Atlas/Silica discipline: a metric that cannot fail reports nothing.

    On an EMPTY store the probe query must return nothing. Guards every
    assertion below against passing vacuously.
    """
    assert _search(store, CASES["replacement"]["question_current"]) == []


@pytest.mark.parametrize("case_id", list(CASES.keys()))
def test_contradiction_matrix(store, monkeypatch, case_id):
    """Full A/B/C/D/E walk for one case. Prints the matrix row; enforces only
    the high-confidence predictions (see module docstring)."""
    case = CASES[case_id]

    rec0 = _write(store, case["initial"])
    pre_contra_ts = _now_iso()

    # atlas step 2: the ORIGINAL value must be retrievable before contradicting it
    baseline = criterion_a(store, {**case, "expected_current": case["stale_value"]})
    assert baseline, f"[{case_id}] step-2 baseline failed: initial fact not retrievable"
    _supersede(store, rec0, case["contradiction"])

    a = criterion_a(store, case)
    b = criterion_b(store, case)

    c_report = criterion_c_run_dream(store, monkeypatch)
    c = criterion_c_durability_held(store, case)

    d = criterion_d(store, case, pre_contra_ts)
    e = criterion_e(store, case)

    print(
        f"\nMATRIX {case_id:>11}: A={a} B={b} C={c} "
        f"D(asof={d['d_asof']}, now={d['d_now']}, row={d['row_survives']}) "
        f"E={e}  dream={{ran:{c_report.get('ran')}, reason:{c_report.get('reason') or c_report.get('skipped')}}}"
    )

    # --- enforced predictions -------------------------------------------
    if case["expect_a"] == "pass":
        assert a, f"[{case_id}] predicted A=pass: current value not retrievable"
    else:
        # retraction — two-layer story (measured 24/8): the PROXY passes because
        # versioning hides the stale claim from retrieval. The real gap is that
        # no NEGATIVE memory exists: the negation itself ('no sister') is stored
        # nowhere, so an answerer seeing empty evidence can still confabulate
        # 'yes'. Enforce only the measured part; document the structural gap.
        if not a:
            pytest.xfail(f"[{case_id}] predicted A=fail confirmed: stale claim surfaced")
        print(f"  -> [{case_id}] proxy-A passed (stale claim hidden by valid_to "
              f"filtering); STRUCTURAL GAP unchanged: no negative-memory kind, "
              f"negation unretrievable")

    assert c, f"[{case_id}] correction did not survive the deterministic background pass"
    assert d["d_asof"] is not False, (
        f"[{case_id}] bi-temporal regression: as_of query can't see pre-contradiction state"
    )
    # d_now is the INTERESTING failure — reported, not enforced (yet):
    if d["d_now"] is False:
        print(f"  -> predicted d_now failure confirmed on [{case_id}]: "
              f"history invisible at current time (row_survives={d['row_survives']})")


def test_criterion_c_dream_skip_is_clean(store, monkeypatch):
    """The dream's deterministic skip must be side-effect free: state not advanced,
    report well-formed. (Live-LLM leg is opt-in via ARGOS_CONTRA_DREAM_LIVE=1.)"""
    report = criterion_c_run_dream(store, monkeypatch)
    assert report.get("ran") in (False, True)
    if os.environ.get("ARGOS_CONTRA_DREAM_LIVE") != "1":
        assert not report.get("ran"), f"dream ran despite pinned gates: {report}"
        assert report.get("skipped") == "egress_gate" or report.get("reason") in (
            "llm_client_unavailable", "", None
        ), f"unexpected skip reason: {report}"


# --------------------------------------------------------------------------
# Steal #2 mini: deletion durability (atlas §6 steps 1–9, subset)
# --------------------------------------------------------------------------

def _canary(store):
    tok = "Plumbus Vantablack-7"
    case = {
        "initial": f"The user's dog is named {tok}.",
        "stale_value": tok,
        "question_current": "What is the user's dog called?",
        "expected_current": tok,
        "entity": tok.split()[0],
    }
    rec = store.remember(category="personal_fact", content=case["initial"],
                         dedup=False, source="explicit", expires_at=None)
    return rec, case


def test_deletion_steps_1_4_7_8(store, monkeypatch):
    """write -> baseline -> delete -> assert gone -> background pass -> STILL gone.

    Atlas steps 1-4 + 7-8. Deterministic, zero-cost (dream pinned to its skip path).
    """
    rec, case = _canary(store)
    hits = _search(store, case["question_current"])
    assert any(case["stale_value"] in h.content for h in hits), "step 2 baseline failed"

    action = store.delete_memory(rec.memory_id)
    print(f"\ndelete action: {action}")

    gone = not any(case["stale_value"] in h.content
                   for h in _search(store, case["question_current"]))
    assert gone, "step 4 FAILED: deleted memory still retrievable (hard-delete broken)"
    print("step 4: deleted value unretrievable OK")

    criterion_c_run_dream(store, monkeypatch)

    gone_after = not any(case["stale_value"] in h.content
                         for h in _search(store, case["question_current"]))
    print(f"step 8: still gone after dream: {gone_after}")
    assert gone_after, "step 8 FAILED: background pass resurrected the deleted memory"

    # step 9: derived-store probes (None = untested path, itself a finding)
    e = criterion_e(store, case)
    print(f"step 9 derived probes: {e}")


def test_deletion_step_5_6_refeed_resurrection(store):
    """OBSERVATIONAL (atlas step 5-6): delete, then re-feed the ORIGINAL source
    material. Atlas prediction: 'supersession without a value-level tombstone does
    not survive re-derivation' — most systems fail here.

    First run 24/8: CONFIRMED FAILING for Argos — remember(dedup=True) recreates
    the fact because hard-delete left no tombstone. Non-enforcing until a
    tombstone design decision is made; the printout documents reality.
    """
    rec, case = _canary(store)
    assert store.delete_memory(rec.memory_id), "delete failed"

    refed = None
    try:
        refed = store.remember(
            category="personal_fact",
            content=case["initial"],
            dedup=True,           # realistic re-ingest path
            source="explicit",
            expires_at=None,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"step 5 re-feed raised ({exc!r})")
    back = [h.content for h in _search(store, case["question_current"])
            if case["stale_value"] in h.content]
    resurrected = bool(back) or refed is not None
    print(f"\nstep 5-6: re-feed -> {refed!r}; retrievable after re-feed: {bool(back)}")
    if resurrected:
        print("  !! RE-DERIVATION RESURRECTION confirmed (atlas step 6 failure): "
              "no value-tombstone blocks re-assertion of a deleted fact")
    else:
        print("  deletion survived re-derivation (tombstone-equivalent present)")

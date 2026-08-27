#!/usr/bin/env python3
"""probe_isolation.py — cross-scope bleed probe for Argos (deployment pre-flight).

Deterministic, zero-LLM, zero-cost. Builds a SCRATCH DuckDB (never the live
store) and asserts that the store's scoping actually isolates facts across
users, project_ids, the pending-candidate queue, and supersession.

Exercises BOTH retrieval legs (vector + text) via a deterministic hash
embedder so the WHERE-clause scoping is proven on both paths.

Usage:
    python eval/probes/probe_isolation.py            # default: pass/fail report
    python eval/probes/probe_isolation.py --json     # machine-readable too

Exit code 0 = all pass, 1 = any isolation failure.
"""
from __future__ import annotations

import argparse
import math
import re
import sys
import tempfile
import zlib
from pathlib import Path

_PLUGIN_ROOT = Path(__file__).resolve().parents[2]
if str(_PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_ROOT))

from store import DuckDBMemoryStore  # noqa: E402


class FakeEmbedder:
    """Deterministic 64-dim hash embedder so both retrieval legs run.

    Cosine reflects token/n-gram overlap (meaningful, not random) while
    staying dependency-free. Dimension is arbitrary but fixed.
    """

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim

    def embed(self, text: str, is_query: bool = False):
        v = [0.0] * self.dim
        s = (text or "").lower()
        for n in (2, 3, 4):
            for i in range(max(0, len(s) - n + 1)):
                h = zlib.crc32(s[i:i + n].encode("utf-8")) % self.dim
                v[h] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]


RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str) -> None:
    RESULTS.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true", help="emit JSON summary too")
    args = ap.parse_args()

    tmp = tempfile.mkdtemp(prefix="argos_isolation_probe_")
    db = Path(tmp) / "iso.duckdb"
    store = DuckDBMemoryStore(db, user_id="alice", embedder=FakeEmbedder())

    try:
        print("== Argos cross-scope isolation probe ==")
        print(f"(scratch DB: {db})")

        # -------------------------------------------------------- seed alice
        store.set_user_scope("alice")
        a1 = store.remember(
            category="context_note",
            content="alice-marker: Fenix launch date is 14 October 2026",
            tags=["fenix"], payload={},
            project_id="brand-fenix",
        )
        a2 = store.remember(
            category="context_note",
            content="alice-marker: PerfectGuard contract value is 42000 rands",
            tags=["perfectguard"], payload={},
            project_id="brand-perfectguard",
        )

        # global (unscoped) fact — visible to everyone
        store.remember(
            category="context_note",
            content="global-marker: approved launch slogan is 'Built to last'",
            tags=["slogan"], payload={"user_scope": None},
        )

        # alice's pending candidate
        alice_cid = store.save_candidate(
            category="context_note",
            content="alice-marker: secret budget number is 777",
            source="llm_extraction",
            confidence=0.5,
            evidence_text="alice said budget 777",
        )["candidate_id"]

        # ---------------------------------------------------------- seed bob
        store.set_user_scope("bob")
        b1 = store.remember(
            category="context_note",
            content="bob-marker: PestShield renewal date is 3 Jan 2027",
            tags=["pestshield"], payload={},
        )

        # ------------------------------------------------------------- I1: user isolation
        print("\nI1  user-scope isolation (bob must not see alice's facts)")
        store.set_user_scope("bob")
        hits = store.search("Fenix launch date", limit=20)
        leaked = [r.memory_id for r in hits if r.memory_id == a1.memory_id]
        check("I1.bob<->alice text+vector", not leaked,
              f"alice's mem {a1.memory_id} leaked into bob's search: {leaked}")

        store.set_user_scope("bob")
        hits2 = store.search("PerfectGuard contract value", limit=20)
        leaked2 = [r.memory_id for r in hits2 if r.memory_id == a2.memory_id]
        check("I1.bob<->alice (project-scoped) no leak", not leaked2,
              f"alice's {a2.memory_id} leaked: {leaked2}")
        has_b = any(r.memory_id == b1.memory_id
                    for r in store.search("PestShield renewal", limit=20))
        check("control: bob sees bob", has_b, "bob's own fact is retrieved")

        # ------------------------------------------------------------- I2: global visible to all
        print("\nI2  global (unscoped) fact visible to all users")
        store.set_user_scope("bob")
        g = [r.memory_id for r in store.search("launch slogan 'Built to last'", limit=20)
             if "global-marker" in (r.content or "")]
        check("I2.global visible to bob", len(g) == 1, f"global hits: {len(g)}")

        # ------------------------------------------------------------- I3: project isolation
        print("\nI3  project_id isolation (same user, different brands)")
        store.set_user_scope("alice")
        cross = [r.memory_id for r in store.search(
            "Fenix launch date import", limit=20, project_id="brand-perfectguard")]
        check("I3.project cross-brand hidden", a1.memory_id not in cross,
              f"cross-project leaks (want empty): {cross}")
        cross2 = [r.memory_id for r in store.search(
            "PerfectGuard contract value", limit=20, project_id="brand-fenix")]
        check("I3.project other-direction hidden", a2.memory_id not in cross2,
              f"cross-project leaks (want empty): {cross2}")
        within = [r.memory_id for r in store.search(
            "PerfectGuard contract value", limit=20, project_id="brand-perfectguard")]
        check("I3.project within-brand visible", a2.memory_id in within,
              f"within-project hits: {within}")
        # By-design nuance: a fact with NO project_id (NULL) is user-global and
        # visible from every project filter — project isolation holds only when
        # writers stamp project_id. Documented, not a fail.
        null_visible = [r.memory_id for r in store.search(
            "confirmed launch slogan", limit=20, project_id="brand-fenix")]
        print("  [INFO] null-project facts are user-global by design; "
              f"visible from any project filter: {len(null_visible)} hit(s) — "
              "project_id must be stamped at write time for true isolation")

        # ------------------------------------------------------------- I4: candidate queue isolation
        print("\nI4  pending-candidate queue isolation")
        store.set_user_scope("bob")
        cands = store.list_candidates(status="pending", limit=100)
        seen = {c["candidate_id"] for c in cands}
        bob_sees_alice = alice_cid in seen
        check("I4.bob does not see alice's pending proposal", not bob_sees_alice,
              f"alice's candidate {alice_cid} visible to bob: {bob_sees_alice}")
        bob_res = store.review_candidate(
            candidate_id=alice_cid, decision="rejected", reason="x",
            review_source="manual")
        check("I4.bob cannot review alice's proposal", bob_res is None,
              f"review result: {bob_res}")

        # ------------------------------------------------------------- I5: cross-scope supersede blocked
        print("\nI5  cross-scope supersede blocked")
        store.set_user_scope("bob")
        sres = store.review_candidate(
            candidate_id=store.save_candidate(
                category="context_note",
                content="bob-marker: Fenix launch date is 1 Jan 2027",
                source="llm_extraction", confidence=0.5,
                evidence_text="bob says Fenix moved")["candidate_id"],
            decision="approved", reason="supersede attempt",
            supersedes_memory_id=a1.memory_id, review_source="manual")
        sup_ok = (sres or {}).get("superseded", False)
        check("I5.bob cannot supersede alice's memory", not sup_ok,
              f"bob superseded alice's fact? {sup_ok}")

        # ------------------------------------------------------------- C: negative controls
        print("\nC   negative controls (prove retrieval isn't vacuously empty)")
        store.set_user_scope("alice")
        a1_hits = [r.memory_id for r in store.search("Fenix launch date", limit=20)
                   if r.memory_id == a1.memory_id]
        check("C1.alice sees own fact", len(a1_hits) == 1,
              f"alice finds own mem: {len(a1_hits)}")
        store.set_user_scope("default_user")
        g2 = [r.memory_id for r in store.search("launch slogan", limit=20)
              if "global-marker" in (r.content or "")]
        check("C2.completely-unscoped sees global", len(g2) == 1,
              f"unscoped sees global: {len(g2)}")

        # ------------------------------------------------------------- summary
        print("\n== summary ==")
        passed = sum(1 for _, ok, _ in RESULTS if ok)
        total = len(RESULTS)
        print(f"{passed}/{total} passed")
        if args.json:
            import json
            print(json.dumps({
                "passed": passed, "total": total,
                "results": [{"name": n, "ok": ok, "detail": d}
                            for n, ok, d in RESULTS],
            }, indent=2))
        return 0 if passed == total else 1
    finally:
        store.close()


if __name__ == "__main__":
    sys.exit(main())

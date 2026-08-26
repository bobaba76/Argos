"""Tests for the self-corpus regression gate (snapshot_store, build_gold, run_gate).

Covers:
1. Snapshot manager: refuses while the service answers on the endpoint,
   creates a versioned snapshot + sha256 manifest, prunes to the last 5,
   and the gate rejects a tampered snapshot.
2. Gold builder: deterministic, one probe per memory, freeze requires
   every line reviewed, manifest records the sha + snapshot.
3. Gate runner: scores structure + bit-stable rerun, PASS on an unchanged
   frozen pair, FAIL when a mock regression (recency boost disabled) is
   injected, and the verdict thresholds.

Run with:
    python -m pytest tests/test_gate.py -v
"""
from __future__ import annotations

import json
import shutil
import socketserver
import sys
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

_plugin_dir = Path(__file__).resolve().parent.parent
if str(_plugin_dir) not in sys.path:
    sys.path.insert(0, str(_plugin_dir))
_eval_dir = _plugin_dir / "eval"
if str(_eval_dir) not in sys.path:
    sys.path.insert(0, str(_eval_dir))

import build_gold  # noqa: E402
import run_gate  # noqa: E402
import snapshot_store  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_TOPICS = [
    # (category, content, days_ago)
    ("personal_fact", "User lives in Springfield", 300),
    ("personal_fact", "User lives in Springfield and works at TechCorp", 5),
    ("personal_fact", "User lives in Bayport", 200),
    ("personal_fact", "User is 38 years old", 150),
    ("personal_fact", "User drives a Toyota Corolla", 90),
    ("personal_fact", "User drives a Toyota Hilux", 10),
    ("personal_fact", "User has a dog named Rex", 60),
    ("personal_fact", "User has a cat named Luna", 250),
    ("personal_fact", "User speaks English and Afrikaans", 400),
    ("personal_fact", "User owns a house in Springfield", 120),
    ("preference", "User prefers dark mode for code editors", 30),
    ("preference", "User prefers concise technical answers", 8),
    ("preference", "User likes Python over JavaScript", 45),
    ("preference", "User prefers coffee over tea", 70),
    ("preference", "User likes morning workouts", 15),
    ("preference", "User prefers Vim over VS Code", 180),
    ("preference", "User likes reading sci-fi novels", 220),
    ("preference", "User prefers typing over voice input", 5),
    ("preference", "User likes cooking on weekends", 95),
    ("preference", "User prefers quiet workspaces", 140),
    ("insight", "User realized consistent sleep improves focus", 55),
    ("insight", "User noticed morning workouts boost productivity", 25),
    ("insight", "User found that deadlines help them focus", 110),
    ("insight", "User learned that short breaks reduce burnout", 75),
    ("insight", "User realized caffeine after 2pm hurts sleep", 12),
    ("insight", "User noticed they work best in 90 minute blocks", 160),
    ("insight", "User found that writing things down reduces anxiety", 210),
    ("insight", "User learned that walking meetings spark ideas", 85),
    ("relationship", "User is married to Sam", 130),
    ("relationship", "User has a close friend named Alex", 65),
    ("relationship", "User's brother is named James", 240),
    ("relationship", "User's manager is called Priya", 20),
    ("relationship", "User has a colleague named Thabo", 100),
    ("relationship", "User's sister lives in Cape Town", 190),
    ("goal", "User wants to learn Rust by end of 2026", 40),
    ("goal", "User aims to save R50000 for a house deposit", 170),
    ("goal", "User plans to run a marathon in 2027", 35),
    ("goal", "User wants to start a side project", 80),
    ("goal", "User aims to read 24 books this year", 50),
    ("goal", "User plans to visit Japan next year", 15),
    ("context_note", "User is working on a migration project this week", 3),
    ("context_note", "User mentioned a tenant moves out on 31 December 2026", 28),
    ("context_note", "User is on leave next Friday", 2),
    ("context_note", "User has a dentist appointment on March 3rd 2026", 60),
    ("context_note", "User is interviewing candidates this month", 9),
    ("context_note", "User's team is hiring a backend engineer", 18),
    ("context_note", "User is testing a new workflow this quarter", 22),
    ("context_note", "User has a flight to London on 15 May 2026", 45),
    ("context_note", "User is renovating the kitchen this month", 33),
    ("context_note", "User is preparing a talk for June 2026", 70),
    ("event", "User started a new job at TechCorp on March 2nd 2026", 30),
    ("event", "User moved to Springfield in January 2026", 90),
    ("event", "User finished a marathon in April 2026", 55),
    ("event", "User launched a side project in May 2026", 25),
    ("event", "User switched from Windows to Linux in 2025", 200),
    ("event", "User graduated in December 2024", 400),
    ("event", "User bought a house in August 2025", 260),
    ("event", "User started learning guitar in February 2026", 40),
    ("event", "User quit coffee in January 2026", 100),
    ("event", "User adopted a dog in March 2026", 35),
    # Identical-content pairs with different ages (content isolated from
    # every other record): the recency-family base signal must rank the
    # newer member first; disabling it (mock regression) flips the tie to
    # insertion order (older first).
    ("personal_fact", "User owns a red bicycle", 400),
    ("personal_fact", "User owns a red bicycle", 5),
    ("preference", "User prefers oat milk in coffee", 400),
    ("preference", "User prefers oat milk in coffee", 5),
    ("personal_fact", "User has a green notebook", 400),
    ("personal_fact", "User has a green notebook", 5),
    ("relationship", "User plays the ukulele", 400),
    ("relationship", "User plays the ukulele", 5),
]

# The newer member of each identical-content pair (targets for the
# mock-regression test — the recency-family signal must keep them at
# rank 1).
_PAIR_CONTENTS = [
    "User owns a red bicycle",
    "User prefers oat milk in coffee",
    "User has a green notebook",
    "User plays the ukulele",
]


@pytest.fixture
def fixture_db(tmp_path):
    """A temp store with ~60 known records across categories and ages."""
    from store import DuckDBMemoryStore

    store = DuckDBMemoryStore(tmp_path / "fixture.duckdb", user_id="default_user")
    now = datetime.now(timezone.utc)
    for cat, content, days_ago in _TOPICS:
        rec = store.remember(category=cat, content=content, dedup=False)
        if rec is not None:
            ts = (now - timedelta(days=days_ago)).isoformat()
            store.connection.execute(
                "UPDATE memory_records SET created_at = ? WHERE memory_id = ?",
                [ts, rec.memory_id],
            )
    try:
        store.connection.execute("CHECKPOINT")
    except Exception:
        pass
    store.close()
    return tmp_path / "fixture.duckdb"


class _HealthHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        self.rfile.readline()
        self.wfile.write((json.dumps({"ok": True}) + "\n").encode("utf-8"))
        self.wfile.flush()


@pytest.fixture
def fake_service_endpoint(tmp_path):
    """A fake memory-service endpoint that answers health probes."""
    server = socketserver.ThreadingTCPServer(("127.0.0.1", 0), _HealthHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    endpoint = tmp_path / "hybrid_memory_service.json"
    endpoint.write_text(json.dumps({
        "host": "127.0.0.1",
        "port": server.server_address[1],
        "token": "test-token",
        "pid": 999999,
    }), encoding="utf-8")
    yield endpoint
    server.shutdown()
    server.server_close()


def _take_snapshot(fixture_db, tmp_path, endpoint=None):
    return snapshot_store.take_snapshot(
        fixture_db, tmp_path / "snapshots", endpoint=endpoint,
    )


def _build_gold(fixture_db, tmp_path, auto_approve=True):
    gold_path = tmp_path / "gold" / "gold_v1.jsonl"
    lines = build_gold.build_gold(
        fixture_db, gold_path, limit=60, seed=42, auto_approve=auto_approve,
    )
    return gold_path, lines


def _gold_with_pair_targets(fixture_db, tmp_path):
    """Gold set + probes targeting the NEWER member of each duplicate pair.

    The recency boost must rank the newer member at rank 1 (older member
    ties on text score); disabling the boost flips the tie to insertion
    order, dropping MRR — the mock regression the gate must catch.
    """
    import duckdb

    gold_path, lines = _build_gold(fixture_db, tmp_path)
    conn = duckdb.connect(str(fixture_db), read_only=True)
    try:
        for content in _PAIR_CONTENTS:
            row = conn.execute(
                "SELECT memory_id, category FROM memory_records "
                "WHERE content = ? ORDER BY created_at DESC LIMIT 1",
                [content],
            ).fetchone()
            if row is None:
                continue
            lines.append({
                "memory_id": row[0],
                "category": row[1],
                "content": content,
                "query": f"what is {content}?",
                "template": "direct",
                "status": "approved",
            })
    finally:
        conn.close()
    with gold_path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    return gold_path, lines


# ---------------------------------------------------------------------------
# Snapshot manager
# ---------------------------------------------------------------------------

class TestSnapshotStore:
    def test_service_running_detects_live_endpoint(self, fake_service_endpoint):
        assert snapshot_store.service_running(fake_service_endpoint) is True

    def test_service_running_false_on_missing_endpoint(self, tmp_path):
        assert snapshot_store.service_running(tmp_path / "nope.json") is False

    def test_take_snapshot_refuses_while_service_running(
        self, fixture_db, tmp_path, fake_service_endpoint
    ):
        with pytest.raises(RuntimeError, match="memory service is running"):
            _take_snapshot(fixture_db, tmp_path, endpoint=fake_service_endpoint)

    def test_take_snapshot_creates_manifest(self, fixture_db, tmp_path):
        manifest = _take_snapshot(fixture_db, tmp_path)
        snap_dir = tmp_path / "snapshots" / manifest["snapshot_id"]
        db_copy = snap_dir / manifest["db_filename"]
        assert db_copy.exists()
        assert manifest["db_sha256"] == snapshot_store._sha256_file(db_copy)
        assert manifest["record_count"] == len(_TOPICS)
        assert manifest["snapshot_id"].startswith("20")

    def test_prune_keeps_last_5(self, fixture_db, tmp_path):
        for _ in range(7):
            _take_snapshot(fixture_db, tmp_path)
        remaining = snapshot_store.list_snapshots(tmp_path / "snapshots")
        assert len(remaining) == 5

    def test_gate_rejects_tampered_snapshot(self, fixture_db, tmp_path):
        manifest = _take_snapshot(fixture_db, tmp_path)
        snap_dir = tmp_path / "snapshots" / manifest["snapshot_id"]
        gold_path, _ = _build_gold(fixture_db, tmp_path)
        # Tamper: append a byte to the snapshot db.
        with (snap_dir / manifest["db_filename"]).open("ab") as f:
            f.write(b"X")
        rc = run_gate.run_gate(
            snap_dir, gold_path, tmp_path / "scores.json",
            compare_path=None, ladder=[5, 10, 20], embedder_model="",
        )
        assert rc == 2


# ---------------------------------------------------------------------------
# Gold builder
# ---------------------------------------------------------------------------

class TestBuildGold:
    def test_gold_deterministic_one_probe_per_memory(self, fixture_db, tmp_path):
        p1, lines1 = _build_gold(fixture_db, tmp_path)
        p2, lines2 = _build_gold(fixture_db, tmp_path)
        assert lines1 == lines2
        assert len(lines1) <= len(_TOPICS)
        assert len(lines1) > 0
        ids = [l["memory_id"] for l in lines1]
        assert len(set(ids)) == len(ids), "one probe per memory"
        for l in lines1:
            assert l["query"] and l["category"] and l["status"] == "approved"

    def test_freeze_requires_all_reviewed(self, fixture_db, tmp_path):
        gold_path, _ = _build_gold(fixture_db, tmp_path, auto_approve=False)
        with pytest.raises(RuntimeError, match="pending review"):
            build_gold.freeze_gold(gold_path, fixture_db)

    def test_freeze_writes_manifest(self, fixture_db, tmp_path):
        gold_path, lines = _build_gold(fixture_db, tmp_path)
        manifest = build_gold.freeze_gold(gold_path, fixture_db)
        assert manifest["approved_count"] == len(lines)
        assert manifest["sha256"] == build_gold.gold_sha256(lines)
        assert (tmp_path / "gold" / "gold_manifest.json").exists()
        # Snapshot linkage: fixture_db is not inside a snapshot dir → None.
        assert manifest["snapshot_id"] is None

    def test_freeze_records_snapshot_linkage(self, fixture_db, tmp_path):
        manifest = _take_snapshot(fixture_db, tmp_path)
        snap_db = tmp_path / "snapshots" / manifest["snapshot_id"] / manifest["db_filename"]
        gold_path, _ = _build_gold(snap_db, tmp_path)
        frozen = build_gold.freeze_gold(gold_path, snap_db)
        assert frozen["snapshot_id"] == manifest["snapshot_id"]
        assert frozen["snapshot_db_sha256"] == manifest["db_sha256"]


# ---------------------------------------------------------------------------
# Gate runner
# ---------------------------------------------------------------------------

class TestGateRunner:
    def test_scores_structure_and_bit_stable(self, fixture_db, tmp_path):
        manifest = _take_snapshot(fixture_db, tmp_path)
        snap_dir = tmp_path / "snapshots" / manifest["snapshot_id"]
        gold_path, _ = _build_gold(fixture_db, tmp_path)
        out1 = tmp_path / "scores1.json"
        out2 = tmp_path / "scores2.json"
        rc1 = run_gate.run_gate(
            snap_dir, gold_path, out1, compare_path=None,
            ladder=[5, 10, 20], embedder_model="",
        )
        rc2 = run_gate.run_gate(
            snap_dir, gold_path, out2, compare_path=None,
            ladder=[5, 10, 20], embedder_model="",
        )
        assert rc1 == 0 and rc2 == 0
        s1 = json.loads(out1.read_text(encoding="utf-8"))
        s2 = json.loads(out2.read_text(encoding="utf-8"))
        # Bit-stable except the wall-clock timestamp.
        s1.pop("timestamp")
        s2.pop("timestamp")
        assert s1 == s2, "rerun on an unchanged pair must be bit-stable"
        assert s1["snapshot_id"] == manifest["snapshot_id"]
        assert set(s1["overall"]) == {"recall@5", "recall@10", "recall@20", "mrr"}
        assert s1["by_category"], "per-category scores present"
        for cat, m in s1["by_category"].items():
            assert set(m) == {"recall@5", "recall@10", "recall@20", "mrr"}

    def test_baseline_recorded_then_pass(self, fixture_db, tmp_path):
        manifest = _take_snapshot(fixture_db, tmp_path)
        snap_dir = tmp_path / "snapshots" / manifest["snapshot_id"]
        gold_path, _ = _build_gold(fixture_db, tmp_path)
        baseline = tmp_path / "gate_baseline.json"
        # First run records the baseline.
        rc = run_gate.run_gate(
            snap_dir, gold_path, tmp_path / "scores.json",
            compare_path=baseline, ladder=[5, 10, 20], embedder_model="",
        )
        assert rc == 0
        assert baseline.exists()
        # Rerun on the unchanged pair → PASS.
        rc = run_gate.run_gate(
            snap_dir, gold_path, tmp_path / "scores2.json",
            compare_path=baseline, ladder=[5, 10, 20], embedder_model="",
        )
        assert rc == 0

    def test_mock_regression_fails_gate(self, fixture_db, tmp_path, monkeypatch):
        """Disabling the recency boost must FAIL the gate (exit 1)."""
        from store import DuckDBMemoryStore

        manifest = _take_snapshot(fixture_db, tmp_path)
        snap_dir = tmp_path / "snapshots" / manifest["snapshot_id"]
        gold_path, _ = _gold_with_pair_targets(fixture_db, tmp_path)
        baseline = tmp_path / "gate_baseline.json"
        assert run_gate.run_gate(
            snap_dir, gold_path, tmp_path / "scores.json",
            compare_path=baseline, ladder=[5, 10, 20], embedder_model="",
        ) == 0
        # Inject the regression: disable the recency-family base signal
        # (recency boost + age decay + dormancy decay) so tied pairs fall
        # back to insertion order (older first).
        monkeypatch.setattr(
            DuckDBMemoryStore, "_recency_boost",
            staticmethod(lambda created_at: 0.0),
        )
        monkeypatch.setattr(DuckDBMemoryStore, "_IMPORTANCE_AGE_DECAY_PER_DAY", 0.0)
        monkeypatch.setattr(DuckDBMemoryStore, "_IMPORTANCE_DORMANCY_DECAY_PER_DAY", 0.0)
        rc = run_gate.run_gate(
            snap_dir, gold_path, tmp_path / "scores_regressed.json",
            compare_path=baseline, ladder=[5, 10, 20], embedder_model="",
        )
        assert rc == 1, "recency-signal regression must FAIL the gate"

    def test_restore_passes_again(self, fixture_db, tmp_path, monkeypatch):
        """After the regression is reverted, the gate PASSes again."""
        from store import DuckDBMemoryStore

        manifest = _take_snapshot(fixture_db, tmp_path)
        snap_dir = tmp_path / "snapshots" / manifest["snapshot_id"]
        gold_path, _ = _gold_with_pair_targets(fixture_db, tmp_path)
        baseline = tmp_path / "gate_baseline.json"
        assert run_gate.run_gate(
            snap_dir, gold_path, tmp_path / "scores.json",
            compare_path=baseline, ladder=[5, 10, 20], embedder_model="",
        ) == 0
        monkeypatch.setattr(
            DuckDBMemoryStore, "_recency_boost",
            staticmethod(lambda created_at: 0.0),
        )
        monkeypatch.setattr(DuckDBMemoryStore, "_IMPORTANCE_AGE_DECAY_PER_DAY", 0.0)
        monkeypatch.setattr(DuckDBMemoryStore, "_IMPORTANCE_DORMANCY_DECAY_PER_DAY", 0.0)
        assert run_gate.run_gate(
            snap_dir, gold_path, tmp_path / "scores_regressed.json",
            compare_path=baseline, ladder=[5, 10, 20], embedder_model="",
        ) == 1
        monkeypatch.undo()
        assert run_gate.run_gate(
            snap_dir, gold_path, tmp_path / "scores_restored.json",
            compare_path=baseline, ladder=[5, 10, 20], embedder_model="",
        ) == 0


class TestVerdictThresholds:
    def _scores(self, overall=None, by_category=None, ladder=(5, 20, 96)):
        return {
            "ladder": list(ladder),
            "overall": overall or {"recall@96": 0.90, "mrr": 0.60},
            "by_category": by_category or {
                "personal_fact": {"recall@96": 0.90, "mrr": 0.60},
            },
        }

    def test_pass_when_identical(self):
        s = self._scores()
        ok, failures = run_gate.gate_verdict(s, s)
        assert ok and failures == []

    def test_pass_on_improvement(self):
        base = self._scores()
        cur = self._scores(overall={"recall@96": 0.95, "mrr": 0.70})
        ok, _ = run_gate.gate_verdict(cur, base)
        assert ok, "improvements never fail the gate"

    def test_fail_category_recall_drop_over_1pp(self):
        base = self._scores()
        cur = self._scores(by_category={
            "personal_fact": {"recall@96": 0.88, "mrr": 0.60},
        })
        ok, failures = run_gate.gate_verdict(cur, base)
        assert not ok
        assert any("personal_fact" in f for f in failures)

    def test_pass_category_recall_drop_within_1pp(self):
        base = self._scores()
        cur = self._scores(by_category={
            "personal_fact": {"recall@96": 0.891, "mrr": 0.60},
        })
        ok, _ = run_gate.gate_verdict(cur, base)
        assert ok, "0.9pp category drop is within tolerance"

    def test_fail_overall_recall_drop_over_0_5pp(self):
        base = self._scores()
        cur = self._scores(overall={"recall@96": 0.894, "mrr": 0.60})
        ok, failures = run_gate.gate_verdict(cur, base)
        assert not ok
        assert any("overall" in f for f in failures)

    def test_fail_mrr_drop_over_0_01(self):
        base = self._scores()
        cur = self._scores(overall={"recall@96": 0.90, "mrr": 0.58})
        ok, failures = run_gate.gate_verdict(cur, base)
        assert not ok
        assert any("MRR" in f for f in failures)

    def test_pass_mrr_drop_within_0_01(self):
        base = self._scores()
        cur = self._scores(overall={"recall@96": 0.90, "mrr": 0.595})
        ok, _ = run_gate.gate_verdict(cur, base)
        assert ok

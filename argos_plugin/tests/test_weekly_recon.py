"""Tests for weekly_recon.py — the Monday 06:00 drift recon (issue #21).

Covers the invocation logic without spinning up the embedder:
- ``find_frozen_pair`` locates the snapshot carrying ``gate_baseline.json``.
- ``main`` returns ERROR (rc 2) when no frozen pair exists.
- Branch B (service UP): reruns the frozen pair via ``run_gate.run_gate``
  and maps rc 0 → silent PASS, rc 1 → reported FAIL.
- Branch A (service DOWN): takes a fresh snapshot and runs the gate vs the
  frozen baseline (``run_gate.run_gate`` + ``snapshot_store.take_snapshot``
  are monkeypatched to keep the test hermetic).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_plugin_dir = Path(__file__).resolve().parent.parent
if str(_plugin_dir) not in sys.path:
    sys.path.insert(0, str(_plugin_dir))
_eval_dir = _plugin_dir / "eval"
if str(_eval_dir) not in sys.path:
    sys.path.insert(0, str(_eval_dir))

import weekly_recon  # noqa: E402
import snapshot_store  # noqa: E402


def _make_snapshot_dir(snapshots_dir: Path, snapshot_id: str, with_baseline: bool):
    d = snapshots_dir / snapshot_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "manifest.json").write_text(json.dumps({
        "snapshot_id": snapshot_id,
        "db_filename": "hybrid_memory.duckdb",
        "db_sha256": "deadbeef",
        "record_count": 10,
    }), encoding="utf-8")
    if with_baseline:
        (d / "gate_baseline.json").write_text("{}", encoding="utf-8")
    return d


# ---------------------------------------------------------------------------
# find_frozen_pair
# ---------------------------------------------------------------------------

class TestFindFrozenPair:
    def test_returns_none_when_empty(self, tmp_path):
        assert weekly_recon.find_frozen_pair(tmp_path / "snaps") is None

    def test_returns_none_without_baseline(self, tmp_path):
        snaps = tmp_path / "snaps"
        _make_snapshot_dir(snaps, "20260828-1200", with_baseline=False)
        assert weekly_recon.find_frozen_pair(snaps) is None

    def test_returns_dir_with_baseline(self, tmp_path):
        snaps = tmp_path / "snaps"
        _make_snapshot_dir(snaps, "20260828-1200", with_baseline=False)
        frozen = _make_snapshot_dir(snaps, "20260828-1800", with_baseline=True)
        assert weekly_recon.find_frozen_pair(snaps) == frozen


# ---------------------------------------------------------------------------
# main: error path + branch B (service up)
# ---------------------------------------------------------------------------

class TestMain:
    def _args(self, tmp_path, snapshots_dir, gold):
        return [
            "--live-db", str(tmp_path / "live.duckdb"),
            "--endpoint", str(tmp_path / "endpoint.json"),
            "--snapshots-dir", str(snapshots_dir),
            "--gold", str(gold),
            "--ladder", "5,20",
        ]

    def test_no_frozen_pair_returns_error(self, tmp_path, capsys):
        gold = tmp_path / "gold.jsonl"
        gold.write_text("", encoding="utf-8")
        rc = weekly_recon.main(self._args(tmp_path, tmp_path / "snaps", gold))
        assert rc == 2
        out = capsys.readouterr().out
        assert "verdict: ERROR (no baseline)" in out

    def test_branch_b_pass_silent(self, tmp_path, monkeypatch):
        snaps = tmp_path / "snaps"
        frozen = _make_snapshot_dir(snaps, "20260828-1800", with_baseline=True)
        gold = tmp_path / "gold.jsonl"
        gold.write_text("", encoding="utf-8")

        monkeypatch.setattr(snapshot_store, "service_running", lambda endpoint: True)
        calls = {}

        def fake_run_gate(snapshot_dir, gold_path, out_path, compare_path=None,
                          ladder=None, user_id="default_user"):
            calls["snapshot_dir"] = Path(snapshot_dir)
            calls["compare_path"] = Path(compare_path) if compare_path else None
            return 0

        monkeypatch.setattr(weekly_recon.run_gate, "run_gate", fake_run_gate)
        rc = weekly_recon.main(self._args(tmp_path, snaps, gold))
        assert rc == 0  # silent on PASS
        # Reran the frozen pair against its own baseline.
        assert calls["snapshot_dir"] == frozen
        assert calls["compare_path"] == frozen / "gate_baseline.json"

    def test_branch_b_fail_reports(self, tmp_path, monkeypatch, capsys):
        snaps = tmp_path / "snaps"
        _make_snapshot_dir(snaps, "20260828-1800", with_baseline=True)
        gold = tmp_path / "gold.jsonl"
        gold.write_text("", encoding="utf-8")

        monkeypatch.setattr(snapshot_store, "service_running", lambda endpoint: True)
        monkeypatch.setattr(weekly_recon.run_gate, "run_gate",
                            lambda *a, **k: 1)
        rc = weekly_recon.main(self._args(tmp_path, snaps, gold))
        assert rc == 1
        out = capsys.readouterr().out
        assert "DRIFT" in out and "verdict: FAIL" in out

    def test_branch_a_fresh_snapshot_then_gate(self, tmp_path, monkeypatch, capsys):
        snaps = tmp_path / "snaps"
        frozen = _make_snapshot_dir(snaps, "20260828-1800", with_baseline=True)
        gold = tmp_path / "gold.jsonl"
        gold.write_text("", encoding="utf-8")

        monkeypatch.setattr(snapshot_store, "service_running", lambda endpoint: False)
        # Fake take_snapshot: creates a new snapshot dir + manifest.
        def fake_take_snapshot(live_db, snapshots_dir, endpoint=None, **kw):
            new_id = "20260829-0600"
            d = _make_snapshot_dir(Path(snapshots_dir), new_id, with_baseline=False)
            return {"snapshot_id": new_id, "record_count": 42}

        monkeypatch.setattr(snapshot_store, "take_snapshot", fake_take_snapshot)
        calls = {}

        def fake_run_gate(snapshot_dir, gold_path, out_path, compare_path=None,
                          ladder=None, user_id="default_user"):
            calls["compare_path"] = Path(compare_path) if compare_path else None
            return 0

        monkeypatch.setattr(weekly_recon.run_gate, "run_gate", fake_run_gate)
        rc = weekly_recon.main(self._args(tmp_path, snaps, gold))
        assert rc == 0
        out = capsys.readouterr().out
        assert "branch A" in out and "verdict: PASS" in out
        # Fresh snapshot's gate was compared against the frozen baseline.
        assert calls["compare_path"] == frozen / "gate_baseline.json"

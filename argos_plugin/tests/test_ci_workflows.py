"""#292: CI workflow definitions — valid YAML, tiers declared, triggers
scoped.

Parses the workflow files (PyYAML is a runtime dep) and pins the
tiered-gate contract:
- Tier 0 (deterministic smoke) runs on EVERY PR — unconditional in the
  existing ci.yml pytest step.
- Tier 1 (curated slice) is a separate non-blocking job that consults
  the checked-in path-trigger list.
- Tier 2 (weekly full gate) is a scheduled + dispatchable workflow that
  runs the existing run_gate.py with a DELTA baseline and reports drift.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EVAL_DIR = _REPO_ROOT / "argos_plugin" / "eval"


def _load_workflow(name: str) -> dict:
    path = _REPO_ROOT / ".github" / "workflows" / name
    assert path.exists(), f"workflow missing: {path}"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _steps_of(wf: dict, job: str) -> list:
    return wf["jobs"][job].get("steps", [])


def _run_text(wf: dict, job: str) -> str:
    return " ".join(s.get("run", "") for s in _load_workflow(name=wf)
                    ["jobs"][job].get("steps", []))


class TestTier0InCI:
    """Tier 0 runs on every PR (seconds, deterministic)."""

    def test_ci_yml_parses(self):
        wf = yaml.safe_load(
            (_REPO_ROOT / ".github" / "workflows" / "ci.yml")
            .read_text(encoding="utf-8")
        )
        assert wf["name"].startswith("CI")

    def test_tier0_smoke_always_in_pytest_step(self):
        wf = yaml.safe_load((_REPO_ROOT / ".github" / "workflows"
                             / "ci.yml").read_text(encoding="utf-8"))
        steps = wf["jobs"]["test"]["steps"]
        pytest_steps = [s.get("run", "") for s in steps
                        if "pytest" in s.get("run", "")]
        assert pytest_steps, "no pytest step in ci.yml"
        assert any("test_retrieval_smoke_gate.py" in s for s in pytest_steps), (
            "Tier 0 smoke must run on EVERY PR (unconditional in ci.yml)"
        )


class TestTier1Workflow:
    """Tier 1: path-triggered, non-blocking, delta-vs-baseline."""

    def test_tier1_job_exists_and_is_non_blocking(self):
        wf = yaml.safe_load((_REPO_ROOT / ".github" / "workflows" / "ci.yml")
                            .read_text(encoding="utf-8"))
        job = wf["jobs"].get("tier1-slice")
        assert job is not None, "tier1-slice job missing from ci.yml"
        assert job.get("continue-on-error") is True, (
            "tier1 must be a non-blocking background check"
        )
        assert int(job.get("timeout-minutes", 0)) >= 15, (
            "tier1 needs a generous timeout (~8 min slice + setup)"
        )

    def test_tier1_job_consults_the_path_list(self):
        wf = yaml.safe_load((_REPO_ROOT / ".github" / "workflows" / "ci.yml")
                            .read_text(encoding="utf-8"))
        job = wf["jobs"]["tier1-slice"]
        run_steps = " ".join(s.get("run", "") for s in job.get("steps", []))
        assert "ci_tier1_slice.py --paths" in run_steps, (
            "tier1 job must consult the checked-in path trigger list"
        )
        assert "test_ci_tier1_slice.py" in run_steps, (
            "tier1 job must run the slice test when triggered"
        )


class TestTier2Workflow:
    """Tier 2: weekly scheduled full-gate job exists and reports drift."""

    def test_weekly_workflow_parses(self):
        wf = yaml.safe_load(
            (_REPO_ROOT / ".github" / "workflows" / "retrieval-weekly.yml")
            .read_text(encoding="utf-8")
        )
        assert "retrieval" in wf["name"].lower()

    def test_weekly_is_on_demand_only_until_runner_exists(self):
        """#403 (2026-09-10): the weekly schedule is deliberately DISABLED -
        no self-hosted runner is registered, and a scheduled job with no
        runner is silently skipped (the one failure mode the provisioning
        preflight cannot catch). On-demand dispatch stays available.
        Re-enable the schedule only together with a registered runner + a
        provisioned artifact dir, and update this pin in the same change.
        Rationale + upgrade path: docs/eval-gate-tier2.md."""
        wf = yaml.safe_load(
            (_REPO_ROOT / ".github" / "workflows" / "retrieval-weekly.yml")
            .read_text(encoding="utf-8")
        )
        on = wf.get(True) or wf.get("on") or {}
        assert "workflow_dispatch" in on, "tier 2 must be dispatchable"
        assert "schedule" not in on, (
            "tier 2 schedule re-enabled without a registered runner (#403) "
            "- see docs/eval-gate-tier2.md"
        )

    def test_weekly_runs_existing_gate_delta_vs_baseline(self):
        wf = yaml.safe_load(
            (_REPO_ROOT / ".github" / "workflows" / "retrieval-weekly.yml")
            .read_text(encoding="utf-8")
        )
        job = wf["jobs"]["weekly-gate"]
        run_steps = " ".join(s.get("run", "") for s in job.get("steps", []))
        # Reuses the existing gate — regression = DELTA vs baseline.
        assert "run_gate.py" in run_steps
        assert "--compare" in run_steps
        # Preflight (dry-report) runs BEFORE the gate.
        assert "ci_tier2_check.py --check" in run_steps
        assert "ci_tier2_check.py --drift-history" in run_steps

    # -- PR #337 review blockers ---------------------------------------------

    def _weekly_gate_steps(self) -> list:
        wf = yaml.safe_load(
            (_REPO_ROOT / ".github" / "workflows" / "retrieval-weekly.yml")
            .read_text(encoding="utf-8")
        )
        return wf["jobs"]["weekly-gate"]["steps"]

    def test_drift_scan_sees_the_gate_output(self):
        """Blocker 1 regression: the drift report must scan the SAME
        directory the gate writes its scores into — otherwise every
        weekly run's drift report is {"runs": 0} and 'reports drift'
        never happens."""
        steps = self._weekly_gate_steps()
        gate_step = next(
            s for s in steps if "run_gate.py" in (s.get("run") or ""))
        drift_step = next(
            s for s in steps if "--drift-history" in (s.get("run") or ""))
        gate_run = gate_step["run"]
        drift_run = drift_step["run"]
        # The gate's --out path...
        out_match = [ln.strip() for ln in gate_run.splitlines()
                     if "--out" in ln]
        assert out_match, "gate step has no --out"
        out_path = out_match[0].split("--out", 1)[1].strip().rstrip("\\")
        # ...must live INSIDE the drift scan dir.
        drift_dir = [ln.strip() for ln in drift_run.splitlines()
                     if "--drift-history" in ln]
        assert drift_dir, "drift step has no --drift-history"
        drift_dir = drift_dir[0].split("--drift-history", 1)[1].strip()
        out_norm = out_path.replace("\\", "/").strip('"').rstrip("/")
        drift_norm = drift_dir.replace("\\", "/").rstrip("/")
        assert out_norm.startswith(drift_norm), (
            f"gate --out ({out_norm}) is outside the drift scan dir "
            f"({drift_norm}) — the drift report would never see a run"
        )

    def test_gate_output_is_timestamped_per_run(self):
        """Re-review BLOCKER regression: a FIXED --out filename is
        overwritten every Monday, so drift_history always sees
        {"runs": 1} and cross-run deltas never accumulate. The output
        filename must contain a varying (per-run) component."""
        steps = self._weekly_gate_steps()
        gate_run = next(
            s["run"] for s in steps if "run_gate.py" in (s.get("run") or ""))
        # The gate step computes a timestamped output filename.
        assert re.search(r"gate_scores_weekly_.*\$\(", gate_run) or \
            re.search(r"GATE_OUT=.*\$\(", gate_run), (
            "gate output filename is fixed — every weekly run overwrites "
            "the same file and drift history can never accumulate"
        )
        # And --out references the computed variable (not a literal).
        out_lines = [ln.strip() for ln in gate_run.splitlines()
                     if "--out" in ln and not ln.strip().startswith("#")]
        assert out_lines, "gate step has no --out"
        assert "$GATE_OUT" in out_lines[0] or "$(date" in out_lines[0], (
            "--out must reference the per-run timestamped filename"
        )

    def test_upload_artifact_path_is_a_glob(self):
        """The upload must cover the per-run timestamped files (a fixed
        path would miss every run but one)."""
        steps = self._weekly_gate_steps()
        upload = next(
            s for s in steps
            if str(s.get("uses", "")).startswith("actions/upload-artifact"))
        path = str((upload.get("with") or {}).get("path", ""))
        assert "gate_scores_weekly_*" in path or "*" in path, (
            f"upload path is not a glob: {path}"
        )

    def test_weekly_permissions_allow_artifact_upload(self):
        """upload-artifact needs the actions: write scope — with only
        contents: read the upload step (if: always()) fails 403."""
        wf = yaml.safe_load(
            (_REPO_ROOT / ".github" / "workflows" / "retrieval-weekly.yml")
            .read_text(encoding="utf-8")
        )
        perms = wf.get("permissions") or {}
        assert perms.get("actions") == "write", (
            "weekly workflow must grant actions: write for upload-artifact"
        )

    def test_tier1_uses_minimal_install(self):
        """The tier1 slice is text-only (embedder=None) — installing the
        full requirements.txt (torch + sentence-transformers) makes the
        job slow enough that concurrency-cancel kills it on iterative
        PRs. Minimal install: duckdb + pydantic + numpy + pytest."""
        wf = yaml.safe_load((_REPO_ROOT / ".github" / "workflows" / "ci.yml")
                            .read_text(encoding="utf-8"))
        job = wf["jobs"]["tier1-slice"]
        install_steps = [s.get("run", "") for s in job["steps"]
                         if "pip install" in (s.get("run") or "")]
        assert install_steps, "tier1 job has no install step"
        joined = " ".join(install_steps)
        assert "requirements.txt" not in joined, (
            "tier1 must NOT install the full requirements.txt (torch/"
            "sentence-transformers) — the slice is text-only"
        )
        assert "duckdb" in joined and "pytest" in joined
        # pydantic is pulled by the conftest's argos alias registration
        # (argos_plugin/__init__.py → service_client → memory_service →
        # config_model → from pydantic import BaseModel). Without it the
        # job dies at collection with ModuleNotFoundError.
        assert "pydantic" in joined, (
            "tier1 minimal install must include pydantic — the conftest "
            "imports argos_plugin/__init__.py whose chain pulls config_model"
        )

    def test_tier1_minimal_install_covers_import_chain(self):
        """The minimal install must satisfy the argos_plugin/__init__.py
        import chain — not just the slice's own file.  This is the
        regression that caused the CI collection failure: the conftest
        registers an 'argos' alias that execs __init__.py, whose chain
        pulls service_client → memory_service → config_model → pydantic.
        Without pydantic in the minimal install, the job dies at
        collection with ModuleNotFoundError."""
        wf = yaml.safe_load((_REPO_ROOT / ".github" / "workflows" / "ci.yml")
                            .read_text(encoding="utf-8"))
        job = wf["jobs"]["tier1-slice"]
        install_run = " ".join(
            s.get("run", "") for s in job["steps"]
            if "pip install" in (s.get("run") or "")
        )
        # Packages named in the workflow's pip install line (strip
        # version pins so 'pydantic==2.13.4' → 'pydantic').
        raw = re.findall(r'pip install\s+(.+)', install_run)[0].split()
        installed = {pkg.split("==")[0].split(">=")[0] for pkg in raw}
        # Heavy packages that must NOT be in the minimal install.
        heavy = {"sentence-transformers", "torch", "pandas",
                 "fastapi", "uvicorn", "PyPDF2", "openpyxl",
                 "python-docx", "PyYAML", "winsdk"}
        assert not (installed & heavy), (
            f"tier1 minimal install must exclude heavy deps, found: "
            f"{installed & heavy}"
        )
        # config_model.py is the file that triggered the regression —
        # it imports pydantic at module level (no try/except guard).
        # The minimal install MUST include pydantic so the conftest's
        # argos alias registration (which execs __init__.py) succeeds.
        config_model = (_REPO_ROOT / "argos_plugin" / "config_model.py")
        assert "from pydantic import" in config_model.read_text(
            encoding="utf-8"), (
            "config_model.py no longer imports pydantic at module level "
            "— the regression anchor may need updating"
        )
        assert "pydantic" in installed, (
            "tier1 minimal install must include pydantic — "
            "config_model.py imports it at module level and the "
            "conftest's argos alias registration execs __init__.py"
        )
        # store_common.py imports numpy at module level (with a
        # try/except fallback). Include it so the slice matches
        # production behaviour rather than silently skipping dedup.
        store_common = (_REPO_ROOT / "argos_plugin" / "store_common.py")
        assert "import numpy" in store_common.read_text(encoding="utf-8"), (
            "store_common.py no longer imports numpy — the regression "
            "anchor may need updating"
        )
        assert "numpy" in installed, (
            "tier1 minimal install must include numpy — "
            "store_common.py imports it at module level"
        )

    def test_drift_scan_finds_a_written_scores_file(self, tmp_path):
        """Functional: a gate_scores file written into the scanned dir
        IS found by drift_history (the actual failure mode of blocker 1)."""
        from eval.ci_tier2_check import drift_history
        scores = {
            "timestamp": "2026-09-07T00:00:00+00:00",
            "probe_count": 1000, "ladder": [5, 20, 96],
            "overall": {"recall@96": 1.0, "mrr": 0.88},
        }
        # Simulate the weekly run: scores written into the scanned dir.
        scanned_dir = tmp_path / "snapshots"
        scanned_dir.mkdir()
        (scanned_dir / "gate_scores_weekly.json").write_text(
            json.dumps(scores), encoding="utf-8")
        rep = drift_history(scanned_dir)
        assert rep["runs"] == 1, (
            "drift scan did not find the run's written output"
        )
        assert rep["history"][0]["mrr"] == 0.88

    def test_drift_history_accumulates_across_runs(self, tmp_path):
        """The deliverable: two timestamped per-run files in the scan
        dir produce a cross-run delta (the fixed-filename failure mode
        made this impossible)."""
        from eval.ci_tier2_check import drift_history
        scanned_dir = tmp_path / "snapshots"
        scanned_dir.mkdir()
        for date, recall, mrr in (
            ("20260901", 1.0, 0.88),
            ("20260908", 0.995, 0.87),
        ):
            (scanned_dir / f"gate_scores_weekly_{date}.json").write_text(
                json.dumps({
                    "timestamp": f"20{date[:4]}-{date[4:6]}-{date[6:]}T00:00:00+00:00",
                    "probe_count": 1000, "ladder": [5, 20, 96],
                    "overall": {"recall@96": recall, "mrr": mrr},
                }), encoding="utf-8")
        rep = drift_history(scanned_dir)
        assert rep["runs"] == 2, (
            "timestamped per-run files must accumulate in the drift history"
        )
        assert rep["history"][1]["drift_vs_previous"] == pytest.approx(-0.005)

    def test_weekly_provisions_gitignored_artifacts_before_preflight(self):
        """Blocker 2 regression: the gate's inputs are gitignored and
        checkout's git clean -ffdx deletes ignored files — a provisioning
        step must run BEFORE preflight, or the weekly gate never
        executes."""
        wf = yaml.safe_load(
            (_REPO_ROOT / ".github" / "workflows" / "retrieval-weekly.yml")
            .read_text(encoding="utf-8")
        )
        steps = wf["jobs"]["weekly-gate"]["steps"]
        names = [s.get("name", "") for s in steps]
        prov_idx = next(
            (i for i, n in enumerate(names) if "Provision" in n), None)
        pre_idx = next(
            (i for i, n in enumerate(names) if "Preflight" in n), None)
        assert prov_idx is not None, (
            "no provisioning step — checkout's git clean -ffdx wipes the "
            "gitignored snapshot/baseline/gold before preflight"
        )
        assert pre_idx is not None
        assert prov_idx < pre_idx, "provisioning must precede preflight"
        # The provisioning step references all three artifact classes.
        prov_run = " ".join(
            s.get("run", "") for s in steps if "Provision" in s.get("name", ""))
        assert "hybrid_memory.duckdb" in prov_run
        assert "gate_baseline.json" in prov_run
        # Pin tracks the current gold re-pin (v3 since the 2026-09-07
        # expiry-aware re-pin); update together with the pin itself.
        assert "gold_v3.jsonl" in prov_run
        # It fails loudly (exit 2) when an artifact is absent.
        assert "exit 2" in prov_run


class TestTier2CheckScript:
    """The tier2 preflight/drift tool behaves correctly."""

    def _write_fixture(self, tmp_path: Path) -> tuple[Path, Path, Path]:
        snap = tmp_path / "snap"
        snap.mkdir()
        db = snap / "hybrid_memory.duckdb"
        db.write_bytes(b"fake-db-bytes")
        (snap / "manifest.json").write_text(json.dumps({
            "db_filename": "hybrid_memory.duckdb",
            "db_sha256": hashlib.sha256(db.read_bytes()).hexdigest(),
        }), encoding="utf-8")
        gold = tmp_path / "gold.jsonl"
        gold.write_text(json.dumps({
            "memory_id": "m1", "category": "personal_fact",
            "content": "c", "query": "q", "template": "direct",
        }) + "\n", encoding="utf-8")
        baseline = snap / "gate_baseline.json"
        baseline.write_text(json.dumps({
            "overall": {"recall@96": 1.0, "mrr": 0.88},
        }), encoding="utf-8")
        return snap, gold, baseline

    def test_preflight_passes_on_coherent_fixture(self, tmp_path):
        from eval.ci_tier2_check import preflight
        snap, gold, baseline = self._write_fixture(tmp_path)
        code, report = preflight(snap, gold, baseline)
        assert code == 0
        assert report["ok"] is True
        assert report["gold_probes"] == 1

    def test_preflight_fails_loud_on_missing_baseline(self, tmp_path):
        from eval.ci_tier2_check import preflight
        snap, gold, _ = self._write_fixture(tmp_path)
        code, report = preflight(snap, gold, tmp_path / "missing.json")
        assert code == 2
        assert report["ok"] is False
        assert any("baseline" in p for p in report["problems"])

    def test_preflight_fails_loud_on_sha_mismatch(self, tmp_path):
        from eval.ci_tier2_check import preflight
        snap, gold, baseline = self._write_fixture(tmp_path)
        # Corrupt the manifest's expected sha.
        manifest = json.loads((snap / "manifest.json").read_text(
            encoding="utf-8"))
        manifest["db_sha256"] = "0" * 64
        (snap / "manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8")
        code, report = preflight(snap, gold, baseline)
        assert code == 2
        assert any("sha mismatch" in p for p in report["problems"])

    def test_drift_history_reports_delta_vs_previous(self, tmp_path):
        from eval.ci_tier2_check import drift_history
        d = tmp_path / "hist"
        d.mkdir()
        for name, recall, mrr in (
            ("gate_scores_run1.json", 1.0, 0.88),
            ("gate_scores_run2.json", 0.995, 0.87),
        ):
            (d / name).write_text(json.dumps({
                "timestamp": "2026-08-01T00:00:00+00:00",
                "probe_count": 1000, "ladder": [5, 20, 96],
                "overall": {"recall@96": recall, "mrr": mrr},
            }), encoding="utf-8")
        rep = drift_history(d)
        assert rep["runs"] == 2
        assert rep["history"][1]["drift_vs_previous"] == pytest.approx(-0.005)

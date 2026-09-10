# Tier-2 retrieval gate (eval)

The Tier-2 gate runs the full recall/bench regression check for the memory
pipeline: the self-corpus gate (`argos_plugin/eval/run_gate.py`) over a frozen
store snapshot + the reviewed gold set, evaluated as a **delta against the
snapshot's recorded baseline** — never as absolute numbers. It sits on top of
the tiered regression pyramid (#292): change-scoped per-PR tests (Tier 0/1) →
this gate (Tier 2).

## Current posture: on-demand only (2026-09-10, #403)

The gate is **on-demand**. The former weekly schedule is disabled: the
workflow targets a self-hosted Windows runner, and none has been registered
for this repo — a scheduled job with no runner is *silently skipped*, the one
failure mode the provisioning preflight cannot catch. A stated on-demand
posture is honest; a dead schedule is not.

## Inputs (current reviewed pin)

- Snapshot: `20260906_162404_018897_1d9c5253` (under
  `argos_plugin/eval/snapshots/` — gitignored, personal memory content).
- Gold: `gold_v3.jsonl` (gitignored; freeze sha `f14794bb…` — see
  `argos_plugin/eval/gold/README.md`).
- Baseline: the snapshot's `gate_baseline.json`.

## Running it on demand

Provision a host with the artifacts. The workflow reads
`ARGOS_GATE_ARTIFACTS_DIR` (default `D:/argos-gate-artifacts`); the host-side
layout mirrors the repo-relative paths (`snapshots/<id>/…` +
`gold/gold_v3.jsonl`). Then either:

1. **Dispatch the workflow**: Actions → *retrieval-weekly-gate (Tier 2)* →
   *Run workflow*. Leave `dry_run` unticked to execute the gate; tick it for a
   preflight-only check (artifacts present + coherent).
2. **Or run it directly** in a maintainer environment:

   ```bash
   python argos_plugin/eval/run_gate.py \
     --snapshot argos_plugin/eval/snapshots/<snapshot_id> \
     --gold argos_plugin/eval/gold/gold_v3.jsonl \
     --out argos_plugin/eval/snapshots/gate_scores_<date>.json \
     --compare argos_plugin/eval/snapshots/<snapshot_id>/gate_baseline.json
   ```

The provisioning step + preflight (`argos_plugin/eval/ci_tier2_check.py
--check`) verify integrity (sha256 vs manifest) and **fail loudly** when
artifacts are missing — never a silent skip.

## Upgrade path: back to a weekly cadence

1. Register a self-hosted runner (`runs-on: [self-hosted, windows]`) on a
   host that holds the artifacts and the model cache.
2. Provision (or point `vars.ARGOS_GATE_ARTIFACTS_DIR` at) the artifact
   directory with the current snapshot + gold pair.
3. Re-enable the `schedule:` block in `.github/workflows/retrieval-weekly.yml`
   and add a missed-run check (e.g. alert when no successful run exists in
   the last 14 days).

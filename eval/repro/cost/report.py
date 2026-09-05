#!/usr/bin/env python3
"""Report generator — reads raw results from both arms and produces
COST_REPORT.md with per-model breakdown tables.

Usage:
    python report.py                 # generate COST_REPORT.md
    python report.py --stdout        # also print to stdout
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from datetime import datetime

HERE = Path(__file__).parent
RESULTS = HERE / "results"
REPORT = HERE / "COST_REPORT.md"

# Prices verified 2026-09-02 via OpenRouter /api/v1/models.
# These are per-token prices from the API; converted to per-1M for display.
# Actual cost figures in results come from usage.cost (credits charged),
# not from these stated prices.
MODELS = {
    "deepseek-v4-flash": {
        "id": "deepseek/deepseek-v4-flash-0731",
        "label": "DeepSeek V4 Flash",
        "price_in_per_1m": 0.065,   # API: 0.000000065/token
        "price_out_per_1m": 0.18,   # API: 0.00000018/token
    },
    "glm-5.3-flash": {
        "id": "z-ai/glm-5.3-flash",
        "label": "GLM 5.3 Flash",
        "price_in_per_1m": 0.075,   # API: 0.000000075/token
        "price_out_per_1m": 0.25,   # API: 0.00000025/token
    },
}


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    recs = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * p / 100
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def token_arm_summary() -> dict:
    """Summarize token arm results per model."""
    out = {}
    for key, info in MODELS.items():
        recs = load_jsonl(RESULTS / f"token_arm_{key}_raw.jsonl")
        if not recs:
            out[key] = None
            continue

        ok = [r for r in recs if not r.get("error")]
        reasoning = [r["reasoning_tokens"] for r in ok]
        total = [r["total_tokens"] for r in ok]
        completion = [r["completion_tokens"] for r in ok]
        cost = [r["cost"] for r in ok]
        wall = [r["wallclock_s"] for r in ok]

        # Per-category breakdown
        cats = {}
        for r in ok:
            cat = r.get("category", "unknown")
            cats.setdefault(cat, []).append(r)

        out[key] = {
            "label": info["label"],
            "n_calls": len(recs),
            "n_ok": len(ok),
            "n_errors": len(recs) - len(ok),
            "median_reasoning": statistics.median(reasoning) if reasoning else 0,
            "p90_reasoning": percentile(reasoning, 90),
            "median_completion": statistics.median(completion) if completion else 0,
            "median_total": statistics.median(total) if total else 0,
            "p90_total": percentile(total, 90),
            "median_cost": statistics.median(cost) if cost else 0,
            "p90_cost": percentile(cost, 90),
            "median_wallclock": statistics.median(wall) if wall else 0,
            "total_cost": sum(cost),
            "cost_per_1000": (sum(cost) / len(ok) * 1000) if ok else 0,
            "by_category": {
                cat: {
                    "n": len(rs),
                    "median_reasoning": statistics.median([r["reasoning_tokens"] for r in rs]),
                    "median_total": statistics.median([r["total_tokens"] for r in rs]),
                    "median_cost": statistics.median([r["cost"] for r in rs]),
                }
                for cat, rs in sorted(cats.items())
            },
        }
    return out


def agentic_arm_summary() -> dict:
    """Summarize agentic arm results per model."""
    out = {}
    for key, info in MODELS.items():
        recs = load_jsonl(RESULTS / f"agentic_arm_{key}_raw.jsonl")
        if not recs:
            out[key] = None
            continue

        n_tasks = len(set(r["task_id"] for r in recs))
        n_runs = len(recs)
        successes = [r for r in recs if r["task_success"]]
        n_success = len(successes)

        tool_calls = [r["total_tool_calls"] for r in recs]
        failed_calls = [r["failed_tool_calls"] for r in recs]
        retries = [r["retries"] for r in recs]
        steps = [r["total_steps"] for r in recs]
        total_tok = [r["cumulative_total_tokens"] for r in recs]
        reason_tok = [r["cumulative_reasoning_tokens"] for r in recs]
        cost = [r["total_cost"] for r in recs]
        wall = [r["total_wallclock_s"] for r in recs]

        success_costs = [r["total_cost"] for r in successes]
        cost_per_success = (sum(success_costs) / n_success) if n_success else float("inf")

        # Per-task breakdown
        per_task = {}
        for r in recs:
            tid = r["task_id"]
            per_task.setdefault(tid, []).append(r)

        out[key] = {
            "label": info["label"],
            "n_tasks": n_tasks,
            "n_runs": n_runs,
            "n_success": n_success,
            "success_rate": n_success / n_runs if n_runs else 0,
            "median_tool_calls": statistics.median(tool_calls) if tool_calls else 0,
            "median_failed_calls": statistics.median(failed_calls) if failed_calls else 0,
            "median_retries": statistics.median(retries) if retries else 0,
            "median_steps": statistics.median(steps) if steps else 0,
            "median_total_tokens": statistics.median(total_tok) if total_tok else 0,
            "median_reasoning_tokens": statistics.median(reason_tok) if reason_tok else 0,
            "median_cost": statistics.median(cost) if cost else 0,
            "p90_cost": percentile(cost, 90),
            "total_cost": sum(cost),
            "cost_per_success": cost_per_success,
            "median_wallclock": statistics.median(wall) if wall else 0,
            "per_task": {
                tid: {
                    "runs": len(rs),
                    "successes": sum(1 for r in rs if r["task_success"]),
                    "median_cost": statistics.median([r["total_cost"] for r in rs]),
                    "median_steps": statistics.median([r["total_steps"] for r in rs]),
                    "median_tools": statistics.median([r["total_tool_calls"] for r in rs]),
                }
                for tid, rs in sorted(per_task.items())
            },
        }
    return out


def generate_report(token: dict, agentic: dict) -> str:
    lines = []
    lines.append("# Real-World Cost Benchmark: DeepSeek V4 Flash vs GLM 5.3 Flash")
    lines.append("")
    lines.append(f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append("")
    lines.append("Measures true end-to-end cost (thinking tokens, output tokens, "
                 "tool calls, retries, errors, cost-per-success) — not stated "
                 "per-token price. Both models at **max reasoning effort** via "
                 "OpenRouter. See `README.md` for full protocol.")
    lines.append("")

    # Model info
    lines.append("## Models")
    lines.append("")
    lines.append("| Model | OpenRouter ID | Input $/M | Output $/M |")
    lines.append("|-------|---------------|-----------|------------|")
    for key, info in MODELS.items():
        lines.append(f"| {info['label']} | `{info['id']}` | "
                     f"${info['price_in_per_1m']:.3f} | "
                     f"${info['price_out_per_1m']:.2f} |")
    lines.append("")
    lines.append("_Stated prices from OpenRouter API (verified 2026-09-02). "
                 "Actual cost figures use `usage.cost` returned by the API, "
                 "not these stated prices._")
    lines.append("")

    # Token arm
    lines.append("## Arm 1: Token Economics (no tools)")
    lines.append("")
    lines.append("Fixed prompt set, no tool calls. Measures the hidden "
                 "thinking-token multiplier.")
    lines.append("")

    has_token = any(token.get(k) for k in MODELS)
    if has_token:
        lines.append("| Model | Calls | Errors | Median reasoning tok | P90 reasoning | "
                     "Median total tok | Median cost/prompt | Cost/1000 prompts |")
        lines.append("|-------|-------|--------|---------------------|---------------|"
                     "------------------|-------------------|-------------------|")
        for key in MODELS:
            s = token.get(key)
            if not s:
                lines.append(f"| {MODELS[key]['label']} | — | — | — | — | — | — | — |")
                continue
            lines.append(
                f"| {s['label']} | {s['n_calls']} | {s['n_errors']} | "
                f"{s['median_reasoning']:.0f} | {s['p90_reasoning']:.0f} | "
                f"{s['median_total']:.0f} | ${s['median_cost']:.4f} | "
                f"${s['cost_per_1000']:.2f} |"
            )
        lines.append("")

        # Per-category breakdown
        lines.append("### Per-category breakdown")
        lines.append("")
        any_cats = any(token.get(k) and token[k].get("by_category") for k in MODELS)
        if any_cats:
            all_cats = set()
            for key in MODELS:
                s = token.get(key)
                if s and s.get("by_category"):
                    all_cats.update(s["by_category"].keys())

            for cat in sorted(all_cats):
                lines.append(f"**{cat}**")
                lines.append("")
                lines.append("| Model | n | Median reasoning | Median total | Median cost |")
                lines.append("|-------|---|-----------------|--------------|-------------|")
                for key in MODELS:
                    s = token.get(key)
                    if s and s.get("by_category", {}).get(cat):
                        c = s["by_category"][cat]
                        lines.append(
                            f"| {s['label']} | {c['n']} | "
                            f"{c['median_reasoning']:.0f} | "
                            f"{c['median_total']:.0f} | "
                            f"${c['median_cost']:.4f} |"
                        )
                    else:
                        lines.append(f"| {MODELS[key]['label']} | — | — | — | — |")
                lines.append("")
    else:
        lines.append("_No token arm results found. Run `run_token_arm.py` first._")
        lines.append("")

    # Agentic arm
    lines.append("## Arm 2: Agentic Coding (with tools)")
    lines.append("")
    lines.append("Small coding tasks in a sandbox with 4 tools "
                 "(read_file, write_file, run_command, list_dir). "
                 "Measures tool-call economics and **cost per successful task**.")
    lines.append("")

    has_agentic = any(agentic.get(k) for k in MODELS)
    if has_agentic:
        lines.append("| Model | Task runs | Success rate | Median tool calls | "
                     "Median failed calls | Median retries | Median total tok | "
                     "Median cost/task | Cost/successful task |")
        lines.append("|-------|-----------|--------------|-------------------|"
                     "--------------------|----------------|------------------|"
                     "------------------|---------------------|")
        for key in MODELS:
            s = agentic.get(key)
            if not s:
                lines.append(f"| {MODELS[key]['label']} | — | — | — | — | — | — | — | — |")
                continue
            cps = f"${s['cost_per_success']:.4f}" if s['cost_per_success'] != float('inf') else "N/A (no successes)"
            lines.append(
                f"| {s['label']} | {s['n_runs']} | "
                f"{s['success_rate']*100:.1f}% | "
                f"{s['median_tool_calls']:.0f} | "
                f"{s['median_failed_calls']:.0f} | "
                f"{s['median_retries']:.0f} | "
                f"{s['median_total_tokens']:.0f} | "
                f"${s['median_cost']:.4f} | "
                f"{cps} |"
            )
        lines.append("")

        # Per-task breakdown
        lines.append("### Per-task breakdown")
        lines.append("")
        any_tasks = any(agentic.get(k) and agentic[k].get("per_task") for k in MODELS)
        if any_tasks:
            all_tasks = set()
            for key in MODELS:
                s = agentic.get(key)
                if s and s.get("per_task"):
                    all_tasks.update(s["per_task"].keys())

            lines.append("| Task | Model | Runs | Successes | Median cost | Median steps | Median tools |")
            lines.append("|------|-------|------|-----------|-------------|--------------|--------------|")
            for tid in sorted(all_tasks):
                for key in MODELS:
                    s = agentic.get(key)
                    if s and s.get("per_task", {}).get(tid):
                        t = s["per_task"][tid]
                        lines.append(
                            f"| {tid} | {s['label']} | {t['runs']} | "
                            f"{t['successes']} | ${t['median_cost']:.4f} | "
                            f"{t['median_steps']:.0f} | {t['median_tools']:.0f} |"
                        )
                    else:
                        lines.append(f"| {tid} | {MODELS[key]['label']} | — | — | — | — | — |")
            lines.append("")

        # Key finding
        lines.append("### Key metric: cost per successful task")
        lines.append("")
        ds = agentic.get("deepseek-v4-flash")
        glm = agentic.get("glm-5.3-flash")
        if ds and glm and ds["cost_per_success"] != float("inf") and glm["cost_per_success"] != float("inf"):
            cheaper = "DeepSeek V4 Flash" if ds["cost_per_success"] < glm["cost_per_success"] else "GLM 5.3 Flash"
            ratio = max(ds["cost_per_success"], glm["cost_per_success"]) / min(ds["cost_per_success"], glm["cost_per_success"])
            lines.append(f"**{cheaper}** is cheaper per successful task by {ratio:.2f}x.")
            lines.append("")
            lines.append(f"- DeepSeek V4 Flash: ${ds['cost_per_success']:.4f}/success "
                         f"({ds['n_success']}/{ds['n_runs']} succeeded)")
            lines.append(f"- GLM 5.3 Flash: ${glm['cost_per_success']:.4f}/success "
                         f"({glm['n_success']}/{glm['n_runs']} succeeded)")
        else:
            lines.append("_Insufficient successful runs to compute cost-per-success for both models._")
        lines.append("")
    else:
        lines.append("_No agentic arm results found. Run `run_agentic_arm.py` first._")
        lines.append("")

    # Caveats
    lines.append("## Caveats")
    lines.append("")
    lines.append("- Reasoning models are nondeterministic even at temperature=0. "
                 "n=3 repetitions capture some variance; medians and P90s are rough.")
    lines.append("- Agentic tool-call behavior varies across runs. Per-task "
                 "breakdown shows the distribution.")
    lines.append("- Both models run at **max effort**. Real-world deployment may "
                 "use lower effort, changing the cost profile significantly.")
    lines.append("- Token arm measures cost only, not answer quality. "
                 "Agentic arm uses deterministic pass/fail verify scripts.")
    lines.append("- OpenRouter-reported `usage.cost` is the source of truth for "
                 "all cost figures (not computed from stated prices).")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stdout", action="store_true",
                    help="Also print report to stdout")
    args = ap.parse_args()

    token = token_arm_summary()
    agentic = agentic_arm_summary()

    report = generate_report(token, agentic)
    REPORT.write_text(report, encoding="utf-8")
    print(f"Report written to {REPORT}")

    if args.stdout:
        print()
        print(report)


if __name__ == "__main__":
    main()

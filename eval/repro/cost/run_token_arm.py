#!/usr/bin/env python3
"""Token-economics arm — measures thinking/output/total tokens and cost
for DeepSeek V4 Flash vs GLM 5.3 Flash on a fixed prompt set, no tools.

Runs each prompt through both models at max reasoning effort, N=3
repetitions per prompt to capture variance. Records per-call usage
breakdowns from OpenRouter and writes raw + summary results.

Usage:
    python run_token_arm.py                    # full run
    python run_token_arm.py --dry-run          # validate prompts, no API
    python run_token_arm.py --reps 5           # custom repetition count
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)

# Verified 2026-09-02 via OpenRouter /api/v1/models:
#   deepseek/deepseek-v4-flash-0731 — supports efforts ["max","high","low"]
#   z-ai/glm-5.3-flash              — supports efforts ["max","high","low"]
# Both support tools + tool_choice. Both return reasoning_tokens in usage.
MODELS = {
    "deepseek-v4-flash": "deepseek/deepseek-v4-flash-0731",
    "glm-5.3-flash": "z-ai/glm-5.3-flash",
}


def load_prompts(path: Path) -> list[dict]:
    prompts = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                prompts.append(json.loads(line))
    return prompts


def prompt_set_hash(prompts: list[dict]) -> str:
    raw = json.dumps(prompts, sort_keys=True).encode()
    return hashlib.sha256(raw).hexdigest()[:16]


def run_prompt(client, model_id: str, prompt_text: str, effort: str):
    """One call — returns per-call record dict."""
    messages = [{"role": "user", "content": prompt_text}]
    result = client.chat(
        model=model_id, messages=messages, reasoning_effort=effort,
    )
    return {
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "reasoning_tokens": result.reasoning_tokens,
        "total_tokens": result.total_tokens,
        "cost": result.cost,
        "wallclock_s": round(result.wallclock_s, 2),
        "retries": result.retries,
        "error": result.error,
        "finish_reason": result.finish_reason,
        "output_chars": len(result.content),
        "output_text": result.content[:500],  # truncate for storage
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true",
                    help="Validate prompts without API calls")
    ap.add_argument("--reps", type=int, default=3,
                    help="Repetitions per prompt (default 3)")
    ap.add_argument("--effort", default="max",
                    help="Reasoning effort: max, high, or low (default max)")
    ap.add_argument("--prompts", default=str(HERE / "prompts.jsonl"))
    ap.add_argument("--model", default=None,
                    help="Only run this model label (e.g. glm-5.3-flash). "
                         "Default: run all models.")
    args = ap.parse_args()

    prompts = load_prompts(Path(args.prompts))
    print(f"Loaded {len(prompts)} prompts from {args.prompts}")
    print(f"Prompt set hash: {prompt_set_hash(prompts)}")

    categories = {}
    for p in prompts:
        categories.setdefault(p["category"], 0)
        categories[p["category"]] += 1
    print(f"Categories: {dict(categories)}")

    # Filter models if --model specified
    models_to_run = MODELS
    if args.model:
        if args.model not in MODELS:
            print(f"ERROR: unknown model '{args.model}'. "
                  f"Available: {list(MODELS.keys())}", file=sys.stderr)
            sys.exit(1)
        models_to_run = {args.model: MODELS[args.model]}

    if args.dry_run:
        print(f"\n[DRY RUN] Prompt set validated. No API calls made.")
        print(f"Would run {len(prompts)} prompts x {len(models_to_run)} models "
              f"x {args.reps} reps = {len(prompts) * len(models_to_run) * args.reps} calls")
        print(f"Effort: {args.effort}")
        return

    from or_client import ORClient
    try:
        client = ORClient()
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    total_calls = len(prompts) * len(models_to_run) * args.reps
    print(f"\nRunning {total_calls} calls (effort={args.effort}, reps={args.reps})")
    print("-" * 70)

    for label, model_id in models_to_run.items():
        out_path = RESULTS / f"token_arm_{label}_raw.jsonl"
        print(f"\n>> {label} ({model_id})")
        call_idx = 0
        with open(out_path, "w", encoding="utf-8") as out:
            for p in prompts:
                for rep in range(args.reps):
                    call_idx += 1
                    t0 = time.time()
                    rec = run_prompt(
                        client, model_id, p["prompt"], args.effort,
                    )
                    rec["prompt_id"] = p["id"]
                    rec["category"] = p["category"]
                    rec["model"] = label
                    rec["model_id"] = model_id
                    rec["rep"] = rep
                    rec["effort"] = args.effort
                    rec["prompt_set_hash"] = prompt_set_hash(prompts)
                    out.write(json.dumps(rec) + "\n")
                    out.flush()

                    elapsed = time.time() - t0
                    status = "OK" if not rec["error"] else f"ERR:{rec['error']}"
                    print(f"  [{call_idx}/{total_calls}] {p['id']} rep{rep} "
                          f"tok={rec['total_tokens']} "
                          f"reason={rec['reasoning_tokens']} "
                          f"cost=${rec['cost']:.4f} "
                          f"{elapsed:.1f}s {status}")

        print(f"   Written: {out_path}")

    print("\nToken arm complete. Run report.py to generate COST_REPORT.md.")


if __name__ == "__main__":
    main()

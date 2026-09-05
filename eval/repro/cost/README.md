# Real-World Cost Benchmark: DeepSeek V4 Flash vs GLM 5.3 Flash

Measures true end-to-end cost (thinking tokens, output tokens, tool calls,
retries, errors, cost-per-success) — not stated per-token price.

## Why this exists

Stated per-token pricing is a theoretical floor that almost no real workload
hits. Reasoning models generate hidden thinking tokens, make tool calls
(some of which fail and trigger retries), and take varying numbers of steps
to complete a task. This benchmark measures the **actual cost of getting
work done**, not the cost of generating a single token.

## Models

| Model | OpenRouter ID | Input $/M | Output $/M |
|-------|---------------|-----------|------------|
| DeepSeek V4 Flash | `deepseek/deepseek-v4-flash-0731` | $0.065 | $0.18 |
| GLM 5.3 Flash | `z-ai/glm-5.3-flash` | $0.075 | $0.25 |

_Prices verified 2026-09-02 via OpenRouter `/api/v1/models`. Actual cost
figures in results come from `usage.cost` (credits charged by OpenRouter),
not from these stated prices._

## Protocol

- **API route**: OpenRouter (`https://openrouter.ai/api/v1/chat/completions`)
- **Reasoning effort**: `max` on every call (both arms). Both models support
  `["max", "high", "low"]` per the OpenRouter API.
- **Temperature**: 0 (best-effort determinism; reasoning models may still vary)
- **Repetitions**: n=3 per prompt/task to capture variance
- **Cost source**: OpenRouter-reported `usage.cost` (credits actually charged)
- **Token capture**: `usage.prompt_tokens`, `usage.completion_tokens`,
  `usage.completion_tokens_details.reasoning_tokens`, `usage.total_tokens`

## Arm 1: Token Economics (no tools)

**File**: `prompts.jsonl` — 40 controlled prompts, 10 each:
- Reasoning (logic puzzles, deduction)
- Coding (write a function to spec, no execution)
- Math (competition-style, show work)
- Instruction-following (summarize, reformat, constrained output)

**Runner**: `run_token_arm.py`
**Output**: `results/token_arm_<model>_raw.jsonl`

Per-call metrics: `prompt_tokens`, `completion_tokens`, `reasoning_tokens`,
`total_tokens`, `cost`, `wallclock_s`, `output_text` (truncated).

## Arm 2: Agentic Coding (with tools)

**File**: `agentic_tasks.jsonl` — 10 small coding tasks, each with:
- Natural language prompt
- Setup commands (sandbox prep)
- Verify script (deterministic pass/fail, exits 0 on success)
- Expected files list

**Runner**: `run_agentic_arm.py`
**Output**: `results/agentic_arm_<model>_raw.jsonl`

**Tools available to the model** (4):
- `read_file(path)` — read a file in the sandbox
- `write_file(path, content)` — write/overwrite a file
- `run_command(command)` — execute a shell command (30s timeout, no network)
- `list_dir(path)` — list directory contents

**Step cap**: 20 (model must complete within 20 tool-call rounds)

Per-run metrics: `task_success`, `total_steps`, `total_tool_calls`,
`successful_tool_calls`, `failed_tool_calls`, `retries` (consecutive
identical tool calls), `cumulative_prompt_tokens`,
`cumulative_completion_tokens`, `cumulative_reasoning_tokens`,
`cumulative_total_tokens`, `total_cost`, `total_wallclock_s`, per-step
breakdown.

**Key derived metric**: `cost_per_success` = total cost across successful
runs / number of successful runs. This is the only number that answers
"which model is cheaper for getting work done."

## How to run

```bash
# 1. Set your API key
export OPENROUTER_API_KEY="sk-or-..."

# 2. Full reproducible run (spends ~$2-5 in credits)
bash eval/repro/cost/verify_cost.sh

# Or run arms separately:
python eval/repro/cost/run_token_arm.py
python eval/repro/cost/run_agentic_arm.py
python eval/repro/cost/report.py

# Dry-run (no API calls, validates prompt/task sets):
python eval/repro/cost/run_token_arm.py --dry-run
python eval/repro/cost/run_agentic_arm.py --dry-run
```

## Output

- `COST_REPORT.md` — summary tables for both arms, per-category and per-task breakdowns, cost-per-successful-task comparison
- `results/*.jsonl` — raw per-call/per-run records (gitignored)

## Estimated run cost

~840 total calls across both arms. At flash-tier prices and ~5K-20K
reasoning tokens per call at max effort:

| Arm | DeepSeek cost | GLM cost |
|-----|--------------|----------|
| Token (240 calls) | ~$0.60 | ~$0.94 |
| Agentic (~600 calls) | ~$1.26 | ~$1.95 |
| **Total** | **~$1.86** | **~$2.89** |

## Caveats

1. **Nondeterminism**: Reasoning models vary across runs even at
   temperature=0. n=3 captures some variance but P90 estimates are rough.
   The report shows medians and P90s, not single-shot numbers.

2. **Effort level**: Both models run at **max effort**. Real-world
   deployment may use lower effort, which would reduce cost (and possibly
   quality) significantly. A future effort-level sweep can be added by
   parameterizing the runners.

3. **Toolset design**: The 4-tool set (read/write/run/list) is deliberately
   minimal. A different toolset would produce different tool-call patterns.
   The benchmark measures model + toolset, not model alone.

4. **Task difficulty**: The 10 agentic tasks are small and self-contained.
   Larger, more complex tasks would likely amplify differences in
   tool-call efficiency and retry behavior.

5. **Sandbox safety**: `run_command` executes in a temp directory only.
   No network access, no writes outside the sandbox. Verify scripts are
   checked in and reviewed.

6. **Model availability**: If a model is rate-limited or deprecated on
   OpenRouter, the harness records the error and reports partial results.

7. **GLM 5.3 Flash multimodal**: Not exercised — both arms are text-only.
   DeepSeek V4 Flash does not have native vision, so this is a fair
   comparison for text/coding workloads.

## Relationship to existing eval

This suite is **separate** from `eval/repro/verify_repro.sh` (which
verifies Argos accuracy claims and should not make paid API calls).
`verify_cost.sh` is the entry point for cost benchmarking and is
intentionally standalone.

## Files

| File | Purpose |
|------|---------|
| `or_client.py` | OpenRouter client with usage capture |
| `prompts.jsonl` | 40 controlled prompts (token arm) |
| `run_token_arm.py` | Token-economics arm runner |
| `agentic_tasks.jsonl` | 10 agentic coding tasks with verify scripts |
| `run_agentic_arm.py` | Agentic arm runner with tool-call loop |
| `report.py` | Generates COST_REPORT.md |
| `verify_cost.sh` | Reproducibility wrapper |
| `README.md` | This file |
| `results/` | Raw output directory (gitignored) |

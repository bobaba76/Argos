# Real-World Cost Benchmark: DeepSeek V4 Flash vs GLM 5.3 Flash

Generated: 2026-09-02 18:05 UTC

Measures true end-to-end cost (thinking tokens, output tokens, tool calls, retries, errors, cost-per-success) — not stated per-token price. Both models at **max reasoning effort** via OpenRouter. See `README.md` for full protocol.

## Models

| Model | OpenRouter ID | Input $/M | Output $/M |
|-------|---------------|-----------|------------|
| DeepSeek V4 Flash | `deepseek/deepseek-v4-flash-0731` | $0.065 | $0.18 |
| GLM 5.3 Flash | `z-ai/glm-5.3-flash` | $0.075 | $0.25 |

_Stated prices from OpenRouter API (verified 2026-09-02). Actual cost figures use `usage.cost` returned by the API, not these stated prices._

## Arm 1: Token Economics (no tools)

Fixed prompt set, no tool calls. Measures the hidden thinking-token multiplier.

| Model | Calls | Errors | Median reasoning tok | P90 reasoning | Median total tok | Median cost/prompt | Cost/1000 prompts |
|-------|-------|--------|---------------------|---------------|------------------|-------------------|-------------------|
| DeepSeek V4 Flash | — | — | — | — | — | — | — |
| GLM 5.3 Flash | 120 | 0 | 1002 | 8467 | 1335 | $0.0004 | $1.06 |

### Per-category breakdown

**coding**

| Model | n | Median reasoning | Median total | Median cost |
|-------|---|-----------------|--------------|-------------|
| DeepSeek V4 Flash | — | — | — | — |
| GLM 5.3 Flash | 30 | 7502 | 8630 | $0.0022 |

**instruction**

| Model | n | Median reasoning | Median total | Median cost |
|-------|---|-----------------|--------------|-------------|
| DeepSeek V4 Flash | — | — | — | — |
| GLM 5.3 Flash | 30 | 319 | 460 | $0.0001 |

**math**

| Model | n | Median reasoning | Median total | Median cost |
|-------|---|-----------------|--------------|-------------|
| DeepSeek V4 Flash | — | — | — | — |
| GLM 5.3 Flash | 30 | 1099 | 1692 | $0.0005 |

**reasoning**

| Model | n | Median reasoning | Median total | Median cost |
|-------|---|-----------------|--------------|-------------|
| DeepSeek V4 Flash | — | — | — | — |
| GLM 5.3 Flash | 30 | 722 | 1096 | $0.0003 |

## Arm 2: Agentic Coding (with tools)

Small coding tasks in a sandbox with 4 tools (read_file, write_file, run_command, list_dir). Measures tool-call economics and **cost per successful task**.

| Model | Task runs | Success rate | Median tool calls | Median failed calls | Median retries | Median total tok | Median cost/task | Cost/successful task |
|-------|-----------|--------------|-------------------|--------------------|----------------|------------------|------------------|---------------------|
| DeepSeek V4 Flash | — | — | — | — | — | — | — | — |
| GLM 5.3 Flash | 30 | 100.0% | 4 | 0 | 0 | 5502 | $0.0004 | $0.0005 |

### Per-task breakdown

| Task | Model | Runs | Successes | Median cost | Median steps | Median tools |
|------|-------|------|-----------|-------------|--------------|--------------|
| a01 | DeepSeek V4 Flash | — | — | — | — | — |
| a01 | GLM 5.3 Flash | 3 | 3 | $0.0003 | 4 | 4 |
| a02 | DeepSeek V4 Flash | — | — | — | — | — |
| a02 | GLM 5.3 Flash | 3 | 3 | $0.0006 | 9 | 8 |
| a03 | DeepSeek V4 Flash | — | — | — | — | — |
| a03 | GLM 5.3 Flash | 3 | 3 | $0.0003 | 5 | 5 |
| a04 | DeepSeek V4 Flash | — | — | — | — | — |
| a04 | GLM 5.3 Flash | 3 | 3 | $0.0002 | 4 | 4 |
| a05 | DeepSeek V4 Flash | — | — | — | — | — |
| a05 | GLM 5.3 Flash | 3 | 3 | $0.0004 | 4 | 4 |
| a06 | DeepSeek V4 Flash | — | — | — | — | — |
| a06 | GLM 5.3 Flash | 3 | 3 | $0.0012 | 6 | 6 |
| a07 | DeepSeek V4 Flash | — | — | — | — | — |
| a07 | GLM 5.3 Flash | 3 | 3 | $0.0003 | 5 | 5 |
| a08 | DeepSeek V4 Flash | — | — | — | — | — |
| a08 | GLM 5.3 Flash | 3 | 3 | $0.0004 | 4 | 4 |
| a09 | DeepSeek V4 Flash | — | — | — | — | — |
| a09 | GLM 5.3 Flash | 3 | 3 | $0.0003 | 5 | 5 |
| a10 | DeepSeek V4 Flash | — | — | — | — | — |
| a10 | GLM 5.3 Flash | 3 | 3 | $0.0006 | 4 | 4 |

### Key metric: cost per successful task

_Insufficient successful runs to compute cost-per-success for both models._

## Caveats

- Reasoning models are nondeterministic even at temperature=0. n=3 repetitions capture some variance; medians and P90s are rough.
- Agentic tool-call behavior varies across runs. Per-task breakdown shows the distribution.
- Both models run at **max effort**. Real-world deployment may use lower effort, changing the cost profile significantly.
- Token arm measures cost only, not answer quality. Agentic arm uses deterministic pass/fail verify scripts.
- OpenRouter-reported `usage.cost` is the source of truth for all cost figures (not computed from stated prices).

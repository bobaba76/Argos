#!/usr/bin/env python3
"""Agentic arm — measures tool-call economics for DeepSeek V4 Flash vs
GLM 5.3 Flash on small coding tasks in a sandbox.

Each task: model gets 4 tools (read_file, write_file, run_command, list_dir)
and a step cap of 20. The harness runs the tool-call loop, executes tool
calls in a temp sandbox, and records per-step token/cost/tool metrics.
After completion (or step cap), a deterministic verify script checks
pass/fail. Cost-per-successful-task is the key derived metric.

Usage:
    python run_agentic_arm.py                 # full run
    python run_agentic_arm.py --dry-run       # validate tasks, no API
    python run_agentic_arm.py --reps 5        # custom repetition count
    python run_agentic_arm.py --step-cap 30   # custom step cap
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import shutil
from pathlib import Path

HERE = Path(__file__).parent
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)

# Verified 2026-09-02 via OpenRouter /api/v1/models:
#   deepseek/deepseek-v4-flash-0731 — supports tools, tool_choice, parallel_tool_calls
#   z-ai/glm-5.3-flash              — supports tools, tool_choice
#   Both support reasoning efforts ["max","high","low"]
MODELS = {
    "deepseek-v4-flash": "deepseek/deepseek-v4-flash-0731",
    "glm-5.3-flash": "z-ai/glm-5.3-flash",
}

SYSTEM_PROMPT = (
    "You are a coding agent working in a sandbox directory. You have access "
    "to tools for reading files, writing files, running shell commands, and "
    "listing directory contents. Complete the task by writing the required "
    "files and verifying they work. Use the run_command tool to execute "
    "Python files and tests. When you are done and have verified the task "
    "is complete, respond with a summary of what you did."
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file in the sandbox.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path to the file."},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file in the sandbox. Creates or overwrites.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path to the file."},
                    "content": {"type": "string", "description": "Content to write."},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Run a shell command in the sandbox directory. Returns stdout, stderr, and exit code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to run."},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List files in the sandbox directory (or a subdirectory).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path. Default: root."},
                },
                "required": [],
            },
        },
    },
]


def load_tasks(path: Path) -> list[dict]:
    tasks = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
    return tasks


def task_set_hash(tasks: list[dict]) -> str:
    raw = json.dumps(tasks, sort_keys=True).encode()
    return hashlib.sha256(raw).hexdigest()[:16]


def execute_tool(name: str, args: dict, sandbox: Path) -> dict:
    """Execute one tool call in the sandbox. Returns result dict."""
    try:
        if name == "read_file":
            path = sandbox / args.get("path", "")
            if not path.exists():
                return {"error": f"File not found: {args.get('path')}"}
            content = path.read_text(encoding="utf-8", errors="replace")
            return {"content": content[:8000]}

        elif name == "write_file":
            path = sandbox / args.get("path", "")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(args.get("content", ""), encoding="utf-8")
            return {"status": "written", "path": args.get("path")}

        elif name == "run_command":
            cmd = args.get("command", "")
            proc = subprocess.run(
                cmd, shell=True, cwd=str(sandbox),
                capture_output=True, text=True, timeout=30,
            )
            stdout = proc.stdout[:8000]
            stderr = proc.stderr[:4000]
            return {
                "stdout": stdout, "stderr": stderr,
                "exit_code": proc.returncode,
            }

        elif name == "list_dir":
            rel = args.get("path", ".")
            target = sandbox / rel
            if not target.exists():
                return {"error": f"Path not found: {rel}"}
            entries = []
            for p in sorted(target.iterdir()):
                entries.append({
                    "name": p.name,
                    "type": "dir" if p.is_dir() else "file",
                    "size": p.stat().st_size if p.is_file() else 0,
                })
            return {"entries": entries}

        else:
            return {"error": f"Unknown tool: {name}"}

    except subprocess.TimeoutExpired:
        return {"error": "Command timed out (30s)"}
    except Exception as e:
        return {"error": str(e)}


def run_task(client, model_id: str, task: dict, effort: str,
             step_cap: int, rep: int) -> dict:
    """Run one task through the tool-call loop. Returns full record."""
    sandbox = Path(tempfile.mkdtemp(prefix=f"agentic_{task['id']}_"))

    # Run setup commands if any
    if task.get("setup"):
        subprocess.run(
            task["setup"], shell=True, cwd=str(sandbox),
            capture_output=True, timeout=30,
        )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task["prompt"]},
    ]

    steps = []
    cumulative = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "reasoning_tokens": 0,
        "total_tokens": 0,
        "cost": 0.0,
        "wallclock_s": 0.0,
    }
    tool_calls_total = 0
    tool_calls_success = 0
    tool_calls_failed = 0
    retries = 0
    last_tool_sig = None

    for step_num in range(1, step_cap + 1):
        result = client.chat(
            model=model_id, messages=messages,
            tools=TOOLS, reasoning_effort=effort,
        )

        # Accumulate usage
        cumulative["prompt_tokens"] += result.prompt_tokens
        cumulative["completion_tokens"] += result.completion_tokens
        cumulative["reasoning_tokens"] += result.reasoning_tokens
        cumulative["total_tokens"] += result.total_tokens
        cumulative["cost"] += result.cost
        cumulative["wallclock_s"] += result.wallclock_s

        step_record = {
            "step": step_num,
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "reasoning_tokens": result.reasoning_tokens,
            "total_tokens": result.total_tokens,
            "cost": result.cost,
            "wallclock_s": round(result.wallclock_s, 2),
            "finish_reason": result.finish_reason,
            "tool_calls": [],
            "error": result.error,
        }

        if result.error:
            steps.append(step_record)
            break

        # If no tool calls, model is done (final answer)
        if not result.tool_calls:
            step_record["final_answer"] = result.content[:500]
            steps.append(step_record)
            break

        # Add assistant message with tool calls to conversation
        assistant_msg = {"role": "assistant", "content": result.content}
        if result.tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc["id"] or f"call_{step_num}_{i}",
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": json.dumps(tc["args"]),
                    },
                }
                for i, tc in enumerate(result.tool_calls)
            ]
        messages.append(assistant_msg)

        # Execute each tool call
        for i, tc in enumerate(result.tool_calls):
            tool_calls_total += 1
            sig = f"{tc['name']}:{json.dumps(tc['args'], sort_keys=True)}"
            if sig == last_tool_sig:
                retries += 1
            last_tool_sig = sig

            exec_result = execute_tool(tc["name"], tc["args"], sandbox)
            is_error = bool(exec_result.get("error"))
            if is_error:
                tool_calls_failed += 1
            else:
                tool_calls_success += 1

            step_record["tool_calls"].append({
                "name": tc["name"],
                "args": tc["args"],
                "result": exec_result,
                "success": not is_error,
            })

            # Feed tool result back to model
            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"] or f"call_{step_num}_{i}",
                "content": json.dumps(exec_result),
            })

        steps.append(step_record)

    # Run verify script
    verify_ok = False
    verify_output = ""
    try:
        proc = subprocess.run(
            task["verify"], shell=True, cwd=str(sandbox),
            capture_output=True, text=True, timeout=30,
        )
        verify_ok = proc.returncode == 0
        verify_output = (proc.stdout + proc.stderr)[:2000]
    except subprocess.TimeoutExpired:
        verify_output = "verify timed out"
    except Exception as e:
        verify_output = f"verify error: {e}"

    # Check expected files
    files_present = {}
    for f in task.get("expected_files", []):
        files_present[f] = (sandbox / f).exists()

    # Cleanup sandbox
    try:
        shutil.rmtree(sandbox)
    except Exception:
        pass

    return {
        "task_id": task["id"],
        "model": model_id,
        "rep": rep,
        "effort": effort,
        "step_cap": step_cap,
        "task_success": verify_ok,
        "verify_output": verify_output.strip()[-500:],
        "files_present": files_present,
        "total_steps": len(steps),
        "total_tool_calls": tool_calls_total,
        "successful_tool_calls": tool_calls_success,
        "failed_tool_calls": tool_calls_failed,
        "retries": retries,
        "cumulative_prompt_tokens": cumulative["prompt_tokens"],
        "cumulative_completion_tokens": cumulative["completion_tokens"],
        "cumulative_reasoning_tokens": cumulative["reasoning_tokens"],
        "cumulative_total_tokens": cumulative["total_tokens"],
        "total_cost": round(cumulative["cost"], 6),
        "total_wallclock_s": round(cumulative["wallclock_s"], 2),
        "steps": steps,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true",
                    help="Validate tasks without API calls")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--effort", default="max",
                    help="Reasoning effort: max, high, or low (default max)")
    ap.add_argument("--step-cap", type=int, default=20)
    ap.add_argument("--tasks", default=str(HERE / "agentic_tasks.jsonl"))
    ap.add_argument("--model", default=None,
                    help="Only run this model label (e.g. glm-5.3-flash). "
                         "Default: run all models.")
    args = ap.parse_args()

    tasks = load_tasks(Path(args.tasks))
    print(f"Loaded {len(tasks)} agentic tasks from {args.tasks}")
    print(f"Task set hash: {task_set_hash(tasks)}")

    # Validate verify scripts parse
    for t in tasks:
        assert t.get("verify"), f"Task {t['id']} missing verify script"
        assert t.get("prompt"), f"Task {t['id']} missing prompt"

    # Filter models if --model specified
    models_to_run = MODELS
    if args.model:
        if args.model not in MODELS:
            print(f"ERROR: unknown model '{args.model}'. "
                  f"Available: {list(MODELS.keys())}", file=sys.stderr)
            sys.exit(1)
        models_to_run = {args.model: MODELS[args.model]}

    if args.dry_run:
        print(f"\n[DRY RUN] Task set validated. No API calls made.")
        total = len(tasks) * len(models_to_run) * args.reps
        print(f"Would run {len(tasks)} tasks x {len(models_to_run)} models "
              f"x {args.reps} reps = {total} task runs "
              f"(~{total * args.step_cap // 2} calls estimated)")
        print(f"Effort: {args.effort}")
        return

    from or_client import ORClient
    try:
        client = ORClient()
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    total_runs = len(tasks) * len(models_to_run) * args.reps
    print(f"\nRunning {total_runs} task runs "
          f"(effort={args.effort}, step_cap={args.step_cap}, reps={args.reps})")
    print("-" * 70)

    run_idx = 0
    for label, model_id in models_to_run.items():
        out_path = RESULTS / f"agentic_arm_{label}_raw.jsonl"
        print(f"\n>> {label} ({model_id})")
        with open(out_path, "w", encoding="utf-8") as out:
            for task in tasks:
                for rep in range(args.reps):
                    run_idx += 1
                    t0 = time.time()
                    print(f"  [{run_idx}/{total_runs}] {task['id']} rep{rep} ...",
                          end=" ", flush=True)
                    record = run_task(
                        client, model_id, task, args.effort,
                        args.step_cap, rep,
                    )
                    record["model_label"] = label
                    record["task_set_hash"] = task_set_hash(tasks)
                    out.write(json.dumps(record) + "\n")
                    out.flush()

                    elapsed = time.time() - t0
                    status = "PASS" if record["task_success"] else "FAIL"
                    print(f"{status} "
                          f"steps={record['total_steps']} "
                          f"tools={record['total_tool_calls']} "
                          f"fails={record['failed_tool_calls']} "
                          f"tok={record['cumulative_total_tokens']} "
                          f"cost=${record['total_cost']:.4f} "
                          f"{elapsed:.1f}s")

        print(f"   Written: {out_path}")

    print("\nAgentic arm complete. Run report.py to generate COST_REPORT.md.")


if __name__ == "__main__":
    main()

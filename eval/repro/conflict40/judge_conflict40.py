#!/usr/bin/env python3
"""Judge for the frozen conflict set (lme_conflict40_v1).

Class-aware rubric (model driven by LME_MODEL, default gpt-4o for protocol
parity; LME_MODEL=deepseek/deepseek-v4-flash-0731 gives ~96%-agreement cheap
judging):
  expectation=current_value -> correct ONLY if hypothesis gives the CURRENT value
  expectation=abstain        -> correct ONLY if hypothesis says no current policy
  expectation=refuse         -> correct ONLY if hypothesis refuses / names authority
  expectation=fact           -> plain factual match

Usage:
  python judge_conflict40.py --hyp hyp_conflict40_v1.jsonl [--limit N]
  # run inside the LongMemEval harness clone, with <hermes-agent root> on sys.path
  # (judge needs `agent.auxiliary_client` for LLM calls; exports come from
  #  the harness environment, e.g. HERMES_HOME/.env sourced into the shell)
"""
import argparse
import json
import os
import sys
import threading
import time

JUDGE_MODEL = os.environ.get("LME_MODEL", "openai/gpt-4o-2024-11-20")

RUBRIC = {
    "current_value": (
        "The question asks for the CURRENT value of something. The user's history "
        "contains an older statement and a LATER plain restatement with a different "
        "value (no explicit 'changed to', no links between them).\n"
        "Mark CORRECT only if the hypothesis gives the CURRENT (later) value.\n"
        "Mark INCORRECT if it gives only the outdated value, or gives both values "
        "without identifying the current one, or cites the outdated value as current, "
        "or abstains ('cannot confirm') when the answer is present in context."
    ),
    "abstain": (
        "The question asks about a policy/feature/rule that was explicitly "
        "discontinued with NO replacement, or a rule scoped to a place/premise that "
        "does not cover the question. There is NO current policy.\n"
        "Mark CORRECT only if the hypothesis says there is no current policy / it no "
        "longer exists / no active rule (any wording that does not endorse the old "
        "rule as current). Mentioning the discontinued rule as history is fine if the "
        "conclusion is 'no current policy'.\n"
        "Mark INCORRECT if it presents the discontinued rule as still current, "
        "invents a replacement, or smooths the two statements into one answer."
    ),
    "refuse": (
        "The question asks whether an action is allowed, or what the authoritative "
        "value is. The context shows the action requires a named authority, or the "
        "relevant document is a DRAFT/unapproved, or the rule is scoped to a team "
        "that does not include the actor. The correct answer is a REFUSAL or a "
        "statement of the required authority.\n"
        "Mark CORRECT only if the hypothesis refuses / declines / states the "
        "authority required / treats the draft as not current.\n"
        "Mark INCORRECT if it performs or endorses the action, treats the draft as "
        "authoritative, or answers as if no authority is required."
    ),
    "fact": (
        "Plain factual question. Mark CORRECT if the hypothesis matches the gold "
        "answer. Mark INCORRECT otherwise (including abstaining when the fact is "
        "present in context)."
    ),
}


def _call_llm(messages, max_tokens=250, timeout=60.0):
    from agent.auxiliary_client import call_llm
    kwargs = {}
    if os.environ.get("LME_PROVIDER"):
        kwargs["provider"] = os.environ["LME_PROVIDER"]
    if os.environ.get("LME_MODEL"):
        kwargs["model"] = os.environ["LME_MODEL"]
    response = call_llm(
        task="conflict40_judge",
        messages=messages,
        temperature=0.0,
        max_tokens=max_tokens,
        timeout=timeout,
        **kwargs,
    )
    if response is None:
        return None
    return str(response.choices[0].message.content).strip()


def judge_one(q, hyp):
    exp = q.get("expectation", "fact")
    rubric = RUBRIC.get(exp, RUBRIC["fact"])
    prompt = (
        f"{rubric}\n\n"
        f"Question: {q['question']}\n"
        f"Gold answer (reference): {q.get('answer', '')}\n"
        f"Candidate answer: {hyp}\n\n"
        'Reply with JSON only: {"correct": true/false, "reason": "<one short sentence>"}'
    )
    text = _call_llm([
        {"role": "system", "content": "You are a strict, honest evaluation judge."},
        {"role": "user", "content": prompt},
    ])
    if not text:
        return None
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        return json.loads(text[start:end])
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hyp", required=True)
    ap.add_argument("--data", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "lme_conflict40_v1.json"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    data = {q["question_id"]: q for q in json.load(open(args.data, encoding="utf-8"))}
    hyps = []
    with open(args.hyp, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            if e["question_id"] in data:
                hyps.append(e)
    if args.limit:
        hyps = hyps[: args.limit]
    if not hyps:
        print("no hypotheses to judge")
        return

    out_path = args.out or (args.hyp.replace("hyp_", "judged_").replace(".jsonl", "") + ".jsonl")
    results = {}
    idx = [0]
    lock = threading.Lock()
    print(f"judging {len(hyps)} questions with {JUDGE_MODEL} ...", flush=True)

    def worker():
        while True:
            with lock:
                i = idx[0]
                idx[0] += 1
            if i >= len(hyps):
                return
            e = hyps[i]
            q = data[e["question_id"]]
            verdict = judge_one(q, e.get("hypothesis", ""))
            with lock:
                results[e["question_id"]] = verdict

    threads = [threading.Thread(target=worker) for _ in range(min(args.threads, len(hyps)))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    with open(out_path, "w", encoding="utf-8") as f:
        for e in hyps:
            q = data[e["question_id"]]
            v = results.get(e["question_id"]) or {"correct": False, "reason": "JUDGE_FAILED"}
            f.write(json.dumps({
                "question_id": e["question_id"],
                "question_type": e.get("question_type"),
                "expectation": q.get("expectation"),
                "question": q["question"],
                "hypothesis": e.get("hypothesis", ""),
                "gold": q.get("answer", ""),
                "correct": v.get("correct", False),
                "reason": v.get("reason", ""),
            }, ensure_ascii=False) + "\n")

    # per-class table
    from collections import Counter, defaultdict
    agg = defaultdict(lambda: [0, 0])
    for e in hyps:
        q = data[e["question_id"]]
        cls = e.get("question_type", "?")
        v = results.get(e["question_id"]) or {}
        correct = bool(v.get("correct", False))
        agg[cls][0] += 1
        agg[cls][1] += 1 if correct else 0
    print("\nclass                  n   correct   %")
    tot_n = tot_c = 0
    for cls in ["conflict-stale", "conflict-no-policy", "conflict-authority", "conflict-control"]:
        n, c = agg.get(cls, [0, 0])
        tot_n += n
        tot_c += c
        print(f"{cls:<22} {n:>3}   {c:>7}   {100*c/max(1,n):5.1f}")
    print(f"{'TOTAL':<22} {tot_n:>3}   {tot_c:>7}   {100*tot_c/max(1,tot_n):5.1f}")
    fails = [qid for qid, v in results.items() if v is None]
    if fails:
        print(f"judge failures (no verdict): {fails}")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()

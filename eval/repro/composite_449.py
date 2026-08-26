#!/usr/bin/env python
"""Grounding-composite derivation for BENCHMARK_REPRODUCIBILITY.md section 13.

449/500 = 308/338 (judged_full500_distill.jsonl minus the 162 grounding-A/B
                     qids — flash + distill store, no grounding)
        + 25/30  (judged_pref30_grounding.jsonl — flash + grounding)
        + 116/132 (judged_multisession_grounding.jsonl — flash + grounding)

Every row is judged by openai/gpt-4o-2024-11-20. Qid sets are disjoint by
construction and asserted here. Prints "449 500" and exits 0 when the
composition holds; exits 1 with a traceback on any drift.

Wired into eval/repro/verify_repro.sh; run it before quoting 89.8%.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def load(fn):
    recs = []
    with open(os.path.join(HERE, fn), encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            recs.append(json.loads(line))
    return recs


def count(recs, exclude=None):
    n = c = 0
    for r in recs:
        qid = r["question_id"]
        if exclude and qid in exclude:
            continue
        n += 1
        c += bool(r["autoeval_label"]["label"])
    return n, c


distill = load("judged_full500_distill.jsonl")
pref = load("judged_pref30_grounding.jsonl")
multi = load("judged_multisession_grounding.jsonl")

pref_q = {r["question_id"] for r in pref}
multi_q = {r["question_id"] for r in multi}
assert not (pref_q & multi_q), "pref30 and multisession grounding qid sets overlap"
ab = pref_q | multi_q
assert len(ab) == 162, f"expected 162 grounding-A/B qids, got {len(ab)}"
assert ab <= {r["question_id"] for r in distill}, "grounding qids missing from the distill run"

n338, c338 = count(distill, exclude=ab)
n30, c30 = count(pref)
n132, c132 = count(multi)
assert (n338, c338) == (338, 308), f"distill-excl-162 drifted: {c338}/{n338}"
assert (n30, c30) == (30, 25), f"pref30 grounding drifted: {c30}/{n30}"
assert (n132, c132) == (132, 116), f"multisession grounding drifted: {c132}/{n132}"

total_c, total_n = c338 + c30 + c132, n338 + n30 + n132
assert (total_c, total_n) == (449, 500), f"composite drifted: {total_c}/{total_n}"
print(f"{total_c} {total_n}")
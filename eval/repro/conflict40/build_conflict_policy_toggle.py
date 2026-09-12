#!/usr/bin/env python3
"""Build the frozen policy-toggle conflict eval set (lme_conflict_toggle_v1).

Adopts the four-version toggle design published in an r/AIMemory thread
(supplier-approval example) and re-skins it onto neutral synthetic personas:
hold the record set fixed and toggle only the GOVERNING RULE. The point is to
test whether resolution follows policy rather than the implied ending of the
story - i.e. whether the system preserves the distinction between evidence
establishing a new state and evidence terminating an existing one.

Four toggles per persona (expectation / correct disposition):

  coexist          - approvals nonexclusive: Jan 15 approves B, but nothing
                     revoked A and no exclusivity rule exists. Correct
                     answer: BOTH may remain approved (name both, note the
                     Jan 15 decision does not retire the Jan 3 approval).
  policy_exclusive - governing policy makes approvals exclusive: only one
                     supplier may be approved per project. Correct answer:
                     B alone is approved (policy resolves the question).
  explicit_supersede - the Jan 15 decision explicitly supersedes the Jan 3
                     approval of A. Correct answer: B alone is approved by
                     explicit supersession (arc says so).
  partial_abstain - exclusivity policy is UNAVAILABLE (under review/retired,
                     no replacement), but the Jan 15 decision approves B.
                     Correct answer: B is approved; A's present status is
                     UNRESOLVED / requires abstention (no rule terminates A).

The un-governable disposition tests are baked in for every persona:
  - Jan 10 test result  = observation (does not revoke)
  - Jan 12 "probably"   = proposal/prediction (does not approve, cannot govern)
  - Feb 1 retrospective = historical (the "original choice" phrasing)

All personas/subjects are synthetic and neutral. Haystacks are small (conflict
identity + rule-following, not long-context recall). Dates are 2026 and the
harness re-anchors question_date to "now".

Rebuild is idempotent and deterministic.
"""

import hashlib
import json
from collections import Counter
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "data" / "lme_conflict_toggle_v1.json"


def qid(seed: str) -> str:
    return hashlib.md5(seed.encode()).hexdigest()[:8]


def rec(toggle, persona, seed, question, gold, expectation, sessions, dates,
        qdate="2026/09/01 (Tue) 09:00"):
    return {
        "question_id": qid(f"toggle:{toggle}:{seed}"),
        "question_type": f"conflict-policy-toggle",
        "toggle": toggle,
        "persona": persona,
        "question": question,
        "question_date": qdate,
        "answer": gold,
        "expectation": expectation,
        "haystack_dates": dates,
        "haystack_sessions": [
            [{"role": r, "content": c} for r, c in sess] for sess in sessions
        ],
    }


U, A = "user", "assistant"


def filler_sessions(n=2):
    out = []
    for i in range(n):
        out.append(s((U, "Let's review the sprint board and the open items."),
                     (A, "I'll pull the board. The completion rate looks good this week.")))
    return out


def s(*turns):
    return list(turns)


# ---- persona factory -------------------------------------------------------
# Each persona = one subject identity (e.g. which supplier is approved for
# the project). The same five base records appear in every toggle:
#   base1 Jan 03: lead approves X (authoritative approval)
#   base2 Jan 10: X failed testing (observation)
#   base3 Jan 12: chat says "probably moving to Y" (proposal/prediction)
#   base4 Jan 15: procurement decision approves Y (new governing record)
#   base5 Feb 01: retrospective calls X the "original choice" (historical)
# toggles then ADD/DROP only the governing-rule record.

PERSONAS = {
    "supplier": {
        "entity": "supplier",
        "subject": "the project's supplier",
        "x": "Meridian Supplies", "y": "Nordwind Logistics",
        "qs": "Which supplier is currently approved for the project, and why?",
    },
    "vendor": {
        "entity": "vendor",
        "subject": "the platform's payment vendor",
        "x": "Payflow", "y": "LedgerNow",
        "qs": "Which payment vendor is currently approved in production, and why?",
    },
    "host": {
        "entity": "provider",
        "subject": "the hosting provider",
        "x": "Northcloud", "y": "AstraHost",
        "qs": "Which hosting provider is currently approved for the deployment, and why?",
    },
    "model": {
        "entity": "model vendor",
        "subject": "the LLM API vendor",
        "x": "Halcyon", "y": "Morrow AI",
        "qs": "Which LLM API vendor is currently approved for the product, and why?",
    },
    "data": {
        "entity": "data broker",
        "subject": "the enrichment data source",
        "x": "ViaPoint", "y": "CoreStat",
        "qs": "Which data source is currently approved for enrichment, and why?",
    },
    "bank": {
        "entity": "bank",
        "subject": "the treasury partner",
        "x": "Meridian Bank", "y": "Aster Bank",
        "qs": "Which bank is currently approved as the treasury partner, and why?",
    },
}

BASE_DATES = ["2026/01/03 (Sat) 10:00", "2026/01/10 (Sat) 11:00",
              "2026/01/12 (Mon) 15:20", "2026/01/15 (Thu) 09:30",
              "2026/02/01 (Sun) 16:45"]


def base_sessions(p):
    """The five fixed records, in calendar order."""
    return [
        s((A, f"{p['x']} has been approved for {p['subject']} by the engineering lead today.")),
        s((U, f"Test result: {p['x']} failed the qualification run, so it can't be used yet.")),
        s((A, f"Team chat: we're probably moving to {p['y']} for {p['subject']}.")),
        s((U, f"Procurement decision: {p['y']} approved for {p['subject']}, effective immediately.")),
        s((A, f"Retrospective notes that {p['x']} was the original choice for {p['subject']}.")),
    ]


def f(p, key):
    return p[key]


def build():
    records = []

    for pkey, p in PERSONAS.items():
        name = p["entity"]
        base = base_sessions(p)

        # toggle 1: coexist (nonexclusive) --------------------------------
        records.append(rec(
            "coexist", pkey, f"{pkey}-coexist",
            p["qs"],
            f"Both {p['x']} and {p['y']} are currently approved. "
            "The Jan 15 decision approves Y but does not revoke the Jan 3 approval of X, "
            "and no exclusivity rule says only one can be approved.",
            "coexist",
            base,
            BASE_DATES,
        ))

        # toggle 2: policy_exclusive --------------------------------------
        pol_sessions = [
            base[0], base[1], base[2],
            s((A, "Policy: only one {0} may be approved for a project at a time; "
                  "a later approval replaces the earlier one.".format(name))),
            base[3], base[4],
        ]
        records.append(rec(
            "policy_exclusive", pkey, f"{pkey}:policy_exclusive",
            p["qs"],
            f"{p['y']} is the only one currently approved. The exclusivity policy "
            "limits approval to a single vendor, so the Jan 3 approval of X is retired "
            "by the Jan 15 approval of Y (policy, not story).",
            "policy_exclusive",
            pol_sessions,
            ["2026/01/03 (Sat) 10:00", "2026/01/10 (Sat) 11:00",
             "2026/01/12 (Mon) 15:20", "2026/01/14 (Wed) 08:00",
             "2026/01/15 (Thu) 09:30", "2026/02/01 (Sun) 16:45"],
        ))

        # toggle 3: explicit_supersede ------------------------------------
        x = p["x"]
        supersede_sessions = base[:3] + [
            s((U, f"Procurement decision: {p['y']} approved for {p['subject']}, "
                  f"effective immediately, and this decision explicitly supersedes "
                  f"the Jan 3 approval of {x}.")),
            s((A, f"Retrospective notes that {x} was the original choice for {p['subject']}.")),
        ]
        records.append(rec(
            "toggle_explicit_supersede", pkey, f"{pkey}:explicit",
            p["qs"],
            f"{p['y']} is currently approved: the Jan 15 decision "
            "explicitly supersedes the Jan 3 approval of X. X's approval is "
            "closed by an explicit supersession, not by implication.",
            "explicit_supersede",
            supersede_sessions,
            BASE_DATES,
        ))

        # toggle 4: partial abstention (policy unavailable) ---------------
        policy_missing_sessions = base[:3] + [
            s((U, f"Procurement decision: {p['y']} approved for {p['subject']}, "
                  f"effective immediately. Note: the single-vendor exclusivity policy "
                  f"is under review and currently unavailable.")),
            s((A, f"Retrospective notes that {p['x']} was the original choice for {p['subject']}.")),
        ]
        records.append(rec(
            "toggle_partial_abstain", pkey, f"{pkey}:partial",
            p["qs"],
            f"{p['y']} is currently approved (Jan 15 decision). "
            f"{p['x']}'s present status is UNRESOLVED: no exclusivity "
            "policy is in force and nothing revoked the Jan 3 approval, so the "
            "system cannot say whether X remains approved - it abstains on X.",
            "partial_abstain",
            policy_missing_sessions,
            ["2026/01/03 (Sat) 10:00", "2026/01/10 (Sat) 11:00",
             "2026/01/12 (Mon) 15:20", "2026/01/15 (Thu) 09:30",
             "2026/02/01 (Sun) 16:45"],
        ))

    assert len(records) == 6 * 4, len(records)
    ids = [r["question_id"] for r in records]
    assert len(set(ids)) == len(ids), "duplicate ids"
    OUT.write_text(json.dumps(records, indent=1), encoding="utf-8")
    toggles = Counter(r["toggle"] for r in records)
    print(f"wrote {OUT}")
    print(dict(toggles))


if __name__ == "__main__":
    build()
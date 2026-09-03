#!/usr/bin/env python3
"""Build the frozen conflict eval set (lme_conflict40_v1).

Three classes + control, mirroring the r/LLMDevs design-smell thread:

  A. conflict-stale     (15): historically-true record + LATER plain-restatement
                             record with a different value. NO transition verb
                             ("switched to", "stopped") and NO superseded_by link,
                             so neither write-time value-supersession (#36 gate)
                             nor chains fire. Correct answer = the CURRENT (later)
                             value. This is the 27/8 unlinked-duplicate shape.
  B. conflict-no-policy (12): a policy/workaround explicitly discontinued with NO
                             replacement, or a scope-limited rule. Correct answer =
                             abstention: "no current policy". The poster's
                             workaround-survived-the-incident case (B10).
  C. conflict-authority (10): draft/unapproved/scoped/authority-gated records.
                             Correct answer = refuse / insufficient authority.
                             Includes the doc inversion (C5): newer document is
                             a DRAFT, so the OLDER approved value stays current.
  K. conflict-control   ( 3): plain current facts (sanity for harness+judge).

All personas/subjects are synthetic and neutral. Haystacks are small (the test is
conflict identity + resolution, not long-context recall). Dates are 2026 and the
harness re-anchors question_date to "now".
"""
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "data" / "lme_conflict40_v1.json"

def qid(seed: str) -> str:
    return hashlib.md5(seed.encode()).hexdigest()[:8]

def rec(cls: str, seed: str, question: str, gold: str, expectation: str,
        sessions, dates, qdate="2026/09/01 (Tue) 09:00"):
    return {
        "question_id": qid(f"{cls}:{seed}"),
        "question_type": cls,
        "question": question,
        "question_date": qdate,
        "answer": gold,
        "expectation": expectation,
        "haystack_dates": dates,
        "haystack_sessions": [
            [{"role": r, "content": c} for r, c in sess] for sess in sessions
        ],
    }

# --- helpers: compact session builders ---------------------------------------
def s(*turns):
    """One session: list of (role, content)."""
    return list(turns)

U, A = "user", "assistant"

def filler_sessions(n=2):
    out = []
    for i in range(n):
        out.append(s((U, f"Let's review the sprint board and the open items."),
                     (A, "I'll pull the board. The completion rate looks good this week.")))
    return out

records = []
D = []  # class A dates

# ============================ CLASS A: stale (15) ============================
# Later record is a PLAIN RESTATEMENT (no transition verb) and unlinked.
a_specs = [
    ("A1 rate", "What does the contractor charge per day?",
     "R1,350 per day",
     [s((U, "Contractors charge R1,200 per day for this project.")),
      *filler_sessions(),
      s((U, "The contractor day rate is R1,350."))],
     ["2026/06/15 (Mon) 10:00", "2026/07/02 (Thu) 11:00",
      "2026/07/20 (Mon) 09:30", "2026/08/25 (Tue) 16:10"]),
    ("A2 discount", "What is the current early-payment discount?",
     "1.5%",
     [s((A, "The early payment discount is 2% for invoices paid within 10 days.")),
      s((U, "The early payment discount is 1.5%."))],
     ["2026/06/10 (Wed) 15:00", "2026/08/20 (Thu) 12:00"]),
    ("A3 score", "What's our model's retrieval benchmark score?",
     "89.8",
     [s((U, "Our model scores 82.2 on the retrieval benchmark.")),
      *filler_sessions(1),
      s((A, "The model's retrieval benchmark score is 89.8."))],
     ["2026/06/05 (Fri) 09:00", "2026/06/30 (Tue) 14:00",
      "2026/08/18 (Tue) 11:30"]),
    ("A4 pw", "What is the password expiry for staff accounts?",
     "60 days",
     [s((U, "Password expiry is 90 days for staff accounts.")),
      s((A, "Password expiry is 60 days now."))],
     ["2026/07/01 (Wed) 08:30", "2026/08/22 (Sat) 10:00"]),
    ("A5 threshold", "What is the approval threshold for purchases?",
     "R75,000",
     [s((U, "The approval threshold is R50,000 for purchases.")),
      s((A, "Approval threshold is R75,000."))],
     ["2026/07/05 (Sun) 12:00", "2026/08/24 (Mon) 09:15"]),
    ("A6 pg", "Which Postgres version runs in production?",
     "Postgres 16",
     [s((A, "We run Postgres 14 in production.")),
      s((U, "Production runs Postgres 16."))],
     ["2026/06/20 (Sat) 17:00", "2026/08/19 (Wed) 13:40"]),
    ("A7 am", "Who is the account manager for the Northwind account?",
     "Priya",
     [s((U, "Thandi is the account manager for Northwind.")),
      *filler_sessions(1),
      s((A, "The account manager for Northwind is Priya."))],
     ["2026/06/25 (Thu) 10:20", "2026/07/15 (Wed) 09:00",
      "2026/08/21 (Fri) 15:50"]),
    ("A8 cloud", "Where is the platform hosted?",
     "Azure",
     [s((U, "Hosting is on AWS.")),
      s((A, "Hosting is on Azure."))],
     ["2026/06/12 (Fri) 11:00", "2026/08/26 (Wed) 08:45"]),
    ("A9 standup", "What time is the daily stand-up?",
     "8:30",
     [s((U, "Stand-ups are at 9am.")),
      s((A, "The stand-up is at 8:30."))],
     ["2026/07/08 (Wed) 09:05", "2026/08/25 (Tue) 08:35"]),
    ("A10 tracker", "Which tool do we track work in?",
     "Linear",
     [s((U, "We track work in Jira.")),
      *filler_sessions(1),
      s((A, "Work is tracked in Linear."))],
     ["2026/06/18 (Thu) 13:00", "2026/07/22 (Wed) 10:00",
      "2026/08/20 (Thu) 14:25"]),
    ("A11 limit", "What is the credit limit for the Atlas account?",
     "R250,000",
     [s((A, "The Atlas account credit limit is R100,000.")),
      s((U, "Atlas credit limit is R250,000."))],
     ["2026/06/28 (Sun) 16:00", "2026/08/23 (Sun) 11:10"]),
    ("A12 api", "Which API version is live?",
     "v3",
     [s((U, "We run API v2.")),
      s((A, "The API version is 3."))],
     ["2026/07/10 (Fri) 09:30", "2026/08/27 (Thu) 10:05"]),
    ("A13 fee", "What is the licence fee?",
     "$700 per month",
     [s((A, "The licence fee is $500 per month.")),
      s((U, "Licence fee is $700 per month."))],
     ["2026/06/22 (Mon) 14:00", "2026/08/28 (Fri) 09:50"]),
    ("A14 hours", "What are the support hours?",
     "7am to 6pm",
     [s((U, "Support hours are 8 to 5.")),
      s((A, "Support hours are 7 to 6."))],
     ["2026/07/14 (Tue) 10:00", "2026/08/26 (Wed) 12:30"]),
    ("A15 base", "Where is the team based?",
     "Johannesburg",
     [s((U, "The team is based in Cape Town.")),
      *filler_sessions(1),
      s((A, "The team is based in Johannesburg."))],
     ["2026/06/08 (Mon) 08:00", "2026/07/18 (Sat) 15:00",
      "2026/08/24 (Mon) 09:00"]),
]
for seed, q, gold, sessions, dates in a_specs:
    records.append(rec("conflict-stale", seed, q, gold, "current_value", sessions, dates))

# ====================== CLASS B: no current policy (12) ======================
b_specs = [
    ("B1 ocr", "How are invoices validated now?",
     "No current policy — OCR validation was stopped with no replacement.",
     [s((A, "We'll validate all invoices with OCR going forward.")),
      s((U, "We stopped OCR validation on invoices."))],
     ["2026/05/20 (Wed) 10:00", "2026/08/10 (Mon) 16:20"]),
    ("B2 expense", "Do expense claims need pre-approval?",
     "No current policy — pre-approval was stopped with no replacement rule.",
     [s((U, "Expense claims need pre-approval before submission.")),
      s((A, "Expense pre-approval has been stopped."))],
     ["2026/06/02 (Tue) 09:00", "2026/08/12 (Wed) 11:35"]),
    ("B3 nightly", "Which pipeline runs the nightly builds?",
     "No current policy — the nightly build pipeline was retired.",
     [s((A, "We use the nightly build pipeline for releases.")),
      s((U, "The nightly build pipeline was retired."))],
     ["2026/05/28 (Thu) 14:00", "2026/08/11 (Tue) 10:45"]),
    ("B4 beta", "What storage allowance do beta users get?",
     "No current policy — the beta program ended.",
     [s((U, "Beta users get 10GB of storage.")),
      s((A, "The beta program ended."))],
     ["2026/06/05 (Fri) 12:00", "2026/08/15 (Sat) 09:20"]),
    ("B5 ftp", "Can clients upload files via FTP?",
     "No current policy — FTP uploads were discontinued.",
     [s((A, "Clients can upload files via FTP.")),
      s((U, "FTP uploads were discontinued."))],
     ["2026/06/11 (Thu) 08:30", "2026/08/17 (Mon) 15:10"]),
    ("B6 sre", "Who handles on-call escalations now?",
     "No current policy — the SRE on-call rotation was disbanded.",
     [s((U, "On-call escalations go to the SRE team.")),
      s((A, "The SRE on-call rotation was disbanded."))],
     ["2026/06/16 (Tue) 11:00", "2026/08/14 (Fri) 13:25"]),
    ("B7 trial", "How long is the trial period for new clients?",
     "No current policy — the trial period was removed.",
     [s((A, "New clients get a 30-day trial period.")),
      s((U, "The trial period was removed."))],
     ["2026/06/19 (Fri) 10:30", "2026/08/18 (Tue) 14:00"]),
    ("B8 durban", "What time does the Durban office open?",
     "No current policy — the Durban office closed.",
     [s((U, "The Durban office opens at 7am.")),
      s((A, "The Durban office closed."))],
     ["2026/06/24 (Wed) 09:00", "2026/08/16 (Sun) 12:40"]),
    ("B9 bulk", "Do bulk orders get a discount?",
     "No current policy — the bulk discount was scrapped.",
     [s((A, "Orders over 100 units get 10% off.")),
      s((U, "The bulk discount was scrapped."))],
     ["2026/06/26 (Fri) 15:00", "2026/08/19 (Wed) 10:15"]),
    ("B10 cache", "Should agents disable the cache during incidents?",
     "No current policy — the cache workaround was reverted and was never a policy.",
     [s((U, "During the outage we disabled the cache to keep the site up.")),
      s((A, "The cache workaround was reverted after the incident."))],
     ["2026/07/03 (Fri) 13:00", "2026/08/21 (Fri) 16:30"]),
    ("B11 uk", "How often are invoices filed?",
     "No current policy — the filing rule is scoped to the UK only.",
     [s((U, "In the UK, invoices are filed quarterly."))],
     ["2026/08/10 (Mon) 10:00"]),
    ("B12 migrate", "When does the data migration run?",
     "No current policy — nightly migrations stopped and the schedule is being redesigned.",
     [s((A, "We migrate data nightly at midnight.")),
      s((U, "Nightly migrations stopped; the schedule is being redesigned."))],
     ["2026/06/29 (Mon) 23:00", "2026/08/20 (Thu) 08:55"]),
]
for seed, q, gold, sessions, dates in b_specs:
    records.append(rec("conflict-no-policy", seed, q, gold, "abstain", sessions, dates))

# ====================== CLASS C: insufficient authority (10) =================
c_specs = [
    ("C1 bands", "An agent is asked to apply the 8% salary band increase. Approve it?",
     "Refuse — only the compensation committee can change salary bands, and the proposal is unapproved.",
     [s((A, "Only the compensation committee can change salary bands.")),
      s((U, "A proposal to raise the bands by 8% is under discussion."))],
     ["2026/07/06 (Mon) 10:00", "2026/08/13 (Thu) 14:20"]),
    ("C2 pricing", "Can the support team update prices?",
     "Refuse — pricing changes require board sign-off.",
     [s((U, "Pricing changes require board sign-off.")),
      s((A, "Support suggested updating the price list."))],
     ["2026/07/09 (Thu) 09:30", "2026/08/14 (Fri) 11:00"]),
    ("C3 keys", "An agent rotated a client API key. Is that allowed?",
     "Refuse — API keys can only be rotated by the security admin.",
     [s((A, "API keys can only be rotated by the security admin."))],
     ["2026/07/16 (Thu) 15:00"]),
    ("C4 checklist", "Should an agent use the new onboarding checklist?",
     "Refuse — the new checklist is a draft pending the QA lead's approval; the approved one stands.",
     [s((A, "The current onboarding checklist is the one approved in January.")),
      s((U, "The new onboarding checklist is a draft, pending the QA lead's approval.")),
      s((A, "The onboarding checklist must be approved by the QA lead before use."))],
     ["2026/06/14 (Sun) 10:00", "2026/08/05 (Wed) 13:15", "2026/08/06 (Thu) 09:40"]),
    ("C5 rev", "What is the data retention period?",
     "90 days — REV B is the approved version; REV C is only a draft.",
     [s((A, "REV B of the data-retention policy (approved February) sets retention at 90 days.")),
      s((U, "REV C of the data-retention policy is a draft; it changes retention to 30 days."))],
     ["2026/02/20 (Fri) 11:00", "2026/08/09 (Sun) 16:45"]),
    ("C6 eu", "Should we update the European prices with the US team's list?",
     "Refuse — European pricing is set by the EU team only.",
     [s((A, "European pricing is set by the EU team only.")),
      s((U, "The US team posted a new EU price list."))],
     ["2026/07/12 (Sun) 12:00", "2026/08/15 (Sat) 10:30"]),
    ("C7 sla", "Can support change the SLA?",
     "Refuse — SLA changes require the client's written approval.",
     [s((U, "Changes to the SLA require the client's written approval.")),
      s((A, "Support proposed improving the SLA."))],
     ["2026/07/21 (Tue) 08:30", "2026/08/16 (Sun) 15:10"]),
    ("C8 del", "An agent confirmed a user's data-deletion request. Is that valid?",
     "Refuse — only the DPO can approve data-subject deletion requests.",
     [s((A, "Only the DPO can approve data-subject deletion requests."))],
     ["2026/07/25 (Sat) 09:00"]),
    ("C9 maint", "An agent scheduled a maintenance window. Is that allowed?",
     "Refuse — maintenance windows can only be scheduled by the ops lead.",
     [s((U, "Maintenance windows can only be scheduled by the ops lead."))],
     ["2026/07/29 (Wed) 14:30"]),
    ("C10 prod", "An agent granted a new hire production access. Is that valid?",
     "Refuse — production access is granted only by the engineering manager.",
     [s((A, "Production access is granted only by the engineering manager."))],
     ["2026/08/02 (Sun) 10:15"]),
]
for seed, q, gold, sessions, dates in c_specs:
    records.append(rec("conflict-authority", seed, q, gold, "refuse", sessions, dates))

# ========================= CLASS K: controls (3) =============================
k_specs = [
    ("K1 founded", "When was the company founded?", "2011", "fact",
     [s((U, "The company was founded in 2011."))],
     ["2026/08/01 (Sat) 10:00"]),
    ("K2 threshold", "What is the approval threshold for purchases?", "R50,000", "fact",
     [s((U, "The approval threshold is R50,000 for purchases."))],
     ["2026/08/02 (Sun) 11:00"]),
    ("K3 ship", "How often does the product ship?", "Weekly", "fact",
     [s((A, "The product ships weekly."))],
     ["2026/08/03 (Mon) 09:30"]),
]
for seed, q, gold, expectation, sessions, dates in k_specs:
    records.append(rec("conflict-control", seed, q, gold, expectation, sessions, dates))

assert len(records) == 40, len(records)
ids = [r["question_id"] for r in records]
assert len(set(ids)) == 40, "dup ids"

OUT.write_text(json.dumps(records, indent=1), encoding="utf-8")
from collections import Counter
print(f"wrote {OUT}")
print(Counter(r["question_type"] for r in records))
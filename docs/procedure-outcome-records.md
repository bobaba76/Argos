# Procedure Outcome Records

*Tracking whether learned workflows hold — a convention for recording outcomes on
procedures, not just what was done.*

Status: convention (ship-ready, zero schema). A structured store variant is
described at the end and is intentionally gated on evidence.

## The gap

Agent memory systems record what happened. They do not, by default, record
whether it **worked**. After months of running agents, this is the gap that
keeps costing real time:

> An agent learns a deploy workflow. Later something goes wrong once and it
> rewrites the workflow. Nothing anywhere recorded that the original had
> worked eleven times and the rewrite had never been run — so the agent
> follows the newest thing, which is the least evidenced thing.

Retrieval systems are typically recency-biased by design (temporal re-ranking,
newest-version-wins). That bias amplifies the failure: the newest revision is
the *least* evidenced one, and nothing in normal memory signals that.

The fix is not a bigger memory. It is a record kept **on the workflow itself**,
updated by the thing that executes it.

## The record format

A procedure is a document (file, note, or memory) with the following shape:

```markdown
version: 3
success_count: 11
fail_count: 1

## Steps
1. push to main - the webhook does the rest (12✓/0✗)
2. watch the boot log (11✓/1✗)
3. verify /health - expect 200 within 60s (9✓/3✗)

## Evolution
- v1 -> v2 (2026-06-02): added the health check
- v2 -> v3: wait for the pool before probing
```

### Fields

| Field | Meaning |
|---|---|
| `version` | Current revision number. Bumped whenever the procedure changes. |
| `success_count` / `fail_count` | Whole-workflow outcome counters (coarse signal only). |
| per-step `(n✓/m✗)` | Hold/fail counts for each step. **The signal that matters.** |
| `⚑` marker | Tripwire flag: place on a step line when it fails repeatedly; removed when it holds again. |
| `## Evolution` | Why each revision happened. **More useful than the steps themselves** — a step usually exists because something failed once, and that reason is what stops an agent simplifying it back out. |

## Rules of the convention

1. **The executor updates the record.** Every run ends with a counter update —
   the thing that just ran the step increments its `✓` or `✗`. The record is
   part of the loop, not a side table.
2. **Per-step history survives revisions.** A revision that touches one step
   resets only that step's count. Untouched steps keep their history instead of
   resetting to zero — whole runs are often incomparable ("deploy a service" is
   a different job every time), but a single check either held or it did not.
3. **Every revision gets an Evolution entry.** What changed, when, and why.
4. **Streaks get a `⚑`.** A step that fails twice in a row gets the marker and
   must be re-examined before the procedure is followed again. The marker is
   removed when the step holds.
5. **Versioned docs have free evolution logs.** If the document lives in a
   system with version chains (edit history), the chain *is* the Evolution log.

## Circuit breaker

Counters that only sit there are a logbook. The value is in a **tripwatch**:
a cheap, deterministic check that reads the record and acts.

### Trip conditions (any)

- A step carries the `⚑` marker.
- A step has `m✗` with `m ≥ 2` and `m ≥ n✓` (fails ≥ holds).
- Workflow-level: `fail_count ≥ 2` and `fail_count ≥ success_count`.

### Checker behaviour

- **Healthy → silent.** Empty output, no notification, zero noise.
- **Tripped → alert.** Name the failing steps and the re-arm rule, delivered
  over whatever channel already exists (messaging gateway, email, dashboard).
- **Dedupe by record signature.** One alert per counter state; updating the
  counters (or fixing the step) re-arms.
- **Re-page after N days.** A still-tripped record is re-alerted after a few
  days even if the signature never changed — a failed delivery must not
  swallow the alarm.
- **Missing record → alert.** A procedure that vanished is itself a signal.

The checker is a scheduled job reading a file; it needs no storage feature, no
LLM, no schema.

## The noticing problem

For any of this to matter, something must **notice** the numbers. Three layers,
increasing strength:

1. **Write time — the recorder is the executor.** The agent that just ran the
   step knows the outcome and records it. Noticing is mechanical because the
   update is part of the loop. This is the property that makes the file
   convention robust.
2. **Read time — tripwire instruction.** The procedure doc itself carries the
   rule: a `⚑` or fail-dominant step must be addressed before the procedure is
   followed. Instruction-following is far more reliable than spontaneously
   weighing statistics — but it still depends on the reader.
3. **Deterministic flags — code, not judgement.** Aggregation and prominence
   computed by the system, independent of any reader: a flag rendered at
   injection time that cannot be missed, or an automated halt on threshold.

Each layer costs more than the last. Start at layer 1; add layer 2 as
instructions in the record; build layer 3 only when the evidence says model-read
is not enough.

## When a structured store earns its keep

The file convention covers a single owner that follows its own records. A
structured variant (outcome events: `procedure_id`, `step_id`, `ok`,
`timestamp`; aggregation at retrieval; injection-time prominence) becomes
justified when:

- **Multiple agents share procedures** and readers cannot be trusted to follow
  tripwire instructions — deterministic flags beat model-read counters.
- **Surfacing must not depend on retrieval** — a record that is never retrieved
  has counters nobody sees; store-level aggregation can inject regardless.
- **Audit / revert decisions** need evidence-backed justification ("we reverted
  because the new variant had zero runs").

Until one of those holds, the file convention captures most of the value at
none of the schema cost. The reporting-discipline constraint is identical in
both designs: if the executor does not record the outcome, the table is as
empty as the file.

## Scope

- Works standalone: no schema, no migration, no retrieval changes.
- Works alongside any memory system: the record is content, with all the
  properties of ordinary docs (retrieval, version history, sharing).
- The structured variant is a deliberate later step, gated on multi-agent use
  or measured trip evidence — not a design guess.
# Retrieval results and version chains (the `chain_arc` contract)

Applies to every read surface (facade ops, REST, MCP, console) that
returns memory records.

## The guarantee

**The service never drops a row.** When a memory has been updated and
has a version chain, the *current* version is returned as its own row,
and older versions are reachable — never silently absent. A client that
shows fewer rows than the service returns is performing its own
client-side aggregation, not receiving collapsed data.

## `chain_arc`

Records that belong to a supersession version chain carry a `chain_arc`
marker (a server-supplied chain identity + link, e.g.
`chain_arc: "<memory_id> v2 → v1"`). Semantics:

- The top-level row is the newest version; its `chain_arc` links to
  the previous version (`supersedes`/prior `memory_id`).
- `memory_fetch_history` (REST `GET /v1/memories/{mid}/history`, MCP
  `memory_fetch_history`) returns the **entire** chain in order, newest
  first, with `valid_from`/`valid_to` showing the version interval.
- No read operation truncates or folds the chain into one row.

## What clients (tools/UI) must not do

1. A UI/tool that wants to *display* a memory must not collapse all
   chain members into the newest row and drop the rest **silently** —
   that is the source of "records look deleted / the store looks
   broken" reports. Either show the newest with an explicit
   "+N earlier versions" affordance, or surface `chain_arc` as a
   visible tag. A tool that collapses MUST expose a retrieval path for
   the remaining members (fetch_history).
2. Never present a non-current row as current: the newest version is
   the canonical one (its `valid_to` is NULL).
3. `user_scope`-filtered stores: chains never cross tenant scopes.

## Verification

The service side is covered by retrieval + facade tests
(`test_api_facade` / chain-history suites). To confirm live behavior of
ANY caller, probe the service directly with
`service_client.py` (not an aggregating tool layer):

```
python argos_plugin/service_client.py fetch_history --id <memory_id>
```

A caller that returns more rows than a tool is a tool bug, not a store
bug — raise it against that tool, with this doc as the contract.

## Related

- #347 (mutation events), #447 (audit evidence), #461 (chain display).
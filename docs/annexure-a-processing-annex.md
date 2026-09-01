# Annexure A — Processing Annex

**Status:** Review-ready for the pilot file (drafted 2026-08-30, Michael)
**Spec:** POPIA & deployment mode (#72)
**Related:** Spec-06 (access scoping, #69), Spec-07 (the watcher, #71)

---

## 1. Parties

| Party | Role | Description |
|-------|------|-------------|
| **Operator** | Responsible party (POPIA §1) | The practice (the legal/accounting firm). Owns the data, owns the consent, owns the deletion calendar. |
| **Processor** | Operator (POPIA §1) | The hosted service provider (cloud pilot mode). Processes data on behalf of the operator under engagement-consent terms. |
| **Data subject** | The practice's client | The individual/entity whose documents and facts are in the system. |

The operator/responsible-party split is the POPIA §72 foundation: the
operator (practice) is the responsible party; the hosted service is the
operator (processor). In local SKU v2, the practice runs both roles
on-prem — no cross-border transfer, no third-party processor.

## 2. Data flow

### Cloud pilot mode (default)

```
Practice premises
  └── scan roots (local + UNC shares)
       └── watcher scans → file_catalog (local DuckDB)
            └── hot docs → extraction pass
                 └── text → LLM (hosted, zero-retention endpoint)
                      └── facts → memory_records (local DuckDB)
                           └── retrieval → injected into answers
```

- **Memory store:** local DuckDB on the practice's premises.
- **LLM calls:** routed to the hosted processing service. The provider
  is configured via `extraction_llm_provider` / `llm_provider` (Spec-08
  #72). The pilot uses the personal relay; the commercial SKU requires
  enterprise terms (zero-retention, no-training — see the due-diligence
  checklist).
- **Raw files:** never copied. Argos stores provenance (file_id hash +
  path pointer) and extracted facts only.
- **Cross-border transfer:** the extraction LLM call sends document
  *text* (not the raw file) to the hosted endpoint. This is within the
  engagement-consent scope for the pilot.

### Local SKU v2 (on-prem)

```
Practice premises
  └── scan roots (local + UNC shares)
       └── watcher scans → file_catalog (local DuckDB)
            └── hot docs → extraction pass
                 └── text → LLM (on-prem endpoint, zero cross-border)
                      └── facts → memory_records (local DuckDB)
                           └── retrieval → injected into answers
```

- **Memory store:** local DuckDB on the practice's premises.
- **LLM calls:** routed to an on-prem endpoint (configured via
  `extraction_llm_provider` / `answering_llm_provider`). Zero
  cross-border transfer — POPIA §72 avoidance.
- **Raw files:** never copied. Same as cloud pilot.
- **Cross-border transfer:** none. All processing on-prem.

Switching modes = config change + restart. `deployment_mode` in the
config (`cloud_pilot` / `local_sku`). The plugin does not move data
itself; the flag controls provider routing and audit labelling.

## 3. Per-user access + audit

- **Access scoping (Spec-06, #69):** role-based allow masks + hidden
  deny list. Deny > allow > wheel. Practice-internal doc class =
  principals-only. Folder layout IS the ACL (spec-06 D2) — mapped at
  catalog time by the watcher (spec-07 D8).
- **Audit log (Spec-06 D4 / Spec-07 D8):** append-only `access_audit`
  table. Every query records identity + timestamp + outcome (granted/
  denied counts, denied scopes, excluded flag). Exportable via the
  service API as JSONL or CSV. Principals-only read. 90-day rotation
  default.
- **Graph guard (Spec-06 D3):** cross-scope neighbours dropped in graph
  traversal — a shared director between Client A and B doesn't surface
  B's facts to a user with only A.

## 4. Retention

- **Calendar:** the file_catalog carries the retention category (5yr
  tax / 7yr company). The calendar is informational — the system never
  auto-deletes.
- **Purge:** principal-operated tool with client permission. The
  practice's responsible party makes the deletion decision; the system
  executes it and records the action.
- **Tombstones:** deleted files are tombstoned in the catalog (status =
  `tombstoned`). Facts sourced from tombstoned files are marked
  `invalidated` (Spec-07 D4) and excluded from retrieval. The catalog
  row is retained for audit forensics.
- **Never auto-delete:** this is a hard constraint, not a default. The
  retention calendar is a display, not a trigger.

## 5. Breach notification

- **Detection:** the audit log is the primary breach-detection surface.
  Unusual query patterns (high deny counts, queries from unexpected
  users, access to practice-internal records by non-principals) are
  visible in the export.
- **Notification:** the operator (practice) is responsible for POPIA
  breach notification to the Information Regulator and affected data
  subjects. The system provides the audit export; the practice makes
  the notification.
- **Forensics:** the audit log + file_catalog + memory_records provide
  a full chain of custody — who accessed what, when, and what facts
  were sourced from which documents.

## 6. DSAR (data subject access request) cooperation

- **Export:** the audit log export (JSONL/CSV) provides the access
  history. The file_catalog provides the document inventory. The
  memory_records provide the extracted facts. All three are exportable
  via the service API.
- **Correction:** facts are versioned (Spec-05 supersession). A
  correction creates a new version; the old version is marked
  `superseded`. The correction history is retained.
- **Deletion:** principal-operated tool with client permission (see
  Retention above). The system does not auto-delete.

## 7. Termination export

- **Full export:** the practice can export the entire memory store
  (DuckDB file) + the file_catalog + the audit log at any time. The
  DuckDB file is a single portable file (P4.3 backup machinery).
- **Provider switching:** switching LLM provider = config change +
  restart. No data migration, no lock-in. The extraction/answering
  provider is a config field, not a code path.
- **Data return:** on termination, the practice exports the DuckDB
  file and takes it with them. The hosted service's copy is deleted
  (the practice's responsible party decision). The system provides the
  export; the practice executes the deletion.

## 8. Provider due-diligence

See `provider-due-diligence-checklist.md` for the zero-retention / no-
training checklist. The pilot may use the personal relay; the
commercial SKU requires enterprise terms + region option.

## 9. Deployment mode switching

| From | To | Action |
|------|----|--------|
| cloud_pilot | local_sku | Set `deployment_mode=local_sku`, `data_residency=local`, configure on-prem LLM endpoints (`extraction_llm_provider`, `answering_llm_provider`). Restart. |
| local_sku | cloud_pilot | Set `deployment_mode=cloud_pilot`, `data_residency=cloud`, configure hosted LLM endpoints. Restart. |

No data migration in either direction — the memory store stays local.
The only change is where the LLM calls are routed.

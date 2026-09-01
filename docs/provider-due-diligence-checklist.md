# Provider Due-Diligence Checklist

**Status:** Review-ready for the pilot file (drafted 2026-08-30, Michael)
**Spec:** POPIA & deployment mode (#72)

---

## Purpose

Before routing any document text or memory operations through an LLM
provider, the operator (practice) must verify that the provider meets
the zero-retention / no-training requirements. This checklist is the
due-diligence artefact for the pilot file.

The pilot may use the personal relay (engagement-consent covers it).
The commercial local SKU v2 requires enterprise terms + a region option
(zero cross-border transfer).

## Checklist

### 1. Data retention

- [ ] **Zero retention:** the provider does not store input text or
      output beyond the time needed to return the response (typically
      <30 days for abuse monitoring, then deleted).
- [ ] **No persistent copies:** the provider confirms in writing
      (enterprise terms) that no persistent copies of input/output are
      retained after the response is returned.
- [ ] **Retention policy documented:** the provider's retention policy
      is documented and available for the practice's records.

### 2. Training on data

- [ ] **No training:** the provider does not train models on the
      practice's data (input or output). Confirmed in enterprise terms.
- [ ] **No fine-tuning:** the provider does not fine-tune models on
      the practice's data.
- [ ] **Opt-out is default:** the provider's default configuration is
      no-training (not an opt-out the practice must remember to set).

### 3. Data residency / cross-border transfer

- [ ] **Region option:** the provider offers a region option (e.g. EU,
      US, ZA) for data processing. For the local SKU v2, the endpoint
      is on-prem (zero cross-border).
- [ ] **POPIA §72 compliance:** if cross-border transfer is involved
      (cloud pilot), the practice has the engagement-consent covering
      it. For the local SKU v2, no cross-border transfer occurs.
- [ ] **Sub-processor disclosure:** the provider discloses any sub-
      processors and their regions.

### 4. Security

- [ ] **Encryption in transit:** TLS 1.2+ for all API calls.
- [ ] **Encryption at rest:** the provider encrypts any transient
      storage (abuse monitoring logs) at rest.
- [ ] **Access controls:** the provider's staff cannot access the
      practice's data without explicit authorisation.
- [ ] **Audit trail:** the provider maintains an audit trail of access
      to the practice's data (for breach notification cooperation).

### 5. Breach notification

- [ ] **Notification commitment:** the provider commits to notifying
      the practice within a defined window (e.g. 72 hours) of a
      confirmed breach.
- [ ] **Forensic cooperation:** the provider cooperates with the
      practice's breach investigation (audit logs, access records).

### 6. Termination

- [ ] **Data return:** on termination, the provider returns any
      retained data (abuse monitoring logs) and confirms deletion.
- [ ] **No lock-in:** the practice can switch providers without data
      migration (the memory store is local; only the LLM endpoint
      changes).
- [ ] **Contract termination:** the enterprise terms specify the
      termination process and data deletion timeline.

### 7. POPIA-specific

- [ ] **Operator vs processor:** the provider is a processor (POPIA
      §1) acting on behalf of the practice (operator / responsible
      party). The operator/processor split is documented in the
      enterprise terms.
- [ ] **Consent chain:** the practice's engagement-consent covers the
      processing. The provider does not seek separate consent from
      data subjects.
- [ ] **DSAR cooperation:** the provider cooperates with data subject
      access requests (export, correction, deletion) within the
      practice's process.

## Provider evaluation record

| Provider | Zero retention | No training | Region option | Enterprise terms | Status |
|----------|---------------|-------------|---------------|-----------------|--------|
| Personal relay (pilot) | N/A (pilot) | N/A (pilot) | N/A | N/A | Pilot use — engagement-consent covers |
| _Enterprise provider 1_ | ☐ | ☐ | ☐ | ☐ | _Pending_ |
| _Enterprise provider 2_ | ☐ | ☐ | ☐ | ☐ | _Pending_ |
| _On-prem endpoint (local SKU v2)_ | ☐ (N/A — on-prem) | ☐ (N/A — on-prem) | ☐ (N/A — on-prem) | ☐ (N/A — on-prem) | _Pending_ |

## Sign-off

This checklist is completed by the practice's responsible party before
routing any production data through a new provider. The completed
checklist is filed in the pilot pack.

---

**Related:** Annexure A (processing annex), Spec-06 (access scoping),
Spec-07 (the watcher), Spec-08 (#72 deployment mode).

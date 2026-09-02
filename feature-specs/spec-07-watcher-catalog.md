# Spec 7 — The watcher: document catalog, extraction & freshness

Status: **IMPLEMENTED 2026-09-01** (batch-12, #71 — `watcher.py`, commits
`93d7fab`/`57c25ba`; closes #68, #70). The APPROVED/parked text below is
historical.

## Problem

The doc tier's engine room is undefined. Facts don't appear in Argos by magic:
something must (1) know every file that exists and where, (2) decide which
files are "hot" enough to extract, (3) extract facts with provenance, (4)
notice when files change, move or die — and keep facts honest afterwards.
Without one coherent design, the pieces (catalog, hot policy, D4 fields,
audit seed, spec-06's folder mapping) get built ad hoc and drift apart. This
spec is the contract that ties them into one machine.

The failures it exists to prevent:

- **Path-based provenance breaks at year-end reorgs** — files move, links die.
- **Duplicate copies** ("Tax Invoice (2).pdf") fragment evidence, duplicate
  extraction, split the audit story.
- **Stale facts survive silently** — an edited file leaves old numbers
  answering as if current.
- **OCR digits lie unflagged** — one misread digit in a scanned amount is a
  career-ending answer.
- **Open Excel files break scans** — locked workbooks must skip, not fail.
- **Freshness is unanswerable** — "when did we last verify this?" needs a
  date, not a shrug.

## Prior decisions (locked in earlier sessions — recorded, not re-litigated)

- **Three tiers:** catalog pass (every file, near-zero cost, no LLM) /
  extraction pass (hot docs only, facts with evidence into Argos) / raw files
  stay in place (Argos stores provenance and paths, never copies).
- **Catalog is a tool surface, never context injection** — locate → read.
- **Poll, not push:** timer scan (10–60 min debounced), stat comparison
  (path, size, mtime).
- **Hot** = recency (current tax/financial year) OR type default (master-data
  per client) OR usage (N touches → auto-promote) OR manual pin (override
  wins).
- **Formats:** PDF (text + OCR fallback for scans), XLSX/CSV (sheet-level
  units, headers encoded into rows — the 26/8 rule), DOCX cheap plain text.
  Live-DB formats (OUT/DAT) and email capture = out.
- **Folder layout IS the ACL** (spec-06 D2) — this spec owns the mapping at
  catalog time.

## Design

### D1 — Stable doc identity: content hash, not path

- `file_id` = SHA-256 of file content (streamed, chunked — no full load). **Path
  is a pointer; hash is the identity.**
- One catalog row per unique hash. A file seen at multiple paths → alias rows
  pointing at one canonical row (kills "Tax Invoice (2).pdf" duplication).
- Move/rename = alias update only — **never re-extraction**.
- **Content edit that changes extractable content = version bump** → re-extract
  → old facts `stale` (D4). An edit SHOULD invalidate; a move should not; **an
  Excel resave that doesn't change the data must NOT** (amended in review,
  30/8).

**Two hashes, two jobs:** `file_id` = SHA-256 of raw bytes (dedup, alias
detection, content-edit detection); `extract_hash` = SHA-256 of the
*extraction input* — the PDF text layer, or sheet values + headers: the exact
payload the extractor receives. **Re-extraction fires only when
`extract_hash` changes.** Excel rewrites file bytes on every save (metadata,
calc chain, zip ordering) without changing data; raw-hash version bumps would
re-extract and churn `current`/`stale` weekly. The fingerprint makes resaves
no-ops while real data edits still invalidate.

### D2 — `file_catalog` table (in the same store)

Columns: `file_id` PK, `canonical_path`, `size`, `mtime`, `first_seen`,
`last_seen`, `status` (active/tombstoned), `client_scope`, `doc_class`,
`doc_type` (pdf/xlsx/csv/docx), `one_line_description`,
`description_method` (heuristic/llm), `hot_flags`, `hot_reason`,
`extract_hash` (last extraction-input fingerprint — the re-extract gate),
`extracted_at`, `last_touch`, `touch_count`, `pinned`.

**Same DuckDB as memory** (decision 1): one backup story (P4.3 machinery),
one audit join, no second file to lose. Catalog rows are tiny — 10k files is
nothing. **Constraint (amended in review, 30/8): the watcher writes only via
the shared memory service — service client, never a direct DB writer.** The
single-owner pattern (one process owns DuckDB) is what killed the lock-jam
class; the scanner must not become a second writer.

### D3 — The scan loop

- Config: `scan_roots[]` (local + UNC shares), `interval` (default 30 min,
  debounced), exclude patterns (`.tmp`, `~$` lock files, thumbs/ds-store).
- Each pass: walk roots with `os.scandir` stat (no full reads), compare
  (path, size, mtime) vs catalog → classify new / changed / moved / unchanged /
  deleted.
- **New:** hash → catalog row + heuristic description.
- **Changed:** hash → content differs? version bump (D1).
- **Deleted:** tombstone — row kept, `status=tombstoned` (the D4 `invalidated`
  trigger; also reorg + audit forensics).
- **Locked/open files (Windows Excel/PDF):** skip + retry next pass. The pass
  never fails.
- **Heuristic description:** filename + folder + first-page text (PDF) / sheet
  names + headers (XLSX) / first lines (CSV) → deterministic one-liner. LLM
  description ONLY when heuristic confidence is low — daily cap (cost gate).
- Manual *rescan now* tool. Windows Change Journals = later upgrade, not v1.

### D4 — Hot policy (locked; wired here)

Recency / type-defaults / usage count ≥ N → auto-promote / pin (override
wins). Hot docs feed the extraction queue; everything else stays
catalog-only. **First run: full catalog for free, extraction only on the hot
set.**

### D5 — Extraction pass (the only LLM spend)

- Per hot doc: extract facts → memory writes with `namespace='document'`,
  `client_scope`, `doc_class` (spec-05) **plus the D4 fields (spec-05 §D4 —
  activated by this spec, closing #68):**
  - `source_doc_id` = file_id (hash — never a path), `source_loc`
    (page/sheet/row), `extraction_method` (`text`/`ocr`/`excel`),
    `extracted_at`, `verified_state`.
- `verified_state` semantics: born-digital text = `current`; OCR-sourced
  **numeric** facts = `unverified` (flagged, lower confidence); version bump →
  old facts `stale`, new facts `current`; doc tombstoned → facts
  `invalidated`; a principal verify action flips `unverified` → `current`
  (records `verified_at`).
- Excel: sheet-level units, headers encoded into each row (no header-mapping
  reliance).
- **Same write path as everything else:** proposal/review gates, value-
  supersession, egress gates. No special bypasses for doc facts.

### D6 — Doc-to-doc conflicts: ride existing machinery, don't build new

- Original vs corrected invoice: the existing write-time regex value
  extractor + cross-category conflict scan already detects revised values →
  conflict downgrades to `pending_user_confirmation`. **Never silent
  auto-supersede across documents** (the 28/8 rule).
- v1 detection = the existing value collides; no dedicated cross-doc diff
  pass (cost/scope). Documented honest limit: conflicts surface only when
  values collide.
- **v1 cheap addition (amended in review, 30/8): filename-pattern conflict
  surfacing.** At extraction time, docs whose names carry correction markers
  (`CORRECTED`, `REVISED`, `(2)`, `v2` …) for the same client + doc type
  produce a review candidate listing both sources. Deterministic regex, zero
  LLM — catches the original-vs-corrected invoice case the value-collision
  scan misses.

### D7 — Freshness answers

With D4 live, every evidence row carries its story: *"VAT number 4780… —
extracted from [file] on [date] via [method]; last verified [date]"*. The
answerer policy from the trust model (spec-04, #39/#40/#43/#35) applies
unchanged: **a `stale` or `unverified` fact is never presented as current.**

### D8 — Audit seed (closes #70)

- `access_audit` table (spec-06 D4) ships with this tier: every query records
  identity + timestamp + outcome; export via service API; principals-only
  read; 90-day rotation default.
- The catalog mapping pass writes `client_scope`/`doc_class` onto every row —
  the pilot's "ACL" (folder layout, spec-06 D2).

## Explicitly OUT of scope

- **Windows Change Journals** — later upgrade (poll suffices for v1 cadence).
- **Email capture** — Outlook "save to folder" rule IS the intake; zero code.
- **Retention automation** — the catalog carries the calendar (5yr tax / 7yr
  company); purge = principal-operated tool with client permission;
  **never auto-delete**.
- **OneDrive placeholders, encrypted files, exotic codecs** — documented v2
  gaps, fail-soft (skipped + logged) in v1.
- **Cloud-vs-local boundary** — its own session (spec-06 OUT list).

## Tests (cheapest falsifying first — deterministic, no LLM)

1. **Identity:** rename → same `file_id`, zero re-extraction; content edit →
   changed extraction input → re-extract queued; **Excel resave with
   identical data → no re-extraction, facts stay `current`.**
2. **Dedup:** two paths, one content → one canonical row + aliases.
3. **Scan classification:** synthetic tree — new/changed/moved/unchanged/
   deleted; tombstone on delete; pass completes with locked files present.
4. **Hot policy precedence:** pin > usage > type-default > recency; N-touch
   auto-promote; first-run extracts only the hot set.
5. **Locked file:** open handle → skip + retry next pass, zero failures.
6. **D4 lifecycle:** version bump → old `stale`; OCR numeric → `unverified`;
   tombstone → `invalidated`; verify action → `current`.
7. **Conflict:** revised value from doc B vs doc A → `pending_user_confirmation`,
   never silent.
8. **Catalog retrieval:** BM25-lite over name+description+dates; date-range +
   client-scope filters.
9. **Full suite green; no watcher config = zero behaviour change.**

## Effort

Largest of the three specs: new table + scanner + streamed hasher + hot policy
+ extraction queue + D4 activation + audit seed + ~25–30 tests. No new deps
(hashlib is stdlib; PDF/XLSX readers already in the venv — pypdf is the only
possible addition, already used by the company parser). One migration family
shared with spec-05/06.

## Decisions (Michael, 30/8 — for sign-off)

Settled previously (recorded above, not re-litigated): tiers, poll-not-push,
hot policy, formats, folder=ACL.

New (recommended in bold):

1. **Catalog lives in the same DuckDB store** (one backup, one audit join) —
   not a separate sidecar file. *Amended in review: watcher writes only via
   the shared service — never a second direct DB writer (lock-jam class).*
2. **Identity = full-content SHA-256** (streamed); version bump on content
   change → re-extract; move/rename never re-extracts. *Amended in review:
   re-extraction gates on the extraction-input fingerprint (`extract_hash`),
   not raw bytes — Excel resaves are no-ops.*
3. **Conflicts ride the existing write-time supersession** (→ pending
   confirmation); no dedicated cross-doc diff pass in v1. *Amended in review:
   + filename-pattern conflict surfacing (correction markers, zero LLM).*
4. **LLM descriptions only on low-confidence heuristics**, daily cap.
5. **Retention: never auto-delete**; purge = principal tool + client
   permission.
6. **Watch cadence 30 min debounced** by default, manual rescan tool.
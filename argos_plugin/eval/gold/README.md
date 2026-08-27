# Gold set — freeze record

The gold set (`gold_v1.jsonl`) and its manifest (`gold_manifest.json`) carry
personal memory content and are gitignored. Only the freeze sha256 of the
validated set is recorded here (commit this file when the set freezes).

| Date | Gold sha256 | Snapshot id | Approved | Rejected |
|------|-------------|-------------|----------|----------|
| 2026-08-26 | `f274be0528d271e335d22ece8cf0a58d14db607632cac46d1ed8373b3e6ee89c` | `20260826_170626_224920_4d612e0f` | 995 | 5 |

Set: 1000 probes sampled from the active store (1227 active at freeze),
template-generated, deterministic seed 42. Recall windows are near-saturated
by design — the gate is a **stability tripwire**; the real-query layer is the
recall ruler.

## Freezing a new pair (process)

1. Stop the memory service, take a snapshot (`eval/snapshot_store.py take`).
2. Build the reviewable set (`eval/build_gold.py --db <snapshot>/hybrid_memory.duckdb`).
3. Review `eval/gold/gold_v1.jsonl` once (status: `approved`/`rejected`).
4. Freeze (`eval/build_gold.py --db <snapshot>/hybrid_memory.duckdb --freeze`).
5. Record the sha256 + snapshot id in the table above and commit this file.
6. Run the gate (`eval/run_gate.py --snapshot <snapshot> --gold eval/gold/gold_v1.jsonl --compare <snapshot>/gate_baseline.json`).

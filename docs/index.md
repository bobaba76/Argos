# Argos

Persistent memory for AI agents, on your own machine.

Argos is a Hermes plugin with a standalone server: a hybrid vector + graph store with local embeddings and an external API (MCP + REST). It remembers facts across sessions, versions every change, and surfaces provenance so you can trust what it recalls.

## Why Argos

- **Local-first.** Your memory database lives on your machine. No cloud calls, no telemetry, no third-party embeddings API.
- **Evidence-oriented.** Every memory carries provenance — where it came from, when, and how confident the system is. You can ask "why was this retrieved?" and get a real answer.
- **Auditable.** Every facade operation is logged. Destructive actions (erase) produce receipts. Access is scoped per principal.
- **Trust model.** Nothing becomes a memory silently. Extraction produces *proposals* — pending until you approve them. Updates chain versions instead of overwriting.

## What's here

This is the public docs site. The [README](https://github.com/bobaba76/Argos/blob/master/README.md) in the repo is the canonical short-form reference; these docs extend it with walkthroughs, API reference, and tuning guidance.

- [Quickstart](quickstart.md) — get Argos running in five minutes.
- [Installation](installation.md) — the real install path (plugin manifest + deploy sync).
- [Configuration](configuration.md) — every setting, default, and description (mirrors `CONFIG_REFERENCE.md`).
- [API reference](api/index.md) — MCP (stdio) and REST (HTTP) surfaces, with the live facade operation allowlist.
- [Tuning](tuning.md) — embedder/reranker matrix and retrieval knobs.
- [Integration guides](integration.md) — adapters for non-Hermes agents (placeholder, tracking #277).
- [FAQ](faq.md) — common questions and troubleshooting.

## License

Business Source License 1.1 (BSL 1.1): free for personal and non-production use; production or commercial use requires a license. Converts to Apache 2.0 on August 21, 2030. Full terms: [`LICENSE.md`](https://github.com/bobaba76/Argos/blob/master/LICENSE.md).

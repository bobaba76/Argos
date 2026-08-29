# tool_compression/ — reference copies, NOT runtime code

This directory is a **reference copy of patched Hermes core tool files**.
It is **NOT Argos runtime code** and **NOT part of the sync path**.

- These files are copies of the patched Hermes core tools (browser, terminal,
  code execution, delegate, memory, skill manager, session search, clarify).
- The live runtime copies live in the Hermes install, not here (see
  `SYNC_HANDOFF.md` for the sync topology).
- Nothing in `argos_plugin/` imports from this directory, and `deploy.py`
  does not copy it to the live install.
- Kept in-tree as a **hook-integration reference**: the memory plugin's
  `pre_llm_call` hook and tool wrappers interact with these tools, so the
  exact patched shape matters for integration work.

Do not edit these files expecting the change to reach a running Hermes —
edit the live install (or the patch source), then refresh this copy if it
is still needed as reference.

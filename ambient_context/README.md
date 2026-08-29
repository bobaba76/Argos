# ambient_context/ — reference copies, NOT runtime code

This directory is a **reference copy of patched Hermes core files** plus the
ambient-context utility modules the plugin's `pre_llm_call` hook calls.
It is **NOT Argos runtime code** and **NOT part of the sync path**.

- `hermes_{time,location,weather,file_activity}.py` — ambient-context
  utility modules that the plugin's `pre_llm_call` hook uses as a reference
  for the ambient context block.
- `modified_{turn_context,conversation_loop,gateway_run,config_defaults}.py`
  — patched Hermes core files (e.g. `modified_gateway_run.py` is ~1.2MB).
- The live runtime copies live in the Hermes install, not here (see
  `SYNC_HANDOFF.md` for the sync topology).
- `argos_plugin/` carries its **own** ambient modules for runtime use; this
  directory is not on any import path and `deploy.py` does not sync it.
- Kept in-tree as **hook-integration reference**: the exact patched shape
  of the Hermes core matters for understanding how the plugin hooks in.

Do not edit these files expecting the change to reach a running Hermes —
edit the live install (or the patch source), then refresh this copy if it
is still needed as reference.

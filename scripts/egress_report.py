#!/usr/bin/env python3
"""Print a 'what leaves this machine' report for the hybrid-memory plugin.

Inventories every plugin-owned LLM egress site, what it sends, the config
flag that gates it, and whether it is live under the current config.
Run from the repo root:

    python scripts/egress_report.py          # human-readable report
    python scripts/egress_report.py --json   # machine-readable JSON
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PLUGIN_DIR = Path(__file__).resolve().parent.parent / "hybrid_memory_plugin"
if str(_PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_DIR))

from egress import (  # noqa: E402
    CONTEXT_NETWORK,
    GROUPS,
    SITES,
    load_config,
    local_only,
    report,
    site_live,
)


def _json_report(cfg: dict) -> dict:
    by_kind = {site["kind"]: site for site in SITES}
    sites = []
    for group_name, kinds in GROUPS:
        for kind in kinds:
            site = by_kind[kind]
            sites.append(
                {
                    "kind": kind,
                    "group": group_name,
                    "payload": site["payload"],
                    "trigger": site["trigger"],
                    "gate": site["gate"],
                    "live": site_live(site, cfg),
                }
            )
    return {
        "local_only": local_only(cfg),
        "sites": sites,
        "context_providers": [
            {"kind": s["kind"], "payload": s["payload"], "note": s["note"]}
            for s in CONTEXT_NETWORK
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json", action="store_true", help="emit machine-readable JSON"
    )
    args = parser.parse_args()
    cfg = load_config()
    if args.json:
        print(json.dumps(_json_report(cfg), indent=2))
    else:
        print(report(cfg))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
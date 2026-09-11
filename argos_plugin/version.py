"""Single source of truth for the Argos plugin version (#360).

Kept as a standalone module (not only a package attribute) so every
import context can read the same value: the package itself, modules
loaded standalone with the plugin dir on sys.path (probes, scripts,
tests), and scripts/deploy.py — which parses this file's literal.

plugin.yaml carries the same version; test_deploy.py pins the two equal,
so a version bump touches one test-visible pair, not N files.
"""

__version__ = "1.0.0"

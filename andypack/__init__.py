"""comfyui-andypack — Animation Coordinator (dependency-aware FFLF resolver)."""

from andypack import api, server  # noqa: F401  (server registers HTTP routes on import)
from andypack.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = "./web"

# Seed `<user>/default/andypack/animations/default.json` from the bundled manifest
# so a fresh install loads with a working manifest. Idempotent and non-destructive;
# never let a seeding hiccup block the pack from loading.
try:
    api.seed_default_manifest()
except Exception:  # pragma: no cover - defensive: loading must not fail on seed
    pass

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

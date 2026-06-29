"""comfyui-andypack — Animation Coordinator (dependency-aware FFLF resolver)."""

from andypack import server  # noqa: F401  (registers HTTP routes on import)
from andypack.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

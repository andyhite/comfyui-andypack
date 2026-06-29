"""ComfyUI entry point for comfyui-andypack.

ComfyUI imports this top-level package from `custom_nodes/<dir>/__init__.py`.
The pack's code lives in the `andypack` subpackage and uses absolute imports
(`from andypack...`), so the repo root must be on `sys.path` — ComfyUI does not
add it automatically. We insert it here, then re-export the symbols ComfyUI reads:
`NODE_CLASS_MAPPINGS`, `NODE_DISPLAY_NAME_MAPPINGS`, and `WEB_DIRECTORY`.

`WEB_DIRECTORY` is resolved by ComfyUI relative to THIS file's directory (the repo
root), so "./web" correctly points at the repo-root `web/` folder.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

from andypack import (  # noqa: E402  (must follow the sys.path insert above)
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
    WEB_DIRECTORY,
)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

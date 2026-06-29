"""ComfyUI node classes — thin wrappers over andypack.resolve / io / images."""

from __future__ import annotations

import os
from datetime import datetime, timezone

from andypack import images, io
from andypack.manifest import load_manifest


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class AnimationManifestLoader:
    CATEGORY = "andypack"
    FUNCTION = "load"
    RETURN_TYPES = ("ANIM_MANIFEST",)
    RETURN_NAMES = ("manifest",)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"manifest_path": ("STRING", {"default": "animations.json"})}}

    @classmethod
    def IS_CHANGED(cls, manifest_path):
        try:
            return os.path.getmtime(manifest_path)
        except OSError:
            return float("nan")

    def load(self, manifest_path):
        return (load_manifest(manifest_path),)


class ConceptImageWriter:
    CATEGORY = "andypack"
    FUNCTION = "write"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("character_dir",)
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "root_dir": ("STRING", {"default": "output/anim"}),
                "character": ("STRING", {"default": "Cortex"}),
            },
            "optional": {
                "identity_positive": ("STRING", {"default": "", "multiline": True}),
                "identity_negative": ("STRING", {"default": "", "multiline": True}),
            },
        }

    def write(self, image, root_dir, character, identity_positive="", identity_negative=""):
        char_dir = os.path.join(root_dir, character)
        images.save_image_png(image, os.path.join(char_dir, "_concept.png"))
        layer = {}
        if identity_positive.strip():
            layer["prompt"] = identity_positive.strip()
        if identity_negative.strip():
            layer["negative"] = identity_negative.strip()
        if layer:
            io.atomic_write_json(os.path.join(char_dir, "_concept.json"), layer)
        return (char_dir,)


NODE_CLASS_MAPPINGS = {
    "AnimationManifestLoader": AnimationManifestLoader,
    "ConceptImageWriter": ConceptImageWriter,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AnimationManifestLoader": "Animation Manifest Loader",
    "ConceptImageWriter": "Concept Image Writer",
}

"""ComfyUI node classes — thin wrappers over andypack.resolve / io / images."""

from __future__ import annotations

import os
from datetime import datetime, timezone

from andypack import images, io
from andypack.manifest import load_manifest
from andypack.resolve import resolve_pose


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


class CharacterPoseSelector:
    CATEGORY = "andypack"
    FUNCTION = "select"
    RETURN_TYPES = ("IMAGE", "BOOLEAN", "STRING", "STRING", "STRING", "ANIM_META")
    RETURN_NAMES = ("source_image", "has_source", "positive", "negative", "output_dir", "meta")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "manifest": ("ANIM_MANIFEST",),
                "root_dir": ("STRING", {"default": "output/anim"}),
                "character": ("STRING", {"default": "Cortex"}),
                "pose": ("STRING", {"default": "base"}),
                "direction": ("STRING", {"default": "E"}),
            }
        }

    def select(self, manifest, root_dir, character, pose, direction):
        r = resolve_pose(manifest, root_dir, character, pose, direction)
        if not r["selectable"]:
            raise RuntimeError(
                f"pose {pose}@{direction} not selectable: blocked_by={r['blocked_by']}"
            )
        src = r["source_image"]
        if src:
            image = images.load_image_tensor(src)
            has_source = True
        else:
            image = images.empty_image()
            has_source = False
        return (image, has_source, r["positive"], r["negative"], r["output_dir"], r["meta"])


class PoseFrameWriter:
    CATEGORY = "andypack"
    FUNCTION = "write"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_dir",)
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "output_dir": ("STRING",),
                "meta": ("ANIM_META",),
            }
        }

    def write(self, image, output_dir, meta):
        # payload first (atomic), sidecar last (atomic) = completion sentinel
        png_path = os.path.join(output_dir, meta["image"])
        images.save_image_png(image, png_path)
        sidecar = io.build_pose_sidecar(meta, created_utc=_utc_now())
        sidecar_path = os.path.join(output_dir, f"{meta['direction']}.json")
        io.atomic_write_json(sidecar_path, sidecar)
        return (output_dir,)


NODE_CLASS_MAPPINGS = {
    "AnimationManifestLoader": AnimationManifestLoader,
    "ConceptImageWriter": ConceptImageWriter,
    "CharacterPoseSelector": CharacterPoseSelector,
    "PoseFrameWriter": PoseFrameWriter,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AnimationManifestLoader": "Animation Manifest Loader",
    "ConceptImageWriter": "Concept Image Writer",
    "CharacterPoseSelector": "Character Pose Selector",
    "PoseFrameWriter": "Pose Frame Writer",
}

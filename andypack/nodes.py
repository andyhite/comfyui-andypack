"""ComfyUI node classes — thin wrappers over andypack.resolve / io / images."""

from __future__ import annotations

import os
from datetime import datetime, timezone

from andypack import api, images, io
from andypack.manifest import load_manifest
from andypack.resolve import resolve_animation, resolve_pose


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class AnimationManifestLoader:
    CATEGORY = "andypack"
    FUNCTION = "load"
    RETURN_TYPES = ("ANIM_MANIFEST",)
    RETURN_NAMES = ("manifest",)

    @classmethod
    def INPUT_TYPES(cls):
        # Combo of manifest files found in user/default/andypack/animations.
        # Falls back to the conventional name so the widget is never empty.
        names = api.list_manifest_names() or ["default.json"]
        return {"required": {"manifest": (names,)}}

    @classmethod
    def IS_CHANGED(cls, manifest):
        try:
            return os.path.getmtime(api.resolve_manifest_path(manifest))
        except OSError:
            return float("nan")

    def load(self, manifest):
        return (load_manifest(api.resolve_manifest_path(manifest)),)


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
                "root_dir": ("STRING", {"default": "output/characters"}),
                "character": ("STRING", {"default": "cortex"}),
            },
            "optional": {
                "identity_positive": ("STRING", {"default": "", "multiline": True}),
                "identity_negative": ("STRING", {"default": "", "multiline": True}),
            },
        }

    def write(self, image, root_dir, character, identity_positive="", identity_negative=""):
        # Character names become path segments — force lowercase snake_case.
        char_dir = os.path.join(root_dir, io.to_snake_case(character))
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
    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING", "ANIM_META")
    RETURN_NAMES = ("source_image", "positive", "negative", "output_dir", "meta")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "manifest": ("ANIM_MANIFEST",),
                "character_dir": ("STRING", {"default": "output/characters/cortex"}),
                "pose": ("STRING", {"default": "base"}),
                "direction": ("STRING", {"default": "E"}),
            }
        }

    def select(self, manifest, character_dir, pose, direction):
        root, character = os.path.split(os.path.normpath(character_dir))
        r = resolve_pose(manifest, root, character, pose, direction)
        if not r["selectable"]:
            raise RuntimeError(
                f"pose {pose}@{direction} not selectable: blocked_by={r['blocked_by']}"
            )
        # On a successful select the `from`-source is always complete, so
        # source_image is always a real image (never the empty sentinel).
        src = r["source_image"]
        image = images.load_image_tensor(src) if src else images.empty_image()
        return (image, r["positive"], r["negative"], r["output_dir"], r["meta"])


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


class CharacterAnimationSelector:
    CATEGORY = "andypack"
    FUNCTION = "select"
    RETURN_TYPES = ("IMAGE", "BOOLEAN", "IMAGE", "BOOLEAN", "STRING", "STRING", "STRING", "ANIM_META")
    RETURN_NAMES = (
        "start_image", "has_start", "end_image", "has_end",
        "positive", "negative", "output_dir", "meta",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "manifest": ("ANIM_MANIFEST",),
                "character_dir": ("STRING", {"default": "output/characters/cortex"}),
                "animation": ("STRING", {"default": "fighting_stance_idle"}),
                "direction": ("STRING", {"default": "E"}),
            }
        }

    def _anchor(self, path):
        if path:
            return images.load_image_tensor(path), True
        return images.empty_image(), False

    def select(self, manifest, character_dir, animation, direction):
        root, character = os.path.split(os.path.normpath(character_dir))
        r = resolve_animation(manifest, root, character, animation, direction)
        if not r["selectable"]:
            raise RuntimeError(
                f"animation {animation}@{direction} not selectable: blocked_by={r['blocked_by']}"
            )
        start_image, has_start = self._anchor(r["start_image"])
        end_image, has_end = self._anchor(r["end_image"])
        return (
            start_image, has_start, end_image, has_end,
            r["positive"], r["negative"], r["output_dir"], r["meta"],
        )


class AnimationFrameWriter:
    CATEGORY = "andypack"
    FUNCTION = "write"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_dir",)
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE",),
                "output_dir": ("STRING",),
                "meta": ("ANIM_META",),
            },
            "optional": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "loop_closure": (["drop_last", "duplicate_first"],),
            },
        }

    def write(self, frames, output_dir, meta, seed=0, loop_closure="drop_last"):
        os.makedirs(output_dir, exist_ok=True)
        # frames: IMAGE batch [B, H, W, C] -> list of single-frame tensors
        batch = [frames[i:i + 1] for i in range(frames.shape[0])]
        if meta.get("loop"):
            batch = io.apply_loop_closure(batch, loop_closure)
        for index, frame in enumerate(batch):
            images.save_image_png(frame, os.path.join(output_dir, io.frame_name(index)))
        count = len(batch)
        full_meta = io.build_animation_meta(
            meta,
            count=count,
            start_frame=io.frame_name(0),
            last_frame=io.frame_name(count - 1),
            seed=seed,
            created_utc=_utc_now(),
        )
        io.atomic_write_json(os.path.join(output_dir, "meta.json"), full_meta)
        return (output_dir,)


NODE_CLASS_MAPPINGS = {
    "AnimationManifestLoader": AnimationManifestLoader,
    "ConceptImageWriter": ConceptImageWriter,
    "CharacterPoseSelector": CharacterPoseSelector,
    "PoseFrameWriter": PoseFrameWriter,
    "CharacterAnimationSelector": CharacterAnimationSelector,
    "AnimationFrameWriter": AnimationFrameWriter,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AnimationManifestLoader": "Animation Manifest Loader",
    "ConceptImageWriter": "Concept Image Writer",
    "CharacterPoseSelector": "Character Pose Selector",
    "PoseFrameWriter": "Pose Frame Writer",
    "CharacterAnimationSelector": "Character Animation Selector",
    "AnimationFrameWriter": "Animation Frame Writer",
}

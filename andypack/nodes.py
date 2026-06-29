"""ComfyUI node classes — thin wrappers over andypack.resolve / io / images."""

from __future__ import annotations

import os
from datetime import datetime, timezone

from andypack import api, images, io
from andypack.manifest import load_manifest
from andypack.resolve import effective_manifest, resolve_animation, resolve_pose


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class AnimationManifestLoader:
    CATEGORY = "andypack"
    FUNCTION = "load"
    RETURN_TYPES = ("ANIM_MANIFEST",)
    RETURN_NAMES = ("MANIFEST",)

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


class CharacterSelector:
    CATEGORY = "andypack"
    FUNCTION = "select"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("CHARACTER_DIR",)

    @classmethod
    def _root(cls):
        return api.characters_dir() or "output/characters"

    @classmethod
    def INPUT_TYPES(cls):
        # Combo of character folders found in <output>/characters.
        names = api.list_subdirs(api.characters_dir()) or ["cortex"]
        return {"required": {"character": (names,)}}

    @classmethod
    def IS_CHANGED(cls, character):
        try:
            return os.path.getmtime(os.path.join(cls._root(), character))
        except OSError:
            return float("nan")

    def select(self, character):
        return (os.path.join(self._root(), character),)


class ConceptImageWriter:
    CATEGORY = "andypack"
    FUNCTION = "write"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("CHARACTER_DIR",)
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                # A directory name under ComfyUI's output dir (joined below).
                "root_dir": ("STRING", {"default": "characters"}),
                "character": ("STRING", {"default": "cortex"}),
            },
            "optional": {
                "identity_positive": ("STRING", {"default": "", "multiline": True}),
                "identity_negative": ("STRING", {"default": "", "multiline": True}),
            },
        }

    def write(self, image, root_dir, character, identity_positive="", identity_negative=""):
        # Character names become path segments — force lowercase snake_case.
        # root_dir is resolved under ComfyUI's output dir (absolute paths pass through).
        char_dir = api.under_output(os.path.join(root_dir, io.to_snake_case(character)))
        images.save_image_png(image, os.path.join(char_dir, "_concept.png"))
        layer = {}
        if identity_positive.strip():
            layer["positive_prompt"] = identity_positive.strip()
        if identity_negative.strip():
            layer["negative_prompt"] = identity_negative.strip()
        if layer:
            io.atomic_write_json(os.path.join(char_dir, "_concept.json"), layer)
        return (char_dir,)


class CharacterPoseSelector:
    CATEGORY = "andypack"
    FUNCTION = "select"
    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING", "ANIM_META")
    RETURN_NAMES = ("SOURCE_IMAGE", "POSITIVE_PROMPT", "NEGATIVE_PROMPT", "OUTPUT_DIR", "META")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "manifest": ("ANIM_MANIFEST",),
                "character_dir": ("STRING", {"default": "output/characters/cortex"}),
                "pose": ("STRING", {"default": "base"}),
                "direction": ("STRING", {"default": "EAST"}),
            }
        }

    def select(self, manifest, character_dir, pose, direction):
        if not character_dir or not pose or not direction:
            raise RuntimeError(
                "CharacterPoseSelector needs character_dir/pose/direction; got "
                f"character_dir={character_dir!r}, pose={pose!r}, direction={direction!r}"
            )
        root, character = api.split_character_dir(character_dir)
        manifest = effective_manifest(manifest, root, character)
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
    RETURN_NAMES = ("OUTPUT_DIR",)
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
    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING", "STRING", "BOOLEAN", "STRING", "ANIM_META")
    RETURN_NAMES = (
        "START_IMAGE", "END_IMAGE",
        "POSITIVE_PROMPT", "NEGATIVE_PROMPT", "IS_FFLF", "OUTPUT_DIR", "META",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "manifest": ("ANIM_MANIFEST",),
                "character_dir": ("STRING", {"default": "output/characters/cortex"}),
                "animation": ("STRING", {"default": "fighting_stance_idle"}),
                "direction": ("STRING", {"default": "EAST"}),
            }
        }

    def select(self, manifest, character_dir, animation, direction):
        if not character_dir or not animation or not direction:
            raise RuntimeError(
                "CharacterAnimationSelector needs character_dir/animation/direction; got "
                f"character_dir={character_dir!r}, animation={animation!r}, direction={direction!r}"
            )
        root, character = api.split_character_dir(character_dir)
        manifest = effective_manifest(manifest, root, character)
        r = resolve_animation(manifest, root, character, animation, direction)
        if not r["selectable"]:
            raise RuntimeError(
                f"animation {animation}@{direction} not selectable: blocked_by={r['blocked_by']}"
            )
        # start_image is always present (every selectable animation has a start
        # source — the I2V seed). A declared end_at makes this an FFLF clip.
        start_image = images.load_image_tensor(r["start_image"])
        if r["end_image"]:
            end_image, is_fflf = images.load_image_tensor(r["end_image"]), True
        else:
            end_image, is_fflf = images.empty_image(), False
        return (
            start_image, end_image,
            r["positive"], r["negative"], is_fflf, r["output_dir"], r["meta"],
        )


class AnimationFrameWriter:
    CATEGORY = "andypack"
    FUNCTION = "write"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("OUTPUT_DIR",)
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
    "CharacterSelector": CharacterSelector,
    "ConceptImageWriter": ConceptImageWriter,
    "CharacterPoseSelector": CharacterPoseSelector,
    "PoseFrameWriter": PoseFrameWriter,
    "CharacterAnimationSelector": CharacterAnimationSelector,
    "AnimationFrameWriter": AnimationFrameWriter,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AnimationManifestLoader": "Animation Manifest Loader",
    "CharacterSelector": "Character Selector",
    "ConceptImageWriter": "Concept Image Writer",
    "CharacterPoseSelector": "Character Pose Selector",
    "PoseFrameWriter": "Pose Frame Writer",
    "CharacterAnimationSelector": "Character Animation Selector",
    "AnimationFrameWriter": "Animation Frame Writer",
}

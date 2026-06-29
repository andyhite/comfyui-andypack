"""ComfyUI node classes — thin wrappers over andypack.resolve / io / images."""

from __future__ import annotations

import os
from datetime import datetime, timezone

from andypack import api, images, io
from andypack.manifest import load_manifest
from andypack.resolve import effective_manifest, resolve_animation, resolve_pose


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


_NO_CHARACTER = "(select character)"


def _mtime(path) -> float:
    try:
        return os.path.getmtime(path) if path else 0.0
    except OSError:
        return 0.0


def _selector_fingerprint(resolved: dict, *image_keys: str) -> str:
    """A change-token for a selector: its merged prompt_hash, selectability, and
    the identity+mtime of each anchor image it consumes. A selector reads the
    rendered tree, so without this ComfyUI would cache its first result and never
    notice a dependency being (re)rendered or a prompt edit going stale."""
    parts = [
        resolved["meta"]["prompt_hash"],
        str(resolved["selectable"]),
    ]
    for key in image_keys:
        path = resolved.get(key)
        parts.append(f"{path}:{_mtime(path)}")
    return "|".join(parts)


def _character_choices():
    """Combo choices for a character dropdown: a placeholder + the folders in the
    characters dir. The placeholder lets the cascade start unselected."""
    return [_NO_CHARACTER, *api.list_subdirs(api.characters_dir())]


def _characters_root():
    return api.characters_dir() or "output/characters"


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
        # character is a real combo of character folders; category/pose/direction
        # are STRING widgets the web extension turns into the cascading combos
        # (category is a UI filter; the node resolves by pose + direction).
        return {
            "required": {
                "manifest": ("ANIM_MANIFEST",),
                "character": (_character_choices(),),
                "category": ("STRING", {"default": ""}),
                "pose": ("STRING", {"default": ""}),
                "direction": ("STRING", {"default": ""}),
            }
        }

    @classmethod
    def IS_CHANGED(cls, manifest, character, category, pose, direction):
        # Re-resolve so the cache reflects the rendered tree (deps generated /
        # re-rendered, prompt edits going stale), not just the widget values.
        if character in ("", _NO_CHARACTER) or not pose or not direction:
            return float("nan")
        root = _characters_root()
        try:
            eff = effective_manifest(manifest, root, character)
            r = resolve_pose(eff, root, character, pose, direction)
        except Exception:
            return float("nan")
        return _selector_fingerprint(r, "source_image")

    def select(self, manifest, character, category, pose, direction):
        if character in ("", _NO_CHARACTER):
            raise RuntimeError("CharacterPoseSelector: select a character first")
        if not pose or not direction:
            raise RuntimeError("CharacterPoseSelector: pick a pose and a direction")
        root = _characters_root()
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
        # Re-render discipline: drop the sidecar (completion sentinel) FIRST so an
        # interrupted rewrite reads as incomplete, then payload, then sidecar last.
        png_path = os.path.join(output_dir, meta["image"])
        sidecar_path = os.path.join(output_dir, f"{meta['direction']}.json")
        io.remove_if_exists(sidecar_path)
        images.save_image_png(image, png_path)
        sidecar = io.build_pose_sidecar(meta, created_utc=_utc_now())
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
                "character": (_character_choices(),),
                "category": ("STRING", {"default": ""}),
                "animation": ("STRING", {"default": ""}),
                "direction": ("STRING", {"default": ""}),
            }
        }

    @classmethod
    def IS_CHANGED(cls, manifest, character, category, animation, direction):
        # Re-resolve so the cache reflects the rendered tree (anchors generated /
        # re-rendered, prompt edits going stale), not just the widget values.
        if character in ("", _NO_CHARACTER) or not animation or not direction:
            return float("nan")
        root = _characters_root()
        try:
            eff = effective_manifest(manifest, root, character)
            r = resolve_animation(eff, root, character, animation, direction)
        except Exception:
            return float("nan")
        return _selector_fingerprint(r, "start_image", "end_image")

    def select(self, manifest, character, category, animation, direction):
        if character in ("", _NO_CHARACTER):
            raise RuntimeError("CharacterAnimationSelector: select a character first")
        if not animation or not direction:
            raise RuntimeError("CharacterAnimationSelector: pick an animation and a direction")
        root = _characters_root()
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
        # Re-render discipline: drop meta.json (the completion sentinel) FIRST and
        # clear any stale frames so an interrupted rewrite reads as incomplete and
        # a shorter clip can't leave orphan higher-index frames behind. meta.json
        # is written LAST (atomic) below.
        meta_path = os.path.join(output_dir, "meta.json")
        io.remove_if_exists(meta_path)
        io.clear_frames(output_dir)
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
        io.atomic_write_json(meta_path, full_meta)
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

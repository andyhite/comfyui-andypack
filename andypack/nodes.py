"""ComfyUI node classes — thin wrappers over andypack.resolve / io / images."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

from andypack import api, images, io, manikins, resolve
from andypack.manifest import collect_warnings, load_manifest
from andypack.resolve import effective_manifest, resolve_animation, resolve_pose


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


_NO_CHARACTER = "(select character)"

# The leaf keys a selector emits in its POSE / ANIMATION dict — the resolved
# values + image tensors. The resolver meta rides along bundled under the private
# `_meta` key (a JSON-safe subset the writer uses to build the sidecar) and is not
# a leaf output. The Unpack nodes fan these out as static, typed outputs; a test
# asserts they stay in sync with the keys the selectors actually build (less
# `_meta`).
POSE_OUTPUT_KEYS = sorted([
    "source_image", "pose_reference", "positive", "negative", "output_dir",
])
ANIMATION_OUTPUT_KEYS = sorted([
    "start_image", "end_image", "positive", "negative",
    "is_fflf", "length", "fps", "width", "height", "shift", "output_dir",
])


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
    """Combo choices for a character dropdown: a placeholder + the character folders
    in the characters dir. Uses the same character detection as the `/characters`
    route (`api.list_characters`), so the combo and the route never disagree. The
    placeholder lets the cascade start unselected."""
    chars = [c["name"] for c in api.list_characters(_characters_root())]
    return [_NO_CHARACTER, *chars]


def _characters_root():
    return api.characters_dir() or "output/characters"


class AnimationManifestLoader:
    CATEGORY = "andypack/Manifest"
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


class CharacterCreator:
    """Persist a character's prompt layer (character.json — no image, no
    provenance) and emit the base-pose job for one direction, pairing the
    reference image (first) with the bundled manikin for that direction (second)
    for a multi-reference FLUX.2 edit. Selector-style: pick a direction, get one
    ANIM_POSE; the base pose is the tree root."""

    CATEGORY = "andypack/Character"
    FUNCTION = "create"
    RETURN_TYPES = ("ANIM_POSE",)
    RETURN_NAMES = ("POSE",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "manifest": ("ANIM_MANIFEST",),
                "image": ("IMAGE",),
                "character": ("STRING", {"default": "cortex"}),
                "direction": (manikins.CANONICAL_DIRECTIONS,),
            },
            "optional": {
                "character_positive": ("STRING", {"default": "", "multiline": True}),
                "character_negative": ("STRING", {"default": "", "multiline": True}),
                # Persist the reference art to `<char>/_reference.png` so the
                # character can be reloaded (CharacterReferenceLoader) and its base
                # re-generated later without re-supplying the original image.
                "save_reference": ("BOOLEAN", {"default": True}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, manifest, image, character, direction,
                   character_positive="", character_negative="", save_reference=True):
        # Re-resolve the base pose so the fingerprint reflects prompt edits going
        # stale, plus the character-layer widgets that this node persists.
        if not character or not direction:
            return float("nan")
        root = _characters_root()
        try:
            char_name = io.to_snake_case(character)
            eff = effective_manifest(manifest, root, char_name)
            r = resolve_pose(eff, root, char_name, "base", direction)
        except Exception:
            return float("nan")
        return "|".join([
            r["meta"]["prompt_hash"], direction,
            character_positive.strip(), character_negative.strip(),
        ])

    def create(self, manifest, image, character, direction,
               character_positive="", character_negative="", save_reference=True):
        if direction not in manikins.CANONICAL_DIRECTIONS:
            raise RuntimeError(f"CharacterCreator: unknown direction {direction!r}")
        root = _characters_root()
        char_name = io.to_snake_case(character)
        layer = {}
        if character_positive.strip():
            layer["positive_prompt"] = character_positive.strip()
        if character_negative.strip():
            layer["negative_prompt"] = character_negative.strip()
        # Persist the character prompt layer (merging over any existing overlay).
        # No image is written and no provenance is stamped — base sidecars carry
        # the tree's provenance.
        existing = resolve.read_character(root, char_name)
        payload = io.build_character(layer, existing=existing)
        io.atomic_write_json(os.path.join(root, char_name, "character.json"), payload)
        # Drop the cached character layer so the resolve below (and descendants)
        # see this write even within the filesystem's mtime resolution window.
        resolve.invalidate_character(root, char_name)
        # Optionally persist the reference art so the character can be reloaded and
        # its base re-generated later (CharacterReferenceLoader) without the user
        # re-supplying the original image. Not a render node — no provenance.
        if save_reference:
            images.save_image_png(image, resolve.reference_image_path(root, char_name))

        eff = effective_manifest(manifest, root, char_name)
        if "base" not in eff.get("poses", {}):
            raise RuntimeError("CharacterCreator: manifest has no 'base' pose")
        if direction not in eff["poses"]["base"]["directions"]:
            raise RuntimeError(f"CharacterCreator: base has no direction {direction!r}")
        r = resolve_pose(eff, root, char_name, "base", direction)
        manikin = images.load_image_tensor(manikins.manikin_path(direction))
        pose = {
            "source_image": image,        # the character reference (first reference)
            "pose_reference": manikin,    # the manikin for this direction (second)
            "positive": r["positive"],
            "negative": r["negative"],
            "output_dir": r["output_dir"],
            "_meta": r["meta"],
        }
        return (pose,)


def _build_pose_bundle(r: dict) -> dict:
    """An ANIM_POSE bundle from a resolve_pose result. A normal pose has no manikin
    (pose_reference is the empty sentinel); on a selectable pose the `from`-source
    is complete, so source_image is a real image."""
    src = r["source_image"]
    image = images.load_image_tensor(src) if src else images.empty_image()
    return {
        "source_image": image,
        "pose_reference": images.empty_image(),
        "positive": r["positive"],
        "negative": r["negative"],
        "output_dir": r["output_dir"],
        "_meta": r["meta"],
    }


def _build_animation_bundle(r: dict) -> dict:
    """An ANIM_ANIMATION bundle from a resolve_animation result — start/end anchor
    images plus the wireable generation params (length/fps/width/height/shift)."""
    start_image = (
        images.load_image_tensor(r["start_image"]) if r["start_image"]
        else images.empty_image()
    )
    if r["end_image"]:
        end_image, is_fflf = images.load_image_tensor(r["end_image"]), True
    else:
        end_image, is_fflf = images.empty_image(), False
    meta = r["meta"]
    as_int = lambda k: int(meta[k]) if meta.get(k) is not None else 0  # noqa: E731
    return {
        "start_image": start_image,
        "end_image": end_image,
        "positive": r["positive"],
        "negative": r["negative"],
        "is_fflf": is_fflf,
        "length": as_int("length"),
        "fps": max(as_int("fps"), 1),
        "width": as_int("width"),
        "height": as_int("height"),
        "shift": float(meta["shift"]) if meta.get("shift") is not None else 0.0,
        "output_dir": r["output_dir"],
        "_meta": meta,
    }


class CharacterReferenceLoader:
    """Reload a character's persisted reference art (`<char>/_reference.png`, saved
    by the Character Creator) as an IMAGE. Feed it back into the Character Creator
    to re-generate base directions later without re-supplying the original concept
    art. Raises if the character has no persisted reference (it was created with
    save_reference off, or predates persistence)."""

    CATEGORY = "andypack/Character"
    FUNCTION = "load"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("REFERENCE_IMAGE",)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"character": (_character_choices(),)}}

    @classmethod
    def IS_CHANGED(cls, character):
        if character in ("", _NO_CHARACTER):
            return float("nan")
        path = resolve.reference_image_path(_characters_root(), character)
        return f"{path}:{_mtime(path)}"

    def load(self, character):
        if character in ("", _NO_CHARACTER):
            raise RuntimeError("CharacterReferenceLoader: select a character first")
        path = resolve.reference_image_path(_characters_root(), character)
        if not os.path.exists(path):
            raise RuntimeError(
                f"CharacterReferenceLoader: {character!r} has no persisted reference "
                f"image (expected {path}); re-run the Character Creator with "
                "save_reference enabled, or supply the reference art directly"
            )
        return (images.load_image_tensor(path),)


class CharacterPoseSelector:
    CATEGORY = "andypack/Pose"
    FUNCTION = "select"
    # One bundled POSE dict (unpack it with Unpack Pose) instead of loose outputs.
    RETURN_TYPES = ("ANIM_POSE",)
    RETURN_NAMES = ("POSE",)

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
        if pose not in manifest.get("poses", {}):
            raise RuntimeError(
                f"CharacterPoseSelector: unknown pose {pose!r} (stale or renamed) — pick a pose"
            )
        if not manifest["poses"][pose].get("from"):
            raise RuntimeError(
                f"CharacterPoseSelector: {pose!r} is a root pose — use the Character Creator node"
            )
        r = resolve_pose(manifest, root, character, pose, direction)
        if not r["selectable"]:
            raise RuntimeError(
                f"pose {pose}@{direction} not selectable: blocked_by={r['blocked_by']}"
            )
        # Bundle the loose outputs into one POSE dict (see POSE_OUTPUT_KEYS). On a
        # successful select the `from`-source is complete, so source_image is real.
        return (_build_pose_bundle(r),)


class PoseFrameWriter:
    CATEGORY = "andypack/Pose"
    FUNCTION = "write"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("OUTPUT_DIR",)
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose": ("ANIM_POSE",),
                "image": ("IMAGE",),
            }
        }

    def write(self, pose, image):
        output_dir = pose["output_dir"]
        meta = pose["_meta"]
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
    CATEGORY = "andypack/Animation"
    FUNCTION = "select"
    # One bundled ANIMATION dict (unpack it with Unpack Animation) instead of outputs.
    RETURN_TYPES = ("ANIM_ANIMATION",)
    RETURN_NAMES = ("ANIMATION",)

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
        if animation not in manifest.get("animations", {}):
            raise RuntimeError(
                f"CharacterAnimationSelector: unknown animation {animation!r} "
                "(stale or renamed) — pick an animation"
            )
        r = resolve_animation(manifest, root, character, animation, direction)
        if not r["selectable"]:
            raise RuntimeError(
                f"animation {animation}@{direction} not selectable: blocked_by={r['blocked_by']}"
            )
        # Bundle the loose outputs into one ANIMATION dict (see ANIMATION_OUTPUT_KEYS).
        # The wireable generation params (length/fps/width/height/shift) drive the
        # WanFirstLastFrameToVideo node + ModelSamplingSD3 directly.
        return (_build_animation_bundle(r),)


class AnimationFrameWriter:
    CATEGORY = "andypack/Animation"
    FUNCTION = "write"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("OUTPUT_DIR",)
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "animation": ("ANIM_ANIMATION",),
                "frames": ("IMAGE",),
            },
            "optional": {
                # Provenance only: the seed that drove the upstream sampler, wired
                # in so meta.json records what produced these frames. forceInput
                # makes it link-only — no widget, and crucially no ComfyUI
                # `control_after_generate` magic (which a `seed`-named widget gets)
                # that would mutate the recorded value out of sync with the sampler.
                "seed": ("INT", {"default": 0, "forceInput": True}),
            },
        }

    def write(self, animation, frames, seed=0):
        output_dir = animation["output_dir"]
        meta = animation["_meta"]
        # Reject an empty frame batch up front, before touching the existing render.
        # Writing it would produce a meta.json with count=0 and a negative-index
        # last_frame ("frame_-0001.png"), which animation_complete reads as
        # "complete" — a corrupt clip masquerading as done, then a FileNotFoundError
        # in any downstream animation that consumes it as an anchor.
        if int(frames.shape[0]) == 0:
            raise RuntimeError(
                "AnimationFrameWriter: received an empty frame batch; nothing to "
                "write (check the upstream sampler)"
            )
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
        # A loop (FFLF start==end) ends on a duplicate of its first frame; drop it
        # so the clip plays seamlessly on repeat. `meta["loop"]` is derived by the
        # resolver, not authored.
        if meta.get("loop") and len(batch) > 1:
            batch = io.apply_loop_closure(batch, drop_last=True)
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


# (key, output name) for each Unpack output, in slot order. The keys must cover
# the selector's leaf keys (POSE_OUTPUT_KEYS / ANIMATION_OUTPUT_KEYS) — a test
# enforces it — so unpacking exposes every leaf the selector produces.
_POSE_UNPACK = (
    ("source_image", "SOURCE_IMAGE"),
    ("pose_reference", "POSE_REFERENCE"),
    ("positive", "POSITIVE_PROMPT"),
    ("negative", "NEGATIVE_PROMPT"),
    ("output_dir", "OUTPUT_DIR"),
)
_ANIMATION_UNPACK = (
    ("start_image", "START_IMAGE"),
    ("end_image", "END_IMAGE"),
    ("positive", "POSITIVE_PROMPT"),
    ("negative", "NEGATIVE_PROMPT"),
    ("is_fflf", "IS_FFLF"),
    ("length", "LENGTH"),
    ("fps", "FPS"),
    ("width", "WIDTH"),
    ("height", "HEIGHT"),
    ("shift", "SHIFT"),
    ("output_dir", "OUTPUT_DIR"),
)


class PoseUnpack:
    """Fan a POSE dict out into its individual typed outputs, while also forwarding
    the whole POSE on — tap the fields you need and pass the rest along."""

    CATEGORY = "andypack/Pose"
    FUNCTION = "unpack"
    RETURN_TYPES = ("ANIM_POSE", "IMAGE", "IMAGE", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("POSE", *(name for _key, name in _POSE_UNPACK))

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"pose": ("ANIM_POSE",)}}

    def unpack(self, pose):
        return (pose, *(pose[key] for key, _name in _POSE_UNPACK))


class AnimationUnpack:
    """Fan an ANIMATION dict out into its individual typed outputs, while also
    forwarding the whole ANIMATION on — tap what you need and pass the rest along."""

    CATEGORY = "andypack/Animation"
    FUNCTION = "unpack"
    RETURN_TYPES = (
        "ANIM_ANIMATION", "IMAGE", "IMAGE", "STRING", "STRING", "BOOLEAN",
        "INT", "INT", "INT", "INT", "FLOAT", "STRING",
    )
    RETURN_NAMES = ("ANIMATION", *(name for _key, name in _ANIMATION_UNPACK))

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"animation": ("ANIM_ANIMATION",)}}

    def unpack(self, animation):
        return (animation, *(animation[key] for key, _name in _ANIMATION_UNPACK))


def _animated_preview(frames, fps) -> dict:
    """Write `frames` to ComfyUI's temp dir as an animated WEBP and return the UI
    payload that makes the node play it. Returns {} outside ComfyUI (no temp dir),
    so the node still works headless / under test."""
    try:
        import folder_paths
    except Exception:
        return {}
    full_dir, name, counter, subfolder, _ = folder_paths.get_save_image_path(
        "andypack_play", folder_paths.get_temp_directory(),
        int(frames.shape[2]), int(frames.shape[1]),
    )
    file = f"{name}_{counter:05}_.webp"
    images.save_animated_webp(frames, os.path.join(full_dir, file), fps)
    return {"images": [{"filename": file, "subfolder": subfolder, "type": "temp"}],
            "animated": (True,)}


class AnimationPlayback:
    """Play a rendered animation at the manifest fps, chaining its dependent clips:
    the start_from dep is prepended and the end_at dep appended (an animation dep
    plays its frames; a pose dep is held for `fps` frames ~1s). An action
    that returns to its start state loops `loops` times before the exit. Uses the
    same cascading selectors as the Character Animation Selector. Shows an in-node
    animated preview and outputs the assembled frame batch + fps."""

    CATEGORY = "andypack/Animation"
    FUNCTION = "play"
    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("FRAMES", "FPS")
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "manifest": ("ANIM_MANIFEST",),
                "character": (_character_choices(),),
                "category": ("STRING", {"default": ""}),
                "animation": ("STRING", {"default": ""}),
                "direction": ("STRING", {"default": ""}),
                "loops": ("INT", {"default": 1, "min": 1, "max": 64}),
            }
        }

    @classmethod
    def IS_CHANGED(cls, manifest, character, category, animation, direction, loops):
        # Re-resolve so the cache reflects the rendered tree (the action and its
        # chained clips being (re)rendered), plus the loop count.
        if character in ("", _NO_CHARACTER) or not animation or not direction:
            return float("nan")
        root = _characters_root()
        try:
            eff = effective_manifest(manifest, root, character)
            fps = resolve.animation_fps(eff, animation)
            segs = resolve.playback_segments(
                eff, root, character, animation, direction, loops=int(loops), fps=fps
            )
        except Exception:
            return float("nan")
        parts = [str(loops)]
        for s in segs:
            src = s["dir"] if s["kind"] == "anim" else s["image"]
            n = s.get("repeat", s.get("count"))
            parts.append(f"{s['kind']}:{n}:{src}:{_mtime(src)}")
        return "|".join(parts)

    def play(self, manifest, character, category, animation, direction, loops):
        if character in ("", _NO_CHARACTER):
            raise RuntimeError("AnimationPlayback: select a character first")
        if not animation or not direction:
            raise RuntimeError("AnimationPlayback: pick an animation and a direction")
        root = _characters_root()
        manifest = effective_manifest(manifest, root, character)
        fps = resolve.animation_fps(manifest, animation)
        segs = resolve.playback_segments(
            manifest, root, character, animation, direction, loops=int(loops), fps=fps
        )
        frames = images.assemble_playback(segs)
        if images.is_empty(frames):
            raise RuntimeError(
                f"AnimationPlayback: no rendered frames for {animation}@{direction}"
            )
        return {"ui": _animated_preview(frames, fps), "result": (frames, fps)}


class ManifestLint:
    """Surface the manifest's non-fatal lint findings (Wan-unfriendly lengths,
    directions outside the canonical list) as text in the graph."""

    CATEGORY = "andypack/Manifest"
    FUNCTION = "lint"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("REPORT",)
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"manifest": ("ANIM_MANIFEST",)}}

    def lint(self, manifest):
        findings = collect_warnings(manifest)
        report = (
            "OK — no lint findings"
            if not findings else "\n".join(f"⚠ {f}" for f in findings)
        )
        return (report,)


class CoverageReport:
    """A status table over every (entity, direction) for a character: what's
    generated, ready, stale, or blocked. Re-runs every queue so it reflects the
    current rendered tree."""

    CATEGORY = "andypack/Diagnostics"
    FUNCTION = "report"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("REPORT", "JSON")
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "manifest": ("ANIM_MANIFEST",),
                "character": (_character_choices(),),
            }
        }

    @classmethod
    def IS_CHANGED(cls, manifest, character):
        return float("nan")  # disk-backed report: always recompute

    def report(self, manifest, character):
        char = "" if character == _NO_CHARACTER else character
        data = api.coverage_report(manifest, _characters_root(), char)
        return (api.format_coverage_table(data), json.dumps(data, indent=2))


class MergedPromptReport:
    """Every entity×direction's fully merged positive/negative prompt — the exact
    cascade output a sampler would receive (identity → globals → entity →
    direction). A debugging aid. Character is optional: leave it on the placeholder
    to preview the manifest's prompts without a character's identity layer."""

    CATEGORY = "andypack/Diagnostics"
    FUNCTION = "report"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("REPORT", "JSON")
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "manifest": ("ANIM_MANIFEST",),
                "character": (_character_choices(),),
            }
        }

    @classmethod
    def IS_CHANGED(cls, manifest, character):
        return float("nan")  # reflects manifest/identity edits: always recompute

    def report(self, manifest, character):
        char = "" if character == _NO_CHARACTER else character
        rows = api.merged_prompt_rows(manifest, _characters_root(), char)
        return (api.format_merged_prompts(rows), json.dumps(rows, indent=2))


class RegenQueue:
    """The selectable-now (ready/stale) (entity, direction) cells in dependency
    order — the work list for a batch regeneration pass."""

    CATEGORY = "andypack/Diagnostics"
    FUNCTION = "build"
    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("QUEUE", "COUNT")
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "manifest": ("ANIM_MANIFEST",),
                "character": (_character_choices(),),
            }
        }

    @classmethod
    def IS_CHANGED(cls, manifest, character):
        return float("nan")  # disk-backed: always recompute

    def build(self, manifest, character):
        char = "" if character == _NO_CHARACTER else character
        queue = api.regen_queue(manifest, _characters_root(), char)
        text = "\n".join(
            f"{item['id']}@{item['direction']} [{item['status']}] ({item['kind']})"
            for item in queue
        )
        return (text, len(queue))


class AutoPoseSelector:
    """Auto-advancing batch selector: emit the NEXT actionable (ready/stale)
    non-root pose in dependency order, as an ANIM_POSE. Wire it like the Character
    Pose Selector (→ Unpack Pose → FLUX edit → Pose Frame Writer) and queue the
    graph repeatedly — each run generates the next pose until none remain (then it
    raises, the natural stop). Root poses (base) are skipped — use the Character
    Creator for those."""

    CATEGORY = "andypack/Pose"
    FUNCTION = "select"
    RETURN_TYPES = ("ANIM_POSE",)
    RETURN_NAMES = ("POSE",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "manifest": ("ANIM_MANIFEST",),
                "character": (_character_choices(),),
            }
        }

    @classmethod
    def IS_CHANGED(cls, manifest, character):
        # Re-run as the tree fills (the next job changes) or a prompt drifts.
        if character in ("", _NO_CHARACTER):
            return float("nan")
        root = _characters_root()
        try:
            eff = effective_manifest(manifest, root, character)
            job = api.next_actionable(eff, root, character, "pose", exclude_root=True)
            if not job:
                return "none"
            r = resolve_pose(eff, root, character, job["id"], job["direction"])
        except Exception:
            return float("nan")
        return f"{job['id']}@{job['direction']}|" + _selector_fingerprint(r, "source_image")

    def select(self, manifest, character):
        if character in ("", _NO_CHARACTER):
            raise RuntimeError("AutoPoseSelector: select a character first")
        root = _characters_root()
        manifest = effective_manifest(manifest, root, character)
        job = api.next_actionable(manifest, root, character, "pose", exclude_root=True)
        if not job:
            raise RuntimeError(
                "AutoPoseSelector: no actionable poses remain — every non-root pose "
                "is generated, blocked on an ungenerated dependency, or stale only "
                "because an upstream pose changed. Generate the base directions with "
                "the Character Creator first, and if a root pose is stale (its prompt "
                "changed) re-run the Character Creator to clear its descendants."
            )
        r = resolve_pose(manifest, root, character, job["id"], job["direction"])
        return (_build_pose_bundle(r),)


class AutoAnimationSelector:
    """Auto-advancing batch selector: emit the NEXT actionable (ready/stale)
    animation in dependency order, as an ANIM_ANIMATION. Wire it like the Character
    Animation Selector (→ Unpack Animation → WanFirstLastFrameToVideo → Animation
    Frame Writer) and queue the graph repeatedly — each run generates the next clip
    until none remain (then it raises, the natural stop)."""

    CATEGORY = "andypack/Animation"
    FUNCTION = "select"
    RETURN_TYPES = ("ANIM_ANIMATION",)
    RETURN_NAMES = ("ANIMATION",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "manifest": ("ANIM_MANIFEST",),
                "character": (_character_choices(),),
            }
        }

    @classmethod
    def IS_CHANGED(cls, manifest, character):
        if character in ("", _NO_CHARACTER):
            return float("nan")
        root = _characters_root()
        try:
            eff = effective_manifest(manifest, root, character)
            job = api.next_actionable(eff, root, character, "animation")
            if not job:
                return "none"
            r = resolve_animation(eff, root, character, job["id"], job["direction"])
        except Exception:
            return float("nan")
        return f"{job['id']}@{job['direction']}|" + _selector_fingerprint(
            r, "start_image", "end_image"
        )

    def select(self, manifest, character):
        if character in ("", _NO_CHARACTER):
            raise RuntimeError("AutoAnimationSelector: select a character first")
        root = _characters_root()
        manifest = effective_manifest(manifest, root, character)
        job = api.next_actionable(manifest, root, character, "animation")
        if not job:
            raise RuntimeError(
                "AutoAnimationSelector: no actionable animations remain — every "
                "animation is generated, blocked on an ungenerated anchor pose, or "
                "stale only because an upstream pose/clip changed (regenerate that "
                "upstream node to clear its dependents)"
            )
        r = resolve_animation(manifest, root, character, job["id"], job["direction"])
        return (_build_animation_bundle(r),)


class MirrorFrameWriter:
    """Synthesize a mirror-mapped direction from its already-generated source
    (e.g. WEST from EAST) by horizontally flipping the rendered payload — no
    sampling. Honors `mirror_map`; writes the completion sentinel last (atomic)."""

    CATEGORY = "andypack/Animation"
    FUNCTION = "write"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("OUTPUT_DIR",)
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "manifest": ("ANIM_MANIFEST",),
                "character": (_character_choices(),),
                "kind": (["animation", "pose"],),
                "id": ("STRING", {"default": ""}),
                "direction": ("STRING", {"default": ""}),
            }
        }

    @classmethod
    def IS_CHANGED(cls, manifest, character, kind, id, direction):
        if character in ("", _NO_CHARACTER) or not id or not direction:
            return float("nan")
        root = _characters_root()
        try:
            eff = effective_manifest(manifest, root, character)
            src_dir = (eff.get("mirror_map") or {}).get(direction)
            if not src_dir:
                return float("nan")
            if kind == "pose":
                src = resolve.pose_image_path(root, character, id, src_dir)
                return f"pose:{src}:{_mtime(src)}"
            d = resolve.animation_frame_dir(root, character, id, src_dir)
            meta = resolve.animation_meta_path(root, character, id, src_dir)
            return f"anim:{d}:{_mtime(meta)}"
        except Exception:
            return float("nan")

    def write(self, manifest, character, kind, id, direction):
        if character in ("", _NO_CHARACTER):
            raise RuntimeError("MirrorFrameWriter: select a character first")
        if not id or not direction:
            raise RuntimeError("MirrorFrameWriter: pick an id and a (mirrored) direction")
        root = _characters_root()
        manifest = effective_manifest(manifest, root, character)
        src_dir = (manifest.get("mirror_map") or {}).get(direction)
        if not src_dir:
            raise RuntimeError(
                f"direction {direction!r} is not in mirror_map; nothing to mirror from"
            )
        if kind == "pose":
            return (self._mirror_pose(manifest, root, character, id, direction, src_dir),)
        return (self._mirror_animation(manifest, root, character, id, direction, src_dir),)

    def _mirror_pose(self, manifest, root, character, pose_id, direction, src_dir):
        src_png = resolve.pose_image_path(root, character, pose_id, src_dir)
        if not os.path.exists(src_png):
            raise RuntimeError(f"source pose {pose_id}@{src_dir} is not generated")
        dst_png = resolve.pose_image_path(root, character, pose_id, direction)
        dst_sidecar = resolve.pose_sidecar_path(root, character, pose_id, direction)
        io.remove_if_exists(dst_sidecar)  # incomplete-first
        images.mirror_png(src_png, dst_png)
        r = resolve_pose(manifest, root, character, pose_id, direction)
        meta = {**r["meta"], "mirrored_from": {"direction": src_dir}}
        io.atomic_write_json(dst_sidecar, io.build_pose_sidecar(meta, created_utc=_utc_now()))
        return r["output_dir"]

    def _mirror_animation(self, manifest, root, character, anim_id, direction, src_dir):
        src_meta = resolve.read_node_meta(manifest, root, character, anim_id, src_dir)
        src_d = resolve.animation_frame_dir(root, character, anim_id, src_dir)
        if not src_meta:
            raise RuntimeError(f"source animation {anim_id}@{src_dir} is not generated")
        frames = sorted(
            n for n in os.listdir(src_d) if n.startswith("frame_") and n.endswith(".png")
        )
        # Reject a source whose meta.json survives but whose frame PNGs are gone:
        # mirroring it would write a count=0 meta with start/last_frame pointing at
        # a nonexistent frame_00000.png, which animation_complete reads as "done" —
        # a corrupt clip that FileNotFoundErrors any downstream anchor. Mirror the
        # AnimationFrameWriter empty-batch guard for this sibling write path.
        if not frames:
            raise RuntimeError(
                f"source animation {anim_id}@{src_dir} has meta.json but no frames "
                "to mirror (cleared or partially deleted)"
            )
        dst_d = resolve.animation_frame_dir(root, character, anim_id, direction)
        os.makedirs(dst_d, exist_ok=True)
        meta_path = resolve.animation_meta_path(root, character, anim_id, direction)
        io.remove_if_exists(meta_path)  # incomplete-first
        io.clear_frames(dst_d)
        for name in frames:
            images.mirror_png(os.path.join(src_d, name), os.path.join(dst_d, name))
        r = resolve_animation(manifest, root, character, anim_id, direction)
        count = len(frames)
        meta = {**r["meta"], "mirrored_from": {"direction": src_dir}}
        full = io.build_animation_meta(
            meta, count=count, start_frame=io.frame_name(0),
            last_frame=io.frame_name(max(count - 1, 0)),
            seed=src_meta.get("seed"), created_utc=_utc_now(),
        )
        io.atomic_write_json(meta_path, full)
        return dst_d


NODE_CLASS_MAPPINGS = {
    "AnimationManifestLoader": AnimationManifestLoader,
    "CharacterCreator": CharacterCreator,
    "CharacterReferenceLoader": CharacterReferenceLoader,
    "CharacterPoseSelector": CharacterPoseSelector,
    "AutoPoseSelector": AutoPoseSelector,
    "PoseFrameWriter": PoseFrameWriter,
    "PoseUnpack": PoseUnpack,
    "CharacterAnimationSelector": CharacterAnimationSelector,
    "AutoAnimationSelector": AutoAnimationSelector,
    "AnimationFrameWriter": AnimationFrameWriter,
    "AnimationUnpack": AnimationUnpack,
    "AnimationPlayback": AnimationPlayback,
    "MirrorFrameWriter": MirrorFrameWriter,
    "ManifestLint": ManifestLint,
    "CoverageReport": CoverageReport,
    "MergedPromptReport": MergedPromptReport,
    "RegenQueue": RegenQueue,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AnimationManifestLoader": "Animation Manifest Loader",
    "CharacterCreator": "Character Creator",
    "CharacterReferenceLoader": "Character Reference Loader",
    "CharacterPoseSelector": "Character Pose Selector",
    "AutoPoseSelector": "Auto Pose Selector (next job)",
    "PoseFrameWriter": "Pose Frame Writer",
    "PoseUnpack": "Unpack Pose",
    "CharacterAnimationSelector": "Character Animation Selector",
    "AutoAnimationSelector": "Auto Animation Selector (next job)",
    "AnimationFrameWriter": "Animation Frame Writer",
    "AnimationUnpack": "Unpack Animation",
    "AnimationPlayback": "Animation Playback",
    "MirrorFrameWriter": "Mirror Frame Writer",
    "ManifestLint": "Animation Manifest Lint",
    "CoverageReport": "Coverage Report",
    "MergedPromptReport": "Prompt Report",
    "RegenQueue": "Regen Queue",
}

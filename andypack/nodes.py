"""ComfyUI node classes — thin wrappers over andypack.resolve / io / images."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import torch

from andypack import api, atlas as _atlas_mod, images, io, manikins, resolve, sprites
from andypack.manifest import load_manifest
from andypack.resolve import effective_manifest, resolve_animation, resolve_pose

try:
    # comfy_execution is only importable inside a running ComfyUI process.
    # Guarded the same way andypack/server.py guards `from server import
    # PromptServer`, so `python -c "import andypack..."`, ruff, and mypy stay
    # green in CI (no ComfyUI installed there).
    from comfy_execution.graph_utils import GraphBuilder, is_link
except Exception:  # pragma: no cover - import-time guard outside ComfyUI
    GraphBuilder = None  # type: ignore[assignment,misc]
    is_link = None  # type: ignore[assignment]


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


def _character_choices():
    """Combo choices for a character dropdown: a placeholder + the character folders
    in the characters dir. Uses the same character detection as the `/characters`
    route (`api.list_characters`), so the combo and the route never disagree. The
    placeholder lets the cascade start unselected."""
    chars = [c["name"] for c in api.list_characters(_characters_root())]
    return [_NO_CHARACTER, *chars]


def _characters_root():
    return api.characters_dir() or "output/characters"


_CARDINAL_4 = ["EAST", "SOUTH", "WEST", "NORTH"]


def _atlas_directions(directions_arg: str) -> list[str]:
    if directions_arg == "cardinal_4":
        return _CARDINAL_4
    # "all" (default) — every direction in the canonical set
    return list(manikins.CANONICAL_DIRECTIONS)


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

        pose = _character_base_pose(
            "CharacterCreator", manifest, root, char_name, direction, image
        )
        return (pose,)


def _character_base_pose(label, manifest, root, char_name, direction, image):
    """Resolve the `base` pose for a character+direction through the effective
    manifest (character overlay applied) and pair the supplied reference `image`
    (first FLUX-edit reference) with the direction's bundled manikin (second) into
    an ANIM_POSE dict. Shared by Character Creator (which persists the prompt layer
    first) and the read-only Character Loader. `label` prefixes error messages with
    the calling node's name."""
    if direction not in manikins.CANONICAL_DIRECTIONS:
        raise RuntimeError(f"{label}: unknown direction {direction!r}")
    eff = effective_manifest(manifest, root, char_name)
    if "base" not in eff.get("poses", {}):
        raise RuntimeError(f"{label}: manifest has no 'base' pose")
    if direction not in eff["poses"]["base"]["directions"]:
        raise RuntimeError(f"{label}: base has no direction {direction!r}")
    r = resolve_pose(eff, root, char_name, "base", direction)
    manikin = images.load_image_tensor(manikins.manikin_path(direction))
    return {
        "source_image": image,        # the character reference (first reference)
        "pose_reference": manikin,    # the manikin for this direction (second)
        "positive": r["positive"],
        "negative": r["negative"],
        "output_dir": r["output_dir"],
        "_meta": r["meta"],
    }


def _build_pose_bundle(r: dict, root: str = "", character: str = "", sweep=None) -> dict:
    """An ANIM_POSE bundle from a resolve_pose result.

    A **root** pose (``meta["from"]`` is None, e.g. ``base``) is a multi-reference
    FLUX edit: source_image = the character's persisted reference art, pose_reference
    = the bundled manikin for its direction — the same pairing Character Creator
    makes, so Auto Pose Selector can drive the base turnaround too. A **derived**
    pose re-poses its `from`-source (single reference); pose_reference stays the
    empty sentinel.

    Raises if a root pose has no persisted reference — silently falling back to a
    blank sentinel would let PoseEditConditioning bake a near-blank reference latent
    with no error (the character must exist first; Character Creator persists the
    reference)."""
    meta = r["meta"]
    if meta.get("from") is None:
        ref_path = resolve.reference_image_path(root, character) if character else ""
        if not ref_path or not os.path.exists(ref_path):
            raise RuntimeError(
                f"_build_pose_bundle: {character!r} has no persisted reference "
                f"image (expected {ref_path or '<no character>'}); re-run the "
                "Character Creator with save_reference enabled, or supply the "
                "reference art directly, before targeting a root pose"
            )
        source = images.load_image_tensor(ref_path)
        direction = meta.get("direction", "")
        pose_reference = (
            images.load_image_tensor(manikins.manikin_path(direction))
            if direction in manikins.CANONICAL_DIRECTIONS
            else images.empty_image()
        )
    else:
        src = r["source_image"]
        source = images.load_image_tensor(src) if src else images.empty_image()
        pose_reference = images.empty_image()
    return {
        "source_image": source,
        "pose_reference": pose_reference,
        "positive": r["positive"],
        "negative": r["negative"],
        "output_dir": r["output_dir"],
        "_meta": meta,
        "_sweep": sweep or {},
    }


def _build_animation_bundle(r: dict, sweep=None) -> dict:
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
        "_sweep": sweep or {},
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


class CharacterPromptLoader:
    """Read a character's authored identity prompts from `character.json` as
    wireable STRINGs, so a txt2img / FLUX edit graph can drive the character's
    own positive/negative without hand-typing them. Unlike
    CharacterReferenceLoader (which needs persisted reference art), this only
    needs the authored layer, so it works before any render exists — e.g. to
    seed the Create stage's reference generation from the character combo."""

    CATEGORY = "andypack/Character"
    FUNCTION = "load"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("POSITIVE", "NEGATIVE")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"character": (_character_choices(),)}}

    @classmethod
    def IS_CHANGED(cls, character):
        if character in ("", _NO_CHARACTER):
            return float("nan")
        path = os.path.join(
            _characters_root(), io.to_snake_case(character), "character.json"
        )
        return f"{path}:{_mtime(path)}"

    def load(self, character):
        if character in ("", _NO_CHARACTER):
            raise RuntimeError("CharacterPromptLoader: select a character first")
        identity = resolve.read_character(
            _characters_root(), io.to_snake_case(character)
        )
        return (
            str(identity.get("positive_prompt", "") or ""),
            str(identity.get("negative_prompt", "") or ""),
        )


class CharacterLoader:
    """Emit the base-pose FLUX-edit job for an EXISTING character + direction,
    pairing the supplied reference `image` (first reference) with the bundled
    manikin (second) — exactly like the Character Creator, but READ-ONLY: it never
    writes character.json, so authored prompts survive. Use it in a Create graph
    that generates the reference art from an authored character.json (via
    CharacterPromptLoader → txt2img) and needs the base pose without re-authoring
    the prompt layer. A missing/empty character.json is not an error — the base
    pose's {character_prompt} just expands to empty."""

    CATEGORY = "andypack/Character"
    FUNCTION = "load"
    RETURN_TYPES = ("ANIM_POSE",)
    RETURN_NAMES = ("POSE",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "manifest": ("ANIM_MANIFEST",),
                "image": ("IMAGE",),
                "character": (_character_choices(),),
                "direction": (manikins.CANONICAL_DIRECTIONS,),
            },
            "optional": {
                # Persist the reference art to `<char>/_reference.png` so the
                # Stage-2 turnaround sweep can reload the root reference. Not a
                # prompt write — character.json is never touched here.
                "save_reference": ("BOOLEAN", {"default": True}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, manifest, image, character, direction, save_reference=True):
        if character in ("", _NO_CHARACTER):
            return float("nan")
        root = _characters_root()
        try:
            char_name = io.to_snake_case(character)
            eff = effective_manifest(manifest, root, char_name)
            r = resolve_pose(eff, root, char_name, "base", direction)
        except Exception:
            return float("nan")
        return "|".join([r["meta"]["prompt_hash"], direction, str(save_reference)])

    def load(self, manifest, image, character, direction, save_reference=True):
        if character in ("", _NO_CHARACTER):
            raise RuntimeError("CharacterLoader: select a character first")
        root = _characters_root()
        char_name = io.to_snake_case(character)
        pose = _character_base_pose(
            "CharacterLoader", manifest, root, char_name, direction, image
        )
        # Read-only w.r.t. the prompt layer; only the reference art may be written.
        if save_reference:
            images.save_image_png(image, resolve.reference_image_path(root, char_name))
        return (pose,)


class PoseSweepSelector:
    """Sweep or spot-fix the pose turnaround. mode=sweep emits the next actionable
    (ready/stale) non-root pose in dependency order — drive it inside a Sweep Loop
    to fill everything; queue the graph repeatedly and each run generates the next
    pose until none remain (then it raises, the natural stop). mode=target force-
    regenerates the named pose@direction (no completeness gate — the point is a
    spot-fix), leaving all others alone.

    With `include_base` on (sweep mode), root poses (base) are emitted too —
    paired with the bundled manikin (a 2-reference edit), so a SINGLE turnaround
    graph (Pose Sweep Selector → Pose Edit Conditioning → sampler → Pose Frame
    Writer) can generate the whole turnaround, base + derived. The character must
    exist first (run Character Creator once to persist its reference art)."""

    CATEGORY = "andypack/Pose"
    FUNCTION = "select"
    # One bundled POSE dict (unpack it with Unpack Pose) instead of loose outputs.
    RETURN_TYPES = ("ANIM_POSE",)
    RETURN_NAMES = ("POSE",)

    @classmethod
    def INPUT_TYPES(cls):
        # character is a real combo of character folders; category/pose/direction
        # are STRING widgets the web extension turns into the cascading combos
        # (pose/direction only matter in target mode; category scopes the sweep).
        return {
            "required": {
                "manifest": ("ANIM_MANIFEST",),
                "character": (_character_choices(),),
                "mode": (["sweep", "target"],),
                "skip_mirrored": ("BOOLEAN", {"default": True}),
                "include_base": ("BOOLEAN", {"default": True}),
                "category": ("STRING", {"default": ""}),
                "pose": ("STRING", {"default": ""}),
                "direction": ("STRING", {"default": ""}),
            },
            "optional": {
                # Wire Sweep Loop Open's flow token here (optional — the node
                # works standalone outside a loop). Accepted and passed through
                # unchanged; its only job is to place this selector (and
                # everything downstream of it, transitively including the
                # writer) on the Open->Close dependency path so Sweep Loop
                # Close's graph walk captures the whole body as the loop.
                "flow": ("SWEEP_FLOW",),
            },
        }

    @classmethod
    def IS_CHANGED(cls, *a, **k):
        # Disk-backed; re-read every execution — the sweep loop depends on this
        # (it must re-run each iteration and always see the current tree state).
        return float("nan")

    @staticmethod
    def _ctx(manifest, character, mode, skip_mirrored, include_base, category, pose, direction):
        return {
            "character": character, "kind": "pose", "mode": mode,
            "exclude_root": not include_base, "category": category or None,
            "skip_mirrored": skip_mirrored,
            "target": (pose, direction) if mode == "target" else None,
            # The writer takes no manifest input, but recomputing REMAINING
            # post-write needs one — stash the already-computed effective
            # manifest here (in-memory only; _sweep is never persisted).
            "manifest": manifest,
        }

    def select(self, manifest, character, mode, skip_mirrored, include_base,
               category, pose, direction, flow=None):
        # `flow` is accepted only so this selector sits on the Sweep Loop
        # Open->Close dependency path; its value is unused (ignored on purpose).
        del flow
        if character in ("", _NO_CHARACTER):
            raise RuntimeError("PoseSweepSelector: select a character first")
        root = _characters_root()
        manifest = effective_manifest(manifest, root, character)
        ctx = self._ctx(manifest, character, mode, skip_mirrored, include_base, category,
                        pose, direction)
        if mode == "target":
            if not pose or not direction:
                raise RuntimeError(
                    "PoseSweepSelector: target mode needs a pose and a direction"
                )
            if pose not in manifest.get("poses", {}):
                raise RuntimeError(
                    f"PoseSweepSelector: unknown pose {pose!r} (stale or renamed) — pick a pose"
                )
            r = resolve_pose(manifest, root, character, pose, direction)
            if not r["selectable"]:
                raise RuntimeError(
                    f"pose {pose}@{direction} not selectable: blocked_by={r['blocked_by']}"
                )
        else:
            job = api.next_actionable(
                manifest, root, character, "pose",
                exclude_root=not include_base, category=category or None,
                skip_mirrored=skip_mirrored,
            )
            if not job:
                raise RuntimeError(
                    "PoseSweepSelector: no actionable poses remain — every non-root pose "
                    "is generated, blocked on an ungenerated dependency, or stale only "
                    "because an upstream pose changed. Generate the base directions with "
                    "the Character Creator first, and if a root pose is stale (its prompt "
                    "changed) re-run the Character Creator to clear its descendants."
                )
            r = resolve_pose(manifest, root, character, job["id"], job["direction"])
        # Bundle the loose outputs into one POSE dict (see POSE_OUTPUT_KEYS), stamped
        # with sweep context so a later writer can recompute remaining work.
        return (_build_pose_bundle(r, root, character, sweep=ctx),)


def _sweep_remaining(bundle: dict) -> int:
    """The count of cells still actionable after a write, for the sweep loop's
    continue-signal. `target` mode (or any bundle with no `_sweep`, e.g. a
    manually-wired single write) returns 0 so the loop runs exactly once;
    `sweep` mode returns the live post-write count so the loop drains then
    stops cleanly at 0. Must be recomputed after each write — dependency depth
    means a write can unblock new cells the pre-write count didn't see."""
    s = bundle.get("_sweep") or {}
    if s.get("mode") != "sweep":
        return 0
    return api.remaining_actionable(
        s["manifest"], _characters_root(), s["character"], s["kind"],
        exclude_root=s.get("exclude_root", False),
        category=s.get("category"), skip_mirrored=s.get("skip_mirrored", False),
    )


class PoseFrameWriter:
    CATEGORY = "andypack/Pose"
    FUNCTION = "write"
    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("OUTPUT_DIR", "REMAINING")
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose": ("ANIM_POSE",),
                "image": ("IMAGE",),
            },
            "optional": {
                "mask": ("MASK",),
                # Provenance only: the seed that drove the upstream sampler.
                # forceInput = link-only, so no `control_after_generate` widget
                # can mutate the recorded value out of sync with the sampler.
                "seed": ("INT", {"default": 0, "forceInput": True}),
            },
        }

    def write(self, pose, image, mask=None, seed=0):
        output_dir = pose["output_dir"]
        meta = pose["_meta"]
        has_alpha = mask is not None or int(image.shape[-1]) == 4
        # Re-render discipline: drop the sidecar (completion sentinel) FIRST so an
        # interrupted rewrite reads as incomplete, then payload, then sidecar last.
        png_path = os.path.join(output_dir, meta["image"])
        sidecar_path = os.path.join(output_dir, f"{meta['direction']}.json")
        io.remove_if_exists(sidecar_path)
        images.save_image_png(image, png_path, mask=mask)
        sidecar = io.build_pose_sidecar(
            meta, created_utc=_utc_now(), has_alpha=has_alpha, seed=seed
        )
        io.atomic_write_json(sidecar_path, sidecar)
        return (output_dir, _sweep_remaining(pose))


class AnimationSweepSelector:
    """Sweep or spot-fix animations. mode=sweep emits the next actionable
    (ready/stale) animation in dependency order — drive it inside a Sweep Loop
    to fill everything; queue the graph repeatedly and each run generates the
    next clip until none remain (then it raises, the natural stop). mode=target
    force-regenerates the named animation@direction (no completeness gate — the
    point is a spot-fix), leaving all others alone.

    Set `category` to scope the sweep to one manifest category (e.g.
    "locomotion", "combat"); leave it empty to sweep all animations. Animations
    have no root/base concept (that's a pose-only distinction), so unlike the
    pose selector there is no `include_base` widget."""

    CATEGORY = "andypack/Animation"
    FUNCTION = "select"
    # One bundled ANIMATION dict (unpack it with Unpack Animation) instead of outputs.
    RETURN_TYPES = ("ANIM_ANIMATION",)
    RETURN_NAMES = ("ANIMATION",)

    @classmethod
    def INPUT_TYPES(cls):
        # character is a real combo of character folders; category/animation/direction
        # are STRING widgets the web extension turns into the cascading combos
        # (animation/direction only matter in target mode; category scopes the sweep).
        return {
            "required": {
                "manifest": ("ANIM_MANIFEST",),
                "character": (_character_choices(),),
                "mode": (["sweep", "target"],),
                "skip_mirrored": ("BOOLEAN", {"default": True}),
                "category": ("STRING", {"default": ""}),
                "animation": ("STRING", {"default": ""}),
                "direction": ("STRING", {"default": ""}),
            },
            "optional": {
                # Wire Sweep Loop Open's flow token here (optional — the node
                # works standalone outside a loop). Accepted and passed through
                # unchanged; its only job is to place this selector (and
                # everything downstream of it, transitively including the
                # writer) on the Open->Close dependency path so Sweep Loop
                # Close's graph walk captures the whole body as the loop.
                "flow": ("SWEEP_FLOW",),
            },
        }

    @classmethod
    def IS_CHANGED(cls, *a, **k):
        # Disk-backed; re-read every execution — the sweep loop depends on this
        # (it must re-run each iteration and always see the current tree state).
        return float("nan")

    @staticmethod
    def _ctx(manifest, character, mode, skip_mirrored, category, animation, direction):
        return {
            "character": character, "kind": "animation", "mode": mode,
            "exclude_root": False, "category": category or None,
            "skip_mirrored": skip_mirrored,
            "target": (animation, direction) if mode == "target" else None,
            # The writer takes no manifest input, but recomputing REMAINING
            # post-write needs one — stash the already-computed effective
            # manifest here (in-memory only; _sweep is never persisted).
            "manifest": manifest,
        }

    def select(self, manifest, character, mode, skip_mirrored, category, animation, direction,
               flow=None):
        # `flow` is accepted only so this selector sits on the Sweep Loop
        # Open->Close dependency path; its value is unused (ignored on purpose).
        del flow
        if character in ("", _NO_CHARACTER):
            raise RuntimeError("AnimationSweepSelector: select a character first")
        root = _characters_root()
        manifest = effective_manifest(manifest, root, character)
        ctx = self._ctx(manifest, character, mode, skip_mirrored, category, animation, direction)
        if mode == "target":
            if not animation or not direction:
                raise RuntimeError(
                    "AnimationSweepSelector: target mode needs an animation and a direction"
                )
            if animation not in manifest.get("animations", {}):
                raise RuntimeError(
                    f"AnimationSweepSelector: unknown animation {animation!r} "
                    "(stale or renamed) — pick an animation"
                )
            r = resolve_animation(manifest, root, character, animation, direction)
            if not r["selectable"]:
                raise RuntimeError(
                    f"animation {animation}@{direction} not selectable: blocked_by={r['blocked_by']}"
                )
        else:
            job = api.next_actionable(
                manifest, root, character, "animation",
                category=category or None, skip_mirrored=skip_mirrored,
            )
            if not job:
                scope = f" in category {category!r}" if category else ""
                raise RuntimeError(
                    f"AnimationSweepSelector: no actionable animations remain{scope} — "
                    "every animation is generated, blocked on an ungenerated anchor "
                    "pose, or stale only because an upstream pose/clip changed "
                    "(regenerate that upstream node to clear its dependents)"
                )
            r = resolve_animation(manifest, root, character, job["id"], job["direction"])
        # Bundle the loose outputs into one ANIMATION dict (see ANIMATION_OUTPUT_KEYS),
        # stamped with sweep context so a later writer can recompute remaining work.
        # The wireable generation params (length/fps/width/height/shift) drive the
        # WanFirstLastFrameToVideo node + ModelSamplingSD3 directly.
        return (_build_animation_bundle(r, sweep=ctx),)


class AnimationFrameWriter:
    CATEGORY = "andypack/Animation"
    FUNCTION = "write"
    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("OUTPUT_DIR", "REMAINING")
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
                "mask": ("MASK",),
                # Loop-seam mitigation: ramp a per-channel color match toward the
                # FIRST frame across the clip, so a start==end loop's drifted
                # closing frames land back on the opening palette. Applied only
                # when the resolver derived `loop` (no-op for non-loop clips).
                "loop_color_match": ("BOOLEAN", {"default": False}),
            },
        }

    def write(self, animation, frames, seed=0, mask=None, loop_color_match=False):
        output_dir = animation["output_dir"]
        meta = animation["_meta"]
        has_alpha = mask is not None or int(frames.shape[-1]) == 4
        # Reject an empty frame batch up front, before touching the existing render.
        # Writing it would produce a meta.json with count=0 and a negative-index
        # last_frame ("frame_-0001.png"), which animation_complete reads as
        # "complete" — a corrupt clip masquerading as done, then a FileNotFoundError
        # in any downstream animation that consumes it as an anchor.
        if images.is_empty(frames):
            raise RuntimeError(
                "AnimationFrameWriter: received an empty or 1x1 sentinel frame batch; "
                "nothing to write (check the upstream sampler)")

        if mask is not None:
            mask = mask if mask.dim() == 3 else mask.unsqueeze(0)
            if int(mask.shape[0]) not in (1, int(frames.shape[0])):
                raise RuntimeError(
                    f"AnimationFrameWriter: mask batch of {int(mask.shape[0])} frames "
                    f"doesn't match the {int(frames.shape[0])}-frame image batch — "
                    "supply one mask per frame, or a single mask to apply to all"
                )

        if loop_color_match and meta.get("loop") and int(frames.shape[0]) > 1:
            frames = images.match_color_ramp(frames, frames[0:1])

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
        # resolver, not authored. Loop closure only drops from the end (drop_last),
        # so batch[i] always corresponds to frames[i] — mask slicing by `index` is
        # safe for both the looping and non-looping paths.
        if meta.get("loop") and len(batch) > 1:
            batch = io.apply_loop_closure(batch, drop_last=True)
        for index, frame in enumerate(batch):
            if mask is None:
                frame_mask = None
            elif int(mask.shape[0]) == 1:
                frame_mask = mask[0:1]
            else:
                frame_mask = mask[index:index + 1]
            images.save_image_png(
                frame,
                os.path.join(output_dir, io.frame_name(index)),
                mask=frame_mask,
            )
        count = len(batch)
        full_meta = io.build_animation_meta(
            meta,
            count=count,
            start_frame=io.frame_name(0),
            last_frame=io.frame_name(count - 1),
            seed=seed,
            created_utc=_utc_now(),
            has_alpha=has_alpha,
        )
        io.atomic_write_json(meta_path, full_meta)
        return (output_dir, _sweep_remaining(animation))


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
    RETURN_TYPES = ("ANIM_POSE", "IMAGE", "IMAGE", "STRING", "STRING", "STRING", "BOOLEAN")
    RETURN_NAMES = ("POSE", *(name for _key, name in _POSE_UNPACK), "HAS_POSE_REFERENCE")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"pose": ("ANIM_POSE",)}}

    def unpack(self, pose):
        # HAS_POSE_REFERENCE: True for a manikin-driven root pose (2-ref FLUX edit),
        # False for a derived re-pose (the reference is the empty sentinel). Lets a
        # single turnaround graph feed the manikin only when present.
        has_ref = not images.is_empty(pose["pose_reference"])
        return (pose, *(pose[key] for key, _name in _POSE_UNPACK), has_ref)


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


def _image_preview(image) -> dict:
    """Write a single IMAGE to ComfyUI's temp dir as a PNG and return the UI payload
    so a sheet/diagnostic node shows its result inline. Returns {} outside ComfyUI."""
    try:
        import folder_paths
    except Exception:
        return {}
    full_dir, name, counter, subfolder, _ = folder_paths.get_save_image_path(
        "andypack_preview", folder_paths.get_temp_directory(),
        int(image.shape[2]), int(image.shape[1]),
    )
    file = f"{name}_{counter:05}_.png"
    images.save_image_png(image, os.path.join(full_dir, file))
    return {"images": [{"filename": file, "subfolder": subfolder, "type": "temp"}]}


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
        table = api.format_coverage_table(data)
        return {"ui": {"text": (table,)}, "result": (table, json.dumps(data, indent=2))}


class SpriteTrimPivot:
    CATEGORY = "andypack/Sprite"
    FUNCTION = "trim"
    RETURN_TYPES = ("IMAGE", "SPRITE_TRIM")
    RETURN_NAMES = ("TRIMMED", "TRIM_DATA")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "alpha_threshold": (
                    "FLOAT",
                    {"default": 0.03, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "trim_mode": (["union", "per_frame"],),
                "pivot": (["center", "bottom_center", "top_center", "custom"],),
            },
            "optional": {
                "pivot_x": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0}),
                "pivot_y": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0}),
                "pad": ("INT", {"default": 0, "min": 0, "max": 256}),
            },
        }

    def trim(
        self,
        image,
        alpha_threshold,
        trim_mode,
        pivot,
        pivot_x=0.5,
        pivot_y=1.0,
        pad=0,
    ):
        out, rects = sprites.trim_batch(image, threshold=alpha_threshold, mode=trim_mode, pad=pad)
        h, w = int(out.shape[1]), int(out.shape[2])
        px, py = sprites.pivot_point(w, h, pivot, custom=(pivot_x, pivot_y))
        for r in rects:
            r["pivot"] = [px, py]
        return (out, {"frames": rects, "trim_mode": trim_mode, "pivot_kind": pivot})


class SpritesheetPacker:
    CATEGORY = "andypack/Sprite"
    FUNCTION = "pack"
    RETURN_TYPES = ("IMAGE", "ANIM_ATLAS")
    RETURN_NAMES = ("SHEET", "ATLAS")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "layout": (["grid", "horizontal", "vertical", "maxrects"],),
                "columns": ("INT", {"default": 0, "min": 0}),
                "padding": ("INT", {"default": 2, "min": 0}),
                "extrude": ("INT", {"default": 0, "min": 0}),
                "power_of_two": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "trim_data": ("SPRITE_TRIM",),
                "fps": ("INT", {"forceInput": True}),
            },
        }

    def pack(
        self,
        image,
        layout,
        columns,
        padding,
        extrude,
        power_of_two,
        trim_data=None,
        fps=None,
    ):
        sheet, atlas_dict = sprites.pack_sheet(
            image,
            layout=layout,
            columns=columns,
            padding=padding,
            extrude=extrude,
            power_of_two=power_of_two,
            trim_data=trim_data,
        )
        if fps is not None:
            duration_ms = round(1000 / max(fps, 1))
            for frame in atlas_dict["frames"]:
                frame["duration_ms"] = duration_ms
        return (sheet, atlas_dict)


class AtlasMetadataWriter:
    CATEGORY = "andypack/Export"
    FUNCTION = "export"
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("OUTPUT_DIR",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "atlas": ("ANIM_ATLAS",),
                "sheet": ("IMAGE",),
                "format": (["json_hash", "json_array", "aseprite",
                             "godot_spriteframes", "unity", "texturepacker", "css"],),
                "name": ("STRING", {"default": ""}),
            },
            "optional": {
                "output_subdir": ("STRING", {"default": "atlas"}),
                "animation": ("ANIM_ANIMATION", {"forceInput": True}),
            },
        }

    def export(
        self,
        atlas,
        sheet,
        format,
        name,
        output_subdir="atlas",
        animation=None,
    ):
        name = (name or "").strip()
        if not name or name != os.path.basename(name) or ".." in name:
            raise RuntimeError(
                "AtlasMetadataWriter: 'name' must be a non-empty bare filename "
                f"(no directory separators), got {name!r}"
            )
        output_dir = os.path.join(api.output_dir() or "output", output_subdir)
        png_path = os.path.join(output_dir, f"{name}.png")
        images.save_image_png(sheet, png_path)
        text, ext = _atlas_mod.serialize(atlas, name, format)
        meta_path = os.path.join(output_dir, f"{name}{ext}")
        io.atomic_write_text(meta_path, text)
        if animation is not None:
            prompt_hash = animation.get("_meta", {}).get("prompt_hash")
            if prompt_hash is not None:
                prov = {"prompt_hash": prompt_hash}
                io.atomic_write_json(
                    os.path.join(output_dir, f"{name}.provenance.json"), prov
                )
        return {"ui": {}, "result": (output_dir,)}


class TurnaroundSheet:
    """Composite every rendered direction of a pose into one unlabeled contact sheet.

    Iterates over CANONICAL_DIRECTIONS in order; each rendered direction loads its
    PNG, each unrendered direction becomes a mid-gray placeholder. Returns a single
    IMAGE tensor suitable for previewing or saving."""

    CATEGORY = "andypack/Diagnostics"
    FUNCTION = "build"
    OUTPUT_NODE = True
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("SHEET",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "manifest": ("ANIM_MANIFEST",),
                "character": (_character_choices(),),
                "pose": ("STRING", {"default": "base"}),
            },
            "optional": {
                "columns": ("INT", {"default": 4, "min": 1, "max": 8}),
                "include_labels": ("BOOLEAN", {"default": True}),
                "cell_size": ("INT", {"default": 0, "min": 0, "max": 2048}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, manifest, character, pose, columns=4,
                   include_labels=True, cell_size=0):
        return float("nan")  # always recompute — reflects the rendered tree on disk

    def build(self, manifest, character, pose, columns=4,
              include_labels=True, cell_size=0):
        if character in ("", _NO_CHARACTER):
            raise RuntimeError("TurnaroundSheet: select a character first")
        root = _characters_root()
        tiles = []
        for direction in manikins.CANONICAL_DIRECTIONS:
            if resolve.pose_complete(root, character, pose, direction):
                path = resolve.pose_image_path(root, character, pose, direction)
                tiles.append(images.load_image_tensor(path))
            else:
                tiles.append(None)
        cell = (cell_size, cell_size) if cell_size > 0 else None
        labels = list(manikins.CANONICAL_DIRECTIONS) if include_labels else None
        sheet = images.contact_sheet(tiles, columns, cell=cell, labels=labels)
        return {"ui": _image_preview(sheet), "result": (sheet,)}


class AnimatedSpriteExport:
    """Export a frame batch as a looping animated GIF, APNG, or WebP.

    Optional onion-skinning composites ghosted neighbor frames for animator QA.
    Shows an in-node animated-WebP preview; writes the chosen format to the
    ComfyUI output directory.
    """

    CATEGORY = "andypack/Export"
    FUNCTION = "export"
    OUTPUT_NODE = True
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("PREVIEW", "OUTPUT_DIR")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "format": (["gif", "apng", "webp"],),
                "loop": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "fps": ("INT", {"default": 12, "forceInput": True}),
                "onion_skin": ("BOOLEAN", {"default": False}),
                "onion_prev": ("INT", {"default": 1, "min": 0, "max": 8}),
                "onion_next": ("INT", {"default": 0, "min": 0, "max": 8}),
                "onion_opacity": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0}),
                "name": ("STRING", {"default": "sprite"}),
            },
        }

    def export(
        self,
        image,
        format,
        loop,
        fps=12,
        onion_skin=False,
        onion_prev=1,
        onion_next=0,
        onion_opacity=0.3,
        name="sprite",
    ):
        fps_safe = max(int(fps), 1)
        frames = image
        if onion_skin:
            frames = images.onion_skin(
                image, int(onion_prev), int(onion_next), float(onion_opacity)
            )
        out_dir = api.output_dir() or "output"
        ext = format
        out_path = os.path.join(out_dir, f"{name}.{ext}")
        if format == "gif":
            images.save_animated_gif(frames, out_path, fps_safe, loop=loop)
        elif format == "apng":
            images.save_animated_apng(frames, out_path, fps_safe, loop=loop)
        else:
            images.save_animated_webp(frames, out_path, fps_safe, loop=loop)
        return {"ui": _animated_preview(frames, fps_safe), "result": (frames, out_dir)}


class FrameRetime:
    """Retime an IMAGE batch to a target fps (uniform resample / trim / pad-hold).
    Wan renders natively at 16fps; game sprites often want 8-12. Wire the source
    FPS from Animation Frames / Unpack Animation, pick a target, and feed the
    result to the packer/exporter with the new FPS."""

    CATEGORY = "andypack/Sprite"
    FUNCTION = "retime"
    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("FRAMES", "FPS")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE",),
                "fps": ("INT", {"default": 16, "min": 1, "forceInput": True}),
                "target_fps": ("INT", {"default": 12, "min": 1, "max": 120}),
                "mode": (["resample", "trim", "pad_hold"],),
            }
        }

    def retime(self, frames, fps, target_fps, mode):
        n = int(frames.shape[0])
        target = max(1, round(n * int(target_fps) / max(int(fps), 1)))
        return (images.retime_batch(frames, target, mode), int(target_fps))


class AnimationSheetBuilder:
    """Pack a full animation into a game-ready sprite sheet: one ROW per rendered
    direction, one COLUMN per frame. Unlike Character Atlas Builder (one frame per
    direction, a turnaround preview), this lays out every frame of every direction
    and emits a frame-accurate ANIM_ATLAS with per-direction tags + fps — feed it
    straight into Atlas Metadata Writer (aseprite / godot get one animation per
    direction). The single-node Stage-3 export."""

    CATEGORY = "andypack/Sprite"
    FUNCTION = "build"
    RETURN_TYPES = ("IMAGE", "ANIM_ATLAS", "STRING")
    RETURN_NAMES = ("SHEET", "ATLAS", "REPORT")
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "manifest": ("ANIM_MANIFEST",),
                "character": (_character_choices(),),
                "animation": ("STRING", {"default": ""}),
                "directions": (["all", "cardinal_4"],),
                "padding": ("INT", {"default": 2, "min": 0}),
                "power_of_two": ("BOOLEAN", {"default": False}),
            }
        }

    @classmethod
    def IS_CHANGED(cls, manifest, character, animation, directions, padding, power_of_two):
        if character in ("", _NO_CHARACTER) or not animation:
            return float("nan")
        root = _characters_root()
        try:
            eff = effective_manifest(manifest, root, character)
            dirs = _atlas_directions(directions)
            pairs = resolve.rendered_directions(eff, root, character, "animation", animation, dirs)
        except Exception:
            return float("nan")
        parts = [str(padding), str(power_of_two)]
        for d, path in pairs:
            meta = resolve.animation_meta_path(root, character, animation, d)
            parts.append(f"{d}:{path}:{_mtime(meta)}")
        return "|".join(parts)

    def build(self, manifest, character, animation, directions, padding, power_of_two):
        if character in ("", _NO_CHARACTER):
            raise RuntimeError("AnimationSheetBuilder: select a character first")
        if not animation:
            raise RuntimeError("AnimationSheetBuilder: pick an animation id")
        root = _characters_root()
        manifest = effective_manifest(manifest, root, character)
        dirs = _atlas_directions(directions)
        pairs = resolve.rendered_directions(manifest, root, character, "animation", animation, dirs)
        if not pairs:
            raise RuntimeError(
                f"AnimationSheetBuilder: no rendered directions for animation "
                f"{animation!r} (tried: {dirs})"
            )
        rows: list[tuple[str, list]] = []
        for d, path in pairs:
            frame_files = sorted(
                n for n in os.listdir(path)
                if n.startswith("frame_") and n.endswith(".png")
            )
            if not frame_files:
                raise RuntimeError(
                    f"AnimationSheetBuilder: {animation!r}@{d} has no frames in {path}"
                )
            frames = [
                images.load_image_tensor(os.path.join(path, fn), keep_alpha=True)
                for fn in frame_files
            ]
            rows.append((d, frames))
        fps = resolve.animation_fps(manifest, animation)
        sheet, atlas = sprites.pack_direction_rows(
            rows, fps=fps, padding=padding, power_of_two=power_of_two
        )
        report = (
            f"{animation}: {len(rows)} directions × up to "
            f"{max(len(f) for _d, f in rows)} frames @ {fps}fps\n"
            + "Rows: " + ", ".join(d for d, _f in rows)
        )
        return {"ui": _image_preview(sheet), "result": (sheet, atlas, report)}


class PoseEditConditioning:
    """Assemble the FLUX.2 pose-edit conditioning for a POSE in one node: text-encode
    the pose prompt, then attach the source image as a reference latent — and the
    manikin too when the pose carries one (a root/base pose). Derived poses have no
    manikin, so only the source is attached. This collapses the ~8-node reference
    chain (2× scale + 2× VAE-encode + 2× ReferenceLatent + ConditioningZeroOut +
    EmptyLatent) into one, and lets a SINGLE turnaround graph handle base + derived
    poses (no separate 1-ref / 2-ref workflows).

    Output: positive (text + reference latents), negative (zeroed), and an empty
    latent sized `width`×`height` (0 = derive from the source image)."""

    CATEGORY = "andypack/Pose"
    FUNCTION = "build"
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose": ("ANIM_POSE",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "megapixels": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 4.0, "step": 0.1}),
                "width": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 16}),
                "height": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 16}),
            }
        }

    def build(self, pose, clip, vae, megapixels, width, height):
        import comfy.utils
        import node_helpers

        def encode(text):
            tokens = clip.tokenize(text)
            return clip.encode_from_tokens_scheduled(tokens)

        def scale_to_mp(image):
            samples = image.movedim(-1, 1)
            _b, _c, h, w = samples.shape
            total = int(megapixels * 1024 * 1024)
            scale = (total / max(1, w * h)) ** 0.5
            tw, th = max(1, round(w * scale)), max(1, round(h * scale))
            s = comfy.utils.common_upscale(samples, tw, th, "lanczos", "disabled")
            return s.movedim(1, -1)

        def ref_latent(cond, image):
            pixels = scale_to_mp(image)[:, :, :, :3]
            latent = vae.encode(pixels)
            return node_helpers.conditioning_set_values(
                cond, {"reference_latents": [latent]}, append=True
            )

        positive = encode(pose["positive"])
        positive = ref_latent(positive, pose["source_image"])
        if not images.is_empty(pose["pose_reference"]):
            positive = ref_latent(positive, pose["pose_reference"])

        # Negative: zero out the positive text conditioning (FLUX has no negative
        # path; kept so the graph wires a valid negative at any CFG).
        negative = []
        for t in encode(pose["negative"] or ""):
            d = t[1].copy()
            if "pooled_output" in d and d["pooled_output"] is not None:
                d["pooled_output"] = torch.zeros_like(d["pooled_output"])
            negative.append([torch.zeros_like(t[0]), d])

        src = pose["source_image"]
        sh, sw = int(src.shape[1]), int(src.shape[2])
        out_w = width if width > 0 else (sw - sw % 16 or 16)
        out_h = height if height > 0 else (sh - sh % 16 or 16)
        # Size the empty output latent via the VAE itself, not a hand-built shape:
        # encode a blank at the target pixel size and zero it. A hardcoded
        # [1, 16, H//8, W//8] assumed an 8x VAE, but flux2-vae compresses 16x, so the
        # canvas came out 2x too large — and each derived pose (an edit of an edit)
        # compounded to 4x+, stretching anatomy. Encoding a blank tracks whatever
        # spatial compression + channel count the VAE actually has.
        latent = {"samples": torch.zeros_like(
            vae.encode(torch.zeros([1, out_h, out_w, 3]))
        )}
        return (positive, negative, latent)


class AnimationFrames:
    """Load a rendered animation clip's frames back as an IMAGE batch (+ its fps) —
    just the raw frames on disk, no dep-chaining or loop semantics. Use it to
    re-process a clip without re-sampling: re-matte, retime (Frame Retime), pack
    (Spritesheet Packer), or re-export."""

    CATEGORY = "andypack/Animation"
    FUNCTION = "load"
    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("FRAMES", "FPS")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "manifest": ("ANIM_MANIFEST",),
                "character": (_character_choices(),),
                "animation": ("STRING", {"default": ""}),
                "direction": ("STRING", {"default": ""}),
            }
        }

    @classmethod
    def IS_CHANGED(cls, manifest, character, animation, direction):
        if character in ("", _NO_CHARACTER) or not animation or not direction:
            return float("nan")
        return _mtime(resolve.animation_meta_path(_characters_root(), character, animation, direction))

    def load(self, manifest, character, animation, direction):
        if character in ("", _NO_CHARACTER):
            raise RuntimeError("AnimationFrames: select a character first")
        if not animation or not direction:
            raise RuntimeError("AnimationFrames: pick an animation and a direction")
        root = _characters_root()
        manifest = effective_manifest(manifest, root, character)
        pairs = resolve.rendered_directions(
            manifest, root, character, "animation", animation, [direction]
        )
        if not pairs:
            raise RuntimeError(
                f"AnimationFrames: {animation!r}@{direction} is not rendered"
            )
        _d, path = pairs[0]
        frame_files = sorted(
            n for n in os.listdir(path)
            if n.startswith("frame_") and n.endswith(".png")
        )
        if not frame_files:
            raise RuntimeError(f"AnimationFrames: no frames in {path}")
        tensors = [
            images.load_image_tensor(os.path.join(path, fn), keep_alpha=True)
            for fn in frame_files
        ]
        max_h = max(t.shape[1] for t in tensors)
        max_w = max(t.shape[2] for t in tensors)
        tensors = [images.pad_to(t, max_h, max_w) for t in tensors]
        frames = torch.cat(tensors, dim=0)
        return (frames, resolve.animation_fps(manifest, animation))


class SweepLoopOpen:
    """Marks the start of a one-press sweep loop and emits the SWEEP_FLOW token
    that brackets it. Wire `flow` into the sweep body's selector (Pose Sweep
    Selector / Animation Sweep Selector both take an optional `flow` input —
    ignored, just passed through) so the whole body sits on the Open->Close
    dependency path; Sweep Loop Close's graph walk needs that to find and
    re-clone the body every iteration.

    Unlike the validation spike (`SpikeLoopOpen`), this node carries NO fixed
    iteration budget — the real loop's continue/stop signal is the writer's
    live `REMAINING` output (recomputed off disk after every write), not a
    Python-side counter. Open here is deliberately trivial: just a distinctive
    flow-control token so nothing else can accidentally wire into that socket.
    """

    CATEGORY = "andypack/Loop"
    FUNCTION = "open"
    RETURN_TYPES = ("SWEEP_FLOW",)
    RETURN_NAMES = ("flow",)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    def open(self):
        return ({},)


class SweepLoopClose:
    """Closes a one-press sweep loop: while the writer's `remaining` is > 0,
    clones the Open->Close subgraph (the loop body) and re-expands it so the
    engine runs another iteration; at `remaining <= 0` it terminates (returns
    a plain result with no `expand` key).

    Mechanic ported verbatim (not re-derived) from the validated spike, before
    it was deleted, and, before that, from ComfyUI 0.26.2's own reference loop
    nodes (`tests/execution/testing_nodes/testing-pack/flow_control.py`
    `TestWhileLoopClose`): `flow` arrives as a raw, unresolved
    `[node_id, output_index]` link (`rawLink: True`) so `flow[0]` recovers the
    Open node's id; `dynprompt`/`unique_id` (hidden) let this node walk the
    CURRENT expanded graph backward from itself to find every dependency
    (`_explore_dependencies`), then forward from Open through that dependency
    set to collect just the loop body (`_collect_contained`); every contained
    node (including this Close, renamed "Recurse") is cloned into a fresh
    `GraphBuilder`, internal links rewired to the clones, and the whole thing
    is returned as `{"result": ..., "expand": graph.finalize()}` so the engine
    splices it in and runs it. See
    docs/superpowers/notes/2026-07-01-loop-spike-findings.md for the full
    contract, citations, and the spike file's original contents.

    No rewiring of the cloned Open's inputs is needed (unlike the reference
    `TestForLoopClose`, which injects a decremented counter into the clone):
    Open here carries no per-iteration state at all (see SweepLoopOpen), and
    `remaining` is recomputed fresh by the cloned writer, off disk, every
    iteration — there is nothing to inject.
    """

    CATEGORY = "andypack/Loop"
    FUNCTION = "close"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("DONE",)
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "flow": ("SWEEP_FLOW", {"forceInput": True, "rawLink": True}),
                "remaining": ("INT", {"forceInput": True}),
            },
            "hidden": {"dynprompt": "DYNPROMPT", "unique_id": "UNIQUE_ID"},
        }

    def _explore_dependencies(self, node_id, dynprompt, upstream) -> None:
        node_info = dynprompt.get_node(node_id)
        if "inputs" not in node_info:
            return
        for _key, value in node_info["inputs"].items():
            if is_link(value):
                parent_id = value[0]
                if parent_id not in upstream:
                    upstream[parent_id] = []
                    self._explore_dependencies(parent_id, dynprompt, upstream)
                upstream[parent_id].append(node_id)

    def _collect_contained(self, node_id, upstream, contained) -> None:
        if node_id not in upstream:
            return
        for child_id in upstream[node_id]:
            if child_id not in contained:
                contained[child_id] = True
                self._collect_contained(child_id, upstream, contained)

    def close(self, flow, remaining: int, dynprompt=None, unique_id=None):
        if remaining <= 0:
            return ("done",)

        assert dynprompt is not None
        assert unique_id is not None

        # `flow` arrived as a rawLink: [open_node_id, output_index].
        open_node_id = flow[0]

        upstream: dict[str, list[str]] = {}
        self._explore_dependencies(unique_id, dynprompt, upstream)

        contained: dict[str, bool] = {}
        self._collect_contained(open_node_id, upstream, contained)
        contained[unique_id] = True
        contained[open_node_id] = True

        graph = GraphBuilder()
        for node_id in contained:
            original = dynprompt.get_node(node_id)
            clone_id = "Recurse" if node_id == unique_id else node_id
            node = graph.node(original["class_type"], clone_id)
            node.set_override_display_id(node_id)
        for node_id in contained:
            original = dynprompt.get_node(node_id)
            clone_id = "Recurse" if node_id == unique_id else node_id
            node = graph.lookup_node(clone_id)
            assert node is not None
            for key, value in original["inputs"].items():
                if is_link(value) and value[0] in contained:
                    parent = graph.lookup_node(value[0])
                    assert parent is not None
                    node.set_input(key, parent.out(value[1]))
                else:
                    node.set_input(key, value)

        my_clone = graph.lookup_node("Recurse")
        assert my_clone is not None
        return {
            "result": ("looping",),
            "expand": graph.finalize(),
        }


NODE_CLASS_MAPPINGS = {
    "AnimationManifestLoader": AnimationManifestLoader,
    "CharacterCreator": CharacterCreator,
    "CharacterReferenceLoader": CharacterReferenceLoader,
    "CharacterPromptLoader": CharacterPromptLoader,
    "CharacterLoader": CharacterLoader,
    "PoseSweepSelector": PoseSweepSelector,
    "PoseFrameWriter": PoseFrameWriter,
    "PoseUnpack": PoseUnpack,
    "PoseEditConditioning": PoseEditConditioning,
    "AnimationSweepSelector": AnimationSweepSelector,
    "AnimationFrameWriter": AnimationFrameWriter,
    "AnimationUnpack": AnimationUnpack,
    "AnimationFrames": AnimationFrames,
    "CoverageReport": CoverageReport,
    "SpriteTrimPivot": SpriteTrimPivot,
    "SpritesheetPacker": SpritesheetPacker,
    "AtlasMetadataWriter": AtlasMetadataWriter,
    "AnimationSheetBuilder": AnimationSheetBuilder,
    "TurnaroundSheet": TurnaroundSheet,
    "AnimatedSpriteExport": AnimatedSpriteExport,
    "FrameRetime": FrameRetime,
    "SweepLoopOpen": SweepLoopOpen,
    "SweepLoopClose": SweepLoopClose,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AnimationManifestLoader": "Animation Manifest Loader",
    "CharacterCreator": "Character Creator",
    "CharacterReferenceLoader": "Character Reference Loader",
    "CharacterPromptLoader": "Character Prompt Loader",
    "CharacterLoader": "Character Loader",
    "PoseSweepSelector": "Pose Sweep Selector",
    "PoseFrameWriter": "Pose Frame Writer",
    "PoseUnpack": "Unpack Pose",
    "PoseEditConditioning": "Pose Edit Conditioning",
    "AnimationSweepSelector": "Animation Sweep Selector",
    "AnimationFrameWriter": "Animation Frame Writer",
    "AnimationUnpack": "Unpack Animation",
    "AnimationFrames": "Animation Frames (load)",
    "CoverageReport": "Coverage Report",
    "SpriteTrimPivot": "Sprite Trim & Pivot",
    "SpritesheetPacker": "Spritesheet Packer",
    "AtlasMetadataWriter": "Atlas Metadata Writer",
    "AnimationSheetBuilder": "Animation Sheet Builder",
    "TurnaroundSheet": "Turnaround Sheet",
    "AnimatedSpriteExport": "Animated Sprite Export",
    "FrameRetime": "Frame Retime",
    "SweepLoopOpen": "Sweep Loop Open",
    "SweepLoopClose": "Sweep Loop Close",
}

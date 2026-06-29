"""Pure FFLF dependency resolver: cascading prompts, completeness, anchors, staleness.

No ComfyUI / torch imports. Reads the rendered tree and `_concept.json` from disk;
the manifest dict is passed in (already validated by andypack.manifest).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import warnings
from typing import Any, Optional

from andypack.manifest import node_kind, validate_manifest

Manifest = dict[str, Any]

_WS = re.compile(r"\s+")
_SEP = "␟"  # UNIT SEPARATOR


# --- cascade: merge, identity, hashing -------------------------------------- #

def merge_layers(*parts: Optional[str]) -> str:
    """Join non-empty POSITIVE cascade layers, general -> specific, with a blank
    line (`\\n\\n`) between each. Each layer is kept verbatim (stripped of
    surrounding whitespace); empty/whitespace-only layers are dropped."""
    return "\n\n".join(part.strip() for part in parts if part and part.strip())


def merge_negative(*parts: Optional[str]) -> str:
    """Merge NEGATIVE cascade layers as comma-separated term lists: split each
    layer on commas, strip terms, case-insensitive dedupe (first occurrence
    wins), re-join with ', '. Negatives are almost always term lists, so this
    collapses duplicate boilerplate across layers."""
    seen: set[str] = set()
    out: list[str] = []
    for part in parts:
        if not part:
            continue
        for raw in part.split(","):
            term = raw.strip()
            if not term:
                continue
            key = term.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(term)
    return ", ".join(out)


def _normalize(text: str) -> str:
    return _WS.sub(" ", text.strip())


def _read_json(path: str) -> Optional[dict]:
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def read_identity(root: str, character: str) -> dict:
    """Per-character identity layer from `_concept.json`, or {} if absent/corrupt."""
    data = _read_json(os.path.join(root, character, "_concept.json"))
    return data or {}


def effective_manifest(manifest: Manifest, root: str, character: str) -> Manifest:
    """The manifest a character actually sees: the base manifest extended with
    the character's own `poses`/`animations` from `_concept.json` (character
    entries override/extend by id). Returns the base manifest unchanged when the
    character defines none. The merged manifest is re-validated, so a bad
    character ref or a cycle raises ManifestError instead of resolving silently
    or looping."""
    identity = read_identity(root, character)
    char_poses = identity.get("poses") or {}
    char_anims = identity.get("animations") or {}
    if not char_poses and not char_anims:
        return manifest
    merged: Manifest = {
        **manifest,
        "poses": {**manifest.get("poses", {}), **char_poses},
        "animations": {**manifest.get("animations", {}), **char_anims},
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # length warnings already surfaced at load
        validate_manifest(merged)
    return merged


def merged_prompts(
    manifest: Manifest, root: str, character: str, kind: str, entity_id: str, direction: str
) -> tuple[str, str]:
    """Cascade identity -> globals[kind] -> entity -> entity.directions[dir]."""
    identity = read_identity(root, character)
    glob = manifest.get("globals", {}).get(kind, {}) or {}
    collection = manifest["poses"] if kind == "pose" else manifest["animations"]
    entity = collection[entity_id]
    dlayer = (entity.get("directions", {}) or {}).get(direction) or {}

    positive = merge_layers(
        identity.get("positive_prompt"), glob.get("positive_prompt"),
        entity.get("positive_prompt"), dlayer.get("positive_prompt"),
    )
    negative = merge_negative(
        identity.get("negative_prompt"), glob.get("negative_prompt"),
        entity.get("negative_prompt"), dlayer.get("negative_prompt"),
    )
    return positive, negative


def compute_prompt_hash(
    manifest: Manifest, root: str, character: str, kind: str, entity_id: str, direction: str
) -> str:
    positive, negative = merged_prompts(manifest, root, character, kind, entity_id, direction)
    raw = _normalize(positive) + _SEP + _normalize(negative)
    return "sha1:" + hashlib.sha1(raw.encode("utf-8")).hexdigest()


# --- paths ------------------------------------------------------------------ #

def _concept_png(root: str, character: str) -> str:
    return os.path.join(root, character, "_concept.png")


def _pose_basedir(root: str, character: str, pose_id: str) -> str:
    return os.path.join(root, character, f"_{pose_id}")


def _pose_png(root: str, character: str, pose_id: str, direction: str) -> str:
    return os.path.join(_pose_basedir(root, character, pose_id), f"{direction}.png")


def _pose_sidecar(root: str, character: str, pose_id: str, direction: str) -> str:
    return os.path.join(_pose_basedir(root, character, pose_id), f"{direction}.json")


def _anim_dir(root: str, character: str, anim_id: str, direction: str) -> str:
    return os.path.join(root, character, anim_id, direction)


def _anim_meta_path(root: str, character: str, anim_id: str, direction: str) -> str:
    return os.path.join(_anim_dir(root, character, anim_id, direction), "meta.json")


# Public path accessors (nodes build payload paths through these instead of
# duplicating the on-disk layout).
def concept_image_path(root: str, character: str) -> str:
    return _concept_png(root, character)


def pose_image_path(root: str, character: str, pose_id: str, direction: str) -> str:
    return _pose_png(root, character, pose_id, direction)


def pose_sidecar_path(root: str, character: str, pose_id: str, direction: str) -> str:
    return _pose_sidecar(root, character, pose_id, direction)


def animation_frame_dir(root: str, character: str, anim_id: str, direction: str) -> str:
    return _anim_dir(root, character, anim_id, direction)


def animation_meta_path(root: str, character: str, anim_id: str, direction: str) -> str:
    return _anim_meta_path(root, character, anim_id, direction)


# --- direction resolution + completeness ------------------------------------ #

def resolved_dir(dep: dict, selected_dir: str) -> str:
    d = dep.get("direction", "same")
    return selected_dir if d in (None, "same") else d


def _count_frames(base: str) -> int:
    try:
        names = os.listdir(base)
    except OSError:
        return 0
    return sum(1 for n in names if n.startswith("frame_") and n.endswith(".png"))


def concept_complete(root: str, character: str) -> bool:
    return os.path.exists(_concept_png(root, character))


def pose_complete(root: str, character: str, pose_id: str, direction: str) -> bool:
    if not os.path.exists(_pose_png(root, character, pose_id, direction)):
        return False
    return _read_json(_pose_sidecar(root, character, pose_id, direction)) is not None


def animation_complete(root: str, character: str, anim_id: str, direction: str) -> bool:
    meta = _read_json(_anim_meta_path(root, character, anim_id, direction))
    if not meta:
        return False
    try:
        need = int(meta["frames"]["count"])
    except (KeyError, TypeError, ValueError):
        return False
    return _count_frames(_anim_dir(root, character, anim_id, direction)) >= need


def node_complete(manifest: Manifest, root: str, character: str, ref: str, direction: str) -> bool:
    kind = node_kind(manifest, ref)
    if kind == "concept":
        return concept_complete(root, character)
    if kind == "pose":
        return pose_complete(root, character, ref, direction)
    return animation_complete(root, character, ref, direction)


def read_node_meta(
    manifest: Manifest, root: str, character: str, ref: str, direction: str
) -> Optional[dict]:
    """The rendered sidecar/meta dict for a node, or None (concept / unrendered)."""
    kind = node_kind(manifest, ref)
    if kind == "concept":
        return None
    if kind == "pose":
        return _read_json(_pose_sidecar(root, character, ref, direction))
    return _read_json(_anim_meta_path(root, character, ref, direction))


def read_rendered_hash(
    manifest: Manifest, root: str, character: str, ref: str, direction: str
) -> Optional[str]:
    meta = read_node_meta(manifest, root, character, ref, direction)
    return meta.get("prompt_hash") if meta else None


def read_render_id(
    manifest: Manifest, root: str, character: str, ref: str, direction: str
) -> Optional[str]:
    """The rendered node's `render_id` (provenance token), or None when the node
    is a concept, unrendered, or was written before provenance existed."""
    meta = read_node_meta(manifest, root, character, ref, direction)
    return meta.get("render_id") if meta else None


def direct_deps(manifest: Manifest, ref: str, direction: str) -> list[tuple[str, str]]:
    """(dep_ref, resolved_dir) for a node's direct dependencies at `direction`:
    a pose's `from`; an animation's effective start_from + end_at. Concept refs
    are included (they have no render_id, recorded as None)."""
    kind = node_kind(manifest, ref)
    if kind == "concept":
        return []
    if kind == "pose":
        frm = manifest["poses"][ref]["from"]
        return [(frm["ref"], resolved_dir(frm, direction))]
    return [(dep["ref"], resolved_dir(dep, direction)) for _slot, dep in animation_deps(manifest, ref)]


def recorded_sources(
    manifest: Manifest, root: str, character: str, ref: str, direction: str
) -> dict[str, Optional[str]]:
    """`"dep_ref@dir" -> dep render_id` for each direct dep, captured at resolve
    time and persisted into the node's meta so `outdated` can later detect a
    consumed source being re-rendered (even with an unchanged prompt)."""
    sources: dict[str, Optional[str]] = {}
    for dep_ref, ddir in direct_deps(manifest, ref, direction):
        sources[f"{dep_ref}@{ddir}"] = read_render_id(manifest, root, character, dep_ref, ddir)
    return sources


# --- FFLF anchors ----------------------------------------------------------- #

def _single_image(manifest: Manifest, root: str, character: str, ref: str, direction: str) -> Optional[str]:
    """A concept/pose dep's single image (used for either FFLF slot)."""
    kind = node_kind(manifest, ref)
    if kind == "concept":
        return _concept_png(root, character)
    if kind == "pose":
        return _pose_png(root, character, ref, direction)
    return None  # animations are not single-image


def pose_source_image(
    manifest: Manifest, root: str, character: str, pose_id: str, direction: str
) -> Optional[str]:
    """The image a pose's FLUX edit consumes — its `from` source."""
    frm = manifest["poses"][pose_id]["from"]
    return _single_image(manifest, root, character, frm["ref"], resolved_dir(frm, direction))


def _animation_frame(
    manifest: Manifest, root: str, character: str, ref: str, direction: str, key: str
) -> Optional[str]:
    meta = _read_json(_anim_meta_path(root, character, ref, direction))
    if not meta or key not in meta:
        return None
    return os.path.join(_anim_dir(root, character, ref, direction), meta[key])


def effective_start_dep(manifest: Manifest, anim_id: str) -> Optional[dict]:
    """An animation's start anchor: its explicit `start_from`, else the
    manifest-level `defaults.start_from` (the I2V seed every animation needs)."""
    anim = manifest["animations"][anim_id]
    return anim.get("start_from") or manifest.get("defaults", {}).get("start_from")


def animation_deps(manifest: Manifest, anim_id: str) -> list[tuple[str, dict]]:
    """(slot, dep) pairs for an animation: the effective start_from (always
    present) plus end_at when declared (FFLF)."""
    deps: list[tuple[str, dict]] = []
    start = effective_start_dep(manifest, anim_id)
    if start:
        deps.append(("start_from", start))
    end = manifest["animations"][anim_id].get("end_at")
    if end:
        deps.append(("end_at", end))
    return deps


def _anchor_from_dep(
    manifest: Manifest, root: str, character: str, dep: Optional[dict], direction: str, frame_key: str
) -> Optional[str]:
    if not dep:
        return None
    ddir = resolved_dir(dep, direction)
    kind = node_kind(manifest, dep["ref"])
    if kind in ("concept", "pose"):
        return _single_image(manifest, root, character, dep["ref"], ddir)
    return _animation_frame(manifest, root, character, dep["ref"], ddir, frame_key)


def start_anchor(manifest: Manifest, root: str, character: str, anim_id: str, direction: str) -> Optional[str]:
    """start_from (or the default) -> dep's LAST frame (animation) or its single image."""
    dep = effective_start_dep(manifest, anim_id)
    return _anchor_from_dep(manifest, root, character, dep, direction, "last_frame")


def end_anchor(manifest: Manifest, root: str, character: str, anim_id: str, direction: str) -> Optional[str]:
    """end_at -> dep's FIRST frame (animation) or its single image (concept/pose)."""
    dep = manifest["animations"][anim_id].get("end_at")
    return _anchor_from_dep(manifest, root, character, dep, direction, "start_frame")


# --- transitive staleness --------------------------------------------------- #

def outdated(manifest: Manifest, root: str, character: str, ref: str, direction: str) -> bool:
    """A COMPLETE node is stale if its own merged-prompt hash drifted or any
    ancestor is outdated. Incompleteness is handled by `blocked`, not here."""
    kind = node_kind(manifest, ref)
    if kind == "concept":
        return False
    if not node_complete(manifest, root, character, ref, direction):
        return False
    meta = read_node_meta(manifest, root, character, ref, direction)
    current = compute_prompt_hash(manifest, root, character, kind, ref, direction)
    if (meta or {}).get("prompt_hash") != current:
        return True
    # Provenance: if this node recorded the render_id of each source it consumed,
    # a source whose current render_id differs (re-rendered, even with an
    # unchanged prompt) makes this node stale. Absent on pre-provenance metas, in
    # which case we fall back to the transitive-hash walk below.
    sources = (meta or {}).get("sources")
    if isinstance(sources, dict):
        for key, recorded in sources.items():
            dep_ref, ddir = key.rsplit("@", 1)
            if read_render_id(manifest, root, character, dep_ref, ddir) != recorded:
                return True
    if kind == "pose":
        frm = manifest["poses"][ref]["from"]
        return outdated(manifest, root, character, frm["ref"], resolved_dir(frm, direction))
    for _slot, dep in animation_deps(manifest, ref):
        if outdated(manifest, root, character, dep["ref"], resolved_dir(dep, direction)):
            return True
    return False


# --- resolve + status ------------------------------------------------------- #

def resolve_pose(manifest: Manifest, root: str, character: str, pose_id: str, direction: str) -> dict:
    pose = manifest["poses"][pose_id]
    frm = pose["from"]
    src_dir = resolved_dir(frm, direction)
    src_complete = node_complete(manifest, root, character, frm["ref"], src_dir)
    positive, negative = merged_prompts(manifest, root, character, "pose", pose_id, direction)
    return {
        "selectable": (direction in pose["directions"]) and src_complete,
        "blocked_by": [] if src_complete else [{"from": frm, "dir": src_dir}],
        "stale": src_complete and outdated(manifest, root, character, frm["ref"], src_dir),
        "source_image": pose_source_image(manifest, root, character, pose_id, direction)
        if src_complete else None,
        "positive": positive,
        "negative": negative,
        "output_dir": _pose_basedir(root, character, pose_id),
        "meta": {
            "kind": "pose", "pose": pose_id, "direction": direction, "from": frm,
            "image": f"{direction}.png", "manifest_version": manifest["version"],
            "prompt_hash": compute_prompt_hash(manifest, root, character, "pose", pose_id, direction),
            "sources": recorded_sources(manifest, root, character, pose_id, direction),
        },
    }


def resolve_animation(manifest: Manifest, root: str, character: str, anim_id: str, direction: str) -> dict:
    anim = manifest["animations"][anim_id]
    defaults = manifest.get("defaults", {})
    blocked_by: list[dict] = []
    stale: list[str] = []
    for slot, dep in animation_deps(manifest, anim_id):
        ddir = resolved_dir(dep, direction)
        if not node_complete(manifest, root, character, dep["ref"], ddir):
            blocked_by.append({slot: dep, "dir": ddir})
            continue
        if outdated(manifest, root, character, dep["ref"], ddir):
            stale.append(slot)
    positive, negative = merged_prompts(manifest, root, character, "animation", anim_id, direction)
    return {
        "selectable": (direction in anim["directions"]) and not blocked_by,
        "blocked_by": blocked_by,
        "stale": stale,
        "start_image": start_anchor(manifest, root, character, anim_id, direction),
        "end_image": end_anchor(manifest, root, character, anim_id, direction),
        "positive": positive,
        "negative": negative,
        "output_dir": _anim_dir(root, character, anim_id, direction),
        "meta": {
            "kind": "animation", "animation": anim_id, "direction": direction,
            "fps": anim.get("fps", defaults.get("fps")),
            "length": anim.get("length", defaults.get("length")),
            "loop": anim.get("loop", False), "manifest_version": manifest["version"],
            "prompt_hash": compute_prompt_hash(manifest, root, character, "animation", anim_id, direction),
            "sources": recorded_sources(manifest, root, character, anim_id, direction),
        },
    }


def status(manifest: Manifest, root: str, character: str, ref: str, direction: str) -> str:
    kind = node_kind(manifest, ref)
    if kind == "pose":
        r = resolve_pose(manifest, root, character, ref, direction)
        own_complete = pose_complete(root, character, ref, direction)
        dep_stale = bool(r["stale"])
    else:
        r = resolve_animation(manifest, root, character, ref, direction)
        own_complete = animation_complete(root, character, ref, direction)
        dep_stale = bool(r["stale"])
    if r["blocked_by"]:
        return "blocked"
    if own_complete:
        return "stale" if outdated(manifest, root, character, ref, direction) else "generated"
    return "stale" if dep_stale else "ready"

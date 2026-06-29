"""Pure FFLF dependency resolver: cascading prompts, completeness, anchors, staleness.

No ComfyUI / torch imports. Reads the rendered tree and `_concept.json` from disk;
the manifest dict is passed in (already validated by andypack.manifest).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Any, Optional

from andypack.manifest import node_kind

Manifest = dict[str, Any]

_WS = re.compile(r"\s+")
_SEP = "␟"  # UNIT SEPARATOR


# --- cascade: merge, identity, hashing -------------------------------------- #

def merge_layers(*parts: Optional[str]) -> str:
    """Join non-empty layers, comma-splitting, case-insensitive dedupe, first wins."""
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
        identity.get("prompt"), glob.get("prompt"), entity.get("prompt"), dlayer.get("prompt")
    )
    negative = merge_layers(
        identity.get("negative"), glob.get("negative"), entity.get("negative"), dlayer.get("negative")
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


def read_rendered_hash(
    manifest: Manifest, root: str, character: str, ref: str, direction: str
) -> Optional[str]:
    kind = node_kind(manifest, ref)
    if kind == "concept":
        return None
    if kind == "pose":
        meta = _read_json(_pose_sidecar(root, character, ref, direction))
    else:
        meta = _read_json(_anim_meta_path(root, character, ref, direction))
    return meta.get("prompt_hash") if meta else None


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


def _anchor(
    manifest: Manifest, root: str, character: str, anim_id: str, direction: str, slot: str, frame_key: str
) -> Optional[str]:
    dep = manifest["animations"][anim_id].get(slot)
    if not dep:
        return None
    ddir = resolved_dir(dep, direction)
    kind = node_kind(manifest, dep["ref"])
    if kind in ("concept", "pose"):
        return _single_image(manifest, root, character, dep["ref"], ddir)
    return _animation_frame(manifest, root, character, dep["ref"], ddir, frame_key)


def start_anchor(manifest: Manifest, root: str, character: str, anim_id: str, direction: str) -> Optional[str]:
    """start_from -> dep's LAST frame (animation) or its single image (concept/pose)."""
    return _anchor(manifest, root, character, anim_id, direction, "start_from", "last_frame")


def end_anchor(manifest: Manifest, root: str, character: str, anim_id: str, direction: str) -> Optional[str]:
    """end_at -> dep's FIRST frame (animation) or its single image (concept/pose)."""
    return _anchor(manifest, root, character, anim_id, direction, "end_at", "start_frame")


# --- transitive staleness --------------------------------------------------- #

def outdated(manifest: Manifest, root: str, character: str, ref: str, direction: str) -> bool:
    """A COMPLETE node is stale if its own merged-prompt hash drifted or any
    ancestor is outdated. Incompleteness is handled by `blocked`, not here."""
    kind = node_kind(manifest, ref)
    if kind == "concept":
        return False
    if not node_complete(manifest, root, character, ref, direction):
        return False
    rendered = read_rendered_hash(manifest, root, character, ref, direction)
    current = compute_prompt_hash(manifest, root, character, kind, ref, direction)
    if rendered != current:
        return True
    if kind == "pose":
        frm = manifest["poses"][ref]["from"]
        return outdated(manifest, root, character, frm["ref"], resolved_dir(frm, direction))
    anim = manifest["animations"][ref]
    for slot in ("start_from", "end_at"):
        dep = anim.get(slot)
        if dep and outdated(manifest, root, character, dep["ref"], resolved_dir(dep, direction)):
            return True
    return False

"""Atomic filesystem writes, meta/sidecar builders, and path-safety (pure stdlib)."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from typing import Any, Optional

_NON_SNAKE = re.compile(r"[^a-z0-9]+")

_RID_SEP = "␟"  # UNIT SEPARATOR


def render_id(prompt_hash: str, created_utc: str) -> str:
    """A per-render identity: changes when the merged prompt drifts (prompt_hash)
    OR when the node is simply re-rendered (created_utc). Descendants record the
    render_id they consumed, so provenance staleness catches an ancestor being
    re-rendered even with an unchanged prompt."""
    raw = f"{prompt_hash}{_RID_SEP}{created_utc}"
    return "rid:" + hashlib.sha1(raw.encode("utf-8")).hexdigest()


def atomic_write_json(path: str, data: dict) -> None:
    """Write JSON to a temp file in the same dir, then atomically replace `path`."""
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def frame_name(index: int) -> str:
    return f"frame_{index:05d}.png"


def remove_if_exists(path: str) -> None:
    """Best-effort unlink; ignores a missing file."""
    try:
        os.unlink(path)
    except OSError:
        pass


def clear_frames(output_dir: str) -> None:
    """Delete every `frame_*.png` in `output_dir` (missing dir is a no-op).

    Used before re-rendering an animation so a shorter clip cannot leave stale
    higher-index frames behind (which would inflate the completeness count and
    orphan frames past `last_frame`).
    """
    try:
        names = os.listdir(output_dir)
    except OSError:
        return
    for name in names:
        if name.startswith("frame_") and name.endswith(".png"):
            remove_if_exists(os.path.join(output_dir, name))


def apply_loop_closure(frames: list, *, drop_first: bool = False, drop_last: bool = False) -> list:
    """Trim seam frame(s) so a clip joins cleanly: drop the leading and/or trailing
    boundary frame. A clip of a single frame (or empty) is returned unchanged."""
    if len(frames) <= 1:
        return frames
    start = 1 if drop_first else 0
    stop = len(frames) - 1 if drop_last else len(frames)
    return frames[start:stop]


# The keys `build_concept_sidecar` owns: the identity layer it (re)writes plus the
# provenance it stamps. Everything else in an existing `_concept.json` (e.g.
# character-authored `poses`/`animations`, which effective_manifest reads) is
# preserved across a re-render rather than clobbered.
_CONCEPT_OWNED_KEYS = ("positive_prompt", "negative_prompt", "prompt_hash", "created_utc", "render_id")


def build_concept_sidecar(layer: dict, created_utc: str, existing: Optional[dict] = None) -> dict:
    """`_concept.json` = the (possibly empty) identity layer + provenance, merged
    over any `existing` sidecar so character-authored fields (e.g. `poses` /
    `animations`) survive a concept re-render. A concept has no merged prompt, so
    its identity layer is hashed as the prompt_hash; the render_id then changes
    when the concept is re-rendered (created_utc) or its identity edited (layer
    hash). Descendants record that render_id, so re-rendering the concept — the
    root of the tree — propagates staleness like any other source.

    The identity widgets are the source of truth for the identity layer: keys the
    new `layer` omits are dropped (so clearing a widget clears the stored value),
    while all non-owned `existing` keys pass through untouched."""
    raw = json.dumps(layer, sort_keys=True, ensure_ascii=False)
    prompt_hash = "sha1:" + hashlib.sha1(raw.encode("utf-8")).hexdigest()
    preserved = {
        k: v for k, v in (existing or {}).items() if k not in _CONCEPT_OWNED_KEYS
    }
    return {
        **preserved,
        **layer,
        "prompt_hash": prompt_hash,
        "created_utc": created_utc,
        "render_id": render_id(prompt_hash, created_utc),
    }


# The keys build_character owns (rewrites from the widgets). Everything else in
# an existing character.json — e.g. the character-authored poses/animations
# overlay that effective_manifest reads — is preserved across a rewrite.
_CHARACTER_OWNED_KEYS = ("positive_prompt", "negative_prompt")


def build_character(layer: dict, existing: Optional[dict] = None) -> dict:
    """character.json = the (possibly empty) character prompt layer, merged over
    any `existing` file so a character-authored `poses`/`animations` overlay
    survives. Unlike the old concept sidecar this carries NO provenance: the
    character is no longer a render node (the reference image is not persisted),
    so the tree's provenance roots at the base pose's own sidecars.

    The widgets are the source of truth for the prompt layer: keys the new
    `layer` omits are dropped (clearing a widget clears the stored value), while
    all non-owned `existing` keys pass through untouched."""
    preserved = {
        k: v for k, v in (existing or {}).items() if k not in _CHARACTER_OWNED_KEYS
    }
    return {**preserved, **layer}


def build_pose_sidecar(meta: dict, created_utc: str) -> dict:
    """Pose sidecar = resolve_pose meta + created_utc + render_id."""
    return {
        **meta,
        "created_utc": created_utc,
        "render_id": render_id(meta["prompt_hash"], created_utc),
    }


def build_animation_meta(
    meta: dict,
    *,
    count: int,
    start_frame: str,
    last_frame: str,
    seed: Optional[int],
    created_utc: str,
) -> dict:
    """Animation meta.json = resolve_animation meta + frame pointers + provenance."""
    full: dict[str, Any] = {
        **meta,
        "seed": seed,
        "frames": {"dir": ".", "pattern": "frame_{:05d}.png", "count": count},
        "start_frame": start_frame,
        "last_frame": last_frame,
        "created_utc": created_utc,
        "render_id": render_id(meta["prompt_hash"], created_utc),
    }
    return full


def to_snake_case(name: str) -> str:
    """Normalize a character name to a lowercase snake_case path segment.

    Lowercases, replaces every run of non-`[a-z0-9]` with a single underscore,
    and trims leading/trailing underscores. Raises ValueError if nothing usable
    remains (so we never build a path from an empty segment).
    """
    slug = _NON_SNAKE.sub("_", name.strip().lower()).strip("_")
    if not slug:
        raise ValueError(f"character name {name!r} has no usable characters")
    return slug


def list_json_names(directory: Optional[str]) -> list[str]:
    """Sorted basenames of `*.json` files directly in `directory`.

    Returns [] when `directory` is None or does not exist. Subdirectories and
    non-JSON files are ignored.
    """
    if not directory:
        return []
    try:
        entries = os.listdir(directory)
    except OSError:
        return []
    return sorted(
        n for n in entries
        if n.endswith(".json") and os.path.isfile(os.path.join(directory, n))
    )


def resolve_under(base: Optional[str], candidate: str) -> str:
    """Resolve a possibly-relative path against `base`.

    Absolute `candidate` passes through unchanged. A relative `candidate` is
    joined onto `base`; if `base` is falsy (e.g. ComfyUI's user dir is
    unavailable outside ComfyUI), `candidate` passes through as-is so it falls
    back to the process CWD.
    """
    if os.path.isabs(candidate) or not base:
        return candidate
    return os.path.join(base, candidate)

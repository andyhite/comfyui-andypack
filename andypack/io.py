"""Atomic filesystem writes, meta/sidecar builders, and path-safety (pure stdlib)."""

from __future__ import annotations

import json
import os
import re
import tempfile
from typing import Any, Optional

_NON_SNAKE = re.compile(r"[^a-z0-9]+")


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


def apply_loop_closure(frames: list, mode: str) -> list:
    """Make a loop seamless: drop the trailing closing frame, or duplicate the first."""
    if not frames:
        return frames
    if mode == "drop_last":
        return frames[:-1]
    if mode == "duplicate_first":
        return list(frames) + [frames[0]]
    raise ValueError(f"unknown loop_closure mode: {mode!r}")


def build_pose_sidecar(meta: dict, created_utc: str) -> dict:
    """Pose sidecar = resolve_pose meta + created_utc."""
    return {**meta, "created_utc": created_utc}


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


def safe_path(root: str, candidate: str) -> Optional[str]:
    """Resolve `candidate` under `root`, rejecting `..`, absolute, and symlink escapes.

    Returns the real absolute path if it is inside `root`, else None.
    """
    root_real = os.path.realpath(root)
    target = os.path.realpath(os.path.join(root_real, candidate))
    if target == root_real or target.startswith(root_real + os.sep):
        return target
    return None

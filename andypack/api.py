"""Pure JSON-payload builders for the anim_coord HTTP routes (stdlib only)."""

from __future__ import annotations

import os
from typing import Any

from andypack.resolve import (
    resolve_animation,
    resolve_pose,
    status,
)

Manifest = dict[str, Any]


def format_blocked(blocked_by: list) -> list[str]:
    """Render resolve blocked_by entries as '<ref>@<dir>' strings."""
    out: list[str] = []
    for entry in blocked_by:
        ddir = entry["dir"]
        for key, dep in entry.items():
            if key == "dir":
                continue
            out.append(f"{dep['ref']}@{ddir}")
    return out


def _is_character(root: str, name: str) -> bool:
    d = os.path.join(root, name)
    if not os.path.isdir(d):
        return False
    if os.path.exists(os.path.join(d, "_concept.png")):
        return True
    try:
        return any(os.path.isdir(os.path.join(d, c)) for c in os.listdir(d))
    except OSError:
        return False


def list_characters(root: str) -> list[dict]:
    """One-level scan of `root` for character directories."""
    try:
        names = sorted(os.listdir(root))
    except OSError:
        return []
    return [{"name": n} for n in names if _is_character(root, n)]


def list_options(manifest: Manifest, root: str, character: str) -> list[dict]:
    """Every (pose|animation, direction) with its UI status and blocked_by."""
    out: list[dict] = []
    for pid, pose in manifest.get("poses", {}).items():
        for direction in pose.get("directions", {}):
            r = resolve_pose(manifest, root, character, pid, direction)
            out.append({
                "kind": "pose", "id": pid, "direction": direction,
                "status": status(manifest, root, character, pid, direction),
                "blocked_by": format_blocked(r["blocked_by"]),
            })
    for aid, anim in manifest.get("animations", {}).items():
        for direction in anim.get("directions", {}):
            r = resolve_animation(manifest, root, character, aid, direction)
            out.append({
                "kind": "animation", "id": aid, "direction": direction,
                "status": status(manifest, root, character, aid, direction),
                "blocked_by": format_blocked(r["blocked_by"]),
            })
    return out

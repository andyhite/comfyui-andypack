"""Pure JSON-payload builders for the anim_coord HTTP routes (stdlib only)."""

from __future__ import annotations

import os
from typing import Any, Optional
from urllib.parse import quote

from andypack import io
from andypack.manifest import node_kind
from andypack.resolve import (
    effective_manifest,
    effective_start_dep,
    read_rendered_hash,
    resolve_animation,
    resolve_pose,
    resolved_dir,
    status,
)

Manifest = dict[str, Any]


def user_default_base() -> Optional[str]:
    """ComfyUI's `user/default` directory, or None when not running in ComfyUI."""
    try:
        import folder_paths
    except Exception:
        return None
    return os.path.join(folder_paths.get_user_directory(), "default")


def manifests_dir() -> Optional[str]:
    """The pack's manifest directory: `user/default/andypack/animations`.

    None when not running in ComfyUI.
    """
    base = user_default_base()
    if base is None:
        return None
    return os.path.join(base, "andypack", "animations")


def list_manifest_names() -> list[str]:
    """Available manifest filenames (e.g. `default.json`) in the manifests dir."""
    return io.list_json_names(manifests_dir())


def resolve_manifest_path(manifest_path: str) -> str:
    """Resolve a manifest path: absolute as-is; a bare name under the manifests dir."""
    return io.resolve_under(manifests_dir(), manifest_path)


def output_dir() -> Optional[str]:
    """ComfyUI's output directory, or None when not running in ComfyUI."""
    try:
        import folder_paths
    except Exception:
        return None
    return folder_paths.get_output_directory()


def under_output(rel: str) -> str:
    """Resolve `rel` under ComfyUI's output dir; absolute paths pass through.

    Outside ComfyUI (no output dir) `rel` falls back to itself (CWD-relative).
    """
    return io.resolve_under(output_dir(), rel)


def characters_dir() -> Optional[str]:
    """The root that holds per-character directories: `<output>/characters`.

    None when not running in ComfyUI.
    """
    base = output_dir()
    return None if base is None else os.path.join(base, "characters")


def list_subdirs(directory: Optional[str]) -> list[str]:
    """Sorted names of immediate subdirectories of `directory` (excluding dotfiles)."""
    if not directory:
        return []
    try:
        entries = os.listdir(directory)
    except OSError:
        return []
    return sorted(
        n for n in entries
        if not n.startswith(".") and os.path.isdir(os.path.join(directory, n))
    )


def split_character_dir(character_dir: str) -> tuple[str, str]:
    """Split a character_dir into (root, character) — the form resolve.* expects."""
    return os.path.split(os.path.normpath(character_dir))


def character_root_and_name(character_dir: str, character_name: str) -> tuple[str, str]:
    """(root, character) from either a full character_dir or a bare character
    name (resolved under the characters dir). Prefers an explicit character_dir."""
    if character_dir:
        return split_character_dir(character_dir)
    if character_name:
        base = characters_dir() or "output/characters"
        return split_character_dir(os.path.join(base, character_name))
    return ("", "")


def manifest_options(manifest: Manifest) -> dict:
    """The selectable structure of a manifest, for frontend combos.

    Returns ``{"poses": {id: [directions]}, "animations": {id: [directions]}}`` —
    derived purely from the manifest (no rendered tree needed), so the frontend
    can populate pose/animation and direction combos before anything is generated.
    """
    def dirs(entity: dict) -> list[str]:
        return list((entity.get("directions") or {}).keys())

    return {
        "poses": {pid: dirs(p) for pid, p in manifest.get("poses", {}).items()},
        "animations": {aid: dirs(a) for aid, a in manifest.get("animations", {}).items()},
    }


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
    """Every (pose|animation, direction) with its UI status and blocked_by.

    Uses the character's effective manifest, so character-specific poses and
    animations appear too.
    """
    manifest = effective_manifest(manifest, root, character)
    out: list[dict] = []
    for pid, pose in manifest.get("poses", {}).items():
        for direction in pose.get("directions", {}):
            r = resolve_pose(manifest, root, character, pid, direction)
            out.append({
                "kind": "pose", "id": pid, "direction": direction,
                "category": pose.get("category"),
                "status": status(manifest, root, character, pid, direction),
                "blocked_by": format_blocked(r["blocked_by"]),
            })
    for aid, anim in manifest.get("animations", {}).items():
        for direction in anim.get("directions", {}):
            r = resolve_animation(manifest, root, character, aid, direction)
            out.append({
                "kind": "animation", "id": aid, "direction": direction,
                "category": anim.get("category"),
                "status": status(manifest, root, character, aid, direction),
                "blocked_by": format_blocked(r["blocked_by"]),
            })
    return out


def _preview(
    manifest: Manifest, root: str, character: str,
    dep_ref: str, dep_dir: str, image_path: Optional[str], stale: bool,
) -> Optional[dict]:
    if not image_path:
        return None
    rel = os.path.relpath(image_path, root)
    version = read_rendered_hash(manifest, root, character, dep_ref, dep_dir) or ""
    url = (
        "/anim_coord/frame?"
        f"root={quote(root, safe='')}&path={quote(rel, safe='')}&v={quote(version, safe='')}"
    )
    return {"ref": dep_ref, "direction": dep_dir, "url": url, "stale": stale}


def resolve_payload(manifest: Manifest, root: str, character: str, ref: str, direction: str) -> dict:
    """Full resolve trimmed to UI fields, with source/dual anchor previews."""
    manifest = effective_manifest(manifest, root, character)
    kind = node_kind(manifest, ref)
    if kind == "pose":
        r = resolve_pose(manifest, root, character, ref, direction)
        frm = manifest["poses"][ref]["from"]
        sdir = resolved_dir(frm, direction)
        return {
            "selectable": r["selectable"],
            "blocked_by": format_blocked(r["blocked_by"]),
            "source_preview": _preview(
                manifest, root, character, frm["ref"], sdir, r["source_image"], bool(r["stale"])
            ),
        }
    r = resolve_animation(manifest, root, character, ref, direction)
    anim = manifest["animations"][ref]
    previews: dict[str, Any] = {"start_preview": None, "end_preview": None}
    for slot, key in (("start_from", "start_preview"), ("end_at", "end_preview")):
        # start_from falls back to the manifest default (so free clips preview
        # their I2V seed too); end_at is only ever explicit.
        dep = effective_start_dep(manifest, ref) if slot == "start_from" else anim.get(slot)
        if not dep:
            continue
        ddir = resolved_dir(dep, direction)
        image = r["start_image"] if slot == "start_from" else r["end_image"]
        previews[key] = _preview(
            manifest, root, character, dep["ref"], ddir, image, slot in r["stale"]
        )
    return {
        "selectable": r["selectable"],
        "blocked_by": format_blocked(r["blocked_by"]),
        **previews,
    }


def frame_path(root: str, rel: str) -> Optional[str]:
    """Confine `rel` under `root` and require it to exist; else None (=> 404)."""
    target = io.safe_path(root, rel)
    if target is None or not os.path.isfile(target):
        return None
    return target

"""Pure JSON-payload builders for the anim_coord HTTP routes (stdlib only)."""

from __future__ import annotations

import os
import shutil
from typing import Any, Optional

from andypack import io
from andypack.manifest import node_kind, topo_order
from andypack.resolve import (
    effective_manifest,
    merged_prompts,
    resolve_animation,
    resolve_pose,
    status,
)

Manifest = dict[str, Any]

# The manifest shipped in the repo, seeded into the user dir on first load.
BUNDLED_MANIFEST = os.path.join(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
    "examples",
    "animations.json",
)


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


def seed_default_manifest() -> bool:
    """Copy the bundled manifest to `<manifests_dir>/default.json` on first load.

    Gives a fresh install a working `default.json` (the name the selectors fall
    back to) without the user authoring one. Idempotent and non-destructive:
    a no-op outside ComfyUI (no manifests dir) or when the file already exists,
    so a user's edited manifest is never clobbered. Returns True iff it wrote.
    """
    dest_dir = manifests_dir()
    if dest_dir is None:
        return False
    dest = os.path.join(dest_dir, "default.json")
    if os.path.exists(dest):
        return False
    os.makedirs(dest_dir, exist_ok=True)
    shutil.copyfile(BUNDLED_MANIFEST, dest)
    return True


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


_COVERAGE_ORDER = ("blocked", "stale", "ready", "generated")


def coverage_report(manifest: Manifest, root: str, character: str) -> dict:
    """Every (entity, direction) with its status, plus per-status counts. Reuses
    the character's effective manifest (character-specific entities included)."""
    rows = list_options(manifest, root, character)
    summary = {key: 0 for key in _COVERAGE_ORDER}
    for row in rows:
        summary[row["status"]] = summary.get(row["status"], 0) + 1
    return {"character": character, "rows": rows, "summary": summary, "total": len(rows)}


def format_coverage_table(report: dict) -> str:
    """Render a coverage_report as a fixed-width text table for a Show-Text node."""
    rows = sorted(
        report["rows"],
        key=lambda r: (r["kind"], r.get("category") or "", r["id"], r["direction"]),
    )
    lines = [f"coverage for {report['character'] or '(none)'} — {report['total']} cells"]
    counts = report["summary"]
    lines.append(
        "  ".join(f"{key}={counts.get(key, 0)}" for key in _COVERAGE_ORDER)
    )
    lines.append("")
    lines.append(f"{'KIND':<10} {'CATEGORY':<12} {'ID':<26} {'DIR':<12} STATUS")
    for r in rows:
        blocked = f"  <- {', '.join(r['blocked_by'])}" if r["blocked_by"] else ""
        lines.append(
            f"{r['kind']:<10} {(r.get('category') or ''):<12} {r['id']:<26} "
            f"{r['direction']:<12} {r['status']}{blocked}"
        )
    return "\n".join(lines)


def regen_queue(manifest: Manifest, root: str, character: str) -> list[dict]:
    """Selectable-now (status `ready` or `stale`) (entity, direction) cells in
    dependency order — the work list for a batch regeneration pass. Blocked cells
    are omitted (their dependencies must be generated first)."""
    manifest = effective_manifest(manifest, root, character)
    out: list[dict] = []
    for ref in topo_order(manifest):
        kind = node_kind(manifest, ref)
        collection = manifest["poses"] if kind == "pose" else manifest["animations"]
        entity = collection.get(ref)
        if not entity:
            continue
        for direction in entity.get("directions", {}) or {}:
            st = status(manifest, root, character, ref, direction)
            if st in ("ready", "stale"):
                out.append({"kind": kind, "id": ref, "direction": direction, "status": st})
    return out


def merged_prompt_rows(manifest: Manifest, root: str, character: str) -> list[dict]:
    """Every (entity, direction) with its fully merged positive/negative prompts —
    the cascade output a sampler would receive. With a character, the character's
    effective manifest and identity layer are folded in; without one (``""``), the
    base manifest is used and no identity layer applies."""
    if character:
        manifest = effective_manifest(manifest, root, character)
    out: list[dict] = []
    for collection, kind in (
        (manifest.get("poses", {}), "pose"),
        (manifest.get("animations", {}), "animation"),
    ):
        for eid, entity in collection.items():
            for direction in entity.get("directions", {}) or {}:
                positive, negative = merged_prompts(manifest, root, character, kind, eid, direction)
                out.append({
                    "kind": kind, "id": eid, "direction": direction,
                    "category": entity.get("category"),
                    "positive": positive, "negative": negative,
                })
    return out


def format_merged_prompts(rows: list[dict]) -> str:
    """Render merged_prompt_rows as a readable block per (entity, direction)."""
    lines = [f"merged prompts — {len(rows)} cells"]
    for r in sorted(
        rows, key=lambda x: (x["kind"], x.get("category") or "", x["id"], x["direction"])
    ):
        cat = f"  ({r['category']})" if r.get("category") else ""
        lines.append("")
        lines.append(f"[{r['kind']}] {r['id']} @ {r['direction']}{cat}")
        lines.append(f"  + {r['positive'] or '(empty)'}")
        lines.append(f"  - {r['negative'] or '(empty)'}")
    return "\n".join(lines)



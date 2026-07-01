"""Pure JSON-payload builders for the anim_coord HTTP routes (stdlib only)."""

from __future__ import annotations

import json
import os
import shutil
import warnings
from typing import Any, Optional

from andypack import io
from andypack.manifest import (
    ManifestError,
    collect_warnings,
    node_kind,
    topo_order,
    validate_manifest,
)
from andypack.resolve import (
    animation_frame_dir,
    effective_manifest,
    invalidate_character,
    pose_image_path,
    reference_image_path,
    resolve_animation,
    resolve_pose,
    resolution_pass,
    stale_locally,
    status,
    status_from_resolved,
)

Manifest = dict[str, Any]


def _safe_effective(manifest: Manifest, root: str, character: str) -> Manifest:
    """`effective_manifest`, but degrade to the base manifest if the character's
    `character.json` overlay is structurally invalid (bad ref / cycle). The
    diagnostics + options reads are read-only views: surfacing the base manifest
    is better than aborting the queued graph with a ManifestError traceback (the
    selector IS_CHANGED hooks already swallow the same raise)."""
    try:
        return effective_manifest(manifest, root, character)
    except ManifestError:
        return manifest

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


def safe_manifest_path(name: str) -> Optional[str]:
    """Resolve a manifest by BARE NAME under the manifests dir, rejecting any
    name that is unsafe or traverses (the HTTP attack surface — unlike
    resolve_manifest_path, which allows absolute paths for trusted node inputs)."""
    if not manifest_name_is_safe(name):
        return None
    base = manifests_dir()
    return None if base is None else os.path.join(base, name)


# --- CRUD helpers for the sidebar GUI (write-capable, path-safe) ------------- #
#
# These back the editor panel. The HTTP routes never take a client filesystem
# path: a manifest is addressed by a *bare basename* (validated by
# `manifest_name_is_safe`) resolved under the server's own manifests dir, and a
# character by a name snake-cased to a single path segment under the characters
# dir. So there is no path the client can point outside the pack's own trees.

def manifest_name_is_safe(name: str) -> bool:
    """A manifest name the save/read routes will accept: a bare `*.json` basename
    with no directory part and no traversal. Rejects '', '.json', any name with a
    path separator or '..', and non-`.json` names."""
    if not name or not name.endswith(".json") or name == ".json":
        return False
    if name != os.path.basename(name):
        return False
    return ".." not in name and "/" not in name and "\\" not in name


def read_manifest_text(name: str) -> Optional[str]:
    """Raw text of a manifest by bare name, or None (unsafe name / absent / no
    manifests dir). The editor loads this verbatim so the user edits real JSON."""
    if not manifest_name_is_safe(name):
        return None
    base = manifests_dir()
    if base is None:
        return None
    try:
        with open(os.path.join(base, name), encoding="utf-8") as fh:
            return fh.read()
    except OSError:
        return None


def save_manifest_text(name: str, text: str) -> dict:
    """Validate `text` as a manifest and atomically write it to `<manifests>/name`.

    Returns `{"ok": True, "warnings": [...]}` on success (lint findings surfaced,
    not fatal), or `{"ok": False, "error": "..."}` without writing on an unsafe
    name, missing manifests dir, malformed JSON, or a structural ManifestError.
    Parsing + validating BEFORE the write means a bad edit can never overwrite a
    working manifest with a broken one."""
    if not manifest_name_is_safe(name):
        return {"ok": False, "error": f"unsafe manifest name {name!r}"}
    base = manifests_dir()
    if base is None:
        return {"ok": False, "error": "manifests dir unavailable (not in ComfyUI)"}
    try:
        data = json.loads(text)
    except ValueError as exc:
        return {"ok": False, "error": f"invalid JSON: {exc}"}
    if not isinstance(data, dict):
        return {"ok": False, "error": "manifest root must be an object"}
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            validate_manifest(data)
        lint = [str(w.message) for w in caught]
    except ManifestError as exc:
        return {"ok": False, "error": str(exc)}
    os.makedirs(base, exist_ok=True)
    io.atomic_write_json(os.path.join(base, name), data)
    return {"ok": True, "warnings": lint or collect_warnings(data)}


def _is_safe_segment(name: str) -> bool:
    """True if `name` is a single, safe path segment — non-empty, not `.`/`..`, not
    absolute, and containing no path separator. Character names addressed by the
    routes must satisfy this so a client value can't traverse out of the characters
    dir (the write paths already snake-case, which always yields a safe segment;
    this guards the read path too)."""
    if not name or name in (".", "..") or os.path.isabs(name):
        return False
    if os.sep in name or (os.altsep and os.altsep in name):
        return False
    return True


def read_character_layer(root: str, name: str) -> dict:
    """The character's `character.json` dict (prompt layer + any overlay), or {}
    when absent/corrupt/unsafe. A fresh disk read (not the resolve cache) so the
    editor always shows what's on disk. Rejects an unsafe `name` (path traversal)
    rather than reading outside the characters dir."""
    if not _is_safe_segment(name):
        return {}
    path = os.path.join(root, name, "character.json")
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def create_character(root: str, name: str) -> dict:
    """Create `<root>/<snake>/character.json` (empty layer) for a new character.

    Idempotent: an existing character is reported, not clobbered. Returns
    `{"ok": True, "name": <snake>, "created": bool}` or `{"ok": False, "error"}`
    on an unusable name."""
    try:
        snake = io.to_snake_case(name)
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}
    path = os.path.join(root, snake, "character.json")
    if os.path.exists(path):
        return {"ok": True, "name": snake, "created": False}
    existing = read_character_layer(root, snake)
    io.atomic_write_json(path, io.build_character({}, existing=existing))
    invalidate_character(root, snake)
    return {"ok": True, "name": snake, "created": True}


def save_character_layer(
    root: str,
    name: str,
    positive: str,
    negative: str,
    overlay: Optional[dict] = None,
) -> dict:
    """Write a character's prompt layer, preserving any poses/animations overlay.

    The widgets/fields are the source of truth: an empty positive/negative drops
    that key (matching the Character Creator node). Snake-cases the name to a path
    segment and invalidates the resolve cache so descendants re-resolve.

    When `overlay` is a dict its `poses`/`animations` keys (when each is a dict)
    are folded into the layer before `build_character`, so they are written to
    character.json. Existing callers pass 4 args (overlay defaults None) —
    unchanged behavior."""
    try:
        snake = io.to_snake_case(name)
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}
    layer: dict = {}
    if positive and positive.strip():
        layer["positive_prompt"] = positive.strip()
    if negative and negative.strip():
        layer["negative_prompt"] = negative.strip()
    if isinstance(overlay, dict):
        if isinstance(overlay.get("poses"), dict):
            layer["poses"] = overlay["poses"]
        if isinstance(overlay.get("animations"), dict):
            layer["animations"] = overlay["animations"]
    existing = read_character_layer(root, snake)
    payload = io.build_character(layer, existing=existing)
    io.atomic_write_json(os.path.join(root, snake, "character.json"), payload)
    invalidate_character(root, snake)
    return {"ok": True, "name": snake}


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


def thumb_path(
    root: str,
    character: str,
    kind: str,
    entity_id: str,
    direction: str,
) -> Optional[str]:
    """Return the on-disk path to the thumbnail source for a rendered cell.

    Validates ``character``, and (for non-reference kinds) ``entity_id`` and
    ``direction``, each via ``_is_safe_segment``. Returns ``None`` when any
    segment is unsafe, the kind is unrecognised, or the file/dir is absent.

    - ``kind="reference"`` → ``<char>/_reference.png``
    - ``kind="pose"``      → ``<char>/_<entity_id>/<direction>.png``
    - ``kind="animation"`` → first ``frame_*.png`` (sorted) in the frame dir
    """
    if not _is_safe_segment(character):
        return None
    if kind != "reference":
        if not _is_safe_segment(entity_id) or not _is_safe_segment(direction):
            return None
    if kind == "reference":
        path = reference_image_path(root, character)
    elif kind == "pose":
        path = pose_image_path(root, character, entity_id, direction)
    elif kind == "animation":
        frame_dir = animation_frame_dir(root, character, entity_id, direction)
        try:
            frames = sorted(
                n for n in os.listdir(frame_dir)
                if n.startswith("frame_") and n.endswith(".png")
            )
        except OSError:
            return None
        if not frames:
            return None
        path = os.path.join(frame_dir, frames[0])
    else:
        return None
    return path if os.path.exists(path) else None


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
        "mirror_map": manifest.get("mirror_map", {}),
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
    if os.path.exists(os.path.join(d, "character.json")):
        return True
    if os.path.exists(os.path.join(d, "_reference.png")):
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
    manifest = _safe_effective(manifest, root, character)
    out: list[dict] = []
    with resolution_pass():
        for pid, pose in manifest.get("poses", {}).items():
            for direction in pose.get("directions", {}):
                r = resolve_pose(manifest, root, character, pid, direction)
                out.append({
                    "kind": "pose", "id": pid, "direction": direction,
                    "category": pose.get("category"),
                    # A root pose (no `from`) is created by the Character Creator,
                    # not the generic pose selector — the frontend filters on this.
                    "root": pose.get("from") is None,
                    "status": status_from_resolved(manifest, root, character, pid, direction, r),
                    "blocked_by": format_blocked(r["blocked_by"]),
                })
        for aid, anim in manifest.get("animations", {}).items():
            for direction in anim.get("directions", {}):
                r = resolve_animation(manifest, root, character, aid, direction)
                out.append({
                    "kind": "animation", "id": aid, "direction": direction,
                    "category": anim.get("category"),
                    "root": False,
                    "status": status_from_resolved(manifest, root, character, aid, direction, r),
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
    manifest = _safe_effective(manifest, root, character)
    out: list[dict] = []
    with resolution_pass():
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


def _actionable_items(
    manifest: Manifest,
    root: str,
    character: str,
    kind: str,
    *,
    exclude_root: bool = False,
    category: Optional[str] = None,
    skip_mirrored: bool = False,
):
    """Yield selectable-now (ready/stale) cells of `kind` in dependency order —
    the shared filter that backs both `next_actionable` and
    `remaining_actionable`.

    `exclude_root` drops root poses (no `from`, e.g. `base`) — they need the
    Character Creator's reference image + manikin, not a generic selector.

    `category` filters to entities whose `category` field matches exactly.

    `skip_mirrored` drops directions that appear as keys in
    `manifest["mirror_map"]` — those directions are derived from a source
    direction and should not be independently queued."""
    eff = _safe_effective(manifest, root, character)
    mirror_keys = set(eff.get("mirror_map", {}).keys()) if skip_mirrored else set()
    for item in regen_queue(eff, root, character):
        if item["kind"] != kind:
            continue
        if exclude_root and kind == "pose":
            pose = eff.get("poses", {}).get(item["id"], {})
            if pose.get("from") is None:
                continue
        if category is not None:
            collection = eff.get("poses", {}) if kind == "pose" else eff.get("animations", {})
            entity = collection.get(item["id"], {})
            if entity.get("category") != category:
                continue
        if item["direction"] in mirror_keys:
            continue
        # Skip a cell that is stale ONLY because of an ancestor it can't fix by
        # re-rendering itself — re-running it wouldn't clear the staleness, so the
        # batch loop would wedge on it forever (e.g. every descendant of a stale
        # root pose that an auto-selector won't regenerate). A `ready` cell, or one
        # stale for its own reasons, is genuinely actionable.
        if item["status"] == "stale" and not stale_locally(
            eff, root, character, item["id"], item["direction"]
        ):
            continue
        yield item


def next_actionable(
    manifest: Manifest,
    root: str,
    character: str,
    kind: str,
    *,
    exclude_root: bool = False,
    category: Optional[str] = None,
    skip_mirrored: bool = False,
) -> Optional[dict]:
    """The first selectable-now (ready/stale) cell of `kind` in dependency order,
    or None when nothing of that kind is actionable. Backs the auto-advancing
    batch selectors: queue the graph repeatedly and each run picks the next job.

    `exclude_root` drops root poses (no `from`, e.g. `base`) — they need the
    Character Creator's reference image + manikin, not a generic selector.

    `category` filters to entities whose `category` field matches exactly.

    `skip_mirrored` drops directions that appear as keys in
    `manifest["mirror_map"]` — those directions are derived from a source
    direction and should not be independently queued."""
    for item in _actionable_items(
        manifest,
        root,
        character,
        kind,
        exclude_root=exclude_root,
        category=category,
        skip_mirrored=skip_mirrored,
    ):
        return item
    return None


def remaining_actionable(
    manifest: Manifest,
    root: str,
    character: str,
    kind: str,
    *,
    exclude_root: bool = False,
    category: Optional[str] = None,
    skip_mirrored: bool = False,
) -> int:
    """Count of currently-actionable cells of `kind`, using the same filter as
    `next_actionable`. Used by the frame writers to drive a one-press sweep
    loop's continue-signal after each render."""
    return sum(
        1
        for _ in _actionable_items(
            manifest,
            root,
            character,
            kind,
            exclude_root=exclude_root,
            category=category,
            skip_mirrored=skip_mirrored,
        )
    )
    return None



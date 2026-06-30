"""Pure FFLF dependency resolver: cascading prompts, completeness, anchors, staleness.

No ComfyUI / torch imports. Reads the rendered tree and `character.json` from disk;
the manifest dict is passed in (already validated by andypack.manifest).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import warnings
from contextlib import contextmanager
from typing import Any, Optional

from andypack.manifest import node_kind, validate_manifest

Manifest = dict[str, Any]

_WS = re.compile(r"\s+")
_SEP = "␟"  # UNIT SEPARATOR
# The opt-in template tokens, matched in a SINGLE pass so a token that appears
# inside an injected value is not re-expanded by a later substitution.
_TEMPLATE_TOKEN = re.compile(
    r"\{(character_prompt|direction_prompt|direction_name|view_phrase)\}"
)


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


_IDENTITY_CACHE: dict[str, tuple[float, dict]] = {}
# Validated character-effective manifests, keyed on a content-derived tuple
# (see _effective_cache_key). Cleared by invalidate_character.
_EFFECTIVE_CACHE: dict[tuple, Manifest] = {}


def _effective_cache_key(manifest: Manifest, identity: dict) -> tuple:
    """A cache key tied to CONTENT, not id(). A base-manifest edit changes the
    serialized poses/animations/globals/defaults/view_phrases, so the key
    changes and a stale merge can't be served. identity is keyed by its id()
    (stable per file version via read_character's mtime cache) plus its size."""
    payload = json.dumps(
        {k: manifest.get(k) for k in
         ("version", "poses", "animations", "globals", "defaults", "view_phrases")},
        sort_keys=True, default=str,
    )
    return (hashlib.sha1(payload.encode("utf-8")).hexdigest(), id(identity), len(identity))


def read_character(root: str, character: str) -> dict:
    """Per-character prompt layer from `character.json`, or {} if absent/corrupt.

    Memoized by path + mtime: the resolve/report hot paths re-read a character's
    `character.json` many times per cell (effective_manifest, merged_prompts,
    outdated, recorded_sources), so caching collapses that to one read per file
    version. A rewrite bumps the mtime, which invalidates the entry; the creator
    node also calls `invalidate_character` explicitly, because a rewrite can land
    within a coarse filesystem's mtime resolution window and leave the mtime
    unchanged. Callers must not mutate the returned dict."""
    path = os.path.join(root, character, "character.json")
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return {}
    cached = _IDENTITY_CACHE.get(path)
    if cached is not None and cached[0] == mtime:
        return cached[1]
    data = _read_json(path) or {}
    _IDENTITY_CACHE[path] = (mtime, data)
    return data


def invalidate_character(root: str, character: str) -> None:
    """Forget the cached character layer (and every effective manifest derived
    from any character layer), forcing the next read to hit disk and re-validate.

    The creator node calls this after rewriting `character.json`: a rewrite bumps
    the mtime but, on a coarse-mtime filesystem (FAT/exFAT, many NFS/SMB shares),
    can land within the mtime resolution window of the prior read — so mtime alone
    can't be trusted to invalidate. Clearing the entry guarantees descendants
    resolve against the new character layer rather than a stale cached copy."""
    _IDENTITY_CACHE.pop(os.path.join(root, character, "character.json"), None)
    # The effective-manifest cache keys on the identity object's id(); a popped
    # entry frees that object, so just drop the whole (small) cache rather than
    # track which entries derived from this character.
    _EFFECTIVE_CACHE.clear()


def effective_manifest(manifest: Manifest, root: str, character: str) -> Manifest:
    """The manifest a character actually sees: the base manifest extended with
    the character's own `poses`/`animations` from `character.json` (character
    entries override/extend by id). Returns the base manifest unchanged when the
    character defines none. The merged manifest is re-validated, so a bad
    character ref or a cycle raises ManifestError instead of resolving silently
    or looping."""
    identity = read_character(root, character)
    # `character.json` is user-authored; tolerate a malformed `poses`/`animations`
    # (e.g. a list) by ignoring it rather than crashing the `{**...}` merge below.
    char_poses = identity.get("poses")
    char_anims = identity.get("animations")
    char_poses = char_poses if isinstance(char_poses, dict) else {}
    char_anims = char_anims if isinstance(char_anims, dict) else {}
    if not char_poses and not char_anims:
        return manifest
    # Cache the merged + validated manifest keyed on the identities of its two
    # inputs: the base manifest object and the cached character dict (read_character
    # returns a stable object until the file changes or is invalidated). This keeps
    # validate_manifest's ref/cycle DFS off the IS_CHANGED hot path, which
    # re-derives the effective manifest on every graph evaluation. The cache is
    # dropped wholesale by invalidate_character when the character layer changes.
    key = _effective_cache_key(manifest, identity)
    cached = _EFFECTIVE_CACHE.get(key)
    if cached is not None:
        return cached
    merged: Manifest = {
        **manifest,
        "poses": {**manifest.get("poses", {}), **char_poses},
        "animations": {**manifest.get("animations", {}), **char_anims},
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # length warnings already surfaced at load
        validate_manifest(merged)
    _EFFECTIVE_CACHE[key] = merged
    return merged


def substitute_variables(
    text: Optional[str], *, positive: bool, identity: dict, direction_layer: dict,
    direction: str, view_phrase: str = "",
) -> Optional[str]:
    """Expand the opt-in template variables in a prompt layer, resolved by field
    context. `{character_prompt}` -> character positive/negative; `{direction_prompt}`
    -> the selected direction's positive/negative; `{direction_name}` -> the bare
    direction name (both contexts); `{view_phrase}` -> the manifest-level
    per-direction camera phrase (positive context only — it is affirmative camera
    language, so it expands to '' in a negative field and never pollutes the
    negative term list). Literal token replacement (not str.format), so unknown
    `{...}` tokens and stray braces survive; absent sources expand to ''.
    Applied per-layer BEFORE the merge so an expanded negative term list dedupes
    against sibling terms and a layer that resolves empty is dropped cleanly.

    Substitution is a SINGLE regex pass, so a token that appears inside an
    injected value (e.g. a literal `{direction_name}` stored in the identity
    layer) is left verbatim instead of being re-expanded by a later replacement —
    expansion is order-independent and never recursive."""
    if not text:
        return text
    field = "positive_prompt" if positive else "negative_prompt"
    values = {
        "character_prompt": (identity.get(field) or "").strip(),
        "direction_prompt": (direction_layer.get(field) or "").strip(),
        "direction_name": direction,
        "view_phrase": (view_phrase or "").strip() if positive else "",
    }
    return _TEMPLATE_TOKEN.sub(lambda m: values[m.group(1)], text)


def merged_prompts(
    manifest: Manifest, root: str, character: str, kind: str, entity_id: str, direction: str
) -> tuple[str, str]:
    """Compile a prompt: merge globals[kind] + entity, then substitute the opt-in
    template variables. Identity and the per-direction layer are NOT merged as
    cascade layers — they surface only via `{character_prompt}` /
    `{direction_prompt}` / `{direction_name}`, resolved by field (positive vs
    negative). Variables resolve in either the global or the entity prompt."""
    identity = read_character(root, character)
    glob = manifest.get("globals", {}).get(kind, {}) or {}
    collection = manifest["poses"] if kind == "pose" else manifest["animations"]
    entity = collection[entity_id]
    dlayer = (entity.get("directions", {}) or {}).get(direction) or {}
    view_phrase = (manifest.get("view_phrases") or {}).get(direction) or ""

    def sub(text: Optional[str], *, positive: bool) -> Optional[str]:
        return substitute_variables(
            text, positive=positive, identity=identity,
            direction_layer=dlayer, direction=direction, view_phrase=view_phrase,
        )

    positive = merge_layers(
        sub(glob.get("positive_prompt"), positive=True),
        sub(entity.get("positive_prompt"), positive=True),
    )
    negative = merge_negative(
        sub(glob.get("negative_prompt"), positive=False),
        sub(entity.get("negative_prompt"), positive=False),
    )
    return positive, negative


def hash_prompts(positive: str, negative: str) -> str:
    """The merged-prompt staleness hash for a (positive, negative) pair. Whitespace-
    normalized so cosmetic edits don't drift it. Callers that already merged the
    prompts hash them directly; `compute_prompt_hash` merges first."""
    raw = _normalize(positive) + _SEP + _normalize(negative)
    return "sha1:" + hashlib.sha1(raw.encode("utf-8")).hexdigest()


def compute_prompt_hash(
    manifest: Manifest, root: str, character: str, kind: str, entity_id: str, direction: str
) -> str:
    return hash_prompts(*merged_prompts(manifest, root, character, kind, entity_id, direction))


# --- paths ------------------------------------------------------------------ #

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
def pose_image_path(root: str, character: str, pose_id: str, direction: str) -> str:
    return _pose_png(root, character, pose_id, direction)


def pose_sidecar_path(root: str, character: str, pose_id: str, direction: str) -> str:
    return _pose_sidecar(root, character, pose_id, direction)


def animation_frame_dir(root: str, character: str, anim_id: str, direction: str) -> str:
    return _anim_dir(root, character, anim_id, direction)


def animation_meta_path(root: str, character: str, anim_id: str, direction: str) -> str:
    return _anim_meta_path(root, character, anim_id, direction)


def reference_image_path(root: str, character: str) -> str:
    """The optional persisted character reference art (`<char>/_reference.png`).

    The reference is the concept image the Character Creator edits into the base
    directions. Persisting it (opt-in on the creator node) lets a character be
    reloaded and its base re-generated later without hunting for the original art.
    It is NOT a render node and carries no provenance — base sidecars still root
    the tree's staleness."""
    return os.path.join(root, character, "_reference.png")


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
    if kind == "pose":
        return pose_complete(root, character, ref, direction)
    return animation_complete(root, character, ref, direction)


def read_node_meta(
    manifest: Manifest, root: str, character: str, ref: str, direction: str
) -> Optional[dict]:
    """The rendered sidecar/meta dict for a node, or None (unrendered)."""
    kind = node_kind(manifest, ref)
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
    """The rendered node's `render_id` (provenance token), or None when unrendered
    or written before provenance existed. The base pose roots the tree, so its
    sidecar render_id is what descendants record and stale against."""
    meta = read_node_meta(manifest, root, character, ref, direction)
    return meta.get("render_id") if meta else None


def direct_deps(manifest: Manifest, ref: str, direction: str) -> list[tuple[str, str]]:
    """(dep_ref, resolved_dir) for a node's direct dependencies at `direction`:
    a pose's `from`; an animation's effective start_from + end_at. A root pose
    (no `from`) has no direct deps."""
    kind = node_kind(manifest, ref)
    if kind == "pose":
        frm = manifest["poses"][ref].get("from")
        return [(frm["ref"], resolved_dir(frm, direction))] if frm else []
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
    """A pose dep's single image (used for either FFLF slot)."""
    kind = node_kind(manifest, ref)
    if kind == "pose":
        return _pose_png(root, character, ref, direction)
    return None  # animations are not single-image


def pose_source_image(
    manifest: Manifest, root: str, character: str, pose_id: str, direction: str
) -> Optional[str]:
    """The image a pose's FLUX edit consumes — its `from` source, or None for a
    root pose (no `from`; the creator node supplies the reference image)."""
    frm = manifest["poses"][pose_id].get("from")
    if not frm:
        return None
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
    if node_kind(manifest, dep["ref"]) == "pose":
        return _single_image(manifest, root, character, dep["ref"], ddir)
    return _animation_frame(manifest, root, character, dep["ref"], ddir, frame_key)


def start_anchor(manifest: Manifest, root: str, character: str, anim_id: str, direction: str) -> Optional[str]:
    """start_from (or the default) -> dep's LAST frame (animation) or its single image."""
    dep = effective_start_dep(manifest, anim_id)
    return _anchor_from_dep(manifest, root, character, dep, direction, "last_frame")


def end_anchor(manifest: Manifest, root: str, character: str, anim_id: str, direction: str) -> Optional[str]:
    """end_at -> dep's FIRST frame (animation) or its single image (pose)."""
    dep = manifest["animations"][anim_id].get("end_at")
    return _anchor_from_dep(manifest, root, character, dep, direction, "start_frame")


# --- transitive staleness --------------------------------------------------- #

_OUTDATED_MEMO: Optional[dict[tuple[str, str, str], bool]] = None


@contextmanager
def resolution_pass():
    """Memoize `outdated()` across one read-only resolve pass. A coverage / regen /
    status report walks many (entity, direction) cells that share ancestors;
    without this each cell re-reads every ancestor's meta and recomputes its
    prompt hash, so a shared root is re-walked once per descendant — O(cells x
    depth). The rendered tree must not change inside the pass — true for the report
    builders, which only read. Nested passes reuse the outer cache; the cache is
    dropped on exit so it can never serve a later render."""
    global _OUTDATED_MEMO
    if _OUTDATED_MEMO is not None:
        yield
        return
    _OUTDATED_MEMO = {}
    try:
        yield
    finally:
        _OUTDATED_MEMO = None


def outdated(manifest: Manifest, root: str, character: str, ref: str, direction: str) -> bool:
    """A COMPLETE node is stale if its own merged-prompt hash drifted or any
    ancestor is outdated. Incompleteness is handled by `blocked`, not here.
    Memoized by (character, ref, direction) within a `resolution_pass()`, where the
    rendered tree is read-only."""
    memo = _OUTDATED_MEMO
    if memo is None:
        return _outdated(manifest, root, character, ref, direction)
    key = (character, ref, direction)
    if key not in memo:
        memo[key] = _outdated(manifest, root, character, ref, direction)
    return memo[key]


def _sources_drifted(
    manifest: Manifest, root: str, character: str, ref: str, direction: str, meta: Optional[dict]
) -> bool:
    """True if the recorded `sources` key-set differs from the current dep set
    (an anchor ref was swapped / added / removed), OR a recorded source's
    render_id drifted (re-rendered). Catches anchor identity changes the
    prompt_hash can't see.

    Malformed keys (no '@', from hand-edited or older metas) are excluded from
    the key-set comparison and the render_id check — the transitive walk covers
    those deps."""
    recorded = (meta or {}).get("sources")
    if not isinstance(recorded, dict):
        return False  # pre-provenance meta: transitive walk still covers deps
    # Only consider well-formed "@"-bearing keys; skip malformed ones as the
    # old per-key loop did (transitive walk below covers those deps).
    recorded_valid = {k: v for k, v in recorded.items() if "@" in k}
    current = recorded_sources(manifest, root, character, ref, direction)
    if set(recorded_valid.keys()) != set(current.keys()):
        return True
    for key, rid in recorded_valid.items():
        dep_ref, ddir = key.rsplit("@", 1)
        if read_render_id(manifest, root, character, dep_ref, ddir) != rid:
            return True
    return False


def stale_locally(manifest: Manifest, root: str, character: str, ref: str, direction: str) -> bool:
    """True if a COMPLETE node is stale for its OWN reasons — its merged prompt hash
    drifted, or a recorded source's `render_id` changed — so re-rendering THIS node
    will clear the drift. A node that is `outdated` only because an ancestor is
    outdated returns False here (re-rendering it wouldn't help). The batch
    auto-selectors use this to avoid wedging on a cell whose staleness they can't
    clear (e.g. every descendant of a stale root pose the selector won't regenerate)."""
    if not node_complete(manifest, root, character, ref, direction):
        return False
    kind = node_kind(manifest, ref)
    meta = read_node_meta(manifest, root, character, ref, direction)
    if (meta or {}).get("prompt_hash") != compute_prompt_hash(
        manifest, root, character, kind, ref, direction
    ):
        return True
    if _sources_drifted(manifest, root, character, ref, direction, meta):
        return True
    return False


def _outdated(manifest: Manifest, root: str, character: str, ref: str, direction: str) -> bool:
    kind = node_kind(manifest, ref)
    if not node_complete(manifest, root, character, ref, direction):
        return False
    meta = read_node_meta(manifest, root, character, ref, direction)
    current = compute_prompt_hash(manifest, root, character, kind, ref, direction)
    if (meta or {}).get("prompt_hash") != current:
        return True
    # Provenance: if this node recorded the render_id of each source it consumed,
    # or if the dep key-set changed (anchor ref swapped / end_at added/removed),
    # this node is stale. Absent on pre-provenance metas, in which case we fall
    # back to the transitive-hash walk below.
    if _sources_drifted(manifest, root, character, ref, direction, meta):
        return True
    if kind == "pose":
        frm = manifest["poses"][ref].get("from")
        if not frm:
            return False
        return outdated(manifest, root, character, frm["ref"], resolved_dir(frm, direction))
    for _slot, dep in animation_deps(manifest, ref):
        if outdated(manifest, root, character, dep["ref"], resolved_dir(dep, direction)):
            return True
    return False


# --- resolve + status ------------------------------------------------------- #

def resolve_pose(manifest: Manifest, root: str, character: str, pose_id: str, direction: str) -> dict:
    pose = manifest["poses"][pose_id]
    frm = pose.get("from")
    if frm:
        src_dir = resolved_dir(frm, direction)
        src_complete = node_complete(manifest, root, character, frm["ref"], src_dir)
        blocked_by = [] if src_complete else [{"from": frm, "dir": src_dir}]
        stale = src_complete and outdated(manifest, root, character, frm["ref"], src_dir)
        source_image = (
            pose_source_image(manifest, root, character, pose_id, direction)
            if src_complete else None
        )
    else:
        # Root pose (e.g. base): no upstream node. The creator node supplies the
        # reference image; there is nothing to block on or go stale against here.
        src_complete, blocked_by, stale, source_image = True, [], False, None
    positive, negative = merged_prompts(manifest, root, character, "pose", pose_id, direction)
    return {
        "selectable": (direction in pose["directions"]) and src_complete,
        "blocked_by": blocked_by,
        "stale": stale,
        "source_image": source_image,
        "positive": positive,
        "negative": negative,
        "output_dir": _pose_basedir(root, character, pose_id),
        "meta": {
            "kind": "pose", "pose": pose_id, "direction": direction, "from": frm,
            "image": f"{direction}.png", "manifest_version": manifest["version"],
            "prompt_hash": hash_prompts(positive, negative),
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
    start_image = start_anchor(manifest, root, character, anim_id, direction)
    end_image = end_anchor(manifest, root, character, anim_id, direction)
    # A clip loops when it begins and ends on the exact same frame — i.e. FFLF
    # whose start and end anchors resolve to the same image. There is no manifest
    # `loop` flag; looping is purely a consequence of the start/end anchors. The
    # writer drops the duplicated final frame so such a clip plays seamlessly.
    is_loop = start_image is not None and start_image == end_image
    return {
        "selectable": (direction in anim["directions"]) and not blocked_by,
        "blocked_by": blocked_by,
        "stale": stale,
        "start_image": start_image,
        "end_image": end_image,
        "positive": positive,
        "negative": negative,
        "output_dir": _anim_dir(root, character, anim_id, direction),
        "meta": {
            "kind": "animation", "animation": anim_id, "direction": direction,
            "fps": anim.get("fps", defaults.get("fps")),
            "length": anim.get("length", defaults.get("length")),
            "width": anim.get("width", defaults.get("width")),
            "height": anim.get("height", defaults.get("height")),
            "shift": anim.get("shift", defaults.get("shift")),
            "loop": is_loop, "manifest_version": manifest["version"],
            "prompt_hash": hash_prompts(positive, negative),
            "sources": recorded_sources(manifest, root, character, anim_id, direction),
        },
    }


def animation_fps(manifest: Manifest, anim_id: str) -> int:
    """The animation's resolved fps (its own, else `defaults.fps`), at least 1.

    A pure manifest lookup — no rendered-tree resolution — so it is cheap enough
    for the playback IS_CHANGED hot path, which runs on every graph evaluation."""
    anim = manifest["animations"][anim_id]
    fps = anim.get("fps", manifest.get("defaults", {}).get("fps"))
    return int(fps or 0) or 1


def playback_segments(
    manifest: Manifest, root: str, character: str, anim_id: str, direction: str,
    *, loops: int, fps: int,
) -> list[dict]:
    """An ordered playback plan for an animation, chaining its start_from/end_at
    deps one level deep. Segments, in play order:

      {"kind": "anim", "dir": <frame_dir>, "repeat": n, "drop_first": b, "drop_last": b}
      {"kind": "hold", "image": <png path>,  "count": n}

    An *animation* anchor contributes its own frames (played once) and drops the
    action's adjacent boundary frame — FFLF cross-wiring makes the action's first
    frame a copy of start_from's last and its last a copy of end_at's first. A
    *pose* anchor is held for `fps` frames (~1s) and nothing is dropped.
    Deps whose frames aren't rendered yet are skipped (and no boundary drop on
    that side). The action repeats `loops` times when it returns to its start
    state (start_from and end_at resolve to the same ref+direction)."""
    anim = manifest["animations"][anim_id]
    start = effective_start_dep(manifest, anim_id)
    end = anim.get("end_at")

    def dep_segment(dep: Optional[dict]) -> tuple[Optional[dict], bool]:
        """(segment, whether the action's boundary frame should be dropped)."""
        if not dep:
            return None, False
        ddir = resolved_dir(dep, direction)
        ref = dep["ref"]
        if not node_complete(manifest, root, character, ref, ddir):
            return None, False
        if node_kind(manifest, ref) == "animation":
            return {
                "kind": "anim", "dir": animation_frame_dir(root, character, ref, ddir),
                "repeat": 1, "drop_first": False, "drop_last": False,
            }, True
        return {"kind": "hold", "image": _single_image(manifest, root, character, ref, ddir),
                "count": max(int(fps), 1)}, False

    pre_seg, drop_first = dep_segment(start)
    post_seg, drop_last = dep_segment(end)

    start_img = start_anchor(manifest, root, character, anim_id, direction)
    end_img = end_anchor(manifest, root, character, anim_id, direction)
    loopable = start_img is not None and start_img == end_img
    action = {
        "kind": "anim", "dir": animation_frame_dir(root, character, anim_id, direction),
        "repeat": max(int(loops), 1) if loopable else 1,
        "drop_first": drop_first, "drop_last": drop_last,
    }
    return [s for s in (pre_seg, action, post_seg) if s]


def status_from_resolved(
    manifest: Manifest, root: str, character: str, ref: str, direction: str, resolved: dict
) -> str:
    """The UI status for a node given its already-resolved dict (from resolve_pose
    / resolve_animation). Callers that have just resolved a cell use this to avoid
    a second full resolve; `status` wraps it for callers that haven't."""
    kind = node_kind(manifest, ref)
    own_complete = (
        pose_complete(root, character, ref, direction) if kind == "pose"
        else animation_complete(root, character, ref, direction)
    )
    if resolved["blocked_by"]:
        return "blocked"
    if own_complete:
        return "stale" if outdated(manifest, root, character, ref, direction) else "generated"
    return "stale" if bool(resolved["stale"]) else "ready"


def status(manifest: Manifest, root: str, character: str, ref: str, direction: str) -> str:
    kind = node_kind(manifest, ref)
    resolved = (
        resolve_pose(manifest, root, character, ref, direction) if kind == "pose"
        else resolve_animation(manifest, root, character, ref, direction)
    )
    return status_from_resolved(manifest, root, character, ref, direction, resolved)

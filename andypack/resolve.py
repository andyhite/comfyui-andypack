"""Pure FFLF dependency resolver.

This module is intentionally free of any ComfyUI / torch imports so it can be
unit-tested in isolation (see CLAUDE.md and spec §5). It decides *what's
selectable*, *what prompt*, and *which two anchor frames* an (animation,
direction) pair should feed the sampler — it does not sample or load images.

Filesystem contract (spec §2): a rendered animation directory counts as
**complete** iff `.complete` exists AND `meta.json` parses AND at least
`frames.count` frame files are present. The `.complete` sentinel is written
last, so a half-rendered directory never reads as a satisfied dependency.

FFLF cross-wiring (the easy-to-invert rule, spec §1):
  - ``start_from`` consumes the dependency's **last_frame** (start where it ended)
  - ``end_at``     consumes the dependency's **start_frame** (end where it began)
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Any, Optional

Manifest = dict[str, Any]
Animation = dict[str, Any]
Dep = dict[str, Any]


# --------------------------------------------------------------------------- #
# Text normalization, negatives, hashing (spec §2, §4)
# --------------------------------------------------------------------------- #

_WS = re.compile(r"\s+")

# Unicode UNIT SEPARATOR — the delimiter the spec hashes positive/negative with.
_SEP = "␟"


def _normalize(text: str) -> str:
    """Strip ends and collapse internal whitespace runs to a single space."""
    return _WS.sub(" ", text.strip())


def compose_negative(manifest: Manifest, anim: Animation, direction: str) -> str:
    """Compose the negative prompt for ``(anim, direction)`` (spec §4).

    Order: global + facial(frontal|default) + animation.negative (if present).
    Joined with ", ", deduped case-insensitively, first occurrence preserved.
    """
    negatives = manifest["negatives"]
    facial = negatives["facial"]

    parts = [negatives["global"]]
    if direction in facial["frontal_directions"]:
        parts.append(facial["frontal"])
    else:
        parts.append(facial["default"])
    anim_negative = anim.get("negative")
    if anim_negative:
        parts.append(anim_negative)

    seen: set[str] = set()
    terms: list[str] = []
    for part in parts:
        for raw in part.split(","):
            term = raw.strip()
            if not term:
                continue
            key = term.lower()
            if key in seen:
                continue
            seen.add(key)
            terms.append(term)
    return ", ".join(terms)


def compute_prompt_hash(manifest: Manifest, anim: Animation, direction: str) -> str:
    """``sha1:`` hash of the normalized positive + composed negative (spec §2)."""
    composed = compose_negative(manifest, anim, direction)
    raw = _normalize(anim["prompt"]) + _SEP + _normalize(composed)
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()
    return f"sha1:{digest}"


# --------------------------------------------------------------------------- #
# Direction resolution + filesystem completeness (spec §5)
# --------------------------------------------------------------------------- #

def resolved_dir(dep: Dep, selected_dir: str) -> str:
    """A dependency's target direction: "same"/omitted follows the selection."""
    d = dep.get("direction", "same")
    return selected_dir if d in (None, "same") else d


def _base_png(root: str, character: str, direction: str) -> str:
    return os.path.join(root, character, "_base", f"{direction}.png")


def _anim_dir(root: str, character: str, anim: str, direction: str) -> str:
    return os.path.join(root, character, anim, direction)


def _read_meta(root: str, character: str, anim: str, direction: str) -> Optional[dict]:
    """Read a rendered dir's meta.json, tolerating missing/corrupt -> None."""
    path = os.path.join(_anim_dir(root, character, anim, direction), "meta.json")
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _count_frames(base: str) -> int:
    try:
        names = os.listdir(base)
    except OSError:
        return 0
    return sum(1 for n in names if n.startswith("frame_") and n.endswith(".png"))


def animation_complete(root: str, character: str, anim: str, direction: str) -> bool:
    """Spec §2/§5: `.complete` sentinel + parseable meta + enough frame files."""
    base = _anim_dir(root, character, anim, direction)
    if not os.path.exists(os.path.join(base, ".complete")):
        return False
    meta = _read_meta(root, character, anim, direction)
    if not meta:
        return False
    try:
        need = int(meta["frames"]["count"])
    except (KeyError, TypeError, ValueError):
        return False
    return _count_frames(base) >= need


def dep_complete(root: str, character: str, dep: Dep, selected_dir: str) -> bool:
    tdir = resolved_dir(dep, selected_dir)
    if dep["ref"] == "base_pose":
        return os.path.exists(_base_png(root, character, tdir))
    return animation_complete(root, character, dep["ref"], tdir)


def is_stale(manifest: Manifest, root: str, character: str, dep: Dep, selected_dir: str) -> bool:
    """A complete non-base dependency is stale if its rendered prompt_hash
    differs from the manifest's current hash (spec §1). base_pose is never stale.
    """
    if dep["ref"] == "base_pose":
        return False
    tdir = resolved_dir(dep, selected_dir)
    meta = _read_meta(root, character, dep["ref"], tdir)
    if not meta:
        return False
    rendered = meta.get("prompt_hash")
    current = compute_prompt_hash(manifest, manifest["animations"][dep["ref"]], tdir)
    return rendered != current


# --------------------------------------------------------------------------- #
# Anchor selection — the FFLF cross-wiring (spec §1, §5)
# --------------------------------------------------------------------------- #

def pick_start_anchor(root: str, character: str, anim: Animation, direction: str) -> Optional[str]:
    """start_from -> dependency's LAST frame (or the base png), else None."""
    dep = anim.get("start_from")
    if not dep:
        return None
    tdir = resolved_dir(dep, direction)
    if dep["ref"] == "base_pose":
        return _base_png(root, character, tdir)
    meta = _read_meta(root, character, dep["ref"], tdir)
    if not meta or "last_frame" not in meta:
        return None
    return os.path.join(_anim_dir(root, character, dep["ref"], tdir), meta["last_frame"])


def pick_end_anchor(root: str, character: str, anim: Animation, direction: str) -> Optional[str]:
    """end_at -> dependency's FIRST frame, else None.

    A ``base_pose`` end anchor is undefined (a static pose has no "start frame")
    and is not used by any animation in the manifest; resolve to None.
    """
    dep = anim.get("end_at")
    if not dep:
        return None
    tdir = resolved_dir(dep, direction)
    if dep["ref"] == "base_pose":
        return None
    meta = _read_meta(root, character, dep["ref"], tdir)
    if not meta or "start_frame" not in meta:
        return None
    return os.path.join(_anim_dir(root, character, dep["ref"], tdir), meta["start_frame"])


# --------------------------------------------------------------------------- #
# resolve() + status() (spec §5, §7)
# --------------------------------------------------------------------------- #

def resolve(manifest: Manifest, root: str, character: str, anim_id: str, direction: str) -> dict:
    """Resolve an (animation, direction) pair against the rendered tree."""
    anim = manifest["animations"][anim_id]
    defaults = manifest.get("defaults", {})

    blocked_by: list[dict] = []
    stale: list[str] = []
    for slot in ("start_from", "end_at"):
        dep = anim.get(slot)
        if not dep:
            continue
        if not dep_complete(root, character, dep, direction):
            blocked_by.append({slot: dep, "dir": resolved_dir(dep, direction)})
            continue
        if is_stale(manifest, root, character, dep, direction):
            stale.append(slot)

    selectable = (direction in anim["directions"]) and not blocked_by

    return {
        "selectable": selectable,
        "blocked_by": blocked_by,
        "stale": stale,
        "start_image": pick_start_anchor(root, character, anim, direction),
        "end_image": pick_end_anchor(root, character, anim, direction),
        "positive": anim["prompt"],
        "negative": compose_negative(manifest, anim, direction),
        "output_dir": _anim_dir(root, character, anim_id, direction),
        "meta": {
            "fps": anim.get("fps", defaults.get("fps")),
            "length": anim.get("length", defaults.get("length")),
            "loop": anim.get("loop", False),
            "prompt_hash": compute_prompt_hash(manifest, anim, direction),
        },
    }


def status(manifest: Manifest, root: str, character: str, anim_id: str, direction: str) -> str:
    """UI status for an (animation, direction) pair (spec §7).

    blocked: missing a dependency.
    stale:   deps satisfied but a dep's prompt_hash is outdated.
    generated: this pair's own output is complete.
    ready:   deps satisfied, not yet generated.
    """
    r = resolve(manifest, root, character, anim_id, direction)
    if r["blocked_by"]:
        return "blocked"
    if r["stale"]:
        return "stale"
    if animation_complete(root, character, anim_id, direction):
        return "generated"
    return "ready"

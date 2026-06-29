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

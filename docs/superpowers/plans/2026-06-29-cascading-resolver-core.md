# Cascading Resolver Core — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the pure-Python core of the Animation Coordinator — `manifest.py` (load/validate/cycle-detect) and `resolve.py` (cascading prompts, completeness, FFLF anchors, transitive staleness, `resolve_pose`/`resolve_animation`/`status`) — fully unit-tested.

**Architecture:** One dependency graph of three node kinds — concept seed → poses → animations. `manifest.py` validates the manifest dict and classifies refs. `resolve.py` walks the rendered tree on disk plus the in-memory manifest to decide selectability, compose cascading prompts, pick FFLF anchor images, and compute transitive staleness. No ComfyUI/torch — these modules are importable and testable standalone.

**Tech Stack:** Python ≥3.10, standard library only (`json`, `os`, `re`, `hashlib`, `warnings`), `pytest`, `ruff`, `mypy`.

**Source of truth:** `docs/superpowers/specs/2026-06-29-cascading-pose-resolver-design.md`. Read it before starting.

## Global Constraints

- `manifest.py` and `resolve.py` MUST NOT import ComfyUI, `torch`, `numpy`, or any non-stdlib package. Keep them pure and unit-testable.
- FFLF rule, never inverted: `start_from` consumes a dependency's **last** frame; `end_at` consumes its **first** frame. Single-image deps (concept/pose) resolve the **same** png for either slot.
- Completion sentinel is the **atomically-written-last** `meta.json`/sidecar — there is no `.complete` file. A node with no parseable meta/sidecar is incomplete.
- Cascade order, general → specific: `identity → globals.{kind} → entity → entity.directions[dir]`. The merge rule is identical for positives and negatives.
- Prompt hash: `"sha1:" + sha1(normalize(merged_positive) + "␟" + normalize(merged_negative))`, where `normalize` strips ends and collapses internal whitespace runs to a single space.
- Staleness is **transitive-on-hash** and never blocks — stale stays selectable.
- Every task ends green: `pytest -q` passes and `ruff check . && mypy andypack` is clean.
- Per-direction prompts: a `directions` value is a map keyed by direction; each value is an optional `{prompt?, negative?}` layer (often `{}`). The map's keys are the selectable directions.

---

## File Structure

- `andypack/__init__.py` — package marker (exists).
- `andypack/manifest.py` — `ManifestError`, `load_manifest`, `validate_manifest`, `node_kind`. Pure.
- `andypack/resolve.py` — cascade/merge/hash, path helpers, completeness, anchors, `outdated`, `resolve_pose`, `resolve_animation`, `status`. Pure.
- `tests/fixtures/manifest.json` — focused manifest: concept → base → fighting_stance → idle → {entry, exit, punch}.
- `tests/conftest.py` — `manifest` fixture + `Tree` builder that renders poses/animations into `tmp_path` with correct (or deliberately stale) hashes.
- `tests/test_manifest.py`, `tests/test_merge.py`, `tests/test_completeness.py`, `tests/test_anchors.py`, `tests/test_staleness.py`, `tests/test_resolve.py` — unit + acceptance tests.
- `examples/animations.json` — production manifest, ported to the new schema (final task).

**Fixture strategy:** rendered trees are built programmatically in `tmp_path` by the `Tree` builder (it computes correct hashes via `resolve.compute_prompt_hash`, so "fresh" fixtures stay fresh and "stale" ones use a sentinel hash). The committed static fixture is only `tests/fixtures/manifest.json`.

---

## Task 1: Scaffolding + focused test manifest

Removes the obsolete step-2 code (built to the old base_pose/facial/`.complete` model) and lands the test manifest every later task loads.

**Files:**
- Delete: `andypack/resolve.py` (old), `tests/test_resolve.py` (old), `tests/fixtures/empty_root/`, `tests/fixtures/idle_root/`, `tests/fixtures/partial_root/`, `tests/fixtures/bare_root/`
- Create: `tests/fixtures/manifest.json`

- [ ] **Step 1: Remove obsolete step-2 artifacts**

```bash
git rm -q -f andypack/resolve.py tests/test_resolve.py 2>/dev/null || rm -f andypack/resolve.py tests/test_resolve.py
rm -rf tests/fixtures/empty_root tests/fixtures/idle_root tests/fixtures/partial_root tests/fixtures/bare_root
```

(`andypack/__init__.py` and `pyproject.toml` stay.)

- [ ] **Step 2: Write the focused test manifest**

Create `tests/fixtures/manifest.json`:

```json
{
  "version": 1,
  "directions": ["E", "SE", "S", "SW", "W", "NW", "N", "NE"],
  "mirror_map": { "W": "E", "SW": "SE", "NW": "NE" },
  "defaults": { "fps": 16, "length": 33 },
  "globals": {
    "animation": { "negative": "blurry, low quality, watermark" },
    "pose": { "negative": "blurry, low quality" }
  },
  "poses": {
    "base": {
      "from": { "ref": "concept" },
      "prompt": "neutral standing pose",
      "directions": {
        "E": { "prompt": "facing right in profile" },
        "SE": { "prompt": "facing down-right at three-quarter" },
        "S": { "prompt": "facing toward the viewer" }
      }
    },
    "fighting_stance": {
      "from": { "ref": "base", "direction": "same" },
      "prompt": "ready fighting stance, fists up",
      "directions": { "E": { "prompt": "guard facing right" } }
    }
  },
  "animations": {
    "fighting_stance_idle": {
      "category": "combat", "loop": true, "length": 33,
      "start_from": { "ref": "fighting_stance" },
      "prompt": "bob gently in a ready guard",
      "directions": { "E": {} }
    },
    "fighting_stance_entry": {
      "category": "combat", "loop": false, "length": 25,
      "start_from": { "ref": "base" },
      "end_at": { "ref": "fighting_stance_idle" },
      "prompt": "rise from standing into a guard",
      "directions": { "E": {} }
    },
    "fighting_stance_exit": {
      "category": "combat", "loop": false, "length": 25,
      "start_from": { "ref": "fighting_stance_idle" },
      "end_at": { "ref": "base" },
      "prompt": "drop from guard back to standing",
      "directions": { "E": {} }
    },
    "punch": {
      "category": "combat", "loop": false, "length": 21,
      "start_from": { "ref": "fighting_stance_idle" },
      "end_at": { "ref": "fighting_stance_idle" },
      "prompt": "throw a straight jab to the right",
      "negative": "both arms extended, extra arm",
      "directions": { "E": {} }
    }
  }
}
```

- [ ] **Step 3: Verify the manifest parses**

Run: `python3 -c "import json; json.load(open('tests/fixtures/manifest.json')); print('ok')"`
Expected: `ok`

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "chore: reset to cascading model; add focused test manifest"
```

---

## Task 2: `manifest.py` — load, validate, classify, cycle-detect

**Files:**
- Create: `andypack/manifest.py`
- Test: `tests/test_manifest.py`

**Interfaces:**
- Produces:
  - `class ManifestError(Exception)`
  - `node_kind(manifest: dict, ref: str) -> str` → `"concept" | "pose" | "animation"`; raises `ManifestError` on unknown ref.
  - `validate_manifest(manifest: dict) -> None` → raises `ManifestError` on structural problems or dependency cycles; emits `warnings.warn` for non-`4n+1` animation lengths.
  - `load_manifest(path: str) -> dict` → parses JSON, validates, returns the dict.

- [ ] **Step 1: Write failing tests**

Create `tests/test_manifest.py`:

```python
import json
import warnings
from pathlib import Path

import pytest

from andypack.manifest import ManifestError, load_manifest, node_kind, validate_manifest

FIX = Path(__file__).parent / "fixtures" / "manifest.json"


def base_manifest():
    return json.loads(FIX.read_text())


def test_load_valid_manifest_returns_dict():
    m = load_manifest(str(FIX))
    assert m["version"] == 1
    assert "fighting_stance_idle" in m["animations"]


def test_node_kind_classifies_each_ref():
    m = base_manifest()
    assert node_kind(m, "concept") == "concept"
    assert node_kind(m, "base") == "pose"
    assert node_kind(m, "punch") == "animation"


def test_node_kind_unknown_ref_raises():
    with pytest.raises(ManifestError):
        node_kind(base_manifest(), "does_not_exist")


def test_validate_rejects_bad_animation_ref():
    m = base_manifest()
    m["animations"]["punch"]["start_from"] = {"ref": "nope"}
    with pytest.raises(ManifestError):
        validate_manifest(m)


def test_validate_rejects_pose_from_animation():
    m = base_manifest()
    m["poses"]["base"]["from"] = {"ref": "punch"}  # a pose may only edit concept/pose
    with pytest.raises(ManifestError):
        validate_manifest(m)


def test_validate_detects_cycle():
    m = base_manifest()
    # base <- fighting_stance and fighting_stance <- base  => cycle
    m["poses"]["base"]["from"] = {"ref": "fighting_stance", "direction": "E"}
    with pytest.raises(ManifestError):
        validate_manifest(m)


def test_validate_warns_on_non_4n_plus_1_length():
    m = base_manifest()
    m["animations"]["punch"]["length"] = 20  # 20 is not 4n+1
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        validate_manifest(m)
    assert any("4n+1" in str(w.message) or "length" in str(w.message) for w in caught)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_manifest.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'andypack.manifest'`

- [ ] **Step 3: Implement `andypack/manifest.py`**

```python
"""Manifest loading, validation, and ref classification (pure stdlib)."""

from __future__ import annotations

import json
import warnings
from typing import Any

Manifest = dict[str, Any]


class ManifestError(Exception):
    """Raised when a manifest is structurally invalid or has a dependency cycle."""


def node_kind(manifest: Manifest, ref: str) -> str:
    """Classify a ref as 'concept', 'pose', or 'animation'."""
    if ref == "concept":
        return "concept"
    if ref in manifest.get("poses", {}):
        return "pose"
    if ref in manifest.get("animations", {}):
        return "animation"
    raise ManifestError(f"unknown ref: {ref!r}")


def _validate_refs(manifest: Manifest) -> None:
    for pid, pose in manifest.get("poses", {}).items():
        frm = pose.get("from")
        if not isinstance(frm, dict) or "ref" not in frm:
            raise ManifestError(f"pose {pid!r} missing 'from.ref'")
        if node_kind(manifest, frm["ref"]) == "animation":
            raise ManifestError(f"pose {pid!r} 'from' must reference concept or a pose")
        if not isinstance(pose.get("directions"), dict):
            raise ManifestError(f"pose {pid!r} missing 'directions' map")
    for aid, anim in manifest.get("animations", {}).items():
        for slot in ("start_from", "end_at"):
            dep = anim.get(slot)
            if dep is not None:
                if not isinstance(dep, dict) or "ref" not in dep:
                    raise ManifestError(f"animation {aid!r} {slot} missing 'ref'")
                node_kind(manifest, dep["ref"])  # raises on unknown
        if not isinstance(anim.get("directions"), dict):
            raise ManifestError(f"animation {aid!r} missing 'directions' map")


def _detect_cycles(manifest: Manifest) -> None:
    edges: dict[str, list[str]] = {}

    def add(node: str, ref: str | None) -> None:
        edges.setdefault(node, [])
        if ref and ref != "concept":
            edges[node].append(ref)

    for pid, pose in manifest.get("poses", {}).items():
        add(pid, pose.get("from", {}).get("ref"))
    for aid, anim in manifest.get("animations", {}).items():
        edges.setdefault(aid, [])
        for slot in ("start_from", "end_at"):
            dep = anim.get(slot)
            if dep:
                add(aid, dep.get("ref"))

    WHITE, GRAY, BLACK = 0, 1, 2
    color = {n: WHITE for n in edges}

    def dfs(node: str) -> None:
        color[node] = GRAY
        for nxt in edges.get(node, []):
            if color.get(nxt, BLACK) == GRAY:
                raise ManifestError(f"dependency cycle: {node} -> {nxt}")
            if color.get(nxt, WHITE) == WHITE:
                dfs(nxt)
        color[node] = BLACK

    for node in list(edges):
        if color[node] == WHITE:
            dfs(node)


def _warn_lengths(manifest: Manifest) -> None:
    default_len = manifest.get("defaults", {}).get("length")
    for aid, anim in manifest.get("animations", {}).items():
        length = anim.get("length", default_len)
        if isinstance(length, int) and (length - 1) % 4 != 0:
            warnings.warn(f"animation {aid!r} length {length} is not 4n+1 (Wan-unfriendly)")


def validate_manifest(manifest: Manifest) -> None:
    """Structural validation + cycle detection. Raises ManifestError on failure."""
    if not isinstance(manifest.get("version"), int):
        raise ManifestError("manifest missing integer 'version'")
    for key in ("poses", "animations"):
        if not isinstance(manifest.get(key), dict):
            raise ManifestError(f"manifest missing '{key}' object")
    _validate_refs(manifest)
    _detect_cycles(manifest)
    _warn_lengths(manifest)


def load_manifest(path: str) -> Manifest:
    """Load, validate, and return the manifest dict."""
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ManifestError("manifest root must be an object")
    validate_manifest(data)
    return data
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_manifest.py -q`
Expected: PASS (7 passed)

- [ ] **Step 5: Lint/type-check + commit**

```bash
ruff check . && mypy andypack
git add andypack/manifest.py tests/test_manifest.py
git commit -m "feat: manifest load/validate/classify with cycle detection"
```

---

## Task 3: `resolve.py` — cascade merge, normalize, hashing

**Files:**
- Create: `andypack/resolve.py`
- Test: `tests/test_merge.py`

**Interfaces:**
- Consumes: nothing (identity is read from disk; `_concept.json` may be absent → `{}`).
- Produces:
  - `merge_layers(*parts: str | None) -> str`
  - `read_identity(root: str, character: str) -> dict`
  - `merged_prompts(manifest, root, character, kind: str, entity_id: str, direction: str) -> tuple[str, str]` — `kind` is `"pose"` or `"animation"`.
  - `compute_prompt_hash(manifest, root, character, kind, entity_id, direction) -> str`

- [ ] **Step 1: Write failing tests**

Create `tests/test_merge.py`:

```python
import hashlib
import json
import re
from pathlib import Path

from andypack.resolve import compute_prompt_hash, merge_layers, merged_prompts, read_identity

FIX = Path(__file__).parent / "fixtures" / "manifest.json"


def base_manifest():
    return json.loads(FIX.read_text())


def _norm(s):
    return re.sub(r"\s+", " ", s.strip())


def test_merge_layers_joins_non_empty_in_order():
    assert merge_layers("a", None, "b", "") == "a, b"


def test_merge_layers_dedupes_case_insensitively_preserving_first():
    assert merge_layers("Blurry, foo", "blurry, bar") == "Blurry, foo, bar"


def test_merge_layers_is_lossless_for_prose_commas():
    assert merge_layers("walks forward, steady pace") == "walks forward, steady pace"


def test_read_identity_absent_returns_empty(tmp_path):
    assert read_identity(str(tmp_path), "Cortex") == {}


def test_read_identity_reads_concept_sidecar(tmp_path):
    char_dir = tmp_path / "Cortex"
    char_dir.mkdir()
    (char_dir / "_concept.json").write_text(json.dumps({"prompt": "a mouthless hero"}))
    assert read_identity(str(tmp_path), "Cortex") == {"prompt": "a mouthless hero"}


def test_merged_prompts_cascades_identity_global_entity_direction(tmp_path):
    char_dir = tmp_path / "Cortex"
    char_dir.mkdir()
    (char_dir / "_concept.json").write_text(json.dumps({"prompt": "a mouthless hero"}))
    m = base_manifest()
    pos, neg = merged_prompts(m, str(tmp_path), "Cortex", "pose", "base", "E")
    # identity -> (no globals.pose positive) -> pose.prompt -> base.directions.E.prompt
    assert pos == "a mouthless hero, neutral standing pose, facing right in profile"
    assert neg == "blurry, low quality"  # globals.pose.negative only


def test_compute_prompt_hash_matches_formula(tmp_path):
    m = base_manifest()
    pos, neg = merged_prompts(m, str(tmp_path), "Cortex", "animation", "punch", "E")
    raw = _norm(pos) + "␟" + _norm(neg)
    expected = "sha1:" + hashlib.sha1(raw.encode("utf-8")).hexdigest()
    assert compute_prompt_hash(m, str(tmp_path), "Cortex", "animation", "punch", "E") == expected
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_merge.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'andypack.resolve'`

- [ ] **Step 3: Implement the cascade portion of `andypack/resolve.py`**

Create `andypack/resolve.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_merge.py -q`
Expected: PASS (7 passed)

- [ ] **Step 5: Lint/type-check + commit**

```bash
ruff check . && mypy andypack
git add andypack/resolve.py tests/test_merge.py
git commit -m "feat: cascading prompt merge + prompt hashing"
```

---

## Task 4: `resolve.py` — paths, completeness, rendered-hash; `Tree` builder

**Files:**
- Modify: `andypack/resolve.py` (append)
- Create: `tests/conftest.py`
- Test: `tests/test_completeness.py`

**Interfaces:**
- Consumes: `compute_prompt_hash` (Task 3), `node_kind` (Task 2).
- Produces:
  - `resolved_dir(dep: dict, selected_dir: str) -> str`
  - `concept_complete(root, character) -> bool`
  - `pose_complete(root, character, pose_id, direction) -> bool`
  - `animation_complete(root, character, anim_id, direction) -> bool`
  - `node_complete(manifest, root, character, ref, direction) -> bool`
  - `read_rendered_hash(manifest, root, character, ref, direction) -> Optional[str]`
  - Test helper `tests/conftest.py::Tree` with `.concept()`, `.identity(**layer)`, `.pose(id, dir, *, stale=False, sidecar=True)`, `.animation(id, dir, *, frames=3, stale=False, meta=True)`, and a `manifest` fixture.

- [ ] **Step 1: Write the `Tree` builder + `manifest` fixture**

Create `tests/conftest.py`:

```python
import json
import os
from pathlib import Path

import pytest

from andypack import resolve

FIX = Path(__file__).parent / "fixtures" / "manifest.json"


@pytest.fixture
def manifest():
    return json.loads(FIX.read_text())


def _write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh)


def _touch(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w").close()


class Tree:
    """Builds a rendered character tree under a root dir, with correct hashes."""

    def __init__(self, manifest, root, character="Cortex"):
        self.m = manifest
        self.root = root
        self.char = character

    def _cdir(self):
        return os.path.join(self.root, self.char)

    def concept(self):
        _touch(os.path.join(self._cdir(), "_concept.png"))
        return self

    def identity(self, **layer):
        _write_json(os.path.join(self._cdir(), "_concept.json"), layer)
        return self

    def pose(self, pose_id, direction, *, stale=False, sidecar=True):
        base = os.path.join(self._cdir(), f"_{pose_id}")
        _touch(os.path.join(base, f"{direction}.png"))
        if sidecar:
            h = "sha1:STALE" if stale else resolve.compute_prompt_hash(
                self.m, self.root, self.char, "pose", pose_id, direction
            )
            _write_json(os.path.join(base, f"{direction}.json"), {
                "kind": "pose", "pose": pose_id, "direction": direction,
                "from": self.m["poses"][pose_id]["from"], "image": f"{direction}.png",
                "prompt_hash": h, "manifest_version": self.m["version"],
            })
        return self

    def animation(self, anim_id, direction, *, frames=3, stale=False, meta=True):
        base = os.path.join(self._cdir(), anim_id, direction)
        for i in range(frames):
            _touch(os.path.join(base, f"frame_{i:05d}.png"))
        if meta:
            h = "sha1:STALE" if stale else resolve.compute_prompt_hash(
                self.m, self.root, self.char, "animation", anim_id, direction
            )
            _write_json(os.path.join(base, "meta.json"), {
                "kind": "animation", "animation": anim_id, "direction": direction,
                "fps": 16, "length": frames,
                "loop": self.m["animations"][anim_id].get("loop", False),
                "prompt_hash": h, "manifest_version": self.m["version"],
                "frames": {"dir": ".", "pattern": "frame_{:05d}.png", "count": frames},
                "start_frame": "frame_00000.png", "last_frame": f"frame_{frames - 1:05d}.png",
            })
        return self


@pytest.fixture
def tree(manifest, tmp_path):
    return Tree(manifest, str(tmp_path / "root"))
```

- [ ] **Step 2: Write failing completeness tests**

Create `tests/test_completeness.py`:

```python
from andypack.resolve import (
    animation_complete,
    concept_complete,
    node_complete,
    pose_complete,
    read_rendered_hash,
    resolved_dir,
)


def test_resolved_dir_same_vs_explicit():
    assert resolved_dir({"ref": "x"}, "E") == "E"
    assert resolved_dir({"ref": "x", "direction": "same"}, "SE") == "SE"
    assert resolved_dir({"ref": "x", "direction": "E"}, "SE") == "E"


def test_concept_complete(tree):
    assert concept_complete(tree.root, tree.char) is False
    tree.concept()
    assert concept_complete(tree.root, tree.char) is True


def test_pose_complete_requires_png_and_sidecar(tree):
    tree.concept().pose("base", "E", sidecar=False)
    assert pose_complete(tree.root, tree.char, "base", "E") is False  # png only
    tree.pose("base", "E")  # now with sidecar
    assert pose_complete(tree.root, tree.char, "base", "E") is True


def test_animation_complete_requires_meta_and_frames(tree):
    tree.concept().animation("fighting_stance_idle", "E", frames=3, meta=False)
    assert animation_complete(tree.root, tree.char, "fighting_stance_idle", "E") is False
    tree.animation("fighting_stance_idle", "E", frames=3)  # meta now present
    assert animation_complete(tree.root, tree.char, "fighting_stance_idle", "E") is True


def test_node_complete_dispatches_by_kind(manifest, tree):
    tree.concept().pose("base", "E").pose("fighting_stance", "E")
    assert node_complete(manifest, tree.root, tree.char, "concept", "E") is True
    assert node_complete(manifest, tree.root, tree.char, "base", "E") is True
    assert node_complete(manifest, tree.root, tree.char, "fighting_stance", "E") is True
    assert node_complete(manifest, tree.root, tree.char, "punch", "E") is False


def test_read_rendered_hash(manifest, tree):
    assert read_rendered_hash(manifest, tree.root, tree.char, "concept", "E") is None
    tree.concept().pose("base", "E")
    h = read_rendered_hash(manifest, tree.root, tree.char, "base", "E")
    assert h is not None and h.startswith("sha1:")
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_completeness.py -q`
Expected: FAIL — `ImportError: cannot import name 'concept_complete'`

- [ ] **Step 4: Append completeness code to `andypack/resolve.py`**

```python
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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_completeness.py -q`
Expected: PASS (6 passed)

- [ ] **Step 6: Lint/type-check + commit**

```bash
ruff check . && mypy andypack
git add andypack/resolve.py tests/conftest.py tests/test_completeness.py
git commit -m "feat: completeness checks + rendered-hash readers + Tree builder"
```

---

## Task 5: `resolve.py` — FFLF anchors

**Files:**
- Modify: `andypack/resolve.py` (append)
- Test: `tests/test_anchors.py`

**Interfaces:**
- Consumes: path helpers, `resolved_dir`, `node_kind`, `_read_json`.
- Produces:
  - `pose_source_image(manifest, root, character, pose_id, direction) -> Optional[str]`
  - `start_anchor(manifest, root, character, anim_id, direction) -> Optional[str]`
  - `end_anchor(manifest, root, character, anim_id, direction) -> Optional[str]`

- [ ] **Step 1: Write failing tests**

Create `tests/test_anchors.py`:

```python
import os

from andypack.resolve import end_anchor, pose_source_image, start_anchor


def test_pose_source_from_concept_is_concept_png(manifest, tree):
    tree.concept()
    src = pose_source_image(manifest, tree.root, tree.char, "base", "E")
    assert src.endswith(os.path.join("Cortex", "_concept.png"))


def test_pose_source_from_pose_is_that_pose_png(manifest, tree):
    tree.concept().pose("base", "E")
    src = pose_source_image(manifest, tree.root, tree.char, "fighting_stance", "E")
    assert src.endswith(os.path.join("_base", "E.png"))


def test_animation_anchor_on_pose_uses_pose_png_for_both_slots(manifest, tree):
    # fighting_stance_idle.start_from = fighting_stance (a pose, single image)
    tree.concept().pose("base", "E").pose("fighting_stance", "E")
    s = start_anchor(manifest, tree.root, tree.char, "fighting_stance_idle", "E")
    assert s.endswith(os.path.join("_fighting_stance", "E.png"))


def test_punch_anchors_cross_wire_fflf(manifest, tree):
    # start_from idle -> idle.last_frame; end_at idle -> idle.start_frame
    tree.concept().pose("base", "E").pose("fighting_stance", "E").animation(
        "fighting_stance_idle", "E", frames=3
    )
    s = start_anchor(manifest, tree.root, tree.char, "punch", "E")
    e = end_anchor(manifest, tree.root, tree.char, "punch", "E")
    assert s.endswith(os.path.join("fighting_stance_idle", "E", "frame_00002.png"))
    assert e.endswith(os.path.join("fighting_stance_idle", "E", "frame_00000.png"))


def test_entry_and_exit_anchors_mix_pose_and_animation(manifest, tree):
    tree.concept().pose("base", "E").pose("fighting_stance", "E").animation(
        "fighting_stance_idle", "E", frames=3
    )
    # entry: start_from base (pose png) ; end_at idle (start_frame)
    assert start_anchor(manifest, tree.root, tree.char, "fighting_stance_entry", "E").endswith(
        os.path.join("_base", "E.png")
    )
    assert end_anchor(manifest, tree.root, tree.char, "fighting_stance_entry", "E").endswith(
        os.path.join("fighting_stance_idle", "E", "frame_00000.png")
    )
    # exit: start_from idle (last_frame) ; end_at base (pose png)
    assert start_anchor(manifest, tree.root, tree.char, "fighting_stance_exit", "E").endswith(
        os.path.join("fighting_stance_idle", "E", "frame_00002.png")
    )
    assert end_anchor(manifest, tree.root, tree.char, "fighting_stance_exit", "E").endswith(
        os.path.join("_base", "E.png")
    )


def test_anchor_none_when_dep_absent(manifest, tree):
    # idle has no end_at
    assert end_anchor(manifest, tree.root, tree.char, "fighting_stance_idle", "E") is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_anchors.py -q`
Expected: FAIL — `ImportError: cannot import name 'start_anchor'`

- [ ] **Step 3: Append anchor code to `andypack/resolve.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_anchors.py -q`
Expected: PASS (6 passed)

- [ ] **Step 5: Lint/type-check + commit**

```bash
ruff check . && mypy andypack
git add andypack/resolve.py tests/test_anchors.py
git commit -m "feat: FFLF anchor resolution (pose/concept single image, animation first/last)"
```

---

## Task 6: `resolve.py` — transitive staleness (`outdated`)

**Files:**
- Modify: `andypack/resolve.py` (append)
- Test: `tests/test_staleness.py`

**Interfaces:**
- Consumes: `node_kind`, `node_complete`, `read_rendered_hash`, `compute_prompt_hash`, `resolved_dir`.
- Produces: `outdated(manifest, root, character, ref, direction) -> bool`

- [ ] **Step 1: Write failing tests**

Create `tests/test_staleness.py`:

```python
from andypack.resolve import outdated


def _full_stance_tree(tree):
    return (
        tree.concept()
        .pose("base", "E")
        .pose("fighting_stance", "E")
        .animation("fighting_stance_idle", "E", frames=3)
    )


def test_concept_never_outdated(manifest, tree):
    tree.concept()
    assert outdated(manifest, tree.root, tree.char, "concept", "E") is False


def test_incomplete_node_is_not_outdated(manifest, tree):
    # nothing rendered -> base is incomplete -> not "stale" (that's blocked territory)
    tree.concept()
    assert outdated(manifest, tree.root, tree.char, "base", "E") is False


def test_fresh_chain_is_not_outdated(manifest, tree):
    _full_stance_tree(tree)
    assert outdated(manifest, tree.root, tree.char, "fighting_stance_idle", "E") is False


def test_own_hash_drift_marks_outdated(manifest, tree):
    _full_stance_tree(tree)
    # re-render fighting_stance with a bogus stored hash
    tree.pose("fighting_stance", "E", stale=True)
    assert outdated(manifest, tree.root, tree.char, "fighting_stance", "E") is True


def test_staleness_is_transitive(manifest, tree):
    # base rendered with a stale hash; idle/punch are otherwise fresh
    tree.concept().pose("base", "E", stale=True).pose("fighting_stance", "E").animation(
        "fighting_stance_idle", "E", frames=3
    )
    # fighting_stance's own hash is fine, but its ancestor (base) is outdated
    assert outdated(manifest, tree.root, tree.char, "fighting_stance", "E") is True
    # ripples all the way to punch (start_from idle -> fighting_stance -> base)
    assert outdated(manifest, tree.root, tree.char, "punch", "E") is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_staleness.py -q`
Expected: FAIL — `ImportError: cannot import name 'outdated'`

- [ ] **Step 3: Append `outdated` to `andypack/resolve.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_staleness.py -q`
Expected: PASS (5 passed)

- [ ] **Step 5: Lint/type-check + commit**

```bash
ruff check . && mypy andypack
git add andypack/resolve.py tests/test_staleness.py
git commit -m "feat: transitive-on-hash staleness"
```

---

## Task 7: `resolve.py` — `resolve_pose`, `resolve_animation`, `status`

**Files:**
- Modify: `andypack/resolve.py` (append)
- Test: `tests/test_resolve.py`

**Interfaces:**
- Consumes: everything above.
- Produces:
  - `resolve_pose(manifest, root, character, pose_id, direction) -> dict`
  - `resolve_animation(manifest, root, character, anim_id, direction) -> dict`
  - `status(manifest, root, character, ref, direction) -> str`
- `resolve_pose` dict keys: `selectable`, `blocked_by` (list — empty or `[{"from": dep, "dir": str}]`), `stale` (bool), `source_image`, `positive`, `negative`, `output_dir`, `meta`.
- `resolve_animation` dict keys: `selectable`, `blocked_by` (list of `{slot: dep, "dir": str}`), `stale` (list of slot names), `start_image`, `end_image`, `positive`, `negative`, `output_dir`, `meta`.
- `status` returns `"blocked" | "stale" | "generated" | "ready"`.

- [ ] **Step 1: Write failing tests**

Create `tests/test_resolve.py`:

```python
import os

from andypack.resolve import resolve_animation, resolve_pose, status


def test_base_pose_ready_when_concept_present(manifest, tree):
    tree.concept()
    r = resolve_pose(manifest, tree.root, tree.char, "base", "E")
    assert r["selectable"] is True
    assert r["blocked_by"] == []
    assert r["source_image"].endswith(os.path.join("Cortex", "_concept.png"))
    assert status(manifest, tree.root, tree.char, "base", "E") == "ready"


def test_base_pose_blocked_when_concept_missing(manifest, tree):
    r = resolve_pose(manifest, tree.root, tree.char, "base", "E")
    assert r["selectable"] is False
    assert status(manifest, tree.root, tree.char, "base", "E") == "blocked"


def test_pose_generated_status(manifest, tree):
    tree.concept().pose("base", "E")
    assert status(manifest, tree.root, tree.char, "base", "E") == "generated"


def test_fighting_stance_unlocks_after_base(manifest, tree):
    tree.concept()
    assert status(manifest, tree.root, tree.char, "fighting_stance", "E") == "blocked"
    tree.pose("base", "E")
    assert status(manifest, tree.root, tree.char, "fighting_stance", "E") == "ready"


def test_punch_blocked_until_idle(manifest, tree):
    tree.concept().pose("base", "E").pose("fighting_stance", "E")
    r = resolve_animation(manifest, tree.root, tree.char, "punch", "E")
    assert r["selectable"] is False
    slots = {k for entry in r["blocked_by"] for k in entry if k != "dir"}
    assert slots == {"start_from", "end_at"}
    assert status(manifest, tree.root, tree.char, "punch", "E") == "blocked"


def test_punch_ready_with_anchors_after_idle(manifest, tree):
    tree.concept().pose("base", "E").pose("fighting_stance", "E").animation(
        "fighting_stance_idle", "E", frames=3
    )
    r = resolve_animation(manifest, tree.root, tree.char, "punch", "E")
    assert r["selectable"] is True
    assert r["start_image"].endswith(os.path.join("fighting_stance_idle", "E", "frame_00002.png"))
    assert r["end_image"].endswith(os.path.join("fighting_stance_idle", "E", "frame_00000.png"))
    assert status(manifest, tree.root, tree.char, "punch", "E") == "ready"
    assert r["meta"]["prompt_hash"].startswith("sha1:")


def test_direction_outside_map_not_selectable(manifest, tree):
    tree.concept().pose("base", "E").pose("fighting_stance", "E").animation(
        "fighting_stance_idle", "E", frames=3
    )
    r = resolve_animation(manifest, tree.root, tree.char, "punch", "S")
    assert r["selectable"] is False  # punch.directions only has E
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_resolve.py -q`
Expected: FAIL — `ImportError: cannot import name 'resolve_pose'`

- [ ] **Step 3: Append resolve/status to `andypack/resolve.py`**

```python
# --- resolve + status ------------------------------------------------------- #

def resolve_pose(manifest: Manifest, root: str, character: str, pose_id: str, direction: str) -> dict:
    pose = manifest["poses"][pose_id]
    frm = pose["from"]
    src_dir = resolved_dir(frm, direction)
    src_complete = node_complete(manifest, root, character, frm["ref"], src_dir)
    positive, negative = merged_prompts(manifest, root, character, "pose", pose_id, direction)
    return {
        "selectable": (direction in pose["directions"]) and src_complete,
        "blocked_by": [] if src_complete else [{"from": frm, "dir": src_dir}],
        "stale": src_complete and outdated(manifest, root, character, frm["ref"], src_dir),
        "source_image": pose_source_image(manifest, root, character, pose_id, direction)
        if src_complete else None,
        "positive": positive,
        "negative": negative,
        "output_dir": _pose_basedir(root, character, pose_id),
        "meta": {
            "kind": "pose", "pose": pose_id, "direction": direction, "from": frm,
            "image": f"{direction}.png", "manifest_version": manifest["version"],
            "prompt_hash": compute_prompt_hash(manifest, root, character, "pose", pose_id, direction),
        },
    }


def resolve_animation(manifest: Manifest, root: str, character: str, anim_id: str, direction: str) -> dict:
    anim = manifest["animations"][anim_id]
    defaults = manifest.get("defaults", {})
    blocked_by: list[dict] = []
    stale: list[str] = []
    for slot in ("start_from", "end_at"):
        dep = anim.get(slot)
        if not dep:
            continue
        ddir = resolved_dir(dep, direction)
        if not node_complete(manifest, root, character, dep["ref"], ddir):
            blocked_by.append({slot: dep, "dir": ddir})
            continue
        if outdated(manifest, root, character, dep["ref"], ddir):
            stale.append(slot)
    positive, negative = merged_prompts(manifest, root, character, "animation", anim_id, direction)
    return {
        "selectable": (direction in anim["directions"]) and not blocked_by,
        "blocked_by": blocked_by,
        "stale": stale,
        "start_image": start_anchor(manifest, root, character, anim_id, direction),
        "end_image": end_anchor(manifest, root, character, anim_id, direction),
        "positive": positive,
        "negative": negative,
        "output_dir": _anim_dir(root, character, anim_id, direction),
        "meta": {
            "kind": "animation", "animation": anim_id, "direction": direction,
            "fps": anim.get("fps", defaults.get("fps")),
            "length": anim.get("length", defaults.get("length")),
            "loop": anim.get("loop", False), "manifest_version": manifest["version"],
            "prompt_hash": compute_prompt_hash(manifest, root, character, "animation", anim_id, direction),
        },
    }


def status(manifest: Manifest, root: str, character: str, ref: str, direction: str) -> str:
    kind = node_kind(manifest, ref)
    if kind == "pose":
        r = resolve_pose(manifest, root, character, ref, direction)
        own_complete = pose_complete(root, character, ref, direction)
        dep_stale = bool(r["stale"])
    else:
        r = resolve_animation(manifest, root, character, ref, direction)
        own_complete = animation_complete(root, character, ref, direction)
        dep_stale = bool(r["stale"])
    if r["blocked_by"]:
        return "blocked"
    if own_complete:
        return "stale" if outdated(manifest, root, character, ref, direction) else "generated"
    return "stale" if dep_stale else "ready"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_resolve.py -q`
Expected: PASS (7 passed)

- [ ] **Step 5: Lint/type-check + commit**

```bash
ruff check . && mypy andypack
git add andypack/resolve.py tests/test_resolve.py
git commit -m "feat: resolve_pose, resolve_animation, status"
```

---

## Task 8: End-to-end acceptance walk + full gate

**Files:**
- Test: `tests/test_acceptance.py`

**Interfaces:**
- Consumes: `status` (Task 7), `outdated` (Task 6), the `Tree` builder.

- [ ] **Step 1: Write the failing acceptance test (spec §8 end-to-end)**

Create `tests/test_acceptance.py`:

```python
from andypack.resolve import status


def test_chain_unlocks_step_by_step(manifest, tree):
    root, char = tree.root, tree.char

    # Only concept present -> only base@E/SE/S selectable; nothing combat yet.
    tree.concept()
    assert status(manifest, root, char, "base", "E") == "ready"
    assert status(manifest, root, char, "fighting_stance", "E") == "blocked"
    assert status(manifest, root, char, "fighting_stance_idle", "E") == "blocked"

    # Generate base -> fighting_stance unlocks.
    tree.pose("base", "E")
    assert status(manifest, root, char, "fighting_stance", "E") == "ready"
    assert status(manifest, root, char, "fighting_stance_idle", "E") == "blocked"

    # Generate fighting_stance -> idle unlocks.
    tree.pose("fighting_stance", "E")
    assert status(manifest, root, char, "fighting_stance_idle", "E") == "ready"
    for combat in ("fighting_stance_entry", "fighting_stance_exit", "punch"):
        assert status(manifest, root, char, combat, "E") == "blocked"

    # Generate idle -> entry/exit/punch unlock.
    tree.animation("fighting_stance_idle", "E", frames=3)
    for combat in ("fighting_stance_entry", "fighting_stance_exit", "punch"):
        assert status(manifest, root, char, combat, "E") == "ready"


def test_editing_base_prompt_makes_whole_subtree_stale_but_selectable(manifest, tree):
    root, char = tree.root, tree.char
    # Fully render the chain fresh.
    tree.concept().pose("base", "E").pose("fighting_stance", "E").animation(
        "fighting_stance_idle", "E", frames=3
    )
    assert status(manifest, root, char, "punch", "E") == "ready"

    # "Edit the base pose prompt": its stored hash no longer matches the manifest.
    tree.pose("base", "E", stale=True)

    # base + everything downstream show stale, but stay selectable (amber).
    assert status(manifest, root, char, "base", "E") == "stale"
    assert status(manifest, root, char, "fighting_stance", "E") == "stale"
    assert status(manifest, root, char, "fighting_stance_idle", "E") == "stale"
    assert status(manifest, root, char, "punch", "E") == "stale"
    from andypack.resolve import resolve_animation
    assert resolve_animation(manifest, root, char, "punch", "E")["selectable"] is True
```

- [ ] **Step 2: Run to verify it fails, then passes against existing code**

Run: `pytest tests/test_acceptance.py -q`
Expected: PASS (the implementation from Tasks 1–7 already satisfies it). If anything fails, fix the implementation, not the test — the failure is a real defect in the resolver.

- [ ] **Step 3: Full suite + lint/type gate**

Run: `pytest -q && ruff check . && mypy andypack`
Expected: all tests pass; ruff clean; `Success: no issues found`.

- [ ] **Step 4: Commit**

```bash
git add tests/test_acceptance.py
git commit -m "test: end-to-end resolver acceptance (chain unlock + transitive stale)"
```

---

## Task 9: Port production `examples/animations.json` + update docs

The resolver core is done and green. This task makes the shipped manifest and the project docs match the new model.

**Files:**
- Modify: `examples/animations.json`
- Modify: `CLAUDE.md`
- Modify: `docs/anim-coord-node-spec.md`

- [ ] **Step 1: Port `examples/animations.json` to the new schema**

Apply these exact transformations to the existing file:

1. Delete the top-level `negatives` block entirely.
2. Add a top-level `globals` block after `defaults`:
   ```json
   "globals": {
     "animation": { "negative": "extra limbs, extra arms, mutated hands, fused fingers, deformed, blurry, low quality, jpeg artifacts, watermark, text" },
     "pose": { "negative": "deformed, blurry, low quality, jpeg artifacts, watermark, text" }
   },
   ```
3. Add a top-level `poses` block (before `animations`):
   ```json
   "poses": {
     "base": {
       "from": { "ref": "concept" },
       "prompt": "a neutral standing pose of the character, arms relaxed at its sides",
       "directions": {
         "E":  { "prompt": "the character facing directly to the right in profile" },
         "SE": { "prompt": "the character facing down-right at a three-quarter view" },
         "S":  { "prompt": "the character facing toward the viewer" }
       }
     },
     "fighting_stance": {
       "from": { "ref": "base", "direction": "same" },
       "prompt": "the character in a ready fighting stance, weight low on staggered feet, both fists raised in a guard in front of the chest and head",
       "directions": { "E": { "prompt": "facing to the right" } }
     }
   },
   ```
4. For **every** animation, replace the list form `"directions": ["X", "Y"]` with the map form `"directions": { "X": {}, "Y": {} }` — same keys, empty layer values. Leave every other animation field unchanged.
5. Re-point the existing combat dependencies from the old literal to the new refs (they are already `fighting_stance_idle`, so only the start anchor of `fighting_stance_idle` and `fighting_stance_entry` change):
   - `fighting_stance_idle`: set `"start_from": { "ref": "fighting_stance" }` (was `base_pose`).
   - `fighting_stance_entry`: set `"start_from": { "ref": "base" }` and keep `"end_at": { "ref": "fighting_stance_idle", "direction": "same" }`.
6. Add a `fighting_stance_exit` animation mirroring entry:
   ```json
   "fighting_stance_exit": {
     "category": "combat",
     "directions": { "E": {} },
     "loop": false, "length": 25,
     "start_from": { "ref": "fighting_stance_idle", "direction": "same" },
     "end_at": { "ref": "base" },
     "prompt": "The character drops out of its fighting stance back into a relaxed standing pose. It lowers its fists, unstaggers its feet, and settles upright, the heavy oversized head returning to a level neutral rest."
   },
   ```
7. Leave all other animations' `start_from`/`end_at`/`prompt`/`negative`/`loop`/`length`/`fps`/`category` exactly as they are (animations that had no deps still have none).

- [ ] **Step 2: Validate the ported manifest with the new loader**

Run:
```bash
python3 -c "from andypack.manifest import load_manifest; m=load_manifest('examples/animations.json'); print(len(m['poses']),'poses', len(m['animations']),'animations')"
```
Expected: prints the pose/animation counts with no `ManifestError` (warnings about non-`4n+1` lengths are acceptable and expected for some clips).

- [ ] **Step 3: Update `CLAUDE.md` non-negotiables**

Replace the "Non-negotiables" and "Build order" sections so they match the new model. Apply these edits:

- The FFLF bullet stays, but append: "Single-image deps (concept/pose) resolve the same image for either slot."
- Replace the writer-atomicity bullet with: "Writer order is atomic: payload first, then the `meta.json`/sidecar written LAST via temp-file + atomic rename. There is no `.complete` file; a dir with no parseable meta/sidecar is incomplete."
- Add a bullet: "Refs are typed: `concept` (seed), a pose id, or an animation id. Poses cascade `identity → globals.pose → pose → direction`; animations cascade `identity → globals.animation → animation → direction`. No facial/global negative special-casing."
- Update the build order list to: 1. `manifest.py`, 2. `resolve.py` + tests (done), 3. pose nodes, 4. animation nodes, 5. server routes, 6. web extension — and note the source of truth is now `docs/superpowers/specs/2026-06-29-cascading-pose-resolver-design.md`.

- [ ] **Step 4: Add a supersession note to `docs/anim-coord-node-spec.md`**

At the very top of `docs/anim-coord-node-spec.md`, insert:
```markdown
> **Superseded (2026-06-29):** the authoritative model is now
> `docs/superpowers/specs/2026-06-29-cascading-pose-resolver-design.md`
> (concept seed → poses → animations, cascading prompts, transitive staleness,
> meta-as-completion-sentinel). This document is retained for history; where the
> two disagree, the cascading-pose-resolver design wins.
```

- [ ] **Step 5: Full gate + commit**

Run: `pytest -q && ruff check . && mypy andypack`
Expected: all green.

```bash
git add examples/animations.json CLAUDE.md docs/anim-coord-node-spec.md
git commit -m "feat: port example manifest to cascading schema; update docs"
```

---

## Self-Review (completed by plan author)

- **Spec coverage:** §3 cascade/merge/hash → Task 3; §3 completeness (meta sentinel, no `.complete`) → Task 4; §5 anchors incl. single-image deps and `exit.end_at: base` → Task 5; §6 transitive staleness → Task 6; §6 `resolve_*`/`status` → Task 7; §8 build-step-2 + end-to-end acceptance → Tasks 7–8; manifest validation/cycles/ref-typing (§6/§7 loader) → Task 2; schema migration (§1, §5) → Task 9. Node/server/web subsystems (§7) are intentionally out of scope for this plan (separate plans).
- **Placeholder scan:** no `TBD`/`TODO`; every code step shows complete code; the Task 9 manifest port uses explicit mechanical rules with full content for the non-mechanical additions.
- **Type consistency:** `node_kind` returns `concept|pose|animation` everywhere; `merged_prompts`/`compute_prompt_hash` take `kind ∈ {pose, animation}`; `resolve_pose.stale` is a bool, `resolve_animation.stale` is a list of slot names, `blocked_by` shapes are consistent between Tasks 5/7 and their tests.

## Out of scope (subsequent plans, gated on this one)

- **Plan 2 — pose nodes:** `CharacterPoseSelector`, `PoseFrameWriter` (writes `{dir}.png` + sidecar, atomic), concept intake. Needs live ComfyUI API via ctx7.
- **Plan 3 — animation nodes:** `CharacterAnimationSelector`, `AnimationFrameWriter` (frames + `meta.json` last, loop closure).
- **Plan 4 — server routes:** `/characters`, `/options`, `/resolve`, `/frame` (path-traversal hardened), concept upload.
- **Plan 5 — web extension:** dynamic combos, status glyphs, source/dual previews, amber stale, auto-refresh on write.

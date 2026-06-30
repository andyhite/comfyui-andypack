# Character Creator Node Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Concept writer with a `CharacterCreator` node that pairs a (non-persisted) character reference image with a per-direction manikin to render all 8 base-pose directions as multi-reference FLUX.2 edits; make the `base` pose the tree root and retire the `concept` node.

**Architecture:** The reserved `concept` ref and `node_kind == "concept"` are deleted. A pose's `from` becomes optional — a pose with no `from` is a *root pose* (`base`). `character.json` keeps only the character prompt layer (no provenance); base sidecars carry provenance. `CharacterCreator` is a selector-style node (direction dropdown → one `ANIM_POSE`) that persists `character.json`, then emits the base job with the manikin attached as a second reference image. The prompt template var `{identity_prompt}` is renamed `{character_prompt}`.

**Tech Stack:** Python 3.10–3.12, pytest, ruff, mypy. Torch (CPU) only via the node/image layer. `resolve.py` and `manifest.py` stay free of torch/ComfyUI imports.

## Global Constraints

- `resolve.py` and `manifest.py` MUST NOT import torch or ComfyUI (`folder_paths`, `server`). Keep image/tensor work in `images.py` / `nodes.py`.
- Atomic write ordering for rendered nodes is unchanged: payload first, sidecar/meta LAST via temp-file + atomic rename. `character.json` is written atomically via `io.atomic_write_json`.
- Canonical 8 directions, in this exact order: `EAST, SOUTH_EAST, SOUTH, SOUTH_WEST, WEST, NORTH_WEST, NORTH, NORTH_EAST`.
- Manikin filename → direction map: `east→EAST`, `south_east→SOUTH_EAST`, `south→SOUTH`, `south_west→SOUTH_WEST`, `west→WEST`, `north_west→NORTH_WEST`, `north→NORTH`, `north_east→NORTH_EAST`.
- Character names are normalized with `io.to_snake_case` before use as a path segment.
- Run the full gate before declaring done: `pytest -q`, `ruff check .`, `mypy andypack`.

---

### Task 1: Manikin asset module + bundled images

**Files:**
- Create: `andypack/assets/manikins/{EAST,SOUTH_EAST,SOUTH,SOUTH_WEST,WEST,NORTH_WEST,NORTH,NORTH_EAST}.png`
- Create: `andypack/manikins.py`
- Test: `tests/test_manikins.py`

**Interfaces:**
- Produces: `manikins.CANONICAL_DIRECTIONS: list[str]`; `manikins.manikin_path(direction: str) -> str` (absolute path to the bundled PNG; raises `RuntimeError` for an unknown/missing direction).

- [ ] **Step 1: Copy the drawn manikins into the package, renamed to canonical UPPERCASE direction names**

```bash
mkdir -p andypack/assets/manikins
cp ~/Desktop/poses/east.png        andypack/assets/manikins/EAST.png
cp ~/Desktop/poses/south_east.png  andypack/assets/manikins/SOUTH_EAST.png
cp ~/Desktop/poses/south.png       andypack/assets/manikins/SOUTH.png
cp ~/Desktop/poses/south_west.png  andypack/assets/manikins/SOUTH_WEST.png
cp ~/Desktop/poses/west.png        andypack/assets/manikins/WEST.png
cp ~/Desktop/poses/north_west.png  andypack/assets/manikins/NORTH_WEST.png
cp ~/Desktop/poses/north.png       andypack/assets/manikins/NORTH.png
cp ~/Desktop/poses/north_east.png  andypack/assets/manikins/NORTH_EAST.png
ls andypack/assets/manikins   # expect 8 PNGs
```

- [ ] **Step 2: Write the failing test**

```python
# tests/test_manikins.py
import os
import pytest
from andypack import manikins


def test_canonical_directions_are_the_eight_in_order():
    assert manikins.CANONICAL_DIRECTIONS == [
        "EAST", "SOUTH_EAST", "SOUTH", "SOUTH_WEST",
        "WEST", "NORTH_WEST", "NORTH", "NORTH_EAST",
    ]


def test_manikin_path_resolves_every_direction_to_an_existing_file():
    for direction in manikins.CANONICAL_DIRECTIONS:
        path = manikins.manikin_path(direction)
        assert os.path.isfile(path), path


def test_manikin_path_rejects_unknown_direction():
    with pytest.raises(RuntimeError, match="manikin"):
        manikins.manikin_path("UP")
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_manikins.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'andypack.manikins'`

- [ ] **Step 4: Write the module**

```python
# andypack/manikins.py
"""Bundled manikin pose references, keyed by canonical direction (pure stdlib)."""

from __future__ import annotations

import os

CANONICAL_DIRECTIONS = [
    "EAST", "SOUTH_EAST", "SOUTH", "SOUTH_WEST",
    "WEST", "NORTH_WEST", "NORTH", "NORTH_EAST",
]

_ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets", "manikins")


def manikin_path(direction: str) -> str:
    """Absolute path to the bundled manikin PNG for `direction`.

    Raises RuntimeError for a direction outside CANONICAL_DIRECTIONS or whose
    asset is missing from the package, so a misconfigured graph fails loudly
    rather than feeding a nonexistent path to the image loader.
    """
    if direction not in CANONICAL_DIRECTIONS:
        raise RuntimeError(f"no manikin for unknown direction {direction!r}")
    path = os.path.join(_ASSETS_DIR, f"{direction}.png")
    if not os.path.isfile(path):
        raise RuntimeError(f"missing manikin asset for {direction!r} (expected {path})")
    return path
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_manikins.py -q`
Expected: PASS (3 passed)

- [ ] **Step 6: Commit**

```bash
git add andypack/manikins.py andypack/assets/manikins tests/test_manikins.py
git commit -m "feat: bundle manikin pose references + manikin_path helper"
```

---

### Task 2: `io.build_character` — character prompt layer with no provenance

**Files:**
- Modify: `andypack/io.py` (add `build_character`; leave `build_concept_sidecar` for now — Task 3 removes it)
- Test: `tests/test_io.py`

**Interfaces:**
- Produces: `io.build_character(layer: dict, existing: Optional[dict] = None) -> dict` — returns the merged dict: every non-owned key from `existing` preserved, then `layer` applied. Owned keys: `positive_prompt`, `negative_prompt`. No `prompt_hash`/`created_utc`/`render_id`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_io.py  (add these)
from andypack import io


def test_build_character_is_just_the_layer_when_no_existing():
    out = io.build_character({"positive_prompt": "a hero", "negative_prompt": "blurry"})
    assert out == {"positive_prompt": "a hero", "negative_prompt": "blurry"}
    assert "render_id" not in out and "prompt_hash" not in out and "created_utc" not in out


def test_build_character_preserves_overlay_and_drops_cleared_keys():
    existing = {
        "positive_prompt": "old", "negative_prompt": "old neg",
        "poses": {"wave": {"from": {"ref": "base"}, "directions": {"EAST": {}}}},
    }
    # New layer omits negative_prompt (widget cleared) — it must be dropped,
    # while the character-authored `poses` overlay survives.
    out = io.build_character({"positive_prompt": "new"}, existing=existing)
    assert out["positive_prompt"] == "new"
    assert "negative_prompt" not in out
    assert out["poses"] == existing["poses"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_io.py -k build_character -q`
Expected: FAIL with `AttributeError: module 'andypack.io' has no attribute 'build_character'`

- [ ] **Step 3: Add the function to `andypack/io.py`**

Add near `build_concept_sidecar`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_io.py -k build_character -q`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add andypack/io.py tests/test_io.py
git commit -m "feat: io.build_character (character prompt layer, no provenance)"
```

---

### Task 3: Retire the concept node — base becomes the tree root

This is the core rename. It is one atomic change because `node_kind`, the seed/fixture `from`, the resolver, and the node layer are tightly coupled; the suite goes green only when all move together. Removes the concept tree node, makes a pose's `from` optional (root poses), renames the identity helpers + template var, repoints `character.json`, and deletes the Concept writer/loader nodes.

**Files:**
- Modify: `andypack/manifest.py` (`node_kind`, `_validate_refs`, `_dependency_edges`)
- Modify: `andypack/resolve.py` (remove concept branches, root-pose handling, rename `read_identity`→`read_character` / `invalidate_identity`→`invalidate_character`, `_TEMPLATE_TOKEN`)
- Modify: `andypack/io.py` (delete `build_concept_sidecar` + `_CONCEPT_OWNED_KEYS`)
- Modify: `andypack/api.py` (`_is_character` → `character.json`)
- Modify: `andypack/nodes.py` (delete `ConceptImageWriter` + `ConceptImageLoader` + their mapping entries; rename `resolve.read_identity` call sites)
- Modify: `tests/conftest.py` (`Tree.identity`→`Tree.character` writing `character.json`; drop `Tree.concept`; `Tree.pose` uses `.get("from")`)
- Modify: `tests/fixtures/manifest.json` (drop `base.from`)
- Modify: `tests/test_nodes.py`, `tests/test_manifest.py`, `tests/test_resolve.py`, `tests/test_provenance.py`, `tests/test_staleness.py`, `tests/test_merge.py`, `tests/test_api.py`, `tests/test_acceptance.py` (vocabulary + dropped concept tests)

**Interfaces:**
- Produces: `resolve.read_character(root, character) -> dict`; `resolve.invalidate_character(root, character) -> None`; template var `{character_prompt}`. `node_kind` returns only `"pose"`/`"animation"`. `resolve_pose` accepts a pose with no `from` (root): `blocked_by=[]`, `source_image=None`, `stale=False`, `meta["from"]=None`, `meta["sources"]={}`.
- Consumes: `io.build_character` (Task 2).

- [ ] **Step 1: Update the fixture so `base` is a root pose**

In `tests/fixtures/manifest.json`, delete the `"from": { "ref": "concept" }` line from `poses.base` (leave the rest of `base` intact). `base` now has no `from`.

- [ ] **Step 2: Update `tests/conftest.py`**

Replace `Tree.concept` and `Tree.identity` with a single `character` builder, and make `Tree.pose` tolerate a missing `from`:

```python
    # delete the concept() method entirely; replace identity() with:
    def character(self, **layer):
        _write_json(os.path.join(self._cdir(), "character.json"), layer)
        return self
```

In `Tree.pose`, change the sidecar `"from"` line to tolerate a root pose:

```python
                "from": self.m["poses"][pose_id].get("from"),
```

- [ ] **Step 3: Update `manifest.py`**

`node_kind` — drop the concept branch:

```python
def node_kind(manifest: Manifest, ref: str) -> str:
    """Classify a ref as 'pose' or 'animation' (raises on an unknown ref)."""
    if ref in manifest.get("poses", {}):
        return "pose"
    if ref in manifest.get("animations", {}):
        return "animation"
    raise ManifestError(f"unknown ref: {ref!r}")
```

`_validate_refs` — make `from` optional, forbid only an animation target:

```python
    for pid, pose in manifest.get("poses", {}).items():
        frm = pose.get("from")
        if frm is not None:
            if not isinstance(frm, dict) or "ref" not in frm:
                raise ManifestError(f"pose {pid!r} 'from' must be an object with a 'ref'")
            if node_kind(manifest, frm["ref"]) == "animation":
                raise ManifestError(f"pose {pid!r} 'from' must reference a pose")
        _validate_directions(f"pose {pid!r}", pose)
```

`_dependency_edges` — drop the concept special-case (`add` already guards a falsy ref); a root pose contributes no edge:

```python
    def add(node: str, ref: str | None) -> None:
        edges.setdefault(node, [])
        if ref:
            edges[node].append(ref)
    ...
    for pid, pose in manifest.get("poses", {}).items():
        add(pid, (pose.get("from") or {}).get("ref"))
```

- [ ] **Step 4: Update `resolve.py`**

Rename the identity helpers and repoint the path (whole `read_identity`/`invalidate_identity` pair):

```python
def read_character(root: str, character: str) -> dict:
    """Per-character prompt layer from `character.json`, or {} if absent/corrupt.
    Memoized by path+mtime (the resolve/report hot paths re-read it many times per
    cell). Callers must not mutate the returned dict."""
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
    from any character layer), forcing the next read to hit disk. The creator
    node calls this after rewriting character.json, because a rewrite can land
    within a coarse filesystem's mtime resolution window."""
    _IDENTITY_CACHE.pop(os.path.join(root, character, "character.json"), None)
    _EFFECTIVE_CACHE.clear()
```

In `effective_manifest` and `merged_prompts`, change `read_identity(` → `read_character(`. The `identity = read_character(...)` local name may stay.

`_TEMPLATE_TOKEN` and `substitute_variables` — rename the token:

```python
_TEMPLATE_TOKEN = re.compile(r"\{(character_prompt|direction_prompt|direction_name)\}")
```

```python
    values = {
        "character_prompt": (identity.get(field) or "").strip(),
        "direction_prompt": (direction_layer.get(field) or "").strip(),
        "direction_name": direction,
    }
```

Delete `_concept_png`, `concept_image_path`, `concept_complete`. Remove every `kind == "concept"` branch:

```python
def node_complete(manifest, root, character, ref, direction):
    kind = node_kind(manifest, ref)
    if kind == "pose":
        return pose_complete(root, character, ref, direction)
    return animation_complete(root, character, ref, direction)


def read_node_meta(manifest, root, character, ref, direction):
    kind = node_kind(manifest, ref)
    if kind == "pose":
        return _read_json(_pose_sidecar(root, character, ref, direction))
    return _read_json(_anim_meta_path(root, character, ref, direction))


def read_render_id(manifest, root, character, ref, direction):
    """The rendered node's `render_id`, or None when unrendered."""
    meta = read_node_meta(manifest, root, character, ref, direction)
    return meta.get("render_id") if meta else None


def _single_image(manifest, root, character, ref, direction):
    if node_kind(manifest, ref) == "pose":
        return _pose_png(root, character, ref, direction)
    return None  # animations are not single-image
```

`pose_source_image` and `direct_deps` — handle a root pose's missing `from`:

```python
def pose_source_image(manifest, root, character, pose_id, direction):
    frm = manifest["poses"][pose_id].get("from")
    if not frm:
        return None
    return _single_image(manifest, root, character, frm["ref"], resolved_dir(frm, direction))
```

```python
    if kind == "pose":
        frm = manifest["poses"][ref].get("from")
        return [(frm["ref"], resolved_dir(frm, direction))] if frm else []
```

`_anchor_from_dep` — `concept` is gone, so the single-image branch keys on `pose` only:

```python
    kind = node_kind(manifest, dep["ref"])
    if kind == "pose":
        return _single_image(manifest, root, character, dep["ref"], ddir)
    return _animation_frame(manifest, root, character, dep["ref"], ddir, frame_key)
```

`outdated` — remove the `kind == "concept": return False` early-out; a root pose recurses into no ancestor:

```python
    if kind == "pose":
        frm = manifest["poses"][ref].get("from")
        if not frm:
            return False
        return outdated(manifest, root, character, frm["ref"], resolved_dir(frm, direction))
```

`resolve_pose` — branch on whether the pose has a `from`:

```python
def resolve_pose(manifest, root, character, pose_id, direction):
    pose = manifest["poses"][pose_id]
    frm = pose.get("from")
    if frm:
        src_dir = resolved_dir(frm, direction)
        src_complete = node_complete(manifest, root, character, frm["ref"], src_dir)
        blocked_by = [] if src_complete else [{"from": frm, "dir": src_dir}]
        stale = src_complete and outdated(manifest, root, character, frm["ref"], src_dir)
        source_image = pose_source_image(manifest, root, character, pose_id, direction) if src_complete else None
    else:
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
```

Also update the module docstring line that mentions `_concept.json` → `character.json`.

- [ ] **Step 5: Update `io.py` — delete the old concept sidecar**

Remove `build_concept_sidecar` and `_CONCEPT_OWNED_KEYS` entirely (Task 2's `build_character` replaces them).

- [ ] **Step 6: Update `api.py` — character-directory marker**

```python
def _is_character(root: str, name: str) -> bool:
    d = os.path.join(root, name)
    if not os.path.isdir(d):
        return False
    if os.path.exists(os.path.join(d, "character.json")):
        return True
    try:
        return any(os.path.isdir(os.path.join(d, c)) for c in os.listdir(d))
    except OSError:
        return False
```

- [ ] **Step 7: Update `nodes.py` — delete the Concept writer/loader**

Delete the entire `ConceptImageWriter` class, the entire `ConceptImageLoader` class, and their entries in `NODE_CLASS_MAPPINGS` and `NODE_DISPLAY_NAME_MAPPINGS`. Anywhere a remaining node calls `resolve.read_identity(`, change it to `resolve.read_character(`. (No remaining node should reference `concept_image_path` or `build_concept_sidecar`.)

- [ ] **Step 8: Update the affected tests (vocabulary + drop concept-node tests)**

In `tests/test_nodes.py`:
- Delete `test_concept_writer_writes_provenance_sidecar`, `test_concept_writer_preserves_authored_poses`, `test_pose_resolve_records_concept_render_id`, `test_concept_image_loader_reads_image_and_identity`, `test_concept_image_loader_missing_concept`.
- In `test_pose_selector_is_changed_tracks_dependency_render`, replace the concept setup with a root-base setup:

```python
def test_pose_selector_is_changed_tracks_dependency_render(manifest, tree, monkeypatch):
    monkeypatch.setattr(nodes, "_characters_root", lambda: tree.root)
    manifest["poses"]["base"]["positive_prompt"] += " {character_prompt}"
    tree.pose("base", "EAST")  # base is the root; render it directly
    before = nodes.CharacterPoseSelector.IS_CHANGED(
        manifest, tree.char, "", "fighting_stance", "EAST"
    )
    tree.character(positive_prompt="a brand new character line")
    after = nodes.CharacterPoseSelector.IS_CHANGED(
        manifest, tree.char, "", "fighting_stance", "EAST"
    )
    assert before != after
```

In every affected test file, apply these mechanical replacements:
- `tree.concept().` → `` (delete the call; `.concept()` no longer exists). Where a test needs base rendered, use `tree.pose("base", "<DIR>")`.
- `tree.identity(` → `tree.character(`
- `resolve.read_identity` → `resolve.read_character`; `resolve.invalidate_identity` → `resolve.invalidate_character`
- `{identity_prompt}` → `{character_prompt}` (in test prompt strings)
- `_concept.json` → `character.json`; `_concept.png` references removed.
- Any assertion that `node_kind(..., "concept")` returns `"concept"` is removed; add `pytest.raises(ManifestError)` for an unknown ref if a test covered that.

- [ ] **Step 9: Add root-pose coverage to `tests/test_manifest.py` and `tests/test_resolve.py`**

```python
# tests/test_manifest.py
from andypack.manifest import topo_order, validate_manifest


def test_root_pose_with_no_from_validates_and_sorts_as_leaf(manifest):
    validate_manifest(manifest)            # fixture base has no `from`
    order = topo_order(manifest)
    assert order.index("base") < order.index("fighting_stance")
```

```python
# tests/test_resolve.py
from andypack import resolve


def test_root_pose_resolves_with_no_source_and_empty_sources(manifest, tree):
    r = resolve.resolve_pose(manifest, tree.root, tree.char, "base", "EAST")
    assert r["selectable"] is True          # EAST in base.directions, no source dep
    assert r["blocked_by"] == [] and r["stale"] is False
    assert r["source_image"] is None
    assert r["meta"]["from"] is None and r["meta"]["sources"] == {}


def test_root_pose_outdated_only_on_own_prompt_drift(manifest, tree):
    tree.pose("base", "EAST")
    assert resolve.outdated(manifest, tree.root, tree.char, "base", "EAST") is False
    manifest["poses"]["base"]["directions"]["EAST"]["positive_prompt"] = "changed"
    assert resolve.outdated(manifest, tree.root, tree.char, "base", "EAST") is True
```

- [ ] **Step 10: Run the full suite and fix fallout**

Run: `pytest -q`
Expected: PASS. Investigate any remaining `_concept`/`identity`/`concept` references the search below reports, and fix:

```bash
grep -rn "concept\|_concept\|read_identity\|invalidate_identity\|identity_prompt\|build_concept" andypack tests | grep -v "character"
```
Expected: no hits (an empty result).

- [ ] **Step 11: Lint + types**

Run: `ruff check . && mypy andypack`
Expected: clean.

- [ ] **Step 12: Commit**

```bash
git add -A
git commit -m "refactor: retire concept node; base is the tree root, character.json holds prompts only"
```

---

### Task 4: `CharacterCreator` node + `pose_reference` in the POSE bundle

**Files:**
- Modify: `andypack/nodes.py` (add `CharacterCreator`; add `pose_reference` to `POSE_OUTPUT_KEYS`, `_POSE_UNPACK`, `PoseUnpack`; `CharacterPoseSelector` sets `pose_reference` empty + rejects root poses; mappings)
- Test: `tests/test_nodes.py`

**Interfaces:**
- Consumes: `manikins.CANONICAL_DIRECTIONS`, `manikins.manikin_path` (Task 1); `io.build_character` (Task 2); `resolve.read_character`/`resolve.invalidate_character`/`resolve_pose` (Task 3).
- Produces: `nodes.CharacterCreator.create(manifest, image, character, direction, character_positive="", character_negative="") -> (pose_dict,)` where `pose_dict` has keys `source_image, pose_reference, positive, negative, output_dir, _meta`. `POSE_OUTPUT_KEYS` includes `"pose_reference"`; `PoseUnpack` gains a `POSE_REFERENCE` (IMAGE) output.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_nodes.py  (add)
from andypack import manikins


def test_character_creator_writes_character_json_without_provenance(manifest, tmp_path, monkeypatch):
    root = str(tmp_path)
    monkeypatch.setattr(nodes, "_characters_root", lambda: root)
    (pose,) = nodes.CharacterCreator().create(
        manifest, _img(), "Cortex", "EAST",
        character_positive="a brave hero", character_negative="blurry",
    )
    data = json.load(open(os.path.join(root, "cortex", "character.json")))
    assert data == {"positive_prompt": "a brave hero", "negative_prompt": "blurry"}
    assert pose["output_dir"].endswith(os.path.join("cortex", "_base"))
    assert pose["_meta"]["pose"] == "base" and pose["_meta"]["direction"] == "EAST"


def test_character_creator_attaches_manikin_as_pose_reference(manifest, tmp_path, monkeypatch):
    monkeypatch.setattr(nodes, "_characters_root", lambda: str(tmp_path))
    (pose,) = nodes.CharacterCreator().create(manifest, _img(), "cortex", "EAST")
    # The manikin rides along as a real (non-empty) second reference image.
    assert pose["pose_reference"] is not None
    assert not images.is_empty(pose["pose_reference"])


def test_character_creator_rejects_unknown_direction(manifest, tmp_path, monkeypatch):
    monkeypatch.setattr(nodes, "_characters_root", lambda: str(tmp_path))
    with pytest.raises(RuntimeError, match="direction"):
        nodes.CharacterCreator().create(manifest, _img(), "cortex", "UP")


def test_pose_selector_rejects_root_pose(manifest, tree, monkeypatch):
    monkeypatch.setattr(nodes, "_characters_root", lambda: tree.root)
    tree.pose("base", "EAST")
    with pytest.raises(RuntimeError, match="root pose"):
        nodes.CharacterPoseSelector().select(manifest, tree.char, "", "base", "EAST")


def test_pose_selector_sets_empty_pose_reference(manifest, tree, monkeypatch):
    monkeypatch.setattr(nodes, "_characters_root", lambda: tree.root)
    tree.pose("base", "EAST")
    (pose,) = nodes.CharacterPoseSelector().select(manifest, tree.char, "", "fighting_stance", "EAST")
    assert images.is_empty(pose["pose_reference"])
```

Also update `test_pose_selector_returns_single_dict` and `test_unpack_outputs_cover_selector_leaf_keys` for the new key (the latter already asserts `{k for k,_ in nodes._POSE_UNPACK} == set(nodes.POSE_OUTPUT_KEYS)` and the RETURN_TYPES length — both will pass once the key is added everywhere). In `_pose_dict` helper, add `"pose_reference": None`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_nodes.py -k "character_creator or root_pose or pose_reference" -q`
Expected: FAIL (`AttributeError: ... 'CharacterCreator'` and `KeyError: 'pose_reference'`).

- [ ] **Step 3: Add `pose_reference` to the bundle plumbing**

In `nodes.py`, add the import and the key:

```python
from andypack import api, images, io, manikins, resolve
```

```python
POSE_OUTPUT_KEYS = sorted([
    "source_image", "pose_reference", "positive", "negative", "output_dir",
])
```

```python
_POSE_UNPACK = (
    ("source_image", "SOURCE_IMAGE"),
    ("pose_reference", "POSE_REFERENCE"),
    ("positive", "POSITIVE_PROMPT"),
    ("negative", "NEGATIVE_PROMPT"),
    ("output_dir", "OUTPUT_DIR"),
)
```

```python
class PoseUnpack:
    ...
    RETURN_TYPES = ("ANIM_POSE", "IMAGE", "IMAGE", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("POSE", *(name for _key, name in _POSE_UNPACK))
```

- [ ] **Step 4: Update `CharacterPoseSelector.select` — empty `pose_reference` + root-pose guard**

After the existing "unknown pose" check, add the guard, and add `pose_reference` to the emitted dict:

```python
        if pose not in manifest.get("poses", {}):
            raise RuntimeError(
                f"CharacterPoseSelector: unknown pose {pose!r} (stale or renamed) — pick a pose"
            )
        if not manifest["poses"][pose].get("from"):
            raise RuntimeError(
                f"CharacterPoseSelector: {pose!r} is a root pose — use the Character Creator node"
            )
        r = resolve_pose(manifest, root, character, pose, direction)
        ...
        pose = {
            "source_image": image,
            "pose_reference": images.empty_image(),
            "positive": r["positive"],
            "negative": r["negative"],
            "output_dir": r["output_dir"],
            "_meta": r["meta"],
        }
        return (pose,)
```

- [ ] **Step 5: Add the `CharacterCreator` class**

Place it where `ConceptImageWriter` used to be:

```python
class CharacterCreator:
    """Persist a character's prompt layer (character.json — no image, no
    provenance) and emit the base-pose job for one direction, pairing the
    reference image (first) with the bundled manikin for that direction (second)
    for a multi-reference FLUX.2 edit. Selector-style: pick a direction, get one
    ANIM_POSE; the base pose is the tree root."""

    CATEGORY = "andypack/Character"
    FUNCTION = "create"
    RETURN_TYPES = ("ANIM_POSE",)
    RETURN_NAMES = ("POSE",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "manifest": ("ANIM_MANIFEST",),
                "image": ("IMAGE",),
                "character": ("STRING", {"default": "cortex"}),
                "direction": (manikins.CANONICAL_DIRECTIONS,),
            },
            "optional": {
                "character_positive": ("STRING", {"default": "", "multiline": True}),
                "character_negative": ("STRING", {"default": "", "multiline": True}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, manifest, image, character, direction,
                   character_positive="", character_negative=""):
        if not character or not direction:
            return float("nan")
        root = _characters_root()
        try:
            char_name = io.to_snake_case(character)
            eff = effective_manifest(manifest, root, char_name)
            r = resolve_pose(eff, root, char_name, "base", direction)
        except Exception:
            return float("nan")
        return "|".join([
            r["meta"]["prompt_hash"], direction,
            character_positive.strip(), character_negative.strip(),
        ])

    def create(self, manifest, image, character, direction,
               character_positive="", character_negative=""):
        if direction not in manikins.CANONICAL_DIRECTIONS:
            raise RuntimeError(f"CharacterCreator: unknown direction {direction!r}")
        root = _characters_root()
        char_name = io.to_snake_case(character)
        layer = {}
        if character_positive.strip():
            layer["positive_prompt"] = character_positive.strip()
        if character_negative.strip():
            layer["negative_prompt"] = character_negative.strip()
        existing = resolve.read_character(root, char_name)
        payload = io.build_character(layer, existing=existing)
        io.atomic_write_json(os.path.join(root, char_name, "character.json"), payload)
        resolve.invalidate_character(root, char_name)

        eff = effective_manifest(manifest, root, char_name)
        if "base" not in eff.get("poses", {}):
            raise RuntimeError("CharacterCreator: manifest has no 'base' pose")
        if direction not in eff["poses"]["base"]["directions"]:
            raise RuntimeError(f"CharacterCreator: base has no direction {direction!r}")
        r = resolve_pose(eff, root, char_name, "base", direction)
        manikin = images.load_image_tensor(manikins.manikin_path(direction))
        pose = {
            "source_image": image,
            "pose_reference": manikin,
            "positive": r["positive"],
            "negative": r["negative"],
            "output_dir": r["output_dir"],
            "_meta": r["meta"],
        }
        return (pose,)
```

- [ ] **Step 6: Register the node**

In `NODE_CLASS_MAPPINGS` add `"CharacterCreator": CharacterCreator,` and in `NODE_DISPLAY_NAME_MAPPINGS` add `"CharacterCreator": "Character Creator",`.

- [ ] **Step 7: Run the node tests**

Run: `pytest tests/test_nodes.py -q`
Expected: PASS (including the updated `test_unpack_outputs_cover_selector_leaf_keys`).

- [ ] **Step 8: Full gate**

Run: `pytest -q && ruff check . && mypy andypack`
Expected: clean.

- [ ] **Step 9: Commit**

```bash
git add andypack/nodes.py tests/test_nodes.py
git commit -m "feat: CharacterCreator node + manikin pose_reference in the POSE bundle"
```

---

### Task 5: Seed manifest — base is root, all 8 directions, multi-reference prompt

**Files:**
- Modify: `examples/animations.json`
- Test: `tests/test_seed.py`

**Interfaces:**
- Produces: a seed manifest whose `base` pose has no `from`, lists all 8 canonical directions, uses a multi-reference `positive_prompt` template, and uses `{character_prompt}` (never `{identity_prompt}`).

- [ ] **Step 1: Write failing tests**

```python
# tests/test_seed.py  (add)
import json
from andypack import api
from andypack.manifest import topo_order, validate_manifest

EXAMPLE = "examples/animations.json"


def test_seed_base_is_root_with_all_eight_directions():
    m = json.loads(open(EXAMPLE, encoding="utf-8").read())
    base = m["poses"]["base"]
    assert "from" not in base                       # base is the tree root
    assert set(base["directions"]) == {
        "EAST", "SOUTH_EAST", "SOUTH", "SOUTH_WEST",
        "WEST", "NORTH_WEST", "NORTH", "NORTH_EAST",
    }
    validate_manifest(m)
    topo_order(m)                                   # no cycle, sorts


def test_seed_uses_character_prompt_token_not_identity():
    raw = open(EXAMPLE, encoding="utf-8").read()
    assert "{identity_prompt}" not in raw
    assert "{character_prompt}" in raw
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_seed.py -k "base_is_root or character_prompt" -q`
Expected: FAIL (base still has `from`, only 5 directions, `{identity_prompt}` present).

- [ ] **Step 3: Edit `examples/animations.json`**

1. In `poses.base`, delete the `"from": { "ref": "concept" }` block.
2. Rewrite `poses.base.positive_prompt` to the multi-reference template (reference order: image 1 = character, image 2 = manikin):

```
"Edit the first image — the character reference — to show that exact character facing {direction_name}, using the gray articulated mannequin in the second image purely as the pose-and-camera-angle guide for the body orientation.\n\nThe character holds a neutral standing pose: standing upright and still, weight even on both feet, legs straight and together, both arms relaxed at the sides. Match the body's facing direction and viewing angle to the mannequin; do not copy the mannequin's gray featureless surface, proportions, or lack of detail.\n\nPreserve the character's identity, design, colors, and proportions exactly as in the first image: {character_prompt}\n\nAs viewed from the {direction_name}: {direction_prompt}"
```

3. Add the three missing `directions` entries to `poses.base` (author them as the left-facing counterparts of the existing right-facing prose). Add after `NORTH`:

```json
        "WEST": {
          "positive_prompt": "a clean full-body side profile, the whole character turned 90 degrees to face the left edge of the frame — the side-on orientation a character has when facing left in a 2D side-scrolling platformer, standing upright with legs straight and together. The head is shown in pure lateral profile, exactly one eye visible on the side facing us. The body is fully sideways, chest toward the left edge, one arm at the near side, the near leg overlapping the far leg, both shoes seen edge-on with the toes pointing left.",
          "negative_prompt": "far eye, second eye, both eyes visible, two eyes"
        },
        "SOUTH_WEST": {
          "positive_prompt": "a three-quarter front view, the character turned about 45 degrees to face toward the lower-left of the frame — halfway between the front view and the left side profile, an orthographic character-turnaround three-quarter front pose, the head angled toward the lower left. Both eyes are in view, the near eye larger and the far eye foreshortened. The near arm is full and forward; the far arm recedes behind the torso with just its edge showing. The legs angle toward the left, both shoes pointing toward the lower left, the far shoe partly behind the near one."
        },
        "NORTH_WEST": {
          "positive_prompt": "a three-quarter back view, the character turned to face mostly away and toward the left edge of the frame — halfway between the back view and the left side profile, an orthographic character-turnaround three-quarter back pose. The rounded rear of the head is toward the viewer and the face is turned away to the far side. The back of the body and clothing is in view; the near arm shows at its side and the far arm is tucked behind the torso. The legs angle toward the left, the shoes seen from behind and the side with the heels showing, toes pointing away to the left.",
          "negative_prompt": "face, facial features, eyes"
        }
```

4. Replace every remaining `{identity_prompt}` in the file with `{character_prompt}` (the `globals.pose` / `globals.animation` negatives and each pose/animation `positive_prompt`). Verify:

```bash
grep -c "{identity_prompt}" examples/animations.json   # expect 0
grep -c "{character_prompt}" examples/animations.json   # expect > 0
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_seed.py -q`
Expected: PASS (existing seed tests + the two new ones).

- [ ] **Step 5: Commit**

```bash
git add examples/animations.json tests/test_seed.py
git commit -m "feat: seed manifest — base roots the tree, all 8 directions, multi-ref prompt"
```

---

### Task 6: Expose root-pose flag to the frontend + hide root poses in the pose selector

**Files:**
- Modify: `andypack/api.py` (`list_options` pose rows get `"root"`)
- Modify: `web/anim_coord.js` (filter root poses out of the `CharacterPoseSelector` id combo)
- Test: `tests/test_api.py`

**Interfaces:**
- Produces: each pose row from `api.list_options` carries `"root": bool` (`True` when the pose has no `from`); animation rows carry `"root": False`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_api.py  (add)
from andypack import api


def test_list_options_marks_root_poses(manifest, tmp_path):
    rows = api.list_options(manifest, str(tmp_path), "cortex")
    by_id = {(r["kind"], r["id"], r["direction"]): r for r in rows}
    assert by_id[("pose", "base", "EAST")]["root"] is True
    assert by_id[("pose", "fighting_stance", "EAST")]["root"] is False
    assert by_id[("animation", "walk", "EAST")]["root"] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_api.py -k marks_root_poses -q`
Expected: FAIL with `KeyError: 'root'`.

- [ ] **Step 3: Add `root` to the rows in `api.list_options`**

In the poses loop, add `"root": pose.get("from") is None` to the appended row dict; in the animations loop, add `"root": False`. (The exact row dict is the one already built with `"kind"`, `"id"`, `"direction"`, `"status"`.)

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_api.py -k marks_root_poses -q`
Expected: PASS.

- [ ] **Step 5: Filter root poses in the web combo**

In `web/anim_coord.js`, in the function that builds the id combo (`buildIdCombo`), where it filters options to the node's kind, also drop root poses for the pose selector. Find the line filtering by kind (it produces the id options from `opts` filtered to `o.kind === cfg.kind`) and change the predicate to also exclude `o.root` when the kind is `pose`:

```javascript
  const ids = opts.filter((o) => o.kind === cfg.kind && !(cfg.kind === "pose" && o.root));
```

(There is no automated test for the JS; verify manually in ComfyUI that `base` no longer appears in the Pose selector's dropdown but still appears in coverage/diagnostics.)

- [ ] **Step 6: Full gate + commit**

```bash
pytest -q && ruff check . && mypy andypack
git add andypack/api.py web/anim_coord.js tests/test_api.py
git commit -m "feat: mark root poses in options; hide base from the pose selector combo"
```

---

### Task 7: Update README

**Files:**
- Modify: `README.md`

**Interfaces:** none (docs only).

- [ ] **Step 1: Rewrite the concept references**

Apply these edits to `README.md`:
- Diagram: replace the `concept["_concept.png ..."] --> base` node with `base` as the root (e.g. `base["base pose<br/>(per direction, from a manikin)"] --> ...`); the character reference + manikin are inputs to base, not a persisted node.
- On-disk layout: remove the `_concept.png` / `_concept.json` lines; add `character.json   character prompt layer { positive_prompt?, negative_prompt? } (no image saved)`.
- Node table: delete the **Concept Image Writer** and **Concept Image Loader** rows; add **Character Creator** — "Write a character's `character.json` prompt layer and emit the base-pose job for one direction, pairing the reference image with the bundled manikin (multi-reference FLUX.2 edit)."
- Workflow steps: replace "Concept Image Writer once per character (uploads `_concept.png`)" with "Character Creator per base direction (reference image + manikin → base pose). The reference image is not persisted — keep it in your graph."
- Replace any `{identity_prompt}` mention with `{character_prompt}`; replace prose describing identity living in `_concept.json` with `character.json`.
- Add a short "Manikins" note: the 8 bundled pose references in `andypack/assets/manikins/` supply the per-direction camera angle; base renders all 8 directly (no mirroring for base).

- [ ] **Step 2: Sanity-check for stale references**

```bash
grep -n "concept\|_concept\|identity_prompt" README.md
```
Expected: no hits.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: README for CharacterCreator + base-as-root + manikins"
```

---

## Self-Review

**Spec coverage:**
- Remove concept tree node / base-as-root / optional `from` → Task 3.
- Reference image not persisted; `character.json` = prompts only, no provenance → Tasks 2 (builder) + 3 (no image written; creator in Task 4 writes no image).
- Base sidecars carry provenance (`sources` empty) → Task 3 (`resolve_pose` meta) + existing `PoseFrameWriter`/`build_pose_sidecar` (unchanged).
- All 8 base directions from manikins → Tasks 1 (assets) + 5 (seed directions) + 4 (pairing).
- Manifest-driven prompts + multi-ref base template → Task 5.
- `{identity_prompt}`→`{character_prompt}`, node inputs `character_positive/negative`, `read_identity`/`invalidate_identity` rename → Tasks 3 + 4.
- Selector-style node, `ANIM_POSE`, `PoseUnpack`/`PoseFrameWriter` reuse → Task 4.
- `pose_reference` leaf + `PoseUnpack` output → Task 4.
- Exclude root poses from `CharacterPoseSelector` → Task 4 (guard) + Task 6 (combo filter, server flag).
- Remove `ConceptImageLoader` → Task 3.
- `api._is_character` marker → Task 3.
- Error handling (unknown direction, missing base/direction, missing manikin) → Tasks 1 + 4.
- Tests across the rename → Tasks 1–6. README → Task 7.

**Placeholder scan:** none — every code/test step shows the actual content; authored prose for the 3 new directions and the multi-ref template is provided verbatim in Task 5.

**Type consistency:** `read_character`/`invalidate_character`, `build_character(layer, existing=)`, `manikin_path(direction)`, `CANONICAL_DIRECTIONS`, the `POSE_OUTPUT_KEYS`/`_POSE_UNPACK`/`PoseUnpack` triple (all gain `pose_reference`/`POSE_REFERENCE` consistently), and `CharacterCreator.create(manifest, image, character, direction, character_positive, character_negative)` are used identically wherever referenced across tasks.

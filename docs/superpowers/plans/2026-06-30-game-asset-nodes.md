# Game-Asset Nodes + Correctness Overhaul Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the 12 verified review findings and add transparency, sprite-sheet/atlas export, and FFLF-native production nodes so comfyui-andypack produces engine-ready video-game character assets.

**Architecture:** Keep `resolve.py`/`manifest.py`/`io.py` pure (no torch). Alpha and all sprite tensor ops live in `images.py` + a new `andypack/sprites.py`; engine-format serializers in a new pure-stdlib `andypack/atlas.py`. New nodes are thin wrappers in `nodes.py` over those layers, following the existing selector/writer patterns (disk-mtime `IS_CHANGED`, payload-first/sidecar-last atomic writes). New socket types are private-key dict bundles like `ANIM_POSE`.

**Tech Stack:** Python 3.10–3.12, PyTorch (CPU in CI), Pillow, numpy, aiohttp (ComfyUI), pytest, ruff, mypy. Vanilla-DOM ComfyUI web extensions (no build step).

## Global Constraints

- `resolve.py` and `manifest.py` and `io.py` MUST NOT import torch/ComfyUI (a test import-guards `resolve.py`).
- Every disk-reading node MUST define `IS_CHANGED` that fingerprints the disk state it reads (mtime), per the `_selector_fingerprint` pattern in `nodes.py:42`.
- Writers MUST drop the completion sidecar/meta FIRST, write payload, then write the sidecar/meta LAST via `io.atomic_write_json` (temp-file + atomic rename). `clear_frames` before re-writing animation frames.
- HTTP routes return JSON only, never file bytes, and take NO client filesystem path: a manifest is a bare basename validated by `api.manifest_name_is_safe`; a character is a name validated by `api._is_safe_segment` (or snake-cased) resolved under the server's own dirs.
- The loop flag is DERIVED (`start_image == end_image`), never authored — no manifest `loop` field.
- FFLF cross-wiring: `start_from` consumes the dep's LAST frame; `end_at` consumes the dep's FIRST frame. Never invert.
- ComfyUI IMAGE tensors are float32 `[B, H, W, C]` in `[0,1]`. RGB = 3-ch; RGBA = 4-ch. `MASK` is `[B, H, W]` in `[0,1]`.
- Commands: test `pytest -q`; lint `ruff check .`; types `mypy andypack`. All three MUST stay green at every commit.
- Commit message footer for every commit:
  ```
  Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01XLyJYgUZjZ5WtWVtB4reWi
  ```
  (Footer omitted from the per-task `git commit` snippets below for brevity — add it to every commit.)
- Work happens on branch `feat/game-asset-nodes` (already created).

## File Structure

**New files:**
- `andypack/sprites.py` — pure tensor/PIL sprite ops: alpha bbox, trim (union/per-frame), pivot, sprite-sheet packing, palette quantize/extract. Imports torch/numpy/PIL. NOT pure-core.
- `andypack/atlas.py` — pure-stdlib engine-format serializers (atlas JSON, Aseprite, Godot SpriteFrames, Unity meta, TexturePacker, CSS). No torch.
- `tests/test_alpha.py`, `tests/test_sprites.py`, `tests/test_atlas.py`, `tests/test_palette.py` — new unit tests.
- `tests/test_fixes.py` — regression tests for the 12 findings (grouped).
- `examples/workflows/*.json` — example graphs (final task).

**Modified files:**
- `andypack/resolve.py` — staleness dep key-set (#1), effective-cache keying (#2), playback loop gate (#8); new read helpers for atlas/state-machine/turnaround.
- `andypack/manifest.py` — id/direction path-segment validation (#7), gen-param presence/positivity validation (#6/#12).
- `andypack/api.py` — route path-safety (#4/#5), `next_actionable` category predicate + `skip_mirrored`, `state_machine`, thumbnail path resolution.
- `andypack/server.py` — route gating (#4/#5), thumbnail route.
- `andypack/io.py` — `has_alpha` in sidecar/meta builders.
- `andypack/images.py` — alpha at the disk boundary, playback modes, timing/recolor helpers, thumbnail.
- `andypack/nodes.py` — fix MirrorFrameWriter IS_CHANGED (#3) + batch mode, empty-sentinel guard (#10), fps clamp (#9), all new node classes + mappings.
- `web/anim_coord.js` — character-combo refresh (#11).
- `web/anim_coord_panel.js` — coverage thumbnails, characters tab, space-safe keys + mirror tally.
- `CLAUDE.md`, `README.md`, `docs/prompting-guide.md` — node count, new nodes/types/categories, moved invariants.

---

## Group 1 — Correctness & security fixes

### Task 1: Anchor-swap staleness (#1)

A complete animation must re-stale when an anchor `ref` is swapped or `end_at` is added/removed. Compare the current dep key-set against the recorded `sources` keys.

**Files:**
- Modify: `andypack/resolve.py` (`_outdated` ~513-544, `stale_locally` ~487-510)
- Test: `tests/test_fixes.py`

**Interfaces:**
- Produces: behavior change only — `outdated(...)` / `stale_locally(...)` now return True when the recorded `sources` key-set differs from the current `recorded_sources(...)` key-set.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_fixes.py
import json, os
from andypack import resolve
from tests.conftest import write_manifest  # existing helper; if absent, build inline

def _render_pose(root, char, pose, direction, manifest):
    # minimal: write a pose png + sidecar so node_complete() is True
    base = os.path.join(root, char, f"_{pose}")
    os.makedirs(base, exist_ok=True)
    open(os.path.join(base, f"{direction}.png"), "wb").close()
    r = resolve.resolve_pose(manifest, root, char, pose, direction)
    from andypack import io
    io.atomic_write_json(os.path.join(base, f"{direction}.json"),
                         io.build_pose_sidecar(r["meta"], created_utc="2026-01-01T00:00:00Z"))

def test_swapping_anchor_ref_restales_animation(tmp_path):
    root = str(tmp_path)
    char = "hero"
    manifest = {
        "version": 1,
        "poses": {
            "base": {"directions": {"EAST": {}}},
            "poseA": {"from": {"ref": "base"}, "directions": {"EAST": {}}},
            "poseB": {"from": {"ref": "base"}, "directions": {"EAST": {}}},
        },
        "animations": {
            "walk": {"start_from": {"ref": "poseA"}, "directions": {"EAST": {}},
                     "length": 5, "fps": 8, "width": 16, "height": 16},
        },
        "defaults": {},
    }
    for p in ("base", "poseA", "poseB"):
        _render_pose(root, char, p, "EAST", manifest)
    # render walk against poseA
    rwalk = resolve.resolve_animation(manifest, root, char, "walk", "EAST")
    from andypack import io
    adir = os.path.join(root, char, "walk", "EAST")
    os.makedirs(adir, exist_ok=True)
    for i in range(5):
        open(os.path.join(adir, io.frame_name(i)), "wb").close()
    io.atomic_write_json(os.path.join(adir, "meta.json"),
        io.build_animation_meta(rwalk["meta"], count=5, start_frame=io.frame_name(0),
                                last_frame=io.frame_name(4), seed=0, created_utc="2026-01-01T00:00:00Z"))
    assert resolve.outdated(manifest, root, char, "walk", "EAST") is False
    # swap the anchor to poseB (already rendered, prompts unchanged)
    manifest["animations"]["walk"]["start_from"] = {"ref": "poseB"}
    assert resolve.outdated(manifest, root, char, "walk", "EAST") is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_fixes.py::test_swapping_anchor_ref_restales_animation -v`
Expected: FAIL (asserts True but gets False on the swap).

- [ ] **Step 3: Implement the dep key-set check**

In `resolve.py`, add a helper and call it from both `_outdated` and `stale_locally`:

```python
def _sources_drifted(manifest, root, character, ref, direction, meta) -> bool:
    """True if the recorded `sources` key-set differs from the current dep set
    (an anchor ref was swapped / added / removed), OR a recorded source's
    render_id drifted (re-rendered). Catches anchor identity changes the
    prompt_hash can't see."""
    recorded = (meta or {}).get("sources")
    if not isinstance(recorded, dict):
        return False  # pre-provenance meta: transitive walk still covers deps
    current = recorded_sources(manifest, root, character, ref, direction)
    if set(recorded.keys()) != set(current.keys()):
        return True
    for key, rid in recorded.items():
        if "@" not in key:
            continue
        dep_ref, ddir = key.rsplit("@", 1)
        if read_render_id(manifest, root, character, dep_ref, ddir) != rid:
            return True
    return False
```

Replace the existing `sources` loop in `_outdated` (lines ~525-535) with `if _sources_drifted(manifest, root, character, ref, direction, meta): return True`, and the analogous loop in `stale_locally` (lines ~502-509) likewise. Keep the prompt-hash check and the transitive walk that follow.

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_fixes.py::test_swapping_anchor_ref_restales_animation tests/test_staleness.py tests/test_provenance.py -q`
Expected: PASS (new test green; no staleness/provenance regressions).

- [ ] **Step 5: Commit**

```bash
git add andypack/resolve.py tests/test_fixes.py
git commit -m "fix: re-stale animation when an anchor ref is swapped or end_at changes (#1)"
```

---

### Task 2: Effective-manifest cache keying (#2)

`_EFFECTIVE_CACHE` keyed on `id(manifest)` can serve a stale merged manifest after a base-manifest edit. Key on content (version + a structural hash) + identity mtime.

**Files:**
- Modify: `andypack/resolve.py` (`effective_manifest` ~120-155, cache decl ~78)
- Test: `tests/test_fixes.py`

**Interfaces:**
- Produces: `effective_manifest` returns the freshly-merged manifest after the base manifest content changes, even when a new dict reuses a freed `id()`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_fixes.py (append)
def test_effective_manifest_reflects_base_edit_with_overlay(tmp_path):
    root = str(tmp_path); char = "hero"
    os.makedirs(os.path.join(root, char), exist_ok=True)
    # character overlay so effective_manifest does the merge+cache path
    from andypack import io
    io.atomic_write_json(os.path.join(root, char, "character.json"),
        {"poses": {"wave": {"from": {"ref": "base"}, "directions": {"EAST": {}}}}})
    resolve.invalidate_character(root, char)
    base1 = {"version": 1, "poses": {"base": {"directions": {"EAST": {}}}},
             "animations": {}, "defaults": {},
             "globals": {"pose": {"positive_prompt": "v1"}}}
    eff1 = resolve.effective_manifest(base1, root, char)
    assert eff1["globals"]["pose"]["positive_prompt"] == "v1"
    # a DIFFERENT base object (simulating a reload) with edited content
    base2 = {"version": 1, "poses": {"base": {"directions": {"EAST": {}}}},
             "animations": {}, "defaults": {},
             "globals": {"pose": {"positive_prompt": "v2"}}}
    eff2 = resolve.effective_manifest(base2, root, char)
    assert eff2["globals"]["pose"]["positive_prompt"] == "v2"
```

Note: this passes today *unless* `id(base2) == id(base1)`. Make the test deterministic by also asserting the cache key is content-derived:

```python
def test_effective_cache_key_is_content_derived():
    from andypack.resolve import _effective_cache_key
    a = {"version": 1, "poses": {}, "animations": {}, "globals": {"x": 1}}
    b = {"version": 1, "poses": {}, "animations": {}, "globals": {"x": 2}}
    assert _effective_cache_key(a, {}) != _effective_cache_key(b, {})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_fixes.py::test_effective_cache_key_is_content_derived -v`
Expected: FAIL ("cannot import name '_effective_cache_key'").

- [ ] **Step 3: Implement content-derived keying**

In `resolve.py`:

```python
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
```

Change `_EFFECTIVE_CACHE` to `dict[tuple, Manifest]` and replace `key = (id(manifest), id(identity))` with `key = _effective_cache_key(manifest, identity)`. `invalidate_character` still `.clear()`s the whole cache.

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_fixes.py -k effective -q && pytest tests/test_resolve.py tests/test_acceptance.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add andypack/resolve.py tests/test_fixes.py
git commit -m "fix: key effective-manifest cache on content, not id() (#2)"
```

---

### Task 3: MirrorFrameWriter IS_CHANGED (#3)

Add a disk-mtime `IS_CHANGED` so re-rendering the source direction re-fires the mirror.

**Files:**
- Modify: `andypack/nodes.py` (`MirrorFrameWriter` ~859)
- Test: `tests/test_nodes.py` or `tests/test_fixes.py`

**Interfaces:**
- Consumes: `resolve.pose_image_path`, `resolve.animation_frame_dir`, `_mtime` (nodes.py), `manifest.get("mirror_map")`.
- Produces: `MirrorFrameWriter.IS_CHANGED(manifest, character, kind, id, direction)` returning a string fingerprint that changes with the source mtime.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_fixes.py (append)
from andypack import nodes
def test_mirror_writer_is_changed_tracks_source_mtime(tmp_path, monkeypatch):
    root = str(tmp_path); char = "hero"
    manifest = {"version": 1, "mirror_map": {"WEST": "EAST"},
                "poses": {"p": {"from": {"ref": "base"}, "directions": {"EAST": {}, "WEST": {}}},
                          "base": {"directions": {"EAST": {}}}},
                "animations": {}, "defaults": {}}
    monkeypatch.setattr(nodes, "_characters_root", lambda: root)
    src = resolve.pose_image_path(root, char, "p", "EAST")
    os.makedirs(os.path.dirname(src), exist_ok=True)
    with open(src, "wb") as fh: fh.write(b"a")
    fp1 = nodes.MirrorFrameWriter.IS_CHANGED(manifest, char, "pose", "p", "WEST")
    os.utime(src, (10**9, 10**9))  # bump mtime
    fp2 = nodes.MirrorFrameWriter.IS_CHANGED(manifest, char, "pose", "p", "WEST")
    assert fp1 != fp2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_fixes.py::test_mirror_writer_is_changed_tracks_source_mtime -v`
Expected: FAIL ("type object 'MirrorFrameWriter' has no attribute 'IS_CHANGED'").

- [ ] **Step 3: Implement IS_CHANGED**

Add to `MirrorFrameWriter`:

```python
    @classmethod
    def IS_CHANGED(cls, manifest, character, kind, id, direction):
        if character in ("", _NO_CHARACTER) or not id or not direction:
            return float("nan")
        root = _characters_root()
        try:
            eff = effective_manifest(manifest, root, character)
            src_dir = (eff.get("mirror_map") or {}).get(direction)
            if not src_dir:
                return float("nan")
            if kind == "pose":
                src = resolve.pose_image_path(root, character, id, src_dir)
                return f"pose:{src}:{_mtime(src)}"
            d = resolve.animation_frame_dir(root, character, id, src_dir)
            meta = resolve.animation_meta_path(root, character, id, src_dir)
            return f"anim:{d}:{_mtime(meta)}"
        except Exception:
            return float("nan")
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_fixes.py::test_mirror_writer_is_changed_tracks_source_mtime tests/test_nodes.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add andypack/nodes.py tests/test_fixes.py
git commit -m "fix: MirrorFrameWriter IS_CHANGED tracks source mtime so it re-mirrors (#3)"
```

---

### Task 4: Route path-traversal gating (#4/#5)

Gate the `manifest` and `character` GET params; drop the client `character_dir` path.

**Files:**
- Modify: `andypack/server.py` (`_manifest_from_request` ~19, `_root_and_char` ~53)
- Modify: `andypack/api.py` (add a safe resolver)
- Test: `tests/test_server.py` / `tests/test_fixes.py`

**Interfaces:**
- Produces: `api.safe_manifest_path(name) -> Optional[str]` (None for unsafe/traversing names); `_root_and_char` resolves the character only by validated bare name under `characters_dir()`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_fixes.py (append)
from andypack import api
def test_safe_manifest_path_rejects_traversal():
    assert api.safe_manifest_path("../../etc/passwd.json") is None
    assert api.safe_manifest_path("/etc/passwd.json") is None
    assert api.safe_manifest_path("default.json") is not None  # bare name ok
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_fixes.py::test_safe_manifest_path_rejects_traversal -v`
Expected: FAIL ("module 'andypack.api' has no attribute 'safe_manifest_path'").

- [ ] **Step 3: Implement**

In `api.py`:

```python
def safe_manifest_path(name: str) -> Optional[str]:
    """Resolve a manifest by BARE NAME under the manifests dir, rejecting any
    name that is unsafe or traverses (the HTTP attack surface — unlike
    resolve_manifest_path, which allows absolute paths for trusted node inputs)."""
    if not manifest_name_is_safe(name):
        return None
    base = manifests_dir()
    return None if base is None else os.path.join(base, name)
```

In `server.py` `_manifest_from_request`, replace `api.resolve_manifest_path(name)` with a gated lookup:

```python
    name = request.query.get("manifest") or "default.json"
    path = api.safe_manifest_path(name)
    if path is None:
        raise web.HTTPBadRequest(
            text=json.dumps({"error": f"unsafe manifest name {name!r}"}),
            content_type="application/json")
    try:
        return load_manifest(path)
    except (OSError, ManifestError) as exc:
        raise web.HTTPBadRequest(
            text=json.dumps({"error": str(exc)}), content_type="application/json") from exc
```

In `server.py` `_root_and_char`, ignore `character_dir` from the client and resolve by bare name only:

```python
    def _root_and_char(request):
        root = api.characters_dir() or ""
        name = request.query.get("character", "")
        if not api._is_safe_segment(name):
            return (root, "")  # unsafe name -> empty character (read-only options degrade)
        return (root, name)
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_fixes.py -k "safe_manifest" tests/test_server.py tests/test_api.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add andypack/api.py andypack/server.py tests/test_fixes.py
git commit -m "fix: gate manifest/character GET params against path traversal (#4,#5)"
```

---

### Task 5: Manifest id/direction path-segment validation (#7)

Reject pose ids, animation ids, and direction names that aren't safe single path segments.

**Files:**
- Modify: `andypack/manifest.py` (`_validate_directions` ~25, `_validate_refs` ~77)
- Test: `tests/test_manifest.py` / `tests/test_fixes.py`

**Interfaces:**
- Produces: `validate_manifest` raises `ManifestError` for an id/direction containing `/`, `\`, or `..`, or equal to `.`/`..`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_fixes.py (append)
import pytest
from andypack.manifest import validate_manifest, ManifestError
def test_unsafe_entity_id_rejected():
    m = {"version": 1, "animations": {"../escape": {"start_from": {"ref": "base"},
         "directions": {"EAST": {}}}},
         "poses": {"base": {"directions": {"EAST": {}}}}, "defaults": {}}
    with pytest.raises(ManifestError):
        validate_manifest(m)
def test_unsafe_direction_name_rejected():
    m = {"version": 1, "poses": {"base": {"directions": {"../x": {}}}},
         "animations": {}, "defaults": {}}
    with pytest.raises(ManifestError):
        validate_manifest(m)
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_fixes.py -k "unsafe_entity or unsafe_direction" -v`
Expected: FAIL (no raise).

- [ ] **Step 3: Implement**

In `manifest.py`:

```python
import os

def _is_safe_segment(name: str) -> bool:
    return bool(name) and name not in (".", "..") and not os.path.isabs(name) \
        and "/" not in name and "\\" not in name and ".." not in name
```

In `_validate_directions`, after confirming `directions` is a dict, validate each name:

```python
    for dname, dlayer in directions.items():
        if not _is_safe_segment(dname):
            raise ManifestError(f"{label} direction name {dname!r} is unsafe (path segment)")
        if not isinstance(dlayer, dict):
            raise ManifestError(...)  # unchanged
```

In `_validate_refs`, at the top of the pose and animation loops, validate ids:

```python
    for pid, pose in manifest.get("poses", {}).items():
        if not _is_safe_segment(pid):
            raise ManifestError(f"pose id {pid!r} is unsafe (path segment)")
        ...
    for aid, anim in manifest.get("animations", {}).items():
        if not _is_safe_segment(aid):
            raise ManifestError(f"animation id {aid!r} is unsafe (path segment)")
        ...
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_fixes.py -k unsafe tests/test_manifest.py tests/test_seed.py -q`
Expected: PASS (seed manifest uses safe ids/directions).

- [ ] **Step 5: Commit**

```bash
git add andypack/manifest.py tests/test_fixes.py
git commit -m "fix: validate entity ids and direction names as safe path segments (#7)"
```

---

### Task 6: Gen-param presence + positivity validation (#6/#9/#12)

Require each animation's effective `width/height/length/fps` to resolve to a positive int; clamp the wireable `fps` output ≥1.

**Files:**
- Modify: `andypack/manifest.py` (`_validate_refs` / new `_validate_animation_gen_params`)
- Modify: `andypack/nodes.py` (`_build_animation_bundle` ~216 fps clamp)
- Test: `tests/test_fixes.py`

**Interfaces:**
- Produces: `validate_manifest` raises when an animation's effective `width/height/length/fps` is missing or ≤0; `_build_animation_bundle` emits `fps >= 1`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_fixes.py (append)
def test_missing_gen_params_rejected():
    m = {"version": 1, "poses": {"base": {"directions": {"EAST": {}}}},
         "animations": {"walk": {"start_from": {"ref": "base"}, "directions": {"EAST": {}}}},
         "defaults": {}}  # no width/height/length/fps anywhere
    with pytest.raises(ManifestError):
        validate_manifest(m)
def test_nonpositive_gen_params_rejected():
    m = {"version": 1, "poses": {"base": {"directions": {"EAST": {}}}},
         "animations": {"walk": {"start_from": {"ref": "base"}, "directions": {"EAST": {}},
            "length": -3, "fps": 8, "width": 16, "height": 16}},
         "defaults": {}}
    with pytest.raises(ManifestError):
        validate_manifest(m)
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_fixes.py -k gen_params -v`
Expected: FAIL (no raise).

- [ ] **Step 3: Implement**

In `manifest.py`, add and call from `_validate_refs` inside the animation loop:

```python
def _require_positive_gen_params(label, anim, defaults):
    for field in ("length", "fps", "width", "height"):
        val = anim.get(field, defaults.get(field))
        if val is None:
            raise ManifestError(f"{label} has no resolvable {field!r} (set it on the "
                                f"animation or in defaults)")
        if not isinstance(val, int) or val <= 0:
            raise ManifestError(f"{label} {field!r} must be a positive integer, got {val!r}")
```

Call `_require_positive_gen_params(f"animation {aid!r}", anim, manifest.get("defaults", {}))` after the existing `_validate_gen_params` call in the animation loop.

In `nodes.py` `_build_animation_bundle`, change the fps emission to clamp:

```python
        "fps": max(as_int("fps"), 1),
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_fixes.py -k gen_params tests/test_manifest.py tests/test_nodes.py tests/test_seed.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add andypack/manifest.py andypack/nodes.py tests/test_fixes.py
git commit -m "fix: require positive resolvable gen-params; clamp fps output >=1 (#6,#9,#12)"
```

---

### Task 7: Playback loopable gate (#8)

Gate `loopable` on resolved `start_image == end_image`, not just same ref.

**Files:**
- Modify: `andypack/resolve.py` (`playback_segments` ~676-680)
- Test: `tests/test_playback.py` / `tests/test_fixes.py`

**Interfaces:**
- Produces: `playback_segments` sets the action `repeat>1` only when the resolved start and end anchor images are the same path.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_fixes.py (append)
def test_playback_not_loopable_when_anchor_frames_differ(tmp_path):
    root = str(tmp_path); char = "hero"
    manifest = {"version": 1,
        "poses": {"base": {"directions": {"EAST": {}}}},
        "animations": {
            "src": {"start_from": {"ref": "base"}, "directions": {"EAST": {}},
                    "length": 4, "fps": 8, "width": 16, "height": 16},
            "clip": {"start_from": {"ref": "src"}, "end_at": {"ref": "src"},
                     "directions": {"EAST": {}}, "length": 4, "fps": 8, "width": 16, "height": 16}},
        "defaults": {}}
    # render src (4 distinct frames so start_frame != last_frame) and clip
    from andypack import io
    for anim, n in (("src", 4), ("clip", 4)):
        d = os.path.join(root, char, anim, "EAST"); os.makedirs(d, exist_ok=True)
        for i in range(n): open(os.path.join(d, io.frame_name(i)), "wb").close()
        r = resolve.resolve_animation(manifest, root, char, anim, "EAST")
        io.atomic_write_json(os.path.join(d, "meta.json"),
            io.build_animation_meta(r["meta"], count=n, start_frame=io.frame_name(0),
                last_frame=io.frame_name(n-1), seed=0, created_utc="2026-01-01T00:00:00Z"))
    segs = resolve.playback_segments(manifest, root, char, "clip", "EAST", loops=3, fps=8)
    action = [s for s in segs if s.get("dir", "").endswith(os.path.join("clip", "EAST"))][0]
    assert action["repeat"] == 1  # start_image (src.last) != end_image (src.first)
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_fixes.py::test_playback_not_loopable_when_anchor_frames_differ -v`
Expected: FAIL (repeat == 3).

- [ ] **Step 3: Implement**

In `playback_segments`, replace the `loopable` computation with a resolved-image check:

```python
    start_img = start_anchor(manifest, root, character, anim_id, direction)
    end_img = end_anchor(manifest, root, character, anim_id, direction)
    loopable = start_img is not None and start_img == end_img
```

(Remove the old `start["ref"] == end["ref"]` ref-based predicate.)

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_fixes.py -k loopable tests/test_playback.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add andypack/resolve.py tests/test_fixes.py
git commit -m "fix: gate playback loop on equal resolved anchor images (#8)"
```

---

### Task 8: AnimationFrameWriter empty-sentinel guard (#10)

Reject the 1×1 `empty_image()` sentinel, not just an empty batch.

**Files:**
- Modify: `andypack/nodes.py` (`AnimationFrameWriter.write` ~448)
- Test: `tests/test_animation_writeback.py` / `tests/test_fixes.py`

**Interfaces:**
- Produces: `AnimationFrameWriter.write` raises `RuntimeError` when `images.is_empty(frames)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_fixes.py (append)
from andypack import images
def test_animation_writer_rejects_empty_sentinel(tmp_path, monkeypatch):
    monkeypatch.setattr(nodes, "_characters_root", lambda: str(tmp_path))
    w = nodes.AnimationFrameWriter()
    anim = {"output_dir": str(tmp_path / "a"),
            "_meta": {"prompt_hash": "sha1:x", "loop": False}}
    with pytest.raises(RuntimeError):
        w.write(anim, images.empty_image())
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_fixes.py::test_animation_writer_rejects_empty_sentinel -v`
Expected: FAIL (writes a 1-frame clip, no raise).

- [ ] **Step 3: Implement**

In `AnimationFrameWriter.write`, replace the `frames.shape[0] == 0` guard with:

```python
        if images.is_empty(frames):
            raise RuntimeError(
                "AnimationFrameWriter: received an empty or 1x1 sentinel frame batch; "
                "nothing to write (check the upstream sampler)")
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_fixes.py::test_animation_writer_rejects_empty_sentinel tests/test_animation_writeback.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add andypack/nodes.py tests/test_fixes.py
git commit -m "fix: reject 1x1 empty-image sentinel in AnimationFrameWriter (#10)"
```

---

### Task 9: Character-combo refresh (#11)

Repopulate the character combo from `/characters` in the web extension; document the refresh.

**Files:**
- Modify: `web/anim_coord.js`
- Modify: `CLAUDE.md` (web section note)
- Test: manual (frontend) — documented in the task.

**Interfaces:**
- Produces: character widgets are repopulated from `GET /anim_coord/characters` on node creation / panel refresh.

- [ ] **Step 1: Add the refresh logic**

In `web/anim_coord.js`, where selector widgets are wired, fetch `/anim_coord/characters` and, for any widget named `character` whose `options.values` is a combo, replace its values with `["(select character)", ...names]` (preserving the current value if still present). Hook it into the existing `refreshAll`/`wire` path so it runs on node add and on the panel's refresh button.

```javascript
async function refreshCharacterCombos(node) {
  const res = await fetch("/anim_coord/characters").then(r => r.json()).catch(() => []);
  const names = (res || []).map(c => c.name);
  const w = node.widgets?.find(w => w.name === "character" && Array.isArray(w.options?.values));
  if (!w) return;
  const prev = w.value;
  w.options.values = ["(select character)", ...names];
  if (!w.options.values.includes(prev)) w.value = w.options.values[0];
}
```

- [ ] **Step 2: Manual verification**

Run ComfyUI, create a character via Character Creator, then add a Character Pose Selector and click the panel refresh: the new character appears in the dropdown without a page reload. Record the result in the commit body.

- [ ] **Step 3: Document**

In `CLAUDE.md` web section, add: "The character combo is repopulated from `/anim_coord/characters` on node add / panel refresh; characters created mid-session appear after a refresh (no full reload)."

- [ ] **Step 4: Commit**

```bash
git add web/anim_coord.js CLAUDE.md
git commit -m "feat: repopulate character combos from /characters on refresh (#11)"
```

---

## Group 2 — Alpha boundary

### Task 10: Alpha-aware `images.py` disk boundary

`save_image_png` writes RGBA when given a mask or a 4-ch image; `load_image_tensor` can keep alpha. RGB path unchanged.

**Files:**
- Modify: `andypack/images.py`
- Test: `tests/test_alpha.py`

**Interfaces:**
- Produces:
  - `to_rgba(image: Tensor, mask: Optional[Tensor] = None) -> Tensor` — `[B,H,W,4]`. If `mask` given, it is the alpha (resized to H×W if needed); else if `image` is 4-ch, returned as-is; else alpha=1.
  - `save_image_png(image, path, mask=None)` — writes RGBA iff `mask` is given or `image` is 4-ch; else RGB.
  - `load_image_tensor(path, keep_alpha=False) -> Tensor` — 4-ch when `keep_alpha` and the PNG has alpha; else 3-ch (white-matte, unchanged default).
  - `alpha_bbox(image: Tensor, threshold: float = 0.03) -> tuple[int,int,int,int] | None` — `(left, top, right, bottom)` over alpha≥threshold (single image or batch union when called per-frame), None if fully transparent.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_alpha.py
import torch, numpy as np
from PIL import Image
from andypack import images

def test_save_png_with_mask_writes_rgba(tmp_path):
    img = torch.ones((1, 4, 4, 3))            # white RGB
    mask = torch.zeros((1, 4, 4)); mask[:, :2, :2] = 1.0
    p = str(tmp_path / "s.png")
    images.save_image_png(img, p, mask=mask)
    with Image.open(p) as im:
        assert im.mode == "RGBA"
        assert im.getpixel((0, 0))[3] == 255 and im.getpixel((3, 3))[3] == 0

def test_save_png_with_rgba_input_preserves_alpha(tmp_path):
    rgba = torch.ones((1, 2, 2, 4)); rgba[..., 3] = 0.0
    p = str(tmp_path / "r.png")
    images.save_image_png(rgba, p)
    with Image.open(p) as im:
        assert im.mode == "RGBA" and im.getpixel((0, 0))[3] == 0

def test_save_png_rgb_unchanged(tmp_path):
    p = str(tmp_path / "rgb.png")
    images.save_image_png(torch.ones((1, 2, 2, 3)), p)
    with Image.open(p) as im:
        assert im.mode == "RGB"

def test_load_keep_alpha_roundtrip(tmp_path):
    rgba = torch.ones((1, 2, 2, 4)); rgba[..., 3] = 0.0
    p = str(tmp_path / "r.png")
    images.save_image_png(rgba, p)
    t = images.load_image_tensor(p, keep_alpha=True)
    assert t.shape[-1] == 4 and float(t[0, 0, 0, 3]) == 0.0
    assert images.load_image_tensor(p).shape[-1] == 3  # default still 3-ch
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_alpha.py -v`
Expected: FAIL (mask/keep_alpha kwargs not supported; RGBA not written).

- [ ] **Step 3: Implement**

In `images.py` add `to_rgba`, `alpha_bbox`, and rewrite `save_image_png` / `load_image_tensor`:

```python
def _alpha_from_mask(mask: torch.Tensor, h: int, w: int) -> torch.Tensor:
    m = mask if mask.dim() == 3 else mask.unsqueeze(0)   # [B,H,W]
    if m.shape[1] != h or m.shape[2] != w:
        m = torch.nn.functional.interpolate(m.unsqueeze(1), size=(h, w),
            mode="bilinear", align_corners=False).squeeze(1)
    return m[0].clamp(0.0, 1.0)

def to_rgba(image: torch.Tensor, mask=None) -> torch.Tensor:
    frame = image[0] if image.dim() == 4 else image     # [H,W,C]
    h, w = int(frame.shape[0]), int(frame.shape[1])
    rgb = frame[..., :3]
    if mask is not None:
        a = _alpha_from_mask(mask, h, w).unsqueeze(-1)
    elif frame.shape[-1] == 4:
        a = frame[..., 3:4]
    else:
        a = torch.ones((h, w, 1), dtype=frame.dtype)
    return torch.cat([rgb, a], dim=-1)

def save_image_png(image: torch.Tensor, path: str, mask=None) -> None:
    frame = image[0] if image.dim() == 4 else image
    has_alpha = mask is not None or frame.shape[-1] == 4
    if has_alpha:
        rgba = to_rgba(image, mask)
        arr = (rgba.clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)
        mode = "RGBA"
    else:
        arr = (frame[..., :3].clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)
        mode = "RGB"
    # ...existing atomic temp-write, with Image.fromarray(arr, mode=mode)

def alpha_bbox(image: torch.Tensor, threshold: float = 0.03):
    frame = image[0] if image.dim() == 4 else image
    if frame.shape[-1] != 4:
        return (0, 0, int(frame.shape[1]), int(frame.shape[0]))
    a = frame[..., 3]
    ys, xs = torch.where(a >= threshold)
    if ys.numel() == 0:
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
```

In `_flatten_to_rgb`, add a sibling `_to_rgba_pil` and branch `load_image_tensor` on `keep_alpha` (return `np.asarray(img.convert("RGBA"))/255` as a 4-ch tensor).

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_alpha.py tests/test_images.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add andypack/images.py tests/test_alpha.py
git commit -m "feat: alpha at the disk boundary (RGBA write/read, mask compositing)"
```

---

### Task 11: `has_alpha` in sidecar/meta builders

**Files:**
- Modify: `andypack/io.py` (`build_pose_sidecar` ~103, `build_animation_meta` ~112)
- Test: `tests/test_io.py`

**Interfaces:**
- Produces: `build_pose_sidecar(meta, created_utc, has_alpha=False)` and `build_animation_meta(..., has_alpha=False)` add `"has_alpha": bool` to the output.

- [ ] **Step 1: Write failing test**

```python
# tests/test_io.py (append)
from andypack import io
def test_sidecar_records_has_alpha():
    meta = {"prompt_hash": "sha1:x", "direction": "EAST"}
    s = io.build_pose_sidecar(meta, created_utc="2026-01-01T00:00:00Z", has_alpha=True)
    assert s["has_alpha"] is True
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_io.py::test_sidecar_records_has_alpha -v`
Expected: FAIL (unexpected kwarg).

- [ ] **Step 3: Implement**

Add `has_alpha: bool = False` param to both builders and include `"has_alpha": has_alpha` in the returned dict.

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_io.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add andypack/io.py tests/test_io.py
git commit -m "feat: record has_alpha in pose sidecar / animation meta"
```

---

### Task 12: Writer `mask` inputs + RGBA threading

**Files:**
- Modify: `andypack/nodes.py` (`PoseFrameWriter` ~330, `AnimationFrameWriter` ~416)
- Test: `tests/test_pose_writeback.py`, `tests/test_animation_writeback.py`

**Interfaces:**
- Consumes: `images.save_image_png(image, path, mask=...)`, `io.build_*_sidecar(..., has_alpha=...)`.
- Produces: both writers accept optional `mask` (MASK); a 4-ch image or a mask yields an RGBA payload and `has_alpha=True` in the sidecar/meta.

- [ ] **Step 1: Write failing test**

```python
# tests/test_pose_writeback.py (append)
import torch, json, os
from andypack import nodes, images
from PIL import Image
def test_pose_writer_writes_rgba_with_mask(tmp_path):
    pose = {"output_dir": str(tmp_path),
            "_meta": {"image": "EAST.png", "direction": "EAST", "prompt_hash": "sha1:x"}}
    mask = torch.zeros((1, 4, 4)); mask[:, :2, :2] = 1.0
    nodes.PoseFrameWriter().write(pose, torch.ones((1, 4, 4, 3)), mask=mask)
    with Image.open(os.path.join(str(tmp_path), "EAST.png")) as im:
        assert im.mode == "RGBA"
    sc = json.load(open(os.path.join(str(tmp_path), "EAST.json")))
    assert sc["has_alpha"] is True
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_pose_writeback.py::test_pose_writer_writes_rgba_with_mask -v`
Expected: FAIL (no `mask` param).

- [ ] **Step 3: Implement**

Add to both writers' `INPUT_TYPES` an `"optional": {"mask": ("MASK",)}`. Thread `mask` through `write(self, pose, image, mask=None)` and `write(self, animation, frames, seed=0, mask=None)`. Compute `has_alpha = mask is not None or int(image.shape[-1]) == 4`; pass `mask=mask` to `save_image_png` (per-frame slice `mask[i:i+1]` for animations) and `has_alpha=has_alpha` to the sidecar/meta builder.

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_pose_writeback.py tests/test_animation_writeback.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add andypack/nodes.py tests/test_pose_writeback.py tests/test_animation_writeback.py
git commit -m "feat: optional mask input on Pose/Animation Frame Writers -> RGBA sprites"
```

---

## Group 3 — Sprite export chain

### Task 13: `sprites.py` trim + pivot core

**Files:**
- Create: `andypack/sprites.py`
- Test: `tests/test_sprites.py`

**Interfaces:**
- Produces:
  - `trim_batch(image, threshold=0.03, mode="union", pad=0) -> tuple[Tensor, list[dict]]` — returns the cropped batch (uniform in `union` mode) and per-frame `{"source_size": [w,h], "offset": [x,y]}`.
  - `pivot_point(w, h, kind, custom=(0.5, 1.0)) -> tuple[int,int]` — pixel pivot; `kind` in `center|bottom_center|top_center|custom`.
  - `SPRITE_TRIM` bundle shape: `{"frames": [{"source_size","offset","pivot"}], "trim_mode", "pivot_kind"}`.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_sprites.py
import torch
from andypack import sprites
def _frame(w, h, box):  # box = (l,t,r,b) opaque
    img = torch.zeros((h, w, 4)); l,t,r,b = box
    img[t:b, l:r, :3] = 1.0; img[t:b, l:r, 3] = 1.0
    return img.unsqueeze(0)
def test_trim_union_crops_to_shared_bbox():
    batch = torch.cat([_frame(8, 8, (1,1,4,4)), _frame(8, 8, (3,3,6,6))], dim=0)
    out, rects = sprites.trim_batch(batch, mode="union")
    assert out.shape[1] == 5 and out.shape[2] == 5      # union bbox (1..6)x(1..6)
    assert len(rects) == 2 and rects[0]["offset"] == [1, 1]
def test_pivot_bottom_center():
    assert sprites.pivot_point(10, 20, "bottom_center") == (5, 20)
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_sprites.py -v`
Expected: FAIL (module missing).

- [ ] **Step 3: Implement `andypack/sprites.py`**

Write `trim_batch` (per-frame `images.alpha_bbox`, union = min/max over frames, crop each frame to the union box padded by `pad`; `per_frame` crops each to its own box and records offsets) and `pivot_point`. Pure torch; import `from andypack import images` for `alpha_bbox`.

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_sprites.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add andypack/sprites.py tests/test_sprites.py
git commit -m "feat: sprites.trim_batch + pivot_point (alpha trim, shared-bbox registration)"
```

---

### Task 14: Sprite Trim & Pivot node

**Files:**
- Modify: `andypack/nodes.py` (new class + mappings)
- Test: `tests/test_nodes.py`

**Interfaces:**
- Consumes: `sprites.trim_batch`, `sprites.pivot_point`.
- Produces: node `SpriteTrimPivot` (category `andypack/Sprite`), `RETURN_TYPES=("IMAGE","SPRITE_TRIM")`, `RETURN_NAMES=("TRIMMED","TRIM_DATA")`, `FUNCTION="trim"`.

- [ ] **Step 1: Write failing test**

```python
# tests/test_nodes.py (append)
import torch
from andypack import nodes
def test_sprite_trim_pivot_node():
    img = torch.zeros((1, 8, 8, 4)); img[2:6, 2:6, :] = 1.0
    out, trim = nodes.SpriteTrimPivot().trim(img, alpha_threshold=0.03,
        trim_mode="union", pivot="bottom_center", pivot_x=0.5, pivot_y=1.0, pad=0)
    assert out.shape[-1] == 4 and trim["frames"][0]["pivot"]
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_nodes.py::test_sprite_trim_pivot_node -v`
Expected: FAIL (class missing).

- [ ] **Step 3: Implement**

```python
class SpriteTrimPivot:
    CATEGORY = "andypack/Sprite"; FUNCTION = "trim"
    RETURN_TYPES = ("IMAGE", "SPRITE_TRIM"); RETURN_NAMES = ("TRIMMED", "TRIM_DATA")
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "alpha_threshold": ("FLOAT", {"default": 0.03, "min": 0.0, "max": 1.0, "step": 0.01}),
            "trim_mode": (["union", "per_frame"],),
            "pivot": (["center", "bottom_center", "top_center", "custom"],),
        }, "optional": {
            "pivot_x": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0}),
            "pivot_y": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0}),
            "pad": ("INT", {"default": 0, "min": 0, "max": 256}),
        }}
    def trim(self, image, alpha_threshold, trim_mode, pivot, pivot_x=0.5, pivot_y=1.0, pad=0):
        out, rects = sprites.trim_batch(image, threshold=alpha_threshold, mode=trim_mode, pad=pad)
        h, w = int(out.shape[1]), int(out.shape[2])
        px, py = sprites.pivot_point(w, h, pivot, custom=(pivot_x, pivot_y))
        for r in rects:
            r["pivot"] = [px, py]
        return (out, {"frames": rects, "trim_mode": trim_mode, "pivot_kind": pivot})
```

Register in `NODE_CLASS_MAPPINGS` (`"SpriteTrimPivot"`) + display name `"Sprite Trim & Pivot"`.

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_nodes.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add andypack/nodes.py tests/test_nodes.py
git commit -m "feat: Sprite Trim & Pivot node"
```

---

### Task 15: `sprites.py` packer core

**Files:**
- Modify: `andypack/sprites.py`
- Test: `tests/test_sprites.py`

**Interfaces:**
- Produces: `pack_sheet(image, layout="grid", columns=0, padding=2, extrude=0, power_of_two=False, trim_data=None) -> tuple[Tensor, dict]` — returns the sheet IMAGE `[1,H,W,C]` and an `ANIM_ATLAS` dict `{"sheet_size":[w,h], "columns":n, "frames":[{"rect":[x,y,w,h], "source_size":[w,h], "offset":[x,y], "pivot":[x,y]|None, "duration_ms":int|None}]}`.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_sprites.py (append)
def test_pack_grid_places_frames():
    batch = torch.zeros((4, 6, 6, 4)); batch[..., :] = 1.0
    sheet, atlas = sprites.pack_sheet(batch, layout="grid", columns=2, padding=1)
    assert sheet.shape[0] == 1 and atlas["columns"] == 2 and len(atlas["frames"]) == 4
    assert atlas["frames"][1]["rect"][0] > atlas["frames"][0]["rect"][0]  # col 2 to the right
def test_pack_power_of_two():
    sheet, _ = sprites.pack_sheet(torch.ones((1, 5, 5, 4)), power_of_two=True)
    h, w = sheet.shape[1], sheet.shape[2]
    assert (h & (h - 1)) == 0 and (w & (w - 1)) == 0
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_sprites.py -k pack -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

Write `pack_sheet`: compute frame cell size (max H/W across the batch, since `trim union` already uniformizes), lay out by `layout` (grid auto-cols = ceil(sqrt(n)) unless `columns>0`; horizontal/vertical strips; maxrects = simple shelf packer for trimmed frames), composite each frame onto an RGBA canvas at its rect with `padding`, optional `extrude` (replicate edge pixels into the padding gutter), optional power-of-two rounding of the final canvas. Pull `duration_ms` from a `fps` passed by the node (Task 16). Reuse `images._resize_batch` only if needed.

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_sprites.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add andypack/sprites.py tests/test_sprites.py
git commit -m "feat: sprites.pack_sheet (grid/strip/maxrects, padding, extrude, POT)"
```

---

### Task 16: Spritesheet Packer node

**Files:**
- Modify: `andypack/nodes.py`
- Test: `tests/test_nodes.py`

**Interfaces:**
- Consumes: `sprites.pack_sheet`.
- Produces: `SpritesheetPacker` (category `andypack/Sprite`), `RETURN_TYPES=("IMAGE","ANIM_ATLAS")`, `RETURN_NAMES=("SHEET","ATLAS")`, `FUNCTION="pack"`. Optional `trim_data (SPRITE_TRIM)`, `fps (INT, forceInput)`, `names (STRING)`.

- [ ] **Step 1: Write failing test**

```python
# tests/test_nodes.py (append)
def test_spritesheet_packer_node():
    batch = torch.ones((4, 6, 6, 4))
    sheet, atlas = nodes.SpritesheetPacker().pack(batch, layout="grid", columns=2,
        padding=1, extrude=0, power_of_two=False, fps=8)
    assert sheet.shape[0] == 1 and atlas["frames"][0]["duration_ms"] == 125
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_nodes.py::test_spritesheet_packer_node -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

Add the class wrapping `sprites.pack_sheet`, passing `trim_data` and computing `duration_ms = round(1000/max(fps,1))` for each frame when `fps` is wired. INPUT_TYPES required: `image`, `layout` (`["grid","horizontal","vertical","maxrects"]`), `columns` (INT default 0), `padding` (INT default 2), `extrude` (INT default 0), `power_of_two` (BOOLEAN default False); optional: `trim_data`, `fps` (INT forceInput), `names` (STRING). Register mappings.

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_nodes.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add andypack/nodes.py tests/test_nodes.py
git commit -m "feat: Spritesheet Packer node"
```

---

### Task 17: `atlas.py` engine-format serializers

**Files:**
- Create: `andypack/atlas.py`
- Test: `tests/test_atlas.py`

**Interfaces:**
- Produces (pure stdlib, all take an `ANIM_ATLAS` dict + `name`, return `str`): `to_json_hash`, `to_json_array`, `to_aseprite`, `to_godot_spriteframes`, `to_unity_meta`, `to_texturepacker`, `to_css`. Plus `serialize(atlas, name, fmt) -> tuple[str, str]` returning `(text, file_extension)`.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_atlas.py
import json
from andypack import atlas
ATLAS = {"sheet_size": [12, 6], "columns": 2, "frames": [
    {"rect": [0,0,6,6], "source_size": [6,6], "offset": [0,0], "pivot": [3,6], "duration_ms": 125},
    {"rect": [6,0,6,6], "source_size": [6,6], "offset": [0,0], "pivot": [3,6], "duration_ms": 125}]}
def test_aseprite_shape():
    text, ext = atlas.serialize(ATLAS, "walk", "aseprite")
    data = json.loads(text)
    assert ext == ".json" and "frames" in data and data["meta"]["size"] == {"w": 12, "h": 6}
def test_godot_is_tres():
    text, ext = atlas.serialize(ATLAS, "walk", "godot_spriteframes")
    assert ext == ".tres" and "SpriteFrames" in text
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_atlas.py -v`
Expected: FAIL (module missing).

- [ ] **Step 3: Implement `andypack/atlas.py`**

Write each serializer per its target's documented shape (Aseprite `{frames, meta}`; TexturePacker hash/array; Godot `SpriteFrames` `.tres` resource; Unity `.meta` sprite sheet; CSS background-position rules) and a `serialize(atlas, name, fmt)` dispatcher mapping `format` → `(text, ext)`.

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_atlas.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add andypack/atlas.py tests/test_atlas.py
git commit -m "feat: atlas.py engine-format serializers (Aseprite/Godot/Unity/TexturePacker/CSS/JSON)"
```

---

### Task 18: Atlas Metadata Writer node

**Files:**
- Modify: `andypack/nodes.py`
- Test: `tests/test_nodes.py`

**Interfaces:**
- Consumes: `atlas.serialize`, `images.save_image_png`, `io.atomic_write_json`/text write, `io.render_id`.
- Produces: `AtlasMetadataWriter` (category `andypack/Export`, `OUTPUT_NODE=True`), `RETURN_TYPES=("STRING",)`, `RETURN_NAMES=("OUTPUT_DIR",)`, `FUNCTION="export"`.

- [ ] **Step 1: Write failing test**

```python
# tests/test_nodes.py (append)
import os
def test_atlas_metadata_writer(tmp_path, monkeypatch):
    monkeypatch.setattr(nodes.api, "output_dir", lambda: str(tmp_path))
    atlas_d = {"sheet_size": [12, 6], "columns": 2, "frames": [
        {"rect": [0,0,6,6], "source_size":[6,6], "offset":[0,0], "pivot":[3,6], "duration_ms":125}]}
    out = nodes.AtlasMetadataWriter().export(atlas_d, torch.ones((1,6,12,4)),
        "aseprite", "walk", output_subdir="atlas")
    d = out["result"][0] if isinstance(out, dict) else out[0]
    assert os.path.exists(os.path.join(d, "walk.png"))
    assert os.path.exists(os.path.join(d, "walk.json"))
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_nodes.py::test_atlas_metadata_writer -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

Class writes the sheet PNG first, then the serialized sidecar LAST (atomic temp+rename via a new `io.atomic_write_text(path, text)` helper — add it next to `atomic_write_json`). Resolve `output_dir` under `api.output_dir()/<output_subdir>`. When an `animation` (ANIM_ANIMATION) is wired, stamp `render_id` from its `_meta["prompt_hash"]`. INPUT_TYPES required: `atlas (ANIM_ATLAS)`, `sheet (IMAGE)`, `format` (combo of the 7), `name (STRING)`; optional: `output_subdir (STRING)`, `animation (ANIM_ANIMATION, forceInput)`. Return `{"ui": {...}, "result": (output_dir,)}` or `(output_dir,)`.

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_nodes.py tests/test_io.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add andypack/nodes.py andypack/io.py tests/test_nodes.py
git commit -m "feat: Atlas Metadata Writer node + io.atomic_write_text"
```

---

### Task 19: Character Atlas Builder node (FFLF-aware)

**Files:**
- Modify: `andypack/resolve.py` (add `rendered_directions` helper), `andypack/nodes.py`
- Test: `tests/test_resolve.py`, `tests/test_nodes.py`

**Interfaces:**
- Produces:
  - `resolve.rendered_directions(manifest, root, character, kind, entity_id, directions) -> list[tuple[str, str]]` — `(direction, frame_dir_or_png)` for each requested direction that is `node_complete`.
  - `CharacterAtlasBuilder` (category `andypack/Sprite`), `RETURN_TYPES=("IMAGE","ANIM_ATLAS","STRING")`, `RETURN_NAMES=("SHEET","ATLAS","REPORT")`, with a disk-mtime `IS_CHANGED`.

- [ ] **Step 1: Write failing test**

```python
# tests/test_resolve.py (append)
def test_rendered_directions_skips_unrendered(tmp_path):
    root = str(tmp_path); char = "hero"
    manifest = {"version": 1, "poses": {"base": {"directions": {"EAST": {}, "WEST": {}}}},
                "animations": {}, "defaults": {}}
    base = os.path.join(root, char, "_base"); os.makedirs(base, exist_ok=True)
    open(os.path.join(base, "EAST.png"), "wb").close()
    from andypack import io
    io.atomic_write_json(os.path.join(base, "EAST.json"),
        io.build_pose_sidecar({"prompt_hash":"sha1:x","direction":"EAST"}, created_utc="t"))
    got = resolve.rendered_directions(manifest, root, char, "pose", "base", ["EAST", "WEST"])
    assert [d for d, _ in got] == ["EAST"]
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_resolve.py::test_rendered_directions_skips_unrendered -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

Add `rendered_directions` to `resolve.py` (loop directions, keep those `node_complete`, return path via `pose_image_path`/`animation_frame_dir`). Add `CharacterAtlasBuilder` to `nodes.py`: resolve directions in canonical order (`manikins.CANONICAL_DIRECTIONS` filtered by `directions` arg `all|cardinal_4|custom`), load each via `images.load_image_tensor(..., keep_alpha=True)` (pose) or frame dirs (animation, first frame per direction), pack one row per direction via `sprites.pack_sheet(layout="grid", columns=<len(dirs)>)`, build a REPORT string of rendered vs skipped. Disk-mtime `IS_CHANGED` summing the resolved paths' mtimes.

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_resolve.py tests/test_nodes.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add andypack/resolve.py andypack/nodes.py tests/test_resolve.py tests/test_nodes.py
git commit -m "feat: Character Atlas Builder (FFLF-aware multi-direction sheet)"
```

---

### Task 20: `sprites.py` palette core

**Files:**
- Modify: `andypack/sprites.py`
- Test: `tests/test_palette.py`

**Interfaces:**
- Produces:
  - `extract_palette(image, colors=32) -> list[tuple[int,int,int]]`.
  - `quantize_to_palette(image, palette, dither="none", preserve_alpha=True) -> Tensor` — nearest-color remap, alpha preserved.
  - `ANIM_PALETTE` bundle shape: `{"colors": [[r,g,b], ...]}`.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_palette.py
import torch
from andypack import sprites
def test_extract_and_lock_palette():
    img = torch.zeros((1, 4, 4, 3)); img[:, :, :2, 0] = 1.0   # red / black
    pal = sprites.extract_palette(img, colors=2)
    assert len(pal) <= 2
    out = sprites.quantize_to_palette(img, pal)
    uniq = {tuple(int(v*255) for v in out[0, y, x, :3]) for y in range(4) for x in range(4)}
    assert uniq.issubset({tuple(c) for c in pal})
def test_quantize_preserves_alpha():
    rgba = torch.ones((1, 2, 2, 4)); rgba[..., 3] = 0.5
    out = sprites.quantize_to_palette(rgba, [(255,255,255)], preserve_alpha=True)
    assert out.shape[-1] == 4 and abs(float(out[0,0,0,3]) - 0.5) < 1e-3
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_palette.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

Use Pillow `Image.quantize`/`convert("P", palette=...)` for extract (median cut) and a numpy nearest-color remap for lock; reattach the original alpha channel when `preserve_alpha`. Dither via PIL's `Dither.FLOYDSTEINBERG` / `NONE` (ordered = simple 4×4 Bayer offset before nearest).

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_palette.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add andypack/sprites.py tests/test_palette.py
git commit -m "feat: sprites palette extract + quantize/lock (alpha-preserving)"
```

---

### Task 21: Palette Quantize & Lock node

**Files:**
- Modify: `andypack/nodes.py`
- Test: `tests/test_nodes.py`

**Interfaces:**
- Consumes: `sprites.extract_palette`, `sprites.quantize_to_palette`.
- Produces: `PaletteQuantizeLock` (category `andypack/Sprite`), `RETURN_TYPES=("IMAGE","ANIM_PALETTE")`, `RETURN_NAMES=("IMAGE","PALETTE")`, `FUNCTION="run"`. Optional `palette (ANIM_PALETTE)`, `preserve_alpha`, `extract_only`.

- [ ] **Step 1: Write failing test**

```python
# tests/test_nodes.py (append)
def test_palette_node_extract_only_passthrough():
    img = torch.ones((1, 2, 2, 3))
    out_img, pal = nodes.PaletteQuantizeLock().run(img, colors=4, dither="none",
        preserve_alpha=True, extract_only=True)
    assert torch.equal(out_img, img) and "colors" in pal
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_nodes.py::test_palette_node_extract_only_passthrough -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

Class: if `palette` wired → lock to its colors; else extract `colors` and (unless `extract_only`) quantize to them. `extract_only` returns the input image unchanged + the extracted `{"colors": [...]}`. Register mappings.

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_nodes.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add andypack/nodes.py tests/test_nodes.py
git commit -m "feat: Palette Quantize & Lock node"
```

---

## Group 4 — FFLF production + mirror economy

### Task 22: `next_actionable` category predicate + `skip_mirrored`

**Files:**
- Modify: `andypack/api.py` (`next_actionable` ~425)
- Test: `tests/test_api.py`

**Interfaces:**
- Produces: `next_actionable(manifest, root, character, kind, *, exclude_root=False, category=None, skip_mirrored=False) -> Optional[dict]`. `category` filters to `entity.get("category") == category`; `skip_mirrored` drops directions present as keys in `manifest.get("mirror_map", {})`.

- [ ] **Step 1: Write failing test**

```python
# tests/test_api.py (append)
from andypack import api
def test_next_actionable_skip_mirrored(tmp_path):
    root = str(tmp_path); char = "hero"
    manifest = {"version": 1, "mirror_map": {"WEST": "EAST"},
        "poses": {"base": {"directions": {"EAST": {}}},
                  "p": {"from": {"ref": "base"}, "directions": {"EAST": {}, "WEST": {}}}},
        "animations": {}, "defaults": {}}
    base = os.path.join(root, char, "_base"); os.makedirs(base, exist_ok=True)
    open(os.path.join(base, "EAST.png"), "wb").close()
    from andypack import io
    io.atomic_write_json(os.path.join(base, "EAST.json"),
        io.build_pose_sidecar({"prompt_hash":"sha1:x","direction":"EAST"}, created_utc="t"))
    job = api.next_actionable(manifest, root, char, "pose", exclude_root=True, skip_mirrored=True)
    assert job["direction"] == "EAST"   # WEST is a mirror target, skipped
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_api.py::test_next_actionable_skip_mirrored -v`
Expected: FAIL (unexpected kwarg).

- [ ] **Step 3: Implement**

Add `category=None, skip_mirrored=False` params. In the loop, `continue` when `category is not None` and the entity's `category` differs, and when `skip_mirrored` and `item["direction"] in eff.get("mirror_map", {})`.

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_api.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add andypack/api.py tests/test_api.py
git commit -m "feat: next_actionable category predicate + skip_mirrored"
```

---

### Task 23: `skip_mirrored` on the Auto selectors

**Files:**
- Modify: `andypack/nodes.py` (`AutoPoseSelector` ~748, `AutoAnimationSelector` ~804)
- Test: `tests/test_nodes.py`

**Interfaces:**
- Consumes: `api.next_actionable(..., skip_mirrored=...)`.
- Produces: both Auto selectors gain a `skip_mirrored` BOOLEAN (default True) input, reflected in `IS_CHANGED` and passed to `next_actionable`.

- [ ] **Step 1: Write failing test**

```python
# tests/test_nodes.py (append)
def test_auto_pose_selector_has_skip_mirrored_input():
    req = nodes.AutoPoseSelector.INPUT_TYPES()["required"]
    assert "skip_mirrored" in req
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_nodes.py::test_auto_pose_selector_has_skip_mirrored_input -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

Add `"skip_mirrored": ("BOOLEAN", {"default": True})` to both Auto selectors' `INPUT_TYPES["required"]`; thread it into `IS_CHANGED` and `select` and pass to `api.next_actionable`. Keep `exclude_root=True` for the pose selector.

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_nodes.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add andypack/nodes.py tests/test_nodes.py
git commit -m "feat: skip_mirrored on Auto Pose/Animation selectors"
```

---

### Task 24: MirrorFrameWriter batch mode + helper lift

**Files:**
- Modify: `andypack/nodes.py` (`MirrorFrameWriter` ~859)
- Test: `tests/test_nodes.py`

**Interfaces:**
- Consumes: existing `_mirror_pose`/`_mirror_animation`.
- Produces: `MirrorFrameWriter` gains `mirror_all` (BOOLEAN, default False) and renames the `id` widget to `entity_id`. With `mirror_all`, it iterates `mirror_map` and mirrors every destination whose source is rendered; returns `(OUTPUT_DIRS, COUNT)`. `RETURN_TYPES=("STRING","INT")`, `RETURN_NAMES=("OUTPUT_DIRS","COUNT")`.

- [ ] **Step 1: Write failing test**

```python
# tests/test_nodes.py (append)
def test_mirror_writer_batch_all(tmp_path, monkeypatch):
    monkeypatch.setattr(nodes, "_characters_root", lambda: str(tmp_path))
    root = str(tmp_path); char = "hero"
    manifest = {"version": 1, "mirror_map": {"WEST": "EAST"},
        "poses": {"base": {"directions": {"EAST": {}}},
                  "p": {"from": {"ref": "base"}, "directions": {"EAST": {}, "WEST": {}}}},
        "animations": {}, "defaults": {}}
    src = resolve.pose_image_path(root, char, "p", "EAST"); os.makedirs(os.path.dirname(src), exist_ok=True)
    images.save_image_png(torch.ones((1,4,4,4)), src)
    from andypack import io
    r = resolve.resolve_pose(manifest, root, char, "p", "EAST")
    io.atomic_write_json(resolve.pose_sidecar_path(root, char, "p", "EAST"),
        io.build_pose_sidecar(r["meta"], created_utc="t"))
    dirs, count = nodes.MirrorFrameWriter().write(manifest, char, "pose", "p", "", mirror_all=True)
    assert count == 1 and os.path.exists(resolve.pose_image_path(root, char, "p", "WEST"))
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_nodes.py::test_mirror_writer_batch_all -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

Rename the `id` INPUT to `entity_id`; update `write(self, manifest, character, kind, entity_id, direction, mirror_all=False)`. When `mirror_all`, loop `manifest["mirror_map"].items()` (target then source), call `_mirror_pose`/`_mirror_animation` for each target whose source is generated (guard with `os.path.exists`/`read_node_meta`), collect output dirs, return `(newline-join(dirs), len(dirs))`. Single-target path returns `(out_dir, 1)`. Update `RETURN_TYPES`/`RETURN_NAMES` and extend the Task-3 `IS_CHANGED` to accept `mirror_all` and `entity_id`.

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_nodes.py tests/test_fixes.py -k mirror -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add andypack/nodes.py tests/test_nodes.py
git commit -m "feat: MirrorFrameWriter batch 'mirror all' mode; rename id->entity_id"
```

---

### Task 25: Manikin Pose Control node

**Files:**
- Modify: `andypack/nodes.py`
- Test: `tests/test_nodes.py`

**Interfaces:**
- Consumes: `manikins.manikin_path`, `manikins.CANONICAL_DIRECTIONS`, `images.load_image_tensor`, `resolve.resolve_pose`.
- Produces: `ManikinPoseControl` (category `andypack/Pose`), `RETURN_TYPES=("IMAGE","STRING","STRING")`, `RETURN_NAMES=("POSE_CONTROL_IMAGE","POSITIVE_PROMPT","DIRECTION_NAME")`, `FUNCTION="control"`.

- [ ] **Step 1: Write failing test**

```python
# tests/test_nodes.py (append)
def test_manikin_pose_control_direction_only(monkeypatch, tmp_path):
    monkeypatch.setattr(nodes, "_characters_root", lambda: str(tmp_path))
    manifest = {"version": 1, "poses": {"base": {"directions": {"EAST": {}}}},
                "animations": {}, "defaults": {}}
    img, pos, dname = nodes.ManikinPoseControl().control(manifest, "(select character)",
        "base", "EAST", direction_only=True)
    assert img.shape[-1] == 3 and dname == "EAST"
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_nodes.py::test_manikin_pose_control_direction_only -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

Class loads `manikins.manikin_path(direction)` as the control image; when not `direction_only`, also resolves `resolve_pose` for the positive prompt (else empty). INPUT_TYPES: required `manifest`, `character` (`_character_choices()`), `pose` (STRING default "base"), `direction` (STRING); optional `direction_only` (BOOLEAN default False). Register mappings.

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_nodes.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add andypack/nodes.py tests/test_nodes.py
git commit -m "feat: Manikin Pose Control node (manikins as ControlNet source)"
```

---

### Task 26: Character Identity Anchor node

**Files:**
- Modify: `andypack/nodes.py`
- Test: `tests/test_nodes.py`

**Interfaces:**
- Consumes: `resolve.reference_image_path`, `resolve.resolve_pose`, `images.load_image_tensor`/`empty_image`/`_resize_batch`.
- Produces: `CharacterIdentityAnchor` (category `andypack/Character`), `RETURN_TYPES=("IMAGE","IMAGE","IMAGE")`, `RETURN_NAMES=("REFERENCE_IMAGE","BASE_DIRECTION_IMAGE","ANCHOR_BATCH")`, with a selector-style mtime `IS_CHANGED`.

- [ ] **Step 1: Write failing test**

```python
# tests/test_nodes.py (append)
def test_character_identity_anchor(monkeypatch, tmp_path):
    monkeypatch.setattr(nodes, "_characters_root", lambda: str(tmp_path))
    root = str(tmp_path); char = "hero"
    images.save_image_png(torch.ones((1,4,4,3)), resolve.reference_image_path(root, char))
    manifest = {"version": 1, "poses": {"base": {"directions": {"EAST": {}}}},
                "animations": {}, "defaults": {}}
    ref, base, batch = nodes.CharacterIdentityAnchor().anchor(manifest, char, "EAST",
        include_reference=True, include_base=False, base_pose="base")
    assert ref.shape[-1] == 3 and batch.shape[0] >= 1
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_nodes.py::test_character_identity_anchor -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

Load the reference (or `empty_image()` if absent); load the base-pose direction image when complete (else empty); concat the present ones into ANCHOR_BATCH (resize to a common size via `images._resize_batch`). INPUT_TYPES: required `manifest`, `character`, `direction` (STRING); optional `include_reference`/`include_base` (BOOLEAN), `base_pose` (STRING default "base"). `IS_CHANGED` fingerprints the reference + base PNG mtimes. Register mappings.

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_nodes.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add andypack/nodes.py tests/test_nodes.py
git commit -m "feat: Character Identity Anchor node (reference + base for IPAdapter)"
```

---

### Task 27: Action Set Selector node

**Files:**
- Modify: `andypack/nodes.py`
- Test: `tests/test_nodes.py`

**Interfaces:**
- Consumes: `api.next_actionable(..., category=...)`, `api.regen_queue`, `_build_animation_bundle`, `resolve.resolve_animation`.
- Produces: `ActionSetSelector` (category `andypack/Animation`), `RETURN_TYPES=("ANIM_ANIMATION","INT","STRING")`, `RETURN_NAMES=("ANIMATION","REMAINING","REPORT")`, with `IS_CHANGED` over the filtered next job.

- [ ] **Step 1: Write failing test**

```python
# tests/test_nodes.py (append)
def test_action_set_selector_input_has_action_set():
    req = nodes.ActionSetSelector.INPUT_TYPES()["required"]
    assert "action_set" in req
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_nodes.py::test_action_set_selector_input_has_action_set -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

Class mirrors `AutoAnimationSelector` but passes `category=action_set or None` to `next_actionable`, counts remaining in-set via `regen_queue` filtered by category, returns `(bundle, remaining, report)`. `action_set` is a STRING widget (default ""). Register mappings.

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_nodes.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add andypack/nodes.py tests/test_nodes.py
git commit -m "feat: Action Set Selector (next job scoped to a category)"
```

---

### Task 28: State Machine Report node

**Files:**
- Modify: `andypack/api.py` (add `state_machine` + `format_state_machine`), `andypack/nodes.py`
- Test: `tests/test_api.py`, `tests/test_nodes.py`

**Interfaces:**
- Produces:
  - `api.state_machine(manifest, root, character) -> dict` — `{"states":[...], "transitions":[{"from","clip","to","loop","directions"}], ...}` derived from each animation's `start_from`/`end_at` refs (same ref+dir = self-loop).
  - `api.format_state_machine(sm) -> str`.
  - `StateMachineReport` (category `andypack/Diagnostics`, `OUTPUT_NODE`), `RETURN_TYPES=("STRING","STRING")`, `RETURN_NAMES=("REPORT","JSON")`, `IS_CHANGED=nan`.

- [ ] **Step 1: Write failing test**

```python
# tests/test_api.py (append)
def test_state_machine_marks_self_loop(tmp_path):
    manifest = {"version": 1, "poses": {"stand": {"directions": {"EAST": {}}}},
        "animations": {"idle": {"start_from": {"ref": "stand"}, "end_at": {"ref": "stand"},
            "directions": {"EAST": {}}, "length": 5, "fps": 8, "width": 16, "height": 16}},
        "defaults": {}}
    sm = api.state_machine(manifest, str(tmp_path), "")
    t = [x for x in sm["transitions"] if x["clip"] == "idle"][0]
    assert t["loop"] is True and t["from"] == "stand" and t["to"] == "stand"
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_api.py::test_state_machine_marks_self_loop -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

`state_machine` walks `animations`, reads `effective_start_dep`/`end_at` refs, marks `loop` when start ref == end ref (and `end_at` present), groups by `category`, lists covered directions; runs inside `resolution_pass()`. `format_state_machine` renders a transition table. The node returns `(format_state_machine(sm), json.dumps(sm, indent=2))`. Register mappings.

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_api.py tests/test_nodes.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add andypack/api.py andypack/nodes.py tests/test_api.py tests/test_nodes.py
git commit -m "feat: State Machine Report (project animator controller from FFLF graph)"
```

---

### Task 29: Turnaround Sheet node

**Files:**
- Modify: `andypack/images.py` (add `contact_sheet`), `andypack/nodes.py`
- Test: `tests/test_images.py`, `tests/test_nodes.py`

**Interfaces:**
- Produces:
  - `images.contact_sheet(tiles, columns, cell=None, labels=None) -> Tensor` — composites tiles (None = placeholder) into one IMAGE.
  - `TurnaroundSheet` (category `andypack/Diagnostics`, `OUTPUT_NODE`), `RETURN_TYPES=("IMAGE",)`, `RETURN_NAMES=("SHEET",)`, `IS_CHANGED=nan`.

- [ ] **Step 1: Write failing test**

```python
# tests/test_images.py (append)
import torch
from andypack import images
def test_contact_sheet_handles_missing_tiles():
    sheet = images.contact_sheet([torch.ones((1,4,4,3)), None], columns=2, cell=(4,4))
    assert sheet.shape[0] == 1 and sheet.shape[2] == 8  # 2 columns x 4px
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_images.py::test_contact_sheet_handles_missing_tiles -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

`contact_sheet` lays tiles in a grid, resizing each to `cell` (or the max tile size), drawing a neutral placeholder for `None`. `TurnaroundSheet` resolves every direction of `pose` in `manikins.CANONICAL_DIRECTIONS` order, loads complete ones (`pose_complete` + `load_image_tensor`), passes the list (None for missing) to `contact_sheet`, returns `{"ui": preview, "result": (sheet,)}`. INPUT_TYPES: required `manifest`, `character`, `pose` (STRING default "base"); optional `columns` (INT default 4), `include_labels` (BOOLEAN), `cell_size` (INT default 0). Register mappings.

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_images.py tests/test_nodes.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add andypack/images.py andypack/nodes.py tests/test_images.py tests/test_nodes.py
git commit -m "feat: Turnaround Sheet node (labeled contact sheet of all directions)"
```

---

## Group 5 — Render-economy / polish

### Task 30: AnimationPlayback modes

**Files:**
- Modify: `andypack/images.py` (add `apply_play_mode`), `andypack/nodes.py` (`AnimationPlayback` ~563)
- Test: `tests/test_images.py`, `tests/test_nodes.py`

**Interfaces:**
- Produces:
  - `images.apply_play_mode(frames, mode, hold_frames=0) -> Tensor` — `loop` (unchanged), `ping_pong` (append reversed interior), `once` (unchanged here; node passes loops=1), `hold_last` (repeat last `hold_frames`).
  - `AnimationPlayback` gains `mode` combo + `hold_frames` INT, reflected in `IS_CHANGED`.

- [ ] **Step 1: Write failing test**

```python
# tests/test_images.py (append)
def test_ping_pong_appends_reversed_interior():
    f = torch.arange(4).float().reshape(4,1,1,1).repeat(1,2,2,3)
    out = images.apply_play_mode(f, "ping_pong")
    assert out.shape[0] == 6   # 0,1,2,3,2,1
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_images.py::test_ping_pong_appends_reversed_interior -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

`apply_play_mode`: `ping_pong` then `cat([f, flip(f[1:-1])])`; `hold_last` then `cat([f, f[-1:].repeat(hold_frames,1,1,1)])`; others return `f`. In `AnimationPlayback.play`, after `assemble_playback`, apply the mode (and for `once`, pass `loops=1` upstream). Add `mode` (`["loop","ping_pong","once","hold_last"]`) and `hold_frames` (INT default 0) to INPUT_TYPES and `IS_CHANGED`.

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_images.py tests/test_nodes.py tests/test_playback.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add andypack/images.py andypack/nodes.py tests/test_images.py tests/test_nodes.py
git commit -m "feat: AnimationPlayback ping_pong/once/hold_last modes"
```

---

### Task 31: Boomerang Loop Writer node

**Files:**
- Modify: `andypack/nodes.py`
- Test: `tests/test_nodes.py`

**Interfaces:**
- Consumes: `io.clear_frames`, `io.frame_name`, `io.build_animation_meta`, `images.save_image_png`, `images.is_empty`.
- Produces: `BoomerangLoopWriter` (category `andypack/Animation`, `OUTPUT_NODE`), `RETURN_TYPES=("STRING",)`, `RETURN_NAMES=("OUTPUT_DIR",)`, `FUNCTION="write"`. Writes A-to-B-to-A so first==last (loop by construction), `meta["loop"]=True`.

- [ ] **Step 1: Write failing test**

```python
# tests/test_nodes.py (append)
def test_boomerang_writes_palindrome(tmp_path, monkeypatch):
    monkeypatch.setattr(nodes, "_characters_root", lambda: str(tmp_path))
    anim = {"output_dir": str(tmp_path / "idle" / "EAST"),
            "_meta": {"prompt_hash": "sha1:x", "loop": False}}
    f = torch.arange(3).float().reshape(3,1,1,1).repeat(1,4,4,3)  # frames 0,1,2
    out = nodes.BoomerangLoopWriter().write(anim, f, mode="boomerang")
    d = out["result"][0] if isinstance(out, dict) else out[0]
    import json
    meta = json.load(open(os.path.join(d, "meta.json")))
    assert meta["loop"] is True and meta["frames"]["count"] == 4  # 0,1,2,1
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_nodes.py::test_boomerang_writes_palindrome -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

`boomerang` builds `cat([frames, frames[1:-1].flip(0)])`; `trim_seam` drops a duplicated boundary. Reject `is_empty`. Write with `clear_frames` + per-frame `save_image_png`; set `meta["loop"]=True` before `build_animation_meta`. INPUT_TYPES: required `animation`, `frames`, `mode` (`["boomerang","trim_seam"]`); optional `drop_turnaround` (BOOLEAN default True), `seed` (INT forceInput). Register mappings.

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_nodes.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add andypack/nodes.py tests/test_nodes.py
git commit -m "feat: Boomerang Loop Writer (seamless loop from a one-way clip)"
```

---

### Task 32: Tween Clip Provider node

**Files:**
- Modify: `andypack/nodes.py`
- Test: `tests/test_nodes.py`

**Interfaces:**
- Consumes: the ANIM_ANIMATION bundle's resolved `start_image`/`end_image` tensors + `_meta` length/fps.
- Produces: `TweenClipProvider` (category `andypack/Animation`), `RETURN_TYPES=("IMAGE","IMAGE","INT","INT")`, `RETURN_NAMES=("START_IMAGE","END_IMAGE","TWEEN_COUNT","FPS")`, `FUNCTION="provide"`, plus a private `_validate_fflf(start, end)` raising on a non-FFLF clip.

- [ ] **Step 1: Write failing test**

```python
# tests/test_nodes.py (append)
def test_tween_requires_fflf():
    import pytest
    with pytest.raises(RuntimeError):
        nodes.TweenClipProvider()._validate_fflf(start=None, end=None)
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_nodes.py::test_tween_requires_fflf -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

`provide(animation, tween_count=0, include_anchors=True)` reads `animation["start_image"]`/`["end_image"]` tensors; `_validate_fflf` raises when either is the empty sentinel or they are equal. `TWEEN_COUNT` defaults to `length-2` snapped to 4n+1 when 0. Return the two anchor images + counts + `fps` from `_meta` (clamped >=1). Register mappings.

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_nodes.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add andypack/nodes.py tests/test_nodes.py
git commit -m "feat: Tween Clip Provider (FFLF anchors -> external interpolation)"
```

---

### Task 33: Frame Timing Normalizer node

**Files:**
- Modify: `andypack/images.py` (add `retime_batch`), `andypack/nodes.py`
- Test: `tests/test_images.py`, `tests/test_nodes.py`

**Interfaces:**
- Produces:
  - `images.retime_batch(frames, target, mode) -> Tensor` — `resample` (nearest-index to target N), `trim` (slice to N), `pad_hold` (repeat last to N).
  - `FrameTimingNormalizer` (category `andypack/Animation`), `RETURN_TYPES=("IMAGE","INT")`, `RETURN_NAMES=("FRAMES","LENGTH")`.

- [ ] **Step 1: Write failing test**

```python
# tests/test_images.py (append)
def test_retime_resample_to_target():
    f = torch.arange(4).float().reshape(4,1,1,1).repeat(1,2,2,3)
    out = images.retime_batch(f, 8, "resample")
    assert out.shape[0] == 8
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_images.py::test_retime_resample_to_target -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

`retime_batch`: resample maps `i -> round(i*(N-1)/(target-1))`; trim slices; pad_hold repeats the last frame. `FrameTimingNormalizer.run` resolves target from `target_length` or the wired `animation` meta length, snaps to 4n+1 when `enforce_4n1`, returns `(frames, N)`. INPUT_TYPES: required `frames`, `mode` (`["resample","trim","pad_hold"]`), `enforce_4n1` (BOOLEAN default True); optional `animation` (ANIM_ANIMATION), `target_length` (INT default 0). Register mappings.

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_images.py tests/test_nodes.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add andypack/images.py andypack/nodes.py tests/test_images.py tests/test_nodes.py
git commit -m "feat: Frame Timing Normalizer (uniform 4n+1 frame counts)"
```

---

### Task 34: Color Variant Batcher node

**Files:**
- Modify: `andypack/images.py` (add `recolor`), `andypack/nodes.py`
- Test: `tests/test_images.py`, `tests/test_nodes.py`

**Interfaces:**
- Produces:
  - `images.recolor(image, spec) -> Tensor` — `spec` is `{"hue": deg, "sat": x, "val": y}` or `{"hex": "#RRGGBB"}`; alpha preserved.
  - `ColorVariantBatcher` (category `andypack/Animation`, `OUTPUT_NODE`), `RETURN_TYPES=("STRING","INT")`, `RETURN_NAMES=("OUTPUT_DIRS","COUNT")`. Writes sibling `<id>__<variant>` targets copying source seed/render_id; disk-mtime `IS_CHANGED`.

- [ ] **Step 1: Write failing test**

```python
# tests/test_images.py (append)
def test_recolor_hue_preserves_alpha():
    rgba = torch.ones((1,2,2,4)); rgba[...,3] = 0.5; rgba[...,0] = 1.0; rgba[...,1:3] = 0.0
    out = images.recolor(rgba, {"hue": 120, "sat": 1.0, "val": 1.0})
    assert out.shape[-1] == 4 and abs(float(out[0,0,0,3]) - 0.5) < 1e-3
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_images.py::test_recolor_hue_preserves_alpha -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

`recolor` converts RGB to HSV (numpy), applies hue rotate / sat-val scale or remaps to a target hue, back to RGB, reattaches alpha. `ColorVariantBatcher` parses a newline `variants` spec (`"name: hue=deg, sat=x, val=y"` or `"name: #RRGGBB"`), reads source pose/animation pixels via the resolver paths (must exist; frame-PNG guard for animations), writes each variant to `<id>__<variant>` siblings using the writer discipline (sidecar/meta last, copy source seed + render_id). INPUT_TYPES mirror `MirrorFrameWriter` + a `variants` (STRING multiline). Register mappings.

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_images.py tests/test_nodes.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add andypack/images.py andypack/nodes.py tests/test_images.py tests/test_nodes.py
git commit -m "feat: Color Variant Batcher (deterministic recolor siblings)"
```

---

### Task 35: Variant Layer Composer node

**Files:**
- Modify: `andypack/nodes.py`
- Test: `tests/test_nodes.py`

**Interfaces:**
- Consumes: `resolve.merge_layers`, `resolve.merge_negative`, `resolve.hash_prompts`.
- Produces: `VariantLayerComposer` (category `andypack/Pose`), `RETURN_TYPES=("ANIM_POSE",)`, `RETURN_NAMES=("POSE",)`, `FUNCTION="compose"`.

- [ ] **Step 1: Write failing test**

```python
# tests/test_nodes.py (append)
def test_variant_layer_composer_recomputes_hash():
    pose = {"source_image": images.empty_image(), "pose_reference": images.empty_image(),
            "positive": "a hero", "negative": "blurry", "output_dir": "/x/_p",
            "_meta": {"prompt_hash": "sha1:old", "image": "EAST.png"}}
    out = nodes.VariantLayerComposer().compose(pose, "gold", "golden armor", "")
    assert "golden armor" in out["positive"] and out["_meta"]["prompt_hash"] != "sha1:old"
    assert out["output_dir"].endswith("__gold")
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_nodes.py::test_variant_layer_composer_recomputes_hash -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

`compose(pose, variant_id, variant_positive, variant_negative="", output_suffix="")` merges via `merge_layers(pose["positive"], variant_positive)` / `merge_negative(pose["negative"], variant_negative)`, recomputes `prompt_hash = resolve.hash_prompts(pos, neg)` (copy `_meta`, replace `prompt_hash`), sets `output_dir = f"{pose['output_dir']}__{output_suffix or variant_id}"`, copies the rest of the bundle. INPUT_TYPES: required `pose (ANIM_POSE)`, `variant_id (STRING)`, `variant_positive (STRING multiline)`; optional `variant_negative (STRING multiline)`, `output_suffix (STRING)`. Register mappings.

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_nodes.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add andypack/nodes.py tests/test_nodes.py
git commit -m "feat: Variant Layer Composer (outfit/equipment cascade layer at wire time)"
```

---

### Task 36: Animated Sprite Export node

**Files:**
- Modify: `andypack/images.py` (add `save_animated_gif`, `save_animated_apng`, `onion_skin`), `andypack/nodes.py`
- Test: `tests/test_images.py`, `tests/test_nodes.py`

**Interfaces:**
- Produces:
  - `images.save_animated_gif(frames, path, fps)`, `images.save_animated_apng(frames, path, fps)`, `images.onion_skin(frames, prev, next, opacity) -> Tensor`.
  - `AnimatedSpriteExport` (category `andypack/Export`, `OUTPUT_NODE`), `RETURN_TYPES=("IMAGE","STRING")`, `RETURN_NAMES=("PREVIEW","OUTPUT_DIR")`, `FUNCTION="export"`.

- [ ] **Step 1: Write failing test**

```python
# tests/test_images.py (append)
def test_save_gif(tmp_path):
    f = torch.ones((3, 4, 4, 3))
    p = str(tmp_path / "a.gif")
    images.save_animated_gif(f, p, 8)
    from PIL import Image
    with Image.open(p) as im:
        assert im.is_animated and im.n_frames == 3
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_images.py::test_save_gif -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

GIF/APNG via Pillow `save_all`; `onion_skin` composites ghosted prev/next frames at reduced opacity. `AnimatedSpriteExport.export` writes to `api.output_dir()/<name>` (or temp for preview), returns `{"ui": _animated_preview(...), "result": (frames, output_dir)}`; returns `{}` ui headless. INPUT_TYPES: required `image`, `format` (`["gif","apng","webp"]`), `loop` (BOOLEAN default True); optional `fps` (INT forceInput default 12), `onion_skin` (BOOLEAN), `onion_prev`/`onion_next` (INT), `onion_opacity` (FLOAT), `name` (STRING). Register mappings.

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_images.py tests/test_nodes.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add andypack/images.py andypack/nodes.py tests/test_images.py tests/test_nodes.py
git commit -m "feat: Animated Sprite Export (GIF/APNG/WebP + onion-skin)"
```

---

## Group 6 — Panel / route features

### Task 37: Thumbnail route

**Files:**
- Modify: `andypack/images.py` (add `thumbnail_data_uri`), `andypack/api.py` (add `thumb_path`), `andypack/server.py` (new route)
- Test: `tests/test_images.py`, `tests/test_api.py`, `tests/test_server.py`

**Interfaces:**
- Produces:
  - `images.thumbnail_data_uri(path, max_px=96) -> str` — `data:image/png;base64,...`.
  - `api.thumb_path(root, character, kind, entity_id, direction) -> Optional[str]` — validated (`_is_safe_segment` on every segment) pose PNG / animation first-frame / reference path; None if unsafe or absent.
  - `GET /anim_coord/thumb?character=&kind=&id=&direction=` then `{"data_uri": "..."}` or 404.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_api.py (append)
def test_thumb_path_rejects_traversal(tmp_path):
    assert api.thumb_path(str(tmp_path), "../x", "pose", "base", "EAST") is None
    assert api.thumb_path(str(tmp_path), "hero", "pose", "..", "EAST") is None
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_api.py::test_thumb_path_rejects_traversal -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

`thumb_path` validates `character`, `entity_id`, `direction` via `_is_safe_segment`, returns the resolved path (`kind=reference` then `_reference.png`; pose then `_<id>/<dir>.png`; animation then first `frame_*.png` in `<id>/<dir>`) or None when unsafe/absent. `thumbnail_data_uri` opens with PIL, `thumbnail((max_px,max_px))`, saves PNG to `BytesIO`, base64-encodes. Server route resolves `characters_dir()`, calls `thumb_path`, returns the data-uri JSON or 404.

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_api.py tests/test_images.py tests/test_server.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add andypack/images.py andypack/api.py andypack/server.py tests/test_api.py tests/test_images.py tests/test_server.py
git commit -m "feat: /anim_coord/thumb base64 thumbnail route (server-resolved, path-safe)"
```

---

### Task 38: Coverage-grid thumbnails (panel)

**Files:**
- Modify: `web/anim_coord_panel.js`
- Test: manual (frontend), documented.

**Interfaces:**
- Consumes: `GET /anim_coord/thumb`.

- [ ] **Step 1: Implement**

In the coverage grid render, for each cell add a lazy `<img loading="lazy">` whose `src` is set from the `data_uri` returned by `/anim_coord/thumb?character=&kind=&id=&direction=`; fall back to the status glyph on 404. Cache per cell to avoid refetching.

- [ ] **Step 2: Manual verification**

Run ComfyUI, open the panel for a character with rendered cells: thumbnails appear; missing cells keep the glyph. Record in the commit body.

- [ ] **Step 3: Commit**

```bash
git add web/anim_coord_panel.js
git commit -m "feat: coverage-grid thumbnails via /anim_coord/thumb"
```

---

### Task 39: Characters tab — reference + overlay editor (panel)

**Files:**
- Modify: `web/anim_coord_panel.js`, `andypack/server.py` (widen `/character/save`), `andypack/api.py` (widen `save_character_layer`)
- Test: `tests/test_api.py`, manual (frontend)

**Interfaces:**
- Produces: `save_character_layer(root, name, positive, negative, overlay=None)` round-trips a `poses`/`animations` overlay.

- [ ] **Step 1: Write failing test**

```python
# tests/test_api.py (append)
def test_save_character_layer_preserves_overlay(tmp_path):
    root = str(tmp_path)
    api.save_character_layer(root, "hero", "p", "n",
        overlay={"poses": {"wave": {"from": {"ref": "base"}, "directions": {"EAST": {}}}}})
    layer = api.read_character_layer(root, "hero")
    assert "wave" in layer.get("poses", {})
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_api.py::test_save_character_layer_preserves_overlay -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

Add `overlay=None` to `save_character_layer`; when a dict, fold its `poses`/`animations` into the `layer` before `build_character`. Widen `/character/save` to read `poses`/`animations` from the body. In the panel Characters tab, render the reference thumbnail (thumb route, `kind=reference`) and a `<textarea>` JSON editor bound to the overlay, saving via the route.

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_api.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add web/anim_coord_panel.js andypack/server.py andypack/api.py tests/test_api.py
git commit -m "feat: Characters tab reference thumbnail + overlay editor"
```

---

### Task 40: Space-safe grid keys + mirror tally (panel)

**Files:**
- Modify: `web/anim_coord_panel.js`, `andypack/api.py` (`manifest_options` includes `mirror_map`)
- Test: `tests/test_api.py`, manual (frontend)

- [ ] **Step 1: Write failing test**

```python
# tests/test_api.py (append)
def test_manifest_options_includes_mirror_map():
    out = api.manifest_options({"version": 1, "poses": {}, "animations": {}, "mirror_map": {"WEST": "EAST"}})
    assert out["mirror_map"] == {"WEST": "EAST"}
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_api.py::test_manifest_options_includes_mirror_map -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

Add `"mirror_map": manifest.get("mirror_map", {})` to `api.manifest_options`. In the panel, replace the space-join/space-split group key with a carried `{kind, category, id}` object so ids/categories with spaces group correctly; mark mirror-target direction cells (keys of `mirror_map`) with a distinct badge; add a per-character `sampled=N mirrored=M` tally.

- [ ] **Step 4: Run tests + manual**

Run: `pytest tests/test_api.py -q` (green). Then open the panel: spaced ids group; mirror cells badged; tally shows. Record manual result in the commit body.

- [ ] **Step 5: Commit**

```bash
git add web/anim_coord_panel.js andypack/api.py tests/test_api.py
git commit -m "feat: space-safe coverage keys + sampled-vs-mirrored tally"
```

---

## Group 7 — Docs + examples

### Task 41: Update CLAUDE.md / README / prompting-guide

**Files:**
- Modify: `CLAUDE.md`, `README.md`, `docs/prompting-guide.md`
- Test: none (docs); `ruff`/`mypy`/`pytest` still green.

- [ ] **Step 1: Update CLAUDE.md**

- Bump the node count (17 then new total) in the module map.
- Add the new categories (`andypack/Sprite`, `andypack/Export`) and every new node to the `nodes.py` description.
- Add `andypack/sprites.py` and `andypack/atlas.py` to the module map.
- Add an "Alpha boundary" invariant: tensors stay 3-ch in the graph; RGBA materializes only at the writer/pack disk boundary (4-ch input or a `MASK` input); `has_alpha` recorded in sidecar/meta.
- Update the staleness invariant to note anchor-identity drift (dep key-set) is now a staleness trigger (#1).
- Update the HTTP-routes invariant to note the thumbnail route + GET path-safety gating (#4/#5).

- [ ] **Step 2: Update README.md**

Add a "Game-asset / sprite export" section: the alpha then trim then pack then atlas-export chain, the turnaround/identity/state-machine nodes, the render-economy nodes, and the mirror economy (`skip_mirrored` + batch mirror).

- [ ] **Step 3: Update docs/prompting-guide.md**

Add a short note on producing alpha cutouts upstream (bring-your-own bg-removal) feeding the writers, and palette-locking for pixel-art consistency.

- [ ] **Step 4: Verify green + commit**

Run: `pytest -q && ruff check . && mypy andypack`
Expected: all green.

```bash
git add CLAUDE.md README.md docs/prompting-guide.md
git commit -m "docs: document game-asset nodes, alpha boundary, and moved invariants"
```

---

### Task 42: Example workflows

**Files:**
- Create: `examples/workflows/sprite_export.json`, `examples/workflows/turnaround.json`
- Test: JSON-parse check.

- [ ] **Step 1: Author the sprite-export graph**

A graph wiring: Manifest Loader then Character Animation Selector then Unpack Animation then (user sampler) then Animation Frame Writer (with mask) then Sprite Trim & Pivot then Spritesheet Packer then Atlas Metadata Writer. Save as `examples/workflows/sprite_export.json`.

- [ ] **Step 2: Author the turnaround graph**

Manifest Loader then Turnaround Sheet (+ Character Identity Anchor demo). Save as `examples/workflows/turnaround.json`.

- [ ] **Step 3: Validate JSON + commit**

Run: `python -c "import json,glob; [json.load(open(f)) for f in glob.glob('examples/workflows/*.json')]"`
Expected: no error.

```bash
git add examples/workflows/
git commit -m "docs: example workflows for sprite export and turnaround"
```

---

## Final verification

- [ ] Run the full suite: `pytest -q` — all green.
- [ ] `ruff check .` — clean.
- [ ] `mypy andypack` — clean.
- [ ] Confirm `resolve.py`/`manifest.py`/`io.py` still import without torch (existing guard test).
- [ ] Confirm the Unpack-key sync test still passes for `ANIM_POSE`/`ANIM_ANIMATION`.
- [ ] Smoke-test in ComfyUI: load a character, render base, mirror, build an atlas, export it; open the panel and confirm thumbnails + tally.

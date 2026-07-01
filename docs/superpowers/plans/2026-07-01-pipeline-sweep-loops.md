# One-press Sweep Loops Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fill a whole pose/animation sweep in one Queue press, and spot-fix individual cells, while keeping the four-workflow shape.

**Architecture:** Add an andypack While-loop bracket (built on ComfyUI 0.26.2 node-expansion) that re-runs the render body until a `REMAINING` count hits zero, and collapse each stage's Auto/Character selectors into one `mode: sweep|target` selector. The loop's continue-signal is recomputed by the writer after each render (dependency depth means it can't be counted upfront). The pure resolve/api core stays torch-free and ComfyUI-free.

**Tech Stack:** Python 3.10–3.12, pytest, ComfyUI 0.26.2 custom nodes, `comfy_execution.graph_utils` (node expansion).

## Global Constraints

- Test: `pytest -q` · Lint: `ruff check .` · Types: `mypy andypack` (all must stay green).
- `resolve.py`, `manifest.py`, `atlas.py` stay free of ComfyUI/torch imports. `io.py` stays torch-free.
- Node count stays at 20: this plan removes 4 selectors and adds 4 nodes (net 0).
- Every animation must have a START image; FFLF cross-wiring (`start_from`→last frame, `end_at`→first frame) is unchanged.
- Selectors' `IS_CHANGED` must keep returning `float("nan")` (disk re-read each execution — the loop depends on it).
- Atomic write ordering unchanged (drop sidecar, write payload, write sidecar last).
- Commit after each task. Work on branch `feat/sweep-loops` (already created).

---

## Task 1: Spike — validate the loop-node contract on ComfyUI 0.26.2

**Why first:** The Open/Close node-expansion contract (how a flow token threads through an `OUTPUT_NODE` body, and where `REMAINING` is sampled so continuation reflects post-write state) is the one real unknown. Validate it with a throwaway before building real nodes. This task produces a findings note and a minimal working loop; its node code is the reference Task 7 productizes.

**Files:**
- Create: `andypack/_spike_loop.py` (throwaway; deleted in Task 7)
- Create: `docs/superpowers/notes/2026-07-01-loop-spike-findings.md`

**Interfaces:**
- Produces (for Task 7): a validated `SweepLoopOpen`/`SweepLoopClose` expansion pattern — the exact `GraphBuilder`/`dynprompt` calls that (a) run the body's output node each iteration, (b) re-read `IS_CHANGED=NaN` nodes per iteration, (c) terminate on `remaining <= 0`.

- [ ] **Step 1: Write a minimal expansion loop**

Create `andypack/_spike_loop.py` with a counter body (no models). Open emits a `SWEEP_FLOW` token; a body node increments a JSON counter file and returns the new count as `remaining_seed - i`; Close expands the Open→Close subgraph while `remaining > 0`.

```python
# andypack/_spike_loop.py  (THROWAWAY — remove in Task 7)
from comfy_execution.graph_utils import GraphBuilder

class SpikeLoopOpen:
    CATEGORY = "andypack/_spike"
    FUNCTION = "open"
    RETURN_TYPES = ("SWEEP_FLOW",)
    RETURN_NAMES = ("flow",)
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"iterations": ("INT", {"default": 3})}}
    def open(self, iterations):
        return ({"iterations": iterations},)

class SpikeLoopClose:
    CATEGORY = "andypack/_spike"
    FUNCTION = "close"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("done",)
    OUTPUT_NODE = True
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"flow": ("SWEEP_FLOW",), "remaining": ("INT", {"forceInput": True})},
            "hidden": {"dynprompt": "DYNPROMPT", "unique_id": "UNIQUE_ID"},
        }
    def close(self, flow, remaining, dynprompt=None, unique_id=None):
        if remaining <= 0:
            return ("done",)
        graph = GraphBuilder()
        # Clone every node on the path from this Close back to its Open, rewire the
        # cloned Close's `remaining` input to the cloned body output, and expand.
        # (This is the exact mechanic to VALIDATE — see reference note in findings.)
        ...
        return {"result": ("looping",), "expand": graph.finalize()}

NODE_CLASS_MAPPINGS = {"SpikeLoopOpen": SpikeLoopOpen, "SpikeLoopClose": SpikeLoopClose}
```

Consult the ComfyUI 0.26.2 reference for the exact subgraph-clone mechanic before finalizing (context7: resolve `ComfyUI` docs, query "node expansion GraphBuilder while loop dynprompt"). If a maintained reference loop pack is easier to vendor than to hand-roll the clone, note that in findings and vendor it under `andypack/`.

- [ ] **Step 2: Load it in the live instance and run a 3-iteration counter graph**

Register the spike temporarily (import `_spike_loop` from `andypack/nodes.py`, or drop into `custom_nodes` copy), restart ComfyUI, build a graph: `SpikeLoopOpen(iterations=3) → CounterBody → SpikeLoopClose`, Queue once.
Expected: the counter file reaches 3; ComfyUI logs three body executions; the run ends without error.

- [ ] **Step 3: Confirm IS_CHANGED re-read**

Give `CounterBody` `IS_CHANGED → float("nan")`; confirm it re-executes each iteration (not cached to the first value).
Expected: counter increments each iteration (proves per-iteration re-read — the property the real selectors rely on).

- [ ] **Step 4: Write findings**

Record in `docs/superpowers/notes/2026-07-01-loop-spike-findings.md`: the exact working Open/Close code, the clone mechanic, how `remaining` is threaded, any gotchas (token typing, `forceInput`, hidden inputs), and whether a reference was vendored. This is Task 7's source of truth.

- [ ] **Step 5: Commit**

```bash
git add andypack/_spike_loop.py docs/superpowers/notes/2026-07-01-loop-spike-findings.md
git commit -m "spike: validate ComfyUI 0.26.2 loop-node expansion contract"
```

---

## Task 2: `remaining_actionable` count in the api layer

**Files:**
- Modify: `andypack/api.py` (near `next_actionable`, api.py:467)
- Test: `tests/test_api.py`

**Interfaces:**
- Consumes: `regen_queue`, `_safe_effective`, `stale_locally` (existing).
- Produces: `api.remaining_actionable(manifest, root, character, kind, *, exclude_root=False, category=None, skip_mirrored=False) -> int` — count of currently-actionable cells of `kind`, using the SAME filter as `next_actionable`. Used by the writers (Task 6) to drive loop continuation.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_api.py`:

```python
def test_remaining_actionable_counts_all_actionable_poses(manifest, tree):
    tree.character(concept="x")  # character exists, no poses rendered yet
    # With nothing rendered, every non-root pose that is selectable-now counts.
    first = api.next_actionable(manifest, tree.root, tree.char, "pose", exclude_root=True)
    n = api.remaining_actionable(manifest, tree.root, tree.char, "pose", exclude_root=True)
    assert first is not None
    assert n >= 1
    # Count must equal the number of distinct actionable cells next_actionable would walk.
    seen = 0
    # render each actionable cell in turn; remaining must strictly decrease toward 0
    prev = n
    while api.next_actionable(manifest, tree.root, tree.char, "pose", exclude_root=True):
        job = api.next_actionable(manifest, tree.root, tree.char, "pose", exclude_root=True)
        tree.pose(job["id"], job["direction"])
        seen += 1
        cur = api.remaining_actionable(manifest, tree.root, tree.char, "pose", exclude_root=True)
        assert cur < prev
        prev = cur
    assert prev == 0
    assert seen == n  # the initial count predicted the total drained (flat dependency case)


def test_remaining_actionable_zero_when_nothing_actionable(manifest, tree):
    assert api.remaining_actionable(manifest, tree.root, tree.char, "animation") == 0
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_api.py::test_remaining_actionable_counts_all_actionable_poses -v`
Expected: FAIL with `AttributeError: module 'andypack.api' has no attribute 'remaining_actionable'`.

- [ ] **Step 3: Implement, factoring the shared filter out of `next_actionable`**

In `andypack/api.py`, extract the per-item predicate so both functions share it (DRY), then add the count:

```python
def _actionable_items(manifest, root, character, kind, *,
                      exclude_root=False, category=None, skip_mirrored=False):
    """Yield actionable cells of `kind` in dependency order (the filter that backs
    both next_actionable and remaining_actionable)."""
    eff = _safe_effective(manifest, root, character)
    mirror_keys = set(eff.get("mirror_map", {}).keys()) if skip_mirrored else set()
    for item in regen_queue(eff, root, character):
        if item["kind"] != kind:
            continue
        if exclude_root and kind == "pose":
            if eff.get("poses", {}).get(item["id"], {}).get("from") is None:
                continue
        if category is not None:
            coll = eff.get("poses", {}) if kind == "pose" else eff.get("animations", {})
            if coll.get(item["id"], {}).get("category") != category:
                continue
        if item["direction"] in mirror_keys:
            continue
        if item["status"] == "stale" and not stale_locally(
            eff, root, character, item["id"], item["direction"]
        ):
            continue
        yield item


def next_actionable(manifest, root, character, kind, *,
                    exclude_root=False, category=None, skip_mirrored=False):
    for item in _actionable_items(manifest, root, character, kind,
                                  exclude_root=exclude_root, category=category,
                                  skip_mirrored=skip_mirrored):
        return item
    return None


def remaining_actionable(manifest, root, character, kind, *,
                         exclude_root=False, category=None, skip_mirrored=False) -> int:
    return sum(1 for _ in _actionable_items(
        manifest, root, character, kind,
        exclude_root=exclude_root, category=category, skip_mirrored=skip_mirrored))
```

Keep `next_actionable`'s existing docstring; the behavior is identical (first of the shared iterator).

- [ ] **Step 4: Run tests to verify pass (including the existing next_actionable suite)**

Run: `pytest tests/test_api.py -q`
Expected: PASS (new tests + all existing `next_actionable` tests unchanged).

- [ ] **Step 5: Commit**

```bash
git add andypack/api.py tests/test_api.py
git commit -m "feat(api): remaining_actionable count; share filter with next_actionable"
```

---

## Task 3: Prove the loop drains a multi-level turnaround

**Why:** The loop's correctness rests on "rendering a cell unblocks its dependents, and the writer's re-count picks them up next iteration." Lock this with a test so a refactor can't silently break depth handling.

**Files:**
- Test: `tests/test_api.py`

**Interfaces:**
- Consumes: `api.next_actionable`, `api.remaining_actionable`, `tree.pose` (existing).

- [ ] **Step 1: Write the failing test**

The fixture manifest has derived poses that depend on `base` (a root pose). Simulate the sweep: render `base` for a direction first, then assert a derived pose that was previously blocked becomes actionable and the count reflects it.

```python
def test_sweep_drains_derived_after_base_renders(manifest, tree):
    tree.character(concept="x")
    d = "SOUTH"
    # Render base (root) for the direction — mirrors Character Creator / include_base.
    tree.pose("base", d)
    # A derived pose depending on base should now be actionable for that direction.
    job = api.next_actionable(manifest, tree.root, tree.char, "pose", exclude_root=True)
    assert job is not None, "a derived pose should unblock once base is rendered"
    before = api.remaining_actionable(manifest, tree.root, tree.char, "pose", exclude_root=True)
    tree.pose(job["id"], job["direction"])
    after = api.remaining_actionable(manifest, tree.root, tree.char, "pose", exclude_root=True)
    assert after == before - 1
```

If the fixture's dependency shape doesn't express a base→derived edge for `SOUTH`, pick the actual root/derived ids from `tests/fixtures/manifest.json` and use those (read the file; do not invent ids).

- [ ] **Step 2: Run to verify it fails or passes meaningfully**

Run: `pytest tests/test_api.py::test_sweep_drains_derived_after_base_renders -v`
Expected: PASS if Task 2 is correct; if it FAILS, the depth handling is wrong — fix `_actionable_items`/`regen_queue` usage before proceeding.

- [ ] **Step 3: Commit**

```bash
git add tests/test_api.py
git commit -m "test(api): sweep drains derived poses after base renders"
```

---

## Task 4: Carry sweep-context in the POSE/ANIMATION bundle

**Files:**
- Modify: `andypack/nodes.py` (`POSE_OUTPUT_KEYS`/`ANIMATION_OUTPUT_KEYS` nodes.py:28-38; `_build_pose_bundle` nodes.py:200; `_build_animation_bundle` nodes.py:237)
- Test: `tests/test_nodes.py`

**Interfaces:**
- Produces: bundles carry a `_sweep` dict: `{"character": str, "kind": "pose"|"animation", "mode": "sweep"|"target", "exclude_root": bool, "category": str|None, "skip_mirrored": bool}`. The writer (Task 6) reads `_sweep` to compute mode-aware `REMAINING`. `_sweep` is metadata (leading underscore), NOT a wireable output — like the existing `_meta`.

- [ ] **Step 1: Write the failing test**

```python
def test_pose_bundle_carries_sweep_context(manifest, tree):
    from andypack import nodes
    tree.character(concept="x"); tree.pose("base", "SOUTH")
    job = __import__("andypack.api", fromlist=["x"]).next_actionable(
        manifest, tree.root, tree.char, "pose", exclude_root=True)
    r = __import__("andypack.resolve", fromlist=["x"]).resolve_pose(
        manifest, tree.root, tree.char, job["id"], job["direction"])
    bundle = nodes._build_pose_bundle(
        r, tree.root, tree.char,
        sweep={"character": tree.char, "kind": "pose", "mode": "sweep",
               "exclude_root": True, "category": None, "skip_mirrored": True})
    assert bundle["_sweep"]["mode"] == "sweep"
    assert bundle["_sweep"]["kind"] == "pose"
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_nodes.py::test_pose_bundle_carries_sweep_context -v`
Expected: FAIL (`_build_pose_bundle` takes no `sweep` kwarg / no `_sweep` key).

- [ ] **Step 3: Implement**

Add an optional `sweep=None` kwarg to both builders and store it under `_sweep` (default `{}`). Do not add it to `POSE_OUTPUT_KEYS`/`ANIMATION_OUTPUT_KEYS` (those are the wireable leaf outputs; `_sweep`, like `_meta`, is internal). Example for the pose builder return dict:

```python
def _build_pose_bundle(r, root="", character="", sweep=None):
    ...
    return {
        # ...existing keys...
        "_meta": r["meta"],
        "_sweep": sweep or {},
    }
```

Mirror for `_build_animation_bundle(r, sweep=None)` (add `character`/`root` params if the animation builder needs them for the writer's re-count; thread from the selector).

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_nodes.py -q`
Expected: PASS (existing bundle tests unaffected — `_sweep` defaults to `{}`).

- [ ] **Step 5: Commit**

```bash
git add andypack/nodes.py tests/test_nodes.py
git commit -m "feat(nodes): thread sweep-context through POSE/ANIMATION bundles"
```

---

## Task 5: Unified `PoseSweepSelector` (mode sweep|target)

**Files:**
- Modify: `andypack/nodes.py` (add `PoseSweepSelector`; remove `AutoPoseSelector` nodes.py:660 and `CharacterPoseSelector` nodes.py:302; update `NODE_CLASS_MAPPINGS` nodes.py:1286 and `NODE_DISPLAY_NAME_MAPPINGS` nodes.py:1308)
- Test: `tests/test_nodes.py`

**Interfaces:**
- Consumes: `api.next_actionable`, `resolve_pose`, `effective_manifest`, `_build_pose_bundle(..., sweep=...)` (Task 4).
- Produces: node `PoseSweepSelector` with widgets `character`, `mode` (`["sweep","target"]`), `skip_mirrored` (BOOL), `include_base` (BOOL), `category` (STRING, target-ignored), `target_pose` (STRING), `target_direction` (STRING). Output `("ANIM_POSE",)`. `sweep` → next actionable; `target` → the named `pose@direction`, force-resolved, `_sweep.mode="target"`.

- [ ] **Step 1: Write the failing tests**

```python
def test_pose_sweep_selector_sweep_emits_next_actionable(manifest, tree):
    from andypack import nodes
    tree.character(concept="x"); tree.pose("base", "SOUTH")
    (pose,) = nodes.PoseSweepSelector().select(
        manifest, tree.char, "sweep", True, True, "", "", "")
    assert pose["_sweep"]["mode"] == "sweep"
    assert pose["output_dir"]  # a real actionable cell was resolved


def test_pose_sweep_selector_target_forces_named_cell(manifest, tree):
    from andypack import nodes
    tree.character(concept="x"); tree.pose("base", "SOUTH")
    # pick a real derived pose id from the fixture; here assume "walk" exists
    (pose,) = nodes.PoseSweepSelector().select(
        manifest, tree.char, "target", True, False, "", "walk", "SOUTH")
    assert pose["_sweep"]["mode"] == "target"
    assert pose["_sweep"]["target"] == ("walk", "SOUTH")
```

(Read `tests/fixtures/manifest.json` and substitute a real derived pose id if `walk` is not present.)

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_nodes.py -k pose_sweep_selector -v`
Expected: FAIL (`PoseSweepSelector` undefined).

- [ ] **Step 3: Implement**

```python
class PoseSweepSelector:
    """Sweep or spot-fix the pose turnaround. mode=sweep emits the next actionable
    pose (drive it inside a Sweep Loop to fill everything); mode=target force-
    regenerates the named pose@direction, leaving all others alone."""
    CATEGORY = "andypack/Pose"
    FUNCTION = "select"
    RETURN_TYPES = ("ANIM_POSE",)
    RETURN_NAMES = ("POSE",)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "manifest": ("ANIM_MANIFEST",),
            "character": (_character_choices(),),
            "mode": (["sweep", "target"],),
            "skip_mirrored": ("BOOLEAN", {"default": True}),
            "include_base": ("BOOLEAN", {"default": True}),
            "category": ("STRING", {"default": ""}),
            "target_pose": ("STRING", {"default": ""}),
            "target_direction": ("STRING", {"default": ""}),
        }}

    @classmethod
    def IS_CHANGED(cls, *a, **k):
        return float("nan")  # disk-backed; re-read every execution (loop depends on this)

    def _ctx(self, character, mode, skip_mirrored, include_base, category, tp, td):
        return {"character": character, "kind": "pose", "mode": mode,
                "exclude_root": not include_base, "category": category or None,
                "skip_mirrored": skip_mirrored,
                "target": (tp, td) if mode == "target" else None}

    def select(self, manifest, character, mode, skip_mirrored, include_base,
               category, target_pose, target_direction):
        if character in ("", _NO_CHARACTER):
            raise RuntimeError("PoseSweepSelector: select a character first")
        root = _characters_root()
        manifest = effective_manifest(manifest, root, character)
        ctx = self._ctx(character, mode, skip_mirrored, include_base, category,
                        target_pose, target_direction)
        if mode == "target":
            if not target_pose or not target_direction:
                raise RuntimeError("PoseSweepSelector: target mode needs target_pose and target_direction")
            r = resolve_pose(manifest, root, character, target_pose, target_direction)
            if not r["selectable"]:
                raise RuntimeError(f"pose {target_pose}@{target_direction} blocked_by={r['blocked_by']}")
        else:
            job = api.next_actionable(manifest, root, character, "pose",
                                      exclude_root=not include_base,
                                      category=category or None, skip_mirrored=skip_mirrored)
            if not job:
                raise RuntimeError("PoseSweepSelector: no actionable poses remain")
            r = resolve_pose(manifest, root, character, job["id"], job["direction"])
        return (_build_pose_bundle(r, root, character, sweep=ctx),)
```

Delete `AutoPoseSelector` and `CharacterPoseSelector`. Update both mapping dicts: remove the two old keys, add `"PoseSweepSelector": PoseSweepSelector` and a display name `"Pose Sweep Selector"`.

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_nodes.py -q && ruff check . && mypy andypack`
Expected: PASS/clean. Fix any test that referenced the removed classes (update to `PoseSweepSelector`).

- [ ] **Step 5: Commit**

```bash
git add andypack/nodes.py tests/test_nodes.py
git commit -m "feat(nodes): unify pose selectors into PoseSweepSelector (sweep|target)"
```

---

## Task 6: Unified `AnimationSweepSelector` (mode sweep|target)

**Files:**
- Modify: `andypack/nodes.py` (add `AnimationSweepSelector`; remove `AutoAnimationSelector` nodes.py:734 and `CharacterAnimationSelector` nodes.py:397; update mappings)
- Test: `tests/test_nodes.py`

**Interfaces:**
- Consumes: `api.next_actionable`, `resolve_animation`, `_build_animation_bundle(..., sweep=...)`.
- Produces: node `AnimationSweepSelector`, widgets `character`, `mode`, `skip_mirrored`, `category`, `target_animation`, `target_direction`. Output `("ANIM_ANIMATION",)`. Same sweep/target semantics as Task 5.

- [ ] **Step 1: Write the failing tests** (mirror Task 5 for animations)

```python
def test_animation_sweep_selector_sweep_emits_next(manifest, tree):
    from andypack import nodes
    tree.character(concept="x")
    # render the pose(s) an animation anchors on so it becomes actionable
    # (use the anchor ids the fixture animation references)
    ...
    (anim,) = nodes.AnimationSweepSelector().select(
        manifest, tree.char, "sweep", True, "", "", "")
    assert anim["_sweep"]["mode"] == "sweep"
```

Fill the `...` by reading the fixture to render the required anchor poses (do not invent ids).

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_nodes.py -k animation_sweep_selector -v`
Expected: FAIL (`AnimationSweepSelector` undefined).

- [ ] **Step 3: Implement** (structure identical to `PoseSweepSelector`, using `resolve_animation`, `kind="animation"`, no `include_base`/`exclude_root`)

Delete `AutoAnimationSelector` and `CharacterAnimationSelector`; update both mapping dicts (remove two keys; add `"AnimationSweepSelector"` + display name `"Animation Sweep Selector"`).

- [ ] **Step 4: Run tests**

Run: `pytest -q && ruff check . && mypy andypack`
Expected: PASS/clean.

- [ ] **Step 5: Commit**

```bash
git add andypack/nodes.py tests/test_nodes.py
git commit -m "feat(nodes): unify animation selectors into AnimationSweepSelector (sweep|target)"
```

---

## Task 7: Writers emit mode-aware `REMAINING`

**Files:**
- Modify: `andypack/nodes.py` (`PoseFrameWriter` nodes.py:363; `AnimationFrameWriter`)
- Test: `tests/test_pose_writeback.py`, `tests/test_animation_writeback.py`

**Interfaces:**
- Consumes: `api.remaining_actionable` (Task 2), bundle `_sweep` (Task 4).
- Produces: `PoseFrameWriter.write` and `AnimationFrameWriter.write` return `(output_dir, remaining)`; `RETURN_TYPES = ("STRING", "INT")`, `RETURN_NAMES = ("OUTPUT_DIR", "REMAINING")`. `remaining` = `0` when `_sweep.mode == "target"` (or `_sweep` empty), else `api.remaining_actionable(...)` recomputed post-write from `_sweep` scope. This wire feeds `SweepLoopClose` (Task 8).

- [ ] **Step 1: Write the failing test**

```python
def test_pose_writer_reports_zero_remaining_in_target_mode(manifest, tree, tmp_path):
    from andypack import nodes, images
    tree.character(concept="x"); tree.pose("base", "SOUTH")
    (pose,) = nodes.PoseSweepSelector().select(
        manifest, tree.char, "target", True, False, "", "walk", "SOUTH")
    img = images.empty_image()  # 1x1x3 placeholder tensor
    out_dir, remaining = nodes.PoseFrameWriter().write(pose, img)
    assert remaining == 0  # target mode never continues the loop


def test_pose_writer_reports_positive_remaining_mid_sweep(manifest, tree):
    from andypack import nodes, images
    tree.character(concept="x"); tree.pose("base", "SOUTH")
    (pose,) = nodes.PoseSweepSelector().select(
        manifest, tree.char, "sweep", True, True, "", "", "")
    out_dir, remaining = nodes.PoseFrameWriter().write(pose, images.empty_image())
    assert remaining >= 0  # count of cells still actionable after this write
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_pose_writeback.py -k remaining -v`
Expected: FAIL (`write` returns a 1-tuple; `ValueError: not enough values to unpack`).

- [ ] **Step 3: Implement**

```python
class PoseFrameWriter:
    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("OUTPUT_DIR", "REMAINING")
    # ...INPUT_TYPES unchanged...
    def write(self, pose, image, mask=None):
        # ...existing write body unchanged, producing output_dir...
        remaining = self._remaining(pose)
        return (output_dir, remaining)

    def _remaining(self, pose):
        s = pose.get("_sweep") or {}
        if s.get("mode") != "sweep":
            return 0
        root = _characters_root()
        return api.remaining_actionable(
            _base_manifest_for(s["character"]), root, s["character"], s["kind"],
            exclude_root=s.get("exclude_root", False),
            category=s.get("category"), skip_mirrored=s.get("skip_mirrored", False))
```

`_base_manifest_for` loads the character's effective manifest the same way selectors do (reuse the existing manifest-load path the nodes already use — e.g. `effective_manifest(_load_default_manifest(), root, character)`; wire whatever the selector used). Mirror the change in `AnimationFrameWriter` with `kind="animation"`.

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_pose_writeback.py tests/test_animation_writeback.py -q`
Expected: PASS. Update any existing writer test that unpacked a single return value to expect two.

- [ ] **Step 5: Commit**

```bash
git add andypack/nodes.py tests/test_pose_writeback.py tests/test_animation_writeback.py
git commit -m "feat(nodes): writers emit mode-aware REMAINING for loop continuation"
```

---

## Task 8: Real `SweepLoopOpen` / `SweepLoopClose` nodes

**Files:**
- Create: loop nodes in `andypack/nodes.py` (category `andypack/Loop`); update mappings
- Delete: `andypack/_spike_loop.py`
- Test: manual/live (node expansion is a runtime concern; unit coverage is the Task 1 spike + Task 9 integration)

**Interfaces:**
- Consumes: the validated expansion pattern from Task 1 findings; `REMAINING` (INT) from the writers (Task 7).
- Produces: `SweepLoopOpen` → `("SWEEP_FLOW",)`; `SweepLoopClose(flow, remaining)` → `("STRING",)` `("DONE",)`, `OUTPUT_NODE = True`. Continues while `remaining > 0`.

- [ ] **Step 1: Port the spike into real nodes**

Copy the validated `SpikeLoopOpen`/`SpikeLoopClose` bodies from Task 1 findings into `nodes.py` as `SweepLoopOpen`/`SweepLoopClose`, category `andypack/Loop`, with docstrings describing the sweep-body bracket. Use the EXACT clone/expand mechanic the spike proved — do not re-derive it.

- [ ] **Step 2: Register and delete the spike**

Add `"SweepLoopOpen"`/`"SweepLoopClose"` to both mapping dicts (display names `"Sweep Loop Open"` / `"Sweep Loop Close"`). Delete `andypack/_spike_loop.py` and remove any temporary import of it.

- [ ] **Step 3: Verify import + mappings**

Run: `python -c "import andypack.nodes as n; assert 'SweepLoopOpen' in n.NODE_CLASS_MAPPINGS and 'SweepLoopClose' in n.NODE_CLASS_MAPPINGS and '_spike_loop' not in dir(n)"`
Expected: no error.

- [ ] **Step 4: Lint/type**

Run: `ruff check . && mypy andypack`
Expected: clean. (`comfy_execution` import may need a `# type: ignore`/guarded import like the existing `PromptServer` guard in `server.py` so `mypy`/CI without ComfyUI still passes — follow that pattern.)

- [ ] **Step 5: Commit**

```bash
git add andypack/nodes.py
git rm andypack/_spike_loop.py
git commit -m "feat(nodes): SweepLoopOpen/Close bracket (productized from spike)"
```

---

## Task 9: Rebuild the four example workflows + docs, validate live

**Files:**
- Modify: `examples/workflows/1a_character_create.json`, `1b_turnaround_batch.json`, `2_animate_fflf.json`, `3_sprite_export.json`, `examples/workflows/README.md`
- Modify: `CLAUDE.md` (node list: reflect the selector unification + loop nodes)

**Interfaces:**
- Consumes: all prior tasks (nodes exist and are registered on the live instance after a restart).

- [ ] **Step 1: Rewire `1b_turnaround_batch.json` as a looped sweep**

Replace `AutoPoseSelector` with `PoseSweepSelector` (`mode=sweep`, `include_base=true`), wrap the body in `SweepLoopOpen` → … → `SweepLoopClose`, and wire `PoseFrameWriter.REMAINING → SweepLoopClose.remaining`. Update the Note node to explain "one press fills the turnaround; set mode=target + a pose/direction to spot-fix one."

- [ ] **Step 2: Rewire `2_animate_fflf.json`** the same way with `AnimationSweepSelector` and `AnimationFrameWriter.REMAINING → SweepLoopClose.remaining`.

- [ ] **Step 3: Update `1a` and `3`**

`1a`: swap any removed selector references; keep it as the create/reference workflow (decision on slimming `base@SOUTH` deferred — leave as-is unless told otherwise). `3`: optionally wrap `AnimationSheetBuilder → AtlasMetadataWriter` in the loop to pack all clips (default: leave per-clip, add a Note that a loop variant is possible) — per open question in the spec.

- [ ] **Step 4: Live validation on the instance**

Restart ComfyUI (`restart_comfyui`), load `1b`, pick a character with a persisted reference, Queue once.
Expected: multiple poses render in a single Queue; the run ends cleanly (no red error) when the turnaround is full. Then set `mode=target`, name one `pose@direction`, Queue: exactly that cell is rewritten, others untouched (check mtimes). Repeat the smoke test for `2`.

- [ ] **Step 5: Update docs**

Update `examples/workflows/README.md` (the table + patterns) and `CLAUDE.md`'s node list (Pose/Animation groups: `PoseSweepSelector`/`AnimationSweepSelector` replace the four; add the `andypack/Loop` group). Run `pytest -q && ruff check . && mypy andypack` one final time.

- [ ] **Step 6: Commit**

```bash
git add examples/workflows/ CLAUDE.md docs/
git commit -m "feat: looped sweep example workflows + docs for sweep/target selectors"
```

---

## Self-Review

**Spec coverage:**
- Loop bracket → Tasks 1, 8. Unified sweep/target selectors → Tasks 5, 6. Post-render REMAINING (dependency depth) → Tasks 2, 3, 7. Bundle context → Task 4. `target` runs once (mode-aware REMAINING) → Task 7. `AutoPoseSelector` REMAINING subsumed → Task 5 (node replaced). Net-zero node count → Tasks 5–8. Four-workflow shape + example rebuild → Task 9. Minor/open questions (slim `1a`, export loop) → Task 9 Step 3, deferred per spec.
- Physical model seam (no mega-graph): respected — no task merges FLUX/WAN.

**Placeholder scan:** The `...` in Task 6 Step 1 and Task 9 are explicit "read the fixture / rewire in the canvas" procedures, not code placeholders; each says exactly what to fill and from where. Loop-node internals are intentionally sourced from the Task 1 spike rather than fabricated — this is the honest handling of the one genuine unknown.

**Type consistency:** `remaining_actionable`/`_actionable_items` signatures match across Tasks 2/7. `_sweep` dict keys (`character`, `kind`, `mode`, `exclude_root`, `category`, `skip_mirrored`, `target`) are consistent Tasks 4/5/6/7. Writer return `(OUTPUT_DIR, REMAINING)` consistent Tasks 7/9. `SWEEP_FLOW` token + `SweepLoopClose(flow, remaining)` consistent Tasks 1/8/9.

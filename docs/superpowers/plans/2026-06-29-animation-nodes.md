# Animation Nodes — Implementation Plan (Plan 3 of 5)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the animation-side ComfyUI nodes — `CharacterAnimationSelector` (resolves FFLF anchors + cascading prompts to feed the WAN sampler) and `AnimationFrameWriter` (writes the frame batch + `meta.json` last, with loop closure) — so generating an animation writes back and unlocks its dependents.

**Architecture:** Same thin-wrapper pattern as Plan 2: nodes marshal ComfyUI types and delegate to `andypack.resolve` (Plan 1) and `andypack.io`/`andypack.images` (Plan 2). Loop-closure and meta construction are pure (`io.py`) and unit-tested; the frame-batch iteration lives in the node.

**Tech Stack:** Python ≥3.10, stdlib; `torch` (provided by ComfyUI) for the frame batch; `pytest`/`ruff`/`mypy`.

**Prerequisites:** Plan 1 (resolver) and Plan 2 (`io.py`, `images.py`, `nodes.py`, package mappings) complete and green.

**Source of truth:** `docs/superpowers/specs/2026-06-29-cascading-pose-resolver-design.md` §5 (anchors), §6 (resolve_animation), §7 (nodes).

## Global Constraints

- FFLF never inverted: `start_image` = `start_anchor` (dep's last frame / single image); `end_image` = `end_anchor` (dep's first frame / single image).
- Writer atomicity: write all `frame_{:05d}.png` first, then `meta.json` last via `io.atomic_write_json`. No `.complete`.
- Loop closure runs only when `meta["loop"]` is true; mode is a node input (`drop_last` | `duplicate_first`), default `drop_last`.
- Empty anchors emit `images.empty_image()` plus a `BOOLEAN` presence flag so downstream graphs branch without inspecting tensor shape.
- Every task ends green: `pytest -q`; `ruff check . && mypy andypack`.

---

## File Structure

- `andypack/nodes.py` — append `CharacterAnimationSelector`, `AnimationFrameWriter`; extend mappings.
- `tests/test_animation_writeback.py` — pure write-back acceptance (frames-as-empty-files + meta via `io`).

(No new modules — this plan reuses `io.py`/`images.py`/`resolve.py`.)

---

## Task 1: `CharacterAnimationSelector` node

**Files:**
- Modify: `andypack/nodes.py` (append before mappings)
- Modify: `andypack/nodes.py` mappings

**Interfaces:**
- Consumes: `andypack.resolve.resolve_animation`, `andypack.images.load_image_tensor`/`empty_image`.
- Produces: `CharacterAnimationSelector` → `("IMAGE", "BOOLEAN", "IMAGE", "BOOLEAN", "STRING", "STRING", "STRING", "ANIM_META")` named `(start_image, has_start, end_image, has_end, positive, negative, output_dir, meta)`.

- [ ] **Step 1: Implement `CharacterAnimationSelector` (insert above `NODE_CLASS_MAPPINGS`)**

```python
from andypack.resolve import resolve_animation


class CharacterAnimationSelector:
    CATEGORY = "andypack"
    FUNCTION = "select"
    RETURN_TYPES = ("IMAGE", "BOOLEAN", "IMAGE", "BOOLEAN", "STRING", "STRING", "STRING", "ANIM_META")
    RETURN_NAMES = (
        "start_image", "has_start", "end_image", "has_end",
        "positive", "negative", "output_dir", "meta",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "manifest": ("ANIM_MANIFEST",),
                "root_dir": ("STRING", {"default": "output/anim"}),
                "character": ("STRING", {"default": "Cortex"}),
                "animation": ("STRING", {"default": "fighting_stance_idle"}),
                "direction": ("STRING", {"default": "E"}),
            }
        }

    def _anchor(self, path):
        if path:
            return images.load_image_tensor(path), True
        return images.empty_image(), False

    def select(self, manifest, root_dir, character, animation, direction):
        r = resolve_animation(manifest, root_dir, character, animation, direction)
        if not r["selectable"]:
            raise RuntimeError(
                f"animation {animation}@{direction} not selectable: blocked_by={r['blocked_by']}"
            )
        start_image, has_start = self._anchor(r["start_image"])
        end_image, has_end = self._anchor(r["end_image"])
        return (
            start_image, has_start, end_image, has_end,
            r["positive"], r["negative"], r["output_dir"], r["meta"],
        )
```

- [ ] **Step 2: Register it in the mappings**

Add to both mapping dicts in `andypack/nodes.py`:

```python
    "CharacterAnimationSelector": CharacterAnimationSelector,
```
```python
    "CharacterAnimationSelector": "Character Animation Selector",
```

- [ ] **Step 3: Lint + commit**

```bash
ruff check andypack/nodes.py
git add andypack/nodes.py
git commit -m "feat: CharacterAnimationSelector node (FFLF anchors + cascading prompts)"
```

---

## Task 2: `AnimationFrameWriter` node

**Files:**
- Modify: `andypack/nodes.py` (append before mappings)
- Modify: `andypack/nodes.py` mappings

**Interfaces:**
- Consumes: `andypack.io.frame_name`/`apply_loop_closure`/`build_animation_meta`/`atomic_write_json`, `andypack.images.save_image_png`.
- Produces: `AnimationFrameWriter` → `("STRING",)` named `(output_dir,)`. Writes `frame_{:05d}.png` for each frame, then `meta.json` last.

- [ ] **Step 1: Implement `AnimationFrameWriter` (insert above the mappings)**

```python
class AnimationFrameWriter:
    CATEGORY = "andypack"
    FUNCTION = "write"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_dir",)
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE",),
                "output_dir": ("STRING",),
                "meta": ("ANIM_META",),
            },
            "optional": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "loop_closure": (["drop_last", "duplicate_first"],),
            },
        }

    def write(self, frames, output_dir, meta, seed=0, loop_closure="drop_last"):
        os.makedirs(output_dir, exist_ok=True)
        # frames: IMAGE batch [B, H, W, C] -> list of single-frame tensors
        batch = [frames[i:i + 1] for i in range(frames.shape[0])]
        if meta.get("loop"):
            batch = io.apply_loop_closure(batch, loop_closure)
        for index, frame in enumerate(batch):
            images.save_image_png(frame, os.path.join(output_dir, io.frame_name(index)))
        count = len(batch)
        full_meta = io.build_animation_meta(
            meta,
            count=count,
            start_frame=io.frame_name(0),
            last_frame=io.frame_name(count - 1),
            seed=seed,
            created_utc=_utc_now(),
        )
        io.atomic_write_json(os.path.join(output_dir, "meta.json"), full_meta)
        return (output_dir,)
```

- [ ] **Step 2: Register it in the mappings**

Add to both mapping dicts:

```python
    "AnimationFrameWriter": AnimationFrameWriter,
```
```python
    "AnimationFrameWriter": "Animation Frame Writer",
```

- [ ] **Step 3: Lint + commit**

```bash
ruff check andypack/nodes.py
git add andypack/nodes.py
git commit -m "feat: AnimationFrameWriter node (frames + meta-last, loop closure)"
```

---

## Task 3: Write-back acceptance — generating idle unlocks punch

**Files:**
- Test: `tests/test_animation_writeback.py`

This mirrors the writer's filesystem contract without torch: it writes frames as empty files and `meta.json` via `io.build_animation_meta`, then asserts `resolve.status` flips the dependents.

**Interfaces:**
- Consumes: `andypack.io`, `andypack.resolve.status`/`compute_prompt_hash`.

- [ ] **Step 1: Write the failing acceptance test**

Create `tests/test_animation_writeback.py`:

```python
import os

from andypack import io
from andypack.resolve import compute_prompt_hash, status


def _write_animation(manifest, root, char, anim_id, direction, count):
    out_dir = os.path.join(root, char, anim_id, direction)
    os.makedirs(out_dir, exist_ok=True)
    for i in range(count):
        open(os.path.join(out_dir, io.frame_name(i)), "w").close()
    base_meta = {
        "kind": "animation", "animation": anim_id, "direction": direction,
        "fps": 16, "length": count, "loop": manifest["animations"][anim_id].get("loop", False),
        "manifest_version": manifest["version"],
        "prompt_hash": compute_prompt_hash(manifest, root, char, "animation", anim_id, direction),
    }
    full = io.build_animation_meta(
        base_meta, count=count, start_frame=io.frame_name(0),
        last_frame=io.frame_name(count - 1), seed=1, created_utc="2026-06-29T00:00:00Z",
    )
    io.atomic_write_json(os.path.join(out_dir, "meta.json"), full)


def test_writing_idle_unlocks_punch(manifest, tree):
    root, char = tree.root, tree.char
    tree.concept().pose("base", "E").pose("fighting_stance", "E")
    assert status(manifest, root, char, "punch", "E") == "blocked"

    _write_animation(manifest, root, char, "fighting_stance_idle", "E", count=3)

    assert status(manifest, root, char, "fighting_stance_idle", "E") == "generated"
    for combat in ("punch", "fighting_stance_entry", "fighting_stance_exit"):
        assert status(manifest, root, char, combat, "E") == "ready"
```

- [ ] **Step 2: Run to verify the contract holds**

Run: `pytest tests/test_animation_writeback.py -q`
Expected: PASS. If it FAILS, the defect is in `io.build_animation_meta` or `resolve` — fix the code, not the test.

- [ ] **Step 3: Full gate + commit**

Run: `pytest -q && ruff check . && mypy andypack`
Expected: all green.

```bash
git add tests/test_animation_writeback.py
git commit -m "test: animation write-back unlocks dependents"
```

---

## Self-Review (completed by plan author)

- **Spec coverage:** §6 `resolve_animation` consumption (anchors + merged prompts + meta) → Task 1; §7 `AnimationFrameWriter` write order + loop closure + frame pointers → Task 2; §8 step-4 acceptance ("writing idle flips punch to ready") → Task 3.
- **Placeholder scan:** none; complete code in every step.
- **Type consistency:** `meta` from `CharacterAnimationSelector` is the `resolve_animation` meta dict; `AnimationFrameWriter` reads `meta["loop"]` and passes the dict to `io.build_animation_meta`, which only adds `seed`/`frames`/`start_frame`/`last_frame`/`created_utc`. `frame_name` zero-pads identically in writer and resolver/tests.

## Notes for the implementer

- WAN clips are typically `4n+1` frames; the loader warns (not errors) on other lengths. `drop_last` suits samplers that emit a closing duplicate of frame 0; `duplicate_first` suits samplers that don't. Expose the choice on the node (done) so the user matches their sampler.
- `frames.shape[0]` is the batch size; if a sampler returns a single `[H,W,C]`, wrap upstream — the node assumes the standard `[B,H,W,C]` IMAGE contract.

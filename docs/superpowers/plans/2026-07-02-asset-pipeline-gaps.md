# Asset Pipeline Gaps Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close every gap from the 2026-07-02 pack review: per-direction pose references (replacing hard-coded manikins), mirror writing on the frame writers, a sentinel-free Wan conditioning node, batch Stage-3 export, and all the smaller fixes (alpha exports, labels, seed parity, mask guard, coverage display, retime, loop color-match, palette lock, union trim, doc drift).

**Architecture:** Every feature keeps the pack's layering: pure logic in `resolve.py`/`manifest.py`/`io.py`/`api.py`/`sprites.py`/`atlas.py` (no ComfyUI/torch imports in the first four), torch/PIL work in `images.py`/`sprites.py`, and thin node wrappers in `nodes.py`. New disk state (pose references) lives under ComfyUI's `user/default/andypack/pose_references/`, resolved server-side like the manifests dir. Mirror writes are precomputed at selector time (`_mirror` jobs stashed in the bundle) so writers never need a manifest input.

**Tech Stack:** Python 3.10+ (stdlib + torch + numpy + Pillow), ComfyUI custom-node API, vanilla-JS frontend extensions. Tests: pytest with the existing `tests/conftest.py` fixtures (`manifest`, `tree`/`Tree`).

## Global Constraints

- `resolve.py`, `manifest.py`, `io.py`, `api.py`, `atlas.py` must stay importable with **no ComfyUI or torch imports** (CI runs them headless).
- ComfyUI-only imports in `nodes.py` (e.g. `comfy_extras`, `folder_paths`, `node_helpers`) must be guarded or function-local, exactly like the existing `GraphBuilder` guard at `nodes.py:15-23`.
- After every task: `pytest -q` (309+ tests, all passing), `ruff check .` (clean), `mypy andypack` (clean). Run all three before each commit.
- Atomic write discipline: payload first, sidecar/meta LAST via `io.atomic_write_json` / `io.atomic_write_text`; drop the sidecar FIRST on rewrite.
- ComfyUI IMAGE tensors stay 3-channel inside the graph; RGBA materializes only at the disk/export boundary.
- Every new node class must be added to BOTH `NODE_CLASS_MAPPINGS` and `NODE_DISPLAY_NAME_MAPPINGS` at the bottom of `nodes.py`.
- New node inputs on EXISTING nodes go in the `optional` section (a new `required` input breaks saved workflows).
- Commit per task with conventional-commit messages. Commit messages end with the Co-Authored-By/Claude-Session trailer per repo convention.
- Read `CLAUDE.md` at the repo root before starting — it documents the invariants these tasks must respect.

---

### Task 1: Dead-reference cleanup (culled-node drift)

**Files:**
- Modify: `web/anim_coord.js:20-25` (SELECTOR_NODES)
- Modify: `andypack/nodes.py:1320-1324` (AnimationFrames docstring)
- Modify: `andypack/sprites.py:531-535` (orphaned section header)

**Interfaces:**
- Consumes: nothing.
- Produces: nothing new — removals only. (Task 18 will re-add a palette section to `sprites.py` with real content.)

- [ ] **Step 1: Remove the dead `AnimationPlayback` entry from the frontend selector map**

In `web/anim_coord.js`, delete the line `AnimationPlayback: { idWidget: "animation", kind: "animation" },` from `SELECTOR_NODES` (the node was culled 2026-06-30; the entry is dead config).

- [ ] **Step 2: Rewrite the AnimationFrames docstring to drop culled-node references**

In `andypack/nodes.py`, replace the `AnimationFrames` class docstring (currently mentions `AnimationPlayback` and `Frame Timing Normalizer`, both removed) with:

```python
    """Load a rendered animation clip's frames back as an IMAGE batch (+ its fps) —
    just the raw frames on disk, no dep-chaining or loop semantics. Use it to
    re-process a clip without re-sampling: re-matte, retime (Frame Retime), pack
    (Spritesheet Packer), or re-export."""
```

- [ ] **Step 3: Remove the orphaned palette section header from sprites.py**

At the bottom of `andypack/sprites.py`, delete the empty trailing section:

```python
# ---------------------------------------------------------------------------
# Palette extraction and quantization
# ---------------------------------------------------------------------------
```

(Task 18 reinstates palette support with actual code; leaving an empty header until then is misleading.)

- [ ] **Step 4: Run checks**

Run: `pytest -q && ruff check . && mypy andypack`
Expected: all pass (no behavior changed).

- [ ] **Step 5: Commit**

```bash
git add web/anim_coord.js andypack/nodes.py andypack/sprites.py
git commit -m "chore: remove references to culled nodes (AnimationPlayback, Frame Timing Normalizer, palette header)"
```

---

### Task 2: Coverage Report displays in-node

**Files:**
- Modify: `andypack/nodes.py:867-894` (CoverageReport.report)
- Modify: `web/anim_coord.js` (add `beforeRegisterNodeDef` handler)
- Test: `tests/test_nodes.py` (update `test_coverage_report_node`)

**Interfaces:**
- Consumes: `api.coverage_report`, `api.format_coverage_table` (existing).
- Produces: `CoverageReport.report()` now returns `{"ui": {"text": (table,)}, "result": (table, json_blob)}` instead of a bare tuple. The JS side renders any node in `TEXT_DISPLAY_NODES` (a `Set` in `web/anim_coord.js`) with a read-only multiline widget on execution — Task 17's `SheetExportAll` adds itself to that set.

- [ ] **Step 1: Update the existing test to expect the ui payload**

In `tests/test_nodes.py`, replace `test_coverage_report_node`:

```python
def test_coverage_report_node(manifest, tree, monkeypatch):
    monkeypatch.setattr(nodes, "_characters_root", lambda: tree.root)
    tree.character()
    out = nodes.CoverageReport().report(manifest, tree.char)
    report, blob = out["result"]
    assert "base" in report
    assert json.loads(blob)["total"] > 0
    # The table is also pushed to the frontend so the node shows it inline.
    assert out["ui"]["text"] == (report,)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_nodes.py::test_coverage_report_node -v`
Expected: FAIL — `report()` returns a tuple, `out["result"]` raises `TypeError`.

- [ ] **Step 3: Return the ui payload from the node**

In `andypack/nodes.py`, change `CoverageReport.report`:

```python
    def report(self, manifest, character):
        char = "" if character == _NO_CHARACTER else character
        data = api.coverage_report(manifest, _characters_root(), char)
        table = api.format_coverage_table(data)
        return {"ui": {"text": (table,)}, "result": (table, json.dumps(data, indent=2))}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_nodes.py::test_coverage_report_node -v`
Expected: PASS.

- [ ] **Step 5: Render the text in-node on the frontend**

In `web/anim_coord.js`, add the import at the top (next to the existing imports):

```js
import { ComfyWidgets } from "../../scripts/widgets.js";
```

Add near `SELECTOR_NODES`:

```js
// Nodes whose execution pushes {"ui": {"text": [...]}} — render it in-node as a
// read-only multiline widget (no third-party Show Text pack needed).
const TEXT_DISPLAY_NODES = new Set(["CoverageReport"]);
```

Add a `beforeRegisterNodeDef` handler to the `app.registerExtension({...})` object (alongside `setup`/`nodeCreated`/`loadedGraphNode`):

```js
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!TEXT_DISPLAY_NODES.has(nodeData.name)) return;
    const prev = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      prev?.apply(this, arguments);
      const text = (message?.text || []).join("");
      let w = (this.widgets || []).find((x) => x.name === "display");
      if (!w) {
        w = ComfyWidgets.STRING(this, "display", ["STRING", { multiline: true }], app).widget;
        w.inputEl.readOnly = true;
        w.inputEl.style.fontFamily = "monospace";
        w.inputEl.style.fontSize = "10px";
      }
      w.value = text;
      this.onResize?.(this.size);
    };
  },
```

- [ ] **Step 6: Run checks and commit**

Run: `pytest -q && ruff check . && mypy andypack`
Expected: all pass.

```bash
git add andypack/nodes.py web/anim_coord.js tests/test_nodes.py
git commit -m "feat(diagnostics): show the coverage table inline on the Coverage Report node"
```

---

### Task 3: AtlasMetadataWriter name validation

**Files:**
- Modify: `andypack/nodes.py:990-1035` (AtlasMetadataWriter.export)
- Test: `tests/test_nodes.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `AtlasMetadataWriter.export` raises `RuntimeError` on an empty or path-bearing `name` instead of writing `.png`/`.json` dotfiles.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_nodes.py`:

```python
def test_atlas_writer_rejects_empty_name(tmp_path, monkeypatch):
    monkeypatch.setattr(nodes.api, "output_dir", lambda: str(tmp_path))
    atlas = {"sheet_size": [4, 4], "columns": 1,
             "frames": [{"rect": [0, 0, 4, 4], "source_size": [4, 4],
                         "offset": [0, 0], "pivot": None, "duration_ms": None}]}
    with pytest.raises(RuntimeError, match="name"):
        nodes.AtlasMetadataWriter().export(atlas, _img(4, 4), "json_hash", "")
    with pytest.raises(RuntimeError, match="name"):
        nodes.AtlasMetadataWriter().export(atlas, _img(4, 4), "json_hash", "../evil")
    assert not os.path.exists(os.path.join(str(tmp_path), "atlas", ".png"))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_nodes.py::test_atlas_writer_rejects_empty_name -v`
Expected: FAIL — no RuntimeError raised (files get written with empty basenames).

- [ ] **Step 3: Add the validation**

At the top of `AtlasMetadataWriter.export` in `andypack/nodes.py`, before any write:

```python
        name = (name or "").strip()
        if not name or name != os.path.basename(name) or ".." in name:
            raise RuntimeError(
                "AtlasMetadataWriter: 'name' must be a non-empty bare filename "
                f"(no directory separators), got {name!r}"
            )
```

- [ ] **Step 4: Run tests, checks, commit**

Run: `pytest tests/test_nodes.py::test_atlas_writer_rejects_empty_name -v && pytest -q && ruff check . && mypy andypack`
Expected: all pass.

```bash
git add andypack/nodes.py tests/test_nodes.py
git commit -m "fix(export): reject empty/path-bearing atlas names in Atlas Metadata Writer"
```

---

### Task 4: Pose seed provenance (parity with the animation writer)

**Files:**
- Modify: `andypack/io.py:120-127` (build_pose_sidecar)
- Modify: `andypack/nodes.py:546-577` (PoseFrameWriter)
- Test: `tests/test_io.py`, `tests/test_nodes.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `io.build_pose_sidecar(meta, created_utc, has_alpha=False, seed=None)` — new keyword-only-by-position `seed` param, recorded in the sidecar. `PoseFrameWriter.write(self, pose, image, mask=None, seed=0)` — new optional link-only `seed` INT input (Task 14 later adds `write_mirrored` after it).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_io.py`:

```python
def test_pose_sidecar_records_seed():
    from andypack import io
    meta = {"kind": "pose", "pose": "base", "direction": "EAST", "prompt_hash": "sha1:x"}
    side = io.build_pose_sidecar(meta, created_utc="2026-01-01T00:00:00Z", seed=1234)
    assert side["seed"] == 1234
```

Add to `tests/test_nodes.py`:

```python
def test_pose_writer_records_seed(tmp_path):
    out = str(tmp_path / "_base")
    meta = {"kind": "pose", "pose": "base", "direction": "EAST", "from": None,
            "image": "EAST.png", "manifest_version": 1, "prompt_hash": "sha1:one"}
    nodes.PoseFrameWriter().write(_pose_dict(meta, out), _img(), seed=42)
    side = json.loads(open(os.path.join(out, "EAST.json")).read())
    assert side["seed"] == 42
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_io.py::test_pose_sidecar_records_seed tests/test_nodes.py::test_pose_writer_records_seed -v`
Expected: FAIL — `build_pose_sidecar` takes no `seed`; `write` takes no `seed`.

- [ ] **Step 3: Implement**

In `andypack/io.py`, change `build_pose_sidecar`:

```python
def build_pose_sidecar(
    meta: dict, created_utc: str, has_alpha: bool = False, seed: Optional[int] = None
) -> dict:
    """Pose sidecar = resolve_pose meta + created_utc + render_id + has_alpha + seed."""
    return {
        **meta,
        "seed": seed,
        "created_utc": created_utc,
        "render_id": render_id(meta["prompt_hash"], created_utc),
        "has_alpha": has_alpha,
    }
```

In `andypack/nodes.py`, add to `PoseFrameWriter.INPUT_TYPES` optional (same pattern and comment as AnimationFrameWriter's seed at `nodes.py:708-711`):

```python
            "optional": {
                "mask": ("MASK",),
                # Provenance only: the seed that drove the upstream sampler.
                # forceInput = link-only, so no `control_after_generate` widget
                # can mutate the recorded value out of sync with the sampler.
                "seed": ("INT", {"default": 0, "forceInput": True}),
            },
```

Change the signature and sidecar build:

```python
    def write(self, pose, image, mask=None, seed=0):
        ...
        sidecar = io.build_pose_sidecar(
            meta, created_utc=_utc_now(), has_alpha=has_alpha, seed=seed
        )
```

- [ ] **Step 4: Run tests, checks, commit**

Run: `pytest -q && ruff check . && mypy andypack`
Expected: all pass.

```bash
git add andypack/io.py andypack/nodes.py tests/test_io.py tests/test_nodes.py
git commit -m "feat(pose): record sampler seed in pose sidecars (parity with animation meta)"
```

---

### Task 5: Animation writer mask-batch guard + single-mask broadcast

**Files:**
- Modify: `andypack/nodes.py:715-764` (AnimationFrameWriter.write)
- Test: `tests/test_nodes.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `AnimationFrameWriter.write` raises `RuntimeError` (before touching the prior render) when the mask batch is neither 1 nor the frame count; a 1-frame mask broadcasts to every frame.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_nodes.py`:

```python
def _mask(n, h=2, w=2):
    return torch.ones((n, h, w), dtype=torch.float32)


def test_animation_writer_rejects_mismatched_mask(tmp_path):
    out = str(tmp_path / "punch" / "EAST")
    meta = {"kind": "animation", "animation": "punch", "direction": "EAST",
            "fps": 16, "length": 5, "loop": False, "manifest_version": 1,
            "prompt_hash": "sha1:abc"}
    writer = nodes.AnimationFrameWriter()
    writer.write(_anim_dict(meta, out), _batch(5))  # good prior render
    with pytest.raises(RuntimeError, match="mask batch"):
        writer.write(_anim_dict(meta, out), _batch(5), mask=_mask(3))
    # The guard fires before clearing, so the prior render survives intact.
    assert os.path.exists(os.path.join(out, "meta.json"))
    assert sum(n.startswith("frame_") for n in os.listdir(out)) == 5


def test_animation_writer_broadcasts_single_mask(tmp_path):
    out = str(tmp_path / "punch" / "EAST")
    meta = {"kind": "animation", "animation": "punch", "direction": "EAST",
            "fps": 16, "length": 3, "loop": False, "manifest_version": 1,
            "prompt_hash": "sha1:abc"}
    nodes.AnimationFrameWriter().write(_anim_dict(meta, out), _batch(3), mask=_mask(1))
    full = json.loads(open(os.path.join(out, "meta.json")).read())
    assert full["has_alpha"] is True
    from PIL import Image
    with Image.open(os.path.join(out, "frame_00002.png")) as img:
        assert img.mode == "RGBA"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_nodes.py::test_animation_writer_rejects_mismatched_mask tests/test_nodes.py::test_animation_writer_broadcasts_single_mask -v`
Expected: first FAILS (no RuntimeError — instead an IndexError deep in `_alpha_from_mask`, after the meta was already deleted); second FAILS (IndexError slicing `mask[2:3]` on a 1-frame mask → empty tensor).

- [ ] **Step 3: Implement the guard and broadcast**

In `AnimationFrameWriter.write`, immediately after the existing empty-batch guard (before `os.makedirs`/`remove_if_exists`):

```python
        if mask is not None:
            mask = mask if mask.dim() == 3 else mask.unsqueeze(0)
            if int(mask.shape[0]) not in (1, int(frames.shape[0])):
                raise RuntimeError(
                    f"AnimationFrameWriter: mask batch of {int(mask.shape[0])} frames "
                    f"doesn't match the {int(frames.shape[0])}-frame image batch — "
                    "supply one mask per frame, or a single mask to apply to all"
                )
```

Change the per-frame mask slice in the write loop:

```python
        for index, frame in enumerate(batch):
            if mask is None:
                frame_mask = None
            elif int(mask.shape[0]) == 1:
                frame_mask = mask[0:1]
            else:
                frame_mask = mask[index:index + 1]
```

- [ ] **Step 4: Run tests, checks, commit**

Run: `pytest -q && ruff check . && mypy andypack`
Expected: all pass.

```bash
git add andypack/nodes.py tests/test_nodes.py
git commit -m "fix(animation): validate mask batch length up front; broadcast a single mask"
```

---

### Task 6: Alpha-preserving animated exports (WebP / APNG / GIF)

**Files:**
- Modify: `andypack/images.py:232-293` (the three `save_animated_*` encoders)
- Test: `tests/test_images.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `save_animated_webp/apng/gif` preserve alpha when the batch is 4-channel (RGBA WebP/APNG; palette-transparency GIF). 3-channel input behavior is unchanged. Internal helper `_pil_frames(frames) -> list[Image.Image]`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_images.py`:

```python
def _rgba_batch(n=2, h=8, w=8):
    """RGBA batch: left half opaque red, right half fully transparent."""
    t = torch.zeros((n, h, w, 4), dtype=torch.float32)
    t[..., 0] = 1.0                 # red
    t[..., : , : w // 2, 3] = 1.0   # left half opaque
    return t


def test_animated_webp_preserves_alpha(tmp_path):
    path = str(tmp_path / "clip.webp")
    images.save_animated_webp(_rgba_batch(), path, fps=8)
    from PIL import Image
    with Image.open(path) as img:
        assert img.mode in ("RGBA", "P") and "A" in img.convert("RGBA").getbands()
        rgba = img.convert("RGBA")
        assert rgba.getpixel((7, 0))[3] == 0      # right half transparent
        assert rgba.getpixel((0, 0))[3] == 255    # left half opaque


def test_animated_apng_preserves_alpha(tmp_path):
    path = str(tmp_path / "clip.png")
    images.save_animated_apng(_rgba_batch(), path, fps=8)
    from PIL import Image
    with Image.open(path) as img:
        rgba = img.convert("RGBA")
        assert rgba.getpixel((7, 0))[3] == 0


def test_animated_gif_preserves_alpha(tmp_path):
    path = str(tmp_path / "clip.gif")
    images.save_animated_gif(_rgba_batch(), path, fps=8)
    from PIL import Image
    with Image.open(path) as img:
        assert "transparency" in img.info
        rgba = img.convert("RGBA")
        assert rgba.getpixel((7, 0))[3] == 0


def test_animated_webp_rgb_unchanged(tmp_path):
    # 3-channel input still writes a plain RGB animation (original behavior).
    path = str(tmp_path / "clip.webp")
    images.save_animated_webp(torch.zeros((2, 8, 8, 3)), path, fps=8)
    from PIL import Image
    with Image.open(path) as img:
        assert img.convert("RGBA").getpixel((0, 0))[3] == 255
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_images.py -k "preserves_alpha" -v`
Expected: FAIL — transparent pixels come back opaque (alpha is sliced off).

- [ ] **Step 3: Implement the shared PIL-frame helper and rewrite the encoders**

In `andypack/images.py`, add above `save_animated_webp`:

```python
def _pil_frames(frames: torch.Tensor) -> list["Image.Image"]:
    """An IMAGE batch as PIL frames — RGBA when the batch carries 4 channels,
    RGB otherwise. The disk boundary is where alpha materializes; everything
    upstream in the graph stays 3-channel."""
    arr = (frames.clamp(0.0, 1.0).cpu().numpy() * 255.0).round().astype(np.uint8)
    mode = "RGBA" if int(frames.shape[-1]) == 4 else "RGB"
    return [Image.fromarray(a, mode=mode) for a in arr]
```

Rewrite the three encoders to use it (keeping signatures, makedirs, duration, and loop-count behavior identical):

```python
def save_animated_webp(
    frames: torch.Tensor, path: str, fps: int, loop: bool = True
) -> None:
    """Encode an IMAGE batch [N, H, W, C] as an animated WEBP at `path`, played at
    `fps`. RGBA batches keep their alpha channel. A single frame writes a still
    WEBP. `loop` True = repeat forever (loop count 0), False = play once."""
    pil = _pil_frames(frames)
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    duration = int(round(1000.0 / max(int(fps), 1)))
    pil[0].save(
        path, format="WEBP", save_all=True, append_images=pil[1:],
        duration=duration, loop=0 if loop else 1, quality=80, method=4,
    )


def save_animated_apng(
    frames: torch.Tensor, path: str, fps: int, loop: bool = True
) -> None:
    """Encode an IMAGE batch as an animated PNG (APNG). RGBA batches keep their
    alpha channel. `loop` True = repeat forever, False = play once."""
    pil = _pil_frames(frames)
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    duration = int(round(1000.0 / max(int(fps), 1)))
    pil[0].save(
        path, format="PNG", save_all=True, append_images=pil[1:],
        duration=duration, loop=0 if loop else 1,
    )


def _gif_frame(img: "Image.Image") -> "tuple[Image.Image, Optional[int]]":
    """Quantize one frame for GIF. RGBA frames reserve palette index 255 for
    fully-transparent pixels (GIF has 1-bit transparency); returns the paletted
    frame and the transparency index (None for opaque frames)."""
    if img.mode != "RGBA":
        return img.convert("RGB").convert("P", palette=Image.ADAPTIVE), None
    alpha = img.getchannel("A")
    p = img.convert("RGB").convert("P", palette=Image.ADAPTIVE, colors=255)
    transparent = alpha.point(lambda a: 255 if a < 128 else 0)
    p.paste(255, transparent)
    return p, 255


def save_animated_gif(
    frames: torch.Tensor, path: str, fps: int, loop: bool = True
) -> None:
    """Encode an IMAGE batch as an animated GIF. RGBA batches get palette
    transparency (alpha < 0.5 -> fully transparent; GIF has no partial alpha).
    `loop` True = repeat forever, False = play once."""
    quantized = [_gif_frame(f) for f in _pil_frames(frames)]
    pil = [q for q, _t in quantized]
    transparency = quantized[0][1]
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    duration_ms = int(round(1000.0 / max(int(fps), 1)))
    kwargs: dict = {}
    if transparency is not None:
        kwargs["transparency"] = transparency
    pil[0].save(
        path, format="GIF", save_all=True, append_images=pil[1:],
        duration=duration_ms, loop=0 if loop else 1, disposal=2, **kwargs,
    )
```

- [ ] **Step 4: Run tests, checks, commit**

Run: `pytest tests/test_images.py -v && pytest -q && ruff check . && mypy andypack`
Expected: all pass. (`AnimatedSpriteExport` needs no change — it passes the tensor straight through, so RGBA batches from `AnimationFrames`' `keep_alpha=True` loads now export with transparency.)

```bash
git add andypack/images.py tests/test_images.py
git commit -m "fix(export): preserve alpha in animated WebP/APNG/GIF exports"
```

---

### Task 7: Contact-sheet labels (TurnaroundSheet include_labels)

**Files:**
- Modify: `andypack/images.py:338-389` (contact_sheet)
- Test: `tests/test_images.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `images.contact_sheet(tiles, columns, cell=None, labels=None)` — `labels` now draws each string onto its cell (top-left, white text with black stroke). `TurnaroundSheet` already passes `labels`; no node change needed.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_images.py`:

```python
def test_contact_sheet_draws_labels():
    tiles = [torch.zeros((1, 32, 32, 3)), None]
    plain = images.contact_sheet(tiles, columns=2)
    labeled = images.contact_sheet(tiles, columns=2, labels=["EAST", "WEST"])
    assert labeled.shape == plain.shape
    # Drawing text must actually change pixels (the widget can't be a no-op).
    assert not torch.equal(plain, labeled)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_images.py::test_contact_sheet_draws_labels -v`
Expected: FAIL — labels are ignored, tensors identical.

- [ ] **Step 3: Implement label drawing**

In `andypack/images.py` `contact_sheet`, replace the docstring line `labels: reserved for future caption support; currently ignored.` with `labels: optional per-tile caption strings, drawn at each cell's top-left (white with a black stroke). Extra labels beyond len(tiles) are ignored.` — then, just before the final `return sheet`, add:

```python
    if labels:
        from PIL import ImageDraw

        arr = (sheet[0].clamp(0.0, 1.0).cpu().numpy() * 255.0).round().astype(np.uint8)
        pil = Image.fromarray(arr, mode="RGB")
        draw = ImageDraw.Draw(pil)
        for idx in range(min(n, len(labels))):
            row, col = divmod(idx, columns)
            draw.text(
                (col * cell_w + 4, row * cell_h + 4), str(labels[idx]),
                fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0),
            )
        sheet = torch.from_numpy(
            np.asarray(pil, dtype=np.float32) / 255.0
        ).unsqueeze(0)
    return sheet
```

- [ ] **Step 4: Run tests, checks, commit**

Run: `pytest -q && ruff check . && mypy andypack`
Expected: all pass (TurnaroundSheet's `include_labels` widget now actually does something).

```bash
git add andypack/images.py tests/test_images.py
git commit -m "feat(diagnostics): implement contact-sheet direction labels (TurnaroundSheet include_labels)"
```

---

### Task 8: Frame Retime node (expose retime_batch)

**Files:**
- Modify: `andypack/nodes.py` (new class + registration)
- Test: `tests/test_nodes.py`

**Interfaces:**
- Consumes: `images.retime_batch(frames, target, mode)` (existing, tested).
- Produces: node class `FrameRetime` — `retime(self, frames, fps, target_fps, mode) -> (IMAGE, INT)`, registered as `"FrameRetime"` / display `"Frame Retime"`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_nodes.py`:

```python
def test_frame_retime_resamples_to_target_fps():
    frames, fps = nodes.FrameRetime().retime(_batch(16), fps=16, target_fps=8, mode="resample")
    assert int(frames.shape[0]) == 8
    assert fps == 8


def test_frame_retime_upsamples():
    frames, fps = nodes.FrameRetime().retime(_batch(4), fps=8, target_fps=16, mode="resample")
    assert int(frames.shape[0]) == 8
    assert fps == 16
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_nodes.py -k frame_retime -v`
Expected: FAIL — `nodes.FrameRetime` doesn't exist.

- [ ] **Step 3: Implement the node**

Add to `andypack/nodes.py` (near `AnimatedSpriteExport`):

```python
class FrameRetime:
    """Retime an IMAGE batch to a target fps (uniform resample / trim / pad-hold).
    Wan renders natively at 16fps; game sprites often want 8-12. Wire the source
    FPS from Animation Frames / Unpack Animation, pick a target, and feed the
    result to the packer/exporter with the new FPS."""

    CATEGORY = "andypack/Sprite"
    FUNCTION = "retime"
    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("FRAMES", "FPS")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE",),
                "fps": ("INT", {"default": 16, "min": 1, "forceInput": True}),
                "target_fps": ("INT", {"default": 12, "min": 1, "max": 120}),
                "mode": (["resample", "trim", "pad_hold"],),
            }
        }

    def retime(self, frames, fps, target_fps, mode):
        n = int(frames.shape[0])
        target = max(1, round(n * int(target_fps) / max(int(fps), 1)))
        return (images.retime_batch(frames, target, mode), int(target_fps))
```

Register in both mappings at the bottom of `nodes.py`:

```python
    "FrameRetime": FrameRetime,
```
```python
    "FrameRetime": "Frame Retime",
```

- [ ] **Step 4: Run tests, checks, commit**

Run: `pytest -q && ruff check . && mypy andypack`
Expected: all pass.

```bash
git add andypack/nodes.py tests/test_nodes.py
git commit -m "feat(sprite): Frame Retime node — resample a clip to a target fps"
```

---

### Task 9: Loop-seam color match

**Files:**
- Modify: `andypack/images.py` (new `match_color_ramp`)
- Modify: `andypack/nodes.py` (AnimationFrameWriter `loop_color_match` flag)
- Test: `tests/test_images.py`, `tests/test_nodes.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `images.match_color_ramp(frames, reference, strength=1.0) -> Tensor` — per-channel mean/std match toward `reference`, linearly ramped from 0 at frame 0 to `strength` at the final frame. `AnimationFrameWriter` gains optional `loop_color_match` BOOLEAN (default False), applied only when the resolver derived `loop`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_images.py`:

```python
def test_match_color_ramp_pins_last_frame_to_reference():
    # Frames drift brighter; the ramp should pull the LAST frame's stats back to
    # frame 0's, while leaving frame 0 untouched.
    frames = torch.rand((5, 8, 8, 3)) * 0.5
    for i in range(5):
        frames[i] = (frames[i] + i * 0.08).clamp(0, 1)  # progressive brightening
    out = images.match_color_ramp(frames, frames[0:1])
    assert torch.equal(out[0], frames[0])  # start untouched
    drift_before = abs(float(frames[-1].mean()) - float(frames[0].mean()))
    drift_after = abs(float(out[-1].mean()) - float(frames[0].mean()))
    assert drift_after < drift_before * 0.2


def test_match_color_ramp_single_frame_noop():
    frames = torch.rand((1, 4, 4, 3))
    assert torch.equal(images.match_color_ramp(frames, frames[0:1]), frames)
```

Add to `tests/test_nodes.py`:

```python
def test_animation_writer_loop_color_match(tmp_path):
    out = str(tmp_path / "spin" / "EAST")
    meta = {"kind": "animation", "animation": "spin", "direction": "EAST",
            "fps": 16, "length": 5, "loop": True, "manifest_version": 1,
            "prompt_hash": "sha1:abc"}
    frames = torch.rand((5, 8, 8, 3)) * 0.3
    frames[-1] = (frames[-1] + 0.5).clamp(0, 1)  # drifted final frame
    nodes.AnimationFrameWriter().write(
        _anim_dict(meta, out), frames, loop_color_match=True
    )
    from PIL import Image
    import numpy as np
    # loop=True drops the duplicated closing frame -> frames 0..3 written; the
    # matched frame 3 should sit close to frame 0's brightness, not the raw drift.
    with Image.open(os.path.join(out, "frame_00003.png")) as img:
        last_mean = np.asarray(img, dtype=np.float32).mean() / 255.0
    assert abs(last_mean - float(frames[0].mean())) < 0.15
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_images.py -k color_ramp -v`
Expected: FAIL — `images.match_color_ramp` doesn't exist.

- [ ] **Step 3: Implement the helper**

Add to `andypack/images.py` (near `retime_batch`):

```python
def match_color_ramp(
    frames: torch.Tensor, reference: torch.Tensor, strength: float = 1.0
) -> torch.Tensor:
    """Per-channel mean/std color match of each frame toward `reference`, ramped
    linearly from 0 at frame 0 to `strength` at the final frame. Hides the
    loop-seam color drift Wan's low-noise expert introduces on start==end clips
    (see docs/prompting-guide.md) without touching the clip's opening frames.
    Only the RGB channels are matched; alpha passes through untouched."""
    n = int(frames.shape[0])
    if n <= 1:
        return frames
    ref = reference[0] if reference.dim() == 4 else reference
    ref_rgb = ref[..., :3]
    ref_mean = ref_rgb.mean(dim=(0, 1))
    ref_std = ref_rgb.std(dim=(0, 1)).clamp_min(1e-6)
    out = frames.clone()
    for i in range(n):
        weight = float(strength) * (i / (n - 1))
        if weight <= 0.0:
            continue
        f = frames[i, ..., :3]
        mean = f.mean(dim=(0, 1))
        std = f.std(dim=(0, 1)).clamp_min(1e-6)
        matched = (f - mean) / std * ref_std + ref_mean
        out[i, ..., :3] = ((1.0 - weight) * f + weight * matched).clamp(0.0, 1.0)
    return out
```

- [ ] **Step 4: Wire the flag into the animation writer**

In `AnimationFrameWriter.INPUT_TYPES` optional, add:

```python
                # Loop-seam mitigation: ramp a per-channel color match toward the
                # FIRST frame across the clip, so a start==end loop's drifted
                # closing frames land back on the opening palette. Applied only
                # when the resolver derived `loop` (no-op for non-loop clips).
                "loop_color_match": ("BOOLEAN", {"default": False}),
```

In `write` (signature becomes `def write(self, animation, frames, seed=0, mask=None, loop_color_match=False):`), after the mask guard and before the batch split:

```python
        if loop_color_match and meta.get("loop") and int(frames.shape[0]) > 1:
            frames = images.match_color_ramp(frames, frames[0:1])
```

- [ ] **Step 5: Run tests, checks, commit**

Run: `pytest -q && ruff check . && mypy andypack`
Expected: all pass.

```bash
git add andypack/images.py andypack/nodes.py tests/test_images.py tests/test_nodes.py
git commit -m "feat(animation): optional loop-seam color match on the frame writer"
```

---

### Task 10: Per-direction pose references — manifest + resolver plumbing

**Files:**
- Modify: `andypack/manifest.py:31-47` (_validate_directions)
- Modify: `andypack/resolve.py` (new `pose_reference_name`; `resolve_pose` meta; staleness)
- Modify: `andypack/api.py` (new `pose_references_dir`)
- Test: `tests/test_manifest.py`, `tests/test_resolve.py`, `tests/test_staleness.py`

**Interfaces:**
- Consumes: `_is_safe_segment` (manifest.py, existing), `user_default_base` (api.py, existing).
- Produces:
  - Manifest schema: a pose's direction layer may carry `"reference_image": "<bare>.png"` — a filename resolved under the pose-references dir. Validated at load.
  - `resolve.pose_reference_name(manifest, pose_id, direction) -> Optional[str]`.
  - `resolve_pose(...)["meta"]["reference_image"]` — the authored filename or None (flows into sidecars automatically).
  - Staleness: a complete pose whose sidecar-recorded `reference_image` differs from the manifest's current value is stale (both `stale_locally` and `_outdated`).
  - `api.pose_references_dir() -> Optional[str]` = `<user>/default/andypack/pose_references` (None outside ComfyUI).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_manifest.py`:

```python
def test_reference_image_must_be_safe_png():
    from andypack.manifest import ManifestError, validate_manifest
    base = {
        "version": 1, "poses": {
            "base": {"directions": {"EAST": {"reference_image": "../evil.png"}}},
        }, "animations": {},
    }
    with pytest.raises(ManifestError, match="reference_image"):
        validate_manifest(base)
    base["poses"]["base"]["directions"]["EAST"]["reference_image"] = "notpng.jpg"
    with pytest.raises(ManifestError, match="reference_image"):
        validate_manifest(base)
    base["poses"]["base"]["directions"]["EAST"]["reference_image"] = "crouch_EAST.png"
    validate_manifest(base)  # valid: bare *.png filename
```

Add to `tests/test_resolve.py`:

```python
def test_pose_reference_name_and_meta(manifest, tree):
    manifest["poses"]["fighting_stance"]["directions"]["EAST"]["reference_image"] = "fs_EAST.png"
    assert resolve.pose_reference_name(manifest, "fighting_stance", "EAST") == "fs_EAST.png"
    assert resolve.pose_reference_name(manifest, "base", "EAST") is None
    r = resolve.resolve_pose(manifest, tree.root, tree.char, "fighting_stance", "EAST")
    assert r["meta"]["reference_image"] == "fs_EAST.png"
```

Add to `tests/test_staleness.py`:

```python
def test_reference_image_drift_makes_pose_stale(manifest, tree):
    tree.pose("base", "EAST").pose("fighting_stance", "EAST")
    assert not resolve.stale_locally(manifest, tree.root, tree.char, "fighting_stance", "EAST")
    # Authoring a reference_image the render didn't use -> stale (own reasons).
    manifest["poses"]["fighting_stance"]["directions"]["EAST"]["reference_image"] = "fs_EAST.png"
    assert resolve.stale_locally(manifest, tree.root, tree.char, "fighting_stance", "EAST")
    assert resolve.outdated(manifest, tree.root, tree.char, "fighting_stance", "EAST")
```

(Note: `Tree.pose` sidecars carry no `reference_image` key, matching a pre-feature render — `.get("reference_image")` returns None, which must compare unequal to the newly-authored name.)

Add to `tests/test_api.py`:

```python
def test_pose_references_dir_outside_comfyui():
    # No folder_paths outside ComfyUI -> None (same degrade as manifests_dir).
    assert api.pose_references_dir() is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_manifest.py::test_reference_image_must_be_safe_png tests/test_resolve.py::test_pose_reference_name_and_meta tests/test_staleness.py::test_reference_image_drift_makes_pose_stale tests/test_api.py::test_pose_references_dir_outside_comfyui -v`
Expected: FAIL — validation missing, `pose_reference_name` / `pose_references_dir` undefined, meta lacks the key.

- [ ] **Step 3: Implement manifest validation**

In `andypack/manifest.py` `_validate_directions`, inside the `for dname, dlayer in directions.items():` loop after the existing dict check:

```python
        ref_img = dlayer.get("reference_image")
        if ref_img is not None:
            if (
                not isinstance(ref_img, str)
                or not ref_img.endswith(".png")
                or ref_img == ".png"
                or not _is_safe_segment(ref_img)
            ):
                raise ManifestError(
                    f"{label} direction {dname!r} 'reference_image' must be a "
                    f"bare *.png filename (resolved under the pose-references "
                    f"dir), got {ref_img!r}"
                )
```

- [ ] **Step 4: Implement resolver support**

In `andypack/resolve.py`, add near `pose_source_image`:

```python
def pose_reference_name(manifest: Manifest, pose_id: str, direction: str) -> Optional[str]:
    """The per-direction custom pose-reference filename authored on the pose's
    direction layer (`reference_image`), or None. A bare *.png filename the nodes
    resolve under the pose-references dir (`user/default/andypack/pose_references`)
    — it replaces the bundled manikin as the second FLUX-edit reference, and on a
    derived pose it ADDS a second reference where there was none."""
    dlayer = (manifest["poses"][pose_id].get("directions", {}) or {}).get(direction) or {}
    return dlayer.get("reference_image") or None
```

In `resolve_pose`, add to the returned `meta` dict (after `"image"`):

```python
            "reference_image": pose_reference_name(manifest, pose_id, direction),
```

In `stale_locally`, after the `_sources_drifted` check:

```python
    if kind == "pose" and (meta or {}).get("reference_image") != pose_reference_name(
        manifest, ref, direction
    ):
        return True
```

In `_outdated`, after the `_sources_drifted` check (before the `if kind == "pose":` transitive walk):

```python
    if kind == "pose" and (meta or {}).get("reference_image") != pose_reference_name(
        manifest, ref, direction
    ):
        return True
```

- [ ] **Step 5: Implement the api dir helper**

In `andypack/api.py`, add after `manifests_dir`:

```python
def pose_references_dir() -> Optional[str]:
    """The pack's pose-reference dir: `user/default/andypack/pose_references`.
    Holds the per-direction reference images that a pose direction layer's
    `reference_image` names. None when not running in ComfyUI."""
    base = user_default_base()
    if base is None:
        return None
    return os.path.join(base, "andypack", "pose_references")
```

- [ ] **Step 6: Run tests, checks, commit**

Run: `pytest -q && ruff check . && mypy andypack`
Expected: all pass.

```bash
git add andypack/manifest.py andypack/resolve.py andypack/api.py tests/test_manifest.py tests/test_resolve.py tests/test_staleness.py tests/test_api.py
git commit -m "feat(manifest): per-direction pose reference_image — schema, resolve meta, staleness"
```

---

### Task 11: Per-direction pose references — node wiring

**Files:**
- Modify: `andypack/nodes.py:183-252` (`_character_base_pose`, `_build_pose_bundle`; new `_pose_references_root`, `_pose_reference_tensor`)
- Test: `tests/test_nodes.py`

**Interfaces:**
- Consumes: `resolve_pose` meta's `reference_image` (Task 10), `api.pose_references_dir` (Task 10), `manikins.manikin_path` (existing).
- Produces:
  - `nodes._pose_references_root() -> str` (api dir or the relative fallback `user/default/andypack/pose_references` — same degrade pattern as `_characters_root`).
  - `nodes._pose_reference_tensor(meta) -> torch.Tensor` — authored reference (loaded, raises if the file is missing) > bundled manikin (root poses) > empty sentinel (derived poses). Used by both `_build_pose_bundle` and `_character_base_pose`, so custom references flow through Character Creator/Loader, the sweep selector, and Pose Edit Conditioning (which already attaches the second reference whenever `pose_reference` is non-empty).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_nodes.py`:

```python
def _write_ref_png(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    images.save_image_png(_img(4, 4), path)


def test_derived_pose_uses_custom_reference(manifest, tree, tmp_path, monkeypatch):
    monkeypatch.setattr(nodes, "_characters_root", lambda: tree.root)
    refs = str(tmp_path / "pose_refs")
    monkeypatch.setattr(nodes, "_pose_references_root", lambda: refs)
    _write_ref_png(os.path.join(refs, "fs_EAST.png"))
    manifest["poses"]["fighting_stance"]["directions"]["EAST"]["reference_image"] = "fs_EAST.png"
    tree.pose("base", "EAST")
    images.save_image_png(
        _img(), resolve.pose_image_path(tree.root, tree.char, "base", "EAST")
    )
    (pose,) = nodes.PoseSweepSelector().select(
        manifest, tree.char, "target", True, False, "", "fighting_stance", "EAST"
    )
    # Derived pose normally has NO second reference; the authored one adds it.
    assert not images.is_empty(pose["pose_reference"])
    assert pose["_meta"]["reference_image"] == "fs_EAST.png"


def test_missing_custom_reference_raises(manifest, tree, tmp_path, monkeypatch):
    monkeypatch.setattr(nodes, "_characters_root", lambda: tree.root)
    monkeypatch.setattr(nodes, "_pose_references_root", lambda: str(tmp_path / "empty"))
    manifest["poses"]["fighting_stance"]["directions"]["EAST"]["reference_image"] = "gone.png"
    tree.pose("base", "EAST")
    images.save_image_png(
        _img(), resolve.pose_image_path(tree.root, tree.char, "base", "EAST")
    )
    with pytest.raises(RuntimeError, match="gone.png"):
        nodes.PoseSweepSelector().select(
            manifest, tree.char, "target", True, False, "", "fighting_stance", "EAST"
        )


def test_root_pose_still_falls_back_to_manikin(manifest, tree, monkeypatch):
    monkeypatch.setattr(nodes, "_characters_root", lambda: tree.root)
    images.save_image_png(
        _img(), resolve.reference_image_path(tree.root, tree.char)
    )
    (pose,) = nodes.PoseSweepSelector().select(
        manifest, tree.char, "target", True, True, "", "base", "EAST"
    )
    assert not images.is_empty(pose["pose_reference"])  # the bundled manikin
    assert pose["_meta"]["reference_image"] is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_nodes.py -k "custom_reference or falls_back_to_manikin" -v`
Expected: first two FAIL (`_pose_references_root` doesn't exist / derived pose gets the empty sentinel); the manikin fallback test may already pass — keep it as a regression guard.

- [ ] **Step 3: Implement the helpers and rewire the bundle builders**

In `andypack/nodes.py`, add after `_characters_root`:

```python
def _pose_references_root():
    return api.pose_references_dir() or os.path.join(
        "user", "default", "andypack", "pose_references"
    )


def _pose_reference_tensor(meta):
    """The pose-reference IMAGE for a resolved pose meta, in precedence order:
    the authored per-direction `reference_image` (from the pose-references dir),
    else the bundled manikin for a ROOT pose's direction, else the empty sentinel
    (a derived pose stays a single-reference edit). A missing authored file
    raises — silently falling back would bake the wrong reference latent."""
    name = meta.get("reference_image")
    if name:
        path = os.path.join(_pose_references_root(), name)
        if not os.path.exists(path):
            raise RuntimeError(
                f"pose reference {name!r} not found (expected {path}); save it "
                "with the Pose Reference Writer, or remove the direction's "
                "reference_image from the manifest"
            )
        return images.load_image_tensor(path)
    direction = meta.get("direction", "")
    if meta.get("from") is None and direction in manikins.CANONICAL_DIRECTIONS:
        return images.load_image_tensor(manikins.manikin_path(direction))
    return images.empty_image()
```

In `_build_pose_bundle`, replace the root-branch manikin block

```python
        direction = meta.get("direction", "")
        pose_reference = (
            images.load_image_tensor(manikins.manikin_path(direction))
            if direction in manikins.CANONICAL_DIRECTIONS
            else images.empty_image()
        )
```

with

```python
        pose_reference = _pose_reference_tensor(meta)
```

and in the derived (`else`) branch, replace `pose_reference = images.empty_image()` with `pose_reference = _pose_reference_tensor(meta)`.

In `_character_base_pose`, replace

```python
    manikin = images.load_image_tensor(manikins.manikin_path(direction))
```

with

```python
    manikin = _pose_reference_tensor(r["meta"])
```

(the variable name stays; it now honors an authored `reference_image` on `base`'s direction layer too).

- [ ] **Step 4: Run tests, checks, commit**

Run: `pytest -q && ruff check . && mypy andypack`
Expected: all pass.

```bash
git add andypack/nodes.py tests/test_nodes.py
git commit -m "feat(pose): per-direction custom pose references override the bundled manikins"
```

---

### Task 12: Manikin Loader + Pose Reference Writer nodes

**Files:**
- Modify: `andypack/nodes.py` (two new classes + registration)
- Test: `tests/test_nodes.py`

**Interfaces:**
- Consumes: `manikins.manikin_path`, `images.load_image_tensor`, `images.save_image_png`, `io.to_snake_case`, `nodes._pose_references_root` (Task 11).
- Produces:
  - `ManikinLoader.load(direction) -> (IMAGE, STRING)` — the bundled manikin + its direction name. This is the pose/camera source for building CUSTOM references: drive an OpenPose/ControlNet-capable model (FLUX.2 has no ControlNet path — SDXL/SD1.5 openpose or a pose-transfer edit both work) per direction, then save each result with the writer.
  - `PoseReferenceWriter.write(image, name, direction, direction_from="") -> (STRING,)` — writes `<pose_references>/<snake(name)>_<DIRECTION>.png` and returns the exact filename to paste into a manifest direction layer's `reference_image`. `direction_from` (link-only STRING) overrides the combo so ManikinLoader's `DIRECTION` output can drive it and the two nodes can never disagree.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_nodes.py`:

```python
def test_manikin_loader_returns_image_and_direction():
    img, direction = nodes.ManikinLoader().load("EAST")
    assert direction == "EAST"
    assert not images.is_empty(img)


def test_pose_reference_writer_writes_and_names(tmp_path, monkeypatch):
    refs = str(tmp_path / "pose_refs")
    monkeypatch.setattr(nodes, "_pose_references_root", lambda: refs)
    (filename,) = nodes.PoseReferenceWriter().write(_img(4, 4), "Crouch Low", "EAST")
    assert filename == "crouch_low_EAST.png"
    assert os.path.exists(os.path.join(refs, filename))


def test_pose_reference_writer_direction_from_overrides(tmp_path, monkeypatch):
    refs = str(tmp_path / "pose_refs")
    monkeypatch.setattr(nodes, "_pose_references_root", lambda: refs)
    (filename,) = nodes.PoseReferenceWriter().write(
        _img(4, 4), "crouch", "EAST", direction_from="WEST"
    )
    assert filename == "crouch_WEST.png"


def test_pose_reference_writer_rejects_bad_direction(tmp_path, monkeypatch):
    monkeypatch.setattr(nodes, "_pose_references_root", lambda: str(tmp_path))
    with pytest.raises(RuntimeError, match="direction"):
        nodes.PoseReferenceWriter().write(_img(4, 4), "crouch", "EAST", direction_from="SIDEWAYS")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_nodes.py -k "manikin_loader or pose_reference_writer" -v`
Expected: FAIL — classes don't exist.

- [ ] **Step 3: Implement both nodes**

Add to `andypack/nodes.py` (after `PoseEditConditioning`):

```python
class ManikinLoader:
    """Load a bundled manikin (the per-direction camera/body-orientation reference)
    as an IMAGE, plus its direction name. The starting point for authoring CUSTOM
    pose references: drive a pose-capable graph (e.g. an OpenPose ControlNet on an
    SDXL/SD1.5 checkpoint — FLUX.2 Klein has no ControlNet path — or a pose-transfer
    edit) once per direction, then persist each result with Pose Reference Writer."""

    CATEGORY = "andypack/Pose"
    FUNCTION = "load"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("MANIKIN", "DIRECTION")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"direction": (manikins.CANONICAL_DIRECTIONS,)}}

    def load(self, direction):
        return (images.load_image_tensor(manikins.manikin_path(direction)), direction)


class PoseReferenceWriter:
    """Save an IMAGE into the pose-references dir as `<name>_<DIRECTION>.png` —
    exactly the filename a pose direction layer's `reference_image` points at.
    Returns that filename so it can be pasted into the manifest. Wire Manikin
    Loader's DIRECTION output into `direction_from` to keep the loader and writer
    on the same direction automatically (it overrides the combo when connected)."""

    CATEGORY = "andypack/Pose"
    FUNCTION = "write"
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("FILENAME",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "name": ("STRING", {"default": "pose"}),
                "direction": (manikins.CANONICAL_DIRECTIONS,),
            },
            "optional": {
                "direction_from": ("STRING", {"default": "", "forceInput": True}),
            },
        }

    def write(self, image, name, direction, direction_from=""):
        d = direction_from or direction
        if d not in manikins.CANONICAL_DIRECTIONS:
            raise RuntimeError(f"PoseReferenceWriter: unknown direction {d!r}")
        try:
            snake = io.to_snake_case(name)
        except ValueError as exc:
            raise RuntimeError(f"PoseReferenceWriter: {exc}") from exc
        filename = f"{snake}_{d}.png"
        images.save_image_png(image, os.path.join(_pose_references_root(), filename))
        return (filename,)
```

Register both in the two mappings:

```python
    "ManikinLoader": ManikinLoader,
    "PoseReferenceWriter": PoseReferenceWriter,
```
```python
    "ManikinLoader": "Manikin Loader",
    "PoseReferenceWriter": "Pose Reference Writer",
```

- [ ] **Step 4: Run tests, checks, commit**

Run: `pytest -q && ruff check . && mypy andypack`
Expected: all pass.

```bash
git add andypack/nodes.py tests/test_nodes.py
git commit -m "feat(pose): Manikin Loader + Pose Reference Writer for authoring custom pose references"
```

---

### Task 13: Mirror jobs precomputed into the bundles

**Files:**
- Modify: `andypack/resolve.py` (new `mirror_targets`)
- Modify: `andypack/nodes.py` (new `_mirror_jobs`; stash `_mirror` in `PoseSweepSelector.select`, `AnimationSweepSelector.select`, `_character_base_pose`)
- Test: `tests/test_resolve.py`, `tests/test_nodes.py`

**Interfaces:**
- Consumes: `resolve_pose` / `resolve_animation` (existing).
- Produces:
  - `resolve.mirror_targets(manifest, direction) -> list[str]` — mirror_map KEYS whose VALUE is `direction` (the map is `mirrored -> source`, e.g. `{"WEST": "EAST"}`).
  - `nodes._mirror_jobs(manifest, root, character, kind, entity_id, direction) -> list[dict]` — one `{"direction", "meta", "output_dir"}` per mirror target the entity declares, each fully resolved (its OWN prompt hash / sources / output dir), computed at select time so the writer needs no manifest.
  - Bundle key `_mirror` (list, possibly empty) on `ANIM_POSE` / `ANIM_ANIMATION` bundles from both selectors and from `_character_base_pose`. Task 14's writers consume it.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_resolve.py`:

```python
def test_mirror_targets(manifest):
    assert resolve.mirror_targets(manifest, "EAST") == ["WEST"]
    assert resolve.mirror_targets(manifest, "SOUTH") == []
```

Add to `tests/test_nodes.py`:

```python
def test_selector_stashes_mirror_jobs(manifest, tree, monkeypatch):
    monkeypatch.setattr(nodes, "_characters_root", lambda: tree.root)
    # Declare WEST on the pose so the EAST render has a mirror target to fill.
    manifest["poses"]["fighting_stance"]["directions"]["WEST"] = {}
    tree.pose("base", "EAST")
    images.save_image_png(
        _img(), resolve.pose_image_path(tree.root, tree.char, "base", "EAST")
    )
    (pose,) = nodes.PoseSweepSelector().select(
        manifest, tree.char, "target", True, False, "", "fighting_stance", "EAST"
    )
    jobs = pose["_mirror"]
    assert [j["direction"] for j in jobs] == ["WEST"]
    assert jobs[0]["meta"]["direction"] == "WEST"
    assert jobs[0]["meta"]["prompt_hash"]  # WEST's own hash, not EAST's
    assert jobs[0]["output_dir"].endswith("_fighting_stance")


def test_selector_mirror_jobs_empty_when_undeclared(manifest, tree, monkeypatch):
    monkeypatch.setattr(nodes, "_characters_root", lambda: tree.root)
    tree.pose("base", "EAST")
    images.save_image_png(
        _img(), resolve.pose_image_path(tree.root, tree.char, "base", "EAST")
    )
    (pose,) = nodes.PoseSweepSelector().select(
        manifest, tree.char, "target", True, False, "", "fighting_stance", "EAST"
    )
    assert pose["_mirror"] == []  # fixture declares only EAST on fighting_stance
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_resolve.py::test_mirror_targets tests/test_nodes.py -k mirror_jobs -v`
Expected: FAIL — `mirror_targets` undefined; bundles have no `_mirror` key.

- [ ] **Step 3: Implement `mirror_targets` in the resolver**

Add to `andypack/resolve.py` near `resolved_dir`:

```python
def mirror_targets(manifest: Manifest, direction: str) -> list[str]:
    """Directions DERIVED from `direction` via `mirror_map` (the map reads
    mirrored -> source, e.g. {"WEST": "EAST"}). These are the cells a writer can
    fill deterministically by horizontally flipping the source render."""
    mirror_map = manifest.get("mirror_map") or {}
    return [mirrored for mirrored, source in mirror_map.items() if source == direction]
```

- [ ] **Step 4: Implement `_mirror_jobs` and stash it**

Add to `andypack/nodes.py` after `_build_animation_bundle`:

```python
def _mirror_jobs(manifest, root, character, kind, entity_id, direction):
    """Resolved write-jobs for every mirror_map direction derived from `direction`
    that `entity_id` declares: each is {"direction", "meta", "output_dir"}, with
    the MIRRORED direction's own prompt hash / sources / output dir. Precomputed
    at select time and stashed on the bundle (`_mirror`) so the writers can
    mirror-write without a manifest input."""
    collection = manifest["poses"] if kind == "pose" else manifest["animations"]
    entity = collection.get(entity_id, {})
    declared = entity.get("directions", {}) or {}
    jobs = []
    for d in resolve.mirror_targets(manifest, direction):
        if d not in declared:
            continue
        r = (
            resolve_pose(manifest, root, character, entity_id, d)
            if kind == "pose"
            else resolve_animation(manifest, root, character, entity_id, d)
        )
        jobs.append({"direction": d, "meta": r["meta"], "output_dir": r["output_dir"]})
    return jobs
```

In `PoseSweepSelector.select`, change the final return to:

```python
        bundle = _build_pose_bundle(r, root, character, sweep=ctx)
        m = r["meta"]
        bundle["_mirror"] = _mirror_jobs(
            manifest, root, character, "pose", m["pose"], m["direction"]
        )
        return (bundle,)
```

In `AnimationSweepSelector.select`, change the final return to:

```python
        bundle = _build_animation_bundle(r, sweep=ctx)
        m = r["meta"]
        bundle["_mirror"] = _mirror_jobs(
            manifest, root, character, "animation", m["animation"], m["direction"]
        )
        return (bundle,)
```

In `_character_base_pose`, add before `return`:

```python
    pose["_mirror"] = _mirror_jobs(eff, root, char_name, "pose", "base", direction)
```

(where `pose` is the dict literal — assign it to a variable first:)

```python
    pose = {
        "source_image": image,
        "pose_reference": manikin,
        "positive": r["positive"],
        "negative": r["negative"],
        "output_dir": r["output_dir"],
        "_meta": r["meta"],
    }
    pose["_mirror"] = _mirror_jobs(eff, root, char_name, "pose", "base", direction)
    return pose
```

- [ ] **Step 5: Run tests, checks, commit**

Run: `pytest -q && ruff check . && mypy andypack`
Expected: all pass.

```bash
git add andypack/resolve.py andypack/nodes.py tests/test_resolve.py tests/test_nodes.py
git commit -m "feat(mirror): precompute resolved mirror-direction jobs into selector bundles"
```

---

### Task 14: `write_mirrored` flag on both frame writers

**Files:**
- Modify: `andypack/nodes.py` (PoseFrameWriter, AnimationFrameWriter — extract `_write_one`/`_write_clip`, add the flag)
- Test: `tests/test_nodes.py`

**Interfaces:**
- Consumes: bundle `_mirror` jobs (Task 13), `io.build_pose_sidecar(..., seed=)` (Task 4), `images.match_color_ramp` (Task 9).
- Produces:
  - `PoseFrameWriter.write(self, pose, image, mask=None, seed=0, write_mirrored=False)`; `AnimationFrameWriter.write(self, animation, frames, seed=0, mask=None, loop_color_match=False, write_mirrored=False)`.
  - When `write_mirrored` is on, each `_mirror` job gets a horizontally-flipped copy written with the mirrored direction's own resolved meta plus `"mirrored_from": <source direction>` — a real render on disk, so anchors, staleness, coverage, and the sheet builders all see the mirrored cells with zero further changes.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_nodes.py`:

```python
def _mirror_pose_dict(tmp_path):
    """A pose bundle for EAST with one precomputed WEST mirror job."""
    out = str(tmp_path / "_fighting_stance")
    east = {"kind": "pose", "pose": "fighting_stance", "direction": "EAST",
            "from": {"ref": "base"}, "image": "EAST.png",
            "manifest_version": 1, "prompt_hash": "sha1:east"}
    west = {"kind": "pose", "pose": "fighting_stance", "direction": "WEST",
            "from": {"ref": "base"}, "image": "WEST.png",
            "manifest_version": 1, "prompt_hash": "sha1:west"}
    d = _pose_dict(east, out)
    d["_mirror"] = [{"direction": "WEST", "meta": west, "output_dir": out}]
    return d, out


def test_pose_writer_mirrors_flipped_copy(tmp_path):
    d, out = _mirror_pose_dict(tmp_path)
    img = _img(2, 4)
    img[0, 0, 0, 0] = 1.0  # asymmetric marker at x=0
    nodes.PoseFrameWriter().write(d, img, write_mirrored=True)
    assert os.path.exists(os.path.join(out, "EAST.png"))
    assert os.path.exists(os.path.join(out, "WEST.png"))
    west_side = json.loads(open(os.path.join(out, "WEST.json")).read())
    assert west_side["prompt_hash"] == "sha1:west"     # WEST's own hash
    assert west_side["mirrored_from"] == "EAST"
    from PIL import Image
    import numpy as np
    with Image.open(os.path.join(out, "WEST.png")) as im:
        arr = np.asarray(im)
    assert arr[0, 3, 0] == 255 and arr[0, 0, 0] == 0   # marker flipped to x=w-1


def test_pose_writer_mirror_off_by_default(tmp_path):
    d, out = _mirror_pose_dict(tmp_path)
    nodes.PoseFrameWriter().write(d, _img(2, 4))
    assert not os.path.exists(os.path.join(out, "WEST.png"))


def test_animation_writer_mirrors_clip(tmp_path):
    out_e = str(tmp_path / "walk" / "EAST")
    out_w = str(tmp_path / "walk" / "WEST")
    east = {"kind": "animation", "animation": "walk", "direction": "EAST",
            "fps": 16, "length": 3, "loop": False, "manifest_version": 1,
            "prompt_hash": "sha1:east"}
    west = dict(east, direction="WEST", prompt_hash="sha1:west")
    d = _anim_dict(east, out_e)
    d["_mirror"] = [{"direction": "WEST", "meta": west, "output_dir": out_w}]
    nodes.AnimationFrameWriter().write(d, _batch(3), write_mirrored=True)
    for out in (out_e, out_w):
        assert sum(n.startswith("frame_") for n in os.listdir(out)) == 3
        assert os.path.exists(os.path.join(out, "meta.json"))
    west_meta = json.loads(open(os.path.join(out_w, "meta.json")).read())
    assert west_meta["prompt_hash"] == "sha1:west"
    assert west_meta["mirrored_from"] == "EAST"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_nodes.py -k mirror -v`
Expected: mirror-write tests FAIL (`write() got an unexpected keyword argument 'write_mirrored'`).

- [ ] **Step 3: Refactor PoseFrameWriter and add the flag**

Rewrite `PoseFrameWriter` in `andypack/nodes.py`:

```python
class PoseFrameWriter:
    CATEGORY = "andypack/Pose"
    FUNCTION = "write"
    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("OUTPUT_DIR", "REMAINING")
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose": ("ANIM_POSE",),
                "image": ("IMAGE",),
            },
            "optional": {
                "mask": ("MASK",),
                # Provenance only: the seed that drove the upstream sampler.
                # forceInput = link-only, so no `control_after_generate` widget
                # can mutate the recorded value out of sync with the sampler.
                "seed": ("INT", {"default": 0, "forceInput": True}),
                # Also write a horizontally-flipped copy into every mirror_map
                # direction derived from this one (precomputed `_mirror` jobs on
                # the bundle) — each with the MIRRORED direction's own resolved
                # sidecar, so anchors/staleness/coverage see a real render.
                "write_mirrored": ("BOOLEAN", {"default": False}),
            },
        }

    @staticmethod
    def _write_one(output_dir, meta, image, mask, seed):
        has_alpha = mask is not None or int(image.shape[-1]) == 4
        # Re-render discipline: drop the sidecar (completion sentinel) FIRST so an
        # interrupted rewrite reads as incomplete, then payload, then sidecar last.
        png_path = os.path.join(output_dir, meta["image"])
        sidecar_path = os.path.join(output_dir, f"{meta['direction']}.json")
        io.remove_if_exists(sidecar_path)
        images.save_image_png(image, png_path, mask=mask)
        sidecar = io.build_pose_sidecar(
            meta, created_utc=_utc_now(), has_alpha=has_alpha, seed=seed
        )
        io.atomic_write_json(sidecar_path, sidecar)

    def write(self, pose, image, mask=None, seed=0, write_mirrored=False):
        output_dir = pose["output_dir"]
        meta = pose["_meta"]
        self._write_one(output_dir, meta, image, mask, seed)
        if write_mirrored:
            flipped = torch.flip(image, dims=[2])
            flipped_mask = (
                torch.flip(mask, dims=[mask.dim() - 1]) if mask is not None else None
            )
            for job in pose.get("_mirror") or []:
                self._write_one(
                    job["output_dir"],
                    {**job["meta"], "mirrored_from": meta["direction"]},
                    flipped, flipped_mask, seed,
                )
        return (output_dir, _sweep_remaining(pose))
```

- [ ] **Step 4: Refactor AnimationFrameWriter and add the flag**

Rewrite `AnimationFrameWriter.write` (keeping `INPUT_TYPES` from Tasks 5/9 and adding `write_mirrored` to optional with the same comment as the pose writer):

```python
    @staticmethod
    def _write_clip(output_dir, meta, frames, mask, seed):
        has_alpha = mask is not None or int(frames.shape[-1]) == 4
        os.makedirs(output_dir, exist_ok=True)
        # Re-render discipline: drop meta.json (the completion sentinel) FIRST and
        # clear any stale frames so an interrupted rewrite reads as incomplete and
        # a shorter clip can't leave orphan higher-index frames behind. meta.json
        # is written LAST (atomic) below.
        meta_path = os.path.join(output_dir, "meta.json")
        io.remove_if_exists(meta_path)
        io.clear_frames(output_dir)
        batch = [frames[i:i + 1] for i in range(frames.shape[0])]
        # A loop (FFLF start==end) ends on a duplicate of its first frame; drop it
        # so the clip plays seamlessly on repeat. Loop closure only drops from the
        # end (drop_last), so batch[i] always corresponds to frames[i] — mask
        # slicing by `index` stays safe for both paths.
        if meta.get("loop") and len(batch) > 1:
            batch = io.apply_loop_closure(batch, drop_last=True)
        for index, frame in enumerate(batch):
            if mask is None:
                frame_mask = None
            elif int(mask.shape[0]) == 1:
                frame_mask = mask[0:1]
            else:
                frame_mask = mask[index:index + 1]
            images.save_image_png(
                frame, os.path.join(output_dir, io.frame_name(index)), mask=frame_mask
            )
        count = len(batch)
        full_meta = io.build_animation_meta(
            meta, count=count, start_frame=io.frame_name(0),
            last_frame=io.frame_name(count - 1), seed=seed,
            created_utc=_utc_now(), has_alpha=has_alpha,
        )
        io.atomic_write_json(meta_path, full_meta)

    def write(self, animation, frames, seed=0, mask=None,
              loop_color_match=False, write_mirrored=False):
        output_dir = animation["output_dir"]
        meta = animation["_meta"]
        # Reject an empty frame batch up front, before touching the existing render.
        if images.is_empty(frames):
            raise RuntimeError(
                "AnimationFrameWriter: received an empty or 1x1 sentinel frame batch; "
                "nothing to write (check the upstream sampler)")
        if mask is not None:
            mask = mask if mask.dim() == 3 else mask.unsqueeze(0)
            if int(mask.shape[0]) not in (1, int(frames.shape[0])):
                raise RuntimeError(
                    f"AnimationFrameWriter: mask batch of {int(mask.shape[0])} frames "
                    f"doesn't match the {int(frames.shape[0])}-frame image batch — "
                    "supply one mask per frame, or a single mask to apply to all"
                )
        if loop_color_match and meta.get("loop") and int(frames.shape[0]) > 1:
            frames = images.match_color_ramp(frames, frames[0:1])
        self._write_clip(output_dir, meta, frames, mask, seed)
        if write_mirrored:
            flipped = torch.flip(frames, dims=[2])
            flipped_mask = (
                torch.flip(mask, dims=[mask.dim() - 1]) if mask is not None else None
            )
            for job in animation.get("_mirror") or []:
                self._write_clip(
                    job["output_dir"],
                    {**job["meta"], "mirrored_from": meta["direction"]},
                    flipped, flipped_mask, seed,
                )
        return (output_dir, _sweep_remaining(animation))
```

(Note the empty-batch and mask guards moved out of the clip helper so they run once, before the prior primary render is touched — preserving the guard-before-clear behavior the existing tests assert.)

- [ ] **Step 5: Run tests, checks, commit**

Run: `pytest -q && ruff check . && mypy andypack`
Expected: all pass, including all pre-existing writer tests.

```bash
git add andypack/nodes.py tests/test_nodes.py
git commit -m "feat(mirror): write_mirrored flag on both frame writers — flip + write mirror_map directions"
```

---

### Task 15: Wan Animation Conditioning node (sentinel-free FFLF/i2v)

**Files:**
- Modify: `andypack/nodes.py` (new `_wan_end_image` helper + `WanAnimationConditioning` class + registration)
- Test: `tests/test_nodes.py`

**Interfaces:**
- Consumes: `ANIM_ANIMATION` bundle keys `positive`, `negative`, `start_image`, `end_image`, `is_fflf`, `width`, `height`, `length`; `images.is_empty`.
- Produces:
  - `nodes._wan_end_image(animation) -> Optional[Tensor]` — the end anchor, or None for a plain-i2v clip; the 1×1 sentinel NEVER passes through.
  - `WanAnimationConditioning.build(animation, clip, vae, batch_size=1) -> (CONDITIONING, CONDITIONING, LATENT)` — text-encodes both prompts and delegates to core `comfy_extras.nodes_wan.WanFirstLastFrameToVideo` (via `getattr(node, node.FUNCTION)` so a core method rename can't break us), omitting `end_image` when not FFLF. `shift` still wires from Unpack Animation into `ModelSamplingSD3` (model-side, not conditioning-side).

- [ ] **Step 1: Write the failing tests (pure helper — the comfy delegation is runtime-only)**

Add to `tests/test_nodes.py`:

```python
def test_wan_end_image_none_for_plain_i2v():
    anim = {"is_fflf": False, "end_image": images.empty_image()}
    assert nodes._wan_end_image(anim) is None


def test_wan_end_image_none_for_sentinel_even_if_flagged():
    # Defense in depth: a mis-built bundle claiming FFLF with a sentinel end must
    # still resolve to None — the sentinel must never reach the sampler.
    anim = {"is_fflf": True, "end_image": images.empty_image()}
    assert nodes._wan_end_image(anim) is None


def test_wan_end_image_passes_real_anchor():
    end = _img(4, 4)
    anim = {"is_fflf": True, "end_image": end}
    assert nodes._wan_end_image(anim) is end


def test_wan_animation_conditioning_registered():
    assert "WanAnimationConditioning" in nodes.NODE_CLASS_MAPPINGS
    assert nodes.WanAnimationConditioning.RETURN_TYPES == (
        "CONDITIONING", "CONDITIONING", "LATENT"
    )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_nodes.py -k wan -v`
Expected: FAIL — `_wan_end_image` / `WanAnimationConditioning` undefined.

- [ ] **Step 3: Check the core node's actual signature (informational)**

If a ComfyUI checkout is available locally, run:
`grep -n "class WanFirstLastFrameToVideo" -A 40 <comfyui>/comfy_extras/nodes_wan.py`
The core node (ComfyUI ≥ 0.3.x) exposes `FUNCTION = "encode"` with signature `encode(self, positive, negative, vae, width, height, length, batch_size, start_image=None, end_image=None, clip_vision_start_image=None, clip_vision_end_image=None)` returning `(positive, negative, latent)`. The implementation below calls it via `getattr(node, node.FUNCTION)` with keyword args only, so it tolerates parameter reordering; if a kwarg name differs on the installed version, adjust the kwargs dict to match — nothing else changes.

- [ ] **Step 4: Implement**

Add to `andypack/nodes.py` (after `PoseEditConditioning`):

```python
def _wan_end_image(animation: dict):
    """The FFLF end anchor from an ANIMATION bundle, or None for a plain-i2v clip.
    The 1x1 sentinel the bundle carries for a missing `end_at` must NEVER reach
    the sampler — anchoring a clip's final frame to a black pixel — so both the
    is_fflf flag and the sentinel shape are checked."""
    if not animation.get("is_fflf"):
        return None
    end = animation.get("end_image")
    if end is None or images.is_empty(end):
        return None
    return end


class WanAnimationConditioning:
    """One-node Wan 2.2 i2v / FFLF conditioning for an ANIMATION bundle: text-
    encode the merged prompts and delegate to the core WanFirstLastFrameToVideo
    with the bundle's width/height/length and start anchor — omitting `end_image`
    entirely when the clip is not FFLF, so one graph handles FFLF, plain i2v, and
    loop clips without switches or sentinel leaks. Wire `shift` (Unpack Animation)
    into ModelSamplingSD3 as before; outputs feed the dual-expert samplers."""

    CATEGORY = "andypack/Animation"
    FUNCTION = "build"
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "animation": ("ANIM_ANIMATION",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
            },
            "optional": {
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 16}),
            },
        }

    def build(self, animation, clip, vae, batch_size=1):
        try:
            from comfy_extras.nodes_wan import WanFirstLastFrameToVideo
        except Exception as exc:  # pragma: no cover - requires a ComfyUI process
            raise RuntimeError(
                "WanAnimationConditioning needs ComfyUI (comfy_extras.nodes_wan "
                "is unavailable outside a running ComfyUI)"
            ) from exc

        def encode(text):
            return clip.encode_from_tokens_scheduled(clip.tokenize(text))

        kwargs = dict(
            positive=encode(animation["positive"]),
            negative=encode(animation["negative"] or ""),
            vae=vae,
            width=int(animation["width"]),
            height=int(animation["height"]),
            length=int(animation["length"]),
            batch_size=int(batch_size),
            start_image=animation["start_image"],
        )
        end = _wan_end_image(animation)
        if end is not None:
            kwargs["end_image"] = end
        core = WanFirstLastFrameToVideo()
        positive, negative, latent = getattr(core, core.FUNCTION)(**kwargs)
        return (positive, negative, latent)
```

Register:

```python
    "WanAnimationConditioning": WanAnimationConditioning,
```
```python
    "WanAnimationConditioning": "Wan Animation Conditioning",
```

- [ ] **Step 5: Run tests, checks, commit**

Run: `pytest -q && ruff check . && mypy andypack`
Expected: all pass.

```bash
git add andypack/nodes.py tests/test_nodes.py
git commit -m "feat(animation): Wan Animation Conditioning — one-node FFLF/i2v conditioning, sentinel-free"
```

---

### Task 16: Union-trim option on Animation Sheet Builder

**Files:**
- Modify: `andypack/sprites.py` (new `union_trim_rows`)
- Modify: `andypack/nodes.py:1151-1233` (AnimationSheetBuilder — `trim` flag)
- Test: `tests/test_sprites.py`

**Interfaces:**
- Consumes: `images.alpha_bbox` (existing).
- Produces: `sprites.union_trim_rows(rows, threshold=0.03, pad=0) -> rows` — crops EVERY frame of every `(direction, [frames])` row to the single union alpha bbox across all frames, preserving cross-direction registration; degrades to a no-op for 3-channel frames or an empty union. `AnimationSheetBuilder` gains optional `trim` BOOLEAN (default False), applied before `pack_direction_rows`. Task 17's export helper reuses both.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_sprites.py`:

```python
def test_union_trim_rows_crops_to_shared_bbox():
    from andypack import sprites
    def frame(y, x):
        f = torch.zeros((1, 16, 16, 4))
        f[0, y, x, 3] = 1.0  # one opaque pixel
        return f
    rows = [("EAST", [frame(4, 4), frame(4, 10)]), ("SOUTH", [frame(9, 6)])]
    out = sprites.union_trim_rows(rows)
    # Union bbox spans y 4..9, x 4..10 -> crop is 6 tall, 7 wide, for EVERY frame.
    for _name, frames in out:
        for f in frames:
            assert (int(f.shape[1]), int(f.shape[2])) == (6, 7)
    # Registration: EAST frame 0's pixel was at (4,4) -> now at (0,0);
    # SOUTH's was at (9,6) -> now at (5,2). Offsets shift identically.
    assert float(out[0][1][0][0, 0, 0, 3]) == 1.0
    assert float(out[1][1][0][0, 5, 2, 3]) == 1.0


def test_union_trim_rows_noop_without_alpha():
    from andypack import sprites
    rows = [("EAST", [torch.zeros((1, 8, 8, 3))])]
    out = sprites.union_trim_rows(rows)
    assert int(out[0][1][0].shape[1]) == 8  # full frame: nothing to trim
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_sprites.py -k union_trim -v`
Expected: FAIL — `union_trim_rows` undefined.

- [ ] **Step 3: Implement the helper**

Add to `andypack/sprites.py` (before `pack_direction_rows`):

```python
def union_trim_rows(
    rows: list[tuple[str, list[Tensor]]],
    threshold: float = 0.03,
    pad: int = 0,
) -> list[tuple[str, list[Tensor]]]:
    """Crop every frame of every row to the single union alpha bbox computed
    across ALL frames of ALL rows — shrinks sheet cells while keeping every
    direction spatially registered (each frame shifts by the same offset).
    3-channel frames make the union the full frame, so the call degrades to a
    no-op. Frames are assumed to share H/W (they come from one render)."""
    flat = [(f[0] if f.dim() == 4 else f) for _name, frames in rows for f in frames]
    if not flat:
        return rows
    h, w = int(flat[0].shape[0]), int(flat[0].shape[1])
    boxes = [b for b in (images.alpha_bbox(f, threshold) for f in flat) if b is not None]
    if not boxes:
        return rows
    left = max(0, min(b[0] for b in boxes) - pad)
    top = max(0, min(b[1] for b in boxes) - pad)
    right = min(w, max(b[2] for b in boxes) + pad)
    bottom = min(h, max(b[3] for b in boxes) + pad)
    if (left, top, right, bottom) == (0, 0, w, h):
        return rows

    def crop(t: Tensor) -> Tensor:
        f = t[0] if t.dim() == 4 else t
        return f[top:bottom, left:right, :].unsqueeze(0)

    return [(name, [crop(f) for f in frames]) for name, frames in rows]
```

- [ ] **Step 4: Wire the flag into AnimationSheetBuilder**

In `AnimationSheetBuilder.INPUT_TYPES`, add an `optional` section:

```python
            "optional": {
                # Union alpha-trim across ALL directions' frames before packing —
                # shrinks cells while keeping every direction registered. Only
                # meaningful for RGBA renders (writers with a MASK connected).
                "trim": ("BOOLEAN", {"default": False}),
            },
```

Update `IS_CHANGED` signature to `(cls, manifest, character, animation, directions, padding, power_of_two, trim=False)` and append `str(trim)` to `parts`. Update `build` signature to `(self, manifest, character, animation, directions, padding, power_of_two, trim=False)` and add, right before the `fps = ...` line:

```python
        if trim:
            rows = sprites.union_trim_rows(rows)
```

- [ ] **Step 5: Run tests, checks, commit**

Run: `pytest -q && ruff check . && mypy andypack`
Expected: all pass.

```bash
git add andypack/sprites.py andypack/nodes.py tests/test_sprites.py
git commit -m "feat(sprite): union alpha-trim option on Animation Sheet Builder"
```

---

### Task 17: Sheet Export All (Stage-3 batch export)

**Files:**
- Modify: `andypack/nodes.py` (extract `_animation_sheet` + `_write_atlas` helpers; new `SheetExportAll` class; AtlasMetadataWriter + AnimationSheetBuilder reuse the helpers)
- Modify: `web/anim_coord.js` (add `"SheetExportAll"` to `TEXT_DISPLAY_NODES`)
- Test: `tests/test_nodes.py`

**Interfaces:**
- Consumes: `resolve.rendered_directions`, `resolve.animation_fps`, `sprites.pack_direction_rows`, `sprites.union_trim_rows` (Task 16), `_atlas_mod.serialize`, `io.atomic_write_text`, `effective_manifest`.
- Produces:
  - `nodes._animation_sheet(manifest, root, character, animation, dirs, padding, power_of_two, trim) -> Optional[tuple[Tensor, dict, list[str]]]` — `(sheet, atlas, row_dirs)` or None when no direction is rendered.
  - `nodes._write_atlas(output_dir, name, sheet, atlas, fmt) -> None` — sheet PNG + serialized metadata (atomic text write).
  - Node `SheetExportAll.export(manifest, character, directions, format, padding, power_of_two, trim=False, output_subdir="atlas") -> {"ui": {"text": (report,)}, "result": (out_dir, report)}` — one sheet + atlas per animation with ≥1 rendered direction, named `<character>_<animation>`; skipped animations listed in the report (no silent truncation).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_nodes.py`:

```python
def _render_animation(tree, anim_id, direction, frames=2):
    tree.animation(anim_id, direction, frames=frames)
    d = resolve.animation_frame_dir(tree.root, tree.char, anim_id, direction)
    for i in range(frames):  # real PNGs so the sheet builder can load them
        images.save_image_png(_img(4, 4), os.path.join(d, f"frame_{i:05d}.png"))


def test_sheet_export_all_writes_every_rendered_animation(manifest, tree, tmp_path, monkeypatch):
    monkeypatch.setattr(nodes, "_characters_root", lambda: tree.root)
    monkeypatch.setattr(nodes.api, "output_dir", lambda: str(tmp_path / "out"))
    tree.pose("base", "EAST").pose("fighting_stance", "EAST")
    _render_animation(tree, "fighting_stance_idle", "EAST")
    _render_animation(tree, "walk", "EAST")
    out = nodes.SheetExportAll().export(
        manifest, tree.char, "all", "json_hash", 2, False
    )
    out_dir, report = out["result"]
    for aid in ("fighting_stance_idle", "walk"):
        assert os.path.exists(os.path.join(out_dir, f"{tree.char}_{aid}.png"))
        assert os.path.exists(os.path.join(out_dir, f"{tree.char}_{aid}.json"))
    # Unrendered animations are reported, not silently dropped.
    assert "punch" in report and "skipped" in report
    assert out["ui"]["text"] == (report,)


def test_sheet_export_all_raises_when_nothing_rendered(manifest, tree, tmp_path, monkeypatch):
    monkeypatch.setattr(nodes, "_characters_root", lambda: tree.root)
    monkeypatch.setattr(nodes.api, "output_dir", lambda: str(tmp_path / "out"))
    tree.character()
    with pytest.raises(RuntimeError, match="no animation"):
        nodes.SheetExportAll().export(manifest, tree.char, "all", "json_hash", 2, False)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_nodes.py -k sheet_export_all -v`
Expected: FAIL — `SheetExportAll` undefined.

- [ ] **Step 3: Extract the shared helpers**

Add to `andypack/nodes.py` (above `AnimationSheetBuilder`):

```python
def _animation_sheet(manifest, root, character, animation, dirs, padding,
                     power_of_two, trim=False):
    """(sheet, atlas, row_dirs) for one animation's rendered directions, or None
    when no direction is rendered. Raises when a rendered direction has no frames
    on disk (a corrupt render — meta present, payload missing)."""
    pairs = resolve.rendered_directions(
        manifest, root, character, "animation", animation, dirs
    )
    if not pairs:
        return None
    rows: list[tuple[str, list]] = []
    for d, path in pairs:
        frame_files = sorted(
            n for n in os.listdir(path)
            if n.startswith("frame_") and n.endswith(".png")
        )
        if not frame_files:
            raise RuntimeError(f"{animation!r}@{d} has no frames in {path}")
        rows.append((d, [
            images.load_image_tensor(os.path.join(path, fn), keep_alpha=True)
            for fn in frame_files
        ]))
    if trim:
        rows = sprites.union_trim_rows(rows)
    fps = resolve.animation_fps(manifest, animation)
    sheet, atlas = sprites.pack_direction_rows(
        rows, fps=fps, padding=padding, power_of_two=power_of_two
    )
    return sheet, atlas, [d for d, _f in rows]


def _write_atlas(output_dir, name, sheet, atlas, fmt):
    """Write a sheet PNG + its serialized atlas metadata (payload first, text
    atomic) under `output_dir` as `<name>.png` / `<name><ext>`."""
    images.save_image_png(sheet, os.path.join(output_dir, f"{name}.png"))
    text, ext = _atlas_mod.serialize(atlas, name, fmt)
    io.atomic_write_text(os.path.join(output_dir, f"{name}{ext}"), text)
```

Rewire `AnimationSheetBuilder.build` to use `_animation_sheet` (replacing its inline pairs/rows/fps/pack block — behavior identical, including the no-rendered-directions RuntimeError which now triggers on `None`):

```python
    def build(self, manifest, character, animation, directions, padding,
              power_of_two, trim=False):
        if character in ("", _NO_CHARACTER):
            raise RuntimeError("AnimationSheetBuilder: select a character first")
        if not animation:
            raise RuntimeError("AnimationSheetBuilder: pick an animation id")
        root = _characters_root()
        manifest = effective_manifest(manifest, root, character)
        dirs = _atlas_directions(directions)
        built = _animation_sheet(
            manifest, root, character, animation, dirs, padding, power_of_two, trim
        )
        if built is None:
            raise RuntimeError(
                f"AnimationSheetBuilder: no rendered directions for animation "
                f"{animation!r} (tried: {dirs})"
            )
        sheet, atlas, row_dirs = built
        fps = resolve.animation_fps(manifest, animation)
        report = (
            f"{animation}: {len(row_dirs)} directions @ {fps}fps\n"
            + "Rows: " + ", ".join(row_dirs)
        )
        return {"ui": _image_preview(sheet), "result": (sheet, atlas, report)}
```

Rewire `AtlasMetadataWriter.export`'s sheet+metadata writes to use `_write_atlas` (keeping the Task 3 name validation and the provenance sidecar):

```python
        output_dir = os.path.join(api.output_dir() or "output", output_subdir)
        _write_atlas(output_dir, name, sheet, atlas, format)
```

- [ ] **Step 4: Implement the node**

Add after `AnimationSheetBuilder`:

```python
class SheetExportAll:
    """Stage-3 batch export: build and write a game sheet + atlas for EVERY
    animation in the character's effective manifest that has at least one
    rendered direction — one queue press instead of one Animation Sheet Builder
    run per animation. Files land in `<output>/<output_subdir>/` as
    `<character>_<animation>.png` + metadata. Animations with nothing rendered
    are listed in the report, never silently dropped."""

    CATEGORY = "andypack/Export"
    FUNCTION = "export"
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("OUTPUT_DIR", "REPORT")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "manifest": ("ANIM_MANIFEST",),
                "character": (_character_choices(),),
                "directions": (["all", "cardinal_4"],),
                "format": (["json_hash", "json_array", "aseprite",
                            "godot_spriteframes", "unity", "texturepacker", "css"],),
                "padding": ("INT", {"default": 2, "min": 0}),
                "power_of_two": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "trim": ("BOOLEAN", {"default": False}),
                "output_subdir": ("STRING", {"default": "atlas"}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, *a, **k):
        return float("nan")  # disk-backed: re-export reflects the rendered tree

    def export(self, manifest, character, directions, format, padding,
               power_of_two, trim=False, output_subdir="atlas"):
        if character in ("", _NO_CHARACTER):
            raise RuntimeError("SheetExportAll: select a character first")
        root = _characters_root()
        manifest = effective_manifest(manifest, root, character)
        dirs = _atlas_directions(directions)
        out_dir = os.path.join(api.output_dir() or "output", output_subdir)
        exported: list[str] = []
        skipped: list[str] = []
        for aid in sorted(manifest.get("animations", {})):
            built = _animation_sheet(
                manifest, root, character, aid, dirs, padding, power_of_two, trim
            )
            if built is None:
                skipped.append(aid)
                continue
            sheet, atlas, row_dirs = built
            name = f"{character}_{aid}"
            _write_atlas(out_dir, name, sheet, atlas, format)
            exported.append(f"{aid}: {len(row_dirs)} direction(s) -> {name}.png")
        if not exported:
            raise RuntimeError(
                "SheetExportAll: no animation has any rendered direction — "
                "run the animation sweep first"
            )
        lines = [f"exported {len(exported)} animation(s) to {out_dir}", *exported]
        if skipped:
            lines.append(f"skipped (nothing rendered): {', '.join(skipped)}")
        report = "\n".join(lines)
        return {"ui": {"text": (report,)}, "result": (out_dir, report)}
```

Register:

```python
    "SheetExportAll": SheetExportAll,
```
```python
    "SheetExportAll": "Sheet Export All",
```

In `web/anim_coord.js`, extend the set from Task 2:

```js
const TEXT_DISPLAY_NODES = new Set(["CoverageReport", "SheetExportAll"]);
```

- [ ] **Step 5: Run tests, checks, commit**

Run: `pytest -q && ruff check . && mypy andypack`
Expected: all pass (including the pre-existing AnimationSheetBuilder tests against the refactor).

```bash
git add andypack/nodes.py web/anim_coord.js tests/test_nodes.py
git commit -m "feat(export): Sheet Export All — one-press Stage-3 sheets+atlases for every rendered animation"
```

---

### Task 18: Reinstate Palette Quantize & Lock

**Files:**
- Modify: `andypack/sprites.py` (new `quantize_to_palette`; numpy import)
- Modify: `andypack/nodes.py` (new `PaletteQuantizeLock` class + registration)
- Test: `tests/test_sprites.py`

**Interfaces:**
- Consumes: PIL quantization (`Image.quantize`), numpy.
- Produces:
  - `sprites.quantize_to_palette(image, colors=16, palette_image=None, dither=False) -> (Tensor, Tensor)` — the quantized batch (same shape; alpha untouched) and a palette swatch IMAGE `[1, 16, colors*16, 3]`.
  - Node `PaletteQuantizeLock.quantize(image, colors, dither, palette_image=None) -> (IMAGE, IMAGE)`, display name **"Palette Quantize & Lock"** (matching `docs/prompting-guide.md:182`, which becomes accurate again).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_sprites.py`:

```python
def test_quantize_to_palette_limits_colors():
    from andypack import sprites
    torch.manual_seed(0)
    batch = torch.rand((2, 16, 16, 3))
    out, swatch = sprites.quantize_to_palette(batch, colors=8)
    assert out.shape == batch.shape
    flat = out.reshape(-1, 3)
    unique = {tuple(px.tolist()) for px in (flat * 255).round().to(torch.int32)}
    assert len(unique) <= 8
    assert tuple(swatch.shape) == (1, 16, 8 * 16, 3)


def test_quantize_to_palette_keeps_alpha():
    from andypack import sprites
    batch = torch.rand((1, 8, 8, 4))
    batch[..., 3] = 0.25
    out, _sw = sprites.quantize_to_palette(batch, colors=4)
    assert torch.equal(out[..., 3], batch[..., 3])


def test_quantize_locks_to_palette_image():
    from andypack import sprites
    # A pure-red palette source forces every output pixel to red.
    palette_src = torch.zeros((1, 8, 8, 3))
    palette_src[..., 0] = 1.0
    batch = torch.rand((1, 8, 8, 3))
    out, _sw = sprites.quantize_to_palette(batch, colors=2, palette_image=palette_src)
    flat = (out.reshape(-1, 3) * 255).round().to(torch.int32)
    for px in {tuple(p.tolist()) for p in flat}:
        assert px[1] == 0 and px[2] == 0  # only reds/blacks from the red source
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_sprites.py -k quantize -v`
Expected: FAIL — `quantize_to_palette` undefined.

- [ ] **Step 3: Implement the helper**

In `andypack/sprites.py`, add to the module imports:

```python
import numpy as np
from PIL import Image as PILImage
```

Add at the bottom of the file:

```python
# ---------------------------------------------------------------------------
# Palette extraction and quantization
# ---------------------------------------------------------------------------


def _rgb_to_pil(frame_rgb: Tensor) -> "PILImage.Image":
    arr = (frame_rgb.clamp(0.0, 1.0).cpu().numpy() * 255.0).round().astype(np.uint8)
    return PILImage.fromarray(arr, mode="RGB")


def quantize_to_palette(
    image: Tensor,
    colors: int = 16,
    palette_image: Optional[Tensor] = None,
    dither: bool = False,
) -> tuple[Tensor, Tensor]:
    """Quantize an IMAGE batch [B, H, W, C] to ONE shared palette (pixel-art /
    limited-color consistency). The palette is built from `palette_image` when
    given (LOCK mode — e.g. another direction's frames or a saved swatch, so
    every direction/animation shares identical colors), else from all frames of
    the batch. Alpha (C == 4) passes through untouched. Returns the quantized
    batch (same shape) and a palette swatch IMAGE [1, 16, colors*16, 3]."""
    b = int(image.shape[0])
    colors = max(2, int(colors))
    if palette_image is not None:
        src_frame = palette_image[0] if palette_image.dim() == 4 else palette_image
        src = _rgb_to_pil(src_frame[..., :3])
    else:
        strip = torch.cat([image[i, :, :, :3] for i in range(b)], dim=1)
        src = _rgb_to_pil(strip)
    pal = src.quantize(colors=colors, method=PILImage.Quantize.MEDIANCUT)
    dith = PILImage.Dither.FLOYDSTEINBERG if dither else PILImage.Dither.NONE
    out = image.clone()
    for i in range(b):
        q = _rgb_to_pil(image[i, :, :, :3]).quantize(palette=pal, dither=dith)
        out[i, :, :, :3] = torch.from_numpy(
            np.asarray(q.convert("RGB"), dtype=np.float32) / 255.0
        )
    raw = (pal.getpalette() or [0, 0, 0])[: colors * 3]
    raw = raw + [0] * (colors * 3 - len(raw))  # PIL may return a short palette
    swatch = torch.zeros((1, 16, colors * 16, 3), dtype=torch.float32)
    for i in range(colors):
        r, g, bl = raw[i * 3: i * 3 + 3]
        swatch[0, :, i * 16:(i + 1) * 16, 0] = r / 255.0
        swatch[0, :, i * 16:(i + 1) * 16, 1] = g / 255.0
        swatch[0, :, i * 16:(i + 1) * 16, 2] = bl / 255.0
    return out, swatch
```

- [ ] **Step 4: Implement the node**

Add to `andypack/nodes.py` (near `SpriteTrimPivot`):

```python
class PaletteQuantizeLock:
    """Force a frame batch onto one shared, limited palette (pixel-art / hand-
    painted consistency). Build the palette from the batch itself, or connect
    `palette_image` to LOCK to an external source (another direction's frames, a
    saved swatch) so every direction and animation shares identical colors —
    preventing the per-direction color drift that reads as a glitch in a sprite
    sheet. Alpha passes through untouched. Run after background removal, before
    packing."""

    CATEGORY = "andypack/Sprite"
    FUNCTION = "quantize"
    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("IMAGE", "PALETTE")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "colors": ("INT", {"default": 16, "min": 2, "max": 256}),
                "dither": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "palette_image": ("IMAGE",),
            },
        }

    def quantize(self, image, colors, dither, palette_image=None):
        out, swatch = sprites.quantize_to_palette(
            image, colors=colors, palette_image=palette_image, dither=dither
        )
        return (out, swatch)
```

Register:

```python
    "PaletteQuantizeLock": PaletteQuantizeLock,
```
```python
    "PaletteQuantizeLock": "Palette Quantize & Lock",
```

- [ ] **Step 5: Run tests, checks, commit**

Run: `pytest -q && ruff check . && mypy andypack`
Expected: all pass. `docs/prompting-guide.md:182` is accurate again with no doc edit.

```bash
git add andypack/sprites.py andypack/nodes.py tests/test_sprites.py
git commit -m "feat(sprite): reinstate Palette Quantize & Lock for pixel-art color consistency"
```

---

### Task 19: Documentation sync (README, CLAUDE.md, manifest schema, prompting guide)

**Files:**
- Modify: `README.md` (node table, sprite-export section, graph wiring, manifest section)
- Modify: `CLAUDE.md` (module map, invariants)
- Modify: `docs/manifest-schema.md` (reference_image)
- Modify: `docs/prompting-guide.md` (Wan wiring note)

**Interfaces:**
- Consumes: everything above.
- Produces: docs matching the shipped node set. No code changes.

- [ ] **Step 1: README node table — add rows (keep the existing table style)**

Add to the node table in `README.md`:

```markdown
| **Manikin Loader** | Load a bundled per-direction manikin as an IMAGE (+ its direction name) — the pose/camera source for authoring custom pose references (drive an OpenPose/ControlNet-capable graph per direction). |
| **Pose Reference Writer** | Save an IMAGE into the pose-references dir (`user/default/andypack/pose_references/`) as `<name>_<DIRECTION>.png` and return the filename — exactly what a pose direction layer's `reference_image` points at. |
| **Wan Animation Conditioning** | One-node Wan 2.2 conditioning from an `ANIM_ANIMATION` bundle: text-encode + core FFLF encode, omitting `end_image` entirely for non-FFLF clips (the sentinel never reaches the sampler). Wire `SHIFT` into `ModelSamplingSD3` as before. |
| **Sheet Export All** | Stage-3 batch export: one sheet + atlas per animation with ≥1 rendered direction, in one queue press. Skipped animations are listed in the report. |
| **Palette Quantize & Lock** | Force frames onto one shared limited palette (optionally locked to a `palette_image`) for pixel-art consistency across directions/animations. |
| **Frame Retime** | Resample a clip to a target fps (resample / trim / pad-hold) before packing/export. |
```

- [ ] **Step 2: README feature notes**

- In the **Manikins** section, add: per-direction `reference_image` on a pose's direction layer overrides the bundled manikin (root poses) or adds a second reference (derived poses); files live in `user/default/andypack/pose_references/`; author them with Manikin Loader → your pose graph → Pose Reference Writer.
- In the writers' descriptions, document `write_mirrored` (flip + write every `mirror_map` direction derived from the written one, with the mirrored direction's own sidecar and a `mirrored_from` provenance key) and `loop_color_match`. Note the caveat from `docs/prompting-guide.md` §4: mirroring is only sound for bilaterally symmetric designs.
- In **Graph wiring → Animations**, replace the manual `WanFirstLastFrameToVideo` wiring paragraph's opening with: recommended path is **Animation Sweep Selector → Wan Animation Conditioning → samplers → Animation Frame Writer** (manual `WanFirstLastFrameToVideo` wiring remains valid; the conditioning node exists so one graph serves FFLF and non-FFLF clips).
- In **Game-asset / sprite export**, mention Sheet Export All (batch), the `trim` option, Palette Quantize & Lock, Frame Retime, and that Animated Sprite Export now preserves alpha.
- Update the culled-node footnote: the pack is no longer exactly 20 nodes — say "the pipeline-essential node set" instead of a count, or update the count to match `NODE_CLASS_MAPPINGS` (count it: 22 existing + 6 new = 28).

- [ ] **Step 3: CLAUDE.md updates**

- Module map: add the six new nodes to the grouped list (Pose: Manikin Loader, Pose Reference Writer; Animation: Wan Animation Conditioning; Sprite: Palette Quantize & Lock, Frame Retime; Export: Sheet Export All) and note the writers' `write_mirrored` / `loop_color_match` flags and `REMAINING` semantics unchanged.
- Invariants: add three bullets —
  - "**Pose references**: a pose direction layer's `reference_image` names a bare `*.png` under `user/default/andypack/pose_references/`; it overrides the bundled manikin (root) or adds a second FLUX reference (derived). Recorded in the sidecar; drift re-stales the cell."
  - "**Mirrored cells are real renders**: `write_mirrored` writes a flipped payload with the MIRRORED direction's own resolved meta (+ `mirrored_from`); nothing downstream special-cases mirrors."
  - "**The 1×1 end-image sentinel must never reach a sampler**: `WanAnimationConditioning` (via `_wan_end_image`) omits `end_image` for non-FFLF clips; don't wire `Unpack Animation.END_IMAGE` into a Wan node without checking `IS_FFLF`."
- Remove the stale note "pure helpers for removed nodes may still linger in `sprites.py`/`api.py` (not exposed)" if it no longer applies, and update the "21 focused nodes" count.

- [ ] **Step 4: docs/manifest-schema.md + docs/prompting-guide.md**

- `docs/manifest-schema.md`: document the direction-layer `reference_image` property (type, bare-`*.png` constraint, resolution dir, staleness behavior, authoring workflow via Manikin Loader + Pose Reference Writer).
- `docs/prompting-guide.md` §1: add one sentence noting the pack now ships **Wan Animation Conditioning**, which performs this wiring (including the leave-`end_image`-unconnected rule) automatically. §3's loop-drift caveat: note the writer's `loop_color_match` flag as the built-in mitigation. §5's palette paragraph needs no change (node reinstated in Task 18) — verify the name matches "Palette Quantize & Lock".

- [ ] **Step 5: Run checks and commit**

Run: `pytest -q && ruff check . && mypy andypack`
Expected: all pass (docs only).

```bash
git add README.md CLAUDE.md docs/manifest-schema.md docs/prompting-guide.md
git commit -m "docs: sync README/CLAUDE/schema/prompting guide with the new node set"
```

---

## Out of scope for this plan (manual follow-ups, need the live pod)

These cannot be verified headless and are deliberately NOT tasks above — do them on the running ComfyUI pod after the code lands (see memory note `comfyui-pod-behavior`):

1. **Sweep-loop scale validation** — run a full animation sweep (~180+ iterations) and watch dynprompt/memory behavior; the Open/Close mechanic is only spike-validated at small counts.
2. **Regenerate `examples/workflows/2_animate_fflf.json`** using Wan Animation Conditioning (drops the `PainterFLF2V` third-party dependency); re-run the 1a → 1b → 2 → 3 chain end-to-end.
3. **Add a pose-reference authoring example workflow** (Manikin Loader → OpenPose ControlNet graph → Pose Reference Writer) once a ControlNet-capable checkpoint is picked.
4. **Visual QA**: mirror flips on an asymmetric character (confirm the symmetric-design caveat is documented in the right places), palette lock on real renders, loop color-match on a real Wan loop, GIF/WebP/APNG transparency in a browser and an engine import.

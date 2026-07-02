# SYBP Character-Asset Workflow Set — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship one generic 4-workflow ComfyUI set that drives create → turnaround → animate → export for all 26 *Shit Your Brain Pants* characters off their pre-authored `character.json` layers, plus the one pack node that lets the create-stage prompt auto-follow the character combo.

**Architecture:** Add a small read-only `CharacterPromptLoader` node to the `comfyui-andypack` pack (returns a character's authored positive/negative as STRINGs). Then author four workflow JSONs in the `shit-your-brain-pants` repo: workflow 1 (Create) is `1a` reworked so a shared style-reference image + the loaded prompt drive a FLUX.2 reference txt2img; workflows 2–4 are `1b`/`2`/`3` retargeted to the SYBP manifest + a SYBP character by two widget edits each.

**Tech Stack:** Python 3.10–3.12, PyTorch (CPU in CI), ComfyUI custom-node pack, pytest / ruff / mypy. ComfyUI UI-format workflow JSON. FLUX.2 Klein 9B (create/turnaround), WAN 2.2 14B i2v (animate).

## Global Constraints

- Node code lives in `andypack/nodes.py`; `resolve.py`/`io.py` stay torch-free — `CharacterPromptLoader` uses only existing `api`/`io`/`resolve` helpers, no torch.
- New nodes MUST be registered in BOTH `NODE_CLASS_MAPPINGS` and `NODE_DISPLAY_NAME_MAPPINGS` in `andypack/nodes.py`.
- All pack changes must pass `pytest -q`, `ruff check .`, and `mypy andypack` (CI runs all three on 3.10/3.11/3.12).
- Workflow JSON must be **UI format** (top-level `nodes`/`links`/`groups`), not API format — API/empty-canvas JSON won't load in the ComfyUI editor.
- Character folder names are snake_case; the combo value is snake-cased via `io.to_snake_case` before use.
- The pack work happens on branch `feat/sybp-workflow-set` (already created, spec committed). The SYBP workflows are a separate git repo at `/Users/user/Code/andyhite/shit-your-brain-pants`.
- Manifest filename referenced by `AnimationManifestLoader` in every SYBP workflow: **`sybp.json`** (must match the file placed under ComfyUI's manifests dir at deploy).
- Default character baked into the SYBP workflow combos: **`cortex`** (switched per run via the dropdown).

---

## Task 1: `CharacterPromptLoader` node (in `comfyui-andypack`)

**Files:**
- Modify: `andypack/nodes.py` — add the class (near `CharacterReferenceLoader`, ~line 304) and register it in both mappings (~lines 1407–1450).
- Test: `tests/test_nodes.py` — append three tests.

**Interfaces:**
- Consumes (existing, verified): `nodes._NO_CHARACTER` (str), `nodes._characters_root()` (→ str), `nodes._mtime(path)` (→ float), `nodes._character_choices()` (→ list), `io.to_snake_case(name)` (→ str), `resolve.read_character(root, character)` (→ dict; returns `{}` when absent/corrupt; keys `positive_prompt`/`negative_prompt`).
- Produces: `nodes.CharacterPromptLoader` with `RETURN_NAMES = ("POSITIVE", "NEGATIVE")` and method `load(self, character) -> tuple[str, str]`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_nodes.py` (the file already imports `json`, `os`, `pytest`, and `nodes`):

```python
# --- CharacterPromptLoader --------------------------------------------------- #

def _write_character(root, name, payload):
    os.makedirs(os.path.join(root, name), exist_ok=True)
    with open(os.path.join(root, name, "character.json"), "w") as f:
        json.dump(payload, f)


def test_character_prompt_loader_returns_authored_prompts(tmp_path, monkeypatch):
    root = str(tmp_path)
    monkeypatch.setattr(nodes, "_characters_root", lambda: root)
    _write_character(root, "cortex", {"positive_prompt": "a brave hero", "negative_prompt": "blurry"})
    # combo value is snake-cased, so a display-cased name still resolves.
    pos, neg = nodes.CharacterPromptLoader().load("Cortex")
    assert pos == "a brave hero"
    assert neg == "blurry"


def test_character_prompt_loader_missing_fields_yield_empty(tmp_path, monkeypatch):
    root = str(tmp_path)
    monkeypatch.setattr(nodes, "_characters_root", lambda: root)
    _write_character(root, "ghost", {"positive_prompt": "x"})
    pos, neg = nodes.CharacterPromptLoader().load("ghost")
    assert pos == "x"
    assert neg == ""


def test_character_prompt_loader_requires_character():
    with pytest.raises(RuntimeError, match="select a character"):
        nodes.CharacterPromptLoader().load(nodes._NO_CHARACTER)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_nodes.py -k character_prompt_loader -v`
Expected: FAIL — `AttributeError: module 'andypack.nodes' has no attribute 'CharacterPromptLoader'`.

- [ ] **Step 3: Add the node class**

In `andypack/nodes.py`, immediately after the `CharacterReferenceLoader` class (after its `load` method, ~line 304), insert:

```python
class CharacterPromptLoader:
    """Read a character's authored identity prompts from `character.json` as
    wireable STRINGs, so a txt2img / FLUX edit graph can drive the character's
    own positive/negative without hand-typing them. Unlike
    CharacterReferenceLoader (which needs persisted reference art), this only
    needs the authored layer, so it works before any render exists — e.g. to
    seed the Create stage's reference generation from the character combo."""

    CATEGORY = "andypack/Character"
    FUNCTION = "load"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("POSITIVE", "NEGATIVE")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"character": (_character_choices(),)}}

    @classmethod
    def IS_CHANGED(cls, character):
        if character in ("", _NO_CHARACTER):
            return float("nan")
        path = os.path.join(
            _characters_root(), io.to_snake_case(character), "character.json"
        )
        return f"{path}:{_mtime(path)}"

    def load(self, character):
        if character in ("", _NO_CHARACTER):
            raise RuntimeError("CharacterPromptLoader: select a character first")
        identity = resolve.read_character(
            _characters_root(), io.to_snake_case(character)
        )
        return (
            str(identity.get("positive_prompt", "") or ""),
            str(identity.get("negative_prompt", "") or ""),
        )
```

- [ ] **Step 4: Register the node in both mappings**

In `andypack/nodes.py`, add to `NODE_CLASS_MAPPINGS` (after the `"CharacterReferenceLoader": CharacterReferenceLoader,` line):

```python
    "CharacterPromptLoader": CharacterPromptLoader,
```

And to `NODE_DISPLAY_NAME_MAPPINGS` (after the `"CharacterReferenceLoader": "Character Reference Loader",` line):

```python
    "CharacterPromptLoader": "Character Prompt Loader",
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest tests/test_nodes.py -k character_prompt_loader -v`
Expected: PASS (3 passed).

- [ ] **Step 6: Run the full checks**

Run: `pytest -q && ruff check . && mypy andypack`
Expected: all pass, no errors. (If `mypy` flags the `load` return, it is inferred `tuple[str, str]` — no annotation needed to match sibling nodes, which are also unannotated.)

- [ ] **Step 7: Commit**

```bash
git add andypack/nodes.py tests/test_nodes.py
git commit -m "feat(nodes): CharacterPromptLoader — authored character.json prompt as STRING outputs

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Workflow 1 — Create (in `shit-your-brain-pants`)

Reworks the pack's `1a_character_create.json` so the character combo + a shared
style reference drive the reference-art txt2img, then persists + renders
`base@SOUTH`. **Depends on Task 1** (uses `CharacterPromptLoader`).

**Files:**
- Create: `/Users/user/Code/andyhite/shit-your-brain-pants/assets/workflows/1_create.json`
- Source template (copy from): `/Users/user/Code/andyhite/comfyui-andypack/examples/workflows/1a_character_create.json`

**Interfaces:**
- Consumes: `CharacterPromptLoader` (Task 1) — outputs `POSITIVE` (slot 0), `NEGATIVE` (slot 1); core nodes `LoadImage`, `VAEEncode`, `ReferenceLatent`; pack nodes `AnimationManifestLoader`, `CharacterCreator`, `PoseEditConditioning`, `PoseFrameWriter`.
- Produces: a UI-format workflow file the ComfyUI editor loads and runs.

- [ ] **Step 1: Create the workflows dir and copy the template**

```bash
mkdir -p /Users/user/Code/andyhite/shit-your-brain-pants/assets/workflows
cp /Users/user/Code/andyhite/comfyui-andypack/examples/workflows/1a_character_create.json \
   /Users/user/Code/andyhite/shit-your-brain-pants/assets/workflows/1_create.json
```

- [ ] **Step 2: Retarget the manifest and CharacterCreator, and drop the hand-typed prompt**

Edit `1_create.json`:

1. Node id `1` (`AnimationManifestLoader`): set `widgets_values` to `["sybp.json"]`.
2. Node id `30` (`CharacterCreator`): set `widgets_values` to `["cortex", "SOUTH", "", "", true]` (character `cortex`; `character_positive`/`character_negative` **empty** so the authored `character.json` is preserved; `save_reference` on).
3. Node id `20` (`CLIPTextEncode`): this stops being hand-typed — its `text` becomes an input wired from `CharacterPromptLoader`. Replace its `inputs` array so `text` is a converted-widget input, and blank its widget value:
   - Set `"inputs"` to:
     ```json
     [
       { "name": "clip", "type": "CLIP", "link": 1 },
       { "name": "text", "type": "STRING", "link": 24, "widget": { "name": "text" } }
     ]
     ```
   - Set `"widgets_values"` to `[""]`.

- [ ] **Step 3: Add the four new front-end nodes**

Append these node objects to the `nodes` array of `1_create.json` (ids continue from the file's `last_node_id` 99; positions are cosmetic):

```json
{
  "id": 60, "type": "LoadImage", "pos": [420, 520], "size": [320, 320],
  "flags": {}, "order": 5, "mode": 0, "inputs": [],
  "outputs": [
    { "name": "IMAGE", "type": "IMAGE", "links": [25], "slot_index": 0 },
    { "name": "MASK", "type": "MASK", "links": [], "slot_index": 1 }
  ],
  "properties": { "Node name for S&R": "LoadImage" },
  "widgets_values": ["sybp_style_ref.png", "image"]
},
{
  "id": 61, "type": "VAEEncode", "pos": [780, 520], "size": [260, 100],
  "flags": {}, "order": 15, "mode": 0,
  "inputs": [
    { "name": "pixels", "type": "IMAGE", "link": 25 },
    { "name": "vae", "type": "VAE", "link": 26 }
  ],
  "outputs": [ { "name": "LATENT", "type": "LATENT", "links": [27], "slot_index": 0 } ],
  "properties": { "Node name for S&R": "VAEEncode" }, "widgets_values": []
},
{
  "id": 62, "type": "ReferenceLatent", "pos": [1080, 300], "size": [260, 80],
  "flags": {}, "order": 22, "mode": 0,
  "inputs": [
    { "name": "conditioning", "type": "CONDITIONING", "link": 28 },
    { "name": "latent", "type": "LATENT", "link": 27 }
  ],
  "outputs": [ { "name": "CONDITIONING", "type": "CONDITIONING", "links": [29, 30], "slot_index": 0 } ],
  "properties": { "Node name for S&R": "ReferenceLatent" }, "widgets_values": []
},
{
  "id": 63, "type": "CharacterPromptLoader", "pos": [40, 1240], "size": [320, 120],
  "flags": {}, "order": 2, "mode": 0, "inputs": [],
  "outputs": [
    { "name": "POSITIVE", "type": "STRING", "links": [24], "slot_index": 0 },
    { "name": "NEGATIVE", "type": "STRING", "links": [], "slot_index": 1 }
  ],
  "properties": { "Node name for S&R": "CharacterPromptLoader" },
  "widgets_values": ["cortex"]
}
```

- [ ] **Step 4: Rewire the conditioning path through ReferenceLatent**

In the `links` array of `1_create.json`:

1. **Remove** the two links out of node 20 that currently feed the sampler/negative directly:
   - `[2, 20, 0, 21, 0, "CONDITIONING"]` (20 → ConditioningZeroOut)
   - `[4, 20, 0, 23, 1, "CONDITIONING"]` (20 → KSampler positive)
2. Change node 20's `outputs[0].links` to `[28]` (it now feeds only ReferenceLatent).
3. **Add** these link tuples (format: `[link_id, from_node, from_slot, to_node, to_slot, type]`):
   ```json
   [24, 63, 0, 20, 1, "STRING"],
   [25, 60, 0, 61, 0, "IMAGE"],
   [26, 12, 0, 61, 1, "VAE"],
   [27, 61, 0, 62, 1, "LATENT"],
   [28, 20, 0, 62, 0, "CONDITIONING"],
   [29, 62, 0, 21, 0, "CONDITIONING"],
   [30, 62, 0, 23, 1, "CONDITIONING"]
   ```
4. Bump top-level `"last_node_id"` to `63` and `"last_link_id"` to `30`.

Result: `CharacterPromptLoader(63).POSITIVE → CLIPTextEncode(20).text`; `CLIPTextEncode(20) → ReferenceLatent(62)`; `LoadImage(60) → VAEEncode(61) → ReferenceLatent(62).latent`; `ReferenceLatent(62) → KSampler(23).positive` and `→ ConditioningZeroOut(21) → KSampler(23).negative`. The rest of `1a` (base-pose branch via CharacterCreator → PoseEditConditioning → KSampler 49 → PoseFrameWriter 52) is unchanged.

- [ ] **Step 5: Update the Note**

Node id `99` (`Note`): set `widgets_values` to:

```json
["STEP 1 (SYBP). Pick a character in BOTH the Character Prompt Loader and Character Creator dropdowns (same value). The loader feeds that character's authored character.json prompt into the txt2img, conditioned on the shared humanoid-brain style reference (LoadImage -> VAEEncode -> ReferenceLatent). character_positive is left blank so the authored character.json is preserved. Then run 2_turnaround to batch the rest. Needs Flux.2 Klein 9B + qwen_3_8b. Drop the style ref PNG in ComfyUI/input/ and set it on the LoadImage node."]
```

- [ ] **Step 6: Verify the JSON parses**

Run:
```bash
python3 -c "import json; d=json.load(open('/Users/user/Code/andyhite/shit-your-brain-pants/assets/workflows/1_create.json')); print(len(d['nodes']),'nodes', len(d['links']),'links')"
```
Expected: `21 nodes 28 links` (17 original + 4 new; 23 original links − 2 removed + 7 added).

- [ ] **Step 7: Validate against ComfyUI (if the pod is up)**

If the ComfyUI pod is reachable via the comfy MCP, validate the graph resolves against live `object_info` (catches a missing `ReferenceLatent`, an unregistered `CharacterPromptLoader`, or a bad link):

Use MCP tool `mcp__comfy_comfyui__validate_workflow` with the file contents.
Expected: valid (all node types + widgets resolve). If `CharacterPromptLoader` is reported missing, the pod is running a pack build without Task 1 deployed — note it and continue (structure is still correct).
If the pod is unreachable, skip this step and confirm by loading `1_create.json` in the ComfyUI editor manually.

- [ ] **Step 8: Commit (in the SYBP repo)**

```bash
cd /Users/user/Code/andyhite/shit-your-brain-pants
git rev-parse --abbrev-ref HEAD   # if this prints the default branch, first: git checkout -b feat/andypack-workflows
git add assets/workflows/1_create.json
git commit -m "feat(workflows): SYBP create workflow (style-ref txt2img + CharacterPromptLoader)"
```

---

## Task 3: Workflows 2–4 — Turnaround / Animate / Export (in `shit-your-brain-pants`)

Straight copies of the pack examples with two widget edits each.

**Files:**
- Create: `assets/workflows/2_turnaround.json` (from `1b_turnaround_batch.json`)
- Create: `assets/workflows/3_animate.json` (from `2_animate_fflf.json`)
- Create: `assets/workflows/4_export.json` (from `3_sprite_export.json`)

**Interfaces:**
- Consumes: pack nodes only (`AnimationManifestLoader`, `PoseSweepSelector`, `AnimationSweepSelector`, `AnimationSheetBuilder`, `AtlasMetadataWriter`, `SweepLoopOpen/Close`, writers) + `PainterFLF2V`/`BiRefNetRMBG` for workflow 3. No dependency on Task 1.
- Produces: three UI-format workflow files.

- [ ] **Step 1: Copy the three templates**

```bash
SRC=/Users/user/Code/andyhite/comfyui-andypack/examples/workflows
DST=/Users/user/Code/andyhite/shit-your-brain-pants/assets/workflows
cp "$SRC/1b_turnaround_batch.json" "$DST/2_turnaround.json"
cp "$SRC/2_animate_fflf.json"      "$DST/3_animate.json"
cp "$SRC/3_sprite_export.json"     "$DST/4_export.json"
```

- [ ] **Step 2: Retarget `2_turnaround.json`**

- `AnimationManifestLoader` node `widgets_values`: `["sybp.json"]`.
- `PoseSweepSelector` node `widgets_values`: change the first element `"ranger"` → `"cortex"` (leave the rest: `["cortex", "sweep", true, true, "", "", ""]`).

- [ ] **Step 3: Retarget `3_animate.json`**

- `AnimationManifestLoader` node `widgets_values`: `["sybp.json"]`.
- `AnimationSweepSelector` node `widgets_values`: change the first element `"ranger"` → `"cortex"` (leave the rest: `["cortex", "sweep", true, "", "", ""]`).

- [ ] **Step 4: Retarget `4_export.json`**

- `AnimationManifestLoader` node `widgets_values`: `["sybp.json"]`.
- `AnimationSheetBuilder` node `widgets_values`: change the first element `"ranger"` → `"cortex"` and set the animation to one that exists in the SYBP manifest, e.g. `["cortex", "walk", "all", 2, false]`.
- `AtlasMetadataWriter` node `widgets_values`: `["aseprite", "cortex_walk", "atlas"]` (keep `aseprite` format; the name is cosmetic).

- [ ] **Step 5: Verify all three parse**

Run:
```bash
for f in 2_turnaround 3_animate 4_export; do
  python3 -c "import json,sys; d=json.load(open('/Users/user/Code/andyhite/shit-your-brain-pants/assets/workflows/$f.json')); \
    m=[n for n in d['nodes'] if n['type']=='AnimationManifestLoader'][0]['widgets_values']; \
    print('$f', 'manifest=', m)"
done
```
Expected: each prints `manifest= ['sybp.json']`.

- [ ] **Step 6: Validate against ComfyUI (if the pod is up)**

For each of the three files, run `mcp__comfy_comfyui__validate_workflow` if the pod is reachable. Expected: valid. Workflow 3 additionally requires `Comfyui-PainterFLF2V` + `comfyui-rmbg` installed on the pod — a validation error naming `PainterFLF2V`/`BiRefNetRMBG` means those custom nodes aren't installed, not a workflow defect. Skip if unreachable and load each in the editor to confirm.

- [ ] **Step 7: Commit (in the SYBP repo)**

```bash
cd /Users/user/Code/andyhite/shit-your-brain-pants
git add assets/workflows/2_turnaround.json assets/workflows/3_animate.json assets/workflows/4_export.json
git commit -m "feat(workflows): SYBP turnaround/animate/export workflows (retargeted to sybp.json)"
```

---

## Task 4: Workflow-set README (in `shit-your-brain-pants`)

**Files:**
- Create: `/Users/user/Code/andyhite/shit-your-brain-pants/assets/workflows/README.md`

- [ ] **Step 1: Write the README**

```markdown
# SYBP character-asset workflows

UI-format ComfyUI workflows for the comfyui-andypack pipeline, tuned for this
project. Load one via ComfyUI **Load** (or drag-and-drop). Run in order; state
is disk-backed under `<output>/characters/<char>/`.

| File | Stage | Model | What it does |
|---|---|---|---|
| `1_create.json`     | Create     | FLUX.2 Klein 9B | Character Prompt Loader feeds the selected character's `character.json` prompt into a style-referenced txt2img (shared humanoid-brain LoadImage → VAEEncode → ReferenceLatent), persists the reference, renders `base@SOUTH`. |
| `2_turnaround.json` | Turnaround | FLUX.2 Klein 9B | One-press Sweep Loop: Pose Sweep Selector (`sweep`, `include_base`) → Pose Edit Conditioning → sampler → Pose Frame Writer. |
| `3_animate.json`    | Animate    | WAN 2.2 14B i2v | One-press Sweep Loop: Animation Sweep Selector → dual hi/lo WAN + lightx2v + pixel-animate LoRAs → PainterFLF2V → BiRefNet alpha → Animation Frame Writer. |
| `4_export.json`     | Export     | — | Animation Sheet Builder (rows=directions × cols=frames) → Atlas Metadata Writer (Aseprite JSON). Change `animation` per clip. |

## Setup
- Manifest: place `animations.json` as **`sybp.json`** under ComfyUI's andypack
  manifests dir; place `characters/*` under `<output>/characters/`.
- `1_create.json`: pick the character in **both** the Character Prompt Loader and
  Character Creator dropdowns (same value); drop the shared style-reference PNG in
  `ComfyUI/input/` and select it on the LoadImage node.
- Requires the `comfyui-andypack` pack (with `CharacterPromptLoader`), plus
  `Comfyui-PainterFLF2V` + `comfyui-rmbg` for `3_animate.json`.

## Notes
- Coverage is EAST-centric by design (platformer). The sweeps render whatever
  each pose/animation lists in the manifest; add WEST / other facings there.
- Export target is Aseprite for manual tweaks; Godot import is downstream.
```

- [ ] **Step 2: Verify it renders**

Run: `python3 -c "print(open('/Users/user/Code/andyhite/shit-your-brain-pants/assets/workflows/README.md').read()[:80])"`
Expected: prints the title line.

- [ ] **Step 3: Commit (in the SYBP repo)**

```bash
cd /Users/user/Code/andyhite/shit-your-brain-pants
git add assets/workflows/README.md
git commit -m "docs(workflows): README for the SYBP character-asset workflow set"
```

---

## Self-Review

**1. Spec coverage** (checked each §):
- §2 `CharacterPromptLoader` → Task 1. ✓
- §3 Create workflow (style ref + ReferenceLatent + prompt loader + preserve authored) → Task 2. ✓
- §4 Turnaround/Animate/Export retarget → Task 3. ✓
- §1 repo layout (workflows in SYBP repo) → Tasks 2–4 write to `shit-your-brain-pants/assets/workflows/`; README → Task 4. ✓
- §6 verification (`pytest`/`ruff`/`mypy`; JSON parse; `validate_workflow`; manifest-filename confirm) → Task 1 Step 6, Task 2 Steps 6–7, Task 3 Steps 5–6. ✓
- §5 manifest follow-ups are explicitly out of scope (user-owned); README §Notes points at the E/W coverage one. ✓
- §7 out-of-scope items (per-character copies, batch driver, sync script, Godot tuning, manifest edits) — none appear as tasks. ✓

**2. Placeholder scan:** No "TBD"/"handle edge cases"/"write tests for the above" — every code and JSON step is literal. The style-reference filename `sybp_style_ref.png` and character `cortex` are concrete defaults (documented as user-swappable), not placeholders.

**3. Type consistency:** `CharacterPromptLoader.load` returns `(str, str)` named `POSITIVE`/`NEGATIVE`; Task 2 wires slot 0 (`POSITIVE`) via link 24 into `CLIPTextEncode.text` and leaves slot 1 unwired — consistent. `read_character`/`to_snake_case`/`_characters_root`/`_mtime`/`_NO_CHARACTER` all match the signatures verified in `andypack`. Link ids 24–30 and node ids 60–63 are unique and below the bumped `last_node_id`/`last_link_id`.

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-02-sybp-workflow-set.md`. Two execution options:

1. **Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?

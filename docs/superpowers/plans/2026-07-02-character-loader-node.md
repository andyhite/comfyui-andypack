# Character Loader Node Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a read-only `CharacterLoader` node that emits the base-pose FLUX-edit job for a character+direction without ever writing `character.json`, so the SYBP Create workflow stops wiping authored prompts.

**Architecture:** Extract the "resolve base pose + pair supplied image with the direction's manikin → ANIM_POSE dict" logic (currently inline in `CharacterCreator.create`) into a shared module-level helper. `CharacterCreator` keeps writing the prompt layer then calls the helper; the new `CharacterLoader` calls the helper only (plus optional reference persistence). `CharacterPromptLoader` (STRING feed) is untouched.

**Tech Stack:** Python 3.10+, ComfyUI custom-node conventions, pytest, ruff, mypy.

## Global Constraints

- `resolve.py` / `manifest.py` stay free of ComfyUI/torch imports (node code lives in `nodes.py`). — this task touches only `nodes.py`, tests, and docs.
- Nodes register in BOTH `NODE_CLASS_MAPPINGS` and `NODE_DISPLAY_NAME_MAPPINGS` in `andypack/nodes.py`.
- CI runs `pytest -q`, `ruff check .`, `mypy andypack` on Python 3.10/3.11/3.12 — all three must stay green.
- Character on a read node is an *existing-only* combo: `_character_choices()` (placeholder `_NO_CHARACTER` + real folders). Not free-text.
- `_reference.png` persistence uses `images.save_image_png(image, resolve.reference_image_path(root, char_name))`.
- Never write `character.json` from the loader under any code path.

---

### Task 1: Extract shared base-pose helper (pure refactor)

**Files:**
- Modify: `andypack/nodes.py` (add helper near the other module-level `_build_*` helpers ~line 195; rewrite the tail of `CharacterCreator.create` ~lines 177–192)
- Test: `tests/test_nodes.py` (existing CharacterCreator tests are the guard — no new test)

**Interfaces:**
- Produces: `_character_base_pose(label: str, manifest, root: str, char_name: str, direction: str, image) -> dict` returning a POSE dict with keys `source_image, pose_reference, positive, negative, output_dir, _meta`.
- Consumes: `effective_manifest`, `resolve_pose` (already imported from `andypack.resolve`), `manikins.CANONICAL_DIRECTIONS`, `manikins.manikin_path`, `images.load_image_tensor`.

- [ ] **Step 1: Add the shared helper**

Insert above `def _build_pose_bundle` (currently line ~195) in `andypack/nodes.py`:

```python
def _character_base_pose(label, manifest, root, char_name, direction, image):
    """Resolve the `base` pose for a character+direction through the effective
    manifest (character overlay applied) and pair the supplied reference `image`
    (first FLUX-edit reference) with the direction's bundled manikin (second) into
    an ANIM_POSE dict. Shared by Character Creator (which persists the prompt layer
    first) and the read-only Character Loader. `label` prefixes error messages with
    the calling node's name."""
    if direction not in manikins.CANONICAL_DIRECTIONS:
        raise RuntimeError(f"{label}: unknown direction {direction!r}")
    eff = effective_manifest(manifest, root, char_name)
    if "base" not in eff.get("poses", {}):
        raise RuntimeError(f"{label}: manifest has no 'base' pose")
    if direction not in eff["poses"]["base"]["directions"]:
        raise RuntimeError(f"{label}: base has no direction {direction!r}")
    r = resolve_pose(eff, root, char_name, "base", direction)
    manikin = images.load_image_tensor(manikins.manikin_path(direction))
    return {
        "source_image": image,        # the character reference (first reference)
        "pose_reference": manikin,    # the manikin for this direction (second)
        "positive": r["positive"],
        "negative": r["negative"],
        "output_dir": r["output_dir"],
        "_meta": r["meta"],
    }
```

- [ ] **Step 2: Rewrite the tail of `CharacterCreator.create` to call the helper**

Replace lines ~177–192 (from `eff = effective_manifest(...)` through `return (pose,)`) with:

```python
        pose = _character_base_pose(
            "CharacterCreator", manifest, root, char_name, direction, image
        )
        return (pose,)
```

Leave everything above it (the direction guard at line ~153, the `character.json` write, `invalidate_character`, and the `save_reference` block) exactly as-is.

- [ ] **Step 3: Run the CharacterCreator tests — must still pass**

Run: `pytest tests/test_nodes.py -k character_creator -v`
Expected: PASS (all 5 CharacterCreator tests: writes_character_json, attaches_manikin, rejects_unknown_direction, persists_reference, can_skip_reference).

- [ ] **Step 4: Lint + types**

Run: `ruff check andypack/nodes.py && mypy andypack`
Expected: no errors.

- [ ] **Step 5: Commit**

```bash
git add andypack/nodes.py
git commit -m "refactor(nodes): extract _character_base_pose helper from CharacterCreator"
```

---

### Task 2: Add the `CharacterLoader` node

**Files:**
- Modify: `andypack/nodes.py` (add class after `CharacterPromptLoader` ~line 343; register in both mapping dicts ~lines 1447–1472)
- Test: `tests/test_nodes.py` (new tests after the CharacterPromptLoader block ~line 784)

**Interfaces:**
- Consumes: `_character_base_pose` (Task 1), `_character_choices`, `_characters_root`, `_NO_CHARACTER`, `io.to_snake_case`, `images.save_image_png`, `resolve.reference_image_path`, `manikins.CANONICAL_DIRECTIONS`.
- Produces: `nodes.CharacterLoader().load(manifest, image, character, direction, save_reference=True) -> (pose_dict,)`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_nodes.py`:

```python
# --- CharacterLoader (read-only base-pose emitter) --------------------------- #

def test_character_loader_emits_base_pose(manifest, tmp_path, monkeypatch):
    monkeypatch.setattr(nodes, "_characters_root", lambda: str(tmp_path))
    (pose,) = nodes.CharacterLoader().load(manifest, _img(), "cortex", "EAST")
    assert pose["_meta"]["pose"] == "base" and pose["_meta"]["direction"] == "EAST"
    assert pose["output_dir"].endswith(os.path.join("cortex", "_base"))
    assert not images.is_empty(pose["source_image"])      # the supplied reference
    assert not images.is_empty(pose["pose_reference"])     # the direction's manikin


def test_character_loader_does_not_write_character_json(manifest, tmp_path, monkeypatch):
    root = str(tmp_path)
    monkeypatch.setattr(nodes, "_characters_root", lambda: root)
    # Author a character.json up front; the loader must leave it byte-for-byte intact.
    os.makedirs(os.path.join(root, "cortex"), exist_ok=True)
    cj = os.path.join(root, "cortex", "character.json")
    with open(cj, "w") as f:
        json.dump({"positive_prompt": "a brave hero", "negative_prompt": "blurry"}, f)
    before = open(cj).read()
    nodes.CharacterLoader().load(manifest, _img(), "cortex", "EAST")
    assert open(cj).read() == before


def test_character_loader_persists_reference_by_default(manifest, tmp_path, monkeypatch):
    root = str(tmp_path)
    monkeypatch.setattr(nodes, "_characters_root", lambda: root)
    nodes.CharacterLoader().load(manifest, _img(3, 4), "cortex", "EAST")
    assert os.path.isfile(resolve.reference_image_path(root, "cortex"))


def test_character_loader_can_skip_reference_persistence(manifest, tmp_path, monkeypatch):
    root = str(tmp_path)
    monkeypatch.setattr(nodes, "_characters_root", lambda: root)
    nodes.CharacterLoader().load(manifest, _img(), "cortex", "EAST", save_reference=False)
    assert not os.path.exists(resolve.reference_image_path(root, "cortex"))


def test_character_loader_rejects_unknown_direction(manifest, tmp_path, monkeypatch):
    monkeypatch.setattr(nodes, "_characters_root", lambda: str(tmp_path))
    with pytest.raises(RuntimeError, match="direction"):
        nodes.CharacterLoader().load(manifest, _img(), "cortex", "UP")


def test_character_loader_requires_character(manifest, tmp_path, monkeypatch):
    monkeypatch.setattr(nodes, "_characters_root", lambda: str(tmp_path))
    with pytest.raises(RuntimeError, match="character"):
        nodes.CharacterLoader().load(manifest, _img(), nodes._NO_CHARACTER, "EAST")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_nodes.py -k character_loader -v`
Expected: FAIL with `AttributeError: module 'andypack.nodes' has no attribute 'CharacterLoader'`.

- [ ] **Step 3: Add the `CharacterLoader` class**

Insert after `CharacterPromptLoader` (ends ~line 342) in `andypack/nodes.py`:

```python
class CharacterLoader:
    """Emit the base-pose FLUX-edit job for an EXISTING character + direction,
    pairing the supplied reference `image` (first reference) with the bundled
    manikin (second) — exactly like the Character Creator, but READ-ONLY: it never
    writes character.json, so authored prompts survive. Use it in a Create graph
    that generates the reference art from an authored character.json (via
    CharacterPromptLoader → txt2img) and needs the base pose without re-authoring
    the prompt layer. A missing/empty character.json is not an error — the base
    pose's {character_prompt} just expands to empty."""

    CATEGORY = "andypack/Character"
    FUNCTION = "load"
    RETURN_TYPES = ("ANIM_POSE",)
    RETURN_NAMES = ("POSE",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "manifest": ("ANIM_MANIFEST",),
                "image": ("IMAGE",),
                "character": (_character_choices(),),
                "direction": (manikins.CANONICAL_DIRECTIONS,),
            },
            "optional": {
                # Persist the reference art to `<char>/_reference.png` so the
                # Stage-2 turnaround sweep can reload the root reference. Not a
                # prompt write — character.json is never touched here.
                "save_reference": ("BOOLEAN", {"default": True}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, manifest, image, character, direction, save_reference=True):
        if character in ("", _NO_CHARACTER):
            return float("nan")
        root = _characters_root()
        try:
            char_name = io.to_snake_case(character)
            eff = effective_manifest(manifest, root, char_name)
            r = resolve_pose(eff, root, char_name, "base", direction)
        except Exception:
            return float("nan")
        return "|".join([r["meta"]["prompt_hash"], direction, str(save_reference)])

    def load(self, manifest, image, character, direction, save_reference=True):
        if character in ("", _NO_CHARACTER):
            raise RuntimeError("CharacterLoader: select a character first")
        root = _characters_root()
        char_name = io.to_snake_case(character)
        pose = _character_base_pose(
            "CharacterLoader", manifest, root, char_name, direction, image
        )
        # Read-only w.r.t. the prompt layer; only the reference art may be written.
        if save_reference:
            images.save_image_png(image, resolve.reference_image_path(root, char_name))
        return (pose,)
```

- [ ] **Step 4: Register the node in both mapping dicts**

In `NODE_CLASS_MAPPINGS` (~line 1449), after the `"CharacterPromptLoader": CharacterPromptLoader,` line add:

```python
    "CharacterLoader": CharacterLoader,
```

In `NODE_DISPLAY_NAME_MAPPINGS` (~line 1472), after the `"CharacterPromptLoader": "Character Prompt Loader",` line add:

```python
    "CharacterLoader": "Character Loader",
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_nodes.py -k character_loader -v`
Expected: PASS (all 6 CharacterLoader tests).

- [ ] **Step 6: Full suite + lint + types**

Run: `pytest -q && ruff check . && mypy andypack`
Expected: all green.

- [ ] **Step 7: Commit**

```bash
git add andypack/nodes.py tests/test_nodes.py
git commit -m "feat(nodes): CharacterLoader — read-only base-pose emitter (no prompt write)"
```

---

### Task 3: Docs (README + CLAUDE.md)

**Files:**
- Modify: `README.md` (node table ~line 163; the create-flow prose ~lines 182–185 if it names the Creator as the only base-pose emitter)
- Modify: `CLAUDE.md` (Character group bullet in the Module map; node count)

- [ ] **Step 1: Add the README node-table row**

In `README.md`, directly after the **Character Creator** row (~line 163) add:

```markdown
| **Character Loader** | Read-only sibling of the Character Creator: emit the base-pose job for an existing character + direction (reference image + manikin → multi-reference FLUX.2 edit) **without writing `character.json`**. Use when the character's prompt layer is already authored and must be preserved (e.g. the SYBP Create workflow generates the reference art from `character.json`, then loads the base pose). Optionally persists the reference art (`save_reference`, default on). |
```

- [ ] **Step 2: Update CLAUDE.md Character group**

In `CLAUDE.md`, find the module-map Character line:

```
  - Character: Character Creator, Character Reference Loader.
```

Replace with (also covers the pre-existing Character Prompt Loader, which the map omitted):

```
  - Character: Character Creator (writes character.json + emits base pose),
    Character Loader (read-only: emits base pose, never writes character.json),
    Character Reference Loader, Character Prompt Loader (character.json prompts
    as STRINGs).
```

Then update the node count in the `nodes.py` line from **20 focused nodes** to **21 focused nodes**.

- [ ] **Step 3: Verify no stale claims**

Run: `grep -n "20 focused nodes" CLAUDE.md`
Expected: no output (the count was bumped).

- [ ] **Step 4: Commit**

```bash
git add README.md CLAUDE.md
git commit -m "docs: document CharacterLoader node"
```

---

### Task 4: Fix the SYBP `1_create.json` workflow (separate repo)

**Files:**
- Modify: `../shit-your-brain-pants/assets/workflows/1_create.json` (node id 30; the Note node id 99)

**Interfaces:**
- Consumes: the new `"CharacterLoader"` node type from Task 2.

- [ ] **Step 1: Repoint node 30 to CharacterLoader and drop prompt widgets**

In node id `30`:
- Change `"type": "CharacterCreator"` → `"type": "CharacterLoader"`.
- Change `"properties"."Node name for S&R"` from `"CharacterCreator"` → `"CharacterLoader"`.
- Change `widgets_values` from `["cortex", "SOUTH", "", "", true]` to `["cortex", "SOUTH", true]` (drop the two empty prompt strings; keep character, direction, save_reference).

Leave the node's `inputs` (`manifest` link 10, `image` link 11) and `outputs` (POSE links 12, 22) unchanged — the interface is identical.

- [ ] **Step 2: Update the Note (node 99) text**

Replace the misleading "character_positive is left blank so the authored character.json is preserved" clause. New `widgets_values[0]`:

```
STEP 1 (SYBP). Pick the same character in the Character Prompt Loader and Character Loader dropdowns. The prompt loader feeds that character's authored character.json prompt into the txt2img, conditioned on the shared humanoid-brain style reference (LoadImage -> VAEEncode -> ReferenceLatent). The Character Loader is READ-ONLY: it emits the base pose from the generated reference without ever writing character.json, so the authored prompts are preserved. Then run 2_turnaround to batch the rest. Needs Flux.2 Klein 9B + qwen_3_8b. Drop the style ref PNG in ComfyUI/input/ and set it on the LoadImage node.
```

- [ ] **Step 3: Validate the JSON parses**

Run: `python -c "import json; json.load(open('../shit-your-brain-pants/assets/workflows/1_create.json')); print('ok')"`
Expected: `ok`.

- [ ] **Step 4: Commit (in the shit-your-brain-pants repo)**

```bash
cd ../shit-your-brain-pants
git add assets/workflows/1_create.json
git commit -m "fix(workflow): 1_create uses read-only CharacterLoader (stop clobbering character.json)"
cd -
```

---

## Self-Review

**Spec coverage:**
- New `CharacterLoader` node (interface, behavior, save_reference, missing-json resolves-anyway) → Task 2. ✓
- Read-only guarantee (no character.json write) → Task 2 Step 1 test `does_not_write_character_json`. ✓
- Shared helper refactor → Task 1. ✓
- Leave CharacterCreator / CharacterPromptLoader unchanged → Task 1 keeps Creator's write path; CharacterPromptLoader not touched. ✓
- Registration + docs → Task 2 Step 4, Task 3. ✓
- Workflow fix → Task 4. ✓
- IS_CHANGED (prompt_hash + direction + save_reference) → Task 2 Step 3. ✓

**Placeholder scan:** No TBD/TODO; every code step shows full code. ✓

**Type consistency:** Helper name `_character_base_pose` used identically in Task 1 (defined + called by Creator) and Task 2 (called by Loader). Node class `CharacterLoader` and type string `"CharacterLoader"` consistent across Tasks 2–4. POSE dict keys match `_build_pose_bundle`/CharacterCreator output shape. ✓

# SYBP character-asset workflow set — design

**Date:** 2026-07-02
**Repos:** `comfyui-andypack` (the pack), `shit-your-brain-pants` (the game/assets)
**Goal:** A reusable set of ComfyUI workflows that drive the full
character → turnaround → animation → sprite-sheet pipeline for the 26
*Shit Your Brain Pants* characters, using this pack's dependency-aware FFLF
resolver and the characters' pre-authored `character.json` identity layers.

---

## 1. Context

- The pack's example workflows (`comfyui-andypack/examples/workflows/`,
  stages `1a`/`1b`/`2`/`3`) already implement the whole pipeline generically,
  parameterized by `manifest` / `character` / `animation` combos that the web
  extension (`web/anim_coord.js`) populates from the loaded manifest.
- The SYBP manifest (`shit-your-brain-pants/assets/animations.json`) and all 26
  `character.json` overlays **validate cleanly** against
  `andypack/manifest.py` (base + each overlay merged the way
  `resolve.effective_manifest` does at resolve time), with zero lint warnings.
- SYBP diverges from the example manifest in one way that matters here: the 26
  characters have **pre-authored identity prompts** in `character.json`, whereas
  the example workflow (`1a`) has the prompt typed by hand into a
  `CLIPTextEncode` widget. So the SYBP Create stage must *read* the authored
  prompt rather than have it typed.
- The game is a **platformer**: coverage is deliberately EAST-centric (E/W
  rendered independently — not auto-mirrored, because designs are asymmetric),
  with SE/SW/S for some idle breaks and NE/NW/N for occasional cutscenes.
  Workflows stay **direction-agnostic** — the sweep selectors iterate whatever
  each pose/animation lists — so coverage is purely a manifest-content decision,
  not a workflow decision.

### Decisions (locked with the user)

| Question | Decision |
|---|---|
| Deliverable shape | **One generic SYBP set** of 4 workflows; character picked from the combo. |
| Create-stage prompt source | **New `CharacterPromptLoader` node** reads `character.json`. |
| Reference art | **FLUX.2 txt2img** from the authored prompt, conditioned on a shared humanoid-brain **style/anatomy reference image** (user-supplied). |
| Export target | **Aseprite** (AnimationSheetBuilder + AtlasMetadataWriter); Godot downstream. |
| Workflow file location | **`shit-your-brain-pants` repo** (`assets/workflows/`); the pack's `examples/` stays generic. |
| Manifest/character deployment | **Manual / already handled**; workflows reference expected paths + manifest filename. |

---

## 2. Component: `CharacterPromptLoader` node (in `comfyui-andypack`)

A small, focused node mirroring `CharacterReferenceLoader`, so the Create
stage's txt2img prompt auto-follows the `character` dropdown and a single
generic workflow serves all 26 characters.

- **Category:** `andypack/Character`
- **`INPUT_TYPES`:** `required = { "character": (_character_choices(),) }` — the
  same dynamic combo the other character-scoped nodes use; the web extension
  repopulates it from `/anim_coord/characters`.
- **`RETURN_TYPES` / `RETURN_NAMES`:** `("STRING", "STRING")` /
  `("POSITIVE", "NEGATIVE")`.
- **Behavior (`load`):** resolve the characters root (`_characters_root()`),
  snake-case the name (`io.to_snake_case`), read the identity layer
  (`resolve.read_character(root, name)`), return
  `(identity.get("positive_prompt", ""), identity.get("negative_prompt", ""))`.
  Raise `RuntimeError` if `character` is empty / the no-character sentinel
  (same guard style as `CharacterReferenceLoader`).
- **`IS_CHANGED`:** keyed on the `character.json` path + mtime (so an edited
  identity re-fires the node), returning `float("nan")` when no character is
  selected.
- **Registration:** add to `NODE_CLASS_MAPPINGS` and
  `NODE_DISPLAY_NAME_MAPPINGS` (display name e.g. "Character Prompt Loader").
- **Deliberately minimal:** returns pure identity text with no prompt-wrapping
  widgets. Framing/anatomy/style come from the shared style-reference image
  wired as a FLUX.2 reference latent (§3), so no framing text is needed here.
- **Tests / checks:** a unit test that a known `character.json` yields its
  `positive_prompt` / `negative_prompt` (and empties when absent); `ruff check
  .` and `mypy andypack` clean. The node is pure (no torch/ComfyUI side effects
  beyond reading JSON via existing helpers), consistent with `resolve`/`io`
  staying torch-free.

No other pack code changes. The node is independently understandable: input is
one character name, output is that character's authored prompt strings, and it
depends only on the existing `api`/`io`/`resolve` character-reading helpers.

---

## 3. Workflow 1 — Create (reworked front-end)

Purpose: from a character's authored prompt + the shared style reference,
generate the character's reference art, persist it (`CharacterCreator`, which
also keeps the authored `character.json` untouched), and render `base@SOUTH`.

Loaders (identical to `1a`): `UNETLoader(flux-2-klein-9b)`,
`CLIPLoader(qwen_3_8b, flux2)`, `VAELoader(flux2-vae)`,
`AnimationManifestLoader(sybp.json)`.

Graph:

```
LoadImage(humanoid-brain style ref) → VAEEncode ─┐
                                                 ReferenceLatent ─┐
CharacterPromptLoader(character) ─ POSITIVE → CLIPTextEncode ─────┘   │
                                                        ConditioningZeroOut ─ (negative)
   → KSampler → VAEDecode → [concept art IMAGE] → PreviewImage
                                              └→ CharacterCreator(
                                                   manifest, image=concept art,
                                                   character=<same combo>, direction=SOUTH,
                                                   character_positive="", character_negative="",
                                                   save_reference=true) → POSE
        → PoseEditConditioning(pose, clip, vae) → KSampler → VAEDecode
             → PreviewImage + PoseFrameWriter(pose, image)
```

Notes:
- `ReferenceLatent` (core node) is the workflow-level form of the
  `reference_latents` append that `PoseEditConditioning` performs internally
  (`vae.encode(pixels)` → `conditioning_set_values(..., {"reference_latents":
  [latent]}, append=True)`). It injects the shared style reference so all
  characters share style/anatomy while the identity prompt differentiates them.
- FLUX.2 Klein has no negative path (CFG 1.0), so the negative is a
  `ConditioningZeroOut` of the positive — the `NEGATIVE` output of
  `CharacterPromptLoader` is left unwired in this stage (kept on the node for
  reuse on the Wan path).
- `character_positive=""` / `character_negative=""` make `CharacterCreator`
  **preserve** the authored `character.json` (verified: `io.build_character`
  merges an empty layer over `existing`, leaving owned keys intact) rather than
  overwriting it.
- The `character` value is set in **both** `CharacterPromptLoader` and
  `CharacterCreator` (two combos, same value) — a minor redundancy; there is no
  wire between them because `CharacterCreator.character` is a widget, not an
  input.
- KSampler settings follow `1a` (4-step, cfg 1.0, euler/simple); the concept-art
  `EmptySD3LatentImage` size follows `1a` (≈768×1024).

---

## 4. Workflows 2–4 — Turnaround / Animate / Export

Direct reuse of the example `1b` / `2` / `3`, with exactly two widget changes
each: `AnimationManifestLoader.manifest → sybp.json` and the selector/builder's
`character → <a SYBP character>`. Everything else is unchanged.

- **2 — Turnaround** (from `1b`): `SweepLoopOpen/Close` wrap
  `PoseSweepSelector(mode=sweep, include_base=on)` → `PoseEditConditioning` →
  sampler → `PoseFrameWriter`. One Queue press fills the whole turnaround across
  every listed direction; `mode=target` + pose/direction spot-fixes one cell.
- **3 — Animate** (from `2`): `SweepLoopOpen/Close` wrap
  `AnimationSweepSelector(mode=sweep)` → dual hi/lo WAN 2.2 14B (+ lightx2v
  4-step + pixel-animate LoRAs) → `PainterFLF2V` → dual-pass ddim → `BiRefNetRMBG
  (invert_output=on)` + `JoinImageWithAlpha` (RGBA cutout) → `AnimationFrameWriter`.
  One press fills every clip; `mode=target` spot-fixes one. Requires
  `Comfyui-PainterFLF2V` + `comfyui-rmbg`.
- **4 — Export** (from `3`): `AnimationSheetBuilder(character, animation, all)` →
  `AtlasMetadataWriter(aseprite, …)`. Per-clip by design; change `animation` per
  run. Output feeds Aseprite for manual tweaks, then Godot.

These are direction-agnostic — the sweep iterates whatever each pose/animation's
`directions` map lists, so the platformer's EAST-centric coverage needs no
workflow change.

---

## 5. Non-blocking manifest follow-ups (owned by the user, not this build)

Recorded here so they aren't lost; none block the workflow set:

1. **`base` prompt vs. the manikin.** The SYBP shared `base` prompt does not
   reference the gray manikin the way the example does ("Match the body pose and
   orientation of the gray mannequin in the second image … never the mannequin's
   gray"). `PoseEditConditioning` attaches the manikin latent for every root/base
   pose regardless, so adding that wording is free per-direction orientation
   signal for the humanoid brains.
2. **E/W coverage.** Bidirectional animations (locomotion/combat) and the anchor
   poses `combat_stance` / `walk_stride` / `run_stride` are currently EAST-only;
   rendering both sides (asymmetric designs can't be auto-mirrored) means adding
   `WEST` (and selectively SE/SW/S, NE/NW/N) to the relevant `directions` maps.
3. **Non-humanoid `base` overrides.** `algorithm` (monolith) and `bureau_drone`
   (floating drone) override `base` but, being root poses, still get a humanoid
   manikin latent attached, which may fight the intended shape. A way to suppress
   the manikin for a specific base could be worth adding — flagged for a separate
   investigation.

---

## 6. Verification

- **Node:** `pytest -q` (incl. the new `CharacterPromptLoader` test),
  `ruff check .`, `mypy andypack` — all clean.
- **Workflows:** JSON well-formedness; and, when the pod is up,
  `validate_workflow` via the comfy MCP against live `object_info` to confirm
  every node type + widget resolves (notably `ReferenceLatent`,
  `CharacterPromptLoader`, `PainterFLF2V`, `BiRefNetRMBG`).
- **One value to confirm at deploy:** the manifest filename referenced by
  `AnimationManifestLoader` must match the file placed under the ComfyUI
  manifests dir (design assumes `sybp.json`).

---

## 7. Out of scope

- Per-character workflow copies (the combo makes them unnecessary).
- A batch/headless queue driver (generic set only, for now).
- A manifest/character sync script (deployment is handled manually).
- Godot-specific export tuning (Aseprite is the handoff; Godot is downstream).
- Editing the SYBP manifest content (the §5 items are the user's call).

# Example Workflows

Complete, UI-format reference workflows for the full character → animation →
sprite-sheet pipeline. Load one via ComfyUI's **Load** (or drag-and-drop); it
opens as a graph you can run. Each carries a **Note** node explaining the step.

Run them in order. State is disk-backed under `<output>/characters/<char>/`, so
each stage reads what the previous one wrote.

| File | Stage | Model | What it does |
|------|-------|-------|--------------|
| `1a_character_create.json` | Create | FLUX.2 Klein 9B | txt2img a character reference, persist it (`CharacterCreator`), and render `base@SOUTH` via `PoseEditConditioning`. |
| `1b_turnaround_batch.json` | Turnaround | FLUX.2 Klein 9B | `AutoPoseSelector(include_base)` → `PoseEditConditioning` → sampler → `PoseFrameWriter`. **Queue repeatedly** to fill every base + derived pose across all directions in ONE graph. |
| `2_animate_fflf.json` | Animate | WAN 2.2 14B i2v | `AutoAnimationSelector` → dual hi/lo (+ lightx2v 4-step + pixel-animate LoRAs) → `PainterFLF2V` → dual-pass ddim → BiRefNet alpha → `AnimationFrameWriter`. **Queue repeatedly** for every clip. |
| `3_sprite_export.json` | Export | — | `AnimationSheetBuilder` (rows = directions, cols = frames) → `AtlasMetadataWriter` (Aseprite). One node builds the game sheet + tagged atlas. |

## Requirements

- **Models** (see `pod/models.txt` in the deploy repo, or `docs/prompting-guide.md`):
  `flux-2-klein-9b` + `qwen_3_8b` encoder + `flux2-vae`; `wan2.2_i2v_high/low_noise_14B`
  + `umt5_xxl` + `wan_2.1_vae` + `clip_vision_h` + `lightx2v` 4-step LoRAs +
  `wan2.2_animate_adapter_model` (the pixel-animate LoRA).
- **Custom nodes**: `Comfyui-PainterFLF2V` (first-last-frame) and `comfyui-rmbg`
  (BiRefNet) for the animation workflow's alpha path, plus this pack.

> Note: `flux-2-klein-9b` requires the **8B-class** `qwen_3_8b` text encoder
> (4096-dim). The 4B repo's `qwen_3_4b` mismatches it (KSampler
> `mat1/mat2 ... 7680 vs 12288`).

## Editing for your character

Set `character` (and `character_positive` in `1a`) to your own values, and
`animation` in `3_sprite_export.json` to the clip you want to pack. The web
extension populates character/animation/direction combos from the loaded manifest.

## Key patterns

- **`PoseEditConditioning`** collapses the whole FLUX edit-conditioning chain
  (text encode + reference latents + zeroed negative + empty latent) into one
  node, and attaches the manikin reference only for base poses — so `1b` handles
  base (2-ref) and derived (1-ref) poses in a single graph.
- **`AnimationSheetBuilder`** packs a whole clip (every frame × every rendered
  direction) with a per-direction tagged atlas — Aseprite/Godot import one
  animation per direction.
- Alpha is baked at the writer boundary: `BiRefNet` (`invert_output=on`) →
  `JoinImageWithAlpha` → RGBA frames.

# Game-character sprite pipeline — Hooded Ranger (8-dir HD) — design

Date: 2026-06-30
Status: approved, building live

Build a set of **three fully-functional ComfyUI workflows** that take andypack's
coordinator nodes (previously skeletons with a "USER SAMPLER GAP") and fill the
gaps with real **Flux.2 Klein 9B** (poses) and **WAN 2.2 14B i2v FFLF** (animation)
sampling, producing a complete demo character end-to-end: an 8-directional HD 2D
hooded-ranger with idle + walk animations, packed into sprite sheets + Aseprite atlas.

## Decisions (from brainstorming)

- **Art target**: HD 2D sprites (high-res rendered, smooth shading). **No** palette
  quantization.
- **Directions**: full 8-way at **eye level** (side-scroller camera; character
  rotated S/SE/E/NE/N/NW/W/SW). The pack's `view_phrases` are already eye-level.
- **No mirroring**: the ranger (bow held one side, quiver slung, hood asymmetry) is
  not bilaterally symmetric → generate all 8 directions natively.
- **Character**: hooded ranger — cloaked archer/rogue, hood, cape, bow, quiver.
- **Demo scope**: full 8-dir turnaround + `idle` + `walk` loops in all 8 dirs
  (~16 WAN clips).
- **Deliverable**: a real demo character generated live, in named workflows the
  user opens and watches, backed by a shared manifest.
- **Animate adapter**: `wan2.2_animate_adapter_model.safetensors` IS the
  styly-agents "Wan2-2-pixel-animate" LoRA — applied as a LoRA to BOTH hi/lo noise
  models; improves animation, works for HD 2D. Keep it.
- **Export**: Aseprite JSON atlas. Plus animated GIF previews for QA.

## Environment (verified live)

- ComfyUI 0.26.2, PyTorch 2.10 cu130, NVIDIA RTX PRO 4000 Blackwell, 23.4 GB VRAM.
- diffusion_models: `flux-2-klein-9b.safetensors`,
  `wan2.2_i2v_high_noise_14B_fp16.safetensors`, `wan2.2_i2v_low_noise_14B_fp16.safetensors`
- loras: `wan2.2_animate_adapter_model.safetensors`,
  `wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors`,
  `..._low_noise.safetensors`
- vae: `flux2-vae.safetensors`, `wan_2.1_vae.safetensors`
- text_encoders: `qwen_3_4b.safetensors` (Flux2), `umt5_xxl_fp16.safetensors` (WAN)
- clip_vision: `clip_vision_h.safetensors` (used by PainterFLF2V; verify at build)
- Custom nodes confirmed installed: `PainterFLF2V` (Comfyui-PainterFLF2V),
  `BiRefNetRMBG`, andypack (33 nodes).

## Architecture: 3 staged workflows + shared manifest

Disk-backed handoff is andypack's model: Stage 1 writes base/stride poses per
direction → Stage 2 reads them as FFLF anchors and writes clips → Stage 3 reads
rendered poses/clips and packs them. Shared connective tissue: a new manifest
`ranger.json` and the on-disk character `ranger`.

Batch driving: andypack selectors are per-cell; the **Auto** selectors
(`AutoPoseSelector`, `AutoAnimationSelector`) auto-advance to the next actionable
(unrendered/stale) cell in dependency order — queue the prompt N times to sweep all
cells.

### Manifest `ranger.json` (version 2)

- `directions`: [EAST, SOUTH_EAST, SOUTH, SOUTH_WEST, WEST, NORTH_WEST, NORTH, NORTH_EAST]
- `mirror_map`: {} (empty — no mirroring)
- `view_phrases`: eye-level per-direction camera language (from prompting-guide §2 table)
- `defaults`: { fps: 16, length: 33, width: 480, height: 832, shift: 5,
  start_from: {ref: "base"} }
- `globals.pose`, `globals.animation` (positive/negative blocks)
- `poses`: `base` (root; manikin-driven, all 8 dirs), `walk_stride`
  (from: {ref: base}; single-ref re-pose, all 8 dirs)
- `animations`:
  - `idle`  → start_from base, end_at base (loop)
  - `walk`  → start_from walk_stride, end_at walk_stride (loop)
- Character prompt (hooded ranger) lives in `character.json` on disk (via
  CharacterCreator) and surfaces through `{character_prompt}`.

### Workflow 1 — `andypack_ranger_1_turnaround` (Flux.2 Klein 9B)

Reference art + 8-dir `base` + 8-dir `walk_stride`.

- Reference: `UNETLoader(flux-2-klein-9b)` + `CLIPLoader(qwen_3_4b, type=flux2)` +
  `VAELoader(flux2-vae)` → `CLIPTextEncode` (ranger prompt) →
  `ConditioningZeroOut` (neg) → `EmptyLatentImage(768x1024)` →
  `KSampler(steps 4, cfg 1.0, euler, simple, denoise 1.0)` → `VAEDecode` →
  feeds `CharacterCreator(character="ranger", direction=SOUTH, save_reference=on)`.
- Base pose per dir: `CharacterCreator`/`AutoPoseSelector` → `PoseUnpack`
  (source_image=reference, pose_reference=manikin, positive, output_dir).
  Multi-ref Flux edit: reference + manikin each `ImageScaleToTotalPixels(~1MP)` →
  chained `ReferenceLatent` → `KSampler` (same Klein settings) → `VAEDecode` →
  `PoseFrameWriter`. Prompt = pack's merged base prompt (guide §2a, manikin scoped
  to pose/orientation only).
- walk_stride per dir: `CharacterPoseSelector(pose=walk_stride)` → `PoseUnpack`
  (source_image = base for that dir) → single-ref Flux re-pose (guide §2b) →
  `PoseFrameWriter`.
- Batch: queue → `AutoPoseSelector` sweeps all unrendered (dir × pose) cells.

Klein settings: 4 steps, cfg 1.0, euler, simple. NO negative path
(`ConditioningZeroOut`). Keep both refs at same total-pixel target.

### Workflow 2 — `andypack_ranger_2_animate` (WAN 2.2 FFLF via PainterFLF2V)

Idle + walk loops, all 8 dirs. Wiring mirrors the user's proven
"2-Character Sprite Animation" workflow.

- `AutoAnimationSelector(character=ranger)` → `AnimationUnpack`
  → start_image, end_image, positive, negative, length, fps, width, height, shift,
  output_dir.
- Models:
  - HIGH: `UNETLoader(wan2.2_i2v_high_noise_14B_fp16)` →
    `LoraLoaderModelOnly(lightx2v_4steps high, 1.0)` →
    `LoraLoaderModelOnly(wan2.2_animate_adapter_model, 1.0)` →
    `ModelSamplingSD3(shift=5)`
  - LOW: `UNETLoader(wan2.2_i2v_low_noise_14B_fp16)` →
    `LoraLoaderModelOnly(lightx2v_4steps low, 1.0)` →
    `LoraLoaderModelOnly(wan2.2_animate_adapter_model, 1.0)` →
    `ModelSamplingSD3(shift=5)`
- `CLIPLoader(umt5_xxl_fp16, type=wan)` → `CLIPTextEncode` pos (from Unpack
  positive) / neg (from Unpack negative).
- `CLIPVisionLoader(clip_vision_h)` + `CLIPVisionEncode(start_image)` →
  PainterFLF2V clip_vision_start/end (reuse same output).
- `PainterFLF2V(positive, negative, vae=wan_2.1_vae, width, height, length,
  batch_size=1, motion_amplitude=1.1, start_image, end_image,
  clip_vision_start/end)` → (positive, negative, latent).
- Dual-pass sampler:
  - `KSamplerAdvanced(model=HIGH, steps 4, cfg 1.0, ddim, simple,
    start_at_step 0, end_at_step 2, return_with_leftover_noise=enable)`
  - `KSamplerAdvanced(model=LOW, steps 4, cfg 1.0, ddim, simple,
    start_at_step 2, end_at_step 4, return_with_leftover_noise=disable)`
- `VAEDecode(wan_2.1_vae)` → `AnimationFrameWriter(animation, frames)`.
  (Loop is derived by the resolver when start==end; writer drops dup final frame.)
- Batch: queue → AutoAnimationSelector sweeps all 8 dir × {idle, walk}.

Settings: 480×832 portrait, length 33 (~2s @16fps), shift 5, cfg 1.0, 4 steps
(2+2), ddim/simple. Note: fp16 14B dual-expert on 24 GB → offloading → minutes/clip.

### Workflow 3 — `andypack_ranger_3_export` (sprite sheets + Aseprite atlas)

- BG removal already applied at render time is optional; if frames are RGB, run
  `BiRefNetRMBG(model=BiRefNet_toonout, background=Alpha)` on loaded frames.
- Per animation: `CharacterAtlasBuilder(character=ranger, kind=anim, id=<idle|walk>,
  directions=all, layout, padding, power_of_two)` → (sheet IMAGE, ANIM_ATLAS) →
  `AtlasMetadataWriter(format=aseprite, name=ranger_<anim>)`.
- Optional QA: `SpriteTrimPivot` before packing; `AnimatedSpriteExport(format=gif,
  fps=16, loop=on)` per animation for a visual loop check.
- No `PaletteQuantizeLock` (HD choice).

## Build order (live, with checkpoints)

1. Write `ranger.json` into the manifests dir; load it; `ManifestLint` +
   `MergedPromptReport` to sanity-check prompts.
2. Build + save WF1. Smoke test: generate reference + `base` for ONE direction
   (SOUTH); show user. Then batch remaining base + walk_stride (all 8).
3. Build + save WF2. Smoke test: 1 clip (idle SOUTH); show user. Then batch the
   rest (~16 clips).
4. Build + save WF3. Pack sheets + Aseprite atlas + GIF previews.

Each workflow name is reported to the user as saved so they can open and watch.

## Risks / open items

- clip_vision_h presence: verify; PainterFLF2V clip-vision inputs are optional, so
  fall back to no-clip-vision if missing.
- Flux multi-ref edit fidelity across back/profile views (guide §2 failure modes:
  faces on back views, eye count on profiles) — rely on affirmative view_phrases +
  manikin orientation; regenerate individual cells as needed.
- WAN loop color drift when start==end (guide §1) — keep clips short (33),
  moderate shift; optional color-match pass if seams show.
- fp16 14B runtime — the long pole; smoke-test before committing to the full batch.
- Aspect consistency Flux(768×1024 ≈0.75) vs WAN(480×832 ≈0.577): PainterFLF2V
  resizes start/end to width/height; minor recompose acceptable. Revisit if
  distortion appears.

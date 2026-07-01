# andypack node-pack review — findings from an end-to-end build

Grounded in a full live e2e on 2026-06-30: hooded-ranger character → 8-direction
turnaround (base + walk_stride, FLUX.2 Klein) → idle + walk animations (WAN 2.2
FFLF via PainterFLF2V, alpha-cut) → sprite sheets + atlases. Every recommendation
below traces to concrete friction hit during that build.

## What worked well (keep as-is)

- The **FFLF resolver + manifest** core: cascade prompts, template vars
  (`{view_phrase}`/`{character_prompt}`/`{direction_prompt}`), gen-param plumbing
  (shift/length/fps/w/h wired straight into `ModelSamplingSD3`/`PainterFLF2V`),
  loop derivation, provenance/staleness. This is the pack's strength and it held up.
- `AnimationManifestLoader`, `CharacterCreator`, `CharacterReferenceLoader`,
  `CharacterPoseSelector`, `PoseUnpack`, `PoseFrameWriter`, `CharacterAnimationSelector`,
  `AnimationUnpack`, `AnimationFrameWriter` — the spine; clean and correct.
- `TurnaroundSheet` and `CoverageReport` — the review/QA loop; used constantly.
- Manikin bundling inside `CharacterCreator` (auto second reference per direction) —
  made the base turnaround Just Work.

## Bugs / rough edges hit (FIX)

1. **`CharacterAtlasBuilder` discarded `pack_sheet`'s atlas** (`sheet, _ = ...`) and
   returned an atlas dict without `sheet_size`/`frames` → `AtlasMetadataWriter` crashed
   with `KeyError: 'sheet_size'` for every format. **Fixed in this session** (merge
   `packed`). Deeper fix: make the ANIM_ATLAS a single shared builder output and add a
   test that *every* atlas producer round-trips through *every* `AtlasMetadataWriter`
   format — the builder/serializer contract is currently unenforced.

2. **`PoseUnpack` returns an empty 1×1 sentinel for a derived pose's `POSE_REFERENCE`.**
   This is the single biggest workflow-structure complication: `base` needs a
   **2-reference** Flux edit (character + manikin) while `walk_stride` needs a
   **1-reference** edit (character only). Because the sentinel can't be told apart in
   the graph, I had to build **two separate turnaround workflows**. Add a
   `HAS_POSE_REFERENCE` (BOOLEAN) or `REF_MODE` (STRING) output so one graph can branch
   — or, better, ship the helper in ADD-2.

3. **`AutoPoseSelector` interleaves root and derived poses** (a derived pose becomes
   actionable as soon as its one base direction exists), so it can't drive a single
   fixed-reference sampler graph. I fell back to explicit `CharacterCreator`(base) and
   `CharacterPoseSelector`(walk_stride) per direction. Give it a `pose_kind`
   (root/derived) filter or the same `REF_MODE` signal.

4. **`TurnaroundSheet` / `CharacterAtlasBuilder` return an IMAGE but aren't preview
   output nodes** — you must hand-wire a `PreviewImage` to see the sheet. Make them
   `OUTPUT_NODE` (emit a ui preview) or document.

## Gaps that forced manual/local work (ADD)

1. **`AnimationSheetBuilder` — full-frame directional sprite sheet. (Highest impact.)**
   `CharacterAtlasBuilder` only packs **one frame per direction** (a turnaround
   preview). Nothing packs a whole animation's *frames × all directions* into a game
   sheet (rows = directions, cols = frames). I built that locally with PIL. A node that
   loads every frame of every rendered direction, packs to `rows_per_direction`, and
   emits a frame-accurate `ANIM_ATLAS` (per-direction frame ranges + fps) consumable by
   `AtlasMetadataWriter` (aseprite/godot) would make Stage 3 a single node. Note
   `AnimationPlayback` already loads one direction's frames as IMAGE — generalize that.

2. **`PoseEditConditioning` — collapse the base/derived split.** A node
   `(pose, clip, vae) → (positive, negative, latent)` that text-encodes the pose prompt,
   `ReferenceLatent`s the source image, and `ReferenceLatent`s the manikin **only when
   present** (skipping the sentinel). One node replaces the ~8-node ref chain and lets a
   **single** turnaround workflow handle base + every derived pose. Removes the trickiest
   wiring in the whole build.

3. **`AnimationFrames` — a clean frame loader.** `(character, animation, direction) →
   IMAGE, fps`, no playback holds/loops semantics (unlike `AnimationPlayback`). Needed to
   re-process rendered clips (re-matte, re-time, re-export, pack) without re-sampling. I
   couldn't reload frames as a batch to add alpha post-hoc, forcing a full re-render.

4. **Alpha convenience (document, maybe thin node).** Engine sprites need alpha; the
   working path was external `BiRefNetRMBG` (needs *all* optional params, and
   `invert_output=true` for this art) → `JoinImageWithAlpha` → RGBA into
   `AnimationFrameWriter.frames`. The writer's own `mask` input *should* also work
   (`has_alpha = mask is not None`) but I only ever fed it while BiRefNet was erroring, so
   re-verify it; if solid, document "connect a MASK for alpha" as the one-liner.

## Consolidation candidates (evaluate — not confident these are dead)

I never needed these in a full character→sheet run; they add node-count surface area.
Keep only those with a real, exercised workflow:
- **Diagnostics sprawl:** `ManifestLint`, `CoverageReport`, `MergedPromptReport`,
  `StateMachineReport`, `RegenQueue` — five report nodes; I only reached for
  `CoverageReport`. Consider folding into one or two richer reports.
- `ActionSetSelector` overlaps `AutoAnimationSelector` + a category arg.
- `TweenClipProvider`, `BoomerangLoopWriter`, `ColorVariantBatcher`,
  `VariantLayerComposer`, `MirrorFrameWriter` — "process rendered frames without
  sampling" utilities. Legitimate ideas, but unproven here; gate them behind a clear
  utilities group or drop the ones without a reference workflow.

## Pack-level, high leverage (WORKFLOWS + DOCS)

- **Ship complete, working, UI-format reference workflows** in `examples/`. The current
  `turnaround.json`/`sprite_export.json` are skeletons with a "USER SAMPLER GAP" — I
  authored the real Flux-pose, WAN-FFLF, and export graphs from scratch. Ship: (a) Flux.2
  Klein reference + turnaround, (b) WAN 2.2 FFLF animation **with the alpha path**, (c)
  sprite-sheet + atlas export. Save them as **UI format** (nodes+links), not API — API
  JSON opens to an empty canvas in the ComfyUI library.
- **Pin the model/encoder pairing in docs:** `flux-2-klein-9b` requires the 8B-class
  `qwen_3_8b` encoder (4096-dim); the 4B repo's `qwen_3_4b` mismatches it
  (`mat1/mat2 ... 7680 vs 12288`). Note the WAN FFLF options (`PainterFLF2V` vs core
  `WanFirstLastFrameToVideo`).

## Suggested priority

- **P0 (most simplification):** `AnimationSheetBuilder`; `PoseEditConditioning`; ship
  complete UI reference workflows.
- **P1:** `PoseUnpack` `REF_MODE` output + `AutoPoseSelector` filter; atlas
  builder/serializer contract test; auto-preview on sheet nodes.
- **P2:** `AnimationFrames` loader; diagnostics consolidation; evaluate the niche
  utility nodes; alpha-path documentation.

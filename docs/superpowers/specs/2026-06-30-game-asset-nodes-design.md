# Game-asset nodes + correctness/completeness overhaul — design

Make comfyui-andypack a comprehensive, correct node pack for producing
video-game character **assets and sprites**: close the correctness/security
findings from the multi-agent review, add transparency/sprite-sheet/atlas
export, and add FFLF-native production nodes (turnarounds, identity anchoring,
state-machine projection, render-economy helpers).

Scope: **everything except** the Companion/normal-map (lit-2D) node, which stays
deferred. The implementation is a **single linear ordered sequence** (no phase
gates); each step ships with its tests.

## Background

The pack is a dependency-aware FFLF resolver (character → animation → direction
from one `animations.json`); it resolves prompts + anchor images + gating + meta
and writes back, but does **not** sample. A 31-agent review confirmed the core is
sound (careful `IS_CHANGED`, atomic writes, derived loop, transitive staleness)
with no critical bugs, but surfaced 12 verified correctness/security gaps and a
structural gap for game assets: the pack resolves and writes **3-channel RGB**
frames and has no alpha / sprite-sheet / atlas / engine-export path.

The source of truth stays the code + `examples/animations.json` + `README.md` +
`docs/prompting-guide.md`. This design adds nodes and fixes; it preserves the
documented invariants in `CLAUDE.md` (FFLF cross-wiring, loop-is-derived,
payload-first/sidecar-last, routes-take-no-client-paths, `resolve.py` torch-free).

## Cross-cutting decisions

1. **Alpha lives at the disk boundary, not in the tensor graph.** ComfyUI IMAGE
   tensors are 3-channel; we do NOT force 4-channel tensors through the graph
   (sentinels, anchors, references, previews stay 3-ch). Instead:
   - The image arriving at a writer/pack node **may already be 4-channel RGBA**
     (the user brought their own background-removal node) — its alpha is
     preserved.
   - **Optionally** a separate `MASK` input sets/overrides the alpha channel.
   - Neither present → today's 3-ch RGB behavior, unchanged.
2. **`images.py` owns all alpha handling.** `resolve.py` stays torch-free and
   alpha-unaware (paths only).
3. **New socket types are dict bundles** in the existing private-key style of
   `ANIM_POSE`/`ANIM_ANIMATION`: `SPRITE_TRIM`, `ANIM_ATLAS`, `ANIM_PALETTE`.
4. **New categories:** `andypack/Sprite`, `andypack/Export`.
5. **Every disk-reading node carries a disk-mtime `IS_CHANGED`** (the
   `_selector_fingerprint` pattern); writers keep payload-first / sidecar-last
   atomic ordering.
6. **TDD** against `pytest`; `ruff` + `mypy andypack` stay green.
7. **Docs updated as invariants move:** `CLAUDE.md`, `README.md`,
   `docs/prompting-guide.md`, and the node count.

## Alpha boundary (`images.py` + writers)

- `save_image_png(image, path, mask=None)`:
  - `mask` supplied → composite as alpha (4th channel), write RGBA. Overrides any
    incoming alpha.
  - else `image` is 4-channel → write RGBA preserving alpha.
  - else 3-channel → RGB (unchanged).
- `save_animated_webp` / frame writes slice `mask`/alpha per frame.
- `load_image_tensor(path, keep_alpha=False)`: default keeps today's white-matte
  3-ch behavior (anchors/references must stay 3-ch); `keep_alpha=True` returns
  4-ch RGBA for trim/pack reads so a saved sprite round-trips with transparency.
- Helpers: `to_rgba(image, mask=None)`, `alpha_bbox(image, threshold)`,
  `composite_alpha`. `empty_image()`/`is_empty` stay 3-ch sentinels.
- `mirror_png` already preserves alpha (no change).
- `PoseFrameWriter` / `AnimationFrameWriter`: add optional `mask` (MASK) input,
  detect 4-ch input, thread through; record `has_alpha` in the sidecar/meta via
  `io.build_pose_sidecar` / `io.build_animation_meta`.

## Correctness & security fixes (done first)

Ordered to de-risk the node work. Each is a verified review finding.

1. **Anchor-staleness (#1, `resolve.py`).** A complete animation is not re-staled
   when an anchor `ref` is swapped or `end_at` is added/removed (prompt-hash
   ignores anchors; provenance loop walks only *recorded* keys; transitive walk
   only checks self-outdatedness of the *new* dep). Fix: compute the current dep
   key-set (`{f"{ref}@{dir}"}` from `animation_deps` / pose `from`) and treat any
   key added/removed vs the recorded `sources` as drift, in `_outdated` and
   `stale_locally`. (Prefer the key-set comparison over folding anchors into the
   prompt hash, to keep the hash a pure-prompt signal.)
2. **`_EFFECTIVE_CACHE` keying (#2, `resolve.py`).** Cache keyed on
   `(id(manifest), id(identity))` can return a stale merged manifest after a
   base-manifest edit (freed-id reuse). Fix: key on a content signal — manifest
   `version` + a cheap structural hash (or the loader's mtime) plus the identity
   mtime — never raw `id()`.
3. **`MirrorFrameWriter` `IS_CHANGED` (#3, `nodes.py`).** No `IS_CHANGED` → caches
   on widget values and won't re-mirror after the source is re-rendered. Fix: add
   `IS_CHANGED` summing the resolved source PNG/frame mtimes (+ mirror_map), like
   the selectors.
4. **Path-traversal trio (#4/#5/#7).**
   - `server.py` GET routes: gate the `manifest` query value through
     `manifest_name_is_safe` (don't reuse `resolve_manifest_path`, which allows
     absolute paths for trusted node inputs).
   - `/options`: drop the client `character_dir` path; resolve the character only
     by validated bare name (`_is_safe_segment`) under `characters_dir()`.
   - `manifest.py`: validate every pose id, animation id, and direction **name**
     as a safe single path segment (no separators, no `..`) in `_validate_refs` /
     `_validate_directions`, mirroring `_is_safe_segment`.
5. **Gen-param validation (#6/#9/#12, `manifest.py` + bundle).** Missing/zero/
   negative `width/height/length/fps` slip through and emit `0` into the Wan
   sampler. Fix: require each animation's effective `width/height/length/fps`
   (per-animation or `defaults`) to resolve to a **positive** int at load (fatal
   ManifestError); fix the modulo edge so negatives are caught; clamp the
   `_build_animation_bundle` `fps` output ≥1 (consistent with `animation_fps`).
6. **Playback loopable gate (#8, `resolve.py`).** `playback_segments` treats
   same-`ref` anchors as loopable even when start/end **frames** differ. Fix:
   gate `loopable` on resolved `start_image == end_image` (same condition
   `resolve_animation` uses for `is_loop`).
7. **Empty-sentinel guard (#10, `nodes.py`).** `AnimationFrameWriter` rejects an
   empty batch but not the 1×1 `empty_image()` sentinel. Fix: reject
   `images.is_empty(frames)` before writing.
8. **Character-combo refresh (#11, `web/` + docs).** Characters created mid-
   session don't appear in other nodes' combos until refresh. Fix: the web
   extension repopulates the character combo via `/characters`; document the
   refresh requirement.

## Existing-node features

- **RGBA write path + `has_alpha`** on both writers (the alpha boundary above).
- **`MirrorFrameWriter` batch mode** — `mirror_all` (or `direction="ALL"`):
  iterate `mirror_map`, mirror every destination whose source side is rendered,
  in one queue run; reuse `_mirror_pose`/`_mirror_animation`; pairs with the
  `IS_CHANGED` fix. Rename the `id` widget off the Python builtin.
- **`skip_mirrored` (default True)** on `AutoPoseSelector` + `AutoAnimationSelector`
  — `next_actionable` drops directions present as keys in `mirror_map` so the
  batch never samples a mirrorable view (≈halves generation). `api` change only.
- **Thumbnail route + coverage-grid thumbnails** — `GET /anim_coord/thumb?...`
  returns a base64 PNG data-URI (pose image / animation first frame) from a
  **server-resolved** `kind/id/direction` (adds the `_is_safe_segment` validation
  `/options` lacks → also closes #5's read vector). Panel renders lazy `<img>`
  per cell. JSON-only, no file bytes, no client FS path.
- **`AnimationPlayback` modes** — `mode` combo `loop | ping_pong | once |
  hold_last` + `hold_frames`. Post-process the assembled batch in `images.py`
  (`torch.flip` interior / repeat last); `playback_segments` unchanged; reflect
  `mode`/`hold_frames` in `IS_CHANGED`. No manifest loop field.
- **Characters tab: reference thumbnail + overlay editor** — show
  `<char>/_reference.png` (thumb route, `kind=reference`) and a JSON editor for
  the `character.json` `poses`/`animations` overlay; save via `/character/save`
  (widen payload; `build_character` already preserves non-owned keys).
- **Space-safe coverage grid keys + sampled-vs-mirrored tally** — carry a
  `{kind,category,id}` object instead of space-join/split; mark mirror-target
  directions distinctly using `mirror_map`.

## New nodes (everything except Companion maps)

### Sprite / export chain

- **Sprite Trim & Pivot** (`andypack/Sprite`). Alpha-bbox crop; `union` mode
  crops every frame to one shared bbox (8-direction cycle stays registered);
  `per_frame` records individual offsets. Computes a pivot (center / bottom-center
  feet / top / custom). Out: trimmed IMAGE batch + `SPRITE_TRIM` (per-frame
  source_size, trim offset, pivot). Inputs: `image (IMAGE RGBA)`,
  `alpha_threshold`, `trim_mode`, `pivot`, optional `pivot_x/pivot_y`, `pad`.
- **Spritesheet Packer** (`andypack/Sprite`). Packs a frame batch into one sheet
  (grid / horizontal / vertical / maxrects), padding + edge-extrude bleed +
  optional power-of-two. Out: `(IMAGE SHEET, ANIM_ATLAS)` where ATLAS carries
  sheet_size + per-frame rect/source_size/trim/pivot/duration_ms (durations from
  wired `fps`). Inputs: `image`, `layout`, `columns`, `padding`, `extrude`,
  `power_of_two`, optional `trim_data (SPRITE_TRIM)`, `fps (forceInput)`, `names`.
- **Character Atlas Builder** (`andypack/Sprite`). FFLF-aware multi-direction
  packer: resolve each direction's `OUTPUT_DIR` via the resolver, gate on
  `node_complete` (skip + report unrendered), pack an 8-direction (or cardinal-4 /
  subset) sheet (one row per direction) in one node; mirror-derived directions
  included automatically (already on disk). Disk-mtime `IS_CHANGED`. Out:
  `(IMAGE SHEET, ANIM_ATLAS, STRING REPORT)`.
- **Atlas Metadata Writer** (`andypack/Export`). Terminal export, OUTPUT_NODE:
  write the sheet PNG first, then the metadata sidecar LAST (atomic). Formats:
  TexturePacker (hash/array), Unity sprite-sheet meta, Godot `SpriteFrames`
  `.tres`, Aseprite JSON, CSS sprites. Records `render_id` provenance when an
  `ANIM_ANIMATION` is wired. In: `atlas (ANIM_ATLAS)`, `sheet (IMAGE)`, `format`,
  `name`, optional `output_subdir`, `animation (forceInput)`.
- **Palette Quantize & Lock** (`andypack/Sprite`). Wire an `ANIM_PALETTE` to
  remap/lock; else median-cut/k-means quantize to N colors and emit the palette.
  Preserve alpha; optional dither; `extract_only` mode. Out: `(IMAGE,
  ANIM_PALETTE)`. Extract once from base, fan the palette across directions.

### FFLF production

- **Manikin Pose Control** (`andypack/Pose`). Expose the bundled per-direction
  manikins (`manikins.manikin_path`) as a first-class `POSE_CONTROL` IMAGE + the
  resolved positive prompt + canonical `DIRECTION_NAME`, to drive a
  ControlNet/DWPose path (not just the Character Creator's FLUX edit slot). Pure
  load.
- **Character Identity Anchor** (`andypack/Character`). Assemble `_reference.png`
  + the already-rendered base pose for the **same** direction into an anchor
  batch for IPAdapter/Redux conditioning (attacks cross-direction identity
  drift). Selector-style mtime `IS_CHANGED`. Out: `(REFERENCE_IMAGE,
  BASE_DIRECTION_IMAGE, ANCHOR_BATCH)`.
- **Action Set Selector (next job)** (`andypack/Animation`). Auto-selector scoped
  to one manifest `category` (locomotion/combat/…): emit the next ready/stale clip
  in that set, report remaining, raise when the set is complete. Add a
  `category`/`predicate` kwarg to `next_actionable` (keep the wedge-avoidance /
  `stale_locally` logic); build via `_build_animation_bundle`; honor
  `effective_manifest`. Out: `(ANIM_ANIMATION, REMAINING, REPORT)`.
- **State Machine Report** (`andypack/Diagnostics`). Project the implicit animator
  controller from the FFLF graph (`start_from`/`end_at` refs = transitions,
  same-ref anchors = self-loop) as a transition table + importable JSON; doubles
  as authoring validation (orphan states, missing exits, direction coverage).
  Read-only inside `resolution_pass`, `nan` `IS_CHANGED`. Out: `(REPORT, JSON)`.
- **Turnaround Sheet** (`andypack/Diagnostics`). Composite every rendered
  direction of a pose (default `base`) into one labeled contact sheet in
  canonical order with placeholders for missing views; OUTPUT_NODE with preview.
  Read-only, `nan` `IS_CHANGED`. Out: `(IMAGE SHEET)`.

### Render-economy / animation

- **Boomerang Loop Writer** (`andypack/Animation`). Synthesize A→B→A from a
  one-way clip (append reversed interior) for seamless idle/breath loops at half
  the Wan cost. Satisfies the loop invariant **by construction** (first frame ==
  last frame ⇒ `start_image == end_image`); does not author a loop flag. Takes a
  live IMAGE input (no stale-cache concern). OUTPUT_NODE.
- **Tween Clip Provider** (`andypack/Animation`). Resolve an FFLF clip's start/end
  anchors (start = dep last_frame, end = dep first_frame — do NOT invert) + emit
  the derived in-between count to feed an external RIFE/FILM node for near-linear
  transitions. Load/resolve only (stays "does not sample"). Validate the clip is
  genuinely FFLF (`start != end`). Out: `(START_IMAGE, END_IMAGE, TWEEN_COUNT,
  FPS)`.
- **Frame Timing Normalizer** (`andypack/Animation`). Retime a rendered batch to
  an exact target count (resample / trim / hold-pad) with optional 4n+1 snap so
  every direction (and mirror/interpolation pair) shares a loop-clean count. Its
  `LENGTH` output drives the writer's meta count so completeness stays exact. Out:
  `(FRAMES, LENGTH)`.
- **Color Variant Batcher** (`andypack/Animation`). Derive team-color/skin
  variants of an already-rendered pose/animation via deterministic recolor (hue
  rotate / sat-val scale / target-hex remap) to sibling targets
  (`<id>__<variant>/…`), copying source seed + render_id (the MirrorFrameWriter
  disk-to-disk pattern). Disk-mtime `IS_CHANGED`. OUTPUT_NODE.
- **Variant Layer Composer** (`andypack/Pose`). Inject an outfit/equipment prompt
  fragment as an extra cascade layer at wire time (reuse `merge_layers` /
  `merge_negative` / `compute_prompt_hash`), recompute `prompt_hash` (staleness
  still tracks), retarget `output_dir` to a `<id>__<variant>` sibling. Passthrough
  ANIM_POSE bundle.
- **Animated Sprite Export** (`andypack/Export`). Export a frame batch as looping
  GIF / APNG / WebP with in-node preview; optional onion-skin (ghosted prev/next
  frames). Reuse `save_animated_webp` / `_animated_preview`; add GIF/APNG via
  Pillow; return `{}` headless. OUTPUT_NODE.

## Deferred (explicitly out of scope)

- **Companion Map Emitter** (normal/emissive/roughness/AO for lit-2D). Adds a
  second sampler pass per asset and targets a narrower audience. Revisit after
  the sprite chain + mirror economy land.

## Testing

- `pytest` TDD per step. New test modules mirror the existing layout
  (`tests/test_<area>.py`): `test_alpha`/`test_images` (RGBA write/round-trip,
  mask override, 3-ch fallback), `test_sprite_trim`, `test_atlas_pack`,
  `test_atlas_export`, `test_palette`, plus regression tests for each of the 12
  fixes (anchor-swap staleness, effective-cache invalidation on base edit, mirror
  `IS_CHANGED`, path-traversal rejection on routes + ids, gen-param validation,
  playback loop gate, empty-sentinel guard).
- The Unpack-key sync test pattern extends to any new bundle that has an Unpack.
- `ruff check .` + `mypy andypack` green; `resolve.py` import-checked to stay
  torch-free.

## Linear execution order

One ordered sequence (each step ships with tests):

1. Correctness & security fixes — groups 1–8 above (pure core + routes first).
2. Alpha boundary in `images.py` + writer `mask`/RGBA + `has_alpha` meta.
3. Sprite export chain nodes — Sprite Trim & Pivot → Spritesheet Packer →
   Character Atlas Builder → Atlas Metadata Writer → Palette Quantize & Lock.
4. FFLF production nodes + `skip_mirrored` + MirrorFrameWriter batch mode —
   Manikin Pose Control, Character Identity Anchor, Action Set Selector, State
   Machine Report, Turnaround Sheet.
5. Render-economy / polish nodes + `AnimationPlayback` modes — Boomerang Loop
   Writer, Tween Clip Provider, Frame Timing Normalizer, Color Variant Batcher,
   Variant Layer Composer, Animated Sprite Export.
6. Panel / route features — thumbnail route + coverage thumbnails, Characters tab
   reference + overlay editor, space-safe grid keys + mirror tally.
7. Docs + example workflows — `CLAUDE.md` (new nodes/types/categories, moved
   invariants, node count), `README.md`, `docs/prompting-guide.md`, example
   graphs for the sprite-export and turnaround flows.

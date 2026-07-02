# comfyui-andypack — Animation Coordinator

A ComfyUI custom-node pack for building large, **direction-aware character
animation sets** from a single manifest. It is a dependency-aware **FFLF**
(first-frame / last-frame) resolver: it drives `character → pose → animation`
selection from one `animations.json`, gates what you can generate on what is
already rendered, and feeds your sampler the right positive/negative prompts and
start/end anchor images.

It does **not** sample or generate. You build the FLUX (pose/frame edits) and
WAN (animation) graph; this pack resolves prompts, reference images, dependency
gating, and completion metadata, and writes the results back to disk so the next
node in the chain unlocks.

---

## Mental model

Everything is one dependency graph of two rendered node kinds, rooted at the
**base** pose:

```mermaid
graph TD
  ref["character reference<br/>(node input, persisted as _reference.png)"] --> base
  manikin["manikin[dir]<br/>(bundled, per direction)"] --> base["base pose<br/>(root, per direction)"]
  base --> stance["fighting_stance pose"]
  stance --> idle["fighting_stance_idle<br/>(WAN loop)"]
  base -. start_from .-> entry["fighting_stance_entry"]
  idle -. end_at .-> entry
  idle -. start_from + end_at .-> punch["punch / kick / …"]
```

- **Base pose** — the tree root. Each of its 8 directions is a FLUX.2 multi-
  reference edit of the **character reference image** (a Character Creator input,
  persisted as `_reference.png` so it can be reloaded) paired with the bundled
  **manikin** for that direction (which supplies the camera angle / body
  orientation). Renders to `_base/{dir}.png` + a sidecar. A character's prompt
  layer lives in `character.json` (no provenance — base sidecars root staleness).
- **Pose** — a per-direction still produced by a FLUX.2 edit of a *source image*
  (another pose). A pose with no `from` is a root pose (base). Renders to
  `_{pose}/{dir}.png` + a sidecar.
- **Animation** — a Wan 2.2 i2v clip. Renders to `{anim}/{dir}/frame_*.png` +
  `meta.json`.

### FFLF cross-wiring

Every animation needs a **start image** (the I2V initial latent): its explicit
`start_from`, otherwise the manifest's `defaults.start_from`. `end_at` is
optional — when present, the clip is FFLF. The cross-wiring is:

- `start_from` consumes the dependency's **last** frame.
- `end_at` consumes the dependency's **first** frame.
- A single-image dep (a pose) resolves the same image for either slot.

**Looping is a consequence of FFLF, not a flag.** There is no `loop` field. A
clip loops when its start and end anchors resolve to the *same image* (e.g.
`start_from` and `end_at` both pointing at one pose) — it begins and ends on the
same frame. The Animation Frame Writer detects that and drops the duplicated
final frame so the clip plays seamlessly on repeat.

### Cascading prompts

Each render's final positive and negative are merged from layers, general →
specific:

```
globals[kind] → entity
```

The character prompt layer (`character.json`) and the per-direction layer are
**not** cascade layers — they surface only via the opt-in template variables,
resolved by field context (positive vs negative):

| Variable | Expands to |
|---|---|
| `{character_prompt}` | the character's `positive_prompt` / `negative_prompt` |
| `{view_phrase}` | the manifest-level `view_phrases[direction]` camera phrase (positive only) |
| `{direction_prompt}` | the entity's own per-direction `positive_prompt` / `negative_prompt` |
| `{direction_name}` | the bare direction name (e.g. `EAST`) |

`{view_phrase}` is the key to the 8-direction workflow: per-direction camera
language lives **once** in the manifest's `view_phrases` map, so every pose and
animation can opt into all 8 directions with empty per-direction layers and still
get correct, affirmative view language — including the turnaround failure-mode
mitigations ("only one eye visible" on profiles, "no face visible" on back
views). The manifest stays **character-agnostic**: identity arrives only via
`{character_prompt}`.

Positives are joined as prose; negatives are merged as a deduped, comma-separated
term list. The merged prompt is hashed into the sidecar/`meta.json` as
`prompt_hash`.

> **FLUX.2 Klein has no negative-prompt path**, so the seed manifest's poses carry
> no negative layer — pose failure modes are mitigated affirmatively in the
> positive (via `view_phrases`). The negative pipeline is for the Wan animation
> path (effective at CFG > 1). See [`docs/prompting-guide.md`](docs/prompting-guide.md).

### Staleness

Staleness is **transitive on the prompt hash**. A complete node is `stale` if its
own merged-prompt hash drifted from what was rendered, **or** any ancestor is
stale. Editing the character prompt layer or any cascade layer ripples
downstream. A stale node stays selectable — it just shows amber so you know to
re-render.

---

## Installation

Clone (or symlink) this repo into your ComfyUI `custom_nodes/` directory:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/andyhite/comfyui-andypack.git
```

Restart ComfyUI. The nodes appear under the **andypack** category, and the web
extensions (`web/anim_coord.js` — dynamic combos; `web/anim_coord_panel.js` — the
sidebar manager) load automatically.

Runtime deps are the ones ComfyUI already provides (`torch`, `numpy`, `Pillow`,
`aiohttp`). No extra install step is required for normal use.

### Where files live

| What | Location |
|---|---|
| Manifests | `ComfyUI/user/default/andypack/animations/*.json` |
| Character output | `ComfyUI/output/characters/<character>/` |

A **character** is any directory under the characters root containing a
`character.json`, a `_reference.png`, a pose dir, or an animation dir. The
reference art is persisted by default (`save_reference` on the Character Creator)
so a character can be reloaded and its base re-generated; turn it off to keep the
reference only in your graph.

```
output/characters/cortex/
  character.json                    character prompt layer { positive_prompt?, negative_prompt? } (no provenance)
  _reference.png                    persisted reference art (optional; Character Reference Loader reads it)
  _base/EAST.png   _base/EAST.json  base pose frame + sidecar (the tree root)
  fighting_stance_idle/EAST/
    frame_00000.png … frame_000NN.png
    meta.json                       written LAST (atomic) = completion sentinel
```

There is no `.complete` file. The sidecar / `meta.json` is written **last** via
temp-file + atomic rename; its presence is the completion signal. A directory
with no parseable meta/sidecar reads as incomplete.

---

## Nodes

All nodes live in the **andypack** category. Custom passthrough types:
`ANIM_MANIFEST` (the loaded, validated manifest), `ANIM_POSE` (a pose job bundle
from a selector to its writer) and `ANIM_ANIMATION` (an animation job bundle).
The `*Unpack` nodes fan a bundle out into individual typed outputs.

| Node | Role |
|---|---|
| **Animation Manifest Loader** | Load + validate `animations.json` (ref typing, cycle detection, `4n+1` length + `view_phrases` lint). Cached by file mtime. |
| **Character Creator** | Write a character's `character.json` prompt layer and emit the base-pose job for one direction, pairing the reference image (`SOURCE_IMAGE`) with the bundled manikin (`POSE_REFERENCE`) for a multi-reference FLUX.2 edit. Optionally persists the reference art (`save_reference`, default on). |
| **Character Loader** | Read-only sibling of the Character Creator: emit the base-pose job for an existing character + direction (reference image + manikin → multi-reference FLUX.2 edit) **without writing `character.json`**. Use when the character's prompt layer is already authored and must be preserved (e.g. the SYBP Create workflow generates the reference art from `character.json`, then loads the base pose). Optionally persists the reference art (`save_reference`, default on). |
| **Character Reference Loader** | Reload a character's persisted reference art (`_reference.png`) as an IMAGE — feed it back into the Character Creator to re-generate base directions later. |
| **Pose Sweep Selector** | `mode=sweep` emits the *next* actionable (ready/stale) pose in dependency order — drive it inside a Sweep Loop to batch-generate every pose in one Queue press; raises when none remain. `include_base=on` also emits root (base) poses paired with their manikin, so one graph (→ Pose Edit Conditioning) drives the whole turnaround. `mode=target` force-regenerates one named `pose@direction` as a spot-fix, ignoring the completeness gate. Takes an optional `flow` (`SWEEP_FLOW`) input to sit inside a Sweep Loop. |
| **Unpack Pose** | Fan an `ANIM_POSE` out into `SOURCE_IMAGE`, `POSE_REFERENCE`, `POSITIVE_PROMPT`, `NEGATIVE_PROMPT`, `OUTPUT_DIR`, `HAS_POSE_REFERENCE` (and forward the bundle). |
| **Pose Frame Writer** | Write `{dir}.png` then the `{dir}.json` sidecar last (atomic). Returns `(OUTPUT_DIR, REMAINING)` — `REMAINING` is the live post-write actionable count (sweep mode) or `0` (target mode), feeding a Sweep Loop's continue/stop signal. Optional `write_mirrored` also writes a horizontally-flipped copy into every `mirror_map` direction derived from the one just written, each with its own sidecar and a `mirrored_from` provenance key (sound only for bilaterally symmetric designs — see [`docs/prompting-guide.md`](docs/prompting-guide.md) §4). |
| **Animation Sweep Selector** | `mode=sweep` emits the *next* actionable animation in dependency order — drive it inside a Sweep Loop to batch-generate every clip in one Queue press; raises when none remain. `category` scopes the sweep to one manifest category. `mode=target` force-regenerates one named `animation@direction` as a spot-fix. Emits an `ANIM_ANIMATION` bundle: `START_IMAGE`, `END_IMAGE`, `IS_FFLF`, merged prompts, plus `LENGTH`/`FPS`/`WIDTH`/`HEIGHT`/`SHIFT` that wire straight into `WanFirstLastFrameToVideo` + `ModelSamplingSD3`. Takes an optional `flow` (`SWEEP_FLOW`) input to sit inside a Sweep Loop. |
| **Unpack Animation** | Fan an `ANIM_ANIMATION` out into its typed outputs (start/end image, prompts, is_fflf, length, fps, width, height, shift, output_dir). |
| **Animation Frame Writer** | Write `frame_{:05d}.png`, trim the duplicate closing frame of a seamless loop, then write `meta.json` last (atomic). Records the sampler `seed`. Returns `(OUTPUT_DIR, REMAINING)` — same sweep-loop continue/stop signal as Pose Frame Writer. Optional `write_mirrored` (same semantics as Pose Frame Writer's) and `loop_color_match` (ramps a per-channel color match toward the first frame across a derived loop clip, mitigating the low-noise expert's color/contrast drift on the `start_image == end_image` seam). |
| **Animation Frames** | Load a rendered clip back as an IMAGE batch (+ fps) to reprocess (re-matte, re-pack, re-export) without re-sampling. |
| **Pose Edit Conditioning** | One-node FLUX.2 pose-edit conditioning: text-encode + source (and manikin-when-present) reference latents + zeroed negative + empty latent → `(positive, negative, latent)`. |
| **Manikin Loader** | Load a bundled per-direction manikin as an IMAGE (+ its direction name) — the pose/camera source for authoring custom pose references (drive an OpenPose/ControlNet-capable graph per direction). |
| **Pose Reference Writer** | Save an IMAGE into the pose-references dir (`user/default/andypack/pose_references/`) as `<name>_<DIRECTION>.png` and return the filename — exactly what a pose direction layer's `reference_image` points at. |
| **Wan Animation Conditioning** | One-node Wan 2.2 conditioning from an `ANIM_ANIMATION` bundle: text-encode + core FFLF encode, omitting `end_image` entirely for non-FFLF clips (the sentinel never reaches the sampler). Wire `SHIFT` into `ModelSamplingSD3` as before. |
| **Coverage Report** | A status table over every `(entity, direction)` for a character: generated / ready / stale / blocked, plus a JSON blob. |
| **Sweep Loop Open** | Marks the start of a one-press sweep loop; emits a `SWEEP_FLOW` token wired into the sweep body's selector (`flow`) and into Sweep Loop Close (`flow`). |
| **Sweep Loop Close** | Closes the loop: while the writer's `REMAINING` (wired to `remaining`) is `> 0`, clones and re-expands the Open→Close body so the engine runs another iteration; terminates cleanly at `remaining <= 0`. |
| **Sheet Export All** | Stage-3 batch export: one sheet + atlas per animation with ≥1 rendered direction, in one queue press. Skipped animations are listed in the report. |
| **Palette Quantize & Lock** | Force frames onto one shared limited palette (optionally locked to a `palette_image`) for pixel-art consistency across directions/animations. |
| **Frame Retime** | Resample a clip to a target fps (resample / trim / pad-hold) before packing/export. |

Most of this is also driveable from the **Andypack sidebar panel** (see below).

### Typical graph

1. **Animation Manifest Loader** → `MANIFEST`.
2. **Character Creator** per base direction (reference image + manikin → base
   pose) → **Unpack Pose** → FLUX.2 multi-reference edit (`SOURCE_IMAGE` first,
   `POSE_REFERENCE` second) → **Pose Frame Writer**. The reference art is
   persisted, so **Character Reference Loader** can re-supply it later.
3. **Pose Sweep Selector** (`mode=target`, one pose/direction) → **Unpack Pose**
   → FLUX.2 edit → **Pose Frame Writer**. Walk poses in dependency order (poses
   that build on `base`).
4. **Animation Sweep Selector** (`mode=target`, one animation/direction) →
   **Unpack Animation** → **WanFirstLastFrameToVideo** (`START_IMAGE`→`start_image`,
   `END_IMAGE`→`end_image`) → KSampler → VAE Decode → **Animation Frame Writer**.

See [Graph wiring](#graph-wiring) for the exact FLUX.2 and Wan node connections.

**Batch generation (one-press sweep loop).** To work through a whole character
without hand-picking every `(entity, direction)` or re-queueing per cell, set the
**Pose Sweep Selector** / **Animation Sweep Selector**'s `mode` to `sweep` and wrap
the body (selector → … → writer) in **Sweep Loop Open** / **Sweep Loop Close**:
wire `Sweep Loop Open.flow` into the selector's optional `flow` input AND into
`Sweep Loop Close.flow`, and the writer's `REMAINING` output into
`Sweep Loop Close.remaining`. Each pick still picks the next actionable job in
dependency order, but now a single Queue press re-runs the body until nothing
remains, instead of requiring a manual re-queue per cell (see
`examples/workflows/1b_turnaround_batch.json` / `2_animate_fflf.json`). Generate
the 8 `base` directions with the Character Creator first (those need the
reference + manikin).

The web extension repopulates the combos with live status glyphs after each
writer run, so newly-unlocked nodes appear without a manual refresh:

> ✅ generated · 🔵 ready · 🟠 stale · 🔴 blocked

---

## Manifest

The manifest is **character-agnostic and identity-free** — per-character prompt
text lives only in each character's `character.json`. See
[`examples/animations.json`](examples/animations.json) for a full, working
manifest (generated by [`scripts/build_seed_manifest.py`](scripts/build_seed_manifest.py),
which is the easiest way to author a large set), and the resolver code
(`andypack/manifest.py`, `andypack/resolve.py`) for the authoritative schema.

Top-level keys: `version`, `directions` (canonical 8-way ordering),
`mirror_map`, `view_phrases` (per-direction camera language, injected via
`{view_phrase}`), `defaults` (`fps` / `length` / `width` / `height` / `shift` /
`start_from`), `globals` (`animation` / `pose` cascade layers), `poses`, and
`animations`. The `base` pose has no `from` (it is the tree root) and lists all 8
directions. Every anchor pose and animation lists all 8 directions too (usually
with empty per-direction layers, leaning on `view_phrases` + the entity prompt).

A character can extend the manifest with its own `poses` / `animations` by adding
them to its `character.json`; the merged manifest is re-validated, so a bad ref or
a cycle is rejected rather than resolved silently.

### Manikins

The 8 bundled pose references in `andypack/assets/manikins/<DIR>.png` (one per
canonical direction) supply the camera angle / body orientation for the base
pose. The Character Creator pairs the character reference image with the matching
manikin as a second FLUX.2 reference, so all 8 base directions are generated
directly — base does not rely on `mirror_map`.

A pose's per-direction layer can also carry its own `reference_image`: a bare
`*.png` filename resolved under `user/default/andypack/pose_references/`. On a
**root** pose (no `from`, e.g. `base`) it *overrides* the bundled manikin as the
second FLUX.2 reference; on a **derived** pose it *adds* a second reference where
there was none. Author these with **Manikin Loader** (load the bundled manikin
for a direction) → your own pose graph (e.g. an OpenPose ControlNet on an
SDXL/SD1.5 checkpoint, since FLUX.2 Klein has no ControlNet path) → **Pose
Reference Writer** (saves `<name>_<DIRECTION>.png` and returns the filename to
paste into the manifest).

---

## HTTP routes

Registered on `PromptServer.instance.routes` when running inside ComfyUI:

All routes return **JSON only** and take **no client filesystem path** — a
manifest is addressed by a validated bare basename resolved under the pack's
manifests dir, a character by a name snake-cased to one segment under the
characters dir. There is nothing to traverse out of.

| Route | Purpose |
|---|---|
| `GET /anim_coord/ping` | Liveness check the frontend uses before enabling inputs. |
| `GET /anim_coord/characters` | List character directories (server-resolved). |
| `GET /anim_coord/manifests` | List available manifest filenames. |
| `GET /anim_coord/manifest_options?manifest=…` | Pose/animation → directions map (no rendered tree needed). |
| `GET /anim_coord/options?manifest=…&character=…` | Every `(pose\|animation, direction)` with its status + `blocked_by`. |
| `GET /anim_coord/manifest?name=…` | Raw manifest JSON text (for the editor). |
| `POST /anim_coord/manifest/save` `{name, content}` | Validate + atomically save a manifest (rejected, not saved, if invalid). |
| `GET /anim_coord/character?character=…` | A character's `character.json` prompt layer. |
| `POST /anim_coord/character/create` `{character}` | Create a character dir + empty `character.json`. |
| `POST /anim_coord/character/save` `{character, positive_prompt, negative_prompt}` | Write a character's prompt layer (preserving any overlay). |

### Sidebar panel

`web/anim_coord_panel.js` registers an **Andypack** sidebar tab
(`app.extensionManager.registerSidebarTab`) with three sections backed by the
routes above:

- **Manifest** — load / edit / validate-and-save a manifest JSON.
- **Characters** — create a character and edit its `character.json` prompt layer.
- **Coverage** — a live status grid over every `(entity, direction)` for a
  character, auto-refreshing after each graph run.

---

## Graph wiring

The pack resolves and writes back; **you** build the FLUX.2 and Wan sampler
graphs. The confirmed node wiring (late 2025 / 2026):

**Poses — FLUX.2 Klein multi-reference edit.** Per reference image
(`SOURCE_IMAGE` first, `POSE_REFERENCE`/manikin second), normalize with
`ImageScaleToTotalPixels` (~1 MP) then chain a `ReferenceLatent` per reference
onto the conditioning. Distilled Klein: 4 steps, guidance ~1.0, Euler + Simple
(never Euler Ancestral). FLUX.2 has no negative prompt — leave `NEGATIVE_PROMPT`
unwired. Pipe the edit → **Pose Frame Writer**.

**Animations — Wan 2.2 14B i2v first-last-frame.** The recommended path is
**Animation Sweep Selector → Wan Animation Conditioning → samplers → Animation
Frame Writer**: the conditioning node text-encodes the bundle's merged prompts
and delegates to the core `WanFirstLastFrameToVideo`, omitting `end_image`
entirely for a non-FFLF clip so one graph serves both FFLF and plain-i2v
animations without a switch. Manual `WanFirstLastFrameToVideo` wiring remains
valid too — driven by the standard Wan 2.2 i2v models
(`wan2.2_i2v_high_noise_14B` + `wan2.2_i2v_low_noise_14B`, `umt5_xxl` text
encoder, `wan_2.1_vae`). Wire `START_IMAGE`→`start_image`,
`END_IMAGE`→`end_image` (leave `end_image` unconnected when `IS_FFLF` is false),
`WIDTH`/`HEIGHT`/`LENGTH` into the node, `SHIFT` into `ModelSamplingSD3`. The
generated clip's final frame equals `end_image`, which is exactly what the FFLF
cross-wiring assumes. Decode → **Animation Frame Writer** (wire the sampler
`seed` in for provenance). See [`docs/prompting-guide.md`](docs/prompting-guide.md)
for prompt structure, the standard Wan negative block, and sampler settings.

---

## Game-asset / sprite export

Once you have rendered poses and animation clips, comfyui-andypack provides a
second layer of nodes for turning those renders into engine-ready game assets.

### Alpha → trim → pack → atlas pipeline

Bring your own background-removal step (any BG remover node in your graph). Feed
its output into the writers' optional **MASK** input (or supply a 4-channel RGBA
image directly). The writers record `has_alpha` in the sidecar/meta and preserve
the transparency throughout the pack chain. ComfyUI IMAGE tensors stay 3-channel
inside the graph; RGBA materializes only at the disk boundary.

From there, the one-node path for a finished clip:

- **Animation Sheet Builder** — the Stage-3 packer. Loads every frame of every
  rendered direction of an animation and packs a game sheet (rows = directions,
  cols = frames) with a per-direction *tagged* atlas (fps included). Optional
  `trim` union-trims transparent padding across ALL directions' frames before
  packing (RGBA renders only), shrinking cells while keeping every direction
  registered. Feed its `ATLAS` + `SHEET` straight into **Atlas Metadata Writer**
  (`aseprite` / `godot_spriteframes` get one animation per direction).
- **Sheet Export All** — the batch form of the above: builds and writes a sheet +
  atlas for *every* animation in the character's manifest that has at least one
  rendered direction, in one queue press, instead of one Animation Sheet Builder
  run per animation. Animations with nothing rendered are listed in the report,
  never silently dropped.

Or the manual chain when you want per-frame control:

1. **Animation Frames** — load a rendered clip back as an IMAGE batch (+ fps) to
   re-process without re-sampling.
2. **Frame Retime** — resample a clip to a target fps (`resample` / `trim` /
   `pad_hold`) before packing/export — Wan renders natively at 16 fps, game
   sprites often want less.
3. **Palette Quantize & Lock** — force every direction/animation onto one shared
   limited palette (optionally locked to a `palette_image`) for pixel-art
   consistency; run after background removal, before packing.
4. **Sprite Trim & Pivot** — trims transparent padding from each frame and records
   the pivot offset so all frames in a sheet stay consistently anchored.
5. **Spritesheet Packer** — packs a batch of frames into a single spritesheet image.
6. **Atlas Metadata Writer** — serializes frame coordinates, pivots, and per-frame
   duration to the engine format of your choice.
7. **Animated Sprite Export** — GIF/APNG/WebP preview of a completed clip; now
   preserves alpha (a 4-channel input, or a connected `MASK`, produces a
   transparent export instead of flattening to RGB).

### Diagnostics & conditioning helpers

- **Turnaround Sheet** — composites all rendered directions for a pose side-by-side
  as a contact sheet. Use it to catch drift between directions before packing.
- **Coverage Report** — a status table over every (entity, direction): generated /
  ready / stale / blocked. The at-a-glance "what's left to render."
- **Pose Edit Conditioning** — collapses the whole FLUX.2 pose-edit conditioning
  into one node: text-encodes the pose prompt, attaches the source image as a
  reference latent (and the manikin too when the pose carries one — a base pose),
  and outputs `(positive, negative, latent)`. With `PoseSweepSelector`'s
  `include_base`, a single turnaround graph handles base (2-ref) + derived (1-ref)
  poses — no separate workflows.
- **Pose Sweep Selector / Animation Sweep Selector — `mode` + `category`** —
  `mode=sweep` emits the next actionable pose/animation in dependency order
  (drive it inside a Sweep Loop for one-press batch fill); `mode=target` force-
  regenerates one named pose@direction or animation@direction as a spot-fix.
  `category` scopes the sweep to one manifest category (e.g. `locomotion`,
  `combat`); leave empty for all.

> The node set was culled (2026-06-30) to the pipeline-essential node set for
> clarity; niche/unused nodes (variant/color batchers, boomerang, tween, extra
> reports, the one-frame Character Atlas Builder) were removed in favor of the
> core create → turnaround → animate → sheet path. Manikin control, mirroring,
> and palette lock were later reinstated as focused nodes (Manikin Loader / Pose
> Reference Writer, the writers' `write_mirrored` flag, Palette Quantize & Lock)
> once real authoring workflows needed them.

---

## Development

```bash
pytest -q                 # tests
ruff check .              # lint
mypy andypack            # types
```

The resolver core (`andypack/resolve.py`, `andypack/manifest.py`,
`andypack/io.py`, `andypack/api.py`) is **pure stdlib** — no ComfyUI or torch
imports — so it is fully unit-testable from the fixtures in `tests/`. The
torch/PIL bridge is isolated in `andypack/images.py`, and the ComfyUI node
classes in `andypack/nodes.py` are thin wrappers over the pure core.

### Layout

| Module | Responsibility |
|---|---|
| `andypack/manifest.py` | Load, validate, ref-classify, cycle-detect. |
| `andypack/resolve.py` | Merge prompts, hash, completeness, anchors, transitive staleness. |
| `andypack/io.py` | Atomic writes, meta/sidecar builders, path safety. |
| `andypack/api.py` | JSON payload builders + manifest/character CRUD for the routes. |
| `andypack/server.py` | aiohttp route registration (read + write). |
| `andypack/nodes.py` | ComfyUI node classes. |
| `andypack/images.py` | torch/PIL ↔ ComfyUI IMAGE tensors. |
| `andypack/manikins.py` | Bundled per-direction manikin pose references. |
| `web/anim_coord.js` | Frontend: dynamic character-scoped combos + status glyphs. |
| `web/anim_coord_panel.js` | Frontend: the Andypack sidebar manager panel. |
| `scripts/build_seed_manifest.py` | Generator for the bundled seed manifest. |

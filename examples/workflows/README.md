# Example Workflows

These are **reference-template** JSON files for ComfyUI's standard UI workflow
format. They contain structurally valid JSON and reference the correct andypack
node `type` names, but their link indices were hand-authored rather than saved
by a live ComfyUI instance.

**Canonicalization step (required before use):** Drop the JSON into ComfyUI
via `Load` (or drag-and-drop), verify the wiring looks correct in the graph
editor, then immediately `Save` the workflow. ComfyUI will re-write the link
indices and slot positions to its canonical form. The hand-authored values here
are close but may need minor reconnection after the first load.

---

## sprite_export.json

**Purpose:** Render a single animation direction to a sprite sheet with atlas
metadata, ready for Godot, Unity, or a JSON-hash texture packer.

### Node graph (left to right)

| # | Node type | Role |
|---|-----------|------|
| 1 | `AnimationManifestLoader` | Load `animations.json`; emits `ANIM_MANIFEST` |
| 2 | `CharacterAnimationSelector` | Pick character / animation / direction; emits `ANIM_ANIMATION` |
| 3 | `AnimationUnpack` | Fan the bundle into `START_IMAGE`, `END_IMAGE`, `FPS`, `LENGTH`, prompts, etc. |
| 4 | *(Note node)* | **User sampler gap** — insert your Wan 2.2 i2v sampler here (see below) |
| 5 | `AnimationFrameWriter` | Write rendered frames to the character output directory; emits `OUTPUT_DIR` |
| 6 | `SpriteTrimPivot` | Alpha-trim the frame batch and record pivot metadata; emits `TRIMMED` + `TRIM_DATA` |
| 7 | `SpritesheetPacker` | Pack trimmed frames into a sprite sheet; emits `SHEET` + `ATLAS` |
| 8 | `AtlasMetadataWriter` | Write `<name>.png` + `<name>.json` atlas to the output directory |

### Sampler gap wiring

The workflow deliberately omits the Wan sampler because model choice and
conditioning details are user-specific. The intended wiring is:

```
AnimationUnpack:START_IMAGE  → WanFirstLastFrameToVideo:start_image
AnimationUnpack:END_IMAGE    → WanFirstLastFrameToVideo:end_image   (FFLF path)
AnimationUnpack:POSITIVE_PROMPT → clip encoder → conditioning
AnimationUnpack:NEGATIVE_PROMPT → clip encoder → negative conditioning
AnimationUnpack:LENGTH / FPS / WIDTH / HEIGHT / SHIFT → sampler params
WanSampler:IMAGE (frames)    → AnimationFrameWriter:frames
WanSampler:IMAGE (frames)    → SpriteTrimPivot:image
```

The `AnimationFrameWriter` node also accepts an optional `MASK` input for
per-frame transparency (connect the sampler's alpha mask output if available)
and a `seed` link (forceInput, link-only) for provenance recording.

### Key widget values (node 2 — CharacterAnimationSelector)

- `character`: pick your character from the combo
- `animation`: the animation id from your manifest (e.g. `walk`)
- `direction`: a direction string (e.g. `SOUTH`, `EAST`)

The web extension populates these as cascading combos when the manifest is
loaded.

---

## turnaround.json

**Purpose:** Visualize all rendered directions of a base pose as a contact
sheet, and assemble an identity anchor batch for IPAdapter/Redux conditioning.

### Node graph

| # | Node type | Role |
|---|-----------|------|
| 1 | `AnimationManifestLoader` | Load `animations.json`; emits `ANIM_MANIFEST` |
| 2 | `TurnaroundSheet` | Contact sheet of every CANONICAL_DIRECTION for the chosen pose; grayed-out cells are unrendered |
| 3 | `CharacterIdentityAnchor` | Assemble reference art + base pose image for one direction into an `ANCHOR_BATCH` for IPAdapter |

### TurnaroundSheet widget values

- `character`: the character whose rendered directions to show
- `pose`: the pose id (default `base`)
- `columns`: how many tiles per row in the contact sheet (default 4)
- `include_labels`: overlay direction name labels on each cell
- `cell_size`: fixed cell size in pixels (0 = auto from largest rendered image)

### CharacterIdentityAnchor outputs

- `REFERENCE_IMAGE`: the character's persisted `_reference.png`
- `BASE_DIRECTION_IMAGE`: the already-rendered base pose PNG for the chosen direction
- `ANCHOR_BATCH`: both concatenated along the batch axis — feed directly to an IPAdapter IMAGE input to fight cross-direction identity drift

---

## Node type reference (all types used in these workflows)

All types are keys in `NODE_CLASS_MAPPINGS` in `andypack/nodes.py`:

- `AnimationManifestLoader`
- `CharacterAnimationSelector`
- `AnimationUnpack`
- `AnimationFrameWriter`
- `SpriteTrimPivot`
- `SpritesheetPacker`
- `AtlasMetadataWriter`
- `TurnaroundSheet`
- `CharacterIdentityAnchor`

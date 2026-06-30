# Character Creator node — design

Replaces the Concept Image Writer with a `CharacterCreator` node that pairs a
character reference image with a per-direction manikin pose reference, so the
base pose's 8 directions render as consistent multi-reference FLUX.2 edits. The
`concept` root ref is renamed to `character` throughout.

## Background

Today the tree root is the `concept`: `ConceptImageWriter` writes `_concept.png`
+ `_concept.json` (identity layer + provenance), and the `base` pose
(`from: {ref: concept}`) generates each direction by single-reference,
text-guided editing of the concept image. Only 5 of 8 base directions are
generated (E, SE, S, NE, N); W/SW/NW are synthesized by horizontal mirror
(`mirror_map` + `MirrorFrameWriter`).

We now have 8 hand-drawn manikin pose references (one per direction). Pairing the
character reference with the matching manikin as a **second** FLUX.2 reference
makes the camera angle / body orientation come from the manikin rather than from
verbose prose alone, for more consistent per-direction renders.

### FLUX.2 Klein multi-reference findings (BFL `flux2` docs via context7)

- Multi-reference editing is first-class: pass an ordered list of references
  (`encode_image_refs(ae, [character_ref, manikin])` / CLI
  `input_images="character.png,manikin.png"`). Both occupy distinct temporal
  slots, so **reference order is semantically meaningful** and can be addressed
  in the prompt ("the first image" / "the second image").
- Multi-ref raises the pixel cap (1024² single → 2024² multi); `match_image_size`
  lets the output track the **first** input's dimensions — so the character
  reference goes first, manikin second.
- The official docs are API/CLI-focused and ship no prose prompting rules. The
  established FLUX.2/Kontext editing convention (folded in, flagged as
  convention): explicitly attribute roles to each reference and describe the
  target viewpoint precisely.

## Decisions

1. **Output model:** selector-style — a `direction` dropdown, a single
   `ANIM_POSE` output. Reuses the existing `PoseUnpack` → `PoseFrameWriter`
   pipeline. Not a batch/list node.
2. **All 8 from manikins:** the `base` pose lists all 8 directions; each renders
   directly from its own manikin. Base no longer uses `mirror_map` (mirroring
   stays available for other poses/animations).
3. **Prompts stay manifest-driven:** the resolver compiles the base pose prompt
   from the manifest exactly as today; the node only adds the manikin as a
   second reference. The base `positive_prompt` template is rewritten for
   multi-reference phrasing.
4. **Full rename `concept` → `character`** across node_kind, manifest ref,
   filenames, and seed manifest.

## Components

### Rename: `concept` → `character` (repo-wide, mechanical)

The reserved root ref `concept` becomes `character`:

- **`manifest.py`** — `node_kind` returns `"character"` for `ref == "character"`;
  the pose-`from` validation message and the `deps_graph` root check
  (`ref != "character"`) updated.
- **`resolve.py`** — every `kind == "concept"` branch → `"character"`;
  `_concept_png` / `concept_image_path` / `concept_complete` →
  `_character_png` / `character_image_path` / `character_complete`; on-disk names
  `_concept.png` / `_concept.json` → `character.png` / `character.json`;
  `read_identity` / `invalidate_identity` / `effective_manifest` read
  `character.json`.
- **`io.py`** — `build_concept_sidecar` → `build_character_sidecar` (see
  Provenance stability).
- **`api.py`** — character enumeration checks for `character.png`.
- **`examples/animations.json`** — base pose `from: {ref: "concept"}` →
  `{ref: "character"}` (`default.json` reseeds from this file).
- **`README.md`** — diagram, node table, on-disk layout, workflow steps.

**No on-disk migration.** Existing `_concept.*` dirs are left orphaned; the
project is early and characters are regenerated. (A one-time rename shim is out
of scope unless requested.)

### New node: `CharacterCreator` (category `andypack/Character`)

Hardwired to the `base` pose. Persists identity + reference, then emits the
base-pose job for the selected direction with the manikin attached.

- **`INPUT_TYPES`**
  - required: `manifest` (ANIM_MANIFEST), `image` (IMAGE — the character
    reference), `character` (STRING — the new name the user types, *not* a combo
    of existing characters), `direction` (combo of the 8 canonical directions,
    from a module constant matching the bundled manikins).
  - optional: `identity_positive`, `identity_negative` (multiline STRING).
- **`RETURN_TYPES`** `("ANIM_POSE",)`, `RETURN_NAMES` `("POSE",)`. Not an
  `OUTPUT_NODE` — it always feeds a sampler downstream.
- **`IS_CHANGED`** — fingerprint over (reference image content hash,
  `identity_positive`, `identity_negative`, `character`, `direction`, resolved
  base-pose source/prompt); `nan` when inputs are invalid (mirrors the existing
  selectors' re-resolve discipline).
- **`create`**
  1. Snake-case the character name; resolve the characters root.
  2. Compute the reference image content hash. Build the identity layer from the
     (stripped) positive/negative prompts.
  3. Write `character.png` and `character.json` via
     `build_character_sidecar` (sidecar last, atomic), preserving any
     character-authored `poses`/`animations` keys. Call `invalidate_identity`.
  4. `effective_manifest(...)`; `resolve_pose(..., "base", direction)`. Guard:
     error if `base` is missing or `direction` is not in `base.directions`.
  5. Load the bundled manikin for `direction`.
  6. Return an `ANIM_POSE` bundle: `source_image` = the input reference tensor
     (first), `pose_reference` = manikin tensor (second), `positive`/`negative`
     from the resolve result, `output_dir` (`_base`), `_meta`.

### Provenance stability (behavior change)

The old writer bumped `render_id` on every write (via a fresh `created_utc`).
Because `CharacterCreator` runs **once per direction**, that would mark
already-rendered base directions stale on every run. `build_character_sidecar`
becomes **content-keyed**: it stores an `image_hash` (added to the owned keys
alongside the identity-layer hash) and only advances `created_utc`/`render_id`
when the identity prompts *or* the reference image content hash differ from the
existing sidecar.

- Generating EAST then SOUTH with the same reference + identity ⇒ unchanged
  `render_id` ⇒ EAST not marked stale.
- Swapping the reference image ⇒ new `image_hash` ⇒ new `render_id` ⇒ all 8 base
  directions (and their descendants) correctly go stale.

The node writes `character.png` only when the content hash changed (idempotent
re-runs don't rewrite the payload).

### Manikin assets + all-8 base directions

- Bundle the 8 drawings into `andypack/assets/manikins/<DIR>.png`
  (`north→NORTH`, `north_east→NORTH_EAST`, `east→EAST`,
  `south_east→SOUTH_EAST`, `south→SOUTH`, `south_west→SOUTH_WEST`,
  `west→WEST`, `north_west→NORTH_WEST`). A `manikin_path(direction)` helper
  resolves them; they ship in the repo and load via
  `images.load_image_tensor`.
- The seed manifest's `base.directions` lists **all 8**. Author
  WEST/SOUTH_WEST/NORTH_WEST direction prompts (only E/SE/S/NE/N exist today).
- The base `positive_prompt` template is rewritten for multi-reference:
  attribute identity/design to "the first image" and body orientation + camera
  angle to "the mannequin in the second image," keeping `{identity_prompt}`,
  `{direction_name}`, `{direction_prompt}`.
- `mirror_map` is retained for other poses/animations; base no longer relies on
  it.

### `ANIM_POSE` bundle + `PoseUnpack`

`ANIM_POSE` gains a `pose_reference` (IMAGE) leaf. `PoseUnpack` exposes a new
`POSE_REFERENCE` output (additive, non-breaking — existing graphs gain a slot).
`CharacterPoseSelector` sets `pose_reference` to an empty image (normal poses
have no manikin). `PoseFrameWriter` is unchanged — it writes only the generated
result. The user wires `SOURCE_IMAGE` + `POSE_REFERENCE` into their external
FLUX.2 multi-reference edit node. The `_POSE_UNPACK` table and the test that
enforces unpack covers every leaf key are updated.

### `ConceptImageLoader` → `CharacterImageLoader`

Reads `character.png` / `character.json`; outputs `CHARACTER_IMAGE`,
`HAS_CHARACTER`, `IDENTITY_POSITIVE`, `IDENTITY_NEGATIVE`. Node mappings and
display names updated (`ConceptImageWriter`→`CharacterCreator` "Character
Creator"; `ConceptImageLoader`→`CharacterImageLoader`).

## Data flow

```
CharacterCreator(image, name, identity±, direction)
  ├─ writes  character.png + character.json (content-stable provenance)
  └─ outputs ANIM_POSE { source_image=ref, pose_reference=manikin[dir],
                         positive, negative, output_dir=_base, _meta }
        → PoseUnpack → (SOURCE_IMAGE, POSE_REFERENCE, POSITIVE, NEGATIVE, …)
            → external FLUX.2 multi-reference edit (ref first, manikin second)
                → PoseFrameWriter → _base/<DIR>.png + <DIR>.json
```

## Error handling

- Unknown/empty character name → `to_snake_case` raises (existing behavior).
- `base` pose missing from the manifest, or `direction` not in
  `base.directions` → explicit `RuntimeError`.
- Missing manikin asset for a direction → `RuntimeError` naming the direction.
- Invalid inputs in `IS_CHANGED` → `nan` (forces re-run, defers the clear error
  to `create`), matching the existing selectors.

## Testing (TDD)

- Rename `concept` → `character` across the test suite.
- Stable provenance: same image + identity ⇒ identical `render_id` across runs;
  changed identity or image ⇒ bumped `render_id`.
- `manikin_path` resolves for all 8 directions; missing asset raises.
- All 8 base directions are selectable from the seed manifest.
- `pose_reference` flows through the bundle and `PoseUnpack`; the unpack-covers-
  leaves test still passes.
- `CharacterImageLoader` round-trips identity + image / reports `HAS_CHARACTER`.

## Out of scope

- The FLUX.2 edit/sampler node itself — the pack still does not sample.
- Automatic migration of existing `_concept.*` directories.
- Per-character manikin overrides (bundled assets only).
```

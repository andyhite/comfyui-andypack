# Character Creator node — design

Replaces the Concept Image Writer with a `CharacterCreator` node and **removes
the concept tree node entirely**. The `base` pose becomes the root of the
dependency tree: it renders each of its 8 directions from a character reference
image (a node input, not persisted) paired with a per-direction manikin pose
reference, as a multi-reference FLUX.2 edit.

## Background

Today the tree root is the `concept`: `ConceptImageWriter` writes `_concept.png`
(the seed image) + `_concept.json` (identity layer + provenance), and the `base`
pose (`from: {ref: concept}`) generates each direction by single-reference,
text-guided editing of that concept image. Only 5 of 8 base directions are
generated (E, SE, S, NE, N); W/SW/NW are synthesized by horizontal mirror
(`mirror_map` + `MirrorFrameWriter`).

We now have 8 hand-drawn manikin pose references (one per direction). Pairing the
character reference with the matching manikin as a **second** FLUX.2 reference
makes the camera angle / body orientation come from the manikin rather than from
prose alone, for more consistent per-direction renders.

The concept stops being a persisted node: the reference image is no longer saved
to the character directory, so the concept's provenance role disappears. The
`base` pose's own sidecars carry the provenance instead — `base` is now the root.

### FLUX.2 Klein multi-reference findings (BFL `flux2` docs via context7)

- Multi-reference editing is first-class: pass an ordered list of references
  (`encode_image_refs(ae, [character_ref, manikin])` / CLI
  `input_images="character.png,manikin.png"`). Both occupy distinct temporal
  slots, so **reference order is semantically meaningful** and can be addressed
  in the prompt ("the first image" / "the second image").
- Multi-ref raises the pixel cap (1024² single → 2024² multi); `match_image_size`
  lets the output track the **first** input's dimensions — so the character
  reference goes first, manikin second.
- Official docs are API/CLI-focused and ship no prose prompting rules. The
  established FLUX.2/Kontext editing convention (folded in, flagged as
  convention): explicitly attribute roles to each reference and describe the
  target viewpoint precisely.

## Decisions

1. **No concept node.** The reserved `concept` ref and `node_kind == "concept"`
   are deleted. A pose's `from` becomes optional; a pose with no `from` is a
   **root pose**. `base` is the (only) root.
2. **Reference image is not persisted.** It's a `CharacterCreator` input used to
   generate the base directions; it lives in the user's graph, not on disk.
   There is no `_concept.png` / `character.png`.
3. **`character.json` keeps identity only — no provenance.** It holds the
   identity layer (`positive_prompt` / `negative_prompt`) and the optional
   character-authored `poses`/`animations` overlay. No `prompt_hash`,
   `created_utc`, or `render_id` (the concept is no longer a render node).
4. **Base sidecars carry the provenance.** `_base/<DIR>.png` + `<DIR>.json` get
   the normal pose provenance via `PoseFrameWriter`; `sources` is empty (base
   has no tree deps).
5. **All 8 base directions from manikins.** `base.directions` lists all 8; base
   no longer uses `mirror_map` (mirroring stays for other poses/animations).
6. **Prompts stay manifest-driven.** The resolver compiles the base prompt from
   the manifest; the node only adds the manikin as a second reference. The base
   `positive_prompt` template is rewritten for multi-reference phrasing.
7. **Selector-style node:** a `direction` dropdown, a single `ANIM_POSE` output,
   reusing `PoseUnpack` → `PoseFrameWriter`.
8. **"concept" terminology is fully retired, replaced by "character."** The
   prompt template variable `{identity_prompt}` becomes `{character_prompt}`, the
   node identity inputs become `character_positive` / `character_negative`, and
   the internal `read_identity` / `invalidate_identity` helpers become
   `read_character` / `invalidate_character`.

## Components

### Remove the concept tree node (`manifest.py`, `resolve.py`)

- **`node_kind`** — drop the `concept` branch; classify only `pose` /
  `animation` (unknown ref still raises).
- **`_validate_refs`** — a pose's `from` is now optional. When present it must be
  a dict with `ref` that resolves to a *pose* (not an animation); when absent the
  pose is a root. Update the error message (no "concept").
- **`_dependency_edges`** — drop the `ref != "concept"` special-case; a pose with
  no `from` contributes no edge (already handled by the `None` ref guard), so it
  sorts as a leaf in `topo_order` / cycle detection.
- **`resolve.py` root-pose handling:**
  - `pose_source_image` / `_single_image`: a pose with no `from` has no
    disk source → returns `None` (the node supplies the reference).
  - `resolve_pose`: when `from` is absent → `blocked_by = []`,
    `source_image = None`, `stale = False` (no source dep); `selectable` is just
    `direction in pose["directions"]`.
  - `outdated`: a root pose recurses into no ancestor — it's stale only on its
    own `prompt_hash` drift (so an identity-text change still re-stales base via
    `{identity_prompt}` in its compiled prompt).
  - `direct_deps` / `recorded_sources`: empty for a root pose.
  - Remove `_concept_png`, `concept_image_path`, `concept_complete`, and the
    `kind == "concept"` branches in `node_complete`, `read_node_meta`,
    `read_render_id`, `_anchor_from_dep`.

### `character.json` = character prompt layer only (`io.py`, `resolve.py`)

- **`io.build_concept_sidecar` → `build_character`** — writes only the character
  prompt layer, merged over any existing file so the character-authored
  `poses`/`animations` overlay survives. **No** `prompt_hash` / `created_utc` /
  `render_id`. Owned keys: `positive_prompt`, `negative_prompt` (a cleared widget
  drops its key); all other existing keys pass through.
- **`read_identity` → `read_character`** — reads `character.json` (memoized by
  path+mtime as today); returns the prompt dict (no `render_id`).
- **`invalidate_identity` → `invalidate_character`** — repointed to
  `character.json`; still called by the node after writing (character + effective-
  manifest caches). Docstring updated (no render_id rationale).
- **`effective_manifest`** — unchanged behavior; reads the `poses`/`animations`
  overlay from `character.json`.

### Prompt template variable rename (`resolve.py`, seed manifest)

- **`substitute_variables` / `_TEMPLATE_TOKEN`** — the opt-in token
  `{identity_prompt}` becomes `{character_prompt}` (resolves to the character
  layer's `positive_prompt` / `negative_prompt` by field, exactly as today).
  `{direction_prompt}` / `{direction_name}` are unchanged.
- **Seed manifest (`examples/animations.json`)** — every `{identity_prompt}`
  occurrence (the `globals.pose` / `globals.animation` negatives, the base and
  per-pose positives) is rewritten to `{character_prompt}`.

### `api.py` character-directory marker

`_is_character` checks for `character.json` (was `_concept.png`); the
"any subdirectory" fallback (a rendered pose/animation dir) is retained.

### New node: `CharacterCreator` (category `andypack/Character`)

Hardwired to the `base` pose (a module constant; validated to exist as a root
pose). Persists identity, then emits the base-pose job for the selected
direction with the manikin attached.

- **`INPUT_TYPES`**
  - required: `manifest` (ANIM_MANIFEST), `image` (IMAGE — the character
    reference), `character` (STRING — the new name the user types, *not* a combo
    of existing characters), `direction` (combo of the 8 canonical directions
    from a module constant matching the bundled manikins).
  - optional: `character_positive`, `character_negative` (multiline STRING).
- **`RETURN_TYPES`** `("ANIM_POSE",)`, `RETURN_NAMES` `("POSE",)`. Not an
  `OUTPUT_NODE` — it always feeds a sampler downstream.
- **`IS_CHANGED`** — fingerprint over (`character_positive`, `character_negative`,
  `character`, `direction`, resolved base prompt); `nan` on invalid inputs
  (mirrors the existing selectors' re-resolve discipline).
- **`create`**
  1. Snake-case the character name; resolve the characters root.
  2. Build the character prompt layer from the (stripped) positive/negative
     prompts.
  3. Write `character.json` via `build_character` (atomic), preserving the
     overlay. Call `invalidate_character`. (No image written, no provenance.)
  4. `effective_manifest(...)`; `resolve_pose(..., "base", direction)`. Guard:
     error if `base` is missing or `direction` not in `base.directions`.
  5. Load the bundled manikin for `direction`.
  6. Return an `ANIM_POSE` bundle: `source_image` = the input reference tensor
     (first), `pose_reference` = manikin tensor (second), `positive`/`negative`
     from the resolve result, `output_dir` (`_base`), `_meta`.

Re-running per direction only rewrites `character.json` (idempotent character
prompt text — no provenance to thrash). A character-prompt change drifts the
base prompt hash, correctly re-staling base (and, via `{character_prompt}`,
downstream).

### Manikin assets + all-8 base directions

- Bundle the 8 drawings into `andypack/assets/manikins/<DIR>.png`
  (`north→NORTH`, `north_east→NORTH_EAST`, `east→EAST`,
  `south_east→SOUTH_EAST`, `south→SOUTH`, `south_west→SOUTH_WEST`,
  `west→WEST`, `north_west→NORTH_WEST`). A `manikin_path(direction)` helper
  resolves them; they ship in the repo and load via `images.load_image_tensor`.
  A missing asset raises a `RuntimeError` naming the direction.
- The seed manifest's `base` pose drops its `from` block and lists all 8
  directions. Author WEST/SOUTH_WEST/NORTH_WEST direction prompts (only
  E/SE/S/NE/N exist today).
- The base `positive_prompt` template is rewritten for multi-reference:
  attribute identity/design to "the first image" and body orientation + camera
  angle to "the mannequin in the second image," keeping `{character_prompt}`,
  `{direction_name}`, `{direction_prompt}`.

### `ANIM_POSE` bundle + `PoseUnpack`

`ANIM_POSE` gains a `pose_reference` (IMAGE) leaf. `PoseUnpack` exposes a new
`POSE_REFERENCE` output (additive — existing graphs gain a slot).
`CharacterPoseSelector` sets `pose_reference` to an empty image (normal poses
have no manikin). `PoseFrameWriter` is unchanged (writes only the generated
result). The user wires `SOURCE_IMAGE` + `POSE_REFERENCE` into their external
FLUX.2 multi-reference edit node. The `_POSE_UNPACK` table and the test enforcing
unpack covers every leaf key are updated.

### Exclude root poses from `CharacterPoseSelector`

`base` (and any pose with no `from`) is created by `CharacterCreator`, not the
generic selector — picking it there would yield an empty source. The pose-list
route/web combo for `CharacterPoseSelector` excludes poses with no `from`.

### Remove `ConceptImageLoader`

Its purpose — reload the persisted concept image for re-editing — is gone (no
persisted reference). Identity text lives in `character.json`; the reference
image must be re-supplied from the user's own graph to regenerate base later
(an accepted consequence of not persisting it). Node mappings and display names
updated (`ConceptImageWriter`→`CharacterCreator` "Character Creator";
`ConceptImageLoader` removed).

## Data flow

```
CharacterCreator(manifest, image, name, identity±, direction)
  ├─ writes  character.json  (character prompt layer only, no provenance)
  └─ outputs ANIM_POSE { source_image=ref, pose_reference=manikin[dir],
                         positive, negative, output_dir=_base, _meta }
        → PoseUnpack → (SOURCE_IMAGE, POSE_REFERENCE, POSITIVE, NEGATIVE, …)
            → external FLUX.2 multi-reference edit (ref first, manikin second)
                → PoseFrameWriter → _base/<DIR>.png + <DIR>.json (provenance)
                    → downstream poses (from: base) / animations, unchanged
```

## Error handling

- Empty/unusable character name → `to_snake_case` raises (existing behavior).
- `base` pose missing, or `direction` not in `base.directions` → explicit
  `RuntimeError`.
- Missing manikin asset for a direction → `RuntimeError` naming the direction.
- Invalid inputs in `IS_CHANGED` → `nan` (forces re-run; the clear error is
  raised in `create`), matching the existing selectors.

## Testing (TDD)

- Rename `_concept.json`/`concept` usage to `character.json` across tests; drop
  concept-node assertions; update `{identity_prompt}` → `{character_prompt}`.
- Root pose: a pose with no `from` validates, sorts as a leaf, resolves with
  empty `blocked_by`/`sources`, and is outdated only on its own prompt drift.
- `character.json` character layer round-trips; the `poses`/`animations` overlay
  survives a rewrite; a cleared widget drops its key.
- `{character_prompt}` substitutes by field (positive/negative); a
  character-prompt change re-stales base (prompt-hash drift) and downstream.
- `manikin_path` resolves for all 8 directions; a missing asset raises.
- All 8 base directions are selectable from the seed manifest.
- `pose_reference` flows through the bundle and `PoseUnpack`; the
  unpack-covers-leaves test passes; `CharacterPoseSelector` sets it empty.
- The pose-list route excludes root poses.

## Out of scope / accepted limitations

- The FLUX.2 edit/sampler node itself — the pack still does not sample.
- The reference image is not tracked for staleness: swapping it and re-running
  only one base direction won't auto-stale the others (re-run all 8).
- Automatic migration of existing `_concept.*` directories.
- Per-character manikin overrides (bundled assets only).
```

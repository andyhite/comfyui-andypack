# Character Loader node — design

**Date:** 2026-07-02
**Status:** approved (design), pending implementation

## Problem

The SYBP `1_create.json` workflow drives base-pose generation from a
pre-authored `character.json`. Two character nodes are wired:

- **Node 63 `CharacterPromptLoader`** — reads the authored prompts, emits
  `POSITIVE`/`NEGATIVE` STRINGs that feed the txt2img concept-art stage
  (character prompt + shared style reference → concept art).
- **Node 30 `CharacterCreator`** — takes the generated concept art as `image`
  and emits the base-pose `ANIM_POSE` for the FLUX edit.

`CharacterCreator.create()` rebuilds the character prompt layer purely from its
two prompt widgets: with them blank, `layer = {}`, and
`io.build_character({}, existing)` **drops** the owned keys
(`positive_prompt`/`negative_prompt`). So every run of node 30 wipes the
authored `character.json`. The workflow Note wrongly assumed blank widgets
preserve prompts; the code (per the CLAUDE.md "widgets are source of truth"
invariant) treats blank as an intentional clear.

The Create stage needs a node that emits the base-pose job **without ever
writing the prompt layer**.

## Approach

Add a new read-only node `CharacterLoader` that replaces `CharacterCreator` in
the Create workflow. `CharacterCreator` (authoring/clobber semantics) and
`CharacterPromptLoader` (the STRING feed) are both left unchanged.

Rejected alternatives:
- **One node emitting POSE + strings:** the strings feed txt2img *before* the
  image exists and POSE needs that image *after* — a single node can't do both
  without a graph cycle.
- **Change `CharacterCreator` to preserve on blank:** contradicts the
  documented "widgets are source of truth" invariant and would silently change
  authoring behavior elsewhere.

## Node: `CharacterLoader`

- **Class:** `CharacterLoader`; **display:** "Character Loader"; **category:**
  `andypack/Character`.
- **Inputs (required):**
  - `manifest` (`ANIM_MANIFEST`)
  - `image` (`IMAGE`) — the reference/concept art (first FLUX-edit reference)
  - `character` — combo of *existing* characters (`_character_choices()`, same
    as `CharacterPromptLoader`); this loads, it does not create, so no free-text
  - `direction` — `manikins.CANONICAL_DIRECTIONS` combo
- **Input (optional):** `save_reference` (`BOOLEAN`, default `True`)
- **Output:** `POSE` (`ANIM_POSE`) — identical bundle shape to
  `CharacterCreator`'s output.

### Behavior

1. Validate `direction` ∈ `CANONICAL_DIRECTIONS` (raise otherwise, mirroring
   Creator).
2. `char_name = io.to_snake_case(character)`.
3. Build the effective manifest (character overlay applied); raise if it has no
   `base` pose or no such `direction` — same guards as Creator.
4. Resolve the `base` pose for `character`+`direction`. A missing/empty
   `character.json` is **not** an error — `{character_prompt}` just expands to
   empty (consistent with `CharacterPromptLoader`/resolve read semantics; the
   combo already gates to real character folders).
5. Pair the supplied `image` (source) with the bundled manikin for the
   direction (pose_reference); emit the `ANIM_POSE`.
6. If `save_reference`, persist `image` → `_reference.png`. This is not a prompt
   write and is required so the Stage-2 turnaround sweep
   (`_build_pose_bundle`, which demands a persisted reference for root poses)
   can reload the root reference. **No `character.json` write happens under any
   path.**

### `IS_CHANGED`

Base-pose `prompt_hash` + `direction` + `save_reference` — `CharacterCreator`'s
fingerprint minus the two prompt-widget terms. Returns `float("nan")` when no
character is selected.

## Cleanup

Extract the "resolve base pose + pair supplied image with the direction's
manikin → POSE dict" logic currently inline in `CharacterCreator.create`
(nodes.py ~182–191) into a small shared helper both nodes call, so the two
node bundles can't drift.

## Out of scope

- No change to `CharacterCreator` clobber/authoring semantics.
- No change to `CharacterPromptLoader`.

## Testing

- New node emits a well-formed `ANIM_POSE` (source_image = supplied image,
  pose_reference = the direction's manikin, positive/negative/output_dir/_meta
  present).
- Running the node does **not** modify `character.json` — authored
  `positive_prompt`/`negative_prompt` survive byte-for-byte.
- `save_reference=True` writes `_reference.png`; `save_reference=False` does not.
- Unknown direction / missing base direction raise.

## Docs / registration

- Register `CharacterLoader` in `NODE_CLASS_MAPPINGS` and
  `NODE_DISPLAY_NAME_MAPPINGS`.
- README: document the node.
- CLAUDE.md: bump the Character group to 4 nodes and note the read-only
  loader.

## Workflow fix (separate repo)

`../shit-your-brain-pants/assets/workflows/1_create.json`: swap node 30 from
`CharacterCreator` to `CharacterLoader`, drop its `character_positive` /
`character_negative` widgets, keep `character` + `direction` + `save_reference`.
Update the Note text to stop claiming blank prompts "preserve" via the Creator.

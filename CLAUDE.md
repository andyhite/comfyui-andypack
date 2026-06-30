# comfyui-andypack — Animation Coordinator

ComfyUI node pack: a dependency-aware FFLF resolver. Drives
character → animation → direction from a single `animations.json`,
gates selection on what's rendered, feeds the sampler positive/negative/
start-image/end-image. It does NOT sample — it resolves and writes back.

## Source of truth
Build strictly to
`docs/superpowers/specs/2026-06-29-cascading-pose-resolver-design.md`.
The older `docs/anim-coord-node-spec.md` is superseded; where they
disagree, the cascading design wins. Example manifest +
schema-by-example: `examples/animations.json`.

## Non-negotiables (these are where it goes wrong)
- Every animation needs a START image (the I2V initial latent): its explicit
  `start_from`, else manifest `defaults.start_from`. The loader rejects an
  animation with neither. `end_at` is optional — when present, it's FFLF.
- FFLF cross-wiring: `start_from` consumes the dep's LAST frame;
  `end_at` consumes the dep's FIRST frame. Do not invert.
  Single-image deps (concept/pose) resolve the same image for either slot.
- Refs are typed: `concept` (seed), a pose id, or an animation id. Prompts
  compile as: merge `globals[kind]` + entity (`merge_layers`/`merge_negative`),
  THEN substitute template variables. No facial/global negative special-casing.
- The identity layer (`_concept.json`) and the per-direction layer are NOT
  merged — they surface ONLY via template variables, resolved by field context
  (positive vs negative): `{identity_prompt}` → concept positive/negative;
  `{direction_prompt}` → that direction's positive/negative; `{direction_name}`
  → the bare direction (e.g. `EAST`), both contexts. Literal `str.replace`
  (unknown `{...}` and stray braces survive; absent → ``). Vars resolve inside
  globals too. Negatives still split/dedupe/drop-empty after substitution.
- Dynamic character-scoped combos need a server route + web/ JS
  extension. Pure-Python INPUT_TYPES cannot do it.
- `/frame` route must reject `..` / absolute / symlink escapes; 404
  anything outside {root}. It serves files — treat it as untrusted.
- Writer order is atomic: payload first, then the `meta.json`/sidecar
  written LAST via temp-file + atomic rename. There is no `.complete`
  file; a dir with no parseable meta/sidecar is incomplete.
- Keep `resolve.py` pure (no ComfyUI/torch imports) so it's unit-testable.

## Build order (gate each on its acceptance test)
1. manifest.py (load + validate + cycle detect) — done
2. resolve.py + tests against tests/fixtures/  ← TDD, before nodes — done
3. pose nodes (CharacterPoseSelector + PoseFrameWriter + concept intake)
4. animation nodes (CharacterAnimationSelector + AnimationFrameWriter)
5. server.py routes
6. web/anim_coord.js

## Commands
- Test: `pytest -q`
- Lint/types: `ruff check . && mypy andypack`

Definition of done = the cascading design's §8 end-to-end acceptance passes.

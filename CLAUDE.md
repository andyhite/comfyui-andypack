# comfyui-andypack — Animation Coordinator

ComfyUI node pack: a dependency-aware FFLF resolver. Drives
character → animation → direction from a single `animations.json`,
gates selection on what's rendered, feeds the sampler positive/negative/
start-image/end-image. It does NOT sample — it resolves and writes back.

## Source of truth
Build strictly to `docs/anim-coord-node-spec.md`. Example manifest +
schema-by-example: `examples/animations.json`.

## Non-negotiables (these are where it goes wrong)
- FFLF cross-wiring: `start_from` consumes the dep's LAST frame;
  `end_at` consumes the dep's FIRST frame. Do not invert.
- Dynamic character-scoped combos need a server route + web/ JS
  extension. Pure-Python INPUT_TYPES cannot do it.
- `/frame` route must reject `..` / absolute / symlink escapes; 404
  anything outside {root}. It serves files — treat it as untrusted.
- Writer order is atomic: frames → meta.json → touch .complete. A
  partial dir must never read as a satisfied dependency.
- Keep `resolve.py` pure (no ComfyUI/torch imports) so it's unit-testable.

## Build order (gate each on its acceptance test in spec §9)
1. manifest.py (load + validate + cycle detect)
2. resolve.py + tests against tests/fixtures/  ← TDD, do this before nodes
3. nodes.py (Selector + FrameWriter)
4. server.py routes
5. web/anim_coord.js

## Commands
- Test: `pytest -q`
- Lint/types: `ruff check . && mypy andypack`

Definition of done = spec §9 end-to-end acceptance passes.

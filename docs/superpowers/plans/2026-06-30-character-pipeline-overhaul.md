# Character pipeline overhaul — plan

Goal: make comfyui-andypack a comprehensive, correct node pack for generating
video-game character assets with FLUX.2 Klein (poses) + Wan 2.2 14B i2v
(animations). Research synthesized in `docs/prompting-guide.md`.

## Verdicts from research
- **FFLF architecture is SOUND** for Wan 2.2 (`WanFirstLastFrameToVideo`,
  `start_from`→`start_image`, `end_at`→`end_image`). No core resolver re-wiring.
- **FLUX.2 Klein has NO negative path** — pose negatives are dead weight (or
  backfire). Fold their intent into affirmative positives.
- **Manifest only authors EAST** for anchor poses + animations → the "all 8
  directions" workflow is not backed by content. This is the #1 gap.

## Phase 1 — Prompt & manifest overhaul (pure core, TDD)
1. `resolve.py`: add `{view_phrase}` template var resolved from manifest-level
   `view_phrases[direction]` (positive context). Keeps per-direction camera
   language DRY so an entity can opt into all 8 directions with empty layers.
   `{direction_prompt}` stays for entity-specific overrides.
2. `manifest.py`: validate `view_phrases` (optional map dir→str). Lint: warn if an
   entity lists a direction with no entity layer AND no `view_phrases` entry.
3. Seed manifest rewrite (`examples/animations.json`):
   - Add `view_phrases` map (8 affirmative camera phrases; back/profile mitigations).
   - `base`: tighten the multi-ref edit prompt (<100 words, affirmative, scope
     manikin to pose-only). Keep all 8 directions but move camera prose to
     `view_phrases`; per-direction layers become thin/empty.
   - Every anchor pose + animation: list **all 8 directions** (empty `{}` layers),
     relying on entity prose + `{view_phrase}` + `{direction_name}`.
   - Pose negatives: empty/minimal (Klein ignores them); fold "one eye" etc. into
     `view_phrases` positives.
   - Animation `globals.negative`: the standard Wan block. Add fixed-camera /
     static-background language to motion prompts.
   - `defaults`: keep fps 16, length 33 (4n+1). Document shift presets in docs.
4. Update all affected tests + fixtures.

## Phase 2 — Node correctness + ergonomics
- Fix real bugs surfaced by review.
- Confirm Mirror writer is usable; document the symmetric-only caveat; consider a
  per-character `mirror` opt-in.
- Keep the pack "does not sample" boundary.

## Phase 3 — Sidebar GUI (new feature)
- New HTTP routes (write-capable, JSON-only, no client paths): read/write manifest
  JSON, read/write character.json, coverage grid, options.
- New `web/` sidebar panel via `app.extensionManager.registerSidebarTab`
  (verify API vs installed frontend). Features: manifest CRUD, character CRUD,
  coverage grid. Stretch: click a coverage cell → set selector widgets + queue.
- Path-safety: validate manifest names; never accept filesystem paths from client.

## Phase 4 — Example workflows + docs
- Ship example ComfyUI workflow JSONs for the FLUX base/pose edit and the Wan FFLF
  animation graph (so the pack is turnkey).
- Fix README/CLAUDE.md drift (routes, glyphs, node table, types).

## Discipline
- TDD where practical; `pytest -q && ruff check . && mypy andypack` green before
  each commit. Logical commits per phase. No user prompts (autonomous).

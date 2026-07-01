# comfyui-andypack — Animation Coordinator

ComfyUI custom-node pack: a dependency-aware FFLF resolver. Drives
character → animation → direction from a single `animations.json`, gates
selection on what's already rendered, and feeds a sampler positive/negative/
start-image/end-image. It does NOT sample — it resolves and writes back.

The source of truth is the code plus `examples/animations.json` (schema-by-example)
and `README.md` (user docs). `docs/prompting-guide.md` holds the researched FLUX.2
Klein / Wan 2.2 i2v prompt structure + ComfyUI settings the seed manifest follows.

## Commands
- Test: `pytest -q`
- Lint: `ruff check .`
- Types: `mypy andypack`
- CI runs all three on Python 3.10/3.11/3.12. Torch is installed CPU-only and
  separately (`pip install torch --index-url https://download.pytorch.org/whl/cpu`),
  then `pip install -r requirements-dev.txt`.

## Module map
- `__init__.py` (repo root) — ComfyUI entry point. Inserts the repo root on
  `sys.path` (so `from andypack...` absolute imports resolve), re-exports
  `NODE_CLASS_MAPPINGS` / `NODE_DISPLAY_NAME_MAPPINGS` / `WEB_DIRECTORY="./web"`.
- `andypack/__init__.py` — imports `server` (registers HTTP routes on import) and
  seeds the bundled manifest into the user dir. Seeding must never block loading.
- `manifest.py` — load / structural-validate / cycle-detect / `topo_order`;
  `node_kind` classifies a ref as pose | animation; validates `view_phrases` and
  gen-params (`length`/`fps`/`width`/`height` ints, `shift` numeric).
- `resolve.py` — the pure FFLF core: cascade prompts, template-var substitution
  (incl. `{view_phrase}`), FFLF anchors, completeness, staleness, status,
  playback plan, `reference_image_path`. **No ComfyUI/torch imports** (keep it so).
- `io.py` — atomic JSON writes, meta/sidecar builders (pose, animation,
  character), `render_id` provenance.
- `images.py` — tensor ↔ PNG conversion.
- `manikins.py` — bundled per-direction manikin pose references (`manikin_path`).
- `api.py` — pure JSON payload builders + manifest/character CRUD helpers
  (`save_manifest_text`, `read/save_character_layer`, `create_character`,
  `manifest_name_is_safe`); resolves paths under ComfyUI's `user`/`output` dirs
  (all return None / degrade outside ComfyUI).
- `server.py` — aiohttp `/anim_coord/*` routes (read + write), registered on import.
- `sprites.py` — pure tensor/PIL sprite ops: trim & pivot, spritesheet packing,
  palette quantize & lock. No ComfyUI/torch side effects beyond tensor I/O.
- `atlas.py` — pure-stdlib engine-format serializers (JSON/XML atlas metadata).
  No ComfyUI/torch imports (keep it so).
- `nodes.py` — the ComfyUI node classes + mappings (**20 focused nodes**), grouped
  into `andypack/<Manifest|Character|Pose|Animation|Diagnostics|Sprite|Export>`.
  The pack was culled to the pipeline-essential set (2026-06-30); pure helpers for
  removed nodes may still linger in `sprites.py`/`api.py` (not exposed).
  - Manifest: Animation Manifest Loader.
  - Character: Character Creator, Character Reference Loader.
  - Pose: **Pose Sweep Selector** (unifies the old Character/Auto Pose Selectors;
    `mode: sweep|target` — sweep emits the next actionable pose in dependency
    order, `include_base` drives the whole turnaround with base manikin-paired;
    target force-regenerates one named pose@direction as a spot-fix), Pose Frame
    Writer (returns `(OUTPUT_DIR, REMAINING)` — `REMAINING` is the sweep loop's
    continue/stop signal), Unpack Pose (has `HAS_POSE_REFERENCE`), **Pose Edit
    Conditioning** (one node = FLUX edit conditioning: text encode + source
    reference latent + manikin latent when present + zeroed negative + empty
    latent).
  - Animation: **Animation Sweep Selector** (unifies the old Character/Auto
    Animation Selectors; `mode: sweep|target`, `category` scope), Animation Frame
    Writer (returns `(OUTPUT_DIR, REMAINING)`, same sweep-loop signal), Unpack
    Animation, **Animation Frames** (load a rendered clip back as an IMAGE batch).
  - Diagnostics: Coverage Report, Turnaround Sheet.
  - Sprite: Sprite Trim & Pivot, Spritesheet Packer, **Animation Sheet Builder**
    (full clip → rows=directions × cols=frames + per-direction tagged atlas: the
    Stage-3 packer).
  - Export: Atlas Metadata Writer, Animated Sprite Export.
  - **Loop**: Sweep Loop Open, Sweep Loop Close — the one-press sweep bracket.
    Open emits a `SWEEP_FLOW` token wired into the sweep selector's optional
    `flow` input and into Close's `flow`; the body's writer's `REMAINING` output
    wires into Close's `remaining`. While `remaining > 0`, Close clones and
    re-expands the Open→Close body so the engine runs another iteration;
    terminates cleanly (no `expand` key) at `remaining <= 0`. Keyed on the
    writer's post-write `REMAINING`, not a fixed iteration count.
- `web/anim_coord.js` — frontend extension for dynamic character-scoped combos
  (pure-Python `INPUT_TYPES` can't populate these; it needs the server routes).
  The character combo is repopulated from `/anim_coord/characters` on node add /
  panel refresh; characters created mid-session appear after a refresh (no full
  reload).
- `web/anim_coord_panel.js` — the Andypack sidebar tab (manifest editor, character
  editor, live coverage grid), backed by the write-capable routes.
- `scripts/build_seed_manifest.py` — generator for `examples/animations.json`.

## Invariants (these are where it goes wrong)
- **Every animation needs a START image** (the I2V initial latent): explicit
  `start_from`, else manifest `defaults.start_from`. The loader rejects an
  animation with neither. `end_at` is optional — when present, it's FFLF.
- **FFLF cross-wiring** (`resolve.py` `start_anchor`/`end_anchor`): `start_from`
  consumes the dep's LAST frame; `end_at` consumes the dep's FIRST frame. Do not
  invert. Single-image deps (a pose) resolve the same image for either slot. This
  maps onto the core `WanFirstLastFrameToVideo` node (`start_from`→`start_image`,
  `end_at`→`end_image`); the clip's final frame equals `end_image`.
- **Prompt compile order**: merge `globals[kind]` + entity layers (`merge_layers`
  for positive = blank-line join; `merge_negative` for negative = comma split +
  case-insensitive dedupe), with template-variable substitution applied per-layer
  *before* the merge.
- **Character (`character.json`) and the per-direction layer are NOT cascade layers.**
  They surface only via opt-in template vars, resolved by field context
  (positive vs negative): `{character_prompt}`, `{direction_prompt}`,
  `{direction_name}`, and `{view_phrase}` (manifest-level `view_phrases[direction]`,
  positive context only). Substitution is a single literal regex pass — unknown
  `{...}` and stray braces survive; absent sources expand to ``; a token inside an
  injected value is NOT re-expanded.
- **FLUX.2 Klein has no negative path**: the seed manifest's poses carry no
  negative layer; pose failure-mode mitigations are affirmative (in `view_phrases`
  / the positive). The negative pipeline is for the Wan animation path (CFG > 1).
  The merge machinery still supports pose negatives — just don't author them.
- **Generation params** (`width`/`height`/`length`/`fps`/`shift`) ride in the
  animation meta (from `defaults` + per-animation override) and surface as wireable
  selector outputs so they drive `WanFirstLastFrameToVideo` / `ModelSamplingSD3`.
  `width`/`height`/`length`/`fps` are now **required** to resolve to a positive int
  per animation — the loader raises a fatal error if any are missing or zero.
- **Atomic write ordering**: write the payload (image/frames) first, then the
  `meta.json` / `.json` sidecar LAST via temp-file + atomic rename. There is no
  `.complete` marker — a dir with no parseable meta/sidecar is treated as
  incomplete. `clear_frames` before re-rendering so a shorter clip can't leave
  stale higher-index frames behind.
- **Staleness** (`outdated`): a complete node is stale if its merged-prompt hash
  drifted, OR a recorded source's `render_id` changed (re-rendered even with an
  unchanged prompt), OR any ancestor is outdated, OR the recorded dependency key-set
  changed (a swapped `start_from`/`end_at` ref, or an `end_at` added/removed —
  anchor-identity drift). The `base` pose roots the tree — its per-direction sidecars
  carry the provenance descendants stale against. `character.json` carries NO
  provenance; editing the character prompt re-stales via prompt-hash drift (it appears
  in compiled prompts through `{character_prompt}`).
- **A loop is derived, never authored**: `resolve_animation` sets `meta["loop"]`
  iff the start and end anchors resolve to the same image (`start_image ==
  end_image`); the writer then drops the duplicated final frame. There is no
  manifest `loop` field — don't add one back.
- **Alpha boundary**: ComfyUI IMAGE tensors stay **3-channel** (RGB) throughout the
  graph. RGBA materializes ONLY at the writer/pack disk boundary — either because a
  4-channel image was explicitly supplied, or because an optional `MASK` input was
  connected. `has_alpha` is recorded in the sidecar/meta when alpha is present. All
  alpha pixel logic lives in `images.py`; `resolve.py` and `manifest.py` are alpha-unaware;
  `io.py` stays torch-free and records the `has_alpha` flag but does no pixel-level alpha work.
- **HTTP routes take no client filesystem paths**: the `/anim_coord/*` routes
  return JSON only and never serve file bytes. Reads enumerate the pack's own
  server-resolved dirs; writes (`manifest/save`, `character/create|save`) address a
  manifest by a validated bare basename (`manifest_name_is_safe`) under the
  manifests dir and a character by a name snake-cased to one segment under the
  characters dir — nothing the client sends escapes those trees. A manifest is
  parsed + `validate_manifest`'d BEFORE it touches disk, so a bad edit is rejected,
  not written over a working file. GET routes that accept a `manifest` or `character`
  query param now gate those values against path traversal (the param must be a bare
  segment with no separators). The new `GET /anim_coord/thumb` route returns a JSON
  base64 data-URI; the path it reads is server-resolved and segment-validated — the
  client sends no filesystem path.
- Routes register on import only inside ComfyUI (`PromptServer` import is guarded);
  `api`/`io` helpers return None when `folder_paths` is unavailable.

## On-disk layout
- Manifests: `<user>/default/andypack/animations/*.json`. `default.json` is seeded
  from `examples/animations.json` on first load (idempotent, never clobbers).
- Characters: `<output>/characters/<char>/` containing `character.json` (prompt
  layer + optional `poses`/`animations` overlay; no provenance), optional
  `_reference.png` (persisted reference art; `save_reference` on the creator),
  `_<pose>/<DIR>.png` + `<DIR>.json` sidecar, and `<anim>/<DIR>/frame_NNNNN.png`
  + `meta.json`.

## mypy config quirks (in pyproject.toml — don't "fix" these)
- No `python_version` pin (targets the running interpreter so it parses newer stubs).
- `explicit_package_bases` + `mypy_path="."` so `mypy andypack` resolves `andypack`
  as top-level despite the repo-root `__init__.py`.

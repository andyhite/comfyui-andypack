# comfyui-andypack — Animation Coordinator

ComfyUI custom-node pack: a dependency-aware FFLF resolver. Drives
character → animation → direction from a single `animations.json`, gates
selection on what's already rendered, and feeds a sampler positive/negative/
start-image/end-image. It does NOT sample — it resolves and writes back.

No design spec is checked in; the source of truth is the code plus
`examples/animations.json` (schema-by-example) and `README.md` (user docs).

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
  `node_kind` classifies a ref as concept | pose | animation.
- `resolve.py` — the pure FFLF core: cascade prompts, FFLF anchors, completeness,
  staleness, status, playback plan. **No ComfyUI/torch imports** (keep it that way).
- `io.py` — atomic JSON writes, meta/sidecar builders (pose, animation, concept),
  `render_id` provenance.
- `images.py` — tensor ↔ PNG conversion.
- `api.py` — pure JSON payload builders for the routes; resolves paths under
  ComfyUI's `user`/`output` dirs (all return None outside ComfyUI).
- `server.py` — aiohttp `/anim_coord/*` routes, registered on import.
- `nodes.py` — the ComfyUI node classes + mappings (15 nodes), grouped into
  `andypack/<Manifest|Concept|Pose|Animation|Diagnostics>` categories.
- `web/anim_coord.js` — frontend extension for dynamic character-scoped combos
  (pure-Python `INPUT_TYPES` can't populate these; it needs the server routes).

## Invariants (these are where it goes wrong)
- **Every animation needs a START image** (the I2V initial latent): explicit
  `start_from`, else manifest `defaults.start_from`. The loader rejects an
  animation with neither. `end_at` is optional — when present, it's FFLF.
- **FFLF cross-wiring** (`resolve.py` `start_anchor`/`end_anchor`): `start_from`
  consumes the dep's LAST frame; `end_at` consumes the dep's FIRST frame. Do not
  invert. Single-image deps (concept/pose) resolve the same image for either slot.
- **Prompt compile order**: merge `globals[kind]` + entity layers (`merge_layers`
  for positive = blank-line join; `merge_negative` for negative = comma split +
  case-insensitive dedupe), with template-variable substitution applied per-layer
  *before* the merge.
- **Identity (`_concept.json`) and the per-direction layer are NOT cascade layers.**
  They surface only via opt-in template vars, resolved by field context
  (positive vs negative): `{identity_prompt}`, `{direction_prompt}`,
  `{direction_name}`. Substitution is literal `str.replace` — unknown `{...}` and
  stray braces survive; absent sources expand to ``.
- **Atomic write ordering**: write the payload (image/frames) first, then the
  `meta.json` / `.json` sidecar LAST via temp-file + atomic rename. There is no
  `.complete` marker — a dir with no parseable meta/sidecar is treated as
  incomplete. `clear_frames` before re-rendering so a shorter clip can't leave
  stale higher-index frames behind.
- **Staleness** (`outdated`): a complete node is stale if its merged-prompt hash
  drifted, OR a recorded source's `render_id` changed (re-rendered even with an
  unchanged prompt), OR any ancestor is outdated. The concept carries its
  `render_id` in `_concept.json` (it has no per-direction meta), so re-rendering
  the concept — the tree root — marks descendants stale too.
- **A loop is derived, never authored**: `resolve_animation` sets `meta["loop"]`
  iff the start and end anchors resolve to the same image (`start_image ==
  end_image`); the writer then drops the duplicated final frame. There is no
  manifest `loop` field — don't add one back.
- **HTTP routes take no client filesystem paths**: the `/anim_coord/*` routes
  return JSON only and never serve file bytes. `/characters` enumerates the pack's
  own `<output>/characters` dir (server-resolved via `api.characters_dir()`), not a
  client-supplied root, so there is nothing to traverse out of.
- Routes register on import only inside ComfyUI (`PromptServer` import is guarded);
  `api`/`io` helpers return None when `folder_paths` is unavailable.

## On-disk layout
- Manifests: `<user>/default/andypack/animations/*.json`. `default.json` is seeded
  from `examples/animations.json` on first load (idempotent, never clobbers).
- Characters: `<output>/characters/<char>/` containing `_concept.png` (+
  `_concept.json`: optional identity layer + provenance, always written),
  `_<pose>/<DIR>.png` + `<DIR>.json` sidecar, and `<anim>/<DIR>/frame_NNNNN.png`
  + `meta.json`.

## mypy config quirks (in pyproject.toml — don't "fix" these)
- No `python_version` pin (targets the running interpreter so it parses newer stubs).
- `explicit_package_bases` + `mypy_path="."` so `mypy andypack` resolves `andypack`
  as top-level despite the repo-root `__init__.py`.

> **Superseded (2026-06-29):** the authoritative model is now
> `docs/superpowers/specs/2026-06-29-cascading-pose-resolver-design.md`
> (concept seed → poses → animations, cascading prompts, transitive staleness,
> meta-as-completion-sentinel). This document is retained for history; where the
> two disagree, the cascading-pose-resolver design wins.

# ComfyUI Node Pack — Animation Coordinator (FFLF / dependency-aware)

Build spec for `comfyui-anim-coord`. Target: a small node pack that drives a
character → animation → direction workflow from a single `animations.json`
manifest, resolves First-Frame/Last-Frame (FFLF) dependencies, gates selection
on what's already rendered, and feeds the sampler a positive prompt, composed
negative, start image, and end image.

The pack is a **resolver over a dependency graph**. It does not sample or
generate — it decides *what's allowed*, *what prompt*, and *which two anchor
frames*, then writes results back so the next dependent animation unlocks.

---

## 1. Concepts

- **Manifest** (`animations.json`): single source of truth. Flat `animations`
  map keyed by id; `category` is a field, not nesting. Absorbs prompt bodies.
- **Dependency**: an animation may declare `start_from` and/or `end_at`, each a
  `{ ref, direction }` pointing at another animation id or the literal
  `base_pose`. `direction: "same"` (or omitted) means "use the selected
  direction"; an explicit direction overrides (e.g. an `SE` move borrowing an
  `E` stance).
- **FFLF cross-wiring** (the one easy-to-invert rule):
  - `start_from` consumes the dependency's **`last_frame`** — you *start where
    it ended*.
  - `end_at` consumes the dependency's **`start_frame`** — you *end where it
    begins*.
  - For a `loop` dependency, `start_frame == last_frame` visually (self-FFLF'd),
    so both resolve to the same guard pose. That's intended.
- **base_pose**: dependency root. The Flux character illustration that has no
  animation to chain from. `start_from: { ref: "base_pose" }` resolves to
  `{root}/{character}/_base/{direction}.png`. Coverage is `E, SE, S`.
- **Gating**: an `(animation, direction)` pair is *selectable* only when every
  dependency it needs exists and is complete for the resolved direction.
- **Staleness**: a dependency is *stale* if its rendered `prompt_hash` differs
  from the current manifest prompt hash. Stale → **warn (amber), still
  selectable** — not blocked.

---

## 2. Filesystem layout & conventions

```
{root}/
  {character}/
    _base/
      E.png  SE.png  S.png                      # Flux init poses (dependency roots)
    {animation}/
      {direction}/
        frame_00000.png ... frame_000NN.png
        meta.json
        .complete                               # sentinel, written LAST
```

`character` is any directory under `{root}` that contains either a `_base/`
folder or at least one animation folder. The selector enumerates characters by
scanning `{root}` one level deep.

### meta.json (written by AnimationFrameWriter)

No sidecar frame copies. `start_frame` / `last_frame` are **relative pointers**
into the frame set.

```json
{
  "animation": "punch",
  "direction": "E",
  "fps": 16,
  "length": 21,
  "loop": false,
  "seed": 848213,
  "prompt_hash": "sha1:a1f3c9…",
  "manifest_version": 1,
  "frames": { "dir": ".", "pattern": "frame_{:05d}.png", "count": 21 },
  "start_frame": "frame_00000.png",
  "last_frame":  "frame_00020.png",
  "created_utc": "2026-06-29T18:22:04Z"
}
```

### Completion & write order (atomicity)

A directory counts as **complete** iff `.complete` exists AND `meta.json` parses
AND `frames.count` files are present. Write order is mandatory:

1. write all `frame_*.png`
2. write `meta.json`
3. `touch .complete`

This makes existence checks safe against a half-rendered or interrupted run — a
partial directory never reads as a satisfied dependency.

### prompt_hash

`sha1(normalize(prompt) + "\u241F" + normalize(negative_composed))`, where
`normalize` strips leading/trailing whitespace and collapses internal runs to a
single space. Stored at render time; compared at resolve time. The negative is
included so changing a directional negative rule also invalidates downstream.

---

## 3. Manifest schema (reference)

Top level:

| key | type | notes |
|-----|------|-------|
| `version` | int | manifest schema version, copied into meta |
| `directions` | string[] | canonical 8-way ordering |
| `mirror_map` | obj | engine-mirrored dirs → their source (W→E, SW→SE, NW→NE); these are **never generated**, documented for tooling/coverage |
| `defaults` | `{fps,length}` | fallback when an animation omits them |
| `base_pose` | `{path, directions}` | `path` is a template under `{character}/`; `directions` is the base coverage |
| `negatives` | obj | composition source (see §4) |
| `animations` | obj | id → animation |

Per animation:

| key | type | required | notes |
|-----|------|----------|-------|
| `category` | string | yes | grouping/label only |
| `directions` | string[] | yes | selectable directions for this animation |
| `prompt` | string | yes | positive body; no character identity (init image anchors it) |
| `negative` | string | no | per-animation negative fragment, appended last |
| `loop` | bool | no (default false) | true → writer self-FFLF's first==last for seamless tiling |
| `length` | int | no (default `defaults.length`) | sampler frame count; keep `4n+1` for Wan |
| `fps` | int | no (default `defaults.fps`) | |
| `start_from` | `{ref, direction?}` | no | dependency for the start anchor |
| `end_at` | `{ref, direction?}` | no | dependency for the end anchor |

`ref` is another animation id or `"base_pose"`. `direction` is `"same"`
(default) or a literal direction.

---

## 4. Negative composition

Compose in this order, join with `", "`, dedupe case-insensitively, preserve
first occurrence:

```
global
  + (direction ∈ negatives.facial.frontal_directions
        ? negatives.facial.frontal
        : negatives.facial.default)
  + animation.negative           # if present
```

The facial split exists because the model hallucinates a face onto a mouthless
head during head motion. Front/three-quarter directions (`S, SE, SW`) drop the
broad `face, facial features` terms (negating the whole face concept fights the
front-of-head render) and keep only the specific feature terms; all other
directions (`E, W, N, NE, NW`) keep the full block.

This composed string is what's hashed and what the writer must persist.

---

## 5. Resolution algorithm

```python
def resolved_dir(dep, selected_dir):
    d = dep.get("direction", "same")
    return selected_dir if d in (None, "same") else d

def dep_complete(root, character, dep, selected_dir):
    tdir = resolved_dir(dep, selected_dir)
    if dep["ref"] == "base_pose":
        return exists(f"{root}/{character}/_base/{tdir}.png")
    return animation_complete(root, character, dep["ref"], tdir)   # .complete + meta + frames

def animation_complete(root, character, anim, direction):
    base = f"{root}/{character}/{anim}/{direction}"
    if not exists(f"{base}/.complete"): return False
    meta = read_json(f"{base}/meta.json")           # tolerate missing/corrupt -> False
    return meta and count_frames(base) >= meta["frames"]["count"]

def resolve(root, character, anim_id, direction):
    a = manifest["animations"][anim_id]
    blocked_by, stale = [], []

    for slot in ("start_from", "end_at"):
        dep = a.get(slot)
        if not dep: continue
        if not dep_complete(root, character, dep, direction):
            blocked_by.append({slot: dep, "dir": resolved_dir(dep, direction)})
            continue
        if dep["ref"] != "base_pose" and is_stale(root, character, dep, direction):
            stale.append(slot)

    selectable = (direction in a["directions"]) and not blocked_by

    return {
        "selectable": selectable,
        "blocked_by": blocked_by,
        "stale": stale,
        "start_image": pick_start_anchor(root, character, a, direction),  # dep.last_frame  | base png | None
        "end_image":   pick_end_anchor(root, character, a, direction),    # dep.start_frame | None
        "positive": a["prompt"],
        "negative": compose_negative(a, direction),
        "output_dir": f"{root}/{character}/{anim_id}/{direction}",
        "meta": { "fps": a.get("fps", D.fps), "length": a.get("length", D.length),
                  "loop": a.get("loop", False), "prompt_hash": hash_of(a, direction) },
    }
```

`pick_start_anchor`: if `start_from.ref == base_pose` → the base png path; else
the dependency's `meta.last_frame` (absolute). `pick_end_anchor`: the
dependency's `meta.start_frame`. Either may be `None`.

---

## 6. Nodes

### 6.1 `AnimationManifestLoader`
- **inputs**: `manifest_path` (STRING).
- **outputs**: `manifest` (ANIM_MANIFEST — custom dict type).
- **does**: load + JSON-schema-validate; build adjacency; **detect cycles**
  (fail loudly with the offending edge); validate every `ref` resolves; warn on
  non-`4n+1` lengths. Cache by `(path, mtime)`.

### 6.2 `CharacterAnimationSelector`
- **inputs**: `manifest` (ANIM_MANIFEST), `root_dir` (STRING),
  `character`/`animation`/`direction` (combos — populated dynamically, §7).
- **outputs**:
  `start_image` (IMAGE, may be empty), `end_image` (IMAGE, may be empty),
  `positive` (STRING), `negative` (STRING), `output_dir` (STRING),
  `meta` (ANIM_META: fps/length/loop/seed/prompt_hash).
- **does**: call `resolve`; load anchor PNGs to IMAGE tensors; raise if a
  selected pair is not `selectable` (defense in depth — UI shouldn't allow it).
- **note**: emit empty IMAGE (1×1 or zero-tensor) plus a companion BOOLEAN
  (`has_start_image` / `has_end_image`) so downstream graph can branch on
  presence without inspecting tensor shape.

### 6.3 `AnimationFrameWriter`
- **inputs**: `frames` (IMAGE batch), `output_dir` (STRING), `meta` (ANIM_META),
  `loop` (BOOLEAN, from meta).
- **does**: write `frame_{:05d}.png`; if `loop`, ensure first==last (drop or
  duplicate per your sampler's convention — make it a config flag
  `loop_closure: "drop_last" | "duplicate_first"`); write `meta.json` with
  resolved `start_frame`/`last_frame` pointers and `prompt_hash`; **then** touch
  `.complete`. Returns `output_dir` so it can fan out to a re-scan.

Custom types: register `ANIM_MANIFEST` and `ANIM_META` as passthrough dict
types so they flow between nodes without serialization surprises.

---

## 7. Dynamic combos + previews (frontend + server)

ComfyUI widget combos are static at definition time. Dependency-aware,
character-scoped dropdowns require a **server route + a frontend extension**.
Pure-Python `INPUT_TYPES` cannot do this — call it out to the implementer.

### Server routes (registered by the pack via `PromptServer.instance.routes`)

- `GET /anim_coord/characters?root=…`
  → `[{ "name": "Cortex" }, …]` (one-level scan of `root`).

- `GET /anim_coord/options?root=…&character=…`
  → for every `(animation, direction in animation.directions)`, the resolve
  result trimmed to UI fields:
  ```json
  { "animation": "punch", "direction": "E",
    "status": "ready" | "generated" | "blocked" | "stale",
    "blocked_by": ["fighting_stance_idle@E"] }
  ```
  - `generated`: this pair's own output is complete.
  - `ready`: deps satisfied, not yet generated.
  - `blocked`: missing a dependency.
  - `stale`: generated/ready but a dep's `prompt_hash` is outdated.

- `GET /anim_coord/resolve?root=…&character=…&animation=…&direction=…`
  → full resolve incl. `start_preview` / `end_preview`:
  ```json
  { "selectable": true,
    "start_preview": { "ref":"fighting_stance_idle","direction":"E",
                       "url":"/anim_coord/frame?...&v=a1f3c9","stale":false },
    "end_preview":   { "ref":"fighting_stance_idle","direction":"E",
                       "url":"/anim_coord/frame?...&v=a1f3c9","stale":false },
    "blocked_by": [] }
  ```
  base_pose deps preview the `_base/{dir}.png`.

- `GET /anim_coord/frame?path=…&v=…`
  → streams a PNG. **Security: resolve `path` against `root`, reject `..`,
  symlinks, and absolute escapes; 404 anything outside `{root}`.** `v` is a
  cache-buster (= dep `prompt_hash` or file mtime) so a re-rendered dependency
  invalidates the browser thumbnail.

### Frontend extension (`web/` registered via `WEB_DIRECTORY`)

- On `character` change → fetch `/options`, repopulate the `animation` and
  `direction` combos. Render each option with a status glyph
  (`✓ generated`, `○ ready`, `⨯ blocked`, `▲ stale`). Optionally hide `blocked`
  behind a "show locked" toggle; default to showing them greyed so the node
  doubles as a progress board.
- On `(animation, direction)` change → fetch `/resolve`, draw two thumbnails in
  the node body: **"Starts from → {ref}.{dir} last frame"** and **"Ends at →
  {ref}.{dir} first frame"**. base_pose → show the base png. No dep → show
  "none / fresh init". Tint a thumbnail **amber** when its `stale` flag is set.
- After an `AnimationFrameWriter` run completes, trigger a re-fetch of
  `/options` (listen on the execution-finished event) so newly unlocked pairs
  appear without a manual refresh.

---

## 8. Behavior decisions (locked)

- Stale dependency → **warn, stay selectable** (amber). Not hard-blocked.
- `base_pose` coverage → **E, SE, S**. Any animation rooted on `base_pose` is
  selectable only in those directions.
- Mirrored directions (W/SW/NW) are **never generated**; engine mirrors at
  runtime. Tooling may surface coverage but must not offer them as render
  targets.
- FFLF endpoints are exactly two (`start_from`, `end_at`). Do **not** generalize
  to N anchors or multiple start images.

---

## 9. Build order & acceptance

1. **ManifestLoader** + schema/cycle validation. *Accept*: rejects a manifest
   with a bad `ref` or a cycle; warns on non-`4n+1` length.
2. **resolve()** core + unit tests against a fixture tree. *Accept*: `punch@E`
   is `blocked` with empty tree; becomes `ready` after a fake complete
   `fighting_stance_idle/E` (frames+meta+`.complete`); `start_image` resolves to
   that dir's `last_frame`, `end_image` to its `start_frame`.
3. **Selector** node wiring (anchors → IMAGE, composed negative, meta out).
4. **FrameWriter** with the mandated write order + loop closure + pointer meta.
   *Accept*: writing `fighting_stance_idle/E` flips `punch@E` to `ready` on the
   next `/options` call.
5. **Server routes** incl. `/frame` path-traversal hardening.
6. **Frontend** dynamic combos, status glyphs, dual previews, amber stale,
   auto-refresh on write.

*End-to-end accept*: with only `_base/E.png` present, the only combat option
offered is `fighting_stance_idle@E` (and `fighting_stance_entry@E`). Generate
the idle → `punch/kick/headbutt/block @E` unlock and preview the idle's guard
frame as both anchors. Edit the stance prompt in the manifest → those four show
amber `stale` but remain selectable.

---

## 10. Open extension points (not in v1)

- Per-character manifest overrides (Cortex cockier punch) layered over the
  generic base — a `characters/{name}.json` patch merged at load.
- Batch/queue mode: enumerate all `ready` pairs and emit a job list.
- Coverage report node: per character, a matrix of animation × direction status.

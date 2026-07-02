# Manifest schema reference

This is the authoritative field-by-field reference for the two JSON files that
drive comfyui-andypack: the **animation manifest** (`animations.json`) and a
**character manifest** (`character.json`). It's written for an agent that needs
to *author* these files for a new character/animation set, not just read them.

The runtime source of truth is `andypack/manifest.py` (structural validation),
`andypack/resolve.py` (cascade/anchor/staleness semantics), and
`andypack/io.py` (`character.json` write rules). `examples/animations.json` is
a complete, working example manifest — use it as a template.

---

## 1. Animation manifest (`animations.json`)

One manifest describes a **character-agnostic** dependency graph of two node
kinds — **poses** (stills) and **animations** (clips) — plus the shared
vocabulary (directions, camera language, defaults) they all draw on. No
character identity or prompt text specific to one character belongs here; that
lives in `character.json` (§2).

### Top-level shape

```json
{
  "version": 2,
  "directions": ["EAST", "SOUTH_EAST", "SOUTH", "SOUTH_WEST", "WEST", "NORTH_WEST", "NORTH", "NORTH_EAST"],
  "mirror_map": { "WEST": "EAST", "SOUTH_WEST": "SOUTH_EAST", "NORTH_WEST": "NORTH_EAST" },
  "view_phrases": { "EAST": "...", "SOUTH": "...", "...": "..." },
  "defaults": { "fps": 16, "length": 33, "width": 832, "height": 480, "shift": 3.0, "start_from": { "ref": "base" } },
  "globals": { "pose": {}, "animation": { "negative_prompt": "..." } },
  "poses": { "<pose_id>": { "...": "..." } },
  "animations": { "<animation_id>": { "...": "..." } }
}
```

| Key | Required | Type | Purpose |
|---|---|---|---|
| `version` | yes | int | Schema/format version. Bump only if you change the manifest's shape; `2` is current. |
| `directions` | recommended | array of strings | The canonical 8-way direction ordering (see §1.1). Used for lint checks and UI ordering — not itself load-bearing for resolution. |
| `mirror_map` | no | object | Declares that a direction's art is derived by flipping another direction's render, so it should be skipped by an automated sweep (see §1.2). Not applied automatically — it's a hint the selectors read. |
| `view_phrases` | no | object | Per-direction camera/orientation language, injected via the `{view_phrase}` template variable (see §1.5). Keys should match `directions`. |
| `defaults` | yes (for `start_from`) | object | Fallback generation params and the fallback I2V start image (see §1.3). |
| `globals` | no | object | The outermost prompt cascade layer, keyed by node kind (`"pose"` / `"animation"`), each `{ positive_prompt?, negative_prompt? }` (see §1.6). |
| `poses` | yes | object | Map of pose id → pose object (see §1.4). |
| `animations` | yes | object | Map of animation id → animation object (see §1.4). |

`load_manifest` / `validate_manifest` (`andypack/manifest.py`) enforce all of
this at load time — a structurally invalid manifest raises `ManifestError`
with a specific message rather than failing later mid-graph.

### 1.1 `directions`

The 8-way turnaround set, by convention the compass points:

```
EAST, SOUTH_EAST, SOUTH, SOUTH_WEST, WEST, NORTH_WEST, NORTH, NORTH_EAST
```

You are not required to use exactly these 8 — a manifest can define fewer or
differently-named directions — but every `directions` map on a pose/animation
(§1.4) and every entry in `view_phrases`/`mirror_map` should agree on the same
vocabulary. `directions` here is the *canonical* list used to lint against
(`collect_warnings` flags a pose/animation direction not present here, and a
`view_phrases` map missing an entry for a canonical direction).

Direction names are used as path segments on disk (`_base/EAST.png`), so they
must be filesystem-safe: no `/`, `\`, `..`, and not empty.

### 1.2 `mirror_map`

```json
"mirror_map": { "WEST": "EAST", "SOUTH_WEST": "SOUTH_EAST", "NORTH_WEST": "NORTH_EAST" }
```

A map of `derived_direction -> source_direction`. It means "art for
`derived_direction` is intended to come from flipping `source_direction`'s
render" (e.g. a horizontally-symmetric character only needs its right-facing
directions rendered; the left-facing ones are mirrored in an image editor or a
post-process step you own). **The pack does not perform the mirroring itself.**
Its only effect: when a sweep selector runs with `skip_mirrored=true`, any
direction that is a *key* in `mirror_map` is skipped from the auto-queue, so a
batch sweep won't burn a render on a direction you plan to derive by mirroring.

Leave this `{}` if every direction is rendered independently (e.g. an
asymmetric character, or when you'd rather render all 8 directly).

### 1.3 `defaults`

```json
"defaults": {
  "fps": 16, "length": 33, "width": 832, "height": 480, "shift": 3.0,
  "start_from": { "ref": "base" }
}
```

| Field | Type | Purpose |
|---|---|---|
| `fps` | int | Fallback frames-per-second for any animation that doesn't override it. |
| `length` | int | Fallback frame count. **Wan-friendly lengths are `4n+1`** (17, 21, 25, 33, 49, …) — a non-`4n+1` length is a non-fatal lint warning, not an error. |
| `width`, `height` | int | Fallback frame dimensions. **Must be divisible by 16** (Wan's VAE downsamples 8x with 2x2 latent patches) — non-multiples are a lint warning. |
| `shift` | int or float | Fallback `ModelSamplingSD3` shift value. |
| `start_from` | `{ "ref": "<pose_or_animation_id>" }` | The fallback I2V seed image for any animation that doesn't declare its own `start_from`. **Every animation must resolve to a `start_from` — explicit or via this default — or the manifest fails validation** (I2V needs a start image). |

Every animation's *effective* `width`/`height`/`length`/`fps` (its own value,
falling back to this) must resolve to a **positive integer** — this is
enforced at load time (`_require_positive_gen_params`), not just linted.
`shift` has no such requirement and may be omitted entirely.

### 1.4 Poses and animations

Both live under top-level `poses` / `animations`, each a map of **id → object**.
The id is used as a path segment on disk and in refs (`{"ref": "<id>"}}`), so it
must be a safe, unique, filesystem-safe string — snake_case is the convention
used throughout the seed manifest (`walk_stride`, `fighting_stance_idle`, …).

#### Pose object

```json
"fighting_stance": {
  "from": { "ref": "base" },
  "category": "anchor",
  "positive_prompt": "Edit the reference image to re-pose the character while keeping the exact same facing and camera angle — still {view_phrase}. ... Keep the character's identity, colors, and design exactly as in the reference: {character_prompt}. Full body, plain background, flat even lighting.",
  "directions": {
    "EAST": {}, "SOUTH_EAST": {}, "SOUTH": {}, "SOUTH_WEST": {},
    "WEST": {}, "NORTH_WEST": {}, "NORTH": {}, "NORTH_EAST": {}
  }
}
```

| Field | Required | Purpose |
|---|---|---|
| `from` | no | `{ "ref": "<pose_id>", "direction"?: "<DIR>" }` — the source pose this pose's FLUX edit is applied to. **Omit `from` only for a root pose** (conventionally named `base`) — it has no upstream node; the Character Creator supplies its source image (the character reference art) directly. `from.ref` must reference another *pose*, not an animation. `from.direction` is optional and defaults to `"same"` (the same direction being rendered) — set it to pull a *different* direction's render as the source. |
| `category` | no | Free-text bucket (e.g. `"anchor"`). Purely a filter key for the sweep selectors' `category` widget — has no effect on resolution/validation. Use it to group poses/animations you want to batch-generate together. |
| `positive_prompt` | no | This pose's own positive prompt layer, merged with `globals.pose.positive_prompt` (general → specific, blank-line joined). Typically references `{view_phrase}` and `{character_prompt}` (see §1.5). |
| `negative_prompt` | no | Same, but merged as a deduped comma-term list. **Omit for FLUX.2 Klein poses** — it has no negative-prompt path; the seed manifest's poses carry none. |
| `directions` | **yes** | Map of `direction name -> per-direction layer object` (see below). This is what makes a direction *selectable* for this pose — a direction absent here can never be picked, even if the source pose has it. |

**Per-direction layer object** (the value in `directions`): `{}` in almost
every case in the seed manifest — but it may carry its own
`positive_prompt` / `negative_prompt`, which is **not** cascaded
automatically; it only appears in the compiled prompt if the entity's own
prompt (or a global) references `{direction_prompt}`. Use a non-empty
direction layer when one specific direction needs a one-off tweak that
`view_phrases` doesn't already cover.

A direction layer may also carry `reference_image` — a **pose-only** field (not
valid on an animation's `directions` entry, though the loader only rejects the
type, not the entity kind):

| Field | Type | Purpose |
|---|---|---|
| `reference_image` | string (bare `*.png`) | Names a file under `user/default/andypack/pose_references/`, resolved by the nodes at render time — never a path (no `/`, `\`, `..`; must end in `.png` and be more than just `.png`). Validated at load time (`_validate_directions`), so a malformed value fails fast with a clear message. |

**Precedence and effect** — on a **root** pose (no `from`, e.g. `base`), the
authored `reference_image` *overrides* the bundled per-direction manikin as the
second FLUX.2 reference; on a **derived** pose (has `from`), it *adds* a second
reference where there would otherwise be none (a derived pose is normally a
single-reference edit). A pose direction with a `reference_image` that doesn't
exist on disk raises at render time rather than silently falling back — the
file must be written before the cell can render.

**Staleness** — `reference_image` is recorded on the pose's sidecar
(`pose_reference_name` feeds the resolved meta); changing which file a
direction points at (adding, removing, or swapping it) changes the recorded
dependency key-set, which re-stales that cell exactly like a swapped
`start_from`/`end_at` ref does for animations.

**Authoring workflow**: **Manikin Loader** (loads the bundled manikin for a
direction as an IMAGE + the direction name) → your own pose-generation graph
(e.g. an OpenPose ControlNet on an SDXL/SD1.5 checkpoint — FLUX.2 Klein has no
ControlNet path — or any pose-transfer edit) → **Pose Reference Writer** (saves
the result as `<name>_<DIRECTION>.png` under the pose-references dir and
returns the filename, ready to paste into this field). Wire the loader's
`DIRECTION` output into the writer's optional `direction_from` input to keep
both nodes locked to the same direction as you iterate.

#### Animation object

```json
"punch": {
  "category": "combat",
  "length": 21,
  "positive_prompt": "The character throws a single straight punch forward: ...",
  "negative_prompt": "both arms extended, second arm raised, flailing arm, ...",
  "directions": { "EAST": {}, "...": {} },
  "start_from": { "ref": "fighting_stance_idle" },
  "end_at": { "ref": "fighting_stance_idle" }
}
```

| Field | Required | Purpose |
|---|---|---|
| `category` | no | Same free-text sweep filter as poses (`"locomotion"`, `"aerial"`, `"stance"`, `"combat"`, `"surface"`, `"reactions"`, `"expression"`, … — pick your own vocabulary). |
| `length`, `fps`, `width`, `height`, `shift` | no | Per-animation overrides of the matching `defaults` field. See §1.3 for constraints (positive int; `4n+1` length and /16 dimensions are lint-only). |
| `positive_prompt` | no | Merged with `globals.animation.positive_prompt`, general → specific. This is the *motion* description — poses describe a static frame, animations describe the motion between/around it. |
| `negative_prompt` | no | Merged with `globals.animation.negative_prompt` as a deduped comma list. This is where Wan's CFG-driven negative pipeline actually matters (unlike FLUX.2 poses). |
| `directions` | **yes** | Same shape as a pose's `directions` — which directions this animation is selectable for. |
| `start_from` | no (see below) | `{ "ref": "<pose_or_animation_id>", "direction"?: "<DIR>" }` — the I2V seed image. **If omitted, `defaults.start_from` is used instead; if *both* are absent, the manifest fails validation** (every animation needs a start image). |
| `end_at` | no | `{ "ref": "<pose_or_animation_id>", "direction"?: "<DIR>" }` — the optional FFLF end anchor. Presence of `end_at` is what makes a clip FFLF (first-frame/last-frame) rather than open-ended I2V. |

**FFLF cross-wiring — do not invert this:**

- `start_from` consumes the dependency's **LAST** frame (or its single image,
  if the dep is a pose).
- `end_at` consumes the dependency's **FIRST** frame (or its single image).
- A clip **loops** — and the writer trims the duplicated closing frame — iff
  `start_from` and `end_at` resolve to the *literal same image*. There is no
  `loop` field to author; it's a pure consequence of what you point the two
  anchors at. To make a looping clip, point both at the same pose (e.g.
  `fighting_stance_idle`'s `start_from`/`end_at` both ref `fighting_stance`).
  To make a "returns to neutral" clip (not a loop), point `start_from` at a
  stance pose and `end_at` at `base` (or omit `end_at` for an open one-way
  clip, like `fall`, `defeat`, `wonder`, `turn_around` in the example).

**Dependency graph note:** `start_from`/`end_at` (and a pose's `from`) form a
DAG over `poses` + `animations`. The manifest is validated to be acyclic
(`_detect_cycles`) — a ref chain that loops back on itself is rejected at load
time. `topo_order` walks this graph to drive dependency-ordered sweeps, so
always design new poses/animations as extensions of existing dependency roots
(usually `base`) rather than free-floating.

### 1.5 Template variables

Four opt-in tokens can appear inside any `positive_prompt` / `negative_prompt`
string (global, pose, or animation level). They are substituted in a single
literal pass — **not** recursive, so a token inside an injected value is left
as literal text, and an unrecognized `{...}` or stray brace just survives
untouched:

| Token | Expands to | Positive-only? |
|---|---|---|
| `{character_prompt}` | The character's own `positive_prompt` / `negative_prompt` from `character.json` (field-matched: positive context pulls the character's positive, negative context pulls its negative). | no |
| `{view_phrase}` | The manifest's `view_phrases[direction]` string for the direction being rendered. | **yes** — expands to `""` in a negative field, since it's inherently affirmative camera language. |
| `{direction_prompt}` | The *entity's own* per-direction layer's `positive_prompt` / `negative_prompt` (the `directions.<DIR>` object on this same pose/animation) — field-matched like `{character_prompt}`. | no |
| `{direction_name}` | The bare direction name (e.g. `EAST`), literal string substitution. | no |

This is the mechanism that keeps the manifest **character-agnostic**: identity
only ever enters a compiled prompt through `{character_prompt}`, and
per-direction camera language is authored **once**, in `view_phrases`, then
referenced via `{view_phrase}` from every pose/animation instead of being
repeated in each entity's per-direction layer.

### 1.6 Prompt cascade & compile order

For a given `(pose_or_animation, direction)`, the final prompt is built as:

1. Take `globals[kind].positive_prompt` (kind = `"pose"` or `"animation"`) and
   the entity's own `positive_prompt`.
2. Substitute template variables (§1.5) into **each layer independently**.
3. Merge, general → specific: `merge_layers(global, entity)` — non-empty
   layers joined with a blank line; empty layers dropped.
4. Same for negative, except `merge_negative` splits each layer on commas,
   strips terms, and case-insensitively dedupes across layers (first
   occurrence wins) before rejoining with `, "`.

The character's `character.json` layer and the entity's own per-direction
layer are **never** cascade layers — they only ever enter through
`{character_prompt}` / `{direction_prompt}` in step 2. If an entity's prompt
never references `{character_prompt}`, the character's prompt text simply
doesn't appear in that render — this is a common authoring mistake to check
for.

The final merged (positive, negative) pair is hashed (whitespace-normalized)
into `prompt_hash`, which is what staleness detection compares against on
every future load — editing any layer in the cascade (including
`character.json`) invalidates every descendant that inherits it.

---

## 2. Character manifest (`character.json`)

One file per character, at `<characters_root>/<snake_case_name>/character.json`
(`<characters_root>` is `ComfyUI/output/characters/` at runtime). It supplies
**only** the character-specific identity layer — never structural data like
directions or gen params — plus an optional extension of the shared manifest's
`poses`/`animations`.

### Shape

```json
{
  "positive_prompt": "a small orange fox-like creature with oversized ears, a fluffy tri-tipped tail, round amber eyes, and a leather satchel slung across one shoulder",
  "negative_prompt": "different creature, wrong colors, missing satchel, extra tails",
  "poses": {
    "cape_flourish": { "from": { "ref": "base" }, "positive_prompt": "...", "directions": { "SOUTH": {} } }
  },
  "animations": {
    "cape_flourish_idle": { "start_from": { "ref": "cape_flourish" }, "end_at": { "ref": "cape_flourish" }, "positive_prompt": "...", "directions": { "SOUTH": {} } }
  }
}
```

| Field | Required | Purpose |
|---|---|---|
| `positive_prompt` | no | The character's visual identity description. Surfaces in a compiled prompt **only** where the entity/global prompt contains `{character_prompt}` — write it once here, reference it everywhere. Omitting the field (or writing `""`) means no identity text is injected anywhere. |
| `negative_prompt` | no | Character-specific negative terms (e.g. "wrong species", "missing accessory"). Same substitution rule, negative context. Typically only relevant on the Wan animation path (FLUX.2 poses have no negative path in the seed setup). |
| `poses` | no | A character-specific *overlay* onto the shared manifest's `poses` map — same object shape as §1.4. Entries here are folded in by id (character entries override/extend the base manifest's), and the merged result is **re-validated** on load, so a bad ref or an introduced cycle is rejected rather than silently resolved. Use this for poses that only make sense for one character (e.g. a signature stance). |
| `animations` | no | Same overlay mechanism for `animations`. |

**`character.json` carries no provenance** (no `prompt_hash`/`created_utc`/
`render_id` fields) — it is not itself a render node. It is *authored*, not
*written by a render*. The `base` pose's own per-direction sidecars are what
root the staleness tree; editing `positive_prompt`/`negative_prompt` here
re-stales every descendant through the ordinary prompt-hash drift mechanism
(because `{character_prompt}` expands into their compiled prompts), not
through any provenance field on this file.

**Only `positive_prompt` and `negative_prompt` are "owned" keys** — if you
(or the tooling that writes this file) rewrite the identity layer, any other
top-level key (i.e. `poses`/`animations`) already present is preserved
untouched. When authoring by hand, this just means: it's safe to include
`poses`/`animations` here even though the Character Creator UI only ever
edits the prompt fields.

### What does *not* go in `character.json`

- `directions`, `view_phrases`, `mirror_map`, `defaults`, `globals` — those
  are manifest-level and shared across every character.
  `width`/`height`/`length`/`fps`/`shift` — animation-level, in the base
  manifest or its overlay entries, never here directly (they'd go inside a
  `poses`/`animations` overlay entry, same as any other pose/animation).
- Any generation/gen-param field on a bare top-level key — this file is
  identity text + an optional structural overlay, nothing else.

### On-disk layout this file participates in

```
output/characters/<name>/
  character.json                    <- this file
  _reference.png                    persisted reference art (optional)
  _base/EAST.png  _base/EAST.json   base pose render + sidecar (root of staleness)
  _<pose_id>/<DIR>.png + .json      other pose renders
  <animation_id>/<DIR>/frame_*.png + meta.json
```

`character.json` itself has no fixed relationship to these render
directories beyond living alongside them — it is read fresh on every resolve,
never treated as a cache.

---

## 3. Minimal worked example

A tiny manifest with one root pose, one derived pose, and one looping
animation — enough structure to be valid and renderable:

```json
{
  "version": 2,
  "directions": ["SOUTH"],
  "view_phrases": { "SOUTH": "in a dead-on front view, facing the camera" },
  "defaults": {
    "fps": 16, "length": 33, "width": 832, "height": 480, "shift": 3.0,
    "start_from": { "ref": "base" }
  },
  "globals": {
    "pose": {},
    "animation": { "negative_prompt": "{character_prompt}, blurry, low quality" }
  },
  "poses": {
    "base": {
      "positive_prompt": "Edit the first image so the same character is shown {view_phrase}. Preserve identity exactly: {character_prompt}. Neutral standing pose, plain background.",
      "directions": { "SOUTH": {} }
    }
  },
  "animations": {
    "idle": {
      "category": "stance",
      "positive_prompt": "The character breathes gently in place, swaying slightly, returning to the starting pose for a seamless loop.",
      "directions": { "SOUTH": {} },
      "start_from": { "ref": "base" },
      "end_at": { "ref": "base" }
    }
  }
}
```

Paired with a character:

```json
{
  "positive_prompt": "a small blue robot with a single glowing eye and stubby articulated arms",
  "negative_prompt": "human, organic, extra limbs"
}
```

`idle`'s `start_from` and `end_at` both point at `base`, so it resolves as a
loop and the frame writer drops the duplicate closing frame automatically —
no `loop` field is or should be authored.

---

## 4. Authoring checklist

- [ ] Every pose/animation id and every direction name is a safe, unique,
      filesystem-legal segment (no `/`, `\`, `..`, not empty).
- [ ] Exactly one pose has no `from` — the root (conventionally `base`) —
      and it lists every direction you intend to use anywhere.
- [ ] Every other pose's `from.ref` points at another pose, never an
      animation.
- [ ] Every animation resolves a `start_from`, either its own or via
      `defaults.start_from`.
- [ ] `end_at`/`start_from` refs never form a cycle back to their own
      pose/animation (directly or transitively).
- [ ] Animation `length` is `4n+1` and `width`/`height` are multiples of 16,
      unless you're intentionally accepting the lint warning.
- [ ] Camera/orientation language lives once in `view_phrases`, referenced via
      `{view_phrase}` — not repeated per pose/animation.
- [ ] Any prompt that should carry character identity references
      `{character_prompt}` explicitly; it is not injected implicitly.
- [ ] Character identity text (`positive_prompt`/`negative_prompt`) lives only
      in `character.json`, never inlined into the shared manifest.
- [ ] FLUX.2-edited poses carry no `negative_prompt` layer (no negative path);
      Wan animations use `negative_prompt` freely.

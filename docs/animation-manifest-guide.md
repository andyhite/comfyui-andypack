# Animation Manifest — Authoring Guide

This explains the `animations.json` format used by the **comfyui-andypack**
Animation Coordinator: what each field means, the rules you must follow, and how
to author a full set of poses and animations for a character roster.

The manifest is **character-agnostic**: it describes poses and animations in the
abstract. Per-character identity (and optional per-character poses/animations)
lives in each character's `_concept.json` (see §8). Author the shared library
here; the character files specialize it.

The authoritative reference manifest is `examples/animations.json`. When in
doubt, copy a working entry from there.

---

## 1. The mental model

There are three kinds of node in one dependency graph:

```
concept seed (_concept.png, uploaded 3/4 art, direction-agnostic)
   │  (per-direction FLUX edit)
   ▼
pose  (a still, e.g. "base", "fighting_stance")  ── from: concept or another pose
   │  (WAN image-to-video)
   ▼
animation  (a clip, e.g. "walk", "punch")        ── start_from / end_at: pose or animation
```

- A **pose** is a single still image produced by editing a *source image* (the
  `from` dependency) with FLUX. `base` is edited from the concept; other poses
  are edited from `base` (or another pose).
- An **animation** is a WAN clip. WAN is **image-to-video (I2V)**: it always
  needs a **start image** (the first frame to animate from). If you also give it
  an **end image**, it runs **first-frame/last-frame (FFLF)** and interpolates
  between them.

**The single most important rule:** every animation needs a start image. See §6.

The pack only *resolves and writes back* — it never samples. It decides what's
selectable, composes the prompts, picks the anchor images, and gates each entry
on whether its dependencies have been rendered yet. Your ComfyUI graph does the
actual FLUX/WAN generation.

---

## 2. Top-level shape

```jsonc
{
  "version": 1,
  "directions": ["EAST", "SOUTH_EAST", "SOUTH", "SOUTH_WEST", "WEST", "NORTH_WEST", "NORTH", "NORTH_EAST"],
  "mirror_map": { "WEST": "EAST", "SOUTH_WEST": "SOUTH_EAST", "NORTH_WEST": "NORTH_EAST" },
  "defaults": { "fps": 16, "length": 33, "start_from": { "ref": "base" } },
  "globals": {
    "animation": { "negative_prompt": "…shared animation negatives…" },
    "pose":      { "negative_prompt": "…shared pose negatives…" }
  },
  "poses":      { "<pose_id>": { … }, … },
  "animations": { "<animation_id>": { … }, … }
}
```

| key | type | required | meaning |
|---|---|---|---|
| `version` | int | yes | Schema version; copied into rendered metadata. Use `1`. |
| `directions` | string[] | yes | The canonical direction vocabulary (see §3). |
| `mirror_map` | object | no | Directions produced by horizontally mirroring another, so you don't author them (e.g. `WEST` is `EAST` mirrored). Documentation/coverage only. |
| `defaults` | object | yes | `fps`, `length`, and **`start_from`** fallbacks (see §6). |
| `globals` | object | no | A shared cascade layer per kind: `globals.animation` and `globals.pose`, each a `{ positive_prompt?, negative_prompt? }`. |
| `poses` | object | yes | `id → Pose` (§4). |
| `animations` | object | yes | `id → Animation` (§5). |

Ids (`<pose_id>`, `<animation_id>`) are `lower_snake_case` and must be unique
across the whole manifest (a ref like `"punch"` resolves to either a pose or an
animation — they share one namespace).

---

## 3. Directions

- Directions are spelled out in `UPPER_SNAKE_CASE`: `EAST`, `SOUTH_EAST`,
  `SOUTH`, `SOUTH_WEST`, `WEST`, `NORTH_WEST`, `NORTH`, `NORTH_EAST`.
- Every pose/animation has a `directions` **map** whose **keys are the
  directions that entry supports**. Each value is an optional per-direction
  prompt layer (usually `{}`).
- A direction key must be one of the top-level `directions`.
- `mirror_map` lists directions you intend to get by mirroring (don't author
  those as separate keys unless an entry genuinely differs when mirrored).

Example: an animation that only faces right is `"directions": { "EAST": {} }`.
A pose authored for three views is `"directions": { "EAST": {…}, "SOUTH_EAST": {…}, "SOUTH": {…} }`.

---

## 4. Pose

```jsonc
"base": {
  "from": { "ref": "concept" },
  "positive_prompt": "a neutral standing pose, arms relaxed at its sides",
  "directions": {
    "EAST":       { "positive_prompt": "facing directly right in profile" },
    "SOUTH_EAST": { "positive_prompt": "facing down-right at a three-quarter view" },
    "SOUTH":      { "positive_prompt": "facing toward the viewer" }
  }
},
"fighting_stance": {
  "from": { "ref": "base", "direction": "same" },
  "positive_prompt": "a ready fighting stance, weight low, fists raised in a guard",
  "directions": { "EAST": { "positive_prompt": "facing right" } }
}
```

| key | type | required | notes |
|---|---|---|---|
| `from` | `{ ref, direction? }` | yes | The source image FLUX edits. `ref` is `"concept"` or another pose id (**never an animation**). `direction` is `"same"` (default — use the selected direction) or a literal direction. A `concept` source ignores direction. |
| `positive_prompt` | string | no | Per-pose positive layer. |
| `negative_prompt` | string | no | Per-pose negative layer. |
| `directions` | `{ <DIR>: Layer }` | yes | Selectable directions; each value an optional `{ positive_prompt?, negative_prompt? }` (often `{}`). |
| `category` | string | no | Optional UI grouping for poses (animations use it heavily; poses without it group under "(all)"). |

A pose is **selectable** once its `from` source has been generated. So `base`
(from concept) unlocks once the concept exists; `fighting_stance` (from `base`)
unlocks once `base` is generated in that direction.

---

## 5. Animation

```jsonc
"punch": {
  "category": "combat",
  "loop": false,
  "length": 21,
  "start_from": { "ref": "fighting_stance_idle" },
  "end_at":     { "ref": "fighting_stance_idle" },
  "positive_prompt": "throw a straight jab to the right…",
  "negative_prompt": "both arms extended, extra arm",
  "directions": { "EAST": {} }
}
```

| key | type | required | notes |
|---|---|---|---|
| `category` | string | recommended | Grouping label for the UI cascade (e.g. `locomotion`, `combat`). Free string; pick a consistent set. |
| `positive_prompt` | string | no | Per-animation positive layer (describe the motion). |
| `negative_prompt` | string | no | Per-animation negative layer. |
| `loop` | bool | no (false) | True for seamless loops (idles, walk cycles). Drives loop-closure on write. |
| `length` | int | no (`defaults.length`) | Frame count. **Keep it `4n+1`** for WAN (e.g. 17, 21, 25, 33, 49); the loader warns otherwise. |
| `fps` | int | no (`defaults.fps`) | Playback fps. |
| `start_from` | `{ ref, direction? }` | no* | The **start image** dependency (I2V seed). `ref` ∈ `"concept"` \| pose id \| animation id. *If omitted, it falls back to `defaults.start_from`.* |
| `end_at` | `{ ref, direction? }` | no | If present, the clip is **FFLF**: this is the end-image dependency. Omit for plain I2V. |
| `directions` | `{ <DIR>: Layer }` | yes | Selectable directions; per-direction layers. |

An animation is **selectable** once its start dependency (and `end_at`, if
present) has been generated in the resolved direction.

---

## 6. Start images & FFLF (the rules that matter most)

**Every animation must resolve a start image.** Either:
- give it an explicit `start_from`, or
- rely on `defaults.start_from` (set to `{ "ref": "base" }` in the reference
  manifest) — any animation without `start_from` seeds from the base pose.

The loader **rejects** a manifest with an animation that has neither. Free clips
(walks, idles, reactions, expressions) generally have **no** `start_from` and
seed from `base`. Transitions and combat moves usually set an explicit one.

**Anchor resolution (do not invert):**
- `start_from` → the dependency's **LAST** frame (for an animation dep) or its
  single image (for a concept/pose dep). This becomes the clip's first frame.
- `end_at` → the dependency's **FIRST** frame (animation dep) or its single
  image (concept/pose dep). This becomes the clip's last frame.
- A concept/pose dependency is a single still, so it resolves the **same** image
  for whichever slot uses it.

This cross-wiring is what makes chains seamless. Example for a looping idle and
the moves around it:

```jsonc
"fighting_stance_idle": { "loop": true,  "start_from": { "ref": "fighting_stance" } },        // seeds from the stance pose; loops
"fighting_stance_entry":{ "loop": false, "start_from": { "ref": "base" }, "end_at": { "ref": "fighting_stance_idle" } }, // base → into the idle's first frame
"fighting_stance_exit": { "loop": false, "start_from": { "ref": "fighting_stance_idle" }, "end_at": { "ref": "base" } }, // idle's last frame → back to base
"punch":                { "loop": false, "start_from": { "ref": "fighting_stance_idle" }, "end_at": { "ref": "fighting_stance_idle" } } // starts at idle's last frame, ends at idle's first → returns to the loop
```

`direction` on any dep is `"same"` (default) or a literal, exactly like a pose's
`from`.

---

## 7. Prompts: compile = merge + substitute

A prompt is compiled in two steps, per axis (positive, negative):

1. **Merge** `globals.<kind>` + the entity prompt (`<kind>` is `pose` or
   `animation`). Positives are kept **verbatim**, joined with a blank line
   (`\n\n`). Negatives are **comma-separated term lists**: split on commas,
   de-duplicated case-insensitively (first wins), re-joined with `", "`.
2. **Substitute** the template variables in the merged text.

The entity prompt is the **composition root** — it decides where the character
identity and the per-direction text land. The identity layer and the
per-direction layer are **not** merged; they are inert until referenced.

### Template variables

Resolved by **field context** — the same token pulls the positive source in a
`positive_prompt` and the negative source in a `negative_prompt`:

| Variable | in a positive prompt | in a negative prompt |
|---|---|---|
| `{identity_prompt}` | concept `positive_prompt` | concept `negative_prompt` |
| `{direction_prompt}` | the direction's `positive_prompt` | its `negative_prompt` |
| `{direction_name}` | the direction's bare name (e.g. `EAST`) | same |

```jsonc
"positive_prompt": "{identity_prompt} mid-stride, running. As viewed from the {direction_name}: {direction_prompt}"
```

Variables resolve in **either** a global or an entity prompt (substitution runs
on the merged text). Replacement is literal: unknown `{...}` tokens and stray
braces are left alone; an empty/absent source expands to nothing — and because
substitution feeds the negative term-list dedupe, an empty `{direction_prompt}`
in a negative can't leave a stray `, ,`. A prompt that references no variable
gets none of that text.

Authoring guidance:
- **`globals.animation` / `globals.pose`**: cross-cutting quality negatives
  (`deformed, blurry, low quality, …`); reference `{identity_prompt}` here to
  fold the character's negative identity into every render.
- **Entity `positive_prompt`**: describe the motion/pose, and splice
  `{identity_prompt}` / `{direction_prompt}` / `{direction_name}` where they
  belong.
- **Per-direction layer**: the angle-specific text that `{direction_prompt}`
  pulls in (and `{direction_name}` names); inert unless referenced.
- **Character identity** (look, colors, "mouthless", etc.) lives in the
  character's `_concept.json` — see §8 — and is pulled in via `{identity_prompt}`.

Never write a field literally called `prompt` or `negative`; the keys are
`positive_prompt` and `negative_prompt`.

---

## 8. Per-character files (`_concept.json`)

Each character directory holds `_concept.png` (the seed art) and an optional
`_concept.json`. That file is the character's **identity layer** and may also
**extend the manifest** with character-specific entities:

```jsonc
{
  "positive_prompt": "a big-headed mouthless hero, off-white skin, blue overalls",
  "negative_prompt": "realistic, photographic",
  "poses":      { "<character_pose_id>": { …same shape as a manifest pose… } },
  "animations": { "<character_anim_id>": { …same shape as a manifest animation… } }
}
```

- `positive_prompt` / `negative_prompt` are the **identity layer** — pulled
  into a render wherever a prompt references `{identity_prompt}` (see §7),
  resolved to the positive in a positive field and the negative in a negative
  field. This is where character look/identity lives; it is referenced, not
  auto-merged.
- `poses` / `animations` (optional) are **merged into the manifest by id** when
  that character is selected. New ids add entries; existing ids override the
  shared ones. Their refs can point at shared *or* character entities. The
  merged manifest is re-validated, so a bad ref or a cycle is rejected.

Author shared, reusable stuff in `animations.json`; reserve `_concept.json` for
what's unique to one character.

---

## 9. Validation rules (the loader enforces these)

A manifest is rejected if:
- `version` isn't an int, or `poses`/`animations` aren't objects.
- A pose's `from.ref` is missing, or points at an **animation**.
- A `start_from`/`end_at`/`from`/`defaults.start_from` `ref` doesn't resolve to a
  known `concept` / pose id / animation id.
- An **animation has no start** — no `start_from` and no `defaults.start_from`.
- There's a **dependency cycle** (e.g. pose A `from` B and B `from` A).
- A pose/animation is missing its `directions` map.

It **warns** (doesn't reject) when an animation `length` isn't `4n+1`.

---

## 10. How to author a full set (recommended workflow)

1. **Pick the directions** the project supports and list them in `directions`;
   note any mirrored ones in `mirror_map`. Most entries will be a subset (often
   just `EAST`, or the three front views for poses).
2. **Set `defaults`**: `fps`, a default `length` (`4n+1`), and
   `start_from: { "ref": "base" }` so free clips seed from base automatically.
3. **Define the pose chain**: at minimum `base` (`from: concept`). Add stance/
   state poses (`from: base`) that animations will seed from
   (`fighting_stance`, `crouch`, `aim`, …). Poses are stills, so keep their
   `positive_prompt` about the static shape.
4. **Fill `globals`** with shared quality negatives (and any always-on positives).
5. **Author animations grouped by `category`.** For each:
   - Write the motion in `positive_prompt`; add motion-specific `negative_prompt`
     only as needed (rely on `globals`).
   - Set `loop: true` for cycles/idles.
   - Choose `length` (`4n+1`) and `fps`.
   - Decide the **start**: free clip → no `start_from` (uses base); state-bound
     clip → `start_from` the relevant pose or idle. Decide whether it should
     return to a state → add `end_at` (FFLF), remembering the cross-wiring in §6.
   - Set `directions` (often `{ "EAST": {} }`).
6. **Keep identity out of the manifest.** Looks/colors/character traits go in
   each `_concept.json`. Use `_concept.json` `poses`/`animations` only for truly
   character-specific moves.
7. **Validate** by loading it (the pack rejects bad refs/cycles/startless
   animations and warns on non-`4n+1` lengths). Fix anything it flags.

A complete, working example covering ~40 animations across 8 categories lives in
`examples/animations.json` — use it as the template.

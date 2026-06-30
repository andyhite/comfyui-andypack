# Animation Coordinator — Cascading Pose Resolver (design)

Status: approved design, supersedes the model in `docs/anim-coord-node-spec.md`.
Date: 2026-06-29.

This revises the Animation Coordinator from a base-pose + facial-negative FFLF
resolver into a **single dependency graph of three node kinds** (concept seed →
poses → animations) with **cascading prompts** and **transitive staleness**. The
pack still only *resolves and writes back* — it never samples or generates. The
ComfyUI graph the user builds drives the actual FLUX (pose/frame edits) and WAN
(animation) generation; this pack manages prompts, reference images, dependency
gating, and completion metadata.

---

## 1. What changed from the original spec

| Original (`anim-coord-node-spec.md`) | This design |
|---|---|
| `base_pose` is a special root (`_base/{dir}.png`, "never stale") | Concept seed (`_concept.png`) is the root; `_base` becomes an ordinary **pose** generated from the concept |
| Anchors come from `base_pose` png or an animation frame | Anchors come from a **concept**, a **pose**, or an **animation** — uniform ref typing |
| Negatives composed from `global + facial(frontal/default) + anim.negative` | **No facial/global special cases.** Cascading positive+negative layers, merged uniformly |
| `prompt`/`negative` are single strings; `directions` is a list | Every cascade layer has optional `prompt`/`negative`; `directions` is a **map** keyed by direction, each value an optional per-direction layer |
| Completion = `.complete` sentinel + `meta.json` + frames | Completion = the **atomically-written-last `meta.json`/sidecar** + payload. No `.complete` |
| Staleness is one-level (direct dep's rendered hash vs manifest) | **Transitive-on-hash**: an edit at any cascade layer or any ancestor ripples downstream |
| Two node kinds (Selector + FrameWriter) | Adds **pose** Selector + Writer and concept handling alongside the animation nodes |

The FFLF cross-wiring rule is unchanged: `start_from` consumes a dependency's
**last** frame; `end_at` consumes its **first** frame.

---

## 2. Node kinds (one dependency graph)

```
_concept.png  (seed: 3/4 concept art, uploaded, direction-agnostic)
   │  + optional _concept.json identity layer { prompt?, negative? }
   ▼  (per-direction FLUX edit)
_base/{dir}.png            pose   from: concept
   ▼  (FLUX edit)
_fighting_stance/{dir}.png pose   from: base
   ▼  (start_from)
fighting_stance_idle       animation (WAN, loop)   start_from: fighting_stance
   ├─ fighting_stance_entry  start_from: base                 · end_at: fighting_stance_idle
   ├─ fighting_stance_exit   start_from: fighting_stance_idle · end_at: base
   └─ punch / kick / …       start_from: fighting_stance_idle · end_at: fighting_stance_idle
```

**Concept seed** — the per-character root. Uploaded image `_concept.png`,
direction-agnostic. Optional sidecar `_concept.json` holds the per-character
**identity layer** (`{ prompt?, negative? }`) that reinforces character details
the generated frames can't see. Referenced by the reserved ref `"concept"`.
Never rendered by the pack, so it has no rendered hash of its own — but its
identity layer participates in every descendant's merged prompt (see §4), so
editing it ripples downstream.

**Pose** — a per-direction still produced by a FLUX edit of a *source image*.
Declares `from` (a `concept` or another pose), a per-pose layer, and a
per-direction layer map. Renders to `_{poseId}/{dir}.png` + a sidecar.

**Animation** — a WAN clip. Declares `start_from`/`end_at` (each a pose or
animation id, or `concept`), a per-animation layer, and a per-direction layer
map. Renders to `{animId}/{dir}/frame_*.png` + `meta.json`.

---

## 3. Filesystem layout & completeness

```
{root}/{character}/
  _concept.png                          seed image (uploaded)
  _concept.json                         optional identity layer { prompt?, negative? }

  _base/E.png   _base/E.json            pose frame + sidecar
  _base/SE.png  _base/SE.json
  _fighting_stance/E.png  _fighting_stance/E.json

  fighting_stance_idle/E/
    frame_00000.png … frame_000NN.png
    meta.json                           written LAST (atomic) = completion sentinel
```

`character` = any directory under `{root}` containing a `_concept.png`, a pose
dir, or an animation dir.

### Completion (atomicity, no `.complete`)

The meta/sidecar is written **last**, via temp file + atomic rename
(`os.replace`), after the payload. Its presence is the completion signal:

- **Concept** complete ⇔ `_concept.png` exists. (`_concept.json` is optional.)
- **Pose** `_{id}/{dir}` complete ⇔ `{dir}.png` exists AND `{dir}.json` parses.
- **Animation** `{id}/{dir}` complete ⇔ `meta.json` parses AND `frames.count`
  frame files are present.

A half-rendered directory has no parseable meta/sidecar and reads as incomplete.
**Re-render discipline:** delete the meta/sidecar first (dir goes incomplete),
write the payload, then write the meta/sidecar last.

### Sidecar / meta shapes

Pose sidecar `_{id}/{dir}.json`:
```json
{
  "kind": "pose",
  "pose": "fighting_stance",
  "direction": "E",
  "from": { "ref": "base", "direction": "E" },
  "image": "E.png",
  "prompt_hash": "sha1:…",
  "manifest_version": 1,
  "created_utc": "2026-06-29T18:22:04Z"
}
```

Animation `meta.json` (unchanged except `.complete` is gone; `prompt_hash` is the
merged-prompt hash):
```json
{
  "kind": "animation",
  "animation": "punch",
  "direction": "E",
  "fps": 16,
  "length": 21,
  "loop": false,
  "seed": 848213,
  "prompt_hash": "sha1:…",
  "manifest_version": 1,
  "frames": { "dir": ".", "pattern": "frame_{:05d}.png", "count": 21 },
  "start_frame": "frame_00000.png",
  "last_frame": "frame_00020.png",
  "created_utc": "2026-06-29T18:22:04Z"
}
```

`start_frame`/`last_frame` are relative pointers into the frame set (no sidecar
frame copies).

---

## 4. Cascading prompts

Every render's final positive and negative are built by merging **layers,
general → specific**. Each layer optionally supplies `prompt` (positive) and/or
`negative`.

**Animation** `anim@dir` layer stack:
```
globals.animation → animations[anim] → animations[anim].directions[dir]
```

**Pose** `pose@dir` layer stack:
```
globals.pose → poses[pose] → poses[pose].directions[dir]
```

Positives merge into the final positive; negatives merge into the final
negative; the merge rule is identical for both axes. Any layer may be omitted.

**Identity is opt-in, not an automatic layer.** The per-character
`_concept.json` identity is no longer prepended to the cascade. Instead, any
layer may reference `{identity_positive}` (→ identity `positive_prompt`) or
`{identity_negative}` (→ identity `negative_prompt`) to splice the identity in
place. Tokens are expanded per-layer **before** the merge — so an expanded
negative term list dedupes against sibling terms — via literal `str.replace`
(unknown `{...}` tokens and stray braces survive; absent identity → `""`). See
`2026-06-29-identity-prompt-variable-design.md`.

### Merge rule

`merge_layers(*parts)`:
1. For each non-empty part, split on `,`.
2. Strip each term; drop empties.
3. Dedupe case-insensitively, preserving first occurrence.
4. Re-join with `", "`.

This is lossless for prose positives (commas just normalize) and collapses
duplicate boilerplate for term-list negatives. It is the same function used for
both positive and negative composition.

### Prompt hash

```
prompt_hash = "sha1:" + sha1( normalize(merged_positive) + "␟" + normalize(merged_negative) )
```
where `normalize` strips ends and collapses internal whitespace runs to one
space, and `␟` is U+241F (UNIT SEPARATOR). The hash is over the **fully merged**
prompt (identity tokens already expanded), so a change at *any* layer — global,
entity, or direction — changes the hash. Identity is read from `_concept.json`
at hash time, so editing identity invalidates only descendants that reference
`{identity_positive}` / `{identity_negative}`, without any special-casing.

---

## 5. Manifest schema

The manifest (`animations.json`) is **character-agnostic and identity-free** —
character identity lives only in the per-character `_concept.json`. Top level:

| key | type | notes |
|---|---|---|
| `version` | int | copied into meta/sidecars |
| `directions` | string[] | canonical 8-way ordering (tooling/coverage) |
| `mirror_map` | obj | engine-mirrored dirs → source; never generated |
| `defaults` | `{fps,length}` | fallback fps/length for animations |
| `globals` | `{ animation?: Layer, pose?: Layer }` | top cascade layer per kind |
| `poses` | obj | id → Pose |
| `animations` | obj | id → Animation |

`Layer = { prompt?: string, negative?: string }`.

**Pose:**
| key | type | req | notes |
|---|---|---|---|
| `from` | `{ ref, direction? }` | yes | source image: `ref` is `"concept"` or a pose id; `direction` is `"same"`(default) or a literal. `concept` source ignores direction |
| `prompt` / `negative` | string | no | per-pose cascade layer |
| `directions` | `{ <dir>: Layer }` | yes | keys = selectable directions; each value an optional per-direction layer (often `{}`) |

**Animation:**
| key | type | req | notes |
|---|---|---|---|
| `category` | string | yes | grouping/label |
| `prompt` / `negative` | string | no | per-animation cascade layer |
| `loop` | bool | no (false) | loop closure / self-FFLF |
| `length` | int | no (`defaults.length`) | keep `4n+1` for Wan |
| `fps` | int | no (`defaults.fps`) | |
| `start_from` | `{ ref, direction? }` | no | start anchor dep; `ref` ∈ pose/animation id or `concept` |
| `end_at` | `{ ref, direction? }` | no | end anchor dep |
| `directions` | `{ <dir>: Layer }` | yes | keys = selectable directions; per-direction layer |

`direction: "same"` (or omitted) on a dep means "use the selected direction"; a
literal overrides.

---

## 6. Resolution algorithm (pure `resolve.py`)

`resolve.py` stays free of ComfyUI/torch imports. It reads the rendered tree and
`_concept.json` from disk; the manifest dict is passed in.

### Ref typing
```python
def node_kind(manifest, ref):
    if ref == "concept": return "concept"
    if ref in manifest["poses"]: return "pose"
    if ref in manifest["animations"]: return "animation"
    raise KeyError(ref)   # loader validates refs; this is defense in depth
```

### Identity + merge + hash
```python
identity = read_layer(f"{root}/{character}/_concept.json")   # {} if absent
merged_positive = merge_layers(identity.prompt, globals[kind].prompt, entity.prompt, entity.directions[dir].prompt)
merged_negative = merge_layers(identity.negative, globals[kind].negative, entity.negative, entity.directions[dir].negative)
prompt_hash = sha1_of(merged_positive, merged_negative)
```

### Completeness (per kind, §3)
```python
concept_complete(root, char)            = exists(_concept.png)
pose_complete(root, char, id, dir)      = exists(_{id}/{dir}.png) and parses(_{id}/{dir}.json)
animation_complete(root, char, id, dir) = parses(meta.json) and count_frames >= meta.frames.count
```

### Anchors
- **start_from**: concept/pose dep → that single png; animation dep → its
  `last_frame`.
- **end_at**: concept/pose dep → that single png; animation dep → its
  `start_frame`.
- A pose's own **source image** (the thing FLUX edits) is the `from` dep's image:
  `concept` → `_concept.png`; pose → `_{from}/{resolved_dir}.png`.

### Transitive staleness (`outdated`)

`outdated` is the staleness predicate for a **complete** node — incompleteness is
a separate axis handled by `blocked`, never by `stale`.
```python
def outdated(manifest, root, char, ref, dir):
    kind = node_kind(manifest, ref)
    if kind == "concept":
        return False                       # not rendered; identity is captured in descendants' hashes
    if not complete(manifest, root, char, ref, dir):
        return False                       # incomplete ⇒ not "stale"; surfaced as blocked/missing elsewhere
    if read_rendered_hash(...) != prompt_hash(manifest, root, char, kind, ref, dir):
        return True                        # own merged-prompt hash drifted
    if kind == "pose":
        return outdated(manifest, root, char, from.ref, resolved_dir(from, dir))
    if kind == "animation":
        return any(outdated(... dep ...) for dep in (start_from, end_at) if dep)
    return False
```
A complete node is stale if its own merged-prompt hash drifted **or** any
ancestor is outdated. (Does not catch "ancestor re-rendered with an unchanged
prompt" — that needs provenance tracking, deferred.)

### `resolve_pose` / `resolve_animation`
Two entry points sharing the helpers above.

`resolve_pose(manifest, root, char, pose_id, dir)` →
```python
{
  "selectable": (dir in pose.directions) and from_complete,
  "blocked_by": [] or [{ "from": from_dep, "dir": resolved_dir }],
  "stale": bool,                       # from-source outdated
  "source_image": <concept/pose png> or None,
  "positive": merged_positive,
  "negative": merged_negative,
  "output_dir": f"{root}/{char}/_{pose_id}/{dir}",
  "meta": { "kind":"pose", "from": from_dep, "prompt_hash": ... },
}
```

`resolve_animation(manifest, root, char, anim_id, dir)` →
```python
{
  "selectable": (dir in anim.directions) and not blocked_by,
  "blocked_by": [ {slot: dep, "dir": resolved_dir}, … ],
  "stale": [ slot for slot in (start_from,end_at) if dep present and outdated(dep) ],
  "start_image": <dep.last_frame | dep png | None>,
  "end_image":   <dep.start_frame | dep png | None>,
  "positive": merged_positive,
  "negative": merged_negative,
  "output_dir": f"{root}/{char}/{anim_id}/{dir}",
  "meta": { "kind":"animation", "fps":…, "length":…, "loop":…, "prompt_hash": … },
}
```

### `status` (UI, per kind)
Evaluated against direct deps (`from` for a pose; `start_from`/`end_at` for an
animation):
```python
if any dep incomplete:              return "blocked"
if own output complete:             return "stale" if outdated(self) else "generated"
# deps complete, own output not yet generated
return "stale" if any(outdated(dep)) else "ready"
```
So a generated node whose own hash drifted is `stale`; a not-yet-generated node
whose inputs are outdated is `stale`; otherwise `generated`/`ready`. `stale`
stays selectable (amber), never blocks.

---

## 7. Nodes (ComfyUI surface)

Custom passthrough dict types: `ANIM_MANIFEST`, `ANIM_META`.

- **`AnimationManifestLoader`** — load + validate; build adjacency over poses +
  animations + concept; **detect cycles**; validate every `from`/`start_from`/
  `end_at` ref resolves to a known node; warn on non-`4n+1` animation lengths.
  Cache by `(path, mtime)`.
- **`CharacterPoseSelector`** — inputs: manifest, root, `character`/`pose`/
  `direction` (dynamic combos). Calls `resolve_pose`; loads the `from` source png
  to an IMAGE. Outputs: `source_image` (IMAGE, may be empty) + `has_source`
  (BOOLEAN), `positive`, `negative`, `output_dir`, `meta`.
- **`PoseFrameWriter`** — inputs: `image` (single IMAGE), `output_dir`, `meta`.
  Writes `{dir}.png`, then the `{dir}.json` sidecar last (atomic). Returns
  `output_dir` for re-scan fan-out.
- **`CharacterAnimationSelector`** — as in the original spec but with merged
  prompts and pose/animation/concept-typed anchors. Outputs `start_image`,
  `end_image` (+ `has_start`/`has_end` BOOLEAN), `positive`, `negative`,
  `output_dir`, `meta`. Raises if a selected pair is not `selectable`.
- **`AnimationFrameWriter`** — writes `frame_{:05d}.png`; loop closure
  (`loop_closure: "drop_last" | "duplicate_first"`); writes `meta.json` last
  (atomic) with `start_frame`/`last_frame` pointers and the merged `prompt_hash`.
  No `.complete`. Returns `output_dir`.
- **Concept intake** — a `ConceptImageWriter` node (or server upload route)
  writes `_concept.png` and an optional `_concept.json` identity layer.

### Server routes (`PromptServer.instance.routes`)
- `GET /anim_coord/characters?root=…`
- `GET /anim_coord/options?root=…&character=…` → every selectable
  `(pose|animation, direction)` with `status` and `blocked_by`.
- `GET /anim_coord/resolve?root=…&character=…&kind=…&id=…&direction=…` → full
  resolve incl. `source_preview` (poses) or `start_preview`/`end_preview`
  (animations), each with a cache-busted `url` and a `stale` flag.
- `GET /anim_coord/frame?path=…&v=…` → streams a PNG. **Security: resolve
  `path` against `root`; reject `..`, symlinks, absolute escapes; 404 anything
  outside `{root}`.** `v` = dep `prompt_hash` or mtime.

### Frontend extension (`WEB_DIRECTORY`)
- Character change → fetch `/options`, repopulate `pose`/`animation` +
  `direction` combos with status glyphs (`✓ generated`, `○ ready`, `⨯ blocked`,
  `▲ stale`).
- Selection change → fetch `/resolve`, draw the source thumbnail (poses) or the
  two FFLF anchor thumbnails (animations); tint amber when `stale`.
- After a writer run → re-fetch `/options` so newly unlocked nodes appear.

---

## 8. Build order & acceptance

Gate each step on its acceptance test.

1. **`manifest.py`** — load + validate + ref typing + cycle detect. *Accept:*
   rejects a bad `ref` or a cycle; warns on non-`4n+1` length.
2. **`resolve.py`** — `merge_layers`, merged-prompt hashing, completeness,
   anchors, `outdated` (transitive), `resolve_pose`/`resolve_animation`,
   `status` — with unit tests against a fixture tree. *Accept (TDD, before
   nodes):*
   - With only `_concept.png`: `base@E` pose is `ready`; blocked if concept
     missing.
   - Generate `base@E` → `fighting_stance@E` pose `ready`; generate it →
     `fighting_stance_idle@E` `ready`.
   - Generate idle → `entry@E`, `exit@E`, `punch@E`, … `ready`. Anchors:
     `punch.start_image` = idle `last_frame`, `punch.end_image` = idle
     `start_frame`; `entry.start_image` = `_base/E.png`, `entry.end_image` =
     idle `start_frame`; `exit.start_image` = idle `last_frame`,
     `exit.end_image` = `_base/E.png`.
   - Edit the `base` pose prompt → `fighting_stance`, `idle`, and all downstream
     animations report `stale` (transitive) but stay selectable.
3. **Pose nodes** — `CharacterPoseSelector` + `PoseFrameWriter` + concept
   intake. *Accept:* writing `_base/E.png` + sidecar flips `fighting_stance@E`
   to `ready`.
4. **Animation nodes** — `CharacterAnimationSelector` + `AnimationFrameWriter`.
   *Accept:* writing `fighting_stance_idle/E` flips `punch@E` to `ready`.
5. **Server routes** incl. `/frame` path-traversal hardening + concept upload.
6. **Frontend** dynamic combos, status glyphs, source/dual previews, amber
   stale, auto-refresh on write.

**End-to-end accept:** with only `_concept.png` present, the only selectable
node is `base@{E,SE,S}`. Walk the chain: generate `base` → `fighting_stance`
unlocks → generate it → `fighting_stance_idle` unlocks → generate idle →
`entry`/`exit`/`punch`/`kick`/`headbutt`/`block` unlock and preview idle's guard
frame as anchors. Edit the `base` pose prompt → the whole stance subtree shows
amber `stale` but stays selectable.

---

## 9. Decisions locked during design

1. Drop `.complete`; the atomically-written-last `meta.json`/sidecar is the
   completion sentinel.
2. Staleness is **transitive-on-hash**; provenance ("ancestor re-rendered, same
   prompt") deferred.
3. Negatives have **no facial/global special-casing** — pure cascade, same
   merge as positives.
4. The concept identity prompt participates in the staleness graph (falls out of
   merged-prompt hashing).
5. Per-character identity lives in a `_concept.json` **sidecar** (graduate to a
   `characters/{name}.json` only if more per-character data appears).

## 10. Not in v1

- Provenance-based staleness (record consumed source hashes).
- Per-character manifest overrides (`characters/{name}.json`).
- Batch/queue mode; coverage-report node.

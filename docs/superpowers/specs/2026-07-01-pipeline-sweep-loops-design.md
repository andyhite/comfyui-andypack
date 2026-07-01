# Design: One-press sweep loops + sweep/target selectors

**Date:** 2026-07-01
**Status:** Draft for review
**Topic:** Make it easy to build a full character (poses + animations) end-to-end,
keeping the four-workflow shape but removing the manual "press Queue N times" grind.

## Problem

The pack already resolves and writes back a full character pipeline
(create → pose turnaround → animation → sprite export), disk-backed and
staleness-aware. But the *unit of interaction* today is "a graph you Queue
repeatedly": each Queue press renders one pose/clip, and you keep pressing until
the auto-selector raises. Producing a whole character means babysitting the
queue across dozens of presses per stage, twice (poses, then clips).

The goal is not to collapse or re-architect the pipeline. It is to make a
**new character easy**: fill a whole sweep in one press, and go back to
spot-fix individual cells that came out wrong without disturbing the good ones.

## Working model (the shape we're keeping)

Four workflows, matching how the user actually works:

1. **Create** — iterate a character reference until it's right (possibly for
   several characters) and persist it.
2. **Pose sweep** — fill every pose for a character: the common poses from the
   base manifest **plus** that character's extras from `character.json`. End
   state: all pose directories filled.
3. **Animation sweep** — same, for animations.
4. **Export** — pack finished clips into sheets + atlas metadata.

This is a deliberate decomposition along two real seams: **pipeline stage** and
**model family**. The second seam is physically forced — verified on the target
box (single 23.4 GB card): FLUX.2 Klein 9B (stills) and WAN 2.2 14B×2 (clips)
cannot co-reside, so ComfyUI must load one family, then the other. No single
mega-graph is possible; four workflows stay.

## What already works (do NOT rebuild)

Confirmed in code — the behaviors the user described are largely present:

- **Base manifest + per-character extras.** Both selectors resolve through
  `effective_manifest`, which overlays `character.json`'s `poses`/`animations`
  onto the base manifest. A sweep already covers common + character-specific
  entities. (`api.py`)
- **"Fill everything, leave the good ones alone."** `next_actionable` walks
  dependency order and only ever yields a cell that is missing or locally stale;
  a complete-and-fine cell is never re-emitted. (`api.py:467`)
- **Spot-fix one cell.** `CharacterPoseSelector` / `CharacterAnimationSelector`
  resolve an exact `id@direction` and do **not** gate on completeness — they
  re-resolve and the writer overwrites, touching nothing else. (`nodes.py:338`)
- **Staleness/provenance.** Editing a prompt re-stales only descendants.
- **Re-evaluation each run.** Every selector's `IS_CHANGED` returns `NaN`, so it
  re-reads disk on each execution — the prerequisite for looping.

So this is an **additive** change, not a rebuild.

## The one real gap: a loop

"Full set" today = press Queue N times. The fix is an in-graph loop so **one
press fills the sweep**. ComfyUI 0.26.2 (verified on the box) supports the
node-expansion / execution-inversion API a custom loop node needs, and no
generic loop node is installed (KJNodes' only "loop" is a video-seam decoder),
so the pack ships its own — as previously decided.

## Design

### Overview

Two changes, both additive to the resolve core (which stays torch-free and
ComfyUI-free):

1. A **generic While-loop bracket** (`SweepLoopOpen` / `SweepLoopClose`) built on
   node expansion, keyed on a `REMAINING` count so it stops cleanly at zero.
2. A **unified per-stage selector** with a `mode` of `sweep` or `target`,
   replacing the split Auto/Character selectors. One pose graph and one
   animation graph then handle both "fill all fast" and "fix this one."

Node-count delta is **net zero** (see Node inventory): four selectors collapse
to two, two loop nodes are added — the pack stays at 20.

### 1. The loop bracket

`SweepLoopOpen` → (body) → `SweepLoopClose`, category `andypack/Loop`.

- `SweepLoopOpen` emits a `SWEEP_FLOW` token threaded through the body to
  `SweepLoopClose`. This token is how the expansion engine identifies the
  enclosed subgraph to re-instantiate per iteration.
- `SweepLoopClose` takes the flow token plus a `REMAINING` (INT). If
  `REMAINING > 0`, it expands a fresh copy of the Open→Close subgraph for the
  next iteration; otherwise it terminates. Because every body node's
  `IS_CHANGED` is `NaN`, each fresh iteration re-reads disk and advances.

The body is the existing chain, unchanged in spirit:

```
SweepLoopOpen → PoseSweepSelector(mode) → PoseEditConditioning → KSampler
             → VAEDecode → PoseFrameWriter → SweepLoopClose
```

(Animation body: `AnimationSweepSelector` → `PainterFLF2V` → dual `KSampler` →
alpha → `AnimationFrameWriter`.)

**Why the continue-signal is post-render, not an upfront count.** Poses have
dependency depth: derived poses become actionable only after `base` renders.
Enumerating "all actionable now" at loop start would miss cells that unblock
mid-sweep, under-rendering the turnaround. So `REMAINING` must be recomputed
*after* each write. The writer is the natural place (it runs last and has just
mutated disk).

### 2. Writers emit `REMAINING`

`PoseFrameWriter` and `AnimationFrameWriter` gain a `REMAINING` (INT) output.
After writing, the writer recomputes `next_actionable(...)` for its stage/scope
and returns the count of cells still actionable. This wire feeds
`SweepLoopClose`. The writers already receive the bundle; the bundle is extended
to carry the **sweep context** it needs to recompute: `character`, `mode`,
scope flags (`skip_mirrored`, `include_base`, `category`), and — in `target`
mode — the target `id@direction`.

### 3. Unified `sweep`/`target` selector

`PoseSweepSelector` (replaces `AutoPoseSelector` + `CharacterPoseSelector`) and
`AnimationSweepSelector` (replaces the animation pair). One `mode` widget:

- **`sweep`** — emit `next_actionable` for the scope (honoring `include_base` /
  `category` / `skip_mirrored`). Drives a full fill.
- **`target`** — emit the exact `id@direction` named in the widgets, resolved
  regardless of completeness (force-regenerate). For spot-fixes.

The selector stamps the sweep context (mode, scope, target) into the emitted
bundle so the writer can compute the mode-aware `REMAINING`.

### 4. `target` mode runs exactly once (the wrinkle, resolved)

The tension: `target` force-redoes a cell that is already "complete," so a
completeness-based `REMAINING` would either loop forever or never start.

Resolution: **`REMAINING` is mode-aware.** In `target` mode the writer returns
`0` unconditionally — the loop runs its single iteration (the forced re-render)
and stops. In `sweep` mode `REMAINING` = live `next_actionable` count, so the
loop fills until the queue drains, then stops cleanly at the last real cell (no
raise, no red error).

Degenerate case: starting a `sweep` when nothing is actionable runs one
iteration whose selector raises the existing informative "none remain" error.
That is acceptable (rare, and the message is useful); clean zero-iteration entry
is a possible later refinement, not MVP.

### `AutoPoseSelector` `REMAINING`

Subsumed: the unified `PoseSweepSelector` provides the count semantics that
`AutoPoseSelector` lacked. No separate output needs bolting onto the old node
because the old node is replaced.

## Final shape per workflow

1. **Create** — unchanged in essence; optionally slimmed to "dial in reference →
   persist," since `base@SOUTH` gets rendered by the pose sweep anyway
   (`include_base`). Optional, low priority.
2. **Pose sweep** — `SweepLoopOpen → PoseSweepSelector(mode) → PoseEditConditioning
   → KSampler → VAEDecode → PoseFrameWriter → SweepLoopClose`. `mode=sweep`
   fills the turnaround in one press; `mode=target` spot-fixes one `pose@dir`.
3. **Animation sweep** — same pattern with the WAN/PainterFLF2V body and
   `AnimationSweepSelector`.
4. **Export** — wrap `AnimationSheetBuilder → AtlasMetadataWriter` in the same
   loop, iterating over all animations so one press packs every sheet. The loop
   condition is "unpacked animations remain," computed the same way.

## Node inventory (delta)

Removed: `AutoPoseSelector`, `CharacterPoseSelector`, `AutoAnimationSelector`,
`CharacterAnimationSelector` (4).
Added: `SweepLoopOpen`, `SweepLoopClose`, `PoseSweepSelector`,
`AnimationSweepSelector` (4).
**Net: 0.** Pack stays at 20 focused nodes. Writers gain one output each (not new
nodes).

## Testing

- **Resolve/api layer (pure, no ComfyUI):** `next_actionable` count semantics
  used for `REMAINING`; mode-aware count (target → 0); dependency-depth cases
  (derived unblocks after base) so the loop provably drains a multi-level
  turnaround. These are unit-testable without torch, as the core already is.
- **Bundle context round-trip:** selector stamps mode/scope/target; writer reads
  them back and computes the right `REMAINING`.
- **Loop bracket:** covered by a spike (below) plus a small integration graph run
  against the live instance (fill a 2-direction, 2-pose fixture; assert all four
  cells written in one press; assert a `target` run rewrites exactly one).

## Risks and the required spike

**Primary risk: the loop-node contract.** The exact Open/Close expansion
contract — how the `SWEEP_FLOW` token threads through an `OUTPUT_NODE` body, and
precisely where `REMAINING` is sampled so continuation reflects post-write
state — is the fiddly part of ComfyUI's execution-inversion model. Before
building the real nodes, spike a minimal `SweepLoopOpen`/`Close` pair around a
trivial body (a counter + a disk-write) on the 0.26.2 instance and confirm:
(a) the body's output node runs each iteration, (b) `IS_CHANGED=NaN` re-reads
per iteration, (c) termination on `REMAINING==0` is clean. If the token-threading
contract differs from the assumption here, the node boundaries may shift; the
*semantics* above (post-render recompute, mode-aware count) hold regardless.

**Secondary:** the existing example workflows reference the removed selectors, so
all four example graphs are rebuilt as part of this work (they were going to be
touched anyway).

## Non-goals

- No cross-character batch driver (user works one character at a time).
- No orchestration layer outside the graphs; the loop lives in the canvas.
- No change to the manifest schema, resolve semantics, FFLF wiring, alpha
  boundary, or disk layout.
- No merging of the FLUX and WAN stages (physically impossible on the hardware).

## Open questions

1. Slim workflow 1 (drop `base@SOUTH`), or leave it as a useful early preview?
2. Export loop: one graph that sweeps all animations, or keep per-clip export as
   the default and add the loop as an option?

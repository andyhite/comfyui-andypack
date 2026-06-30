# Template variables in prompts: `{identity_prompt}`, `{direction_prompt}`, `{direction_name}`

Amends the cascade defined in
`2026-06-29-cascading-pose-resolver-design.md` §4. Where the two disagree,
this document wins for prompt composition.

## Problem

The entity (pose/animation) prompt should be the **composition root**: the
author writes a self-contained template and decides *where* the character
identity and the per-direction text land. Today identity and the direction
layer are auto-merged as fixed cascade layers, so the author cannot place them.

## Model

A pose or animation has an optional `positive_prompt` / `negative_prompt`.
A direction (under an entity) and the character concept (`_concept.json`) also
have optional positive/negative prompts — but those are **inert on their own**.
They surface only when an entity (or global) prompt references them by variable.

**Order of operations** (per axis):

1. Merge `globals[kind]` + entity prompts, exactly as today
   (`merge_layers` for positive, `merge_negative` for negative). The direction
   layer and the identity layer are **no longer merged**.
2. Substitute the template variables in the merged text, resolved by **field
   context** (positive vs negative).

| Variable | positive context | negative context |
|---|---|---|
| `{identity_prompt}` | concept `positive_prompt` | concept `negative_prompt` |
| `{direction_prompt}` | direction `positive_prompt` | direction `negative_prompt` |
| `{direction_name}` | the direction's bare name (`EAST`) | same |

Variables resolve in **either** a global or an entity prompt, since
substitution runs on the merged result.

### Substitution semantics

- Substitution is applied **per-layer before the merge**. The result is
  identical to merge-then-substitute for the author's stated order, but a layer
  whose variables resolve to empty (e.g. a global that is only
  `{direction_prompt}` for a direction with no negative) is cleanly dropped
  instead of leaving a blank line or stray `, ,`.
- Replacement is **literal token substitution** (`str.replace`), not
  `str.format`: unknown `{...}` tokens and stray braces survive untouched; an
  absent/empty source expands to `""`.
- Negatives keep the existing term-list treatment after substitution: split on
  commas, dedupe case-insensitively (first occurrence wins), drop empty tokens.
  So an empty `{direction_prompt}` cannot leave a stray `, ,`, and an expanded
  `{identity_prompt}` that is itself a comma list dedupes against siblings.

## Implementation

A single chokepoint: `merged_prompts` in `andypack/resolve.py`.

```python
def substitute_variables(text, *, positive, identity, direction_layer, direction):
    if not text:
        return text
    field = "positive_prompt" if positive else "negative_prompt"
    ident = (identity.get(field) or "").strip()
    dprompt = (direction_layer.get(field) or "").strip()
    return (text.replace("{identity_prompt}", ident)
                .replace("{direction_prompt}", dprompt)
                .replace("{direction_name}", direction))
```

`merged_prompts` merges only `globals[kind]` + entity (positive via
`merge_layers`, negative via `merge_negative`), running each layer through
`substitute_variables` in the matching field context first.

## Consequences (free, from the chokepoint)

- **Staleness still works.** `compute_prompt_hash` calls `merged_prompts`, so
  the hash is over the substituted text. Editing identity, a direction prompt,
  or a global changes the hash of exactly the entities that reference (or
  merge) it — marking them `stale`. No special-casing.
- **Reports reflect final text.** `api.merged_prompt_rows` /
  `format_merged_prompts` show the compiled prompts.
- **No node changes.** Nodes build prompts only through `merged_prompts`. (The
  concept-intake node's `identity_positive` / `identity_negative` *inputs* are
  unrelated UI fields that write the `_concept.json` identity layer.)

## Out of scope (YAGNI)

No general templating engine and no further variables (character name, etc.)
for now — only the three tokens above.

## Acceptance

1. positive = merged `globals[kind].positive_prompt` + entity `positive_prompt`,
   with the direction layer **not** auto-appended.
2. `{direction_prompt}` in an entity (or global) positive injects the selected
   direction's `positive_prompt`; in a negative, its `negative_prompt`.
3. `{direction_name}` injects the bare direction name in both contexts.
4. `{identity_prompt}` injects concept positive in a positive field, concept
   negative in a negative field.
5. A negative `{direction_prompt}` that resolves empty leaves no stray `, ,`;
   an expanded `{identity_prompt}` comma list dedupes against sibling terms.
6. Variables referenced inside a `globals[kind]` prompt resolve too.
7. Unknown `{foo}` tokens and literal braces survive untouched; empty/absent
   sources expand to `""`.
8. Editing a referenced source (identity / direction / global) flips a rendered
   dependent that references it to `stale`.

## Docs to update with the code

- `CLAUDE.md` — cascade lines + variable note.
- `2026-06-29-cascading-pose-resolver-design.md` §4 layer stacks + hash note.
- `docs/animation-manifest-guide.md`.
- `examples/animations.json` — pose templates using the variables.

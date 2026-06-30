# Opt-in identity via `{identity_positive}` / `{identity_negative}`

Amends the cascade defined in
`2026-06-29-cascading-pose-resolver-design.md` §4. Where the two disagree,
this document wins for identity placement.

## Problem

Today the per-character identity layer (`_concept.json`'s `positive_prompt` /
`negative_prompt`) is **auto-merged as the top cascade layer** in
`merged_prompts` (`andypack/resolve.py`). The author cannot control *where* the
identity text lands in a prompt — it is always prepended (positives) or folded
into the front of the term list (negatives). For prompts like
`"a wide shot of <character> running through rain"`, the identity belongs mid-
sentence, not bolted on the front.

## Behavior change (intentional, breaking)

The identity layer is **removed from the automatic cascade**. Identity is now
**opt-in only**: its text appears solely where a prompt layer references it.

- Cascade becomes `globals[kind] → entity → entity.directions[dir]` — no
  identity layer.
- `{identity_positive}` in any layer expands to the character's identity
  `positive_prompt`.
- `{identity_negative}` in any layer expands to the character's identity
  `negative_prompt`.

A character that previously relied on auto-prepended identity will lose it from
every prompt until the manifest references the tokens. This is the intended
migration.

## Substitution semantics

Tokens are expanded **per-layer, before the merge** — not on the final merged
string. This matters for negatives: `merge_negative` splits on commas and
dedupes terms, so `{identity_negative}` expanding to `"blurry, deformed"`
*before* the merge lets those terms participate in dedupe like any other.
Expanding after the merge would treat the whole token as one opaque term and
skip dedupe. Positives are equivalent either way; per-layer keeps it uniform.

Replacement is **literal token substitution** (`str.replace`), not
`str.format`:
- Unknown tokens and stray `{`/`}` in prompts are left untouched (no crash).
- An absent or empty identity field expands the token to `""`.

## Implementation

A single chokepoint: `merged_prompts` in `andypack/resolve.py`.

```python
def substitute_identity(text: Optional[str], identity: dict) -> Optional[str]:
    if not text:
        return text
    pos = (identity.get("positive_prompt") or "").strip()
    neg = (identity.get("negative_prompt") or "").strip()
    return text.replace("{identity_positive}", pos).replace("{identity_negative}", neg)
```

`merged_prompts` drops the identity layer from both `merge_layers` /
`merge_negative` calls and maps each remaining layer through
`substitute_identity(..., identity)` first.

## Consequences (free, from the chokepoint)

- **Staleness still works.** `compute_prompt_hash` calls `merged_prompts`, so
  the hash is over the substituted text. Editing a character's identity changes
  every dependent's merged prompt → hash → marks them `stale`. No special-casing.
- **Reports reflect final text.** `api.merged_prompt_rows` /
  `format_merged_prompts` show the expanded prompts.
- **No node changes.** Nodes build prompts only through `merged_prompts`.

## Out of scope (YAGNI)

No general templating engine, no additional variables, no per-layer escape
syntax — only the two literal tokens.

## Acceptance

1. A layer containing `{identity_positive}` renders with the identity positive
   text spliced in place; identity is **not** also prepended.
2. With no token referenced anywhere, the merged positive/negative contain
   **none** of the identity text.
3. `{identity_negative}` expanding to a comma list is deduped against sibling
   negative terms.
4. Empty/absent identity → token expands to `""`, no stray crash, no `{...}`
   left in output for the known tokens.
5. Unknown `{foo}` tokens and literal braces survive untouched.
6. Editing identity `positive_prompt` flips a rendered dependent that
   references `{identity_positive}` to `stale`; a dependent that does **not**
   reference it is unaffected.

## Docs to update with the code

- `CLAUDE.md` — cascade lines and the negative special-casing note.
- `2026-06-29-cascading-pose-resolver-design.md` §4 layer stacks + hash note.
- `docs/animation-manifest-guide.md`.
- `examples/animations.json` — add a usage example.

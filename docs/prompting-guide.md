# Prompting & generation guide — FLUX.2 Klein (poses) + Wan 2.2 14B i2v (animations)

Authoritative reference for how this pack's prompt templates and recommended
ComfyUI graphs are structured. Synthesized from BFL FLUX.2 docs, the Wan 2.2
model card / ComfyUI docs, and community guides (late 2025 / 2026). Where a value
is community-sourced rather than vendor-confirmed it is flagged inline.

---

## 1. Wan 2.2 14B i2v — first-last-frame (FFLF) is natively supported

The pack's `end_at = end frame` architecture is **correct and natively supported**.
There is **no** separate FLF checkpoint for Wan 2.2 14B (unlike Wan 2.1, which
shipped `wan2.1_flf2v_720p_14B`) — you reuse the standard i2v models.

**Node: `WanFirstLastFrameToVideo`** (core, `comfy_extras/nodes_wan.py`).
- Required: `positive`, `negative` (CONDITIONING), `vae`, `width`, `height`,
  `length`, `batch_size`.
- Optional: `start_image` (IMAGE), `end_image` (IMAGE),
  `clip_vision_start_image`, `clip_vision_end_image`.
- Outputs: `positive`, `negative` (CONDITIONING), `latent` (LATENT).

**Resolver mapping:** `start_from` → `start_image`; `end_at` → `end_image`. With
no `end_at`, leave `end_image` unconnected — the node degrades to plain i2v, so a
single node handles both cases. The generated clip's literal final frame equals
the supplied `end_image`, which is exactly why the FFLF cross-wiring
(`start_from` → dep LAST frame, `end_at` → dep FIRST frame, loop when
`start_image == end_image` then drop the duplicate final frame) is sound. **Keep
all of it.** The pack now ships **Wan Animation Conditioning**, a one-node
wrapper that performs this exact wiring automatically — including the
leave-`end_image`-unconnected rule for a non-FFLF clip — from an
`ANIM_ANIMATION` bundle; manual `WanFirstLastFrameToVideo` wiring (below)
remains valid for graphs that need more control.

**Models** (`ComfyUI/models/diffusion_models/`): `wan2.2_i2v_high_noise_14B` +
`wan2.2_i2v_low_noise_14B` (fp16 or fp8_scaled); text encoder
`umt5_xxl_fp8_e4m3fn_scaled.safetensors` (CLIPLoader); VAE
`wan_2.1_vae.safetensors` (the 14B path uses the Wan **2.1** VAE — not the 2.2 5B
VAE). CLIP-vision sockets are vestigial from the 2.1 FLF model; leave unconnected.

**Loop color drift caveat:** when `start_image == end_image`, the low-noise expert
can drift color/contrast and make the seam visible. Mitigate: keep loop clips
≤49 frames, keep `shift` moderate (3.0 @ 480p), optional output color-match pass
— the Animation Frame Writer's `loop_color_match` flag is the built-in version
of that pass: it ramps a per-channel color match toward the first frame across
a derived loop clip, applied only when the resolver derived `loop` (a no-op on
non-loop clips).

---

## 2. FLUX.2 Klein — pose-edit prompt structure

Core rules for every Klein edit prompt:
- **Prose, not tags.** Flowing sentences. Front-load the subject (earlier words
  weigh more).
- **40–70 words** (hard cap 512 tokens); >100 words is a failure mode.
- **State PRESERVE + CHANGE explicitly:** *Keep [X] identical to the first image
  while [change Y].*
- **NO negative prompts.** FLUX.2 is guidance-distilled — no negative path.
  Negatives can backfire. Convert every negative into affirmative description
  ("sharp focus" not "no blur"; name the real colors to hold them).
- **Reference order matters** — earlier images get more attention. Refer to images
  only as "the first image" / "the second image" and give each an explicit ROLE.

### 2a. Base pose (multi-reference: character = image 1, gray manikin = image 2)

> The same character from the first image, redrawn to exactly match the body pose
> and orientation of the gray articulated mannequin in the second image. Use the
> mannequin only for pose and camera orientation — do not copy its gray color or
> featureless surface. Keep the character's face, hairstyle, colors, outfit, and
> design identical to the first image: {character_prompt}. {direction_prompt}.
> Full-body shot, plain neutral background, flat even studio lighting, clean
> character-turnaround sheet style.

Scope image 2 to **pose/orientation only** so the manikin's gray doesn't bleed.

### 2b. Re-posing an anchor pose (single reference)

> The same character from the reference image, repositioned into {pose description}
> while keeping the same {direction_name}-facing orientation. Keep the face,
> hairstyle, colors, outfit, and all design details identical to the reference
> image: {character_prompt}. {direction_prompt}. Full-body shot, plain neutral
> background, flat even studio lighting.

### Per-direction VIEW phrase (name the camera explicitly to fight front/3-4 bias)

| Dir | Phrase |
|-----|--------|
| SOUTH (front) | viewed directly from the front, facing the camera |
| SOUTH_EAST | viewed from the front-right three-quarter angle |
| EAST (right) | in full right-side profile, only one eye visible |
| NORTH_EAST | viewed from the back-right three-quarter angle, face turned away |
| NORTH (back) | viewed directly from behind — back of the head and hair to camera, no face visible |
| NORTH_WEST | viewed from the back-left three-quarter angle, face turned away |
| WEST (left) | in full left-side profile, only one eye visible |
| SOUTH_WEST | viewed from the front-left three-quarter angle |

Back views (NORTH/NE/NW): describe back-of-head/hair affirmatively (Klein has no
negative to suppress a face). Profiles (E/W): "only one eye visible."

### ComfyUI settings
| Variant | Steps | Guidance | Sampler | Scheduler |
|---|---|---|---|---|
| Klein distilled (4-step) | 4 | 1.0–1.5 | Euler | Simple |
| Klein/dev base | 20–24 | 3.5–5.0 *(community-sourced — verify)* | Euler | Simple |

Never use Euler Ancestral on the distilled model. Klein has no prompt upsampling.
Multi-ref wiring: each reference `ImageScaleToTotalPixels` (~1MP) → `ReferenceLatent`,
chained — keep both refs at the **same** total-pixel target so neither dominates.

---

## 3. Wan 2.2 i2v — animation motion prompt structure

Structure (official Wan2.2 order): **Subject + Scene + Motion + Aesthetic
(camera + lighting) + Stylization.** For i2v the start frame fixes appearance —
spend the prompt on **MOTION + CAMERA**, not static description. ~80–120 words,
motion first and longest, strong concrete verbs, a COMPLETE arc.

> [subject anchor], [primary action — a strong verb describing a complete arc that
> returns to start], [secondary limb/body detail], fixed camera / locked-off shot,
> plain static background, even flat lighting, clean crisp silhouette, game sprite
> style.

Rules: always add "fixed camera / static camera" for sprites (Wan drifts
otherwise); keep background explicitly static; for a loop describe a full cycle
ending where it began; don't re-describe static appearance; don't request fps;
don't exceed ~120 words.

**Standard negative block (use verbatim; only effective at CFG > 1):**

> Bright tones, overexposed, static, blurred details, subtitles, style, works,
> paintings, images, static, overall gray, worst quality, low quality, JPEG
> compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly
> drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture,
> messy background, three legs, many people in the background, walking backwards

Keep "static / still picture / walking backwards" — they fight frozen/reversed clips.

**Frames / fps / resolution:** native **16 fps**; frames must be **4n+1**
(81≈5s, 49≈3s, 33≈2s, 25≈1.5s, 17≈1s — use 33–49 for 2–3s clips); buckets
832×480 (`shift 3.0`) or 1280×720 (`shift 5.0`). 480p is plenty for one small
character; interpolate to higher fps in post (RIFE/FILM).

**Samplers:**
| Path | Steps | Split | CFG | Sampler | Sched | Shift |
|---|---|---|---|---|---|---|
| Final quality (negatives work) | 20–30 | ~half/half | 3.5–5.0 | Euler | Simple | 3 (480p) / 5 (720p) |
| Fast iterate (4-step Lightning LoRA on both experts) | 4 | 2+2 | 1.0 | Euler | Simple | 5 |

Dual-expert MoE: high-noise expert early steps, low-noise late, switch at
boundary ≈0.9. **At CFG=1 the negative prompt is ignored** — use the multi-step
path whenever negatives matter.

**Seamless loop:** same image to both `start_image` and `end_image`; prompt a
motion arc that returns to start; drop the duplicated final frame (the writer
already does this when the resolver derives `loop`).

---

## 4. Directional turnaround strategy

- **Mirroring** (E→W, NE→NW, SE→SW) is sound **only for bilaterally symmetric
  designs.** It breaks on one-handed props, slung items, eyepatch/scar, hair
  parting, faction logos/text (mirror-reversed), handedness. Make it an opt-in
  per-character flag; default to native generation.
- Generate the 5 unique directions (S, SE, E, NE, N) natively, mirror the rest —
  only when flagged symmetric.
- Vary ONLY the VIEW phrase across panels; keep the character block identical.
- Failure modes: face on back views (describe back-of-head; drive manikin
  orientation), wrong eye count on profiles ("only one eye visible"). Klein has no
  negative — rely on affirmative phrasing + the manikin.
- Anchor-then-interpolate is validated (Sprite Sheet Diffusion, arXiv 2412.03685):
  clean key poses first, short interpolated spans, regenerate (don't append) a
  span when shortening — matches the pack's `clear_frames` invariant.

---

## 5. Engine-ready sprite export tips

**Alpha cutouts (bring your own BG removal).** To produce transparent sprites
directly usable in a game engine, run a background-removal node upstream of the
Pose Frame Writer or Animation Frame Writer and connect its mask to the writer's
optional **MASK** input (or supply a pre-composited RGBA image). The writers
preserve the alpha channel at the disk boundary and record `has_alpha` in the
sidecar/meta; everything inside the graph stays 3-channel (RGB). Chain into
Sprite Trim & Pivot → Spritesheet Packer for the full pipeline.

**Palette lock for pixel-art consistency.** If your character uses a restricted
palette (pixel-art, limited-color, or hand-painted look), apply **Palette Quantize
& Lock** after background removal and before packing. This forces every direction
and animation to the same quantized color table, preventing per-direction color
drift that would be visually obvious in a sprite sheet.

---

## 6. ComfyUI sidebar tab API (CRUD manifest panel)

> Re-verify signatures against the installed frontend version (the API has churned).

```js
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

app.registerExtension({
  name: "andypack.manifestPanel",
  async setup() {
    app.extensionManager.registerSidebarTab({
      id: "andypackManifest",
      icon: "pi pi-list",          // PrimeIcons "pi pi-*" or Material "mdi mdi-*"
      title: "Andypack",
      tooltip: "Manifest & character manager",
      type: "custom",
      render: (el) => { /* el is an HTMLElement you own — build DOM into it */ },
    });
  },
});
```

- HTTP CRUD via `api.fetchApi(route, { method, headers, body })` (prefixes the
  base path, carries auth). Routes return JSON only and take no client paths —
  the panel sends manifest *data*, never filesystem paths.
- Toasts: `app.extensionManager.toast.add({ severity, summary, detail, life })`
  (`severity`: success | info | warn | error).
- Dialogs: `app.extensionManager.dialog.prompt({ title, message })` /
  `.confirm({ title, message })`; for rich forms, render your own DOM in the
  sidebar element.

---

## Open questions / flagged uncertainties
- Base-Klein CFG/steps (3.5–5.0 / 20–24) are community-sourced — verify on the
  actual checkpoint.
- Manikin gray-color bleed is plausible but undocumented — mitigated by scoping
  image 2 to "pose only" + naming real colors; test empirically.
- Wan loop color drift (start==end) has no canonical fix — color-match is a
  workaround.
- ComfyUI sidebar/dialog/toast method names must be re-confirmed against the
  user's installed frontend version.

"""Torch/PIL bridge for ComfyUI IMAGE tensors. Isolated so the rest of the pack stays pure."""

from __future__ import annotations

import base64
import math
import os
import tempfile
from io import BytesIO
from typing import Optional

import numpy as np
import torch
from PIL import Image


# Background the alpha channel is composited over when flattening RGBA/LA art to
# the 3-channel RGB that ComfyUI IMAGE tensors carry. White suits character art on
# a transparent background; `.convert("RGB")` alone would composite onto black.
_MATTE = (255, 255, 255)


def thumbnail_data_uri(path: str, max_px: int = 96) -> str:
    """Open *path*, shrink to fit ``max_px`` × ``max_px`` (preserving aspect
    ratio), and return a ``data:image/png;base64,...`` URI string.

    Raises/propagates if the file can't be opened — the caller guards existence.
    """
    buf = BytesIO()
    with Image.open(path) as img:
        img.thumbnail((max_px, max_px))
        img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def load_image_tensor(path: str, keep_alpha: bool = False) -> torch.Tensor:
    """Load a PNG into a ComfyUI IMAGE tensor [1, H, W, C] float32 in [0, 1].

    When ``keep_alpha`` is False (the default) images with transparency are
    alpha-composited over a white matte and returned as 3-ch RGB — unchanged
    from the original behavior.  When ``keep_alpha`` is True, a 4-ch RGBA
    tensor is returned; RGB-only PNGs get a full alpha channel (α=1).
    """
    with Image.open(path) as img:
        if keep_alpha:
            rgba = img.convert("RGBA")
            arr = np.asarray(rgba, dtype=np.float32) / 255.0
        else:
            rgb = _flatten_to_rgb(img)
            arr = np.asarray(rgb, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def _flatten_to_rgb(img: "Image.Image") -> "Image.Image":
    # Paletted images may carry transparency in their palette — promote to RGBA.
    if img.mode == "P":
        img = img.convert("RGBA") if "transparency" in img.info else img.convert("RGB")
    if img.mode in ("RGBA", "LA"):
        rgba = img.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (*_MATTE, 255))
        return Image.alpha_composite(bg, rgba).convert("RGB")
    return img.convert("RGB")


def _alpha_from_mask(mask: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """Resize a ComfyUI MASK [B,H,W] to (h, w) and return the first frame [H,W]."""
    m = mask if mask.dim() == 3 else mask.unsqueeze(0)
    if m.shape[1] != h or m.shape[2] != w:
        m = torch.nn.functional.interpolate(
            m.unsqueeze(1), size=(h, w), mode="bilinear", align_corners=False
        ).squeeze(1)
    return m[0].clamp(0.0, 1.0)


def to_rgba(image: torch.Tensor, mask: "torch.Tensor | None" = None) -> torch.Tensor:
    """Return a [1,H,W,4] RGBA tensor from the first frame of an IMAGE batch.

    - If *mask* ([B,H,W]) is given, it becomes the alpha channel (resized if needed).
    - Else if *image* is already 4-ch, the existing alpha is kept.
    - Else alpha is set to 1 (fully opaque).
    """
    frame = image[0] if image.dim() == 4 else image
    h = int(frame.shape[0])
    w = int(frame.shape[1])
    rgb = frame[..., :3]
    if mask is not None:
        a = _alpha_from_mask(mask, h, w).unsqueeze(-1)
    elif frame.shape[-1] == 4:
        a = frame[..., 3:4]
    else:
        a = torch.ones((h, w, 1), dtype=frame.dtype)
    return torch.cat([rgb, a], dim=-1).unsqueeze(0)


def alpha_bbox(
    image: torch.Tensor,
    threshold: float = 0.03,
) -> "tuple[int, int, int, int] | None":
    """Return (left, top, right, bottom) of pixels where alpha >= threshold.

    For a 3-ch image the entire rectangle is returned.  Returns None when the
    image is fully transparent (no pixels at or above the threshold).
    """
    frame = image[0] if image.dim() == 4 else image
    if frame.shape[-1] != 4:
        return (0, 0, int(frame.shape[1]), int(frame.shape[0]))
    a = frame[..., 3]
    ys, xs = torch.where(a >= threshold)
    if ys.numel() == 0:
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def save_image_png(
    image: torch.Tensor,
    path: str,
    mask: "torch.Tensor | None" = None,
) -> None:
    """Atomically save the first batch item of an IMAGE tensor as a PNG.

    Writes RGBA when *mask* is provided or when *image* is already 4-ch;
    otherwise writes RGB (original behavior, fully backward-compatible).
    """
    frame = image[0] if image.dim() == 4 else image
    has_alpha = mask is not None or frame.shape[-1] == 4
    if has_alpha:
        rgba = to_rgba(image, mask)[0]
        arr = (rgba.clamp(0.0, 1.0).cpu().numpy() * 255.0).round().astype(np.uint8)
    else:
        arr = (frame[..., :3].clamp(0.0, 1.0).cpu().numpy() * 255.0).round().astype(np.uint8)
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, suffix=".png.tmp")
    os.close(fd)
    try:
        Image.fromarray(arr).save(tmp, format="PNG")
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def pad_to(tensor: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Zero-pad a [1, H, W, C] tensor to (height, width), top-left anchored.

    New pixels are filled with zeros (transparent for RGBA).  Do NOT use for
    resizing — bilinear resize is in ``_resize_batch``; this is strictly for
    making tensors with DIFFERENT sizes uniform before ``torch.cat`` so no
    pixel content is distorted.  A no-op when the tensor already matches.
    """
    _, h, w, c = tensor.shape
    if h == height and w == width:
        return tensor
    out = torch.zeros((1, height, width, c), dtype=tensor.dtype)
    out[:, :h, :w, :] = tensor
    return out


def empty_image() -> torch.Tensor:
    """A 1x1 black image — the 'no anchor present' sentinel."""
    return torch.zeros((1, 1, 1, 3), dtype=torch.float32)


def is_empty(image: torch.Tensor) -> bool:
    """True when `image` is the empty sentinel (no real pixels) — e.g. what
    assemble_playback returns when no segment had readable frames. Real frames
    carry their source PNG dimensions, so a >1x1 batch is never the sentinel."""
    return image.shape[0] == 0 or (image.shape[1] <= 1 and image.shape[2] <= 1)


def _resize_batch(batch: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Bilinear-resize an IMAGE batch [N, H, W, C] to (height, width); a no-op when
    it already matches."""
    if batch.shape[1] == height and batch.shape[2] == width:
        return batch
    nchw = batch.permute(0, 3, 1, 2)
    resized = torch.nn.functional.interpolate(
        nchw, size=(height, width), mode="bilinear", align_corners=False
    )
    return resized.permute(0, 2, 3, 1).contiguous()


def match_color_ramp(
    frames: torch.Tensor, reference: torch.Tensor, strength: float = 1.0
) -> torch.Tensor:
    """Per-channel mean/std color match of each frame toward `reference`, ramped
    linearly from 0 at frame 0 to `strength` at the final frame. Hides the
    loop-seam color drift Wan's low-noise expert introduces on start==end clips
    (see docs/prompting-guide.md) without touching the clip's opening frames.
    Only the RGB channels are matched; alpha passes through untouched."""
    n = int(frames.shape[0])
    if n <= 1:
        return frames
    ref = reference[0] if reference.dim() == 4 else reference
    ref_rgb = ref[..., :3]
    ref_mean = ref_rgb.mean(dim=(0, 1))
    ref_std = ref_rgb.std(dim=(0, 1)).clamp_min(1e-6)
    out = frames.clone()
    for i in range(n):
        weight = float(strength) * (i / (n - 1))
        if weight <= 0.0:
            continue
        f = frames[i, ..., :3]
        mean = f.mean(dim=(0, 1))
        std = f.std(dim=(0, 1)).clamp_min(1e-6)
        matched = (f - mean) / std * ref_std + ref_mean
        out[i, ..., :3] = ((1.0 - weight) * f + weight * matched).clamp(0.0, 1.0)
    return out


def retime_batch(
    frames: torch.Tensor, target: int, mode: str = "resample"
) -> torch.Tensor:
    """Retime an IMAGE batch [N, H, W, C] to a target frame count.

    Args:
        frames: Tensor of shape [N, H, W, C].
        target: Desired output frame count; clamped to >=1.
        mode: One of ``"resample"``, ``"trim"``, or ``"pad_hold"``.

            - ``resample``: Maps each output index ``i`` to source index
              ``round(i * (N-1) / (target-1))`` (when target > 1; a single
              output takes frame 0). Handles target > N by duplicating frames
              and target < N by dropping frames, uniformly.
            - ``trim``: Returns the first ``min(target, N)`` frames.  When
              target > N all N frames are returned; frames cannot be invented
              by trimming.
            - ``pad_hold``: Returns the first ``target`` frames when
              target <= N; otherwise appends the last frame repeated until
              the batch reaches ``target``.

    Returns:
        Retimed tensor [target', H, W, C].  For ``trim`` when target > N,
        target' equals N (not target).
    """
    n = int(frames.shape[0])
    target = max(1, int(target))
    if n == 0:
        return frames
    if mode == "resample":
        if target == 1:
            indices = [0]
        else:
            indices = [round(i * (n - 1) / (target - 1)) for i in range(target)]
        return torch.stack([frames[idx] for idx in indices])
    if mode == "trim":
        return frames[:min(target, n)]
    if mode == "pad_hold":
        if target <= n:
            return frames[:target]
        pad_count = target - n
        return torch.cat([frames, frames[-1:].repeat(pad_count, 1, 1, 1)])
    raise ValueError(f"retime_batch: unknown mode {mode!r}")


def _pil_frames(frames: torch.Tensor) -> list["Image.Image"]:
    """An IMAGE batch as PIL frames — RGBA when the batch carries 4 channels,
    RGB otherwise. The disk boundary is where alpha materializes; everything
    upstream in the graph stays 3-channel."""
    arr = (frames.clamp(0.0, 1.0).cpu().numpy() * 255.0).round().astype(np.uint8)
    mode = "RGBA" if int(frames.shape[-1]) == 4 else "RGB"
    return [Image.fromarray(a, mode=mode) for a in arr]


def save_animated_webp(
    frames: torch.Tensor, path: str, fps: int, loop: bool = True
) -> None:
    """Encode an IMAGE batch [N, H, W, C] as an animated WEBP at `path`, played at
    `fps`. RGBA batches keep their alpha channel. A single frame writes a still
    WEBP. `loop` True = repeat forever (loop count 0), False = play once."""
    pil = _pil_frames(frames)
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    duration = int(round(1000.0 / max(int(fps), 1)))
    pil[0].save(
        path, format="WEBP", save_all=True, append_images=pil[1:],
        duration=duration, loop=0 if loop else 1, quality=80, method=4,
    )


def _gif_frame(img: "Image.Image") -> "tuple[Image.Image, Optional[int]]":
    """Quantize one frame for GIF. RGBA frames reserve palette index 255 for
    fully-transparent pixels (GIF has 1-bit transparency); returns the paletted
    frame and the transparency index (None for opaque frames)."""
    if img.mode != "RGBA":
        return img.convert("RGB").convert("P", palette=Image.ADAPTIVE), None  # type: ignore[attr-defined]
    alpha = img.getchannel("A")
    p = img.convert("RGB").convert("P", palette=Image.ADAPTIVE, colors=255)  # type: ignore[attr-defined]
    transparent = alpha.point(lambda a: 255 if a < 128 else 0)
    p.paste(255, transparent)
    return p, 255


def save_animated_gif(
    frames: torch.Tensor, path: str, fps: int, loop: bool = True
) -> None:
    """Encode an IMAGE batch as an animated GIF. RGBA batches get palette
    transparency (alpha < 0.5 -> fully transparent; GIF has no partial alpha).
    `loop` True = repeat forever, False = play once."""
    quantized = [_gif_frame(f) for f in _pil_frames(frames)]
    pil = [q for q, _t in quantized]
    transparency = quantized[0][1]
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    duration_ms = int(round(1000.0 / max(int(fps), 1)))
    kwargs: dict = {}
    if transparency is not None:
        kwargs["transparency"] = transparency
    pil[0].save(
        path, format="GIF", save_all=True, append_images=pil[1:],
        duration=duration_ms, loop=0 if loop else 1, disposal=2, **kwargs,
    )


def save_animated_apng(
    frames: torch.Tensor, path: str, fps: int, loop: bool = True
) -> None:
    """Encode an IMAGE batch as an animated PNG (APNG). RGBA batches keep their
    alpha channel. `loop` True = repeat forever, False = play once."""
    pil = _pil_frames(frames)
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    duration = int(round(1000.0 / max(int(fps), 1)))
    pil[0].save(
        path, format="PNG", save_all=True, append_images=pil[1:],
        duration=duration, loop=0 if loop else 1,
    )


def onion_skin(
    frames: torch.Tensor,
    prev: int,
    next: int,  # noqa: A002 — intentionally shadows builtin to match public API
    opacity: float,
) -> torch.Tensor:
    """Composite ghosted neighbor frames for animator QA.

    For each frame *i*, compute a ghost = mean of up-to-``prev`` preceding frames
    and up-to-``next`` following frames.  The output is a linear blend::

        out[i] = (1 - opacity) * frame[i] + opacity * ghost[i]

    When frame *i* has no neighbors (edges, or prev=next=0), ghost equals frame[i]
    so the output is unchanged.

    Args:
        frames:  IMAGE batch [N, H, W, C] float in [0, 1].
        prev:    Number of preceding frames to include in the ghost.
        next:    Number of following frames to include in the ghost.
        opacity: Blend weight of the ghost (0 = no ghost, 1 = ghost only).

    Returns:
        Blended tensor [N, H, W, C] clamped to [0, 1].
    """
    n = int(frames.shape[0])
    if n == 0 or (int(prev) == 0 and int(next) == 0):
        return frames
    out = frames.clone()
    opacity_f = float(opacity)
    prev_n = int(prev)
    next_n = int(next)
    for i in range(n):
        lo = max(0, i - prev_n)
        hi = min(n, i + next_n + 1)
        neighbor_indices = [j for j in range(lo, hi) if j != i]
        if not neighbor_indices:
            continue
        ghost = torch.stack([frames[j] for j in neighbor_indices]).mean(dim=0)
        out[i] = (1.0 - opacity_f) * frames[i] + opacity_f * ghost
    return out.clamp(0.0, 1.0)


def contact_sheet(
    tiles: list,
    columns: int,
    cell: "Optional[tuple[int, int]]" = None,
    labels: "Optional[list[str]]" = None,
) -> torch.Tensor:
    """Composite IMAGE tiles into a grid, with mid-gray placeholders for None slots.

    Args:
        tiles: list of IMAGE tensors ``[1, H, W, C]`` or ``None`` for unrendered
            slots. None entries become a neutral mid-gray placeholder cell.
        columns: number of grid columns (>=1). Rows = ceil(len(tiles)/columns).
        cell: optional ``(cell_w, cell_h)`` target size per tile in pixels.
            When None, the cell size is the maximum H and W across all non-None
            tiles, falling back to (64, 64) when every tile is None.
        labels: optional per-tile caption strings, drawn at each cell's top-left
            (white with a black stroke). Extra labels beyond len(tiles) are ignored.

    Returns:
        IMAGE tensor ``[1, rows*cell_h, columns*cell_w, 3]``.
    """
    n = len(tiles)
    if n == 0:
        return torch.zeros((1, 1, 1, 3), dtype=torch.float32)

    columns = max(1, int(columns))
    rows = math.ceil(n / columns)

    if cell is not None:
        cell_w, cell_h = int(cell[0]), int(cell[1])
    else:
        non_none = [t for t in tiles if t is not None]
        if non_none:
            cell_h = max(int(t.shape[1]) for t in non_none)
            cell_w = max(int(t.shape[2]) for t in non_none)
        else:
            cell_h, cell_w = 64, 64

    placeholder = torch.full((1, cell_h, cell_w, 3), 0.5, dtype=torch.float32)
    sheet = torch.zeros((1, rows * cell_h, columns * cell_w, 3), dtype=torch.float32)

    for idx, tile in enumerate(tiles):
        row = idx // columns
        col = idx % columns
        y = row * cell_h
        x = col * cell_w
        if tile is None:
            sheet[:, y:y + cell_h, x:x + cell_w, :] = placeholder
        else:
            resized = _resize_batch(tile[..., :3], cell_h, cell_w)
            sheet[:, y:y + cell_h, x:x + cell_w, :] = resized

    if labels:
        from PIL import ImageDraw

        arr = (sheet[0].clamp(0.0, 1.0).cpu().numpy() * 255.0).round().astype(np.uint8)
        pil = Image.fromarray(arr, mode="RGB")
        draw = ImageDraw.Draw(pil)
        for idx in range(min(n, len(labels))):
            row, col = divmod(idx, columns)
            draw.text(
                (col * cell_w + 4, row * cell_h + 4), str(labels[idx]),
                fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0),
            )
        sheet = torch.from_numpy(
            np.asarray(pil, dtype=np.float32) / 255.0
        ).unsqueeze(0)
    return sheet

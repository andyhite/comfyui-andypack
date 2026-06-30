"""Sprite trim, pivot, sheet-packing, and palette helpers for game asset export.

No ComfyUI or PromptServer imports — torch/numpy/PIL only.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch import Tensor

from andypack import images

# 4×4 ordered-dither Bayer threshold matrix, values in [-0.5, 0.5).
_BAYER_4X4: np.ndarray = (
    np.array(
        [
            [0, 8, 2, 10],
            [12, 4, 14, 6],
            [3, 11, 1, 9],
            [15, 7, 13, 5],
        ],
        dtype=np.float32,
    )
    / 16.0
    - 0.5
)

# ANIM_PALETTE bundle shape used by AnimPaletteExtractNode (Task 21).
# {"colors": [[r, g, b], ...]}  — r/g/b are ints in [0, 255].
ANIM_PALETTE: dict[str, list[list[int]]] = {"colors": []}


def trim_batch(
    image: Tensor,
    threshold: float = 0.03,
    mode: str = "union",
    pad: int = 0,
) -> tuple[Tensor, list[dict]]:
    """Trim transparent borders from an IMAGE batch [B, H, W, C].

    Returns the cropped batch and per-frame metadata with keys:

    - ``source_size`` — [w, h] of the original (pre-crop) frame.
    - ``offset`` — [x, y] top-left of the crop box in the original frame.
    - ``crop_size`` — [w, h] of the content crop before any zero-padding.
      In ``union`` mode this equals the output tensor [W, H] (all frames share
      the union box).  In ``per_frame`` mode this is each frame's own content
      dimensions; a fully-transparent frame reports ``[0, 0]``.

    Parameters
    ----------
    image:
        ComfyUI IMAGE tensor ``[B, H, W, C]`` float32 in [0, 1].  RGBA
        (C == 4) is required for meaningful alpha trimming; a 3-ch image
        returns the full rectangle.
    threshold:
        Alpha value below which a pixel is considered transparent.
    mode:
        ``"union"`` — one shared bounding box across all frames (min left/top,
        max right/bottom); every frame is cropped to it so the batch stays
        spatially registered.  Fully-transparent frames contribute nothing to
        the union.  The output tensor has uniform H/W across all frames.

        ``"per_frame"`` — each frame is cropped to its own bounding box.
        Frames whose crops differ in size are zero-padded to the largest crop
        so the output can still be returned as a single stacked Tensor.
    pad:
        Expand the crop box by this many pixels on each edge (clamped to image
        bounds).
    """
    h = int(image.shape[1])
    w = int(image.shape[2])
    b = int(image.shape[0])

    # Collect per-frame bboxes; None means fully transparent.
    raw: list[tuple[int, int, int, int] | None] = [
        images.alpha_bbox(image[i], threshold) for i in range(b)
    ]

    if mode == "union":
        valid = [bb for bb in raw if bb is not None]
        if not valid:
            # Fully transparent batch — return unchanged with zero offsets.
            rects: list[dict] = [
                {"source_size": [w, h], "offset": [0, 0], "crop_size": [w, h]}
                for _ in range(b)
            ]
            return image, rects

        left = max(0, min(bb[0] for bb in valid) - pad)
        top = max(0, min(bb[1] for bb in valid) - pad)
        right = min(w, max(bb[2] for bb in valid) + pad)
        bottom = min(h, max(bb[3] for bb in valid) + pad)

        rects = [
            {
                "source_size": [w, h],
                "offset": [left, top],
                "crop_size": [right - left, bottom - top],
            }
            for _ in range(b)
        ]
        return image[:, top:bottom, left:right, :], rects

    # per_frame mode — compute each frame's own crop box.
    crops: list[Tensor] = []
    rects = []
    for i in range(b):
        bbox = raw[i]
        if bbox is None:
            fl, ft, fr, fb = 0, 0, w, h
        else:
            fl, ft, fr, fb = bbox
        fl = max(0, fl - pad)
        ft = max(0, ft - pad)
        fr = min(w, fr + pad)
        fb = min(h, fb + pad)
        cw = 0 if raw[i] is None else fr - fl
        ch = 0 if raw[i] is None else fb - ft
        crops.append(image[i, ft:fb, fl:fr, :])
        rects.append({"source_size": [w, h], "offset": [fl, ft], "crop_size": [cw, ch]})

    # Zero-pad crops to the largest dimensions so torch.stack succeeds.
    max_h = max(int(c.shape[0]) for c in crops)
    max_w = max(int(c.shape[1]) for c in crops)
    nc = int(image.shape[3])
    padded: list[Tensor] = []
    for crop in crops:
        ch = int(crop.shape[0])
        cw = int(crop.shape[1])
        if ch < max_h or cw < max_w:
            frame = torch.zeros((max_h, max_w, nc), dtype=crop.dtype)
            frame[:ch, :cw, :] = crop
            padded.append(frame)
        else:
            padded.append(crop)
    return torch.stack(padded, dim=0), rects


def pivot_point(
    w: int,
    h: int,
    kind: str,
    custom: tuple[float, float] = (0.5, 1.0),
) -> tuple[int, int]:
    """Return a pixel (x, y) pivot for a sprite of size (w, h).

    Parameters
    ----------
    w, h:
        Sprite width and height in pixels.
    kind:
        ``"center"`` → (w//2, h//2); ``"bottom_center"`` → (w//2, h);
        ``"top_center"`` → (w//2, 0); ``"custom"`` → (round(w*cx), round(h*cy)).
    custom:
        (cx, cy) relative coordinates used when *kind* is ``"custom"``.
    """
    if kind == "center":
        return (w // 2, h // 2)
    if kind == "bottom_center":
        return (w // 2, h)
    if kind == "top_center":
        return (w // 2, 0)
    if kind == "custom":
        cx, cy = custom
        return (round(w * cx), round(h * cy))
    raise ValueError(f"Unknown pivot kind: {kind!r}")


def _next_power_of_two(n: int) -> int:
    """Return the smallest power of two >= n (minimum 1)."""
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def _compute_grid_rects(
    num_frames: int,
    cell_w: int,
    cell_h: int,
    columns: int,
    padding: int,
) -> tuple[list[tuple[int, int, int, int]], int, int, int]:
    """Return (rects, num_cols, canvas_w, canvas_h) for a grid layout.

    Each rect is (x, y, w, h) — the content placement box on the canvas.
    """
    cols = columns if columns > 0 else math.ceil(math.sqrt(num_frames))
    rows = math.ceil(num_frames / cols)
    canvas_w = cols * cell_w + (cols + 1) * padding
    canvas_h = rows * cell_h + (rows + 1) * padding
    rects: list[tuple[int, int, int, int]] = []
    for i in range(num_frames):
        col = i % cols
        row = i // cols
        x = padding + col * (cell_w + padding)
        y = padding + row * (cell_h + padding)
        rects.append((x, y, cell_w, cell_h))
    return rects, cols, canvas_w, canvas_h


def _compute_horizontal_rects(
    num_frames: int,
    cell_w: int,
    cell_h: int,
    padding: int,
) -> tuple[list[tuple[int, int, int, int]], int, int, int]:
    """Return (rects, num_cols, canvas_w, canvas_h) for a single-row layout."""
    canvas_w = num_frames * cell_w + (num_frames + 1) * padding
    canvas_h = cell_h + 2 * padding
    rects: list[tuple[int, int, int, int]] = []
    for i in range(num_frames):
        x = padding + i * (cell_w + padding)
        y = padding
        rects.append((x, y, cell_w, cell_h))
    return rects, num_frames, canvas_w, canvas_h


def _compute_vertical_rects(
    num_frames: int,
    cell_w: int,
    cell_h: int,
    padding: int,
) -> tuple[list[tuple[int, int, int, int]], int, int, int]:
    """Return (rects, num_cols, canvas_w, canvas_h) for a single-column layout."""
    canvas_w = cell_w + 2 * padding
    canvas_h = num_frames * cell_h + (num_frames + 1) * padding
    rects: list[tuple[int, int, int, int]] = []
    for i in range(num_frames):
        x = padding
        y = padding + i * (cell_h + padding)
        rects.append((x, y, cell_w, cell_h))
    return rects, 1, canvas_w, canvas_h


def _compute_maxrects_rects(
    num_frames: int,
    cell_w: int,
    cell_h: int,
    columns: int,
    padding: int,
) -> tuple[list[tuple[int, int, int, int]], int, int, int]:
    """Simple shelf/row bin-packer.  For uniform cells this degrades gracefully
    to a grid-like arrangement.  Returns (rects, num_cols, canvas_w, canvas_h).
    """
    if num_frames == 0:
        return [], 0, 0, 0

    # Choose a target sheet width: respect `columns` hint when given, else
    # estimate from sqrt to produce a roughly square layout.
    if columns > 0:
        cols = columns
    else:
        cols = math.ceil(math.sqrt(num_frames))

    shelf_w = cols * cell_w + (cols + 1) * padding

    rects: list[tuple[int, int, int, int]] = []
    shelf_x = padding
    shelf_y = padding
    shelf_height = 0
    actual_cols = 0
    current_row_cols = 0

    for _ in range(num_frames):
        # Start a new shelf when the frame won't fit horizontally.
        if shelf_x + cell_w + padding > shelf_w:
            shelf_y += shelf_height + padding
            shelf_x = padding
            shelf_height = 0
            actual_cols = max(actual_cols, current_row_cols)
            current_row_cols = 0

        rects.append((shelf_x, shelf_y, cell_w, cell_h))
        shelf_x += cell_w + padding
        shelf_height = max(shelf_height, cell_h)
        current_row_cols += 1

    actual_cols = max(actual_cols, current_row_cols)
    canvas_w = shelf_w
    canvas_h = shelf_y + shelf_height + padding
    return rects, actual_cols, canvas_w, canvas_h


def _extrude_frame(
    canvas: Tensor,
    frame: Tensor,
    x: int,
    y: int,
    cell_w: int,
    cell_h: int,
    extrude: int,
    padding: int,
) -> None:
    """Replicate the frame's edge pixels outward by `extrude` px into the gutter."""
    # Clamp to both the canvas edge AND the gutter width so bleed never reaches
    # a neighbouring frame's content box.
    ex = min(extrude, x, padding)
    ey = min(extrude, y, padding)
    # Canvas height and width for clamping right/bottom bleed.
    ch = int(canvas.shape[0])
    cw = int(canvas.shape[1])
    ex_r = min(extrude, cw - (x + cell_w), padding)
    ey_b = min(extrude, ch - (y + cell_h), padding)

    # Left bleed: replicate column x of frame leftward.
    if ex > 0:
        col = frame[:, 0:1, :]  # [H, 1, C]
        canvas[y: y + cell_h, x - ex: x, :] = col.expand(-1, ex, -1)
    # Right bleed: replicate last column rightward.
    if ex_r > 0:
        col = frame[:, cell_w - 1: cell_w, :]
        canvas[y: y + cell_h, x + cell_w: x + cell_w + ex_r, :] = col.expand(-1, ex_r, -1)
    # Top bleed: replicate row y of frame upward.
    if ey > 0:
        row = frame[0:1, :, :]  # [1, W, C]
        canvas[y - ey: y, x: x + cell_w, :] = row.expand(ey, -1, -1)
    # Bottom bleed: replicate last row downward.
    if ey_b > 0:
        row = frame[cell_h - 1: cell_h, :, :]
        canvas[y + cell_h: y + cell_h + ey_b, x: x + cell_w, :] = row.expand(ey_b, -1, -1)
    # Corner bleeds (top-left, top-right, bottom-left, bottom-right).
    if ex > 0 and ey > 0:
        corner = frame[0, 0, :]
        canvas[y - ey: y, x - ex: x, :] = corner
    if ex_r > 0 and ey > 0:
        corner = frame[0, cell_w - 1, :]
        canvas[y - ey: y, x + cell_w: x + cell_w + ex_r, :] = corner
    if ex > 0 and ey_b > 0:
        corner = frame[cell_h - 1, 0, :]
        canvas[y + cell_h: y + cell_h + ey_b, x - ex: x, :] = corner
    if ex_r > 0 and ey_b > 0:
        corner = frame[cell_h - 1, cell_w - 1, :]
        canvas[y + cell_h: y + cell_h + ey_b, x + cell_w: x + cell_w + ex_r, :] = corner


def pack_sheet(
    image: Tensor,
    layout: str = "grid",
    columns: int = 0,
    padding: int = 2,
    extrude: int = 0,
    power_of_two: bool = False,
    trim_data: Optional[dict] = None,
) -> tuple[Tensor, dict]:
    """Pack an IMAGE batch [B, H, W, C] into a sprite sheet [1, H', W', 4].

    Parameters
    ----------
    image:
        ComfyUI IMAGE batch ``[B, H, W, C]`` float32 in [0, 1].  RGBA (C=4)
        or RGB (C=3) — RGB frames are treated as fully opaque.
    layout:
        ``"grid"`` — rows × columns grid (auto cols = ceil(sqrt(B)));
        ``"horizontal"`` — single row;
        ``"vertical"`` — single column;
        ``"maxrects"`` — simple shelf/row bin-packer (gracefully grid-like for
        uniform cells).
    columns:
        Column hint; ``<=0`` means auto (grid / maxrects only).
    padding:
        Gutter in pixels between cells (and around the border).
    extrude:
        Replicate each frame's edge pixels outward by this many pixels into the
        gutter (bleed to prevent texture-filter seams).  0 = disabled.
    power_of_two:
        Round the final canvas W and H up to the next power of two.
    trim_data:
        Optional ``SPRITE_TRIM`` bundle (from Task 14) — a dict with a
        ``"frames"`` list of per-frame dicts containing ``source_size``,
        ``offset``, ``pivot``, and ``crop_size``.  When absent, all frames get
        default metadata.

    Returns
    -------
    tuple[Tensor, dict]
        ``(sheet, atlas)`` where ``sheet`` is ``[1, H', W', 4]`` and ``atlas``
        is ``{"sheet_size": [w, h], "columns": n, "frames": [...]}``.
    """
    b = int(image.shape[0])
    cell_h = int(image.shape[1])
    cell_w = int(image.shape[2])
    nc = int(image.shape[3])

    # Promote to RGBA if needed.
    if nc == 3:
        alpha = torch.ones((b, cell_h, cell_w, 1), dtype=image.dtype)
        image_rgba = torch.cat([image, alpha], dim=-1)
    else:
        image_rgba = image

    # Compute layout-specific placement rects and canvas size.
    if layout == "horizontal":
        rects, num_cols, canvas_w, canvas_h = _compute_horizontal_rects(
            b, cell_w, cell_h, padding
        )
    elif layout == "vertical":
        rects, num_cols, canvas_w, canvas_h = _compute_vertical_rects(
            b, cell_w, cell_h, padding
        )
    elif layout == "maxrects":
        rects, num_cols, canvas_w, canvas_h = _compute_maxrects_rects(
            b, cell_w, cell_h, columns, padding
        )
    else:  # default: grid
        rects, num_cols, canvas_w, canvas_h = _compute_grid_rects(
            b, cell_w, cell_h, columns, padding
        )

    # Apply power-of-two rounding before compositing.
    if power_of_two:
        canvas_w = _next_power_of_two(canvas_w)
        canvas_h = _next_power_of_two(canvas_h)

    # Allocate transparent RGBA canvas.
    canvas = torch.zeros((canvas_h, canvas_w, 4), dtype=image.dtype)

    for i in range(b):
        x, y, cw, ch = rects[i]
        frame = image_rgba[i]  # [cell_h, cell_w, 4]

        # Optional extrude bleed into gutter.
        if extrude > 0:
            _extrude_frame(canvas, frame, x, y, cw, ch, extrude, padding)

        # Composite frame onto canvas.
        canvas[y: y + ch, x: x + cw, :] = frame

    # Build atlas metadata per frame.
    frame_entries: list[dict] = []
    trim_frames: list[dict] = (trim_data or {}).get("frames", [])
    for i, (x, y, cw, ch) in enumerate(rects):
        if i < len(trim_frames):
            tf = trim_frames[i]
            source_size: list[int] = tf.get("source_size", [cell_w, cell_h])
            offset: list[int] = tf.get("offset", [0, 0])
            raw_pivot = tf.get("pivot")
            pivot: Optional[list[int]] = list(raw_pivot) if raw_pivot is not None else None
        else:
            source_size = [cell_w, cell_h]
            offset = [0, 0]
            pivot = None

        frame_entries.append({
            "rect": [x, y, cw, ch],
            "source_size": source_size,
            "offset": offset,
            "pivot": pivot,
            "duration_ms": None,
        })

    atlas: dict = {
        "sheet_size": [canvas_w, canvas_h],
        "columns": num_cols,
        "frames": frame_entries,
    }

    sheet = canvas.unsqueeze(0)
    return sheet, atlas


# ---------------------------------------------------------------------------
# Palette extraction and quantization
# ---------------------------------------------------------------------------


def extract_palette(
    image: Tensor,
    colors: int = 32,
) -> list[tuple[int, int, int]]:
    """Extract up to *colors* representative RGB colors using Pillow median-cut.

    Parameters
    ----------
    image:
        ComfyUI IMAGE tensor [B, H, W, C] float32 in [0, 1].  Only the first
        batch item is used; alpha is ignored for extraction.
    colors:
        Maximum palette size passed to ``Image.quantize``.

    Returns
    -------
    list[tuple[int, int, int]]
        Up to *colors* (r, g, b) tuples with values in [0, 255].  Duplicate
        entries produced by quantization are removed, so the list may be
        shorter than *colors*.
    """
    frame = image[0]
    rgb_np = (
        (frame[..., :3].clamp(0.0, 1.0).cpu().numpy() * 255.0)
        .round()
        .astype(np.uint8)
    )
    pil_img = Image.fromarray(rgb_np, mode="RGB")
    quantized = pil_img.quantize(colors=colors, method=Image.Quantize.MEDIANCUT)
    pal_data: list[int] = quantized.getpalette() or []
    n = min(colors, len(pal_data) // 3)
    seen: set[tuple[int, int, int]] = set()
    result: list[tuple[int, int, int]] = []
    for i in range(n):
        r, g, b = pal_data[i * 3], pal_data[i * 3 + 1], pal_data[i * 3 + 2]
        c = (r, g, b)
        if c not in seen:
            seen.add(c)
            result.append(c)
    return result


def _nearest_color_np(
    rgb_np: np.ndarray,
    pal_np: np.ndarray,
) -> np.ndarray:
    """Map each pixel in *rgb_np* [H, W, 3] uint8 to the nearest palette color.

    Returns [H, W, 3] uint8 where every pixel value is exactly a palette entry.
    """
    h, w = rgb_np.shape[:2]
    pixels = rgb_np.reshape(-1, 3).astype(np.float32)
    pal_f = pal_np.astype(np.float32)
    diff = pixels[:, None, :] - pal_f[None, :, :]  # [N, K, 3]
    dists = (diff * diff).sum(axis=-1)  # [N, K]
    indices = dists.argmin(axis=-1)  # [N]
    return pal_np[indices].reshape(h, w, 3)


def _pil_palette_image(palette: list[tuple[int, int, int]]) -> Image.Image:
    """Build a 1×1 P-mode PIL image carrying *palette* (padded to 256 colors).

    Unused palette slots are filled with the last real palette color so that
    Floyd-Steinberg dithering never maps a pixel to an off-palette entry like
    ``(0, 0, 0)`` (what zero-padding would produce for non-black palettes).
    """
    flat: list[int] = [v for c in palette for v in c]
    last_color: list[int] = list(palette[-1]) if palette else [0, 0, 0]
    remaining = 768 - len(flat)
    repeats = remaining // 3
    flat += last_color * repeats
    pal_img: Image.Image = Image.new("P", (1, 1))
    pal_img.putpalette(flat)
    return pal_img


def _quantize_floyd_steinberg(
    rgb_np: np.ndarray,
    palette: list[tuple[int, int, int]],
) -> np.ndarray:
    """Dither *rgb_np* [H, W, 3] uint8 against *palette* via PIL Floyd-Steinberg."""
    pil_img = Image.fromarray(rgb_np, mode="RGB")
    pal_img = _pil_palette_image(palette)
    quantized = pil_img.quantize(
        palette=pal_img, dither=Image.Dither.FLOYDSTEINBERG
    )
    return np.array(quantized.convert("RGB"), dtype=np.uint8)


def _quantize_ordered(
    rgb_np: np.ndarray,
    pal_np: np.ndarray,
    h: int,
    w: int,
) -> np.ndarray:
    """4×4 Bayer ordered dither: offset each pixel before nearest-color snap."""
    bayer = np.tile(_BAYER_4X4, (math.ceil(h / 4), math.ceil(w / 4)))[:h, :w]
    offset = (bayer * 32.0)[:, :, None]  # scale to ±16 in [0,255] space
    dithered = np.clip(rgb_np.astype(np.float32) + offset, 0.0, 255.0)
    return _nearest_color_np(dithered.astype(np.uint8), pal_np)


def quantize_to_palette(
    image: Tensor,
    palette: list[tuple[int, int, int]],
    dither: str = "none",
    preserve_alpha: bool = True,
) -> Tensor:
    """Remap every pixel in *image* to the nearest color in *palette*.

    Parameters
    ----------
    image:
        ComfyUI IMAGE tensor [B, H, W, C] float32 in [0, 1].  Only the first
        batch item is processed.
    palette:
        List of (r, g, b) tuples with values in [0, 255].
    dither:
        ``"none"`` — pure nearest-color mapping (no dithering);
        ``"floyd_steinberg"`` — PIL Floyd-Steinberg error diffusion;
        ``"ordered"`` — 4×4 Bayer ordered dither (offset applied before
        nearest-color snap).
    preserve_alpha:
        When True and the input is 4-channel RGBA, the original alpha channel
        is reattached to the output unchanged.

    Returns
    -------
    Tensor
        [1, H, W, C] float32 in [0, 1].  C == 4 when *preserve_alpha* is True
        and the input had 4 channels; otherwise C == 3.
    """
    frame = image[0]  # [H, W, C]
    h = int(frame.shape[0])
    w = int(frame.shape[1])
    nc = int(frame.shape[2])

    alpha: Optional[Tensor] = None
    if preserve_alpha and nc == 4:
        alpha = frame[..., 3:4].clone()

    rgb_np = (
        (frame[..., :3].clamp(0.0, 1.0).cpu().numpy() * 255.0)
        .round()
        .astype(np.uint8)
    )
    pal_np = np.array(palette, dtype=np.uint8)

    if dither == "floyd_steinberg":
        out_np = _quantize_floyd_steinberg(rgb_np, palette)
    elif dither == "ordered":
        out_np = _quantize_ordered(rgb_np, pal_np, h, w)
    else:
        out_np = _nearest_color_np(rgb_np, pal_np)

    out_f = torch.from_numpy(out_np.astype(np.float32) / 255.0)  # [H, W, 3]

    if alpha is not None:
        out_f = torch.cat([out_f, alpha], dim=-1)

    return out_f.unsqueeze(0)  # [1, H, W, C]

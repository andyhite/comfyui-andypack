"""Sprite trim, pivot, sheet-packing, and palette helpers for game asset export.

No ComfyUI or PromptServer imports — torch/numpy/PIL only.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor

from andypack import images


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


def pack_direction_rows(
    rows: list[tuple[str, list[Tensor]]],
    fps: int = 16,
    padding: int = 2,
    power_of_two: bool = False,
) -> tuple[Tensor, dict]:
    """Pack a multi-direction animation into a sprite sheet: one ROW per direction,
    one COLUMN per frame. This is the game-ready layout that ``pack_sheet`` (a flat
    grid) can't guarantee when frame counts differ per direction.

    Parameters
    ----------
    rows:
        Ordered ``[(direction_name, [frame_tensor, ...]), ...]``. Each frame tensor
        is an IMAGE ``[1, H, W, C]`` (or ``[H, W, C]``); RGB is promoted to opaque
        RGBA. Rows may have different frame counts — short rows are left-padded with
        transparent cells so every row spans the full column count and stays aligned.
    fps:
        Playback rate; written into every frame's ``duration_ms`` (``1000/fps``).
    padding, power_of_two:
        As in :func:`pack_sheet`.

    Returns
    -------
    tuple[Tensor, dict]
        ``(sheet [1, H', W', 4], atlas)``. The atlas is a superset of the
        ``pack_sheet`` contract — ``sheet_size``/``columns``/``frames`` — plus
        ``fps`` and ``tags`` (``[{"name": dir, "from": i, "to": j}, ...]``, one per
        direction) so ``AtlasMetadataWriter`` can emit per-direction animations.
    """
    if not rows:
        raise ValueError("pack_direction_rows: no directions to pack")

    def _norm(t: Tensor) -> Tensor:
        f = t[0] if t.dim() == 4 else t
        if f.shape[-1] == 3:
            alpha = torch.ones((*f.shape[:-1], 1), dtype=f.dtype)
            f = torch.cat([f, alpha], dim=-1)
        return f

    norm_rows = [(name, [_norm(t) for t in frames]) for name, frames in rows]
    cell_h = max((f.shape[0] for _n, fs in norm_rows for f in fs), default=1)
    cell_w = max((f.shape[1] for _n, fs in norm_rows for f in fs), default=1)
    cols = max((len(fs) for _n, fs in norm_rows), default=0)
    n_rows = len(norm_rows)

    canvas_w = cols * cell_w + (cols + 1) * padding
    canvas_h = n_rows * cell_h + (n_rows + 1) * padding
    if power_of_two:
        canvas_w = _next_power_of_two(canvas_w)
        canvas_h = _next_power_of_two(canvas_h)
    canvas = torch.zeros((canvas_h, canvas_w, 4), dtype=torch.float32)

    duration_ms = int(round(1000.0 / fps)) if fps > 0 else 100
    frame_entries: list[dict] = []
    tags: list[dict] = []
    idx = 0
    for r, (name, frames) in enumerate(norm_rows):
        y = padding + r * (cell_h + padding)
        first = idx
        for c, frame in enumerate(frames):
            x = padding + c * (cell_w + padding)
            fh, fw = int(frame.shape[0]), int(frame.shape[1])
            canvas[y: y + fh, x: x + fw, :] = frame.to(torch.float32)
            frame_entries.append({
                "rect": [x, y, cell_w, cell_h],
                "source_size": [cell_w, cell_h],
                "offset": [0, 0],
                "pivot": [cell_w // 2, cell_h],
                "duration_ms": duration_ms,
            })
            idx += 1
        if frames:
            tags.append({"name": name, "from": first, "to": idx - 1})

    atlas: dict = {
        "sheet_size": [canvas_w, canvas_h],
        "columns": cols,
        "frames": frame_entries,
        "fps": fps,
        "tags": tags,
    }
    return canvas.unsqueeze(0), atlas


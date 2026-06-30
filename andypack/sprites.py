"""Sprite trim and pivot helpers for game asset export.

No ComfyUI or PromptServer imports — torch/PIL only.
"""

from __future__ import annotations

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

"""Torch/PIL bridge for ComfyUI IMAGE tensors. Isolated so the rest of the pack stays pure."""

from __future__ import annotations

import math
import os
import tempfile
from typing import Optional

import numpy as np
import torch
from PIL import Image


# Background the alpha channel is composited over when flattening RGBA/LA art to
# the 3-channel RGB that ComfyUI IMAGE tensors carry. White suits character art on
# a transparent background; `.convert("RGB")` alone would composite onto black.
_MATTE = (255, 255, 255)


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


def mirror_png(src: str, dst: str) -> None:
    """Atomically write `dst` as the horizontal mirror of the PNG at `src`.

    Used to synthesize a mirror-mapped direction (e.g. WEST from EAST) without
    re-generating it. Transparency is preserved so a mirrored frame still
    composites correctly downstream.
    """
    with Image.open(src) as img:
        flipped = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        flipped.load()
    directory = os.path.dirname(dst) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, suffix=".png.tmp")
    os.close(fd)
    try:
        flipped.save(tmp, format="PNG")
        os.replace(tmp, dst)
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


def _load_frames_dir(directory: str) -> "torch.Tensor | None":
    """Load `frame_*.png` from a directory as a [N, H, W, C] batch (sorted by
    name), or None when the directory holds no frames."""
    try:
        names = sorted(
            n for n in os.listdir(directory)
            if n.startswith("frame_") and n.endswith(".png")
        )
    except OSError:
        return None
    if not names:
        return None
    frames = [load_image_tensor(os.path.join(directory, n)) for n in names]
    return torch.cat(frames, dim=0)


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


def assemble_playback(segments: list) -> torch.Tensor:
    """Concatenate a playback plan (see resolve.playback_segments) into one IMAGE
    batch. `anim` segments load their frame dir, tile `repeat` times, then drop the
    seam boundary frame(s); `hold` segments repeat a single image `count` times.
    Segments with no readable frames are skipped.

    Segments may differ in resolution — a held pose anchor is often
    authored at a different size than the action frames. Every segment is conformed
    to a single target (the first `anim` segment's size — the action's own frames —
    else the first part) so torch.cat doesn't crash on mismatched H/W."""
    parts: list[torch.Tensor] = []
    target: tuple[int, int] | None = None  # (H, W) from the first anim segment
    for seg in segments:
        if seg["kind"] == "anim":
            batch = _load_frames_dir(seg["dir"])
            if batch is None:
                continue
            repeat = max(int(seg.get("repeat", 1)), 1)
            if repeat > 1:
                batch = batch.repeat(repeat, 1, 1, 1)
            if seg.get("drop_first") and batch.shape[0] > 1:
                batch = batch[1:]
            if seg.get("drop_last") and batch.shape[0] > 1:
                batch = batch[:-1]
            if target is None:
                target = (int(batch.shape[1]), int(batch.shape[2]))
            parts.append(batch)
        else:  # hold a single image for `count` frames
            img = load_image_tensor(seg["image"])
            parts.append(img.repeat(max(int(seg.get("count", 1)), 1), 1, 1, 1))
    if not parts:
        return empty_image()
    if target is None:
        target = (int(parts[0].shape[1]), int(parts[0].shape[2]))
    parts = [_resize_batch(p, target[0], target[1]) for p in parts]
    return torch.cat(parts, dim=0)


def save_animated_webp(frames: torch.Tensor, path: str, fps: int) -> None:
    """Encode an IMAGE batch [N, H, W, C] as an animated WEBP at `path`, played at
    `fps`. A single frame writes a still WEBP."""
    arr = (frames.clamp(0.0, 1.0).cpu().numpy() * 255.0).round().astype(np.uint8)
    pil = [Image.fromarray(a, mode="RGB") for a in arr]
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    duration = int(round(1000.0 / max(int(fps), 1)))
    pil[0].save(
        path, format="WEBP", save_all=True, append_images=pil[1:],
        duration=duration, loop=0, quality=80, method=4,
    )


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
        labels: reserved for future caption support; currently ignored.

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

    return sheet

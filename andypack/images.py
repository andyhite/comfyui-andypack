"""Torch/PIL bridge for ComfyUI IMAGE tensors. Isolated so the rest of the pack stays pure."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import torch
from PIL import Image


# Background the alpha channel is composited over when flattening RGBA/LA art to
# the 3-channel RGB that ComfyUI IMAGE tensors carry. White suits character art on
# a transparent background; `.convert("RGB")` alone would composite onto black.
_MATTE = (255, 255, 255)


def load_image_tensor(path: str) -> torch.Tensor:
    """Load a PNG into a ComfyUI IMAGE tensor [1, H, W, C] float32 in [0, 1].

    Images with transparency (RGBA / LA / paletted-with-alpha) are alpha-composited
    over a white matte rather than dropped onto black, so concept art on a
    transparent background flattens cleanly.
    """
    with Image.open(path) as img:
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


def save_image_png(image: torch.Tensor, path: str) -> None:
    """Atomically save the first batch item of an IMAGE tensor as a PNG."""
    frame = image[0] if image.dim() == 4 else image
    arr = (frame.clamp(0.0, 1.0).cpu().numpy() * 255.0).round().astype(np.uint8)
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, suffix=".png.tmp")
    os.close(fd)
    try:
        Image.fromarray(arr, mode="RGB").save(tmp, format="PNG")
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
        flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
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


def empty_image() -> torch.Tensor:
    """A 1x1 black image — the 'no anchor present' sentinel."""
    return torch.zeros((1, 1, 1, 3), dtype=torch.float32)

"""Tests for andypack.sprites: palette extraction and quantize-to-palette."""

import torch

from andypack import sprites


def test_extract_and_lock_palette() -> None:
    img = torch.zeros((1, 4, 4, 3))
    img[:, :, :2, 0] = 1.0  # left half red, right half black
    pal = sprites.extract_palette(img, colors=2)
    assert len(pal) <= 2
    out = sprites.quantize_to_palette(img, pal)
    uniq = {
        tuple(int(v * 255) for v in out[0, y, x, :3])
        for y in range(4)
        for x in range(4)
    }
    assert uniq.issubset({tuple(c) for c in pal})


def test_quantize_preserves_alpha() -> None:
    rgba = torch.ones((1, 2, 2, 4))
    rgba[..., 3] = 0.5
    out = sprites.quantize_to_palette(rgba, [(255, 255, 255)], preserve_alpha=True)
    assert out.shape[-1] == 4 and abs(float(out[0, 0, 0, 3]) - 0.5) < 1e-3

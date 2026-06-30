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
        tuple(round(float(out[0, y, x, c]) * 255) for c in range(3))
        for y in range(4)
        for x in range(4)
    }
    assert uniq.issubset({tuple(c) for c in pal})


def test_quantize_preserves_alpha() -> None:
    rgba = torch.ones((1, 2, 2, 4))
    rgba[..., 3] = 0.5
    out = sprites.quantize_to_palette(rgba, [(255, 255, 255)], preserve_alpha=True)
    assert out.shape[-1] == 4 and abs(float(out[0, 0, 0, 3]) - 0.5) < 1e-3


def test_floyd_steinberg_membership_non_black_palette() -> None:
    """FS dithering must never produce a pixel outside the caller palette.

    A palette of red + green with no black entry previously allowed zero-padded
    slots to be chosen for dark pixels, emitting (0, 0, 0) which is off-palette.
    """
    palette = [(255, 0, 0), (0, 255, 0)]
    palette_set = {tuple(c) for c in palette}
    # gradient image: top row bright, bottom row dark — exercises the dark-pixel path
    h, w = 8, 8
    img = torch.zeros((1, h, w, 3))
    for row in range(h):
        img[0, row, :, 0] = row / (h - 1)  # red channel gradient
        img[0, row, :, 1] = 1.0 - row / (h - 1)  # green channel gradient
    out = sprites.quantize_to_palette(img, palette, dither="floyd_steinberg")
    assert out.shape[-1] == 3
    for y in range(h):
        for x in range(w):
            pixel = tuple(round(float(out[0, y, x, c]) * 255) for c in range(3))
            assert pixel in palette_set, (
                f"FS dither produced off-palette pixel {pixel} at ({y},{x})"
            )


def test_quantize_rgb_input_produces_3ch_output() -> None:
    """A 3-channel input must yield a 3-channel output (no alpha reattachment)."""
    rgb = torch.ones((1, 4, 4, 3)) * 0.5
    out = sprites.quantize_to_palette(rgb, [(128, 128, 128)], preserve_alpha=False)
    assert out.shape[-1] == 3


def test_quantize_rgba_no_preserve_alpha_produces_3ch_output() -> None:
    """preserve_alpha=False on RGBA input must drop the alpha channel."""
    rgba = torch.ones((1, 4, 4, 4))
    rgba[..., 3] = 0.7
    out = sprites.quantize_to_palette(rgba, [(255, 255, 255)], preserve_alpha=False)
    assert out.shape[-1] == 3

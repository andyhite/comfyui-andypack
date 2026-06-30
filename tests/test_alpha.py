"""Tests for alpha-aware disk boundary in images.py (Task 10)."""

import torch
from PIL import Image

from andypack import images


def test_save_png_with_mask_writes_rgba(tmp_path):
    img = torch.ones((1, 4, 4, 3))
    mask = torch.zeros((1, 4, 4))
    mask[:, :2, :2] = 1.0
    p = str(tmp_path / "s.png")
    images.save_image_png(img, p, mask=mask)
    with Image.open(p) as im:
        assert im.mode == "RGBA"
        assert im.getpixel((0, 0))[3] == 255
        assert im.getpixel((3, 3))[3] == 0


def test_save_png_with_rgba_input_preserves_alpha(tmp_path):
    rgba = torch.ones((1, 2, 2, 4))
    rgba[..., 3] = 0.0
    p = str(tmp_path / "r.png")
    images.save_image_png(rgba, p)
    with Image.open(p) as im:
        assert im.mode == "RGBA"
        assert im.getpixel((0, 0))[3] == 0


def test_save_png_rgb_unchanged(tmp_path):
    p = str(tmp_path / "rgb.png")
    images.save_image_png(torch.ones((1, 2, 2, 3)), p)
    with Image.open(p) as im:
        assert im.mode == "RGB"


def test_load_keep_alpha_roundtrip(tmp_path):
    rgba = torch.ones((1, 2, 2, 4))
    rgba[..., 3] = 0.0
    p = str(tmp_path / "r.png")
    images.save_image_png(rgba, p)
    t = images.load_image_tensor(p, keep_alpha=True)
    assert t.shape[-1] == 4
    assert float(t[0, 0, 0, 3]) == 0.0
    assert images.load_image_tensor(p).shape[-1] == 3


def test_to_rgba_with_mask(tmp_path):
    img = torch.ones((1, 4, 4, 3))
    mask = torch.zeros((1, 4, 4))
    mask[:, :2, :2] = 1.0
    out = images.to_rgba(img, mask=mask)
    assert out.shape == (4, 4, 4)
    assert float(out[0, 0, 3]) == 1.0
    assert float(out[3, 3, 3]) == 0.0


def test_to_rgba_from_rgba_input():
    rgba = torch.ones((1, 2, 2, 4))
    rgba[..., 3] = 0.5
    out = images.to_rgba(rgba)
    assert out.shape == (2, 2, 4)
    assert abs(float(out[0, 0, 3]) - 0.5) < 1e-5


def test_to_rgba_from_rgb_gets_full_alpha():
    rgb = torch.ones((1, 3, 3, 3)) * 0.5
    out = images.to_rgba(rgb)
    assert out.shape == (3, 3, 4)
    assert float(out[0, 0, 3]) == 1.0


def test_alpha_bbox_rgb_returns_full_rect():
    img = torch.ones((1, 4, 6, 3))
    result = images.alpha_bbox(img)
    assert result == (0, 0, 6, 4)


def test_alpha_bbox_rgba_finds_opaque_region():
    img = torch.zeros((1, 4, 4, 4))
    img[0, 1:3, 1:3, 3] = 1.0
    result = images.alpha_bbox(img)
    assert result == (1, 1, 3, 3)


def test_alpha_bbox_fully_transparent_returns_none():
    img = torch.zeros((1, 4, 4, 4))
    result = images.alpha_bbox(img)
    assert result is None


def test_load_keep_alpha_false_on_rgba_png_gives_3ch(tmp_path):
    rgba = torch.ones((1, 2, 2, 4))
    rgba[..., 3] = 0.5
    p = str(tmp_path / "semi.png")
    images.save_image_png(rgba, p)
    t = images.load_image_tensor(p, keep_alpha=False)
    assert t.shape[-1] == 3


def test_load_keep_alpha_on_rgb_png_gives_4ch(tmp_path):
    rgb = torch.ones((1, 2, 2, 3)) * 0.5
    p = str(tmp_path / "rgb.png")
    images.save_image_png(rgb, p)
    t = images.load_image_tensor(p, keep_alpha=True)
    assert t.shape[-1] == 4
    assert float(t[0, 0, 0, 3]) == 1.0

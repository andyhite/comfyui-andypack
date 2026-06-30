"""Tests for andypack.sprites: trim_batch and pivot_point."""

import torch

from andypack import sprites


def _frame(w: int, h: int, box: tuple[int, int, int, int]) -> torch.Tensor:
    """Return a [1, H, W, 4] RGBA frame with an opaque rectangle at *box*."""
    img = torch.zeros((h, w, 4))
    left, top, right, bottom = box
    img[top:bottom, left:right, :3] = 1.0
    img[top:bottom, left:right, 3] = 1.0
    return img.unsqueeze(0)


def test_trim_union_crops_to_shared_bbox() -> None:
    batch = torch.cat([_frame(8, 8, (1, 1, 4, 4)), _frame(8, 8, (3, 3, 6, 6))], dim=0)
    out, rects = sprites.trim_batch(batch, mode="union")
    # Union bbox is (1..6) x (1..6) → 5 × 5
    assert out.shape[1] == 5 and out.shape[2] == 5
    assert len(rects) == 2 and rects[0]["offset"] == [1, 1]


def test_pivot_bottom_center() -> None:
    assert sprites.pivot_point(10, 20, "bottom_center") == (5, 20)

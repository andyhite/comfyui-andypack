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
    # crop_size must equal the output tensor [W, H] for every frame in union mode.
    out_w = int(out.shape[2])
    out_h = int(out.shape[1])
    for rect in rects:
        assert rect["crop_size"] == [out_w, out_h], rect


def test_trim_per_frame_different_content_sizes() -> None:
    """Per-frame mode: each frame gets its own crop_size; output tensor is uniform."""
    # Frame 0: 2 × 2 content box at (1,1)–(3,3)
    # Frame 1: 4 × 4 content box at (0,0)–(4,4)
    frame0 = _frame(8, 8, (1, 1, 3, 3))
    frame1 = _frame(8, 8, (0, 0, 4, 4))
    batch = torch.cat([frame0, frame1], dim=0)
    out, rects = sprites.trim_batch(batch, mode="per_frame")

    # Output tensor must be uniformly sized (zero-padded to the larger crop).
    assert out.shape[0] == 2
    assert out.shape[1] == out.shape[1]  # uniform H across frames
    assert out.shape[2] == out.shape[2]  # uniform W across frames

    # Each frame's crop_size reflects its own content, NOT the padded tensor size.
    assert rects[0]["crop_size"] == [2, 2], rects[0]
    assert rects[1]["crop_size"] == [4, 4], rects[1]

    # Offsets stay correct.
    assert rects[0]["offset"] == [1, 1], rects[0]
    assert rects[1]["offset"] == [0, 0], rects[1]

    # source_size is always the original frame dimensions.
    assert rects[0]["source_size"] == [8, 8]
    assert rects[1]["source_size"] == [8, 8]

    # The output tensor is padded to the larger crop (4 × 4).
    assert int(out.shape[1]) == 4
    assert int(out.shape[2]) == 4


def test_pivot_bottom_center() -> None:
    assert sprites.pivot_point(10, 20, "bottom_center") == (5, 20)


def test_pack_grid_places_frames() -> None:
    batch = torch.zeros((4, 6, 6, 4))
    batch[..., :] = 1.0
    sheet, atlas = sprites.pack_sheet(batch, layout="grid", columns=2, padding=1)
    assert sheet.shape[0] == 1 and atlas["columns"] == 2 and len(atlas["frames"]) == 4
    assert atlas["frames"][1]["rect"][0] > atlas["frames"][0]["rect"][0]  # col 2 right of col 1


def test_pack_power_of_two() -> None:
    sheet, _ = sprites.pack_sheet(torch.ones((1, 5, 5, 4)), power_of_two=True)
    h, w = sheet.shape[1], sheet.shape[2]
    assert (h & (h - 1)) == 0 and (w & (w - 1)) == 0

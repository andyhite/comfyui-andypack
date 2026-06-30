import pytest
import numpy as np
from PIL import Image

from andypack import images


def test_load_image_composites_alpha_over_white(tmp_path):
    # Fully-transparent black pixel must flatten to white, not black.
    p = tmp_path / "concept.png"
    Image.new("RGBA", (2, 2), (0, 0, 0, 0)).save(p)
    t = images.load_image_tensor(str(p))
    assert t.shape == (1, 2, 2, 3)
    assert float(t.min()) == 1.0  # composited onto the white matte


def test_load_image_opaque_rgb_unchanged(tmp_path):
    p = tmp_path / "flat.png"
    Image.new("RGB", (2, 2), (10, 20, 30)).save(p)
    t = images.load_image_tensor(str(p))
    assert np.allclose(t[0, 0, 0].numpy(), np.array([10, 20, 30]) / 255.0, atol=1e-3)


def test_mirror_png_flips_horizontally(tmp_path):
    src = tmp_path / "src.png"
    dst = tmp_path / "dst.png"
    # left column red, right column blue
    arr = np.zeros((1, 2, 3), dtype=np.uint8)
    arr[0, 0] = (255, 0, 0)
    arr[0, 1] = (0, 0, 255)
    Image.fromarray(arr, "RGB").save(src)

    images.mirror_png(str(src), str(dst))
    out = np.asarray(Image.open(dst).convert("RGB"))
    assert tuple(out[0, 0]) == (0, 0, 255)  # columns swapped
    assert tuple(out[0, 1]) == (255, 0, 0)


def _frames_dir(tmp_path, name, count):
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        Image.new("RGB", (2, 2), (i, i, i)).save(d / f"frame_{i:05d}.png")
    return str(d)


def test_assemble_playback_repeats_holds_and_drops_seams(tmp_path):
    pre = _frames_dir(tmp_path, "pre", 3)
    action = _frames_dir(tmp_path, "act", 4)
    hold_png = tmp_path / "pose.png"
    Image.new("RGB", (2, 2), (9, 9, 9)).save(hold_png)

    batch = images.assemble_playback([
        {"kind": "anim", "dir": pre, "repeat": 1, "drop_first": False, "drop_last": False},
        {"kind": "anim", "dir": action, "repeat": 2, "drop_first": True, "drop_last": True},
        {"kind": "hold", "image": str(hold_png), "count": 5},
    ])
    # pre(3) + action(4*2, minus first & last = 6) + hold(5) = 14
    assert batch.shape[0] == 3 + 6 + 5
    assert batch.shape[1:] == (2, 2, 3)


def test_assemble_playback_skips_empty_dirs(tmp_path):
    empty = str(tmp_path / "empty")
    (tmp_path / "empty").mkdir()
    batch = images.assemble_playback([{"kind": "anim", "dir": empty, "repeat": 1}])
    assert batch.shape[0] == 1  # empty_image sentinel, nothing to concat


def test_assemble_playback_conforms_mismatched_resolutions(tmp_path):
    # A held anchor authored at a different size than the action frames must not
    # crash torch.cat — every segment is resized to a common target.
    action = _frames_dir(tmp_path, "act", 3)  # 2x2 frames (see _frames_dir)
    big_hold = tmp_path / "concept.png"
    Image.new("RGB", (8, 6), (9, 9, 9)).save(big_hold)  # 6h x 8w, different size
    batch = images.assemble_playback([
        {"kind": "hold", "image": str(big_hold), "count": 2},
        {"kind": "anim", "dir": action, "repeat": 1, "drop_first": False, "drop_last": False},
    ])
    assert batch.shape[0] == 2 + 3
    assert batch.shape[1:] == (2, 2, 3)  # conformed to the action's frame size


def test_save_animated_webp_writes_all_frames(tmp_path):
    import torch
    frames = torch.stack([
        torch.full((2, 2, 3), v / 255.0, dtype=torch.float32) for v in (10, 20, 30)
    ])
    out = tmp_path / "clip.webp"
    images.save_animated_webp(frames, str(out), fps=12)
    assert out.exists()
    with Image.open(out) as im:
        assert getattr(im, "n_frames", 1) == 3


def test_contact_sheet_handles_missing_tiles():
    import torch
    sheet = images.contact_sheet([torch.ones((1, 4, 4, 3)), None], columns=2, cell=(4, 4))
    assert sheet.shape[0] == 1 and sheet.shape[2] == 8  # 2 columns x 4 px wide


def test_contact_sheet_all_none_returns_placeholder_grid():
    sheet = images.contact_sheet([None, None], columns=2, cell=(8, 6))
    assert sheet.shape == (1, 6, 16, 3)
    assert float(sheet.mean()) == pytest.approx(0.5)


def test_contact_sheet_infers_cell_from_tile_max():
    import torch
    tile = torch.zeros((1, 10, 20, 3))
    sheet = images.contact_sheet([tile, None], columns=2)
    assert sheet.shape[1] == 10 and sheet.shape[2] == 40


def test_ping_pong_appends_reversed_interior():
    import torch
    f = torch.arange(4).float().reshape(4, 1, 1, 1).repeat(1, 2, 2, 3)
    out = images.apply_play_mode(f, "ping_pong")
    assert out.shape[0] == 6  # 0,1,2,3,2,1

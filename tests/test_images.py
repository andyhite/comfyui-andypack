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


def test_retime_resample_to_target():
    import torch
    f = torch.arange(4).float().reshape(4, 1, 1, 1).repeat(1, 2, 2, 3)
    out = images.retime_batch(f, 8, "resample")
    assert out.shape[0] == 8


def test_retime_resample_identity():
    import torch
    f = torch.zeros((5, 2, 2, 3))
    out = images.retime_batch(f, 5, "resample")
    assert out.shape[0] == 5


def test_retime_resample_single_output():
    import torch
    f = torch.zeros((4, 2, 2, 3))
    out = images.retime_batch(f, 1, "resample")
    assert out.shape[0] == 1


def test_retime_trim_truncates():
    import torch
    f = torch.zeros((8, 2, 2, 3))
    out = images.retime_batch(f, 4, "trim")
    assert out.shape[0] == 4


def test_retime_trim_over_n_returns_all():
    import torch
    f = torch.zeros((4, 2, 2, 3))
    out = images.retime_batch(f, 10, "trim")
    assert out.shape[0] == 4  # can't invent frames via trim


def test_retime_pad_hold_pads_last_frame():
    import torch
    f = torch.zeros((3, 2, 2, 3))
    f[2] = 0.5  # last frame has distinct value
    out = images.retime_batch(f, 6, "pad_hold")
    assert out.shape[0] == 6
    assert float(out[5].mean()) == pytest.approx(0.5)


def test_retime_pad_hold_truncates_when_over():
    import torch
    f = torch.zeros((6, 2, 2, 3))
    out = images.retime_batch(f, 4, "pad_hold")
    assert out.shape[0] == 4


def test_retime_target_clamped_to_1():
    import torch
    f = torch.zeros((4, 2, 2, 3))
    out = images.retime_batch(f, 0, "resample")
    assert out.shape[0] == 1


def test_retime_empty_batch_returns_unchanged():
    import torch
    f = torch.zeros((0, 2, 2, 3))
    out = images.retime_batch(f, 4, "resample")
    assert out.shape[0] == 0


def test_recolor_hue_preserves_alpha():
    import torch
    rgba = torch.ones((1, 2, 2, 4))
    rgba[..., 3] = 0.5
    rgba[..., 0] = 1.0
    rgba[..., 1:3] = 0.0
    out = images.recolor(rgba, {"hue": 120, "sat": 1.0, "val": 1.0})
    assert out.shape[-1] == 4 and abs(float(out[0, 0, 0, 3]) - 0.5) < 1e-3


def test_recolor_hue_rotates_color():
    import torch
    # Pure red pixel: h=0, s=1, v=1 → rotate +120° → green
    rgb = torch.zeros((1, 1, 1, 3))
    rgb[..., 0] = 1.0
    out = images.recolor(rgb, {"hue": 120, "sat": 1.0, "val": 1.0})
    assert out.shape == (1, 1, 1, 3)
    assert float(out[0, 0, 0, 1]) > 0.9  # green channel high


def test_recolor_hex_remaps_hue():
    import torch
    # Pure red → hex blue (#0000FF target hue), sat+val preserved
    rgb = torch.zeros((1, 1, 1, 3))
    rgb[..., 0] = 1.0
    out = images.recolor(rgb, {"hex": "#0000FF"})
    assert out.shape == (1, 1, 1, 3)
    # Blue hue: b channel should dominate
    assert float(out[0, 0, 0, 2]) > 0.9


def test_save_gif(tmp_path):
    import torch
    from PIL import Image
    # Use distinct frames so PIL does not deduplicate them.
    f = torch.stack([torch.full((4, 4, 3), v, dtype=torch.float32) for v in (0.1, 0.4, 0.7)])
    p = str(tmp_path / "a.gif")
    images.save_animated_gif(f, p, 8)
    with Image.open(p) as im:
        assert im.is_animated and im.n_frames == 3


def test_save_gif_loop_false_differs(tmp_path):
    """loop=False must produce a non-infinite GIF (differs from loop=True)."""
    import torch
    from PIL import Image
    f = torch.stack([torch.full((4, 4, 3), v, dtype=torch.float32) for v in (0.1, 0.4, 0.7)])
    p_loop = str(tmp_path / "loop.gif")
    p_once = str(tmp_path / "once.gif")
    images.save_animated_gif(f, p_loop, 8, loop=True)
    images.save_animated_gif(f, p_once, 8, loop=False)
    with Image.open(p_loop) as im_loop:
        loop_val = im_loop.info.get("loop")
    with Image.open(p_once) as im_once:
        once_val = im_once.info.get("loop")
    # Infinite GIF (loop=True) → loop count 0; play-once (loop=False) must differ.
    assert loop_val != once_val


def test_save_apng(tmp_path):
    import torch
    # Use distinct frames so PIL APNG does not deduplicate them.
    f = torch.stack([
        torch.full((4, 4, 3), v, dtype=torch.float32) for v in [0.1, 0.4, 0.7, 1.0]
    ])
    p = str(tmp_path / "b.png")
    images.save_animated_apng(f, p, 12)
    from PIL import Image
    with Image.open(p) as im:
        assert im.n_frames == 4


def test_onion_skin_shape_preserved():
    import torch
    f = torch.rand((5, 8, 8, 3))
    out = images.onion_skin(f, prev=1, next=1, opacity=0.3)
    assert out.shape == f.shape


def test_onion_skin_no_neighbors_returns_frames():
    import torch
    f = torch.rand((3, 4, 4, 3))
    out = images.onion_skin(f, prev=0, next=0, opacity=0.5)
    assert torch.allclose(out, f)


def test_onion_skin_blends_neighbors():
    import torch
    # frame 0 = zeros, frame 1 = ones, frame 2 = zeros
    # For frame 1, with prev=1 next=1, ghost = mean(frame0, frame2) = 0
    # out[1] = (1-0.5)*ones + 0.5*zeros = 0.5
    f = torch.zeros((3, 2, 2, 3))
    f[1] = 1.0
    out = images.onion_skin(f, prev=1, next=1, opacity=0.5)
    assert abs(float(out[1].mean()) - 0.5) < 1e-4

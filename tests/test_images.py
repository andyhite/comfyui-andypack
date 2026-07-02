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


def _frames_dir(tmp_path, name, count):
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        Image.new("RGB", (2, 2), (i, i, i)).save(d / f"frame_{i:05d}.png")
    return str(d)


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


def test_save_animated_gif_rgba_batch(tmp_path):
    """A 4-ch RGBA frame batch must not crash — alpha is now preserved via palette transparency."""
    import torch
    from PIL import Image
    f = torch.stack([torch.full((4, 4, 4), v, dtype=torch.float32) for v in (0.1, 0.4, 0.7)])
    p = str(tmp_path / "rgba.gif")
    images.save_animated_gif(f, p, 8)
    with Image.open(p) as im:
        # Pillow merges consecutive frame bytes when identical (e.g., fully transparent frames).
        assert im.n_frames == 2


def test_save_animated_webp_rgba_batch(tmp_path):
    """A 4-ch RGBA frame batch must not crash — alpha is dropped to RGB for WebP."""
    import torch
    from PIL import Image
    f = torch.stack([torch.full((4, 4, 4), v, dtype=torch.float32) for v in (0.1, 0.4, 0.7)])
    p = str(tmp_path / "rgba.webp")
    images.save_animated_webp(f, p, 8)
    with Image.open(p) as im:
        assert getattr(im, "n_frames", 1) == 3


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


# --- thumbnail_data_uri -------------------------------------------------------

def test_thumbnail_data_uri_returns_valid_png(tmp_path):
    import base64
    from io import BytesIO
    p = tmp_path / "source.png"
    Image.new("RGB", (200, 150), (255, 0, 0)).save(p)
    uri = images.thumbnail_data_uri(str(p))
    assert uri.startswith("data:image/png;base64,")
    raw = base64.b64decode(uri[len("data:image/png;base64,"):])
    # Valid PNG magic bytes
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"
    thumb = Image.open(BytesIO(raw))
    assert max(thumb.size) <= 96


def test_thumbnail_data_uri_respects_max_px(tmp_path):
    from io import BytesIO
    import base64
    p = tmp_path / "big.png"
    Image.new("RGB", (300, 300), (0, 128, 0)).save(p)
    uri = images.thumbnail_data_uri(str(p), max_px=32)
    raw = base64.b64decode(uri[len("data:image/png;base64,"):])
    thumb = Image.open(BytesIO(raw))
    assert max(thumb.size) <= 32


def test_thumbnail_data_uri_preserves_aspect_ratio(tmp_path):
    from io import BytesIO
    import base64
    p = tmp_path / "wide.png"
    Image.new("RGB", (200, 100), (0, 0, 255)).save(p)
    uri = images.thumbnail_data_uri(str(p), max_px=96)
    raw = base64.b64decode(uri[len("data:image/png;base64,"):])
    thumb = Image.open(BytesIO(raw))
    # Aspect ratio of 2:1 must be preserved
    w, h = thumb.size
    assert abs(w / h - 2.0) < 0.1


# --- Alpha-preserving animated exports ---

def _rgba_batch(n=2, h=8, w=8):
    """RGBA batch: left half opaque red, right half fully transparent."""
    import torch
    t = torch.zeros((n, h, w, 4), dtype=torch.float32)
    t[..., 0] = 1.0                 # red
    t[..., : , : w // 2, 3] = 1.0   # left half opaque
    return t


def test_animated_webp_preserves_alpha(tmp_path):
    path = str(tmp_path / "clip.webp")
    images.save_animated_webp(_rgba_batch(), path, fps=8)
    from PIL import Image
    with Image.open(path) as img:
        assert img.mode in ("RGBA", "P") and "A" in img.convert("RGBA").getbands()
        rgba = img.convert("RGBA")
        assert rgba.getpixel((7, 0))[3] == 0      # right half transparent
        assert rgba.getpixel((0, 0))[3] == 255    # left half opaque


def test_animated_apng_preserves_alpha(tmp_path):
    path = str(tmp_path / "clip.png")
    images.save_animated_apng(_rgba_batch(), path, fps=8)
    from PIL import Image
    with Image.open(path) as img:
        rgba = img.convert("RGBA")
        assert rgba.getpixel((7, 0))[3] == 0


def test_animated_gif_preserves_alpha(tmp_path):
    path = str(tmp_path / "clip.gif")
    images.save_animated_gif(_rgba_batch(), path, fps=8)
    from PIL import Image
    with Image.open(path) as img:
        assert "transparency" in img.info
        rgba = img.convert("RGBA")
        assert rgba.getpixel((7, 0))[3] == 0


def test_animated_webp_rgb_unchanged(tmp_path):
    # 3-channel input still writes a plain RGB animation (original behavior).
    import torch
    path = str(tmp_path / "clip.webp")
    images.save_animated_webp(torch.zeros((2, 8, 8, 3)), path, fps=8)
    from PIL import Image
    with Image.open(path) as img:
        assert img.convert("RGBA").getpixel((0, 0))[3] == 255

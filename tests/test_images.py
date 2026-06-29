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

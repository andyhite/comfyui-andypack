import json
import os

import torch
from PIL import Image

from andypack import api, images, io, nodes
from andypack.resolve import compute_prompt_hash, pose_image_path, status


def _img(h=2, w=2):
    return torch.zeros((1, h, w, 3), dtype=torch.float32)


def _batch(n, h=2, w=2):
    return torch.zeros((n, h, w, 3), dtype=torch.float32)


def _write_animation(manifest, root, char, anim_id, direction, count):
    out_dir = os.path.join(root, char, anim_id, direction)
    os.makedirs(out_dir, exist_ok=True)
    for i in range(count):
        open(os.path.join(out_dir, io.frame_name(i)), "w").close()
    base_meta = {
        "kind": "animation", "animation": anim_id, "direction": direction,
        "fps": 16, "length": count, "loop": False,
        "manifest_version": manifest["version"],
        "prompt_hash": compute_prompt_hash(manifest, root, char, "animation", anim_id, direction),
    }
    full = io.build_animation_meta(
        base_meta, count=count, start_frame=io.frame_name(0),
        last_frame=io.frame_name(count - 1), seed=1, created_utc="2026-06-29T00:00:00Z",
    )
    io.atomic_write_json(os.path.join(out_dir, "meta.json"), full)


def test_writing_idle_unlocks_punch(manifest, tree):
    root, char = tree.root, tree.char
    tree.pose("base", "EAST").pose("fighting_stance", "EAST")
    assert status(manifest, root, char, "punch", "EAST") == "blocked"

    _write_animation(manifest, root, char, "fighting_stance_idle", "EAST", count=3)

    assert status(manifest, root, char, "fighting_stance_idle", "EAST") == "generated"
    for combat in ("punch", "fighting_stance_entry", "fighting_stance_exit"):
        assert status(manifest, root, char, combat, "EAST") == "ready"


def test_animation_writer_writes_rgba_with_mask(tmp_path):
    frame_count = 3
    animation = {
        "output_dir": str(tmp_path),
        "_meta": {
            "kind": "animation",
            "animation": "idle",
            "direction": "EAST",
            "fps": 16,
            "length": frame_count,
            "loop": False,
            "manifest_version": "1.0",
            "prompt_hash": "sha1:x",
        },
    }
    frames = torch.ones((frame_count, 4, 4, 3))
    mask = torch.zeros((frame_count, 4, 4))
    mask[:, :2, :2] = 1.0
    nodes.AnimationFrameWriter().write(animation, frames, mask=mask)
    for i in range(frame_count):
        fname = os.path.join(str(tmp_path), f"frame_{i:05d}.png")
        with Image.open(fname) as im:
            assert im.mode == "RGBA"
    meta = json.load(open(os.path.join(str(tmp_path), "meta.json")))
    assert meta["has_alpha"] is True


# --- mode-aware REMAINING ----------------------------------------------------- #

def test_animation_writer_reports_zero_remaining_in_target_mode(manifest, tree, monkeypatch):
    monkeypatch.setattr(nodes, "_characters_root", lambda: tree.root)
    tree.pose("base", "EAST").pose("fighting_stance", "EAST")
    images.save_image_png(
        _img(), pose_image_path(tree.root, tree.char, "fighting_stance", "EAST")
    )
    # fighting_stance_idle is the first actionable animation: render it so `punch`
    # (its start/end anchor) becomes selectable.
    idle_dir = os.path.join(tree.root, tree.char, "fighting_stance_idle", "EAST")
    os.makedirs(idle_dir, exist_ok=True)
    for i in range(3):
        images.save_image_png(_img(), os.path.join(idle_dir, io.frame_name(i)))
    (anim,) = nodes.AnimationSweepSelector().select(
        manifest, tree.char, "sweep", True, "", "", ""
    )
    out_dir, remaining = nodes.AnimationFrameWriter().write(anim, _batch(3))
    assert anim["_meta"]["animation"] == "fighting_stance_idle"

    (anim2,) = nodes.AnimationSweepSelector().select(
        manifest, tree.char, "target", True, "", "punch", "EAST"
    )
    out_dir2, remaining2 = nodes.AnimationFrameWriter().write(anim2, _batch(3))
    assert remaining2 == 0  # target mode never continues the loop


def test_animation_writer_reports_live_remaining_mid_sweep(manifest, tree, monkeypatch):
    monkeypatch.setattr(nodes, "_characters_root", lambda: tree.root)
    tree.pose("base", "EAST").pose("fighting_stance", "EAST")
    images.save_image_png(
        _img(), pose_image_path(tree.root, tree.char, "fighting_stance", "EAST")
    )
    (anim,) = nodes.AnimationSweepSelector().select(
        manifest, tree.char, "sweep", True, "", "", ""
    )
    out_dir, remaining = nodes.AnimationFrameWriter().write(anim, _batch(3))
    assert remaining >= 0
    sweep = anim["_sweep"]
    expected = api.remaining_actionable(
        sweep["manifest"], tree.root, sweep["character"], sweep["kind"],
        exclude_root=sweep.get("exclude_root", False),
        category=sweep.get("category"), skip_mirrored=sweep.get("skip_mirrored", False),
    )
    assert remaining == expected

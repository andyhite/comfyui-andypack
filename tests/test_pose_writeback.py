import json
import os

import torch
from PIL import Image

from andypack import api, images, io, nodes, resolve
from andypack.resolve import status


def test_writing_base_sidecar_unlocks_fighting_stance(manifest, tree):
    root, char = tree.root, tree.char
    tree.character()
    assert status(manifest, root, char, "fighting_stance", "EAST") == "blocked"

    # Simulate PoseFrameWriter's write-back for base@EAST: payload then sidecar last.
    out_dir = os.path.join(root, char, "_base")
    os.makedirs(out_dir, exist_ok=True)
    open(os.path.join(out_dir, "EAST.png"), "w").close()  # payload (image bytes elsewhere)
    from andypack.resolve import compute_prompt_hash
    meta = {
        "kind": "pose", "pose": "base", "direction": "EAST",
        "from": manifest["poses"]["base"].get("from"), "image": "EAST.png",
        "manifest_version": manifest["version"],
        "prompt_hash": compute_prompt_hash(manifest, root, char, "pose", "base", "EAST"),
    }
    sidecar = io.build_pose_sidecar(meta, created_utc="2026-06-29T00:00:00Z")
    io.atomic_write_json(os.path.join(out_dir, "EAST.json"), sidecar)

    assert status(manifest, root, char, "base", "EAST") == "generated"
    assert status(manifest, root, char, "fighting_stance", "EAST") == "ready"


def test_pose_writer_writes_rgba_with_mask(tmp_path):
    pose = {
        "output_dir": str(tmp_path),
        "_meta": {"image": "EAST.png", "direction": "EAST", "prompt_hash": "sha1:x"},
    }
    mask = torch.zeros((1, 4, 4))
    mask[:, :2, :2] = 1.0
    nodes.PoseFrameWriter().write(pose, torch.ones((1, 4, 4, 3)), mask=mask)
    with Image.open(os.path.join(str(tmp_path), "EAST.png")) as im:
        assert im.mode == "RGBA"
    sc = json.load(open(os.path.join(str(tmp_path), "EAST.json")))
    assert sc["has_alpha"] is True


# --- mode-aware REMAINING ----------------------------------------------------- #

def test_pose_writer_reports_zero_remaining_in_target_mode(manifest, tree, monkeypatch):
    monkeypatch.setattr(nodes, "_characters_root", lambda: tree.root)
    tree.character()
    tree.pose("base", "EAST")
    images.save_image_png(_placeholder(), resolve.pose_image_path(
        tree.root, tree.char, "base", "EAST"))
    (pose,) = nodes.PoseSweepSelector().select(
        manifest, tree.char, "target", True, False, "", "fighting_stance", "EAST"
    )
    out_dir, remaining = nodes.PoseFrameWriter().write(pose, _placeholder())
    assert remaining == 0  # target mode never continues the loop


def test_pose_writer_reports_live_remaining_mid_sweep(manifest, tree, monkeypatch):
    monkeypatch.setattr(nodes, "_characters_root", lambda: tree.root)
    # Character exists with a persisted reference, but no base direction rendered
    # yet: with include_base, base is the next actionable root pose.
    tree.character()
    images.save_image_png(_placeholder(), resolve.reference_image_path(tree.root, tree.char))
    (pose,) = nodes.PoseSweepSelector().select(
        manifest, tree.char, "sweep", True, True, "", "", ""
    )
    out_dir, remaining = nodes.PoseFrameWriter().write(pose, _placeholder())
    assert remaining >= 0
    # The writer's count matches a direct post-write call with the same scope.
    sweep = pose["_sweep"]
    expected = api.remaining_actionable(
        sweep["manifest"], tree.root, sweep["character"], sweep["kind"],
        exclude_root=sweep.get("exclude_root", False),
        category=sweep.get("category"), skip_mirrored=sweep.get("skip_mirrored", False),
    )
    assert remaining == expected


def _placeholder():
    return images.empty_image()

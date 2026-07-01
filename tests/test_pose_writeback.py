import json
import os

import torch
from PIL import Image

from andypack import io, nodes
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

import os

from andypack import io
from andypack.resolve import status


def test_writing_base_sidecar_unlocks_fighting_stance(manifest, tree):
    root, char = tree.root, tree.char
    tree.concept()
    assert status(manifest, root, char, "fighting_stance", "E") == "blocked"

    # Simulate PoseFrameWriter's write-back for base@E: payload then sidecar last.
    out_dir = os.path.join(root, char, "_base")
    os.makedirs(out_dir, exist_ok=True)
    open(os.path.join(out_dir, "E.png"), "w").close()  # payload (image bytes elsewhere)
    from andypack.resolve import compute_prompt_hash
    meta = {
        "kind": "pose", "pose": "base", "direction": "E",
        "from": manifest["poses"]["base"]["from"], "image": "E.png",
        "manifest_version": manifest["version"],
        "prompt_hash": compute_prompt_hash(manifest, root, char, "pose", "base", "E"),
    }
    sidecar = io.build_pose_sidecar(meta, created_utc="2026-06-29T00:00:00Z")
    io.atomic_write_json(os.path.join(out_dir, "E.json"), sidecar)

    assert status(manifest, root, char, "base", "E") == "generated"
    assert status(manifest, root, char, "fighting_stance", "E") == "ready"

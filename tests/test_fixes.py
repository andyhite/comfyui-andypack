import os

from andypack import resolve


def _render_pose(root, char, pose, direction, manifest):
    # minimal: write a pose png + sidecar so node_complete() is True
    base = os.path.join(root, char, f"_{pose}")
    os.makedirs(base, exist_ok=True)
    open(os.path.join(base, f"{direction}.png"), "wb").close()
    r = resolve.resolve_pose(manifest, root, char, pose, direction)
    from andypack import io
    io.atomic_write_json(os.path.join(base, f"{direction}.json"),
                         io.build_pose_sidecar(r["meta"], created_utc="2026-01-01T00:00:00Z"))


def test_swapping_anchor_ref_restales_animation(tmp_path):
    root = str(tmp_path)
    char = "hero"
    manifest = {
        "version": 1,
        "poses": {
            "base": {"directions": {"EAST": {}}},
            "poseA": {"from": {"ref": "base"}, "directions": {"EAST": {}}},
            "poseB": {"from": {"ref": "base"}, "directions": {"EAST": {}}},
        },
        "animations": {
            "walk": {"start_from": {"ref": "poseA"}, "directions": {"EAST": {}},
                     "length": 5, "fps": 8, "width": 16, "height": 16},
        },
        "defaults": {},
    }
    for p in ("base", "poseA", "poseB"):
        _render_pose(root, char, p, "EAST", manifest)
    # render walk against poseA
    rwalk = resolve.resolve_animation(manifest, root, char, "walk", "EAST")
    from andypack import io
    adir = os.path.join(root, char, "walk", "EAST")
    os.makedirs(adir, exist_ok=True)
    for i in range(5):
        open(os.path.join(adir, io.frame_name(i)), "wb").close()
    io.atomic_write_json(os.path.join(adir, "meta.json"),
        io.build_animation_meta(rwalk["meta"], count=5, start_frame=io.frame_name(0),
                                last_frame=io.frame_name(4), seed=0, created_utc="2026-01-01T00:00:00Z"))
    assert resolve.outdated(manifest, root, char, "walk", "EAST") is False
    # swap the anchor to poseB (already rendered, prompts unchanged)
    manifest["animations"]["walk"]["start_from"] = {"ref": "poseB"}
    assert resolve.outdated(manifest, root, char, "walk", "EAST") is True

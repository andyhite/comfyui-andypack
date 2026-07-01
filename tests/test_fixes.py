import os

import pytest

from andypack import api
from andypack import images
from andypack import resolve
from andypack import io
from andypack import nodes
from andypack.manifest import validate_manifest, ManifestError


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


def test_effective_cache_key_is_content_derived():
    from andypack.resolve import _effective_cache_key
    a = {"version": 1, "poses": {}, "animations": {}, "globals": {"x": 1}}
    b = {"version": 1, "poses": {}, "animations": {}, "globals": {"x": 2}}
    assert _effective_cache_key(a, {}) != _effective_cache_key(b, {})


def test_effective_manifest_reflects_base_edit_with_overlay(tmp_path):
    root = str(tmp_path)
    char = "hero"
    os.makedirs(os.path.join(root, char), exist_ok=True)
    # character overlay so effective_manifest does the merge+cache path
    io.atomic_write_json(os.path.join(root, char, "character.json"),
        {"poses": {"wave": {"from": {"ref": "base"}, "directions": {"EAST": {}}}}})
    resolve.invalidate_character(root, char)
    base1 = {"version": 1, "poses": {"base": {"directions": {"EAST": {}}}},
             "animations": {}, "defaults": {},
             "globals": {"pose": {"positive_prompt": "v1"}}}
    eff1 = resolve.effective_manifest(base1, root, char)
    assert eff1["globals"]["pose"]["positive_prompt"] == "v1"
    # a DIFFERENT base object (simulating a reload) with edited content
    base2 = {"version": 1, "poses": {"base": {"directions": {"EAST": {}}}},
             "animations": {}, "defaults": {},
             "globals": {"pose": {"positive_prompt": "v2"}}}
    eff2 = resolve.effective_manifest(base2, root, char)
    assert eff2["globals"]["pose"]["positive_prompt"] == "v2"


def test_mirror_writer_is_changed_tracks_source_mtime(tmp_path, monkeypatch):
    root = str(tmp_path)
    char = "hero"
    manifest = {"version": 1, "mirror_map": {"WEST": "EAST"},
                "poses": {"p": {"from": {"ref": "base"}, "directions": {"EAST": {}, "WEST": {}}},
                          "base": {"directions": {"EAST": {}}}},
                "animations": {}, "defaults": {}}
    monkeypatch.setattr(nodes, "_characters_root", lambda: root)
    src = resolve.pose_image_path(root, char, "p", "EAST")
    os.makedirs(os.path.dirname(src), exist_ok=True)
    with open(src, "wb") as fh:
        fh.write(b"a")
    fp1 = nodes.MirrorFrameWriter.IS_CHANGED(manifest, char, "pose", "p", "WEST")
    os.utime(src, (10**9, 10**9))  # bump mtime
    fp2 = nodes.MirrorFrameWriter.IS_CHANGED(manifest, char, "pose", "p", "WEST")
    assert fp1 != fp2


def test_safe_manifest_path_rejects_traversal(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "manifests_dir", lambda: str(tmp_path))
    assert api.safe_manifest_path("../../etc/passwd.json") is None
    assert api.safe_manifest_path("/etc/passwd.json") is None
    assert api.safe_manifest_path("default.json") is not None  # bare name ok


def test_unsafe_entity_id_rejected():
    m = {"version": 1, "animations": {"../escape": {"start_from": {"ref": "base"},
         "directions": {"EAST": {}}}},
         "poses": {"base": {"directions": {"EAST": {}}}}, "defaults": {}}
    with pytest.raises(ManifestError):
        validate_manifest(m)


def test_unsafe_direction_name_rejected():
    m = {"version": 1, "poses": {"base": {"directions": {"../x": {}}}},
         "animations": {}, "defaults": {}}
    with pytest.raises(ManifestError):
        validate_manifest(m)


def test_missing_gen_params_rejected():
    m = {"version": 1, "poses": {"base": {"directions": {"EAST": {}}}},
         "animations": {"walk": {"start_from": {"ref": "base"}, "directions": {"EAST": {}}}},
         "defaults": {}}  # no width/height/length/fps anywhere
    with pytest.raises(ManifestError):
        validate_manifest(m)


def test_nonpositive_gen_params_rejected():
    m = {"version": 1, "poses": {"base": {"directions": {"EAST": {}}}},
         "animations": {"walk": {"start_from": {"ref": "base"}, "directions": {"EAST": {}},
            "length": -3, "fps": 8, "width": 16, "height": 16}},
         "defaults": {}}
    with pytest.raises(ManifestError):
        validate_manifest(m)


def test_playback_not_loopable_when_anchor_frames_differ(tmp_path):
    root = str(tmp_path)
    char = "hero"
    manifest = {"version": 1,
        "poses": {"base": {"directions": {"EAST": {}}}},
        "animations": {
            "src": {"start_from": {"ref": "base"}, "directions": {"EAST": {}},
                    "length": 4, "fps": 8, "width": 16, "height": 16},
            "clip": {"start_from": {"ref": "src"}, "end_at": {"ref": "src"},
                     "directions": {"EAST": {}}, "length": 4, "fps": 8, "width": 16, "height": 16}},
        "defaults": {}}
    # render src (4 distinct frames so start_frame != last_frame) and clip
    for anim, n in (("src", 4), ("clip", 4)):
        d = os.path.join(root, char, anim, "EAST")
        os.makedirs(d, exist_ok=True)
        for i in range(n):
            open(os.path.join(d, io.frame_name(i)), "wb").close()
        r = resolve.resolve_animation(manifest, root, char, anim, "EAST")
        io.atomic_write_json(os.path.join(d, "meta.json"),
            io.build_animation_meta(r["meta"], count=n, start_frame=io.frame_name(0),
                last_frame=io.frame_name(n - 1), seed=0, created_utc="2026-01-01T00:00:00Z"))
    segs = resolve.playback_segments(manifest, root, char, "clip", "EAST", loops=3, fps=8)
    action = [s for s in segs if s.get("dir", "").endswith(os.path.join("clip", "EAST"))][0]
    assert action["repeat"] == 1  # start_image (src.last) != end_image (src.first)


def test_animation_writer_rejects_empty_sentinel(tmp_path, monkeypatch):
    monkeypatch.setattr(nodes, "_characters_root", lambda: str(tmp_path))
    w = nodes.AnimationFrameWriter()
    anim = {"output_dir": str(tmp_path / "a"),
            "_meta": {"prompt_hash": "sha1:x", "loop": False}}
    with pytest.raises(RuntimeError):
        w.write(anim, images.empty_image())

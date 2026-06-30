import json
import os

import pytest
import torch

from andypack import images, nodes, resolve


def _img(h=2, w=2):
    return torch.zeros((1, h, w, 3), dtype=torch.float32)


def _batch(n, h=2, w=2):
    return torch.zeros((n, h, w, 3), dtype=torch.float32)


def _pose_dict(meta, output_dir, image=None):
    """An ANIM_POSE dict as the selector emits it (the writer reads `output_dir`
    and the bundled `_meta`)."""
    return {
        "source_image": image, "pose_reference": None, "positive": "", "negative": "",
        "output_dir": output_dir, "_meta": meta,
    }


def _anim_dict(meta, output_dir):
    """An ANIM_ANIMATION dict (writer reads `output_dir` + the bundled `_meta`)."""
    return {
        "start_image": None, "end_image": None, "positive": "", "negative": "",
        "is_fflf": False, "length": meta.get("length", 0), "fps": meta.get("fps", 0),
        "output_dir": output_dir, "_meta": meta,
    }


# --- re-render discipline ---------------------------------------------------- #

def test_animation_rewrite_clears_stale_frames(tmp_path):
    out = str(tmp_path / "punch" / "EAST")
    meta = {
        "kind": "animation", "animation": "punch", "direction": "EAST",
        "fps": 16, "length": 5, "loop": False, "manifest_version": 1,
        "prompt_hash": "sha1:abc",
    }
    writer = nodes.AnimationFrameWriter()
    writer.write(_anim_dict(meta, out), _batch(5))
    assert sum(n.startswith("frame_") for n in os.listdir(out)) == 5

    # A shorter re-render must not leave the old frame_00003/00004 behind.
    writer.write(_anim_dict(dict(meta, length=3), out), _batch(3))
    frames = sorted(n for n in os.listdir(out) if n.startswith("frame_"))
    assert frames == ["frame_00000.png", "frame_00001.png", "frame_00002.png"]
    import json
    full = json.loads(open(os.path.join(out, "meta.json")).read())
    assert full["frames"]["count"] == 3
    assert full["last_frame"] == "frame_00002.png"


def test_animation_writer_drops_last_frame_for_loop(tmp_path):
    out = str(tmp_path / "spin" / "EAST")
    meta = {
        "kind": "animation", "animation": "spin", "direction": "EAST",
        "fps": 16, "length": 5, "loop": True, "manifest_version": 1,
        "prompt_hash": "sha1:abc",
    }
    nodes.AnimationFrameWriter().write(_anim_dict(meta, out), _batch(5))
    frames = sorted(n for n in os.listdir(out) if n.startswith("frame_"))
    assert frames == ["frame_00000.png", "frame_00001.png", "frame_00002.png", "frame_00003.png"]
    full = json.loads(open(os.path.join(out, "meta.json")).read())
    assert full["frames"]["count"] == 4  # duplicate closing frame dropped


def test_pose_rewrite_replaces_sidecar(tmp_path):
    out = str(tmp_path / "_base")
    meta = {
        "kind": "pose", "pose": "base", "direction": "EAST",
        "from": None, "image": "EAST.png",
        "manifest_version": 1, "prompt_hash": "sha1:one",
    }
    writer = nodes.PoseFrameWriter()
    writer.write(_pose_dict(meta, out), _img())
    assert os.path.exists(os.path.join(out, "EAST.png"))

    writer.write(_pose_dict(dict(meta, prompt_hash="sha1:two"), out), _img())
    import json
    side = json.loads(open(os.path.join(out, "EAST.json")).read())
    assert side["prompt_hash"] == "sha1:two"


# --- selector IS_CHANGED fingerprint ---------------------------------------- #

def test_pose_selector_is_changed_tracks_dependency_render(manifest, tree, monkeypatch):
    monkeypatch.setattr(nodes, "_characters_root", lambda: tree.root)
    # The character layer is opt-in, so the prompt must reference it to ripple.
    manifest["poses"]["fighting_stance"]["positive_prompt"] += " {character_prompt}"
    tree.pose("base", "EAST")  # base (root) rendered -> fighting_stance selectable
    before = nodes.CharacterPoseSelector.IS_CHANGED(
        manifest, tree.char, "", "fighting_stance", "EAST"
    )
    # Editing the character layer must move the fingerprint (prompt_hash folds it in).
    tree.character(positive_prompt="a brand new character line")
    after = nodes.CharacterPoseSelector.IS_CHANGED(
        manifest, tree.char, "", "fighting_stance", "EAST"
    )
    assert before != after


def test_animation_selector_is_changed_tracks_anchor_render(manifest, tree, monkeypatch):
    monkeypatch.setattr(nodes, "_characters_root", lambda: tree.root)
    tree.pose("base", "EAST").pose("fighting_stance", "EAST")
    before = nodes.CharacterAnimationSelector.IS_CHANGED(
        manifest, tree.char, "", "punch", "EAST"
    )  # idle not rendered -> blocked, no anchor image
    tree.animation("fighting_stance_idle", "EAST", frames=3)
    after = nodes.CharacterAnimationSelector.IS_CHANGED(
        manifest, tree.char, "", "punch", "EAST"
    )  # idle rendered -> selectable, anchors resolve
    assert before != after


def test_selector_is_changed_volatile_without_character(manifest, monkeypatch):
    # No character chosen -> always re-run (NaN != NaN), never a stale cache hit.
    token = nodes.CharacterPoseSelector.IS_CHANGED(manifest, "", "", "base", "EAST")
    assert token != token


# --- new utility / writer nodes --------------------------------------------- #

def test_animation_writer_rejects_empty_batch(tmp_path):
    # An empty frame batch must raise, not write a count=0 "complete" meta with a
    # negative-index last_frame that downstream anchors would chase to a missing file.
    out = str(tmp_path / "punch" / "EAST")
    meta = {
        "kind": "animation", "animation": "punch", "direction": "EAST",
        "fps": 16, "length": 0, "loop": False, "manifest_version": 1,
        "prompt_hash": "sha1:abc",
    }
    with pytest.raises(RuntimeError, match="empty frame batch"):
        nodes.AnimationFrameWriter().write(_anim_dict(meta, out), _batch(0))
    assert not os.path.exists(os.path.join(out, "meta.json"))


def test_animation_writer_empty_batch_keeps_prior_render(tmp_path):
    # The empty-batch guard fires before clearing, so a prior good render survives.
    out = str(tmp_path / "punch" / "EAST")
    meta = {
        "kind": "animation", "animation": "punch", "direction": "EAST",
        "fps": 16, "length": 3, "loop": False, "manifest_version": 1,
        "prompt_hash": "sha1:abc",
    }
    nodes.AnimationFrameWriter().write(_anim_dict(meta, out), _batch(3))
    with pytest.raises(RuntimeError):
        nodes.AnimationFrameWriter().write(_anim_dict(meta, out), _batch(0))
    frames = sorted(n for n in os.listdir(out) if n.startswith("frame_"))
    assert frames == ["frame_00000.png", "frame_00001.png", "frame_00002.png"]


def test_manifest_lint_reports_findings(manifest):
    manifest["poses"]["base"]["directions"]["UP"] = {}  # unknown direction
    (report,) = nodes.ManifestLint().lint(manifest)
    assert "UP" in report


def test_manifest_lint_clean(manifest):
    (report,) = nodes.ManifestLint().lint(manifest)
    assert report.startswith("OK")


def test_coverage_report_node(manifest, tree, monkeypatch):
    monkeypatch.setattr(nodes, "_characters_root", lambda: tree.root)
    tree.character()
    report, blob = nodes.CoverageReport().report(manifest, tree.char)
    assert "base" in report
    assert json.loads(blob)["total"] > 0


def test_merged_prompt_report_node(manifest, tree, monkeypatch):
    monkeypatch.setattr(nodes, "_characters_root", lambda: tree.root)
    report, blob = nodes.MergedPromptReport().report(manifest, nodes._NO_CHARACTER)
    assert "[pose] base @ EAST" in report
    data = json.loads(blob)
    assert any(r["id"] == "punch" and r["negative"] for r in data)


def test_regen_queue_node(manifest, tree, monkeypatch):
    monkeypatch.setattr(nodes, "_characters_root", lambda: tree.root)
    tree.character()
    text, count = nodes.RegenQueue().build(manifest, tree.char)
    assert count >= 1
    assert "base@EAST" in text


def test_mirror_writer_pose(manifest, tree, monkeypatch):
    monkeypatch.setattr(nodes, "_characters_root", lambda: tree.root)
    tree.character()
    images.save_image_png(_img(), resolve.pose_image_path(tree.root, tree.char, "base", "EAST"))
    (out_dir,) = nodes.MirrorFrameWriter().write(manifest, tree.char, "pose", "base", "WEST")
    assert os.path.exists(resolve.pose_image_path(tree.root, tree.char, "base", "WEST"))
    side = json.loads(open(resolve.pose_sidecar_path(tree.root, tree.char, "base", "WEST")).read())
    assert side["mirrored_from"]["direction"] == "EAST"
    assert side["render_id"].startswith("rid:")


def test_mirror_writer_animation(manifest, tree, monkeypatch):
    monkeypatch.setattr(nodes, "_characters_root", lambda: tree.root)
    # real frame PNGs for the EAST source, plus its meta (conftest writes meta)
    tree.animation("fighting_stance_idle", "EAST", frames=3)
    src_dir = resolve.animation_frame_dir(tree.root, tree.char, "fighting_stance_idle", "EAST")
    for i in range(3):
        images.save_image_png(_img(), os.path.join(src_dir, f"frame_{i:05d}.png"))
    (out_dir,) = nodes.MirrorFrameWriter().write(
        manifest, tree.char, "animation", "fighting_stance_idle", "WEST"
    )
    meta = json.loads(open(resolve.animation_meta_path(
        tree.root, tree.char, "fighting_stance_idle", "WEST")).read())
    assert meta["frames"]["count"] == 3
    assert meta["mirrored_from"]["direction"] == "EAST"


def test_animation_selector_outputs_length_and_fps(manifest, tree, monkeypatch):
    monkeypatch.setattr(nodes, "_characters_root", lambda: tree.root)
    tree.pose("base", "EAST").pose("fighting_stance", "EAST").animation(
        "fighting_stance_idle", "EAST", frames=3
    )
    idle = resolve.animation_frame_dir(tree.root, tree.char, "fighting_stance_idle", "EAST")
    for i in range(3):  # real PNGs so the anchor frames load
        images.save_image_png(_img(), os.path.join(idle, f"frame_{i:05d}.png"))
    (anim,) = nodes.CharacterAnimationSelector().select(manifest, tree.char, "", "punch", "EAST")
    assert anim["length"] == manifest["animations"]["punch"]["length"]  # 21
    assert anim["fps"] == manifest["defaults"]["fps"]                   # 16 (punch inherits default)
    assert anim["_meta"]["animation"] == "punch"


# --- dict outputs + getters -------------------------------------------------- #

def test_pose_selector_returns_single_dict(manifest, tree, monkeypatch):
    monkeypatch.setattr(nodes, "_characters_root", lambda: tree.root)
    # Select a non-root pose: base (root) is created via the Character Creator.
    tree.pose("base", "EAST")  # base rendered so fighting_stance has a source
    # Write a real PNG at base@EAST so the selector can load it as the source image.
    images.save_image_png(_img(), resolve.pose_image_path(tree.root, tree.char, "base", "EAST"))
    (pose,) = nodes.CharacterPoseSelector().select(manifest, tree.char, "", "fighting_stance", "EAST")
    # Leaf outputs are selectable; the resolver meta rides bundled under `_meta`
    # (not a leaf output) and the image rides along as a tensor.
    assert isinstance(pose["positive"], str)
    assert pose["source_image"].shape[0] == 1
    assert pose["_meta"]["pose"] == "fighting_stance" and pose["_meta"]["image"] == "EAST.png"
    # Drift guard: the public (non-`_meta`) keys are exactly the leaf-output keys.
    assert sorted(k for k in pose if k != "_meta") == nodes.POSE_OUTPUT_KEYS


def test_animation_selector_returns_single_dict(manifest, tree, monkeypatch):
    monkeypatch.setattr(nodes, "_characters_root", lambda: tree.root)
    tree.pose("base", "EAST").pose("fighting_stance", "EAST").animation(
        "fighting_stance_idle", "EAST", frames=3
    )
    idle = resolve.animation_frame_dir(tree.root, tree.char, "fighting_stance_idle", "EAST")
    for i in range(3):
        images.save_image_png(_img(), os.path.join(idle, f"frame_{i:05d}.png"))
    (anim,) = nodes.CharacterAnimationSelector().select(manifest, tree.char, "", "punch", "EAST")
    assert anim["_meta"]["animation"] == "punch"
    assert isinstance(anim["is_fflf"], bool)
    assert isinstance(anim["length"], int) and isinstance(anim["fps"], int)
    assert anim["start_image"].shape[0] == 1 and anim["end_image"].shape[0] == 1
    assert sorted(k for k in anim if k != "_meta") == nodes.ANIMATION_OUTPUT_KEYS


def test_pose_unpack_forwards_dict_and_fans_out_typed_outputs():
    pose = {"positive": "hello", "negative": "blurry", "source_image": _img(),
            "pose_reference": _img(), "output_dir": "/x", "_meta": {}}
    passthrough, img, manikin, pos, neg, out_dir = nodes.PoseUnpack().unpack(pose)
    assert nodes.PoseUnpack.RETURN_NAMES == (
        "POSE", "SOURCE_IMAGE", "POSE_REFERENCE", "POSITIVE_PROMPT", "NEGATIVE_PROMPT", "OUTPUT_DIR"
    )
    assert passthrough is pose  # whole POSE forwarded on, unchanged
    assert img.shape[0] == 1 and manikin.shape[0] == 1
    assert (pos, neg, out_dir) == ("hello", "blurry", "/x")


def test_animation_unpack_forwards_dict_and_fans_out_typed_outputs():
    anim = {"start_image": _img(), "end_image": _img(), "positive": "p", "negative": "n",
            "is_fflf": True, "length": 21, "fps": 16, "output_dir": "/a", "_meta": {}}
    passthrough, start, end, pos, neg, fflf, length, fps, out_dir = (
        nodes.AnimationUnpack().unpack(anim)
    )
    assert nodes.AnimationUnpack.RETURN_NAMES == (
        "ANIMATION", "START_IMAGE", "END_IMAGE", "POSITIVE_PROMPT", "NEGATIVE_PROMPT",
        "IS_FFLF", "LENGTH", "FPS", "OUTPUT_DIR",
    )
    assert passthrough is anim  # whole ANIMATION forwarded on, unchanged
    assert start.shape[0] == 1 and end.shape[0] == 1
    assert (pos, neg, fflf, length, fps, out_dir) == ("p", "n", True, 21, 16, "/a")


def test_unpack_outputs_cover_selector_leaf_keys():
    # Every leaf key a selector emits must have an Unpack output (and vice versa);
    # the declared types line up slot-for-slot after the leading passthrough.
    assert {k for k, _n in nodes._POSE_UNPACK} == set(nodes.POSE_OUTPUT_KEYS)
    assert {k for k, _n in nodes._ANIMATION_UNPACK} == set(nodes.ANIMATION_OUTPUT_KEYS)
    assert nodes.PoseUnpack.RETURN_TYPES[0] == "ANIM_POSE"
    assert nodes.AnimationUnpack.RETURN_TYPES[0] == "ANIM_ANIMATION"
    assert len(nodes.PoseUnpack.RETURN_TYPES) == len(nodes._POSE_UNPACK) + 1
    assert len(nodes.AnimationUnpack.RETURN_TYPES) == len(nodes._ANIMATION_UNPACK) + 1


def test_animation_playback_chains_and_loops(manifest, tree, monkeypatch):
    monkeypatch.setattr(nodes, "_characters_root", lambda: tree.root)
    tree.pose("base", "EAST").pose("fighting_stance", "EAST").animation(
        "fighting_stance_idle", "EAST", frames=3
    ).animation("punch", "EAST", frames=3)
    # Tree writes empty frame files; the playback node loads pixels, so write real
    # PNGs for the idle (chained) and punch (action) frames.
    for anim in ("fighting_stance_idle", "punch"):
        d = resolve.animation_frame_dir(tree.root, tree.char, anim, "EAST")
        for i in range(3):
            images.save_image_png(_img(), os.path.join(d, f"frame_{i:05d}.png"))

    out = nodes.AnimationPlayback().play(manifest, tree.char, "", "punch", "EAST", 3)
    frames, fps = out["result"]
    assert fps == manifest["defaults"]["fps"]  # punch inherits the default fps (16)
    # idle(3) + punch(3*3, minus dropped first & last seam = 7) + idle(3) = 13
    assert frames.shape[0] == 3 + 7 + 3


def test_animation_playback_raises_when_unrendered(manifest, tree, monkeypatch):
    # Nothing rendered: the action dir has no frames and the (incomplete) deps are
    # skipped, so assemble_playback yields the empty sentinel. play() must raise,
    # not emit a bogus 1x1 black frame (the old `shape[0] == 0` guard never fired).
    monkeypatch.setattr(nodes, "_characters_root", lambda: tree.root)
    with pytest.raises(RuntimeError, match="no rendered frames"):
        nodes.AnimationPlayback().play(manifest, tree.char, "", "punch", "EAST", 1)


def test_leaf_output_keys_exclude_private_meta():
    assert "_meta" not in nodes.POSE_OUTPUT_KEYS
    assert "_meta" not in nodes.ANIMATION_OUTPUT_KEYS


def test_mirror_writer_rejects_unmapped_direction(manifest, tree, monkeypatch):
    monkeypatch.setattr(nodes, "_characters_root", lambda: tree.root)
    tree.character()
    try:
        nodes.MirrorFrameWriter().write(manifest, tree.char, "pose", "base", "EAST")
    except RuntimeError as e:
        assert "mirror_map" in str(e)
    else:
        raise AssertionError("expected RuntimeError for an unmapped direction")


def test_mirror_animation_rejects_frameless_source(manifest, tree, monkeypatch):
    # Source meta survives but its frame PNGs are gone (cleared / partially
    # deleted). Mirroring it would write a count=0 "complete" meta whose
    # start/last_frame point at a nonexistent frame_00000.png. Must raise instead.
    monkeypatch.setattr(nodes, "_characters_root", lambda: tree.root)
    tree.animation("fighting_stance_idle", "EAST", frames=3)
    src_dir = resolve.animation_frame_dir(tree.root, tree.char, "fighting_stance_idle", "EAST")
    for n in [f for f in os.listdir(src_dir) if f.startswith("frame_")]:
        os.remove(os.path.join(src_dir, n))
    with pytest.raises(RuntimeError, match="no frames"):
        nodes.MirrorFrameWriter().write(
            manifest, tree.char, "animation", "fighting_stance_idle", "WEST"
        )
    assert not os.path.exists(
        resolve.animation_meta_path(tree.root, tree.char, "fighting_stance_idle", "WEST")
    )


def test_pose_selector_rejects_unknown_id(manifest, tree, monkeypatch):
    # A stale/renamed serialized id must give a friendly error, not a raw KeyError.
    monkeypatch.setattr(nodes, "_characters_root", lambda: tree.root)
    tree.character()
    with pytest.raises(RuntimeError, match="unknown pose"):
        nodes.CharacterPoseSelector().select(manifest, tree.char, "", "ghost_pose", "EAST")


def test_animation_selector_rejects_unknown_id(manifest, tree, monkeypatch):
    monkeypatch.setattr(nodes, "_characters_root", lambda: tree.root)
    tree.character()
    with pytest.raises(RuntimeError, match="unknown animation"):
        nodes.CharacterAnimationSelector().select(manifest, tree.char, "", "ghost_anim", "EAST")


def test_animation_selector_tolerates_missing_start_anchor(manifest, tree, monkeypatch):
    # A selectable animation whose start_from dep is a complete animation whose meta
    # lacks 'last_frame' resolves start_image to None; the selector must fall back to
    # the empty sentinel, not call load_image_tensor(None).
    monkeypatch.setattr(nodes, "_characters_root", lambda: tree.root)
    tree.pose("base", "EAST")
    # Real PNG for the end_at -> base anchor (the conftest touch leaves it empty).
    images.save_image_png(_img(), resolve.pose_image_path(tree.root, tree.char, "base", "EAST"))
    tree.animation("fighting_stance_idle", "EAST", frames=3)
    # Real frames so the dep is complete + anchors could load, but drop last_frame.
    idle = resolve.animation_frame_dir(tree.root, tree.char, "fighting_stance_idle", "EAST")
    for i in range(3):
        images.save_image_png(_img(), os.path.join(idle, f"frame_{i:05d}.png"))
    meta_path = resolve.animation_meta_path(tree.root, tree.char, "fighting_stance_idle", "EAST")
    meta = json.loads(open(meta_path).read())
    del meta["last_frame"]
    with open(meta_path, "w") as fh:
        json.dump(meta, fh)
    # fighting_stance_exit: start_from -> fighting_stance_idle (anim), end_at -> base.
    (anim,) = nodes.CharacterAnimationSelector().select(
        manifest, tree.char, "", "fighting_stance_exit", "EAST"
    )
    assert images.is_empty(anim["start_image"])  # empty sentinel, no crash




# --- CharacterCreator + pose_reference -------------------------------------- #

def test_character_creator_writes_character_json_without_provenance(manifest, tmp_path, monkeypatch):
    root = str(tmp_path)
    monkeypatch.setattr(nodes, "_characters_root", lambda: root)
    (pose,) = nodes.CharacterCreator().create(
        manifest, _img(), "Cortex", "EAST",
        character_positive="a brave hero", character_negative="blurry",
    )
    data = json.load(open(os.path.join(root, "cortex", "character.json")))
    assert data == {"positive_prompt": "a brave hero", "negative_prompt": "blurry"}
    assert pose["output_dir"].endswith(os.path.join("cortex", "_base"))
    assert pose["_meta"]["pose"] == "base" and pose["_meta"]["direction"] == "EAST"


def test_character_creator_attaches_manikin_as_pose_reference(manifest, tmp_path, monkeypatch):
    monkeypatch.setattr(nodes, "_characters_root", lambda: str(tmp_path))
    (pose,) = nodes.CharacterCreator().create(manifest, _img(), "cortex", "EAST")
    # The manikin rides along as a real (non-empty) second reference image.
    assert pose["pose_reference"] is not None
    assert not images.is_empty(pose["pose_reference"])


def test_character_creator_rejects_unknown_direction(manifest, tmp_path, monkeypatch):
    monkeypatch.setattr(nodes, "_characters_root", lambda: str(tmp_path))
    with pytest.raises(RuntimeError, match="direction"):
        nodes.CharacterCreator().create(manifest, _img(), "cortex", "UP")


def test_pose_selector_rejects_root_pose(manifest, tree, monkeypatch):
    monkeypatch.setattr(nodes, "_characters_root", lambda: tree.root)
    tree.pose("base", "EAST")
    with pytest.raises(RuntimeError, match="root pose"):
        nodes.CharacterPoseSelector().select(manifest, tree.char, "", "base", "EAST")


def test_pose_selector_sets_empty_pose_reference(manifest, tree, monkeypatch):
    monkeypatch.setattr(nodes, "_characters_root", lambda: tree.root)
    tree.pose("base", "EAST")
    images.save_image_png(_img(), resolve.pose_image_path(tree.root, tree.char, "base", "EAST"))
    (pose,) = nodes.CharacterPoseSelector().select(manifest, tree.char, "", "fighting_stance", "EAST")
    assert images.is_empty(pose["pose_reference"])

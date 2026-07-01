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
    with pytest.raises(RuntimeError, match="empty or 1x1 sentinel frame batch"):
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


def test_state_machine_report_node(manifest, monkeypatch, tmp_path):
    monkeypatch.setattr(nodes, "_characters_root", lambda: str(tmp_path))
    report, blob = nodes.StateMachineReport().report(manifest, nodes._NO_CHARACTER)
    assert "state machine" in report.lower()
    data = json.loads(blob)
    assert "states" in data and "transitions" in data
    # The fixture manifest has animations (punch, fighting_stance_idle, etc.)
    # so the state machine must contain transitions.
    assert len(data["transitions"]) > 0


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
    out_dir, _ = nodes.MirrorFrameWriter().write(manifest, tree.char, "pose", "base", "WEST")
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
    out_dir, _ = nodes.MirrorFrameWriter().write(
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
    # Generation params for the WanFirstLastFrameToVideo node ride along as wireable
    # INT/FLOAT outputs (from the manifest defaults / per-animation override).
    assert isinstance(anim["width"], int) and isinstance(anim["height"], int)
    assert isinstance(anim["shift"], float)
    assert anim["start_image"].shape[0] == 1 and anim["end_image"].shape[0] == 1
    assert sorted(k for k in anim if k != "_meta") == nodes.ANIMATION_OUTPUT_KEYS


def test_pose_unpack_forwards_dict_and_fans_out_typed_outputs():
    pose = {"positive": "hello", "negative": "blurry", "source_image": _img(),
            "pose_reference": _img(), "output_dir": "/x", "_meta": {}}
    passthrough, img, manikin, pos, neg, out_dir, has_ref = nodes.PoseUnpack().unpack(pose)
    assert nodes.PoseUnpack.RETURN_NAMES == (
        "POSE", "SOURCE_IMAGE", "POSE_REFERENCE", "POSITIVE_PROMPT", "NEGATIVE_PROMPT",
        "OUTPUT_DIR", "HAS_POSE_REFERENCE",
    )
    assert passthrough is pose  # whole POSE forwarded on, unchanged
    assert img.shape[0] == 1 and manikin.shape[0] == 1
    assert (pos, neg, out_dir) == ("hello", "blurry", "/x")
    assert has_ref is True  # pose_reference is a real image (a manikin-driven pose)
    # a derived pose (empty sentinel reference) reports False
    pose2 = {**pose, "pose_reference": images.empty_image()}
    assert nodes.PoseUnpack().unpack(pose2)[-1] is False


def test_animation_unpack_forwards_dict_and_fans_out_typed_outputs():
    anim = {"start_image": _img(), "end_image": _img(), "positive": "p", "negative": "n",
            "is_fflf": True, "length": 21, "fps": 16, "width": 832, "height": 480,
            "shift": 3.0, "output_dir": "/a", "_meta": {}}
    (passthrough, start, end, pos, neg, fflf, length, fps,
     width, height, shift, out_dir) = nodes.AnimationUnpack().unpack(anim)
    assert nodes.AnimationUnpack.RETURN_NAMES == (
        "ANIMATION", "START_IMAGE", "END_IMAGE", "POSITIVE_PROMPT", "NEGATIVE_PROMPT",
        "IS_FFLF", "LENGTH", "FPS", "WIDTH", "HEIGHT", "SHIFT", "OUTPUT_DIR",
    )
    assert passthrough is anim  # whole ANIMATION forwarded on, unchanged
    assert start.shape[0] == 1 and end.shape[0] == 1
    assert (pos, neg, fflf, length, fps, width, height, shift, out_dir) == (
        "p", "n", True, 21, 16, 832, 480, 3.0, "/a"
    )


def test_unpack_outputs_cover_selector_leaf_keys():
    # Every leaf key a selector emits must have an Unpack output (and vice versa);
    # the declared types line up slot-for-slot after the leading passthrough.
    assert {k for k, _n in nodes._POSE_UNPACK} == set(nodes.POSE_OUTPUT_KEYS)
    assert {k for k, _n in nodes._ANIMATION_UNPACK} == set(nodes.ANIMATION_OUTPUT_KEYS)
    assert nodes.PoseUnpack.RETURN_TYPES[0] == "ANIM_POSE"
    assert nodes.AnimationUnpack.RETURN_TYPES[0] == "ANIM_ANIMATION"
    # PoseUnpack: passthrough POSE + leaf keys + a computed HAS_POSE_REFERENCE.
    assert len(nodes.PoseUnpack.RETURN_TYPES) == len(nodes._POSE_UNPACK) + 2
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

    out = nodes.AnimationPlayback().play(
        manifest, tree.char, "", "punch", "EAST", 3, mode="loop", hold_frames=0
    )
    frames, fps = out["result"]
    assert fps == manifest["defaults"]["fps"]  # punch inherits the default fps (16)
    # idle(3) + punch(1x, drop_first+drop_last = 1) + idle(3) = 7
    # punch is not loopable: idle.last_frame != idle.start_frame (3 distinct frames)
    assert frames.shape[0] == 3 + 1 + 3


def test_animation_playback_raises_when_unrendered(manifest, tree, monkeypatch):
    # Nothing rendered: the action dir has no frames and the (incomplete) deps are
    # skipped, so assemble_playback yields the empty sentinel. play() must raise,
    # not emit a bogus 1x1 black frame (the old `shape[0] == 0` guard never fired).
    monkeypatch.setattr(nodes, "_characters_root", lambda: tree.root)
    with pytest.raises(RuntimeError, match="no rendered frames"):
        nodes.AnimationPlayback().play(
            manifest, tree.char, "", "punch", "EAST", 1, mode="loop", hold_frames=0
        )


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


# --- reference persistence + CharacterReferenceLoader ----------------------- #

def test_character_creator_persists_reference_by_default(manifest, tmp_path, monkeypatch):
    root = str(tmp_path)
    monkeypatch.setattr(nodes, "_characters_root", lambda: root)
    nodes.CharacterCreator().create(manifest, _img(3, 4), "cortex", "EAST")
    ref = resolve.reference_image_path(root, "cortex")
    assert os.path.isfile(ref)  # _reference.png written so base can be re-generated


def test_character_creator_can_skip_reference_persistence(manifest, tmp_path, monkeypatch):
    root = str(tmp_path)
    monkeypatch.setattr(nodes, "_characters_root", lambda: root)
    nodes.CharacterCreator().create(manifest, _img(), "cortex", "EAST", save_reference=False)
    assert not os.path.exists(resolve.reference_image_path(root, "cortex"))


def test_reference_loader_roundtrips_the_saved_image(manifest, tmp_path, monkeypatch):
    root = str(tmp_path)
    monkeypatch.setattr(nodes, "_characters_root", lambda: root)
    nodes.CharacterCreator().create(manifest, _img(5, 6), "cortex", "EAST")
    (img,) = nodes.CharacterReferenceLoader().load("cortex")
    assert img.shape[0] == 1 and img.shape[1] == 5 and img.shape[2] == 6


def test_reference_loader_raises_when_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(nodes, "_characters_root", lambda: str(tmp_path))
    with pytest.raises(RuntimeError, match="reference"):
        nodes.CharacterReferenceLoader().load("ghost")


def test_reference_loader_requires_a_character(monkeypatch, tmp_path):
    monkeypatch.setattr(nodes, "_characters_root", lambda: str(tmp_path))
    with pytest.raises(RuntimeError, match="character"):
        nodes.CharacterReferenceLoader().load("(select character)")


# --- auto-advancing batch selectors ----------------------------------------- #

def test_auto_pose_selector_emits_next_actionable_non_root_pose(manifest, tree, monkeypatch):
    monkeypatch.setattr(nodes, "_characters_root", lambda: tree.root)
    # base generated -> fighting_stance is the next actionable (non-root) pose.
    tree.pose("base", "EAST")
    images.save_image_png(_img(), resolve.pose_image_path(tree.root, tree.char, "base", "EAST"))
    (pose,) = nodes.AutoPoseSelector().select(
        manifest, tree.char, skip_mirrored=True, include_base=False
    )
    assert pose["_meta"]["pose"] == "fighting_stance"
    assert sorted(k for k in pose if k != "_meta") == nodes.POSE_OUTPUT_KEYS


def test_auto_pose_selector_raises_when_nothing_actionable(manifest, tree, monkeypatch):
    monkeypatch.setattr(nodes, "_characters_root", lambda: tree.root)
    tree.character()  # base is a root pose (excluded); nothing else actionable
    with pytest.raises(RuntimeError, match="no actionable poses"):
        nodes.AutoPoseSelector().select(
            manifest, tree.char, skip_mirrored=True, include_base=False
        )


def test_auto_pose_selector_include_base_emits_root_with_manikin(manifest, tree, monkeypatch):
    monkeypatch.setattr(nodes, "_characters_root", lambda: tree.root)
    # Character exists with a persisted reference, but no base direction rendered
    # yet: with include_base, base is the next actionable root pose and it comes
    # bundled with its manikin (2-reference edit).
    tree.character()
    images.save_image_png(_img(), resolve.reference_image_path(tree.root, tree.char))
    (pose,) = nodes.AutoPoseSelector().select(
        manifest, tree.char, skip_mirrored=True, include_base=True
    )
    assert pose["_meta"]["pose"] == "base"
    assert not images.is_empty(pose["source_image"])   # the reference art
    assert not images.is_empty(pose["pose_reference"])  # the direction's manikin


def test_auto_animation_selector_emits_next_actionable_animation(manifest, tree, monkeypatch):
    monkeypatch.setattr(nodes, "_characters_root", lambda: tree.root)
    tree.pose("base", "EAST").pose("fighting_stance", "EAST")
    # fighting_stance is the I2V start anchor for fighting_stance_idle — write a
    # real PNG there so the bundle can load it.
    images.save_image_png(
        _img(), resolve.pose_image_path(tree.root, tree.char, "fighting_stance", "EAST")
    )
    (anim, remaining) = nodes.AutoAnimationSelector().select(
        manifest, tree.char, skip_mirrored=True, category=""
    )
    assert anim["_meta"]["animation"] == "fighting_stance_idle"
    assert sorted(k for k in anim if k != "_meta") == nodes.ANIMATION_OUTPUT_KEYS
    assert remaining >= 1


def test_auto_animation_selector_raises_when_nothing_actionable(manifest, tree, monkeypatch):
    monkeypatch.setattr(nodes, "_characters_root", lambda: tree.root)
    tree.character()
    with pytest.raises(RuntimeError, match="no actionable animations"):
        nodes.AutoAnimationSelector().select(
            manifest, tree.char, skip_mirrored=True, category=""
        )


def test_pose_selector_sets_empty_pose_reference(manifest, tree, monkeypatch):
    monkeypatch.setattr(nodes, "_characters_root", lambda: tree.root)
    tree.pose("base", "EAST")
    images.save_image_png(_img(), resolve.pose_image_path(tree.root, tree.char, "base", "EAST"))
    (pose,) = nodes.CharacterPoseSelector().select(manifest, tree.char, "", "fighting_stance", "EAST")
    assert images.is_empty(pose["pose_reference"])


# --- SpriteTrimPivot node ---------------------------------------------------- #

def test_sprite_trim_pivot_node():
    img = torch.zeros((1, 8, 8, 4))
    img[0, 2:6, 2:6, :] = 1.0
    out, trim = nodes.SpriteTrimPivot().trim(
        img,
        alpha_threshold=0.03,
        trim_mode="union",
        pivot="bottom_center",
        pivot_x=0.5,
        pivot_y=1.0,
        pad=0,
    )
    assert out.shape[-1] == 4
    assert trim["frames"][0]["pivot"]


# --- SpritesheetPacker node -------------------------------------------------- #

def test_spritesheet_packer_node():
    batch = torch.ones((4, 6, 6, 4))
    sheet, atlas = nodes.SpritesheetPacker().pack(batch, layout="grid", columns=2,
        padding=1, extrude=0, power_of_two=False, fps=8)
    assert sheet.shape[0] == 1 and atlas["frames"][0]["duration_ms"] == 125


# --- AtlasMetadataWriter node ----------------------------------------------- #

def test_atlas_metadata_writer(tmp_path, monkeypatch):
    monkeypatch.setattr(nodes.api, "output_dir", lambda: str(tmp_path))
    atlas_d = {"sheet_size": [12, 6], "columns": 2, "frames": [
        {"rect": [0, 0, 6, 6], "source_size": [6, 6], "offset": [0, 0],
         "pivot": [3, 6], "duration_ms": 125}]}
    out = nodes.AtlasMetadataWriter().export(
        atlas_d, torch.ones((1, 6, 12, 4)),
        "aseprite", "walk", output_subdir="atlas")
    d = out["result"][0] if isinstance(out, dict) else out[0]
    assert os.path.exists(os.path.join(d, "walk.png"))
    assert os.path.exists(os.path.join(d, "walk.json"))


# --- CharacterAtlasBuilder -------------------------------------------------- #

def test_character_atlas_builder_renders_pose_sheet(manifest, tree, monkeypatch):
    monkeypatch.setattr(nodes, "_characters_root", lambda: tree.root)
    tree.pose("base", "EAST")
    images.save_image_png(
        _img(4, 4),
        resolve.pose_image_path(tree.root, tree.char, "base", "EAST"),
    )
    res = nodes.CharacterAtlasBuilder().build(
        manifest, tree.char, "pose", "base", "all", "per_direction_rows", 2, False
    )
    sheet, atlas, report = res["result"] if isinstance(res, dict) else res
    assert sheet.shape[0] == 1
    assert atlas["directions"] == ["EAST"]
    assert atlas["columns"] == 1
    assert atlas["sheet_size"]  # merged from pack_sheet — serializable by AtlasMetadataWriter
    assert "Rendered:" in report
    assert "EAST" in report


def test_character_atlas_builder_raises_when_nothing_rendered(manifest, tree, monkeypatch):
    monkeypatch.setattr(nodes, "_characters_root", lambda: tree.root)
    tree.character()
    with pytest.raises(RuntimeError, match="no rendered directions"):
        nodes.CharacterAtlasBuilder().build(
            manifest, tree.char, "pose", "base", "all", "grid", 0, False
        )


def test_character_atlas_builder_is_changed_volatile_without_character(manifest, monkeypatch):
    monkeypatch.setattr(nodes, "_characters_root", lambda: "output/characters")
    token = nodes.CharacterAtlasBuilder.IS_CHANGED(
        manifest, "", "pose", "", "all", "grid", 0, False
    )
    assert token != token


def test_character_atlas_builder_pads_mismatched_direction_sizes(manifest, tree, monkeypatch):
    """Directions with different H/W are zero-padded before torch.cat — no crash."""
    monkeypatch.setattr(nodes, "_characters_root", lambda: tree.root)
    # Write sidecars via Tree so node_complete is True for both directions, then
    # overwrite the placeholder PNGs with real RGBA images of different sizes.
    tree.pose("base", "EAST").pose("base", "SOUTH")
    east_path = resolve.pose_image_path(tree.root, tree.char, "base", "EAST")
    south_path = resolve.pose_image_path(tree.root, tree.char, "base", "SOUTH")
    # EAST: 6×6 RGBA, SOUTH: 8×10 RGBA — mismatched, would crash torch.cat without padding
    images.save_image_png(torch.zeros((1, 6, 6, 4)), east_path)
    images.save_image_png(torch.zeros((1, 8, 10, 4)), south_path)
    res = nodes.CharacterAtlasBuilder().build(
        manifest, tree.char, "pose", "base", "all", "grid", 0, False
    )
    sheet, atlas, report = res["result"] if isinstance(res, dict) else res
    # Sheet must be a single-image batch (pack_sheet always returns B=1).
    assert sheet.shape[0] == 1
    # Atlas must record exactly the two directions that were complete.
    assert len(atlas["directions"]) == 2
    assert "EAST" in atlas["directions"]
    assert "SOUTH" in atlas["directions"]


def test_palette_node_extract_only_passthrough():
    img = torch.ones((1, 2, 2, 3))
    out_img, pal = nodes.PaletteQuantizeLock().run(img, colors=4, dither="none",
        preserve_alpha=True, extract_only=True)
    assert torch.equal(out_img, img) and "colors" in pal


def test_auto_pose_selector_has_skip_mirrored_input():
    req = nodes.AutoPoseSelector.INPUT_TYPES()["required"]
    assert "skip_mirrored" in req


def test_manikin_pose_control_direction_only(monkeypatch, tmp_path):
    monkeypatch.setattr(nodes, "_characters_root", lambda: str(tmp_path))
    manifest = {"version": 1, "poses": {"base": {"directions": {"EAST": {}}}},
                "animations": {}, "defaults": {}}
    img, pos, dname = nodes.ManikinPoseControl().control(manifest, "(select character)",
        "base", "EAST", direction_only=True)
    assert img.shape[-1] == 3 and dname == "EAST"


def test_mirror_writer_batch_all(tmp_path, monkeypatch):
    from andypack import io
    monkeypatch.setattr(nodes, "_characters_root", lambda: str(tmp_path))
    root = str(tmp_path)
    char = "hero"
    manifest = {
        "version": 1,
        "mirror_map": {"WEST": "EAST"},
        "poses": {
            "base": {"directions": {"EAST": {}}},
            "p": {"from": {"ref": "base"}, "directions": {"EAST": {}, "WEST": {}}},
        },
        "animations": {},
        "defaults": {},
    }
    src = resolve.pose_image_path(root, char, "p", "EAST")
    os.makedirs(os.path.dirname(src), exist_ok=True)
    images.save_image_png(torch.ones((1, 4, 4, 4)), src)
    r = resolve.resolve_pose(manifest, root, char, "p", "EAST")
    io.atomic_write_json(
        resolve.pose_sidecar_path(root, char, "p", "EAST"),
        io.build_pose_sidecar(r["meta"], created_utc="t"),
    )
    dirs, count = nodes.MirrorFrameWriter().write(
        manifest, char, "pose", "p", "", mirror_all=True
    )
    assert count == 1
    assert os.path.exists(resolve.pose_image_path(root, char, "p", "WEST"))


# --- AnimationSheetBuilder / AnimationFrames -------------------------------- #

def _render_clip(tree, anim, direction, n=3):
    """Render `n` real RGBA frames for anim@direction (base + stance prereqs)."""
    tree.pose("base", direction).pose("fighting_stance", direction).animation(
        anim, direction, frames=n
    )
    d = resolve.animation_frame_dir(tree.root, tree.char, anim, direction)
    for i in range(n):
        images.save_image_png(_img(4, 4), os.path.join(d, f"frame_{i:05d}.png"))


def test_animation_sheet_builder_packs_direction_rows(manifest, tree, monkeypatch):
    monkeypatch.setattr(nodes, "_characters_root", lambda: tree.root)
    for direction in ("EAST", "SOUTH"):
        _render_clip(tree, "fighting_stance_idle", direction, n=3)
    res = nodes.AnimationSheetBuilder().build(
        manifest, tree.char, "fighting_stance_idle", "all", 2, False
    )
    sheet, atlas_d, report = res["result"] if isinstance(res, dict) else res
    assert sheet.shape[0] == 1 and sheet.shape[-1] == 4       # single RGBA batch
    assert atlas_d["sheet_size"] and atlas_d["fps"] > 0
    assert len(atlas_d["frames"]) == 6                        # 2 dirs × 3 frames
    names = [t["name"] for t in atlas_d["tags"]]
    assert "EAST" in names and "SOUTH" in names               # one tag per direction
    assert "fighting_stance_idle" in report


def test_animation_frames_loads_clip(manifest, tree, monkeypatch):
    monkeypatch.setattr(nodes, "_characters_root", lambda: tree.root)
    _render_clip(tree, "fighting_stance_idle", "EAST", n=3)
    frames, fps = nodes.AnimationFrames().load(
        manifest, tree.char, "fighting_stance_idle", "EAST"
    )
    assert frames.shape[0] == 3 and frames.shape[-1] == 4
    assert fps > 0


def test_animation_frames_raises_when_unrendered(manifest, tree, monkeypatch):
    monkeypatch.setattr(nodes, "_characters_root", lambda: tree.root)
    tree.character()
    with pytest.raises(RuntimeError, match="not rendered"):
        nodes.AnimationFrames().load(manifest, tree.char, "fighting_stance_idle", "EAST")


# --- CharacterIdentityAnchor ------------------------------------------------ #

def test_auto_animation_selector_has_category_scope():
    # ActionSetSelector was folded into AutoAnimationSelector as a `category` input.
    req = nodes.AutoAnimationSelector.INPUT_TYPES()["required"]
    assert "category" in req
    assert not hasattr(nodes, "ActionSetSelector")


def test_character_identity_anchor(monkeypatch, tmp_path):
    monkeypatch.setattr(nodes, "_characters_root", lambda: str(tmp_path))
    root = str(tmp_path)
    char = "hero"
    images.save_image_png(
        torch.ones((1, 4, 4, 3)), resolve.reference_image_path(root, char)
    )
    manifest = {
        "version": 1,
        "poses": {"base": {"directions": {"EAST": {}}}},
        "animations": {},
        "defaults": {},
    }
    ref, base, batch = nodes.CharacterIdentityAnchor().anchor(
        manifest, char, "EAST",
        include_reference=True, include_base=False, base_pose="base",
    )
    assert ref.shape[-1] == 3 and batch.shape[0] >= 1


# --- TurnaroundSheet -------------------------------------------------------- #

def test_turnaround_sheet_raises_for_placeholder_character(manifest, tree, monkeypatch):
    monkeypatch.setattr(nodes, "_characters_root", lambda: tree.root)
    with pytest.raises(RuntimeError, match="character"):
        nodes.TurnaroundSheet().build(manifest, nodes._NO_CHARACTER, "base")


def test_turnaround_sheet_returns_image_sheet(manifest, tree, monkeypatch):
    monkeypatch.setattr(nodes, "_characters_root", lambda: tree.root)
    tree.pose("base", "EAST")
    images.save_image_png(
        _img(4, 4),
        resolve.pose_image_path(tree.root, tree.char, "base", "EAST"),
    )
    out = nodes.TurnaroundSheet().build(manifest, tree.char, "base")
    sheet = out["result"][0]
    assert sheet.shape[0] == 1 and sheet.ndim == 4  # [1, H, W, C]


def test_turnaround_sheet_all_unrendered_returns_placeholders(manifest, tree, monkeypatch):
    monkeypatch.setattr(nodes, "_characters_root", lambda: tree.root)
    tree.character()  # character exists but nothing rendered
    out = nodes.TurnaroundSheet().build(manifest, tree.char, "base", columns=4)
    sheet = out["result"][0]
    # 8 directions in 4 columns = 2 rows; all cells are mid-gray placeholder
    assert sheet.shape[0] == 1
    assert sheet.shape[3] == 3  # 3-ch RGB


def test_turnaround_sheet_is_changed_always_nan(manifest, tree, monkeypatch):
    monkeypatch.setattr(nodes, "_characters_root", lambda: tree.root)
    token = nodes.TurnaroundSheet.IS_CHANGED(manifest, tree.char, "base")
    assert token != token  # NaN != NaN


def test_turnaround_sheet_node_registered():
    assert "TurnaroundSheet" in nodes.NODE_CLASS_MAPPINGS
    assert "TurnaroundSheet" in nodes.NODE_DISPLAY_NAME_MAPPINGS


# --- BoomerangLoopWriter ---------------------------------------------------- #

def test_boomerang_writes_palindrome(tmp_path, monkeypatch):
    monkeypatch.setattr(nodes, "_characters_root", lambda: str(tmp_path))
    anim = {
        "output_dir": str(tmp_path / "idle" / "EAST"),
        "_meta": {"prompt_hash": "sha1:x", "loop": False},
    }
    f = torch.arange(3).float().reshape(3, 1, 1, 1).repeat(1, 4, 4, 3)
    out = nodes.BoomerangLoopWriter().write(anim, f, mode="boomerang")
    d = out["result"][0] if isinstance(out, dict) else out[0]
    meta = json.load(open(os.path.join(d, "meta.json")))
    assert meta["loop"] is True and meta["frames"]["count"] == 4  # 0,1,2,1


# --- TweenClipProvider ------------------------------------------------------ #

def test_tween_requires_fflf():
    with pytest.raises(RuntimeError):
        nodes.TweenClipProvider()._validate_fflf(start=None, end=None)


# --- FrameTimingNormalizer -------------------------------------------------- #

def test_frame_timing_normalizer_registered():
    assert "FrameTimingNormalizer" in nodes.NODE_CLASS_MAPPINGS
    assert "FrameTimingNormalizer" in nodes.NODE_DISPLAY_NAME_MAPPINGS


def test_frame_timing_normalizer_resample_to_target():
    f = torch.zeros((5, 2, 2, 3))
    out, length = nodes.FrameTimingNormalizer().run(
        f, "resample", enforce_4n1=False, target_length=9
    )
    assert out.shape[0] == 9
    assert length == 9


def test_frame_timing_normalizer_length_output_matches_frames():
    # LENGTH must equal the actual emitted count, not the requested target.
    f = torch.zeros((4, 2, 2, 3))
    out, length = nodes.FrameTimingNormalizer().run(
        f, "trim", enforce_4n1=False, target_length=10
    )
    assert length == int(out.shape[0])
    assert length == 4  # trim can't invent frames


def test_frame_timing_normalizer_4n1_snap_up():
    # 6 -> nearest 4n+1 above = 9
    f = torch.zeros((5, 2, 2, 3))
    out, length = nodes.FrameTimingNormalizer().run(
        f, "pad_hold", enforce_4n1=True, target_length=6
    )
    assert out.shape[0] == 9
    assert length == 9


def test_frame_timing_normalizer_4n1_no_snap_on_aligned():
    # 33 is already 4n+1 (32 % 4 == 0); must not change.
    f = torch.zeros((30, 2, 2, 3))
    out, length = nodes.FrameTimingNormalizer().run(
        f, "resample", enforce_4n1=True, target_length=33
    )
    assert length == 33


def test_frame_timing_normalizer_target_from_animation():
    anim = {"_meta": {"length": 17, "fps": 16}, "output_dir": ""}
    f = torch.zeros((5, 2, 2, 3))
    out, length = nodes.FrameTimingNormalizer().run(
        f, "resample", enforce_4n1=False, animation=anim
    )
    assert length == 17


def test_frame_timing_normalizer_target_length_overrides_animation():
    anim = {"_meta": {"length": 33, "fps": 16}, "output_dir": ""}
    f = torch.zeros((5, 2, 2, 3))
    out, length = nodes.FrameTimingNormalizer().run(
        f, "resample", enforce_4n1=False, animation=anim, target_length=9
    )
    assert length == 9  # explicit target_length wins


def test_frame_timing_normalizer_noop_when_no_target():
    f = torch.zeros((7, 2, 2, 3))
    out, length = nodes.FrameTimingNormalizer().run(
        f, "resample", enforce_4n1=False
    )
    assert length == 7  # pass-through when no target specified


def test_frame_timing_normalizer_animation_zero_length_falls_back():
    # When animation._meta.length is 0 the node should pass through.
    anim = {"_meta": {"length": 0, "fps": 16}, "output_dir": ""}
    f = torch.zeros((5, 2, 2, 3))
    out, length = nodes.FrameTimingNormalizer().run(
        f, "resample", enforce_4n1=False, animation=anim
    )
    assert length == 5


# --- ColorVariantBatcher ----------------------------------------------------- #

def test_color_variant_batcher_pose_writes_sibling_png(manifest, tree, monkeypatch):
    monkeypatch.setattr(nodes, "_characters_root", lambda: tree.root)
    tree.character()
    # tree.pose() touches an empty PNG then writes the sidecar; overwrite with a
    # real PNG afterwards so load_image_tensor can open it.
    tree.pose("base", "EAST")
    images.save_image_png(
        _img(), resolve.pose_image_path(tree.root, tree.char, "base", "EAST")
    )

    variants_text = "red: hue=0, sat=1.0, val=1.0\nblue: hue=240, sat=1.0, val=1.0"
    dirs, count = nodes.ColorVariantBatcher().write(
        manifest, tree.char, "pose", "base", "EAST", variants_text
    )

    assert count == 2
    assert os.path.exists(
        resolve.pose_image_path(tree.root, tree.char, "base__red", "EAST")
    )
    assert os.path.exists(
        resolve.pose_image_path(tree.root, tree.char, "base__blue", "EAST")
    )
    # Sidecar must have variant_of provenance
    red_sidecar = json.loads(
        open(resolve.pose_sidecar_path(tree.root, tree.char, "base__red", "EAST")).read()
    )
    assert red_sidecar["variant_of"] == {"id": "base", "variant": "red"}
    assert red_sidecar.get("render_id", "").startswith("rid:")


def test_color_variant_batcher_animation_writes_sibling_frames(manifest, tree, monkeypatch):
    monkeypatch.setattr(nodes, "_characters_root", lambda: tree.root)
    tree.character()
    tree.animation("fighting_stance_idle", "EAST", frames=3)
    src_d = resolve.animation_frame_dir(tree.root, tree.char, "fighting_stance_idle", "EAST")
    for i in range(3):
        images.save_image_png(_img(), os.path.join(src_d, f"frame_{i:05d}.png"))

    dirs, count = nodes.ColorVariantBatcher().write(
        manifest, tree.char, "animation", "fighting_stance_idle", "EAST",
        "green: #00FF00"
    )

    assert count == 1
    dst_d = resolve.animation_frame_dir(
        tree.root, tree.char, "fighting_stance_idle__green", "EAST"
    )
    assert os.path.exists(os.path.join(dst_d, "frame_00000.png"))
    meta = json.loads(
        open(resolve.animation_meta_path(
            tree.root, tree.char, "fighting_stance_idle__green", "EAST"
        )).read()
    )
    assert meta["frames"]["count"] == 3
    assert meta["variant_of"] == {"id": "fighting_stance_idle", "variant": "green"}


def test_color_variant_batcher_raises_when_source_missing(manifest, tree, monkeypatch):
    monkeypatch.setattr(nodes, "_characters_root", lambda: tree.root)
    tree.character()
    with pytest.raises(RuntimeError, match="not generated"):
        nodes.ColorVariantBatcher().write(
            manifest, tree.char, "pose", "base", "EAST", "red: hue=0, sat=1, val=1"
        )


def test_color_variant_batcher_raises_on_empty_variants(manifest, tree, monkeypatch):
    monkeypatch.setattr(nodes, "_characters_root", lambda: tree.root)
    tree.character()
    tree.pose("base", "EAST")
    images.save_image_png(
        _img(), resolve.pose_image_path(tree.root, tree.char, "base", "EAST")
    )
    with pytest.raises(RuntimeError, match="no valid variants"):
        nodes.ColorVariantBatcher().write(
            manifest, tree.char, "pose", "base", "EAST", "   \n  "
        )


def test_color_variant_batcher_is_changed_nan_without_id(manifest, tree, monkeypatch):
    monkeypatch.setattr(nodes, "_characters_root", lambda: tree.root)
    token = nodes.ColorVariantBatcher.IS_CHANGED(
        manifest, tree.char, "pose", "", "EAST", "red: hue=0"
    )
    assert token != token  # NaN


def test_variant_layer_composer_recomputes_hash():
    pose = {
        "source_image": images.empty_image(),
        "pose_reference": images.empty_image(),
        "positive": "a hero",
        "negative": "blurry",
        "output_dir": "/x/_p",
        "_meta": {"prompt_hash": "sha1:old", "image": "EAST.png"},
    }
    (out,) = nodes.VariantLayerComposer().compose(pose, "gold", "golden armor", "")
    assert "golden armor" in out["positive"]
    assert out["_meta"]["prompt_hash"] != "sha1:old"
    assert out["output_dir"].endswith("__gold")
    # Input bundle must NOT be mutated
    assert pose["_meta"]["prompt_hash"] == "sha1:old"


# --- AnimatedSpriteExport --------------------------------------------------- #

def test_animated_sprite_export_registered():
    assert "AnimatedSpriteExport" in nodes.NODE_CLASS_MAPPINGS
    assert nodes.NODE_DISPLAY_NAME_MAPPINGS["AnimatedSpriteExport"] == "Animated Sprite Export"


def test_animated_sprite_export_writes_gif(tmp_path, monkeypatch):
    monkeypatch.setattr(nodes.api, "output_dir", lambda: str(tmp_path))
    # Use distinct frames so PIL does not deduplicate them.
    frames = torch.stack([
        torch.full((4, 4, 3), v, dtype=torch.float32) for v in (0.1, 0.4, 0.7)
    ])
    result = nodes.AnimatedSpriteExport().export(
        frames, format="gif", loop=True, fps=8, name="hero"
    )
    out_file = tmp_path / "hero.gif"
    assert out_file.exists()
    # result dict has ui (may be {}) and result tuple
    frames_out, out_dir = result["result"]
    assert frames_out.shape == frames.shape
    assert out_dir == str(tmp_path)
    from PIL import Image
    with Image.open(str(out_file)) as im:
        assert im.is_animated and im.n_frames == 3


def test_animated_sprite_export_onion_skin_changes_frames(tmp_path, monkeypatch):
    monkeypatch.setattr(nodes.api, "output_dir", lambda: str(tmp_path))
    # frame0=0, frame1=1, frame2=0 — onion skin on frame1 should darken it
    f = torch.zeros((3, 2, 2, 3))
    f[1] = 1.0
    result = nodes.AnimatedSpriteExport().export(
        f, format="gif", loop=True, fps=8,
        onion_skin=True, onion_prev=1, onion_next=1, onion_opacity=0.5, name="oss"
    )
    frames_out, _ = result["result"]
    # frame1 should be blended: 0.5*1 + 0.5*mean(0,0) = 0.5
    assert abs(float(frames_out[1].mean()) - 0.5) < 1e-4


def test_animated_sprite_export_webp(tmp_path, monkeypatch):
    monkeypatch.setattr(nodes.api, "output_dir", lambda: str(tmp_path))
    frames = torch.stack([
        torch.full((4, 4, 3), v, dtype=torch.float32) for v in [0.2, 0.5, 0.8]
    ])
    nodes.AnimatedSpriteExport().export(frames, format="webp", loop=True, fps=12, name="clip")
    assert (tmp_path / "clip.webp").exists()

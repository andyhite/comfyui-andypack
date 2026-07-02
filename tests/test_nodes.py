import json
import math
import os

import pytest
import torch

from andypack import api, images, nodes, resolve


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

def test_pose_sweep_selector_is_changed_always_reruns(manifest, tree, monkeypatch):
    # PoseSweepSelector.IS_CHANGED unconditionally returns NaN — it is disk-backed
    # and must re-read every execution (the sweep loop depends on this), so it
    # always re-runs regardless of widget values or rendered-tree state. This
    # replaces the old CharacterPoseSelector behavior of fingerprinting the
    # resolved prompt/source-image to detect drift; the unified selector instead
    # just always re-executes, which is a strictly stronger guarantee.
    monkeypatch.setattr(nodes, "_characters_root", lambda: tree.root)
    manifest["poses"]["fighting_stance"]["positive_prompt"] += " {character_prompt}"
    tree.pose("base", "EAST")  # base (root) rendered -> fighting_stance selectable
    before = nodes.PoseSweepSelector.IS_CHANGED(
        manifest, tree.char, "target", True, False, "", "fighting_stance", "EAST"
    )
    tree.character(positive_prompt="a brand new character line")
    after = nodes.PoseSweepSelector.IS_CHANGED(
        manifest, tree.char, "target", True, False, "", "fighting_stance", "EAST"
    )
    assert math.isnan(before) and math.isnan(after)


def test_animation_sweep_selector_is_changed_always_reruns(manifest, tree, monkeypatch):
    # AnimationSweepSelector.IS_CHANGED unconditionally returns NaN — it is
    # disk-backed and must re-read every execution (the sweep loop depends on
    # this), so it always re-runs regardless of widget values or rendered-tree
    # state. This replaces the old CharacterAnimationSelector behavior of
    # fingerprinting the resolved start/end anchor images to detect drift; the
    # unified selector instead just always re-executes, a strictly stronger
    # guarantee.
    monkeypatch.setattr(nodes, "_characters_root", lambda: tree.root)
    tree.pose("base", "EAST").pose("fighting_stance", "EAST")
    before = nodes.AnimationSweepSelector.IS_CHANGED(
        manifest, tree.char, "target", True, "", "punch", "EAST"
    )  # idle not rendered -> blocked, no anchor image
    tree.animation("fighting_stance_idle", "EAST", frames=3)
    after = nodes.AnimationSweepSelector.IS_CHANGED(
        manifest, tree.char, "target", True, "", "punch", "EAST"
    )  # idle rendered -> selectable, anchors resolve
    assert math.isnan(before) and math.isnan(after)


def test_selector_is_changed_volatile_without_character(manifest, monkeypatch):
    # No character chosen -> always re-run (NaN != NaN), never a stale cache hit.
    token = nodes.PoseSweepSelector.IS_CHANGED(manifest, "", "target", True, False, "", "base", "EAST")
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


def test_coverage_report_node(manifest, tree, monkeypatch):
    monkeypatch.setattr(nodes, "_characters_root", lambda: tree.root)
    tree.character()
    out = nodes.CoverageReport().report(manifest, tree.char)
    report, blob = out["result"]
    assert "base" in report
    assert json.loads(blob)["total"] > 0
    # The table is also pushed to the frontend so the node shows it inline.
    assert out["ui"]["text"] == (report,)


def test_animation_selector_outputs_length_and_fps(manifest, tree, monkeypatch):
    monkeypatch.setattr(nodes, "_characters_root", lambda: tree.root)
    tree.pose("base", "EAST").pose("fighting_stance", "EAST").animation(
        "fighting_stance_idle", "EAST", frames=3
    )
    idle = resolve.animation_frame_dir(tree.root, tree.char, "fighting_stance_idle", "EAST")
    for i in range(3):  # real PNGs so the anchor frames load
        images.save_image_png(_img(), os.path.join(idle, f"frame_{i:05d}.png"))
    (anim,) = nodes.AnimationSweepSelector().select(
        manifest, tree.char, "target", True, "", "punch", "EAST"
    )
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
    (pose,) = nodes.PoseSweepSelector().select(
        manifest, tree.char, "target", True, False, "", "fighting_stance", "EAST"
    )
    # Leaf outputs are selectable; the resolver meta rides bundled under `_meta`
    # (not a leaf output) and the image rides along as a tensor.
    assert isinstance(pose["positive"], str)
    assert pose["source_image"].shape[0] == 1
    assert pose["_meta"]["pose"] == "fighting_stance" and pose["_meta"]["image"] == "EAST.png"
    # Drift guard: the public (non-`_meta`) keys are exactly the leaf-output keys.
    assert sorted(k for k in pose if not k.startswith("_")) == nodes.POSE_OUTPUT_KEYS


def test_animation_selector_returns_single_dict(manifest, tree, monkeypatch):
    monkeypatch.setattr(nodes, "_characters_root", lambda: tree.root)
    tree.pose("base", "EAST").pose("fighting_stance", "EAST").animation(
        "fighting_stance_idle", "EAST", frames=3
    )
    idle = resolve.animation_frame_dir(tree.root, tree.char, "fighting_stance_idle", "EAST")
    for i in range(3):
        images.save_image_png(_img(), os.path.join(idle, f"frame_{i:05d}.png"))
    (anim,) = nodes.AnimationSweepSelector().select(
        manifest, tree.char, "target", True, "", "punch", "EAST"
    )
    assert anim["_meta"]["animation"] == "punch"
    assert isinstance(anim["is_fflf"], bool)
    assert isinstance(anim["length"], int) and isinstance(anim["fps"], int)
    # Generation params for the WanFirstLastFrameToVideo node ride along as wireable
    # INT/FLOAT outputs (from the manifest defaults / per-animation override).
    assert isinstance(anim["width"], int) and isinstance(anim["height"], int)
    assert isinstance(anim["shift"], float)
    assert anim["start_image"].shape[0] == 1 and anim["end_image"].shape[0] == 1
    assert sorted(k for k in anim if not k.startswith("_")) == nodes.ANIMATION_OUTPUT_KEYS


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


def test_leaf_output_keys_exclude_private_meta():
    assert "_meta" not in nodes.POSE_OUTPUT_KEYS
    assert "_meta" not in nodes.ANIMATION_OUTPUT_KEYS


def test_pose_bundle_carries_sweep_context(manifest, tree):
    tree.character(concept="x")
    tree.pose("base", "EAST")
    images.save_image_png(_img(), resolve.pose_image_path(tree.root, tree.char, "base", "EAST"))
    job = api.next_actionable(manifest, tree.root, tree.char, "pose", exclude_root=True)
    r = resolve.resolve_pose(manifest, tree.root, tree.char, job["id"], job["direction"])
    bundle = nodes._build_pose_bundle(
        r, tree.root, tree.char,
        sweep={"character": tree.char, "kind": "pose", "mode": "sweep",
               "exclude_root": True, "category": None, "skip_mirrored": True})
    assert bundle["_sweep"]["mode"] == "sweep"
    assert bundle["_sweep"]["kind"] == "pose"
    # Not a wireable leaf output — like `_meta`.
    assert "_sweep" not in nodes.POSE_OUTPUT_KEYS


def test_animation_bundle_defaults_sweep_to_empty_dict(manifest, tree, monkeypatch):
    monkeypatch.setattr(nodes, "_characters_root", lambda: tree.root)
    tree.pose("base", "EAST").pose("fighting_stance", "EAST").animation(
        "fighting_stance_idle", "EAST", frames=3
    )
    idle = resolve.animation_frame_dir(tree.root, tree.char, "fighting_stance_idle", "EAST")
    for i in range(3):
        images.save_image_png(_img(), os.path.join(idle, f"frame_{i:05d}.png"))
    r = resolve.resolve_animation(manifest, tree.root, tree.char, "punch", "EAST")
    bundle = nodes._build_animation_bundle(r)
    assert bundle["_sweep"] == {}
    assert "_sweep" not in nodes.ANIMATION_OUTPUT_KEYS


def test_pose_selector_rejects_unknown_id(manifest, tree, monkeypatch):
    # A stale/renamed serialized id must give a friendly error, not a raw KeyError.
    monkeypatch.setattr(nodes, "_characters_root", lambda: tree.root)
    tree.character()
    with pytest.raises(RuntimeError, match="unknown pose"):
        nodes.PoseSweepSelector().select(
            manifest, tree.char, "target", True, False, "", "ghost_pose", "EAST"
        )


def test_animation_selector_rejects_unknown_id(manifest, tree, monkeypatch):
    monkeypatch.setattr(nodes, "_characters_root", lambda: tree.root)
    tree.character()
    with pytest.raises(RuntimeError, match="unknown animation"):
        nodes.AnimationSweepSelector().select(
            manifest, tree.char, "target", True, "", "ghost_anim", "EAST"
        )


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
    (anim,) = nodes.AnimationSweepSelector().select(
        manifest, tree.char, "target", True, "", "fighting_stance_exit", "EAST"
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


def test_pose_sweep_selector_target_mode_can_force_a_root_pose(manifest, tree, monkeypatch):
    # DESIGN CHANGE from the old CharacterPoseSelector, which rejected root poses
    # (steering the user to the Character Creator instead). The unified selector's
    # target mode is an explicit spot-fix tool that force-resolves "regardless of
    # completeness" (per the sweep-loops design doc) with no carve-out for root
    # poses — and _build_pose_bundle already knows how to pair a root pose with
    # its manikin. So targeting ("base", "EAST") now succeeds instead of raising —
    # PROVIDED the character has a persisted reference (seeded here); without one
    # there is nothing to pair with the manikin, and the bundle builder now raises
    # rather than silently emitting a blank sentinel (see the test right below).
    monkeypatch.setattr(nodes, "_characters_root", lambda: tree.root)
    tree.pose("base", "EAST")
    images.save_image_png(_img(), resolve.pose_image_path(tree.root, tree.char, "base", "EAST"))
    images.save_image_png(_img(), resolve.reference_image_path(tree.root, tree.char))
    (pose,) = nodes.PoseSweepSelector().select(
        manifest, tree.char, "target", True, False, "", "base", "EAST"
    )
    assert pose["_meta"]["pose"] == "base"
    assert pose["_sweep"]["target"] == ("base", "EAST")
    assert not images.is_empty(pose["source_image"])  # the persisted reference art


def test_pose_sweep_selector_target_mode_root_pose_requires_reference(manifest, tree, monkeypatch):
    # Without a persisted reference, a root-pose target must raise rather than
    # silently bake a blank 1x1 sentinel into source_image (the old, footgun-prone
    # behavior) — the character must exist first (Character Creator persists the
    # reference art).
    monkeypatch.setattr(nodes, "_characters_root", lambda: tree.root)
    tree.pose("base", "EAST")
    images.save_image_png(_img(), resolve.pose_image_path(tree.root, tree.char, "base", "EAST"))
    with pytest.raises(RuntimeError, match="reference"):
        nodes.PoseSweepSelector().select(
            manifest, tree.char, "target", True, False, "", "base", "EAST"
        )


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

def test_pose_sweep_selector_emits_next_actionable_non_root_pose(manifest, tree, monkeypatch):
    monkeypatch.setattr(nodes, "_characters_root", lambda: tree.root)
    # base generated -> fighting_stance is the next actionable (non-root) pose.
    tree.pose("base", "EAST")
    images.save_image_png(_img(), resolve.pose_image_path(tree.root, tree.char, "base", "EAST"))
    (pose,) = nodes.PoseSweepSelector().select(
        manifest, tree.char, "sweep", True, False, "", "", ""
    )
    assert pose["_meta"]["pose"] == "fighting_stance"
    assert pose["_sweep"]["mode"] == "sweep"
    assert sorted(k for k in pose if not k.startswith("_")) == nodes.POSE_OUTPUT_KEYS


def test_pose_sweep_selector_raises_when_nothing_actionable(manifest, tree, monkeypatch):
    monkeypatch.setattr(nodes, "_characters_root", lambda: tree.root)
    tree.character()  # base is a root pose (excluded); nothing else actionable
    with pytest.raises(RuntimeError, match="no actionable poses"):
        nodes.PoseSweepSelector().select(
            manifest, tree.char, "sweep", True, False, "", "", ""
        )


def test_pose_sweep_selector_include_base_emits_root_with_manikin(manifest, tree, monkeypatch):
    monkeypatch.setattr(nodes, "_characters_root", lambda: tree.root)
    # Character exists with a persisted reference, but no base direction rendered
    # yet: with include_base, base is the next actionable root pose and it comes
    # bundled with its manikin (2-reference edit).
    tree.character()
    images.save_image_png(_img(), resolve.reference_image_path(tree.root, tree.char))
    (pose,) = nodes.PoseSweepSelector().select(
        manifest, tree.char, "sweep", True, True, "", "", ""
    )
    assert pose["_meta"]["pose"] == "base"
    assert not images.is_empty(pose["source_image"])   # the reference art
    assert not images.is_empty(pose["pose_reference"])  # the direction's manikin


def test_animation_sweep_selector_emits_next_actionable_animation(manifest, tree, monkeypatch):
    monkeypatch.setattr(nodes, "_characters_root", lambda: tree.root)
    tree.pose("base", "EAST").pose("fighting_stance", "EAST")
    # fighting_stance is the I2V start anchor for fighting_stance_idle — write a
    # real PNG there so the bundle can load it.
    images.save_image_png(
        _img(), resolve.pose_image_path(tree.root, tree.char, "fighting_stance", "EAST")
    )
    # NOTE: the old AutoAnimationSelector returned (anim, remaining) — a 2-tuple
    # with a REMAINING count. The unified selector drops REMAINING (the writer
    # computes remaining work in a later task), so it returns a 1-tuple now.
    (anim,) = nodes.AnimationSweepSelector().select(
        manifest, tree.char, "sweep", True, "", "", ""
    )
    assert anim["_meta"]["animation"] == "fighting_stance_idle"
    assert anim["_sweep"]["mode"] == "sweep"
    assert sorted(k for k in anim if not k.startswith("_")) == nodes.ANIMATION_OUTPUT_KEYS


def test_animation_sweep_selector_raises_when_nothing_actionable(manifest, tree, monkeypatch):
    monkeypatch.setattr(nodes, "_characters_root", lambda: tree.root)
    tree.character()
    with pytest.raises(RuntimeError, match="no actionable animations"):
        nodes.AnimationSweepSelector().select(
            manifest, tree.char, "sweep", True, "", "", ""
        )


def test_pose_selector_sets_empty_pose_reference(manifest, tree, monkeypatch):
    monkeypatch.setattr(nodes, "_characters_root", lambda: tree.root)
    tree.pose("base", "EAST")
    images.save_image_png(_img(), resolve.pose_image_path(tree.root, tree.char, "base", "EAST"))
    (pose,) = nodes.PoseSweepSelector().select(
        manifest, tree.char, "target", True, False, "", "fighting_stance", "EAST"
    )
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


def test_atlas_writer_rejects_empty_name(tmp_path, monkeypatch):
    monkeypatch.setattr(nodes.api, "output_dir", lambda: str(tmp_path))
    atlas = {"sheet_size": [4, 4], "columns": 1,
             "frames": [{"rect": [0, 0, 4, 4], "source_size": [4, 4],
                         "offset": [0, 0], "pivot": None, "duration_ms": None}]}
    with pytest.raises(RuntimeError, match="name"):
        nodes.AtlasMetadataWriter().export(atlas, _img(4, 4), "json_hash", "")
    with pytest.raises(RuntimeError, match="name"):
        nodes.AtlasMetadataWriter().export(atlas, _img(4, 4), "json_hash", "../evil")
    assert not os.path.exists(os.path.join(str(tmp_path), "atlas", ".png"))


# --- CharacterAtlasBuilder -------------------------------------------------- #

def test_pose_sweep_selector_input_shape():
    req = nodes.PoseSweepSelector.INPUT_TYPES()["required"]
    assert list(req.keys()) == [
        "manifest", "character", "mode", "skip_mirrored", "include_base",
        "category", "pose", "direction",
    ]
    assert req["mode"] == (["sweep", "target"],)
    assert req["skip_mirrored"] == ("BOOLEAN", {"default": True})
    assert req["include_base"] == ("BOOLEAN", {"default": True})


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

def test_animation_sweep_selector_input_shape():
    # ActionSetSelector was folded into the animation selector as a `category`
    # input (first via AutoAnimationSelector, now via the unified
    # AnimationSweepSelector).
    req = nodes.AnimationSweepSelector.INPUT_TYPES()["required"]
    assert list(req.keys()) == [
        "manifest", "character", "mode", "skip_mirrored",
        "category", "animation", "direction",
    ]
    assert not hasattr(nodes, "ActionSetSelector")


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

# --- TweenClipProvider ------------------------------------------------------ #

# --- FrameTimingNormalizer -------------------------------------------------- #

# --- ColorVariantBatcher ----------------------------------------------------- #

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


# --- CharacterPromptLoader --------------------------------------------------- #

def _write_character(root, name, payload):
    os.makedirs(os.path.join(root, name), exist_ok=True)
    with open(os.path.join(root, name, "character.json"), "w") as f:
        json.dump(payload, f)


def test_character_prompt_loader_returns_authored_prompts(tmp_path, monkeypatch):
    root = str(tmp_path)
    monkeypatch.setattr(nodes, "_characters_root", lambda: root)
    _write_character(root, "cortex", {"positive_prompt": "a brave hero", "negative_prompt": "blurry"})
    # combo value is snake-cased, so a display-cased name still resolves.
    pos, neg = nodes.CharacterPromptLoader().load("Cortex")
    assert pos == "a brave hero"
    assert neg == "blurry"


def test_character_prompt_loader_missing_fields_yield_empty(tmp_path, monkeypatch):
    root = str(tmp_path)
    monkeypatch.setattr(nodes, "_characters_root", lambda: root)
    _write_character(root, "ghost", {"positive_prompt": "x"})
    pos, neg = nodes.CharacterPromptLoader().load("ghost")
    assert pos == "x"
    assert neg == ""


def test_character_prompt_loader_requires_character():
    with pytest.raises(RuntimeError, match="select a character"):
        nodes.CharacterPromptLoader().load(nodes._NO_CHARACTER)


# --- CharacterLoader (read-only base-pose emitter) --------------------------- #

def test_character_loader_emits_base_pose(manifest, tmp_path, monkeypatch):
    monkeypatch.setattr(nodes, "_characters_root", lambda: str(tmp_path))
    (pose,) = nodes.CharacterLoader().load(manifest, _img(), "cortex", "EAST")
    assert pose["_meta"]["pose"] == "base" and pose["_meta"]["direction"] == "EAST"
    assert pose["output_dir"].endswith(os.path.join("cortex", "_base"))
    assert not images.is_empty(pose["source_image"])      # the supplied reference
    assert not images.is_empty(pose["pose_reference"])     # the direction's manikin


def test_character_loader_does_not_write_character_json(manifest, tmp_path, monkeypatch):
    root = str(tmp_path)
    monkeypatch.setattr(nodes, "_characters_root", lambda: root)
    # Author a character.json up front; the loader must leave it byte-for-byte intact.
    os.makedirs(os.path.join(root, "cortex"), exist_ok=True)
    cj = os.path.join(root, "cortex", "character.json")
    with open(cj, "w") as f:
        json.dump({"positive_prompt": "a brave hero", "negative_prompt": "blurry"}, f)
    before = open(cj).read()
    nodes.CharacterLoader().load(manifest, _img(), "cortex", "EAST")
    assert open(cj).read() == before


def test_character_loader_persists_reference_by_default(manifest, tmp_path, monkeypatch):
    root = str(tmp_path)
    monkeypatch.setattr(nodes, "_characters_root", lambda: root)
    nodes.CharacterLoader().load(manifest, _img(3, 4), "cortex", "EAST")
    assert os.path.isfile(resolve.reference_image_path(root, "cortex"))


def test_character_loader_can_skip_reference_persistence(manifest, tmp_path, monkeypatch):
    root = str(tmp_path)
    monkeypatch.setattr(nodes, "_characters_root", lambda: root)
    nodes.CharacterLoader().load(manifest, _img(), "cortex", "EAST", save_reference=False)
    assert not os.path.exists(resolve.reference_image_path(root, "cortex"))


def test_character_loader_rejects_unknown_direction(manifest, tmp_path, monkeypatch):
    monkeypatch.setattr(nodes, "_characters_root", lambda: str(tmp_path))
    with pytest.raises(RuntimeError, match="direction"):
        nodes.CharacterLoader().load(manifest, _img(), "cortex", "UP")


def test_character_loader_requires_character(manifest, tmp_path, monkeypatch):
    monkeypatch.setattr(nodes, "_characters_root", lambda: str(tmp_path))
    with pytest.raises(RuntimeError, match="character"):
        nodes.CharacterLoader().load(manifest, _img(), nodes._NO_CHARACTER, "EAST")

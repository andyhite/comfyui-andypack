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
        "source_image": image, "positive": "", "negative": "",
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
        "from": {"ref": "concept"}, "image": "EAST.png",
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
    tree.concept()
    # Identity is opt-in, so the prompt must reference it for an edit to ripple.
    manifest["poses"]["base"]["positive_prompt"] += " {identity_prompt}"
    before = nodes.CharacterPoseSelector.IS_CHANGED(
        manifest, tree.char, "", "base", "EAST"
    )
    # Rendering the dependency (concept already there) vs editing identity must
    # move the fingerprint — selectable/prompt_hash/source mtime are folded in.
    tree.identity(positive_prompt="a brand new identity line")
    after = nodes.CharacterPoseSelector.IS_CHANGED(
        manifest, tree.char, "", "base", "EAST"
    )
    assert before != after


def test_animation_selector_is_changed_tracks_anchor_render(manifest, tree, monkeypatch):
    monkeypatch.setattr(nodes, "_characters_root", lambda: tree.root)
    tree.concept().pose("base", "EAST").pose("fighting_stance", "EAST")
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

def test_concept_writer_writes_provenance_sidecar(tmp_path, monkeypatch):
    # Even with no identity text, the concept sidecar is written with a render_id so
    # re-rendering the concept can propagate staleness to its descendants.
    monkeypatch.setattr(nodes, "_characters_root", lambda: str(tmp_path))
    (char_dir,) = nodes.ConceptImageWriter().write(_img(), "cortex")
    side = json.loads(open(os.path.join(char_dir, "_concept.json")).read())
    assert side["render_id"].startswith("rid:")


def test_concept_writer_preserves_authored_poses(tmp_path, monkeypatch):
    # Re-rendering the concept must not clobber character-authored poses/animations
    # stored in _concept.json (which effective_manifest folds into the manifest).
    monkeypatch.setattr(nodes, "_characters_root", lambda: str(tmp_path))
    (char_dir,) = nodes.ConceptImageWriter().write(_img(), "cortex", identity_positive="hero")
    side_path = os.path.join(char_dir, "_concept.json")
    side = json.loads(open(side_path).read())
    side["poses"] = {"wave": {"from": {"ref": "concept"}, "directions": {"EAST": {}}}}
    with open(side_path, "w") as fh:
        json.dump(side, fh)

    # Re-render the concept (new identity text).
    nodes.ConceptImageWriter().write(_img(), "cortex", identity_positive="reworked hero")
    after = json.loads(open(side_path).read())
    assert after["poses"] == {"wave": {"from": {"ref": "concept"}, "directions": {"EAST": {}}}}
    assert after["positive_prompt"] == "reworked hero"


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


def test_pose_resolve_records_concept_render_id(manifest, tmp_path, monkeypatch):
    # A pose's recorded sources capture the concept's render_id, closing the gap
    # where re-rendering the concept left descendants looking fresh.
    root = str(tmp_path)
    monkeypatch.setattr(nodes, "_characters_root", lambda: root)
    (char_dir,) = nodes.ConceptImageWriter().write(_img(), "cortex")
    rid = json.loads(open(os.path.join(char_dir, "_concept.json")).read())["render_id"]
    r = resolve.resolve_pose(manifest, root, "cortex", "base", "EAST")
    assert r["meta"]["sources"]["concept@EAST"] == rid


def test_concept_image_loader_reads_image_and_identity(tree, monkeypatch):
    monkeypatch.setattr(nodes, "_characters_root", lambda: tree.root)
    images.save_image_png(_img(), resolve.concept_image_path(tree.root, tree.char))
    tree.identity(positive_prompt="a brave hero", negative_prompt="blurry")
    image, has, pos, neg = nodes.ConceptImageLoader().load(tree.char)
    assert has is True and image.shape[0] == 1
    assert pos == "a brave hero" and neg == "blurry"


def test_concept_image_loader_missing_concept(tree, monkeypatch):
    monkeypatch.setattr(nodes, "_characters_root", lambda: tree.root)
    image, has, pos, neg = nodes.ConceptImageLoader().load(tree.char)
    assert has is False and pos == "" and neg == ""


def test_manifest_lint_reports_findings(manifest):
    manifest["poses"]["base"]["directions"]["UP"] = {}  # unknown direction
    (report,) = nodes.ManifestLint().lint(manifest)
    assert "UP" in report


def test_manifest_lint_clean(manifest):
    (report,) = nodes.ManifestLint().lint(manifest)
    assert report.startswith("OK")


def test_coverage_report_node(manifest, tree, monkeypatch):
    monkeypatch.setattr(nodes, "_characters_root", lambda: tree.root)
    tree.concept()
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
    tree.concept()
    text, count = nodes.RegenQueue().build(manifest, tree.char)
    assert count >= 1
    assert "base@EAST" in text


def test_mirror_writer_pose(manifest, tree, monkeypatch):
    monkeypatch.setattr(nodes, "_characters_root", lambda: tree.root)
    tree.concept()
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
    tree.concept().pose("base", "EAST").pose("fighting_stance", "EAST").animation(
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
    tree.concept()
    images.save_image_png(_img(), resolve.concept_image_path(tree.root, tree.char))
    (pose,) = nodes.CharacterPoseSelector().select(manifest, tree.char, "", "base", "EAST")
    # Leaf outputs are selectable; the resolver meta rides bundled under `_meta`
    # (not a leaf output) and the image rides along as a tensor.
    assert isinstance(pose["positive"], str)
    assert pose["source_image"].shape[0] == 1
    assert pose["_meta"]["pose"] == "base" and pose["_meta"]["image"] == "EAST.png"
    # Drift guard: the public (non-`_meta`) keys are exactly the leaf-output keys.
    assert sorted(k for k in pose if k != "_meta") == nodes.POSE_OUTPUT_KEYS


def test_animation_selector_returns_single_dict(manifest, tree, monkeypatch):
    monkeypatch.setattr(nodes, "_characters_root", lambda: tree.root)
    tree.concept().pose("base", "EAST").pose("fighting_stance", "EAST").animation(
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
            "output_dir": "/x", "_meta": {}}
    passthrough, img, pos, neg, out_dir = nodes.PoseUnpack().unpack(pose)
    assert nodes.PoseUnpack.RETURN_NAMES == (
        "POSE", "SOURCE_IMAGE", "POSITIVE_PROMPT", "NEGATIVE_PROMPT", "OUTPUT_DIR"
    )
    assert passthrough is pose  # whole POSE forwarded on, unchanged
    assert img.shape[0] == 1
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
    tree.concept().pose("base", "EAST").pose("fighting_stance", "EAST").animation(
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
    tree.concept()
    try:
        nodes.MirrorFrameWriter().write(manifest, tree.char, "pose", "base", "EAST")
    except RuntimeError as e:
        assert "mirror_map" in str(e)
    else:
        raise AssertionError("expected RuntimeError for an unmapped direction")

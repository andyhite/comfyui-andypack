import json
import os

import torch

from andypack import images, nodes, resolve


def _img(h=2, w=2):
    return torch.zeros((1, h, w, 3), dtype=torch.float32)


def _batch(n, h=2, w=2):
    return torch.zeros((n, h, w, 3), dtype=torch.float32)


# --- re-render discipline ---------------------------------------------------- #

def test_animation_rewrite_clears_stale_frames(tmp_path):
    out = str(tmp_path / "punch" / "EAST")
    meta = {
        "kind": "animation", "animation": "punch", "direction": "EAST",
        "fps": 16, "length": 5, "loop": False, "manifest_version": 1,
        "prompt_hash": "sha1:abc",
    }
    writer = nodes.AnimationFrameWriter()
    writer.write(_batch(5), out, meta)
    assert sum(n.startswith("frame_") for n in os.listdir(out)) == 5

    # A shorter re-render must not leave the old frame_00003/00004 behind.
    writer.write(_batch(3), out, dict(meta, length=3))
    frames = sorted(n for n in os.listdir(out) if n.startswith("frame_"))
    assert frames == ["frame_00000.png", "frame_00001.png", "frame_00002.png"]
    import json
    full = json.loads(open(os.path.join(out, "meta.json")).read())
    assert full["frames"]["count"] == 3
    assert full["last_frame"] == "frame_00002.png"


def test_pose_rewrite_replaces_sidecar(tmp_path):
    out = str(tmp_path / "_base")
    meta = {
        "kind": "pose", "pose": "base", "direction": "EAST",
        "from": {"ref": "concept"}, "image": "EAST.png",
        "manifest_version": 1, "prompt_hash": "sha1:one",
    }
    writer = nodes.PoseFrameWriter()
    writer.write(_img(), out, meta)
    assert os.path.exists(os.path.join(out, "EAST.png"))

    writer.write(_img(), out, dict(meta, prompt_hash="sha1:two"))
    import json
    side = json.loads(open(os.path.join(out, "EAST.json")).read())
    assert side["prompt_hash"] == "sha1:two"


# --- selector IS_CHANGED fingerprint ---------------------------------------- #

def test_pose_selector_is_changed_tracks_dependency_render(manifest, tree, monkeypatch):
    monkeypatch.setattr(nodes, "_characters_root", lambda: tree.root)
    tree.concept()
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
    out = nodes.CharacterAnimationSelector().select(manifest, tree.char, "", "punch", "EAST")
    # order: START, END, POS, NEG, IS_FFLF, LENGTH, FPS, OUTPUT_DIR, META
    assert out[5] == manifest["animations"]["punch"]["length"]  # 21
    assert out[6] == manifest["defaults"]["fps"]                # 16 (punch inherits default)
    assert out[8]["animation"] == "punch"


def test_mirror_writer_rejects_unmapped_direction(manifest, tree, monkeypatch):
    monkeypatch.setattr(nodes, "_characters_root", lambda: tree.root)
    tree.concept()
    try:
        nodes.MirrorFrameWriter().write(manifest, tree.char, "pose", "base", "EAST")
    except RuntimeError as e:
        assert "mirror_map" in str(e)
    else:
        raise AssertionError("expected RuntimeError for an unmapped direction")

from andypack.resolve import (
    animation_complete,
    concept_complete,
    node_complete,
    pose_complete,
    read_rendered_hash,
    resolved_dir,
)


def test_resolved_dir_same_vs_explicit():
    assert resolved_dir({"ref": "x"}, "E") == "E"
    assert resolved_dir({"ref": "x", "direction": "same"}, "SE") == "SE"
    assert resolved_dir({"ref": "x", "direction": "E"}, "SE") == "E"


def test_concept_complete(tree):
    assert concept_complete(tree.root, tree.char) is False
    tree.concept()
    assert concept_complete(tree.root, tree.char) is True


def test_pose_complete_requires_png_and_sidecar(tree):
    tree.concept().pose("base", "E", sidecar=False)
    assert pose_complete(tree.root, tree.char, "base", "E") is False  # png only
    tree.pose("base", "E")  # now with sidecar
    assert pose_complete(tree.root, tree.char, "base", "E") is True


def test_animation_complete_requires_meta_and_frames(tree):
    tree.concept().animation("fighting_stance_idle", "E", frames=3, meta=False)
    assert animation_complete(tree.root, tree.char, "fighting_stance_idle", "E") is False
    tree.animation("fighting_stance_idle", "E", frames=3)  # meta now present
    assert animation_complete(tree.root, tree.char, "fighting_stance_idle", "E") is True


def test_node_complete_dispatches_by_kind(manifest, tree):
    tree.concept().pose("base", "E").pose("fighting_stance", "E")
    assert node_complete(manifest, tree.root, tree.char, "concept", "E") is True
    assert node_complete(manifest, tree.root, tree.char, "base", "E") is True
    assert node_complete(manifest, tree.root, tree.char, "fighting_stance", "E") is True
    assert node_complete(manifest, tree.root, tree.char, "punch", "E") is False


def test_read_rendered_hash(manifest, tree):
    assert read_rendered_hash(manifest, tree.root, tree.char, "concept", "E") is None
    tree.concept().pose("base", "E")
    h = read_rendered_hash(manifest, tree.root, tree.char, "base", "E")
    assert h is not None and h.startswith("sha1:")

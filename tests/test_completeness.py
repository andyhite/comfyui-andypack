from andypack.resolve import (
    animation_complete,
    node_complete,
    pose_complete,
    resolved_dir,
)


def test_resolved_dir_same_vs_explicit():
    assert resolved_dir({"ref": "x"}, "EAST") == "EAST"
    assert resolved_dir({"ref": "x", "direction": "same"}, "SOUTH_EAST") == "SOUTH_EAST"
    assert resolved_dir({"ref": "x", "direction": "EAST"}, "SOUTH_EAST") == "EAST"


def test_pose_complete_requires_png_and_sidecar(tree):
    tree.pose("base", "EAST", sidecar=False)
    assert pose_complete(tree.root, tree.char, "base", "EAST") is False  # png only
    tree.pose("base", "EAST")  # now with sidecar
    assert pose_complete(tree.root, tree.char, "base", "EAST") is True


def test_animation_complete_requires_meta_and_frames(tree):
    tree.animation("fighting_stance_idle", "EAST", frames=3, meta=False)
    assert animation_complete(tree.root, tree.char, "fighting_stance_idle", "EAST") is False
    tree.animation("fighting_stance_idle", "EAST", frames=3)  # meta now present
    assert animation_complete(tree.root, tree.char, "fighting_stance_idle", "EAST") is True


def test_node_complete_dispatches_by_kind(manifest, tree):
    tree.pose("base", "EAST").pose("fighting_stance", "EAST")
    assert node_complete(manifest, tree.root, tree.char, "base", "EAST") is True
    assert node_complete(manifest, tree.root, tree.char, "fighting_stance", "EAST") is True
    assert node_complete(manifest, tree.root, tree.char, "punch", "EAST") is False

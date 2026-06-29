import os

from andypack.resolve import end_anchor, pose_source_image, start_anchor


def test_pose_source_from_concept_is_concept_png(manifest, tree):
    tree.concept()
    src = pose_source_image(manifest, tree.root, tree.char, "base", "EAST")
    assert src.endswith(os.path.join("Cortex", "_concept.png"))


def test_pose_source_from_pose_is_that_pose_png(manifest, tree):
    tree.concept().pose("base", "EAST")
    src = pose_source_image(manifest, tree.root, tree.char, "fighting_stance", "EAST")
    assert src.endswith(os.path.join("_base", "EAST.png"))


def test_animation_anchor_on_pose_uses_pose_png_for_both_slots(manifest, tree):
    # fighting_stance_idle.start_from = fighting_stance (a pose, single image)
    tree.concept().pose("base", "EAST").pose("fighting_stance", "EAST")
    s = start_anchor(manifest, tree.root, tree.char, "fighting_stance_idle", "EAST")
    assert s.endswith(os.path.join("_fighting_stance", "EAST.png"))


def test_punch_anchors_cross_wire_fflf(manifest, tree):
    # start_from idle -> idle.last_frame; end_at idle -> idle.start_frame
    tree.concept().pose("base", "EAST").pose("fighting_stance", "EAST").animation(
        "fighting_stance_idle", "EAST", frames=3
    )
    s = start_anchor(manifest, tree.root, tree.char, "punch", "EAST")
    e = end_anchor(manifest, tree.root, tree.char, "punch", "EAST")
    assert s.endswith(os.path.join("fighting_stance_idle", "EAST", "frame_00002.png"))
    assert e.endswith(os.path.join("fighting_stance_idle", "EAST", "frame_00000.png"))


def test_entry_and_exit_anchors_mix_pose_and_animation(manifest, tree):
    tree.concept().pose("base", "EAST").pose("fighting_stance", "EAST").animation(
        "fighting_stance_idle", "EAST", frames=3
    )
    # entry: start_from base (pose png) ; end_at idle (start_frame)
    assert start_anchor(manifest, tree.root, tree.char, "fighting_stance_entry", "EAST").endswith(
        os.path.join("_base", "EAST.png")
    )
    assert end_anchor(manifest, tree.root, tree.char, "fighting_stance_entry", "EAST").endswith(
        os.path.join("fighting_stance_idle", "EAST", "frame_00000.png")
    )
    # exit: start_from idle (last_frame) ; end_at base (pose png)
    assert start_anchor(manifest, tree.root, tree.char, "fighting_stance_exit", "EAST").endswith(
        os.path.join("fighting_stance_idle", "EAST", "frame_00002.png")
    )
    assert end_anchor(manifest, tree.root, tree.char, "fighting_stance_exit", "EAST").endswith(
        os.path.join("_base", "EAST.png")
    )


def test_anchor_none_when_dep_absent(manifest, tree):
    # idle has no end_at
    assert end_anchor(manifest, tree.root, tree.char, "fighting_stance_idle", "EAST") is None


def test_animation_without_start_from_defaults_to_base(manifest, tree):
    # walk declares no start_from -> uses defaults.start_from (base) as the I2V seed
    tree.concept().pose("base", "EAST")
    s = start_anchor(manifest, tree.root, tree.char, "walk", "EAST")
    assert s.endswith(os.path.join("_base", "EAST.png"))
    # walk has no end_at -> plain I2V, no FFLF end anchor
    assert end_anchor(manifest, tree.root, tree.char, "walk", "EAST") is None

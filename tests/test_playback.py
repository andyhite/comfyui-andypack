from andypack.resolve import animation_frame_dir, playback_segments, pose_image_path


def test_animation_anchors_chain_and_drop_action_boundaries(manifest, tree):
    # punch: start_from=idle (anim), end_at=idle (anim). Both rendered.
    tree.concept().pose("base", "EAST").pose("fighting_stance", "EAST").animation(
        "fighting_stance_idle", "EAST", frames=3
    ).animation("punch", "EAST", frames=3)
    root, char = tree.root, tree.char

    segs = playback_segments(manifest, root, char, "punch", "EAST", loops=3, fps=16)

    assert [s["kind"] for s in segs] == ["anim", "anim", "anim"]
    idle_dir = animation_frame_dir(root, char, "fighting_stance_idle", "EAST")
    prepend, action, append = segs
    assert prepend["dir"] == idle_dir and prepend["repeat"] == 1
    assert append["dir"] == idle_dir
    # punch returns to its start state (idle->idle) -> loops N; its boundary frames
    # duplicate the neighbour animation frames, so they're dropped.
    assert action["dir"] == animation_frame_dir(root, char, "punch", "EAST")
    assert action["repeat"] == 3
    assert action["drop_first"] is True and action["drop_last"] is True


def test_pose_anchor_is_held_for_fps_frames_and_not_dropped(manifest, tree):
    # walk: no start_from -> default base (pose); no end_at.
    tree.concept().pose("base", "EAST").animation("walk", "EAST", frames=3)
    root, char = tree.root, tree.char

    segs = playback_segments(manifest, root, char, "walk", "EAST", loops=2, fps=8)

    assert [s["kind"] for s in segs] == ["hold", "anim"]
    hold, action = segs
    assert hold["image"] == pose_image_path(root, char, "base", "EAST")
    assert hold["count"] == 8  # held for `fps` frames
    # not a self-returning clip (no end_at) -> plays once; pose anchor -> no drop.
    assert action["repeat"] == 1
    assert action["drop_first"] is False and action["drop_last"] is False


def test_unrendered_anchors_are_skipped_but_loops_still_apply(manifest, tree):
    # Only punch rendered; its idle anchor is missing.
    tree.concept().animation("punch", "EAST", frames=3)
    root, char = tree.root, tree.char

    segs = playback_segments(manifest, root, char, "punch", "EAST", loops=3, fps=16)

    assert [s["kind"] for s in segs] == ["anim"]  # no prepend/append
    (action,) = segs
    assert action["repeat"] == 3  # loopability is structural, not dep-gated
    assert action["drop_first"] is False and action["drop_last"] is False

import os

from andypack.resolve import resolve_animation, resolve_pose, status


def test_base_pose_ready_when_concept_present(manifest, tree):
    tree.concept()
    r = resolve_pose(manifest, tree.root, tree.char, "base", "EAST")
    assert r["selectable"] is True
    assert r["blocked_by"] == []
    assert r["source_image"].endswith(os.path.join("Cortex", "_concept.png"))
    assert status(manifest, tree.root, tree.char, "base", "EAST") == "ready"


def test_base_pose_blocked_when_concept_missing(manifest, tree):
    r = resolve_pose(manifest, tree.root, tree.char, "base", "EAST")
    assert r["selectable"] is False
    assert status(manifest, tree.root, tree.char, "base", "EAST") == "blocked"


def test_pose_generated_status(manifest, tree):
    tree.concept().pose("base", "EAST")
    assert status(manifest, tree.root, tree.char, "base", "EAST") == "generated"


def test_fighting_stance_unlocks_after_base(manifest, tree):
    tree.concept()
    assert status(manifest, tree.root, tree.char, "fighting_stance", "EAST") == "blocked"
    tree.pose("base", "EAST")
    assert status(manifest, tree.root, tree.char, "fighting_stance", "EAST") == "ready"


def test_punch_blocked_until_idle(manifest, tree):
    tree.concept().pose("base", "EAST").pose("fighting_stance", "EAST")
    r = resolve_animation(manifest, tree.root, tree.char, "punch", "EAST")
    assert r["selectable"] is False
    slots = {k for entry in r["blocked_by"] for k in entry if k != "dir"}
    assert slots == {"start_from", "end_at"}
    assert status(manifest, tree.root, tree.char, "punch", "EAST") == "blocked"


def test_punch_ready_with_anchors_after_idle(manifest, tree):
    tree.concept().pose("base", "EAST").pose("fighting_stance", "EAST").animation(
        "fighting_stance_idle", "EAST", frames=3
    )
    r = resolve_animation(manifest, tree.root, tree.char, "punch", "EAST")
    assert r["selectable"] is True
    assert r["start_image"].endswith(os.path.join("fighting_stance_idle", "EAST", "frame_00002.png"))
    assert r["end_image"].endswith(os.path.join("fighting_stance_idle", "EAST", "frame_00000.png"))
    assert status(manifest, tree.root, tree.char, "punch", "EAST") == "ready"
    assert r["meta"]["prompt_hash"].startswith("sha1:")


def test_direction_outside_map_not_selectable(manifest, tree):
    tree.concept().pose("base", "EAST").pose("fighting_stance", "EAST").animation(
        "fighting_stance_idle", "EAST", frames=3
    )
    r = resolve_animation(manifest, tree.root, tree.char, "punch", "SOUTH")
    assert r["selectable"] is False  # punch.directions only has EAST


def test_free_clip_blocked_until_base_then_starts_from_it(manifest, tree):
    # walk has no start_from -> default base. It must be blocked until base exists
    # (I2V needs a start image), then it starts from base with no FFLF end.
    tree.concept()
    r = resolve_animation(manifest, tree.root, tree.char, "walk", "EAST")
    assert r["selectable"] is False
    assert status(manifest, tree.root, tree.char, "walk", "EAST") == "blocked"

    tree.pose("base", "EAST")
    r2 = resolve_animation(manifest, tree.root, tree.char, "walk", "EAST")
    assert r2["selectable"] is True
    assert r2["start_image"].endswith(os.path.join("_base", "EAST.png"))
    assert r2["end_image"] is None  # plain I2V, no FFLF
    assert status(manifest, tree.root, tree.char, "walk", "EAST") == "ready"

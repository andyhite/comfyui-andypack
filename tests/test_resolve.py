import os

import pytest

from andypack.manifest import ManifestError
from andypack.resolve import (
    effective_manifest,
    resolve_animation,
    resolve_pose,
    status,
)


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


def test_loop_is_derived_from_matching_anchors(manifest, tree):
    # A clip whose start and end anchors are the SAME single image is a loop.
    tree.identity(animations={
        "spin_loop": {
            "category": "x", "directions": {"EAST": {}},
            "start_from": {"ref": "base"}, "end_at": {"ref": "base"},
        }
    })
    tree.concept().pose("base", "EAST")
    eff = effective_manifest(manifest, tree.root, tree.char)
    r = resolve_animation(eff, tree.root, tree.char, "spin_loop", "EAST")
    assert r["meta"]["loop"] is True

    # punch starts on idle's last frame and ends on idle's first frame — different
    # images, so it is NOT a loop (no manifest flag involved).
    r2 = resolve_animation(manifest, tree.root, tree.char, "punch", "EAST")
    assert r2["meta"]["loop"] is False


def test_effective_manifest_merges_character_entities(manifest, tree):
    tree.identity(
        poses={"special_pose": {"from": {"ref": "base"}, "directions": {"EAST": {}}}},
        animations={
            "special_move": {
                "category": "combat", "directions": {"EAST": {}},
                "start_from": {"ref": "fighting_stance"},
            }
        },
    )
    eff = effective_manifest(manifest, tree.root, tree.char)
    # character entities present alongside the main ones
    assert "special_pose" in eff["poses"] and "base" in eff["poses"]
    assert "special_move" in eff["animations"] and "punch" in eff["animations"]
    # the base manifest is not mutated
    assert "special_pose" not in manifest["poses"]


def test_effective_manifest_no_character_entities_returns_same(manifest, tree):
    tree.identity(positive_prompt="just an identity prompt")  # no poses/animations
    assert effective_manifest(manifest, tree.root, tree.char) is manifest


def test_effective_manifest_rejects_bad_character_ref(manifest, tree):
    # a character pose referencing an unknown ref must fail, not resolve silently
    tree.identity(poses={"bad": {"from": {"ref": "nope"}, "directions": {"EAST": {}}}})
    with pytest.raises(ManifestError):
        effective_manifest(manifest, tree.root, tree.char)


def test_character_animation_is_resolvable(manifest, tree):
    tree.identity(
        animations={
            "special_move": {
                "category": "combat", "directions": {"EAST": {}},
                "start_from": {"ref": "base"}, "positive_prompt": "a special move",
            }
        }
    )
    eff = effective_manifest(manifest, tree.root, tree.char)
    tree.concept()
    assert status(eff, tree.root, tree.char, "special_move", "EAST") == "blocked"
    tree.pose("base", "EAST")
    assert status(eff, tree.root, tree.char, "special_move", "EAST") == "ready"

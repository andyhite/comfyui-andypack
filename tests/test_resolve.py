import os

import pytest

from andypack import resolve
from andypack.manifest import ManifestError
from andypack.resolve import (
    animation_fps,
    effective_manifest,
    resolve_animation,
    resolve_pose,
    status,
    status_from_resolved,
)


def test_root_base_pose_is_ready_with_no_source(manifest, tree):
    # base is the tree root: ready for any declared direction, nothing to block
    # on, and no disk source (the creator node supplies the reference image).
    r = resolve_pose(manifest, tree.root, tree.char, "base", "EAST")
    assert r["selectable"] is True
    assert r["blocked_by"] == []
    assert r["source_image"] is None
    assert status(manifest, tree.root, tree.char, "base", "EAST") == "ready"


def test_base_pose_not_selectable_for_undeclared_direction(manifest, tree):
    # WEST is not in the fixture base.directions, so even a root pose isn't
    # selectable there.
    r = resolve_pose(manifest, tree.root, tree.char, "base", "WEST")
    assert r["selectable"] is False


def test_pose_generated_status(manifest, tree):
    tree.pose("base", "EAST")
    assert status(manifest, tree.root, tree.char, "base", "EAST") == "generated"


def test_fighting_stance_unlocks_after_base(manifest, tree):
    tree.character()
    assert status(manifest, tree.root, tree.char, "fighting_stance", "EAST") == "blocked"
    tree.pose("base", "EAST")
    assert status(manifest, tree.root, tree.char, "fighting_stance", "EAST") == "ready"


def test_punch_blocked_until_idle(manifest, tree):
    tree.pose("base", "EAST").pose("fighting_stance", "EAST")
    r = resolve_animation(manifest, tree.root, tree.char, "punch", "EAST")
    assert r["selectable"] is False
    slots = {k for entry in r["blocked_by"] for k in entry if k != "dir"}
    assert slots == {"start_from", "end_at"}
    assert status(manifest, tree.root, tree.char, "punch", "EAST") == "blocked"


def test_punch_ready_with_anchors_after_idle(manifest, tree):
    tree.pose("base", "EAST").pose("fighting_stance", "EAST").animation(
        "fighting_stance_idle", "EAST", frames=3
    )
    r = resolve_animation(manifest, tree.root, tree.char, "punch", "EAST")
    assert r["selectable"] is True
    assert r["start_image"].endswith(os.path.join("fighting_stance_idle", "EAST", "frame_00002.png"))
    assert r["end_image"].endswith(os.path.join("fighting_stance_idle", "EAST", "frame_00000.png"))
    assert status(manifest, tree.root, tree.char, "punch", "EAST") == "ready"
    assert r["meta"]["prompt_hash"].startswith("sha1:")


def test_direction_outside_map_not_selectable(manifest, tree):
    tree.pose("base", "EAST").pose("fighting_stance", "EAST").animation(
        "fighting_stance_idle", "EAST", frames=3
    )
    r = resolve_animation(manifest, tree.root, tree.char, "punch", "SOUTH")
    assert r["selectable"] is False  # punch.directions only has EAST


def test_free_clip_blocked_until_base_then_starts_from_it(manifest, tree):
    # walk has no start_from -> default base. It must be blocked until base exists
    # (I2V needs a start image), then it starts from base with no FFLF end.
    tree.character()
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
    tree.character(animations={
        "spin_loop": {
            "category": "x", "directions": {"EAST": {}},
            "start_from": {"ref": "base"}, "end_at": {"ref": "base"},
        }
    })
    tree.pose("base", "EAST")
    eff = effective_manifest(manifest, tree.root, tree.char)
    r = resolve_animation(eff, tree.root, tree.char, "spin_loop", "EAST")
    assert r["meta"]["loop"] is True

    # punch starts on idle's last frame and ends on idle's first frame — different
    # images, so it is NOT a loop (no manifest flag involved).
    r2 = resolve_animation(manifest, tree.root, tree.char, "punch", "EAST")
    assert r2["meta"]["loop"] is False


def test_effective_manifest_merges_character_entities(manifest, tree):
    tree.character(
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
    tree.character(positive_prompt="just an identity prompt")  # no poses/animations
    assert effective_manifest(manifest, tree.root, tree.char) is manifest


def test_effective_manifest_tolerates_malformed_poses(manifest, tree):
    # `character.json` is user-authored; a `poses`/`animations` that isn't an
    # object (here a list) must be ignored, not crash the `{**...}` merge.
    tree.character(poses=["oops"], animations="nope")
    assert effective_manifest(manifest, tree.root, tree.char) is manifest


def test_effective_manifest_rejects_bad_character_ref(manifest, tree):
    # a character pose referencing an unknown ref must fail, not resolve silently
    tree.character(poses={"bad": {"from": {"ref": "nope"}, "directions": {"EAST": {}}}})
    with pytest.raises(ManifestError):
        effective_manifest(manifest, tree.root, tree.char)


def test_character_animation_is_resolvable(manifest, tree):
    tree.character(
        animations={
            "special_move": {
                "category": "combat", "directions": {"EAST": {}},
                "start_from": {"ref": "base"}, "positive_prompt": "a special move",
            }
        }
    )
    eff = effective_manifest(manifest, tree.root, tree.char)
    tree.character()
    assert status(eff, tree.root, tree.char, "special_move", "EAST") == "blocked"
    tree.pose("base", "EAST")
    assert status(eff, tree.root, tree.char, "special_move", "EAST") == "ready"


def test_status_from_resolved_matches_status(manifest, tree):
    # status_from_resolved (used to avoid a second resolve on report paths) must
    # agree with status() across the lifecycle states.
    tree.character()  # base ready, fighting_stance blocked
    for ref in ("base", "fighting_stance", "punch"):
        r = resolve_pose(manifest, tree.root, tree.char, ref, "EAST") if ref != "punch" \
            else resolve_animation(manifest, tree.root, tree.char, ref, "EAST")
        assert status_from_resolved(manifest, tree.root, tree.char, ref, "EAST", r) == \
            status(manifest, tree.root, tree.char, ref, "EAST")


def test_animation_fps_uses_manifest_then_default(manifest):
    # punch declares no fps -> inherits defaults.fps (16); a per-animation fps wins.
    assert animation_fps(manifest, "punch") == manifest["defaults"]["fps"]
    manifest["animations"]["punch"]["fps"] = 24
    assert animation_fps(manifest, "punch") == 24


def test_animation_fps_floor_is_one(manifest):
    manifest["animations"]["punch"]["fps"] = 0
    assert animation_fps(manifest, "punch") == 1


def test_read_character_cache_refreshes_on_rewrite(tree):
    # The identity cache is keyed by mtime, so a rewrite is observed (never stale).
    tree.character(positive_prompt="first")
    assert resolve.read_character(tree.root, tree.char) == {"positive_prompt": "first"}
    path = os.path.join(tree.root, tree.char, "character.json")
    with open(path, "w") as fh:
        fh.write('{"positive_prompt": "second"}')
    os.utime(path, (10**9 + 100, 10**9 + 100))  # force a distinct mtime
    assert resolve.read_character(tree.root, tree.char) == {"positive_prompt": "second"}


def test_invalidate_character_defeats_unchanged_mtime(tree):
    # The coarse-mtime case: a rewrite that lands on the SAME mtime would be served
    # stale by the mtime cache; explicit invalidation forces a fresh read.
    tree.character(positive_prompt="first")
    path = os.path.join(tree.root, tree.char, "character.json")
    mtime = os.path.getmtime(path)
    assert resolve.read_character(tree.root, tree.char) == {"positive_prompt": "first"}
    with open(path, "w") as fh:
        fh.write('{"positive_prompt": "second"}')
    os.utime(path, (mtime, mtime))  # pin mtime: cache alone can't see the change
    assert resolve.read_character(tree.root, tree.char) == {"positive_prompt": "first"}
    resolve.invalidate_character(tree.root, tree.char)
    assert resolve.read_character(tree.root, tree.char) == {"positive_prompt": "second"}


def test_effective_manifest_caches_until_invalidated(manifest, tree):
    tree.character(poses={
        "char_pose": {"from": {"ref": "base"}, "directions": {"EAST": {}}},
    })
    first = resolve.effective_manifest(manifest, tree.root, tree.char)
    assert "char_pose" in first["poses"]
    # A second call returns the same validated object (no re-merge / re-validate).
    assert resolve.effective_manifest(manifest, tree.root, tree.char) is first
    # Invalidation (a character-layer change) rebuilds it from the current layer.
    resolve.invalidate_character(tree.root, tree.char)
    rebuilt = resolve.effective_manifest(manifest, tree.root, tree.char)
    assert rebuilt is not first and "char_pose" in rebuilt["poses"]


def test_resolution_pass_memoizes_without_changing_results(manifest, tree):
    # The memo must be transparent: same result inside and outside a pass, and it
    # must not leak past the context.
    tree.pose("base", "EAST", stale=True)
    bare = resolve.outdated(manifest, tree.root, tree.char, "base", "EAST")
    with resolve.resolution_pass():
        assert resolve._OUTDATED_MEMO is not None
        memoed = resolve.outdated(manifest, tree.root, tree.char, "base", "EAST")
        # A repeat call is served from the memo (same value).
        assert resolve.outdated(manifest, tree.root, tree.char, "base", "EAST") == memoed
    assert resolve._OUTDATED_MEMO is None  # dropped on exit
    assert memoed == bare is True


def test_rendered_directions_skips_unrendered(tmp_path):
    root = str(tmp_path)
    char = "hero"
    manifest = {
        "version": 1,
        "poses": {"base": {"directions": {"EAST": {}, "WEST": {}}}},
        "animations": {},
        "defaults": {},
    }
    base = os.path.join(root, char, "_base")
    os.makedirs(base, exist_ok=True)
    open(os.path.join(base, "EAST.png"), "wb").close()
    from andypack import io
    io.atomic_write_json(
        os.path.join(base, "EAST.json"),
        io.build_pose_sidecar(
            {"prompt_hash": "sha1:x", "direction": "EAST"}, created_utc="t"
        ),
    )
    got = resolve.rendered_directions(manifest, root, char, "pose", "base", ["EAST", "WEST"])
    assert [d for d, _ in got] == ["EAST"]

import json
import os

from andypack import resolve
from andypack.resolve import outdated


def _full_stance_tree(tree):
    return (
        tree.character()
        .pose("base", "EAST")
        .pose("fighting_stance", "EAST")
        .animation("fighting_stance_idle", "EAST", frames=3)
    )


def test_root_pose_never_outdated_via_ancestors(manifest, tree):
    # base is the tree root: with a matching prompt hash it is never outdated
    # (it has no ancestor to inherit staleness from).
    tree.pose("base", "EAST")
    assert outdated(manifest, tree.root, tree.char, "base", "EAST") is False


def test_incomplete_node_is_not_outdated(manifest, tree):
    # nothing rendered -> base is incomplete -> not "stale" (that's blocked territory)
    tree.character()
    assert outdated(manifest, tree.root, tree.char, "base", "EAST") is False


def test_fresh_chain_is_not_outdated(manifest, tree):
    _full_stance_tree(tree)
    assert outdated(manifest, tree.root, tree.char, "fighting_stance_idle", "EAST") is False


def test_own_hash_drift_marks_outdated(manifest, tree):
    _full_stance_tree(tree)
    # re-render fighting_stance with a bogus stored hash
    tree.pose("fighting_stance", "EAST", stale=True)
    assert outdated(manifest, tree.root, tree.char, "fighting_stance", "EAST") is True


def test_malformed_sources_key_does_not_raise(manifest, tree):
    # A sources key lacking '@' (older/hand-edited meta) must be skipped, not crash
    # outdated() — the transitive-hash walk still covers that dependency.
    tree.pose("base", "EAST")
    sidecar_path = os.path.splitext(
        resolve.pose_image_path(tree.root, tree.char, "base", "EAST")
    )[0] + ".json"
    side = json.loads(open(sidecar_path).read())
    side["sources"] = {"concept_no_at_sign": "rid:whatever"}  # malformed key
    with open(sidecar_path, "w") as fh:
        json.dump(side, fh)
    # base's own hash still matches and, as the root pose, it has no ancestor.
    assert outdated(manifest, tree.root, tree.char, "base", "EAST") is False


def test_staleness_is_transitive(manifest, tree):
    # base rendered with a stale hash; idle/punch are otherwise fresh.
    # `outdated` is the staleness predicate for a COMPLETE node (spec §6), so
    # punch must be rendered for its transitive staleness to be observable here
    # (an unrendered punch is the `blocked`/`ready` axis, never `outdated`).
    tree.pose("base", "EAST", stale=True).pose("fighting_stance", "EAST").animation(
        "fighting_stance_idle", "EAST", frames=3
    ).animation("punch", "EAST", frames=3)
    # fighting_stance's own hash is fine, but its ancestor (base) is outdated
    assert outdated(manifest, tree.root, tree.char, "fighting_stance", "EAST") is True
    # ripples all the way to punch (start_from idle -> fighting_stance -> base)
    assert outdated(manifest, tree.root, tree.char, "punch", "EAST") is True


def test_reference_image_drift_makes_pose_stale(manifest, tree):
    tree.pose("base", "EAST").pose("fighting_stance", "EAST")
    assert not resolve.stale_locally(manifest, tree.root, tree.char, "fighting_stance", "EAST")
    # Authoring a reference_image the render didn't use -> stale (own reasons).
    manifest["poses"]["fighting_stance"]["directions"]["EAST"]["reference_image"] = "fs_EAST.png"
    assert resolve.stale_locally(manifest, tree.root, tree.char, "fighting_stance", "EAST")
    assert resolve.outdated(manifest, tree.root, tree.char, "fighting_stance", "EAST")

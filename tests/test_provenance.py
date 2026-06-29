import json
import os

from andypack import io, resolve


def _pose_sidecar(tree, pose_id, direction, *, rid, sources=None):
    """Write a pose payload + sidecar with an explicit render_id (and optional
    recorded sources), bypassing the conftest Tree so we control provenance."""
    base = os.path.join(tree.root, tree.char, f"_{pose_id}")
    os.makedirs(base, exist_ok=True)
    open(os.path.join(base, f"{direction}.png"), "w").close()
    data = {
        "kind": "pose", "pose": pose_id, "direction": direction,
        "from": tree.m["poses"][pose_id]["from"], "image": f"{direction}.png",
        "manifest_version": tree.m["version"],
        "prompt_hash": resolve.compute_prompt_hash(
            tree.m, tree.root, tree.char, "pose", pose_id, direction
        ),
        "render_id": rid,
    }
    if sources is not None:
        data["sources"] = sources
    with open(os.path.join(base, f"{direction}.json"), "w") as fh:
        json.dump(data, fh)


def test_render_id_changes_on_rerender_same_prompt():
    a = io.render_id("sha1:abc", "2026-06-29T00:00:00Z")
    b = io.render_id("sha1:abc", "2026-06-29T00:00:01Z")  # later re-render
    assert a != b  # same prompt, different render => different identity


def test_provenance_flags_rerendered_ancestor(manifest, tree):
    tree.concept()
    _pose_sidecar(tree, "base", "EAST", rid="rid:A")
    _pose_sidecar(tree, "fighting_stance", "EAST", rid="rid:X", sources={"base@EAST": "rid:A"})
    # Recorded source render_id matches base's -> not stale.
    assert resolve.outdated(manifest, tree.root, tree.char, "fighting_stance", "EAST") is False

    # base re-rendered with the SAME prompt (prompt_hash unchanged) but a new
    # render_id. Transitive-hash staleness can't see this; provenance can.
    _pose_sidecar(tree, "base", "EAST", rid="rid:B")
    assert resolve.outdated(manifest, tree.root, tree.char, "fighting_stance", "EAST") is True


def test_pre_provenance_meta_falls_back_to_transitive(manifest, tree):
    # Sidecars without a `sources` key (older renders) must still resolve via the
    # transitive-hash walk, not crash.
    tree.concept().pose("base", "EAST").pose("fighting_stance", "EAST")
    assert resolve.outdated(manifest, tree.root, tree.char, "fighting_stance", "EAST") is False

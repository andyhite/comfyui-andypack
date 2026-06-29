import os

from andypack import api


def test_list_characters_finds_dirs_with_concept(tree):
    tree.concept()  # creates <root>/Cortex/_concept.png
    os.makedirs(os.path.join(tree.root, "NotAChar"), exist_ok=True)  # empty -> excluded
    names = [c["name"] for c in api.list_characters(tree.root)]
    assert names == ["Cortex"]


def test_format_blocked_renders_ref_at_dir():
    blocked = [{"start_from": {"ref": "fighting_stance_idle"}, "dir": "E"},
               {"end_at": {"ref": "fighting_stance_idle"}, "dir": "E"}]
    assert api.format_blocked(blocked) == ["fighting_stance_idle@E", "fighting_stance_idle@E"]


def test_list_options_reports_status_and_blocked(manifest, tree):
    tree.concept()  # only concept present
    opts = {(o["kind"], o["id"], o["direction"]): o for o in api.list_options(manifest, tree.root, tree.char)}

    assert opts[("pose", "base", "E")]["status"] == "ready"
    assert opts[("pose", "fighting_stance", "E")]["status"] == "blocked"
    assert opts[("animation", "punch", "E")]["status"] == "blocked"
    assert opts[("animation", "punch", "E")]["blocked_by"] == [
        "fighting_stance_idle@E", "fighting_stance_idle@E"
    ]
    # base covers three directions
    assert {k for k in opts if k[0] == "pose" and k[1] == "base"} == {
        ("pose", "base", "E"), ("pose", "base", "SE"), ("pose", "base", "S")
    }


def test_resolve_payload_pose_has_source_preview(manifest, tree):
    tree.concept()
    p = api.resolve_payload(manifest, tree.root, tree.char, "base", "E")
    assert p["selectable"] is True
    assert p["source_preview"]["ref"] == "concept"
    assert "/anim_coord/frame?" in p["source_preview"]["url"]


def test_resolve_payload_animation_has_dual_previews(manifest, tree):
    tree.concept().pose("base", "E").pose("fighting_stance", "E").animation(
        "fighting_stance_idle", "E", frames=3
    )
    p = api.resolve_payload(manifest, tree.root, tree.char, "punch", "E")
    assert p["selectable"] is True
    assert p["start_preview"]["ref"] == "fighting_stance_idle"
    assert p["end_preview"]["ref"] == "fighting_stance_idle"
    assert p["start_preview"]["url"].count("path=") == 1


def test_frame_path_confines_to_root(tree):
    tree.concept()
    ok = api.frame_path(tree.root, os.path.join("Cortex", "_concept.png"))
    assert ok is not None and ok.endswith("_concept.png")
    assert api.frame_path(tree.root, "../escape.png") is None
    assert api.frame_path(tree.root, "Cortex/missing.png") is None  # 404: doesn't exist


def test_user_default_base_is_none_outside_comfyui():
    # folder_paths is a ComfyUI-only module; absent here -> None.
    assert api.user_default_base() is None


def test_manifests_dir_and_list_are_empty_without_comfyui():
    assert api.manifests_dir() is None
    assert api.list_manifest_names() == []


def test_resolve_manifest_path_passthrough_without_comfyui():
    # No ComfyUI base -> relative path falls back to itself (CWD-relative).
    assert api.resolve_manifest_path("default.json") == "default.json"
    assert api.resolve_manifest_path("/abs/animations.json") == "/abs/animations.json"

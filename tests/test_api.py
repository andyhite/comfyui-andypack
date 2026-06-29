import os

from andypack import api


def test_list_characters_finds_dirs_with_concept(tree):
    tree.concept()  # creates <root>/Cortex/_concept.png
    os.makedirs(os.path.join(tree.root, "NotAChar"), exist_ok=True)  # empty -> excluded
    names = [c["name"] for c in api.list_characters(tree.root)]
    assert names == ["Cortex"]


def test_format_blocked_renders_ref_at_dir():
    blocked = [{"start_from": {"ref": "fighting_stance_idle"}, "dir": "EAST"},
               {"end_at": {"ref": "fighting_stance_idle"}, "dir": "EAST"}]
    assert api.format_blocked(blocked) == ["fighting_stance_idle@EAST", "fighting_stance_idle@EAST"]


def test_list_options_reports_status_and_blocked(manifest, tree):
    tree.concept()  # only concept present
    opts = {(o["kind"], o["id"], o["direction"]): o for o in api.list_options(manifest, tree.root, tree.char)}

    assert opts[("pose", "base", "EAST")]["status"] == "ready"
    assert opts[("pose", "fighting_stance", "EAST")]["status"] == "blocked"
    assert opts[("animation", "punch", "EAST")]["status"] == "blocked"
    assert opts[("animation", "punch", "EAST")]["blocked_by"] == [
        "fighting_stance_idle@EAST", "fighting_stance_idle@EAST"
    ]
    # base covers three directions
    assert {k for k in opts if k[0] == "pose" and k[1] == "base"} == {
        ("pose", "base", "EAST"), ("pose", "base", "SOUTH_EAST"), ("pose", "base", "SOUTH")
    }


def test_resolve_payload_pose_has_source_preview(manifest, tree):
    tree.concept()
    p = api.resolve_payload(manifest, tree.root, tree.char, "base", "EAST")
    assert p["selectable"] is True
    assert p["source_preview"]["ref"] == "concept"
    assert "/anim_coord/frame?" in p["source_preview"]["url"]


def test_resolve_payload_animation_has_dual_previews(manifest, tree):
    tree.concept().pose("base", "EAST").pose("fighting_stance", "EAST").animation(
        "fighting_stance_idle", "EAST", frames=3
    )
    p = api.resolve_payload(manifest, tree.root, tree.char, "punch", "EAST")
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


def test_split_character_dir():
    assert api.split_character_dir("output/characters/cortex") == ("output/characters", "cortex")
    assert api.split_character_dir("output/characters/cortex/") == ("output/characters", "cortex")
    assert api.split_character_dir("cortex") == ("", "cortex")


def test_characters_dir_is_none_outside_comfyui():
    assert api.characters_dir() is None


def test_list_subdirs(tmp_path):
    (tmp_path / "cortex").mkdir()
    (tmp_path / "boss").mkdir()
    (tmp_path / ".hidden").mkdir()
    (tmp_path / "notes.txt").write_text("x")
    assert api.list_subdirs(str(tmp_path)) == ["boss", "cortex"]
    assert api.list_subdirs(None) == []
    assert api.list_subdirs(str(tmp_path / "nope")) == []


def test_manifest_options_maps_ids_to_directions(manifest):
    opts = api.manifest_options(manifest)
    assert opts["poses"]["base"] == ["EAST", "SOUTH_EAST", "SOUTH"]
    assert opts["poses"]["fighting_stance"] == ["EAST"]
    assert opts["animations"]["punch"] == ["EAST"]
    assert set(opts["animations"]) >= {"fighting_stance_idle", "punch", "fighting_stance_entry"}


def test_resolve_manifest_path_passthrough_without_comfyui():
    # No ComfyUI base -> relative path falls back to itself (CWD-relative).
    assert api.resolve_manifest_path("default.json") == "default.json"
    assert api.resolve_manifest_path("/abs/animations.json") == "/abs/animations.json"

import json
import os

from andypack import io


def test_atomic_write_json_writes_and_replaces(tmp_path):
    p = str(tmp_path / "sub" / "meta.json")
    io.atomic_write_json(p, {"a": 1})
    assert json.loads(open(p).read()) == {"a": 1}
    # no leftover temp files in the directory
    assert os.listdir(tmp_path / "sub") == ["meta.json"]


def test_frame_name_zero_pads():
    assert io.frame_name(0) == "frame_00000.png"
    assert io.frame_name(42) == "frame_00042.png"


def test_loop_closure_drop_last():
    assert io.apply_loop_closure([1, 2, 3, 4], drop_last=True) == [1, 2, 3]


def test_loop_closure_drop_first():
    assert io.apply_loop_closure([1, 2, 3, 4], drop_first=True) == [2, 3, 4]


def test_loop_closure_drop_both():
    assert io.apply_loop_closure([1, 2, 3, 4], drop_first=True, drop_last=True) == [2, 3]


def test_loop_closure_no_flags_is_identity():
    assert io.apply_loop_closure([1, 2, 3]) == [1, 2, 3]


def test_loop_closure_single_frame_unchanged():
    assert io.apply_loop_closure([7], drop_first=True, drop_last=True) == [7]


def test_build_pose_sidecar_carries_meta_plus_timestamp():
    meta = {"kind": "pose", "pose": "base", "direction": "EAST",
            "from": {"ref": "concept"}, "image": "EAST.png",
            "manifest_version": 1, "prompt_hash": "sha1:abc"}
    side = io.build_pose_sidecar(meta, created_utc="2026-06-29T00:00:00Z")
    assert side["prompt_hash"] == "sha1:abc"
    assert side["image"] == "EAST.png"
    assert side["created_utc"] == "2026-06-29T00:00:00Z"


def test_build_animation_meta_adds_frame_pointers():
    meta = {"kind": "animation", "animation": "punch", "direction": "EAST",
            "fps": 16, "length": 21, "loop": False,
            "manifest_version": 1, "prompt_hash": "sha1:abc"}
    full = io.build_animation_meta(meta, count=21, start_frame="frame_00000.png",
                                   last_frame="frame_00020.png", seed=7,
                                   created_utc="2026-06-29T00:00:00Z")
    assert full["frames"] == {"dir": ".", "pattern": "frame_{:05d}.png", "count": 21}
    assert full["start_frame"] == "frame_00000.png"
    assert full["last_frame"] == "frame_00020.png"
    assert full["seed"] == 7


def test_remove_if_exists_is_idempotent(tmp_path):
    p = tmp_path / "x.json"
    p.write_text("{}")
    io.remove_if_exists(str(p))
    assert not p.exists()
    io.remove_if_exists(str(p))  # missing -> no error


def test_clear_frames_removes_only_frame_pngs(tmp_path):
    (tmp_path / "frame_00000.png").write_text("x")
    (tmp_path / "frame_00001.png").write_text("x")
    (tmp_path / "meta.json").write_text("{}")
    (tmp_path / "note.txt").write_text("x")
    io.clear_frames(str(tmp_path))
    assert sorted(os.listdir(tmp_path)) == ["meta.json", "note.txt"]


def test_clear_frames_missing_dir_is_noop(tmp_path):
    io.clear_frames(str(tmp_path / "nope"))  # no raise


def test_resolve_under_joins_relative_to_base():
    assert io.resolve_under("/comfy/user/default", "animations.json") == (
        os.path.join("/comfy/user/default", "animations.json")
    )


def test_resolve_under_absolute_passes_through():
    assert io.resolve_under("/comfy/user/default", "/abs/animations.json") == "/abs/animations.json"


def test_resolve_under_no_base_passes_through():
    assert io.resolve_under(None, "animations.json") == "animations.json"


def test_list_json_names_sorted_basenames(tmp_path):
    (tmp_path / "combat.json").write_text("{}")
    (tmp_path / "default.json").write_text("{}")
    (tmp_path / "notes.txt").write_text("x")
    (tmp_path / "sub").mkdir()
    assert io.list_json_names(str(tmp_path)) == ["combat.json", "default.json"]


def test_list_json_names_missing_or_none_returns_empty(tmp_path):
    assert io.list_json_names(None) == []
    assert io.list_json_names(str(tmp_path / "nope")) == []


def test_to_snake_case_lowercases_and_separates():
    assert io.to_snake_case("Cortex") == "cortex"
    assert io.to_snake_case("My Character") == "my_character"
    assert io.to_snake_case("boss-fight 2") == "boss_fight_2"


def test_to_snake_case_trims_and_collapses():
    assert io.to_snake_case("  Spaced Out  ") == "spaced_out"
    assert io.to_snake_case("a__b--c") == "a_b_c"
    assert io.to_snake_case("Águila!") == "guila"  # non-ascii/punct dropped


def test_to_snake_case_rejects_empty_result():
    import pytest
    with pytest.raises(ValueError):
        io.to_snake_case("!!!")
    with pytest.raises(ValueError):
        io.to_snake_case("")


def test_build_character_is_just_the_layer_when_no_existing():
    out = io.build_character({"positive_prompt": "a hero", "negative_prompt": "blurry"})
    assert out == {"positive_prompt": "a hero", "negative_prompt": "blurry"}
    assert "render_id" not in out and "prompt_hash" not in out and "created_utc" not in out


def test_build_character_preserves_overlay_and_drops_cleared_keys():
    existing = {
        "positive_prompt": "old", "negative_prompt": "old neg",
        "poses": {"wave": {"from": {"ref": "base"}, "directions": {"EAST": {}}}},
    }
    # New layer omits negative_prompt (widget cleared) — it must be dropped,
    # while the character-authored `poses` overlay survives.
    out = io.build_character({"positive_prompt": "new"}, existing=existing)
    assert out["positive_prompt"] == "new"
    assert "negative_prompt" not in out
    assert out["poses"] == existing["poses"]


def test_sidecar_records_has_alpha():
    meta = {"prompt_hash": "sha1:x", "direction": "EAST"}
    s = io.build_pose_sidecar(meta, created_utc="2026-01-01T00:00:00Z", has_alpha=True)
    assert s["has_alpha"] is True

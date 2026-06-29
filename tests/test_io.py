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
    assert io.apply_loop_closure([1, 2, 3, 4], "drop_last") == [1, 2, 3]


def test_loop_closure_duplicate_first():
    assert io.apply_loop_closure([1, 2, 3], "duplicate_first") == [1, 2, 3, 1]


def test_build_pose_sidecar_carries_meta_plus_timestamp():
    meta = {"kind": "pose", "pose": "base", "direction": "E",
            "from": {"ref": "concept"}, "image": "E.png",
            "manifest_version": 1, "prompt_hash": "sha1:abc"}
    side = io.build_pose_sidecar(meta, created_utc="2026-06-29T00:00:00Z")
    assert side["prompt_hash"] == "sha1:abc"
    assert side["image"] == "E.png"
    assert side["created_utc"] == "2026-06-29T00:00:00Z"


def test_build_animation_meta_adds_frame_pointers():
    meta = {"kind": "animation", "animation": "punch", "direction": "E",
            "fps": 16, "length": 21, "loop": False,
            "manifest_version": 1, "prompt_hash": "sha1:abc"}
    full = io.build_animation_meta(meta, count=21, start_frame="frame_00000.png",
                                   last_frame="frame_00020.png", seed=7,
                                   created_utc="2026-06-29T00:00:00Z")
    assert full["frames"] == {"dir": ".", "pattern": "frame_{:05d}.png", "count": 21}
    assert full["start_frame"] == "frame_00000.png"
    assert full["last_frame"] == "frame_00020.png"
    assert full["seed"] == 7


def test_safe_path_allows_inside(tmp_path):
    root = str(tmp_path)
    target = io.safe_path(root, "Cortex/_base/E.png")
    assert target is not None and target.startswith(os.path.realpath(root))


def test_safe_path_rejects_dotdot(tmp_path):
    assert io.safe_path(str(tmp_path), "../../etc/passwd") is None


def test_safe_path_rejects_absolute(tmp_path):
    assert io.safe_path(str(tmp_path), "/etc/passwd") is None


def test_safe_path_rejects_symlink_escape(tmp_path):
    outside = tmp_path.parent / "outside_secret"
    outside.mkdir()
    (tmp_path / "link").symlink_to(outside)
    assert io.safe_path(str(tmp_path), "link/secret.txt") is None


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

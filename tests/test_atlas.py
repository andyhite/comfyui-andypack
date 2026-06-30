"""Tests for andypack.atlas engine-format serializers."""

from __future__ import annotations

import json

from andypack import atlas

ATLAS: dict = {
    "sheet_size": [12, 6],
    "columns": 2,
    "frames": [
        {
            "rect": [0, 0, 6, 6],
            "source_size": [6, 6],
            "offset": [0, 0],
            "pivot": [3, 6],
            "duration_ms": 125,
        },
        {
            "rect": [6, 0, 6, 6],
            "source_size": [6, 6],
            "offset": [0, 0],
            "pivot": [3, 6],
            "duration_ms": 125,
        },
    ],
}


def test_aseprite_shape() -> None:
    text, ext = atlas.serialize(ATLAS, "walk", "aseprite")
    data = json.loads(text)
    assert ext == ".json"
    assert "frames" in data
    assert data["meta"]["size"] == {"w": 12, "h": 6}


def test_godot_is_tres() -> None:
    text, ext = atlas.serialize(ATLAS, "walk", "godot_spriteframes")
    assert ext == ".tres"
    assert "SpriteFrames" in text


def test_json_hash_keys() -> None:
    text, ext = atlas.serialize(ATLAS, "walk", "json_hash")
    data = json.loads(text)
    assert ext == ".json"
    assert "walk_0.png" in data["frames"]
    assert "walk_1.png" in data["frames"]
    assert data["meta"]["image"] == "walk.png"
    assert data["meta"]["size"] == {"w": 12, "h": 6}


def test_json_array_structure() -> None:
    text, ext = atlas.serialize(ATLAS, "walk", "json_array")
    data = json.loads(text)
    assert ext == ".json"
    assert isinstance(data["frames"], list)
    assert len(data["frames"]) == 2
    assert data["frames"][0]["filename"] == "walk_0.png"


def test_texturepacker_is_hash() -> None:
    text_tp, _ = atlas.serialize(ATLAS, "walk", "texturepacker")
    text_jh, _ = atlas.serialize(ATLAS, "walk", "json_hash")
    assert text_tp == text_jh


def test_css_rules() -> None:
    text, ext = atlas.serialize(ATLAS, "walk", "css")
    assert ext == ".css"
    assert ".walk" in text
    assert ".walk-0" in text
    assert ".walk-1" in text
    assert "background-position" in text


def test_unity_meta_yaml() -> None:
    text, ext = atlas.serialize(ATLAS, "walk", "unity")
    assert ext == ".meta"
    assert "TextureImporter" in text
    assert "spriteSheet" in text


def test_serialize_unknown_fmt_raises() -> None:
    try:
        atlas.serialize(ATLAS, "walk", "bogus_format")
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert "bogus_format" in str(exc)


def test_aseprite_frame_count() -> None:
    text, _ = atlas.serialize(ATLAS, "walk", "aseprite")
    data = json.loads(text)
    assert len(data["frames"]) == 2


def test_godot_contains_frame_rects() -> None:
    text, _ = atlas.serialize(ATLAS, "walk", "godot_spriteframes")
    # Both frame rects should appear in the .tres text.
    assert "0, 0, 6, 6" in text
    assert "6, 0, 6, 6" in text


def test_json_hash_frame_fields() -> None:
    text, _ = atlas.serialize(ATLAS, "walk", "json_hash")
    data = json.loads(text)
    frame = data["frames"]["walk_0.png"]
    assert "frame" in frame
    assert frame["frame"] == {"x": 0, "y": 0, "w": 6, "h": 6}
    assert "spriteSourceSize" in frame
    assert "sourceSize" in frame
    assert "duration" in frame


def test_duration_none_handled() -> None:
    atlas_no_dur: dict = {
        "sheet_size": [6, 6],
        "columns": 1,
        "frames": [
            {
                "rect": [0, 0, 6, 6],
                "source_size": [6, 6],
                "offset": [0, 0],
                "pivot": None,
                "duration_ms": None,
            }
        ],
    }
    text, _ = atlas.serialize(atlas_no_dur, "idle", "aseprite")
    data = json.loads(text)
    # duration should still be present (defaulting to 0 or 100)
    assert "duration" in list(data["frames"].values())[0]

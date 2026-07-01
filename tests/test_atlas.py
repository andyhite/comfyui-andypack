"""Tests for andypack.atlas engine-format serializers."""

from __future__ import annotations

import hashlib
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
    # Y-flip: sheet_size=[12,6], frame rect [0,0,6,6] -> unity_y = 6-0-6 = 0
    assert "y: 0" in text
    # GUID determinism: two calls with the same name produce identical output.
    text2, _ = atlas.serialize(ATLAS, "walk", "unity")
    assert text == text2
    # GUID value is pinned to the MD5 of the name so a regression to hash()
    # would be caught immediately.
    expected_guid = hashlib.md5(b"walk").hexdigest()
    assert expected_guid in text


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


# --- Producer/serializer contract + per-direction tags ---------------------- #

import torch  # noqa: E402

from andypack import sprites  # noqa: E402


def _producer_atlases() -> dict:
    """One atlas from every ANIM_ATLAS producer in the pack."""
    _s1, a_flat = sprites.pack_sheet(torch.ones((2, 4, 4, 4)), layout="grid", columns=2)
    _s2, a_rows = sprites.pack_direction_rows(
        [("EAST", [torch.ones((1, 4, 4, 4))] * 2), ("SOUTH", [torch.ones((1, 4, 4, 4))])],
        fps=16,
    )
    return {"pack_sheet": a_flat, "pack_direction_rows": a_rows}


def test_every_producer_serializes_in_every_format() -> None:
    # Guards the producer→serializer contract: every atlas a producer emits must
    # serialize in every format (every producer carries sheet_size + frames).
    for producer, a in _producer_atlases().items():
        for fmt in atlas._FMT_MAP:
            text, ext = atlas.serialize(a, "clip", fmt)
            assert text, f"{producer} -> {fmt} produced empty output"
            assert ext.startswith(".")
            if ext == ".json":
                json.loads(text)  # must be valid JSON


def test_direction_tags_become_aseprite_frametags() -> None:
    _s, a = sprites.pack_direction_rows(
        [("EAST", [torch.ones((1, 4, 4, 4))] * 2), ("SOUTH", [torch.ones((1, 4, 4, 4))] * 3)],
        fps=16,
    )
    ase = json.loads(atlas.serialize(a, "walk", "aseprite")[0])
    tags = ase["meta"]["frameTags"]
    assert [t["name"] for t in tags] == ["EAST", "SOUTH"]
    assert (tags[0]["from"], tags[0]["to"]) == (0, 1)
    assert (tags[1]["from"], tags[1]["to"]) == (2, 4)


def test_direction_tags_become_godot_animations() -> None:
    _s, a = sprites.pack_direction_rows(
        [("EAST", [torch.ones((1, 4, 4, 4))]), ("SOUTH", [torch.ones((1, 4, 4, 4))])],
        fps=16,
    )
    tres = atlas.serialize(a, "walk", "godot_spriteframes")[0]
    assert '"name": &"EAST"' in tres
    assert '"name": &"SOUTH"' in tres


def test_pack_direction_rows_atlas_shape() -> None:
    sheet, a = sprites.pack_direction_rows(
        [("EAST", [torch.ones((1, 8, 6, 4))] * 3), ("SOUTH", [torch.ones((1, 8, 6, 4))] * 2)],
        fps=20, padding=1,
    )
    assert sheet.shape[0] == 1 and sheet.shape[-1] == 4  # single RGBA batch
    assert a["fps"] == 20
    assert a["columns"] == 3            # widest row
    assert len(a["frames"]) == 5        # 3 + 2
    assert a["frames"][0]["duration_ms"] == 50  # 1000/20
    assert [t["name"] for t in a["tags"]] == ["EAST", "SOUTH"]

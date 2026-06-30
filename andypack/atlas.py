"""Engine-format serializers for ANIM_ATLAS dicts produced by sprites.pack_sheet.

Pure stdlib — no torch / numpy / ComfyUI imports.

Each public serializer takes:
    atlas : dict  — {"sheet_size":[w,h], "columns":n, "frames":[...]}
    name  : str   — base name used for filenames / selectors

and returns a str (the serialized text).

The ``serialize`` dispatcher returns ``(text, file_extension)``.
"""

from __future__ import annotations

import json
from typing import Any


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _sheet_wh(atlas: dict) -> tuple[int, int]:
    w, h = atlas["sheet_size"]
    return int(w), int(h)


def _frame_rect(frame: dict) -> tuple[int, int, int, int]:
    x, y, w, h = frame["rect"]
    return int(x), int(y), int(w), int(h)


def _source_size(frame: dict) -> tuple[int, int]:
    sw, sh = frame["source_size"]
    return int(sw), int(sh)


def _offset(frame: dict) -> tuple[int, int]:
    ox, oy = frame["offset"]
    return int(ox), int(oy)


def _duration(frame: dict, default_ms: int = 100) -> int:
    ms = frame.get("duration_ms")
    return int(ms) if ms is not None else default_ms


# ---------------------------------------------------------------------------
# TexturePacker-style JSON (hash)
# ---------------------------------------------------------------------------


def to_json_hash(atlas: dict, name: str) -> str:
    """Serialize atlas as TexturePacker JSON-hash format."""
    sw, sh = _sheet_wh(atlas)
    frames: dict[str, Any] = {}
    for i, frame in enumerate(atlas["frames"]):
        fx, fy, fw, fh = _frame_rect(frame)
        ssw, ssh = _source_size(frame)
        ox, oy = _offset(frame)
        dur = _duration(frame)
        frames[f"{name}_{i}.png"] = {
            "frame": {"x": fx, "y": fy, "w": fw, "h": fh},
            "spriteSourceSize": {"x": ox, "y": oy, "w": fw, "h": fh},
            "sourceSize": {"w": ssw, "h": ssh},
            "duration": dur,
        }
    data: dict[str, Any] = {
        "frames": frames,
        "meta": {
            "image": f"{name}.png",
            "size": {"w": sw, "h": sh},
        },
    }
    return json.dumps(data, indent=2)


# ---------------------------------------------------------------------------
# TexturePacker-style JSON (array)
# ---------------------------------------------------------------------------


def to_json_array(atlas: dict, name: str) -> str:
    """Serialize atlas as TexturePacker JSON-array format."""
    sw, sh = _sheet_wh(atlas)
    frames: list[dict[str, Any]] = []
    for i, frame in enumerate(atlas["frames"]):
        fx, fy, fw, fh = _frame_rect(frame)
        ssw, ssh = _source_size(frame)
        ox, oy = _offset(frame)
        dur = _duration(frame)
        frames.append(
            {
                "filename": f"{name}_{i}.png",
                "frame": {"x": fx, "y": fy, "w": fw, "h": fh},
                "spriteSourceSize": {"x": ox, "y": oy, "w": fw, "h": fh},
                "sourceSize": {"w": ssw, "h": ssh},
                "duration": dur,
            }
        )
    data: dict[str, Any] = {
        "frames": frames,
        "meta": {
            "image": f"{name}.png",
            "size": {"w": sw, "h": sh},
        },
    }
    return json.dumps(data, indent=2)


# ---------------------------------------------------------------------------
# Aseprite JSON
# ---------------------------------------------------------------------------


def to_aseprite(atlas: dict, name: str) -> str:
    """Serialize atlas as Aseprite {frames, meta} JSON."""
    sw, sh = _sheet_wh(atlas)
    frames: dict[str, Any] = {}
    for i, frame in enumerate(atlas["frames"]):
        fx, fy, fw, fh = _frame_rect(frame)
        ssw, ssh = _source_size(frame)
        ox, oy = _offset(frame)
        dur = _duration(frame)
        key = f"{name} {i}.aseprite"
        frames[key] = {
            "frame": {"x": fx, "y": fy, "w": fw, "h": fh},
            "rotated": False,
            "trimmed": ox != 0 or oy != 0,
            "spriteSourceSize": {"x": ox, "y": oy, "w": fw, "h": fh},
            "sourceSize": {"w": ssw, "h": ssh},
            "duration": dur,
        }
    data: dict[str, Any] = {
        "frames": frames,
        "meta": {
            "app": "https://www.aseprite.org/",
            "version": "1.3",
            "image": f"{name}.png",
            "format": "RGBA8888",
            "size": {"w": sw, "h": sh},
            "scale": "1",
        },
    }
    return json.dumps(data, indent=2)


# ---------------------------------------------------------------------------
# Godot SpriteFrames .tres
# ---------------------------------------------------------------------------


def to_godot_spriteframes(atlas: dict, name: str) -> str:
    """Serialize atlas as a Godot 4 SpriteFrames .tres resource."""
    frames = atlas["frames"]
    num_frames = len(frames)
    # load_steps = 1 (ext_resource) + num_frames (AtlasTexture sub_resources) + 1 (resource)
    load_steps = 1 + num_frames + 1

    lines: list[str] = []
    lines.append(
        f'[gd_resource type="SpriteFrames" load_steps={load_steps} format=3]'
    )
    lines.append("")
    lines.append(
        f'[ext_resource type="Texture2D" path="res://{name}.png" id="1"]'
    )
    lines.append("")

    for i, frame in enumerate(frames):
        fx, fy, fw, fh = _frame_rect(frame)
        lines.append(f'[sub_resource type="AtlasTexture" id="AtlasTexture_{i}"]')
        lines.append('atlas = ExtResource("1")')
        lines.append(f"region = Rect2({fx}, {fy}, {fw}, {fh})")
        lines.append("")

    lines.append("[resource]")

    # Build the animation array inline (Godot 4 .tres style).
    anim_frames_parts: list[str] = []
    for i, frame in enumerate(frames):
        dur_ms = _duration(frame)
        dur_sec = dur_ms / 1000.0
        anim_frames_parts.append(
            '{"duration": '
            + f"{dur_sec:.6f}"
            + f', "texture": SubResource("AtlasTexture_{i}")'
            + "}"
        )

    anim_frames_str = ", ".join(anim_frames_parts)
    fps = 1000.0 / _duration(frames[0]) if frames else 8.0
    lines.append(
        "animations = [{"
        + f'"frames": [{anim_frames_str}], '
        + '"loop": true, '
        + f'"name": &"{name}", '
        + f'"speed": {fps:.1f}'
        + "}]"
    )

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Unity sprite-sheet .meta YAML
# ---------------------------------------------------------------------------


def to_unity_meta(atlas: dict, name: str) -> str:
    """Serialize atlas as a Unity TextureImporter sprite-sheet .meta YAML."""
    lines: list[str] = []
    lines.append("fileFormatVersion: 2")
    lines.append(f"guid: {_name_guid(name)}")
    lines.append("TextureImporter:")
    lines.append("  serializedVersion: 12")
    lines.append("  spriteImportMode: 2")
    lines.append("  spriteMeshType: 1")
    lines.append("  spriteSheet:")
    lines.append("    serializedVersion: 2")
    lines.append("    sprites:")

    sw, sh = _sheet_wh(atlas)
    for i, frame in enumerate(atlas["frames"]):
        fx, fy, fw, fh = _frame_rect(frame)
        # Unity's coordinate system has Y flipped (origin at bottom-left).
        unity_y = sh - fy - fh
        lines.append(f"    - name: {name}_{i}")
        lines.append("      rect:")
        lines.append("        serializedVersion: 2")
        lines.append(f"        x: {fx}")
        lines.append(f"        y: {unity_y}")
        lines.append(f"        width: {fw}")
        lines.append(f"        height: {fh}")
        lines.append("      alignment: 0")
        lines.append("      pivot: {x: 0.5, y: 0}")

    return "\n".join(lines) + "\n"


def _name_guid(name: str) -> str:
    """Produce a deterministic pseudo-GUID from a name string."""
    raw = abs(hash(name)) % (16**32)
    hex_str = f"{raw:032x}"
    return (
        f"{hex_str[0:8]}-"
        f"{hex_str[8:12]}-"
        f"{hex_str[12:16]}-"
        f"{hex_str[16:20]}-"
        f"{hex_str[20:32]}"
    )


# ---------------------------------------------------------------------------
# CSS background-position rules
# ---------------------------------------------------------------------------


def to_css(atlas: dict, name: str) -> str:
    """Serialize atlas as CSS sprite rules with background-position per frame."""
    lines: list[str] = []
    lines.append(f".{name} {{")
    lines.append(f"  background-image: url('{name}.png');")
    lines.append("  background-repeat: no-repeat;")
    lines.append("  display: inline-block;")
    lines.append("}")
    lines.append("")

    for i, frame in enumerate(atlas["frames"]):
        fx, fy, fw, fh = _frame_rect(frame)
        lines.append(f".{name}-{i} {{")
        lines.append(f"  background-position: -{fx}px -{fy}px;")
        lines.append(f"  width: {fw}px;")
        lines.append(f"  height: {fh}px;")
        lines.append("}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# TexturePacker alias
# ---------------------------------------------------------------------------


def to_texturepacker(atlas: dict, name: str) -> str:
    """Alias for to_json_hash (standard TexturePacker hash format)."""
    return to_json_hash(atlas, name)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_FMT_MAP: dict[str, tuple[Any, str]] = {
    "json_hash": (to_json_hash, ".json"),
    "json_array": (to_json_array, ".json"),
    "aseprite": (to_aseprite, ".json"),
    "godot_spriteframes": (to_godot_spriteframes, ".tres"),
    "unity": (to_unity_meta, ".meta"),
    "texturepacker": (to_texturepacker, ".json"),
    "css": (to_css, ".css"),
}


def serialize(atlas: dict, name: str, fmt: str) -> tuple[str, str]:
    """Serialize *atlas* into engine format *fmt*.

    Parameters
    ----------
    atlas:
        ANIM_ATLAS dict produced by ``sprites.pack_sheet``.
    name:
        Base name for the sprite sheet (used in filenames and selectors).
    fmt:
        One of: ``json_hash``, ``json_array``, ``aseprite``,
        ``godot_spriteframes``, ``unity``, ``texturepacker``, ``css``.

    Returns
    -------
    tuple[str, str]
        ``(text, file_extension)`` where ``file_extension`` includes the
        leading dot (e.g. ``".json"``).

    Raises
    ------
    ValueError
        If *fmt* is not a recognised format name.
    """
    if fmt not in _FMT_MAP:
        known = ", ".join(sorted(_FMT_MAP))
        raise ValueError(
            f"Unknown atlas format {fmt!r}. Known formats: {known}"
        )
    fn, ext = _FMT_MAP[fmt]
    text: str = fn(atlas, name)
    return text, ext

import json
import os
from pathlib import Path

import pytest

from andypack import resolve

FIX = Path(__file__).parent / "fixtures" / "manifest.json"


@pytest.fixture
def manifest():
    return json.loads(FIX.read_text())


def _write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh)


def _touch(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w").close()


class Tree:
    """Builds a rendered character tree under a root dir, with correct hashes."""

    def __init__(self, manifest, root, character="Cortex"):
        self.m = manifest
        self.root = root
        self.char = character

    def _cdir(self):
        return os.path.join(self.root, self.char)

    def concept(self):
        _touch(os.path.join(self._cdir(), "_concept.png"))
        return self

    def identity(self, **layer):
        _write_json(os.path.join(self._cdir(), "_concept.json"), layer)
        return self

    def pose(self, pose_id, direction, *, stale=False, sidecar=True):
        base = os.path.join(self._cdir(), f"_{pose_id}")
        _touch(os.path.join(base, f"{direction}.png"))
        if sidecar:
            h = "sha1:STALE" if stale else resolve.compute_prompt_hash(
                self.m, self.root, self.char, "pose", pose_id, direction
            )
            _write_json(os.path.join(base, f"{direction}.json"), {
                "kind": "pose", "pose": pose_id, "direction": direction,
                "from": self.m["poses"][pose_id]["from"], "image": f"{direction}.png",
                "prompt_hash": h, "manifest_version": self.m["version"],
            })
        return self

    def animation(self, anim_id, direction, *, frames=3, stale=False, meta=True):
        base = os.path.join(self._cdir(), anim_id, direction)
        for i in range(frames):
            _touch(os.path.join(base, f"frame_{i:05d}.png"))
        if meta:
            h = "sha1:STALE" if stale else resolve.compute_prompt_hash(
                self.m, self.root, self.char, "animation", anim_id, direction
            )
            _write_json(os.path.join(base, "meta.json"), {
                "kind": "animation", "animation": anim_id, "direction": direction,
                "fps": 16, "length": frames, "loop": False,
                "prompt_hash": h, "manifest_version": self.m["version"],
                "frames": {"dir": ".", "pattern": "frame_{:05d}.png", "count": frames},
                "start_frame": "frame_00000.png", "last_frame": f"frame_{frames - 1:05d}.png",
            })
        return self


@pytest.fixture
def tree(manifest, tmp_path):
    return Tree(manifest, str(tmp_path / "root"))

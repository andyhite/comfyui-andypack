import json
import os

from andypack import api


def test_bundled_manifest_exists_and_is_valid_json():
    # The seed source ships in the repo and must be a loadable manifest.
    with open(api.BUNDLED_MANIFEST, encoding="utf-8") as fh:
        data = json.load(fh)
    assert "version" in data


def test_seed_is_noop_outside_comfyui():
    # No manifests dir (folder_paths absent) -> nothing seeded.
    assert api.seed_default_manifest() is False


def test_seed_copies_bundled_manifest_into_empty_dir(tmp_path, monkeypatch):
    dest_dir = tmp_path / "animations"
    monkeypatch.setattr(api, "manifests_dir", lambda: str(dest_dir))

    assert api.seed_default_manifest() is True

    dest = dest_dir / "default.json"
    assert dest.is_file()
    # Content matches the bundled source byte-for-byte.
    with open(api.BUNDLED_MANIFEST, "rb") as src:
        assert dest.read_bytes() == src.read()


def test_seed_does_not_clobber_existing_manifest(tmp_path, monkeypatch):
    dest_dir = tmp_path / "animations"
    os.makedirs(dest_dir)
    dest = dest_dir / "default.json"
    dest.write_text('{"version": "user-edited"}', encoding="utf-8")
    monkeypatch.setattr(api, "manifests_dir", lambda: str(dest_dir))

    assert api.seed_default_manifest() is False
    assert json.loads(dest.read_text())["version"] == "user-edited"

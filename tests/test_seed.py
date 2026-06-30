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


def test_seed_base_is_root_with_all_eight_directions():
    import json as _json
    from andypack.manifest import topo_order, validate_manifest
    m = _json.loads(open("examples/animations.json", encoding="utf-8").read())
    base = m["poses"]["base"]
    assert "from" not in base                       # base is the tree root
    assert set(base["directions"]) == {
        "EAST", "SOUTH_EAST", "SOUTH", "SOUTH_WEST",
        "WEST", "NORTH_WEST", "NORTH", "NORTH_EAST",
    }
    validate_manifest(m)
    topo_order(m)                                   # no cycle, sorts


def test_seed_uses_character_prompt_token_not_identity():
    raw = open("examples/animations.json", encoding="utf-8").read()
    assert "{identity_prompt}" not in raw
    assert "{character_prompt}" in raw


ALL_DIRS = {
    "EAST", "SOUTH_EAST", "SOUTH", "SOUTH_WEST",
    "WEST", "NORTH_WEST", "NORTH", "NORTH_EAST",
}


def _seed():
    import json as _json
    return _json.loads(open("examples/animations.json", encoding="utf-8").read())


def test_seed_every_pose_and_animation_lists_all_eight_directions():
    # The intended workflow generates EVERY anchor pose and animation in EVERY one
    # of the 8 directions, so every entity must list all 8 (lean on view_phrases +
    # entity prose, not per-direction layers).
    m = _seed()
    for kind in ("poses", "animations"):
        for eid, entity in m[kind].items():
            assert set(entity["directions"]) == ALL_DIRS, f"{kind} {eid!r}"


def test_seed_view_phrases_cover_all_eight_directions():
    m = _seed()
    assert set(m["view_phrases"]) == ALL_DIRS
    # No lint findings (every canonical direction has a phrase; lengths are 4n+1).
    from andypack.manifest import collect_warnings
    assert collect_warnings(m) == []


def test_seed_poses_carry_no_klein_hostile_negatives():
    # FLUX.2 Klein has no negative path: the pose globals carry no negative layer
    # and no pose authors a per-direction/entity negative.
    m = _seed()
    assert not m["globals"].get("pose", {}).get("negative_prompt")
    for pid, pose in m["poses"].items():
        assert "negative_prompt" not in pose, pid
        for d, layer in pose["directions"].items():
            assert "negative_prompt" not in layer, f"{pid}@{d}"


def test_seed_animation_globals_carry_standard_wan_negative():
    m = _seed()
    neg = m["globals"]["animation"]["negative_prompt"]
    # Hallmarks of the standard Wan 2.2 block that fight frozen/reversed clips.
    for term in ("still picture", "walking backwards", "{character_prompt}"):
        assert term in neg


def test_seed_in_place_loops_are_fflf_pinned():
    # Animations whose prompt promises a seamless loop and that return to their
    # start pose must set end_at == start_from (a single-image dep resolves the same
    # image for both slots -> derived loop -> the writer drops the duplicate frame).
    m = _seed()
    for aid in ("standing_idle", "crouch_idle", "fighting_stance_idle", "jump_apex",
                "wall_cling", "block", "talk", "cheer", "edge_teeter"):
        anim = m["animations"][aid]
        assert anim.get("end_at"), f"{aid} should be FFLF-pinned"
        assert anim["end_at"]["ref"] == anim["start_from"]["ref"], aid


def test_seed_non_returning_clips_are_start_only():
    # A 180° turn and a downward wall-slide do NOT return to their start, so they
    # must NOT be same-image FFLF loops (that would pin a contradictory end frame).
    m = _seed()
    for aid in ("turn_around", "wall_slide"):
        assert m["animations"][aid].get("end_at") is None, aid


def test_seed_pose_prompts_stay_under_token_cap():
    # FLUX.2 Klein caps at 512 tokens; keep pose prompts well under (word count is a
    # rough proxy). Guards against the templates ballooning again.
    import tempfile
    from andypack.resolve import merged_prompts
    root = tempfile.mkdtemp()
    os.makedirs(os.path.join(root, "cortex"))
    with open(os.path.join(root, "cortex", "character.json"), "w") as fh:
        json.dump({"positive_prompt": "a mouthless cartoon hero, orange body"}, fh)
    m = _seed()
    for pid in m["poses"]:
        for d in m["poses"][pid]["directions"]:
            pos, _ = merged_prompts(m, root, "cortex", "pose", pid, d)
            assert len(pos.split()) < 130, f"{pid}@{d} prompt too long: {len(pos.split())} words"


def test_seed_every_animation_has_a_start_image_source():
    # Every animation needs an I2V start image: an explicit start_from or the
    # manifest default. validate_manifest enforces it, but assert here too.
    m = _seed()
    default_start = m["defaults"].get("start_from")
    for aid, anim in m["animations"].items():
        assert anim.get("start_from") or default_start, aid

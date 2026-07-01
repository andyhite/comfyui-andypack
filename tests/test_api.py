import os

from andypack import api


def test_list_characters_finds_dirs_with_concept(tree):
    tree.character()  # creates <root>/Cortex/_concept.png
    os.makedirs(os.path.join(tree.root, "NotAChar"), exist_ok=True)  # empty -> excluded
    names = [c["name"] for c in api.list_characters(tree.root)]
    assert names == ["Cortex"]


def test_format_blocked_renders_ref_at_dir():
    blocked = [{"start_from": {"ref": "fighting_stance_idle"}, "dir": "EAST"},
               {"end_at": {"ref": "fighting_stance_idle"}, "dir": "EAST"}]
    assert api.format_blocked(blocked) == ["fighting_stance_idle@EAST", "fighting_stance_idle@EAST"]


def test_list_options_reports_status_and_blocked(manifest, tree):
    tree.character()  # only concept present
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
    # animation options carry their category for the multi-level UI
    assert opts[("animation", "punch", "EAST")]["category"] == "combat"


def test_list_options_includes_character_specific_entities(manifest, tree):
    tree.character(
        animations={
            "special_move": {
                "category": "combat", "directions": {"EAST": {}},
                "start_from": {"ref": "base"},
            }
        }
    )
    ids = {(o["kind"], o["id"]) for o in api.list_options(manifest, tree.root, tree.char)}
    assert ("animation", "special_move") in ids  # character-defined
    assert ("animation", "punch") in ids  # main manifest still present


def test_coverage_report_counts_by_status(manifest, tree):
    tree.character()  # only concept -> base ready, deeper poses/anims blocked
    rep = api.coverage_report(manifest, tree.root, tree.char)
    assert rep["total"] == len(rep["rows"])
    assert rep["summary"]["ready"] >= 1   # base@{EAST,SOUTH_EAST,SOUTH}
    assert rep["summary"]["blocked"] >= 1  # fighting_stance / punch
    table = api.format_coverage_table(rep)
    assert "STATUS" in table and "base" in table


def test_regen_queue_is_dependency_ordered_and_skips_blocked(manifest, tree):
    tree.character()
    queue = api.regen_queue(manifest, tree.root, tree.char)
    cells = [(q["id"], q["direction"]) for q in queue]
    # only base (ready) is actionable now; blocked downstream is omitted
    assert ("base", "EAST") in cells
    assert ("fighting_stance", "EAST") not in cells
    assert all(q["status"] in ("ready", "stale") for q in queue)

    # after base is generated, fighting_stance becomes actionable and follows base
    tree.pose("base", "EAST")
    q2 = api.regen_queue(manifest, tree.root, tree.char)
    ids = [q["id"] for q in q2]
    assert "fighting_stance" in ids


def test_user_default_base_is_none_outside_comfyui():
    # folder_paths is a ComfyUI-only module; absent here -> None.
    assert api.user_default_base() is None


def test_manifests_dir_and_list_are_empty_without_comfyui():
    assert api.manifests_dir() is None
    assert api.list_manifest_names() == []


# --- CRUD helpers for the sidebar GUI --------------------------------------- #

def test_manifest_name_is_safe_rejects_traversal_and_non_json():
    assert api.manifest_name_is_safe("default.json")
    assert api.manifest_name_is_safe("my-walks.json")
    assert not api.manifest_name_is_safe("../escape.json")
    assert not api.manifest_name_is_safe("sub/dir.json")
    assert not api.manifest_name_is_safe("notjson.txt")
    assert not api.manifest_name_is_safe("")
    assert not api.manifest_name_is_safe(".json")


def test_save_and_read_manifest_text_roundtrip(tmp_path, monkeypatch):
    import json
    mdir = tmp_path / "animations"
    monkeypatch.setattr(api, "manifests_dir", lambda: str(mdir))
    good = json.dumps({"version": 1, "poses": {}, "animations": {}})
    res = api.save_manifest_text("walks.json", good)
    assert res["ok"] is True
    assert (mdir / "walks.json").is_file()
    assert json.loads(api.read_manifest_text("walks.json"))["version"] == 1


def test_save_manifest_rejects_invalid_json_and_bad_manifest(tmp_path, monkeypatch):
    mdir = tmp_path / "animations"
    monkeypatch.setattr(api, "manifests_dir", lambda: str(mdir))
    bad_json = api.save_manifest_text("x.json", "{not json")
    assert bad_json["ok"] is False and bad_json["error"]
    bad_manifest = api.save_manifest_text("x.json", '{"version": "nope"}')
    assert bad_manifest["ok"] is False and "version" in bad_manifest["error"]
    # Neither bad write created the file.
    assert not (mdir / "x.json").exists()


def test_save_manifest_rejects_unsafe_name(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "manifests_dir", lambda: str(tmp_path))
    res = api.save_manifest_text("../escape.json", '{"version":1,"poses":{},"animations":{}}')
    assert res["ok"] is False
    assert not (tmp_path.parent / "escape.json").exists()


def test_save_manifest_surfaces_lint_warnings(tmp_path, monkeypatch):
    import json
    mdir = tmp_path / "animations"
    monkeypatch.setattr(api, "manifests_dir", lambda: str(mdir))
    # A 4n+1-violating length is a non-fatal lint warning.
    m = {
        "version": 1, "poses": {},
        "animations": {"a": {"length": 30, "fps": 8, "width": 16, "height": 16,
                              "start_from": {"ref": "a"},
                              "directions": {"EAST": {}}}},
        "defaults": {},
    }
    # 'a' refs itself -> cycle; use a valid pose dep instead.
    m["poses"] = {"p": {"directions": {"EAST": {}}}}
    m["animations"]["a"]["start_from"] = {"ref": "p"}
    res = api.save_manifest_text("w.json", json.dumps(m))
    assert res["ok"] is True
    assert any("4n+1" in w for w in res["warnings"])


def test_create_and_save_character_layer(tmp_path):
    root = str(tmp_path / "characters")
    created = api.create_character(root, "Cortex Prime")
    assert created["ok"] is True
    assert created["name"] == "cortex_prime"  # snake-cased folder
    assert os.path.isfile(os.path.join(root, "cortex_prime", "character.json"))

    saved = api.save_character_layer(root, "cortex_prime",
                                     positive="a mouthless hero", negative="realistic")
    assert saved["ok"] is True
    layer = api.read_character_layer(root, "cortex_prime")
    assert layer["positive_prompt"] == "a mouthless hero"
    assert layer["negative_prompt"] == "realistic"


def test_save_character_layer_preserves_overlay_and_clears_emptied(tmp_path):
    import json
    root = str(tmp_path / "characters")
    cdir = os.path.join(root, "cortex")
    os.makedirs(cdir)
    # Pre-existing file with a character-authored poses overlay + a positive prompt.
    with open(os.path.join(cdir, "character.json"), "w") as fh:
        json.dump({"positive_prompt": "old", "poses": {"x": {"directions": {}}}}, fh)
    api.save_character_layer(root, "cortex", positive="new hero", negative="")
    layer = api.read_character_layer(root, "cortex")
    assert layer["positive_prompt"] == "new hero"   # updated
    assert "negative_prompt" not in layer            # emptied -> dropped
    assert layer["poses"] == {"x": {"directions": {}}}  # overlay preserved


def test_read_character_layer_absent_returns_empty(tmp_path):
    assert api.read_character_layer(str(tmp_path), "ghost") == {}


# --- next_actionable (batch auto-selectors) --------------------------------- #

def test_next_actionable_pose_skips_root_and_follows_dependency_order(manifest, tree):
    tree.character()  # only character layer; base not generated yet
    # base is a ROOT pose (needs the Character Creator) -> excluded for poses.
    assert api.next_actionable(manifest, tree.root, tree.char, "pose", exclude_root=True) is None
    # Generate base -> fighting_stance (a non-root pose) becomes the next pose job.
    tree.pose("base", "EAST")
    job = api.next_actionable(manifest, tree.root, tree.char, "pose", exclude_root=True)
    assert job["id"] == "fighting_stance" and job["direction"] == "EAST"


def test_next_actionable_animation_picks_first_ready(manifest, tree):
    tree.pose("base", "EAST").pose("fighting_stance", "EAST")
    # fighting_stance_idle (start_from fighting_stance) is now ready.
    job = api.next_actionable(manifest, tree.root, tree.char, "animation")
    assert job["kind"] == "animation" and job["id"] == "fighting_stance_idle"


def test_next_actionable_none_when_nothing_actionable(manifest, tree):
    tree.character()
    assert api.next_actionable(manifest, tree.root, tree.char, "animation") is None


def test_next_actionable_skips_ancestor_only_stale_to_avoid_wedge(manifest, tree):
    # base (a root pose the pose auto-selector excludes) is stale (prompt drift);
    # fighting_stance is freshly rendered but reads `stale` via ancestor recursion.
    # next_actionable must NOT return fighting_stance (re-rendering it can't clear
    # the staleness) — otherwise the batch loop wedges on it forever. With nothing
    # else actionable it returns None, so the auto-selector stops loudly.
    tree.pose("base", "EAST", stale=True)        # root pose stale
    tree.pose("fighting_stance", "EAST")          # fresh, but ancestor-only stale
    from andypack.resolve import status, stale_locally
    assert status(manifest, tree.root, tree.char, "fighting_stance", "EAST") == "stale"
    assert not stale_locally(manifest, tree.root, tree.char, "fighting_stance", "EAST")
    job = api.next_actionable(manifest, tree.root, tree.char, "pose", exclude_root=True)
    assert job is None  # the ancestor-only-stale descendant is skipped, no wedge


def test_next_actionable_returns_locally_stale_cell(manifest, tree):
    # A cell stale for its OWN reason (prompt-hash drift) IS actionable.
    tree.pose("base", "EAST")                      # base fresh/current
    tree.pose("fighting_stance", "EAST", stale=True)  # own hash drifted
    from andypack.resolve import stale_locally
    assert stale_locally(manifest, tree.root, tree.char, "fighting_stance", "EAST")
    job = api.next_actionable(manifest, tree.root, tree.char, "pose", exclude_root=True)
    assert job["id"] == "fighting_stance"


def test_remaining_actionable_counts_all_actionable_poses(manifest, tree):
    # The fixture manifest has exactly one non-root pose (fighting_stance, one
    # direction: EAST); base is root and excluded. Render base first so
    # fighting_stance becomes actionable (mirrors
    # test_next_actionable_pose_skips_root_and_follows_dependency_order).
    tree.character(concept="x")
    tree.pose("base", "EAST")
    first = api.next_actionable(manifest, tree.root, tree.char, "pose", exclude_root=True)
    n = api.remaining_actionable(manifest, tree.root, tree.char, "pose", exclude_root=True)
    assert first is not None
    assert n >= 1
    # Count must equal the number of distinct actionable cells next_actionable would walk.
    seen = 0
    # render each actionable cell in turn; remaining must strictly decrease toward 0
    prev = n
    while api.next_actionable(manifest, tree.root, tree.char, "pose", exclude_root=True):
        job = api.next_actionable(manifest, tree.root, tree.char, "pose", exclude_root=True)
        tree.pose(job["id"], job["direction"])
        seen += 1
        cur = api.remaining_actionable(manifest, tree.root, tree.char, "pose", exclude_root=True)
        assert cur < prev
        prev = cur
    assert prev == 0
    assert seen == n  # the initial count predicted the total drained (flat dependency case)


def test_remaining_actionable_zero_when_nothing_actionable(manifest, tree):
    assert api.remaining_actionable(manifest, tree.root, tree.char, "animation") == 0


def test_read_character_layer_rejects_path_traversal(tmp_path):
    import json
    # A character.json planted OUTSIDE the characters root must not be reachable
    # via a traversal name.
    outside = tmp_path / "secret"
    outside.mkdir()
    (outside / "character.json").write_text(json.dumps({"secret": "leaked"}))
    root = str(tmp_path / "characters")
    os.makedirs(root)
    assert api.read_character_layer(root, "../secret") == {}
    assert api.read_character_layer(root, "..") == {}
    assert api.read_character_layer(root, "/etc") == {}
    assert api.read_character_layer(root, "a/b") == {}


def test_characters_dir_is_none_outside_comfyui():
    assert api.characters_dir() is None


def test_manifest_options_maps_ids_to_directions(manifest):
    opts = api.manifest_options(manifest)
    assert opts["poses"]["base"] == ["EAST", "SOUTH_EAST", "SOUTH"]
    assert opts["poses"]["fighting_stance"] == ["EAST"]
    assert opts["animations"]["punch"] == ["EAST"]
    assert set(opts["animations"]) >= {"fighting_stance_idle", "punch", "fighting_stance_entry"}


def test_manifest_options_includes_mirror_map():
    out = api.manifest_options({"version": 1, "poses": {}, "animations": {}, "mirror_map": {"WEST": "EAST"}})
    assert out["mirror_map"] == {"WEST": "EAST"}


def test_resolve_manifest_path_passthrough_without_comfyui():
    # No ComfyUI base -> relative path falls back to itself (CWD-relative).
    assert api.resolve_manifest_path("default.json") == "default.json"
    assert api.resolve_manifest_path("/abs/animations.json") == "/abs/animations.json"


def test_list_options_marks_root_poses(manifest, tmp_path):
    rows = api.list_options(manifest, str(tmp_path), "cortex")
    by_id = {(r["kind"], r["id"], r["direction"]): r for r in rows}
    assert by_id[("pose", "base", "EAST")]["root"] is True
    assert by_id[("pose", "fighting_stance", "EAST")]["root"] is False
    assert by_id[("animation", "walk", "EAST")]["root"] is False


def test_next_actionable_skip_mirrored(tmp_path):
    # Render base in BOTH EAST and WEST so that pose `p` is ready in both.
    # p's direction order is WEST first, then EAST — confirming WEST is genuinely
    # queued before EAST and would be returned without the skip guard.
    root = str(tmp_path)
    char = "hero"
    manifest = {
        "version": 1,
        "mirror_map": {"WEST": "EAST"},
        "poses": {
            "base": {"directions": {"EAST": {}, "WEST": {}}},
            "p": {"from": {"ref": "base"}, "directions": {"WEST": {}, "EAST": {}}},
        },
        "animations": {},
        "defaults": {},
    }
    from andypack import io, resolve
    base_dir = os.path.join(root, char, "_base")
    os.makedirs(base_dir, exist_ok=True)
    for direction in ("EAST", "WEST"):
        open(os.path.join(base_dir, f"{direction}.png"), "wb").close()
        correct_hash = resolve.compute_prompt_hash(
            manifest, root, char, "pose", "base", direction
        )
        io.atomic_write_json(
            os.path.join(base_dir, f"{direction}.json"),
            io.build_pose_sidecar(
                {
                    "prompt_hash": correct_hash,
                    "direction": direction,
                    "kind": "pose",
                    "pose": "base",
                    "from": None,
                    "manifest_version": manifest["version"],
                },
                created_utc="t",
            ),
        )
    # WEST is first in p's direction order and not blocked — proves it is
    # genuinely actionable; the skip guard is what removes it.
    job_no_skip = api.next_actionable(
        manifest, root, char, "pose", exclude_root=True, skip_mirrored=False
    )
    assert job_no_skip is not None
    assert job_no_skip["direction"] == "WEST"
    # With skip_mirrored=True, WEST (a mirror key) is skipped and EAST is next.
    job_skip = api.next_actionable(
        manifest, root, char, "pose", exclude_root=True, skip_mirrored=True
    )
    assert job_skip is not None
    assert job_skip["direction"] == "EAST"


def test_next_actionable_category_filters(tmp_path):
    # Two ready animations in different categories; the category param must
    # return only the matching one.
    root = str(tmp_path)
    char = "hero"
    manifest = {
        "version": 1,
        "poses": {
            "base": {"directions": {"EAST": {}}},
        },
        "animations": {
            "walk": {
                "category": "locomotion",
                "start_from": {"ref": "base"},
                "directions": {"EAST": {}},
                "length": 33,
                "fps": 16,
                "width": 832,
                "height": 480,
            },
            "punch": {
                "category": "combat",
                "start_from": {"ref": "base"},
                "directions": {"EAST": {}},
                "length": 33,
                "fps": 16,
                "width": 832,
                "height": 480,
            },
        },
        "defaults": {},
    }
    # Render base/EAST so both animations become ready.
    from andypack import io, resolve
    base_dir = os.path.join(root, char, "_base")
    os.makedirs(base_dir, exist_ok=True)
    open(os.path.join(base_dir, "EAST.png"), "wb").close()
    correct_hash = resolve.compute_prompt_hash(
        manifest, root, char, "pose", "base", "EAST"
    )
    io.atomic_write_json(
        os.path.join(base_dir, "EAST.json"),
        io.build_pose_sidecar(
            {
                "prompt_hash": correct_hash,
                "direction": "EAST",
                "kind": "pose",
                "pose": "base",
                "from": None,
                "manifest_version": manifest["version"],
            },
            created_utc="t",
        ),
    )
    combat = api.next_actionable(manifest, root, char, "animation", category="combat")
    assert combat is not None
    assert combat["id"] == "punch"
    locomotion = api.next_actionable(
        manifest, root, char, "animation", category="locomotion"
    )
    assert locomotion is not None
    assert locomotion["id"] == "walk"


# --- thumb_path ----------------------------------------------------------------

def test_thumb_path_rejects_traversal(tmp_path):
    assert api.thumb_path(str(tmp_path), "../x", "pose", "base", "EAST") is None
    assert api.thumb_path(str(tmp_path), "hero", "pose", "..", "EAST") is None


def test_thumb_path_returns_none_for_missing_pose(tmp_path):
    assert api.thumb_path(str(tmp_path), "hero", "pose", "base", "EAST") is None


def test_thumb_path_returns_pose_path_when_exists(tmp_path):
    from andypack import resolve
    root = str(tmp_path)
    character = "hero"
    pose_id = "base"
    direction = "EAST"
    path = resolve.pose_image_path(root, character, pose_id, direction)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w").close()
    result = api.thumb_path(root, character, "pose", pose_id, direction)
    assert result == path


def test_thumb_path_returns_none_for_empty_animation_dir(tmp_path):
    root = str(tmp_path)
    assert api.thumb_path(root, "hero", "animation", "walk", "EAST") is None


def test_thumb_path_returns_first_frame_for_animation(tmp_path):
    from andypack import resolve
    root = str(tmp_path)
    character = "hero"
    anim_id = "walk"
    direction = "EAST"
    frame_dir = resolve.animation_frame_dir(root, character, anim_id, direction)
    os.makedirs(frame_dir, exist_ok=True)
    for name in ("frame_00001.png", "frame_00000.png"):
        open(os.path.join(frame_dir, name), "w").close()
    result = api.thumb_path(root, character, "animation", anim_id, direction)
    assert result == os.path.join(frame_dir, "frame_00000.png")


def test_thumb_path_returns_none_for_missing_reference(tmp_path):
    assert api.thumb_path(str(tmp_path), "hero", "reference", "", "") is None


def test_thumb_path_returns_reference_path_when_exists(tmp_path):
    from andypack import resolve
    root = str(tmp_path)
    character = "hero"
    path = resolve.reference_image_path(root, character)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w").close()
    result = api.thumb_path(root, character, "reference", "", "")
    assert result == path


def test_thumb_path_returns_none_for_unknown_kind(tmp_path):
    assert api.thumb_path(str(tmp_path), "hero", "bogus", "x", "EAST") is None


def test_save_character_layer_preserves_overlay(tmp_path):
    root = str(tmp_path)
    api.save_character_layer(root, "hero", "p", "n",
        overlay={"poses": {"wave": {"from": {"ref": "base"}, "directions": {"EAST": {}}}}})
    layer = api.read_character_layer(root, "hero")
    assert "wave" in layer.get("poses", {})

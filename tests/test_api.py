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


def test_merged_prompt_rows_apply_globals_and_entity(manifest, tree):
    rows = api.merged_prompt_rows(manifest, tree.root, "")
    base = next(r for r in rows if r["kind"] == "pose"
                and r["id"] == "base" and r["direction"] == "EAST")
    assert "neutral standing pose" in base["positive"]      # entity layer
    assert "facing right in profile" not in base["positive"]  # direction inert unless referenced
    assert "blurry" in base["negative"]                    # globals.pose negative

    punch = next(r for r in rows if r["id"] == "punch" and r["direction"] == "EAST")
    assert "extra arm" in punch["negative"]   # entity negative
    assert "watermark" in punch["negative"]   # globals.animation negative


def test_merged_prompt_rows_splice_referenced_identity(manifest, tree):
    # Identity is opt-in: it appears only where {character_prompt} is referenced.
    tree.character(positive_prompt="a brave hero")
    manifest["poses"]["base"]["positive_prompt"] = "{character_prompt} in a neutral pose"
    rows = api.merged_prompt_rows(manifest, tree.root, tree.char)
    base = next(r for r in rows if r["id"] == "base" and r["direction"] == "EAST")
    assert "a brave hero in a neutral pose" in base["positive"]


def test_merged_prompt_rows_exclude_unreferenced_identity(manifest, tree):
    tree.character(positive_prompt="a brave hero")
    rows = api.merged_prompt_rows(manifest, tree.root, tree.char)
    base = next(r for r in rows if r["id"] == "base" and r["direction"] == "EAST")
    assert "a brave hero" not in base["positive"]  # not referenced -> absent


def test_format_merged_prompts_groups_each_cell(manifest):
    rows = [{"kind": "pose", "id": "base", "direction": "EAST",
             "category": "anchor", "positive": "p", "negative": "n"}]
    text = api.format_merged_prompts(rows)
    assert "[pose] base @ EAST" in text
    assert "+ p" in text and "- n" in text


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


def test_diagnostics_degrade_on_invalid_character_overlay(manifest, tree):
    # A character character.json with a structurally bad pose (unknown source ref)
    # makes effective_manifest's re-validation raise; the read paths must degrade
    # to the base manifest rather than abort the queued graph with a traceback.
    tree.character()
    tree.character(poses={"oops": {"from": {"ref": "nope"}, "directions": {"EAST": {}}}})
    ids = {o["id"] for o in api.list_options(manifest, tree.root, tree.char)}
    assert "base" in ids       # base manifest is still reported
    assert "oops" not in ids   # the invalid overlay is dropped, not crashed on
    # The other report builders share the same fallback.
    assert api.coverage_report(manifest, tree.root, tree.char)["total"] > 0
    assert {q["id"] for q in api.regen_queue(manifest, tree.root, tree.char)}  # no raise
    assert api.merged_prompt_rows(manifest, tree.root, tree.char)  # no raise


def test_manifests_dir_and_list_are_empty_without_comfyui():
    assert api.manifests_dir() is None
    assert api.list_manifest_names() == []


def test_split_character_dir():
    assert api.split_character_dir("output/characters/cortex") == ("output/characters", "cortex")
    assert api.split_character_dir("output/characters/cortex/") == ("output/characters", "cortex")
    assert api.split_character_dir("cortex") == ("", "cortex")


def test_character_root_and_name_prefers_explicit_dir():
    assert api.character_root_and_name("output/characters/cortex", "") == (
        "output/characters", "cortex"
    )


def test_character_root_and_name_from_bare_name_uses_characters_root():
    # outside ComfyUI characters_dir() is None -> falls back to 'output/characters'
    assert api.character_root_and_name("", "cortex") == ("output/characters", "cortex")


def test_character_root_and_name_empty_when_nothing_given():
    assert api.character_root_and_name("", "") == ("", "")


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


def test_list_options_marks_root_poses(manifest, tmp_path):
    rows = api.list_options(manifest, str(tmp_path), "cortex")
    by_id = {(r["kind"], r["id"], r["direction"]): r for r in rows}
    assert by_id[("pose", "base", "EAST")]["root"] is True
    assert by_id[("pose", "fighting_stance", "EAST")]["root"] is False
    assert by_id[("animation", "walk", "EAST")]["root"] is False


def test_next_actionable_skip_mirrored(tmp_path):
    root = str(tmp_path)
    char = "hero"
    manifest = {
        "version": 1,
        "mirror_map": {"WEST": "EAST"},
        "poses": {
            "base": {"directions": {"EAST": {}}},
            "p": {"from": {"ref": "base"}, "directions": {"EAST": {}, "WEST": {}}},
        },
        "animations": {},
        "defaults": {},
    }
    base = os.path.join(root, char, "_base")
    os.makedirs(base, exist_ok=True)
    open(os.path.join(base, "EAST.png"), "wb").close()
    from andypack import io, resolve
    correct_hash = resolve.compute_prompt_hash(manifest, root, char, "pose", "base", "EAST")
    io.atomic_write_json(
        os.path.join(base, "EAST.json"),
        io.build_pose_sidecar(
            {"prompt_hash": correct_hash, "direction": "EAST", "kind": "pose",
             "pose": "base", "from": None, "manifest_version": manifest["version"]},
            created_utc="t",
        ),
    )
    job = api.next_actionable(
        manifest, root, char, "pose", exclude_root=True, skip_mirrored=True
    )
    assert job is not None
    assert job["direction"] == "EAST"  # WEST is a mirror target, skipped

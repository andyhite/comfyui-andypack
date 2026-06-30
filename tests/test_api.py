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

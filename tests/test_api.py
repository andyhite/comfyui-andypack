import os

from andypack import api


def test_list_characters_finds_dirs_with_concept(tree):
    tree.concept()  # creates <root>/Cortex/_concept.png
    os.makedirs(os.path.join(tree.root, "NotAChar"), exist_ok=True)  # empty -> excluded
    names = [c["name"] for c in api.list_characters(tree.root)]
    assert names == ["Cortex"]


def test_format_blocked_renders_ref_at_dir():
    blocked = [{"start_from": {"ref": "fighting_stance_idle"}, "dir": "E"},
               {"end_at": {"ref": "fighting_stance_idle"}, "dir": "E"}]
    assert api.format_blocked(blocked) == ["fighting_stance_idle@E", "fighting_stance_idle@E"]


def test_list_options_reports_status_and_blocked(manifest, tree):
    tree.concept()  # only concept present
    opts = {(o["kind"], o["id"], o["direction"]): o for o in api.list_options(manifest, tree.root, tree.char)}

    assert opts[("pose", "base", "E")]["status"] == "ready"
    assert opts[("pose", "fighting_stance", "E")]["status"] == "blocked"
    assert opts[("animation", "punch", "E")]["status"] == "blocked"
    assert opts[("animation", "punch", "E")]["blocked_by"] == [
        "fighting_stance_idle@E", "fighting_stance_idle@E"
    ]
    # base covers three directions
    assert {k for k in opts if k[0] == "pose" and k[1] == "base"} == {
        ("pose", "base", "E"), ("pose", "base", "SE"), ("pose", "base", "S")
    }

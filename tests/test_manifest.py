import json
import warnings
from pathlib import Path

import pytest

from andypack.manifest import (
    ManifestError,
    collect_warnings,
    load_manifest,
    node_kind,
    topo_order,
    validate_manifest,
)

FIX = Path(__file__).parent / "fixtures" / "manifest.json"


def base_manifest():
    return json.loads(FIX.read_text())


def test_load_valid_manifest_returns_dict():
    m = load_manifest(str(FIX))
    assert m["version"] == 1
    assert "fighting_stance_idle" in m["animations"]


def test_node_kind_classifies_each_ref():
    m = base_manifest()
    assert node_kind(m, "concept") == "concept"
    assert node_kind(m, "base") == "pose"
    assert node_kind(m, "punch") == "animation"


def test_node_kind_unknown_ref_raises():
    with pytest.raises(ManifestError):
        node_kind(base_manifest(), "does_not_exist")


def test_validate_rejects_bad_animation_ref():
    m = base_manifest()
    m["animations"]["punch"]["start_from"] = {"ref": "nope"}
    with pytest.raises(ManifestError):
        validate_manifest(m)


def test_validate_rejects_pose_from_animation():
    m = base_manifest()
    m["poses"]["base"]["from"] = {"ref": "punch"}  # a pose may only edit concept/pose
    with pytest.raises(ManifestError):
        validate_manifest(m)


def test_validate_detects_cycle():
    m = base_manifest()
    # base <- fighting_stance and fighting_stance <- base  => cycle
    m["poses"]["base"]["from"] = {"ref": "fighting_stance", "direction": "EAST"}
    with pytest.raises(ManifestError):
        validate_manifest(m)


def test_validate_rejects_animation_without_any_start():
    m = base_manifest()
    # walk has no explicit start_from; removing the default leaves it with no
    # I2V start image at all -> must be rejected.
    del m["defaults"]["start_from"]
    with pytest.raises(ManifestError):
        validate_manifest(m)


def test_validate_rejects_non_dict_direction_value():
    m = base_manifest()
    m["poses"]["base"]["directions"]["EAST"] = "facing right"  # string, not a layer
    with pytest.raises(ManifestError, match="EAST"):
        validate_manifest(m)


def test_validate_rejects_non_int_length():
    m = base_manifest()
    m["animations"]["punch"]["length"] = "long"
    with pytest.raises(ManifestError, match="length"):
        validate_manifest(m)


def test_validate_rejects_non_int_fps_on_defaults():
    m = base_manifest()
    m["defaults"]["fps"] = "fast"
    with pytest.raises(ManifestError, match="fps"):
        validate_manifest(m)


def test_validate_warns_on_non_4n_plus_1_length():
    m = base_manifest()
    m["animations"]["punch"]["length"] = 20  # 20 is not 4n+1
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        validate_manifest(m)
    assert any("4n+1" in str(w.message) or "length" in str(w.message) for w in caught)


def test_collect_warnings_flags_unknown_direction():
    m = base_manifest()
    m["poses"]["base"]["directions"]["UP"] = {}  # not in canonical 'directions'
    findings = collect_warnings(m)
    assert any("UP" in f and "canonical" in f for f in findings)


def test_collect_warnings_clean_manifest_is_empty():
    assert collect_warnings(base_manifest()) == []


def test_topo_order_places_dependencies_first():
    order = topo_order(base_manifest())
    pos = {ref: i for i, ref in enumerate(order)}
    # base -> fighting_stance -> fighting_stance_idle -> punch
    assert pos["base"] < pos["fighting_stance"] < pos["fighting_stance_idle"] < pos["punch"]
    # walk has no explicit start_from -> depends on defaults.start_from (base)
    assert pos["base"] < pos["walk"]

import hashlib
import json
import re
from pathlib import Path

from andypack.resolve import (
    compute_prompt_hash,
    merge_layers,
    merge_negative,
    merged_prompts,
    read_identity,
)

FIX = Path(__file__).parent / "fixtures" / "manifest.json"


def base_manifest():
    return json.loads(FIX.read_text())


def _norm(s):
    return re.sub(r"\s+", " ", s.strip())


def test_merge_layers_joins_non_empty_with_blank_line():
    assert merge_layers("a", None, "b", "") == "a\n\nb"


def test_merge_layers_keeps_each_layer_verbatim():
    # layers are joined as-is (no comma-splitting, no dedupe)
    assert merge_layers("Blurry, foo", "blurry, bar") == "Blurry, foo\n\nblurry, bar"


def test_merge_layers_preserves_prose_commas():
    assert merge_layers("walks forward, steady pace") == "walks forward, steady pace"


def test_merge_layers_strips_each_layer():
    assert merge_layers("  a  ", "\n b \n") == "a\n\nb"


def test_merge_negative_splits_dedupes_and_comma_joins():
    # negatives are comma-separated term lists: split, case-insensitive dedupe
    # (first wins), re-join with ", "
    assert merge_negative("Blurry, foo", None, "blurry, bar", "") == "Blurry, foo, bar"


def test_merge_negative_is_empty_for_no_layers():
    assert merge_negative(None, "", "   ") == ""


def test_read_identity_absent_returns_empty(tmp_path):
    assert read_identity(str(tmp_path), "Cortex") == {}


def test_read_identity_reads_concept_sidecar(tmp_path):
    char_dir = tmp_path / "Cortex"
    char_dir.mkdir()
    (char_dir / "_concept.json").write_text(json.dumps({"positive_prompt": "a mouthless hero"}))
    assert read_identity(str(tmp_path), "Cortex") == {"positive_prompt": "a mouthless hero"}


def _identity(tmp_path, **layer):
    char_dir = tmp_path / "Cortex"
    char_dir.mkdir()
    (char_dir / "_concept.json").write_text(json.dumps(layer))
    return str(tmp_path)


def test_merged_prompts_cascade_excludes_identity_unless_referenced(tmp_path):
    # Identity is opt-in: it is NOT auto-prepended. The cascade is
    # globals[kind] -> entity -> direction, with no identity layer.
    root = _identity(tmp_path, positive_prompt="a mouthless hero", negative_prompt="ugly")
    m = base_manifest()
    pos, neg = merged_prompts(m, root, "Cortex", "pose", "base", "EAST")
    assert pos == "neutral standing pose\n\nfacing right in profile"
    assert "a mouthless hero" not in pos
    assert neg == "blurry, low quality"  # globals.pose.negative only; identity ugly absent
    assert "ugly" not in neg


def test_identity_positive_token_splices_in_place(tmp_path):
    root = _identity(tmp_path, positive_prompt="a mouthless hero")
    m = base_manifest()
    m["poses"]["base"]["directions"]["EAST"]["positive_prompt"] = (
        "a wide shot of {identity_positive} standing"
    )
    pos, _ = merged_prompts(m, root, "Cortex", "pose", "base", "EAST")
    assert pos == "neutral standing pose\n\na wide shot of a mouthless hero standing"
    assert pos.count("a mouthless hero") == 1  # spliced once, not also prepended


def test_identity_negative_token_expands_then_dedupes(tmp_path):
    # {identity_negative} expands BEFORE the term-merge, so its terms dedupe
    # against sibling negative terms ("blurry" shared with globals.animation).
    root = _identity(tmp_path, negative_prompt="blurry, ugly")
    m = base_manifest()
    m["animations"]["punch"]["negative_prompt"] = "{identity_negative}, extra arm"
    _, neg = merged_prompts(m, root, "Cortex", "animation", "punch", "EAST")
    # globals.animation(blurry, low quality, watermark) + punch(blurry, ugly, extra arm)
    assert neg == "blurry, low quality, watermark, ugly, extra arm"


def test_identity_token_with_empty_identity_expands_to_blank(tmp_path):
    root = _identity(tmp_path)  # no identity fields
    m = base_manifest()
    m["poses"]["base"]["directions"]["EAST"]["positive_prompt"] = "shot of {identity_positive}here"
    pos, _ = merged_prompts(m, root, "Cortex", "pose", "base", "EAST")
    assert "{identity_positive}" not in pos
    assert pos == "neutral standing pose\n\nshot of here"


def test_unknown_token_and_literal_braces_survive(tmp_path):
    root = _identity(tmp_path, positive_prompt="hero")
    m = base_manifest()
    m["poses"]["base"]["directions"]["EAST"]["positive_prompt"] = "shot of {unknown} {thing}"
    pos, _ = merged_prompts(m, root, "Cortex", "pose", "base", "EAST")
    assert "{unknown}" in pos and "{thing}" in pos


def test_compute_prompt_hash_matches_formula(tmp_path):
    m = base_manifest()
    pos, neg = merged_prompts(m, str(tmp_path), "Cortex", "animation", "punch", "EAST")
    raw = _norm(pos) + "␟" + _norm(neg)
    expected = "sha1:" + hashlib.sha1(raw.encode("utf-8")).hexdigest()
    assert compute_prompt_hash(m, str(tmp_path), "Cortex", "animation", "punch", "EAST") == expected

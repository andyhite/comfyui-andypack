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


def test_positive_merges_globals_and_entity_not_direction(tmp_path):
    # The compiled positive merges globals[kind] + entity only. The direction
    # layer is inert unless referenced via {direction_prompt}.
    root = _identity(tmp_path, positive_prompt="a mouthless hero", negative_prompt="ugly")
    m = base_manifest()
    pos, neg = merged_prompts(m, root, "Cortex", "pose", "base", "EAST")
    assert pos == "neutral standing pose"          # entity only; globals.pose has no positive
    assert "facing right in profile" not in pos    # direction NOT auto-appended
    assert "a mouthless hero" not in pos           # identity inert unless referenced
    assert neg == "blurry, low quality"            # globals.pose.negative; identity inert
    assert "ugly" not in neg


def test_direction_prompt_and_name_inject_into_positive(tmp_path):
    root = _identity(tmp_path)
    m = base_manifest()
    m["poses"]["base"]["positive_prompt"] = (
        "neutral standing pose. As viewed from the {direction_name}: {direction_prompt}"
    )
    pos, _ = merged_prompts(m, root, "Cortex", "pose", "base", "EAST")
    assert pos == "neutral standing pose. As viewed from the EAST: facing right in profile"


def test_identity_prompt_resolves_by_field_context(tmp_path):
    root = _identity(tmp_path, positive_prompt="a mouthless hero", negative_prompt="ugly")
    m = base_manifest()
    m["poses"]["base"]["positive_prompt"] = "a wide shot of {identity_prompt} standing"
    m["globals"]["pose"]["negative_prompt"] = "{identity_prompt}, low quality"
    pos, neg = merged_prompts(m, root, "Cortex", "pose", "base", "EAST")
    assert pos == "a wide shot of a mouthless hero standing"  # positive -> concept positive
    assert "ugly" in neg and "a mouthless hero" not in neg    # negative -> concept negative
    assert neg == "ugly, low quality"


def test_identity_negative_expands_then_dedupes(tmp_path):
    # In a negative field {identity_prompt} expands to the concept negative,
    # then the term-merge dedupes it against siblings ("blurry" shared).
    root = _identity(tmp_path, negative_prompt="blurry, ugly")
    m = base_manifest()
    m["animations"]["punch"]["negative_prompt"] = "{identity_prompt}, extra arm"
    _, neg = merged_prompts(m, root, "Cortex", "animation", "punch", "EAST")
    # globals.animation(blurry, low quality, watermark) + punch(blurry, ugly, extra arm)
    assert neg == "blurry, low quality, watermark, ugly, extra arm"


def test_empty_direction_negative_leaves_no_stray_comma(tmp_path):
    # punch@EAST has no direction negative, so {direction_prompt} -> "" and the
    # empty term is dropped (no stray ", ,").
    root = _identity(tmp_path)
    m = base_manifest()
    m["animations"]["punch"]["negative_prompt"] = "{direction_prompt}, extra arm"
    _, neg = merged_prompts(m, root, "Cortex", "animation", "punch", "EAST")
    assert ", ," not in neg
    assert neg == "blurry, low quality, watermark, extra arm"


def test_variables_resolve_inside_globals(tmp_path):
    # Substitution runs on the merged text, so a global may reference variables.
    root = _identity(tmp_path, negative_prompt="signature-flaw")
    m = base_manifest()
    m["globals"]["pose"]["negative_prompt"] = "{identity_prompt}, {direction_name}-artifact"
    _, neg = merged_prompts(m, root, "Cortex", "pose", "base", "SOUTH")
    assert neg == "signature-flaw, SOUTH-artifact"


def test_unknown_tokens_and_empty_sources_survive(tmp_path):
    root = _identity(tmp_path)  # no identity fields
    m = base_manifest()
    m["poses"]["base"]["positive_prompt"] = "{identity_prompt}shot of {unknown} {thing}"
    pos, _ = merged_prompts(m, root, "Cortex", "pose", "base", "EAST")
    assert pos == "shot of {unknown} {thing}"  # identity empty; unknown tokens untouched


def test_compute_prompt_hash_matches_formula(tmp_path):
    m = base_manifest()
    pos, neg = merged_prompts(m, str(tmp_path), "Cortex", "animation", "punch", "EAST")
    raw = _norm(pos) + "␟" + _norm(neg)
    expected = "sha1:" + hashlib.sha1(raw.encode("utf-8")).hexdigest()
    assert compute_prompt_hash(m, str(tmp_path), "Cortex", "animation", "punch", "EAST") == expected

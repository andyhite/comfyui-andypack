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
    (char_dir / "_concept.json").write_text(json.dumps({"prompt": "a mouthless hero"}))
    assert read_identity(str(tmp_path), "Cortex") == {"prompt": "a mouthless hero"}


def test_merged_prompts_cascades_identity_global_entity_direction(tmp_path):
    char_dir = tmp_path / "Cortex"
    char_dir.mkdir()
    (char_dir / "_concept.json").write_text(json.dumps({"prompt": "a mouthless hero"}))
    m = base_manifest()
    pos, neg = merged_prompts(m, str(tmp_path), "Cortex", "pose", "base", "EAST")
    # identity -> (no globals.pose positive) -> pose.prompt -> base.directions.EAST.prompt
    assert pos == "a mouthless hero\n\nneutral standing pose\n\nfacing right in profile"
    assert neg == "blurry, low quality"  # globals.pose.negative only (single layer)


def test_merged_prompts_negative_dedupes_across_layers(tmp_path):
    char_dir = tmp_path / "Cortex"
    char_dir.mkdir()
    # identity negative shares a term with globals.animation negative ("blurry")
    (char_dir / "_concept.json").write_text(json.dumps({"negative": "blurry, ugly"}))
    m = base_manifest()
    pos, neg = merged_prompts(m, str(tmp_path), "Cortex", "animation", "punch", "EAST")
    # identity(blurry, ugly) + globals.animation(blurry, low quality, watermark)
    #   + punch(both arms extended, extra arm), comma-deduped, first wins
    assert neg == (
        "blurry, ugly, low quality, watermark, both arms extended, extra arm"
    )


def test_compute_prompt_hash_matches_formula(tmp_path):
    m = base_manifest()
    pos, neg = merged_prompts(m, str(tmp_path), "Cortex", "animation", "punch", "EAST")
    raw = _norm(pos) + "␟" + _norm(neg)
    expected = "sha1:" + hashlib.sha1(raw.encode("utf-8")).hexdigest()
    assert compute_prompt_hash(m, str(tmp_path), "Cortex", "animation", "punch", "EAST") == expected

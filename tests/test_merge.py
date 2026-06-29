import hashlib
import json
import re
from pathlib import Path

from andypack.resolve import compute_prompt_hash, merge_layers, merged_prompts, read_identity

FIX = Path(__file__).parent / "fixtures" / "manifest.json"


def base_manifest():
    return json.loads(FIX.read_text())


def _norm(s):
    return re.sub(r"\s+", " ", s.strip())


def test_merge_layers_joins_non_empty_in_order():
    assert merge_layers("a", None, "b", "") == "a, b"


def test_merge_layers_dedupes_case_insensitively_preserving_first():
    assert merge_layers("Blurry, foo", "blurry, bar") == "Blurry, foo, bar"


def test_merge_layers_is_lossless_for_prose_commas():
    assert merge_layers("walks forward, steady pace") == "walks forward, steady pace"


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
    pos, neg = merged_prompts(m, str(tmp_path), "Cortex", "pose", "base", "E")
    # identity -> (no globals.pose positive) -> pose.prompt -> base.directions.E.prompt
    assert pos == "a mouthless hero, neutral standing pose, facing right in profile"
    assert neg == "blurry, low quality"  # globals.pose.negative only


def test_compute_prompt_hash_matches_formula(tmp_path):
    m = base_manifest()
    pos, neg = merged_prompts(m, str(tmp_path), "Cortex", "animation", "punch", "E")
    raw = _norm(pos) + "␟" + _norm(neg)
    expected = "sha1:" + hashlib.sha1(raw.encode("utf-8")).hexdigest()
    assert compute_prompt_hash(m, str(tmp_path), "Cortex", "animation", "punch", "E") == expected

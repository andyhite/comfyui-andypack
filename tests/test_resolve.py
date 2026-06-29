"""Unit tests for andypack.resolve — the pure FFLF dependency resolver.

Gated on spec §9 step-2 acceptance:
  - punch@E is `blocked` with an empty tree;
  - becomes `ready` after a complete fighting_stance_idle/E;
  - start_image -> that dir's last_frame, end_image -> its start_frame.
"""

import hashlib
import json
import os
import re
import shutil
from pathlib import Path

from andypack.resolve import (
    compose_negative,
    compute_prompt_hash,
    resolve,
    resolved_dir,
    status,
)

FIX = Path(__file__).parent / "fixtures"
MANIFEST = json.loads(
    (Path(__file__).parents[1] / "examples" / "animations.json").read_text()
)

EMPTY_ROOT = str(FIX / "empty_root")      # base pngs only, nothing rendered
IDLE_ROOT = str(FIX / "idle_root")        # + complete fighting_stance_idle/E
PARTIAL_ROOT = str(FIX / "partial_root")  # frames+meta but no .complete
BARE_ROOT = str(FIX / "bare_root")        # character dir, no _base


def _norm(s):
    return re.sub(r"\s+", " ", s.strip())


# --- resolved_dir -----------------------------------------------------------

def test_resolved_dir_same_uses_selected():
    assert resolved_dir({"ref": "x", "direction": "same"}, "E") == "E"
    assert resolved_dir({"ref": "x"}, "SE") == "SE"


def test_resolved_dir_explicit_overrides():
    assert resolved_dir({"ref": "x", "direction": "E"}, "SE") == "E"


# --- compose_negative (spec §4) --------------------------------------------

def test_compose_negative_non_frontal_uses_default_facial():
    idle = MANIFEST["animations"]["fighting_stance_idle"]
    neg = compose_negative(MANIFEST, idle, "E")
    # E is not a frontal direction -> full facial block (face, facial features)
    assert "face, facial features" in neg
    assert neg.startswith(MANIFEST["negatives"]["global"])


def test_compose_negative_frontal_drops_broad_face_terms():
    idle = MANIFEST["animations"]["fighting_stance_idle"]
    neg = compose_negative(MANIFEST, idle, "S")
    # S is frontal -> drop "face, facial features", keep specific feature terms
    assert "face, facial features" not in neg
    assert "mouth" in neg and "teeth" in neg


def test_compose_negative_appends_animation_negative():
    punch = MANIFEST["animations"]["punch"]
    neg = compose_negative(MANIFEST, punch, "E")
    assert neg.endswith(punch["negative"])


def test_compose_negative_dedupes_case_insensitively_preserving_first():
    fake_mani = {
        "negatives": {
            "global": "blurry, Low Quality",
            "facial": {"frontal_directions": ["S"], "frontal": "mouth", "default": "blurry, face"},
        }
    }
    anim = {"prompt": "p", "negative": "low quality, watermark"}
    neg = compose_negative(fake_mani, anim, "E")
    # "blurry" (global) deduped against facial.default; "low quality" deduped vs "Low Quality"
    assert neg == "blurry, Low Quality, face, watermark"


# --- compute_prompt_hash (spec §2) -----------------------------------------

def test_compute_prompt_hash_matches_documented_formula():
    idle = MANIFEST["animations"]["fighting_stance_idle"]
    neg = compose_negative(MANIFEST, idle, "E")
    raw = _norm(idle["prompt"]) + "␟" + _norm(neg)
    expected = "sha1:" + hashlib.sha1(raw.encode("utf-8")).hexdigest()
    assert compute_prompt_hash(MANIFEST, idle, "E") == expected


# --- spec §9 step-2 acceptance: punch@E ------------------------------------

def test_punch_blocked_on_empty_tree():
    r = resolve(MANIFEST, EMPTY_ROOT, "Cortex", "punch", "E")
    assert r["selectable"] is False
    assert status(MANIFEST, EMPTY_ROOT, "Cortex", "punch", "E") == "blocked"
    # both FFLF slots point at the unrendered fighting_stance_idle
    slots = {k for entry in r["blocked_by"] for k in entry if k != "dir"}
    assert slots == {"start_from", "end_at"}


def test_punch_ready_after_idle_complete():
    r = resolve(MANIFEST, IDLE_ROOT, "Cortex", "punch", "E")
    assert r["selectable"] is True
    assert r["blocked_by"] == []
    assert status(MANIFEST, IDLE_ROOT, "Cortex", "punch", "E") == "ready"


def test_punch_anchors_cross_wire_fflf():
    # start_from consumes the dep's LAST frame; end_at consumes its FIRST frame.
    r = resolve(MANIFEST, IDLE_ROOT, "Cortex", "punch", "E")
    dep_dir = os.path.join("fighting_stance_idle", "E")
    assert r["start_image"].endswith(os.path.join(dep_dir, "frame_00002.png"))
    assert r["end_image"].endswith(os.path.join(dep_dir, "frame_00000.png"))
    assert os.path.exists(r["start_image"])
    assert os.path.exists(r["end_image"])


# --- atomicity: a partial dir never reads as satisfied ----------------------

def test_partial_dir_without_complete_sentinel_blocks():
    r = resolve(MANIFEST, PARTIAL_ROOT, "Cortex", "punch", "E")
    assert r["selectable"] is False
    assert status(MANIFEST, PARTIAL_ROOT, "Cortex", "punch", "E") == "blocked"


# --- base_pose gating (spec §8: E, SE, S coverage) -------------------------

def test_idle_ready_when_base_pose_present():
    r = resolve(MANIFEST, EMPTY_ROOT, "Cortex", "fighting_stance_idle", "E")
    assert r["selectable"] is True
    assert r["start_image"].endswith(os.path.join("_base", "E.png"))


def test_idle_blocked_when_base_pose_missing():
    r = resolve(MANIFEST, BARE_ROOT, "Cortex", "fighting_stance_idle", "E")
    assert r["selectable"] is False
    slots = {k for entry in r["blocked_by"] for k in entry if k != "dir"}
    assert slots == {"start_from"}


def test_direction_outside_animation_directions_not_selectable():
    # fighting_stance_idle only offers E; W is never a render target.
    r = resolve(MANIFEST, EMPTY_ROOT, "Cortex", "fighting_stance_idle", "W")
    assert r["selectable"] is False


# --- staleness: warn (amber) but stay selectable (spec §1, §8) -------------

def test_fresh_dep_not_stale():
    r = resolve(MANIFEST, IDLE_ROOT, "Cortex", "punch", "E")
    assert r["stale"] == []


def test_stale_dep_warns_but_stays_selectable(tmp_path):
    root = tmp_path / "stale_root"
    shutil.copytree(IDLE_ROOT, root)
    meta_path = root / "Cortex" / "fighting_stance_idle" / "E" / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta["prompt_hash"] = "sha1:0000000000000000000000000000000000000000"
    meta_path.write_text(json.dumps(meta))

    r = resolve(MANIFEST, str(root), "Cortex", "punch", "E")
    assert r["selectable"] is True            # stale does NOT block
    assert sorted(r["stale"]) == ["end_at", "start_from"]
    assert status(MANIFEST, str(root), "Cortex", "punch", "E") == "stale"

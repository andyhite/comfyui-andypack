#!/usr/bin/env python3
"""Generate examples/animations.json — the bundled, character-AGNOSTIC seed
manifest — from compact Python definitions.

Why a generator: every pose and animation must list all 8 canonical directions,
and the per-direction camera language must live ONCE (in `view_phrases`) so it
stays consistent and the manifest carries no character-specific anatomy. Authoring
that by hand is error-prone; this script makes the 8-direction expansion and the
shared prompt suffixes mechanical.

Run:  python scripts/build_seed_manifest.py
Output is committed as examples/animations.json (the repo's seed source). The
resolver/manifest code is the source of truth for schema; this is convenience.

Prompt structure follows docs/prompting-guide.md (FLUX.2 Klein for poses: prose,
affirmative, no negatives, manikin = pose-only; Wan 2.2 i2v for animations:
motion-forward, fixed camera, static background, standard negative block).
"""

from __future__ import annotations

import json
import os

DIRECTIONS = [
    "EAST", "SOUTH_EAST", "SOUTH", "SOUTH_WEST",
    "WEST", "NORTH_WEST", "NORTH", "NORTH_EAST",
]

# Affirmative, character-AGNOSTIC per-direction camera language. Orientation only
# (no posture words — anchor poses override the posture). Carries the turnaround
# failure-mode mitigations: "exactly one eye" on profiles, "no face visible" on
# back views (FLUX.2 has no negative prompt, so these must be affirmative).
VIEW_PHRASES = {
    "EAST": "in full right-side profile, facing the right edge, only one eye visible",
    "WEST": "in full left-side profile, facing the left edge, only one eye visible",
    "SOUTH": "in a dead-on front view, facing the camera, both eyes visible",
    "NORTH": "in a dead-on back view, facing directly away, back of the head to "
             "the camera, no face visible",
    "SOUTH_EAST": "from a front-right three-quarter angle, facing the lower-right",
    "SOUTH_WEST": "from a front-left three-quarter angle, facing the lower-left",
    "NORTH_EAST": "from a back-right three-quarter angle, mostly facing away, "
                  "no face visible",
    "NORTH_WEST": "from a back-left three-quarter angle, mostly facing away, "
                  "no face visible",
}

# Base pose = a FLUX.2 multi-reference edit: image 1 = the character reference
# (identity), image 2 = the gray manikin (pose/orientation only). Affirmative,
# names no character anatomy (that arrives via {character_prompt}); camera comes
# from {view_phrase}.
BASE_POSITIVE = (
    "Edit the first image (the character reference) so the same character is shown "
    "{view_phrase}. Match the body pose and orientation of the gray mannequin in "
    "the second image, but keep the character's own colors and design — never the "
    "mannequin's gray. Preserve the character's identity exactly: {character_prompt}. "
    "Relaxed neutral standing pose, full body, plain background, flat even lighting."
)

# Anchor pose = a single-reference FLUX.2 edit that re-poses the character while
# holding the SAME facing/camera as its source. {pose} is the entity description.
ANCHOR_TEMPLATE = (
    "Edit the reference image to re-pose the character while keeping the exact same "
    "facing and camera angle — still {view_phrase}. {pose} Keep the character's "
    "identity, colors, and design exactly as in the reference: {character_prompt}. "
    "Full body, plain background, flat even lighting."
)

# (id, pose-description sentence) — character-agnostic, facing-relative (never
# "to the right"). Each re-poses from `base`, same direction.
ANCHOR_POSES = [
    ("walk_stride",
     "The character is frozen at a clear contact moment of a forward walking "
     "stride: the lead leg forward and planted heel-down, the trailing leg extended "
     "behind and rolling off the toe, the torso upright and balanced over the lead "
     "leg, the arms swung into natural front-to-back opposition."),
    ("run_stride",
     "The character is frozen mid-run at a full sprint, the torso pitched forward "
     "into its momentum: one knee driven high and forward, the other leg extended "
     "fully behind in an explosive push-off, both feet off the ground, the arms "
     "bent sharp and thrown into strong opposition."),
    ("crouch",
     "The character is held in a low braced crouch: the knees bent deeply and the "
     "hips sunk low, the torso leaned in over the legs and compacted toward the "
     "ground, the arms drawn loosely in toward the body, the head lowered."),
    ("jump_tuck",
     "The character is held airborne at the peak of a jump, suspended off the "
     "ground with the legs tucked up beneath the body and the knees gathered toward "
     "the torso, the body compact and lifted, the arms raised slightly with the "
     "upward momentum."),
    ("fighting_stance",
     "The character holds a ready fighting stance: weight low and balanced on "
     "staggered feet with the knees bent, the torso poised, both arms raised in a "
     "fists-up guard in front of the chest and head, the lead fist forward and "
     "lower and the rear fist tucked in close beside the head."),
    ("wall_cling_pose",
     "The character clings flat against a vertical wall, the limbs braced and "
     "gripping the surface, the body kept close and pressed flat to the wall, held "
     "off the ground."),
    ("climb_reach",
     "The character is held mid-climb on a vertical surface, caught at the top of a "
     "pull: one arm reached high overhead and gripping, the opposite arm pulled in "
     "low, one knee lifted to push from below while the other leg extends down, the "
     "body drawn close and vertical against the surface."),
]

# Animations are Wan 2.2 i2v: the start frame already fixes appearance + facing,
# so prompts are MOTION-forward and facing-relative. This suffix is appended to
# every animation positive (fixed camera + static background for clean sprites).
SPRITE_SUFFIX = (
    " Fixed locked-off camera, no camera movement, no panning or zooming. Plain "
    "solid static background, even flat lighting, crisp readable silhouette, clean "
    "game-sprite animation."
)

# Standard Wan 2.2 negative block (English equivalent of the official Chinese
# block). Effective only at CFG > 1 (the full multi-step path, not the 4-step
# distill). {character_prompt} folds in the character's own negative terms.
WAN_NEGATIVE = (
    "{character_prompt}, overexposed, static, blurred details, subtitles, overall "
    "gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, "
    "extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, "
    "misshapen limbs, fused fingers, still picture, messy background, three legs, "
    "many people in the background, walking backwards, extra limbs, mutated hands, "
    "watermark, text"
)

# (id, category, length, start_from, end_at, motion-prose, extra_negative)
# end_at None => start-only i2v. start_from None => defaults.start_from (base).
# Motion prose is facing-relative ("forward", "in the direction it faces").
A = None
ANIMATIONS = [
    ("walk", "locomotion", 33, "walk_stride", "walk_stride",
     "The character walks forward at a steady, even pace in a clear alternating "
     "gait: each leg lifts at the hip, the knee bends to clear the ground, the foot "
     "swings through and plants heel-first then rolls onto the toe as the weight "
     "transfers and the opposite leg repeats. The hips rock subtly with each weight "
     "shift, the torso stays upright, the arms swing in natural opposition, and the "
     "head bobs gently once per step. One full two-step stride cycle that returns "
     "to the exact starting pose for a seamless loop.", A),
    ("run", "locomotion", 25, "run_stride", "run_stride",
     "The character runs forward at full speed, the torso pitched slightly forward. "
     "The legs drive in a fast springy alternating stride, each knee lifting high "
     "and pushing off hard with a brief fully airborne moment between footfalls. "
     "The arms pump in strong opposition, elbows bent sharp, and the body rises and "
     "falls with each push-off. One full stride cycle that returns to the starting "
     "pose for a seamless loop.", A),
    ("skid_stop", "locomotion", 21, "run_stride", "base",
     "The character brakes hard out of a run: it plants both feet forward and leans "
     "its weight sharply back against the momentum, the legs stiffening and the feet "
     "skidding a short distance, the arms swinging out to catch its balance, then "
     "settles upright into a neutral standing stance as the slide ends.", A),
    ("dash", "locomotion", 17, "base", "base",
     "The character bursts forward in a sudden low ground dash: it drops low and "
     "coils onto its back leg, then drives off explosively to shoot ahead at speed, "
     "the body kept low and streamlined and the arms swept back, then eases up and "
     "settles back into a neutral standing stance.", A),
    ("slide", "locomotion", 21, "run_stride", "base",
     "The character drops into a low feet-first ground slide: from a run it ducks "
     "down sharply, kicks one leg forward and drops its hips low to skim along the "
     "ground with the trailing leg folded under and the torso leaned back, then "
     "plants its feet and rises into a neutral standing stance.", A),
    ("jump_launch", "aerial", 17, "base", "jump_tuck",
     "The character launches a jump: it first crouches low, the knees bending deeply "
     "to load the leap, then explodes upward as both legs extend and push off the "
     "ground hard, the body stretching tall, and as it leaves the ground it tucks "
     "its legs up beneath it with the arms lifting in the upward rush.", A),
    ("jump_apex", "aerial", 25, "jump_tuck", "jump_tuck",
     "The character hovers at the very top of its jump arc, floating nearly "
     "weightless as the upward and downward momentum cancel out, the legs tucked up "
     "beneath it and the arms held slightly raised, with only a faint floating "
     "drift of motion, holding the airborne peak for a seamless loop.", A),
    ("fall", "aerial", 17, "jump_tuck", A,
     "The character falls downward through the air, gaining speed as it drops: from "
     "the tucked airborne pose the legs extend and reach down toward the ground, the "
     "arms drift upward against the rising air, and the torso braces for the coming "
     "impact in a continuous downward plummet.", A),
    ("land", "aerial", 17, "fall", "base",
     "The character lands and absorbs the impact of a fall: the feet hit the ground "
     "and the knees bend deeply to cushion it, the torso compressing down and the "
     "head dipping, then the legs straighten, the body springs back up, and it "
     "settles into a neutral standing stance.", A),
    ("double_jump", "aerial", 17, "jump_tuck", "jump_tuck",
     "The character snaps into a second jump in mid-air: from the airborne tuck the "
     "body gives a sharp renewed upward burst as the legs kick down and then re-tuck "
     "beneath it, the arms swinging up with the fresh lift, returning to the tucked "
     "airborne pose.", A),
    ("crouch_entry", "stance", 17, "base", "crouch",
     "The character ducks from standing down into a low crouch: the knees bend "
     "deeply and the hips sink, the whole body folding and compacting downward, the "
     "torso leaning in over the legs and the head lowering, settling into a low "
     "braced hunkered pose.", A),
    ("crouch_exit", "stance", 17, "crouch", "base",
     "The character rises out of a low crouch back up to standing: the knees "
     "straighten and the hips lift, the torso extending upward and the head rising, "
     "until the body settles into a neutral upright standing stance.", A),
    ("standing_idle", "stance", 49, "base", "base",
     "The character stands in place in a relaxed settled stance: the chest rises and "
     "falls with slow steady breathing and the body sways very gently, the arms "
     "hanging loosely and almost still, the head drifting in tiny movements. Only "
     "quiet ambient idle motion, returning to the starting pose for a seamless loop.", A),
    ("crouch_idle", "stance", 49, "crouch", "crouch",
     "The character holds a low crouch in place, staying down with the knees bent "
     "and the hips low: the body breathes with a slow rise and fall and sways very "
     "gently, the arms hanging loosely and the head lowered. Calm and holding low, "
     "returning to the starting pose for a seamless loop.", A),
    ("fighting_stance_entry", "combat", 25, "base", "fighting_stance_idle",
     "The character shifts from a relaxed standing pose into a ready fighting "
     "stance: it staggers its feet, bends its knees to drop its weight low, and "
     "raises both arms into a fists-up guard in front of the chest and head, the "
     "lead fist forward and lower, then locks into the braced poised guard.", A),
    ("fighting_stance_idle", "combat", 33, "fighting_stance", "fighting_stance",
     "The character holds a ready fighting stance, weight low on staggered bent "
     "knees with both fists up in a guard in front of the chest and head: it bobs "
     "and sways very gently in place with light springy ready energy, shifting its "
     "weight subtly while the guard stays intact. Returning to the starting pose "
     "for a seamless loop.", A),
    ("fighting_stance_exit", "combat", 25, "fighting_stance_idle", "base",
     "The character drops out of its fighting stance back into a relaxed standing "
     "pose: it lowers its fists, unstaggers its feet, and settles upright into a "
     "level neutral rest.", A),
    ("punch", "combat", 21, "fighting_stance_idle", "fighting_stance_idle",
     "The character throws a single straight punch forward: it winds up by rolling "
     "its lead shoulder back and loading its weight onto the rear leg, then "
     "explosively drives its lead arm straight forward into a sharp jab at full "
     "extension while the opposite arm stays tucked tight against the chest as a "
     "fixed guard, then retracts and resettles into a neutral guard. The torso and "
     "hips pivot into the punch; a single committed strike with a crisp recovery.",
     "both arms extended, second arm raised, flailing arm, floating arm, waving "
     "arms, morphing arm, extra arm, two-armed punch"),
    ("kick", "combat", 21, "fighting_stance_idle", "fighting_stance_idle",
     "The character throws a single quick forward kick: it shifts its weight back "
     "onto its rear leg, then snaps its front leg up and out in a sharp straight "
     "kick to full extension before retracting and planting back down as the body "
     "resettles into a neutral guard, the arms swinging slightly to counterbalance.",
     A),
    ("headbutt", "combat", 25, "fighting_stance_idle", "fighting_stance_idle",
     "The character lunges into a headbutt: it rears its head back and drops its "
     "stance to load up, then thrusts the whole head forward and slightly down in a "
     "powerful slam with the body and hips driving behind it, then snaps back "
     "upright and resettles into a neutral guard.", A),
    ("block", "combat", 33, "fighting_stance", "fighting_stance",
     "From the ready guard the character braces into a defensive block: it plants "
     "its feet, tenses low and compact, and raises both arms crossed in front of "
     "the head and torso to shield itself, holds the braced stance with faint "
     "tension and breathing, then lowers back to the ready guard. Returning to the "
     "starting pose for a seamless loop.", A),
    ("stomp", "combat", 17, "jump_tuck", "jump_tuck",
     "From an airborne tuck the character drops straight down in a hard ground "
     "stomp: it drives both feet down together to slam onto the surface below, the "
     "body compressing on impact, then rebounds sharply upward and re-tucks its "
     "legs, returning to the airborne pose.", A),
    ("wall_cling", "surface", 33, "wall_cling_pose", "wall_cling_pose",
     "The character clings to a vertical wall, gripping the surface and holding in "
     "place with the limbs braced and the body pressed close and flat, with almost "
     "no motion — only a faint shift of weight and slight tension as it maintains "
     "the grip. Holding steady for a seamless loop.", A),
    # Start-only on purpose: a wall slide translates downward and does NOT return
    # to its start, so it must not be a same-image FFLF loop (the prompt avoids
    # claiming one).
    ("wall_slide", "surface", 25, "wall_cling_pose", A,
     "The character presses flat against a vertical wall and slides slowly "
     "downward, the limbs braced against the surface to drag and slow the descent, "
     "the body kept close and flat, sliding down at a steady, controlled rate.", A),
    ("wall_jump", "surface", 17, "wall_cling_pose", "jump_tuck",
     "The character pushes off a vertical wall in a single explosive burst: both "
     "legs coil and then kick hard against the wall to launch the body up and away "
     "from it, the arms swinging up with the leap as the body springs out into the "
     "air and tucks its legs beneath it.", A),
    ("climb", "surface", 33, "climb_reach", "climb_reach",
     "The character climbs steadily upward: it reaches up with one arm to grip, "
     "pulls itself up as the legs push from below, then reaches with the opposite "
     "arm and repeats, hauling the body upward in a clear even alternating climb "
     "cycle that returns to the starting pose for a seamless loop.", A),
    ("hurt", "reactions", 17, "base", "base",
     "The character flinches and recoils from a hit: the upper body jerks sharply "
     "backward and the head snaps back with the impact, the arms flailing out "
     "briefly, then the body rocks forward and recovers its balance back toward a "
     "neutral stance. A quick stagger-and-recover.", A),
    ("defeat", "reactions", 33, "base", A,
     "The character is defeated: it staggers, the knees buckling and the legs "
     "giving way, the torso sagging and the head slumping forward, then the whole "
     "body folds downward and collapses limp to the ground in a final stagger.", A),
    ("surprise", "reactions", 21, "base", "base",
     "The character reacts to a sudden jolt of surprise: the whole body stiffens "
     "and pops upward with a small hop, straightening tall, the arms flicking "
     "outward and the head rising sharply, then everything settles back down into a "
     "neutral standing stance.", A),
    ("disappointment", "reactions", 25, "base", "base",
     "The character deflates with disappointment: the body sags and the shoulders "
     "drop as the head droops slowly downward in a tired letdown, then it settles "
     "into a slumped neutral stance. A soft gentle slump.", A),
    ("edge_teeter", "reactions", 41, "base", "base",
     "The character stands at the very edge of a ledge and wobbles to keep its "
     "balance: the body tips forward over the drop and then rocks back, the arms "
     "windmilling to stay upright, never quite falling, the head swaying with the "
     "off-balance motion in an unsteady teeter that loops back and forth.", A),
    ("idle_break", "expression", 33, "base", "base",
     "The character does a small idle fidget: it shifts its weight from one foot to "
     "the other and gives the body a brief loose settle, the arms swaying slightly "
     "and the head craning forward a little then easing back, before returning to a "
     "relaxed neutral standing stance.", A),
    ("talk", "expression", 41, "base", "base",
     "The character speaks animatedly in conversation: the head bobs and tilts "
     "gently with the rhythm of speech, the arms make small expressive gestures, "
     "and the body shifts its weight lightly from side to side in lively "
     "conversational motion, returning to the starting pose for a seamless loop.", A),
    ("wonder", "expression", 25, "base", A,
     "The character is struck with overwhelming wonder: the body rears back and the "
     "head tips upward in astonishment as both arms fling wide and high, then it "
     "holds the wide-open marvelling pose at full stretch.", A),
    ("cheer", "expression", 33, "base", "base",
     "The character celebrates with excitement, bouncing up and down in place as "
     "both arms pump and wave triumphantly overhead and the head bobs with the "
     "joyful energy, then settling back down. Returning to the starting pose for a "
     "seamless loop.", A),
    # Start-only on purpose: a 180° turn ends facing the OPPOSITE way, so it must
    # not be pinned back to the start (base@dir) — that would be a contradictory
    # same-image loop. The single base@dir we have can only anchor the start frame.
    ("turn_around", "expression", 21, "base", A,
     "The character pivots in place to face the opposite direction: the body "
     "rotates around as the legs step and reposition beneath it and the head swings "
     "through the turn, settling to face the other way in a clean controlled "
     "in-place turnaround.", A),
]


def all_dirs() -> dict:
    """`directions` map with all 8 canonical directions and empty layers — the
    per-direction camera language comes from view_phrases, the pose/motion from the
    entity prompt, so an empty layer is the norm."""
    return {d: {} for d in DIRECTIONS}


def build() -> dict:
    poses: dict = {
        "base": {
            "positive_prompt": BASE_POSITIVE,
            "directions": all_dirs(),
        },
    }
    for pose_id, pose_desc in ANCHOR_POSES:
        poses[pose_id] = {
            "from": {"ref": "base"},
            "category": "anchor",
            "positive_prompt": ANCHOR_TEMPLATE.replace("{pose}", pose_desc),
            "directions": all_dirs(),
        }

    animations: dict = {}
    for aid, category, length, start_from, end_at, motion, extra_neg in ANIMATIONS:
        entry: dict = {
            "category": category,
            "length": length,
            "positive_prompt": motion + SPRITE_SUFFIX,
            "directions": all_dirs(),
        }
        if start_from is not None:
            entry["start_from"] = {"ref": start_from}
        if end_at is not None:
            entry["end_at"] = {"ref": end_at}
        if extra_neg is not None:
            entry["negative_prompt"] = extra_neg
        animations[aid] = entry

    return {
        "version": 2,
        "directions": DIRECTIONS,
        "mirror_map": {
            "WEST": "EAST",
            "SOUTH_WEST": "SOUTH_EAST",
            "NORTH_WEST": "NORTH_EAST",
        },
        "view_phrases": VIEW_PHRASES,
        "defaults": {
            # Wan 2.2 i2v sprite defaults: 16 fps, 4n+1 length, the 480p bucket
            # (832x480) with shift 3.0 (the calmer, more controllable motion the
            # research recommends for a single small character). Per-animation
            # `width`/`height`/`length`/`shift` overrides are honored.
            "fps": 16,
            "length": 33,
            "width": 832,
            "height": 480,
            "shift": 3.0,
            "start_from": {"ref": "base"},
        },
        "globals": {
            # FLUX.2 Klein has NO negative path — poses carry no negative layer.
            "pose": {},
            # Wan 2.2 i2v uses the standard negative block (effective at CFG > 1).
            "animation": {"negative_prompt": WAN_NEGATIVE},
        },
        "poses": poses,
        "animations": animations,
    }


def main() -> None:
    manifest = build()
    repo_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    dest = os.path.join(repo_root, "examples", "animations.json")
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    print(f"wrote {dest}  ({len(manifest['poses'])} poses, "
          f"{len(manifest['animations'])} animations, {len(DIRECTIONS)} directions)")


if __name__ == "__main__":
    main()

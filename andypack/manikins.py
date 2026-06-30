"""Bundled manikin pose references, keyed by canonical direction (pure stdlib)."""

from __future__ import annotations

import os

CANONICAL_DIRECTIONS = [
    "EAST", "SOUTH_EAST", "SOUTH", "SOUTH_WEST",
    "WEST", "NORTH_WEST", "NORTH", "NORTH_EAST",
]

_ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets", "manikins")


def manikin_path(direction: str) -> str:
    """Absolute path to the bundled manikin PNG for `direction`.

    Raises RuntimeError for a direction outside CANONICAL_DIRECTIONS or whose
    asset is missing from the package, so a misconfigured graph fails loudly
    rather than feeding a nonexistent path to the image loader.
    """
    if direction not in CANONICAL_DIRECTIONS:
        raise RuntimeError(f"no manikin for unknown direction {direction!r}")
    path = os.path.join(_ASSETS_DIR, f"{direction}.png")
    if not os.path.isfile(path):
        raise RuntimeError(f"missing manikin asset for {direction!r} (expected {path})")
    return path

import os

import pytest

from andypack import manikins


def test_canonical_directions_are_the_eight_in_order():
    assert manikins.CANONICAL_DIRECTIONS == [
        "EAST", "SOUTH_EAST", "SOUTH", "SOUTH_WEST",
        "WEST", "NORTH_WEST", "NORTH", "NORTH_EAST",
    ]


def test_manikin_path_resolves_every_direction_to_an_existing_file():
    for direction in manikins.CANONICAL_DIRECTIONS:
        path = manikins.manikin_path(direction)
        assert os.path.isfile(path), path


def test_manikin_path_rejects_unknown_direction():
    with pytest.raises(RuntimeError, match="manikin"):
        manikins.manikin_path("UP")

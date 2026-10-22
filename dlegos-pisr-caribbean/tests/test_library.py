import pytest
import torch

from dlegos_pisr import (
    CubeSpec,
    SpacetimeCube,
    critical_success_index,
    fractions_skill_score,
    iter_tiles,
    stitch_tiles,
)


def test_cube_validates_metadata_and_selects_variable():
    values = torch.zeros(2, 2, 3, 4, 5)
    cube = SpacetimeCube(values, CubeSpec(("temperature", "wind"), times=("a", "b", "c")))
    assert cube.variable("wind").shape == (2, 3, 4, 5)
    with pytest.raises(KeyError):
        cube.variable("rain")


def test_tile_round_trip_with_overlap():
    field = torch.arange(2 * 13 * 17, dtype=torch.float32).reshape(2, 13, 17)
    pieces = list(iter_tiles(field, tile_size=(6, 7), overlap=(2, 3)))
    reconstructed = stitch_tiles(pieces, (13, 17))
    torch.testing.assert_close(reconstructed, field)


def test_precipitation_scores_on_perfect_event_field():
    field = torch.tensor([[0., 2., 17.], [0., 5., 0.]])
    assert critical_success_index(field, field, 16).item() == 1
    assert fractions_skill_score(field, field, 16, window=3).item() == pytest.approx(1.0)


def test_no_event_csi_is_one():
    zeros = torch.zeros(2, 5, 5)
    assert critical_success_index(zeros, zeros, 1).item() == 1

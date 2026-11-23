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


def test_csi_integer_input_keeps_fractional_score():
    prediction = torch.tensor([[1, 1], [0, 0]], dtype=torch.int16)
    target = torch.tensor([[1, 0], [1, 0]], dtype=torch.int16)
    score = critical_success_index(prediction, target, 1)
    assert score.is_floating_point()
    assert score.item() == pytest.approx(1 / 3)


def test_continuous_integer_metrics_do_not_wrap_before_subtracting():
    from dlegos_pisr import mae, rmse

    prediction = torch.tensor([1, 2], dtype=torch.uint8)
    target = torch.tensor([3, 6], dtype=torch.uint8)
    assert mae(prediction, target).item() == pytest.approx(3)
    assert rmse(prediction, target).item() == pytest.approx(10**0.5)


@pytest.mark.parametrize("window", [0, -3, 2, 4, True, 3.5])
def test_fss_rejects_uncentred_or_invalid_windows(window):
    field = torch.ones(3, 3)
    with pytest.raises(ValueError, match="positive odd"):
        fractions_skill_score(field, field, 1, window=window)


def test_fss_integer_and_float_fields_use_centred_zero_padded_windows():
    prediction = torch.tensor([[1, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=torch.int16)
    target = torch.tensor([[0, 1, 0], [0, 0, 0], [0, 0, 0]], dtype=torch.float64)
    # Four centres see the first event, six the second, and four see both.
    score = fractions_skill_score(prediction, target, 1, window=3)
    assert score.dtype == torch.float64
    assert score.item() == pytest.approx(0.8)
    assert fractions_skill_score(prediction, target, 1, window=1).item() == 0


def test_no_event_fss_is_one_and_one_sided_event_scores_zero():
    from dlegos_pisr import precipitation_scores

    truth = torch.zeros(2, 3, 3, dtype=torch.int16)
    perfect = precipitation_scores(truth, truth, thresholds=(1,), fss_window=3)
    assert all(score.item() == 1 for score in perfect.values())
    forecast = truth.clone()
    forecast[0, 1, 1] = 2
    missed = precipitation_scores(forecast, truth, thresholds=(1,), fss_window=3)
    assert all(score.item() == 0 for score in missed.values())


def test_tile_round_trip_integer_overlap_does_not_overflow():
    field = torch.full((5, 5), 200, dtype=torch.uint8)
    pieces = list(iter_tiles(field, tile_size=3, overlap=2))
    torch.testing.assert_close(stitch_tiles(pieces, (5, 5)), field.to(torch.get_default_dtype()))


def test_tiling_small_grid_and_rejects_invalid_coverage():
    from dlegos_pisr import Tile

    field = torch.ones(1, 2)
    pieces = list(iter_tiles(field, tile_size=4, overlap=3))
    assert len(pieces) == 1
    torch.testing.assert_close(stitch_tiles(pieces, (1, 2)), field)
    with pytest.raises(ValueError, match="uncovered"):
        stitch_tiles(pieces, (2, 2))
    with pytest.raises(ValueError, match="inside"):
        stitch_tiles([(Tile(-1, 0, 0, 2), field)], (1, 2))
    with pytest.raises(ValueError, match="must not be empty"):
        list(iter_tiles(torch.empty(0, 2), tile_size=4))


def test_both_models_run_on_a_selected_cube_time():
    from dlegos_pisr import DLegosPISR, MultiVarEDSR

    cube = SpacetimeCube(torch.rand(1, 2, 3, 4, 6), CubeSpec(("rain", "temperature")))
    frame = cube.values[:, :, 0]
    for model_type in (DLegosPISR, MultiVarEDSR):
        extra = {"block_px": 2} if model_type is DLegosPISR else {}
        model = model_type(in_ch=2, out_ch=2, width=4, blocks=2, scale=2, **extra)
        model.eval()
        with torch.inference_mode():
            prediction = model(frame)
        assert prediction.shape == (1, 2, 8, 12)
        assert torch.isfinite(prediction).all()

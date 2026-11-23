"""Synthetic tests of evaluator mechanics; no real precipitation is evaluated."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import xarray as xr


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_precipcorrdiff.py"
SPEC = importlib.util.spec_from_file_location("evaluate_precipcorrdiff", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def cube(units="mm/day"):
    return xr.DataArray(
        np.arange(1, 13, dtype=float).reshape(3, 2, 2),
        dims=("time", "lat", "lon"),
        coords={
            "time": np.arange("2016-01-01", "2016-01-04", dtype="datetime64[D]"),
            "lat": [17.5, 18.0],
            "lon": [-77.0, -76.5],
        },
        attrs={"units": units},
        name="precipitation",
    )


def save_pair(tmp_path, prediction, reference):
    paths = [tmp_path / name for name in ("prediction.nc", "reference.nc")]
    for path, data in zip(paths, (prediction, reference)):
        data.to_netcdf(path, engine="scipy")
    return paths


@pytest.mark.parametrize(
    ("unit", "factor"),
    [("mm d-1", 1.0), ("m/day", 1000.0), ("mm h-1", 24.0),
     ("kg m**-2 s**-1", 86400.0)],
)
def test_converts_units_and_aligns_reversed_axis_order(tmp_path, unit, factor):
    ref = cube()
    prediction = ((ref + 2) / factor).transpose("time", "lon", "lat")
    prediction = prediction.rename({"lat": "latitude", "lon": "longitude"})
    prediction.attrs["units"] = unit
    result = MODULE.evaluate(*save_pair(tmp_path, prediction, ref))
    assert result["n"] == 12
    assert result["mae_mm_day"] == pytest.approx(2)
    assert result["rmse_mm_day"] == pytest.approx(2)
    assert result["bias_mm_day"] == pytest.approx(2)
    assert result["provenance"]["prediction_units"] == unit


@pytest.mark.parametrize("unit", [None, "", "mm", "m", "kg m-2", "mm/month", "mm since 2016"])
def test_rejects_missing_or_ambiguous_reference_units(tmp_path, unit):
    ref = cube(unit) if unit is not None else cube()
    if unit is None:
        ref.attrs = {}
    with pytest.raises(ValueError, match="reference: unsupported or ambiguous"):
        MODULE.evaluate(*save_pair(tmp_path, cube(), ref))


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda a: a.rename({"lat": "y"}), "expected exactly"),
        (lambda a: a.drop_vars("lat"), "explicit one-dimensional"),
        (lambda a: a.assign_coords(lat=[18, 19]), "coordinates must match"),
        (lambda a: a.isel(lon=slice(None, None, -1)), "coordinates must match"),
        (lambda a: a.assign_coords(lon=[-77, -77]), "unique"),
        (lambda a: a.assign_coords(lat=[17.5, np.nan]), "nonmissing"),
        (lambda a: a.assign_coords(
            time=np.arange("2016-01-02", "2016-01-05", dtype="datetime64[D]")
        ), "coordinates must match"),
    ],
)
def test_rejects_invalid_or_mismatched_grids(tmp_path, mutate, message):
    with pytest.raises(ValueError, match=message):
        MODULE.evaluate(*save_pair(tmp_path, cube(), mutate(cube())))


def test_pairwise_missing_observations(tmp_path):
    pred, ref = cube(), cube()
    pred.values[0, 0, 0] = np.nan
    ref.values[0, 0, 1] = np.nan
    result = MODULE.evaluate(*save_pair(tmp_path, pred, ref))
    assert result["n"] == 10 and result["rmse_mm_day"] == 0


def test_rejects_negative_paired_rate(tmp_path):
    pred, ref = cube(), cube()
    pred.values[0, 0, 0] = -1
    with pytest.raises(ValueError, match="nonnegative"):
        MODULE.evaluate(*save_pair(tmp_path, pred, ref))


def test_cli_no_metrics_file_for_incompatible_data(tmp_path):
    prediction, reference = save_pair(tmp_path, cube(), cube("mm"))
    destination = tmp_path / "report.json"
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--prediction", str(prediction),
         "--reference", str(reference), "--output", str(destination)],
        capture_output=True, check=False, text=True,
    )
    assert completed.returncode != 0
    assert "unsupported or ambiguous" in completed.stderr
    assert not destination.exists()


def test_cli_writes_valid_json_for_matching_data(tmp_path):
    prediction, reference = save_pair(tmp_path, cube(), cube())
    destination = tmp_path / "report.json"
    subprocess.run(
        [sys.executable, str(SCRIPT), "--prediction", str(prediction),
         "--reference", str(reference), "--output", str(destination)],
        capture_output=True, check=True, text=True,
    )
    report = json.loads(destination.read_text())
    assert report["n"] == 12
    assert report["provenance"]["metric_units"] == "mm/day"
    assert report["mae_mm_day"] == 0

"""Evaluate released PrecipCorrDiff output against an independent reference."""

import argparse
import json
from pathlib import Path

import numpy as np
import xarray as xr


DIMS = ("time", "lat", "lon")
ALIASES = {"latitude": "lat", "longitude": "lon"}
MM_PER_DAY = {
    "mm/day": 1.0,
    "mm/d": 1.0,
    "mm day-1": 1.0,
    "mm d-1": 1.0,
    "mm day^-1": 1.0,
    "mm d^-1": 1.0,
    "m/day": 1000.0,
    "m/d": 1000.0,
    "m day-1": 1000.0,
    "m d-1": 1000.0,
    "mm/h": 24.0,
    "mm/hr": 24.0,
    "mm/hour": 24.0,
    "mm h-1": 24.0,
    "mm hr-1": 24.0,
    "mm hour-1": 24.0,
    "kg m-2 s-1": 86400.0,
    "kg m^-2 s^-1": 86400.0,
    "kg m**-2 s**-1": 86400.0,
}


def _prepare(data, role):
    # Recognize only explicit axis aliases; no regridding or inferred coordinates.
    rename = {old: new for old, new in ALIASES.items() if old in data.dims}
    if any(new in data.coords or new in data.dims for new in rename.values()):
        raise ValueError(f"{role}: conflicting coordinate aliases")
    data = data.rename(rename)
    if set(data.dims) != set(DIMS):
        raise ValueError(f"{role}: expected exactly time, latitude and longitude dimensions")
    for dim in DIMS:
        if dim not in data.coords or data.coords[dim].dims != (dim,) or dim not in data.indexes:
            raise ValueError(f"{role}: {dim} needs an explicit one-dimensional coordinate")
        index = data.indexes[dim]
        if not len(index) or not index.is_unique or index.hasnans:
            raise ValueError(f"{role}: {dim} coordinate must be nonempty, unique and nonmissing")
        if dim in ("lat", "lon") and (
            not np.issubdtype(data[dim].dtype, np.number)
            or not np.isfinite(data[dim].values).all()
        ):
            raise ValueError(f"{role}: {dim} coordinate must be finite and numeric")

    source_unit = data.attrs.get("units")
    unit = " ".join(source_unit.lower().split()) if isinstance(source_unit, str) else ""
    if unit not in MM_PER_DAY:
        raise ValueError(f"{role}: unsupported or ambiguous precipitation units: {source_unit!r}; "
                         "supply explicit rate units")
    return data.transpose(*DIMS), source_unit, MM_PER_DAY[unit]


def evaluate(prediction, reference, variable="precipitation"):
    with xr.open_dataset(prediction, decode_times={variable: False}) as pred, \
         xr.open_dataset(reference, decode_times={variable: False}) as ref:
        if variable not in pred.data_vars or variable not in ref.data_vars:
            raise ValueError("Required precipitation variable missing in prediction or reference")
        x, pred_unit, pred_factor = _prepare(pred[variable], "prediction")
        y, ref_unit, ref_factor = _prepare(ref[variable], "reference")
        try:
            x, y = xr.align(x, y, join="exact", copy=False)
        except ValueError as exc:
            raise ValueError("Prediction/reference coordinates must match exactly") from exc
        x = x.values.astype("float64") * pred_factor
        y = y.values.astype("float64") * ref_factor

    good = np.isfinite(x) & np.isfinite(y)
    if good.sum() < 10:
        raise ValueError("Insufficient valid paired observations")
    if np.any(x[good] < 0) or np.any(y[good] < 0):
        raise ValueError("Paired precipitation rates must be nonnegative")
    delta = x[good] - y[good]
    return {
        "n": int(good.sum()),
        "mae_mm_day": float(np.abs(delta).mean()),
        "rmse_mm_day": float(np.sqrt(np.mean(delta**2))),
        "bias_mm_day": float(delta.mean()),
        "provenance": {
            "prediction": str(prediction),
            "reference": str(reference),
            "variable": variable,
            "prediction_units": pred_unit,
            "reference_units": ref_unit,
            "metric_units": "mm/day",
        },
        "note": "Verify independent reference provenance and matching temporal and spatial support separately.",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prediction", required=True)
    p.add_argument("--reference", required=True)
    p.add_argument("--variable", default="precipitation")
    p.add_argument("--output", default="precipcorrdiff_metrics.json")
    args = p.parse_args()
    out = evaluate(args.prediction, args.reference, args.variable)
    report = json.dumps(out, indent=2, allow_nan=False)
    Path(args.output).write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()

# PrecipCorrDiff benchmark protocol (unrun)

Paper: https://arxiv.org/abs/2610.19994 (preprint).
Released outputs: https://doi.org/10.5281/zenodo.21819376 (~792.5 MB ZIP).

The archive contains downscaled precipitation output for India. It does not contain verified Caribbean predictions or confirmed model weights or training code. The Zenodo description says 2014–2018; the archive filename says 2014–2019. Inspect the actual NetCDF time coordinates.

## Phase 1: audit the released India output

1. Record the archive's SHA-256 digest. Inventory the NetCDF variables, time range, grid, precipitation units and temporal support.
2. Obtain the matching original IMD 0.25-degree input and an independent reference (finer gridded observations or a documented equivalent). Record the licences, processing steps and reference provenance.
3. Verify that both inputs represent the same intervals and spatial support. Run from `dlegos-pisr-caribbean/`:

   ```sh
   python scripts/evaluate_precipcorrdiff.py --prediction output.nc --reference reference.nc
   ```

   The evaluator accepts a rectilinear precipitation cube with explicit one-dimensional `time`, `lat` and `lon` coordinates. `latitude` and `longitude` are accepted aliases. Variable axis order may differ; coordinate labels and order must still match exactly. The evaluator does not re-grid, infer units or establish that a reference is independent. It normalizes supported precipitation rates (for example, `mm/day`, `m/day`, `mm h-1` and `kg m-2 s-1`) to `mm/day`. Missing units, plain accumulated depths such as `mm`, and other ambiguous or unsupported units are rejected. A rate conversion does not make different sampling or accumulation intervals comparable.
4. The evaluator reports paired MAE, RMSE and prediction-minus-reference bias in `mm/day`. For the full study, also plan threshold-dependent CSI, tail errors and uncertainty. Those analyses are not implemented by this evaluator.

## Phase 2: test Caribbean transfer

The India release cannot be evaluated as Jamaican predictions. Obtain Caribbean 0.25-degree precipitation and independent finer-resolution station, radar or satellite observations. Obtain a licensed implementation of the method or implement it independently. Compare with bilinear interpolation, SRGAN/CNN and D-LEGOS under identical train/test splits.

## Tests

The evaluator requires NumPy and xarray 2025.10 or newer. Non-standard calendars can require `cftime`. The tests also use SciPy for writing synthetic NetCDF files and pytest:

```sh
python -m pip install 'xarray>=2025.10' scipy cftime pytest
python -m pytest -q tests/test_precipcorrdiff.py
```

The tests exercise the evaluator with synthetic NetCDF pairs. They do not establish model skill. The real-data audit, training, inference and transfer experiment remain unrun.

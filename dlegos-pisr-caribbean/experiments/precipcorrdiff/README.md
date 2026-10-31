# PrecipCorrDiff benchmark — protocol (unrun)

Paper: https://arxiv.org/abs/2610.19994 (preprint).
Released outputs: https://doi.org/10.5281/zenodo.21819376 (~792.5 MB ZIP).
The archive contains **India** downscaled precipitation output, NOT Caribbean predictions, not confirmed model weights or training code. Zenodo description says 2014–2018, archive filename says 2014–2019; inspect actual NetCDF timestamps.

## Phase 1: reproducible India output audit
1. Download Zenodo archive and record SHA256 and NetCDF time, latitude, longitude, units.
2. Obtain matching original IMD 0.25-degree input and **independent** ground observations or 0.0625-degree reference, with licences and provenance.
3. Run `scripts/evaluate_precipcorrdiff.py --prediction output.nc --reference reference.nc` only after matching coordinates, variable names, dates and units.
4. Report MAE, RMSE, bias, threshold-dependent CSI, tail errors, and uncertainty; do not compare without compatible reference support.

## Phase 2: Caribbean method transfer
The Indian output cannot be evaluated as Jamaican predictions. Acquire Caribbean 0.25-degree precipitation and independent finer-resolution station/radar/satellite truth. Obtain a licensed implementation of the actual method or independently implement it, and compare to bilinear, SRGAN/CNN, and D-LEGOS under identical train/test splits. Mark all stages **unrun** until data and executable method are available.

**No training, inference, or numerical experiment has been performed by adding this protocol.**

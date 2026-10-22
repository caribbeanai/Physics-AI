# D-LEGOS-PISR Python library

This repository contains an installable Python package for modular,
physics-informed super-resolution of gridded atmospheric fields. The package
currently exposes the existing EDSR-like and D-LEGOS-PISR PyTorch models,
physical consistency utilities, a named spatiotemporal cube, spatial tiling,
and precipitation verification metrics.

## Install

From `dlegos-pisr-caribbean/`:

```bash
python -m pip install -e .
python -m pip install -e '.[dev]'
pytest
```

Install `.[data]` to work with NetCDF/ERA5 helpers and `.[research]` for the
experiment stack. PyTorch is the only core dependency.

## Quick start

```python
import torch
from dlegos_pisr import CubeSpec, SpacetimeCube, DLegosPISR, precipitation_scores

# Cube layout: batch, variables, time, latitude, longitude
x = torch.randn(2, 5, 3, 32, 48)
cube = SpacetimeCube(
    x,
    CubeSpec(
        variables=("t2m", "d2m", "sp", "u10", "v10"),
        lat_spacing_deg=0.25,
        lon_spacing_deg=0.25,
    ),
)

# The current super-resolution networks are 2-D. Select/reshape time steps
# explicitly for training rather than silently changing tensor dimensions.
model = DLegosPISR(in_ch=5, out_ch=5, scale=4)
one_time = cube.values[:, :, 0]
high_resolution = model(one_time)

rain_pred = torch.rand(2, 3, 64, 96) * 20
rain_true = torch.rand_like(rain_pred) * 20
scores = precipitation_scores(rain_pred, rain_true, thresholds=(1, 5, 16))
print({name: value.item() for name, value in scores.items()})
```

## Tiling large domains

`iter_tiles(tensor, tile_size, overlap)` yields `(Tile, tensor_tile)` pairs
for arrays whose final two dimensions are latitude and longitude.
`stitch_tiles(...)` reconstructs the full grid with uniform overlap averaging.
No padding or boundary conditions are invented. Ensure the tiles cover the
entire domain.

## Physics and evaluation

The existing `physics_terms`, `normalized_physical_consistency`, and
`interface_consistency` functions are available from the package root. The
precipitation metrics report critical success index (CSI) and fractions skill
score (FSS) for thresholded fields. Thresholds use the same units as the data.
They are verification tools; they do not by themselves establish forecast
skill or hydrological value.

The attached NowcastNet/HRRR evaluation paper motivates evaluating heavy-event
skill alongside aggregate-field errors. Its results are specific to its study
region, data, model, and verification design. This library neither implements
NowcastNet nor transfers those reported results to the Caribbean.

Reference: Das et al. (2024), “Hybrid physics-AI outperforms numerical weather
prediction for extreme precipitation nowcasting,” *npj Climate and Atmospheric
Science*, 7, 282. https://doi.org/10.1038/s41612-024-00834-8

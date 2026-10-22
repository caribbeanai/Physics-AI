# D-LEGOS-PISR Caribbean

Reproducible research scaffold for **modular physics-informed super-resolution of Caribbean atmospheric fields**.

## Scientific question
Can locally coupled, physics-constrained D-LEGOS blocks reconstruct fine-scale Caribbean atmospheric fields more accurately and consistently than standard super-resolution or monolithic PISR?

## Primary experiment
ERA5 Caribbean domain: **30°N–5°N, 90°W–55°W**. Variables: 2 m temperature, 2 m dew point, surface pressure, 10 m u wind, 10 m v wind. Target: 0.25°. Primary controlled degradation: 1.0° → 0.25° (4×). Additional tests: 0.5° → 0.25° and 2.0° → 0.25°.

Splits: train 1981–2015, validation 2016, test 2017–2018, plus temporal holdout 2019–2025.

## Models
1. Bilinear interpolation
2. Multivariate EDSR-like baseline
3. PISR: multivariate SR + differentiable near-surface physics losses
4. D-LEGOS-PISR: PISR + modular block message passing + interface consistency

## Metrics
RMSE/MAE by variable; physical consistency residuals; D-LEGOS Interface Physical Consistency (IPC); later SSIM/spectral diagnostics and extreme-event skill.

## Python library
The `dlegos_pisr` package now provides installable model APIs, named spatiotemporal cubes, overlap-aware spatial tiling, physics utilities, and precipitation CSI/FSS metrics. See [docs/PYTHON_LIBRARY.md](docs/PYTHON_LIBRARY.md) for installation and examples.

Install from this directory with `python -m pip install -e .`.

## Research integrity
No scientific result should be reported until ERA5 has been downloaded and the full benchmark has run. Library smoke tests validate software behavior only.

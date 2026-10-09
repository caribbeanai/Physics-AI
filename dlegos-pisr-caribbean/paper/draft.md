# D-LEGOS-PISR: Modular Physics-Informed Super-Resolution of Caribbean Atmospheric Fields

**Adrian Dunkley**

## Abstract
High-resolution atmospheric information is essential for climate-risk analysis in small-island and complex-terrain regions, yet global reanalyses and climate-model outputs remain coarse relative to many Caribbean applications. Recent physics-informed super-resolution work has shown that hydrostatic primitive-equation constraints can improve physical consistency. We propose D-LEGOS-PISR, a modular extension in which atmospheric space is represented as interacting local physical blocks. Each block reconstructs a multivariate atmospheric state while satisfying differentiable physics constraints, and adjacent blocks exchange information subject to an interface-consistency penalty.

## Core experiment
ERA5: 2 m temperature, 2 m dew-point temperature, surface pressure, 10 m u wind, 10 m v wind. Caribbean domain: 5°N–30°N, 90°W–55°W. Primary task: controlled 1.0°→0.25° super-resolution. Additional scale tests: 0.5°→0.25° and 2.0°→0.25°. Train 1981–2015, validation 2016, primary test 2017–2018, temporal holdout 2019–2025.

## Proposed objective
L = lambda_r L_rec + lambda_p L_phys + lambda_I L_IPC + lambda_m L_multiscale.

The scientific comparison is bilinear interpolation vs multivariate EDSR-like SR vs monolithic PISR vs D-LEGOS-PISR. Primary evaluation uses variable-wise RMSE/MAE, physics residual consistency, and a new Interface Physical Consistency measure. Extreme winds, heat extremes, and tropical-cyclone case studies follow after the primary benchmark.

## Research integrity
Results remain intentionally unfilled until the ERA5 benchmark is executed under the fixed configuration. Synthetic software tests are not climate results.

## References
Xu, C., Sumbul, G., Porta, H., Béchaz, M., Schemm, S., & Tuia, D. (2026). Physics-Informed Super-Resolution of Atmospheric Data. arXiv:2607.18877.

Hersbach, H. et al. (2020). The ERA5 global reanalysis. Quarterly Journal of the Royal Meteorological Society, 146, 1999–2049.

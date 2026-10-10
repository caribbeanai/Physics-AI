# BootLoops Validation Protocol for D-LEGOS-PISR

This protocol applies the previously used BootLoops-style reproducibility and claim-check workflow to every remaining experiment in the D-LEGOS-PISR Caribbean study.

## Scope
Apply to:
1. 1.0° -> 0.25° primary reconstruction
2. 0.5° -> 0.25° scale test
3. 2.0° -> 0.25° scale test
4. 2019-2025 temporal holdout
5. coastline/island subsets
6. heat-extreme evaluation
7. extreme-wind evaluation
8. tropical-cyclone case studies
9. ablations: no-physics, local-physics-only, interface-only, multiscale
10. any later ERA5-Land or satellite validation

## Required checks per experiment
- Freeze config, random seed, dataset window, variables and domain.
- Save input manifest and data hashes where feasible.
- Save predictions, metrics and model checkpoint references.
- Recompute headline metrics independently from saved predictions.
- Run deliberate-wrong-input tests so validation code is shown to fail when inputs are corrupted.
- Run unit tests for physical residuals and interface metric.
- Compare every numerical statement intended for the paper against a machine-readable result.
- Distinguish software smoke-test outputs from scientific results.
- Record failed runs and exclusions; never silently drop them.
- Preserve null/negative results.

## Publication claim gate
A numerical claim may enter the manuscript only when:
1. the associated experiment completed,
2. the result is saved,
3. the metric can be independently recomputed,
4. the claim-check script passes,
5. the statistical comparison is available where the claim is comparative,
6. the exact dataset/time period is stated.

## Statistical checks
For paired model comparisons across matched samples:
- paired bootstrap confidence intervals for metric differences;
- Wilcoxon signed-rank as a robustness check when distributional assumptions are doubtful;
- report effect size and uncertainty, not p-values alone;
- correct for multiple comparisons when many models/variables are tested.

## Reproducibility outputs
Each run creates:
- config.yaml
- environment.txt
- git_commit.txt
- metrics.json
- predictions manifest
- validation_report.json
- claim_checks.json
- logs/

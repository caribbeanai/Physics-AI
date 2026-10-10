# Validation

The D-LEGOS-PISR study applies the same BootLoops-style validation approach previously used for the user's scientific experiments.

Every experiment must save its fixed configuration, outputs and machine-readable metrics. Headline metrics are recomputed independently before they are allowed into the manuscript. Deliberate wrong-input tests are required to prove that validators fail when inputs are corrupted. Null and negative results remain in the record.

The experiment matrix in `experiment_matrix.yaml` defines the required remaining runs and prevents post-hoc omission of unfavorable experiments.

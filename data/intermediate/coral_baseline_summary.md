# CORAL Covariance-Alignment Diagnostic Baseline

Shallow unsupervised CORAL is included as one representative covariate-alignment baseline, not as an exhaustive domain-adaptation benchmark.

Protocol: N=100, 34-feature Q0-normalised table; five official source-training splits; fixed RidgeCV log-life regressor; full unlabeled target covariates for CORAL moment alignment; target labels used only for scoring. Predictions are clipped to the source-training log-life range as a conservative extrapolation policy.

This diagnostic intentionally keeps the downstream learner fixed. The comparison is therefore naive Ridge transfer vs CORAL-aligned Ridge transfer, not a replacement for the model-champion cross-dataset table.

## Summary

- CORAL improves R2 over naive Ridge transfer in **6/12** directions.
- CORAL yields positive R2 in **2/12** directions.
- In rank-collapsed / CP-only directions, CORAL yields positive R2 in **0/4** directions.
- Interpretation should be directional and regime-aware: CORAL modifies unlabeled covariate geometry; it does not use target labels and does not directly repair conditional shift.

## Direction-Level Results

| Direction | Regime | Naive Ridge R2 | CORAL Ridge R2 | Delta R2 | Naive MAE | CORAL MAE | Delta MAE | Reference Table 3 model | Reference Table 3 R2 |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|
| hust -> matr | cp_interval_only | -7.240 | -3.814 | 3.425 | 945.9 | 728.1 | -217.8 | gaussian_process | -3.600 |
| luh -> hust | cp_interval_only | -7.187 | -12.087 | -4.900 | 551.1 | 858.9 | 307.9 | random_forest | -0.763 |
| matr -> hust | cp_interval_only | -9.673 | -7.825 | 1.847 | 755.1 | 760.4 | 5.3 | gaussian_process | -7.937 |
| sandia -> matr | cp_interval_only | -26.638 | -2.331 | 24.308 | 1657.7 | 494.9 | -1162.7 | gaussian_process | -0.802 |
| luh -> matr | offset_dominant_residual_only | -4.810 | -3.105 | 1.705 | 806.5 | 618.0 | -188.5 | stacking | -0.275 |
| sandia -> hust | offset_dominant_residual_only | -8.256 | -14.679 | -6.423 | 648.0 | 1045.3 | 397.3 | catboost | -2.027 |
| hust -> luh | salvageable_linear_recovers | -0.747 | -4.745 | -3.998 | 471.7 | 931.1 | 459.4 | pls | -0.562 |
| hust -> sandia | salvageable_linear_recovers | 0.312 | -0.185 | -0.497 | 807.3 | 1149.6 | 342.3 | xgboost | -0.231 |
| luh -> sandia | salvageable_linear_recovers | 0.007 | -0.121 | -0.128 | 768.6 | 821.7 | 53.1 | xgboost | 0.499 |
| matr -> luh | salvageable_linear_recovers | -6.326 | 0.134 | 6.460 | 949.0 | 335.4 | -613.6 | stacking | 0.004 |
| matr -> sandia | salvageable_linear_recovers | -1.054 | 0.167 | 1.221 | 1433.0 | 746.5 | -686.5 | xgboost | 0.046 |
| sandia -> luh | salvageable_linear_recovers | 0.373 | -1.052 | -1.425 | 204.9 | 315.6 | 110.7 | pls | 0.494 |

## Caveat

CORAL is a mean/covariance-alignment diagnostic. It is useful as a standard unsupervised alignment falsifier, but it is not a claim that all modern domain-adaptation methods have been exhausted.

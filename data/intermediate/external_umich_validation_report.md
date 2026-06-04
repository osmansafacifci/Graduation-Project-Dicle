# UMich Held-Out External Validation

Frozen thresholds: `configs/regime_thresholds_frozen.yaml`.
Feature table: `data/intermediate/features_sop12_four_dataset_plus_umich_capnorm.csv`.

UMich is treated as held-out external validation, not as a fifth member of the main benchmark.

## Within-UMich Baseline

| model | MAE_mean | SMAPE_mean | R2_mean |
| --- | --- | --- | --- |
| random_forest | 31.425 | 9.879 | 0.258 |
| gaussian_process | 31.126 | 9.720 | 0.139 |
| xgboost | 34.203 | 10.722 | 0.128 |

## Eight New External Directions

| experiment | model | life_ratio_target_over_source | raw_R2 | pearson_r | rank_signal_class | linear_R2 | predicted_super_regime | frozen_threshold_match |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hust_to_umich | random_forest | 0.224 | -540.627 | -0.016 | rank_signal_collapsed | 0.003 | cp_interval_only | True |
| luh_to_umich | gaussian_process | 0.814 | -2.459 | -0.461 | negative_or_inverted_signal | 0.128 | cp_interval_only | True |
| matr_to_umich | catboost | 0.467 | -32.869 | 0.150 | weak_rank_signal | 0.026 | offset_dominant_residual_only | False |
| sandia_to_umich | catboost | 0.906 | -137.227 | -0.079 | rank_signal_collapsed | 0.007 | cp_interval_only | True |
| umich_to_hust | elastic_net | 4.468 | -17.120 | -0.198 | negative_or_inverted_signal | 0.008 | cp_interval_only | True |
| umich_to_luh | xgboost | 1.229 | -0.307 | 0.055 | rank_signal_collapsed | 0.090 | cp_interval_only | True |
| umich_to_matr | random_forest | 2.143 | -1.331 | 0.063 | rank_signal_collapsed | 0.005 | cp_interval_only | True |
| umich_to_sandia | stacking | 1.104 | -0.159 | -0.070 | rank_signal_collapsed | 0.073 | cp_interval_only | True |

Frozen-threshold agreement: **7/8 directions**.

Interpretation: a match means the rank-signal threshold predicted the deployment response class: strong/moderate should permit meaningful linear k-shot recovery, weak should be offset-dominant, and collapsed/inverted should remain CP-interval-only for point prediction.

## Outputs

- `outputs/results_v2_external_umich_validation/external_umich_best_cross_models.csv`
- `outputs/results_v2_external_umich_validation/external_umich_direction_summary.csv`
- `outputs/results_v2_external_umich_validation/external_umich_seed_diagnostics.csv`
- `outputs/results_v2_external_umich_validation/external_umich_predictions.csv`

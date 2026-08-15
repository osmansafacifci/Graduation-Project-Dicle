# Feature-Effect Transport

TreeSHAP contributions are decomposed in log-cycle space; Spearman and cycle-space R2 are sensitivity checks. Positive group contributions carry source-model rank signal on the target cohort; negative groups oppose that ranking.

## Pilot directions
| source | target | model | pearson_log_mean | spearman_cycle_mean | r2_cycle_mean | dominant_positive_group | dominant_positive_contribution | dominant_negative_group | dominant_negative_contribution | rank_signal_class |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hust | luh | catboost | 0.018 | 0.003 | -4.112 | curvature_acceleration | 0.116 | capacity_level_retention | -0.049 | strong_rank_signal |
| hust | luh | random_forest | 0.241 | 0.095 | -3.921 | curvature_acceleration | 0.138 | variability_events | -0.046 | strong_rank_signal |
| hust | matr | catboost | 0.268 | 0.211 | -4.992 | trend_decay_timing | 0.206 | curvature_acceleration | -0.025 | negative_or_inverted_signal |
| hust | matr | random_forest | 0.375 | 0.32 | -4.555 | trend_decay_timing | 0.228 | capacity_level_retention | -0.021 | negative_or_inverted_signal |
| luh | hust | catboost | -0.171 | -0.16 | -3.655 | trend_decay_timing | 0.037 | curvature_acceleration | -0.107 | rank_signal_collapsed |
| luh | hust | random_forest | 0.012 | -0.013 | -0.763 | trend_decay_timing | 0.092 | curvature_acceleration | -0.075 | rank_signal_collapsed |
| matr | hust | catboost | -0.1 | -0.064 | -10.355 | capacity_level_retention | 0.081 | curvature_acceleration | -0.219 | negative_or_inverted_signal |
| matr | hust | random_forest | -0.15 | -0.112 | -10.538 | trend_decay_timing | 0.076 | curvature_acceleration | -0.29 | negative_or_inverted_signal |

## All directions, direction-level summary
| source | target | model | pearson_log_mean | spearman_cycle_mean | dominant_positive_group | dominant_negative_group | rank_signal_class |
| --- | --- | --- | --- | --- | --- | --- | --- |
| sandia | luh | catboost | 0.931 | 0.909 | trend_decay_timing | spectral_entropy | strong_rank_signal |
| luh | sandia | random_forest | 0.89 | 0.754 | variability_events | spectral_entropy | strong_rank_signal |
| luh | sandia | catboost | 0.887 | 0.725 | trend_decay_timing | curvature_acceleration | strong_rank_signal |
| sandia | luh | random_forest | 0.829 | 0.833 | trend_decay_timing | curvature_acceleration | strong_rank_signal |
| hust | sandia | catboost | 0.527 | 0.481 | trend_decay_timing | capacity_level_retention | strong_rank_signal |
| matr | luh | catboost | 0.519 | 0.446 | curvature_acceleration | variability_events | moderate_rank_signal |
| matr | luh | random_forest | 0.464 | 0.209 | curvature_acceleration | variability_events | moderate_rank_signal |
| matr | sandia | catboost | 0.462 | 0.42 | trend_decay_timing | spectral_entropy | moderate_rank_signal |
| hust | sandia | random_forest | 0.434 | 0.561 | trend_decay_timing | spectral_entropy | strong_rank_signal |
| hust | matr | random_forest | 0.375 | 0.32 | trend_decay_timing | capacity_level_retention | negative_or_inverted_signal |
| luh | matr | catboost | 0.325 | 0.143 | trend_decay_timing | capacity_level_retention | weak_rank_signal |
| sandia | matr | catboost | 0.319 | 0.148 | trend_decay_timing | curvature_acceleration | rank_signal_collapsed |
| luh | matr | random_forest | 0.3 | -0.048 | trend_decay_timing | curvature_acceleration | weak_rank_signal |
| hust | matr | catboost | 0.268 | 0.211 | trend_decay_timing | curvature_acceleration | negative_or_inverted_signal |
| sandia | matr | random_forest | 0.266 | 0.013 | trend_decay_timing | curvature_acceleration | rank_signal_collapsed |
| matr | sandia | random_forest | 0.258 | 0.233 | curvature_acceleration | spectral_entropy | moderate_rank_signal |
| hust | luh | random_forest | 0.241 | 0.095 | curvature_acceleration | variability_events | strong_rank_signal |
| sandia | hust | catboost | 0.183 | 0.113 | variability_events | curvature_acceleration | weak_rank_signal |
| sandia | hust | random_forest | 0.034 | -0.069 | spectral_entropy | trend_decay_timing | weak_rank_signal |
| hust | luh | catboost | 0.018 | 0.003 | curvature_acceleration | capacity_level_retention | strong_rank_signal |
| luh | hust | random_forest | 0.012 | -0.013 | trend_decay_timing | curvature_acceleration | rank_signal_collapsed |
| matr | hust | catboost | -0.1 | -0.064 | capacity_level_retention | curvature_acceleration | negative_or_inverted_signal |
| matr | hust | random_forest | -0.15 | -0.112 | trend_decay_timing | curvature_acceleration | negative_or_inverted_signal |
| luh | hust | catboost | -0.171 | -0.16 | trend_decay_timing | curvature_acceleration | rank_signal_collapsed |

## Group contribution means
| source | target | model | capacity_level_retention | curvature_acceleration | spectral_entropy | trend_decay_timing | variability_events |
| --- | --- | --- | --- | --- | --- | --- | --- |
| hust | luh | catboost | -0.049 | 0.116 | -0.022 | 0.015 | -0.042 |
| hust | luh | random_forest | 0.022 | 0.138 | -0.004 | 0.131 | -0.046 |
| hust | matr | catboost | -0.01 | -0.025 | 0.01 | 0.206 | 0.086 |
| hust | matr | random_forest | -0.021 | 0.093 | 0.006 | 0.228 | 0.069 |
| hust | sandia | catboost | 0.01 | 0.063 | 0.104 | 0.281 | 0.069 |
| hust | sandia | random_forest | 0.026 | 0.149 | 0.02 | 0.201 | 0.037 |
| luh | hust | catboost | 0.0 | -0.107 | -0.065 | 0.037 | -0.036 |
| luh | hust | random_forest | 0.008 | -0.075 | -0.037 | 0.092 | 0.024 |
| luh | matr | catboost | 0.03 | 0.049 | 0.055 | 0.154 | 0.038 |
| luh | matr | random_forest | 0.023 | 0.01 | 0.011 | 0.15 | 0.106 |
| luh | sandia | catboost | 0.306 | -0.007 | -0.004 | 0.394 | 0.198 |
| luh | sandia | random_forest | 0.162 | 0.069 | 0.003 | 0.311 | 0.346 |
| matr | hust | catboost | 0.081 | -0.219 | -0.021 | 0.053 | 0.005 |
| matr | hust | random_forest | 0.052 | -0.29 | -0.015 | 0.076 | 0.026 |
| matr | luh | catboost | 0.049 | 0.494 | -0.015 | 0.064 | -0.075 |
| matr | luh | random_forest | 0.013 | 0.466 | -0.009 | 0.075 | -0.082 |
| matr | sandia | catboost | 0.06 | 0.119 | 0.031 | 0.208 | 0.045 |
| matr | sandia | random_forest | 0.022 | 0.173 | -0.013 | 0.07 | 0.005 |
| sandia | hust | catboost | 0.017 | -0.042 | 0.021 | 0.014 | 0.173 |
| sandia | hust | random_forest | -0.033 | 0.08 | 0.14 | -0.085 | -0.068 |
| sandia | luh | catboost | 0.281 | 0.062 | -0.161 | 0.538 | 0.21 |
| sandia | luh | random_forest | 0.222 | -0.001 | 0.013 | 0.306 | 0.289 |
| sandia | matr | catboost | 0.032 | -0.009 | 0.077 | 0.126 | 0.092 |
| sandia | matr | random_forest | 0.036 | 0.004 | 0.053 | 0.092 | 0.081 |

# Four-Dataset Conditional-Shift Diagnostics

Feature table: `features_sop12_four_dataset_capnorm.csv`; N=100; censored cells excluded from regression diagnostics.

## Pairwise Feature-Slope Shift
| pair | log_life_offset_b_minus_a | life_ratio_b_over_a | slope_shifted_features | n_features | slope_shifted_share |
| --- | --- | --- | --- | --- | --- |
| hust_vs_luh | -1.291 | 0.275 | 26.000 | 34.000 | 0.765 |
| hust_vs_sandia | -1.398 | 0.247 | 25.000 | 34.000 | 0.735 |
| matr_vs_sandia | -0.663 | 0.515 | 19.000 | 34.000 | 0.559 |
| matr_vs_luh | -0.556 | 0.573 | 17.000 | 34.000 | 0.500 |
| matr_vs_hust | 0.735 | 2.085 | 14.000 | 34.000 | 0.412 |
| sandia_vs_luh | 0.107 | 1.113 | 3.000 | 34.000 | 0.088 |

## Direction-Level Source-Prediction Diagnostics
| experiment | model | raw_R2 | pearson_r | rank_signal_class | residual_R2 | linear_R2 | adapter_class | slope_shifted_share | life_ratio_target_over_source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hust_to_luh | pls | -0.562 | 0.772 | strong_rank_signal | 0.424 | 0.610 | linear_recovers_predictive_transfer | 0.765 | 0.275 |
| hust_to_matr | gaussian_process | -3.600 | -0.114 | negative_or_inverted_signal | -0.000 | 0.008 | offset_dominant_repair | 0.412 | 0.480 |
| hust_to_sandia | xgboost | -0.231 | 0.577 | strong_rank_signal | 0.216 | 0.366 | linear_recovers_predictive_transfer | 0.735 | 0.247 |
| luh_to_hust | random_forest | -0.763 | 0.015 | rank_signal_collapsed | -0.027 | 0.021 | offset_dominant_repair | 0.765 | 3.636 |
| luh_to_matr | stacking | -0.275 | 0.107 | weak_rank_signal | -0.008 | 0.020 | offset_dominant_repair | 0.500 | 1.744 |
| luh_to_sandia | xgboost | 0.499 | 0.913 | strong_rank_signal | 0.545 | 0.842 | linear_recovers_predictive_transfer | 0.088 | 0.898 |
| matr_to_hust | gaussian_process | -7.937 | -0.120 | negative_or_inverted_signal | -0.181 | 0.025 | center_repaired_but_low_rank | 0.412 | 2.085 |
| matr_to_luh | stacking | 0.004 | 0.407 | moderate_rank_signal | 0.074 | 0.193 | limited_repair | 0.500 | 0.573 |
| matr_to_sandia | xgboost | 0.046 | 0.427 | moderate_rank_signal | 0.078 | 0.261 | linear_recovers_predictive_transfer | 0.559 | 0.515 |
| sandia_to_hust | catboost | -2.027 | 0.187 | weak_rank_signal | -0.489 | 0.039 | center_repaired_but_low_rank | 0.735 | 4.047 |
| sandia_to_luh | pls | 0.494 | 0.862 | strong_rank_signal | 0.625 | 0.750 | linear_recovers_predictive_transfer | 0.088 | 1.113 |
| sandia_to_matr | gaussian_process | -0.802 | -0.006 | rank_signal_collapsed | -0.164 | 0.003 | center_repaired_but_low_rank | 0.559 | 1.941 |

## Strongest Calibrated Directions
| experiment | model | raw_R2 | pearson_r | rank_signal_class | residual_R2 | linear_R2 | adapter_class | slope_shifted_share | life_ratio_target_over_source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| luh_to_sandia | xgboost | 0.499 | 0.913 | strong_rank_signal | 0.545 | 0.842 | linear_recovers_predictive_transfer | 0.088 | 0.898 |
| sandia_to_luh | pls | 0.494 | 0.862 | strong_rank_signal | 0.625 | 0.750 | linear_recovers_predictive_transfer | 0.088 | 1.113 |
| hust_to_luh | pls | -0.562 | 0.772 | strong_rank_signal | 0.424 | 0.610 | linear_recovers_predictive_transfer | 0.765 | 0.275 |
| hust_to_sandia | xgboost | -0.231 | 0.577 | strong_rank_signal | 0.216 | 0.366 | linear_recovers_predictive_transfer | 0.735 | 0.247 |

## Weakest Rank-Signal Directions
| experiment | model | raw_R2 | pearson_r | rank_signal_class | residual_R2 | linear_R2 | adapter_class | slope_shifted_share | life_ratio_target_over_source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| matr_to_hust | gaussian_process | -7.937 | -0.120 | negative_or_inverted_signal | -0.181 | 0.025 | center_repaired_but_low_rank | 0.412 | 2.085 |
| hust_to_matr | gaussian_process | -3.600 | -0.114 | negative_or_inverted_signal | -0.000 | 0.008 | offset_dominant_repair | 0.412 | 0.480 |
| sandia_to_matr | gaussian_process | -0.802 | -0.006 | rank_signal_collapsed | -0.164 | 0.003 | center_repaired_but_low_rank | 0.559 | 1.941 |
| luh_to_hust | random_forest | -0.763 | 0.015 | rank_signal_collapsed | -0.027 | 0.021 | offset_dominant_repair | 0.765 | 3.636 |

Interpretation: `pearson_r` is the source model's rank signal on the target dataset. Linear calibration can exploit positive rank signal; residual-mean calibration mostly repairs the target center. Directions with low or negative rank signal are concept/conditional-shift failures even if target calibration reduces MAE.

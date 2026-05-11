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
| hust_to_luh | pls | -0.373 | 0.766 | strong_rank_signal | 0.443 | 0.600 | linear_recovers_predictive_transfer | 0.765 | 0.275 |
| hust_to_matr | gaussian_process | -3.600 | -0.130 | negative_or_inverted_signal | -0.000 | 0.015 | offset_dominant_repair | 0.412 | 0.480 |
| hust_to_sandia | xgboost | -0.296 | 0.520 | strong_rank_signal | 0.188 | 0.296 | linear_recovers_predictive_transfer | 0.735 | 0.247 |
| luh_to_hust | random_forest | -0.766 | 0.012 | rank_signal_collapsed | -0.028 | 0.021 | offset_dominant_repair | 0.765 | 3.636 |
| luh_to_matr | stacking | -0.313 | 0.102 | weak_rank_signal | -0.012 | 0.019 | offset_dominant_repair | 0.500 | 1.744 |
| luh_to_sandia | xgboost | 0.492 | 0.912 | strong_rank_signal | 0.542 | 0.840 | linear_recovers_predictive_transfer | 0.088 | 0.898 |
| matr_to_hust | gaussian_process | -7.937 | -0.120 | negative_or_inverted_signal | -0.181 | 0.025 | center_repaired_but_low_rank | 0.412 | 2.085 |
| matr_to_luh | stacking | 0.012 | 0.446 | moderate_rank_signal | 0.082 | 0.236 | limited_repair | 0.500 | 0.573 |
| matr_to_sandia | catboost | 0.041 | 0.522 | strong_rank_signal | 0.067 | 0.343 | linear_recovers_predictive_transfer | 0.559 | 0.515 |
| sandia_to_hust | catboost | -1.431 | 0.188 | weak_rank_signal | -0.319 | 0.042 | center_repaired_but_low_rank | 0.735 | 4.047 |
| sandia_to_luh | pls | 0.477 | 0.859 | strong_rank_signal | 0.613 | 0.744 | linear_recovers_predictive_transfer | 0.088 | 1.113 |
| sandia_to_matr | gaussian_process | -0.799 | -0.006 | rank_signal_collapsed | -0.167 | 0.003 | center_repaired_but_low_rank | 0.559 | 1.941 |

## Strongest Calibrated Directions
| experiment | model | raw_R2 | pearson_r | rank_signal_class | residual_R2 | linear_R2 | adapter_class | slope_shifted_share | life_ratio_target_over_source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| luh_to_sandia | xgboost | 0.492 | 0.912 | strong_rank_signal | 0.542 | 0.840 | linear_recovers_predictive_transfer | 0.088 | 0.898 |
| sandia_to_luh | pls | 0.477 | 0.859 | strong_rank_signal | 0.613 | 0.744 | linear_recovers_predictive_transfer | 0.088 | 1.113 |
| hust_to_luh | pls | -0.373 | 0.766 | strong_rank_signal | 0.443 | 0.600 | linear_recovers_predictive_transfer | 0.765 | 0.275 |
| matr_to_sandia | catboost | 0.041 | 0.522 | strong_rank_signal | 0.067 | 0.343 | linear_recovers_predictive_transfer | 0.559 | 0.515 |

## Weakest Rank-Signal Directions
| experiment | model | raw_R2 | pearson_r | rank_signal_class | residual_R2 | linear_R2 | adapter_class | slope_shifted_share | life_ratio_target_over_source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hust_to_matr | gaussian_process | -3.600 | -0.130 | negative_or_inverted_signal | -0.000 | 0.015 | offset_dominant_repair | 0.412 | 0.480 |
| matr_to_hust | gaussian_process | -7.937 | -0.120 | negative_or_inverted_signal | -0.181 | 0.025 | center_repaired_but_low_rank | 0.412 | 2.085 |
| sandia_to_matr | gaussian_process | -0.799 | -0.006 | rank_signal_collapsed | -0.167 | 0.003 | center_repaired_but_low_rank | 0.559 | 1.941 |
| luh_to_hust | random_forest | -0.766 | 0.012 | rank_signal_collapsed | -0.028 | 0.021 | offset_dominant_repair | 0.765 | 3.636 |

Interpretation: `pearson_r` is the source model's rank signal on the target dataset. Linear calibration can exploit positive rank signal; residual-mean calibration mostly repairs the target center. Directions with low or negative rank signal are concept/conditional-shift failures even if target calibration reduces MAE.

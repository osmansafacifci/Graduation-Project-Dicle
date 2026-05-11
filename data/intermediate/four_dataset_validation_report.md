# Four-Dataset Extension Validation

## Check Summary
| check | status | details |
| --- | --- | --- |
| exists:data/intermediate/features_sop12_four_dataset.csv | PASS | /Users/osmancifci/Downloads/Graduation-Project-Dicle/data/intermediate/features_sop12_four_dataset.csv |
| exists:data/intermediate/features_sop12_four_dataset_capnorm.csv | PASS | /Users/osmancifci/Downloads/Graduation-Project-Dicle/data/intermediate/features_sop12_four_dataset_capnorm.csv |
| exists:data/intermediate/four_dataset_manifest.csv | PASS | /Users/osmancifci/Downloads/Graduation-Project-Dicle/data/intermediate/four_dataset_manifest.csv |
| exists:data/intermediate/sandia_cell_audit.csv | PASS | /Users/osmancifci/Downloads/Graduation-Project-Dicle/data/intermediate/sandia_cell_audit.csv |
| exists:data/intermediate/luh_cell_audit.csv | PASS | /Users/osmancifci/Downloads/Graduation-Project-Dicle/data/intermediate/luh_cell_audit.csv |
| exists:outputs/results_v2_four_dataset_within_34feat_log/results_summary.csv | PASS | /Users/osmancifci/Downloads/Graduation-Project-Dicle/outputs/results_v2_four_dataset_within_34feat_log/results_summary.csv |
| exists:outputs/results_v2_four_dataset_cross_34feat_capnorm_log/results_summary.csv | PASS | /Users/osmancifci/Downloads/Graduation-Project-Dicle/outputs/results_v2_four_dataset_cross_34feat_capnorm_log/results_summary.csv |
| matr:cell_counts | PASS | actual=(135, 129, 6), expected=(135, 129, 6) |
| hust:cell_counts | PASS | actual=(77, 77, 0), expected=(77, 77, 0) |
| sandia:cell_counts | PASS | actual=(61, 50, 11), expected=(61, 50, 11) |
| luh:cell_counts | PASS | actual=(108, 106, 2), expected=(108, 106, 2) |
| sandia_primary_subset | PASS | n_primary=61, soc_windows=['0-100'] |
| luh_audit_definition | PASS | ok=108, alignment_methods=['log_age_capacity'] |
| capnorm_shape_matches_raw | PASS | raw=(762, 41), capnorm=(762, 41) |
| capacity_normalized_flags | PASS | raw=[np.int64(0)], capnorm=[np.int64(1)] |
| capnorm_keeps_ids_and_labels | PASS | dataset/cell/window/q0/label/censor columns unchanged |
| split_completeness | PASS | missing=[], bad=[] |
| within_result_matrix_complete | PASS | rows=56, expected=56 |
| cross_result_matrix_complete | PASS | rows=168, expected=168 |
| cross_has_12_directions | PASS | directions=['hust_to_luh', 'hust_to_matr', 'hust_to_sandia', 'luh_to_hust', 'luh_to_matr', 'luh_to_sandia', 'matr_to_hust', 'matr_to_luh', 'matr_to_sandia', 'sandia_to_hust', 'sandia_to_luh', 'sandia_to_matr'] |
| within_n100_positive_best_r2 | PASS | hust_to_hust=0.340, luh_to_luh=0.769, matr_to_matr=0.575, sandia_to_sandia=0.940 |

## Dataset Counts
| dataset | cells | modeling_cells | censored_cells |
| --- | --- | --- | --- |
| hust | 77 | 77 | 0 |
| luh | 108 | 106 | 2 |
| matr | 135 | 129 | 6 |
| sandia | 61 | 50 | 11 |

## Best Within-Dataset Results at N=100
| experiment | model | MAE_mean | SMAPE_mean | R2_mean |
| --- | --- | --- | --- | --- |
| hust_to_hust | random_forest | 177.972 | 12.157 | 0.340 |
| luh_to_luh | gaussian_process | 115.813 | 18.440 | 0.769 |
| matr_to_matr | catboost | 171.662 | 23.671 | 0.575 |
| sandia_to_sandia | xgboost | 120.767 | 23.436 | 0.940 |

## Best Naive Cross-Dataset Results at N=100
| experiment | model | MAE_mean | SMAPE_mean | R2_mean |
| --- | --- | --- | --- | --- |
| hust_to_luh | pls | 352.754 | 104.975 | -0.373 |
| hust_to_matr | gaussian_process | 720.379 | 69.406 | -3.600 |
| hust_to_sandia | xgboost | 1193.096 | 124.730 | -0.296 |
| luh_to_hust | random_forest | 300.655 | 21.018 | -0.766 |
| luh_to_matr | stacking | 343.007 | 42.028 | -0.313 |
| luh_to_sandia | xgboost | 415.345 | 46.643 | 0.492 |
| matr_to_hust | gaussian_process | 763.008 | 67.094 | -7.937 |
| matr_to_luh | stacking | 332.796 | 64.884 | 0.012 |
| matr_to_sandia | catboost | 714.950 | 93.371 | 0.041 |
| sandia_to_hust | catboost | 342.133 | 25.006 | -1.431 |
| sandia_to_luh | pls | 185.042 | 35.347 | 0.477 |
| sandia_to_matr | gaussian_process | 361.407 | 50.155 | -0.799 |

Notes: Sandia primary is restricted to 0-100 SOC-window cells. Luh uses the 108 standard-cycling cells; all parsed through `log_age_capacity` alignment.

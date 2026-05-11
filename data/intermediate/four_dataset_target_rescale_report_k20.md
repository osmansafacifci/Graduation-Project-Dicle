# Four-Dataset Target Calibration Summary (k=20)

Source file: outputs/results_v2_four_dataset_target_rescale/results_summary.csv

## Naive-Best Model, Then Target Adapters
| experiment | source | target | naive_best_model | baseline_R2 | baseline_MAE | residual_R2 | residual_MAE | residual_delta_R2 | residual_delta_MAE | linear_R2 | linear_MAE | linear_delta_R2 | linear_delta_MAE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hust_to_luh | hust | luh | pls | -0.373 | 352.754 | 0.403 | 244.039 | 0.776 | -108.715 | 0.440 | 209.213 | 0.813 | -143.541 |
| hust_to_matr | hust | matr | gaussian_process | -3.600 | 720.379 | -0.052 | 282.390 | 3.549 | -437.988 | -1.168 | 305.079 | 2.432 | -415.300 |
| hust_to_sandia | hust | sandia | xgboost | -0.296 | 1193.096 | 0.108 | 759.894 | 0.404 | -433.202 | 0.185 | 732.640 | 0.481 | -460.456 |
| luh_to_hust | luh | hust | random_forest | -0.766 | 300.655 | -0.107 | 237.285 | 0.658 | -63.369 | -0.114 | 237.413 | 0.651 | -63.242 |
| luh_to_matr | luh | matr | stacking | -0.313 | 343.007 | -0.063 | 277.055 | 0.250 | -65.953 | -0.183 | 288.564 | 0.130 | -54.443 |
| luh_to_sandia | luh | sandia | xgboost | 0.492 | 415.345 | 0.499 | 551.092 | 0.007 | 135.747 | 0.784 | 329.232 | 0.291 | -86.113 |
| matr_to_hust | matr | hust | gaussian_process | -7.937 | 763.008 | -0.266 | 253.266 | 7.671 | -509.742 | -0.108 | 236.918 | 7.829 | -526.090 |
| matr_to_luh | matr | luh | stacking | 0.012 | 332.796 | 0.028 | 328.509 | 0.016 | -4.287 | 0.164 | 296.538 | 0.152 | -36.258 |
| matr_to_sandia | matr | sandia | catboost | 0.041 | 714.950 | -0.023 | 825.149 | -0.063 | 110.199 | 0.201 | 682.627 | 0.160 | -32.323 |
| sandia_to_hust | sandia | hust | catboost | -1.431 | 342.133 | -0.418 | 266.228 | 1.013 | -75.905 | -0.078 | 235.965 | 1.353 | -106.168 |
| sandia_to_luh | sandia | luh | pls | 0.477 | 185.042 | 0.592 | 197.730 | 0.115 | 12.687 | 0.716 | 160.614 | 0.239 | -24.428 |
| sandia_to_matr | sandia | matr | gaussian_process | -0.799 | 361.407 | -0.236 | 307.266 | 0.563 | -54.140 | -0.189 | 295.767 | 0.609 | -65.639 |

## Best Adapter/Model Per Direction
| experiment | source | target | adapter_type | model | baseline_R2 | adapted_R2 | delta_R2 | baseline_MAE | adapted_MAE | delta_MAE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hust_to_luh | hust | luh | linear | pls | -0.373 | 0.440 | 0.813 | 352.754 | 209.213 | -143.541 |
| hust_to_matr | hust | matr | residual_mean | random_forest | -4.548 | 0.052 | 4.600 | 794.761 | 264.908 | -529.853 |
| hust_to_sandia | hust | sandia | linear | xgboost | -0.296 | 0.185 | 0.481 | 1193.096 | 732.640 | -460.456 |
| luh_to_hust | luh | hust | linear | pls | -13309470266126.436 | -0.070 | 13309470266126.363 | 999998510.442 | 233.562 | -999998276.879 |
| luh_to_matr | luh | matr | residual_mean | gaussian_process | -1.076 | -0.051 | 1.024 | 389.067 | 282.309 | -106.758 |
| luh_to_sandia | luh | sandia | linear | elastic_net | -159122594809.130 | 0.853 | 159122594809.983 | 190018301.182 | 195.618 | -190018105.564 |
| matr_to_hust | matr | hust | linear | gaussian_process | -7.937 | -0.108 | 7.829 | 763.008 | 236.918 | -526.090 |
| matr_to_luh | matr | luh | linear | stacking | 0.012 | 0.164 | 0.152 | 332.796 | 296.538 | -36.258 |
| matr_to_sandia | matr | sandia | linear | catboost | 0.041 | 0.201 | 0.160 | 714.950 | 682.627 | -32.323 |
| sandia_to_hust | sandia | hust | linear | catboost | -1.431 | -0.078 | 1.353 | 342.133 | 235.965 | -106.168 |
| sandia_to_luh | sandia | luh | linear | catboost | 0.116 | 0.718 | 0.602 | 259.692 | 161.388 | -98.304 |
| sandia_to_matr | sandia | matr | linear | catboost | -3.781 | -0.091 | 3.690 | 646.331 | 285.383 | -360.948 |

Interpretation: the first table is the conservative audit table because it fixes the model selected by naive cross-dataset R2 before applying target calibration. The second table is useful as an upper envelope but should be described as adapter/model selection, not as a pre-registered protocol.

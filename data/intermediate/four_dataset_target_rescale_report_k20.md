# Four-Dataset Target Calibration Summary (k=20)

Source file: outputs/results_v2_four_dataset_target_rescale/results_summary.csv

## Naive-Best Model, Then Target Adapters
| experiment | source | target | naive_best_model | baseline_R2 | baseline_MAE | residual_R2 | residual_MAE | residual_delta_R2 | residual_delta_MAE | linear_R2 | linear_MAE | linear_delta_R2 | linear_delta_MAE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hust_to_luh | hust | luh | pls | -0.562 | 405.327 | 0.382 | 250.789 | 0.944 | -154.538 | 0.555 | 205.503 | 1.117 | -199.824 |
| hust_to_matr | hust | matr | gaussian_process | -3.600 | 720.361 | -0.052 | 282.363 | 3.548 | -437.998 | -1.183 | 304.885 | 2.417 | -415.476 |
| hust_to_sandia | hust | sandia | xgboost | -0.231 | 1163.071 | 0.139 | 743.910 | 0.371 | -419.161 | 0.263 | 697.569 | 0.494 | -465.501 |
| luh_to_hust | luh | hust | random_forest | -0.763 | 300.383 | -0.106 | 237.201 | 0.657 | -63.183 | -0.112 | 237.200 | 0.651 | -63.183 |
| luh_to_matr | luh | matr | stacking | -0.275 | 334.381 | -0.058 | 275.736 | 0.217 | -58.645 | -0.171 | 288.297 | 0.104 | -46.085 |
| luh_to_sandia | luh | sandia | xgboost | 0.499 | 414.473 | 0.503 | 548.975 | 0.004 | 134.502 | 0.786 | 324.888 | 0.288 | -89.585 |
| matr_to_hust | matr | hust | gaussian_process | -7.937 | 763.007 | -0.266 | 253.267 | 7.671 | -509.740 | -0.108 | 236.918 | 7.829 | -526.089 |
| matr_to_luh | matr | luh | stacking | 0.004 | 334.060 | 0.019 | 330.319 | 0.015 | -3.741 | 0.137 | 303.303 | 0.133 | -30.757 |
| matr_to_sandia | matr | sandia | xgboost | 0.046 | 730.672 | -0.011 | 818.862 | -0.057 | 88.190 | 0.029 | 697.953 | -0.016 | -32.719 |
| sandia_to_hust | sandia | hust | catboost | -2.027 | 384.653 | -0.611 | 279.997 | 1.416 | -104.656 | -0.094 | 236.886 | 1.933 | -147.768 |
| sandia_to_luh | sandia | luh | pls | 0.494 | 181.764 | 0.605 | 194.820 | 0.110 | 13.056 | 0.723 | 159.004 | 0.229 | -22.760 |
| sandia_to_matr | sandia | matr | gaussian_process | -0.802 | 361.478 | -0.233 | 306.867 | 0.569 | -54.610 | -0.190 | 295.804 | 0.612 | -65.674 |

## Best Adapter/Model Per Direction
| experiment | source | target | adapter_type | model | baseline_R2 | adapted_R2 | delta_R2 | baseline_MAE | adapted_MAE | delta_MAE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hust_to_luh | hust | luh | linear | pls | -0.562 | 0.555 | 1.117 | 405.327 | 205.503 | -199.824 |
| hust_to_matr | hust | matr | residual_mean | random_forest | -4.554 | 0.052 | 4.606 | 795.287 | 264.948 | -530.340 |
| hust_to_sandia | hust | sandia | linear | xgboost | -0.231 | 0.263 | 0.494 | 1163.071 | 697.569 | -465.501 |
| luh_to_hust | luh | hust | linear | elastic_net | -13276786456691.275 | -0.053 | 13276786456691.225 | 998007836.135 | 231.875 | -998007604.260 |
| luh_to_matr | luh | matr | residual_mean | gaussian_process | -1.076 | -0.051 | 1.024 | 389.067 | 282.309 | -106.758 |
| luh_to_sandia | luh | sandia | linear | elastic_net | -159104379814.571 | 0.853 | 159104379815.424 | 190001126.683 | 195.691 | -190000930.992 |
| matr_to_hust | matr | hust | linear | gaussian_process | -7.937 | -0.108 | 7.829 | 763.007 | 236.918 | -526.089 |
| matr_to_luh | matr | luh | linear | stacking | 0.004 | 0.137 | 0.133 | 334.060 | 303.303 | -30.757 |
| matr_to_sandia | matr | sandia | linear | stacking | 0.041 | 0.150 | 0.109 | 719.906 | 677.832 | -42.074 |
| sandia_to_hust | sandia | hust | linear | pls | -86.434 | -0.094 | 86.340 | 1246.460 | 232.687 | -1013.773 |
| sandia_to_luh | sandia | luh | linear | pls | 0.494 | 0.723 | 0.229 | 181.764 | 159.004 | -22.760 |
| sandia_to_matr | sandia | matr | linear | catboost | -3.964 | -0.095 | 3.868 | 650.153 | 286.823 | -363.330 |

Interpretation: the first table is the conservative audit table because it fixes the model selected by naive cross-dataset R2 before applying target calibration. The second table is useful as an upper envelope but should be described as adapter/model selection, not as a pre-registered protocol.

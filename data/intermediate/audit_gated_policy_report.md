# Prospective k=20 Audit-Gated Policy

Design: 20 target cells serve as both audit and adapter-fit cells; CP calibration and test cells are disjoint. Results are averaged over 200 random partitions per seed and five source-model seeds.

## Gate Decision Rates
| source | target | model | full_target_r_mean | audit_r_mean | audit_r_sd | gate_matches_full_fraction | gate_rate_linear | gate_rate_residual_mean | gate_rate_cp_interval_only |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hust | luh | pls | 0.772 | 0.788 | 0.067 | 1.0 | 1.0 | 0.0 | 0.0 |
| hust | matr | gaussian_process | -0.114 | -0.141 | 0.171 | 0.945 | 0.01 | 0.045 | 0.945 |
| hust | sandia | xgboost | 0.577 | 0.57 | 0.153 | 0.867 | 0.841 | 0.155 | 0.004 |
| luh | hust | random_forest | 0.015 | 0.004 | 0.248 | 0.677 | 0.058 | 0.295 | 0.647 |
| luh | matr | stacking | 0.107 | 0.071 | 0.301 | 0.489 | 0.144 | 0.354 | 0.502 |
| luh | sandia | xgboost | 0.913 | 0.912 | 0.054 | 1.0 | 1.0 | 0.0 | 0.0 |
| matr | hust | gaussian_process | -0.12 | -0.11 | 0.202 | 0.865 | 0.009 | 0.126 | 0.865 |
| matr | luh | stacking | 0.407 | 0.43 | 0.173 | 0.75 | 0.586 | 0.374 | 0.04 |
| matr | sandia | xgboost | 0.427 | 0.444 | 0.312 | 0.712 | 0.45 | 0.445 | 0.105 |
| sandia | hust | catboost | 0.187 | 0.181 | 0.209 | 0.567 | 0.138 | 0.525 | 0.337 |
| sandia | luh | pls | 0.862 | 0.865 | 0.042 | 1.0 | 1.0 | 0.0 | 0.0 |
| sandia | matr | gaussian_process | -0.006 | 0.004 | 0.218 | 0.676 | 0.047 | 0.277 | 0.676 |

## Audit-Gated Policy @ 90% CP
| source | target | model | R2_mean | MAE_mean | SMAPE_mean | coverage_mean | median_width_mean | point_available_fraction | gate_matches_full_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sandia | luh | pls | 0.717 | 160.533 | 42.05 | 0.951 | 1015.179 | 1.0 | 1.0 |
| luh | sandia | xgboost | 0.598 | 321.453 | 81.66 | 0.94 | 2381.067 | 1.0 | 1.0 |
| hust | luh | pls | 0.541 | 206.556 | 48.262 | 0.953 | 1299.976 | 1.0 | 1.0 |
| matr | luh | stacking | 0.109 | 307.763 | 63.913 | 0.952 | 1628.365 | 0.96 | 0.75 |
| luh | matr | stacking | -0.087 | 277.112 | 35.939 | 0.953 | 1916.363 | 0.498 | 0.489 |
| luh | hust | random_forest | -0.107 | 235.619 | 16.038 | 0.952 | 1164.222 | 0.353 | 0.677 |
| matr | hust | gaussian_process | -0.181 | 245.85 | 16.746 | 0.949 | 1179.86 | 0.135 | 0.865 |
| sandia | matr | gaussian_process | -0.339 | 313.991 | 39.346 | 0.953 | 1943.975 | 0.324 | 0.676 |
| hust | sandia | xgboost | -0.536 | 695.362 | 112.608 | 0.936 | 3981.242 | 0.996 | 0.867 |
| sandia | hust | catboost | -0.568 | 275.096 | 18.524 | 0.954 | 1405.93 | 0.663 | 0.567 |
| matr | sandia | xgboost | -0.973 | 698.223 | 94.88 | 0.936 | 4666.158 | 0.895 | 0.712 |
| hust | matr | gaussian_process | -3.3 | 329.827 | 38.863 | 0.953 | 1907.203 | 0.055 | 0.945 |

## Fixed-Policy Comparators @ 90% CP
| source | target | scenario | R2_mean | MAE_mean | SMAPE_mean | coverage_mean | median_width_mean | point_available_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hust | luh | always_linear | 0.541 | 206.556 | 48.262 | 0.953 | 1299.976 | 1.0 |
| hust | luh | always_residual_mean | 0.382 | 249.131 | 55.055 | 0.951 | 1387.361 | 1.0 |
| hust | luh | no_adaptation | -0.581 | 405.619 | 108.729 | 0.954 | 1873.994 | 1.0 |
| hust | luh | oracle_best_point | 0.557 | 206.295 | 49.2 | 0.953 | 1273.925 | 1.0 |
| hust | matr | always_linear | -1.373 | 300.079 | 38.44 | 0.953 | 2397.611 | 1.0 |
| hust | matr | always_residual_mean | -0.066 | 283.856 | 36.674 | 0.953 | 1850.83 | 1.0 |
| hust | matr | no_adaptation | -3.689 | 720.053 | 69.38 | 0.951 | 2308.327 | 1.0 |
| hust | matr | oracle_best_point | -0.064 | 283.467 | 36.656 | 0.954 | 1850.691 | 1.0 |
| hust | sandia | always_linear | -0.546 | 695.279 | 114.574 | 0.936 | 3960.146 | 1.0 |
| hust | sandia | always_residual_mean | -0.631 | 748.065 | 96.576 | 0.937 | 4367.081 | 1.0 |
| hust | sandia | no_adaptation | -3.032 | 1161.087 | 123.565 | 0.938 | 3204.826 | 1.0 |
| hust | sandia | oracle_best_point | -0.422 | 691.985 | 112.031 | 0.939 | 4030.9 | 1.0 |
| luh | hust | always_linear | -0.14 | 238.119 | 16.226 | 0.952 | 1170.343 | 1.0 |
| luh | hust | always_residual_mean | -0.108 | 237.192 | 16.172 | 0.952 | 1158.681 | 1.0 |
| luh | hust | no_adaptation | -0.807 | 300.465 | 20.995 | 0.954 | 1435.921 | 1.0 |
| luh | hust | oracle_best_point | -0.069 | 232.521 | 15.859 | 0.953 | 1145.974 | 1.0 |
| luh | matr | always_linear | -0.231 | 290.981 | 37.268 | 0.952 | 2050.405 | 1.0 |
| luh | matr | always_residual_mean | -0.074 | 278.056 | 36.096 | 0.954 | 1907.214 | 1.0 |
| luh | matr | no_adaptation | -0.286 | 334.022 | 41.117 | 0.955 | 1705.846 | 1.0 |
| luh | matr | oracle_best_point | -0.061 | 277.374 | 35.917 | 0.955 | 1962.917 | 1.0 |
| luh | sandia | always_linear | 0.598 | 321.453 | 81.66 | 0.94 | 2381.067 | 1.0 |
| luh | sandia | always_residual_mean | 0.112 | 552.574 | 78.605 | 0.935 | 3425.536 | 1.0 |
| luh | sandia | no_adaptation | 0.454 | 414.054 | 46.839 | 0.935 | 3849.252 | 1.0 |
| luh | sandia | oracle_best_point | 0.618 | 323.603 | 81.413 | 0.944 | 2428.888 | 1.0 |
| matr | hust | always_linear | -0.121 | 237.496 | 16.22 | 0.952 | 1171.27 | 1.0 |
| matr | hust | always_residual_mean | -0.283 | 254.903 | 17.366 | 0.949 | 1177.584 | 1.0 |
| matr | hust | no_adaptation | -8.264 | 763.463 | 67.118 | 0.952 | 2529.796 | 1.0 |
| matr | hust | oracle_best_point | -0.083 | 234.081 | 15.973 | 0.954 | 1153.939 | 1.0 |
| matr | luh | always_linear | 0.119 | 303.717 | 63.77 | 0.952 | 1622.965 | 1.0 |
| matr | luh | always_residual_mean | 0.011 | 329.096 | 64.801 | 0.952 | 1722.489 | 1.0 |
| matr | luh | no_adaptation | -0.001 | 333.596 | 64.959 | 0.953 | 1696.939 | 1.0 |
| matr | luh | oracle_best_point | 0.136 | 301.999 | 63.491 | 0.954 | 1616.357 | 1.0 |
| matr | sandia | always_linear | -0.856 | 703.021 | 94.186 | 0.936 | 4634.1 | 1.0 |
| matr | sandia | always_residual_mean | -0.914 | 822.628 | 103.979 | 0.936 | 4714.293 | 1.0 |
| matr | sandia | no_adaptation | -0.295 | 729.103 | 92.281 | 0.936 | 4942.781 | 1.0 |
| matr | sandia | oracle_best_point | -0.501 | 692.927 | 93.951 | 0.942 | 4541.665 | 1.0 |
| sandia | hust | always_linear | -0.116 | 237.592 | 16.214 | 0.954 | 1148.732 | 1.0 |
| sandia | hust | always_residual_mean | -0.618 | 279.879 | 18.828 | 0.953 | 1443.662 | 1.0 |
| sandia | hust | no_adaptation | -2.141 | 385.638 | 28.761 | 0.954 | 1903.536 | 1.0 |
| sandia | hust | oracle_best_point | -0.105 | 236.597 | 16.143 | 0.955 | 1149.3 | 1.0 |
| sandia | luh | always_linear | 0.717 | 160.533 | 42.05 | 0.951 | 1015.179 | 1.0 |
| sandia | luh | always_residual_mean | 0.599 | 196.231 | 40.239 | 0.952 | 1207.385 | 1.0 |
| sandia | luh | no_adaptation | 0.494 | 181.817 | 34.134 | 0.953 | 1443.733 | 1.0 |
| sandia | luh | oracle_best_point | 0.718 | 160.786 | 41.906 | 0.952 | 1015.984 | 1.0 |
| sandia | matr | always_linear | -0.207 | 296.873 | 38.004 | 0.954 | 1939.155 | 1.0 |
| sandia | matr | always_residual_mean | -0.242 | 306.473 | 38.851 | 0.953 | 1931.519 | 1.0 |
| sandia | matr | no_adaptation | -0.817 | 361.595 | 50.217 | 0.953 | 2362.944 | 1.0 |
| sandia | matr | oracle_best_point | -0.118 | 291.051 | 37.442 | 0.955 | 1889.172 | 1.0 |

Interpretation note: CP-only gate draws use a residual-mean centre for conformal intervals but mark point-prediction metrics as unavailable.

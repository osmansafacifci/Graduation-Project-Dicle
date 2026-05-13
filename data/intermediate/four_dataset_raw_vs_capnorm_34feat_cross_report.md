# Four-Dataset Raw vs Capacity-Normalized Cross Ablation

- Raw summary: `outputs/results_v2_four_dataset_cross_34feat_raw_log/results_summary.csv`
- Capnorm summary: `outputs/results_v2_four_dataset_cross_34feat_capnorm_log/results_summary.csv`
- Scope: 34 features, log-target, all seven models, ordered source->target pairs.

## Aggregate Counts

- Model/window rows compared: 168
- Capnorm improves MAE rows: 105/168
- Capnorm improves R2 rows: 102/168
- Best-direction/window rows: 24
- Capnorm improves best-model MAE: 14/24
- Capnorm improves best-model R2: 15/24

## Best Model by Direction and Window

| Direction | N | Raw Best | Raw MAE | Raw R2 | Capnorm Best | Capnorm MAE | Capnorm R2 | Delta MAE | Delta R2 |
|---|---:|---|---:|---:|---|---:|---:|---:|---:|
| `hust_to_luh` | 50 | gaussian_process | 910.3 | -4.941 | elastic_net | 508.3 | -1.442 | -401.9 | +3.500 |
| `hust_to_luh` | 100 | gaussian_process | 910.3 | -4.941 | pls | 405.3 | -0.562 | -504.9 | +4.379 |
| `hust_to_matr` | 50 | random_forest | 574.3 | -2.062 | gaussian_process | 720.4 | -3.601 | +146.1 | -1.539 |
| `hust_to_matr` | 100 | gaussian_process | 569.2 | -2.052 | gaussian_process | 720.4 | -3.600 | +151.1 | -1.548 |
| `hust_to_sandia` | 50 | gaussian_process | 1260.8 | -0.420 | gaussian_process | 1260.5 | -0.419 | -0.4 | +0.001 |
| `hust_to_sandia` | 100 | gaussian_process | 1300.9 | -0.533 | xgboost | 1163.1 | -0.231 | -137.8 | +0.301 |
| `luh_to_hust` | 50 | catboost | 642.2 | -5.461 | random_forest | 351.5 | -1.568 | -290.7 | +3.893 |
| `luh_to_hust` | 100 | xgboost | 532.7 | -3.857 | random_forest | 300.4 | -0.763 | -232.3 | +3.093 |
| `luh_to_matr` | 50 | stacking | 283.2 | -0.104 | stacking | 288.1 | -0.172 | +4.9 | -0.068 |
| `luh_to_matr` | 100 | stacking | 276.9 | 0.007 | stacking | 334.4 | -0.275 | +57.5 | -0.282 |
| `luh_to_sandia` | 50 | catboost | 502.7 | 0.227 | xgboost | 531.3 | 0.267 | +28.5 | +0.040 |
| `luh_to_sandia` | 100 | xgboost | 469.3 | 0.320 | xgboost | 414.5 | 0.499 | -54.8 | +0.178 |
| `matr_to_hust` | 50 | gaussian_process | 781.7 | -8.135 | gaussian_process | 783.1 | -8.158 | +1.4 | -0.024 |
| `matr_to_hust` | 100 | gaussian_process | 781.2 | -8.125 | gaussian_process | 763.0 | -7.937 | -18.1 | +0.188 |
| `matr_to_luh` | 50 | catboost | 330.8 | 0.021 | catboost | 326.9 | 0.048 | -3.8 | +0.027 |
| `matr_to_luh` | 100 | stacking | 336.8 | 0.021 | stacking | 334.1 | 0.004 | -2.7 | -0.017 |
| `matr_to_sandia` | 50 | pls | 463.7 | 0.558 | gaussian_process | 807.8 | -0.003 | +344.1 | -0.561 |
| `matr_to_sandia` | 100 | catboost | 737.7 | 0.034 | xgboost | 730.7 | 0.046 | -7.0 | +0.012 |
| `sandia_to_hust` | 50 | catboost | 544.0 | -4.656 | catboost | 513.8 | -4.151 | -30.1 | +0.505 |
| `sandia_to_hust` | 100 | catboost | 369.2 | -1.811 | catboost | 384.7 | -2.027 | +15.4 | -0.215 |
| `sandia_to_luh` | 50 | pls | 332.6 | -0.352 | pls | 327.2 | -0.300 | -5.4 | +0.052 |
| `sandia_to_luh` | 100 | catboost | 249.8 | 0.129 | pls | 181.8 | 0.494 | -68.0 | +0.365 |
| `sandia_to_matr` | 50 | gaussian_process | 401.2 | -1.079 | gaussian_process | 410.4 | -1.227 | +9.3 | -0.148 |
| `sandia_to_matr` | 100 | gaussian_process | 360.1 | -0.923 | gaussian_process | 361.5 | -0.802 | +1.4 | +0.121 |

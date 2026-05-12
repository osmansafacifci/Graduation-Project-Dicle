# Four-Dataset Raw vs Capacity-Normalized Cross Ablation

- Raw summary: `outputs/results_v2_four_dataset_cross_34feat_raw_log/results_summary.csv`
- Capnorm summary: `outputs/results_v2_four_dataset_cross_34feat_capnorm_log/results_summary.csv`
- Scope: 34 features, log-target, all seven models, ordered source->target pairs.

## Aggregate Counts

- Model/window rows compared: 168
- Capnorm improves MAE rows: 103/168
- Capnorm improves R2 rows: 99/168
- Best-direction/window rows: 24
- Capnorm improves best-model MAE: 14/24
- Capnorm improves best-model R2: 15/24

## Best Model by Direction and Window

| Direction | N | Raw Best | Raw MAE | Raw R2 | Capnorm Best | Capnorm MAE | Capnorm R2 | Delta MAE | Delta R2 |
|---|---:|---|---:|---:|---|---:|---:|---:|---:|
| `hust_to_luh` | 50 | gaussian_process | 910.3 | -4.941 | elastic_net | 508.3 | -1.442 | -401.9 | +3.500 |
| `hust_to_luh` | 100 | gaussian_process | 910.3 | -4.941 | pls | 352.8 | -0.373 | -557.5 | +4.568 |
| `hust_to_matr` | 50 | random_forest | 574.3 | -2.062 | gaussian_process | 720.4 | -3.601 | +146.1 | -1.539 |
| `hust_to_matr` | 100 | gaussian_process | 569.2 | -2.052 | gaussian_process | 720.4 | -3.600 | +151.2 | -1.549 |
| `hust_to_sandia` | 50 | gaussian_process | 1260.8 | -0.420 | gaussian_process | 1260.5 | -0.419 | -0.4 | +0.001 |
| `hust_to_sandia` | 100 | gaussian_process | 1300.9 | -0.533 | xgboost | 1193.1 | -0.296 | -107.8 | +0.237 |
| `luh_to_hust` | 50 | catboost | 642.2 | -5.461 | random_forest | 354.3 | -1.602 | -287.9 | +3.859 |
| `luh_to_hust` | 100 | xgboost | 532.7 | -3.857 | random_forest | 300.7 | -0.766 | -232.0 | +3.091 |
| `luh_to_matr` | 50 | stacking | 283.2 | -0.104 | stacking | 287.8 | -0.173 | +4.6 | -0.069 |
| `luh_to_matr` | 100 | stacking | 276.9 | 0.007 | stacking | 343.0 | -0.313 | +66.1 | -0.320 |
| `luh_to_sandia` | 50 | catboost | 502.7 | 0.227 | xgboost | 531.7 | 0.266 | +28.9 | +0.039 |
| `luh_to_sandia` | 100 | xgboost | 469.3 | 0.320 | xgboost | 415.3 | 0.492 | -53.9 | +0.172 |
| `matr_to_hust` | 50 | gaussian_process | 781.7 | -8.135 | gaussian_process | 783.1 | -8.158 | +1.4 | -0.024 |
| `matr_to_hust` | 100 | gaussian_process | 781.2 | -8.125 | gaussian_process | 763.0 | -7.937 | -18.1 | +0.188 |
| `matr_to_luh` | 50 | catboost | 330.8 | 0.021 | catboost | 330.4 | 0.027 | -0.4 | +0.006 |
| `matr_to_luh` | 100 | stacking | 336.8 | 0.021 | stacking | 332.8 | 0.012 | -4.0 | -0.009 |
| `matr_to_sandia` | 50 | pls | 463.7 | 0.558 | gaussian_process | 807.8 | -0.003 | +344.1 | -0.561 |
| `matr_to_sandia` | 100 | catboost | 737.7 | 0.034 | catboost | 715.0 | 0.041 | -22.7 | +0.007 |
| `sandia_to_hust` | 50 | catboost | 544.0 | -4.656 | catboost | 622.9 | -5.799 | +79.0 | -1.143 |
| `sandia_to_hust` | 100 | catboost | 369.2 | -1.811 | catboost | 342.1 | -1.431 | -27.1 | +0.380 |
| `sandia_to_luh` | 50 | pls | 332.6 | -0.352 | pls | 327.5 | -0.303 | -5.1 | +0.050 |
| `sandia_to_luh` | 100 | catboost | 249.8 | 0.129 | pls | 185.0 | 0.477 | -64.7 | +0.348 |
| `sandia_to_matr` | 50 | gaussian_process | 401.2 | -1.079 | gaussian_process | 410.4 | -1.227 | +9.3 | -0.148 |
| `sandia_to_matr` | 100 | gaussian_process | 360.1 | -0.923 | gaussian_process | 361.4 | -0.799 | +1.3 | +0.124 |

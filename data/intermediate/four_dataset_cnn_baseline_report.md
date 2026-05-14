# Four-Dataset 1D-CNN Baseline

Model: dependency-light NumPy Conv1D -> ReLU -> temporal average pooling + global mean/max pooling -> dense ReLU.
Input: cycles 2..N, channels `[Q_discharge/q0, diff(Q_discharge/q0)]`; target: standardized log(cycle_life).

## Within-Dataset N=100

| target | MAE_mean | SMAPE_mean | R2_mean | R2_pooled_ci95_lower | R2_pooled_ci95_upper |
| ------ | -------- | ---------- | ------- | -------------------- | -------------------- |
| hust   | 265.024  | 18.129     | -0.407  | -0.789               | -0.127               |
| luh    | 91.416   | 13.070     | 0.814   | 0.694                | 0.895                |
| matr   | 150.598  | 20.105     | 0.613   | 0.459                | 0.748                |
| sandia | 222.048  | 35.895     | 0.732   | 0.336                | 0.950                |

## Naive Cross-Dataset N=100

| experiment     | MAE_mean | SMAPE_mean | R2_mean |
| -------------- | -------- | ---------- | ------- |
| luh_to_hust    | 406.205  | 33.989     | -2.981  |
| matr_to_hust   | 758.380  | 70.476     | -9.153  |
| sandia_to_hust | 1778.362 | 74.932     | -43.478 |
| sandia_to_luh  | 241.941  | 47.674     | 0.286   |
| hust_to_luh    | 688.494  | 88.010     | -3.035  |
| matr_to_luh    | 722.292  | 81.345     | -4.600  |
| luh_to_matr    | 694.005  | 66.197     | -3.865  |
| hust_to_matr   | 737.785  | 70.171     | -3.890  |
| sandia_to_matr | 2538.896 | 125.624    | -50.425 |
| luh_to_sandia  | 493.866  | 62.772     | 0.352   |
| hust_to_sandia | 1152.029 | 121.634    | -0.320  |
| matr_to_sandia | 1066.998 | 103.994    | -0.408  |

Interpretation: use this as a deep-learning baseline only. The main paper claim remains the transfer-regime and calibration protocol, not CNN architecture novelty.
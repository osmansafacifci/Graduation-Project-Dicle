# Four-Dataset Mamba Library Backbone (PyTorch)

Input: cycles 2..N, channels `[Q_discharge/q0, diff(Q_discharge/q0)]`.
Purpose: official mamba-ssm deep sequence backbone sensitivity check using the same splits, metrics, clipping, and checkpointing as the CNN/Transformer runners.
Run details: hyperparameter grid, epoch caps, and clipping policy are recorded in the output `results_config.json`.

## Within-Dataset N=100

| target | MAE_mean | SMAPE_mean | R2_mean | R2_cluster_ci95_lower | R2_cluster_ci95_upper |
| ------ | -------- | ---------- | ------- | --------------------- | --------------------- |
| hust   | 246.179  | 16.947     | -0.170  | -0.547                | 0.041                 |
| luh    | 123.386  | 19.174     | 0.736   | 0.519                 | 0.857                 |
| matr   | 221.312  | 28.572     | 0.024   | -0.617                | 0.453                 |
| sandia | 165.344  | 26.912     | 0.787   | 0.383                 | 0.983                 |

## Naive Cross-Dataset N=100

| experiment     | MAE_mean | SMAPE_mean | R2_mean |
| -------------- | -------- | ---------- | ------- |
| hust_to_luh    | 869.586  | 97.386     | -4.800  |
| hust_to_matr   | 742.568  | 72.183     | -3.997  |
| hust_to_sandia | 1267.901 | 132.601    | -0.513  |
| luh_to_hust    | 901.902  | 86.574     | -11.327 |
| luh_to_matr    | 366.164  | 48.734     | -0.896  |
| luh_to_sandia  | 466.354  | 52.315     | 0.266   |
| matr_to_hust   | 766.132  | 69.995     | -8.318  |
| matr_to_luh    | 359.794  | 79.562     | -0.378  |
| matr_to_sandia | 676.971  | 98.750     | -0.059  |
| sandia_to_hust | 1354.909 | 61.519     | -27.793 |
| sandia_to_luh  | 257.649  | 47.116     | 0.183   |
| sandia_to_matr | 2131.087 | 117.163    | -36.962 |

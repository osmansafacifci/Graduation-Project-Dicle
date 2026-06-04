# Four-Dataset Simple Transformer Baseline (PyTorch)

Model: Linear input projection + sinusoidal position encoding + TransformerEncoderLayer x4 + global mean/max pooling.
Input: cycles 2..N, channels `[Q_discharge/q0, diff(Q_discharge/q0)]`.
Purpose: bounded second deep-backbone check; not a new headline architecture or domain-adaptation method.

## Within-Dataset N=100

| target | MAE_mean | SMAPE_mean | R2_mean | R2_cluster_ci95_lower | R2_cluster_ci95_upper |
| ------ | -------- | ---------- | ------- | --------------------- | --------------------- |
| hust   | 243.357  | 16.745     | -0.185  | -0.592                | 0.110                 |
| luh    | 110.994  | 17.500     | 0.792   | 0.661                 | 0.877                 |
| matr   | 238.365  | 32.704     | 0.195   | -0.165                | 0.408                 |
| sandia | 189.375  | 34.587     | 0.853   | 0.754                 | 0.956                 |

## Naive Cross-Dataset N=100 (best per target by R2)

| experiment     | MAE_mean | SMAPE_mean | R2_mean |
| -------------- | -------- | ---------- | ------- |
| luh_to_hust    | 352.766  | 27.359     | -1.859  |
| matr_to_hust   | 746.935  | 65.723     | -7.734  |
| sandia_to_hust | 1359.743 | 61.525     | -28.393 |
| matr_to_luh    | 379.277  | 72.964     | -0.301  |
| sandia_to_luh  | 450.361  | 72.906     | -1.748  |
| hust_to_luh    | 684.462  | 88.449     | -2.722  |
| luh_to_matr    | 653.274  | 63.640     | -3.199  |
| hust_to_matr   | 700.887  | 67.526     | -3.595  |
| sandia_to_matr | 2028.411 | 113.874    | -34.263 |
| luh_to_sandia  | 667.126  | 82.264     | 0.256   |
| matr_to_sandia | 727.418  | 94.506     | 0.067   |
| hust_to_sandia | 1334.510 | 129.886    | -0.651  |

Interpretation: use this as SI-level defensive evidence that the rank-signal regime taxonomy is not unique to a convolutional sequence model. The quick grid is intentionally small to avoid architecture fishing.
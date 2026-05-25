# Four-Dataset Simple Transformer Baseline (PyTorch)

Model: Linear input projection + sinusoidal position encoding + TransformerEncoderLayer x4 + global mean/max pooling.
Input: cycles 2..N, channels `[Q_discharge/q0, diff(Q_discharge/q0)]`.
Purpose: bounded second deep-backbone check; not a new headline architecture or domain-adaptation method.

## Within-Dataset N=100

| target | MAE_mean | SMAPE_mean | R2_mean | R2_cluster_ci95_lower | R2_cluster_ci95_upper |
| ------ | -------- | ---------- | ------- | --------------------- | --------------------- |
| hust   | 238.874  | 16.385     | -0.115  | -0.429                | 0.084                 |
| luh    | 133.117  | 22.246     | 0.742   | 0.605                 | 0.836                 |
| matr   | 278.802  | 38.518     | -0.000  | -0.168                | 0.073                 |
| sandia | 159.311  | 34.144     | 0.927   | 0.829                 | 0.971                 |

## Naive Cross-Dataset N=100 (best per target by R2)

| experiment     | MAE_mean | SMAPE_mean | R2_mean |
| -------------- | -------- | ---------- | ------- |
| luh_to_hust    | 265.295  | 18.323     | -0.493  |
| matr_to_hust   | 816.702  | 74.007     | -8.907  |
| sandia_to_hust | 1110.852 | 53.480     | -19.375 |
| matr_to_luh    | 362.062  | 70.574     | -0.188  |
| sandia_to_luh  | 479.199  | 75.372     | -2.464  |
| hust_to_luh    | 753.229  | 92.878     | -3.292  |
| luh_to_matr    | 733.674  | 69.485     | -3.912  |
| hust_to_matr   | 759.572  | 71.518     | -4.123  |
| sandia_to_matr | 1765.100 | 107.126    | -26.303 |
| luh_to_sandia  | 692.841  | 75.669     | 0.172   |
| matr_to_sandia | 720.159  | 92.826     | -0.014  |
| hust_to_sandia | 1203.330 | 126.799    | -0.328  |

Interpretation: use this as SI-level defensive evidence that the rank-signal regime taxonomy is not unique to a convolutional sequence model. The quick grid is intentionally small to avoid architecture fishing.
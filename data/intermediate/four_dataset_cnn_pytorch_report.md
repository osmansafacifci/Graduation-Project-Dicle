# Four-Dataset 1D-CNN Baseline (PyTorch)

Model: PyTorch Conv1D x2 -> ReLU -> global mean/max pool -> dense + dropout -> scalar log-life.
Input: cycles 2..N, channels `[Q_discharge/q0, diff(Q_discharge/q0)]`.
Inner 5-fold CV over (filters, learning rate). Cluster bootstrap by cell_id for pooled CIs.

## Within-Dataset N=100

| target | MAE_mean | SMAPE_mean | R2_mean | R2_cluster_ci95_lower | R2_cluster_ci95_upper |
| ------ | -------- | ---------- | ------- | --------------------- | --------------------- |
| hust   | 246.898  | 16.916     | -0.174  | -0.440                | -0.028                |
| luh    | 117.598  | 18.835     | 0.761   | 0.624                 | 0.857                 |
| matr   | 196.321  | 25.019     | 0.305   | -0.250                | 0.587                 |
| sandia | 164.650  | 23.915     | 0.881   | 0.754                 | 0.950                 |

## Naive Cross-Dataset N=100 (best per target by R2)

| experiment     | MAE_mean | SMAPE_mean | R2_mean |
| -------------- | -------- | ---------- | ------- |
| luh_to_hust    | 238.346  | 16.205     | -0.167  |
| matr_to_hust   | 642.677  | 55.202     | -6.209  |
| sandia_to_hust | 1624.847 | 70.667     | -36.816 |
| sandia_to_luh  | 228.606  | 44.294     | 0.343   |
| matr_to_luh    | 464.861  | 66.213     | -1.831  |
| hust_to_luh    | 632.044  | 87.075     | -2.027  |
| hust_to_matr   | 728.666  | 69.600     | -3.785  |
| luh_to_matr    | 739.590  | 69.281     | -4.186  |
| sandia_to_matr | 2474.567 | 124.321    | -47.806 |
| luh_to_sandia  | 587.198  | 64.206     | 0.314   |
| matr_to_sandia | 800.609  | 95.292     | 0.002   |
| hust_to_sandia | 1154.978 | 122.795    | -0.289  |

Interpretation: this is the PyTorch deep-learning baseline. Within-dataset HP search
via inner CV mirrors the protocol used for XGBoost/CatBoost in the classical lineup.
Cross-dataset failure pattern is the relevant axis for the rank-signal regime claim.
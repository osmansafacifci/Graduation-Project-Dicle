# K-Shot Scaling Figure Summary

Inputs:
- CP k sweep: `outputs/results_v2_four_dataset_conformal/paper_cp_k_sweep.csv`
- LODO k sweep: `data/intermediate/four_dataset_lodo_source_expert_k_sweep.csv`

Main 90% CP result at k=20:

| Protocol | Mean coverage | Median width | Finite interval fraction |
|---|---:|---:|---:|
| Linear-adapted CP | 0.903 | 1372 | 1.000 |
| Residual-adapted CP | 0.911 | 1466 | 1.000 |
| Target CP | 0.909 | 2868 | 1.000 |

LODO MAE reduction from k=5 to k=20:

| Target | MAE at k=5 | MAE at k=20 | Reduction |
|---|---:|---:|---:|
| Sandia | 470.3 | 277.5 | 192.7 |
| Luh/KIT | 207.6 | 178.2 | 29.4 |
| MATR | 299.6 | 274.7 | 24.9 |
| HUST | 257.5 | 236.0 | 21.5 |

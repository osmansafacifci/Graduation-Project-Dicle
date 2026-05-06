# Paper CP Summary

Primary policy: 90% MAPIE split CP; target rows use k_target=20; adapted rows use residual-mean k_adapter=20.

| Scenario                         | Direction    | Model         | Coverage | Median width | MAE   | sMAPE | R2      | Runs |
| -------------------------------- | ------------ | ------------- | -------- | ------------ | ----- | ----- | ------- | ---- |
| Within-dataset CP                | HUST         | catboost      | 0.933    | 918.8        | 181.8 | 12.3  | 0.284   | 5    |
| Within-dataset CP                | HUST         | random_forest | 0.967    | 941.1        | 178.0 | 12.2  | 0.340   | 5    |
| Within-dataset CP                | MATR         | catboost      | 0.920    | 1118.6       | 171.7 | 23.7  | 0.575   | 5    |
| Within-dataset CP                | MATR         | random_forest | 0.860    | 954.6        | 183.2 | 25.8  | 0.520   | 5    |
| Naive source-calibrated cross CP | HUST -> MATR | catboost      | 0.270    | 918.8        | 670.9 | 66.4  | -3.065  | 5    |
| Naive source-calibrated cross CP | HUST -> MATR | random_forest | 0.305    | 941.1        | 610.2 | 62.6  | -2.403  | 5    |
| Naive source-calibrated cross CP | MATR -> HUST | catboost      | 0.190    | 1118.6       | 861.0 | 79.5  | -9.995  | 5    |
| Naive source-calibrated cross CP | MATR -> HUST | random_forest | 0.148    | 954.6        | 891.7 | 84.3  | -10.929 | 5    |
| Target-domain CP, no adapter     | HUST -> MATR | catboost      | 0.903    | 2074.8       | 670.3 | 66.3  | -3.104  | 100  |
| Target-domain CP, no adapter     | HUST -> MATR | random_forest | 0.908    | 1904.4       | 609.7 | 62.6  | -2.374  | 100  |
| Target-domain CP, no adapter     | MATR -> HUST | catboost      | 0.902    | 2514.0       | 859.0 | 79.3  | -9.912  | 100  |
| Target-domain CP, no adapter     | MATR -> HUST | random_forest | 0.885    | 2568.1       | 895.7 | 84.7  | -11.089 | 100  |
| Residual-mean target-adapted CP  | HUST -> MATR | catboost      | 0.905    | 1302.3       | 276.4 | 35.7  | -0.020  | 100  |
| Residual-mean target-adapted CP  | HUST -> MATR | random_forest | 0.909    | 1281.4       | 276.6 | 35.9  | -0.027  | 100  |
| Residual-mean target-adapted CP  | MATR -> HUST | catboost      | 0.908    | 998.7        | 246.4 | 16.8  | -0.194  | 100  |
| Residual-mean target-adapted CP  | MATR -> HUST | random_forest | 0.907    | 1057.2       | 269.5 | 18.4  | -0.407  | 100  |

## Adapter Improvement Over Target-Only CP

| direction    | model         | coverage_no_adapter | coverage_adapted | median_width_no_adapter | median_width_adapted | median_width_reduction_pct | MAE_no_adapter | MAE_adapted | MAE_reduction_pct | R2_no_adapter | R2_adapted | winkler_no_adapter | winkler_adapted | winkler_reduction_pct |
| ------------ | ------------- | ------------------- | ---------------- | ----------------------- | -------------------- | -------------------------- | -------------- | ----------- | ----------------- | ------------- | ---------- | ------------------ | --------------- | --------------------- |
| HUST -> MATR | catboost      | 0.903               | 0.905            | 2074.8                  | 1302.3               | 37.2                       | 670.3          | 276.4       | 58.8              | -3.104        | -0.020     | 2175.5             | 1742.9          | 19.9                  |
| HUST -> MATR | random_forest | 0.908               | 0.909            | 1904.4                  | 1281.4               | 32.7                       | 609.7          | 276.6       | 54.6              | -2.374        | -0.027     | 1994.5             | 1690.3          | 15.3                  |
| MATR -> HUST | catboost      | 0.902               | 0.908            | 2514.0                  | 998.7                | 60.3                       | 859.0          | 246.4       | 71.3              | -9.912        | -0.194     | 2715.8             | 1171.5          | 56.9                  |
| MATR -> HUST | random_forest | 0.885               | 0.907            | 2568.1                  | 1057.2               | 58.8                       | 895.7          | 269.5       | 69.9              | -11.089       | -0.407     | 2806.0             | 1238.6          | 55.9                  |

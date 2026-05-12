# Four-Dataset LODO Source-Expert Transfer (k=20)

Best protocol per held-out target at N=100:

| Target | Protocol | Model | Adapter | k | MAE | sMAPE | R2 | Notes |
|---|---|---|---|---:|---:|---:|---:|---|
| hust | pooled_erm_kshot | elastic_net | linear | 20 | 237.1 | 16.16 | -0.115 |  |
| luh | pooled_erm_kshot | catboost | linear | 20 | 177.6 | 49.98 | 0.667 |  |
| matr | source_expert_convex | source_primary_experts | residual_mean | 20 | 278.0 | 35.92 | -0.018 | w_hust=0.24; w_sandia=0.07; w_luh=0.69 |
| sandia | pooled_erm_kshot | stacking | linear | 20 | 272.5 | 69.10 | 0.862 |  |

## Best Protocol by Target-Calibration Size

| Target | k | Protocol | Model | Adapter | MAE | R2 |
|---|---:|---|---|---|---:|---:|
| hust | 5 | pooled_erm_kshot | elastic_net | residual_mean | 263.6 | -0.357 |
| hust | 10 | pooled_erm_kshot | elastic_net | residual_mean | 253.0 | -0.256 |
| hust | 15 | pooled_erm_kshot | elastic_net | linear | 239.6 | -0.153 |
| hust | 20 | pooled_erm_kshot | elastic_net | linear | 237.1 | -0.115 |
| luh | 5 | pooled_erm_kshot | catboost | linear | 207.3 | 0.535 |
| luh | 10 | pooled_erm_kshot | catboost | linear | 185.8 | 0.631 |
| luh | 15 | pooled_erm_kshot | elastic_net | linear | 187.0 | 0.646 |
| luh | 20 | pooled_erm_kshot | catboost | linear | 177.6 | 0.667 |
| matr | 5 | source_expert_convex | source_primary_experts | residual_mean | 299.0 | -0.189 |
| matr | 10 | source_expert_convex | source_primary_experts | none | 285.2 | -0.076 |
| matr | 15 | source_expert_convex | source_primary_experts | residual_mean | 279.9 | -0.046 |
| matr | 20 | source_expert_convex | source_primary_experts | residual_mean | 278.0 | -0.018 |
| sandia | 5 | pooled_erm_kshot | gaussian_process | none | 470.5 | 0.628 |
| sandia | 10 | pooled_erm_kshot | catboost | linear | 330.6 | 0.733 |
| sandia | 15 | pooled_erm_kshot | stacking | linear | 294.0 | 0.814 |
| sandia | 20 | pooled_erm_kshot | stacking | linear | 272.5 | 0.862 |

Protocol counts in the k-report candidate set:

## hust
- pooled_erm_kshot / elastic_net / linear k=20: R2=-0.115, MAE=237.1
- source_model_select / source_model_experts / linear k=20: R2=-0.118, MAE=237.1
- source_expert_select / source_primary_experts / linear k=20: R2=-0.118, MAE=236.8

## luh
- pooled_erm_kshot / catboost / linear k=20: R2=0.667, MAE=177.6
- pooled_erm_kshot / elastic_net / linear k=20: R2=0.665, MAE=183.8
- source_model_select / source_model_experts / linear k=20: R2=0.653, MAE=180.3

## matr
- source_expert_convex / source_primary_experts / residual_mean k=20: R2=-0.018, MAE=278.0
- source_expert_convex / source_primary_experts / none k=20: R2=-0.020, MAE=279.2
- pooled_erm_kshot / xgboost / linear k=20: R2=-0.028, MAE=277.8

## sandia
- pooled_erm_kshot / stacking / linear k=20: R2=0.862, MAE=272.5
- pooled_erm_kshot / catboost / linear k=20: R2=0.851, MAE=283.2
- pooled_erm_kshot / random_forest / linear k=20: R2=0.827, MAE=309.4


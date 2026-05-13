# Four-Dataset LODO Source-Expert Transfer (k=20)

Best protocol per held-out target at N=100:

| Target | Protocol | Model | Adapter | k | MAE | sMAPE | R2 | Notes |
|---|---|---|---|---:|---:|---:|---:|---|
| hust | source_model_select | source_model_experts | linear | 20 | 236.0 | 16.05 | -0.110 | selected=luh; model=xgboost; w_sandia=0.34; w_luh=0.66 |
| luh | source_model_select | source_model_experts | linear | 20 | 178.2 | 45.02 | 0.660 | selected=sandia; model=pls; w_matr=0.15; w_hust=0.08; w_sandia=0.77 |
| matr | pooled_erm_kshot | xgboost | linear | 20 | 274.7 | 35.62 | -0.003 |  |
| sandia | pooled_erm_kshot | stacking | linear | 20 | 277.5 | 68.99 | 0.855 |  |

## Best Protocol by Target-Calibration Size

| Target | k | Protocol | Model | Adapter | MAE | R2 |
|---|---:|---|---|---|---:|---:|
| hust | 5 | pooled_erm_kshot | elastic_net | residual_mean | 257.5 | -0.289 |
| hust | 10 | pooled_erm_kshot | elastic_net | residual_mean | 246.7 | -0.186 |
| hust | 15 | pooled_erm_kshot | elastic_net | linear | 239.0 | -0.149 |
| hust | 20 | source_model_select | source_model_experts | linear | 236.0 | -0.110 |
| luh | 5 | pooled_erm_kshot | stacking | linear | 207.6 | 0.541 |
| luh | 10 | source_model_select | source_model_experts | linear | 183.9 | 0.621 |
| luh | 15 | source_model_select | source_model_experts | linear | 182.0 | 0.640 |
| luh | 20 | source_model_select | source_model_experts | linear | 178.2 | 0.660 |
| matr | 5 | source_expert_convex | source_primary_experts | residual_mean | 299.6 | -0.196 |
| matr | 10 | source_expert_convex | source_primary_experts | none | 285.4 | -0.079 |
| matr | 15 | source_expert_convex | source_primary_experts | residual_mean | 280.0 | -0.047 |
| matr | 20 | pooled_erm_kshot | xgboost | linear | 274.7 | -0.003 |
| sandia | 5 | pooled_erm_kshot | gaussian_process | none | 470.3 | 0.630 |
| sandia | 10 | pooled_erm_kshot | catboost | linear | 334.4 | 0.723 |
| sandia | 15 | pooled_erm_kshot | stacking | linear | 297.3 | 0.808 |
| sandia | 20 | pooled_erm_kshot | stacking | linear | 277.5 | 0.855 |

Protocol counts in the k-report candidate set:

## hust
- source_model_select / source_model_experts / linear k=20: R2=-0.110, MAE=236.0
- pooled_erm_kshot / elastic_net / linear k=20: R2=-0.113, MAE=237.0
- source_expert_select / source_primary_experts / linear k=20: R2=-0.129, MAE=237.6

## luh
- source_model_select / source_model_experts / linear k=20: R2=0.660, MAE=178.2
- pooled_erm_kshot / elastic_net / linear k=20: R2=0.658, MAE=186.0
- pooled_erm_kshot / pls / linear k=20: R2=0.638, MAE=182.9

## matr
- pooled_erm_kshot / xgboost / linear k=20: R2=-0.003, MAE=274.7
- source_expert_convex / source_primary_experts / residual_mean k=20: R2=-0.021, MAE=278.1
- source_expert_convex / source_primary_experts / none k=20: R2=-0.022, MAE=279.1

## sandia
- pooled_erm_kshot / stacking / linear k=20: R2=0.855, MAE=277.5
- pooled_erm_kshot / catboost / linear k=20: R2=0.844, MAE=287.4
- pooled_erm_kshot / random_forest / linear k=20: R2=0.826, MAE=309.9


# LODO Main Panel and Supplementary Tables

Main-text panel: `outputs/results_v2_four_dataset_lodo_source_expert/paper_lodo_main_panel.png`.

## Main Panel Summary

| Target | Naive single-source MAE | Pooled k=0 MAE | LODO k=20 MAE | Within oracle MAE | Reduction vs best k=0 (%) | LODO k=20 R2 |
|---|---|---|---|---|---|---|
| MATR | 389.1 | 840.0 | 274.7 | 171.7 | 29.4 | -0.003 |
| HUST | 875.9 | 598.9 | 236.0 | 178.0 | 60.6 | -0.110 |
| Sandia | 625.4 | 369.8 | 277.5 | 120.8 | 25.0 | 0.855 |
| Luh/KIT | 323.1 | 207.4 | 178.2 | 108.8 | 14.1 | 0.660 |

## Best Protocol by k

| Target | k | Protocol | Model | Adapter | MAE | R2 |
|---|---|---|---|---|---|---|
| HUST | 5 | pooled_erm_kshot | elastic_net | residual_mean | 257.5 | -0.289 |
| HUST | 10 | pooled_erm_kshot | elastic_net | residual_mean | 246.7 | -0.186 |
| HUST | 15 | pooled_erm_kshot | elastic_net | linear | 239.0 | -0.149 |
| HUST | 20 | source_model_select | source_model_experts | linear | 236.0 | -0.110 |
| Luh/KIT | 5 | pooled_erm_kshot | pls | none | 206.8 | 0.346 |
| Luh/KIT | 10 | source_model_select | source_model_experts | linear | 183.9 | 0.621 |
| Luh/KIT | 15 | source_model_select | source_model_experts | linear | 182.0 | 0.640 |
| Luh/KIT | 20 | source_model_select | source_model_experts | linear | 178.2 | 0.660 |
| MATR | 5 | source_expert_convex | source_primary_experts | residual_mean | 299.6 | -0.196 |
| MATR | 10 | source_expert_convex | source_primary_experts | residual_mean | 284.1 | -0.082 |
| MATR | 15 | source_expert_convex | source_primary_experts | residual_mean | 280.0 | -0.047 |
| MATR | 20 | pooled_erm_kshot | xgboost | linear | 274.7 | -0.003 |
| Sandia | 5 | pooled_erm_kshot | xgboost | none | 371.6 | 0.608 |
| Sandia | 10 | pooled_erm_kshot | stacking | linear | 327.2 | 0.723 |
| Sandia | 15 | pooled_erm_kshot | stacking | linear | 297.3 | 0.808 |
| Sandia | 20 | pooled_erm_kshot | stacking | linear | 277.5 | 0.855 |

## Top k=20 / Baseline Protocols by Target

| Target | Rank | Protocol | Model | Adapter | k | MAE | R2 |
|---|---|---|---|---|---|---|---|
| HUST | 1 | source_model_select | source_model_experts | linear | 20 | 236.0 | -0.110 |
| HUST | 2 | pooled_erm_kshot | elastic_net | linear | 20 | 237.0 | -0.113 |
| HUST | 3 | source_expert_select | source_primary_experts | linear | 20 | 237.6 | -0.129 |
| HUST | 4 | pooled_erm_kshot | gaussian_process | linear | 20 | 237.6 | -0.130 |
| HUST | 5 | pooled_erm_kshot | pls | linear | 20 | 238.1 | -0.130 |
| HUST | 6 | pooled_erm_kshot | xgboost | linear | 20 | 238.8 | -0.143 |
| Luh/KIT | 1 | source_model_select | source_model_experts | linear | 20 | 178.2 | 0.660 |
| Luh/KIT | 2 | pooled_erm_kshot | pls | linear | 20 | 182.9 | 0.638 |
| Luh/KIT | 3 | source_model_select | source_model_experts | none | 20 | 184.8 | 0.547 |
| Luh/KIT | 4 | pooled_erm_kshot | elastic_net | linear | 20 | 186.0 | 0.658 |
| Luh/KIT | 5 | pooled_erm_kshot | stacking | linear | 20 | 187.2 | 0.636 |
| Luh/KIT | 6 | pooled_erm_kshot | catboost | linear | 20 | 194.4 | 0.595 |
| MATR | 1 | pooled_erm_kshot | xgboost | linear | 20 | 274.7 | -0.003 |
| MATR | 2 | pooled_erm_kshot | random_forest | linear | 20 | 277.6 | -0.041 |
| MATR | 3 | pooled_erm_kshot | stacking | linear | 20 | 277.8 | -0.036 |
| MATR | 4 | source_expert_convex | source_primary_experts | residual_mean | 20 | 278.1 | -0.021 |
| MATR | 5 | source_expert_convex | source_primary_experts | none | 20 | 279.1 | -0.022 |
| MATR | 6 | source_expert_convex | source_primary_experts | linear | 20 | 281.2 | -0.037 |
| Sandia | 1 | pooled_erm_kshot | stacking | linear | 20 | 277.5 | 0.855 |
| Sandia | 2 | pooled_erm_kshot | catboost | linear | 20 | 287.4 | 0.844 |
| Sandia | 3 | pooled_erm_kshot | random_forest | linear | 20 | 309.9 | 0.826 |
| Sandia | 4 | pooled_erm_kshot | xgboost | linear | 20 | 316.7 | 0.818 |
| Sandia | 5 | pooled_erm | xgboost | none | 0 | 369.8 | 0.608 |
| Sandia | 6 | pooled_erm_kshot | xgboost | none | 20 | 380.1 | 0.600 |

## Protocol Family Winners

| Target | Protocol | Model | Adapter | k | MAE | R2 |
|---|---|---|---|---|---|---|
| HUST | source_model_select | source_model_experts | linear | 20 | 236.0 | -0.110 |
| HUST | pooled_erm_kshot | elastic_net | linear | 20 | 237.0 | -0.113 |
| HUST | source_expert_select | source_primary_experts | linear | 20 | 237.6 | -0.129 |
| HUST | source_expert_convex | source_primary_experts | linear | 20 | 240.0 | -0.180 |
| HUST | source_expert_uniform | source_primary_experts | none | 0 | 422.3 | -2.324 |
| HUST | pooled_erm | stacking | none | 0 | 598.9 | -6.168 |
| HUST | source_expert_single | catboost | none | 0 | 875.9 | -10.355 |
| Luh/KIT | source_model_select | source_model_experts | linear | 20 | 178.2 | 0.660 |
| Luh/KIT | pooled_erm_kshot | pls | linear | 20 | 182.9 | 0.638 |
| Luh/KIT | pooled_erm | pls | none | 0 | 207.4 | 0.346 |
| Luh/KIT | source_expert_select | source_primary_experts | linear | 20 | 280.0 | 0.218 |
| Luh/KIT | source_expert_convex | source_primary_experts | linear | 20 | 287.4 | 0.114 |
| Luh/KIT | source_expert_single | xgboost | none | 0 | 323.1 | -0.326 |
| Luh/KIT | source_expert_uniform | source_primary_experts | none | 0 | 365.1 | -0.054 |
| MATR | pooled_erm_kshot | xgboost | linear | 20 | 274.7 | -0.003 |
| MATR | source_expert_convex | source_primary_experts | residual_mean | 20 | 278.1 | -0.021 |
| MATR | source_expert_select | source_primary_experts | residual_mean | 20 | 283.6 | -0.055 |
| MATR | source_model_select | source_model_experts | residual_mean | 20 | 287.9 | -0.125 |
| MATR | source_expert_single | gaussian_process | none | 0 | 389.1 | -1.076 |
| MATR | source_expert_uniform | source_primary_experts | none | 0 | 597.6 | -2.405 |
| MATR | pooled_erm | catboost | none | 0 | 840.0 | -5.372 |
| Sandia | pooled_erm_kshot | stacking | linear | 20 | 277.5 | 0.855 |
| Sandia | pooled_erm | xgboost | none | 0 | 369.8 | 0.608 |
| Sandia | source_expert_single | gaussian_process | none | 0 | 625.4 | -0.110 |
| Sandia | source_expert_select | source_primary_experts | none | 15 | 630.2 | -0.088 |
| Sandia | source_expert_convex | source_primary_experts | linear | 20 | 734.7 | 0.117 |
| Sandia | source_expert_uniform | source_primary_experts | none | 0 | 857.9 | 0.062 |
| Sandia | source_model_select | source_model_experts | linear | 15 | 1714592.2 | -1653720744.466 |

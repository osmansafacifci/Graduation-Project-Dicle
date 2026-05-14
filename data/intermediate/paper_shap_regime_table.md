# SHAP x Conditional-Regime Table

SHAP values are from the source dataset's within-dataset primary model; 
conditional regimes are direction-specific source -> target slope classes.

| Direction | SHAP model | Shifted SHAP mass | Top-10 shifted/stable | Raw R² | Linear R² | Rank regime | Source-important shifted features |
|---|---|---:|---:|---:|---:|---|---|
| Sandia -> HUST | xgboost | 90.3% | 8/2 | -2.027 | 0.039 | weak_rank_signal | Qdis_N (54.8%); mad_Qdis (8.8%); slope_linear (7.3%) |
| Luh/KIT -> HUST | catboost | 88.9% | 10/0 | -0.763 | 0.021 | rank_signal_collapsed | slope_last_quarter (7.6%); slope_linear (6.3%); poly2_b (5.8%) |
| Sandia -> MATR | xgboost | 83.3% | 6/4 | -0.802 | 0.003 | rank_signal_collapsed | Qdis_N (54.8%); mad_Qdis (8.8%); slope_linear (7.3%) |
| HUST -> Luh/KIT | random_forest | 74.2% | 8/2 | -0.562 | 0.610 | strong_rank_signal | Qdis_N (28.2%); Qdis_cycle10 (10.6%); poly2_a (6.7%) |
| Luh/KIT -> MATR | catboost | 65.0% | 8/2 | -0.275 | 0.020 | weak_rank_signal | slope_last_quarter (7.6%); slope_linear (6.3%); poly2_b (5.8%) |
| HUST -> Sandia | random_forest | 61.8% | 6/4 | -0.231 | 0.366 | strong_rank_signal | Qdis_N (28.2%); Qdis_cycle10 (10.6%); sample_entropy (3.2%) |
| HUST -> MATR | random_forest | 54.8% | 3/7 | -3.600 | 0.008 | negative_or_inverted_signal | Qdis_N (28.2%); linearity_r2 (14.3%); spectral_entropy (2.4%) |
| MATR -> HUST | catboost | 54.6% | 6/4 | -7.937 | 0.025 | negative_or_inverted_signal | accel_mean (11.5%); poly2_c (10.7%); slope_last_quarter (7.1%) |
| MATR -> Luh/KIT | catboost | 44.9% | 3/7 | 0.004 | 0.193 | moderate_rank_signal | accel_mean (11.5%); slope_last_quarter (7.1%); mad_Qdis (5.9%) |
| MATR -> Sandia | catboost | 44.2% | 3/7 | 0.046 | 0.261 | moderate_rank_signal | accel_mean (11.5%); mad_Qdis (5.9%); sample_entropy (4.2%) |
| Luh/KIT -> Sandia | catboost | 6.5% | 0/10 | 0.499 | 0.842 | strong_rank_signal | sample_entropy (3.6%); cycle_to_98pct (2.8%) |
| Sandia -> Luh/KIT | xgboost | 2.7% | 0/10 | 0.494 | 0.750 | strong_rank_signal | cycle_to_98pct (1.3%); sample_entropy (1.3%); n_capacity_jumps (0.1%) |

Top-10 shifted/stable counts are computed over the ten highest source-SHAP features for each direction.
High shifted SHAP mass means the source model relies on features whose
feature -> log-life slope changes significantly in the target dataset. Low shifted
mass with positive rank signal is the most transfer-friendly regime.

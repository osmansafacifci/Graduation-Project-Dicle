# Four-Dataset Geometric Shift

- Feature table: `data/intermediate/features_sop12_four_dataset_capnorm.csv`
- Datasets: matr, hust, sandia, luh
- Windows: 50, 100
- Feature sets: 34

## N=50
### Feature Set 34 (34 features)
- `matr_vs_hust`: AUC=0.978, MMD=0.553, Mahalanobis=7.82; top shifts: linearity_r2 (1.34), knee_cycle (1.26), fft_top3_energy_ratio (1.16), mad_Qdis (1.15), spectral_entropy (1.12), skewness_Qdis (1.09), kurtosis_Qdis (1.05), accel_mean (1.03)
- `matr_vs_sandia`: AUC=0.959, MMD=0.810, Mahalanobis=11.76; top shifts: cycle_to_99pct (2.04), Qdis_cycle10 (1.86), retention_ratio (1.85), delta_Qdis (1.85), mean_diff (1.85), cycle_to_98pct (1.83), Qdis_N (1.83), poly2_b (1.72)
- `matr_vs_luh`: AUC=1.000, MMD=0.875, Mahalanobis=26.47; top shifts: cycle_to_99pct (1.97), Qdis_cycle10 (1.76), cycle_to_98pct (1.73), linearity_r2 (1.68), poly2_b (1.68), accel_mean (1.68), retention_ratio (1.66), delta_Qdis (1.66)
- `hust_vs_sandia`: AUC=0.988, MMD=0.763, Mahalanobis=11.22; top shifts: cycle_to_99pct (1.85), cycle_to_98pct (1.64), retention_ratio (1.61), mean_diff (1.61), delta_Qdis (1.61), Qdis_cycle10 (1.61), Qdis_N (1.60), slope_first_quarter (1.59)
- `hust_vs_luh`: AUC=1.000, MMD=0.815, Mahalanobis=31.05; top shifts: cycle_to_99pct (1.98), accel_std (1.89), accel_max_abs (1.81), autocorr_lag1 (1.78), fft_top3_energy_ratio (1.69), cycle_to_98pct (1.68), spectral_entropy (1.67), poly2_b (1.59)
- `sandia_vs_luh`: AUC=0.976, MMD=0.502, Mahalanobis=7.75; top shifts: spectral_entropy (1.49), autocorr_lag1 (1.46), fft_top3_energy_ratio (1.42), knee_cycle (1.33), sample_entropy (1.31), linearity_r2 (1.25), Qdis_cycle10 (1.12), std_diff (1.03)

## N=100
### Feature Set 34 (34 features)
- `matr_vs_hust`: AUC=0.981, MMD=0.431, Mahalanobis=6.71; top shifts: accel_mean (1.13), knee_cycle (0.78), linearity_r2 (0.74), cycle_to_99pct (0.67), autocorr_lag1 (0.65), mad_Qdis (0.56), skewness_Qdis (0.56), fft_top3_energy_ratio (0.48)
- `matr_vs_sandia`: AUC=0.977, MMD=0.831, Mahalanobis=12.41; top shifts: cycle_to_99pct (2.13), retention_ratio (1.91), delta_Qdis (1.91), mean_diff (1.91), Qdis_N (1.88), cycle_to_98pct (1.88), Qdis_cycle10 (1.86), slope_linear (1.82)
- `matr_vs_luh`: AUC=1.000, MMD=0.811, Mahalanobis=22.61; top shifts: cycle_to_99pct (1.97), cycle_to_98pct (1.95), Qdis_cycle10 (1.76), autocorr_lag1 (1.73), pos_neg_diff_ratio (1.71), slope_first_quarter (1.70), accel_mean (1.70), linearity_r2 (1.66)
- `hust_vs_sandia`: AUC=1.000, MMD=0.765, Mahalanobis=11.36; top shifts: cycle_to_99pct (1.91), retention_ratio (1.71), mean_diff (1.71), delta_Qdis (1.71), cycle_to_98pct (1.70), Qdis_N (1.69), pos_neg_diff_ratio (1.68), slope_linear (1.64)
- `hust_vs_luh`: AUC=1.000, MMD=0.773, Mahalanobis=37.96; top shifts: cycle_to_98pct (1.97), cycle_to_99pct (1.96), accel_std (1.95), autocorr_lag1 (1.93), accel_max_abs (1.89), pos_neg_diff_ratio (1.81), sample_entropy (1.76), slope_first_quarter (1.60)
- `sandia_vs_luh`: AUC=0.975, MMD=0.473, Mahalanobis=9.75; top shifts: autocorr_lag1 (1.78), fft_top3_energy_ratio (1.29), sample_entropy (1.25), spectral_entropy (1.19), Qdis_cycle10 (1.12), kurtosis_Qdis (1.10), max_drop (1.02), pos_neg_diff_ratio (1.00)


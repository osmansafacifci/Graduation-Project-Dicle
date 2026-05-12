# Four-Dataset Geometric Shift

- Feature table: `data/intermediate/features_sop12_four_dataset.csv`
- Datasets: matr, hust, sandia, luh
- Windows: 50, 100
- Feature sets: 34

## N=50
### Feature Set 34 (34 features)
- `matr_vs_hust`: AUC=1.000, MMD=0.647, Mahalanobis=16.92; top shifts: Qdis_cycle10 (2.03), poly2_a (2.02), Qdis_N (2.02), linearity_r2 (1.34), knee_cycle (1.26), mad_Qdis (1.25), fft_top3_energy_ratio (1.16), spectral_entropy (1.12)
- `matr_vs_sandia`: AUC=0.997, MMD=0.834, Mahalanobis=17.01; top shifts: cycle_to_99pct (2.04), poly2_a (1.90), Qdis_cycle10 (1.90), Qdis_N (1.89), retention_ratio (1.85), cycle_to_98pct (1.83), delta_Qdis (1.80), mean_diff (1.80)
- `matr_vs_luh`: AUC=0.996, MMD=0.889, Mahalanobis=234.32; top shifts: poly2_a (2.01), Qdis_cycle10 (2.01), Qdis_N (2.00), cycle_to_99pct (1.97), cycle_to_98pct (1.73), linearity_r2 (1.68), poly2_b (1.68), retention_ratio (1.66)
- `hust_vs_sandia`: AUC=0.984, MMD=0.783, Mahalanobis=12.43; top shifts: cycle_to_99pct (1.85), poly2_a (1.65), Qdis_cycle10 (1.65), cycle_to_98pct (1.64), Qdis_N (1.63), retention_ratio (1.61), delta_Qdis (1.59), mean_diff (1.59)
- `hust_vs_luh`: AUC=1.000, MMD=0.845, Mahalanobis=178.87; top shifts: poly2_a (2.02), Qdis_cycle10 (2.02), Qdis_N (2.01), cycle_to_99pct (1.98), accel_std (1.79), autocorr_lag1 (1.78), fft_top3_energy_ratio (1.69), cycle_to_98pct (1.68)
- `sandia_vs_luh`: AUC=0.979, MMD=0.527, Mahalanobis=6.15; top shifts: spectral_entropy (1.49), autocorr_lag1 (1.46), fft_top3_energy_ratio (1.42), knee_cycle (1.33), sample_entropy (1.31), linearity_r2 (1.25), kurtosis_Qdis (1.01), pos_neg_diff_ratio (0.97)

## N=100
### Feature Set 34 (34 features)
- `matr_vs_hust`: AUC=1.000, MMD=0.571, Mahalanobis=15.98; top shifts: Qdis_cycle10 (2.03), poly2_a (2.02), Qdis_N (2.00), accel_mean (1.06), knee_cycle (0.78), linearity_r2 (0.74), mad_Qdis (0.67), cycle_to_99pct (0.67)
- `matr_vs_sandia`: AUC=0.979, MMD=0.854, Mahalanobis=13.87; top shifts: cycle_to_99pct (2.13), retention_ratio (1.91), poly2_a (1.90), delta_Qdis (1.90), mean_diff (1.90), Qdis_cycle10 (1.90), Qdis_N (1.88), cycle_to_98pct (1.88)
- `matr_vs_luh`: AUC=1.000, MMD=0.848, Mahalanobis=231.45; top shifts: poly2_a (2.01), Qdis_cycle10 (2.01), Qdis_N (1.99), cycle_to_99pct (1.97), cycle_to_98pct (1.95), autocorr_lag1 (1.73), pos_neg_diff_ratio (1.71), slope_first_quarter (1.69)
- `hust_vs_sandia`: AUC=1.000, MMD=0.786, Mahalanobis=13.19; top shifts: cycle_to_99pct (1.91), retention_ratio (1.71), cycle_to_98pct (1.70), delta_Qdis (1.69), mean_diff (1.69), pos_neg_diff_ratio (1.68), slope_linear (1.65), poly2_a (1.65)
- `hust_vs_luh`: AUC=1.000, MMD=0.834, Mahalanobis=188.38; top shifts: poly2_a (2.02), Qdis_cycle10 (2.02), Qdis_N (1.99), cycle_to_98pct (1.97), cycle_to_99pct (1.96), autocorr_lag1 (1.93), accel_std (1.91), pos_neg_diff_ratio (1.81)
- `sandia_vs_luh`: AUC=0.980, MMD=0.491, Mahalanobis=8.88; top shifts: autocorr_lag1 (1.78), fft_top3_energy_ratio (1.29), sample_entropy (1.25), spectral_entropy (1.19), kurtosis_Qdis (1.10), pos_neg_diff_ratio (1.00), linearity_r2 (0.99), max_drop (0.97)


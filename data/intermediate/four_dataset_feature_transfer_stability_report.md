# Four-Dataset Feature Transfer Stability

- Feature table: `data/intermediate/features_sop12_four_dataset_capnorm.csv`
- Splits: `splits/sop_v2_four_dataset`
- k-target residual adapter: 20

## N=100
### `hust_to_luh`
- Class counts: {'weak_or_mixed': 17, 'relationship_unstable': 10, 'scale_shift_fragile': 7}
- Top least-fragile features:
  - `exp_decay_k`: score=+0.789, shift_z=1.69, rho=(-0.12, -0.93), adapted_R2=+0.473, class=weak_or_mixed
  - `slope_linear`: score=+0.720, shift_z=1.78, rho=(+0.12, +0.93), adapted_R2=+0.446, class=weak_or_mixed
  - `mad_Qdis`: score=+0.624, shift_z=1.51, rho=(-0.13, -0.92), adapted_R2=+0.374, class=weak_or_mixed
  - `Qdis_N`: score=+0.612, shift_z=1.91, rho=(+0.10, +0.93), adapted_R2=+0.404, class=weak_or_mixed
  - `mean_diff`: score=+0.580, shift_z=1.93, rho=(+0.08, +0.92), adapted_R2=+0.396, class=weak_or_mixed
  - `delta_Qdis`: score=+0.579, shift_z=1.93, rho=(+0.08, +0.92), adapted_R2=+0.395, class=weak_or_mixed
  - `retention_ratio`: score=+0.557, shift_z=1.93, rho=(+0.08, +0.92), adapted_R2=+0.385, class=weak_or_mixed
  - `knee_cycle`: score=+0.442, shift_z=0.22, rho=(-0.09, -0.13), adapted_R2=-0.070, class=weak_or_mixed
  - `kurtosis_Qdis`: score=+0.302, shift_z=1.29, rho=(+0.30, +0.29), adapted_R2=-0.055, class=weak_or_mixed
  - `slope_ratio`: score=+0.258, shift_z=1.14, rho=(-0.36, -0.22), adapted_R2=-0.034, class=weak_or_mixed

### `hust_to_matr`
- Class counts: {'relationship_unstable': 21, 'weak_or_mixed': 13}
- Top least-fragile features:
  - `Qdis_N`: score=+0.489, shift_z=0.09, rho=(+0.10, +0.08), adapted_R2=+0.007, class=weak_or_mixed
  - `cycle_to_98pct`: score=+0.476, shift_z=0.12, rho=(+0.20, +0.21), adapted_R2=-0.051, class=weak_or_mixed
  - `slope_linear`: score=+0.463, shift_z=0.19, rho=(+0.12, +0.11), adapted_R2=-0.005, class=weak_or_mixed
  - `exp_decay_k`: score=+0.463, shift_z=0.19, rho=(-0.12, -0.11), adapted_R2=-0.016, class=weak_or_mixed
  - `slope_ratio`: score=+0.422, shift_z=0.23, rho=(-0.36, -0.27), adapted_R2=-0.479, class=weak_or_mixed
  - `autocorr_lag1`: score=+0.356, shift_z=0.69, rho=(+0.15, +0.07), adapted_R2=-0.130, class=weak_or_mixed
  - `kurtosis_Qdis`: score=+0.348, shift_z=0.49, rho=(+0.30, +0.14), adapted_R2=-6.136, class=weak_or_mixed
  - `linearity_r2`: score=+0.345, shift_z=0.79, rho=(-0.40, -0.30), adapted_R2=+0.006, class=weak_or_mixed
  - `mad_Qdis`: score=+0.307, shift_z=0.58, rho=(-0.13, -0.39), adapted_R2=+0.014, class=weak_or_mixed
  - `cycle_to_99pct`: score=+0.299, shift_z=0.71, rho=(+0.10, +0.29), adapted_R2=-0.045, class=weak_or_mixed

### `hust_to_sandia`
- Class counts: {'weak_or_mixed': 16, 'relationship_unstable': 12, 'scale_shift_fragile': 6}
- Top least-fragile features:
  - `cycle_to_98pct`: score=+0.706, shift_z=3.06, rho=(+0.20, +0.45), adapted_R2=+0.397, class=scale_shift_fragile
  - `accel_mean`: score=+0.422, shift_z=0.07, rho=(-0.17, -0.31), adapted_R2=-1.647, class=weak_or_mixed
  - `knee_cycle`: score=+0.407, shift_z=0.23, rho=(-0.09, -0.20), adapted_R2=-0.094, class=weak_or_mixed
  - `mad_Qdis`: score=+0.372, shift_z=2.35, rho=(-0.13, -0.76), adapted_R2=+0.271, class=weak_or_mixed
  - `poly2_a`: score=+0.350, shift_z=0.88, rho=(+0.15, +0.11), adapted_R2=-0.525, class=weak_or_mixed
  - `slope_ratio`: score=+0.327, shift_z=0.97, rho=(-0.36, -0.30), adapted_R2=-0.100, class=weak_or_mixed
  - `kurtosis_Qdis`: score=+0.315, shift_z=0.35, rho=(+0.30, +0.56), adapted_R2=-0.106, class=weak_or_mixed
  - `linearity_r2`: score=+0.301, shift_z=0.92, rho=(-0.40, -0.28), adapted_R2=-0.092, class=weak_or_mixed
  - `autocorr_lag1`: score=+0.298, shift_z=1.27, rho=(+0.15, +0.18), adapted_R2=-0.149, class=weak_or_mixed
  - `sample_entropy`: score=+0.246, shift_z=1.37, rho=(+0.25, +0.39), adapted_R2=-0.024, class=weak_or_mixed

### `luh_to_hust`
- Class counts: {'weak_or_mixed': 16, 'relationship_unstable': 10, 'scale_shift_fragile': 7, 'stable_candidate': 1}
- Top least-fragile features:
  - `knee_cycle`: score=+0.442, shift_z=0.22, rho=(-0.13, -0.09), adapted_R2=-0.085, class=weak_or_mixed
  - `Qdis_N`: score=+0.425, shift_z=1.91, rho=(+0.93, +0.10), adapted_R2=-0.365, class=weak_or_mixed
  - `slope_linear`: score=+0.424, shift_z=1.78, rho=(+0.93, +0.12), adapted_R2=-0.108, class=weak_or_mixed
  - `exp_decay_k`: score=+0.422, shift_z=1.69, rho=(-0.93, -0.12), adapted_R2=-0.072, class=weak_or_mixed
  - `delta_Qdis`: score=+0.421, shift_z=1.93, rho=(+0.92, +0.08), adapted_R2=-0.443, class=weak_or_mixed
  - `mean_diff`: score=+0.421, shift_z=1.93, rho=(+0.92, +0.08), adapted_R2=-0.440, class=weak_or_mixed
  - `retention_ratio`: score=+0.420, shift_z=1.93, rho=(+0.92, +0.08), adapted_R2=-0.490, class=weak_or_mixed
  - `mad_Qdis`: score=+0.416, shift_z=1.51, rho=(-0.92, -0.13), adapted_R2=-0.056, class=weak_or_mixed
  - `Qdis_cycle10`: score=+0.392, shift_z=2.49, rho=(+0.83, +0.07), adapted_R2=-95.345, class=weak_or_mixed
  - `slope_first_quarter`: score=+0.361, shift_z=2.63, rho=(+0.84, +0.05), adapted_R2=-27.290, class=weak_or_mixed

### `luh_to_matr`
- Class counts: {'weak_or_mixed': 13, 'relationship_unstable': 8, 'scale_shift_fragile': 7, 'stable_candidate': 6}
- Top least-fragile features:
  - `range_Qdis`: score=+0.866, shift_z=0.48, rho=(-0.92, -0.54), adapted_R2=-0.098, class=stable_candidate
  - `max_drop`: score=+0.739, shift_z=0.15, rho=(-0.85, -0.07), adapted_R2=-2.240, class=stable_candidate
  - `std_diff`: score=+0.700, shift_z=0.15, rho=(-0.69, -0.26), adapted_R2=-0.352, class=stable_candidate
  - `accel_std`: score=+0.621, shift_z=0.16, rho=(-0.63, -0.18), adapted_R2=-0.028, class=stable_candidate
  - `mad_Qdis`: score=+0.598, shift_z=1.81, rho=(-0.92, -0.39), adapted_R2=+0.046, class=weak_or_mixed
  - `cycle_to_95pct`: score=+0.590, shift_z=1.39, rho=(+0.89, +0.15), adapted_R2=-0.025, class=weak_or_mixed
  - `slope_last_quarter`: score=+0.550, shift_z=1.31, rho=(+0.89, +0.47), adapted_R2=+0.046, class=weak_or_mixed
  - `variance_Qdis`: score=+0.547, shift_z=0.63, rho=(-0.93, -0.57), adapted_R2=-0.081, class=stable_candidate
  - `accel_max_abs`: score=+0.517, shift_z=0.15, rho=(-0.60, -0.11), adapted_R2=-0.281, class=stable_candidate
  - `Qdis_N`: score=+0.439, shift_z=2.18, rho=(+0.93, +0.08), adapted_R2=+0.033, class=weak_or_mixed

### `luh_to_sandia`
- Class counts: {'stable_candidate': 18, 'weak_or_mixed': 12, 'relationship_unstable': 3, 'scale_shift_fragile': 1}
- Top least-fragile features:
  - `range_Qdis`: score=+1.755, shift_z=0.23, rho=(-0.92, -0.69), adapted_R2=+0.387, class=stable_candidate
  - `retention_ratio`: score=+1.689, shift_z=0.00, rho=(+0.92, +0.58), adapted_R2=+0.364, class=stable_candidate
  - `delta_Qdis`: score=+1.689, shift_z=0.00, rho=(+0.92, +0.57), adapted_R2=+0.366, class=stable_candidate
  - `poly2_b`: score=+1.674, shift_z=0.06, rho=(+0.86, +0.38), adapted_R2=+0.403, class=stable_candidate
  - `mean_diff`: score=+1.648, shift_z=0.00, rho=(+0.92, +0.57), adapted_R2=+0.345, class=stable_candidate
  - `slope_linear`: score=+1.637, shift_z=0.18, rho=(+0.93, +0.73), adapted_R2=+0.334, class=stable_candidate
  - `exp_decay_k`: score=+1.531, shift_z=0.17, rho=(-0.93, -0.73), adapted_R2=+0.289, class=stable_candidate
  - `cycle_to_95pct`: score=+1.501, shift_z=0.59, rho=(+0.89, +0.45), adapted_R2=+0.320, class=stable_candidate
  - `mad_Qdis`: score=+1.438, shift_z=0.18, rho=(-0.92, -0.76), adapted_R2=+0.253, class=stable_candidate
  - `Qdis_N`: score=+1.375, shift_z=0.05, rho=(+0.93, +0.63), adapted_R2=+0.204, class=stable_candidate

### `matr_to_hust`
- Class counts: {'relationship_unstable': 21, 'stable_candidate': 8, 'weak_or_mixed': 5}
- Top least-fragile features:
  - `Qdis_N`: score=+0.610, shift_z=0.09, rho=(+0.08, +0.10), adapted_R2=-0.323, class=stable_candidate
  - `cycle_to_98pct`: score=+0.548, shift_z=0.12, rho=(+0.21, +0.20), adapted_R2=-0.042, class=stable_candidate
  - `exp_decay_k`: score=+0.533, shift_z=0.19, rho=(-0.11, -0.12), adapted_R2=-0.061, class=stable_candidate
  - `slope_linear`: score=+0.532, shift_z=0.19, rho=(+0.11, +0.12), adapted_R2=-0.080, class=stable_candidate
  - `linearity_r2`: score=+0.479, shift_z=0.79, rho=(-0.30, -0.40), adapted_R2=-0.085, class=stable_candidate
  - `mad_Qdis`: score=+0.473, shift_z=0.58, rho=(-0.39, -0.13), adapted_R2=-0.043, class=stable_candidate
  - `cycle_to_99pct`: score=+0.438, shift_z=0.71, rho=(+0.29, +0.10), adapted_R2=-0.144, class=stable_candidate
  - `slope_ratio`: score=+0.422, shift_z=0.23, rho=(-0.27, -0.36), adapted_R2=-0.035, class=weak_or_mixed
  - `autocorr_lag1`: score=+0.356, shift_z=0.69, rho=(+0.07, +0.15), adapted_R2=-0.065, class=weak_or_mixed
  - `kurtosis_Qdis`: score=+0.348, shift_z=0.49, rho=(+0.14, +0.30), adapted_R2=-0.063, class=weak_or_mixed

### `matr_to_luh`
- Class counts: {'weak_or_mixed': 18, 'relationship_unstable': 8, 'scale_shift_fragile': 7, 'stable_candidate': 1}
- Top least-fragile features:
  - `slope_last_quarter`: score=+1.092, shift_z=1.31, rho=(+0.47, +0.89), adapted_R2=+0.387, class=weak_or_mixed
  - `mad_Qdis`: score=+0.819, shift_z=1.81, rho=(-0.39, -0.92), adapted_R2=+0.330, class=weak_or_mixed
  - `Qdis_N`: score=+0.540, shift_z=2.18, rho=(+0.08, +0.93), adapted_R2=+0.327, class=weak_or_mixed
  - `exp_decay_k`: score=+0.476, shift_z=1.93, rho=(-0.11, -0.93), adapted_R2=+0.304, class=weak_or_mixed
  - `slope_linear`: score=+0.472, shift_z=2.03, rho=(+0.11, +0.93), adapted_R2=+0.310, class=weak_or_mixed
  - `spectral_entropy`: score=+0.402, shift_z=0.39, rho=(+0.15, +0.23), adapted_R2=-0.051, class=weak_or_mixed
  - `slope_ratio`: score=+0.399, shift_z=0.49, rho=(-0.27, -0.22), adapted_R2=-0.062, class=weak_or_mixed
  - `fft_top3_energy_ratio`: score=+0.366, shift_z=0.65, rho=(-0.19, -0.26), adapted_R2=-0.073, class=weak_or_mixed
  - `knee_cycle`: score=+0.343, shift_z=0.92, rho=(-0.27, -0.13), adapted_R2=-0.069, class=stable_candidate
  - `kurtosis_Qdis`: score=+0.313, shift_z=0.75, rho=(+0.14, +0.29), adapted_R2=-0.070, class=weak_or_mixed

### `matr_to_sandia`
- Class counts: {'weak_or_mixed': 15, 'scale_shift_fragile': 9, 'relationship_unstable': 8, 'stable_candidate': 2}
- Top least-fragile features:
  - `mad_Qdis`: score=+0.547, shift_z=2.79, rho=(-0.39, -0.76), adapted_R2=+0.228, class=weak_or_mixed
  - `slope_last_quarter`: score=+0.516, shift_z=0.56, rho=(+0.47, +0.72), adapted_R2=-14891791090.974, class=stable_candidate
  - `knee_cycle`: score=+0.476, shift_z=0.26, rho=(-0.27, -0.20), adapted_R2=-0.011, class=stable_candidate
  - `cycle_to_98pct`: score=+0.470, shift_z=3.48, rho=(+0.21, +0.45), adapted_R2=+0.271, class=scale_shift_fragile
  - `slope_ratio`: score=+0.418, shift_z=0.45, rho=(-0.27, -0.30), adapted_R2=-0.116, class=weak_or_mixed
  - `autocorr_lag1`: score=+0.371, shift_z=0.52, rho=(+0.07, +0.18), adapted_R2=-0.108, class=weak_or_mixed
  - `range_Qdis`: score=+0.345, shift_z=0.52, rho=(-0.54, -0.69), adapted_R2=-0.112, class=weak_or_mixed
  - `std_diff`: score=+0.340, shift_z=0.02, rho=(-0.26, -0.57), adapted_R2=-0.123, class=weak_or_mixed
  - `variance_Qdis`: score=+0.329, shift_z=0.60, rho=(-0.57, -0.73), adapted_R2=-0.110, class=weak_or_mixed
  - `accel_std`: score=+0.323, shift_z=0.01, rho=(-0.18, -0.53), adapted_R2=-0.096, class=weak_or_mixed

### `sandia_to_hust`
- Class counts: {'weak_or_mixed': 16, 'relationship_unstable': 12, 'scale_shift_fragile': 6}
- Top least-fragile features:
  - `cycle_to_98pct`: score=+0.833, shift_z=3.06, rho=(+0.45, +0.20), adapted_R2=+0.002, class=scale_shift_fragile
  - `slope_linear`: score=+0.567, shift_z=2.76, rho=(+0.73, +0.12), adapted_R2=-2.757, class=weak_or_mixed
  - `retention_ratio`: score=+0.566, shift_z=3.13, rho=(+0.58, +0.08), adapted_R2=-8.741, class=scale_shift_fragile
  - `delta_Qdis`: score=+0.565, shift_z=3.12, rho=(+0.57, +0.08), adapted_R2=-8.425, class=scale_shift_fragile
  - `mean_diff`: score=+0.565, shift_z=3.12, rho=(+0.57, +0.08), adapted_R2=-8.355, class=scale_shift_fragile
  - `exp_decay_k`: score=+0.560, shift_z=2.72, rho=(-0.73, -0.12), adapted_R2=-1.958, class=weak_or_mixed
  - `Qdis_N`: score=+0.558, shift_z=3.02, rho=(+0.63, +0.10), adapted_R2=-6.889, class=scale_shift_fragile
  - `mad_Qdis`: score=+0.463, shift_z=2.35, rho=(-0.76, -0.13), adapted_R2=-0.040, class=weak_or_mixed
  - `accel_mean`: score=+0.422, shift_z=0.07, rho=(-0.31, -0.17), adapted_R2=-0.189, class=weak_or_mixed
  - `knee_cycle`: score=+0.407, shift_z=0.23, rho=(-0.20, -0.09), adapted_R2=-0.105, class=weak_or_mixed

### `sandia_to_luh`
- Class counts: {'weak_or_mixed': 16, 'stable_candidate': 14, 'relationship_unstable': 3, 'scale_shift_fragile': 1}
- Top least-fragile features:
  - `slope_linear`: score=+2.647, shift_z=0.18, rho=(+0.73, +0.93), adapted_R2=+0.745, class=stable_candidate
  - `exp_decay_k`: score=+2.623, shift_z=0.17, rho=(-0.73, -0.93), adapted_R2=+0.739, class=stable_candidate
  - `Qdis_N`: score=+2.587, shift_z=0.05, rho=(+0.63, +0.93), adapted_R2=+0.735, class=stable_candidate
  - `retention_ratio`: score=+2.579, shift_z=0.00, rho=(+0.58, +0.92), adapted_R2=+0.732, class=stable_candidate
  - `mean_diff`: score=+2.568, shift_z=0.00, rho=(+0.57, +0.92), adapted_R2=+0.732, class=stable_candidate
  - `delta_Qdis`: score=+2.563, shift_z=0.00, rho=(+0.57, +0.92), adapted_R2=+0.730, class=stable_candidate
  - `mad_Qdis`: score=+2.439, shift_z=0.18, rho=(-0.76, -0.92), adapted_R2=+0.707, class=stable_candidate
  - `range_Qdis`: score=+2.265, shift_z=0.23, rho=(-0.69, -0.92), adapted_R2=+0.675, class=stable_candidate
  - `cycle_to_95pct`: score=+1.775, shift_z=0.59, rho=(+0.45, +0.89), adapted_R2=+0.530, class=stable_candidate
  - `cycle_to_98pct`: score=+1.771, shift_z=0.23, rho=(+0.45, +0.86), adapted_R2=+0.295, class=stable_candidate

### `sandia_to_matr`
- Class counts: {'weak_or_mixed': 13, 'scale_shift_fragile': 9, 'relationship_unstable': 8, 'stable_candidate': 4}
- Top least-fragile features:
  - `range_Qdis`: score=+0.911, shift_z=0.52, rho=(-0.69, -0.54), adapted_R2=-0.113, class=stable_candidate
  - `cycle_to_98pct`: score=+0.774, shift_z=3.48, rho=(+0.45, +0.21), adapted_R2=-0.020, class=scale_shift_fragile
  - `mad_Qdis`: score=+0.654, shift_z=2.79, rho=(-0.76, -0.39), adapted_R2=+0.061, class=weak_or_mixed
  - `variance_Qdis`: score=+0.512, shift_z=0.60, rho=(-0.73, -0.57), adapted_R2=-0.109, class=stable_candidate
  - `slope_linear`: score=+0.499, shift_z=3.16, rho=(+0.73, +0.11), adapted_R2=-0.367, class=scale_shift_fragile
  - `exp_decay_k`: score=+0.491, shift_z=3.13, rho=(-0.73, -0.11), adapted_R2=-0.308, class=scale_shift_fragile
  - `Qdis_N`: score=+0.469, shift_z=3.55, rho=(+0.63, +0.08), adapted_R2=-0.231, class=scale_shift_fragile
  - `cycle_to_95pct`: score=+0.456, shift_z=2.77, rho=(+0.45, +0.15), adapted_R2=-0.050, class=weak_or_mixed
  - `knee_cycle`: score=+0.425, shift_z=0.26, rho=(-0.20, -0.27), adapted_R2=+0.000, class=weak_or_mixed
  - `slope_ratio`: score=+0.418, shift_z=0.45, rho=(-0.30, -0.27), adapted_R2=-738890.622, class=weak_or_mixed


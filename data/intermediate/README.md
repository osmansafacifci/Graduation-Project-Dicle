# `data/intermediate/` — Manifest

This directory contains **derived feature tables, audit CSVs, and per-analysis
result files**. It is the single Phase A → Phase B interface: anyone who has
this directory plus `splits/` and the code in `0_data/ 1_features/ 2_models/
3_analysis/` can reproduce every paper-facing number in the repository
without re-touching the raw `.pkl` / `.csv` files of the source datasets.

Raw cell-level files live under `data/raw/` (git-ignored) and are obtained
once from the dataset providers documented in the root [`README.md`](../../README.md).

All files here are derived products. Each file's lineage is summarised below.

## Layout (~130 files; ~10 MB total)

### Feature tables — the primary Phase A output

| File | Rows × cols | What it is | Built by |
|---|---:|---|---|
| `features_sop12_matr.csv` | 270 × 41 | MATR 34 features at N∈{50,100}; raw (no Q0 normalisation) | `1_features/build_features.py` |
| `features_sop12_hust.csv` | 154 × 41 | HUST 34 features at N∈{50,100} | same |
| `features_sop12_sandia.csv` | 172 × 41 | Sandia (all SOC subsets) 34 features at N∈{50,100} | same |
| `features_sop12_luh.csv` | 216 × 41 | Luh/KIT 34 features at N∈{50,100} | same |
| `features_sop12_combined.csv` | 424 × 41 | Two-dataset MATR + HUST union; backwards-compatible filename | same |
| `features_sop12_four_dataset.csv` | 762 × 41 | Four-dataset union (MATR + HUST + Sandia 0-100 SOC subset + Luh) | `1_features/build_four_dataset_table.py` |
| `features_sop12_four_dataset_capnorm.csv` | 762 × 41 | Same as above but with raw-capacity features divided by per-cell Q0 (SOP §2.3) | `1_features/build_features.py --capacity-normalize` |
| `features_sop12_{matr,hust,sandia,luh}_capnorm.csv` | per-dataset | Per-dataset capnorm variants for ablations | same |
| `feature_set_sop12.txt`, `feature_set_24.txt` | small text | Feature-name lists for `--features-from` ablations | hand-maintained |

Column schema for all `features_sop12_*.csv` files:
- Metadata: `dataset, cell_id, n_cycles, q0, cycle_life, is_censored, capacity_normalized`
- 12 SOP features: `Qdis_N, delta_Qdis, retention_ratio, slope_linear, variance_Qdis, range_Qdis, max_drop, std_diff, skewness_Qdis, slope_ratio, Qdis_cycle10, mean_diff`
- 12 shape/decay: `poly2_a, poly2_b, poly2_c, exp_decay_k, cycle_to_99pct, cycle_to_98pct, cycle_to_95pct, slope_first_quarter, slope_last_quarter, autocorr_lag1, knee_cycle, n_capacity_jumps`
- 10 entropy/FFT/2nd-deriv: `accel_mean, accel_std, accel_max_abs, linearity_r2, kurtosis_Qdis, fft_top3_energy_ratio, spectral_entropy, sample_entropy, pos_neg_diff_ratio, mad_Qdis`

### Per-cycle tidy tables

| File | Built by | Used by |
|---|---|---|
| `matr_cycles_tidy.csv` | `0_data/build_matr_audit.py` | feature builder + DMD pilot |
| `hust_cycles_tidy.csv` | `0_data/build_hust_audit.py` | same |
| `sandia_cycles_tidy.csv` | Phase A Sandia notebook | same |
| `luh_cycles_tidy.csv` | Phase A Luh notebook | same |

### Audit tables — for documenting label and censoring choices

- `matr_cell_audit_strict.csv` — 135-cell strict audit; only duplicate continuation-source dedup applied
- `matr_cell_audit_replication.csv` — 112-cell notebook-era audit (legacy)
- `matr_retention_summary.csv` — per-batch retention at 90/85/80% thresholds
- `hust_threshold_audit.csv`, `hust_threshold_summary.csv` — HUST per-cell Q0/EOL audit + summary
- `four_dataset_manifest.csv` — single-table summary of all four datasets used in the paper

### Analysis outputs (paper-facing)

Grouped by analysis script:

| Prefix | Built by | What |
|---|---|---|
| `shift_metrics*.json`, `shift_report*.txt` | `3_analysis/shift_metrics.py` | MMD + Mahalanobis, raw + capnorm |
| `four_dataset_geometric_shift_*` | `3_analysis/four_dataset_geometric_shift.py` | All-pairs MMD/Mahalanobis + discriminator AUC |
| `feature_transfer_stability*` | `3_analysis/feature_transfer_stability.py` | Per-feature direction-aware stability score |
| `four_dataset_feature_transfer_stability_*` | `3_analysis/four_dataset_feature_transfer_stability.py` | All-pairs stability score |
| `concept_shift_diagnostics.json` | `3_analysis/concept_shift_diagnostics.py` | KS test on cycle_life + constant-bias residual decomposition |
| `conditional_shift_*` | `3_analysis/conditional_shift_decomposition.py` | Two-dataset centered-log slope tests with BH-FDR + alpha/beta calibration |
| `four_dataset_conditional_shift_*` | `3_analysis/conditional_shift_four_dataset.py` | Four-dataset pairwise slope shifts + rank-signal regime labels |
| `koopman_dmd_*` | `3_analysis/koopman_dmd_pilot.py` | Hankel-DMD eigenvalues + operator transfer (two-dataset) |
| `four_dataset_koopman_dmd_*` | same with four-dataset CLI flags | Four-dataset version |
| `shap_feature_importance_*` | `3_analysis/shap_feature_importance.py` | Two-dataset SHAP × transfer-class join |
| `four_dataset_shap_feature_importance_*` | same with four-dataset flags | Four-dataset SHAP |
| `paper_shap_regime_*` | `3_analysis/build_shap_regime_table.py` | SHAP × regime joined table |
| `survival_censoring*` | `3_analysis/survival_censoring.py` | Kaplan-Meier + log-rank (two-dataset) |
| `four_dataset_survival_censoring_*` | `3_analysis/survival_censoring_four_dataset.py` | Same with RMST bootstrap CIs |
| `four_dataset_target_rescale_*` | `3_analysis/summarize_target_rescaling.py` | k-shot adapter tables |
| `four_dataset_lodo_source_expert_*` | `3_analysis/lodo_source_expert_transfer.py` | LODO source-expert protocol |
| `four_dataset_validation_*` | `3_analysis/validate_four_dataset_extension.py` | Reproducibility sanity checks (19 assertions) |
| `paper_kshot_*` | `3_analysis/plot_kshot_scaling.py` | k-shot scaling figure data |
| `paper_kshot_cp_scaling_summary.csv` | same | CP coverage curve across k |
| `four_dataset_cnn_pytorch_report.md` | `2_models/run_cnn_pytorch.py` | PyTorch CNN baseline summary |
| `paper_shap_regime_*` | `3_analysis/build_shap_regime_table.py` | SHAP × rank-signal regime joined table |

### Conventions

- All paths in scripts are relative to the project root and computed from
  `Path(__file__).resolve().parents[…]`. Move scripts only if you update the
  hard-coded path depth.
- All CSVs use UTF-8, comma-separated, with a single header row.
- Float columns are written with full precision; manuscript-facing tables
  format with `.3f` (R²) / `.1f` (MAE) / `.1%` (coverage) at presentation
  time, not at write time.
- Per-direction CP results aggregate across (5 seeds × 20 adapter repeats);
  see [`docs/MANUSCRIPT_POSITIONING.md`](../../docs/MANUSCRIPT_POSITIONING.md)
  §"CP outlier aggregation note" for the median-vs-mean policy.

### Provenance

Every file in this directory is derivable from `data/raw/` plus the code at
the tip of the corresponding commit. The Zenodo deposit at v1.0.0 freezes
both the code and the derived CSVs. See [`docs/ZENODO.md`](../../docs/ZENODO.md)
for the deposit workflow and [`docs/REPRODUCIBILITY.md`](../../docs/REPRODUCIBILITY.md)
for the end-to-end reproduction recipe.

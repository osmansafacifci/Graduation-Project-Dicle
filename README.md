# Battery Lifetime Prediction — SOP-Compliant Pipeline

Early-cycle lifetime prediction for lithium-ion cells (Severson/MATR and HUST
public datasets), built around an SOP that fixes label semantics, feature
definitions, and split protocols. Companion to a thesis on cross-dataset
transfer of capacity-only features.

> **Status**
> Within-dataset, cross-dataset, distribution-shift quantification,
> feature-transfer/XAI diagnostics, survival/censoring sensitivity,
> concept-shift diagnostics, target-side recalibration baselines, and standard
> MAPIE split conformal prediction are all in place.

---

## Headline numbers

**Within-dataset (primary configuration: 34 capacity-only features + log-target,
5 seeds, N=100, best model by mean R² within the fixed protocol):**

| Dataset | Best model | MAE [bootstrap 95% CI] | sMAPE [bootstrap 95% CI] | R² [bootstrap 95% CI] |
|---|---|---|---|---|
| MATR | CatBoost | **171.7 [110.4, 243.2]** | 23.7 [15.5, 33.5] | **0.575 [0.256, 0.732]** |
| HUST | Random Forest | **178.0 [112.1, 253.7]** | 12.2 [7.7, 17.3] | **0.340 [-0.579, 0.690]** |

Bootstrap intervals are averaged across the 5 seed-specific test-cell
bootstrap intervals; seed-to-seed standard deviations remain in
`outputs/results_v2_34feat_log/results_summary.csv`.

**Cross-dataset transfer (best per direction across all feature-set ablations):**

| Direction | Naïve R² | After 2-param rescaling on k=20 target cells |
|---|---|---|
| MATR → HUST | −10 to −20 | **−0.02 to −0.13** (50–500× MSE reduction) |
| HUST → MATR | −1.5 to −3 | **−0.05 to −0.11** |

Naïve transfer fails on every (model, feature-set, direction) combination,
even after capacity normalization aligns 71% of the feature-space shift
(Mahalanobis 13 → 3.75). The failure is concept shift (cycle-life
distributions differ), not covariate shift; a 2-parameter mean-and-slope
correction fit on a small target labeled set recovers the bulk of the loss.

---

## Repo layout

```
.
├── 0_data/                  # raw-data fetch + audit
│   ├── download_data.py
│   ├── build_matr_audit.py
│   └── build_hust_audit.py
├── 1_features/              # 34-feature capacity-only feature table
│   └── build_features.py
├── 2_models/                # within-/cross-dataset experiments
│   ├── generate_splits.py
│   ├── vif_screening.py
│   ├── run_experiments.py
│   └── metrics_utils.py
├── 3_analysis/              # shift quantification, recalibration
│   ├── shift_metrics.py
│   ├── feature_transfer_stability.py
│   ├── shap_feature_importance.py
│   ├── survival_censoring.py
│   ├── concept_shift_diagnostics.py
│   ├── target_rescaling.py
│   ├── conformal_prediction.py
│   └── summarize_conformal_results.py
├── notebooks/
│   └── run_pipeline_colab.ipynb     # Phase A (Drive-mounted, heavy I/O)
├── data/
│   ├── raw/                 # gitignored — fetched at runtime
│   └── intermediate/        # committed audit + feature CSVs (~MB)
├── splits/sop_v2/           # 70/15/15 cell splits, 5 seeds × 2 datasets
├── outputs/                 # all experiment results, JSON + summary CSV
├── legacy/                  # archived earlier code — see legacy/README.md
├── run_pipeline.py          # orchestrator (--status, --resume, --phase, --stages)
├── requirements.txt
└── README.md
```

The `legacy/` folder is preserved but **not** part of the active pipeline. It
contains earlier experiments, the Severson Nature Energy 2019 reference code,
and exploratory notebooks. See `legacy/README.md`.

---

## Two-phase workflow

| Phase | Where | Why | Time |
|---|---|---|---|
| **A — Extract** | Colab | Reads ~15–20 GB of raw `.pkl` from Drive, builds 34-feature CSVs (~500 KB) | ~5–10 min |
| **B — Model + Analysis** | Local | Reads only the feature CSVs Phase A produced; pure CPU | seconds–minutes per experiment |

Phase A is run only when feature definitions change. Phase B is iterated freely
on the committed CSVs without re-touching the raw data.

### Phase A — Colab

Drive layout expected:
```
MyDrive/
├── Braatz_NatEnergy2019/    # batch1.pkl, batch2.pkl, batch3.pkl  (MATR)
└── HUST/                    # 1-1.pkl ... 10-8.pkl                 (HUST, 77 cells)
```

Open in Colab and `Runtime → Run all`:
<https://colab.research.google.com/github/osmansafacifci/Graduation-Project-Dicle/blob/main/notebooks/run_pipeline_colab.ipynb>

The last cell triggers a ZIP download containing the audit + feature CSVs.
Unzip into `data/intermediate/` locally and commit.

### Phase B — Local

```bash
pip install -r requirements.txt
python run_pipeline.py --status                  # show what's done / missing
python run_pipeline.py --phase model             # splits + VIF + within-dataset experiments
python run_pipeline.py --phase analysis          # shift + XAI + survival + concept + rescaling + CP
```

Or call individual scripts:

```bash
# 7-model within-dataset experiments (default = log-target, 34 features)
python 2_models/run_experiments.py

# Iterative VIF drop (writes vif_kept_features.txt)
python 2_models/vif_screening.py --drop

# Cross-dataset transfer with capacity normalization
python 1_features/build_features.py --capacity-normalize
python 2_models/run_experiments.py --cross-dataset --log-target \
    --output-dir outputs/results_v2_cross_34_capnorm

# Distribution shift quantification
python 3_analysis/shift_metrics.py
python 3_analysis/shift_metrics.py --capacity-normalize
python 3_analysis/feature_transfer_stability.py
python 3_analysis/shap_feature_importance.py
python 3_analysis/survival_censoring.py

# Concept-shift diagnostics + target-mean rescaling
python 3_analysis/concept_shift_diagnostics.py
python 3_analysis/target_rescaling.py

# Standard split conformal prediction (MAPIE)
python 3_analysis/conformal_prediction.py
python 3_analysis/summarize_conformal_results.py
```

---

## Pipeline blocks (SOP §1–§7 mapping)

| SOP § | Stage | Script | Outputs |
|---|---|---|---|
| §1.1–§1.4 | Labels: Q0 = median(QD, cycles 2–5); EOL @ 0.85·Q0; censoring tracked | `0_data/build_*_audit.py` | `data/intermediate/*_audit*.csv`, `*_cycles_tidy.csv` |
| §2.1–§2.4 | 34 capacity-only features (12 SOP + 12 shape/decay + 10 entropy/FFT/2nd-deriv); z-score; VIF report; `--capacity-normalize` toggle | `1_features/build_features.py`, `2_models/vif_screening.py` | `features_sop12_combined.csv`, `vif_report.txt` |
| §3 | 70/15/15 cell-level split, 5 seeds, lifetime-quartile-stratified | `2_models/generate_splits.py` | `splits/sop_v2/*.json` |
| §4 | 7-model within-dataset lineup (Elastic Net, PLS, Random Forest, XGBoost, CatBoost, Gaussian Process, Stacking); `--log-target`, `--pca`, `--features-from` flags | `2_models/run_experiments.py` | `outputs/results_v2*/...` |
| §5.2 | Cross-dataset transfer (MATR ↔ HUST), three feature-set ablations × two directions | `2_models/run_experiments.py --cross-dataset` | `outputs/results_v2_cross_*/...` |
| §6.3 | Distribution shift: MMD with RBF + median bandwidth, Mahalanobis with pooled covariance, per-feature attribution | `3_analysis/shift_metrics.py` | `data/intermediate/shift_metrics*.json`, `shift_report*.txt` |
| **§6.3+** | Feature transfer/stability: per-feature shift, correlation stability, univariate transfer, residual-mean-adapted transfer | `3_analysis/feature_transfer_stability.py` | `data/intermediate/feature_transfer_stability*` |
| **§6.3++** | SHAP/XAI bridge: primary within-dataset model attributions joined to transfer-stability classes | `3_analysis/shap_feature_importance.py` | `data/intermediate/shap_feature_importance*`, `outputs/results_v2_shap/...` |
| **§6+** | Survival/censoring sensitivity: Kaplan-Meier curves, log-rank test, lower-bound imputation for censored MATR cells | `3_analysis/survival_censoring.py` | `data/intermediate/survival_censoring*`, `outputs/results_v2_survival/...` |
| **§6+** | Concept-shift diagnostics: cycle-life KS test + per-cell residual constant-bias decomposition | `3_analysis/concept_shift_diagnostics.py` | `data/intermediate/concept_shift_diagnostics.json` |
| **§7−** | Target-mean rescaling baseline (k=5/10/20 calibration cells) | `3_analysis/target_rescaling.py` | `outputs/results_v2_target_rescale/...` |
| §7 | Standard split conformal prediction with MAPIE (within split CP; cross source-calibrated diagnostic; cross target-calibrated CP; residual-mean target-adapted CP with separate target calibration; optional linear sensitivity) | `3_analysis/conformal_prediction.py`, `3_analysis/summarize_conformal_results.py` | `outputs/results_v2_conformal/...` |

---

## Detailed results

### Within-dataset

Primary configuration: 34 features + `--log-target` + z-score (no further
preprocessing). Mean R² across 5 seeds at N=100:

| Model | MATR R² | HUST R² |
|---|---|---|
| Elastic Net | 0.30 | 0.20 |
| PLS | 0.43 | 0.19 |
| Random Forest | 0.52 | **0.34** |
| XGBoost | 0.53 | 0.29 |
| **CatBoost** | **0.58** | 0.28 |
| Gaussian Process | 0.36 | 0.26 |
| Stacking (RF + XGB + CatBoost → Elastic Net meta) | 0.54 | 0.27 |

Reference points:
- Severson 2019 (voltage-curve features, MATR): R² ≈ 0.85–0.92
- Capacity-only literature ceiling (MATR): R² ≈ 0.6–0.7

R² is variance-bounded by lifetime spread. HUST's narrower spread (1500–3000
cycles) caps R² lower than MATR's (133–2066). HUST sMAPE (≈12%) is half of
MATR's (≈24%) — absolute error is in fact smaller in relative terms.

#### Ablation matrix (MATR best R²; HUST best R²; N=100)

| Configuration | MATR | HUST | Notes |
|---|---|---|---|
| 12 SOP, no log | 0.371 | 0.307 | original SOP §2 baseline |
| 12 SOP + log | 0.410 | 0.299 | log target rescues Elastic Net |
| 12 SOP + VIF drop (5) + log | 0.351 | 0.405 | strongest reduction |
| 24 feat + log | 0.480 | 0.367 | + 12 shape/decay features |
| 24 feat + VIF drop (8) + log | 0.515 | 0.234 | strong MATR ablation, worst HUST |
| 24 feat + PCA(0.95) + log | 0.411 | 0.433 | best HUST ablation, MATR linears blow up |
| **34 feat + log** (primary) | **0.575** | **0.340** | adds 10 entropy/FFT/2nd-deriv features |

VIF and PCA results are reported as ablations only — applying them per-dataset
would amount to cherry-picking. The primary configuration uses the same recipe
across both datasets.

### Cross-dataset transfer (§5.2)

Trained on each dataset's training split, tested on the *full* uncensored
target dataset. Best per direction at N=100:

| Direction | Feature set | Best model | MAE | R² |
|---|---|---|---|---|
| MATR → HUST | 12 / 24 / 34 | Gaussian Process | 781 | **−8.13** |
| HUST → MATR | 12 | Random Forest | **518** | **−1.53** |
| HUST → MATR | 24 | Random Forest | 552 | −1.80 |
| HUST → MATR | 34 | Gaussian Process | 569 | −2.05 |

Three observations:

1. **All transfers fail catastrophically.** R² < 0 means worse than predicting
   the target's mean cycle-life. This holds across all feature-set sizes and
   all seven models.
2. **Asymmetry.** HUST → MATR (R² ≈ −1.5) is salvageable; MATR → HUST (R² ≈ −8)
   is not. MATR has wider lifetime spread and absorbs the HUST training signal
   as a coarse prior; HUST is too narrow.
3. **More features hurt transfer.** The 34-feature set is the within-dataset
   primary (R² = 0.575) but transfers worse than the 12-feature SOP set —
   within-dataset accuracy and transferability trade off in opposite
   directions.

### Distribution shift quantification (§6.3)

| Setting | Feature set | MMD | Mahalanobis |
|---|---|---|---|
| Raw features | 12 | 0.71 | 13.1 |
| Raw features | 34 | 0.57 | 16.0 |
| Q0-normalized | 12 | 0.51 | **3.75** |
| Q0-normalized | 34 | 0.43 | 6.71 |

`Qdis_cycle10` alone contributes 10.84 σ of pooled-z mean shift (MATR Q0
≈ 1.07 Ah, HUST Q0 ≈ 1.20 Ah; tight within-dataset clustering at cycle 10).
SOP §2.3 capacity normalization closes 71% of the geometric gap.

### Feature Transfer/Stability

Feature-level analysis combines covariate shift, Spearman relationship
stability, univariate within-dataset usefulness, and residual-mean-adapted
univariate cross transfer. The most fragile features are the absolute-capacity
scale features:

| Feature | Raw shift σ | Q0-normalized shift σ | Class |
|---|---:|---:|---|
| `Qdis_cycle10` | 10.84 | 0.20 | scale-shift fragile |
| `poly2_a` | 10.67 | 0.21 | scale-shift fragile |
| `Qdis_N` | 8.21 | 0.09 | scale-shift fragile |

Least-fragile candidates include `cycle_to_98pct`, `exp_decay_k`,
`slope_linear`, `linearity_r2`, and `cycle_to_99pct`. This is the bridge to
SHAP/XAI: separate features that are important within-domain from features
whose distributions and target relationships remain stable across datasets.

### SHAP/XAI bridge

Primary SHAP analysis explains the within-dataset champion models at N=100:
MATR CatBoost and HUST Random Forest, across the five official splits. SHAP
values are in log-cycle prediction space; model metrics are reported in cycle
space and match the headline table.

| Dataset/model | Most important SHAP features | Transfer interpretation |
|---|---|---|
| HUST Random Forest | `Qdis_N`, `linearity_r2`, `Qdis_cycle10`, `poly2_a`, `slope_ratio` | The top feature is useful within HUST but scale-shift fragile; `Qdis_cycle10` and `poly2_a` also fail by scale shift. |
| MATR CatBoost | `accel_mean`, `poly2_c`, `slope_last_quarter`, `range_Qdis`, `variance_Qdis` | Several top MATR signals are relationship-unstable across datasets, especially `accel_mean`, `slope_last_quarter`, and `range_Qdis`. |

This strengthens the transfer story: the models do learn meaningful
within-domain signals, but many high-attribution features are exactly the
features that do not carry stable cross-dataset semantics.

### Geometric alignment is not prediction alignment

Re-running cross-dataset with capacity-normalized features:

| Direction | Feature set | Raw R² | Capnorm R² |
|---|---|---|---|
| MATR → HUST | 12 | −8.13 | −8.11 |
| HUST → MATR | 12 | **−1.53** | **−3.82** |
| HUST → MATR | 34 | −2.05 | −3.60 |

Capacity normalization closed 71% of the feature-space shift but did not
improve transfer; HUST → MATR even degraded. The absolute-capacity gap was
carrying dataset-identity signal that the regressor leaned on. **This is the
classical covariate-vs-concept-shift distinction** — covariate alignment
without concept alignment.

### Concept-shift diagnostics

(a) **KS two-sample test on cycle_life marginals**: KS = 0.827, p = 3.6e-34.
- MATR: n=129, mean=778, std=361, range [133, 2066]
- HUST: n=77, mean=1490, std=274, range [829, 2024]
- HUST cycles are e^0.65 ≈ 1.92× longer-lived on average.

(b) **Constant-bias decomposition** (CatBoost, N=100):

| Direction | Raw R² | After +constant bias | Constant share of SS |
|---|---|---|---|
| MATR → HUST | −11.16 | **−0.03** | **91.5%** |
| HUST → MATR | −2.73 | **+0.04** | **74.2%** |

Most of the cross-dataset failure is *just a constant offset*. The right
constant brings R² to nearly zero from R² = −11.

### Target-mean rescaling fix

Linear correction `y_corrected = a · y_raw + b` fit on k random target cells,
scored on the rest. Averaged across 5 seeds × 20 calibration draws.

**MATR → HUST** (n_target = 77):

| Model | Baseline R² | k=5 | k=10 | **k=20** |
|---|---|---|---|---|
| CatBoost | −10.00 | −1.58 | −0.29 | **−0.13** |
| **PLS** | −19.89 | −3.83 | −0.18 | **−0.02** |
| XGBoost | −10.48 | −1.33 | −0.24 | −0.13 |

**HUST → MATR** (n_target = 129):

| Model | Baseline R² | k=5 | k=10 | **k=20** |
|---|---|---|---|---|
| **Stacking** | −2.66 | −0.71 | −0.25 | **−0.05** |
| CatBoost | −3.06 | −0.87 | −0.30 | −0.07 |
| Random Forest | −2.40 | −0.99 | −0.66 | −0.06 |

Tree-based models recover from R² ≈ −10 to ≈ −0.05 with k = 20 calibration
cells — a 50× to 500× MSE reduction. R² ≈ 0 means the rescaled model now
matches the target's marginal-mean predictor, consistent with the 91.5% /
74.2% constant-bias share found in the residual decomposition.

### Conformal Prediction

Primary policy: 90% MAPIE split CP, N=100, CatBoost + Random Forest. Target
rows use `k_target=20`; adapted rows use residual-mean `k_adapter=20` on a
disjoint target subset before CP calibration.

| Scenario | Coverage | Median width | R² | Interpretation |
|---|---:|---:|---:|---|
| Within-dataset CP | 0.86–0.97 | 919–1119 | 0.28–0.58 | CP behaves normally within dataset |
| Source-calibrated cross CP | 0.15–0.31 | 919–1119 | −10.93 to −2.40 | Source CP under-covers under shift |
| Target-domain CP, no adapter | 0.89–0.91 | 1904–2568 | −11.09 to −2.37 | Coverage restored, intervals huge |
| Residual-mean target-adapted CP | 0.91 | 999–1302 | −0.41 to −0.02 | Center repaired, intervals much narrower |

At `k_adapter=20`, `k_target=20`, residual-mean adaptation reduces median
interval width by 33–60% and MAE by 55–71% relative to target-domain CP
without the adapter. Paper-ready outputs are in
`outputs/results_v2_conformal/paper_cp_summary.*`.

---

## Censoring

| Dataset | Cells | Censored at 0.85 × Q0 | Modeling cells |
|---|---|---|---|
| MATR (b1+b2+b3 strict merge) | 135 | 6 (4.4%) | 129 |
| HUST | 77 | 0 (0%) | 77 |

Censored cells are kept in the feature table with `cycle_life = NaN`;
experiments drop them.

Survival sensitivity analysis keeps those six MATR cells as right-censored
observations. Kaplan-Meier/log-rank results still show a large MATR-vs-HUST
lifetime gap: MATR KM median 773 cycles vs HUST 1513 cycles, log-rank
χ² = 61.2, p = 5.2e-15. Even the conservative lower-bound imputation
where censored MATR cells fail at their censoring times moves the MATR mean
only from 778 to 802 cycles; HUST remains 1490 cycles.

---

## Reproducing a single number

Every reported number can be recomputed from the committed feature CSVs in
`data/intermediate/` plus the split files in `splits/sop_v2/`. No raw `.pkl`
files are needed once Phase A has produced these.

```bash
# Within-dataset primary (MATR R² = 0.575, HUST R² = 0.340)
python 2_models/run_experiments.py --log-target \
    --output-dir outputs/results_v2_34feat_log

# Distribution shift (MMD = 0.71, Mahalanobis = 13.1; capnorm: 0.51, 3.75)
python 3_analysis/shift_metrics.py
python 3_analysis/shift_metrics.py --capacity-normalize
python 3_analysis/feature_transfer_stability.py
python 3_analysis/shap_feature_importance.py
python 3_analysis/survival_censoring.py

# Constant-bias decomposition (91.5% / 74.2% finding)
python 3_analysis/concept_shift_diagnostics.py

# Target-mean rescaling (R² = -10 → -0.02 with k=20)
python 3_analysis/target_rescaling.py

# Standard split conformal prediction (MAPIE)
python 3_analysis/conformal_prediction.py
python 3_analysis/summarize_conformal_results.py

# Optional sensitivity: include the two-parameter linear target adapter
python 3_analysis/conformal_prediction.py --adapter-types residual_mean linear
```

Each script writes a JSON with the per-seed numbers and a summary CSV.

---

## Roadmap

- [x] §1 Labels (Q0, EOL @ 0.85 × Q0, censoring tracking)
- [x] §2 Features (12 SOP + 12 shape/decay + 10 entropy/FFT/2nd-deriv = 34, capacity-only)
- [x] §2.3 Capacity normalization (`--capacity-normalize`)
- [x] §2.4 VIF screening (report-only by default; `--drop` for ablation)
- [x] §3 Splits (70/15/15, 5 seeds, lifetime-quartile-stratified)
- [x] §4 Within-dataset experiments (7 models, log-target, optional PCA, optional VIF subset)
- [x] §5.2 Cross-dataset experiments (raw + capacity-normalized, three feature-set ablations)
- [x] §6.3 Shift metrics (MMD, Mahalanobis, per-feature attribution)
- [x] §6.3+ Feature transfer/stability analysis
- [x] §6.3++ SHAP/XAI attribution joined to transfer-stability classes
- [x] §6+ Survival/censoring sensitivity analysis
- [x] §6+ Concept-shift diagnostics (KS test, per-cell residual constant-bias decomposition)
- [x] §7− Target-mean rescaling baseline (k = 5 / 10 / 20)
- [x] §7 Conformal prediction (Split CP, target recalibration with valid intervals)

---

## Background

- Severson et al. 2019, *Data-driven prediction of battery cycle life before
  capacity degradation*, Nature Energy.
  [doi:10.1038/s41560-019-0356-8](https://doi.org/10.1038/s41560-019-0356-8)
- Ma et al. 2022, *Real-time personalized health status prediction of
  lithium-ion batteries using deep transfer learning*, EES (HUST dataset).
- BatteryML — Microsoft's reference data preprocessing for both datasets:
  <https://github.com/microsoft/BatteryML>

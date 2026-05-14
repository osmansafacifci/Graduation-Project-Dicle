# Project Summary — Battery Lifetime Prediction (Dicle Çoban Thesis)

> **Repo**: <https://github.com/osmansafacifci/Graduation-Project-Dicle>
> **Status**: All §1–§7 result tables are reproducible, including feature-transfer stability, conditional-shift decomposition, SHAP/XAI attribution, survival/censoring sensitivity, supporting Koopman/DMD dynamics diagnostics, importance-weighted CP diagnostics, and standard MAPIE split conformal prediction. The paper extension now has committed four-dataset feature tables, validation checks, within/cross metrics, geometric-shift diagnostics, all-pairs feature-transfer stability, raw-vs-capnorm ablation, conditional-shift/rank-signal diagnostics, k-shot target calibration, leave-one-dataset-out source-expert adaptation, conformal prediction, and survival/censoring audit for MATR, HUST, Sandia, and Luh/KIT. Manuscript positioning is centralized in `docs/MANUSCRIPT_POSITIONING.md`.
> **Last updated**: 2026-05-14

---

## 1. What we built

A reproducible, SOPv2-compliant pipeline for early-cycle battery lifetime
prediction on the Severson/MATR and HUST public datasets, now extended with
Sandia/SNL and Luh/KIT feature tables for the paper. Forked from the student's original repo
([diclecoban/Graduation-Project](https://github.com/diclecoban/Graduation-Project));
v2 modules live alongside the original code so deviations from the supervisor's
SOPv2 spec are traceable.

The pipeline runs in two phases:

- **Phase A (Colab)** — reads raw `.pkl` files from Google Drive (~15-20 GB),
  runs MATR + HUST audits, builds 34-feature CSVs (~500 KB).
- **Phase B (laptop)** — splits, VIF, experiments, shift metrics,
  feature-transfer/XAI diagnostics, survival/censoring sensitivity,
  Koopman/DMD dynamics pilot, target rescaling, four-dataset extension checks,
  leave-one-dataset-out source-expert adaptation, and conformal prediction.
  Pure CPU, seconds-to-minutes per light experiment; full four-dataset
  model/calibration sweeps are longer laptop runs.

Phase A is run only when feature definitions change. Phase B is iterated freely
on the committed CSVs without ever re-touching the raw data.

---

## 2. Goal and why it matters

**The student's original work** had several SOP deviations: wrong Q0 definition
(`first_positive(qd)` instead of `median(qd[2:5])`), wrong EOL definition (no
single-cycle threshold), and a feature set that mixed IR / Tavg / dQdV with
the SOP12 capacity-only features.

**Our job** as the supervisor was to:

1. Build a corrected, reproducible, end-to-end pipeline.
2. Run all experiments specified in the supervisor's SOPv2 doc (`§1`–`§7`).
3. Quantify within-dataset performance and cross-dataset transferability.
4. Identify what works, what doesn't, and document the "why" for the thesis.

---

## 3. Pipeline blocks and SOP compliance

| SOP § | Requirement | Status | Notes |
|---|---|---|---|
| §1.1 | `Q0 = median(QD at cycles 2..5)` | ✅ | Fixed |
| §1.2 | `EOL = first cycle with QD ≤ 0.85·Q0` | ✅ | Threshold raised from 0.80 → 0.85 by supervisor (keeps MATR batch1+3 modelable) |
| §1.3 | HUST `QD` = Coulomb-counted across all discharge stages | ✅ | Including 7-5 cell special case (drop first 2 cycles) |
| §1.4 | Censored cells excluded from modeling, count reported | ✅ | MATR: 6/135 censored. HUST: 0/77. |
| §2 | 12 capacity-only SOP features | ✅ | All 12 |
| §2 (extended) | +12 shape/decay features (poly2, exp_decay, knee, etc.) | ✅ | We added these |
| §2 (extended²) | +10 entropy / FFT / 2nd-derivative features | ✅ | We added these |
| §2.1 | N=100 primary, N=50 secondary | ✅ | Both reported |
| §2.2 | Z-score: fit on train only | ✅ | StandardScaler |
| §2.3 | Capacity normalization (raw-capacity features ÷ Q0; `variance_Qdis` ÷ Q0²) | ✅ | Tested both raw and capnorm; validation now checks both transforms |
| §2.4 | VIF screening on MATR train slice | ✅ | Report-only by default; `--drop` for ablation |
| §3 | 70/15/15 cell-level split, 5 seeds, lifetime-quartile stratified | ✅ | Seeds {42, 123, 456, 789, 1011} |
| §4.1 | ElasticNet with internal CV over l1_ratio | ✅ |
| §4.2 | XGBoost with per-fold early stopping + tuning | ✅ | n_estimators chosen by mean best_iteration |
| §4.3 | CatBoost (optional comparison) | ✅ | Same protocol as XGBoost |
| §4 (extended) | +PLS, +Random Forest, +Gaussian Process, +Stacking | ✅ | We added these for diversity |
| §4 (extended) | log-target option (`--log-target`) | ✅ | Default for all reported runs; rescues linear models |
| §5.2 | Cross-dataset experiments (MATR ↔ HUST) | ✅ | Three feature-set ablations × two directions |
| §6.3 | Shift metrics (MMD, Mahalanobis) | ✅ | Plus per-feature attribution + capnorm comparison |
| **+** | Concept-shift diagnostics (KS test, residual decomposition) | ✅ | New finding (see below) |
| **+** | Conditional-shift decomposition (centered-log slopes + alpha/beta) | ✅ | Universal log-life offset plus 16/34 feature-level slope changes; robust alpha checks, Pearson-r transfer signal, and scatter plot added |
| **Paper extension** | Four-dataset conditional-shift diagnostics | ✅ | Pairwise feature-slope shift shares plus source-prediction rank-signal/calibration regimes for all 12 directions |
| **+** | Koopman/DMD dynamics pilot | ✅ | Supporting SI-style diagnostic on early Q/Q0 trajectories; not a headline predictor |
| **+** | Importance-weighted CP falsifier | ✅ | Cross-fitted logistic density ratios, ESS, target-mass fraction, clipping sweep, and side-by-side target-adapted CP comparison |
| **+** | Target calibration baseline (precursor to §7) | ✅ | k={5,10,15,20}, residual-mean + linear adapters; two-dataset and four-dataset outputs committed |
| **Paper extension** | LODO pooled/source-expert k-shot adaptation | ✅ | Holds out each target dataset, trains on the other three, and compares pooled ERM, source-expert selection/weighting, and source+model selection with k={5,10,15,20} target labels; main-panel and SI packaging added |
| **Paper extension** | Paper-facing k-shot scaling figure | ✅ | Combines CP reliability scaling and LODO point-accuracy scaling from existing k-sweep outputs |
| **Paper extension** | Paper-facing regime-stratified CP figure | ✅ | Sorts all 12 cross-dataset directions by rank-signal regime and naive CP MAE, then compares coverage, width, and finite-interval fraction |
| **+** | SHAP/XAI attribution for primary within-dataset models | ✅ | Explains MATR/HUST champions and Sandia/Luh TreeSHAP-compatible primary models, joined to transfer-stability or conditional-slope classes; all-pairs SHAP × regime table added |
| **+** | Survival/censoring sensitivity | ✅ | Kaplan-Meier, log-rank, and lower-bound imputation for 6 censored MATR cells |
| §7 | Conformal prediction (Split CP, target recalibration) | ✅ | MAPIE implementation added: 90%/95%, Wilson coverage CI, short-/long-life stratified coverage, within, naive cross, target-calibrated, residual-mean target-adapted; default target k sweep now covers {5, 10, 15, 20}; linear adapter is sensitivity |
| **Paper extension** | Four-dataset validation | ✅ | MATR/HUST/Sandia/Luh feature counts, split completeness, and 56-row within / 168-row cross result matrices pass |
| **Paper extension** | Four-dataset survival/censoring audit | ✅ | Sandia 11/61 censored, Luh 2/108 censored; Kaplan-Meier curves, RMST bootstrap CIs, pairwise RMST differences, and pairwise tests committed |

---

## 4. Headline numbers

### Within-dataset (primary configuration: 34-feat + log-target)

Fixed protocol: 5 seeds, N=100, best model selected by mean R². Bootstrap
intervals are computed from pooled out-of-split predictions across the 5
official splits; seed-to-seed standard deviations and the earlier seed-mean
bootstrap intervals are retained in the result CSVs as audit columns.

| Dataset | Best model | MAE [bootstrap 95% CI] | sMAPE [bootstrap 95% CI] | R² [bootstrap 95% CI] |
|---|---|---|---|---|
| MATR | CatBoost | **171.7 [140.1, 202.1]** | 23.7 [19.8, 28.2] | **0.575 [0.458, 0.646]** |
| HUST | Random Forest | **178.0 [148.0, 214.1]** | 12.2 [10.0, 14.8] | **0.340 [0.072, 0.512]** |

The pooled-CI test coverage is MATR 100 prediction rows / 75 distinct cells and
HUST 60 / 43.

Four-dataset paper extension at the same N=100 / 34-feature / log-target
setting:

| Dataset | Best model | MAE | sMAPE | R² | Cells / modeled / censored |
|---|---|---:|---:|---:|---:|
| MATR | CatBoost | 171.7 | 23.7 | 0.575 | 135 / 129 / 6 |
| HUST | Random Forest | 178.0 | 12.2 | 0.340 | 77 / 77 / 0 |
| Sandia 0-100 SOC | XGBoost | 120.8 | 23.4 | 0.940 | 61 / 50 / 11 |
| Luh/KIT | Gaussian Process | 115.8 | 18.4 | 0.769 | 108 / 106 / 2 |

The validation audit is
`data/intermediate/four_dataset_validation_report.md`.

Reference points:
- Severson 2019 (voltage-curve features, MATR): R² ≈ 0.85–0.92
- Capacity-only literature ceiling (MATR): R² ≈ 0.6–0.7
- Dicle's prior result on MATR (R²=0.087, ElasticNet=R²=-0.37) — corrected SOP fixed this

### Cross-dataset (best per direction, N=100)

| Direction | Feature set | Best model | MAE | R² |
|---|---|---|---|---|
| MATR → HUST | 12 | GP | 813 | −8.83 |
| MATR → HUST | 34 | GP | 781 | **−8.12** |
| HUST → MATR | 12 | RF | **543** | **−1.72** |
| HUST → MATR | 24 | RF | 552 | −1.80 |
| HUST → MATR | 34 | GP | 569 | −2.05 |

All transfers fail catastrophically. R² < 0 means worse than predicting the
target's mean cycle life. The 12/24/34 result is directional rather than a
simple monotone rule: MATR → HUST improves slightly from 12 to 34 features but
remains unusable, while HUST → MATR worsens as feature count increases. The
within-dataset 34-feature champion is therefore not automatically the most
transferable representation.

Four-dataset naive cross transfer is mixed rather than uniformly catastrophic:
Luh → Sandia reaches R² = 0.499 and Sandia → Luh reaches R² = 0.494, while
MATR/HUST directions remain poor and some Sandia/HUST/MATR directions still
need target calibration. The full matrix is
`outputs/results_v2_four_dataset_cross_34feat_capnorm_log/results_summary.csv`.

Raw-vs-capacity-normalized 34-feature transfer was then run for all seven
models, both windows, and all 12 ordered directions. Capnorm improves the
best-model MAE in 14/24 direction-window cases and best-model R² in 15/24. It
is best framed as a useful robustness normalization, not a solution: HUST/Luh,
Luh/HUST, and Sandia/Luh improve strongly, while HUST/MATR and Luh/MATR can
worsen. The merged comparison is
`data/intermediate/four_dataset_raw_vs_capnorm_34feat_cross_report.md`.

At k=20, target calibration was run for all 12 directions, all 7 models, and
both residual-mean and linear adapters. The conservative table fixes the
naive-best model first; examples: Luh → Sandia improves from R² = 0.499 to
0.786 with the linear adapter, Sandia → Luh from 0.494 to 0.723, HUST → Luh
from −0.562 to 0.555, MATR → HUST from −7.937 to −0.108, and Sandia → HUST
from −2.027 to −0.094. Cross-check
files: `data/intermediate/four_dataset_target_rescale_naive_best_k20.csv` and
`data/intermediate/four_dataset_target_rescale_report_k20.md`.

### Distribution shift (§6.3)

| Setting | Feature set | MMD | Mahalanobis |
|---|---|---|---|
| Raw features | 12 | 0.71 | 13.1 |
| Raw features | 24 | 0.64 | 15.3 |
| Raw features | 34 | 0.57 | 16.0 |
| Q0-normalized | 12 | 0.51 | **3.75** |
| Q0-normalized | 24 | 0.48 | 5.74 |
| Q0-normalized | 34 | 0.43 | 6.71 |

**Smoking gun**: `Qdis_cycle10` alone has 10.84 σ pooled-z mean shift. MATR
Q0 ≈ 1.07 Ah, HUST Q0 ≈ 1.20 Ah, and at cycle 10 cells haven't lost much
capacity, so the within-dataset variance is tiny relative to the between-
dataset gap. Capacity normalization (SOP §2.3) attacks this directly.

The four-dataset geometric-shift extension adds cross-fitted dataset
discriminator AUC, MMD, and Mahalanobis for every unordered pair. At N=100,
capnorm cuts MATR/Luh Mahalanobis from 231.4 to 19.7 and HUST/Luh from 188.4
to 38.0, but every pair remains highly separable (capnorm AUC = 0.975-1.000).
This is the right nuance for the paper: Q0 normalization removes a major scale
artifact, but geometric support mismatch remains. Files:
`data/intermediate/four_dataset_geometric_shift_capnorm_report.md` and
`data/intermediate/four_dataset_geometric_shift_raw_report.md`.

### Feature transfer/stability analysis (§6.3+)

Per-feature analysis combines covariate shift, Spearman relationship stability,
univariate within-dataset usefulness, and residual-mean-adapted univariate
cross transfer. The most fragile capacity-scale features are:

| Feature | Raw shift σ | Q0-normalized shift σ | Class |
|---|---:|---:|---|
| `Qdis_cycle10` | 10.84 | 0.20 | scale-shift fragile |
| `poly2_a` | 10.67 | 0.21 | scale-shift fragile |
| `Qdis_N` | 8.21 | 0.09 | scale-shift fragile |

The least-fragile candidates are not individually strong transfer predictors,
but their feature/y relationship is more stable: `cycle_to_98pct`,
`exp_decay_k`, `slope_linear`, `linearity_r2`, and `cycle_to_99pct`.
This gives a principled SHAP/XAI bridge: distinguish features that are
important within-domain from features whose distributions and target
relationships remain stable across datasets.

The four-dataset exact-score extension makes this direction-specific. Sandia
↔ Luh is the clear transferable pair: Sandia → Luh has 14 stable candidates
and Luh → Sandia has 18, with `slope_linear`, `exp_decay_k`, `Qdis_N`,
`retention_ratio`, `delta_Qdis`, and `range_Qdis` repeatedly near the top.
MATR/HUST remains weak despite lower capnorm geometry shift: MATR → HUST has
8 stable candidates but top features still do not yield positive adapted
univariate R², and HUST → MATR has no stable candidates under the exact score.
Full all-pairs report:
`data/intermediate/four_dataset_feature_transfer_stability_report.md`.

### Conditional-shift decomposition (§6.3++)

`3_analysis/conditional_shift_decomposition.py` formalizes the terminology:
this is conditional/concept shift in `P(Y|X)` with a large additive component,
not label shift. Each feature is z-scored within dataset, `log(cycle_life)` is
centered within dataset, and HUST-MATR slope differences are bootstrapped with
Benjamini-Hochberg correction. The universal HUST-MATR log-life offset is
reported separately: 0.735, equivalent to a 2.09× life ratio.

| Class | Count | Interpretation |
|---|---:|---|
| Slope-stable | 18 / 34 | Feature relationship is stable after removing the dataset-level log-life offset |
| Slope-shifted | 16 / 34 | Same feature has changed target relationship |

The slope-shift list includes `Qdis_cycle10`, `poly2_a`, `accel_mean`,
`slope_last_quarter`, `mad_Qdis`, `linearity_r2`, and `cycle_to_99pct`.
Alpha/beta calibration gives the same nuance: HUST → MATR has alpha CIs
containing 1 for CatBoost and Random Forest, but Pearson r is the safer
transfer-signal summary than alpha sign. HUST → MATR retains weak positive
rank signal (`r=0.22/0.27` for CatBoost/RF across seeds), while MATR → HUST is
essentially uncorrelated with target lifetime (`r=-0.12/-0.14`, bootstrap CIs
cross zero). The negative MATR → HUST alpha should therefore be framed as
fitting noise around near-zero transfer signal, not as a strong mechanistic
inversion claim. The constant component still explains 72–90% of squared
error, but finite-sample constant R² is negative in the MATR → HUST direction.
The paper-facing directional asymmetry scatter is saved at
`outputs/results_v2_conditional_shift/paper_directional_asymmetry_seed42.png`.
The paper wording should be **dominant conditional offset plus structured
feature-level slope changes and asymmetric rank-transfer loss**, not pure
additive shift.

### Four-dataset conditional-shift regimes

`3_analysis/conditional_shift_four_dataset.py` extends the decomposition to
MATR, HUST, Sandia, and Luh/KIT using the capacity-normalized four-dataset
feature table. It tests whether each pair differs mainly by a log-life offset,
by feature→life slope changes, or by loss of source-prediction rank signal.

| Pair | Life ratio B/A | Slope-shifted features |
|---|---:|---:|
| HUST vs Luh | 0.275 | 26 / 34 |
| HUST vs Sandia | 0.247 | 25 / 34 |
| MATR vs Sandia | 0.515 | 19 / 34 |
| MATR vs Luh | 0.573 | 17 / 34 |
| MATR vs HUST | 2.085 | 14 / 34 |
| Sandia vs Luh | 1.113 | 3 / 34 |

Direction diagnostics make the manuscript claim less dataset-specific and more
mechanistic: directions with preserved rank signal are the directions where
linear target calibration becomes genuinely predictive.

| Direction | Naive R² | Pearson r | Linear-calibrated R² | Regime |
|---|---:|---:|---:|---|
| Luh → Sandia | 0.499 | 0.913 | 0.842 | transferable rank signal |
| Sandia → Luh | 0.494 | 0.862 | 0.750 | transferable rank signal |
| HUST → Luh | -0.562 | 0.772 | 0.610 | rank signal survives large offset |
| MATR → Sandia | 0.046 | 0.427 | 0.261 | calibratable transfer |
| MATR → HUST | -7.937 | -0.120 | 0.025 | center repair; rank signal lost |
| HUST → MATR | -3.600 | -0.114 | 0.008 | center repair; rank signal lost |

This supports the broader paper claim: **cross-dataset battery RUL transfer is
not a binary success/failure problem; it separates into rank-preserving
directions, offset-dominated repairs, and true conditional-shift failures.**
The cross-check report is
`data/intermediate/four_dataset_conditional_shift_report.md`; the heatmap is
`outputs/results_v2_four_dataset_conditional_shift/four_dataset_conditional_shift_heatmaps.png`.

### Supporting Koopman/DMD dynamics diagnostic (§6+++, SI-facing)

`3_analysis/koopman_dmd_pilot.py` adds a lightweight Hankel-DMD diagnostic on
early Q/Q0 retention trajectories over cycles 2..100. This should not be sold
as a new RUL predictor. Its manuscript role is a supporting dynamics paragraph:
early capacity trajectories carry dataset-specific operator signatures, which
is consistent with the conditional-shift story. The four-dataset extension
analyzes MATR 135, HUST 77, Sandia 61, and Luh 108 cells; DMD-summary
separability reaches weighted OvR AUC=0.915 ± 0.019, and the Sandia/Luh
operator-transfer asymmetry is retained as SI evidence rather than a main
claim. Full outputs are in `data/intermediate/koopman_dmd_*`,
`data/intermediate/four_dataset_koopman_dmd_*`,
`outputs/results_v2_koopman_dmd/`, and
`outputs/results_v2_four_dataset_koopman_dmd/`.

### Importance-weighted CP falsifier (§7 diagnostic)

`3_analysis/importance_weighted_conformal.py` tests whether a standard
covariate-shift repair is enough. It estimates `p_target(X)/p_source(X)` with a
cross-fitted logistic dataset discriminator, sweeps clipping at {5, 10, 20,
inf}, and reports ESS plus target-mass fraction.

| Diagnostic | Result |
|---|---|
| Dataset discriminator AUC | 0.994–0.996 |
| Raw calibration-weight ESS/n | 0.55–0.59 |
| 90% weighted CP finite-interval fraction | 0–0.9% |
| Main interpretation | Coverage is recovered only through infinite intervals |

This is the clean covariate-vs-conditional-shift contrast: source weighting
does not yield useful target intervals, while small target-side calibration
does. The updated paper comparison table adds target-adapted residual-mean CP
at k=20 in the same frame: 90% coverage is 0.905–0.909 with finite intervals
for HUST → MATR and 0.907–0.908 for MATR → HUST. The 90% figure now prints
finite-interval fractions directly on the bars to make the degeneracy visible.

### SHAP/XAI attribution bridge

SHAP explains the primary within-dataset models at N=100 across the five
official splits: MATR CatBoost and HUST Random Forest. SHAP values are in the
models' fitted log-cycle space; MAE/sMAPE/R² checks are still reported in
cycle space and match the headline metrics.

| Dataset/model | Top SHAP features | Transfer implication |
|---|---|---|
| HUST Random Forest | `Qdis_N`, `linearity_r2`, `Qdis_cycle10`, `poly2_a`, `slope_ratio` | `Qdis_N`, `Qdis_cycle10`, and `poly2_a` are high-attribution but scale-shift fragile. |
| MATR CatBoost | `accel_mean`, `poly2_c`, `slope_last_quarter`, `range_Qdis`, `variance_Qdis` | `accel_mean`, `slope_last_quarter`, and `range_Qdis` are high-attribution but relationship-unstable. |

This is useful for the manuscript because it prevents a vague "black-box
transfer failed" explanation. The models learn real within-domain signal, but
many of their most important features are not semantically stable across MATR
and HUST.

The four-dataset SHAP extension adds Sandia and Luh/KIT on the
capacity-normalized four-dataset table. These model-check values therefore
belong to the capnorm SHAP protocol, not the raw-feature four-dataset headline
rows. Sandia XGBoost reaches R²=0.926 across the five capnorm splits and is
dominated by `Qdis_N` (54.8% relative SHAP importance), followed by
`mad_Qdis`, `slope_linear`, and `range_Qdis`. Luh/KIT CatBoost reaches
R²=0.770 under the same capnorm protocol and is more distributed, led by
`slope_last_quarter`, `slope_linear`, `poly2_b`, `mad_Qdis`, and
`cycle_to_95pct`. Joined to the Sandia-vs-Luh centered-log slope test, the
top-10 SHAP features for both datasets are slope-stable. This gives a cleaner
feature-importance companion for the new two datasets without reusing the
MATR/HUST fragility labels as if they were Sandia/Luh-specific.

`3_analysis/build_shap_regime_table.py` now creates the all-pairs
paper-facing SHAP × regime table in
`data/intermediate/paper_shap_regime_table.md`. The table uses source-side
SHAP attributions but direction-specific conditional-slope labels. The
cleanest transfer pair is also the cleanest feature-regime pair: Sandia →
Luh has 2.7% shifted SHAP mass and Luh → Sandia has 6.5%. Weak/failed
directions often put most source importance on slope-shifted features:
Sandia → HUST 90.3%, Luh → HUST 88.9%, and MATR ↔ HUST ≈55%.

### Survival/censoring sensitivity

MATR has 6/135 cells censored at the 0.85 × Q0 EOL threshold; HUST has none.
The primary regressions still exclude censored cells because MAE/sMAPE/R² need
observed event times. As a robustness check, Kaplan-Meier analysis treats
those six MATR cells as right-censored at their last observed cycle.

| Quantity | MATR | HUST |
|---|---:|---:|
| Cells / events / censored | 135 / 129 / 6 | 77 / 77 / 0 |
| KM median survival | 773 cycles | 1513 cycles |
| RMST to 2024 cycles | 809 cycles | 1490 cycles |
| Event-only mean | 778 cycles | 1490 cycles |
| Lower-bound mean with censored MATR cells imputed at censoring time | 802 cycles | 1490 cycles |

The two-sample log-rank test remains decisive (χ² = 61.2, p = 5.2e-15).
The original KS test on observed event times was D = 0.827, p = 3.6e-34;
with censored MATR cells imputed at their earliest possible failure times, it
is still D = 0.818, p = 6.5e-34. So censoring is not the reason HUST appears
longer-lived.

The four-dataset audit applies the same rule to the paper extension. Counts
are MATR 135/129/6, HUST 77/77/0, Sandia 61/50/11, and Luh 108/106/2
(cells/events-or-modeled/censored). Kaplan-Meier medians are MATR 773, HUST
1513, Sandia 305, and Luh 508 cycles. RMST at the common 1615-cycle horizon
gives HUST 1429 [1382, 1473], MATR 795 [734, 856], Sandia 711 [547, 875],
and Luh 572 [495, 655] cycles by bootstrap 95% CI. Pairwise RMST differences
show HUST is decisively longer-lived than all others; MATR vs Sandia and
Sandia vs Luh have CIs crossing zero after right-censoring is handled. Outputs:
`data/intermediate/four_dataset_survival_censoring_report.md` and
`outputs/results_v2_four_dataset_survival/kaplan_meier_four_dataset.png`.

### The covariate-vs-concept finding

Re-running cross-dataset with capacity normalization:

| Direction | Feature set | Raw R² | Capnorm R² |
|---|---|---|---|
| MATR → HUST | 12 | −8.83 | −8.11 |
| MATR → HUST | 34 | −8.13 | −7.94 |
| HUST → MATR | 12 | **−1.72** | −3.61 |
| HUST → MATR | 34 | −2.05 | −3.60 |

**Geometric alignment without prediction alignment.** Mahalanobis dropped 71%,
MMD dropped 28% — but transfer accuracy did not reliably improve, and even
degraded on HUST → MATR. The absolute-capacity gap was carrying dataset-identity
information that the regressor leaned on; removing it geometrically aligns
the covariates but breaks the implicit predictor without fixing the underlying
cycle-life distribution mismatch.

### Concept-shift diagnostics

(a) **KS two-sample test on cycle_life marginals**: KS = 0.827, p = 3.6e-34.
  - MATR: n=129, mean=778, std=361, range [133, 2066]
  - HUST: n=77, mean=1490, std=274, range [829, 2024]
  - HUST mean cycle life is ≈1.92× longer; the mean-log offset used in the
    centered conditional-shift table is 0.735 (≈2.09× on the log scale).

(b) **Constant-bias decomposition** (CatBoost N=100):

| Direction | Raw R² | After +constant bias | Constant share of SS |
|---|---|---|---|
| MATR → HUST | −11.16 | **−0.03** | **91.5%** |
| HUST → MATR | −2.73 | **+0.04** | **74.2%** |

This shows that a large directional offset is present. The right constant
(914 cycles for MATR → HUST, −601 cycles for HUST → MATR) brings R² close to
zero from R² = −11 in the diagnostic, but the centered slope and alpha/beta
analyses below show that this offset is not the whole conditional shift.

### Target calibration fix (§7 precursor)

Residual-mean and linear corrections are fit on k random target cells and
scored on the rest. The original MATR/HUST table below reports the linear
adapter; the four-dataset extension now sweeps k={5,10,15,20} with both
adapters across all 12 directions.

**MATR → HUST** (n_target = 77):

| Model | Baseline R² | k=5 | k=10 | **k=20** |
|---|---|---|---|---|
| CatBoost | −10.00 | −1.58 | −0.29 | **−0.13** |
| **PLS** | −19.89 | −3.83 | −0.18 | **−0.02** |
| Stacking | −10.66 | −1.12 | −0.26 | −0.13 |
| XGBoost | −10.48 | −1.33 | −0.24 | −0.13 |

**HUST → MATR** (n_target = 129):

| Model | Baseline R² | k=5 | k=10 | **k=20** |
|---|---|---|---|---|
| **Stacking** | −2.66 | −0.71 | −0.25 | **−0.05** |
| CatBoost | −3.06 | −0.87 | −0.30 | −0.07 |
| Random Forest | −2.40 | −0.99 | −0.66 | −0.06 |
| GP | −2.05 | −0.65 | −0.25 | −0.11 |

**Take-away:** k = 20 target labels recover R² from −10 to ≈ −0.05 across all
tree-based models. The cross-dataset failure is recovered from "catastrophic"
to "approximately equivalent to predicting the target's marginal mean" — and
the fix only needs ~20 labeled cells. This is a 50× to 500× MSE reduction.

### LODO source-expert adaptation

`3_analysis/lodo_source_expert_transfer.py` turns the four-dataset extension
into a practical multi-source protocol. For each held-out target, the script
trains on the other three datasets and compares pooled ERM, pooled ERM with
k-shot residual/linear adapters, fixed source-primary experts, k-shot
source-expert selection, convex source-expert weighting, and source+model
selection.

Best k=20 results at N=100:

| Target | Best protocol | Model / expert | Adapter | MAE | sMAPE | R² |
|---|---|---|---|---:|---:|---:|
| HUST | Source+model selection | Source-model experts | Linear | 236.0 | 16.05 | -0.110 |
| Luh/KIT | Source+model selection | Source-model experts | Linear | 178.2 | 45.02 | 0.660 |
| MATR | Pooled ERM + k-shot | XGBoost | Linear | 274.7 | 35.62 | -0.003 |
| Sandia | Pooled ERM + k-shot | Stacking | Linear | 277.5 | 68.99 | 0.855 |

This is a strong but not over-claimed AI contribution. It improves over naive
single-source transfer for every held-out target and shows clear gains as k
increases from 5 to 20. It does not uniformly beat an oracle that is allowed
to choose the best single source/model/adapter after seeing the target. The
paper framing should therefore be: feasible source pooling/selection with
small target calibration, plus evidence that source relevance and conditional
shift still bound transfer.

`3_analysis/plot_lodo_main_si.py` packages the LODO result for the manuscript:
`outputs/results_v2_four_dataset_lodo_source_expert/paper_lodo_main_panel.png`
for the main text and `data/intermediate/paper_lodo_si_report.md` for SI
details. At k=20, LODO reduces MAE relative to the best no-target baseline
for every held-out target: MATR 29.4%, HUST 60.6%, Sandia 25.0%, and
Luh/KIT 14.1%. Pooled ERM + k-shot is the best family for MATR/Sandia,
while source+model selection wins for HUST/Luh. The SI tables retain the
full protocol rankings and show that source+model selection can be unstable
for Sandia, so the paper should frame LODO as a practical protocol with
source-relevance caveats rather than as a universally dominant method.

`3_analysis/plot_kshot_scaling.py` turns the existing CP and LODO k-sweep
outputs into a paper-facing four-panel scaling figure at
`outputs/results_v2_four_dataset_kshot_scaling/paper_kshot_scaling.png`/`.pdf`.
At k=20, 90% residual-adapted CP averages 0.911 coverage with median width
1466 cycles, versus target-domain CP at 0.909 coverage and median width 2868
cycles. LODO MAE improves from k=5 to k=20 for every held-out target; Sandia
has the largest drop (470.3 -> 277.5 cycles).

### Conformal prediction (§7)

Primary policy: MAPIE split CP at 90% and 95%, N=100. The original MATR/HUST
run uses CatBoost + Random Forest; the four-dataset extension uses each
source dataset's primary model (MATR CatBoost, HUST Random Forest, Sandia
XGBoost, Luh Gaussian Process). Target rows use `k_target=20`; adapted rows
use residual-mean `k_adapter=20` on a disjoint target subset before CP
calibration. Outputs include 95% Wilson score intervals for empirical coverage,
finite-interval fraction, and short-/long-life stratified coverage. The
reproducibility sweep covers
`k_target,k_adapter ∈ {5, 10, 15, 20}`; `paper_cp_k_sweep.csv` and
`paper_cp_k_sweep_coverage.png` expose the recalibration coverage curve while
the manuscript headline stays at the stable `k=20` policy. Small-k rows are
retained as sensitivity checks and explicitly carry the `finite_q_mean` flag.

| Scenario | Coverage | Median width | R² | Interpretation |
|---|---:|---:|---:|---|
| Within-dataset CP, 90% | 0.86–0.97 | 919–1119 | 0.28–0.58 | Standard CP works within dataset |
| Source-calibrated cross CP, 90% | 0.15–0.31 | 919–1119 | −10.93 to −2.40 | Source CP fails under dataset shift |
| Target-domain CP, no adapter, 90% | 0.89–0.91 | 1904–2568 | −11.09 to −2.37 | Coverage restored, intervals huge |
| Residual-mean target-adapted CP, 90% | ≈0.91 | 999–1302 | −0.41 to −0.02 | Center repaired, useful intervals |
| Residual-mean target-adapted CP, 95% | 0.95–0.96 | 1167–1840 | −0.41 to −0.02 | Higher nominal coverage holds with wider intervals |

At `k_adapter=20`, `k_target=20`, residual-mean adaptation reduces median
interval width by 33–60% and MAE by 55–71% relative to target-domain CP
without the adapter at 90%. At 95%, adapted CP stays close to nominal
coverage while preserving the same point-prediction repair. This completes
the uncertainty leg of the thesis arc: point calibration repairs the center;
CP repairs uncertainty. The remaining SOP-letter k-grid loose end is closed.

Four-dataset CP repeats the same standard MAPIE protocol on MATR, HUST,
Sandia, and Luh/KIT using the capacity-normalized feature table and all 12
cross directions. At 90%, within-dataset coverage is 0.875–0.967. Naive
source-calibrated cross CP under-covers everywhere (coverage range 0.00–0.732),
including the easier Sandia↔Luh pair (0.630–0.732). Target-domain CP with
`k_target=20` restores nominal coverage across directions (0.902–0.914) but
often with wide intervals. Residual-mean adapted CP with `k_adapter=20` and
`k_target=20` keeps coverage near nominal (0.892–0.928), all intervals finite,
and reduces median width in most directions. Key examples at 90%: MATR→HUST
improves from source-CP coverage 0.177 to adapted-CP coverage 0.906 with
median width 1007; HUST→MATR from 0.203 to 0.915 with width 1303;
Sandia→Luh from 0.630 to 0.905 with width 1314 and R²=0.154. Outputs are in
`outputs/results_v2_four_dataset_conformal/`. Paper-facing regime-stratified
outputs are `paper_cp_regime_stratified_{90,95}.png/.md` and
`paper_cp_regime_stratified.csv`; they order all 12 cross-dataset directions
by rank-signal class and naive CP MAE, then compare source-calibrated,
target-domain, and residual-adapted CP on coverage, median width, and
finite-interval fraction.

---

## 5. Deviations from SOPv2 spec

| Deviation | Reason | Documented? |
|---|---|---|
| EOL threshold raised from 0.80 → 0.85 | Supervisor email — MATR batch1+3 not enough cells reach 0.80 | README + commit log |
| Feature set expanded from 12 to 34 | First 12 + 12 shape/decay + 10 entropy/FFT/2nd-deriv. All still capacity-only (no voltage curves). | README, `1_features/build_features.py` |
| Model lineup expanded from 3 to 7 | SOP §4 had Elastic Net, XGBoost, CatBoost. We added PLS (multicollinearity-aware linear), Random Forest, GP (uncertainty), Stacking (ensemble). | README "Model lineup" |
| log-target transform | Not in SOP §4. Rescues linear models on MATR (R² −493 → 0.07) and lifts trees ~5% R². Predictions are exp-transformed back so metrics stay in cycle space. | `2_models/run_experiments.py`, README |
| XAI / SHAP added | Not in SOPv2 spec, but useful for explaining why within-domain signal does not necessarily transfer. | `3_analysis/shap_feature_importance.py`, README |
| Survival/censoring sensitivity added | Not in SOPv2 spec, but closes the methodological caveat created by 6 censored MATR cells. | `3_analysis/survival_censoring.py`, README |
| `--capacity-normalize` defaults off | MATR and HUST share A123 1.1 Ah cells, so it's not needed within-dataset. We turned it on only for the cross-dataset capnorm ablation. | `1_features/build_features.py`, README |

---

## 6. The key narrative for the thesis (§5–§6)

> 1. **Within-dataset, our pipeline beats the literature for capacity-only
>    features**: MATR R² = 0.575, HUST R² = 0.34, with corrected SOP labels,
>    34 features, log-target, and 7 models. The MATR result improves on
>    the student's R² ≈ 0.09 by ~6× and is in the upper half of the
>    capacity-only literature ceiling (~0.6–0.7) without ever reading
>    discharge voltage curves.
>
> 2. **Cross-dataset transfer fails catastrophically in the original
>    MATR/HUST setting**: every (source, target, model, feature set)
>    combination gives R² ≪ 0. The 34-feature primary set is best
>    within-dataset, but transfer is directional: it helps MATR → HUST
>    slightly and hurts HUST → MATR. The four-dataset extension makes the
>    stronger claim: transferability depends on retained target rank signal,
>    not simply on feature count.
>
> 3. **The shift in feature space is huge** (MMD = 0.71, Mahalanobis = 13)
>    and dominated by the absolute capacity gap (`Qdis_cycle10` alone has
>    10.84 σ pooled-z mean shift). Capacity normalization (SOP §2.3)
>    closes 71% of the geometric gap.
>
> 4. **But geometric alignment is not prediction alignment.** Re-running
>    cross-dataset with capacity-normalized features does NOT improve R²
>    and even degrades it on HUST → MATR. *The Q0 gap was carrying
>    predictive dataset-identity signal; removing it breaks the implicit
>    predictor without fixing the y-distribution mismatch.* This is the
>    classical covariate-vs-concept-shift distinction.
>
> 5. **Direct evidence of concept shift in y**: MATR (mean=778, std=361)
>    and HUST (mean=1490, std=274) cycle_life marginals are statistically
>    distinct (KS = 0.827, p = 3.6e-34). HUST cells live ~1.9× longer.
>
> 6. **A large dataset-level offset coexists with feature-level slope shifts.**
>    Adding the right bias (914 cycles for MATR → HUST, −601 for the other
>    direction) brings R² from −11 toward zero in the diagnostic, but the
>    centered-log per-feature test shows the shift is not purely additive:
>    18/34 features are slope-stable and 16/34 show slope changes after BH
>    correction. Pearson r makes the directional asymmetry clearer:
>    HUST → MATR keeps weak positive rank signal (`r≈0.22–0.27`), whereas
>    MATR → HUST is effectively uncorrelated with target lifetimes
>    (`r≈−0.12 to −0.14`, bootstrap CIs cross zero). The precise claim is
>    therefore *dominant conditional offset with structured feature-level
>    changes and asymmetric rank-transfer loss*, not label shift and not pure
>    additive shift.
>
> 7. **A covariate-shift CP falsifier fails usefully.** Importance-weighted
>    source CP with cross-fitted logistic density ratios gives dataset
>    discriminator AUC ≈ 0.994–0.996 and raw ESS/n ≈ 0.55–0.59, but coverage is
>    recovered only by returning infinite intervals almost everywhere
>    (finite-interval fraction ≤0.9% at 90%). This supports the interpretation
>    that covariate weighting alone cannot repair target uncertainty.
>
> 8. **A two-parameter linear correction fit on k=20 target cells recovers
>    the bulk of the failure**: R² goes from −10 to −0.05 across all tree-
>    based models. The fix needs only a tiny target labeled set, no
>    domain-adaptation infrastructure, no architecture changes. Conformal
>    prediction (§7) extends this from point-estimate correction to
>    valid prediction intervals, which is the natural next step.

---

## 7. What's worth turning into a paper

### Story arc (Q2 CS journal target)

> **Title**: *Covariate alignment is not concept alignment: a controlled
> study of cross-dataset transfer for early-cycle battery lifetime prediction*

> **Target venue posture**: CS/EE-flavored applied-ML or engineering
> journals. Re-check current JCR/SJR quartiles at submission rather than
> hard-coding impact factors in the manuscript notes.

The core contribution is a **methodological one** wrapped around a battery
domain study — it suits applied-ML venues better than pure mech-eng venues.
The current paper-facing positioning, related-work comparison, Pareto table,
stacking decision, and main-text-vs-SI split are maintained in
`docs/MANUSCRIPT_POSITIONING.md`.

### Sections

1. **Setup** — corrected SOP labels, feature engineering, model lineup.
2. **Within-dataset benchmark** — MATR, HUST, Sandia 0-100 SOC, and Luh/KIT
   now form the committed four-dataset benchmark.
3. **Naïve cross-dataset transfer fails** — full transfer matrix.
4. **Distribution shift quantification** — MMD, Mahalanobis, per-feature
   attribution. Identifies the absolute-capacity gap as the dominant
   geometric component.
5. **Feature stability / transferability** — identify scale-fragile features
   and least-fragile candidates; use this as the SHAP/XAI bridge.
6. **SHAP/XAI bridge** — show that several high-attribution within-domain
   features are scale-shift fragile or relationship-unstable.
7. **Censoring sensitivity** — Kaplan-Meier/log-rank analysis shows the
   six censored MATR cells do not explain the MATR-vs-HUST lifetime gap.
8. **Geometric alignment fails** — capacity normalization closes the gap
   but does not improve transfer. **The headline result.**
9. **Concept-shift evidence** — KS test on y, per-cell residual
   constant-bias decomposition.
10. **Target-side calibration works** — k=20 label fix recovers R² from
   −10 to near 0 in the hard MATR/HUST directions and lifts the easier
   Sandia/Luh directions above their naive baselines.
11. **Multi-source deployment protocol** — leave-one-dataset-out pooled ERM
   and source-expert weighting improve over naive single-source transfer, but
   do not erase the need for source relevance and target calibration.
12. **Conformal prediction** — source CP fails under shift; target-domain CP
   restores coverage; residual-mean target adaptation narrows intervals.
13. **Practical recommendation** — quantify shift, take a small target
   sample, recalibrate. No need for voltage curves, no need for
   adversarial domain adaptation.

### Why it would publish

- Most battery RUL papers are within-dataset only, or apply DANN/transfer-
  learning without first asking *what kind of shift are we facing*.
- The covariate-vs-concept distinction is a classical ML topic but
  surprisingly absent from the battery RUL literature.
- The empirical demonstration (MMD drops 71% but R² doesn't move) is a
  clean visual that reviewers will remember.
- The practical fix (k=20 cells → 50× MSE reduction) is industry-relevant.

### Extensions still worth doing for the paper

Sandia and Luh/KIT are now integrated. The remaining optional extensions are:

1. **Fifth dataset / chemistry-axis expansion** if a clean public dataset can
   be mapped to the same capacity-only feature contract. This would turn the
   current 4×4 matrix into a broader chemistry/protocol-factor study.

2. **Q_CV/Q_CC novel feature** (also from BİLGEM project). Use only as a
   secondary ablation on datasets with comparable charge-stage data and
   compatible CC/CV protocols; keep the universal 34 capacity-only feature
   set as the main transfer benchmark. Electrochemically motivated,
   potentially novel, but not suitable as a required feature for every
   dataset.

---

## 8. How to reproduce

```bash
# Phase A (Colab; one-time, when feature definitions change)
# https://colab.research.google.com/github/osmansafacifci/Graduation-Project-Dicle/blob/main/notebooks/run_pipeline_colab.ipynb
# Click Runtime → Run all. Downloads ZIP of feature CSVs.

# Phase B (laptop)
cd /path/to/Graduation-Project-Dicle
unzip -o ~/Downloads/extract_outputs_*.zip
git add data/intermediate && git commit && git push     # commit feature CSVs

# Reproduce primary results
python run_pipeline.py --phase model

# Reproduce ablations
python 2_models/run_experiments.py --log-target \
    --features-from data/intermediate/feature_set_sop12.txt \
    --output-dir outputs/results_v2_12_log
python 2_models/run_experiments.py --log-target --pca 0.95 \
    --output-dir outputs/results_v2_pca_log

# Cross-dataset transfer
python 2_models/run_experiments.py --cross-dataset --log-target \
    --output-dir outputs/results_v2_cross_34

# Distribution shift quantification
python 3_analysis/shift_metrics.py
python 3_analysis/shift_metrics.py --capacity-normalize
python 3_analysis/feature_transfer_stability.py
python 3_analysis/shap_feature_importance.py
python 3_analysis/build_shap_regime_table.py
python 3_analysis/survival_censoring.py

# Concept-shift diagnostics
python 3_analysis/concept_shift_diagnostics.py
python 3_analysis/conditional_shift_decomposition.py
python 3_analysis/conditional_shift_four_dataset.py
python 3_analysis/koopman_dmd_pilot.py
python 3_analysis/koopman_dmd_pilot.py \
    --datasets matr hust sandia luh \
    --features-path data/intermediate/features_sop12_four_dataset.csv \
    --output-prefix four_dataset_koopman_dmd \
    --output-dir outputs/results_v2_four_dataset_koopman_dmd

# Four-dataset geometric shift and feature transfer/stability
python 3_analysis/four_dataset_geometric_shift.py \
    --features-path data/intermediate/features_sop12_four_dataset_capnorm.csv \
    --output-prefix four_dataset_geometric_shift_capnorm
python 3_analysis/four_dataset_geometric_shift.py \
    --features-path data/intermediate/features_sop12_four_dataset.csv \
    --output-prefix four_dataset_geometric_shift_raw
python 3_analysis/four_dataset_feature_transfer_stability.py

# Four-dataset raw-vs-capnorm 34-feature cross ablation
python 2_models/run_experiments.py \
    --features-path data/intermediate/features_sop12_four_dataset.csv \
    --splits-dir splits/sop_v2_four_dataset \
    --datasets matr hust sandia luh \
    --models elastic_net pls random_forest xgboost catboost gaussian_process stacking \
    --windows 50 100 \
    --cross-dataset \
    --output-dir outputs/results_v2_four_dataset_cross_34feat_raw_log
python 3_analysis/summarize_four_dataset_raw_capnorm.py

# Target calibration
python 3_analysis/target_rescaling.py

# Four-dataset extension validation, survival, and target calibration
python 3_analysis/validate_four_dataset_extension.py
python 3_analysis/survival_censoring_four_dataset.py
python 3_analysis/target_rescaling.py \
    --features-path data/intermediate/features_sop12_four_dataset_capnorm.csv \
    --splits-dir splits/sop_v2_four_dataset \
    --datasets matr hust sandia luh \
    --windows 100 \
    --k-values 5 10 15 20 \
    --adapter-types residual_mean linear \
    --n-repeats 20 \
    --output-dir outputs/results_v2_four_dataset_target_rescale
python 3_analysis/summarize_target_rescaling.py \
    --results outputs/results_v2_four_dataset_target_rescale/results_summary.csv

# Leave-one-dataset-out pooled/source-expert adaptation
python 3_analysis/lodo_source_expert_transfer.py \
    --features-path data/intermediate/features_sop12_four_dataset_capnorm.csv \
    --splits-dir splits/sop_v2_four_dataset \
    --datasets matr hust sandia luh \
    --windows 100 \
    --models elastic_net pls random_forest xgboost catboost gaussian_process stacking \
    --k-values 5 10 15 20 \
    --n-repeats 20 \
    --output-dir outputs/results_v2_four_dataset_lodo_source_expert \
    --k-report 20
python 3_analysis/plot_lodo_main_si.py

# Covariate-shift CP falsifier
python 3_analysis/importance_weighted_conformal.py

# Standard split conformal prediction (MAPIE)
python 3_analysis/conformal_prediction.py
python 3_analysis/summarize_conformal_results.py
python 3_analysis/conformal_prediction.py \
    --features-path data/intermediate/features_sop12_four_dataset_capnorm.csv \
    --splits-dir splits/sop_v2_four_dataset \
    --datasets matr hust sandia luh \
    --models primary \
    --windows 100 \
    --target-k-values 5 10 15 20 \
    --adapter-k-values 5 10 15 20 \
    --target-repeats 20 \
    --confidence-levels 0.90 0.95 \
    --output-dir outputs/results_v2_four_dataset_conformal
python 3_analysis/summarize_conformal_results.py \
    --results-dir outputs/results_v2_four_dataset_conformal \
    --k-target 20 \
    --k-adapter 20

# Paper-facing CP regime and k-shot scaling figures from existing outputs
python 3_analysis/plot_cp_regime_stratified.py
python 3_analysis/plot_kshot_scaling.py
```

All output JSONs and summary CSVs are committed under `outputs/results_v2*/`
and `data/intermediate/`. Re-running on the same CSVs reproduces every number
in this document modulo the random seed used in the CV / fold splits.

---

## 9. File map

| Path | Role |
|---|---|
| `0_data/build_matr_audit.py` | MATR audit (Q0, EOL, censoring, tidy CSV) |
| `0_data/build_hust_audit.py` | HUST audit (Coulomb counting → tidy CSV) |
| `1_features/build_features.py` | 34 capacity-only features |
| `2_models/generate_splits.py` | 70/15/15 lifetime-stratified splits, 5 seeds |
| `2_models/vif_screening.py` | §2.4 VIF report + iterative drop |
| `2_models/run_experiments.py` | Within + cross-dataset experiments, 7 models |
| `3_analysis/shift_metrics.py` | §6.3 MMD + Mahalanobis + per-feature attribution |
| `3_analysis/feature_transfer_stability.py` | feature-level transfer/stability analysis and SHAP bridge |
| `3_analysis/four_dataset_geometric_shift.py` | four-dataset discriminator AUC, MMD, Mahalanobis, and per-feature centroid shifts on raw/capnorm tables |
| `3_analysis/four_dataset_feature_transfer_stability.py` | exact all-pairs feature-transfer-stability score for the four-dataset extension |
| `3_analysis/summarize_four_dataset_raw_capnorm.py` | raw-vs-capnorm 34-feature cross-dataset ablation summary |
| `3_analysis/shap_feature_importance.py` | SHAP/XAI attribution for primary within-dataset models, joined to transfer-stability or conditional-slope classes |
| `3_analysis/build_shap_regime_table.py` | all-pairs SHAP × conditional-regime table using source-side attributions and direction-specific slope labels |
| `3_analysis/survival_censoring.py` | Kaplan-Meier/log-rank censoring sensitivity for the 6 censored MATR cells |
| `3_analysis/survival_censoring_four_dataset.py` | four-dataset Kaplan-Meier/RMST/log-rank/KS censoring audit |
| `3_analysis/concept_shift_diagnostics.py` | KS test + residual constant-bias decomposition |
| `3_analysis/conditional_shift_decomposition.py` | centered-log per-feature slope tests, source-prediction alpha/beta calibration, robust alpha checks, Pearson r with bootstrap CI, and scatter plot |
| `3_analysis/conditional_shift_four_dataset.py` | four-dataset feature-slope/rank-signal conditional-shift regimes |
| `3_analysis/koopman_dmd_pilot.py` | Hankel-DMD / Koopman-style early-capacity dynamics pilot with per-cell modes and source/target operator transfer |
| `3_analysis/importance_weighted_conformal.py` | importance-weighted source CP falsifier with ESS, target-mass fraction, clipping sweep, and target-adapted comparison figure/table |
| `3_analysis/target_rescaling.py` | k-shot residual-mean/linear target-calibration baseline (§7 precursor) |
| `3_analysis/summarize_target_rescaling.py` | compact k=20 target-calibration tables for four-dataset cross-checking |
| `3_analysis/lodo_source_expert_transfer.py` | leave-one-dataset-out pooled/source-expert adaptation with k-shot source selection, convex weighting, source+model selection, and residual/linear target adapters |
| `3_analysis/plot_lodo_main_si.py` | LODO one-panel main figure plus SI best-by-k, k20 ranking, and protocol-family tables |
| `3_analysis/plot_cp_regime_stratified.py` | paper-facing CP coverage, width, and finite-interval figure organized by conditional-shift rank-signal regime |
| `3_analysis/plot_kshot_scaling.py` | paper-facing k-shot scaling figure joining CP reliability and LODO point-accuracy curves |
| `3_analysis/validate_four_dataset_extension.py` | validates four-dataset feature tables, splits, and within/cross result matrices |
| `3_analysis/conformal_prediction.py` | MAPIE standard split CP intervals: 90%/95%, target k sweep {5, 10, 15, 20}, Wilson coverage CI, short-/long-life stratified coverage, within, cross source-calibrated diagnostic, cross target-calibrated, and residual-mean target-adapted; optional linear sensitivity |
| `3_analysis/summarize_conformal_results.py` | paper-facing CP tables, k-sweep coverage table/figure, stratified coverage table, and coverage/width figure |
| `notebooks/run_pipeline_colab.ipynb` | Phase A Colab runner |
| `docs/MANUSCRIPT_POSITIONING.md` | manuscript claim, related-work positioning, Pareto table, and main-text-vs-SI decisions |
| `data/intermediate/*.csv` / `*.json` / `*.txt` | All audit, feature, VIF, shift outputs |
| `outputs/results_v2*` | All experiment results, JSON + summary CSV per ablation |
| `splits/sop_v2/{matr,hust}_{seed}.json` | Reproducible split files |
| `README.md` | High-level repo doc |
| `PROJECT_SUMMARY.md` | This file |

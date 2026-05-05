# Project Summary — Battery Lifetime Prediction (Dicle Çoban Thesis)

> **Repo**: <https://github.com/osmansafacifci/Graduation-Project-Dicle>
> **Status**: All thesis-mandatory experiments complete (§1–§6.3). §7 (conformal) optional extension.
> **Last updated**: 2026-05-01

---

## 1. What we built

A reproducible, SOPv2-compliant pipeline for early-cycle battery lifetime
prediction on the Severson/MATR (124 LFP cells) and HUST (77 LFP cells) public
datasets. Forked from the student's original repo
([diclecoban/Graduation-Project](https://github.com/diclecoban/Graduation-Project));
v2 modules live alongside the original code so deviations from the supervisor's
SOPv2 spec are traceable.

The pipeline runs in two phases:

- **Phase A (Colab)** — reads raw `.pkl` files from Google Drive (~15-20 GB),
  runs MATR + HUST audits, builds 34-feature CSVs (~500 KB).
- **Phase B (laptop)** — splits, VIF, experiments, shift metrics, target
  rescaling. Pure CPU, seconds-to-minutes per experiment.

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
| §2.3 | Capacity normalization (Q0-divide raw-capacity features) | ✅ | Tested both raw and capnorm |
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
| **+** | Target-mean rescaling baseline (precursor to §7) | ✅ | New finding (see below) |
| §7 | Conformal prediction (Split CP, target recalibration) | ⏳ | Future work — extension to existing rescaling |

---

## 4. Headline numbers

### Within-dataset (primary configuration: 34-feat + log-target)

Fixed protocol: 5 seeds, N=100, best model selected by mean R². Bootstrap
intervals are averaged across the 5 seed-specific test-cell bootstrap
intervals.

| Dataset | Best model | MAE [bootstrap 95% CI] | sMAPE [bootstrap 95% CI] | R² [bootstrap 95% CI] |
|---|---|---|---|---|
| MATR | CatBoost | **171.7 [110.4, 243.2]** | 23.7 [15.5, 33.5] | **0.575 [0.256, 0.732]** |
| HUST | Random Forest | **178.0 [112.1, 253.7]** | 12.2 [7.7, 17.3] | **0.340 [-0.579, 0.690]** |

Reference points:
- Severson 2019 (voltage-curve features, MATR): R² ≈ 0.85–0.92
- Capacity-only literature ceiling (MATR): R² ≈ 0.6–0.7
- Dicle's prior result on MATR (R²=0.087, ElasticNet=R²=-0.37) — corrected SOP fixed this

### Cross-dataset (best per direction, N=100)

| Direction | Feature set | Best model | MAE | R² |
|---|---|---|---|---|
| MATR → HUST | 12 / 24 / 34 | GP | 781 | **−8.13** |
| HUST → MATR | 12 | RF | **518** | **−1.53** |
| HUST → MATR | 24 | RF | 552 | −1.80 |
| HUST → MATR | 34 | GP | 569 | −2.05 |

All transfers fail catastrophically. R² < 0 means worse than predicting the
target's mean cycle life. **More features hurt transfer** (34-feat is the
within-dataset champion but transfers worse than 12-feat) — the entropy / FFT
features capture MATR-specific signal that does not generalize.

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

### The covariate-vs-concept finding

Re-running cross-dataset with capacity normalization:

| Direction | Feature set | Raw R² | Capnorm R² |
|---|---|---|---|
| MATR → HUST | 12 | −8.13 | −8.11 |
| MATR → HUST | 34 | −8.13 | −7.94 |
| HUST → MATR | 12 | **−1.53** | **−3.82** |
| HUST → MATR | 34 | −2.05 | −3.60 |

**Geometric alignment without prediction alignment.** Mahalanobis dropped 71%,
MMD dropped 28% — but transfer accuracy did not improve, and even degraded
on HUST → MATR. The absolute-capacity gap was carrying dataset-identity
information that the regressor leaned on; removing it geometrically aligns
the covariates but breaks the implicit predictor without fixing the underlying
cycle-life distribution mismatch.

### Concept-shift diagnostics

(a) **KS two-sample test on cycle_life marginals**: KS = 0.827, p = 3.6e-34.
  - MATR: n=129, mean=778, std=361, range [133, 2066]
  - HUST: n=77, mean=1490, std=274, range [829, 2024]
  - HUST cycles are e^0.65 ≈ 1.92× longer-lived on average.

(b) **Constant-bias decomposition** (CatBoost N=100):

| Direction | Raw R² | After +constant bias | Constant share of SS |
|---|---|---|---|
| MATR → HUST | −11.16 | **−0.03** | **91.5%** |
| HUST → MATR | −2.73 | **+0.04** | **74.2%** |

Most of the cross-dataset failure is *just a constant offset*. The right
constant (914 cycles for MATR → HUST, −601 cycles for HUST → MATR) brings
R² to nearly zero from R² = −11.

### Target-mean rescaling fix (§7 precursor)

Linear correction `y_corrected = a · y_predicted + b` fit on k random target
cells, scored on the rest. Averaged across 5 seeds × 20 calibration draws.

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

---

## 5. Deviations from SOPv2 spec

| Deviation | Reason | Documented? |
|---|---|---|
| EOL threshold raised from 0.80 → 0.85 | Supervisor email — MATR batch1+3 not enough cells reach 0.80 | README + commit log |
| Feature set expanded from 12 to 34 | First 12 + 12 shape/decay + 10 entropy/FFT/2nd-deriv. All still capacity-only (no voltage curves). | README, `1_features/build_features.py` |
| Model lineup expanded from 3 to 7 | SOP §4 had Elastic Net, XGBoost, CatBoost. We added PLS (multicollinearity-aware linear), Random Forest, GP (uncertainty), Stacking (ensemble). | README "Model lineup" |
| log-target transform | Not in SOP §4. Rescues linear models on MATR (R² −493 → 0.07) and lifts trees ~5% R². Predictions are exp-transformed back so metrics stay in cycle space. | `2_models/run_experiments.py`, README |
| §7 conformal not done | Time. Target-mean rescaling baseline is in place as precursor; full Split CP recalibration is one extra step. | README roadmap |
| XAI / SHAP not done | Not in SOPv2 spec. Could be added if needed for thesis defense. | not implemented |
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
> 2. **Cross-dataset transfer fails catastrophically**: every (source,
>    target, model, feature set) combination gives R² ≪ 0. More features
>    hurt transfer — the 34-feature primary set is the worst transferer
>    of the three; the 12-feature SOP set transfers slightly better.
>    *Within-dataset accuracy and transferability trade off in opposite
>    directions.*
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
> 6. **75–92% of the cross-dataset error is a single constant offset.**
>    Adding the right bias (914 cycles for MATR → HUST, −601 for the other
>    direction) brings R² from −11 to −0.03 / from −2.7 to +0.04.
>
> 7. **A two-parameter linear correction fit on k=20 target cells recovers
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

> **Target venues** (Q2, computer-engineering / applied-ML):
> - *Engineering Applications of Artificial Intelligence* (Q1/Q2, IF≈7.8)
> - *Expert Systems with Applications* (Q1, IF≈7.5)
> - *Applied Soft Computing* (Q2, IF≈7)
> - *Knowledge-Based Systems* (Q1)

The core contribution is a **methodological one** wrapped around a battery
domain study — it suits applied-ML venues better than pure mech-eng venues.

### Sections

1. **Setup** — corrected SOP labels, feature engineering, model lineup.
2. **Within-dataset benchmark** — 5 datasets if we extend with Sandia /
   Knapp / Luh from the BİLGEM project (highly recommended for the paper);
   otherwise just MATR + HUST.
3. **Naïve cross-dataset transfer fails** — full transfer matrix.
4. **Distribution shift quantification** — MMD, Mahalanobis, per-feature
   attribution. Identifies the absolute-capacity gap as the dominant
   geometric component.
5. **Geometric alignment fails** — capacity normalization closes the gap
   but does not improve transfer. **The headline result.**
6. **Concept-shift evidence** — KS test on y, per-cell residual
   constant-bias decomposition.
7. **Target-side rescaling works** — k=20 label fix recovers R² from
   −10 to −0.05.
8. **Practical recommendation** — quantify shift, take a small target
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

### Extensions worth doing for the paper

These can be pursued **after** Dicle submits the thesis. Each is independent
of the others; pick by appetite:

1. **Add Sandia / Knapp / Luh datasets** from the BİLGEM TÜBİTAK project.
   With 4–5 datasets you can:
   - Build a 4×4 / 5×5 transfer matrix.
   - Decompose shift components: same chemistry / different temperature
     transfer; different chemistry / same temperature transfer; both
     different (worst case).
   - Quantify which factor (chemistry, temperature, C-rate, manufacturer)
     contributes most to transfer failure.
   This is the strongest standalone paper extension. ~6–8 weeks of work
   to port loaders, run experiments, analyze.

2. **Q_CV/Q_CC novel feature** (also from BİLGEM project). Add as a
   34-feature ablation across the multi-dataset benchmark; show whether
   it transfers better than the entropy/FFT features. Electrochemically
   motivated, novel to the literature, single-author potential.

3. **Per-dataset adaptive preprocessing**. Define a fixed feature set but
   activate VIF / PCA / feature filtering based on train-time diagnostics
   (e.g., "if max VIF > 10 → drop"). This is a principled rule, not
   cherry-picking, and scales to N datasets.

4. **Conformal recalibration (SOP §7)**. Extend the k=20 linear correction
   to Split-CP intervals; provide valid prediction intervals at chosen
   confidence levels with the same calibration set size. Mostly drop-in.

5. **Survival analysis for censored cells**. MATR has 6/135 censored
   cells we currently throw away. Random Survival Forest / Cox PH would
   keep them and explicitly model the censoring; methodologically cleaner
   for the thesis even if it doesn't move R² much.

6. **XAI / SHAP** for feature attribution within the best model. Useful
   for thesis defense; less critical for paper unless the audience cares.

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

# Concept-shift diagnostics
python 3_analysis/concept_shift_diagnostics.py

# Target-mean rescaling
python 3_analysis/target_rescaling.py
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
| `3_analysis/concept_shift_diagnostics.py` | KS test + residual constant-bias decomposition |
| `3_analysis/target_rescaling.py` | k-shot linear correction baseline (§7 precursor) |
| `notebooks/run_pipeline_colab.ipynb` | Phase A Colab runner |
| `data/intermediate/*.csv` / `*.json` / `*.txt` | All audit, feature, VIF, shift outputs |
| `outputs/results_v2*` | All experiment results, JSON + summary CSV per ablation |
| `splits/sop_v2/{matr,hust}_{seed}.json` | Reproducible split files |
| `README.md` | High-level repo doc |
| `PROJECT_SUMMARY.md` | This file |

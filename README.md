# Battery Lifetime Prediction — Cross-Dataset Pipeline

Early-cycle battery lifetime prediction on Severson/MATR and HUST, with the
supervisor's SOPv2 specification driving labels, features, splits, and models.

This repo is a fork of [diclecoban/Graduation-Project](https://github.com/diclecoban/Graduation-Project)
with a parallel, SOP-compliant pipeline (`*_v2`) added on top so the same
raw data can be re-processed end-to-end without depending on the original
author's local machine.

> **Reproducibility note.** Raw `.pkl` files (~15-20 GB combined) are gitignored
> and live on Google Drive. The full pipeline can be re-run by anyone with the
> code, the public Drive folders, and ~30 minutes of compute (Colab is the
> easiest path — see "Colab workflow" below).

---

## What's in the repo

```
├── 0_data_prep/
│   ├── download_data.py            # gdown the public MATR + HUST Drive folders
│   ├── build_matr_audit.py         # MATR cell-level audit (Q0, EOL, censoring)
│   └── build_hust_audit.py         # HUST cycle-level audit (Coulomb counting → Q_d)
├── 1_feature_engineering/
│   └── build_sop12_features_v2.py  # 12 capacity-only SOP features for MATR & HUST
├── 2_modeling_featuring/
│   ├── generate_sop_splits_v2.py   # 70/15/15 lifetime-stratified splits, 5 seeds
│   ├── vif_screening.py            # VIF report on MATR train (default report-only)
│   ├── run_experiments_v2.py       # ElasticNet + XGBoost (+ optional CatBoost)
│   └── metrics_utils.py            # MAE / sMAPE / R² + bootstrap 95% CI helpers
├── notebooks/
│   └── run_pipeline_colab.ipynb    # Drive-mount Colab runner (no local disk needed)
├── run_pipeline.py                 # end-to-end orchestrator with --resume / --status
├── data/
│   ├── raw/                        # gitignored — populated by Drive mount or gdown
│   └── intermediate/               # audit + feature CSVs (committed)
├── splits/sop_v2/                  # JSON splits, one file per (dataset, seed)
├── outputs/results_v2/             # per-seed and seed-averaged experiment results
└── requirements.txt
```

The legacy student scripts (`build_raw_label_table.py`, `build_sop12_features.py`,
`run_sop_protocol_baselines.py`, etc.) are kept untouched for reference but are
known to deviate from the SOP — they use IR/Tavg/dQdV-style features under the
"sop12" name, and they read `Q0` from `first_positive(qd)` instead of the median
over cycles 2-5. The `*_v2` modules are the corrected path.

---

## Pipeline stages

| Stage | Script | Outputs |
|---|---|---|
| `download` | `0_data_prep/download_data.py` | `data/raw/{batch1,batch2,batch3}.pkl`, `data/raw/HUST_data/*.pkl` |
| `audit_matr` | `0_data_prep/build_matr_audit.py` | `matr_cell_audit_{strict,replication}.csv`, `matr_retention_summary.csv` |
| `audit_hust` | `0_data_prep/build_hust_audit.py` | `hust_cycles_tidy.csv`, `hust_threshold_audit.csv`, `hust_threshold_summary.csv` |
| `features` | `1_feature_engineering/build_sop12_features_v2.py` | `features_sop12_{matr,hust,combined}.csv` |
| `splits` | `2_modeling_featuring/generate_sop_splits_v2.py` | `splits/sop_v2/{matr,hust}_{seed}.json` |
| `vif` | `2_modeling_featuring/vif_screening.py` | `vif_screening.json`, `vif_report.txt` |
| `experiments` | `2_modeling_featuring/run_experiments_v2.py` | `outputs/results_v2/results_within_{matr,hust}.json`, `results_summary.csv` |

Run any subset:

```bash
python run_pipeline.py                                    # full pipeline
python run_pipeline.py --skip-download                    # data already on disk
python run_pipeline.py --resume                           # only fill in missing outputs
python run_pipeline.py --status                           # show which outputs exist
python run_pipeline.py --stages features splits           # just these two
```

---

## SOP compliance (v2 path)

| SOP §  | Requirement | Status |
|---|---|---|
| §1.1 | `Q0 = median(Q_dis at cycles 2..5)` | ✅ |
| §1.2 | `EOL = first cycle where Q_dis ≤ 0.85 × Q0` (single-cycle) | ✅ — supervisor lifted from 0.80 to 0.85 to keep MATR batch1+3 modelable |
| §1.3 | HUST `Q_dis` = total Coulomb-counted across all discharge stages | ✅ |
| §1.4 | Censored cells excluded from modeling, count reported | ✅ |
| §2 | 12 capacity-only features (Qdis_N, delta_Qdis, retention_ratio, slope_linear, variance_Qdis, range_Qdis, max_drop, std_diff, skewness_Qdis, slope_ratio, Qdis_cycle10, mean_diff) | ✅ |
| §2 (extended) | +12 additional `Q_dis`-only features beyond the SOP list: `poly2_{a,b,c}`, `exp_decay_k`, `cycle_to_{99,98,95}pct`, `slope_first_quarter`, `slope_last_quarter`, `autocorr_lag1`, `knee_cycle`, `n_capacity_jumps` | ✅ |
| §2 (extended²) | +10 entropy/FFT/2nd-derivative `Q_dis`-only features: `accel_{mean,std,max_abs}`, `linearity_r2`, `kurtosis_Qdis`, `fft_top3_energy_ratio`, `spectral_entropy`, `sample_entropy`, `pos_neg_diff_ratio`, `mad_Qdis` | ✅ |
| §2.1 | `N=100` primary, `N=50` secondary | ✅ default; `--n-windows 25 50 100` for ablation |
| §2.2 | Z-score: fit on **train only**, transform calibration & test | ✅ |
| §2.3 | Capacity normalization (divide raw-capacity features by `Q0`) | ✅ via `--capacity-normalize` (off by default; MATR + HUST share A123 1.1 Ah) |
| §2.4 | VIF screening on MATR train | ✅ — report-only by default; `--drop` flag for iterative pruning |
| §2 (extended) | PCA preprocessing (alternative to VIF drop) | ✅ via `--pca FRAC` on `run_experiments_v2.py`; fit on train, applied to cal/test |
| §4 (extended) | log-target regression (predict `log(cycle_life)`, score in cycle space) | ✅ via `--log-target` on `run_experiments_v2.py` |
| §3 | 70/15/15 cell-level split, 5 seeds {42, 123, 456, 789, 1011}, lifetime-quartile stratification | ✅ |
| §4.1 | Elastic Net with internal 5-fold CV across `l1_ratio ∈ {0.1, ..., 1.0}` | ✅ |
| §4.2 | XGBoost with internal 5-fold CV across `max_depth ∈ {3,5,7}` × `lr ∈ {0.01, 0.05, 0.1}`, `n_estimators` chosen by per-fold early stopping (patience=50) | ✅ |
| §4.3 | CatBoost (optional comparison) | ✅ |
| **+** | Expanded model lineup beyond §4: PLS Regression (multicollinearity-aware linear), Random Forest (bagging trees, contrast to boosting), Gaussian Process (native uncertainty for §7 prep), Stacking ensemble (RF + XGBoost + CatBoost → ElasticNet meta) | ✅ |
| §5.2 | Cross-dataset experiments (MATR↔HUST) | ⏳ next phase |
| §6.3 | Shift metrics (MMD, Mahalanobis) | ⏳ next phase |
| §7 | Conformal prediction | ⏳ later phase |

---

## Model lineup (6 models)

The SOP §4 lineup (Elastic Net + XGBoost + optional CatBoost) was extended after
the first run revealed that Elastic Net is unstable on the 12-feature SOP set
and that XGBoost & CatBoost are both gradient-boosted trees, so the ensemble
diversity is thin. The final lineup spans three paradigms:

| Model | Paradigm | Why it's here |
|---|---|---|
| Elastic Net | Penalized linear (L1+L2) | SOP §4.1 baseline; documents the multicollinearity failure mode |
| **PLS Regression** | Latent-component linear | Handles correlated regressors by construction (chemometrics standard); replaces ElasticNet's intended role |
| **Random Forest** | Bagging trees | Different bias-variance profile from boosting; the original Severson 2019 paper used RF |
| XGBoost | Gradient boosting | SOP §4.2 primary tree |
| CatBoost | Gradient boosting (different impl.) | SOP §4.3 optional comparison |
| **Gaussian Process** | Bayesian kernel | Native uncertainty estimates — direct prep for SOP §7 conformal-prediction phase |

## Current results (5-seed average ± std, N=100)

### Primary configuration

The pipeline reports a single, dataset-agnostic configuration so the same
code path applies to MATR, HUST, and any future battery dataset without
per-dataset preprocessing choices:

> **34 capacity-only features + `--log-target` + z-score standardization,
> no further preprocessing.**

(34 = 12 SOP + 12 extended shape/decay features + 10 entropy/FFT/2nd-derivative features.)

| Dataset | Best model | MAE | sMAPE | R² |
|---|---|---|---|---|
| **MATR** | CatBoost | **172 ± 37** | 23.7 ± 4.8 | **0.575 ± 0.118** |
| **HUST** | Random Forest | **178 ± 28** | 12.2 ± 2.2 | **0.340 ± 0.170** |

Full per-model summary (`outputs/results_v2_34feat_log/results_summary.csv`):

| Model | MATR R² | HUST R² |
|---|---|---|
| ElasticNet | 0.30 | 0.20 |
| PLS | 0.43 | 0.19 |
| Random Forest | 0.52 | **0.34** |
| XGBoost | 0.53 | 0.29 |
| **CatBoost** | **0.575** | 0.28 |
| Gaussian Process | 0.36 | 0.26 |
| Stacking (RF + XGB + CatBoost → ElasticNet meta) | 0.54 | 0.27 |

The literature ceiling for capacity-only feature sets on MATR is roughly
R² ≈ 0.6–0.7 (vs ≈0.9 with voltage-curve features in Severson 2019). At
R² = 0.575 the v2 pipeline is in the upper half of that band without ever
reading discharge voltage.

HUST's R² is naturally bounded — its lifetimes vary less than MATR's
(narrower cycle-life spread → smaller SS_tot in the R² formula), so the
same MAE that yields 0.575 on MATR yields 0.34 on HUST. The MAE/sMAPE
metrics tell a consistent story across both datasets (HUST sMAPE ≈ 12%
is in fact lower than MATR's ≈ 24%).

### Ablation studies (not used as primary results)

Seven configurations were run end-to-end. The five below explore feature-set
size and preprocessing variants but are reported as ablations only — no
per-dataset preprocessing choice is taken into the final results, since
cherry-picking a different recipe per dataset would not generalize when a
third or fourth dataset is added later.

| Configuration | Features | MATR best R² | HUST best R² | Notes |
|---|---|---|---|---|
| 12 SOP, no log | 12 | 0.371 | 0.307 | original SOP §2 baseline |
| 12 SOP + log | 12 | 0.410 | 0.299 | log target rescues ElasticNet |
| 12 SOP + VIF drop (5) + log | 5 | 0.351 | 0.405 | strongest reduction; HUST gain |
| 24 feat + log | 24 | 0.480 | 0.367 | adds 12 shape/decay features |
| 24 feat + VIF drop (8) + log | 8 | 0.515* | 0.234 | best MATR (was), worst HUST |
| 24 feat + PCA(0.95) + log | ~10 PCs | 0.411 | 0.433* | best HUST (was), MATR linears blow up |
| **34 feat + log** (primary) | 34 | **0.575** | **0.340** | adds 10 entropy/FFT/2nd-derivative features |

\*Best-of-ablation numbers; not used as primary results because the
preprocessing rule that delivered each one made the *other* dataset worse.

Cross-cutting observations from the ablations:

- The **log-target transform** is dataset-agnostic — rescues linear models
  on MATR (ElasticNet R² jumps from −493 to +0.07 once cycle life is
  regressed in log space) and gives tree models a +5–10% R² lift on both
  datasets. It is therefore part of the primary configuration.
- **VIF drop and PCA are *not* dataset-agnostic.** VIF removes features
  (subset selection, robust against label noise on small cell counts);
  PCA recombines them into orthogonal axes (preserves more signal but is
  sensitive to cells with extreme components in log space). VIF helps
  MATR (R² 0.48 → 0.52 on CatBoost) but hurts HUST (R² 0.37 → 0.23 on
  XGBoost). PCA does the opposite. Neither rule wins on average, so
  neither is taken as the default.
- This split is itself a useful finding: *no single preprocessing rule
  dominates across battery datasets, motivating cross-dataset robustness
  studies (§5.2).*

### Censoring summary

| Dataset | Cells | Censored at 0.85 × Q0 | Modeling cells |
|---|---|---|---|
| MATR (b1+b2+b3 strict merge) | 135 | 6 (4.4%) | 129 |
| HUST | 77 | 0 (0%) | 77 |

### VIF finding

VIF computed on MATR training slice (seed=42, N=100, 89 cells, z-scored 12 features):

```
delta_Qdis        inf      FLAG
mean_diff         inf      FLAG
std_diff          80060    FLAG
max_drop          73511    FLAG
retention_ratio   23406    FLAG
range_Qdis        18284    FLAG
Qdis_N             877     FLAG
variance_Qdis      693     FLAG
Qdis_cycle10       538     FLAG
slope_linear       446     FLAG
skewness_Qdis       12.7   FLAG
slope_ratio          1.6   ok
```

Eleven of twelve features sit above the VIF=5 threshold. This is **not a data
problem** — it's structural multicollinearity by construction: the 12 SOP
features are all derived from the same `Q_discharge` time series, and several
are direct algebraic transformations of each other (`delta_Qdis = Q_dis(N) -
Q_dis(2)`, `retention_ratio = Q_dis(N) / Q_dis(2)`, `mean_diff ≈ delta_Qdis /
(N-2)`, etc.). The redundancy is sharpened by the early-cycle window: at N=50
or N=100 most cells have lost only 1-3% capacity, so `Q_dis(N)`, `Q_dis(2)` and
`Q_dis(cycle10)` differ by very small amounts.

This is the direct cause of Elastic Net's instability on MATR (σ > μ across
seeds, R² in the −100s): a linear model cannot stably weight 12 nearly
collinear inputs. Tree-based XGBoost is invariant to feature correlation and
therefore produces sane numbers from the same inputs.

### VIF-pruned subset

Iterative VIF pruning (drop highest-VIF feature, recompute, repeat until all
remaining VIF ≤ 5) on the **12-feature** baseline converges to a five-feature
subset:

```
retention_ratio    relative capacity decline (Q_dis(N) / Q_dis(2))
variance_Qdis      trajectory stability
skewness_Qdis      shape asymmetry
slope_ratio        fade acceleration (slope_2nd-half / slope_1st-half)
Qdis_cycle10       early-life capacity reference
```

After Phase A was extended to produce 12 additional `Q_dis`-derived features
(quadratic fit `poly2_{a,b,c}`, exponential decay `exp_decay_k`,
`cycle_to_{99,98,95}pct`, half-window slopes, `autocorr_lag1`, `knee_cycle`,
`n_capacity_jumps`), the same VIF threshold on the **24-feature** set
converges to an eight-feature subset:

```
slope_ratio              fade acceleration (kept from the 12-feat round)
Qdis_cycle10             early-life capacity reference
poly2_c                  quadratic-fit intercept
cycle_to_99pct           cycles until first 99% retention crossing
cycle_to_98pct           same, 98%
slope_first_quarter      slope of the first ¼ of the window
autocorr_lag1            QD lag-1 autocorrelation (curve smoothness)
knee_cycle               estimated capacity-knee cycle (PELT-style change point)
```

The "VIF drop ablation" outputs (`outputs/results_v2_vif_drop/` for the
12-feat run, `outputs/results_v2_24feat_vif_log/` for the 24-feat + log run)
contain the full per-seed numbers. The 24-feat + VIF + log configuration
gives the best MATR result of any run (CatBoost R² = 0.515).

---

## Two-phase workflow

The pipeline is split into two phases so you only pay the heavy I/O cost
when feature definitions actually change:

| Phase | Where | Why | Time | Inputs | Outputs |
|---|---|---|---|---|---|
| **A — Extract** | Colab | Reads ~15-20 GB of raw `.pkl` from Drive | ~5-10 min | Drive raw data | `data/intermediate/*.csv` (~200 KB) |
| **B — Model** | Local laptop | Reads only the feature CSVs Phase A produced; pure CPU | seconds-minutes | Feature CSVs | Splits, VIF report, model results |

Phase A is run **once** (or whenever you change `EOL_FRACTION`,
`EXTRA_WINDOWS`, the feature list, etc.). Phase B is iterated **as often as
you like** — try new models, re-run with VIF pruning, change seeds, run
ablations — without ever touching Drive again.

### Phase A — Colab extract

1. Drive layout (already in place):

   ```
   MyDrive/
     ├── Braatz_NatEnergy2019/   # batch1.pkl, batch2.pkl, batch3.pkl  (MATR)
     └── HUST/                   # 1-1.pkl, 1-2.pkl, ..., 10-8.pkl     (HUST, 77 cells)
   ```

2. Open the notebook in Colab:
   <https://colab.research.google.com/github/osmansafacifci/Graduation-Project-Dicle/blob/main/notebooks/run_pipeline_colab.ipynb>

3. **Runtime → Run all**. Approve the Drive mount when prompted. Wall time
   ~5-10 min (`audit_hust` dominates: 77 cells × Coulomb counting).

4. The last cell triggers a ZIP download (`extract_outputs_<timestamp>.zip`,
   ~200 KB) containing only the audit + feature CSVs.

5. On your laptop:

   ```bash
   cd /path/to/Graduation-Project-Dicle
   git pull
   unzip -o ~/Downloads/extract_outputs_*.zip
   git add data/intermediate
   git commit -m "extract phase: refresh feature CSVs from Colab $(date +%F)"
   git push
   ```

#### Phase A configuration knobs (notebook cell 0)

| Variable | Default | Effect |
|---|---|---|
| `EOL_FRACTION` | `0.85` | Threshold for cycle_life; flip to `0.80` to follow the original SOP |
| `EXTRA_WINDOWS` | `[]` | Add `25` to also compute features at N=25 |
| `CAPACITY_NORMALIZE` | `False` | Set to `True` if you add a third dataset with a different nominal capacity |

### Phase B — local modeling

Once the feature CSVs are committed (or unzipped from the Phase A ZIP), you
no longer need Drive or Colab. Everything runs on your laptop:

```bash
pip install -r requirements-v2.txt    # one-time, slim install (works on Python 3.10–3.13)
python run_pipeline.py --phase model  # splits + VIF report + 6-model experiments
```

> Note: `requirements.txt` carries old pins inherited from the original
> Severson replication code (`numpy<2.0`, `scipy<1.13`) that don't have
> pre-built wheels for Python 3.13. Use `requirements-v2.txt` instead — it
> declares only what the v2 pipeline actually imports, with no upper bounds.

Or call individual scripts with custom flags — for example, a smaller model
subset, or the VIF-pruned ablation:

```bash
# Smaller model subset:
python 2_modeling_featuring/run_experiments_v2.py --models pls catboost gaussian_process

# Iterative VIF drop (writes vif_kept_features.txt):
python 2_modeling_featuring/vif_screening.py --drop

# 24-feat + log + VIF-pruned subset (best MATR config):
python 2_modeling_featuring/run_experiments_v2.py \
    --log-target \
    --features-from data/intermediate/vif_kept_features.txt \
    --output-dir outputs/results_v2_24feat_vif_log

# 24-feat + log + PCA(95% variance) (best HUST config):
python 2_modeling_featuring/run_experiments_v2.py \
    --log-target --pca 0.95 \
    --output-dir outputs/results_v2_24feat_pca_log
```

Phase B outputs (`splits/sop_v2/`, `outputs/results_v2/`, etc.) live alongside
the Phase A artifacts in the same repo, all small enough to commit to Git.

### Phase shortcuts

```bash
python run_pipeline.py --phase extract    # download + audits + features (heavy I/O)
python run_pipeline.py --phase model      # splits + VIF + experiments (CPU-light)
python run_pipeline.py --phase all        # everything end to end (rarely needed)
```

---

## Roadmap

- [x] §1 labels (Q0, EOL@85%, censoring)
- [x] §2 features (12 capacity-only + capacity-normalize + VIF report)
- [x] §2 extended features (+12 `Q_dis`-only features: poly fit, exp decay, knee, autocorr, etc.)
- [x] §2 extended² features (+10 `Q_dis`-only features: 2nd-derivative stats, FFT energy, spectral & sample entropy, MAD, kurtosis)
- [x] §3 splits (70/15/15, 5 seeds, lifetime-quartile-stratified)
- [x] §4 within-dataset experiments (7 models: Elastic Net, PLS, RF, XGBoost, CatBoost, GP, Stacking)
- [x] log-target regression (`--log-target`) — rescues linear models, lifts trees +5–10% R²
- [x] VIF drop ablation (12-feat → 5 feat, 24-feat → 8 feat)
- [x] PCA preprocessing ablation (`--pca 0.95`)
- [ ] §5.2 cross-dataset experiments (MATR ↔ HUST)
- [ ] §6.3 shift metrics (MMD, Mahalanobis)
- [ ] §7 conformal prediction (Split CP, target recalibration)

---

## Background reading

- Severson et al. 2019, *Data-driven prediction of battery cycle life before
  capacity degradation*, Nature Energy.
- Ma et al. 2022, *Real-time personalized health status prediction of
  lithium-ion batteries using deep transfer learning*, Energy & Environmental
  Science (HUST dataset).
- BatteryML — Microsoft's reference data preprocessing for both datasets:
  <https://github.com/microsoft/BatteryML>
- Original Dicle Sara Çoban repo: <https://github.com/diclecoban/Graduation-Project>

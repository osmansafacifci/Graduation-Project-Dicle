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
| §2.1 | `N=100` primary, `N=50` secondary | ✅ default; `--n-windows 25 50 100` for ablation |
| §2.2 | Z-score: fit on **train only**, transform calibration & test | ✅ |
| §2.3 | Capacity normalization (divide raw-capacity features by `Q0`) | ✅ via `--capacity-normalize` (off by default; MATR + HUST share A123 1.1 Ah) |
| §2.4 | VIF screening on MATR train | ✅ — report-only by default; `--drop` flag for iterative pruning |
| §3 | 70/15/15 cell-level split, 5 seeds {42, 123, 456, 789, 1011}, lifetime-quartile stratification | ✅ |
| §4.1 | Elastic Net with internal 5-fold CV across `l1_ratio ∈ {0.1, ..., 1.0}` | ✅ |
| §4.2 | XGBoost with internal 5-fold CV across `max_depth ∈ {3,5,7}` × `lr ∈ {0.01, 0.05, 0.1}`, `n_estimators` chosen by per-fold early stopping (patience=50) | ✅ |
| §4.3 | CatBoost (optional comparison) | ✅ |
| **+** | Expanded model lineup beyond §4: PLS Regression (multicollinearity-aware linear), Random Forest (bagging trees, contrast to boosting), Gaussian Process (native uncertainty for §7 prep) | ✅ |
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

## Current results (5-seed average ± std)

From the first end-to-end Colab run (2026-04-26, ElasticNet + XGBoost only).
A new run with the full 6-model lineup is needed to refresh this table.

```
dataset  experiment    model        N    MAE                 sMAPE             R²
matr     matr_to_matr  elastic_net  50   733.7  ± 872.7      40.28 ± 2.89      −135.0  ± 269.0   ⚠ unstable
matr     matr_to_matr  elastic_net  100  1147.6 ± 1766.9     39.81 ± 3.15      −493.1  ± 986.3   ⚠ unstable
matr     matr_to_matr  xgboost      50   287.9  ± 42.4       34.92 ± 4.92      −0.46   ± 0.42
matr     matr_to_matr  xgboost      100  248.6  ± 45.3       33.04 ± 5.33       0.15   ± 0.21
hust     hust_to_hust  elastic_net  50   222.5  ± 26.8       14.95 ± 1.89       0.05   ± 0.17
hust     hust_to_hust  elastic_net  100  203.1  ± 33.4       13.72 ± 2.18       0.21   ± 0.21
hust     hust_to_hust  xgboost      50   200.1  ± 30.6       13.65 ± 2.34       0.08   ± 0.28
hust     hust_to_hust  xgboost      100  194.8  ± 27.3       13.31 ± 2.18       0.23   ± 0.20
```

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
therefore produces sane numbers from the same inputs. A VIF-pruned ablation
(see "VIF drop ablation" below) is planned to confirm Elastic Net recovers
once the redundant features are removed.

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
pip install -r requirements.txt    # one-time
python run_pipeline.py --phase model              # splits + VIF report + 6-model experiments
```

Or call individual scripts with custom flags — for example, a smaller model
subset, or the VIF-pruned ablation:

```bash
python 2_modeling_featuring/run_experiments_v2.py --models pls catboost gaussian_process
python 2_modeling_featuring/vif_screening.py --drop
python 2_modeling_featuring/run_experiments_v2.py \
    --features-from data/intermediate/vif_kept_features.txt \
    --output-dir outputs/results_v2_vif_drop
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
- [x] §3 splits (70/15/15, 5 seeds, lifetime-quartile-stratified)
- [x] §4 within-dataset experiments (Elastic Net, XGBoost, CatBoost optional)
- [ ] VIF drop ablation (separate experiment branch)
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

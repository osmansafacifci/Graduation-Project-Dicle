# Reproducibility

Single-page recipe to reproduce every paper-facing number in the repository.
Anything beyond this page is convenience; anything missing here is a bug —
please file an issue.

## What is reproducible

- All **within-dataset** metrics (MATR/HUST/Sandia/Luh) at N∈{50,100}
- All **cross-dataset** transfer matrices at N=100
- All **distribution-shift** metrics (MMD, Mahalanobis, per-feature shifts)
- All **conditional-shift** decompositions (centered-log slopes, BH-FDR, alpha/beta)
- All **k-shot target calibration** sweeps (k∈{5,10,15,20}, residual-mean + linear)
- All **conformal prediction** outputs (within, source-CP, target-CP,
  target-adapted CP — 12 directions × 2 confidence levels)
- All **SHAP × regime** join tables
- All **survival / Kaplan-Meier / RMST** outputs
- All **Hankel-DMD pilot** outputs
- All **LODO source-expert** outputs
- The **PyTorch 1D-CNN** baseline (deterministic up to MPS non-determinism;
  pass `--device cpu` for bit-exact runs)

## What is **not** reproducible from this repo alone

- Raw cell-level `.pkl` / `.csv` files of the four datasets — see
  [`README.md`](../README.md) §Datasets for source URLs and DOIs. These must
  be downloaded once into `data/raw/` (git-ignored). After the Phase A
  notebooks have built the per-cycle tidy CSVs, the rest of the pipeline
  runs end-to-end from `data/intermediate/` alone.

## Quick reproduction (laptop, no GPU)

```bash
# 0. clone + venv
git clone https://github.com/osmansafacifci/Graduation-Project-Dicle.git
cd Graduation-Project-Dicle
python -m venv .venv
source .venv/bin/activate

# 1. pinned dependencies (exact-version archival)
#    Includes torch==2.12.0 for the PyTorch CNN baseline and
#    SciencePlots==2.2.0 for manuscript-style matplotlib figures.
pip install -r requirements-pinned.txt

# 2. status — see which stages are already populated by the committed CSVs
python run_pipeline.py --status

# 3. core reproduction (~5–10 min on a recent laptop)
python 2_models/run_experiments.py --log-target \
    --output-dir outputs/results_v2_34feat_log
python 3_analysis/shift_metrics.py
python 3_analysis/concept_shift_diagnostics.py
python 3_analysis/conditional_shift_decomposition.py
python 3_analysis/target_rescaling.py

# 4. paper-facing aggregated artefacts (~2 min)
python 3_analysis/summarize_conformal_results.py \
    --results-dir outputs/results_v2_four_dataset_conformal \
    --k-target 20 --k-adapter 20
python 3_analysis/plot_cp_regime_stratified.py
python 3_analysis/plot_kshot_scaling.py
```

After step 3, your numbers should match the tables in
[`README.md`](../README.md) and [`PROJECT_SUMMARY.md`](../PROJECT_SUMMARY.md)
within rounding (R² to 3 decimals; MAE/sMAPE to 1 decimal). Any larger drift
is a regression and should be reported.

## Full reproduction (including all paper extensions)

```bash
# Four-dataset CP sweep (12 directions × {residual_mean, linear} × k∈{5,10,15,20})
python 3_analysis/conformal_prediction.py \
    --features-path data/intermediate/features_sop12_four_dataset_capnorm.csv \
    --splits-dir splits/sop_v2_four_dataset \
    --datasets matr hust sandia luh \
    --models primary \
    --windows 100 \
    --target-k-values 5 10 15 20 \
    --adapter-k-values 5 10 15 20 \
    --adapter-types residual_mean linear \
    --target-repeats 20 \
    --confidence-levels 0.90 0.95 \
    --output-dir outputs/results_v2_four_dataset_conformal

# LODO source-expert protocol
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

# PyTorch CNN baseline (Apple Silicon: ~60 min; CPU only: ~5–6 hours)
python 2_models/run_cnn_pytorch.py            # default: 80 units, full HP grid

# Aggregate-only rebuild from existing checkpoints (no retraining)
python 2_models/run_cnn_pytorch.py --aggregate-only
```

## Determinism notes

- All classical models use a fixed `random_state` seed per seed in
  `SEEDS = [42, 123, 456, 789, 1011]`.
- XGBoost and CatBoost early-stopping picks an `n_estimators` rounded from
  the mean of per-fold best iterations; bit-exactness can drift slightly
  between machines if `KFold` shuffling produces different fold boundaries
  due to floating-point hash-table iteration order. R² agreement to 3
  decimals is the practical guarantee.
- The PyTorch CNN uses `torch.manual_seed(seed)` and (when available) MPS
  acceleration. MPS reduces some operations non-deterministically by ≤4th
  decimal; pass `--device cpu` for fully deterministic CPU-only runs at
  ~5–6× wall-clock cost.
- Bootstrap CIs in `metrics_utils.bootstrap_metric_ci` are seeded with the
  per-row seed argument; default seed=42 in `aggregate_summary` paths.

## Phase A — raw data ingestion (one-time)

Phase A is the only step that requires the raw `.pkl` / `.csv` files. It is
intentionally separate from the rest of the pipeline so that nobody has to
re-touch raw data after the per-cycle tidy CSVs and feature tables are
committed under `data/intermediate/`. See [`notebooks/`](../notebooks/) for
the Colab notebooks that build `*_cycles_tidy.csv` from each dataset's raw
distribution. The committed `data/intermediate/` directory already contains
the products of Phase A; the notebooks are documented for completeness and
for community members who wish to regenerate from scratch with different
EOL thresholds or windows.

## Sanity check

Once you have run the pipeline locally, this script should print "PASS" for
every assertion in the paper validation harness:

```bash
python 3_analysis/validate_four_dataset_extension.py
```

If any assertion fails, the most likely causes are: (a) a different
dependency version that produced slightly different floating-point output,
(b) raw data version mismatch (the canonical version is documented in
[`data/intermediate/four_dataset_manifest.csv`](../data/intermediate/four_dataset_manifest.csv)),
or (c) a recent code change that has not yet been reflected in the validator.

## How to cite

See [`README.md`](../README.md) §"How to cite" and [`CITATION.cff`](../CITATION.cff).

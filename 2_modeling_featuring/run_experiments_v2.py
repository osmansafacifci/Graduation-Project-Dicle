"""
Within-dataset experiments on the corrected SOP12 feature set.

Setup (per the supervisor's email):
- Datasets: MATR (within-MATR), HUST (within-HUST)
- Models:   Elastic Net (CV), XGBoost (CV-tuned)
- Windows:  N ∈ {50, 100}
- Splits:   5 seeds × 70/15/15 (cell-level, lifetime-stratified, censored excluded)
- Standardization: Z-score; fit StandardScaler on TRAIN ONLY,
                   transform calibration and test sets with the same scaler.
- Metrics:  MAE, sMAPE, R², bootstrap 95% CI (per-seed and averaged)
- Target:   cycle_life (single-cycle EOL @ 0.85 × Q0; computed in the feature builder)

Inputs:
    data/intermediate/features_sop12_combined.csv  (build_sop12_features_v2.py)
    splits/sop_v2/{matr,hust}_{seed}.json          (generate_sop_splits_v2.py)

Outputs:
    outputs/results_v2/results_within_matr.json
    outputs/results_v2/results_within_hust.json
    outputs/results_v2/results_summary.csv         (compact human-readable)

Usage:
    python 2_modeling_featuring/run_experiments_v2.py
    python 2_modeling_featuring/run_experiments_v2.py --models elastic_net
    python 2_modeling_featuring/run_experiments_v2.py --windows 100
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler

# scoped to this directory so we can `from metrics_utils import ...`
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from metrics_utils import bootstrap_metric_ci, compute_metrics  # noqa: E402

# Quiet ElasticNetCV's convergence warnings on small folds (~6-8 cells).
warnings.filterwarnings("ignore", category=ConvergenceWarning)

PROJECT_ROOT = HERE.parent
FEATURES_PATH = PROJECT_ROOT / "data" / "intermediate" / "features_sop12_combined.csv"
SPLITS_DIR = PROJECT_ROOT / "splits" / "sop_v2"
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results_v2"

SEEDS = [42, 123, 456, 789, 1011]
DEFAULT_WINDOWS = [50, 100]
DEFAULT_MODELS = ["elastic_net", "xgboost"]

SOP12_FEATURE_COLS = [
    "Qdis_N", "delta_Qdis", "retention_ratio", "slope_linear",
    "variance_Qdis", "range_Qdis", "max_drop", "std_diff",
    "skewness_Qdis", "slope_ratio", "Qdis_cycle10", "mean_diff",
]


# ---------- model helpers ----------

def fit_elastic_net(X_train_scaled: np.ndarray, y_train: np.ndarray, *, seed: int):
    # CV-tuned over l1_ratio and alpha (alphas=None lets sklearn pick the path)
    n_samples = len(y_train)
    cv = max(2, min(5, n_samples - 1))
    model = ElasticNetCV(
        l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
        cv=cv,
        max_iter=50000,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X_train_scaled, y_train)
    return model


def fit_xgboost(X_train_scaled: np.ndarray, y_train: np.ndarray, *, seed: int):
    """XGBoost tuning per SOP §4.2:
        - max_depth ∈ {3, 5, 7}
        - learning_rate ∈ {0.01, 0.05, 0.1}
        - n_estimators chosen via early stopping (patience=50)
        - Inner 5-fold CV; within each fold, fit on fold-train and evaluate on
          fold-val as eval_set so early stopping triggers per fold.

    Strategy:
        1. For each (max_depth, learning_rate), run 5-fold CV. In each fold,
           fit with n_estimators_max=2000 and early_stopping_rounds=50 against
           the held-out fold; record val MAE at best_iteration and best_iteration.
        2. Pick (max_depth, learning_rate) that minimizes mean fold val MAE.
        3. Refit final model on full train with n_estimators = round(mean of
           per-fold best_iterations) — no eval_set, no early stopping at refit
           time (the iteration count is already chosen).
    """
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise SystemExit(
            "xgboost not installed. Add it to your environment (pip install xgboost)."
        ) from exc

    from sklearn.model_selection import KFold

    n_samples = len(y_train)
    n_folds = max(2, min(5, n_samples - 1))
    n_estimators_max = 2000
    patience = 50

    param_grid = [(d, lr) for d in (3, 5, 7) for lr in (0.01, 0.05, 0.1)]

    best_mae = float("inf")
    best_params: dict = {}
    best_iter_per_fold: list[int] = []
    best_fold_maes: list[float] = []

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = list(kf.split(X_train_scaled))

    for max_depth, lr in param_grid:
        fold_maes: list[float] = []
        fold_best_iters: list[int] = []
        for train_idx, val_idx in folds:
            X_tr, X_v = X_train_scaled[train_idx], X_train_scaled[val_idx]
            y_tr, y_v = y_train[train_idx], y_train[val_idx]
            model = XGBRegressor(
                n_estimators=n_estimators_max,
                max_depth=max_depth,
                learning_rate=lr,
                random_state=seed,
                n_jobs=-1,
                tree_method="hist",
                objective="reg:squarederror",
                early_stopping_rounds=patience,
                eval_metric="mae",
                verbosity=0,
            )
            model.fit(X_tr, y_tr, eval_set=[(X_v, y_v)], verbose=False)
            pred_v = model.predict(X_v)
            fold_maes.append(float(np.mean(np.abs(y_v - pred_v))))
            best_iter = int(getattr(model, "best_iteration", n_estimators_max - 1)) + 1
            fold_best_iters.append(best_iter)

        mean_mae = float(np.mean(fold_maes))
        if mean_mae < best_mae:
            best_mae = mean_mae
            best_params = {"max_depth": max_depth, "learning_rate": lr}
            best_iter_per_fold = fold_best_iters
            best_fold_maes = fold_maes

    n_estimators_final = max(50, int(round(float(np.mean(best_iter_per_fold)))))
    final_model = XGBRegressor(
        n_estimators=n_estimators_final,
        max_depth=best_params["max_depth"],
        learning_rate=best_params["learning_rate"],
        random_state=seed,
        n_jobs=-1,
        tree_method="hist",
        objective="reg:squarederror",
        verbosity=0,
    )
    final_model.fit(X_train_scaled, y_train, verbose=False)

    info = {
        **best_params,
        "n_estimators": n_estimators_final,
        "cv_folds": n_folds,
        "cv_patience": patience,
        "cv_n_estimators_max": n_estimators_max,
        "cv_mae_per_fold": best_fold_maes,
        "cv_mae_mean": float(best_mae),
        "cv_best_iter_per_fold": best_iter_per_fold,
    }
    return final_model, info


# ---------- per-seed evaluation ----------

def evaluate_split(
    df: pd.DataFrame,
    split: dict,
    *,
    n_cycles: int,
    models: list[str],
    seed: int,
) -> dict:
    """Train on `split['train']`, evaluate on `split['test']`. Returns per-model metrics."""
    subset = df[(df["n_cycles"] == n_cycles) & (df["is_censored"] == 0)].copy()

    train_df = subset[subset["cell_id"].isin(split["train"])]
    cal_df = subset[subset["cell_id"].isin(split["calibration"])]
    test_df = subset[subset["cell_id"].isin(split["test"])]

    if len(train_df) < 5 or len(test_df) < 2:
        return {"_skipped": True, "reason": "too few cells",
                "train_cells": int(len(train_df)), "test_cells": int(len(test_df))}

    X_train = train_df[SOP12_FEATURE_COLS].to_numpy()
    y_train = train_df["cycle_life"].to_numpy()
    X_cal = cal_df[SOP12_FEATURE_COLS].to_numpy()
    y_cal = cal_df["cycle_life"].to_numpy()
    X_test = test_df[SOP12_FEATURE_COLS].to_numpy()
    y_test = test_df["cycle_life"].to_numpy()

    # Z-score: fit on TRAIN, transform cal/test
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_cal_s = scaler.transform(X_cal) if len(X_cal) else X_cal
    X_test_s = scaler.transform(X_test)

    out: dict = {
        "n_cycles": int(n_cycles),
        "train_cells": int(len(train_df)),
        "calibration_cells": int(len(cal_df)),
        "test_cells": int(len(test_df)),
    }

    if "elastic_net" in models:
        enet = fit_elastic_net(X_train_s, y_train, seed=seed)
        pred = enet.predict(X_test_s)
        m = compute_metrics(y_test, pred)
        m["bootstrap_95_ci"] = bootstrap_metric_ci(y_test, pred, seed=seed)
        m["best_alpha"] = float(enet.alpha_)
        m["best_l1_ratio"] = float(enet.l1_ratio_)
        if len(X_cal_s):
            cal_pred = enet.predict(X_cal_s)
            m["calibration_MAE"] = float(np.mean(np.abs(y_cal - cal_pred)))
        out["elastic_net"] = m

    if "xgboost" in models:
        xgb_model, xgb_info = fit_xgboost(X_train_s, y_train, seed=seed)
        pred = xgb_model.predict(X_test_s)
        m = compute_metrics(y_test, pred)
        m["bootstrap_95_ci"] = bootstrap_metric_ci(y_test, pred, seed=seed)
        m["tuning"] = xgb_info  # max_depth, learning_rate, n_estimators, CV stats
        if len(X_cal_s):
            cal_pred = xgb_model.predict(X_cal_s)
            m["calibration_MAE"] = float(np.mean(np.abs(y_cal - cal_pred)))
        out["xgboost"] = m

    return out


# ---------- aggregation ----------

def aggregate_seeds(per_seed: dict, model: str, n_cycles: int) -> dict:
    mae, smape, r2 = [], [], []
    for seed, seed_block in per_seed.items():
        model_block = seed_block.get(str(n_cycles), {}).get(model)
        if isinstance(model_block, dict) and "MAE" in model_block:
            mae.append(model_block["MAE"])
            smape.append(model_block["SMAPE"])
            r2.append(model_block["R2"])
    if not mae:
        return {}
    return {
        "MAE_mean": float(np.mean(mae)),
        "MAE_std": float(np.std(mae)),
        "SMAPE_mean": float(np.mean(smape)),
        "SMAPE_std": float(np.std(smape)),
        "R2_mean": float(np.mean(r2)),
        "R2_std": float(np.std(r2)),
        "n_seeds": len(mae),
    }


# ---------- main ----------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", default=["matr", "hust"], choices=["matr", "hust"])
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, choices=["elastic_net", "xgboost"])
    parser.add_argument("--windows", type=int, nargs="+", default=DEFAULT_WINDOWS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not FEATURES_PATH.exists():
        print(f"[error] {FEATURES_PATH} missing — run build_sop12_features_v2.py first.")
        return 1
    df = pd.read_csv(FEATURES_PATH)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []

    for dataset in args.datasets:
        sub = df[df["dataset"] == dataset]
        if sub.empty:
            print(f"[skip] dataset={dataset}: no rows in features table")
            continue

        print(f"\n========== within-{dataset} ==========")
        bundle = {
            "protocol": "SOP_within_dataset_v2",
            "dataset": dataset,
            "feature_set": "sop12 (capacity-only, 12 features)",
            "feature_columns": SOP12_FEATURE_COLS,
            "target": "cycle_life @ 0.85 * Q0 (single-cycle EOL)",
            "split_ratios": list((0.70, 0.15, 0.15)),
            "standardization": "z-score, fit on train, transform cal/test",
            "seeds": SEEDS,
            "windows": args.windows,
            "models": args.models,
            "per_seed": {},
            "averaged": {},
        }

        for seed in SEEDS:
            split_path = SPLITS_DIR / f"{dataset}_{seed}.json"
            if not split_path.exists():
                print(f"[warn] missing split file: {split_path}")
                continue
            with split_path.open() as f:
                split = json.load(f)

            per_window: dict[str, dict] = {}
            for n in args.windows:
                print(f"  seed={seed} N={n}")
                result = evaluate_split(sub, split, n_cycles=n, models=args.models, seed=seed)
                per_window[str(n)] = result
            bundle["per_seed"][str(seed)] = per_window

        # average across seeds
        for n in args.windows:
            for model in args.models:
                agg = aggregate_seeds(bundle["per_seed"], model, n)
                if agg:
                    key = f"{dataset}_to_{dataset}_{model}_N{n}"
                    bundle["averaged"][key] = agg
                    summary_rows.append({
                        "dataset": dataset,
                        "experiment": f"{dataset}_to_{dataset}",
                        "model": model,
                        "n_cycles": n,
                        **agg,
                    })

        out_path = RESULTS_DIR / f"results_within_{dataset}.json"
        with out_path.open("w") as f:
            json.dump(bundle, f, indent=2)
        print(f"\n[save] {out_path}")

    # combined human-readable summary
    summary_df = pd.DataFrame(summary_rows)
    summary_path = RESULTS_DIR / "results_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"[save] {summary_path}")

    print("\n=== SUMMARY (averaged across 5 seeds) ===")
    print(f"{'experiment':<18} {'model':<14} {'N':>3}   {'MAE':>14}  {'sMAPE':>14}  {'R²':>14}")
    print("-" * 90)
    for _, r in summary_df.sort_values(["dataset", "n_cycles", "model"]).iterrows():
        print(
            f"{r['experiment']:<18} {r['model']:<14} {int(r['n_cycles']):>3}   "
            f"{r['MAE_mean']:>7.1f}±{r['MAE_std']:<5.1f}  "
            f"{r['SMAPE_mean']:>7.2f}±{r['SMAPE_std']:<5.2f}  "
            f"{r['R2_mean']:>7.3f}±{r['R2_std']:<5.3f}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())

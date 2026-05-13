"""
Within- and cross-dataset experiments on the corrected capacity-only feature set.

Setup (per the supervisor's email):
- Datasets: MATR (within-MATR), HUST (within-HUST)
- Models:   Elastic Net, PLS, Random Forest, XGBoost, CatBoost, GP, Stacking
- Windows:  N ∈ {50, 100}
- Splits:   5 seeds × 70/15/15 (cell-level, lifetime-stratified, censored excluded)
- Standardization: Z-score; fit StandardScaler on TRAIN ONLY,
                   transform calibration and test sets with the same scaler.
- Metrics:  MAE, sMAPE, R², bootstrap 95% CI (per-seed and averaged)
- Target:   cycle_life (single-cycle EOL @ 0.85 × Q0; computed in the feature builder)

Inputs:
    data/intermediate/features_sop12_combined.csv  (build_features.py)
    splits/sop_v2/{matr,hust}_{seed}.json          (generate_splits.py)

Outputs:
    outputs/results_v2_34feat_log/results_within_matr.json
    outputs/results_v2_34feat_log/results_within_hust.json
    outputs/results_v2_34feat_log/results_summary.csv  (headline protocol)

Usage:
    python 2_models/run_experiments.py            # default: 34 features + log-target
    python 2_models/run_experiments.py --models elastic_net
    python 2_models/run_experiments.py --windows 100
    python 2_models/run_experiments.py --no-log-target --output-dir outputs/results_v2_no_log
    python 2_models/run_experiments.py \
        --features-path data/intermediate/features_sop12_four_dataset.csv \
        --splits-dir splits/sop_v2_four_dataset \
        --datasets matr hust sandia luh \
        --output-dir outputs/results_v2_four_dataset_within_34feat_log
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
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler

# scoped to this directory so we can `from metrics_utils import ...`
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from metrics_utils import bootstrap_metric_ci, compute_metrics, to_cycles  # noqa: E402

# Quiet ElasticNetCV's convergence warnings on small folds (~6-8 cells).
warnings.filterwarnings("ignore", category=ConvergenceWarning)

PROJECT_ROOT = HERE.parent
FEATURES_PATH = PROJECT_ROOT / "data" / "intermediate" / "features_sop12_combined.csv"
SPLITS_DIR = PROJECT_ROOT / "splits" / "sop_v2"
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results_v2_34feat_log"

SEEDS = [42, 123, 456, 789, 1011]
DEFAULT_WINDOWS = [50, 100]
ALL_DATASETS = ["matr", "hust", "sandia", "luh"]
ALL_MODELS = ["elastic_net", "pls", "random_forest", "xgboost", "catboost", "gaussian_process", "stacking"]
DEFAULT_MODELS = ALL_MODELS  # full lineup (stacking included)

SOP12_FEATURE_COLS = [
    "Qdis_N", "delta_Qdis", "retention_ratio", "slope_linear",
    "variance_Qdis", "range_Qdis", "max_drop", "std_diff",
    "skewness_Qdis", "slope_ratio", "Qdis_cycle10", "mean_diff",
]

EXTENDED_FEATURE_COLS = [
    "poly2_a", "poly2_b", "poly2_c", "exp_decay_k",
    "cycle_to_99pct", "cycle_to_98pct", "cycle_to_95pct",
    "slope_first_quarter", "slope_last_quarter",
    "autocorr_lag1", "knee_cycle", "n_capacity_jumps",
]

# Reserved (non-feature) CSV columns
META_COLS = {
    "dataset", "cell_id", "n_cycles", "q0", "cycle_life",
    "is_censored", "capacity_normalized",
}


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


def fit_pls(X_train_scaled: np.ndarray, y_train: np.ndarray, *, seed: int):
    """PLS regression: latent-component projection — multicollinearity-aware
    linear baseline. Tunes n_components via inner 5-fold CV on MAE.
    """
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.model_selection import KFold

    n_samples, n_features = X_train_scaled.shape
    n_folds = max(2, min(5, n_samples - 1))
    candidates = [c for c in (1, 2, 3, 4, 5, 6, 8, 10, 12) if 1 <= c <= min(n_features, n_samples - 1)]

    best_mae = float("inf")
    best_n: int = candidates[0]
    cv_scores: dict[int, float] = {}

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = list(kf.split(X_train_scaled))

    for n_comp in candidates:
        fold_maes: list[float] = []
        for tr_idx, va_idx in folds:
            X_tr, X_va = X_train_scaled[tr_idx], X_train_scaled[va_idx]
            y_tr, y_va = y_train[tr_idx], y_train[va_idx]
            model = PLSRegression(n_components=n_comp, scale=False)
            model.fit(X_tr, y_tr)
            pred = model.predict(X_va).ravel()
            fold_maes.append(float(np.mean(np.abs(y_va - pred))))
        cv_mae = float(np.mean(fold_maes))
        cv_scores[n_comp] = cv_mae
        if cv_mae < best_mae:
            best_mae = cv_mae
            best_n = n_comp

    final = PLSRegression(n_components=best_n, scale=False)
    final.fit(X_train_scaled, y_train)
    info = {
        "n_components": int(best_n),
        "cv_folds": n_folds,
        "cv_mae_mean": float(best_mae),
        "cv_mae_per_n_components": {int(k): float(v) for k, v in cv_scores.items()},
        "candidates": candidates,
    }
    return final, info


def fit_random_forest(X_train_scaled: np.ndarray, y_train: np.ndarray, *, seed: int):
    """Random Forest: bagging-based ensemble — methodological alternative
    to gradient boosting. GridSearchCV over depth and leaf size.
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import GridSearchCV

    n_samples = len(y_train)
    cv = max(2, min(5, n_samples - 1))
    base = RandomForestRegressor(
        n_estimators=500,
        random_state=seed,
        n_jobs=-1,
    )
    grid = GridSearchCV(
        base,
        param_grid={
            "max_depth": [None, 5, 10, 20],
            "min_samples_leaf": [1, 2, 5],
        },
        cv=cv,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )
    grid.fit(X_train_scaled, y_train)
    info = {
        **grid.best_params_,
        "n_estimators": 500,
        "cv_folds": cv,
        "cv_mae_mean": float(-grid.best_score_),
    }
    return grid.best_estimator_, info


def fit_gaussian_process(X_train_scaled: np.ndarray, y_train: np.ndarray, *, seed: int):
    """Gaussian Process Regression with RBF + White noise kernel.
    Native uncertainty (useful prep for SOP §7 conformal-prediction phase).
    Internal kernel hyperparameter optimization makes outer tuning unnecessary.
    """
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel

    kernel = ConstantKernel(1.0, (1e-2, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
    kernel = kernel + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-6, 1e2))

    model = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        n_restarts_optimizer=5,
        random_state=seed,
    )
    model.fit(X_train_scaled, y_train)
    info = {
        "kernel": str(model.kernel_),
        "log_marginal_likelihood": float(model.log_marginal_likelihood(model.kernel_.theta)),
        "n_restarts_optimizer": 5,
    }
    return model, info


def fit_catboost(X_train_scaled: np.ndarray, y_train: np.ndarray, *, seed: int):
    """CatBoost (SOP §4.3 — optional comparison model).

    Same outer protocol as XGBoost: per-fold early stopping in 5-fold CV
    over (depth, learning_rate); refit on full train at mean best iteration.
    """
    try:
        from catboost import CatBoostRegressor
    except ImportError as exc:
        raise SystemExit(
            "catboost not installed. Add it to your environment (pip install catboost)."
        ) from exc

    from sklearn.model_selection import KFold

    n_samples = len(y_train)
    n_folds = max(2, min(5, n_samples - 1))
    n_estimators_max = 2000
    patience = 50

    param_grid = [(d, lr) for d in (4, 6, 8) for lr in (0.01, 0.05, 0.1)]

    best_mae = float("inf")
    best_params: dict = {}
    best_iter_per_fold: list[int] = []
    best_fold_maes: list[float] = []

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = list(kf.split(X_train_scaled))

    for depth, lr in param_grid:
        fold_maes: list[float] = []
        fold_best_iters: list[int] = []
        for train_idx, val_idx in folds:
            X_tr, X_v = X_train_scaled[train_idx], X_train_scaled[val_idx]
            y_tr, y_v = y_train[train_idx], y_train[val_idx]
            model = CatBoostRegressor(
                iterations=n_estimators_max,
                depth=depth,
                learning_rate=lr,
                loss_function="MAE",
                random_seed=seed,
                early_stopping_rounds=patience,
                verbose=False,
                allow_writing_files=False,
            )
            model.fit(X_tr, y_tr, eval_set=(X_v, y_v), verbose=False)
            pred_v = model.predict(X_v)
            fold_maes.append(float(np.mean(np.abs(y_v - pred_v))))
            best_iter = int(model.get_best_iteration() or n_estimators_max - 1) + 1
            fold_best_iters.append(best_iter)

        mean_mae = float(np.mean(fold_maes))
        if mean_mae < best_mae:
            best_mae = mean_mae
            best_params = {"depth": depth, "learning_rate": lr}
            best_iter_per_fold = fold_best_iters
            best_fold_maes = fold_maes

    iterations_final = max(50, int(round(float(np.mean(best_iter_per_fold)))))
    final_model = CatBoostRegressor(
        iterations=iterations_final,
        depth=best_params["depth"],
        learning_rate=best_params["learning_rate"],
        loss_function="MAE",
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
    )
    final_model.fit(X_train_scaled, y_train, verbose=False)

    info = {
        **best_params,
        "iterations": iterations_final,
        "cv_folds": n_folds,
        "cv_patience": patience,
        "cv_iterations_max": n_estimators_max,
        "cv_mae_per_fold": best_fold_maes,
        "cv_mae_mean": float(best_mae),
        "cv_best_iter_per_fold": best_iter_per_fold,
    }
    return final_model, info


def fit_stacking(X_train_scaled: np.ndarray, y_train: np.ndarray, *, seed: int):
    """Stacking ensemble: CatBoost + Random Forest + XGBoost as base learners,
    Elastic Net as meta-learner. Base learners are fit on K-fold splits of the
    training set; their out-of-fold predictions feed the meta-learner. Final
    base learners are then refit on the full training set so the meta-learner
    can score new data via base→meta in one pass.

    Lighter base-learner configurations than the standalone protocols above —
    the diversity gain is what carries stacking, not razor-tuned individual
    models, and re-running each model's full CV inside StackingRegressor
    would multiply runtime ~5×.
    """
    try:
        from xgboost import XGBRegressor
        from catboost import CatBoostRegressor
    except ImportError as exc:
        raise SystemExit("xgboost and catboost both required for stacking") from exc

    from sklearn.ensemble import RandomForestRegressor, StackingRegressor
    from sklearn.model_selection import KFold

    n_samples = len(y_train)
    n_folds = max(2, min(5, n_samples - 1))

    base = [
        ("rf", RandomForestRegressor(
            n_estimators=500, max_depth=None, min_samples_leaf=1,
            random_state=seed, n_jobs=-1,
        )),
        ("xgb", XGBRegressor(
            n_estimators=400, max_depth=5, learning_rate=0.05,
            random_state=seed, n_jobs=-1, tree_method="hist",
            objective="reg:squarederror", verbosity=0,
        )),
        ("cat", CatBoostRegressor(
            iterations=400, depth=6, learning_rate=0.05,
            loss_function="MAE", random_seed=seed,
            verbose=False, allow_writing_files=False,
        )),
    ]
    meta = ElasticNetCV(
        l1_ratio=[0.1, 0.5, 0.9, 1.0], cv=n_folds, max_iter=20000,
        random_state=seed, n_jobs=-1,
    )
    model = StackingRegressor(
        estimators=base, final_estimator=meta,
        cv=KFold(n_splits=n_folds, shuffle=True, random_state=seed),
        n_jobs=1,  # base learners already parallelize; nested n_jobs=-1 thrashes
        passthrough=False,
    )
    model.fit(X_train_scaled, y_train)
    info = {
        "base_learners": [name for name, _ in base],
        "meta_learner": "ElasticNetCV",
        "cv_folds": n_folds,
        "meta_alpha": float(model.final_estimator_.alpha_),
        "meta_l1_ratio": float(model.final_estimator_.l1_ratio_),
    }
    return model, info


# ---------- per-seed evaluation ----------

def evaluate_split(
    df: pd.DataFrame,
    split: dict,
    *,
    n_cycles: int,
    models: list[str],
    seed: int,
    log_target: bool = False,
    pca_variance: float | None = None,
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

    # Optional log-transform of the target. We fit on log space but always score
    # in the original cycle space so MAE / sMAPE / R² remain interpretable across
    # runs with and without --log-target.
    if log_target:
        y_train_fit = np.log(y_train)
        y_cal_fit = np.log(y_cal) if len(y_cal) else y_cal
    else:
        y_train_fit = y_train
        y_cal_fit = y_cal

    # Z-score: fit on TRAIN, transform cal/test
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_cal_s = scaler.transform(X_cal) if len(X_cal) else X_cal
    X_test_s = scaler.transform(X_test)

    # Optional PCA: fit on TRAIN (z-scored), transform cal/test with same
    # projection. n_components = float in (0,1) keeps enough PCs to explain
    # that variance fraction; int >= 1 fixes the component count.
    pca_info: dict | None = None
    if pca_variance is not None:
        n_comp = pca_variance if pca_variance >= 1 else float(pca_variance)
        pca = PCA(n_components=n_comp, random_state=seed)
        X_train_s = pca.fit_transform(X_train_s)
        if len(X_cal_s):
            X_cal_s = pca.transform(X_cal_s)
        X_test_s = pca.transform(X_test_s)
        pca_info = {
            "n_components_in": int(X_train.shape[1]),
            "n_components_out": int(pca.n_components_),
            "explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
            "variance_target": float(pca_variance),
        }

    out: dict = {
        "n_cycles": int(n_cycles),
        "train_cells": int(len(train_df)),
        "calibration_cells": int(len(cal_df)),
        "test_cells": int(len(test_df)),
    }
    if pca_info is not None:
        out["pca"] = pca_info

    if "elastic_net" in models:
        enet = fit_elastic_net(X_train_s, y_train_fit, seed=seed)
        pred = to_cycles(enet.predict(X_test_s), log_target=log_target)
        m = compute_metrics(y_test, pred)
        m["bootstrap_95_ci"] = bootstrap_metric_ci(y_test, pred, seed=seed)
        m["best_alpha"] = float(enet.alpha_)
        m["best_l1_ratio"] = float(enet.l1_ratio_)
        if len(X_cal_s):
            cal_pred = to_cycles(enet.predict(X_cal_s), log_target=log_target)
            m["calibration_MAE"] = float(np.mean(np.abs(y_cal - cal_pred)))
        out["elastic_net"] = m

    if "pls" in models:
        pls_model, pls_info = fit_pls(X_train_s, y_train_fit, seed=seed)
        pred = to_cycles(pls_model.predict(X_test_s).ravel(), log_target=log_target)
        m = compute_metrics(y_test, pred)
        m["bootstrap_95_ci"] = bootstrap_metric_ci(y_test, pred, seed=seed)
        m["tuning"] = pls_info
        if len(X_cal_s):
            cal_pred = to_cycles(pls_model.predict(X_cal_s).ravel(), log_target=log_target)
            m["calibration_MAE"] = float(np.mean(np.abs(y_cal - cal_pred)))
        out["pls"] = m

    if "random_forest" in models:
        rf_model, rf_info = fit_random_forest(X_train_s, y_train_fit, seed=seed)
        pred = to_cycles(rf_model.predict(X_test_s), log_target=log_target)
        m = compute_metrics(y_test, pred)
        m["bootstrap_95_ci"] = bootstrap_metric_ci(y_test, pred, seed=seed)
        m["tuning"] = rf_info
        if len(X_cal_s):
            cal_pred = to_cycles(rf_model.predict(X_cal_s), log_target=log_target)
            m["calibration_MAE"] = float(np.mean(np.abs(y_cal - cal_pred)))
        out["random_forest"] = m

    if "gaussian_process" in models:
        gp_model, gp_info = fit_gaussian_process(X_train_s, y_train_fit, seed=seed)
        pred = to_cycles(gp_model.predict(X_test_s), log_target=log_target)
        m = compute_metrics(y_test, pred)
        m["bootstrap_95_ci"] = bootstrap_metric_ci(y_test, pred, seed=seed)
        m["tuning"] = gp_info
        if len(X_cal_s):
            cal_pred = to_cycles(gp_model.predict(X_cal_s), log_target=log_target)
            m["calibration_MAE"] = float(np.mean(np.abs(y_cal - cal_pred)))
        out["gaussian_process"] = m

    if "xgboost" in models:
        xgb_model, xgb_info = fit_xgboost(X_train_s, y_train_fit, seed=seed)
        pred = to_cycles(xgb_model.predict(X_test_s), log_target=log_target)
        m = compute_metrics(y_test, pred)
        m["bootstrap_95_ci"] = bootstrap_metric_ci(y_test, pred, seed=seed)
        m["tuning"] = xgb_info  # max_depth, learning_rate, n_estimators, CV stats
        if len(X_cal_s):
            cal_pred = to_cycles(xgb_model.predict(X_cal_s), log_target=log_target)
            m["calibration_MAE"] = float(np.mean(np.abs(y_cal - cal_pred)))
        out["xgboost"] = m

    if "catboost" in models:
        cb_model, cb_info = fit_catboost(X_train_s, y_train_fit, seed=seed)
        pred = to_cycles(cb_model.predict(X_test_s), log_target=log_target)
        m = compute_metrics(y_test, pred)
        m["bootstrap_95_ci"] = bootstrap_metric_ci(y_test, pred, seed=seed)
        m["tuning"] = cb_info  # depth, learning_rate, iterations, CV stats
        if len(X_cal_s):
            cal_pred = to_cycles(cb_model.predict(X_cal_s), log_target=log_target)
            m["calibration_MAE"] = float(np.mean(np.abs(y_cal - cal_pred)))
        out["catboost"] = m

    if "stacking" in models:
        st_model, st_info = fit_stacking(X_train_s, y_train_fit, seed=seed)
        pred = to_cycles(st_model.predict(X_test_s), log_target=log_target)
        m = compute_metrics(y_test, pred)
        m["bootstrap_95_ci"] = bootstrap_metric_ci(y_test, pred, seed=seed)
        m["tuning"] = st_info
        if len(X_cal_s):
            cal_pred = to_cycles(st_model.predict(X_cal_s), log_target=log_target)
            m["calibration_MAE"] = float(np.mean(np.abs(y_cal - cal_pred)))
        out["stacking"] = m

    return out


def evaluate_cross_dataset(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    source_split: dict,
    *,
    n_cycles: int,
    models: list[str],
    seed: int,
    log_target: bool = False,
    pca_variance: float | None = None,
) -> dict:
    """Train on `source_split['train']` (cells in source_df), test on the
    full uncensored target dataset at the same n_cycles window.

    Test set is the entire target dataset (not the seed-specific test split)
    so transfer numbers reflect generalization to a *new dataset*, not just
    a held-out fold inside the target. Variability across seeds reflects
    the source training set; the target test set is fixed.
    """
    src_subset = source_df[(source_df["n_cycles"] == n_cycles) & (source_df["is_censored"] == 0)].copy()
    tgt_subset = target_df[(target_df["n_cycles"] == n_cycles) & (target_df["is_censored"] == 0)].copy()

    train_df = src_subset[src_subset["cell_id"].isin(source_split["train"])]
    test_df = tgt_subset

    if len(train_df) < 5 or len(test_df) < 2:
        return {"_skipped": True, "reason": "too few cells",
                "train_cells": int(len(train_df)), "test_cells": int(len(test_df))}

    X_train = train_df[SOP12_FEATURE_COLS].to_numpy()
    y_train = train_df["cycle_life"].to_numpy()
    X_test = test_df[SOP12_FEATURE_COLS].to_numpy()
    y_test = test_df["cycle_life"].to_numpy()

    if log_target:
        y_train_fit = np.log(y_train)
    else:
        y_train_fit = y_train

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Same guard on the inputs — extreme target features (e.g. spectral entropy
    # = 0 on a saturated signal) can produce inf/nan after scaling.
    X_train_s = np.clip(np.nan_to_num(X_train_s, nan=0.0, posinf=0.0, neginf=0.0), -1e6, 1e6)
    X_test_s = np.clip(np.nan_to_num(X_test_s, nan=0.0, posinf=0.0, neginf=0.0), -1e6, 1e6)

    pca_info: dict | None = None
    if pca_variance is not None:
        n_comp = pca_variance if pca_variance >= 1 else float(pca_variance)
        pca = PCA(n_components=n_comp, random_state=seed)
        X_train_s = pca.fit_transform(X_train_s)
        X_test_s = pca.transform(X_test_s)
        pca_info = {
            "n_components_in": int(X_train.shape[1]),
            "n_components_out": int(pca.n_components_),
            "explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
        }

    out: dict = {
        "n_cycles": int(n_cycles),
        "train_cells": int(len(train_df)),
        "test_cells": int(len(test_df)),
    }
    if pca_info is not None:
        out["pca"] = pca_info

    fitters = {
        "elastic_net": fit_elastic_net,
        "pls": fit_pls,
        "random_forest": fit_random_forest,
        "gaussian_process": fit_gaussian_process,
        "xgboost": fit_xgboost,
        "catboost": fit_catboost,
        "stacking": fit_stacking,
    }

    for name in models:
        fit_fn = fitters.get(name)
        if fit_fn is None:
            continue
        result = fit_fn(X_train_s, y_train_fit, seed=seed)
        if isinstance(result, tuple):
            model, info = result
        else:
            model = result
            info = None

        raw_pred = model.predict(X_test_s)
        if hasattr(raw_pred, "ravel"):
            raw_pred = raw_pred.ravel()
        pred = to_cycles(raw_pred, log_target=log_target)
        m = compute_metrics(y_test, pred)
        m["bootstrap_95_ci"] = bootstrap_metric_ci(y_test, pred, seed=seed)
        if name == "elastic_net":
            m["best_alpha"] = float(model.alpha_)
            m["best_l1_ratio"] = float(model.l1_ratio_)
        elif info is not None:
            m["tuning"] = info
        out[name] = m

    return out


# ---------- aggregation ----------

def aggregate_seeds(per_seed: dict, model: str, n_cycles: int) -> dict:
    mae, smape, r2 = [], [], []
    ci_bounds = {
        "MAE": {"lower": [], "upper": []},
        "SMAPE": {"lower": [], "upper": []},
        "R2": {"lower": [], "upper": []},
    }
    for seed, seed_block in per_seed.items():
        model_block = seed_block.get(str(n_cycles), {}).get(model)
        if isinstance(model_block, dict) and "MAE" in model_block:
            mae.append(model_block["MAE"])
            smape.append(model_block["SMAPE"])
            r2.append(model_block["R2"])
            boot = model_block.get("bootstrap_95_ci", {})
            for metric in ci_bounds:
                metric_ci = boot.get(metric, {})
                if "lower" in metric_ci and "upper" in metric_ci:
                    ci_bounds[metric]["lower"].append(metric_ci["lower"])
                    ci_bounds[metric]["upper"].append(metric_ci["upper"])
    if not mae:
        return {}
    # Each seed has its own test-cell bootstrap interval. The summary reports
    # the mean lower/upper bounds across seeds so docs can cite one interval
    # while keeping seed-to-seed standard deviation as a separate quantity.
    def add_ci(out: dict, metric: str) -> None:
        bounds = ci_bounds[metric]
        if bounds["lower"] and bounds["upper"]:
            out[f"{metric}_ci95_lower"] = float(np.mean(bounds["lower"]))
            out[f"{metric}_ci95_upper"] = float(np.mean(bounds["upper"]))

    out = {
        "MAE_mean": float(np.mean(mae)),
        "MAE_std": float(np.std(mae)),
    }
    add_ci(out, "MAE")
    out.update({
        "SMAPE_mean": float(np.mean(smape)),
        "SMAPE_std": float(np.std(smape)),
    })
    add_ci(out, "SMAPE")
    out.update({
        "R2_mean": float(np.mean(r2)),
        "R2_std": float(np.std(r2)),
    })
    add_ci(out, "R2")
    out["n_seeds"] = len(mae)
    return out


# ---------- main ----------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", default=["matr", "hust"], choices=ALL_DATASETS)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, choices=ALL_MODELS,
                        help="Models to train. Default: full 7-model lineup.")
    parser.add_argument("--windows", type=int, nargs="+", default=DEFAULT_WINDOWS,
                        help="Prediction windows N. Default: 50 100 (per SOP §2.1). "
                             "Optional ablation: add 25.")
    parser.add_argument("--features-path", type=Path, default=FEATURES_PATH,
                        help="Feature table to use. Default: data/intermediate/features_sop12_combined.csv")
    parser.add_argument("--splits-dir", type=Path, default=SPLITS_DIR,
                        help="Directory containing {dataset}_{seed}.json split files. Default: splits/sop_v2")
    parser.add_argument("--features-from", type=Path, default=None,
                        help="Path to a text file with one feature name per line. "
                             "If omitted, every non-metadata column in the CSV is used. "
                             "Use data/intermediate/vif_kept_features.txt for the VIF-pruned ablation.")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Directory to write results into. "
                             "Defaults to outputs/results_v2_34feat_log/. Use a separate path "
                             "(e.g. outputs/results_v2_vif_drop/) for ablations.")
    parser.add_argument("--pca", type=float, default=None,
                        help="Apply PCA after standardization. Float in (0,1) = "
                             "explained-variance threshold (e.g. 0.95). "
                             "Int >= 1 = fixed number of components. "
                             "Fit on train only; same projection applied to cal/test.")
    parser.add_argument("--cross-dataset", action="store_true",
                        help="Run cross-dataset transfer experiments instead of within-dataset. "
                             "Trains on each --datasets entry's train split and tests on the FULL "
                             "uncensored other dataset (one direction per ordered pair). "
                             "5-seed averaging is over the source training split.")
    parser.add_argument("--log-target", action="store_true", default=True,
                        help="Train on log(cycle_life) and report metrics in original cycle space "
                             "(predictions are exp-transformed before scoring). "
                             "Default: enabled for the headline protocol.")
    parser.add_argument("--no-log-target", action="store_false", dest="log_target",
                        help="Disable the log-target transform for no-log ablations.")
    return parser.parse_args()


def main() -> int:
    global SOP12_FEATURE_COLS  # may be rewritten if --features-from is used
    args = parse_args()
    features_path = args.features_path if args.features_path.is_absolute() else PROJECT_ROOT / args.features_path
    splits_dir = args.splits_dir if args.splits_dir.is_absolute() else PROJECT_ROOT / args.splits_dir
    if not features_path.exists():
        print(f"[error] {features_path} missing — run build_features.py first.")
        return 1
    df = pd.read_csv(features_path)

    # Resolve feature subset.
    # Default = every numeric column in the CSV that isn't metadata (so when the
    # feature table is extended, we don't have to update this script).
    available = [c for c in df.columns if c not in META_COLS]
    feature_cols: list[str] = list(available)
    feature_subset_source = f"auto-detected from CSV ({len(feature_cols)} features)"
    if args.features_from is not None:
        if not args.features_from.exists():
            print(f"[error] --features-from {args.features_from} not found.")
            return 1
        listed = [line.strip() for line in args.features_from.read_text().splitlines() if line.strip()]
        unknown = [f for f in listed if f not in available]
        if unknown:
            print(f"[error] features-from contains columns not in the CSV: {unknown}")
            print(f"        available columns: {available}")
            return 1
        if not listed:
            print(f"[error] --features-from {args.features_from} is empty.")
            return 1
        feature_cols = listed
        feature_subset_source = f"loaded from {args.features_from} ({len(feature_cols)} features)"

    # Override the module-level constant for evaluate_split() consumers
    SOP12_FEATURE_COLS = feature_cols

    results_dir = args.output_dir if args.output_dir is not None else RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"[setup] features_path: {features_path.relative_to(PROJECT_ROOT)}")
    print(f"[setup] splits_dir: {splits_dir.relative_to(PROJECT_ROOT)}")
    print(f"[setup] features: {feature_subset_source} -> {feature_cols}")
    print(f"[setup] output_dir: {results_dir}")

    summary_rows: list[dict] = []

    if args.cross_dataset:
        # Cross-dataset transfer: train on each dataset's train split, test on
        # the *full* uncensored other dataset. One ordered pair per direction.
        pairs = [(s, t) for s in args.datasets for t in args.datasets if s != t]
        if not pairs:
            print("[error] --cross-dataset needs at least 2 datasets in --datasets.")
            return 1

        for src, tgt in pairs:
            src_df = df[df["dataset"] == src]
            tgt_df = df[df["dataset"] == tgt]
            if src_df.empty or tgt_df.empty:
                print(f"[skip] {src}->{tgt}: a dataset is empty")
                continue

            print(f"\n========== cross {src} -> {tgt} ==========")
            bundle = {
                "protocol": "SOP_cross_dataset_v2",
                "source": src,
                "target": tgt,
                "feature_set": feature_subset_source,
                "feature_columns": list(feature_cols),
                "target_label": "cycle_life @ 0.85 * Q0 (single-cycle EOL)",
                "log_target": bool(args.log_target),
                "pca_variance": args.pca,
                "test_set": "full uncensored target dataset (fixed across seeds)",
                "standardization": "z-score, fit on source train, transform target test",
                "seeds": SEEDS,
                "windows": args.windows,
                "models": args.models,
                "per_seed": {},
                "averaged": {},
            }

            for seed in SEEDS:
                split_path = splits_dir / f"{src}_{seed}.json"
                if not split_path.exists():
                    print(f"[warn] missing split file: {split_path}")
                    continue
                with split_path.open() as f:
                    src_split = json.load(f)

                per_window: dict[str, dict] = {}
                for n in args.windows:
                    print(f"  seed={seed} N={n}")
                    result = evaluate_cross_dataset(
                        src_df, tgt_df, src_split,
                        n_cycles=n, models=args.models, seed=seed,
                        log_target=args.log_target, pca_variance=args.pca,
                    )
                    per_window[str(n)] = result
                bundle["per_seed"][str(seed)] = per_window

            for n in args.windows:
                for model in args.models:
                    agg = aggregate_seeds(bundle["per_seed"], model, n)
                    if agg:
                        key = f"{src}_to_{tgt}_{model}_N{n}"
                        bundle["averaged"][key] = agg
                        summary_rows.append({
                            "dataset": tgt,         # row labeled by where evaluation happens
                            "experiment": f"{src}_to_{tgt}",
                            "model": model,
                            "n_cycles": n,
                            **agg,
                        })

            out_path = results_dir / f"results_cross_{src}_to_{tgt}.json"
            with out_path.open("w") as f:
                json.dump(bundle, f, indent=2)
            print(f"\n[save] {out_path}")

        # combined human-readable summary
        summary_df = pd.DataFrame(summary_rows)
        summary_path = results_dir / "results_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"[save] {summary_path}")

        print("\n=== CROSS-DATASET SUMMARY (5-seed source training) ===")
        print(f"{'experiment':<20} {'model':<16} {'N':>3}   {'MAE':>14}  {'sMAPE':>14}  {'R²':>14}")
        print("-" * 95)
        for _, r in summary_df.sort_values(["experiment", "n_cycles", "model"]).iterrows():
            print(
                f"{r['experiment']:<20} {r['model']:<16} {int(r['n_cycles']):>3}   "
                f"{r['MAE_mean']:>7.1f}±{r['MAE_std']:<5.1f}  "
                f"{r['SMAPE_mean']:>7.2f}±{r['SMAPE_std']:<5.2f}  "
                f"{r['R2_mean']:>7.3f}±{r['R2_std']:<5.3f}"
            )
        return 0

    for dataset in args.datasets:
        sub = df[df["dataset"] == dataset]
        if sub.empty:
            print(f"[skip] dataset={dataset}: no rows in features table")
            continue

        print(f"\n========== within-{dataset} ==========")
        bundle = {
            "protocol": "SOP_within_dataset_v2",
            "dataset": dataset,
            "feature_set": feature_subset_source,
            "feature_columns": list(feature_cols),
            "target": "cycle_life @ 0.85 * Q0 (single-cycle EOL)",
            "log_target": bool(args.log_target),
            "pca_variance": args.pca,
            "split_ratios": list((0.70, 0.15, 0.15)),
            "standardization": "z-score, fit on train, transform cal/test",
            "seeds": SEEDS,
            "windows": args.windows,
            "models": args.models,
            "per_seed": {},
            "averaged": {},
        }

        for seed in SEEDS:
            split_path = splits_dir / f"{dataset}_{seed}.json"
            if not split_path.exists():
                print(f"[warn] missing split file: {split_path}")
                continue
            with split_path.open() as f:
                split = json.load(f)

            per_window: dict[str, dict] = {}
            for n in args.windows:
                print(f"  seed={seed} N={n}")
                result = evaluate_split(sub, split, n_cycles=n, models=args.models, seed=seed, log_target=args.log_target, pca_variance=args.pca)
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

        out_path = results_dir / f"results_within_{dataset}.json"
        with out_path.open("w") as f:
            json.dump(bundle, f, indent=2)
        print(f"\n[save] {out_path}")

    # combined human-readable summary
    summary_df = pd.DataFrame(summary_rows)
    summary_path = results_dir / "results_summary.csv"
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

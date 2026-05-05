"""
Run grouped cross-validation for the top models on the engineered features.

Each fold keeps all windows from the same cell within either the training
or validation slice (GroupKFold on ``cell_id``), preventing leakage across
the n_cycles windows of a single battery.

Usage:
    python 2_modeling_featuring/cross_validate_models.py \
        --dataset data/intermediate/features_top8_cycles.csv \
        --n-splits 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Dict

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
INTERMEDIATE_DIR = DATA_DIR / "intermediate"
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"
DEFAULT_DATASET = INTERMEDIATE_DIR / "features_top8_cycles.csv"
DEFAULT_OUTPUT = RESULTS_DIR / "results_top8_cv_metrics.json"

BASE_FEATURES = [
    "IR_delta",
    "dQd_slope",
    "Qd_mean",
    "IR_slope",
    "Tavg_mean",
    "IR_mean",
    "Qd_std",
    "IR_std",
]

FEATURE_SETS = {
    "with_qd_std": BASE_FEATURES,
    "without_qd_std": [feat for feat in BASE_FEATURES if feat != "Qd_std"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cross-validate battery lifetime models.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help=f"Feature CSV path (default: {DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Where to store the aggregated CV metrics (JSON).",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of folds for GroupKFold.",
    )
    return parser.parse_args()


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Dataset bulunamadı: {path}")
    df = pd.read_csv(path)
    df = df.copy()
    df["cycle_life"] = pd.to_numeric(df["cycle_life"], errors="coerce")
    return df.dropna(subset=["cycle_life"])


def symmetric_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_pred - y_true)
    with np.errstate(divide="ignore", invalid="ignore"):
        smape = np.where(denom == 0, 0.0, diff / denom)
    return float(np.mean(smape) * 100.0)


def build_models() -> Dict[str, Dict[str, object]]:
    rf_params = dict(
        n_estimators=400,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )
    xgb_params = dict(
        n_estimators=800,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=2.0,
        reg_alpha=1.0,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )
    cat_params = dict(
        iterations=400,
        learning_rate=0.05,
        depth=6,
        loss_function="MAE",
        random_seed=42,
        verbose=False,
    )
    enet_params = dict(
        l1_ratio=[0.1, 0.5, 0.9],
        alphas=np.logspace(-4, 0, 20),
        cv=5,
        max_iter=50000,
        tol=1e-3,
        random_state=42,
    )
    return {
        "random_forest": {
            "label": "Random Forest",
            "builder": lambda: RandomForestRegressor(**rf_params),
        },
        "xgboost": {
            "label": "XGBoost",
            "builder": lambda: XGBRegressor(**xgb_params),
        },
        "catboost": {
            "label": "CatBoost",
            "builder": lambda: CatBoostRegressor(**cat_params),
        },
        "elastic_net": {
            "label": "ElasticNet",
            "builder": lambda: make_pipeline(StandardScaler(), ElasticNetCV(**enet_params)),
        },
    }


def aggregate_metrics(fold_metrics: list[dict[str, float]]) -> dict:
    aggregated = {}
    for metric in ["MAE", "R2", "MAPE", "SMAPE"]:
        values = [fold[metric] for fold in fold_metrics if metric in fold]
        if not values:
            continue
        aggregated[metric] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        }
    return aggregated


def evaluate_group_cv(
    df: pd.DataFrame,
    feature_cols: list[str],
    builder: Callable[[], object],
    n_splits: int,
) -> dict[int, dict]:
    results: dict[int, dict] = {}
    for n_cycles in (25, 50, 100):
        subset = df[df["n_cycles"] == n_cycles]
        groups = subset["cell_id"].values
        if subset.empty or len(np.unique(groups)) < n_splits:
            continue
        cols = [c for c in feature_cols if c in subset.columns]
        X = subset[cols].to_numpy()
        y = subset["cycle_life"].to_numpy()

        fold_stats: list[dict[str, float]] = []
        group_kfold = GroupKFold(n_splits=n_splits)

        for train_idx, test_idx in group_kfold.split(X, y, groups):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model = builder()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            fold_stats.append(
                {
                    "MAE": float(mean_absolute_error(y_test, y_pred)),
                    "R2": float(r2_score(y_test, y_pred)),
                    "MAPE": float(np.mean(np.abs((y_test - y_pred) / y_test)) * 100.0),
                    "SMAPE": symmetric_mape(y_test, y_pred),
                }
            )

        if fold_stats:
            results[n_cycles] = aggregate_metrics(fold_stats)
    return results


def main() -> None:
    args = parse_args()
    df = load_dataset(args.dataset)
    models = build_models()

    summary: dict[str, dict[str, dict[int, dict]]] = {}

    for model_key, cfg in models.items():
        summary[model_key] = {}
        for feature_key, feature_list in FEATURE_SETS.items():
            metrics = evaluate_group_cv(df, feature_list, cfg["builder"], args.n_splits)
            summary[model_key][feature_key] = metrics
            print(f"{cfg['label']} ({feature_key}) -> {metrics}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)
    print(f"Saved CV metrics to {args.output}")


if __name__ == "__main__":
    main()

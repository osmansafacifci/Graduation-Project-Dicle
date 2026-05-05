"""
Create feature importance graphs for RF, XGB, CatBoost on 25/50/100 cycles.

Usage:
    python 2_modeling_featuring/plot_feature_importance.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
INTERMEDIATE_DIR = DATA_DIR / "intermediate"
DEFAULT_DATASET = INTERMEDIATE_DIR / "features_top8_cycles.csv"
PLOTS_DIR = PROJECT_ROOT / "plots"

FEATURES = [
    "IR_delta",
    "dQd_slope",
    "Qd_mean",
    "IR_slope",
    "Tavg_mean",
    "IR_mean",
    "Qd_std",
    "IR_std",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="CSV that stores the engineered features.",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=PLOTS_DIR,
        help="Output directory for images.",
    )
    return parser.parse_args()


def load_dataset(dataset: Path) -> pd.DataFrame:
    df = pd.read_csv(dataset)
    df = df.dropna(subset=["cycle_life"])
    return df


def evaluate_importance(
    df: pd.DataFrame,
    feature_cols: list[str],
    n_cycles: int,
    model_builder: Callable[[], object],
) -> pd.Series:
    subset = df[df["n_cycles"] == n_cycles]
    if len(subset) < 10:
        raise ValueError(f"Too few samples for n_cycles={n_cycles}")
    X = subset[feature_cols]
    y = subset["cycle_life"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = model_builder()
    model.fit(X_train, y_train)
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        importances = model.get_feature_importance(type="FeatureImportance")
    return pd.Series(importances, index=feature_cols).sort_values(ascending=True)


def plot_importance(series: pd.Series, model_name: str, n_cycles: int, plots_dir: Path) -> None:
    plots_dir.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh(series.index, series.values)
    ax.set_title(f"{model_name} Feature Importance (n_cycles={n_cycles})")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(plots_dir / f"feature_importance_{model_name.lower()}_{n_cycles}.png", dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if not args.dataset.exists():
        raise SystemExit(f"Dataset not found: {args.dataset}")

    df = load_dataset(args.dataset)
    feature_cols = [c for c in FEATURES if c in df.columns]

    models: Dict[str, Dict] = {
        "RandomForest": {
            "builder": lambda: RandomForestRegressor(
                n_estimators=400,
                min_samples_leaf=2,
                max_features="sqrt",
                random_state=42,
                n_jobs=-1,
            )
        },
        "XGBoost": {
            "builder": lambda: XGBRegressor(
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
        },
        "CatBoost": {
            "builder": lambda: CatBoostRegressor(
                iterations=400,
                learning_rate=0.05,
                depth=6,
                loss_function="MAE",
                random_seed=42,
                verbose=False,
            )
        },
    }

    for model_name, cfg in models.items():
        for n_cycles in (25, 50, 100):
            try:
                series = evaluate_importance(df, feature_cols, n_cycles, cfg["builder"])
            except ValueError:
                continue
            plot_importance(series, model_name, n_cycles, args.plots_dir)


if __name__ == "__main__":
    main()

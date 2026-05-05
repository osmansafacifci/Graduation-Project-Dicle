"""
Compute feature ablation curves for the eight-feature pipeline and
materialize the results as JSON + heatmap PNGs.

Usage:
    python 2_modeling_featuring/make_feature_ablation.py

Outputs:
    - outputs/results/results_feature_ablation.json
    - plots/feature_ablation_heatmaps.png
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
SPLITS_DIR = DATA_DIR / "splits"
PLOTS_DIR = PROJECT_ROOT / "plots"
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"
RESULTS_JSON = RESULTS_DIR / "results_feature_ablation.json"

TRAIN_PATH = SPLITS_DIR / "features_top8_cycles_train.csv"
CALIBRATION_PATH = SPLITS_DIR / "features_top8_cycles_calibration.csv"
TEST_PATH = SPLITS_DIR / "features_top8_cycles_test.csv"

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
TARGET = "cycle_life"
WINDOWS = [25, 50, 100]


def load_split(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Missing data split: {path}")
    return pd.read_csv(path)


def get_model_defs():
    return {
        "Random Forest": lambda: RandomForestRegressor(
            n_estimators=600,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
        "XGBoost": lambda: XGBRegressor(
            n_estimators=800,
            max_depth=6,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=2,
            reg_alpha=1,
            objective="reg:squarederror",
            n_jobs=-1,
            random_state=42,
        ),
        "CatBoost": lambda: CatBoostRegressor(
            depth=6,
            learning_rate=0.05,
            iterations=600,
            loss_function="MAE",
            random_seed=42,
            verbose=False,
        ),
        "ElasticNet": lambda: make_pipeline(
            StandardScaler(),
            ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=42, max_iter=10000),
        ),
    }


def prepare_data():
    train = load_split(TRAIN_PATH)
    calibration = load_split(CALIBRATION_PATH)
    test = load_split(TEST_PATH)
    train = train.dropna(subset=[TARGET])
    calibration = calibration.dropna(subset=[TARGET])
    test = test.dropna(subset=[TARGET])
    if calibration.empty:
        print("Calibration split mevcut ama bu analizde egitime dahil edilmiyor.")
    return train, test


def evaluate_model(
    model, train_df: pd.DataFrame, test_df: pd.DataFrame, keep_features: List[str]
) -> float:
    X_train = train_df[keep_features].to_numpy()
    y_train = train_df[TARGET].to_numpy()
    X_test = test_df[keep_features].to_numpy()
    y_test = test_df[TARGET].to_numpy()

    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return float(mean_absolute_error(y_test, preds))


def main():
    train_only, test = prepare_data()
    model_factories = get_model_defs()

    rows = []
    for model_name, factory in model_factories.items():
        for window in WINDOWS:
            train_slice = train_only[train_only["n_cycles"] == window].reset_index(drop=True)
            test_slice = test[test["n_cycles"] == window].reset_index(drop=True)

            if train_slice.empty or test_slice.empty:
                continue

            for removed in [None] + FEATURES:
                features_to_use = [f for f in FEATURES if f != removed]
                # Skip degenerate case where we drop everything
                if not features_to_use:
                    continue

                model = factory()
                mae = evaluate_model(model, train_slice, test_slice, features_to_use)

                rows.append(
                    {
                        "model": model_name,
                        "n_cycles": window,
                        "feature_removed": removed or "All features",
                        "mae": mae,
                    }
                )

    if not rows:
        raise SystemExit("No ablation rows computed; check data splits.")

    results_df = pd.DataFrame(rows)
    RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_JSON.write_text(results_df.to_json(orient="records", indent=2))

    PLOTS_DIR.mkdir(exist_ok=True)
    sns.set_theme(style="whitegrid")
    models_order = list(model_factories.keys())
    fig, axes = plt.subplots(
        1, len(models_order), figsize=(18, 6), constrained_layout=True
    )

    if len(models_order) == 1:
        axes = [axes]

    feature_order = ["All features"] + FEATURES

    for ax, model_name in zip(axes, models_order):
        pivot = (
            results_df[results_df["model"] == model_name]
            .pivot(index="feature_removed", columns="n_cycles", values="mae")
            .reindex(feature_order)
        )
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".1f",
            cmap="Blues_r",
            cbar=False,
            ax=ax,
        )
        ax.set_title(f"{model_name} MAE by feature removal")
        ax.set_xlabel("n_cycles window")
        ax.set_ylabel("")
    output_path = PLOTS_DIR / "feature_ablation_heatmaps.png"
    fig.savefig(output_path, dpi=220)
    print(f"Saved heatmaps to {output_path}")


if __name__ == "__main__":
    main()

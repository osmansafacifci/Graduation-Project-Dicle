"""
CatBoost baseline model on the eight-feature dataset.

Steps:
  * load ``data/intermediate/features_top8_cycles.csv``
  * filter the feature list and drop rows with NaNs
  * train CatBoostRegressor for n_cycles = 25/50/100
  * print MAE/R2 metrics plus per-feature importances

Run:
    python 2_modeling_featuring/2_6_CatBoost_baseline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "intermediate" / "features_top8_cycles.csv"

CANDIDATE_FEATURES = [
    "IR_delta",
    "dQd_slope",
    "Qd_mean",
    "IR_slope",
    "Tavg_mean",
    "IR_mean",
    "Qd_std",
    "IR_std",
]


def load_dataset() -> tuple[pd.DataFrame, list[str]]:
    if not DATA_PATH.exists():
        print(f"Can't find dataset: {DATA_PATH}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(DATA_PATH)
    if "cycle_life" not in df.columns:
        raise ValueError("Beklenen 'cycle_life' kolonu veri setinde yok.")

    df = df.copy()
    df["cycle_life"] = pd.to_numeric(df["cycle_life"], errors="coerce")

    feature_cols = [col for col in CANDIDATE_FEATURES if col in df.columns]
    df = df.dropna(subset=["cycle_life"] + feature_cols)

    return df, feature_cols


def train_catboost(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    iterations: int = 400,
    depth: int = 6,
    learning_rate: float = 0.05,
) -> tuple[CatBoostRegressor, dict[str, float], pd.DataFrame, pd.Series]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = CatBoostRegressor(
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        loss_function="MAE",
        random_seed=42,
        verbose=False,
    )
    model.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=False)

    y_pred = model.predict(X_test)
    metrics = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "R2": r2_score(y_test, y_pred),
    }
    return model, metrics, X_test, y_test


def main() -> None:
    try:
        df, feature_cols = load_dataset()
    except Exception as exc:  # pragma: no cover - CLI output
        print(f"Veri yükleme hatası: {exc}", file=sys.stderr)
        sys.exit(1)

    print("Kullanılan feature sayısı:", len(feature_cols))
    print(feature_cols)

    results: dict[int, dict[str, float]] = {}

    for n_cycles in (25, 50, 100):
        df_subset = df[df["n_cycles"] == n_cycles].copy()
        print("\n" + "=" * 60)
        print(f"n_cycles = {n_cycles} için CatBoost modeli ({len(df_subset)} satır)")
        print("=" * 60)

        if len(df_subset) < 10:
            print("Yeterli veri yok, atlanıyor.")
            continue

        X = df_subset[feature_cols]
        y = df_subset["cycle_life"]

        model, metrics, X_test, y_test = train_catboost(X, y)
        results[n_cycles] = metrics

        print(f"MAE: {metrics['MAE']:.2f}")
        print(f"R2 : {metrics['R2']:.4f}")

        importances = model.get_feature_importance(type="FeatureImportance")
        importance_series = pd.Series(importances, index=feature_cols).sort_values(
            ascending=False
        )

        print("\nÖzellik önemleri:")
        for feat, score in importance_series.items():
            print(f"  {feat:15s}: {score:.4f}")

        comparison = pd.DataFrame(
            {
                "y_test": y_test.reset_index(drop=True),
                "y_pred": model.predict(X_test),
            }
        )
        comparison["abs_error"] = np.abs(comparison["y_test"] - comparison["y_pred"])
        print("\nİlk 5 tahmin:")
        print(comparison.head())

    print("\n=== ÖZET SONUÇLAR ===")
    for n_cycles, metric in results.items():
        print(
            f"n_cycles = {n_cycles:3d} -> MAE = {metric['MAE']:.2f}, R2 = {metric['R2']:.4f}"
        )


if __name__ == "__main__":
    main()

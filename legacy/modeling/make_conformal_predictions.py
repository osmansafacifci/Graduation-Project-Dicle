"""
Generate conformal prediction intervals for each model using train/calibration/test splits.

Train on the train CSV, calibrate residual quantiles on the calibration CSV, and
evaluate coverage on the fixed test CSV. Produces per-model PNG plots and JSON
summaries capturing quantiles, coverage, and accuracy metrics for each cycle window.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from metrics_utils import bootstrap_metric_ci, compute_metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SPLITS_DIR = DATA_DIR / "splits"
DEFAULT_TRAIN = SPLITS_DIR / "features_top8_cycles_train.csv"
DEFAULT_CALIBRATION = SPLITS_DIR / "features_top8_cycles_calibration.csv"
DEFAULT_TEST = SPLITS_DIR / "features_top8_cycles_test.csv"
PLOTS_DIR = PROJECT_ROOT / "plots"
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"

FEATURE_COLS = [
    "IR_delta",
    "dQd_slope",
    "Qd_mean",
    "IR_slope",
    "Tavg_mean",
    "IR_mean",
    "Qd_std",
    "IR_std",
]
CYCLES = (25, 50, 100)
MODELS = ("random_forest", "xgboost", "catboost", "elastic_net")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create conformal prediction plots for each model.")
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN, help="Train CSV path.")
    parser.add_argument(
        "--calibration",
        type=Path,
        default=DEFAULT_CALIBRATION,
        help="Calibration CSV path.",
    )
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST, help="Test CSV path.")
    parser.add_argument("--alpha", type=float, default=0.1, help="Miscoverage rate (default 0.1 => 90% interval).")
    return parser.parse_args()


def load_split(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Missing CSV: {path}")
    df = pd.read_csv(path)
    df = df.copy()
    df["cycle_life"] = pd.to_numeric(df["cycle_life"], errors="coerce")
    return df.dropna(subset=["cycle_life"])


def build_model(name: str):
    if name == "random_forest":
        return RandomForestRegressor(
            n_estimators=400,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        )
    if name == "xgboost":
        return XGBRegressor(
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
    if name == "catboost":
        return CatBoostRegressor(
            iterations=600,
            learning_rate=0.05,
            depth=6,
            loss_function="MAE",
            random_seed=42,
            verbose=False,
        )
    if name == "elastic_net":
        return make_pipeline(
            StandardScaler(),
            ElasticNetCV(
                l1_ratio=[0.1, 0.5, 0.9],
                alphas=np.logspace(-4, 0, 20),
                max_iter=10000,
                cv=5,
                random_state=42,
            ),
        )
    raise ValueError(f"Unknown model: {name}")


def conformal_quantile(residuals: np.ndarray, alpha: float) -> float:
    if residuals.size == 0:
        return float("nan")
    try:
        return float(np.quantile(residuals, 1 - alpha, method="higher"))
    except TypeError:
        return float(np.quantile(residuals, 1 - alpha, interpolation="higher"))


def run_conformal(
    train_df: pd.DataFrame,
    calibration_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_name: str,
    alpha: float,
) -> Dict[int, dict]:
    summary: Dict[int, dict] = {}
    for n_cycles in CYCLES:
        train_subset = train_df[train_df["n_cycles"] == n_cycles]
        calibration_subset = calibration_df[calibration_df["n_cycles"] == n_cycles]
        test_subset = test_df[test_df["n_cycles"] == n_cycles]
        if train_subset.empty or calibration_subset.empty or test_subset.empty:
            continue

        model = build_model(model_name)
        model.fit(train_subset[FEATURE_COLS], train_subset["cycle_life"])

        cal_pred = model.predict(calibration_subset[FEATURE_COLS])
        cal_res = np.abs(cal_pred - calibration_subset["cycle_life"].to_numpy())
        q = conformal_quantile(cal_res, alpha)

        test_pred = model.predict(test_subset[FEATURE_COLS])
        lower = test_pred - q
        upper = test_pred + q
        actual = test_subset["cycle_life"].to_numpy()

        summary[n_cycles] = {
            "quantile": q,
            "coverage": float(np.mean((actual >= lower) & (actual <= upper))),
            "metrics": compute_metrics(actual, test_pred),
            "bootstrap_95_ci": bootstrap_metric_ci(actual, test_pred),
            "num_test": int(len(actual)),
            "actual": actual.tolist(),
            "pred": test_pred.tolist(),
            "lower": lower.tolist(),
            "upper": upper.tolist(),
        }
    return summary


def plot_results(summary: Dict[int, dict], model_name: str) -> None:
    if not summary:
        print(f"No data to plot for {model_name}")
        return

    PLOTS_DIR.mkdir(exist_ok=True)
    colors = {25: "#1b9e77", 50: "#d95f02", 100: "#7570b3"}
    plt.figure(figsize=(5.8, 5.2))
    ax = plt.gca()
    labels_handled = set()
    for n_cycles, data in summary.items():
        actual = np.asarray(data["actual"])
        pred = np.asarray(data["pred"])
        lower = np.asarray(data["lower"])
        upper = np.asarray(data["upper"])
        yerr = np.vstack((pred - lower, upper - pred))
        ax.errorbar(
            actual,
            pred,
            yerr=yerr,
            fmt="o",
            ecolor=colors.get(n_cycles, "#333333"),
            color=colors.get(n_cycles, "#333333"),
            alpha=0.7,
            label=None if n_cycles in labels_handled else f"n_cycles={n_cycles}",
        )
        labels_handled.add(n_cycles)

    min_val = min(min(data["actual"]) for data in summary.values())
    max_val = max(max(data["actual"]) for data in summary.values())
    ax.plot([min_val, max_val], [min_val, max_val], color="#555555", linestyle="--", linewidth=1)
    ax.set_xlabel("Actual cycle life")
    ax.set_ylabel("Predicted cycle life")
    ax.set_title(f"Conformal prediction intervals – {model_name}")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out_path = PLOTS_DIR / f"conformal_{model_name}.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved {out_path}")


def save_summary(summary: Dict[int, dict], model_name: str, alpha: float) -> None:
    serializable = {
        str(n): {
            "quantile": stats["quantile"],
            "coverage": stats["coverage"],
            "MAE": stats["metrics"]["MAE"],
            "SMAPE": stats["metrics"]["SMAPE"],
            "R2": stats["metrics"]["R2"],
            "bootstrap_95_ci": stats["bootstrap_95_ci"],
            "num_test": stats["num_test"],
        }
        for n, stats in summary.items()
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_json = RESULTS_DIR / f"results_conformal_{model_name}.json"
    payload = {"alpha": alpha, "per_window": serializable}
    with out_json.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)
    print(f"Saved {out_json}")


def main() -> None:
    args = parse_args()
    train_df = load_split(args.train)
    calibration_df = load_split(args.calibration)
    test_df = load_split(args.test)

    for model_name in MODELS:
        summary = run_conformal(
            train_df, calibration_df, test_df, model_name, args.alpha
        )
        plot_results(summary, model_name)
        save_summary(summary, model_name, args.alpha)


if __name__ == "__main__":
    main()

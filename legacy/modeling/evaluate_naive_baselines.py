"""
Evaluate naive baseline predictors (mean, batch-only, cycle-count-only) on the fixed test split.

Usage:
    python 2_modeling_featuring/evaluate_naive_baselines.py \
        --train data/splits/features_top8_cycles_train.csv \
        --test data/splits/features_top8_cycles_test.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from metrics_utils import bootstrap_metric_ci, compute_metrics
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SPLITS_DIR = DATA_DIR / "splits"
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"
DEFAULT_TRAIN = SPLITS_DIR / "features_top8_cycles_train.csv"
DEFAULT_TEST = SPLITS_DIR / "features_top8_cycles_test.csv"
OUT_JSON = RESULTS_DIR / "results_naive_baselines.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute naive baseline metrics on the test split.")
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN, help="Training CSV path.")
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST, help="Test CSV path.")
    return parser.parse_args()


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Missing CSV: {path}")
    df = pd.read_csv(path)
    df = df.copy()
    df["cycle_life"] = pd.to_numeric(df["cycle_life"], errors="coerce")
    df.dropna(subset=["cycle_life"], inplace=True)
    return df


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_pred - y_true)
    with np.errstate(divide="ignore", invalid="ignore"):
        smape_vals = np.where(denom == 0, 0.0, diff / denom)
    return float(np.mean(smape_vals) * 100.0)


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    with np.errstate(divide="ignore", invalid="ignore"):
        vals = np.abs((y_true - y_pred) / y_true)
    vals = vals[~np.isnan(vals) & ~np.isinf(vals)]
    return float(np.mean(vals) * 100.0)


def evaluate_baselines(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    results = []
    y_test = test_df["cycle_life"].to_numpy()

    # Mean predictor
    mean_value = float(train_df["cycle_life"].mean())
    preds_mean = np.full_like(y_test, mean_value)
    mean_metrics = compute_metrics(y_test, preds_mean)
    mean_ci = bootstrap_metric_ci(y_test, preds_mean)
    results.append(
        {
            "Baseline": "Mean predictor",
            "MAE": mean_metrics["MAE"],
            "SMAPE (%)": mean_metrics["SMAPE"],
            "R2": mean_metrics["R2"],
            "MAE 95% CI": f"{mean_ci['MAE']['lower']:.2f} - {mean_ci['MAE']['upper']:.2f}",
            "SMAPE 95% CI": (
                f"{mean_ci['SMAPE']['lower']:.2f} - {mean_ci['SMAPE']['upper']:.2f}"
            ),
            "R2 95% CI": f"{mean_ci['R2']['lower']:.2f} - {mean_ci['R2']['upper']:.2f}",
        }
    )

    # Batch-only predictor
    train_batches = train_df.copy()
    train_batches["batch"] = train_batches["cell_id"].str[:2]
    batch_means = train_batches.groupby("batch")["cycle_life"].mean().to_dict()
    preds_batch = test_df["cell_id"].str[:2].map(batch_means).to_numpy()
    batch_metrics = compute_metrics(y_test, preds_batch)
    batch_ci = bootstrap_metric_ci(y_test, preds_batch)
    results.append(
        {
            "Baseline": "Batch-only predictor",
            "MAE": batch_metrics["MAE"],
            "SMAPE (%)": batch_metrics["SMAPE"],
            "R2": batch_metrics["R2"],
            "MAE 95% CI": f"{batch_ci['MAE']['lower']:.2f} - {batch_ci['MAE']['upper']:.2f}",
            "SMAPE 95% CI": (
                f"{batch_ci['SMAPE']['lower']:.2f} - {batch_ci['SMAPE']['upper']:.2f}"
            ),
            "R2 95% CI": f"{batch_ci['R2']['lower']:.2f} - {batch_ci['R2']['upper']:.2f}",
        }
    )

    # Cycle-count-only predictor
    cycle_means = train_df.groupby("n_cycles")["cycle_life"].mean().to_dict()
    preds_cycle = test_df["n_cycles"].map(cycle_means).to_numpy()
    cycle_metrics = compute_metrics(y_test, preds_cycle)
    cycle_ci = bootstrap_metric_ci(y_test, preds_cycle)
    results.append(
        {
            "Baseline": "Cycle-count-only predictor",
            "MAE": cycle_metrics["MAE"],
            "SMAPE (%)": cycle_metrics["SMAPE"],
            "R2": cycle_metrics["R2"],
            "MAE 95% CI": f"{cycle_ci['MAE']['lower']:.2f} - {cycle_ci['MAE']['upper']:.2f}",
            "SMAPE 95% CI": (
                f"{cycle_ci['SMAPE']['lower']:.2f} - {cycle_ci['SMAPE']['upper']:.2f}"
            ),
            "R2 95% CI": f"{cycle_ci['R2']['lower']:.2f} - {cycle_ci['R2']['upper']:.2f}",
        }
    )

    df_results = pd.DataFrame(results)
    return df_results


def main() -> None:
    args = parse_args()
    train_df = load_csv(args.train)
    test_df = load_csv(args.test)

    df_results = evaluate_baselines(train_df, test_df)
    print("Naive baseline performance on test split:")
    print(df_results.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(df_results.to_json(orient="records", indent=2))
    print(f"Saved JSON summary to {OUT_JSON}")

    # Plot metrics
    plots_dir = PROJECT_ROOT / "plots"
    plots_dir.mkdir(exist_ok=True)
    metrics = ["MAE", "R2", "SMAPE (%)"]
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    axes = axes.ravel()
    labels = df_results["Baseline"].tolist()
    x = np.arange(len(labels))
    colors = ["#1b9e77", "#d95f02", "#7570b3"]
    for ax, metric in zip(axes, metrics):
        ax.bar(x, df_results[metric], color=colors)
        ax.set_title(metric)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    out_plot = plots_dir / "naive_baselines_metrics.png"
    plt.savefig(out_plot, dpi=200)
    plt.close()
    print(f"Saved {out_plot}")


if __name__ == "__main__":
    main()

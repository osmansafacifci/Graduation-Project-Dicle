"""
Generate consolidated figures for the report:
1) conformal_prediction_intervals.png
2) smape_across_windows.png
3) mape_across_windows.png
4) dqdv_curves_example.png
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SPLITS_DIR = DATA_DIR / "splits"
DATASET_ROOT = (
    PROJECT_ROOT
    / "data-driven-prediction-of-battery-cycle-life-before-capacity-degradation-master"
    / "dataset"
)
MAT_PATH = DATASET_ROOT / "2017-05-12_batchdata_updated_struct_errorcorrect.mat"
TRAIN_CSV = SPLITS_DIR / "features_top8_cycles_train.csv"
CALIBRATION_CSV = SPLITS_DIR / "features_top8_cycles_calibration.csv"
TEST_CSV = SPLITS_DIR / "features_top8_cycles_test.csv"
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"
CV_RESULTS = RESULTS_DIR / "results_top8_cv_metrics.json"
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
CYCLES = (25, 50, 100)


def load_split(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.copy()
    df["cycle_life"] = pd.to_numeric(df["cycle_life"], errors="coerce")
    return df.dropna(subset=["cycle_life"])


def conformal_quantile(residuals: np.ndarray, alpha: float = 0.1) -> float:
    if residuals.size == 0:
        return float("nan")
    try:
        return float(np.quantile(residuals, 1 - alpha, method="higher"))
    except TypeError:
        return float(np.quantile(residuals, 1 - alpha, interpolation="higher"))


def make_conformal_plot() -> None:
    train_df = load_split(TRAIN_CSV)
    calibration_df = load_split(CALIBRATION_CSV)
    test_df = load_split(TEST_CSV)
    alpha = 0.1

    plt.figure(figsize=(6, 5))
    ax = plt.gca()
    colors = {25: "#1b9e77", 50: "#d95f02", 100: "#7570b3"}

    for n_cycles in CYCLES:
        train_subset = train_df[train_df["n_cycles"] == n_cycles]
        calibration_subset = calibration_df[calibration_df["n_cycles"] == n_cycles]
        test_subset = test_df[test_df["n_cycles"] == n_cycles]
        if train_subset.empty or calibration_subset.empty or test_subset.empty:
            continue

        model = CatBoostRegressor(
            iterations=600,
            learning_rate=0.05,
            depth=6,
            loss_function="MAE",
            random_seed=42,
            verbose=False,
        )
        model.fit(train_subset[FEATURES], train_subset["cycle_life"])

        cal_pred = model.predict(calibration_subset[FEATURES])
        q = conformal_quantile(
            np.abs(cal_pred - calibration_subset["cycle_life"].to_numpy()), alpha
        )

        test_pred = model.predict(test_subset[FEATURES])
        lower = test_pred - q
        upper = test_pred + q
        actual = test_subset["cycle_life"].to_numpy()
        ax.errorbar(
            actual,
            test_pred,
            yerr=np.vstack((test_pred - lower, upper - test_pred)),
            fmt="o",
            label=f"n={n_cycles}",
            color=colors[n_cycles],
            alpha=0.75,
        )

    min_val = min(test_df["cycle_life"])
    max_val = max(test_df["cycle_life"])
    ax.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="#444", linewidth=1)
    ax.set_xlabel("Actual cycle life")
    ax.set_ylabel("Predicted cycle life")
    ax.set_title("Conformal prediction intervals (CatBoost)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out_path = PLOTS_DIR / "conformal_prediction_intervals.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved {out_path}")


def load_cv_metrics() -> Dict[str, dict]:
    with CV_RESULTS.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def make_metric_plot(metric_key: str, out_name: str, ylabel: str) -> None:
    data = load_cv_metrics()
    models = {
        "random_forest": "Random Forest",
        "xgboost": "XGBoost",
        "catboost": "CatBoost",
        "elastic_net": "ElasticNet",
    }
    rows = []
    for model_key, feature_sets in data.items():
        if model_key not in models:
            continue
        metrics = feature_sets.get("with_qd_std", {})
        for n_str, values in metrics.items():
            metric_val = values.get(metric_key)
            if isinstance(metric_val, dict):
                metric_val = metric_val.get("mean")
            if metric_val is None:
                continue
            rows.append(
                {"Model": models[model_key], "n_cycles": int(n_str), "value": float(metric_val)}
            )
    df = pd.DataFrame(rows)
    if df.empty:
        print(f"No data for {metric_key}")
        return
    plt.figure(figsize=(6, 4))
    for model, group in df.groupby("Model"):
        group_sorted = group.sort_values("n_cycles")
        plt.plot(group_sorted["n_cycles"], group_sorted["value"], marker="o", label=model)
    plt.xlabel("n_cycles window")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} vs early-cycle window")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_path = PLOTS_DIR / out_name
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved {out_path}")


def make_dqdv_plot() -> None:
    cycles_to_plot = [0, 10, 20]
    with h5py.File(MAT_PATH, "r") as f:
        batch = f["batch"]
        cycles_group = f[batch["cycles"][0, 0]]
        plt.figure(figsize=(6, 4))
        for idx in cycles_to_plot:
            ref_v = cycles_group["V"][idx, 0]
            dqdv_key = "discharge_dQdV" if "discharge_dQdV" in cycles_group else "dQdV"
            if dqdv_key not in cycles_group:
                continue
            ref_dqdv = cycles_group[dqdv_key][idx, 0]
            dataset_v = f[ref_v]
            dataset_dqdv = f[ref_dqdv]
            if dataset_v.attrs.get("MATLAB_empty", 0) or dataset_dqdv.attrs.get(
                "MATLAB_empty", 0
            ):
                continue
            voltage = np.array(dataset_v).reshape(-1)
            dqdv = np.array(dataset_dqdv).reshape(-1)
            plt.plot(voltage, dqdv, label=f"Cycle {idx+1}")
    plt.xlabel("Voltage (V)")
    plt.ylabel("dQ/dV")
    plt.title("Representative dQ/dV curves (cell b1c0)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_path = PLOTS_DIR / "dqdv_curves_example.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved {out_path}")


def main() -> None:
    PLOTS_DIR.mkdir(exist_ok=True)
    make_conformal_plot()
    make_metric_plot("SMAPE", "smape_across_windows.png", "SMAPE (%)")
    make_metric_plot("MAPE", "mape_across_windows.png", "MAPE (%)")
    make_dqdv_plot()


if __name__ == "__main__":
    main()

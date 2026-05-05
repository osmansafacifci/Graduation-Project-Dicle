"""
Emit CSV series that match the animated plots (MAPE/SMAPE lines,
conformal intervals, dQ/dV curves).

Usage:
    python 2_modeling_featuring/make_animation_series.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import h5py
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SPLITS_DIR = DATA_DIR / "splits"
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"
CV_RESULTS = RESULTS_DIR / "results_top8_cv_metrics.json"
TRAIN_CSV = SPLITS_DIR / "features_top8_cycles_train.csv"
CALIBRATION_CSV = SPLITS_DIR / "features_top8_cycles_calibration.csv"
TEST_CSV = SPLITS_DIR / "features_top8_cycles_test.csv"
DATASET_ROOT = (
    PROJECT_ROOT
    / "data-driven-prediction-of-battery-cycle-life-before-capacity-degradation-master"
    / "dataset"
)
MAT_PATH = DATASET_ROOT / "2017-05-12_batchdata_updated_struct_errorcorrect.mat"
OUTPUT_DIR = PROJECT_ROOT / "animation_csvs"

CYCLES = [25, 50, 100]
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


def load_split(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.copy()
    df["cycle_life"] = pd.to_numeric(df["cycle_life"], errors="coerce")
    return df.dropna(subset=["cycle_life"])


def load_cv_metrics() -> Dict[str, dict]:
    if not CV_RESULTS.exists():
        raise SystemExit(f"Missing CV metrics file: {CV_RESULTS}")
    return json.loads(CV_RESULTS.read_text())


def save_metric_series(metric_key: str, filename: str) -> None:
    data = load_cv_metrics()
    rows: List[dict] = []
    for model_key, feature_sets in data.items():
        metrics = feature_sets.get("with_qd_std", {})
        for n_str, values in metrics.items():
            metric_val = values.get(metric_key)
            if isinstance(metric_val, dict):
                metric_val = metric_val.get("mean")
            if metric_val is None:
                continue
            rows.append(
                {
                    "model": model_key,
                    "n_cycles": int(n_str),
                    metric_key.lower(): float(metric_val),
                }
            )
    df = pd.DataFrame(rows).sort_values(["model", "n_cycles"])
    OUTPUT_DIR.mkdir(exist_ok=True)
    df.to_csv(OUTPUT_DIR / filename, index=False)


def conformal_quantile(residuals: np.ndarray, alpha: float = 0.1) -> float:
    if residuals.size == 0:
        return float("nan")
    return float(np.quantile(residuals, 1 - alpha, method="higher"))


def save_conformal_series() -> None:
    train_df = load_split(TRAIN_CSV)
    calibration_df = load_split(CALIBRATION_CSV)
    test_df = load_split(TEST_CSV)

    rows = []
    alpha = 0.1
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
        actual = test_subset["cycle_life"].to_numpy()
        for act, pred in zip(actual, test_pred):
            rows.append(
                {
                    "n_cycles": n_cycles,
                    "actual": float(act),
                    "prediction": float(pred),
                    "lower": float(pred - q),
                    "upper": float(pred + q),
                }
            )
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "conformal_intervals.csv", index=False)


def save_dqdv_series() -> None:
    cycles_to_plot = [0, 10, 20]
    rows = []
    if not MAT_PATH.exists():
        raise SystemExit(f"Missing MAT dataset: {MAT_PATH}")
    with h5py.File(MAT_PATH, "r") as f:
        batch = f["batch"]
        cycles_group = f[batch["cycles"][0, 0]]
        dqdv_key = "discharge_dQdV" if "discharge_dQdV" in cycles_group else "dQdV"

        for idx in cycles_to_plot:
            voltage_ds = f[cycles_group["V"][idx, 0]]
            dqdv_ds = f[cycles_group[dqdv_key][idx, 0]]
            voltage = np.array(voltage_ds).reshape(-1)
            dqdv = np.array(dqdv_ds).reshape(-1)
            for v, d in zip(voltage, dqdv):
                rows.append({"cycle_index": idx + 1, "voltage": float(v), "dqdv": float(d)})
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "dqdv_curves.csv", index=False)


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    save_metric_series("MAPE", "mape_across_windows.csv")
    save_metric_series("SMAPE", "smape_across_windows.csv")
    save_conformal_series()
    save_dqdv_series()
    print(f"Animation CSVs stored in {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()

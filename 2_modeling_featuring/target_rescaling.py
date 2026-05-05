"""
Target-mean rescaling baseline for cross-dataset transfer (precursor to SOP §7).

For each cross-dataset direction and each model, we already showed naive
transfer collapses (R² ≪ 0). The covariate-vs-concept-shift analysis in
§6.3 suggested the failure is concept shift (P(y|x) differs), not covariate
shift (P(x) differs). This script tests the simplest possible concept-shift
remedy: a 2-parameter linear correction fit on a small target calibration
subset.

Algorithm:
    1. Train each model on the source dataset's training split (same as the
       cross-dataset experiment).
    2. Predict on the full target dataset (same as before).
    3. Sample k cells from the target as a calibration subset; fit
           y_corrected = a * y_predicted + b
       via OLS on those k cells.
    4. Apply (a, b) to the remaining target cells; score MAE / R² there.
    5. Repeat with `--n-repeats` random calibration draws and average.

Inputs:
    data/intermediate/features_sop12_combined.csv   (34 features by default)
    splits/sop_v2/{matr,hust}_{seed}.json

Outputs:
    outputs/results_v2_target_rescale_<feature_set>/results_summary.csv
    outputs/results_v2_target_rescale_<feature_set>/results_<src>_to_<tgt>.json

Usage:
    python 2_modeling_featuring/target_rescaling.py
    python 2_modeling_featuring/target_rescaling.py --features-from data/intermediate/feature_set_sop12.txt
    python 2_modeling_featuring/target_rescaling.py --k-values 5 10 20 --n-repeats 20
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
from sklearn.metrics import mean_absolute_error, r2_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_experiments_v2 import (  # noqa: E402
    META_COLS, SEEDS,
    fit_catboost, fit_elastic_net, fit_gaussian_process,
    fit_pls, fit_random_forest, fit_stacking, fit_xgboost,
)
from metrics_utils import compute_metrics  # noqa: E402

warnings.filterwarnings("ignore", category=ConvergenceWarning)

PROJECT_ROOT = HERE.parent
FEATURES_PATH = PROJECT_ROOT / "data" / "intermediate" / "features_sop12_combined.csv"
SPLITS_DIR = PROJECT_ROOT / "splits" / "sop_v2"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "results_v2_target_rescale"

ALL_MODELS = ["elastic_net", "pls", "random_forest", "xgboost", "catboost", "gaussian_process", "stacking"]
DEFAULT_WINDOWS = [50, 100]
DEFAULT_DATASETS = ["matr", "hust"]
DEFAULT_KS = [5, 10, 20]
DEFAULT_N_REPEATS = 20

FITTERS = {
    "elastic_net": fit_elastic_net,
    "pls": fit_pls,
    "random_forest": fit_random_forest,
    "gaussian_process": fit_gaussian_process,
    "xgboost": fit_xgboost,
    "catboost": fit_catboost,
    "stacking": fit_stacking,
}


def safe_pred(model, X):
    raw = model.predict(X)
    if hasattr(raw, "ravel"):
        raw = raw.ravel()
    raw = np.nan_to_num(raw, nan=0.0, posinf=1e9, neginf=-1e9)
    return np.clip(raw, -1e9, 1e9)


def linear_recalibration(y_pred: np.ndarray, y_true: np.ndarray) -> tuple[float, float]:
    """OLS fit y_true = a * y_pred + b. Falls back to (1, mean shift) if pred has zero variance."""
    if np.std(y_pred) < 1e-12:
        return 1.0, float(np.mean(y_true) - np.mean(y_pred))
    a, b = np.polyfit(y_pred, y_true, 1)
    return float(a), float(b)


def rescale_block(
    y_pred_target: np.ndarray,
    y_true_target: np.ndarray,
    k_values: list[int],
    n_repeats: int,
    seed: int,
) -> dict:
    """For each k, repeat sample(k cells) → fit linear → score on rest, average."""
    rng = np.random.default_rng(seed)
    n = len(y_pred_target)
    out: dict = {}
    for k in k_values:
        if k >= n - 1:
            continue
        maes, smapes, r2s, a_vals, b_vals = [], [], [], [], []
        for _ in range(n_repeats):
            cal_idx = rng.choice(n, size=k, replace=False)
            test_idx = np.setdiff1d(np.arange(n), cal_idx)
            a, b = linear_recalibration(y_pred_target[cal_idx], y_true_target[cal_idx])
            corrected = a * y_pred_target[test_idx] + b
            metrics = compute_metrics(y_true_target[test_idx], corrected)
            maes.append(metrics["MAE"])
            smapes.append(metrics["SMAPE"])
            r2s.append(metrics["R2"])
            a_vals.append(a)
            b_vals.append(b)
        out[str(k)] = {
            "k": int(k),
            "n_repeats": int(n_repeats),
            "MAE_mean": float(np.mean(maes)),
            "MAE_std": float(np.std(maes)),
            "SMAPE_mean": float(np.mean(smapes)),
            "SMAPE_std": float(np.std(smapes)),
            "R2_mean": float(np.mean(r2s)),
            "R2_std": float(np.std(r2s)),
            "slope_mean": float(np.mean(a_vals)),
            "intercept_mean": float(np.mean(b_vals)),
        }
    return out


def evaluate_direction(
    df: pd.DataFrame,
    src: str,
    tgt: str,
    feature_cols: list[str],
    *,
    n_cycles: int,
    seed: int,
    models: list[str],
    log_target: bool,
    k_values: list[int],
    n_repeats: int,
) -> dict:
    """Train on source's training split, score on target with and without rescaling."""
    from sklearn.preprocessing import StandardScaler

    src_df = df[(df["dataset"] == src) & (df["n_cycles"] == n_cycles) & (df["is_censored"] == 0)]
    tgt_df = df[(df["dataset"] == tgt) & (df["n_cycles"] == n_cycles) & (df["is_censored"] == 0)]
    split_path = SPLITS_DIR / f"{src}_{seed}.json"
    if not split_path.exists():
        return {"_skipped": True, "reason": "missing split"}
    with split_path.open() as f:
        src_split = json.load(f)

    train_df = src_df[src_df["cell_id"].isin(src_split["train"])]
    if len(train_df) < 5 or len(tgt_df) < max(k_values) + 2:
        return {"_skipped": True, "reason": "too few cells"}

    X_train = train_df[feature_cols].to_numpy(dtype=float)
    y_train = train_df["cycle_life"].to_numpy(dtype=float)
    X_tgt = tgt_df[feature_cols].to_numpy(dtype=float)
    y_tgt = tgt_df["cycle_life"].to_numpy(dtype=float)

    y_train_fit = np.log(y_train) if log_target else y_train

    def _to_cycles(p):
        out = np.exp(p) if log_target else p
        return np.clip(np.nan_to_num(out, nan=1.0, posinf=1e9, neginf=1.0), 1.0, 1e9)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_tgt_s = scaler.transform(X_tgt)
    X_train_s = np.clip(np.nan_to_num(X_train_s, nan=0.0, posinf=0.0, neginf=0.0), -1e6, 1e6)
    X_tgt_s = np.clip(np.nan_to_num(X_tgt_s, nan=0.0, posinf=0.0, neginf=0.0), -1e6, 1e6)

    out: dict = {
        "n_cycles": int(n_cycles),
        "train_cells": int(len(train_df)),
        "target_cells": int(len(tgt_df)),
        "models": {},
    }
    for name in models:
        fitter = FITTERS.get(name)
        if fitter is None:
            continue
        result = fitter(X_train_s, y_train_fit, seed=seed)
        model = result[0] if isinstance(result, tuple) else result
        y_pred_target = _to_cycles(safe_pred(model, X_tgt_s))

        baseline = compute_metrics(y_tgt, y_pred_target)
        rescaled = rescale_block(y_pred_target, y_tgt, k_values, n_repeats, seed=seed)

        out["models"][name] = {"baseline": baseline, "rescaled": rescaled}
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS, choices=DEFAULT_DATASETS)
    parser.add_argument("--models", nargs="+", default=ALL_MODELS, choices=ALL_MODELS)
    parser.add_argument("--windows", type=int, nargs="+", default=DEFAULT_WINDOWS)
    parser.add_argument("--features-from", type=Path, default=None,
                        help="Same as run_experiments_v2.py — restrict feature columns. "
                             "Default: all 34 from the CSV.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--log-target", action="store_true", default=True)
    parser.add_argument("--no-log-target", action="store_false", dest="log_target")
    parser.add_argument("--k-values", type=int, nargs="+", default=DEFAULT_KS,
                        help="Calibration set sizes to sweep. Default: 5 10 20.")
    parser.add_argument("--n-repeats", type=int, default=DEFAULT_N_REPEATS,
                        help="Random calibration draws to average over. Default: 20.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not FEATURES_PATH.exists():
        print(f"[error] {FEATURES_PATH} missing — run build_sop12_features_v2.py first.")
        return 1
    df = pd.read_csv(FEATURES_PATH)

    available = [c for c in df.columns if c not in META_COLS]
    feature_cols = list(available)
    feature_subset_source = f"auto-detected from CSV ({len(feature_cols)} features)"
    if args.features_from is not None:
        if not args.features_from.exists():
            print(f"[error] --features-from {args.features_from} not found.")
            return 1
        listed = [line.strip() for line in args.features_from.read_text().splitlines() if line.strip()]
        unknown = [f for f in listed if f not in available]
        if unknown:
            print(f"[error] features-from contains columns not in the CSV: {unknown}")
            return 1
        feature_cols = listed
        feature_subset_source = f"loaded from {args.features_from} ({len(feature_cols)} features)"

    out_dir = args.output_dir if args.output_dir is not None else DEFAULT_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[setup] features: {feature_subset_source}")
    print(f"[setup] output_dir: {out_dir}")
    print(f"[setup] k_values: {args.k_values}, n_repeats: {args.n_repeats}, log_target: {args.log_target}")

    pairs = [(s, t) for s in args.datasets for t in args.datasets if s != t]
    summary_rows: list[dict] = []

    for src, tgt in pairs:
        print(f"\n========== {src} -> {tgt} ==========")
        bundle = {
            "protocol": "target_rescaling_v2",
            "source": src,
            "target": tgt,
            "feature_set": feature_subset_source,
            "feature_columns": list(feature_cols),
            "log_target": bool(args.log_target),
            "k_values": args.k_values,
            "n_repeats": args.n_repeats,
            "seeds": SEEDS,
            "windows": args.windows,
            "models": args.models,
            "per_seed": {},
        }
        for seed in SEEDS:
            per_window: dict[str, dict] = {}
            for n in args.windows:
                print(f"  seed={seed} N={n}")
                per_window[str(n)] = evaluate_direction(
                    df, src, tgt, feature_cols,
                    n_cycles=n, seed=seed,
                    models=args.models, log_target=args.log_target,
                    k_values=args.k_values, n_repeats=args.n_repeats,
                )
            bundle["per_seed"][str(seed)] = per_window

        # Average across seeds for each (model, n, k)
        for n in args.windows:
            for model in args.models:
                base_mae, base_r2, base_smape = [], [], []
                rescaled_acc: dict[int, dict[str, list[float]]] = {k: {"MAE": [], "R2": [], "SMAPE": []} for k in args.k_values}
                for seed in SEEDS:
                    block = bundle["per_seed"][str(seed)].get(str(n), {}).get("models", {}).get(model)
                    if not isinstance(block, dict):
                        continue
                    base = block.get("baseline", {})
                    if "MAE" in base:
                        base_mae.append(base["MAE"]); base_r2.append(base["R2"]); base_smape.append(base["SMAPE"])
                    for k_str, k_block in block.get("rescaled", {}).items():
                        k = int(k_str)
                        if k in rescaled_acc:
                            rescaled_acc[k]["MAE"].append(k_block["MAE_mean"])
                            rescaled_acc[k]["R2"].append(k_block["R2_mean"])
                            rescaled_acc[k]["SMAPE"].append(k_block["SMAPE_mean"])
                if not base_mae:
                    continue
                row = {
                    "experiment": f"{src}_to_{tgt}",
                    "model": model,
                    "n_cycles": n,
                    "baseline_MAE": float(np.mean(base_mae)),
                    "baseline_MAE_std": float(np.std(base_mae)),
                    "baseline_SMAPE": float(np.mean(base_smape)),
                    "baseline_R2": float(np.mean(base_r2)),
                    "baseline_R2_std": float(np.std(base_r2)),
                }
                for k in args.k_values:
                    if rescaled_acc[k]["MAE"]:
                        row[f"k{k}_MAE"] = float(np.mean(rescaled_acc[k]["MAE"]))
                        row[f"k{k}_MAE_std"] = float(np.std(rescaled_acc[k]["MAE"]))
                        row[f"k{k}_SMAPE"] = float(np.mean(rescaled_acc[k]["SMAPE"]))
                        row[f"k{k}_R2"] = float(np.mean(rescaled_acc[k]["R2"]))
                        row[f"k{k}_R2_std"] = float(np.std(rescaled_acc[k]["R2"]))
                summary_rows.append(row)

        out_path = out_dir / f"results_{src}_to_{tgt}.json"
        with out_path.open("w") as f:
            json.dump(bundle, f, indent=2)
        print(f"[save] {out_path}")

    summary_df = pd.DataFrame(summary_rows)
    summary_path = out_dir / "results_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"[save] {summary_path}")

    print("\n=== TARGET-RESCALING SUMMARY (5 seeds × n_repeats cal draws) ===")
    cols = ["experiment", "model", "n_cycles", "baseline_R2"] + [f"k{k}_R2" for k in args.k_values]
    cols = [c for c in cols if c in summary_df.columns]
    print(summary_df[cols].sort_values(["experiment", "n_cycles", "model"]).to_string(index=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())

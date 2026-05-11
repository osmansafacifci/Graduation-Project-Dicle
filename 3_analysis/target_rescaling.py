"""
Target-mean rescaling baseline for cross-dataset transfer (precursor to SOP §7).

For each cross-dataset direction and each model, we already showed naive
transfer often collapses. The covariate-vs-concept-shift analysis in §6.3
suggested the failure is conditional/concept shift (P(y|x) differs), not just
covariate shift (P(x) differs). This script tests two small-target-label
point-calibration adapters:

    residual_mean: y_corrected = y_predicted + mean(y_true - y_predicted)
    linear:        y_corrected = a * y_predicted + b

Algorithm:
    1. Train each model on the source dataset's training split (same as the
       cross-dataset experiment).
    2. Predict on the full target dataset (same as before).
    3. Sample k cells from the target as a calibration subset; fit the requested
       adapter(s) on those k cells.
    4. Apply (a, b) to the remaining target cells; score MAE / R² there.
    5. Repeat with `--n-repeats` random calibration draws and average.

Inputs:
    data/intermediate/features_sop12_combined.csv   (34 features by default)
    splits/sop_v2/{matr,hust}_{seed}.json

Outputs:
    outputs/results_v2_target_rescale/results_summary.csv
    outputs/results_v2_target_rescale/results_<src>_to_<tgt>.json

Usage:
    python 3_analysis/target_rescaling.py
    python 3_analysis/target_rescaling.py --features-from data/intermediate/feature_set_sop12.txt
    python 3_analysis/target_rescaling.py --k-values 5 10 15 20 --n-repeats 20
    python 3_analysis/target_rescaling.py \
        --features-path data/intermediate/features_sop12_four_dataset_capnorm.csv \
        --splits-dir splits/sop_v2_four_dataset \
        --datasets matr hust sandia luh \
        --windows 100 \
        --adapter-types residual_mean linear \
        --output-dir outputs/results_v2_four_dataset_target_rescale
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
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(PROJECT_ROOT / "2_models"))
from run_experiments import (  # noqa: E402
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
ALL_DATASETS = ["matr", "hust", "sandia", "luh"]
DEFAULT_KS = [5, 10, 15, 20]
DEFAULT_N_REPEATS = 20
ADAPTER_TYPES = ["residual_mean", "linear"]

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


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def fit_point_adapter(y_pred: np.ndarray, y_true: np.ndarray, adapter_type: str) -> tuple[float, float]:
    """Return (slope, intercept) for the requested target-side point adapter."""
    if adapter_type == "residual_mean":
        return 1.0, float(np.mean(y_true - y_pred))
    if adapter_type != "linear":
        raise ValueError(f"unknown adapter_type={adapter_type}")
    if np.std(y_pred) < 1e-12:
        return 1.0, float(np.mean(y_true) - np.mean(y_pred))
    a, b = np.polyfit(y_pred, y_true, 1)
    return float(a), float(b)


def rescale_block(
    y_pred_target: np.ndarray,
    y_true_target: np.ndarray,
    k_values: list[int],
    adapter_types: list[str],
    n_repeats: int,
    seed: int,
) -> dict:
    """For each adapter/k, repeat sample(k cells) → fit adapter → score on rest."""
    rng = np.random.default_rng(seed)
    n = len(y_pred_target)
    out: dict = {}
    for adapter_type in adapter_types:
        out[adapter_type] = {}
        for k in k_values:
            if k >= n - 1:
                continue
            maes, smapes, r2s, a_vals, b_vals = [], [], [], [], []
            for _ in range(n_repeats):
                cal_idx = rng.choice(n, size=k, replace=False)
                test_idx = np.setdiff1d(np.arange(n), cal_idx)
                a, b = fit_point_adapter(y_pred_target[cal_idx], y_true_target[cal_idx], adapter_type)
                corrected = a * y_pred_target[test_idx] + b
                corrected = np.clip(np.nan_to_num(corrected, nan=1.0, posinf=1e9, neginf=1.0), 1.0, 1e9)
                metrics = compute_metrics(y_true_target[test_idx], corrected)
                maes.append(metrics["MAE"])
                smapes.append(metrics["SMAPE"])
                r2s.append(metrics["R2"])
                a_vals.append(a)
                b_vals.append(b)
            out[adapter_type][str(k)] = {
                "adapter_type": adapter_type,
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
    adapter_types: list[str],
    n_repeats: int,
    splits_dir: Path,
) -> dict:
    """Train on source's training split, score on target with and without rescaling."""
    from sklearn.preprocessing import StandardScaler

    src_df = df[(df["dataset"] == src) & (df["n_cycles"] == n_cycles) & (df["is_censored"] == 0)]
    tgt_df = df[(df["dataset"] == tgt) & (df["n_cycles"] == n_cycles) & (df["is_censored"] == 0)]
    split_path = splits_dir / f"{src}_{seed}.json"
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
        rescaled = rescale_block(y_pred_target, y_tgt, k_values, adapter_types, n_repeats, seed=seed)

        out["models"][name] = {"baseline": baseline, "rescaled": rescaled}
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features-path", type=Path, default=FEATURES_PATH,
                        help="Feature table to use. Default: data/intermediate/features_sop12_combined.csv")
    parser.add_argument("--splits-dir", type=Path, default=SPLITS_DIR,
                        help="Directory containing {dataset}_{seed}.json split files. Default: splits/sop_v2")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS, choices=ALL_DATASETS)
    parser.add_argument("--models", nargs="+", default=ALL_MODELS, choices=ALL_MODELS)
    parser.add_argument("--windows", type=int, nargs="+", default=DEFAULT_WINDOWS)
    parser.add_argument("--features-from", type=Path, default=None,
                        help="Same as run_experiments.py — restrict feature columns. "
                             "Default: all 34 from the CSV.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--log-target", action="store_true", default=True)
    parser.add_argument("--no-log-target", action="store_false", dest="log_target")
    parser.add_argument("--k-values", type=int, nargs="+", default=DEFAULT_KS,
                        help="Calibration set sizes to sweep. Default: 5 10 15 20.")
    parser.add_argument("--adapter-types", nargs="+", default=["residual_mean", "linear"], choices=ADAPTER_TYPES,
                        help="Target-side point adapters to evaluate. Default: residual_mean linear.")
    parser.add_argument("--n-repeats", type=int, default=DEFAULT_N_REPEATS,
                        help="Random calibration draws to average over. Default: 20.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    features_path = args.features_path if args.features_path.is_absolute() else PROJECT_ROOT / args.features_path
    splits_dir = args.splits_dir if args.splits_dir.is_absolute() else PROJECT_ROOT / args.splits_dir
    if not features_path.exists():
        print(f"[error] {features_path} missing — run build_features.py first.")
        return 1
    df = pd.read_csv(features_path)

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
    print(f"[setup] features_path: {display_path(features_path)}")
    print(f"[setup] splits_dir: {display_path(splits_dir)}")
    print(f"[setup] features: {feature_subset_source}")
    print(f"[setup] output_dir: {out_dir}")
    print(f"[setup] k_values: {args.k_values}, adapter_types={args.adapter_types}, n_repeats: {args.n_repeats}, log_target: {args.log_target}")

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
                    k_values=args.k_values, adapter_types=args.adapter_types,
                    n_repeats=args.n_repeats, splits_dir=splits_dir,
                )
            bundle["per_seed"][str(seed)] = per_window

        # Average across seeds for each (model, n, k)
        for n in args.windows:
            for model in args.models:
                base_mae, base_r2, base_smape = [], [], []
                rescaled_acc: dict[str, dict[int, dict[str, list[float]]]] = {
                    adapter: {k: {"MAE": [], "R2": [], "SMAPE": [], "slope": [], "intercept": []} for k in args.k_values}
                    for adapter in args.adapter_types
                }
                for seed in SEEDS:
                    block = bundle["per_seed"][str(seed)].get(str(n), {}).get("models", {}).get(model)
                    if not isinstance(block, dict):
                        continue
                    base = block.get("baseline", {})
                    if "MAE" in base:
                        base_mae.append(base["MAE"]); base_r2.append(base["R2"]); base_smape.append(base["SMAPE"])
                    for adapter_type, adapter_block in block.get("rescaled", {}).items():
                        for k_str, k_block in adapter_block.items():
                            k = int(k_str)
                            if adapter_type in rescaled_acc and k in rescaled_acc[adapter_type]:
                                rescaled_acc[adapter_type][k]["MAE"].append(k_block["MAE_mean"])
                                rescaled_acc[adapter_type][k]["R2"].append(k_block["R2_mean"])
                                rescaled_acc[adapter_type][k]["SMAPE"].append(k_block["SMAPE_mean"])
                                rescaled_acc[adapter_type][k]["slope"].append(k_block["slope_mean"])
                                rescaled_acc[adapter_type][k]["intercept"].append(k_block["intercept_mean"])
                if not base_mae:
                    continue
                for adapter_type in args.adapter_types:
                    for k in args.k_values:
                        acc = rescaled_acc[adapter_type][k]
                        if not acc["MAE"]:
                            continue
                        summary_rows.append({
                            "experiment": f"{src}_to_{tgt}",
                            "source": src,
                            "target": tgt,
                            "model": model,
                            "n_cycles": n,
                            "adapter_type": adapter_type,
                            "k": int(k),
                            "baseline_MAE": float(np.mean(base_mae)),
                            "baseline_MAE_std": float(np.std(base_mae)),
                            "baseline_SMAPE": float(np.mean(base_smape)),
                            "baseline_R2": float(np.mean(base_r2)),
                            "baseline_R2_std": float(np.std(base_r2)),
                            "adapted_MAE": float(np.mean(acc["MAE"])),
                            "adapted_MAE_std": float(np.std(acc["MAE"])),
                            "adapted_SMAPE": float(np.mean(acc["SMAPE"])),
                            "adapted_R2": float(np.mean(acc["R2"])),
                            "adapted_R2_std": float(np.std(acc["R2"])),
                            "adapter_slope_mean": float(np.mean(acc["slope"])),
                            "adapter_intercept_mean": float(np.mean(acc["intercept"])),
                            "delta_MAE": float(np.mean(acc["MAE"]) - np.mean(base_mae)),
                            "delta_R2": float(np.mean(acc["R2"]) - np.mean(base_r2)),
                        })

        out_path = out_dir / f"results_{src}_to_{tgt}.json"
        with out_path.open("w") as f:
            json.dump(bundle, f, indent=2)
        print(f"[save] {out_path}")

    summary_df = pd.DataFrame(summary_rows)
    summary_path = out_dir / "results_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"[save] {summary_path}")

    print("\n=== TARGET-RESCALING SUMMARY (5 seeds × n_repeats cal draws) ===")
    cols = ["experiment", "model", "n_cycles", "adapter_type", "k", "baseline_R2", "adapted_R2", "delta_R2", "baseline_MAE", "adapted_MAE"]
    cols = [c for c in cols if c in summary_df.columns]
    print(summary_df[cols].sort_values(["experiment", "n_cycles", "model", "adapter_type", "k"]).to_string(index=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())

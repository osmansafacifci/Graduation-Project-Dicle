"""
Concept-shift diagnostics — companions to §6.3 shift_metrics.py.

Two analyses that document *why* feature-side capacity-normalization fails
to improve cross-dataset transfer (covariate-vs-concept-shift narrative):

  (a) Target marginal distribution comparison
      Histogram-style summary of cycle_life on each dataset (uncensored
      cells), plus a Kolmogorov–Smirnov two-sample test on the same
      values. KS rejects the null that both samples come from the same
      distribution → direct evidence of concept shift in y-space.

  (c) Per-cell residual analysis on the worst cross-dataset transfers
      For each direction (matr→hust, hust→matr) and the best within-
      dataset model on the source, train on source train cells and
      predict on the full target dataset. Save per-cell (true, pred,
      residual) triples + the systematic-bias decomposition:
          residual = y_true - y_pred
                   = (mean_true - mean_pred) + (residual - constant_bias)
      so we can see how much of the failure is a constant offset (one-shift
      problem) vs. genuine non-linear mismatch.

Inputs:
    data/intermediate/features_sop12_combined.csv
    splits/sop_v2/{matr,hust}_{seed}.json     (best seed = 42 by default)

Outputs:
    data/intermediate/concept_shift_diagnostics.json
    data/intermediate/concept_shift_diagnostics.txt

Usage:
    python 2_modeling_featuring/concept_shift_diagnostics.py
    python 2_modeling_featuring/concept_shift_diagnostics.py --seed 42 --model catboost
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_experiments_v2 import (  # noqa: E402
    META_COLS,
    fit_catboost, fit_elastic_net, fit_gaussian_process,
    fit_pls, fit_random_forest, fit_stacking, fit_xgboost,
)
from metrics_utils import compute_metrics  # noqa: E402

warnings.filterwarnings("ignore", category=ConvergenceWarning)

PROJECT_ROOT = HERE.parent
FEATURES_PATH = PROJECT_ROOT / "data" / "intermediate" / "features_sop12_combined.csv"
SPLITS_DIR = PROJECT_ROOT / "splits" / "sop_v2"
INTERMEDIATE_DIR = PROJECT_ROOT / "data" / "intermediate"

FITTERS = {
    "elastic_net": fit_elastic_net,
    "pls": fit_pls,
    "random_forest": fit_random_forest,
    "gaussian_process": fit_gaussian_process,
    "xgboost": fit_xgboost,
    "catboost": fit_catboost,
    "stacking": fit_stacking,
}


# ---------- (a) target distribution + KS ----------

def target_distribution_summary(df: pd.DataFrame) -> dict:
    out = {}
    for ds in ("matr", "hust"):
        sub = df[(df["dataset"] == ds) & (df["is_censored"] == 0) & (df["n_cycles"] == 100)]
        y = sub["cycle_life"].to_numpy(dtype=float)
        out[ds] = {
            "n": int(len(y)),
            "min": float(np.min(y)),
            "max": float(np.max(y)),
            "mean": float(np.mean(y)),
            "median": float(np.median(y)),
            "std": float(np.std(y)),
            "p10": float(np.percentile(y, 10)),
            "p25": float(np.percentile(y, 25)),
            "p75": float(np.percentile(y, 75)),
            "p90": float(np.percentile(y, 90)),
        }

    matr_y = df[(df["dataset"] == "matr") & (df["is_censored"] == 0) & (df["n_cycles"] == 100)]["cycle_life"].to_numpy()
    hust_y = df[(df["dataset"] == "hust") & (df["is_censored"] == 0) & (df["n_cycles"] == 100)]["cycle_life"].to_numpy()
    ks_stat, ks_p = ks_2samp(matr_y, hust_y)
    out["KS_test"] = {
        "statistic": float(ks_stat),
        "p_value": float(ks_p),
        "interpretation": (
            "Marginals differ (p<0.05) → direct concept-shift evidence in y."
            if ks_p < 0.05
            else "Cannot reject same-distribution null."
        ),
    }
    out["log_ratio"] = {
        "mean_log_ratio": float(np.log(out["hust"]["mean"] / out["matr"]["mean"])),
        "std_log_ratio": float(np.log(out["hust"]["std"] / out["matr"]["std"])),
    }
    return out


# ---------- (c) per-cell residual ----------

def safe_pred(model, X: np.ndarray) -> np.ndarray:
    raw = model.predict(X)
    if hasattr(raw, "ravel"):
        raw = raw.ravel()
    raw = np.nan_to_num(raw, nan=0.0, posinf=1e9, neginf=-1e9)
    return np.clip(raw, -1e9, 1e9)


def residual_analysis(
    df: pd.DataFrame,
    src: str,
    tgt: str,
    feature_cols: list[str],
    *,
    n_cycles: int,
    seed: int,
    model_name: str,
    log_target: bool,
) -> dict:
    src_df = df[(df["dataset"] == src) & (df["n_cycles"] == n_cycles) & (df["is_censored"] == 0)]
    tgt_df = df[(df["dataset"] == tgt) & (df["n_cycles"] == n_cycles) & (df["is_censored"] == 0)]
    split_path = SPLITS_DIR / f"{src}_{seed}.json"
    with split_path.open() as f:
        src_split = json.load(f)
    train_df = src_df[src_df["cell_id"].isin(src_split["train"])]

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

    fitter = FITTERS[model_name]
    result = fitter(X_train_s, y_train_fit, seed=seed)
    model = result[0] if isinstance(result, tuple) else result
    y_pred = _to_cycles(safe_pred(model, X_tgt_s))

    residuals = y_tgt - y_pred
    constant_bias = float(np.mean(residuals))
    residuals_minus_bias = residuals - constant_bias

    raw_metrics = compute_metrics(y_tgt, y_pred)
    bias_corrected_metrics = compute_metrics(y_tgt, y_pred + constant_bias)

    # SS decomposition: how much of the residual variance is just the constant?
    ss_total = float(np.sum(residuals ** 2))
    ss_after_constant = float(np.sum(residuals_minus_bias ** 2))
    constant_share = 1.0 - (ss_after_constant / ss_total) if ss_total > 1e-12 else 0.0

    return {
        "source": src,
        "target": tgt,
        "n_cycles": int(n_cycles),
        "model": model_name,
        "seed": int(seed),
        "n_target_cells": int(len(y_tgt)),
        "raw_metrics": raw_metrics,
        "constant_bias": constant_bias,
        "constant_share_of_ss": float(constant_share),
        "bias_corrected_metrics": bias_corrected_metrics,
        "residual_stats": {
            "mean": constant_bias,
            "median": float(np.median(residuals)),
            "std": float(np.std(residuals)),
            "min": float(np.min(residuals)),
            "max": float(np.max(residuals)),
            "abs_mean": float(np.mean(np.abs(residuals))),
        },
        # Save the (y_true, y_pred) cloud for downstream plots
        "per_cell": [
            {"y_true": float(yt), "y_pred": float(yp), "residual": float(yt - yp)}
            for yt, yp in zip(y_tgt, y_pred)
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default="catboost", choices=list(FITTERS.keys()))
    parser.add_argument("--n-cycles", type=int, default=100)
    parser.add_argument("--features-from", type=Path, default=None)
    parser.add_argument("--no-log-target", action="store_false", dest="log_target")
    parser.add_argument("--log-target", action="store_true", default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not FEATURES_PATH.exists():
        print(f"[error] {FEATURES_PATH} missing.")
        return 1
    df = pd.read_csv(FEATURES_PATH)

    available = [c for c in df.columns if c not in META_COLS]
    if args.features_from is not None and args.features_from.exists():
        feature_cols = [line.strip() for line in args.features_from.read_text().splitlines() if line.strip()]
    else:
        feature_cols = list(available)

    print(f"[diagnostics] n_features={len(feature_cols)}, model={args.model}, seed={args.seed}, N={args.n_cycles}")

    payload: dict = {
        "protocol": "concept_shift_diagnostics_v2",
        "feature_count": len(feature_cols),
        "model": args.model,
        "seed": args.seed,
        "n_cycles": args.n_cycles,
        "log_target": bool(args.log_target),
    }

    print("\n========== (a) target marginal distribution + KS test ==========")
    payload["target_distribution"] = target_distribution_summary(df)
    td = payload["target_distribution"]
    print(f"  MATR cycle_life: n={td['matr']['n']}, mean={td['matr']['mean']:.0f}, "
          f"std={td['matr']['std']:.0f}, [{td['matr']['min']:.0f}, {td['matr']['max']:.0f}]")
    print(f"  HUST cycle_life: n={td['hust']['n']}, mean={td['hust']['mean']:.0f}, "
          f"std={td['hust']['std']:.0f}, [{td['hust']['min']:.0f}, {td['hust']['max']:.0f}]")
    print(f"  KS test: stat={td['KS_test']['statistic']:.4f}, p={td['KS_test']['p_value']:.2e}")
    print(f"  log(mean_HUST/mean_MATR)={td['log_ratio']['mean_log_ratio']:+.3f}  "
          f"log(std_HUST/std_MATR)={td['log_ratio']['std_log_ratio']:+.3f}")

    print(f"\n========== (c) per-cell residual analysis (model={args.model}) ==========")
    payload["residual_analysis"] = []
    for src, tgt in [("matr", "hust"), ("hust", "matr")]:
        print(f"\n--- {src} -> {tgt} ---")
        block = residual_analysis(
            df, src, tgt, feature_cols,
            n_cycles=args.n_cycles, seed=args.seed,
            model_name=args.model, log_target=args.log_target,
        )
        payload["residual_analysis"].append(block)
        print(f"  n_target={block['n_target_cells']}")
        print(f"  raw           : MAE={block['raw_metrics']['MAE']:.1f}  R²={block['raw_metrics']['R2']:+.3f}")
        print(f"  bias_corrected: MAE={block['bias_corrected_metrics']['MAE']:.1f}  "
              f"R²={block['bias_corrected_metrics']['R2']:+.3f}  (constant_bias={block['constant_bias']:+.1f})")
        print(f"  constant share of SS: {block['constant_share_of_ss']*100:.1f}%")

    out_json = INTERMEDIATE_DIR / "concept_shift_diagnostics.json"
    with out_json.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[save] {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

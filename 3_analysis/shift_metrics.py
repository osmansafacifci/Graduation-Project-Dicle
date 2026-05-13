"""
SOP §6.3: Quantify covariate shift between MATR and HUST in feature space.

Two summary statistics, each computed on z-scored features (z-score fit on
the *pooled* MATR+HUST training cells so neither dataset is privileged):

  1. Maximum Mean Discrepancy (MMD²) with an RBF kernel and median-distance
     bandwidth. Sample-size unbiased estimator. Higher = more shift.
  2. Mahalanobis distance between dataset centroids, using a pooled
     covariance with a small ridge (1e-6 × I) to stay invertible at higher
     dimensionality. Larger = more shift along high-variance axes.

Bonus per-feature attribution:
  - Absolute mean shift |μ_MATR − μ_HUST| in pooled-z-score units, sorted
    so the biggest contributors to MMD/Mahalanobis are obvious.

Inputs:
    data/intermediate/features_sop12_combined.csv

Outputs:
    data/intermediate/shift_metrics.json   (full numbers, all feature sets)
    data/intermediate/shift_report.txt     (human-readable summary)

Usage:
    python 3_analysis/shift_metrics.py
    python 3_analysis/shift_metrics.py --feature-sets 12 24 34
    python 3_analysis/shift_metrics.py --windows 100
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERMEDIATE_DIR = PROJECT_ROOT / "data" / "intermediate"

META_COLS = {
    "dataset", "cell_id", "n_cycles", "q0", "cycle_life",
    "is_censored", "capacity_normalized",
}

SOP12_FEATURES = [
    "Qdis_N", "delta_Qdis", "retention_ratio", "slope_linear",
    "variance_Qdis", "range_Qdis", "max_drop", "std_diff",
    "skewness_Qdis", "slope_ratio", "Qdis_cycle10", "mean_diff",
]

EXTENDED_FEATURES = [
    "poly2_a", "poly2_b", "poly2_c", "exp_decay_k",
    "cycle_to_99pct", "cycle_to_98pct", "cycle_to_95pct",
    "slope_first_quarter", "slope_last_quarter",
    "autocorr_lag1", "knee_cycle", "n_capacity_jumps",
]

EXTENDED2_FEATURES = [
    "accel_mean", "accel_std", "accel_max_abs",
    "linearity_r2", "kurtosis_Qdis",
    "fft_top3_energy_ratio", "spectral_entropy", "sample_entropy",
    "pos_neg_diff_ratio", "mad_Qdis",
]

FEATURE_SETS = {
    "12": SOP12_FEATURES,
    "24": SOP12_FEATURES + EXTENDED_FEATURES,
    "34": SOP12_FEATURES + EXTENDED_FEATURES + EXTENDED2_FEATURES,
}

# Capacity-unit features that the SOP §2.3 normalize-by-Q0 step would scale.
# Mirrors CAPACITY_RAW_FEATURES in build_features.py; keep in sync.
CAPACITY_RAW_FEATURES = {
    "Qdis_N", "delta_Qdis", "slope_linear", "Qdis_cycle10", "max_drop",
    "mean_diff", "std_diff", "range_Qdis",
    "poly2_a", "poly2_b", "poly2_c",
    "slope_first_quarter", "slope_last_quarter",
    "accel_mean", "accel_std", "accel_max_abs",
    "mad_Qdis",
}
CAPACITY_VARIANCE_FEATURES = {"variance_Qdis"}


# ---------- helpers ----------

def median_heuristic_bandwidth(X: np.ndarray, max_pairs: int = 5000, seed: int = 0) -> float:
    """RBF bandwidth = median pairwise Euclidean distance over a subsample of
    pairs (to keep this O(N²) computation cheap on small datasets)."""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    if n < 2:
        return 1.0
    if n * (n - 1) // 2 <= max_pairs:
        i, j = np.triu_indices(n, k=1)
    else:
        idx_a = rng.integers(0, n, size=max_pairs)
        idx_b = rng.integers(0, n, size=max_pairs)
        keep = idx_a != idx_b
        i, j = idx_a[keep], idx_b[keep]
    diffs = X[i] - X[j]
    dists = np.sqrt(np.sum(diffs * diffs, axis=1))
    median = float(np.median(dists))
    return median if median > 1e-12 else 1.0


def mmd2_rbf(X: np.ndarray, Y: np.ndarray, sigma: float | None = None) -> dict:
    """Unbiased MMD² estimator with RBF kernel k(x, y) = exp(-||x-y||² / (2σ²))."""
    if sigma is None:
        Z = np.vstack([X, Y])
        sigma = median_heuristic_bandwidth(Z)
    gamma = 1.0 / (2.0 * sigma * sigma)

    def _rbf(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        sq = (
            np.sum(A * A, axis=1, keepdims=True)
            + np.sum(B * B, axis=1, keepdims=True).T
            - 2.0 * A @ B.T
        )
        sq = np.maximum(sq, 0.0)
        return np.exp(-gamma * sq)

    n = X.shape[0]
    m = Y.shape[0]
    Kxx = _rbf(X, X)
    Kyy = _rbf(Y, Y)
    Kxy = _rbf(X, Y)
    np.fill_diagonal(Kxx, 0.0)
    np.fill_diagonal(Kyy, 0.0)

    sum_xx = float(Kxx.sum()) / (n * (n - 1)) if n > 1 else 0.0
    sum_yy = float(Kyy.sum()) / (m * (m - 1)) if m > 1 else 0.0
    sum_xy = float(Kxy.sum()) / (n * m) if n > 0 and m > 0 else 0.0

    mmd2 = sum_xx + sum_yy - 2.0 * sum_xy
    return {"MMD2": float(mmd2), "MMD": float(np.sqrt(max(mmd2, 0.0))), "sigma": float(sigma)}


def mahalanobis_centroid_distance(X: np.ndarray, Y: np.ndarray, ridge: float = 1e-6) -> dict:
    """Distance between dataset means under the pooled within-group covariance.
    A small ridge keeps the matrix invertible when feature count approaches
    sample count (relevant for 34-feature × ~50-cell HUST slices)."""
    n, p = X.shape
    m, _ = Y.shape
    mu_x = X.mean(axis=0)
    mu_y = Y.mean(axis=0)
    diff = mu_x - mu_y

    # Pooled covariance — bias-corrected weights
    cov_x = np.cov(X, rowvar=False, bias=False) if n > 1 else np.zeros((p, p))
    cov_y = np.cov(Y, rowvar=False, bias=False) if m > 1 else np.zeros((p, p))
    if n + m - 2 <= 0:
        pooled = np.eye(p)
    else:
        pooled = ((n - 1) * cov_x + (m - 1) * cov_y) / (n + m - 2)
    pooled_reg = pooled + ridge * np.eye(p)

    try:
        inv = np.linalg.pinv(pooled_reg)
        d2 = float(diff @ inv @ diff)
    except np.linalg.LinAlgError:
        d2 = float("nan")

    return {
        "Mahalanobis2": d2,
        "Mahalanobis": float(np.sqrt(max(d2, 0.0))) if np.isfinite(d2) else float("nan"),
        "ridge": ridge,
    }


def per_feature_shift(X: np.ndarray, Y: np.ndarray, feature_names: list[str]) -> list[dict]:
    """|μ_X − μ_Y| / pooled_std per feature. The features whose centroid
    moved the most are the ones a non-domain-adapted transfer model is
    going to be most confused by."""
    n, p = X.shape
    m, _ = Y.shape
    mu_x = X.mean(axis=0)
    mu_y = Y.mean(axis=0)
    var_x = X.var(axis=0, ddof=1) if n > 1 else np.zeros(p)
    var_y = Y.var(axis=0, ddof=1) if m > 1 else np.zeros(p)
    pooled_var = ((n - 1) * var_x + (m - 1) * var_y) / max(n + m - 2, 1)
    pooled_std = np.sqrt(np.maximum(pooled_var, 1e-12))
    abs_z = np.abs(mu_x - mu_y) / pooled_std

    rows = [
        {"feature": name, "abs_mean_shift_z": float(abs_z[i]),
         "mu_matr": float(mu_x[i]), "mu_hust": float(mu_y[i]),
         "pooled_std": float(pooled_std[i])}
        for i, name in enumerate(feature_names)
    ]
    rows.sort(key=lambda r: -r["abs_mean_shift_z"])
    return rows


# ---------- main ----------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--feature-sets", nargs="+", default=["12", "24", "34"], choices=list(FEATURE_SETS))
    parser.add_argument("--windows", type=int, nargs="+", default=[50, 100])
    parser.add_argument("--top-k", type=int, default=8,
                        help="How many top per-feature shifts to print in the summary.")
    parser.add_argument("--capacity-normalize", action="store_true",
                        help="Divide raw-capacity features by Q0 and capacity-variance "
                             "features by Q0^2 before computing shift. Mirrors "
                             "--capacity-normalize on the feature builder; lets us measure "
                             "how much shift the absolute-capacity scale gap between "
                             "datasets contributes.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    features_path = INTERMEDIATE_DIR / "features_sop12_combined.csv"
    if not features_path.exists():
        print(f"[error] {features_path} missing — run build_features.py first.")
        return 1
    df = pd.read_csv(features_path)

    available = set(df.columns) - META_COLS
    payload: dict = {
        "protocol": "SOP_§6.3_shift_metrics_v2",
        "capacity_normalize": bool(args.capacity_normalize),
        "results": [],
    }
    report_lines: list[str] = []
    norm_tag = "(capacity-normalized: capacity features divided by q0; variance by q0^2)" if args.capacity_normalize else "(raw features)"
    report_lines.append(f"Distribution shift between MATR and HUST {norm_tag}")
    report_lines.append("=" * 80)

    if args.capacity_normalize:
        # Apply the Q0/Q0^2 normalization on the fly so we don't need to
        # rebuild the two-dataset feature table.
        for col in df.columns:
            if col in CAPACITY_RAW_FEATURES:
                df[col] = df[col] / df["q0"]
            elif col in CAPACITY_VARIANCE_FEATURES:
                df[col] = df[col] / (df["q0"] ** 2)

    for fs_name in args.feature_sets:
        full_list = FEATURE_SETS[fs_name]
        feature_cols = [c for c in full_list if c in available]
        missing = [c for c in full_list if c not in available]
        if missing:
            print(f"[warn] feature set {fs_name}: missing in CSV, skipping: {missing}")
        if not feature_cols:
            continue

        for n in args.windows:
            sub = df[(df["n_cycles"] == n) & (df["is_censored"] == 0)]
            matr = sub[sub["dataset"] == "matr"]
            hust = sub[sub["dataset"] == "hust"]
            if matr.empty or hust.empty:
                continue

            X_raw = matr[feature_cols].to_numpy(dtype=float)
            Y_raw = hust[feature_cols].to_numpy(dtype=float)

            # Pooled z-score so neither dataset dominates the scale
            pooled = np.vstack([X_raw, Y_raw])
            mean = pooled.mean(axis=0)
            std = pooled.std(axis=0, ddof=1)
            std = np.where(std < 1e-12, 1.0, std)
            X = (X_raw - mean) / std
            Y = (Y_raw - mean) / std

            mmd = mmd2_rbf(X, Y)
            maha = mahalanobis_centroid_distance(X, Y)
            per_feat = per_feature_shift(X, Y, feature_cols)

            entry = {
                "feature_set": fs_name,
                "n_features": len(feature_cols),
                "n_cycles": int(n),
                "n_matr": int(len(matr)),
                "n_hust": int(len(hust)),
                "MMD": mmd,
                "Mahalanobis": maha,
                "per_feature_shift": per_feat,
            }
            payload["results"].append(entry)

            header = f"\n[fs={fs_name} | N={n} | n_matr={len(matr)} n_hust={len(hust)}]"
            report_lines.append(header)
            report_lines.append(
                f"  MMD       = {mmd['MMD']:.4f}  (MMD² = {mmd['MMD2']:.4f}, σ_RBF = {mmd['sigma']:.3f})"
            )
            report_lines.append(
                f"  Mahalanobis = {maha['Mahalanobis']:.3f}  (d² = {maha['Mahalanobis2']:.3f}, ridge = {maha['ridge']})"
            )
            report_lines.append(f"  Top {args.top_k} per-feature mean shifts (in pooled-z units):")
            for row in per_feat[: args.top_k]:
                report_lines.append(
                    f"    {row['feature']:<24}  {row['abs_mean_shift_z']:>6.3f}  "
                    f"(μ_MATR={row['mu_matr']:>+7.3f}, μ_HUST={row['mu_hust']:>+7.3f})"
                )

            print(header)
            print(f"  MMD={mmd['MMD']:.4f}  Mahalanobis={maha['Mahalanobis']:.3f}")
            for row in per_feat[: args.top_k]:
                print(f"    {row['feature']:<24} z={row['abs_mean_shift_z']:.3f}")

    suffix = "_capnorm" if args.capacity_normalize else ""
    out_json = INTERMEDIATE_DIR / f"shift_metrics{suffix}.json"
    with out_json.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[save] {out_json}")

    out_report = INTERMEDIATE_DIR / f"shift_report{suffix}.txt"
    out_report.write_text("\n".join(report_lines) + "\n")
    print(f"[save] {out_report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

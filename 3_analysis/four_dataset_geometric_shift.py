"""
Four-dataset geometric-shift diagnostics.

This is the four-dataset analogue of shift_metrics.py. For each dataset pair,
window, and feature set it reports:

  1. Cross-fitted logistic dataset-discriminator AUC.
  2. RBF-kernel MMD with median-distance bandwidth.
  3. Mahalanobis centroid distance under pooled covariance.
  4. Per-feature centroid shifts in pooled-z units.

The discriminator is deliberately cross-fitted so the AUC is a support-overlap
diagnostic rather than an in-sample separability score.

Inputs:
    data/intermediate/features_sop12_four_dataset_capnorm.csv

Outputs:
    data/intermediate/four_dataset_geometric_shift_summary.csv
    data/intermediate/four_dataset_geometric_shift_feature_shifts.csv
    data/intermediate/four_dataset_geometric_shift.json
    data/intermediate/four_dataset_geometric_shift_report.md

Usage:
    python 3_analysis/four_dataset_geometric_shift.py
    python 3_analysis/four_dataset_geometric_shift.py \
        --features-path data/intermediate/features_sop12_four_dataset.csv \
        --output-prefix four_dataset_geometric_shift_raw
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
INTERMEDIATE_DIR = PROJECT_ROOT / "data" / "intermediate"
sys.path.insert(0, str(HERE))
from shift_metrics import FEATURE_SETS, META_COLS, mmd2_rbf, mahalanobis_centroid_distance  # noqa: E402

DEFAULT_FEATURES_PATH = INTERMEDIATE_DIR / "features_sop12_four_dataset_capnorm.csv"
ALL_DATASETS = ["matr", "hust", "sandia", "luh"]


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def pooled_zscore_pair(
    left: pd.DataFrame,
    right: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    x_raw = left[feature_cols].to_numpy(dtype=float)
    y_raw = right[feature_cols].to_numpy(dtype=float)
    pooled = np.vstack([x_raw, y_raw])
    mean = pooled.mean(axis=0)
    std = pooled.std(axis=0, ddof=1)
    std = np.where(std < 1e-12, 1.0, std)
    return (x_raw - mean) / std, (y_raw - mean) / std


def crossfit_discriminator_auc(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    seed: int,
) -> dict:
    Z = np.vstack([X, Y])
    labels = np.concatenate([np.zeros(len(X), dtype=int), np.ones(len(Y), dtype=int)])
    counts = np.bincount(labels)
    if len(counts) < 2 or counts.min() < 3:
        return {"auc_mean": float("nan"), "auc_std": float("nan"), "folds": 0}

    n_splits = int(min(5, counts.min()))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    aucs: list[float] = []
    for train_idx, test_idx in cv.split(Z, labels):
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=1.0,
                solver="liblinear",
                class_weight="balanced",
                max_iter=5000,
                random_state=seed,
            ),
        )
        clf.fit(Z[train_idx], labels[train_idx])
        proba = clf.predict_proba(Z[test_idx])[:, 1]
        auc = float(roc_auc_score(labels[test_idx], proba))
        aucs.append(max(auc, 1.0 - auc))
    return {
        "auc_mean": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0,
        "folds": n_splits,
    }


def per_feature_shift_generic(
    X: np.ndarray,
    Y: np.ndarray,
    feature_cols: list[str],
    left_name: str,
    right_name: str,
) -> list[dict]:
    delta = X.mean(axis=0) - Y.mean(axis=0)
    rows = []
    for i, feature in enumerate(feature_cols):
        rows.append(
            {
                "feature": feature,
                "abs_mean_shift_z": float(abs(delta[i])),
                f"mu_{left_name}": float(X[:, i].mean()),
                f"mu_{right_name}": float(Y[:, i].mean()),
            }
        )
    rows.sort(key=lambda row: -row["abs_mean_shift_z"])
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features-path", type=Path, default=DEFAULT_FEATURES_PATH)
    parser.add_argument("--datasets", nargs="+", default=ALL_DATASETS, choices=ALL_DATASETS)
    parser.add_argument("--windows", type=int, nargs="+", default=[50, 100])
    parser.add_argument("--feature-sets", nargs="+", default=["34"], choices=list(FEATURE_SETS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--output-prefix", default="four_dataset_geometric_shift")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    features_path = resolve_path(args.features_path)
    if not features_path.exists():
        print(f"[error] missing feature table: {features_path}")
        return 1

    df = pd.read_csv(features_path)
    available = set(df.columns) - META_COLS
    summary_rows: list[dict] = []
    feature_rows: list[dict] = []
    json_rows: list[dict] = []
    report_lines = [
        "# Four-Dataset Geometric Shift",
        "",
        f"- Feature table: `{features_path.relative_to(PROJECT_ROOT)}`",
        f"- Datasets: {', '.join(args.datasets)}",
        f"- Windows: {', '.join(str(w) for w in args.windows)}",
        f"- Feature sets: {', '.join(args.feature_sets)}",
        "",
    ]

    for n_cycles in args.windows:
        report_lines.append(f"## N={n_cycles}")
        for feature_set in args.feature_sets:
            full_features = FEATURE_SETS[feature_set]
            feature_cols = [feature for feature in full_features if feature in available]
            if not feature_cols:
                print(f"[warn] no available features for set {feature_set}")
                continue
            report_lines.append(f"### Feature Set {feature_set} ({len(feature_cols)} features)")
            for left_name, right_name in combinations(args.datasets, 2):
                left = df[
                    (df["dataset"] == left_name)
                    & (df["n_cycles"] == n_cycles)
                    & (df["is_censored"] == 0)
                ].copy()
                right = df[
                    (df["dataset"] == right_name)
                    & (df["n_cycles"] == n_cycles)
                    & (df["is_censored"] == 0)
                ].copy()
                if left.empty or right.empty:
                    continue

                X, Y = pooled_zscore_pair(left, right, feature_cols)
                auc = crossfit_discriminator_auc(X, Y, seed=args.seed)
                mmd = mmd2_rbf(X, Y)
                maha = mahalanobis_centroid_distance(X, Y)
                shifts = per_feature_shift_generic(X, Y, feature_cols, left_name, right_name)

                pair = f"{left_name}_vs_{right_name}"
                row = {
                    "pair": pair,
                    "dataset_a": left_name,
                    "dataset_b": right_name,
                    "n_cycles": int(n_cycles),
                    "feature_set": feature_set,
                    "n_features": len(feature_cols),
                    "n_a": int(len(left)),
                    "n_b": int(len(right)),
                    "discriminator_auc_mean": auc["auc_mean"],
                    "discriminator_auc_std": auc["auc_std"],
                    "discriminator_folds": auc["folds"],
                    "MMD": mmd["MMD"],
                    "MMD2": mmd["MMD2"],
                    "rbf_sigma": mmd["sigma"],
                    "Mahalanobis": maha["Mahalanobis"],
                    "Mahalanobis2": maha["Mahalanobis2"],
                }
                summary_rows.append(row)
                json_rows.append({**row, "per_feature_shift": shifts})
                for rank, shift in enumerate(shifts, start=1):
                    feature_rows.append(
                        {
                            "pair": pair,
                            "dataset_a": left_name,
                            "dataset_b": right_name,
                            "n_cycles": int(n_cycles),
                            "feature_set": feature_set,
                            "rank": rank,
                            **shift,
                        }
                    )

                top_features = ", ".join(
                    f"{shift['feature']} ({shift['abs_mean_shift_z']:.2f})"
                    for shift in shifts[: args.top_k]
                )
                report_lines.append(
                    f"- `{pair}`: AUC={auc['auc_mean']:.3f}, "
                    f"MMD={mmd['MMD']:.3f}, Mahalanobis={maha['Mahalanobis']:.2f}; "
                    f"top shifts: {top_features}"
                )
                print(
                    f"[{n_cycles} | fs={feature_set}] {pair}: "
                    f"AUC={auc['auc_mean']:.3f} MMD={mmd['MMD']:.3f} "
                    f"Mahalanobis={maha['Mahalanobis']:.2f}"
                )
            report_lines.append("")

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["n_cycles", "feature_set", "discriminator_auc_mean", "MMD"],
        ascending=[True, True, False, False],
    )
    feature_df = pd.DataFrame(feature_rows)

    out_summary = INTERMEDIATE_DIR / f"{args.output_prefix}_summary.csv"
    out_features = INTERMEDIATE_DIR / f"{args.output_prefix}_feature_shifts.csv"
    out_json = INTERMEDIATE_DIR / f"{args.output_prefix}.json"
    out_report = INTERMEDIATE_DIR / f"{args.output_prefix}_report.md"
    summary_df.to_csv(out_summary, index=False)
    feature_df.to_csv(out_features, index=False)
    with out_json.open("w") as f:
        json.dump(
            {
                "protocol": "four_dataset_geometric_shift_v1",
                "features_path": str(features_path.relative_to(PROJECT_ROOT)),
                "datasets": args.datasets,
                "windows": args.windows,
                "feature_sets": args.feature_sets,
                "results": json_rows,
            },
            f,
            indent=2,
            allow_nan=False,
        )
    out_report.write_text("\n".join(report_lines) + "\n")

    print(f"[save] {out_summary}")
    print(f"[save] {out_features}")
    print(f"[save] {out_json}")
    print(f"[save] {out_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
Feature-level transfer/stability analysis for the MATR <-> HUST study.

This script answers a narrower question than global MMD/Mahalanobis:

    Which early-cycle features look stable enough to transfer across datasets,
    and which are dataset-specific or fragile?

It combines four evidence streams for each feature:

  1. Covariate stability: absolute MATR/HUST mean shift in pooled-z units,
     with and without capacity normalization.
  2. Label-relationship stability: Spearman correlation with cycle_life in
     MATR vs HUST and whether the sign agrees.
  3. Within-dataset usefulness: univariate log-target Ridge model trained
     and tested within each source split.
  4. Cross-dataset transferability: the same univariate source model scored
     on the target dataset, before and after a residual-mean target adapter.

The residual-mean adapter uses k target labels and is evaluated on disjoint
target cells, repeated over random target calibration draws. It mirrors the
primary target-adapted CP point-correction choice without adding model
complexity.

Inputs:
    data/intermediate/features_sop12_combined.csv
    splits/sop_v2/{matr,hust}_{seed}.json

Outputs:
    data/intermediate/feature_transfer_stability.csv
    data/intermediate/feature_transfer_stability_detailed.csv
    data/intermediate/feature_transfer_stability.json
    data/intermediate/feature_transfer_stability_report.txt

Usage:
    python 3_analysis/feature_transfer_stability.py
    python 3_analysis/feature_transfer_stability.py --windows 100 --k-target 20
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
INTERMEDIATE_DIR = PROJECT_ROOT / "data" / "intermediate"
SPLITS_DIR = PROJECT_ROOT / "splits" / "sop_v2"
sys.path.insert(0, str(PROJECT_ROOT / "2_models"))
from metrics_utils import compute_metrics  # noqa: E402
from run_experiments import META_COLS, SEEDS  # noqa: E402

FEATURES_PATH = INTERMEDIATE_DIR / "features_sop12_combined.csv"

CAPACITY_RAW_FEATURES = {
    "Qdis_N", "delta_Qdis", "slope_linear", "Qdis_cycle10", "max_drop",
    "mean_diff", "std_diff", "range_Qdis", "variance_Qdis",
    "poly2_a", "poly2_b", "poly2_c",
    "slope_first_quarter", "slope_last_quarter",
    "accel_mean", "accel_std", "accel_max_abs",
    "mad_Qdis",
}


def safe_cycles_from_log(pred_log: np.ndarray) -> np.ndarray:
    pred_log = np.clip(pred_log, np.log(1.0), np.log(1e9))
    return np.clip(np.nan_to_num(np.exp(pred_log), nan=1.0, posinf=1e9, neginf=1.0), 1.0, 1e9)


def load_split(dataset: str, seed: int) -> dict:
    path = SPLITS_DIR / f"{dataset}_{seed}.json"
    with path.open() as f:
        return json.load(f)


def stable_seed(*parts: object) -> int:
    text = "|".join(str(p) for p in parts)
    return sum((i + 1) * ord(ch) for i, ch in enumerate(text)) % (2**32 - 1)


def dataset_window(df: pd.DataFrame, dataset: str, n_cycles: int) -> pd.DataFrame:
    return df[(df["dataset"] == dataset) & (df["n_cycles"] == n_cycles) & (df["is_censored"] == 0)].copy()


def split_source_frames(sub: pd.DataFrame, split: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = sub[sub["cell_id"].isin(split["train"])].copy()
    test = sub[sub["cell_id"].isin(split["test"])].copy()
    return train, test


def pooled_mean_shift_z(matr: pd.DataFrame, hust: pd.DataFrame, feature: str) -> float:
    x = matr[feature].to_numpy(dtype=float)
    y = hust[feature].to_numpy(dtype=float)
    n = len(x)
    m = len(y)
    var_x = float(np.var(x, ddof=1)) if n > 1 else 0.0
    var_y = float(np.var(y, ddof=1)) if m > 1 else 0.0
    pooled_var = ((n - 1) * var_x + (m - 1) * var_y) / max(n + m - 2, 1)
    pooled_std = float(np.sqrt(max(pooled_var, 1e-12)))
    if pooled_std < 1e-12:
        return 0.0
    return float(abs(np.mean(x) - np.mean(y)) / pooled_std)


def spearman_corr(df: pd.DataFrame, feature: str) -> float:
    corr = df[[feature, "cycle_life"]].corr(method="spearman").iloc[0, 1]
    return float(corr) if np.isfinite(corr) else float("nan")


def fit_univariate_log_ridge(train_df: pd.DataFrame, feature: str) -> tuple[RidgeCV, StandardScaler]:
    X_train = train_df[[feature]].to_numpy(dtype=float)
    y_train = np.log(train_df["cycle_life"].to_numpy(dtype=float))
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    model = RidgeCV(alphas=np.logspace(-4, 4, 25))
    model.fit(X_train_s, y_train)
    return model, scaler


def predict_univariate(model: RidgeCV, scaler: StandardScaler, df: pd.DataFrame, feature: str) -> np.ndarray:
    X = df[[feature]].to_numpy(dtype=float)
    X_s = scaler.transform(X)
    return safe_cycles_from_log(model.predict(X_s))


def residual_mean_adapted_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    k_target: int,
    repeats: int,
    seed: int,
) -> list[dict]:
    n = len(y_true)
    if k_target >= n - 1:
        return []
    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    rows: list[dict] = []
    for repeat in range(repeats):
        cal_idx = rng.choice(indices, size=k_target, replace=False)
        test_idx = np.setdiff1d(indices, cal_idx)
        shift = float(np.mean(y_true[cal_idx] - y_pred[cal_idx]))
        adapted = np.clip(y_pred[test_idx] + shift, 1.0, 1e9)
        metrics = compute_metrics(y_true[test_idx], adapted)
        rows.append(
            {
                "repeat": repeat,
                "residual_shift": shift,
                "MAE": metrics["MAE"],
                "SMAPE": metrics["SMAPE"],
                "R2": metrics["R2"],
            }
        )
    return rows


def summarize_feature(rows: pd.DataFrame, feature: str, n_cycles: int, shift_rows: dict) -> dict:
    sub = rows[(rows["feature"] == feature) & (rows["n_cycles"] == n_cycles)]
    within = sub[sub["experiment"] == "within_univariate"]
    cross_raw = sub[sub["experiment"] == "cross_univariate_raw"]
    cross_adapt = sub[sub["experiment"] == "cross_univariate_residual_mean"]
    out = {
        "feature": feature,
        "n_cycles": int(n_cycles),
        **shift_rows,
        "within_R2_mean": float(within["R2"].mean()),
        "within_MAE_mean": float(within["MAE"].mean()),
        "raw_cross_R2_mean": float(cross_raw["R2"].mean()),
        "raw_cross_MAE_mean": float(cross_raw["MAE"].mean()),
        "adapted_cross_R2_mean": float(cross_adapt["R2"].mean()),
        "adapted_cross_MAE_mean": float(cross_adapt["MAE"].mean()),
        "adapted_cross_SMAPE_mean": float(cross_adapt["SMAPE"].mean()),
        "adapted_residual_shift_mean": float(cross_adapt["residual_shift"].mean()),
        "n_within_runs": int(len(within)),
        "n_cross_raw_runs": int(len(cross_raw)),
        "n_cross_adapted_runs": int(len(cross_adapt)),
    }
    score = 0.0
    score += max(0.0, out["within_R2_mean"]) * 1.0
    score += max(0.0, out["adapted_cross_R2_mean"]) * 2.0
    score += 0.5 if out["spearman_sign_agree"] else 0.0
    score -= min(out["abs_mean_shift_z_raw"], 5.0) * 0.15
    score -= min(out["spearman_abs_delta"], 1.0) * 0.5
    out["transfer_stability_score"] = float(score)
    if out["spearman_sign_agree"] and out["abs_mean_shift_z_raw"] < 1.0 and out["within_R2_mean"] > 0.05:
        out["stability_class"] = "stable_candidate"
    elif out["abs_mean_shift_z_raw"] >= 3.0:
        out["stability_class"] = "scale_shift_fragile"
    elif not out["spearman_sign_agree"]:
        out["stability_class"] = "relationship_unstable"
    else:
        out["stability_class"] = "weak_or_mixed"
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--windows", type=int, nargs="+", default=[100])
    parser.add_argument("--datasets", nargs="+", default=["matr", "hust"], choices=["matr", "hust"])
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--k-target", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=12)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not FEATURES_PATH.exists():
        print(f"[error] missing {FEATURES_PATH}")
        return 1
    df = pd.read_csv(FEATURES_PATH)
    feature_cols = [c for c in df.columns if c not in META_COLS]
    detailed_rows: list[dict] = []
    summary_rows: list[dict] = []

    for n_cycles in args.windows:
        matr = dataset_window(df, "matr", n_cycles)
        hust = dataset_window(df, "hust", n_cycles)
        capnorm = df.copy()
        for col in CAPACITY_RAW_FEATURES:
            if col in capnorm.columns:
                capnorm[col] = capnorm[col] / capnorm["q0"]
        matr_cap = dataset_window(capnorm, "matr", n_cycles)
        hust_cap = dataset_window(capnorm, "hust", n_cycles)

        shift_by_feature: dict[str, dict] = {}
        for feature in feature_cols:
            rho_matr = spearman_corr(matr, feature)
            rho_hust = spearman_corr(hust, feature)
            shift_by_feature[feature] = {
                "abs_mean_shift_z_raw": pooled_mean_shift_z(matr, hust, feature),
                "abs_mean_shift_z_capnorm": pooled_mean_shift_z(matr_cap, hust_cap, feature),
                "spearman_matr": rho_matr,
                "spearman_hust": rho_hust,
                "spearman_abs_delta": (
                    float(abs(rho_matr - rho_hust))
                    if np.isfinite(rho_matr) and np.isfinite(rho_hust)
                    else float("nan")
                ),
                "spearman_sign_agree": bool(np.sign(rho_matr) == np.sign(rho_hust))
                if np.isfinite(rho_matr) and np.isfinite(rho_hust)
                else False,
            }

        for source in args.datasets:
            for target in args.datasets:
                if source == target:
                    continue
                target_df = dataset_window(df, target, n_cycles)
                y_target = target_df["cycle_life"].to_numpy(dtype=float)
                for seed in args.seeds:
                    split = load_split(source, seed)
                    source_df = dataset_window(df, source, n_cycles)
                    train_df, source_test_df = split_source_frames(source_df, split)
                    if len(train_df) < 5 or len(source_test_df) < 2 or len(target_df) < args.k_target + 2:
                        continue
                    for feature in feature_cols:
                        model, scaler = fit_univariate_log_ridge(train_df, feature)
                        pred_within = predict_univariate(model, scaler, source_test_df, feature)
                        y_within = source_test_df["cycle_life"].to_numpy(dtype=float)
                        within_metrics = compute_metrics(y_within, pred_within)
                        detailed_rows.append(
                            {
                                "experiment": "within_univariate",
                                "feature": feature,
                                "n_cycles": n_cycles,
                                "source": source,
                                "target": source,
                                "seed": seed,
                                "repeat": np.nan,
                                "k_target": 0,
                                "residual_shift": np.nan,
                                **within_metrics,
                            }
                        )

                        pred_target = predict_univariate(model, scaler, target_df, feature)
                        raw_metrics = compute_metrics(y_target, pred_target)
                        detailed_rows.append(
                            {
                                "experiment": "cross_univariate_raw",
                                "feature": feature,
                                "n_cycles": n_cycles,
                                "source": source,
                                "target": target,
                                "seed": seed,
                                "repeat": np.nan,
                                "k_target": 0,
                                "residual_shift": np.nan,
                                **raw_metrics,
                            }
                        )

                        for row in residual_mean_adapted_metrics(
                            y_target,
                            pred_target,
                            k_target=args.k_target,
                            repeats=args.repeats,
                            seed=stable_seed(feature, source, target, seed, n_cycles, args.k_target),
                        ):
                            detailed_rows.append(
                                {
                                    "experiment": "cross_univariate_residual_mean",
                                    "feature": feature,
                                    "n_cycles": n_cycles,
                                    "source": source,
                                    "target": target,
                                    "seed": seed,
                                    "k_target": args.k_target,
                                    **row,
                                }
                            )

        detailed_df = pd.DataFrame(detailed_rows)
        for feature in feature_cols:
            summary_rows.append(summarize_feature(detailed_df, feature, n_cycles, shift_by_feature[feature]))

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["n_cycles", "transfer_stability_score"], ascending=[True, False]
    )
    detailed_df = pd.DataFrame(detailed_rows)

    out_summary = INTERMEDIATE_DIR / "feature_transfer_stability.csv"
    out_detailed = INTERMEDIATE_DIR / "feature_transfer_stability_detailed.csv"
    out_json = INTERMEDIATE_DIR / "feature_transfer_stability.json"
    out_report = INTERMEDIATE_DIR / "feature_transfer_stability_report.txt"

    summary_df.to_csv(out_summary, index=False)
    detailed_df.to_csv(out_detailed, index=False)
    payload = {
        "protocol": "feature_transfer_stability_v1",
        "windows": args.windows,
        "seeds": args.seeds,
        "k_target": args.k_target,
        "repeats": args.repeats,
        "feature_columns": feature_cols,
        "summary": summary_df.replace({np.nan: None}).to_dict(orient="records"),
    }
    with out_json.open("w") as f:
        json.dump(payload, f, indent=2, allow_nan=False)

    lines: list[str] = []
    for n_cycles in args.windows:
        block = summary_df[summary_df["n_cycles"] == n_cycles]
        lines.append(f"Feature transfer/stability analysis (N={n_cycles})")
        lines.append("=" * 80)
        lines.append("\nLeast-fragile transfer candidates:")
        for _, row in block.head(args.top_k).iterrows():
            lines.append(
                f"  {row['feature']:<24} score={row['transfer_stability_score']:+.3f} "
                f"shift_z={row['abs_mean_shift_z_raw']:.2f} "
                f"rho(M,H)=({row['spearman_matr']:+.2f},{row['spearman_hust']:+.2f}) "
                f"adapted_R2={row['adapted_cross_R2_mean']:+.3f} "
                f"class={row['stability_class']}"
            )
        lines.append("\nMost fragile by raw mean shift:")
        for _, row in block.sort_values("abs_mean_shift_z_raw", ascending=False).head(args.top_k).iterrows():
            lines.append(
                f"  {row['feature']:<24} shift_z={row['abs_mean_shift_z_raw']:.2f} "
                f"capnorm_z={row['abs_mean_shift_z_capnorm']:.2f} "
                f"adapted_R2={row['adapted_cross_R2_mean']:+.3f} "
                f"class={row['stability_class']}"
            )
        lines.append("")

    out_report.write_text("\n".join(lines))
    print(f"[save] {out_summary}")
    print(f"[save] {out_detailed}")
    print(f"[save] {out_json}")
    print(f"[save] {out_report}")
    print("\n" + "\n".join(lines[: 2 * args.top_k + 8]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

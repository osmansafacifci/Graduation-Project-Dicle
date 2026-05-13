"""
All-pairs feature-transfer-stability scores for the four-dataset extension.

This is a direction-aware four-dataset version of feature_transfer_stability.py.
It keeps the same scoring recipe:

    score = max(0, within_R2)
          + 2 * max(0, adapted_cross_R2)
          + 0.5 if source/target Spearman signs agree
          - 0.15 * min(abs_mean_shift_z, 5)
          - 0.5 * min(abs_delta_spearman, 1)

The model is intentionally univariate Ridge on log(cycle_life). The goal is
not to produce a best predictor; it is to identify features whose relationship
with lifetime is least fragile under source->target transfer.

Inputs:
    data/intermediate/features_sop12_four_dataset_capnorm.csv
    splits/sop_v2_four_dataset/{dataset}_{seed}.json

Outputs:
    data/intermediate/four_dataset_feature_transfer_stability.csv
    data/intermediate/four_dataset_feature_transfer_stability_detailed.csv
    data/intermediate/four_dataset_feature_transfer_stability.json
    data/intermediate/four_dataset_feature_transfer_stability_report.md

Usage:
    python 3_analysis/four_dataset_feature_transfer_stability.py
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
SPLITS_DIR = PROJECT_ROOT / "splits" / "sop_v2_four_dataset"
sys.path.insert(0, str(PROJECT_ROOT / "2_models"))
from metrics_utils import compute_metrics, to_cycles  # noqa: E402
from run_experiments import META_COLS, SEEDS  # noqa: E402

DEFAULT_FEATURES_PATH = INTERMEDIATE_DIR / "features_sop12_four_dataset_capnorm.csv"
DEFAULT_RAW_FEATURES_PATH = INTERMEDIATE_DIR / "features_sop12_four_dataset.csv"
ALL_DATASETS = ["matr", "hust", "sandia", "luh"]


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def stable_seed(*parts: object) -> int:
    text = "|".join(str(part) for part in parts)
    return sum((i + 1) * ord(ch) for i, ch in enumerate(text)) % (2**32 - 1)


def load_split(splits_dir: Path, dataset: str, seed: int) -> dict:
    path = splits_dir / f"{dataset}_{seed}.json"
    with path.open() as f:
        return json.load(f)


def dataset_window(df: pd.DataFrame, dataset: str, n_cycles: int) -> pd.DataFrame:
    return df[(df["dataset"] == dataset) & (df["n_cycles"] == n_cycles) & (df["is_censored"] == 0)].copy()


def split_source_frames(sub: pd.DataFrame, split: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = sub[sub["cell_id"].isin(split["train"])].copy()
    test = sub[sub["cell_id"].isin(split["test"])].copy()
    return train, test


def pooled_mean_shift_z(source_df: pd.DataFrame, target_df: pd.DataFrame, feature: str) -> float:
    x = source_df[feature].to_numpy(dtype=float)
    y = target_df[feature].to_numpy(dtype=float)
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
    return to_cycles(model.predict(scaler.transform(X)), log_target=True)


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
        rows.append({"repeat": repeat, "residual_shift": shift, **metrics})
    return rows


def classify_feature(row: dict) -> str:
    if (
        row["spearman_sign_agree"]
        and row["abs_mean_shift_z"] < 1.0
        and row["within_R2_mean"] > 0.05
    ):
        return "stable_candidate"
    if row["abs_mean_shift_z"] >= 3.0:
        return "scale_shift_fragile"
    if not row["spearman_sign_agree"]:
        return "relationship_unstable"
    return "weak_or_mixed"


def summarize_pair_feature(
    detailed_df: pd.DataFrame,
    *,
    source: str,
    target: str,
    feature: str,
    n_cycles: int,
    shift_row: dict,
) -> dict:
    within = detailed_df[
        (detailed_df["experiment"] == "within_univariate")
        & (detailed_df["source"] == source)
        & (detailed_df["feature"] == feature)
        & (detailed_df["n_cycles"] == n_cycles)
    ]
    cross_raw = detailed_df[
        (detailed_df["experiment"] == "cross_univariate_raw")
        & (detailed_df["source"] == source)
        & (detailed_df["target"] == target)
        & (detailed_df["feature"] == feature)
        & (detailed_df["n_cycles"] == n_cycles)
    ]
    cross_adapt = detailed_df[
        (detailed_df["experiment"] == "cross_univariate_residual_mean")
        & (detailed_df["source"] == source)
        & (detailed_df["target"] == target)
        & (detailed_df["feature"] == feature)
        & (detailed_df["n_cycles"] == n_cycles)
    ]
    out = {
        "source": source,
        "target": target,
        "direction": f"{source}_to_{target}",
        "feature": feature,
        "n_cycles": int(n_cycles),
        **shift_row,
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
    score -= min(out["abs_mean_shift_z"], 5.0) * 0.15
    score -= min(out["spearman_abs_delta"], 1.0) * 0.5
    out["transfer_stability_score"] = float(score)
    out["stability_class"] = classify_feature(out)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features-path", type=Path, default=DEFAULT_FEATURES_PATH)
    parser.add_argument("--raw-features-path", type=Path, default=DEFAULT_RAW_FEATURES_PATH)
    parser.add_argument("--splits-dir", type=Path, default=SPLITS_DIR)
    parser.add_argument("--datasets", nargs="+", default=ALL_DATASETS, choices=ALL_DATASETS)
    parser.add_argument("--windows", type=int, nargs="+", default=[100])
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--k-target", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output-prefix", default="four_dataset_feature_transfer_stability")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    features_path = resolve_path(args.features_path)
    raw_features_path = resolve_path(args.raw_features_path)
    splits_dir = resolve_path(args.splits_dir)
    if not features_path.exists():
        print(f"[error] missing feature table: {features_path}")
        return 1
    if not splits_dir.exists():
        print(f"[error] missing splits directory: {splits_dir}")
        return 1

    df = pd.read_csv(features_path)
    raw_df = pd.read_csv(raw_features_path) if raw_features_path.exists() else None
    feature_cols = [col for col in df.columns if col not in META_COLS]
    detailed_rows: list[dict] = []
    summary_rows: list[dict] = []
    model_cache: dict[tuple[str, int, int, str], tuple[RidgeCV, StandardScaler, pd.DataFrame]] = {}

    print(f"[setup] features_path: {features_path.relative_to(PROJECT_ROOT)}")
    print(f"[setup] splits_dir: {splits_dir.relative_to(PROJECT_ROOT)}")
    print(f"[setup] features: {len(feature_cols)}")

    for n_cycles in args.windows:
        shift_by_pair_feature: dict[tuple[str, str, str], dict] = {}
        for source in args.datasets:
            src_df = dataset_window(df, source, n_cycles)
            raw_src_df = dataset_window(raw_df, source, n_cycles) if raw_df is not None else None
            for target in args.datasets:
                if source == target:
                    continue
                tgt_df = dataset_window(df, target, n_cycles)
                raw_tgt_df = dataset_window(raw_df, target, n_cycles) if raw_df is not None else None
                for feature in feature_cols:
                    rho_source = spearman_corr(src_df, feature)
                    rho_target = spearman_corr(tgt_df, feature)
                    shift = {
                        "abs_mean_shift_z": pooled_mean_shift_z(src_df, tgt_df, feature),
                        "abs_mean_shift_z_raw_table": (
                            pooled_mean_shift_z(raw_src_df, raw_tgt_df, feature)
                            if raw_src_df is not None and raw_tgt_df is not None and feature in raw_src_df.columns
                            else float("nan")
                        ),
                        "spearman_source": rho_source,
                        "spearman_target": rho_target,
                        "spearman_abs_delta": (
                            float(abs(rho_source - rho_target))
                            if np.isfinite(rho_source) and np.isfinite(rho_target)
                            else float("nan")
                        ),
                        "spearman_sign_agree": (
                            bool(np.sign(rho_source) == np.sign(rho_target))
                            if np.isfinite(rho_source) and np.isfinite(rho_target)
                            else False
                        ),
                    }
                    shift_by_pair_feature[(source, target, feature)] = shift

        for source in args.datasets:
            source_df = dataset_window(df, source, n_cycles)
            for seed in args.seeds:
                split = load_split(splits_dir, source, seed)
                train_df, source_test_df = split_source_frames(source_df, split)
                if len(train_df) < 5 or len(source_test_df) < 2:
                    continue
                y_within = source_test_df["cycle_life"].to_numpy(dtype=float)
                for feature in feature_cols:
                    cache_key = (source, seed, n_cycles, feature)
                    model, scaler = fit_univariate_log_ridge(train_df, feature)
                    model_cache[cache_key] = (model, scaler, source_test_df)
                    pred_within = predict_univariate(model, scaler, source_test_df, feature)
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
                            **compute_metrics(y_within, pred_within),
                        }
                    )

        for source in args.datasets:
            for target in args.datasets:
                if source == target:
                    continue
                target_df = dataset_window(df, target, n_cycles)
                y_target = target_df["cycle_life"].to_numpy(dtype=float)
                if len(target_df) < args.k_target + 2:
                    continue
                print(f"[run] N={n_cycles} {source}->{target}")
                for seed in args.seeds:
                    for feature in feature_cols:
                        cache_key = (source, seed, n_cycles, feature)
                        if cache_key not in model_cache:
                            continue
                        model, scaler, _source_test_df = model_cache[cache_key]
                        pred_target = predict_univariate(model, scaler, target_df, feature)
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
                                **compute_metrics(y_target, pred_target),
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
        for source in args.datasets:
            for target in args.datasets:
                if source == target:
                    continue
                for feature in feature_cols:
                    summary_rows.append(
                        summarize_pair_feature(
                            detailed_df,
                            source=source,
                            target=target,
                            feature=feature,
                            n_cycles=n_cycles,
                            shift_row=shift_by_pair_feature[(source, target, feature)],
                        )
                    )

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["n_cycles", "direction", "transfer_stability_score"],
        ascending=[True, True, False],
    )
    detailed_df = pd.DataFrame(detailed_rows)

    out_summary = INTERMEDIATE_DIR / f"{args.output_prefix}.csv"
    out_detailed = INTERMEDIATE_DIR / f"{args.output_prefix}_detailed.csv"
    out_json = INTERMEDIATE_DIR / f"{args.output_prefix}.json"
    out_report = INTERMEDIATE_DIR / f"{args.output_prefix}_report.md"
    summary_df.to_csv(out_summary, index=False)
    detailed_df.to_csv(out_detailed, index=False)
    with out_json.open("w") as f:
        json.dump(
            {
                "protocol": "four_dataset_feature_transfer_stability_v1",
                "features_path": str(features_path.relative_to(PROJECT_ROOT)),
                "raw_features_path": (
                    str(raw_features_path.relative_to(PROJECT_ROOT)) if raw_features_path.exists() else None
                ),
                "splits_dir": str(splits_dir.relative_to(PROJECT_ROOT)),
                "datasets": args.datasets,
                "windows": args.windows,
                "seeds": args.seeds,
                "k_target": args.k_target,
                "repeats": args.repeats,
                "feature_columns": feature_cols,
                "summary": summary_df.replace({np.nan: None}).to_dict(orient="records"),
            },
            f,
            indent=2,
            allow_nan=False,
        )

    lines = [
        "# Four-Dataset Feature Transfer Stability",
        "",
        f"- Feature table: `{features_path.relative_to(PROJECT_ROOT)}`",
        f"- Splits: `{splits_dir.relative_to(PROJECT_ROOT)}`",
        f"- k-target residual adapter: {args.k_target}",
        "",
    ]
    for n_cycles in args.windows:
        lines.append(f"## N={n_cycles}")
        for direction in sorted(summary_df.loc[summary_df["n_cycles"] == n_cycles, "direction"].unique()):
            block = summary_df[
                (summary_df["n_cycles"] == n_cycles)
                & (summary_df["direction"] == direction)
            ].sort_values("transfer_stability_score", ascending=False)
            class_counts = block["stability_class"].value_counts().to_dict()
            lines.append(f"### `{direction}`")
            lines.append(f"- Class counts: {class_counts}")
            lines.append("- Top least-fragile features:")
            for _, row in block.head(args.top_k).iterrows():
                lines.append(
                    f"  - `{row['feature']}`: score={row['transfer_stability_score']:+.3f}, "
                    f"shift_z={row['abs_mean_shift_z']:.2f}, "
                    f"rho=({row['spearman_source']:+.2f}, {row['spearman_target']:+.2f}), "
                    f"adapted_R2={row['adapted_cross_R2_mean']:+.3f}, "
                    f"class={row['stability_class']}"
                )
            lines.append("")

    out_report.write_text("\n".join(lines) + "\n")
    print(f"[save] {out_summary}")
    print(f"[save] {out_detailed}")
    print(f"[save] {out_json}")
    print(f"[save] {out_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

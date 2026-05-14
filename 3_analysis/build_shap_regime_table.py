#!/usr/bin/env python3
"""Build SHAP x conditional-regime tables for the four-dataset paper extension.

The join is intentionally directional:
  - SHAP values come from the source dataset's within-dataset primary model.
  - Conditional-slope regimes come from the source -> target pair.

This gives a paper-facing answer to: are the features that matter within a
source dataset stable or slope-shifted when that source is transferred to a
target dataset?
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INTERMEDIATE_DIR = ROOT / "data/intermediate"

DEFAULT_TWO_DATASET_SHAP = INTERMEDIATE_DIR / "shap_feature_importance.csv"
DEFAULT_FOUR_DATASET_SHAP = INTERMEDIATE_DIR / "four_dataset_shap_feature_importance.csv"
DEFAULT_CONDITIONAL_SLOPES = INTERMEDIATE_DIR / "four_dataset_conditional_shift_feature_slopes.csv"
DEFAULT_DIRECTION_SUMMARY = INTERMEDIATE_DIR / "four_dataset_conditional_shift_direction_summary.csv"
DEFAULT_TRANSFER_STABILITY = INTERMEDIATE_DIR / "four_dataset_feature_transfer_stability.csv"

DATASETS = ["matr", "hust", "sandia", "luh"]
DATASET_LABELS = {
    "matr": "MATR",
    "hust": "HUST",
    "sandia": "Sandia",
    "luh": "Luh/KIT",
}


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_source_shap(two_dataset_path: Path, four_dataset_path: Path, n_cycles: int) -> pd.DataFrame:
    frames = []
    for path in [two_dataset_path, four_dataset_path]:
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        raise FileNotFoundError("No SHAP summary CSVs found.")

    shap = pd.concat(frames, ignore_index=True)
    shap = shap[shap["n_cycles"].eq(n_cycles)].copy()
    shap = shap[shap["dataset"].isin(DATASETS)].copy()
    shap = shap.sort_values(
        ["dataset", "feature", "n_seed_runs", "R2_mean"],
        ascending=[True, True, False, False],
    )
    shap = shap.drop_duplicates(["dataset", "feature"], keep="first")
    required = set(DATASETS)
    present = set(shap["dataset"].unique())
    missing = sorted(required - present)
    if missing:
        raise ValueError(f"Missing source SHAP summaries for datasets: {missing}")

    keep = [
        "dataset",
        "model",
        "feature",
        "mean_abs_shap_log_cycles_mean",
        "mean_abs_shap_log_cycles_std",
        "relative_importance_mean",
        "relative_importance_std",
        "rank_mean",
        "rank_std",
        "top5_rate",
        "top10_rate",
        "MAE_mean",
        "SMAPE_mean",
        "R2_mean",
        "n_seed_runs",
    ]
    keep = [col for col in keep if col in shap.columns]
    return shap[keep].rename(
        columns={
            "dataset": "source",
            "model": "source_shap_model",
            "MAE_mean": "source_within_MAE",
            "SMAPE_mean": "source_within_SMAPE",
            "R2_mean": "source_within_R2",
        }
    )


def build_ordered_regimes(slopes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for source in DATASETS:
        for target in DATASETS:
            if source == target:
                continue
            mask_direct = slopes["dataset_a"].eq(source) & slopes["dataset_b"].eq(target)
            mask_reverse = slopes["dataset_a"].eq(target) & slopes["dataset_b"].eq(source)
            pair_df = slopes[mask_direct | mask_reverse].copy()
            if pair_df.empty:
                raise ValueError(f"No conditional-slope rows found for {source}->{target}")
            direct = bool(mask_direct.loc[pair_df.index].iloc[0])
            for _, row in pair_df.iterrows():
                if direct:
                    slope_source = row["slope_a"]
                    slope_target = row["slope_b"]
                    delta = row["delta_slope_b_minus_a"]
                    offset = row["log_life_offset_b_minus_a"]
                    ratio = row["life_ratio_b_over_a"]
                else:
                    slope_source = row["slope_b"]
                    slope_target = row["slope_a"]
                    delta = -row["delta_slope_b_minus_a"]
                    offset = -row["log_life_offset_b_minus_a"]
                    ratio = 1.0 / row["life_ratio_b_over_a"]

                rows.append(
                    {
                        "source": source,
                        "target": target,
                        "direction": f"{source}_to_{target}",
                        "direction_label": f"{DATASET_LABELS[source]} -> {DATASET_LABELS[target]}",
                        "pair": row["pair"],
                        "feature": row["feature"],
                        "source_slope": slope_source,
                        "target_slope": slope_target,
                        "delta_slope_target_minus_source": delta,
                        "delta_slope_fdr_bh": row["delta_slope_fdr_bh"],
                        "slope_shift_significant": bool(row["slope_shift_significant"]),
                        "conditional_shift_class": row["shift_class"],
                        "log_life_offset_target_minus_source": offset,
                        "life_ratio_target_over_source": ratio,
                    }
                )
    return pd.DataFrame(rows)


def concise_feature_list(group: pd.DataFrame, n: int = 3, min_pct: float = 0.1) -> str:
    rows = group[group["relative_importance_pct"].ge(min_pct)].sort_values(
        "relative_importance_mean",
        ascending=False,
    ).head(n)
    if rows.empty:
        return "-"
    parts = []
    for _, row in rows.iterrows():
        parts.append(f"{row['feature']} ({100 * row['relative_importance_mean']:.1f}%)")
    return "; ".join(parts)


def build_tables(
    source_shap: pd.DataFrame,
    ordered_regimes: pd.DataFrame,
    direction_summary: pd.DataFrame,
    transfer_stability: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    feature = ordered_regimes.merge(source_shap, on=["source", "feature"], how="left", validate="many_to_one")
    if feature["relative_importance_mean"].isna().any():
        missing = feature.loc[feature["relative_importance_mean"].isna(), ["source", "feature"]].drop_duplicates()
        raise ValueError(f"Missing SHAP rows for source/feature pairs:\n{missing.to_string(index=False)}")

    stability_cols = [
        "source",
        "target",
        "feature",
        "abs_mean_shift_z",
        "spearman_source",
        "spearman_target",
        "spearman_abs_delta",
        "spearman_sign_agree",
        "transfer_stability_score",
        "stability_class",
        "adapted_cross_R2_mean",
    ]
    stability_cols = [col for col in stability_cols if col in transfer_stability.columns]
    feature = feature.merge(
        transfer_stability[stability_cols],
        on=["source", "target", "feature"],
        how="left",
        validate="one_to_one",
    )
    feature["relative_importance_pct"] = 100.0 * feature["relative_importance_mean"]
    feature["source_importance_rank_in_direction"] = (
        feature.groupby(["source", "target"])["relative_importance_mean"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    feature["is_top10_source_shap"] = feature["source_importance_rank_in_direction"].le(10)
    feature["is_top5_source_shap"] = feature["source_importance_rank_in_direction"].le(5)
    feature["is_slope_shifted"] = feature["conditional_shift_class"].eq("slope_shifted")

    summary_rows = []
    for (source, target), group in feature.groupby(["source", "target"], sort=False):
        shifted = group[group["is_slope_shifted"]]
        stable = group[~group["is_slope_shifted"]]
        top10 = group[group["is_top10_source_shap"]]
        top10_shifted = top10[top10["is_slope_shifted"]]
        top10_stable = top10[~top10["is_slope_shifted"]]
        summary_rows.append(
            {
                "source": source,
                "target": target,
                "direction": f"{source}_to_{target}",
                "direction_label": f"{DATASET_LABELS[source]} -> {DATASET_LABELS[target]}",
                "source_shap_model": group["source_shap_model"].iloc[0],
                "source_within_R2": group["source_within_R2"].iloc[0],
                "shap_mass_slope_shifted_pct": shifted["relative_importance_pct"].sum(),
                "shap_mass_slope_stable_pct": stable["relative_importance_pct"].sum(),
                "top10_slope_shifted_count": int(top10_shifted.shape[0]),
                "top10_slope_stable_count": int(top10_stable.shape[0]),
                "top_shifted_shap_features": concise_feature_list(shifted, 3),
                "top_stable_shap_features": concise_feature_list(stable, 3),
            }
        )
    direction = pd.DataFrame(summary_rows)

    direction_cols = [
        "source",
        "target",
        "model",
        "raw_R2",
        "pearson_r",
        "linear_R2",
        "rank_signal_class",
        "adapter_class",
        "slope_shifted_share",
        "n_slope_shifted_features",
        "life_ratio_target_over_source",
    ]
    direction_cols = [col for col in direction_cols if col in direction_summary.columns]
    direction = direction.merge(
        direction_summary[direction_cols].rename(columns={"model": "cross_best_model"}),
        on=["source", "target"],
        how="left",
        validate="one_to_one",
    )
    direction = direction.sort_values(
        ["shap_mass_slope_shifted_pct", "raw_R2"],
        ascending=[False, True],
    )

    markdown = build_markdown(direction)
    feature = feature.sort_values(["source", "target", "source_importance_rank_in_direction", "feature"])
    return feature, direction, markdown


def build_markdown(direction: pd.DataFrame) -> str:
    show = direction.copy()
    lines = [
        "# SHAP x Conditional-Regime Table",
        "",
        "SHAP values are from the source dataset's within-dataset primary model; ",
        "conditional regimes are direction-specific source -> target slope classes.",
        "",
        "| Direction | SHAP model | Shifted SHAP mass | Top-10 shifted/stable | Raw R² | Linear R² | Rank regime | Source-important shifted features |",
        "|---|---|---:|---:|---:|---:|---|---|",
    ]
    for _, row in show.iterrows():
        lines.append(
            f"| {row['direction_label']} | {row['source_shap_model']} | "
            f"{row['shap_mass_slope_shifted_pct']:.1f}% | "
            f"{int(row['top10_slope_shifted_count'])}/{int(row['top10_slope_stable_count'])} | "
            f"{row['raw_R2']:.3f} | {row['linear_R2']:.3f} | "
            f"{row['rank_signal_class']} | {row['top_shifted_shap_features']} |"
        )
    lines.extend(
        [
            "",
            "Top-10 shifted/stable counts are computed over the ten highest source-SHAP features for each direction.",
            "High shifted SHAP mass means the source model relies on features whose",
            "feature -> log-life slope changes significantly in the target dataset. Low shifted",
            "mass with positive rank signal is the most transfer-friendly regime.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--two-dataset-shap", type=Path, default=DEFAULT_TWO_DATASET_SHAP)
    parser.add_argument("--four-dataset-shap", type=Path, default=DEFAULT_FOUR_DATASET_SHAP)
    parser.add_argument("--conditional-slopes", type=Path, default=DEFAULT_CONDITIONAL_SLOPES)
    parser.add_argument("--direction-summary", type=Path, default=DEFAULT_DIRECTION_SUMMARY)
    parser.add_argument("--transfer-stability", type=Path, default=DEFAULT_TRANSFER_STABILITY)
    parser.add_argument("--n-cycles", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=INTERMEDIATE_DIR)
    args = parser.parse_args()

    source_shap = load_source_shap(args.two_dataset_shap, args.four_dataset_shap, args.n_cycles)
    slopes = pd.read_csv(args.conditional_slopes)
    direction_summary = pd.read_csv(args.direction_summary)
    transfer_stability = pd.read_csv(args.transfer_stability)

    ordered_regimes = build_ordered_regimes(slopes)
    feature_table, direction_table, markdown = build_tables(
        source_shap,
        ordered_regimes,
        direction_summary,
        transfer_stability,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = args.output_dir / "paper_shap_regime_feature_table.csv"
    direction_path = args.output_dir / "paper_shap_regime_direction_summary.csv"
    report_path = args.output_dir / "paper_shap_regime_table.md"

    feature_table.to_csv(feature_path, index=False)
    direction_table.to_csv(direction_path, index=False)
    report_path.write_text(markdown, encoding="utf-8")

    print(f"[save] {display_path(feature_path)}")
    print(f"[save] {display_path(direction_path)}")
    print(f"[save] {display_path(report_path)}")
    print()
    print(markdown)


if __name__ == "__main__":
    main()

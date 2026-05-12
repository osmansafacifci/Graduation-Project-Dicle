"""
Summarize the four-dataset raw-vs-capacity-normalized cross-dataset ablation.

Inputs:
    outputs/results_v2_four_dataset_cross_34feat_raw_log/results_summary.csv
    outputs/results_v2_four_dataset_cross_34feat_capnorm_log/results_summary.csv

Outputs:
    data/intermediate/four_dataset_raw_vs_capnorm_34feat_cross.csv
    data/intermediate/four_dataset_raw_vs_capnorm_34feat_cross_best.csv
    data/intermediate/four_dataset_raw_vs_capnorm_34feat_cross_report.md

Usage:
    python 3_analysis/summarize_four_dataset_raw_capnorm.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERMEDIATE_DIR = PROJECT_ROOT / "data" / "intermediate"
RAW_SUMMARY = PROJECT_ROOT / "outputs" / "results_v2_four_dataset_cross_34feat_raw_log" / "results_summary.csv"
CAPNORM_SUMMARY = PROJECT_ROOT / "outputs" / "results_v2_four_dataset_cross_34feat_capnorm_log" / "results_summary.csv"


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def add_source_target(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    parts = out["experiment"].str.split("_to_", expand=True)
    out["source"] = parts[0]
    out["target"] = parts[1]
    out["direction"] = out["experiment"]
    return out


def best_rows(df: pd.DataFrame, protocol: str) -> pd.DataFrame:
    rows: list[dict] = []
    for (direction, n_cycles), block in df.groupby(["direction", "n_cycles"], sort=True):
        # R2 is the main ranking metric; MAE breaks ties.
        sorted_block = block.sort_values(["R2_mean", "MAE_mean"], ascending=[False, True])
        best = sorted_block.iloc[0].to_dict()
        rows.append(
            {
                "protocol": protocol,
                "direction": direction,
                "source": best["source"],
                "target": best["target"],
                "n_cycles": int(n_cycles),
                "best_model": best["model"],
                "best_MAE_mean": best["MAE_mean"],
                "best_SMAPE_mean": best["SMAPE_mean"],
                "best_R2_mean": best["R2_mean"],
            }
        )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-summary", type=Path, default=RAW_SUMMARY)
    parser.add_argument("--capnorm-summary", type=Path, default=CAPNORM_SUMMARY)
    parser.add_argument("--output-prefix", default="four_dataset_raw_vs_capnorm_34feat_cross")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    raw_path = resolve_path(args.raw_summary)
    cap_path = resolve_path(args.capnorm_summary)
    if not raw_path.exists():
        print(f"[error] missing raw summary: {raw_path}")
        return 1
    if not cap_path.exists():
        print(f"[error] missing capnorm summary: {cap_path}")
        return 1

    raw = add_source_target(pd.read_csv(raw_path))
    cap = add_source_target(pd.read_csv(cap_path))
    key_cols = ["direction", "source", "target", "model", "n_cycles"]
    metric_cols = ["MAE_mean", "SMAPE_mean", "R2_mean", "MAE_std", "SMAPE_std", "R2_std"]
    merged = raw[key_cols + metric_cols].merge(
        cap[key_cols + metric_cols],
        on=key_cols,
        suffixes=("_raw", "_capnorm"),
        how="inner",
    )
    merged["delta_MAE_capnorm_minus_raw"] = merged["MAE_mean_capnorm"] - merged["MAE_mean_raw"]
    merged["delta_SMAPE_capnorm_minus_raw"] = merged["SMAPE_mean_capnorm"] - merged["SMAPE_mean_raw"]
    merged["delta_R2_capnorm_minus_raw"] = merged["R2_mean_capnorm"] - merged["R2_mean_raw"]
    merged["MAE_pct_change_capnorm_vs_raw"] = np.where(
        np.abs(merged["MAE_mean_raw"]) > 1e-12,
        100.0 * merged["delta_MAE_capnorm_minus_raw"] / merged["MAE_mean_raw"],
        np.nan,
    )
    merged["capnorm_better_MAE"] = merged["delta_MAE_capnorm_minus_raw"] < 0
    merged["capnorm_better_R2"] = merged["delta_R2_capnorm_minus_raw"] > 0

    raw_best = best_rows(raw, "raw")
    cap_best = best_rows(cap, "capnorm")
    best = raw_best.merge(
        cap_best,
        on=["direction", "source", "target", "n_cycles"],
        suffixes=("_raw", "_capnorm"),
    )
    best["delta_best_MAE_capnorm_minus_raw"] = best["best_MAE_mean_capnorm"] - best["best_MAE_mean_raw"]
    best["delta_best_R2_capnorm_minus_raw"] = best["best_R2_mean_capnorm"] - best["best_R2_mean_raw"]
    best["best_MAE_pct_change_capnorm_vs_raw"] = np.where(
        np.abs(best["best_MAE_mean_raw"]) > 1e-12,
        100.0 * best["delta_best_MAE_capnorm_minus_raw"] / best["best_MAE_mean_raw"],
        np.nan,
    )
    best["capnorm_better_best_MAE"] = best["delta_best_MAE_capnorm_minus_raw"] < 0
    best["capnorm_better_best_R2"] = best["delta_best_R2_capnorm_minus_raw"] > 0

    out_detail = INTERMEDIATE_DIR / f"{args.output_prefix}.csv"
    out_best = INTERMEDIATE_DIR / f"{args.output_prefix}_best.csv"
    out_report = INTERMEDIATE_DIR / f"{args.output_prefix}_report.md"
    merged.sort_values(["direction", "n_cycles", "model"]).to_csv(out_detail, index=False)
    best.sort_values(["direction", "n_cycles"]).to_csv(out_best, index=False)

    lines = [
        "# Four-Dataset Raw vs Capacity-Normalized Cross Ablation",
        "",
        f"- Raw summary: `{raw_path.relative_to(PROJECT_ROOT)}`",
        f"- Capnorm summary: `{cap_path.relative_to(PROJECT_ROOT)}`",
        "- Scope: 34 features, log-target, all seven models, ordered source->target pairs.",
        "",
        "## Aggregate Counts",
        "",
        f"- Model/window rows compared: {len(merged)}",
        f"- Capnorm improves MAE rows: {int(merged['capnorm_better_MAE'].sum())}/{len(merged)}",
        f"- Capnorm improves R2 rows: {int(merged['capnorm_better_R2'].sum())}/{len(merged)}",
        f"- Best-direction/window rows: {len(best)}",
        f"- Capnorm improves best-model MAE: {int(best['capnorm_better_best_MAE'].sum())}/{len(best)}",
        f"- Capnorm improves best-model R2: {int(best['capnorm_better_best_R2'].sum())}/{len(best)}",
        "",
        "## Best Model by Direction and Window",
        "",
        "| Direction | N | Raw Best | Raw MAE | Raw R2 | Capnorm Best | Capnorm MAE | Capnorm R2 | Delta MAE | Delta R2 |",
        "|---|---:|---|---:|---:|---|---:|---:|---:|---:|",
    ]
    for _, row in best.sort_values(["direction", "n_cycles"]).iterrows():
        lines.append(
            f"| `{row['direction']}` | {int(row['n_cycles'])} | "
            f"{row['best_model_raw']} | {row['best_MAE_mean_raw']:.1f} | {row['best_R2_mean_raw']:.3f} | "
            f"{row['best_model_capnorm']} | {row['best_MAE_mean_capnorm']:.1f} | {row['best_R2_mean_capnorm']:.3f} | "
            f"{row['delta_best_MAE_capnorm_minus_raw']:+.1f} | {row['delta_best_R2_capnorm_minus_raw']:+.3f} |"
        )
    lines.append("")
    out_report.write_text("\n".join(lines))

    print(f"[save] {out_detail}")
    print(f"[save] {out_best}")
    print(f"[save] {out_report}")
    print(
        "[summary] capnorm better best MAE "
        f"{int(best['capnorm_better_best_MAE'].sum())}/{len(best)}, "
        f"best R2 {int(best['capnorm_better_best_R2'].sum())}/{len(best)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

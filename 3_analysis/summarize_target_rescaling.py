"""
Paper-facing summaries for k-shot target calibration.

Reads the long-form output of target_rescaling.py and writes two compact
cross-check tables:
    1. For each source->target direction, choose the naive-best model by
       baseline R2, then show residual-mean and linear k=20 adapters.
    2. For each direction, choose the best k=20 adapter/model by adapted R2.

Usage:
    python 3_analysis/summarize_target_rescaling.py \
        --results outputs/results_v2_four_dataset_target_rescale/results_summary.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = PROJECT_ROOT / "outputs" / "results_v2_four_dataset_target_rescale" / "results_summary.csv"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "intermediate"


def markdown_table(df: pd.DataFrame, float_digits: int | None = None) -> str:
    formatted = df.copy()
    if float_digits is not None:
        for col in formatted.select_dtypes(include=[np.number]).columns:
            formatted[col] = formatted[col].map(lambda x: "" if pd.isna(x) else f"{x:.{float_digits}f}")
    formatted = formatted.astype(object).where(pd.notna(formatted), "")
    header = "| " + " | ".join(map(str, formatted.columns)) + " |"
    separator = "| " + " | ".join(["---"] * len(formatted.columns)) + " |"
    body = ["| " + " | ".join(map(str, row)) + " |" for row in formatted.to_numpy()]
    return "\n".join([header, separator, *body])


def build_naive_best_summary(df: pd.DataFrame, k: int) -> pd.DataFrame:
    candidates = (
        df[df["n_cycles"].eq(100)]
        .drop_duplicates(["experiment", "model"])
        .sort_values(["experiment", "baseline_R2"], ascending=[True, False])
        .groupby("experiment", as_index=False)
        .head(1)
    )
    rows = []
    for _, base in candidates.iterrows():
        subset = df[
            df["experiment"].eq(base["experiment"])
            & df["model"].eq(base["model"])
            & df["n_cycles"].eq(100)
            & df["k"].eq(k)
        ]
        row = {
            "experiment": base["experiment"],
            "source": base["source"],
            "target": base["target"],
            "naive_best_model": base["model"],
            "baseline_R2": base["baseline_R2"],
            "baseline_MAE": base["baseline_MAE"],
        }
        for adapter in ["residual_mean", "linear"]:
            match = subset[subset["adapter_type"].eq(adapter)]
            if match.empty:
                continue
            item = match.iloc[0]
            prefix = "residual" if adapter == "residual_mean" else "linear"
            row[f"{prefix}_R2"] = item["adapted_R2"]
            row[f"{prefix}_MAE"] = item["adapted_MAE"]
            row[f"{prefix}_delta_R2"] = item["delta_R2"]
            row[f"{prefix}_delta_MAE"] = item["delta_MAE"]
        rows.append(row)
    return pd.DataFrame(rows).sort_values("experiment").reset_index(drop=True)


def build_best_adapter_summary(df: pd.DataFrame, k: int) -> pd.DataFrame:
    cols = [
        "experiment",
        "source",
        "target",
        "adapter_type",
        "model",
        "baseline_R2",
        "adapted_R2",
        "delta_R2",
        "baseline_MAE",
        "adapted_MAE",
        "delta_MAE",
    ]
    return (
        df[df["n_cycles"].eq(100) & df["k"].eq(k)]
        .sort_values(["experiment", "adapted_R2"], ascending=[True, False])
        .groupby("experiment", as_index=False)
        .head(1)[cols]
        .sort_values("experiment")
        .reset_index(drop=True)
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--k", type=int, default=20)
    args = parser.parse_args()

    df = pd.read_csv(args.results)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    naive_best = build_naive_best_summary(df, args.k)
    best_adapter = build_best_adapter_summary(df, args.k)

    out_naive = args.out_dir / f"four_dataset_target_rescale_naive_best_k{args.k}.csv"
    out_best = args.out_dir / f"four_dataset_target_rescale_best_adapter_k{args.k}.csv"
    out_report = args.out_dir / f"four_dataset_target_rescale_report_k{args.k}.md"
    naive_best.to_csv(out_naive, index=False)
    best_adapter.to_csv(out_best, index=False)

    lines = [
        f"# Four-Dataset Target Calibration Summary (k={args.k})",
        "",
        "Source file: "
        + str(args.results.relative_to(PROJECT_ROOT) if args.results.is_relative_to(PROJECT_ROOT) else args.results),
        "",
        "## Naive-Best Model, Then Target Adapters",
        markdown_table(naive_best, float_digits=3),
        "",
        "## Best Adapter/Model Per Direction",
        markdown_table(best_adapter, float_digits=3),
        "",
        "Interpretation: the first table is the conservative audit table because it fixes the model selected by naive cross-dataset R2 before applying target calibration. The second table is useful as an upper envelope but should be described as adapter/model selection, not as a pre-registered protocol.",
    ]
    out_report.write_text("\n".join(lines) + "\n")

    print(f"[save] {out_naive.relative_to(PROJECT_ROOT)}")
    print(f"[save] {out_best.relative_to(PROJECT_ROOT)}")
    print(f"[save] {out_report.relative_to(PROJECT_ROOT)}")
    print(markdown_table(naive_best, float_digits=3))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

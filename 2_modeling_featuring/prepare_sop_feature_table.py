"""Prepare a SOP-oriented feature table from an existing CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sop_utils import FEATURE_SETS, ensure_columns_present


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "data" / "intermediate" / "features_top8_cycles.csv"
DEFAULT_SUPPLEMENT = PROJECT_ROOT / "data" / "intermediate" / "features_early_cycles.csv"
DEFAULT_LABELS = PROJECT_ROOT / "data" / "intermediate" / "raw_label_table.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "intermediate" / "features_sop12_transition.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a SOP-style feature CSV.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--supplement",
        type=Path,
        default=DEFAULT_SUPPLEMENT,
        help="Optional second CSV to merge by cell_id/n_cycles for richer SOP features.",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=DEFAULT_LABELS,
        help="Optional label CSV to merge by cell_id.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--feature-set",
        type=str,
        default="sop12_transition",
        choices=sorted(FEATURE_SETS),
    )
    parser.add_argument(
        "--label-column",
        type=str,
        default="cycle_life",
        help="Target column to keep in the exported CSV.",
    )
    parser.add_argument(
        "--keep-extra-columns",
        nargs="*",
        default=[],
        help="Optional extra columns to preserve (for example: eol_80_cycle is_censored).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise SystemExit(f"Input CSV not found: {args.input}")

    df = pd.read_csv(args.input)
    if args.supplement is not None and args.supplement.exists():
        supplement_df = pd.read_csv(args.supplement)
        base_keys = ["cell_id", "n_cycles"]
        base_only_cols = [
            column
            for column in supplement_df.columns
            if column not in set(base_keys) | set(df.columns)
        ]
        if base_only_cols:
            df = df.merge(
                supplement_df[base_keys + base_only_cols],
                on=base_keys,
                how="left",
            )

    if args.labels is not None and args.labels.exists():
        labels_df = pd.read_csv(args.labels)
        label_only_cols = [column for column in labels_df.columns if column != "cell_id" and column not in df.columns]
        if label_only_cols:
            df = df.merge(labels_df[["cell_id"] + label_only_cols], on="cell_id", how="left")

    feature_columns = FEATURE_SETS[args.feature_set]
    id_columns = ["cell_id"]
    if "dataset_prefix" in df.columns:
        id_columns.append("dataset_prefix")
    keep_columns = [*id_columns, "n_cycles", args.label_column, *feature_columns, *args.keep_extra_columns]
    ensure_columns_present(df, keep_columns)

    exported = df[keep_columns].copy()
    exported.sort_values(["cell_id", "n_cycles"], inplace=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    exported.to_csv(args.output, index=False)
    print(
        f"Saved {len(exported)} rows with feature set '{args.feature_set}' to {args.output}"
    )


if __name__ == "__main__":
    main()

"""Generate observed-only JSON splits for censored SOP targets."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from split_protocol_utils import DEFAULT_SEEDS, build_split_payload, save_split_json


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = PROJECT_ROOT / "data" / "intermediate" / "features_sop12_transition.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "splits" / "observed_only"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate observed-only SOP splits.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--dataset-prefix", type=str, default="b1")
    parser.add_argument("--censor-column", type=str, default="is_censored_80pct_q0")
    parser.add_argument("--label-column", type=str, default="eol_80pct_q0_label")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--calibration-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.dataset.exists():
        raise SystemExit(f"Dataset not found: {args.dataset}")

    df = pd.read_csv(args.dataset)
    required = {"cell_id", args.label_column, args.censor_column}
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise SystemExit(f"Dataset missing required columns: {missing}")

    subset = df[
        df["cell_id"].astype(str).str.startswith(args.dataset_prefix)
        & (pd.to_numeric(df[args.censor_column], errors="coerce") == 0)
    ].copy()
    subset.dropna(subset=[args.label_column], inplace=True)
    if subset.empty:
        raise SystemExit("No observed-only rows found for the requested dataset/censor pair.")

    cell_records = (
        subset.groupby("cell_id", as_index=False)[args.label_column]
        .first()
        .rename(columns={args.label_column: "cycle_life"})
        .sort_values("cell_id")
        .to_dict("records")
    )
    print(
        f"{args.dataset_prefix}: {len(cell_records)} uncensored cells "
        f"using label '{args.label_column}' and censor '{args.censor_column}'"
    )

    scenario_dir = args.output_dir / f"{args.dataset_prefix}_{args.label_column}"
    for seed in args.seeds:
        payload = build_split_payload(
            cell_records,
            train_ratio=args.train_ratio,
            calibration_ratio=args.calibration_ratio,
            test_ratio=args.test_ratio,
            seed=seed,
            stratify_bins=1,
        )
        output_path = scenario_dir / f"{args.dataset_prefix}_{seed}.json"
        save_split_json(payload, output_path)
        print(
            f"Saved {output_path} "
            f"(train={len(payload['train'])}, calibration={len(payload['calibration'])}, test={len(payload['test'])})"
        )


if __name__ == "__main__":
    main()

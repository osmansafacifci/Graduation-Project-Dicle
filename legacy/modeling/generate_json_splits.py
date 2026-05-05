"""
Generate SOP-compliant JSON splits for one or more datasets.

Rules:
- cell-ID-based splitting only
- 70/15/15 train/calibration/test
- 5 fixed seeds
- stratification by cycle-life quartiles
"""

from __future__ import annotations

import argparse
from pathlib import Path

from split_protocol_utils import (
    DEFAULT_SEEDS,
    build_cell_records,
    build_split_payload,
    load_feature_rows,
    save_split_json,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = PROJECT_ROOT / "data" / "intermediate" / "features_top8_cycles.csv"
DEFAULT_SPLIT_DIR = PROJECT_ROOT / "splits"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate stratified JSON splits for battery datasets.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--dataset-prefixes", nargs="+", default=["b1", "b2"])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    feature_rows = load_feature_rows(args.dataset)

    for dataset_prefix in args.dataset_prefixes:
        cell_records = build_cell_records(feature_rows, dataset_prefix)
        print(f"{dataset_prefix}: {len(cell_records)} cells")
        for seed in args.seeds:
            payload = build_split_payload(cell_records, seed=seed)
            output_path = args.output_dir / f"{dataset_prefix}_{seed}.json"
            save_split_json(payload, output_path)
            print(
                f"Saved {output_path} "
                f"(train={len(payload['train'])}, calibration={len(payload['calibration'])}, test={len(payload['test'])})"
            )


if __name__ == "__main__":
    main()

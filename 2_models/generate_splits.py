"""
Generate 70/15/15 cell-level splits for MATR and HUST, stratified by lifetime.

- 5 seeds: {42, 123, 456, 789, 1011}
- Cell-id based (no within-cell leakage between train/cal/test)
- Stratified by lifetime quartiles (within each quartile, apply 70/15/15)
- Censored cells are excluded before splitting (count reported in metadata)

Inputs:
    data/intermediate/features_sop12_combined.csv  (output of build_features.py)

Outputs:
    splits/sop_v2/{matr,hust}_{seed}.json   (5 seeds × 2 datasets = 10 files)

Usage:
    python 2_models/generate_splits.py
    python 2_models/generate_splits.py \
        --features-path data/intermediate/features_sop12_four_dataset.csv \
        --output-dir splits/sop_v2_four_dataset
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURES_PATH = PROJECT_ROOT / "data" / "intermediate" / "features_sop12_combined.csv"
SPLITS_DIR = PROJECT_ROOT / "splits" / "sop_v2"

SEEDS = [42, 123, 456, 789, 1011]
RATIOS = (0.70, 0.15, 0.15)


def make_splits_for_dataset(df: pd.DataFrame, dataset: str) -> dict[int, dict]:
    """For one dataset, return {seed: split_dict} using lifetime-quartile stratification."""
    # Use N=100 row per cell as the canonical anchor (cycle_life is per-cell, not per-N)
    pivot_n = 100 if 100 in df["n_cycles"].unique() else int(df["n_cycles"].min())
    cells = df[df["n_cycles"] == pivot_n][["cell_id", "cycle_life", "is_censored"]].copy()
    modeling = cells[cells["is_censored"] == 0].copy()
    censored = cells[cells["is_censored"] == 1]

    print(f"[{dataset}] cells with N={pivot_n}: {len(cells)} "
          f"(modeling: {len(modeling)}, censored excluded: {len(censored)})")

    # Lifetime quartile stratification
    if len(modeling) >= 4:
        modeling = modeling.assign(
            lifetime_bin=pd.qcut(modeling["cycle_life"], q=4, labels=False, duplicates="drop")
        )
    else:
        modeling = modeling.assign(lifetime_bin=0)

    out: dict[int, dict] = {}
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        train_ids: list[str] = []
        cal_ids: list[str] = []
        test_ids: list[str] = []
        for _, group in modeling.groupby("lifetime_bin"):
            ids = group["cell_id"].to_numpy().copy()
            rng.shuffle(ids)
            n = len(ids)
            n_train = max(1, int(round(n * RATIOS[0])))
            n_cal = max(1, int(round(n * RATIOS[1])))
            n_test = n - n_train - n_cal
            if n_test < 1:
                n_test = 1
                n_train = n - n_cal - n_test
            train_ids.extend(ids[:n_train].tolist())
            cal_ids.extend(ids[n_train:n_train + n_cal].tolist())
            test_ids.extend(ids[n_train + n_cal:].tolist())

        out[seed] = {
            "dataset": dataset,
            "seed": seed,
            "ratios": list(RATIOS),
            "stratified_by": "cycle_life_quartiles",
            "censored_excluded": True,
            "num_total_cells": int(len(cells)),
            "num_modeling_cells": int(len(modeling)),
            "num_censored_excluded": int(len(censored)),
            "train": sorted(train_ids),
            "calibration": sorted(cal_ids),
            "test": sorted(test_ids),
        }
        print(f"  seed={seed}: train={len(train_ids)} / cal={len(cal_ids)} / test={len(test_ids)}")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features-path", type=Path, default=FEATURES_PATH,
                        help="Feature table to split. Default: data/intermediate/features_sop12_combined.csv")
    parser.add_argument("--output-dir", type=Path, default=SPLITS_DIR,
                        help="Directory for split JSON files. Default: splits/sop_v2")
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Optional dataset subset. Default: all datasets in the feature table.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    features_path = args.features_path if args.features_path.is_absolute() else PROJECT_ROOT / args.features_path
    splits_dir = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    if not features_path.exists():
        print(f"[error] {features_path} missing — run build_features.py first.")
        return 1
    df = pd.read_csv(features_path)
    splits_dir.mkdir(parents=True, exist_ok=True)

    datasets = sorted(df["dataset"].unique()) if args.datasets is None else args.datasets
    print(f"[setup] features_path: {features_path.relative_to(PROJECT_ROOT)}")
    print(f"[setup] output_dir: {splits_dir.relative_to(PROJECT_ROOT)}")
    for dataset in datasets:
        print(f"\n[{dataset}]")
        sub = df[df["dataset"] == dataset]
        if sub.empty:
            print(f"  [skip] no rows")
            continue
        splits = make_splits_for_dataset(sub, dataset=dataset)
        for seed, split in splits.items():
            out = splits_dir / f"{dataset}_{seed}.json"
            with out.open("w") as f:
                json.dump(split, f, indent=2)
            print(f"  -> {out.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

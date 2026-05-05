"""
Split the engineered feature table into train/calibration/test partitions.

The script groups rows by ``cell_id`` so that every early-cycle window for
the same battery ends up in the same split, avoiding label leakage.

Usage:
    python 2_modeling_featuring/split_train_val_test.py \
        --source data/intermediate/features_top8_cycles.csv \
        --ratios 0.7 0.15 0.15 \
        --seed 42
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
INTERMEDIATE_DIR = DATA_DIR / "intermediate"
SPLITS_DIR = DATA_DIR / "splits"
DEFAULT_SOURCE = INTERMEDIATE_DIR / "features_top8_cycles.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser( # Reading the argument
        description="Create deterministic train/calibration/test splits from the feature table."
    )
    parser.add_argument(
        "--source", 
        type=Path,
        default=DEFAULT_SOURCE,
        help=f"Input CSV to split (default: {DEFAULT_SOURCE})",
    )
    parser.add_argument(
        "--ratios",
        type=float,
        nargs=3,
        metavar=("TRAIN", "CALIBRATION", "TEST"),
        default=(0.7, 0.15, 0.15), # I used to make train and test (0.8, 0.2)
        help="Split ratios that sum to 1.0 (default: 0.7 0.15 0.15)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for the RNG used to shuffle cell IDs.",
    )
    return parser.parse_args() #read the argument


def normalize_ratios(ratios: Iterable[float]) -> Tuple[float, float, float]: # to make sure that the sum of the ratios is 1.0
    ratios = np.asarray(list(ratios), dtype=float) # ratios -> np dizisi
    if (ratios <= 0).any():
        raise ValueError("All ratios must be positive.")
    total = ratios.sum()
    if not np.isclose(total, 1.0): # if sum isnt 1.0 it rearrange them by dividing the ratios of the total.
        ratios = ratios / total
    return tuple(float(r) for r in ratios)


def allocate_counts(num_cells: int, ratios: Tuple[float, float, float]) -> Tuple[int, int, int]:
    raw_counts = np.array([ratio * num_cells for ratio in ratios], dtype=float)
    counts = np.floor(raw_counts).astype(int) # floor = aşağı yuvarlama fonk.
    remainder = num_cells - counts.sum()
    if remainder > 0: # açıkta kalan piller varsa;
        ordering = np.argsort(raw_counts - counts)[::-1] 
        for idx in ordering[:remainder]:
            counts[idx] += 1

# Note to myself if I forget the logic of allocate_counts: 
# Yukarıdaki mantık şu: aşağı yuvarlayarak birkaç tane açıkta kalan pil oldu.
# Benim bu açıkta kalanları üç gruptan birine vermem lazım
# Yuvarlarken en çok kayıp verdiğim gruba koyarak durumları eşitlemeye çalıştım.
# Ör: 2.9 -> 2 ve 3.5 -> 3 yapmışsam; raw counts - counts = 2.9 - 2 = 0.9 ve 3.5 - 3 = 0.5 buldum.
# Kalan pilleri sonucu fazla olana yani 0.9 olana verdim.

    # Make sure that no split is empty if possible. If anyone is 0 I give 1 battery from most battery split.
    for idx in range(len(counts)):
        if counts[idx] == 0:
            donor = int(np.argmax(counts))
            if counts[donor] <= 1:
                continue
            counts[idx] += 1
            counts[donor] -= 1

    if counts.sum() != num_cells: # Last control; to make no mistake
        raise RuntimeError("Failed to allocate cell counts across splits.")
    return tuple(int(c) for c in counts)


def write_split(
    df: pd.DataFrame, cells: set[str], stem: str, split_name: str, output_dir: Path
) -> tuple[Path, int]:
    subset = df[df["cell_id"].isin(cells)].copy() # copy the cell_id part
    out_path = output_dir / f"{stem}_{split_name}.csv"
    subset.to_csv(out_path, index=False) # bunu yeni öğrendim index = false yaparsam başa gereksiz numara eklemiyor.
    return out_path, subset.shape[0]


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    if not source.exists():
        raise SystemExit(f"Source CSV not found: {source}")

    ratios = normalize_ratios(args.ratios)
    df = pd.read_csv(source)
    if "cell_id" not in df.columns:
        raise SystemExit("Input CSV must include a 'cell_id' column.")

    cells = np.array(sorted(df["cell_id"].unique()))
    rng = np.random.default_rng(args.seed)
    rng.shuffle(cells)

    train_count, val_count, test_count = allocate_counts(len(cells), ratios)
    train_cells = set(cells[:train_count])
    val_cells = set(cells[train_count : train_count + val_count])
    test_cells = set(cells[train_count + val_count : train_count + val_count + test_count])
    # I don't have to write test_count variable. Kafam karışmasın diye yazdım. 
    # Temiz bir kod oluşturmak için silinebilir.

    stem = source.stem
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    outputs = [
        ("train", train_cells),
        ("calibration", val_cells),
        ("test", test_cells),
    ]

    for split_name, split_cells in outputs:
        path, row_count = write_split(df, split_cells, stem, split_name, SPLITS_DIR)
        print(f"Wrote {len(split_cells)} cells / {row_count} rows to {path}")

    print(
        f"Split complete: train={len(train_cells)} cells, "
        f"calibration={len(val_cells)} cells, test={len(test_cells)} cells"
    )


if __name__ == "__main__":
    main()

"""
Generate a light-weight feature table with the eight preferred signals.

The script reads the available ``batch*.pkl`` files (produced from the
MATLAB batch sources), derives the requested statistics for
``n_cycles`` = 25/50/100, and writes the result to
``features_top8_cycles.csv`` at the project root.

Usage:
    python 1_feature_engineering/build_features_top8.py
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERMEDIATE_DIR = DATA_DIR / "intermediate"
PKL_PATHS = (
    RAW_DIR / "batch1.pkl",
    RAW_DIR / "batch2.pkl",
    RAW_DIR / "batch3_varcharge.pkl",
)
OUT_PATH = INTERMEDIATE_DIR / "features_top8_cycles.csv"
N_CYCLE_WINDOWS: Iterable[int] = (25, 50, 100)


def slope(values: np.ndarray) -> float:
    """Return the linear slope of ``values`` across cycles."""
    if values.size <= 1 or np.allclose(values, values[0]):
        return 0.0
    x = np.arange(values.size)
    return float(np.polyfit(x, values, 1)[0])


def delta(values: np.ndarray) -> float:
    """Return last - first; 0 when insufficient samples."""
    if values.size == 0:
        return 0.0
    return float(values[-1] - values[0])


def load_batches() -> Dict[str, dict]:
    """Load and combine every batch pickle listed in ``PKL_PATHS``."""
    combined: Dict[str, dict] = {}
    missing = []
    for path in PKL_PATHS:
        if not path.exists():
            missing.append(path)
            continue
        with path.open("rb") as fp:
            batch_data: Dict[str, dict] = pickle.load(fp)
        overlap = combined.keys() & batch_data.keys()
        if overlap:
            overlap_str = ", ".join(sorted(overlap))
            raise ValueError(f"Duplicate cell IDs across pickles: {overlap_str}")
        combined.update(batch_data)

    if not combined:
        missing_paths = ", ".join(str(p) for p in PKL_PATHS)
        raise SystemExit(f"No batch pickles available. Expected: {missing_paths}")

    if missing:
        missing_str = ", ".join(str(p) for p in missing)
        print(f"Warning: skipped missing pickles: {missing_str}")

    return combined


def main() -> None:
    bat_dict = load_batches()

    rows: list[dict[str, float | int | str]] = []

    for cell_id, cell_data in bat_dict.items():
        cycle_life_raw = cell_data.get("cycle_life")
        if cycle_life_raw is None or len(cycle_life_raw) == 0:
            continue
        cycle_life_arr = np.asarray(cycle_life_raw, dtype=float).ravel()
        if cycle_life_arr.size == 0:
            continue
        cycle_life = float(cycle_life_arr[0])

        summary = cell_data["summary"]
        qd = np.asarray(summary["QD"], dtype=float)
        ir = np.asarray(summary["IR"], dtype=float)
        tavg = np.asarray(summary["Tavg"], dtype=float)

        for window in N_CYCLE_WINDOWS:
            max_idx = min(window, qd.size, ir.size, tavg.size)
            if max_idx < 2:
                # Skip cells that do not have enough cycles for this window.
                continue

            qd_win = qd[:max_idx]
            ir_win = ir[:max_idx]
            tavg_win = tavg[:max_idx]

            rows.append(
                {
                    "cell_id": cell_id,
                    "dataset_prefix": cell_id.split("c", 1)[0],
                    "n_cycles": window,
                    "cycle_life": cycle_life,
                    "Qd_mean": float(np.mean(qd_win)),
                    "Qd_std": float(np.std(qd_win)),
                    "IR_mean": float(np.mean(ir_win)),
                    "IR_std": float(np.std(ir_win)),
                    "IR_delta": delta(ir_win),
                    "IR_slope": slope(ir_win),
                    "Tavg_mean": float(np.mean(tavg_win)),
                    "dQd_slope": slope(qd_win),
                }
            )

    if not rows:
        raise SystemExit("No rows were generated. Check the pickle contents.")

    df = pd.DataFrame(rows)
    df.sort_values(["cell_id", "n_cycles"], inplace=True)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(df)} rows to {OUT_PATH}")


if __name__ == "__main__":
    main()

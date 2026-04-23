"""Build SOP feature tables for MATR and HUST.

MATR supports the full 12-feature SOP schema. HUST does not contain internal
resistance or temperature signals, so it supports the capacity + dQ/dV common
subset used for external cross-dataset evaluation.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
INTERMEDIATE_DIR = PROJECT_ROOT / "data" / "intermediate"
DEFAULT_HUST_DIR = Path("/Users/diclesaracoban/Downloads/HUST_data")
WINDOWS = (25, 50, 100)

SOP12_FEATURES = [
    "IR_delta",
    "dQd_slope",
    "Qd_mean",
    "IR_slope",
    "Tavg_mean",
    "IR_mean",
    "Qd_std",
    "IR_std",
    "dqdv_peak_delta",
    "dqdv_peak_std",
    "dqdv_area_delta",
    "dqdv_peakpos_delta",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build SOP12 and SOP-common feature tables.")
    parser.add_argument("--hust-dir", type=Path, default=DEFAULT_HUST_DIR)
    parser.add_argument("--labels", type=Path, default=INTERMEDIATE_DIR / "raw_label_table.csv")
    parser.add_argument("--matr-output", type=Path, default=INTERMEDIATE_DIR / "features_matr_sop12.csv")
    parser.add_argument("--hust-output", type=Path, default=INTERMEDIATE_DIR / "features_hust_sop_common.csv")
    parser.add_argument(
        "--combined-output",
        type=Path,
        default=INTERMEDIATE_DIR / "features_matr_hust_sop_common.csv",
    )
    return parser.parse_args()


def slope(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if values.size <= 1 or np.allclose(values, values[0]):
        return 0.0
    return float(np.polyfit(np.arange(values.size), values, 1)[0])


def delta(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    return float(values[-1] - values[0])


def summarize_curve_features(curves: list[np.ndarray], cycle_nums: list[int]) -> dict[str, float]:
    peaks: list[float] = []
    areas: list[float] = []
    peak_positions: list[float] = []
    kept_cycles: list[int] = []

    for curve, cycle_num in zip(curves, cycle_nums):
        finite = np.asarray(curve, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size < 2:
            continue
        peak_idx = int(np.nanargmax(finite))
        peaks.append(float(np.nanmax(finite)))
        areas.append(float(np.nansum(finite)))
        peak_positions.append(float(peak_idx / max(1, finite.size - 1)))
        kept_cycles.append(cycle_num)

    if len(peaks) < 2:
        return {
            "dqdv_peak_delta": float("nan"),
            "dqdv_peak_std": float("nan"),
            "dqdv_area_delta": float("nan"),
            "dqdv_peakpos_delta": float("nan"),
        }

    peaks_arr = np.asarray(peaks, dtype=float)
    areas_arr = np.asarray(areas, dtype=float)
    pos_arr = np.asarray(peak_positions, dtype=float)
    return {
        "dqdv_peak_delta": float(peaks_arr[-1] - peaks_arr[0]),
        "dqdv_peak_std": float(np.nanstd(peaks_arr)),
        "dqdv_area_delta": float(areas_arr[-1] - areas_arr[0]),
        "dqdv_peakpos_delta": float(pos_arr[-1] - pos_arr[0]),
    }


def load_matr_batches() -> dict[str, Any]:
    combined: dict[str, Any] = {}
    for path in [RAW_DIR / "batch1.pkl", RAW_DIR / "batch2.pkl", RAW_DIR / "batch3_varcharge.pkl"]:
        if not path.exists():
            continue
        with path.open("rb") as handle:
            batch = pickle.load(handle)
        overlap = set(combined) & set(batch)
        if overlap:
            raise ValueError(f"Duplicate cell IDs across raw batches: {sorted(overlap)}")
        combined.update(batch)
    if not combined:
        raise SystemExit("No MATR raw pickle files found.")
    return combined


def build_matr_features(labels_path: Path) -> pd.DataFrame:
    cells = load_matr_batches()
    rows: list[dict[str, float | int | str]] = []
    for cell_id, cell_data in sorted(cells.items()):
        summary = cell_data["summary"]
        qd = np.asarray(summary["QD"], dtype=float).ravel()
        ir = np.asarray(summary["IR"], dtype=float).ravel()
        tavg = np.asarray(summary["Tavg"], dtype=float).ravel()

        for window in WINDOWS:
            max_idx = min(window, qd.size, ir.size, tavg.size)
            if max_idx < 2:
                continue
            qd_win = qd[:max_idx]
            ir_win = ir[:max_idx]
            tavg_win = tavg[:max_idx]

            curves: list[np.ndarray] = []
            cycle_nums: list[int] = []
            for idx in range(window):
                cycle = cell_data["cycles"].get(str(idx))
                if cycle is None:
                    continue
                curves.append(np.asarray(cycle.get("dQdV", []), dtype=float).ravel())
                cycle_nums.append(idx)

            rows.append(
                {
                    "cell_id": cell_id,
                    "dataset_prefix": "matr",
                    "source_batch": cell_id.split("c", 1)[0],
                    "n_cycles": window,
                    "Qd_mean": float(np.nanmean(qd_win)),
                    "Qd_std": float(np.nanstd(qd_win)),
                    "IR_mean": float(np.nanmean(ir_win)),
                    "IR_std": float(np.nanstd(ir_win)),
                    "IR_delta": delta(ir_win),
                    "IR_slope": slope(ir_win),
                    "Tavg_mean": float(np.nanmean(tavg_win)),
                    "dQd_slope": slope(qd_win),
                    **summarize_curve_features(curves, cycle_nums),
                }
            )

    df = pd.DataFrame(rows)
    if labels_path.exists():
        labels = pd.read_csv(labels_path)
        label_columns = [
            column
            for column in labels.columns
            if column != "dataset_prefix" and column not in {"source_batch"}
        ]
        df = df.merge(labels[label_columns], on="cell_id", how="left")
    df["cycle_life"] = df["eol_85pct_q0_label"]
    return df


def hust_dqdv_curve(cycle_df: pd.DataFrame) -> np.ndarray:
    discharge = cycle_df[cycle_df["Status"].astype(str).str.contains("discharge", case=False, na=False)]
    if discharge.empty:
        return np.asarray([], dtype=float)
    v = pd.to_numeric(discharge["Voltage (V)"], errors="coerce").to_numpy(dtype=float)
    q = pd.to_numeric(discharge["Capacity (mAh)"], errors="coerce").to_numpy(dtype=float) / 1000.0
    valid = np.isfinite(v) & np.isfinite(q)
    v = v[valid]
    q = q[valid]
    if v.size < 5:
        return np.asarray([], dtype=float)
    order = np.argsort(v)
    v = v[order]
    q = q[order]
    unique_v, unique_idx = np.unique(v, return_index=True)
    unique_q = q[unique_idx]
    if unique_v.size < 5 or np.ptp(unique_v) == 0:
        return np.asarray([], dtype=float)
    grid = np.linspace(float(unique_v[0]), float(unique_v[-1]), 200)
    q_grid = np.interp(grid, unique_v, unique_q)
    dqdv = np.gradient(q_grid, grid)
    return np.abs(dqdv[np.isfinite(dqdv)])


def first_eol_80(qd: np.ndarray) -> tuple[float, int]:
    positive = qd[np.isfinite(qd) & (qd > 0)]
    if positive.size == 0:
        return float("nan"), 1
    q0 = float(positive[0])
    hits = np.where(np.isfinite(qd) & (qd > 0) & (qd <= 0.8 * q0))[0]
    if hits.size:
        return float(hits[0] + 1), 0
    return float(len(qd) + 1), 1


def first_eol_pct(qd: np.ndarray, fraction: float) -> tuple[float, int]:
    positive = qd[np.isfinite(qd) & (qd > 0)]
    if positive.size == 0:
        return float("nan"), 1
    q0 = float(positive[0])
    hits = np.where(np.isfinite(qd) & (qd > 0) & (qd <= fraction * q0))[0]
    if hits.size:
        return float(hits[0] + 1), 0
    return float(len(qd) + 1), 1


def build_hust_features(hust_dir: Path) -> pd.DataFrame:
    if not hust_dir.exists():
        raise SystemExit(f"HUST directory not found: {hust_dir}")

    rows: list[dict[str, float | int | str]] = []
    for path in sorted(hust_dir.glob("*.pkl")):
        with path.open("rb") as handle:
            raw = pickle.load(handle)
        raw_cell_id, cell = next(iter(raw.items()))
        cell_id = f"hust_{raw_cell_id}"
        cycles = sorted(int(cycle) for cycle in cell["dq"])
        qd = np.asarray([float(cell["dq"][cycle]) / 1000.0 for cycle in cycles], dtype=float)
        cycle_life_80, is_censored_80 = first_eol_pct(qd, 0.80)
        cycle_life_85, is_censored_85 = first_eol_pct(qd, 0.85)

        for window in WINDOWS:
            max_idx = min(window, len(cycles), qd.size)
            if max_idx < 2:
                continue
            selected_cycles = cycles[:max_idx]
            curves = [hust_dqdv_curve(cell["data"][cycle]) for cycle in selected_cycles]
            qd_win = qd[:max_idx]
            rows.append(
                {
                    "cell_id": cell_id,
                    "dataset_prefix": "hust",
                    "source_batch": "hust",
                    "n_cycles": window,
                    "cycle_life": cycle_life_85,
                    "eol_80pct_q0_label": cycle_life_80,
                    "is_censored_80pct_q0": is_censored_80,
                    "eol_85pct_q0_label": cycle_life_85,
                    "is_censored_85pct_q0": is_censored_85,
                    "Qd_mean": float(np.nanmean(qd_win)),
                    "Qd_std": float(np.nanstd(qd_win)),
                    "dQd_slope": slope(qd_win),
                    "IR_delta": float("nan"),
                    "IR_slope": float("nan"),
                    "Tavg_mean": float("nan"),
                    "IR_mean": float("nan"),
                    "IR_std": float("nan"),
                    **summarize_curve_features(curves, selected_cycles),
                }
            )
    return pd.DataFrame(rows)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved {len(df)} rows to {path}")


def main() -> None:
    args = parse_args()
    matr = build_matr_features(args.labels)
    hust = build_hust_features(args.hust_dir)
    common_features = [
        "Qd_mean",
        "Qd_std",
        "dQd_slope",
        "dqdv_peak_delta",
        "dqdv_peak_std",
        "dqdv_area_delta",
        "dqdv_peakpos_delta",
    ]
    common_columns = [
        "cell_id",
        "dataset_prefix",
        "source_batch",
        "n_cycles",
        "cycle_life",
        "eol_80pct_q0_label",
        "is_censored_80pct_q0",
        "eol_85pct_q0_label",
        "is_censored_85pct_q0",
        *common_features,
    ]

    write_csv(matr, args.matr_output)
    write_csv(hust[common_columns], args.hust_output)
    write_csv(pd.concat([matr[common_columns], hust[common_columns]], ignore_index=True), args.combined_output)


if __name__ == "__main__":
    main()

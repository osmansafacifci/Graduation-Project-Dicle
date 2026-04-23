"""
Build the same early-cycle feature table from BatteryML processed pickles.

This is intended for external datasets such as HUST. It supports both
BatteryML-style BatteryData pickles and the HUST pickles shaped as
{cell_id: {"rul": ..., "dq": ..., "data": ...}}. The output schema matches
features_top8_cycles.csv and includes dataset_prefix so it can be used by the
cross-dataset evaluator.

Example:
    python 1_feature_engineering/build_batteryml_top8_features.py \
        --input-dir data/processed/HUST \
        --dataset-prefix hust \
        --output data/intermediate/features_hust_top8_cycles.csv
"""

from __future__ import annotations

import argparse
import math
import pickle
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "processed" / "HUST"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "intermediate" / "features_hust_top8_cycles.csv"
N_CYCLE_WINDOWS: Iterable[int] = (25, 50, 100)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create top8-compatible features from BatteryML processed pickles."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dataset-prefix", type=str, default="hust")
    parser.add_argument("--windows", nargs="+", type=int, default=list(N_CYCLE_WINDOWS))
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


def get_value(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def array_or_empty(values: Any) -> np.ndarray:
    if values is None:
        return np.asarray([], dtype=float)
    arr = np.asarray(values, dtype=float).ravel()
    return arr[np.isfinite(arr)]


def cycle_capacity(cycle: Any) -> float:
    values = array_or_empty(get_value(cycle, "discharge_capacity_in_Ah"))
    positive = values[values > 0]
    return float(np.nanmax(positive)) if positive.size else float("nan")


def cycle_temperature(cycle: Any) -> float:
    values = array_or_empty(get_value(cycle, "temperature_in_C"))
    return float(np.nanmean(values)) if values.size else float("nan")


def cycle_resistance(cycle: Any) -> float:
    value = get_value(cycle, "internal_resistance_in_ohm")
    if value is None:
        return float("nan")
    arr = array_or_empty(value)
    return float(np.nanmean(arr)) if arr.size else float("nan")


def first_positive(values: np.ndarray) -> float:
    positive = values[np.isfinite(values) & (values > 0)]
    return float(positive[0]) if positive.size else float("nan")


def first_eol_80(qd: np.ndarray) -> tuple[float, int]:
    q0 = first_positive(qd)
    if not math.isfinite(q0):
        return float("nan"), 1
    hits = np.where(np.isfinite(qd) & (qd > 0) & (qd <= 0.8 * q0))[0]
    if hits.size:
        return float(hits[0] + 1), 0
    return float(len(qd) + 1), 1


def normalize_batteries(raw: Any, fallback_cell_id: str) -> list[tuple[str, Any]]:
    if get_value(raw, "cycle_data") is not None:
        return [(str(get_value(raw, "cell_id", fallback_cell_id)), raw)]
    if isinstance(raw, dict):
        batteries = []
        for key, value in raw.items():
            if get_value(value, "cycle_data") is not None or (
                isinstance(value, dict) and "dq" in value
            ):
                batteries.append((str(get_value(value, "cell_id", key)), value))
        if batteries:
            return batteries
    raise ValueError("Pickle does not look like BatteryML BatteryData.")


def build_rows_for_hust_cell(
    cell_id: str,
    battery: dict[str, Any],
    *,
    dataset_prefix: str,
    windows: list[int],
) -> list[dict[str, float | int | str]]:
    dq = battery["dq"]
    cycles = sorted(int(cycle) for cycle in dq)
    qd = np.asarray([float(dq[cycle]) / 1000.0 for cycle in cycles], dtype=float)
    cycle_life, is_censored = first_eol_80(qd)

    rows: list[dict[str, float | int | str]] = []
    for window in windows:
        max_idx = min(window, qd.size)
        if max_idx < 2:
            continue
        qd_win = qd[:max_idx]
        rows.append(
            {
                "cell_id": cell_id,
                "dataset_prefix": dataset_prefix,
                "n_cycles": window,
                "cycle_life": cycle_life,
                "eol_80pct_q0_label": cycle_life,
                "is_censored_80pct_q0": is_censored,
                "Qd_mean": float(np.nanmean(qd_win)),
                "Qd_std": float(np.nanstd(qd_win)),
                "IR_mean": float("nan"),
                "IR_std": float("nan"),
                "IR_delta": float("nan"),
                "IR_slope": float("nan"),
                "Tavg_mean": float("nan"),
                "dQd_slope": slope(qd_win),
            }
        )
    return rows


def build_rows_for_battery(
    cell_id: str,
    battery: Any,
    *,
    dataset_prefix: str,
    windows: list[int],
) -> list[dict[str, float | int | str]]:
    if isinstance(battery, dict) and "dq" in battery:
        return build_rows_for_hust_cell(
            cell_id,
            battery,
            dataset_prefix=dataset_prefix,
            windows=windows,
        )

    cycle_data = list(get_value(battery, "cycle_data", []))
    cycle_data.sort(key=lambda cycle: int(get_value(cycle, "cycle_number", 0)))
    qd = np.asarray([cycle_capacity(cycle) for cycle in cycle_data], dtype=float)
    tavg = np.asarray([cycle_temperature(cycle) for cycle in cycle_data], dtype=float)
    ir = np.asarray([cycle_resistance(cycle) for cycle in cycle_data], dtype=float)
    cycle_life, is_censored = first_eol_80(qd)

    rows: list[dict[str, float | int | str]] = []
    for window in windows:
        max_idx = min(window, qd.size, tavg.size, ir.size)
        if max_idx < 2:
            continue
        qd_win = qd[:max_idx]
        ir_win = ir[:max_idx]
        tavg_win = tavg[:max_idx]
        rows.append(
            {
                "cell_id": cell_id,
                "dataset_prefix": dataset_prefix,
                "n_cycles": window,
                "cycle_life": cycle_life,
                "eol_80pct_q0_label": cycle_life,
                "is_censored_80pct_q0": is_censored,
                "Qd_mean": float(np.nanmean(qd_win)),
                "Qd_std": float(np.nanstd(qd_win)),
                "IR_mean": float(np.nanmean(ir_win)),
                "IR_std": float(np.nanstd(ir_win)),
                "IR_delta": delta(ir_win),
                "IR_slope": slope(ir_win),
                "Tavg_mean": float(np.nanmean(tavg_win)),
                "dQd_slope": slope(qd_win),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    if not args.input_dir.exists():
        raise SystemExit(f"Input directory not found: {args.input_dir}")

    rows: list[dict[str, float | int | str]] = []
    for path in sorted(args.input_dir.rglob("*.pkl")):
        with path.open("rb") as handle:
            raw = pickle.load(handle)
        for cell_id, battery in normalize_batteries(raw, path.stem):
            if not cell_id.startswith(args.dataset_prefix):
                cell_id = f"{args.dataset_prefix}_{cell_id}"
            rows.extend(
                build_rows_for_battery(
                    cell_id,
                    battery,
                    dataset_prefix=args.dataset_prefix,
                    windows=args.windows,
                )
            )

    if not rows:
        raise SystemExit(f"No feature rows were generated from {args.input_dir}")

    df = pd.DataFrame(rows).sort_values(["cell_id", "n_cycles"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df)} rows to {args.output}")


if __name__ == "__main__":
    main()

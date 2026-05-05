"""
HUST cell-level cycle/threshold audit.

Faithful port of the HUST_Standalone_Preprocess_and_Audit notebook
(https://drive.google.com/drive/folders/1RVASMPuhWPbgJQE1G4z1856Na4jpxqPf).
This script does not depend on BatteryML — it reads the raw HUST per-cell pkls
directly and computes per-cycle total discharge capacity by integrating
current × time over each cycle's discharge segment.

Inputs (from HUST Drive folder; place under data/raw/HUST_data/):
    data/raw/HUST_data/1-1.pkl
    data/raw/HUST_data/1-2.pkl
    ...
    (77 cells total; pkl structure: pickle.load(f)[cell_id]['data'][cycle_number]
     -> pandas.DataFrame with columns 'Current (mA)', 'Time (s)', 'Voltage (V)')

Outputs:
    data/intermediate/hust_cycles_tidy.csv
        long table, one row per (cell_id, cycle); ~146k rows for 77 cells
    data/intermediate/hust_threshold_audit.csv
        per-cell summary (Q0, observed_cycles, min retention, EOL @ 90/85/80%)
    data/intermediate/hust_threshold_summary.csv
        global summary line (median/min retention, pct reaching each threshold)

For HUST, per-cycle discharge capacity is the TOTAL discharge across all
discharge stages in that cycle (multi-stage protocol). Special case: cell 7-5
drops cycles 1-2 and renumbers (matches BatteryML preprocessor convention).

Q0 = median(Q_discharge[cycles 2..5]).
EOL = first cycle (>= 2) where Q_discharge <= threshold * Q0.

Usage:
    python 0_data/build_hust_audit.py
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HUST_DIR = PROJECT_ROOT / "data" / "raw" / "HUST_data"
OUT_DIR = PROJECT_ROOT / "data" / "intermediate"

# Discharge-rate metadata for the 77 HUST cells (from BatteryML preprocessor notes)
DISCHARGE_RATES: dict[str, list[int]] = {
    "1-1": [5, 1, 1], "1-2": [5, 1, 2], "1-3": [5, 1, 3], "1-4": [5, 1, 4], "1-5": [5, 1, 5],
    "1-6": [5, 2, 1], "1-7": [5, 2, 2], "1-8": [5, 2, 3], "2-2": [5, 2, 5], "2-3": [5, 3, 1],
    "2-4": [5, 3, 2], "2-5": [5, 3, 3], "2-6": [5, 3, 4], "2-7": [5, 3, 5], "2-8": [5, 4, 1],
    "3-1": [5, 4, 2], "3-2": [5, 4, 3], "3-3": [5, 4, 4], "3-4": [5, 4, 5], "3-5": [5, 5, 1],
    "3-6": [5, 5, 2], "3-7": [5, 5, 3], "3-8": [5, 5, 4], "4-1": [5, 5, 5], "4-2": [4, 1, 1],
    "4-3": [4, 1, 2], "4-4": [4, 1, 3], "4-5": [4, 1, 4], "4-6": [4, 1, 5], "4-7": [4, 2, 1],
    "4-8": [4, 2, 2], "5-1": [4, 2, 3], "5-2": [4, 2, 4], "5-3": [4, 2, 5], "5-4": [4, 3, 1],
    "5-5": [4, 3, 2], "5-6": [4, 3, 3], "5-7": [4, 3, 4], "6-1": [4, 4, 1], "6-2": [4, 4, 2],
    "6-3": [4, 4, 3], "6-4": [4, 4, 4], "6-5": [4, 4, 5], "6-6": [4, 5, 1], "6-8": [4, 5, 3],
    "7-1": [4, 5, 4], "7-2": [4, 5, 5], "7-3": [3, 1, 1], "7-4": [3, 1, 2], "7-5": [3, 1, 3],
    "7-6": [3, 1, 4], "7-7": [3, 1, 5], "7-8": [3, 2, 1], "8-1": [3, 2, 2], "8-2": [3, 2, 3],
    "8-3": [3, 2, 4], "8-4": [3, 2, 5], "8-5": [3, 3, 1], "8-6": [3, 3, 2], "8-7": [3, 3, 3],
    "8-8": [3, 3, 4], "9-1": [3, 3, 5], "9-2": [3, 4, 1], "9-3": [3, 4, 2], "9-4": [3, 4, 3],
    "9-5": [3, 4, 4], "9-6": [3, 4, 5], "9-7": [3, 5, 1], "9-8": [3, 5, 2], "10-1": [3, 5, 3],
    "10-2": [3, 5, 4], "10-3": [3, 5, 5], "10-4": [2, 4, 1], "10-5": [2, 5, 2], "10-6": [2, 3, 3],
    "10-7": [2, 2, 4], "10-8": [2, 1, 5],
}


# ---------- per-cycle capacity (Coulomb counting) ----------

def calc_capacity_ah(current_a, time_s, *, is_charge: bool) -> np.ndarray:
    current_a = np.asarray(current_a, dtype=float)
    time_s = np.asarray(time_s, dtype=float)
    if len(current_a) != len(time_s) or len(current_a) == 0:
        return np.array([], dtype=float)
    q = np.zeros(len(current_a), dtype=float)
    for i in range(1, len(current_a)):
        dt = time_s[i] - time_s[i - 1]
        if not np.isfinite(dt) or dt < 0:
            dt = 0.0
        if is_charge and current_a[i] > 0:
            q[i] = q[i - 1] + current_a[i] * dt / 3600.0
        elif (not is_charge) and current_a[i] < 0:
            q[i] = q[i - 1] - current_a[i] * dt / 3600.0
        else:
            q[i] = q[i - 1]
    return q


def final_discharge_capacity_ah(df: pd.DataFrame) -> float:
    I = df["Current (mA)"].to_numpy(dtype=float) / 1000.0
    t = df["Time (s)"].to_numpy(dtype=float)
    qd = calc_capacity_ah(I, t, is_charge=False)
    return float(qd[-1]) if len(qd) else float("nan")


def final_charge_capacity_ah(df: pd.DataFrame) -> float:
    I = df["Current (mA)"].to_numpy(dtype=float) / 1000.0
    t = df["Time (s)"].to_numpy(dtype=float)
    qc = calc_capacity_ah(I, t, is_charge=True)
    return float(qc[-1]) if len(qc) else float("nan")


def load_hust_cell(pkl_path: Path) -> pd.DataFrame:
    with pkl_path.open("rb") as f:
        obj = pickle.load(f)
    if not (isinstance(obj, dict) and len(obj) == 1):
        raise ValueError(f"Unexpected pkl structure in {pkl_path.name}")
    cell_id = next(iter(obj.keys()))
    payload = obj[cell_id]
    if not (isinstance(payload, dict) and "data" in payload):
        raise ValueError(f"Unexpected payload structure in {pkl_path.name}")
    cell_data = payload["data"]

    rows = []
    for cyc in sorted(cell_data.keys()):
        df = cell_data[cyc]
        rows.append({
            "cell_id": cell_id,
            "cycle": int(cyc),
            "Q_discharge": final_discharge_capacity_ah(df),
            "Q_charge": final_charge_capacity_ah(df),
            "n_points": len(df),
            "v_min": float(df["Voltage (V)"].min()) if "Voltage (V)" in df else float("nan"),
            "v_max": float(df["Voltage (V)"].max()) if "Voltage (V)" in df else float("nan"),
            "t_max_s": float(df["Time (s)"].max()) if "Time (s)" in df else float("nan"),
        })
    out = pd.DataFrame(rows).sort_values("cycle").reset_index(drop=True)

    # BatteryML special case: drop the first 2 cycles for cell 7-5, renumber from 1
    if cell_id == "7-5":
        out = out[out["cycle"] >= 3].copy()
        out["cycle"] = np.arange(1, len(out) + 1)

    rates = DISCHARGE_RATES.get(cell_id, [float("nan")] * 3)
    out["dchg_rate_1"], out["dchg_rate_2"], out["dchg_rate_3"] = rates
    return out


# ---------- audit / EOL recomputation ----------

def compute_q0_from_series(q) -> float:
    q = np.asarray(q, dtype=float).ravel()
    if len(q) < 5:
        return float("nan")
    vals = q[1:5]
    vals = vals[np.isfinite(vals) & (vals > 0)]
    return float(np.median(vals)) if len(vals) else float("nan")


def first_eol_cycle(q, q0: float, threshold_fraction: float) -> float:
    if not np.isfinite(q0) or q0 <= 0:
        return float("nan")
    q = np.asarray(q, dtype=float).ravel()
    thr = threshold_fraction * q0
    for i in range(1, len(q)):  # start at cycle 2 (index 1)
        if np.isfinite(q[i]) and q[i] > 0 and q[i] <= thr:
            return float(i + 1)
    return float("nan")


def audit_thresholds_hust(hust_cycles: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cell_id, g in hust_cycles.groupby("cell_id"):
        g = g.sort_values("cycle").copy()
        q = g["Q_discharge"].to_numpy(dtype=float)
        valid = np.isfinite(q) & (q > 0)
        q0 = compute_q0_from_series(q)
        min_q = float(np.min(q[valid])) if np.any(valid) else float("nan")
        min_ret = float(min_q / q0) if np.isfinite(min_q) and np.isfinite(q0) and q0 > 0 else float("nan")
        rows.append({
            "cell_id": cell_id,
            "observed_cycles": int(g["cycle"].max()),
            "q0": q0,
            "min_q_valid": min_q,
            "min_retention": min_ret,
            "min_retention_pct": 100 * min_ret if np.isfinite(min_ret) else float("nan"),
            "eol_90": first_eol_cycle(q, q0, 0.90),
            "eol_85": first_eol_cycle(q, q0, 0.85),
            "eol_80": first_eol_cycle(q, q0, 0.80),
            "reaches_90pct": bool(np.isfinite(first_eol_cycle(q, q0, 0.90))),
            "reaches_85pct": bool(np.isfinite(first_eol_cycle(q, q0, 0.85))),
            "reaches_80pct": bool(np.isfinite(first_eol_cycle(q, q0, 0.80))),
            "dchg_rate_1": g["dchg_rate_1"].iloc[0] if "dchg_rate_1" in g else float("nan"),
            "dchg_rate_2": g["dchg_rate_2"].iloc[0] if "dchg_rate_2" in g else float("nan"),
            "dchg_rate_3": g["dchg_rate_3"].iloc[0] if "dchg_rate_3" in g else float("nan"),
        })
    return pd.DataFrame(rows).sort_values("cell_id").reset_index(drop=True)


# ---------- main ----------

def main() -> int:
    if not HUST_DIR.exists():
        print(f"[error] HUST data directory not found: {HUST_DIR}")
        print("\nRun first:")
        print("    python 0_data/download_data.py --only hust")
        return 1

    pkl_files = sorted(HUST_DIR.glob("*.pkl"))
    if not pkl_files:
        print(f"[error] no .pkl files in {HUST_DIR}")
        return 1
    print(f"[load] found {len(pkl_files)} HUST pkl files")

    all_cells, failed = [], []
    for path in pkl_files:
        try:
            all_cells.append(load_hust_cell(path))
        except Exception as exc:
            failed.append((path.name, str(exc)))

    if not all_cells:
        print("[error] no cells loaded successfully")
        for name, err in failed[:10]:
            print(f"  {name}: {err}")
        return 1

    hust_cycles = pd.concat(all_cells, ignore_index=True)
    print(f"[ok]   loaded cells={hust_cycles['cell_id'].nunique()}, rows={len(hust_cycles)}, failed={len(failed)}")
    if failed:
        for name, err in failed[:5]:
            print(f"  [warn] {name}: {err}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cycles_path = OUT_DIR / "hust_cycles_tidy.csv"
    hust_cycles.to_csv(cycles_path, index=False)
    print(f"[save] {cycles_path}  ({len(hust_cycles):,} rows)")

    audit = audit_thresholds_hust(hust_cycles)
    audit_path = OUT_DIR / "hust_threshold_audit.csv"
    audit.to_csv(audit_path, index=False)
    print(f"[save] {audit_path}  ({len(audit)} cells)")

    summary = pd.DataFrame([{
        "n_cells": len(audit),
        "median_min_retention_pct": audit["min_retention_pct"].median(),
        "min_of_min_retention_pct": audit["min_retention_pct"].min(),
        "max_of_min_retention_pct": audit["min_retention_pct"].max(),
        "n_reach_90": int(audit["reaches_90pct"].sum()),
        "n_reach_85": int(audit["reaches_85pct"].sum()),
        "n_reach_80": int(audit["reaches_80pct"].sum()),
        "pct_reach_90": 100 * audit["reaches_90pct"].mean(),
        "pct_reach_85": 100 * audit["reaches_85pct"].mean(),
        "pct_reach_80": 100 * audit["reaches_80pct"].mean(),
    }])
    summary_path = OUT_DIR / "hust_threshold_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"[save] {summary_path}")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

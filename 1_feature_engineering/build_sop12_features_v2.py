"""
Build the SOP-compliant 12-feature table from Q_discharge series only.

This is the corrected feature set requested by the supervisor's email
(2026-04-20): twelve capacity-trajectory features computed entirely from
the per-cycle Q_discharge time series. No IR, no temperature, no dQdV.
The same 12 features are computed for MATR and HUST so cross-dataset
comparisons stay on a single, identical feature space.

Definitions (using cycles 2..N, cycle 1 dropped due to formation effect;
indices below are 1-indexed cycle numbers that map to 0-indexed array
positions [1..N-1]):
     1. Qdis_N           = Q_dis(N)
     2. delta_Qdis       = Q_dis(N) - Q_dis(2)
     3. retention_ratio  = Q_dis(N) / Q_dis(2)
     4. slope_linear     = OLS slope of Q_dis(2:N) vs cycle index
     5. variance_Qdis    = Var(Q_dis(2:N))
     6. range_Qdis       = max(Q_dis(2:N)) - min(Q_dis(2:N))
     7. max_drop         = max over c in 2..N-1 of (Q_dis(c) - Q_dis(c+1))
     8. std_diff         = Std(Δ Q_dis)
     9. skewness_Qdis    = skew(Q_dis(2:N))
    10. slope_ratio      = slope(N/2..N) / slope(2..N/2)
    11. Qdis_cycle10     = Q_dis(10)
    12. mean_diff        = Mean(Δ Q_dis)

Labels:
    Q0          = median(Q_dis at cycles 2..5)
    cycle_life  = first cycle c where Q_dis(c) <= 0.85 * Q0
                  (single-cycle definition; censored if never crossed)
    is_censored = 1 if cycle_life is NaN, else 0
Censored cells are kept in the table with cycle_life = NaN; experiments drop them.

Inputs:
    MATR: data/raw/{batch1,batch2,batch3}.pkl  (loaded + merged via the audit logic)
    HUST: data/intermediate/hust_cycles_tidy.csv  (produced by build_hust_audit.py)

Outputs:
    data/intermediate/features_sop12_matr.csv      (~2 rows per cell × 2 N values)
    data/intermediate/features_sop12_hust.csv
    data/intermediate/features_sop12_combined.csv  (both datasets, dataset='matr'|'hust')

Usage:
    python 1_feature_engineering/build_sop12_features_v2.py
    python 1_feature_engineering/build_sop12_features_v2.py --eol-fraction 0.80
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import skew

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
INTERMEDIATE_DIR = PROJECT_ROOT / "data" / "intermediate"

DEFAULT_N_WINDOWS = (50, 100)
DEFAULT_EOL_FRACTION = 0.85

SOP12_FEATURE_COLS = [
    "Qdis_N", "delta_Qdis", "retention_ratio", "slope_linear",
    "variance_Qdis", "range_Qdis", "max_drop", "std_diff",
    "skewness_Qdis", "slope_ratio", "Qdis_cycle10", "mean_diff",
]

# Per SOP §2.3: raw-capacity features that should be divided by Q0 when a
# dataset has a different nominal capacity (so the same model can score across
# datasets with mismatched cell chemistries / sizes).
# retention_ratio and slope_ratio are already ratios — no normalization needed.
CAPACITY_RAW_FEATURES = [
    "Qdis_N", "delta_Qdis", "Qdis_cycle10", "max_drop", "mean_diff",
]

# Same merge metadata as build_matr_audit.py (kept in sync intentionally)
BATCH1_CONTINUATION_FROM_BATCH2 = {
    "b1c0": {"source_cell": "b2c7", "add_len": 662},
    "b1c1": {"source_cell": "b2c8", "add_len": 981},
    "b1c2": {"source_cell": "b2c9", "add_len": 1060},
    "b1c3": {"source_cell": "b2c15", "add_len": 208},
    "b1c4": {"source_cell": "b2c16", "add_len": 482},
}


# ---------- shared label utilities ----------

def compute_q0(qd) -> float:
    qd = np.asarray(qd, dtype=float).ravel()
    if len(qd) < 5:
        return float("nan")
    vals = qd[1:5]
    vals = vals[np.isfinite(vals) & (vals > 0)]
    return float(np.median(vals)) if len(vals) else float("nan")


def compute_cycle_life(qd, q0: float, fraction: float) -> float:
    """Single-cycle EOL: first cycle (>=2, 1-indexed) where Q <= fraction * Q0."""
    if not np.isfinite(q0) or q0 <= 0:
        return float("nan")
    qd = np.asarray(qd, dtype=float).ravel()
    threshold = fraction * q0
    for i in range(1, len(qd)):  # start at index 1 (cycle 2)
        if np.isfinite(qd[i]) and qd[i] > 0 and qd[i] <= threshold:
            return float(i + 1)
    return float("nan")


# ---------- 12 SOP features ----------

def _ols_slope(values: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values), dtype=float)
    return float(np.polyfit(x, values, 1)[0])


def compute_sop12(qd: np.ndarray, n: int) -> dict | None:
    """Compute 12 features over cycles 2..N. Returns None if not enough cycles."""
    qd = np.asarray(qd, dtype=float).ravel()
    max_idx = min(n, len(qd))
    if max_idx < 10:
        return None  # need at least 10 cycles for cycle10 + half-window slopes
    window = qd[1:max_idx]   # cycles 2..N (0-indexed 1..max_idx-1)
    if len(window) < 2:
        return None

    diffs = np.diff(window)
    half = len(window) // 2
    s_first = _ols_slope(window[:half]) if len(window[:half]) >= 2 else 1e-10
    s_second = _ols_slope(window[half:]) if len(window[half:]) >= 2 else 0.0
    slope_ratio = (s_second / s_first) if abs(s_first) >= 1e-10 else 0.0

    q_n = float(qd[max_idx - 1])
    q_2 = float(qd[1])

    return {
        "Qdis_N": q_n,
        "delta_Qdis": q_n - q_2,
        "retention_ratio": (q_n / q_2) if q_2 > 0 else float("nan"),
        "slope_linear": _ols_slope(window),
        "variance_Qdis": float(np.var(window)),
        "range_Qdis": float(np.max(window) - np.min(window)),
        "max_drop": float(np.max(-diffs)) if len(diffs) > 0 else 0.0,
        "std_diff": float(np.std(diffs)) if len(diffs) > 0 else 0.0,
        "skewness_Qdis": float(skew(window)) if len(window) >= 3 else 0.0,
        "slope_ratio": slope_ratio,
        "Qdis_cycle10": float(qd[9]) if len(qd) >= 10 else float("nan"),
        "mean_diff": float(np.mean(diffs)) if len(diffs) > 0 else 0.0,
    }


# ---------- MATR loading ----------

def load_matr_qd_series() -> dict[str, np.ndarray]:
    """Load batch1+batch2+batch3 pkls, merge continuations, return {cell_id: QD_array}."""
    paths = {b: RAW_DIR / f"{b}.pkl" for b in ("batch1", "batch2", "batch3")}
    missing = [p for p in paths.values() if not p.exists()]
    if missing:
        raise SystemExit(
            f"missing MATR pkl files: {[str(p.relative_to(PROJECT_ROOT)) for p in missing]}\n"
            "Run: python 0_data_prep/download_data.py --only matr"
        )
    with paths["batch1"].open("rb") as f:
        b1 = pickle.load(f)
    with paths["batch2"].open("rb") as f:
        b2 = pickle.load(f)
    with paths["batch3"].open("rb") as f:
        b3 = pickle.load(f)
    print(f"[matr] batch1={len(b1)}, batch2={len(b2)}, batch3={len(b3)}")

    continuation_sources = {v["source_cell"] for v in BATCH1_CONTINUATION_FROM_BATCH2.values()}
    qd_by_cell: dict[str, np.ndarray] = {}

    for cell_id, cell in b1.items():
        qd = np.asarray(cell["summary"]["QD"], dtype=float).ravel()
        if cell_id in BATCH1_CONTINUATION_FROM_BATCH2:
            src = BATCH1_CONTINUATION_FROM_BATCH2[cell_id]["source_cell"]
            extra = np.asarray(b2[src]["summary"]["QD"], dtype=float).ravel()
            qd = np.hstack([qd, extra])
        qd_by_cell[f"matr_{cell_id}"] = qd

    for cell_id, cell in b2.items():
        if cell_id in continuation_sources:
            continue
        qd_by_cell[f"matr_{cell_id}"] = np.asarray(cell["summary"]["QD"], dtype=float).ravel()

    for cell_id, cell in b3.items():
        qd_by_cell[f"matr_{cell_id}"] = np.asarray(cell["summary"]["QD"], dtype=float).ravel()

    print(f"[matr] merged cells: {len(qd_by_cell)}")
    return qd_by_cell


# ---------- HUST loading ----------

def load_hust_qd_series() -> dict[str, np.ndarray]:
    """Read hust_cycles_tidy.csv and pivot to {hust_<id>: Q_discharge_array_per_cycle}."""
    tidy_path = INTERMEDIATE_DIR / "hust_cycles_tidy.csv"
    if not tidy_path.exists():
        raise SystemExit(
            f"missing {tidy_path}. Run: python 0_data_prep/build_hust_audit.py"
        )
    df = pd.read_csv(tidy_path)
    qd_by_cell: dict[str, np.ndarray] = {}
    for cell_id, g in df.groupby("cell_id"):
        g = g.sort_values("cycle")
        qd_by_cell[f"hust_{cell_id}"] = g["Q_discharge"].to_numpy(dtype=float)
    print(f"[hust] loaded cells: {len(qd_by_cell)}")
    return qd_by_cell


# ---------- table assembly ----------

def build_feature_rows(
    qd_by_cell: dict[str, np.ndarray],
    *,
    dataset: str,
    n_windows: tuple[int, ...],
    eol_fraction: float,
    capacity_normalize: bool = False,
) -> list[dict]:
    rows = []
    for cell_id in sorted(qd_by_cell):
        qd = qd_by_cell[cell_id]
        q0 = compute_q0(qd)
        if not np.isfinite(q0) or q0 <= 0:
            continue
        cycle_life = compute_cycle_life(qd, q0, eol_fraction)
        is_censored = int(np.isnan(cycle_life))
        for n in n_windows:
            if len(qd) < n:
                continue
            feats = compute_sop12(qd, n)
            if feats is None:
                continue
            if capacity_normalize and q0 > 0:
                # Per SOP §2.3: divide raw-capacity features by Q0 to put a
                # different-nominal-capacity dataset on the same scale.
                for col in CAPACITY_RAW_FEATURES:
                    feats[col] = feats[col] / q0
            row = {
                "dataset": dataset,
                "cell_id": cell_id,
                "n_cycles": n,
                "q0": q0,
                "cycle_life": cycle_life,
                "is_censored": is_censored,
                "capacity_normalized": int(capacity_normalize),
                **feats,
            }
            rows.append(row)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eol-fraction", type=float, default=DEFAULT_EOL_FRACTION,
                        help="EOL threshold fraction of Q0 (SOP=0.85). Default: 0.85")
    parser.add_argument("--n-windows", type=int, nargs="+", default=list(DEFAULT_N_WINDOWS),
                        help="Prediction windows N. Default: 50 100. Add 25 for ablation.")
    parser.add_argument("--capacity-normalize", action="store_true",
                        help="Divide raw-capacity features (Qdis_N, delta_Qdis, Qdis_cycle10, "
                             "max_drop, mean_diff) by Q0. Use when adding a third dataset with "
                             "different nominal capacity. Off by default (MATR + HUST share 1.1Ah).")
    parser.add_argument("--matr-only", action="store_true")
    parser.add_argument("--hust-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
    n_windows = tuple(args.n_windows)
    eol = args.eol_fraction

    matr_rows: list[dict] = []
    hust_rows: list[dict] = []

    cap_norm = args.capacity_normalize
    cap_msg = "Q0-normalized" if cap_norm else "raw"

    if not args.hust_only:
        print(f"\n[matr] building features (N={n_windows}, EOL={eol:.2f}*Q0, capacity={cap_msg})")
        matr_qd = load_matr_qd_series()
        matr_rows = build_feature_rows(
            matr_qd, dataset="matr", n_windows=n_windows,
            eol_fraction=eol, capacity_normalize=cap_norm,
        )
        out = INTERMEDIATE_DIR / "features_sop12_matr.csv"
        pd.DataFrame(matr_rows).to_csv(out, index=False)
        print(f"[matr] saved {len(matr_rows)} rows -> {out}")

    if not args.matr_only:
        print(f"\n[hust] building features (N={n_windows}, EOL={eol:.2f}*Q0, capacity={cap_msg})")
        hust_qd = load_hust_qd_series()
        hust_rows = build_feature_rows(
            hust_qd, dataset="hust", n_windows=n_windows,
            eol_fraction=eol, capacity_normalize=cap_norm,
        )
        out = INTERMEDIATE_DIR / "features_sop12_hust.csv"
        pd.DataFrame(hust_rows).to_csv(out, index=False)
        print(f"[hust] saved {len(hust_rows)} rows -> {out}")

    if matr_rows or hust_rows:
        combined = pd.DataFrame(matr_rows + hust_rows)
        out = INTERMEDIATE_DIR / "features_sop12_combined.csv"
        combined.to_csv(out, index=False)
        print(f"\n[combined] saved {len(combined)} rows -> {out}")

        for ds, df in combined.groupby("dataset"):
            for n in sorted(df["n_cycles"].unique()):
                sub = df[df["n_cycles"] == n]
                cens = int(sub["is_censored"].sum())
                tot = len(sub)
                print(f"  {ds} N={n}: {tot} cells, {tot - cens} reach EOL ({100*(tot-cens)/tot:.1f}%), {cens} censored")
    return 0


if __name__ == "__main__":
    sys.exit(main())

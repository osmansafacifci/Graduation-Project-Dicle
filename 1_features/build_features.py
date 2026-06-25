"""
Build the SOP-compliant 34-feature table from Q_discharge series only.

The first 12 columns are the corrected capacity-trajectory features requested
by the supervisor's email (2026-04-20). The active thesis feature table adds
22 more capacity-only shape/decay, frequency, entropy, and second-derivative
features. No IR, no temperature, no dQdV. The same feature definitions are
computed for MATR and HUST so cross-dataset comparisons stay on a single,
identical feature space.

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
    data/intermediate/features_sop12_combined.csv  (34 features; dataset='matr'|'hust')

Usage:
    python 1_features/build_features.py
    python 1_features/build_features.py --eol-fraction 0.80
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from shared.battery_utils import compute_cycle_life, compute_q0  # noqa: E402
from shared.constants import (  # noqa: E402
    BATCH1_CONTINUATION_FROM_BATCH2,
    CAPACITY_RAW_FEATURES,
    CAPACITY_VARIANCE_FEATURES,
    EXTENDED2_FEATURE_COLS,
    EXTENDED_FEATURE_COLS,
    SOP12_FEATURE_COLS,
)
RAW_DIR = PROJECT_ROOT / "data" / "raw"
INTERMEDIATE_DIR = PROJECT_ROOT / "data" / "intermediate"

DEFAULT_N_WINDOWS = (50, 100)
DEFAULT_EOL_FRACTION = 0.85

ALL_FEATURE_COLS = SOP12_FEATURE_COLS + EXTENDED_FEATURE_COLS + EXTENDED2_FEATURE_COLS  # 34 total


# ---------- 12 SOP features ----------

def _ols_slope(values: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values), dtype=float)
    return float(np.polyfit(x, values, 1)[0])


def compute_extended(qd: np.ndarray, n: int, q0: float) -> dict:
    """Twelve additional QD-only features capturing curve shape and decay dynamics.

    All derived from cycles 2..N of the Q_discharge series; safe defaults when
    fits fail (no exception escapes).
    """
    max_idx = min(n, len(qd))
    window = qd[1:max_idx]
    cycles = np.arange(2, max_idx + 1, dtype=float)
    diffs = np.diff(window) if len(window) > 1 else np.array([0.0])

    # Quadratic polynomial fit: Q(c) = a + b*c + c2*c²
    try:
        c2, b, a = np.polyfit(cycles, window, 2)
    except Exception:
        a, b, c2 = float("nan"), 0.0, 0.0

    # Exponential fade: Q(c) = q0_fit * exp(-k*c)
    try:
        from scipy.optimize import curve_fit  # imported lazily

        def _exp_model(x, q_init, k):
            return q_init * np.exp(-k * x)

        popt, _ = curve_fit(
            _exp_model, cycles, window,
            p0=[max(window[0], 1e-6), 1e-3],
            maxfev=2000,
        )
        exp_decay_k = float(popt[1])
        if not np.isfinite(exp_decay_k):
            exp_decay_k = 0.0
    except Exception:
        exp_decay_k = 0.0

    # Cycle-to-retention-threshold landmarks (1-indexed cycle number; saturates at N if never crossed)
    def _cycle_to(thr_frac: float) -> float:
        thr = thr_frac * q0
        for i, q in enumerate(window):
            if np.isfinite(q) and q < thr:
                return float(i + 2)  # i is 0-indexed within window which starts at cycle 2
        return float(max_idx)

    cycle_to_99 = _cycle_to(0.99)
    cycle_to_98 = _cycle_to(0.98)
    cycle_to_95 = _cycle_to(0.95)

    # Multi-scale slopes
    quarter = max(2, len(window) // 4)
    slope_first = _ols_slope_local(window[:quarter])
    slope_last = _ols_slope_local(window[-quarter:])

    # Lag-1 autocorrelation of cycle-to-cycle ΔQ
    if len(diffs) > 2:
        d = diffs - np.mean(diffs)
        denom = float(np.dot(d, d))
        autocorr = float(np.dot(d[:-1], d[1:]) / denom) if denom > 1e-12 else 0.0
        if not np.isfinite(autocorr):
            autocorr = 0.0
    else:
        autocorr = 0.0

    # Knee detection: cycle with maximum perpendicular distance from the
    # straight line between the first and last point of the window.
    if len(window) >= 5:
        x_norm = (cycles - cycles[0]) / max(cycles[-1] - cycles[0], 1e-9)
        y_norm = (window - np.min(window)) / max(np.max(window) - np.min(window), 1e-9)
        line_y = y_norm[0] + (y_norm[-1] - y_norm[0]) * x_norm
        knee_idx = int(np.argmax(np.abs(y_norm - line_y)))
        knee_cycle = float(cycles[knee_idx])
    else:
        knee_cycle = float(cycles[-1])

    # Cycle-to-cycle drops larger than 1% of Q0
    threshold_jump = 0.01 * q0
    n_jumps = int(np.sum(diffs < -threshold_jump)) if len(diffs) > 0 else 0

    return {
        "poly2_a": float(a),
        "poly2_b": float(b),
        "poly2_c": float(c2),
        "exp_decay_k": exp_decay_k,
        "cycle_to_99pct": cycle_to_99,
        "cycle_to_98pct": cycle_to_98,
        "cycle_to_95pct": cycle_to_95,
        "slope_first_quarter": slope_first,
        "slope_last_quarter": slope_last,
        "autocorr_lag1": autocorr,
        "knee_cycle": knee_cycle,
        "n_capacity_jumps": float(n_jumps),
    }


def _ols_slope_local(values: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values), dtype=float)
    return float(np.polyfit(x, values, 1)[0])


def _sample_entropy(series: np.ndarray, m: int = 2, r_factor: float = 0.2) -> float:
    """SampEn(m, r=r_factor*std, N). Pure-numpy implementation suitable for
    short windows (N=50..100). Returns 0.0 on degenerate input."""
    series = np.asarray(series, dtype=float).ravel()
    n = len(series)
    if n < m + 2:
        return 0.0
    sd = float(np.std(series))
    if sd <= 0:
        return 0.0
    r = r_factor * sd

    def _phi(mm: int) -> int:
        templates = np.array([series[i:i + mm] for i in range(n - mm + 1)])
        # Chebyshev distance (sup-norm) — count pairs within tolerance
        count = 0
        for i in range(len(templates) - 1):
            d = np.max(np.abs(templates[i + 1:] - templates[i]), axis=1)
            count += int(np.sum(d <= r))
        return count

    a = _phi(m + 1)
    b = _phi(m)
    if a == 0 or b == 0:
        return 0.0
    return float(-np.log(a / b))


def compute_extended2(qd: np.ndarray, n: int, q0: float) -> dict:
    """Ten additional QD-only features beyond compute_extended():
    second-derivative statistics, frequency-domain descriptors, complexity,
    and outlier-robust spread. Same robustness contract — exceptions are
    swallowed and a sensible default is returned."""
    max_idx = min(n, len(qd))
    window = qd[1:max_idx]

    if len(window) < 3:
        return {col: 0.0 for col in EXTENDED2_FEATURE_COLS}

    diffs1 = np.diff(window)                                  # 1st derivative
    diffs2 = np.diff(diffs1) if len(diffs1) > 1 else np.array([0.0])  # 2nd derivative

    accel_mean = float(np.mean(diffs2)) if len(diffs2) else 0.0
    accel_std = float(np.std(diffs2)) if len(diffs2) else 0.0
    accel_max_abs = float(np.max(np.abs(diffs2))) if len(diffs2) else 0.0

    # R² of straight-line fit to Q_dis(2..N)
    cycles = np.arange(len(window), dtype=float)
    try:
        slope, intercept = np.polyfit(cycles, window, 1)
        pred = slope * cycles + intercept
        ss_res = float(np.sum((window - pred) ** 2))
        ss_tot = float(np.sum((window - np.mean(window)) ** 2))
        linearity_r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0
    except Exception:
        linearity_r2 = 0.0

    # Higher-order shape
    try:
        kurt = float(kurtosis(window, fisher=True, bias=False))
        if not np.isfinite(kurt):
            kurt = 0.0
    except Exception:
        kurt = 0.0

    # Frequency-domain — FFT magnitudes, drop the DC bin (mean component
    # is captured by Qdis_cycle10 / mean_diff)
    try:
        spectrum = np.abs(np.fft.rfft(window - np.mean(window)))
        if len(spectrum) > 1:
            mags = spectrum[1:]                                   # drop DC
            total_energy = float(np.sum(mags ** 2)) + 1e-12
            top3 = np.sort(mags)[::-1][:3]
            fft_top3_energy_ratio = float(np.sum(top3 ** 2) / total_energy)
            # Spectral entropy — Shannon entropy of normalized power
            p = (mags ** 2) / total_energy
            p = p[p > 0]
            spectral_entropy = float(-np.sum(p * np.log(p))) if len(p) else 0.0
        else:
            fft_top3_energy_ratio = 0.0
            spectral_entropy = 0.0
    except Exception:
        fft_top3_energy_ratio = 0.0
        spectral_entropy = 0.0

    # Complexity / regularity of the QD trajectory
    try:
        samp_ent = _sample_entropy(window, m=2, r_factor=0.2)
    except Exception:
        samp_ent = 0.0

    # Direction balance of cycle-to-cycle deltas
    if len(diffs1) > 0:
        n_neg = int(np.sum(diffs1 < 0))
        n_pos = int(np.sum(diffs1 > 0))
        pos_neg_ratio = float(n_pos / max(n_neg, 1))
    else:
        pos_neg_ratio = 0.0

    # Outlier-robust spread
    try:
        med = float(np.median(window))
        mad = float(np.median(np.abs(window - med)))
    except Exception:
        mad = 0.0

    return {
        "accel_mean": accel_mean,
        "accel_std": accel_std,
        "accel_max_abs": accel_max_abs,
        "linearity_r2": linearity_r2,
        "kurtosis_Qdis": kurt,
        "fft_top3_energy_ratio": fft_top3_energy_ratio,
        "spectral_entropy": spectral_entropy,
        "sample_entropy": samp_ent,
        "pos_neg_diff_ratio": pos_neg_ratio,
        "mad_Qdis": mad,
    }


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

def load_matr_qd_series_from_tidy() -> dict[str, np.ndarray]:
    """Read matr_cycles_tidy.csv → {cell_id: QD_array}. Fast path, no pkls needed."""
    tidy_path = INTERMEDIATE_DIR / "matr_cycles_tidy.csv"
    if not tidy_path.exists():
        return {}
    df = pd.read_csv(tidy_path)
    qd_by_cell: dict[str, np.ndarray] = {}
    for cell_id, g in df.groupby("cell_id"):
        g = g.sort_values("cycle")
        qd_by_cell[str(cell_id)] = g["Q_discharge"].to_numpy(dtype=float)
    print(f"[matr] loaded {len(qd_by_cell)} cells from {tidy_path.name}")
    return qd_by_cell


def load_matr_qd_series_from_pkls() -> dict[str, np.ndarray]:
    """Fallback: load batch1+batch2+batch3 pkls, merge continuations, return {cell_id: QD_array}."""
    paths = {b: RAW_DIR / f"{b}.pkl" for b in ("batch1", "batch2", "batch3")}
    missing = [p for p in paths.values() if not p.exists()]
    if missing:
        raise SystemExit(
            f"missing MATR sources. Either run audit_matr to produce matr_cycles_tidy.csv, "
            f"or place these pkls under data/raw/: "
            f"{[str(p.relative_to(PROJECT_ROOT)) for p in missing]}"
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


def load_matr_qd_series() -> dict[str, np.ndarray]:
    """Try the tidy CSV first (fast, no Drive needed); fall back to pkls."""
    qd = load_matr_qd_series_from_tidy()
    if qd:
        return qd
    print("[matr] matr_cycles_tidy.csv missing, falling back to raw pkls")
    return load_matr_qd_series_from_pkls()


# ---------- HUST loading ----------

def load_hust_qd_series() -> dict[str, np.ndarray]:
    """Read hust_cycles_tidy.csv and pivot to {hust_<id>: Q_discharge_array_per_cycle}."""
    tidy_path = INTERMEDIATE_DIR / "hust_cycles_tidy.csv"
    if not tidy_path.exists():
        raise SystemExit(
            f"missing {tidy_path}. Run: python 0_data/build_hust_audit.py"
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
            feats.update(compute_extended(qd, n, q0))
            feats.update(compute_extended2(qd, n, q0))
            if capacity_normalize and q0 > 0:
                # Per SOP §2.3: divide raw-capacity features by Q0 to put a
                # different-nominal-capacity dataset on the same scale.
                for col in CAPACITY_RAW_FEATURES:
                    feats[col] = feats[col] / q0
                # Variance has capacity² units, so use Q0² for dimensional
                # consistency with the Q/Q0 scale.
                for col in CAPACITY_VARIANCE_FEATURES:
                    feats[col] = feats[col] / (q0 ** 2)
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
                        help="Divide raw-capacity features by Q0 and variance features by Q0^2. "
                             "Use when adding a third dataset with "
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

"""
MATR cell-level audit: provided cycle_life vs historical 80% × Q0 EOL.

Faithful port of the Cell_Audit_MATR_Provided_vs_Recomputed_EOL notebook
(https://drive.google.com/drive/folders/19wCEj4hr54QtARns1HVOX0alsFlkjt2P).
This script does not depend on the student's existing build_raw_label_table.py
or build_sop12_features.py — it reads the three Severson batches directly from
data/raw/ and produces the same CSVs the notebook saved to Drive.

Inputs (from MATR Drive folder; place under data/raw/):
    data/raw/batch1.pkl   (46 cells, 2017-05-12)
    data/raw/batch2.pkl   (48 cells, 2017-06-30)
    data/raw/batch3.pkl   (46 cells, 2018-04-12)   <- standard batch3, NOT varcharge

Outputs:
    data/intermediate/matr_cell_audit_strict.csv
        135-cell strict audit (only duplicate-continuation-source dedup)
    data/intermediate/matr_cell_audit_replication.csv
        112-cell replication-style audit (with student's noisy-cell exclusions)
    data/intermediate/matr_retention_summary.csv
        per-batch retention summary at 90/85/80% thresholds (raw batch1+batch3)

Q0 = median(QD[cycles 2..5]).
This audit intentionally reproduces the supervisor notebook's historical
80% × Q0, 3-consecutive-cycle EOL diagnostic. Modeling labels are recomputed
in `1_features/build_features.py` with the thesis protocol:
single-cycle EOL at 0.85 × Q0.

Usage:
    python 0_data/build_matr_audit.py
"""

from __future__ import annotations

import copy
import gc
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUT_DIR = PROJECT_ROOT / "data" / "intermediate"

BATCH1_PATH = RAW_DIR / "batch1.pkl"
BATCH2_PATH = RAW_DIR / "batch2.pkl"
BATCH3_PATH = RAW_DIR / "batch3.pkl"

# Batch1 cells whose cycling continues into batch2 (per the Severson 2019 paper)
BATCH1_CONTINUATION_FROM_BATCH2 = {
    "b1c0": {"source_cell": "b2c7", "add_len": 662},
    "b1c1": {"source_cell": "b2c8", "add_len": 981},
    "b1c2": {"source_cell": "b2c9", "add_len": 1060},
    "b1c3": {"source_cell": "b2c15", "add_len": 208},
    "b1c4": {"source_cell": "b2c16", "add_len": 482},
}

# Replication-style (notebook-era) exclusions
REPL_EXCLUDE_BATCH1_NOT_FINISHED = {"b1c8", "b1c10", "b1c12", "b1c13", "b1c22"}
REPL_EXCLUDE_BATCH2_NOISY = {"b2c29", "b2c40", "b2c42", "b2c44", "b2c45", "b2c46", "b2c47"}
REPL_EXCLUDE_BATCH1_NOISY = {"b1c18", "b1c38", "b1c40", "b1c43", "b1c44"}
REPL_EXCLUDE_BATCH3_NOISY = {"b3c37", "b3c2", "b3c23", "b3c32", "b3c42", "b3c43"}


# ---------- helpers ----------

def load_pickle(path: Path) -> dict:
    with path.open("rb") as f:
        return pickle.load(f)


def compute_q0(qd) -> float:
    """Q0 = median of valid Q_discharge over cycles 2-5 (0-indexed: indices 1..4)."""
    qd = np.asarray(qd, dtype=float).ravel()
    if len(qd) < 5:
        return float("nan")
    vals = qd[1:5]
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if len(vals) == 0:
        return float("nan")
    return float(np.median(vals))


def compute_eol(qd, q0: float, threshold_fraction: float = 0.80, k_consecutive: int = 3) -> float:
    """EOL = first cycle where QD stays <= threshold for k_consecutive cycles. 1-indexed."""
    if not np.isfinite(q0) or q0 <= 0:
        return float("nan")
    qd = np.asarray(qd, dtype=float).ravel()
    threshold = threshold_fraction * q0
    for i in range(len(qd) - k_consecutive + 1):
        window = qd[i:i + k_consecutive]
        if np.all(np.isfinite(window)) and np.all(window <= threshold):
            return float(i + 1)
    return float("nan")


def safe_scalar(x) -> float:
    try:
        arr = np.asarray(x).ravel()
        if arr.size == 0:
            return float("nan")
        return float(arr[0])
    except Exception:
        return float("nan")


# ---------- merging ----------

def merge_batches(batch1: dict, batch2: dict, batch3: dict, *, strict_keep_all: bool = True):
    """Merge MATR batches with low memory overhead.

    strict_keep_all=True   : keep all cells, drop only the 5 batch2 continuation sources.
    strict_keep_all=False  : also apply replication-era noisy/non-finished exclusions.
    """
    continuation_sources = {v["source_cell"] for v in BATCH1_CONTINUATION_FROM_BATCH2.values()}
    merged: dict[str, dict] = {}
    audit_rows: list[dict] = []

    # batch1
    for cell_id, cell in batch1.items():
        keep, reason = True, ""
        if not strict_keep_all:
            if cell_id in REPL_EXCLUDE_BATCH1_NOT_FINISHED:
                keep, reason = False, "replication_exclude_not_finished"
            elif cell_id in REPL_EXCLUDE_BATCH1_NOISY:
                keep, reason = False, "replication_exclude_noisy"

        merged_id = f"matr_{cell_id}"
        if keep:
            if cell_id in BATCH1_CONTINUATION_FROM_BATCH2:
                spec = BATCH1_CONTINUATION_FROM_BATCH2[cell_id]
                source = batch2[spec["source_cell"]]
                target = copy.deepcopy(cell)
                if "cycle_life" in target:
                    target["cycle_life"] = np.asarray(target["cycle_life"]) + spec["add_len"]
                for key in target["summary"]:
                    if key == "cycle":
                        offset = len(target["summary"][key])
                        target["summary"][key] = np.hstack(
                            (target["summary"][key], source["summary"][key] + offset)
                        )
                    else:
                        target["summary"][key] = np.hstack(
                            (target["summary"][key], source["summary"][key])
                        )
                merged[merged_id] = target
            else:
                merged[merged_id] = cell
        audit_rows.append(dict(merged_id=merged_id, orig_batch="batch1", orig_cell_id=cell_id, kept=keep, reason=reason))

    # batch2
    for cell_id, cell in batch2.items():
        keep, reason = True, ""
        if cell_id in continuation_sources:
            keep, reason = False, "duplicate_continuation_source"
        elif not strict_keep_all and cell_id in REPL_EXCLUDE_BATCH2_NOISY:
            keep, reason = False, "replication_exclude_noisy"
        merged_id = f"matr_{cell_id}"
        if keep:
            merged[merged_id] = cell
        audit_rows.append(dict(merged_id=merged_id, orig_batch="batch2", orig_cell_id=cell_id, kept=keep, reason=reason))

    # batch3
    for cell_id, cell in batch3.items():
        keep, reason = True, ""
        if not strict_keep_all and cell_id in REPL_EXCLUDE_BATCH3_NOISY:
            keep, reason = False, "replication_exclude_noisy"
        merged_id = f"matr_{cell_id}"
        if keep:
            merged[merged_id] = cell
        audit_rows.append(dict(merged_id=merged_id, orig_batch="batch3", orig_cell_id=cell_id, kept=keep, reason=reason))

    return merged, pd.DataFrame(audit_rows)


# ---------- audit table ----------

def build_label_audit(cells_dict: dict, audit_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for merged_id, cell in sorted(cells_dict.items()):
        qd = np.asarray(cell["summary"]["QD"], dtype=float).ravel()
        q0 = compute_q0(qd)
        thr = 0.8 * q0 if np.isfinite(q0) else float("nan")
        eol = compute_eol(qd, q0)
        provided = safe_scalar(cell.get("cycle_life", float("nan")))
        rows.append({
            "cell_id": merged_id,
            "provided_cycle_life": provided,
            "observed_cycles": len(qd),
            "q0": q0,
            "threshold_80pct_q0": thr,
            "min_qd": float(np.nanmin(qd)) if len(qd) else float("nan"),
            "eol_recomputed": eol,
            "is_censored_recomputed": int(np.isnan(eol)),
        })
    df = pd.DataFrame(rows)
    df = df.merge(
        audit_df[["merged_id", "orig_batch", "orig_cell_id", "kept", "reason"]],
        left_on="cell_id", right_on="merged_id", how="left",
    ).drop(columns=["merged_id"])
    df["provided_minus_recomputed"] = df["provided_cycle_life"] - df["eol_recomputed"]
    return df.sort_values(["orig_batch", "orig_cell_id"]).reset_index(drop=True)


# ---------- per-batch retention summary ----------

def summarize_batch_depth(batch_dict: dict, batch_name: str) -> pd.DataFrame:
    rows = []
    for cell_id, cell in sorted(batch_dict.items()):
        qd = np.asarray(cell["summary"]["QD"], dtype=float).ravel()
        q0 = compute_q0(qd)
        if not np.isfinite(q0) or q0 <= 0:
            rows.append(dict(batch=batch_name, cell_id=cell_id, q0=float("nan"),
                             observed_cycles=len(qd), min_qd_valid=float("nan"),
                             min_retention=float("nan"), min_retention_pct=float("nan"),
                             reaches_90pct=False, reaches_85pct=False, reaches_80pct=False))
            continue
        qd_valid = qd[np.isfinite(qd) & (qd > 0)]
        if len(qd_valid) == 0:
            min_qd, min_ret = float("nan"), float("nan")
        else:
            min_qd = float(np.min(qd_valid))
            min_ret = float(min_qd / q0)
        rows.append({
            "batch": batch_name,
            "cell_id": cell_id,
            "q0": q0,
            "observed_cycles": len(qd),
            "min_qd_valid": min_qd,
            "min_retention": min_ret,
            "min_retention_pct": 100 * min_ret if np.isfinite(min_ret) else float("nan"),
            "reaches_90pct": bool(np.isfinite(min_ret) and min_ret <= 0.90),
            "reaches_85pct": bool(np.isfinite(min_ret) and min_ret <= 0.85),
            "reaches_80pct": bool(np.isfinite(min_ret) and min_ret <= 0.80),
        })
    return pd.DataFrame(rows)


# ---------- main ----------

def main() -> int:
    missing = [p for p in (BATCH1_PATH, BATCH2_PATH, BATCH3_PATH) if not p.exists()]
    if missing:
        print("[error] missing MATR pkl files:")
        for m in missing:
            print(f"        {m}")
        print("\nRun first:")
        print("    python 0_data/download_data.py --only matr")
        return 1

    print(f"[load] {BATCH1_PATH}")
    batch1 = load_pickle(BATCH1_PATH)
    print(f"[load] {BATCH2_PATH}")
    batch2 = load_pickle(BATCH2_PATH)
    print(f"[load] {BATCH3_PATH}")
    batch3 = load_pickle(BATCH3_PATH)
    print(f"[ok]   batch1={len(batch1)}, batch2={len(batch2)}, batch3={len(batch3)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # strict
    print("\n[merge] strict (keep all cells except duplicate continuation sources)")
    strict_cells, strict_audit = merge_batches(batch1, batch2, batch3, strict_keep_all=True)
    print(f"        merged_cells={len(strict_cells)}")
    audit_strict = build_label_audit(strict_cells, strict_audit)
    out_strict = OUT_DIR / "matr_cell_audit_strict.csv"
    audit_strict.to_csv(out_strict, index=False)
    print(f"[save]  {out_strict}  ({len(audit_strict)} rows)")
    del strict_cells
    gc.collect()

    # replication
    print("\n[merge] replication-style (with notebook-era exclusions)")
    repl_cells, repl_audit = merge_batches(batch1, batch2, batch3, strict_keep_all=False)
    print(f"        merged_cells={len(repl_cells)}")
    audit_repl = build_label_audit(repl_cells, repl_audit)
    out_repl = OUT_DIR / "matr_cell_audit_replication.csv"
    audit_repl.to_csv(out_repl, index=False)
    print(f"[save]  {out_repl}  ({len(audit_repl)} rows)")
    del repl_cells
    gc.collect()

    # tidy per-cycle QD CSV (mirrors hust_cycles_tidy.csv schema). Once committed,
    # any future feature engineering can run locally without re-reading the pkls.
    print("\n[tidy] writing matr_cycles_tidy.csv (per-cycle QD for all merged cells)")
    strict_cells, _ = merge_batches(batch1, batch2, batch3, strict_keep_all=True)
    tidy_rows = []
    for cell_id, cell in sorted(strict_cells.items()):
        qd = np.asarray(cell["summary"]["QD"], dtype=float).ravel()
        for cycle_idx, q in enumerate(qd, start=1):
            tidy_rows.append({"cell_id": cell_id, "cycle": cycle_idx, "Q_discharge": float(q)})
    tidy_df = pd.DataFrame(tidy_rows)
    tidy_path = OUT_DIR / "matr_cycles_tidy.csv"
    tidy_df.to_csv(tidy_path, index=False)
    print(f"[save]  {tidy_path}  ({len(tidy_df):,} rows)")
    del strict_cells
    gc.collect()

    # retention summary on raw batches (no merge / no exclusion)
    print("\n[summary] per-batch retention (raw batch1 + batch3)")
    df = pd.concat(
        [summarize_batch_depth(batch1, "batch1"),
         summarize_batch_depth(batch2, "batch2"),
         summarize_batch_depth(batch3, "batch3")],
        ignore_index=True,
    )
    summary = (
        df.groupby("batch")
          .agg(
              n_cells=("cell_id", "count"),
              median_min_retention_pct=("min_retention_pct", "median"),
              min_of_min_retention_pct=("min_retention_pct", "min"),
              max_of_min_retention_pct=("min_retention_pct", "max"),
              n_reach_90=("reaches_90pct", "sum"),
              n_reach_85=("reaches_85pct", "sum"),
              n_reach_80=("reaches_80pct", "sum"),
          )
          .reset_index()
    )
    for thr in (90, 85, 80):
        summary[f"pct_reach_{thr}"] = 100 * summary[f"n_reach_{thr}"] / summary["n_cells"]

    out_summary = OUT_DIR / "matr_retention_summary.csv"
    summary.to_csv(out_summary, index=False)
    print(f"[save]  {out_summary}")
    print(summary.to_string(index=False))

    # quick per-batch censoring breakdown for the strict merged audit
    print("\n[overview] strict audit censoring by batch:")
    print(audit_strict.groupby(["orig_batch", "is_censored_recomputed"]).size().unstack(fill_value=0).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())

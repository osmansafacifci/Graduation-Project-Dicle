"""
VIF (Variance Inflation Factor) screening per SOP §2.4.

Default mode is REPORT-ONLY: compute VIF on the MATR train slice, flag
any feature with VIF > 5, but do NOT drop it. The supervisor reviews the
report and can re-run with --drop if iterative pruning is desired.

Algorithm:
  1. Take MATR training cells from a chosen split (default seed=42), N=100 rows.
     Use uncensored cells only (matches the modeling protocol).
  2. Standardize the 12 SOP features (z-score on this MATR train slice).
  3. Compute VIF for each feature: VIF_i = 1 / (1 - R²_i), where R²_i comes
     from OLS regressing feature i on the remaining features.
  4. Default: print all VIFs, mark those above the threshold as flagged,
     write a report file. NO features are removed.
  5. With --drop: iteratively drop the feature with the highest VIF and
     recompute until all VIF ≤ threshold; persist the kept feature list
     for run_experiments_v2.py.

Inputs:
  data/intermediate/features_sop12_combined.csv
  splits/sop_v2/matr_<seed>.json   (default seed=42; configurable via --seed)

Outputs (always):
  data/intermediate/vif_screening.json  (full VIF report + iteration history)
  data/intermediate/vif_report.txt      (human-readable summary)

Outputs (only with --drop):
  data/intermediate/vif_kept_features.txt   (surviving features, one per line)

Usage:
  python 2_modeling_featuring/vif_screening.py             # report only (default)
  python 2_modeling_featuring/vif_screening.py --drop      # iterative drop
  python 2_modeling_featuring/vif_screening.py --seed 42 --threshold 5.0 --n-cycles 100
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERMEDIATE_DIR = PROJECT_ROOT / "data" / "intermediate"
SPLITS_DIR = PROJECT_ROOT / "splits" / "sop_v2"

SOP12_FEATURES = [
    "Qdis_N", "delta_Qdis", "retention_ratio", "slope_linear",
    "variance_Qdis", "range_Qdis", "max_drop", "std_diff",
    "skewness_Qdis", "slope_ratio", "Qdis_cycle10", "mean_diff",
]

# Reserved (non-feature) CSV columns; everything else gets a VIF.
META_COLS = {
    "dataset", "cell_id", "n_cycles", "q0", "cycle_life",
    "is_censored", "capacity_normalized",
}


def compute_vif(X: np.ndarray, feature_names: list[str]) -> dict[str, float]:
    """VIF_i = 1 / (1 - R²_i), where i is regressed on all other features."""
    vifs: dict[str, float] = {}
    n_features = X.shape[1]
    if n_features < 2:
        return {feature_names[0]: 1.0} if n_features == 1 else {}
    for i, name in enumerate(feature_names):
        y = X[:, i]
        X_others = np.delete(X, i, axis=1)
        if np.var(y) < 1e-12:
            vifs[name] = float("inf")
            continue
        model = LinearRegression().fit(X_others, y)
        r2 = float(model.score(X_others, y))
        # numerical guard: clip R^2 just below 1 to avoid divide-by-zero
        if r2 >= 0.999999:
            vifs[name] = float("inf")
        else:
            vifs[name] = float(1.0 / (1.0 - r2))
    return vifs


def iterative_vif_drop(
    X: np.ndarray,
    feature_names: list[str],
    threshold: float,
) -> tuple[list[str], list[str], list[dict]]:
    """Drop features one at a time (highest VIF first) until max VIF <= threshold."""
    kept = list(feature_names)
    removed: list[str] = []
    history: list[dict] = []

    while True:
        col_idx = [feature_names.index(name) for name in kept]
        X_curr = X[:, col_idx]
        vifs = compute_vif(X_curr, kept)
        history.append({
            "iteration": len(history),
            "n_features": len(kept),
            "vif": {k: (None if not np.isfinite(v) else v) for k, v in vifs.items()},
        })
        if not vifs:
            break
        worst_feature = max(vifs, key=lambda k: vifs[k])
        worst_vif = vifs[worst_feature]
        if not np.isfinite(worst_vif):
            # degenerate column — drop and continue
            kept.remove(worst_feature)
            removed.append(worst_feature)
            history[-1]["dropped"] = worst_feature
            history[-1]["dropped_vif"] = None
            continue
        if worst_vif <= threshold:
            history[-1]["dropped"] = None
            break
        kept.remove(worst_feature)
        removed.append(worst_feature)
        history[-1]["dropped"] = worst_feature
        history[-1]["dropped_vif"] = worst_vif
        if not kept:
            break

    return kept, removed, history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=42,
                        help="Which MATR split seed to use for the train slice. Default: 42.")
    parser.add_argument("--threshold", type=float, default=5.0, help="VIF cutoff. Default: 5.0.")
    parser.add_argument("--n-cycles", type=int, default=100,
                        help="Window N to use for VIF (must be in features_sop12_combined.csv). Default: 100.")
    parser.add_argument("--drop", action="store_true",
                        help="Iteratively drop features with VIF > threshold. "
                             "Default is report-only (no features removed).")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    features_path = INTERMEDIATE_DIR / "features_sop12_combined.csv"
    if not features_path.exists():
        print(f"[error] {features_path} missing — run build_sop12_features_v2.py first.")
        return 1

    split_path = SPLITS_DIR / f"matr_{args.seed}.json"
    if not split_path.exists():
        print(f"[error] {split_path} missing — run generate_sop_splits_v2.py first.")
        return 1
    with split_path.open() as f:
        split = json.load(f)

    df = pd.read_csv(features_path)
    feature_cols = [c for c in df.columns if c not in META_COLS]
    matr = df[df["dataset"] == "matr"]
    train_subset = matr[
        (matr["n_cycles"] == args.n_cycles)
        & (matr["is_censored"] == 0)
        & (matr["cell_id"].isin(split["train"]))
    ]
    if len(train_subset) < 5:
        print(f"[error] only {len(train_subset)} train rows at N={args.n_cycles}; need >= 5.")
        return 1

    X = train_subset[feature_cols].to_numpy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    mode = "DROP" if args.drop else "REPORT-ONLY"
    print(f"[vif] mode={mode}, training slice: dataset=matr, seed={args.seed}, "
          f"N={args.n_cycles}, n_train={len(train_subset)} cells, threshold={args.threshold}")
    initial_vif = compute_vif(X_scaled, feature_cols)

    print("\n[vif] initial VIF values (sorted desc):")
    flagged: list[str] = []
    report_lines: list[str] = []
    report_lines.append(f"VIF screening on MATR train (seed={args.seed}, N={args.n_cycles})")
    report_lines.append(f"n_train_cells={len(train_subset)}, threshold={args.threshold}")
    report_lines.append("")
    report_lines.append(f"{'feature':<18}  {'VIF':>10}  flag")
    report_lines.append("-" * 42)
    for name, v in sorted(initial_vif.items(), key=lambda kv: -kv[1]):
        v_str = "inf" if not np.isfinite(v) else f"{v:.3f}"
        flag = "FLAG (>{thr})".format(thr=args.threshold) if v > args.threshold else ""
        if flag:
            flagged.append(name)
        line = f"{name:<18}  {v_str:>10}  {flag}"
        report_lines.append(line)
        print(f"        {line}")

    if flagged:
        print(f"\n[vif] flagged (VIF > {args.threshold}): {flagged}")
    else:
        print(f"\n[vif] no features exceed threshold {args.threshold}.")

    INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
    payload: dict = {
        "mode": mode,
        "seed": args.seed,
        "n_cycles": args.n_cycles,
        "threshold": args.threshold,
        "n_train_cells": int(len(train_subset)),
        "all_features": feature_cols,
        "initial_vif": {k: (None if not np.isfinite(v) else v) for k, v in initial_vif.items()},
        "flagged_features": flagged,
    }

    if args.drop:
        kept, removed, history = iterative_vif_drop(X_scaled, feature_cols, args.threshold)
        print("\n[vif] dropping order (highest VIF first, recomputed each step):")
        for entry in history:
            if entry.get("dropped"):
                v = entry.get("dropped_vif")
                v_str = "inf" if v is None else f"{v:.2f}"
                print(f"        iter {entry['iteration']}: drop {entry['dropped']} (VIF={v_str}), "
                      f"{entry['n_features']} features remained at start of iter")
        print(f"\n[vif] final kept ({len(kept)}): {kept}")
        print(f"[vif] removed ({len(removed)}): {removed}")
        payload["kept_features"] = kept
        payload["removed_features"] = removed
        payload["iterations"] = history

        out_kept = INTERMEDIATE_DIR / "vif_kept_features.txt"
        out_kept.write_text("\n".join(kept) + "\n")
        print(f"[save] {out_kept}")
    else:
        # report-only: keep all features (the modeling pipeline still uses all 12)
        payload["kept_features"] = feature_cols
        payload["removed_features"] = []
        payload["iterations"] = []
        report_lines.append("")
        report_lines.append("Mode = REPORT-ONLY. No features dropped. Re-run with --drop to prune.")

    out_json = INTERMEDIATE_DIR / "vif_screening.json"
    with out_json.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"[save] {out_json}")

    out_report = INTERMEDIATE_DIR / "vif_report.txt"
    out_report.write_text("\n".join(report_lines) + "\n")
    print(f"[save] {out_report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

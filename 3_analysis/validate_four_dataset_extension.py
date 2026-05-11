"""
Validation checks for the four-dataset paper extension.

This script does not train models. It verifies that the committed extension
artifacts match the intended methodology:
    - Sandia primary subset is 0-100 SOC only.
    - Luh uses the 108-cell standard-cycling subset.
    - Four-dataset feature tables have expected cell/censoring counts.
    - Capacity-normalized table differs from the raw table only as intended.
    - Split files exist for all four datasets/seeds.
    - Full within/cross result summaries exist and contain complete matrices.

Outputs:
    data/intermediate/four_dataset_validation_summary.csv
    data/intermediate/four_dataset_validation_report.md

Usage:
    python 3_analysis/validate_four_dataset_extension.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERMEDIATE_DIR = PROJECT_ROOT / "data" / "intermediate"
SPLITS_DIR = PROJECT_ROOT / "splits" / "sop_v2_four_dataset"
WITHIN_SUMMARY = PROJECT_ROOT / "outputs" / "results_v2_four_dataset_within_34feat_log" / "results_summary.csv"
CROSS_SUMMARY = PROJECT_ROOT / "outputs" / "results_v2_four_dataset_cross_34feat_capnorm_log" / "results_summary.csv"

RAW_FEATURES = INTERMEDIATE_DIR / "features_sop12_four_dataset.csv"
CAPNORM_FEATURES = INTERMEDIATE_DIR / "features_sop12_four_dataset_capnorm.csv"
MANIFEST = INTERMEDIATE_DIR / "four_dataset_manifest.csv"
SANDIA_AUDIT = INTERMEDIATE_DIR / "sandia_cell_audit.csv"
LUH_AUDIT = INTERMEDIATE_DIR / "luh_cell_audit.csv"

DATASETS = ["matr", "hust", "sandia", "luh"]
SEEDS = [42, 123, 456, 789, 1011]
MODELS = ["elastic_net", "pls", "random_forest", "xgboost", "catboost", "gaussian_process", "stacking"]
WINDOWS = [50, 100]


def check(condition: bool, name: str, details: str) -> dict:
    return {"check": name, "status": "PASS" if condition else "FAIL", "details": details}


def markdown_table(df: pd.DataFrame, float_digits: int | None = None) -> str:
    """Small dependency-free Markdown table writer."""
    formatted = df.copy()
    if float_digits is not None:
        for col in formatted.select_dtypes(include=[np.number]).columns:
            formatted[col] = formatted[col].map(lambda x: "" if pd.isna(x) else f"{x:.{float_digits}f}")
    formatted = formatted.astype(object).where(pd.notna(formatted), "")
    header = "| " + " | ".join(map(str, formatted.columns)) + " |"
    separator = "| " + " | ".join(["---"] * len(formatted.columns)) + " |"
    body = ["| " + " | ".join(map(str, row)) + " |" for row in formatted.to_numpy()]
    return "\n".join([header, separator, *body])


def main() -> int:
    rows: list[dict] = []
    required = [RAW_FEATURES, CAPNORM_FEATURES, MANIFEST, SANDIA_AUDIT, LUH_AUDIT, WITHIN_SUMMARY, CROSS_SUMMARY]
    for path in required:
        rows.append(check(path.exists(), f"exists:{path.relative_to(PROJECT_ROOT)}", str(path)))
    if any(row["status"] == "FAIL" for row in rows):
        out = pd.DataFrame(rows)
        out.to_csv(INTERMEDIATE_DIR / "four_dataset_validation_summary.csv", index=False)
        print(out.to_string(index=False))
        return 1

    raw = pd.read_csv(RAW_FEATURES)
    capnorm = pd.read_csv(CAPNORM_FEATURES)
    manifest = pd.read_csv(MANIFEST)
    sandia_audit = pd.read_csv(SANDIA_AUDIT)
    luh_audit = pd.read_csv(LUH_AUDIT)
    within = pd.read_csv(WITHIN_SUMMARY)
    cross = pd.read_csv(CROSS_SUMMARY)

    # Dataset/subset definitions.
    raw_cells = raw[raw["n_cycles"] == 100].drop_duplicates(["dataset", "cell_id"])
    counts = raw_cells.groupby("dataset").agg(
        cells=("cell_id", "nunique"),
        modeling_cells=("is_censored", lambda s: int((s == 0).sum())),
        censored_cells=("is_censored", lambda s: int((s == 1).sum())),
    )
    expected_counts = {
        "matr": (135, 129, 6),
        "hust": (77, 77, 0),
        "sandia": (61, 50, 11),
        "luh": (108, 106, 2),
    }
    for ds, expected in expected_counts.items():
        actual = tuple(int(counts.loc[ds, col]) for col in ["cells", "modeling_cells", "censored_cells"])
        rows.append(check(actual == expected, f"{ds}:cell_counts", f"actual={actual}, expected={expected}"))

    sandia_primary_cells = set(raw_cells.loc[raw_cells["dataset"] == "sandia", "cell_id"])
    sandia_soc = sandia_audit.loc[sandia_audit["cell_id"].isin(sandia_primary_cells), "soc_window"].astype(str)
    rows.append(check(
        len(sandia_primary_cells) == 61 and sandia_soc.eq("0-100").all(),
        "sandia_primary_subset",
        f"n_primary={len(sandia_primary_cells)}, soc_windows={sorted(sandia_soc.unique())}",
    ))

    luh_ok = luh_audit["parse_status"].eq("ok").sum()
    luh_alignment = sorted(luh_audit["alignment_method"].dropna().unique())
    rows.append(check(
        luh_ok == 108 and luh_alignment == ["log_age_capacity"],
        "luh_audit_definition",
        f"ok={luh_ok}, alignment_methods={luh_alignment}",
    ))

    # Capacity normalization.
    rows.append(check(raw.shape == capnorm.shape, "capnorm_shape_matches_raw", f"raw={raw.shape}, capnorm={capnorm.shape}"))
    rows.append(check(
        set(raw["capacity_normalized"].unique()) == {0} and set(capnorm["capacity_normalized"].unique()) == {1},
        "capacity_normalized_flags",
        f"raw={sorted(raw['capacity_normalized'].unique())}, capnorm={sorted(capnorm['capacity_normalized'].unique())}",
    ))
    id_cols = ["dataset", "cell_id", "n_cycles", "q0", "cycle_life", "is_censored"]
    ids_match = raw[id_cols].fillna(-1).equals(capnorm[id_cols].fillna(-1))
    rows.append(check(ids_match, "capnorm_keeps_ids_and_labels", "dataset/cell/window/q0/label/censor columns unchanged"))

    # Splits.
    missing_splits = []
    bad_splits = []
    for ds in DATASETS:
        for seed in SEEDS:
            p = SPLITS_DIR / f"{ds}_{seed}.json"
            if not p.exists():
                missing_splits.append(str(p.relative_to(PROJECT_ROOT)))
                continue
            split = json.loads(p.read_text())
            total = len(split["train"]) + len(split["calibration"]) + len(split["test"])
            expected_modeling = expected_counts[ds][1]
            if total != expected_modeling:
                bad_splits.append(f"{ds}_{seed}: total={total}, expected={expected_modeling}")
    rows.append(check(not missing_splits and not bad_splits, "split_completeness", f"missing={missing_splits}, bad={bad_splits}"))

    # Result completeness.
    expected_within = len(DATASETS) * len(MODELS) * len(WINDOWS)
    expected_cross = len(DATASETS) * (len(DATASETS) - 1) * len(MODELS) * len(WINDOWS)
    rows.append(check(len(within) == expected_within, "within_result_matrix_complete", f"rows={len(within)}, expected={expected_within}"))
    rows.append(check(len(cross) == expected_cross, "cross_result_matrix_complete", f"rows={len(cross)}, expected={expected_cross}"))
    rows.append(check(
        cross["experiment"].nunique() == 12,
        "cross_has_12_directions",
        f"directions={sorted(cross['experiment'].unique())}",
    ))

    # Headline sanity: within-dataset should have at least one positive-R2 model
    # for each dataset at N=100; naive transfer is allowed to be poor.
    best_within = within[within["n_cycles"] == 100].groupby("experiment")["R2_mean"].max()
    rows.append(check(
        (best_within > 0).all(),
        "within_n100_positive_best_r2",
        ", ".join(f"{k}={v:.3f}" for k, v in best_within.items()),
    ))

    summary = pd.DataFrame(rows)
    out_csv = INTERMEDIATE_DIR / "four_dataset_validation_summary.csv"
    out_md = INTERMEDIATE_DIR / "four_dataset_validation_report.md"
    summary.to_csv(out_csv, index=False)

    best_within_rows = (
        within[within["n_cycles"] == 100]
        .sort_values(["experiment", "R2_mean"], ascending=[True, False])
        .groupby("experiment", as_index=False)
        .head(1)
        [["experiment", "model", "MAE_mean", "SMAPE_mean", "R2_mean"]]
    )
    best_cross_rows = (
        cross[cross["n_cycles"] == 100]
        .sort_values(["experiment", "R2_mean"], ascending=[True, False])
        .groupby("experiment", as_index=False)
        .head(1)
        [["experiment", "model", "MAE_mean", "SMAPE_mean", "R2_mean"]]
    )

    lines = [
        "# Four-Dataset Extension Validation",
        "",
        "## Check Summary",
        markdown_table(summary),
        "",
        "## Dataset Counts",
        markdown_table(counts.reset_index()),
        "",
        "## Best Within-Dataset Results at N=100",
        markdown_table(best_within_rows, float_digits=3),
        "",
        "## Best Naive Cross-Dataset Results at N=100",
        markdown_table(best_cross_rows, float_digits=3),
        "",
        "Notes: Sandia primary is restricted to 0-100 SOC-window cells. Luh uses the 108 standard-cycling cells; all parsed through `log_age_capacity` alignment.",
    ]
    out_md.write_text("\n".join(lines) + "\n")

    print(f"[save] {out_csv.relative_to(PROJECT_ROOT)}")
    print(f"[save] {out_md.relative_to(PROJECT_ROOT)}")
    print(summary.to_string(index=False))
    return 0 if (summary["status"] == "PASS").all() else 1


if __name__ == "__main__":
    raise SystemExit(main())

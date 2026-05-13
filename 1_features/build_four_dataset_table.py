"""
Build clean four-dataset feature tables for the paper extension.

Primary dataset definitions:
    - MATR/HUST: existing committed `features_sop12_combined.csv`
    - Sandia: only 0-100 SOC-window cells from the Sandia audit
    - Luh/KIT: all successfully extracted standard-cycling cells

Outputs:
    data/intermediate/features_sop12_four_dataset.csv
    data/intermediate/features_sop12_four_dataset_capnorm.csv
    data/intermediate/four_dataset_manifest.csv

The cap-normalized table follows the same `capacity_normalize` behavior as
`1_features/build_features.py`: raw-capacity columns are divided by each row's
Q0, capacity-variance columns by Q0², and `capacity_normalized` is set to 1.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERMEDIATE_DIR = PROJECT_ROOT / "data" / "intermediate"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_features import CAPACITY_RAW_FEATURES, CAPACITY_VARIANCE_FEATURES  # noqa: E402

BASE_COMBINED = INTERMEDIATE_DIR / "features_sop12_combined.csv"
SANDIA_FEATURES = INTERMEDIATE_DIR / "features_sop12_sandia.csv"
SANDIA_AUDIT = INTERMEDIATE_DIR / "sandia_cell_audit.csv"
LUH_FEATURES = INTERMEDIATE_DIR / "features_sop12_luh.csv"

FOUR_DATASET_OUT = INTERMEDIATE_DIR / "features_sop12_four_dataset.csv"
FOUR_DATASET_CAPNORM_OUT = INTERMEDIATE_DIR / "features_sop12_four_dataset_capnorm.csv"
MANIFEST_OUT = INTERMEDIATE_DIR / "four_dataset_manifest.csv"


def require(path: Path) -> None:
    if not path.exists():
        raise SystemExit(f"[error] missing {path.relative_to(PROJECT_ROOT)}")


def load_primary_sandia() -> pd.DataFrame:
    features = pd.read_csv(SANDIA_FEATURES)
    audit = pd.read_csv(SANDIA_AUDIT)
    if "soc_window" not in audit.columns:
        raise SystemExit("[error] sandia_cell_audit.csv has no soc_window column")
    primary_cells = set(
        audit.loc[
            (audit["parse_status"] == "ok") & (audit["soc_window"].astype(str) == "0-100"),
            "cell_id",
        ]
    )
    filtered = features[features["cell_id"].isin(primary_cells)].copy()
    filtered["dataset"] = "sandia"
    return filtered


def capacity_normalize(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    missing = [col for col in [*CAPACITY_RAW_FEATURES, *CAPACITY_VARIANCE_FEATURES] if col not in out.columns]
    if missing:
        raise SystemExit(f"[error] missing expected capacity feature columns: {missing}")
    for col in CAPACITY_RAW_FEATURES:
        out[col] = out[col] / out["q0"]
    for col in CAPACITY_VARIANCE_FEATURES:
        out[col] = out[col] / (out["q0"] ** 2)
    out["capacity_normalized"] = 1
    return out


def manifest_rows(df: pd.DataFrame, table_name: str) -> list[dict]:
    rows: list[dict] = []
    for (dataset, n_cycles), group in df.groupby(["dataset", "n_cycles"], sort=True):
        cells = group.drop_duplicates("cell_id")
        censored = int(cells["is_censored"].sum())
        rows.append(
            {
                "table": table_name,
                "dataset": dataset,
                "n_cycles": int(n_cycles),
                "rows": int(len(group)),
                "cells": int(cells["cell_id"].nunique()),
                "modeling_cells": int(len(cells) - censored),
                "censored_cells": censored,
                "capacity_normalized": int(group["capacity_normalized"].iloc[0]),
            }
        )
    return rows


def main() -> int:
    for path in [BASE_COMBINED, SANDIA_FEATURES, SANDIA_AUDIT, LUH_FEATURES]:
        require(path)

    base = pd.read_csv(BASE_COMBINED)
    sandia = load_primary_sandia()
    luh = pd.read_csv(LUH_FEATURES)
    luh["dataset"] = "luh"

    expected_cols = list(base.columns)
    for name, frame in [("sandia", sandia), ("luh", luh)]:
        if list(frame.columns) != expected_cols:
            raise SystemExit(
                f"[error] {name} feature columns do not match MATR/HUST base table"
            )

    combined = pd.concat([base, sandia, luh], ignore_index=True)
    combined = combined.sort_values(["dataset", "cell_id", "n_cycles"]).reset_index(drop=True)
    combined.to_csv(FOUR_DATASET_OUT, index=False)

    capnorm = capacity_normalize(combined)
    capnorm.to_csv(FOUR_DATASET_CAPNORM_OUT, index=False)

    manifest = pd.DataFrame(
        manifest_rows(combined, "features_sop12_four_dataset.csv")
        + manifest_rows(capnorm, "features_sop12_four_dataset_capnorm.csv")
    )
    manifest.to_csv(MANIFEST_OUT, index=False)

    print(f"[save] {FOUR_DATASET_OUT.relative_to(PROJECT_ROOT)} rows={len(combined)}")
    print(f"[save] {FOUR_DATASET_CAPNORM_OUT.relative_to(PROJECT_ROOT)} rows={len(capnorm)}")
    print(f"[save] {MANIFEST_OUT.relative_to(PROJECT_ROOT)}")
    print(manifest.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

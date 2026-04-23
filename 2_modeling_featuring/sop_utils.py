"""Helpers for SOP-style feature, target, and censoring configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


TOP8_FEATURES = [
    "IR_delta",
    "dQd_slope",
    "Qd_mean",
    "IR_slope",
    "Tavg_mean",
    "IR_mean",
    "Qd_std",
    "IR_std",
]

# Transition mapping based on the richer early-cycle table that already exists
# in the repository. This can be replaced with the final SOP list if needed.
SOP12_TRANSITION_FEATURES = [
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

FEATURE_SETS = {
    "top8": TOP8_FEATURES,
    "top7_no_qd_std": [feature for feature in TOP8_FEATURES if feature != "Qd_std"],
    "sop12_transition": SOP12_TRANSITION_FEATURES,
    "sop12": SOP12_TRANSITION_FEATURES,
    "sop_common_capacity_dqdv": [
        "Qd_mean",
        "Qd_std",
        "dQd_slope",
        "dqdv_peak_delta",
        "dqdv_peak_std",
        "dqdv_area_delta",
        "dqdv_peakpos_delta",
    ],
}


@dataclass(frozen=True)
class PreparedDataset:
    frame: pd.DataFrame
    target_column: str
    censor_summary: dict[str, float | int] | None


def parse_feature_columns(explicit_columns: str | None) -> list[str] | None:
    if explicit_columns is None:
        return None
    columns = [col.strip() for col in explicit_columns.split(",") if col.strip()]
    return columns or None


def resolve_feature_columns(
    df: pd.DataFrame,
    *,
    feature_set: str = "top8",
    explicit_columns: list[str] | None = None,
) -> list[str]:
    columns = explicit_columns or FEATURE_SETS.get(feature_set)
    if columns is None:
        available = ", ".join(sorted(FEATURE_SETS))
        raise SystemExit(f"Unknown feature set '{feature_set}'. Available: {available}")
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise SystemExit(
            "Dataset does not contain the requested feature columns: "
            f"{missing}. Available columns: {list(df.columns)}"
        )
    return list(columns)


def parse_dataset_label_map(raw_value: str | None) -> dict[str, str]:
    if raw_value is None:
        return {}
    mapping: dict[str, str] = {}
    for item in raw_value.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise SystemExit(
                "dataset-label-map must use 'dataset_prefix:column' pairs, "
                f"got: {item}"
            )
        dataset_prefix, label_column = item.split(":", 1)
        dataset_prefix = dataset_prefix.strip()
        label_column = label_column.strip()
        if not dataset_prefix or not label_column:
            raise SystemExit(f"Invalid dataset-label-map pair: {item}")
        mapping[dataset_prefix] = label_column
    return mapping


def _infer_dataset_prefix(series: pd.Series) -> pd.Series:
    prefixes = series.astype(str).str.extract(r"^(b\d+)")[0]
    if prefixes.isna().any():
        raise SystemExit("Could not infer dataset prefix from every cell_id.")
    return prefixes


def _apply_dataset_specific_target(
    df: pd.DataFrame,
    *,
    default_label_column: str,
    dataset_label_map: dict[str, str],
) -> str:
    target_column = "_target_label"
    df[target_column] = pd.NA

    if default_label_column not in df.columns:
        raise SystemExit(f"Default label column not found: {default_label_column}")

    for prefix, group in df.groupby("dataset_prefix").groups.items():
        label_column = dataset_label_map.get(prefix, default_label_column)
        if label_column not in df.columns:
            raise SystemExit(
                f"Label column '{label_column}' for dataset '{prefix}' not found in dataset."
            )
        df.loc[group, target_column] = df.loc[group, label_column]

    df[target_column] = pd.to_numeric(df[target_column], errors="coerce")
    return target_column


def prepare_dataset(
    df: pd.DataFrame,
    *,
    default_label_column: str = "cycle_life",
    dataset_label_map: dict[str, str] | None = None,
    censor_column: str | None = None,
    drop_censored: bool = False,
) -> PreparedDataset:
    prepared = df.copy()
    if "dataset_prefix" in prepared.columns and prepared["dataset_prefix"].notna().all():
        prepared["dataset_prefix"] = prepared["dataset_prefix"].astype(str)
    else:
        prepared["dataset_prefix"] = _infer_dataset_prefix(prepared["cell_id"])
    prepared.dropna(subset=["cell_id", "n_cycles"], inplace=True)

    target_column = _apply_dataset_specific_target(
        prepared,
        default_label_column=default_label_column,
        dataset_label_map=dataset_label_map or {},
    )
    prepared.dropna(subset=[target_column], inplace=True)

    censor_summary = None
    if censor_column is not None:
        if censor_column not in prepared.columns:
            raise SystemExit(f"Censor column not found: {censor_column}")
        censor_values = prepared[censor_column].astype(bool)
        censor_summary = {
            "column": censor_column,
            "num_rows": int(len(prepared)),
            "num_censored": int(censor_values.sum()),
            "censor_rate": float(censor_values.mean()) if len(prepared) else 0.0,
        }
        if drop_censored:
            prepared = prepared.loc[~censor_values].copy()

    return PreparedDataset(
        frame=prepared,
        target_column=target_column,
        censor_summary=censor_summary,
    )


def load_prepared_dataset(
    path,
    *,
    default_label_column: str = "cycle_life",
    dataset_label_map: dict[str, str] | None = None,
    censor_column: str | None = None,
    drop_censored: bool = False,
) -> PreparedDataset:
    dataset = pd.read_csv(path)
    return prepare_dataset(
        dataset,
        default_label_column=default_label_column,
        dataset_label_map=dataset_label_map,
        censor_column=censor_column,
        drop_censored=drop_censored,
    )


def ensure_columns_present(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise SystemExit(f"Dataset missing required columns: {missing}")

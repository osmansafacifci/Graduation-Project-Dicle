"""Centralised constants used across data, feature, model, and analysis scripts.

Keeping these in one place prevents silent divergence when a column name,
feature list, or domain constant is updated in one file but not in others.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Random seeds (cell-level stratified splits)
# ---------------------------------------------------------------------------
SEEDS: list[int] = [42, 123, 456, 789, 1011]

# ---------------------------------------------------------------------------
# Reserved (non-feature) CSV columns
# ---------------------------------------------------------------------------
META_COLS: set[str] = {
    "dataset", "cell_id", "n_cycles", "q0", "cycle_life",
    "is_censored", "capacity_normalized",
}

# ---------------------------------------------------------------------------
# Feature column lists
# ---------------------------------------------------------------------------
SOP12_FEATURE_COLS: list[str] = [
    "Qdis_N", "delta_Qdis", "retention_ratio", "slope_linear",
    "variance_Qdis", "range_Qdis", "max_drop", "std_diff",
    "skewness_Qdis", "slope_ratio", "Qdis_cycle10", "mean_diff",
]

EXTENDED_FEATURE_COLS: list[str] = [
    "poly2_a", "poly2_b", "poly2_c", "exp_decay_k",
    "cycle_to_99pct", "cycle_to_98pct", "cycle_to_95pct",
    "slope_first_quarter", "slope_last_quarter",
    "autocorr_lag1", "knee_cycle", "n_capacity_jumps",
]

EXTENDED2_FEATURE_COLS: list[str] = [
    "accel_mean", "accel_std", "accel_max_abs",
    "linearity_r2", "kurtosis_Qdis",
    "fft_top3_energy_ratio", "spectral_entropy", "sample_entropy",
    "pos_neg_diff_ratio", "mad_Qdis",
]

ALL_FEATURE_COLS: list[str] = (
    SOP12_FEATURE_COLS + EXTENDED_FEATURE_COLS + EXTENDED2_FEATURE_COLS
)  # 34 total

# ---------------------------------------------------------------------------
# Capacity-unit features (divided by Q0 for cross-chemistry normalisation)
# ---------------------------------------------------------------------------
CAPACITY_RAW_FEATURES: set[str] = {
    "Qdis_N", "delta_Qdis", "slope_linear", "Qdis_cycle10", "max_drop",
    "mean_diff", "std_diff", "range_Qdis",
    "poly2_a", "poly2_b", "poly2_c",
    "slope_first_quarter", "slope_last_quarter",
    "accel_mean", "accel_std", "accel_max_abs",
    "mad_Qdis",
}
CAPACITY_VARIANCE_FEATURES: set[str] = {"variance_Qdis"}

# ---------------------------------------------------------------------------
# Batch-continuation metadata (Severson 2019)
# ---------------------------------------------------------------------------
BATCH1_CONTINUATION_FROM_BATCH2: dict[str, dict[str, object]] = {
    "b1c0": {"source_cell": "b2c7", "add_len": 662},
    "b1c1": {"source_cell": "b2c8", "add_len": 981},
    "b1c2": {"source_cell": "b2c9", "add_len": 1060},
    "b1c3": {"source_cell": "b2c15", "add_len": 208},
    "b1c4": {"source_cell": "b2c16", "add_len": 482},
}

# ---------------------------------------------------------------------------
# Model / dataset enumerations
# ---------------------------------------------------------------------------
ALL_MODELS: list[str] = [
    "elastic_net", "pls", "random_forest", "xgboost",
    "catboost", "gaussian_process", "stacking",
]
ALL_DATASETS: list[str] = ["matr", "hust", "sandia", "luh"]

PRIMARY_MODEL_BY_DATASET: dict[str, str] = {
    "matr": "catboost",
    "hust": "random_forest",
    "sandia": "xgboost",
    "luh": "gaussian_process",
}

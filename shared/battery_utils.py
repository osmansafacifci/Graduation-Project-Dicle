"""Shared utility functions for battery data processing.

Contains helpers that were duplicated across 0_data/, 1_features/,
and 3_analysis/ scripts.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Q0 / cycle-life computation
# ---------------------------------------------------------------------------

def compute_q0(qd) -> float:
    """Q0 = median of valid Q_discharge over cycles 2-5 (0-indexed: indices 1..4)."""
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


def compute_eol(
    qd,
    q0: float,
    threshold_fraction: float = 0.80,
    k_consecutive: int = 3,
) -> float:
    """EOL = first cycle where QD stays <= threshold for k_consecutive cycles.

    1-indexed cycle number.  Used by the audit scripts for the historical
    3-consecutive-cycle diagnostic.
    """
    if not np.isfinite(q0) or q0 <= 0:
        return float("nan")
    qd = np.asarray(qd, dtype=float).ravel()
    threshold = threshold_fraction * q0
    for i in range(len(qd) - k_consecutive + 1):
        window = qd[i : i + k_consecutive]
        if np.all(np.isfinite(window)) and np.all(window <= threshold):
            return float(i + 1)
    return float("nan")


# ---------------------------------------------------------------------------
# Model prediction sanitisation
# ---------------------------------------------------------------------------

def safe_pred(model, X: np.ndarray) -> np.ndarray:
    """Predict and sanitise: ravel, replace NaN/inf, clip to [-1e9, 1e9]."""
    raw = model.predict(X)
    if hasattr(raw, "ravel"):
        raw = raw.ravel()
    raw = np.nan_to_num(raw, nan=0.0, posinf=1e9, neginf=-1e9)
    return np.clip(raw, -1e9, 1e9)


# ---------------------------------------------------------------------------
# Path display
# ---------------------------------------------------------------------------

def display_path(path: Path, project_root: Path | None = None) -> str:
    """Return *path* relative to *project_root*, or absolute if outside."""
    if project_root is None:
        return str(path)
    try:
        return str(path.relative_to(project_root))
    except ValueError:
        return str(path)


# ---------------------------------------------------------------------------
# Split / data-window helpers
# ---------------------------------------------------------------------------

def load_split(splits_dir: Path, dataset: str, seed: int) -> dict:
    """Load a canonical ``{train, calibration, test}`` cell-ID split JSON."""
    path = splits_dir / f"{dataset}_{seed}.json"
    with path.open() as f:
        return json.load(f)


def dataset_window(
    df: pd.DataFrame, dataset: str, n_cycles: int,
) -> pd.DataFrame:
    """Slice *df* to one dataset + prediction window, uncensored cells only."""
    return df[
        (df["dataset"] == dataset)
        & (df["n_cycles"] == n_cycles)
        & (df["is_censored"] == 0)
    ].copy()


# ---------------------------------------------------------------------------
# Target-side point adapter
# ---------------------------------------------------------------------------

def fit_point_adapter(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    adapter_type: str,
) -> tuple[float, float]:
    """Return ``(slope, intercept)`` for the requested target-side point adapter."""
    if adapter_type == "residual_mean":
        return 1.0, float(np.mean(y_true - y_pred))
    if adapter_type != "linear":
        raise ValueError(f"unknown adapter_type={adapter_type}")
    if np.std(y_pred) < 1e-12:
        return 1.0, float(np.mean(y_true) - np.mean(y_pred))
    rank_warning = getattr(getattr(np, "exceptions", np), "RankWarning", Warning)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=rank_warning)
        a, b = np.polyfit(y_pred, y_true, 1)
    return float(a), float(b)


# ---------------------------------------------------------------------------
# Deterministic seed from string parts
# ---------------------------------------------------------------------------

def stable_seed(*parts: object) -> int:
    """Produce a deterministic seed from arbitrary hashable parts."""
    text = "|".join(str(p) for p in parts)
    return sum((i + 1) * ord(ch) for i, ch in enumerate(text)) % (2**32 - 1)

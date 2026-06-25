"""Shared metric helpers for battery-life experiments."""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score


def to_cycles(
    pred: np.ndarray,
    *,
    log_target: bool,
    min_cycle: float = 1.0,
    max_cycle: float = 1e9,
) -> np.ndarray:
    """Convert model predictions to bounded cycle-life values.

    Most experiments fit models in log(cycle_life) space and score in cycle
    space. Linear and GP-style models can produce extreme log predictions under
    cross-dataset shift; clipping before exponentiation keeps metrics finite
    without hiding that the prediction is catastrophically wrong.
    """
    arr = np.asarray(pred, dtype=float)
    if log_target:
        log_min = float(np.log(min_cycle))
        log_max = float(np.log(max_cycle))
        arr = np.nan_to_num(arr, nan=log_min, posinf=log_max, neginf=log_min)
        out = np.exp(np.clip(arr, log_min, log_max))
    else:
        out = arr
    out = np.nan_to_num(out, nan=min_cycle, posinf=max_cycle, neginf=min_cycle)
    return np.clip(out, min_cycle, max_cycle)


def fit_with_threaded_joblib(fitter, X_train: np.ndarray, y_train: np.ndarray, *, seed: int):
    """Fit a project model while avoiding loky process spawning in analysis scripts."""
    try:
        from joblib import parallel_backend
    except ImportError:
        return fitter(X_train, y_train, seed=seed)

    with parallel_backend("threading"):
        return fitter(X_train, y_train, seed=seed)


def symmetric_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_pred - y_true)
    with np.errstate(divide="ignore", invalid="ignore"):
        smape = np.where(denom == 0, 0.0, diff / denom)
    return float(np.mean(smape) * 100.0)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "SMAPE": symmetric_mape(y_true, y_pred),
        "R2": float(r2_score(y_true, y_pred)),
    }


def bootstrap_metric_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")
    if len(y_true) < 2:
        return {
            "MAE": {"lower": float("nan"), "upper": float("nan")},
            "SMAPE": {"lower": float("nan"), "upper": float("nan")},
            "R2": {"lower": float("nan"), "upper": float("nan")},
        }

    rng = np.random.default_rng(seed)
    alpha = 1.0 - confidence_level
    lower_q = alpha / 2.0
    upper_q = 1.0 - lower_q

    mae_scores: list[float] = []
    smape_scores: list[float] = []
    r2_scores: list[float] = []

    n = len(y_true)
    indices = np.arange(n)
    for _ in range(n_bootstrap):
        sample_idx = rng.choice(indices, size=n, replace=True)
        sample_true = y_true[sample_idx]
        sample_pred = y_pred[sample_idx]
        sample_metrics = compute_metrics(sample_true, sample_pred)
        mae_scores.append(sample_metrics["MAE"])
        smape_scores.append(sample_metrics["SMAPE"])
        r2_scores.append(sample_metrics["R2"])

    return {
        "MAE": {
            "lower": float(np.quantile(mae_scores, lower_q)),
            "upper": float(np.quantile(mae_scores, upper_q)),
        },
        "SMAPE": {
            "lower": float(np.quantile(smape_scores, lower_q)),
            "upper": float(np.quantile(smape_scores, upper_q)),
        },
        "R2": {
            "lower": float(np.quantile(r2_scores, lower_q)),
            "upper": float(np.quantile(r2_scores, upper_q)),
        },
    }

"""Shared metric helpers for battery-life experiments."""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score


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

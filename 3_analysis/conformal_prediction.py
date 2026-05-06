"""
Standard split conformal prediction for the SOPv2 battery-life pipeline.

This script intentionally uses MAPIE's standard split conformal regressor with
the plain residual-quantile split-CP recipe:

    train point predictor on training cells
    calibrate q_hat on held-out calibration residuals
    report intervals [y_hat - q_hat, y_hat + q_hat] on test cells

The cross-dataset modes are labeled by their calibration domain:

  1. within_split_cp
       Train/calibrate/test within the same dataset using the existing
       70/15/15 cell splits.

  2. cross_source_calibrated_cp
       Train and calibrate on the source dataset, then test on the full
       target dataset. This is a diagnostic; exchangeability is not expected
       under dataset shift.

  3. cross_target_calibrated_cp
       Train on the source dataset, calibrate on k labeled target cells, then
       test on the remaining target cells. This is standard split CP with a
       target-domain calibration set, not a new conformal method.

  4. cross_target_adapted_cp
       Train on the source dataset, fit a small target-domain point adapter
       on k_adapter labeled target cells. The default adapter is residual_mean,
       i.e. a constant residual correction y_adapted = y_source + mean residual.
       A two-parameter linear adapter is available as a sensitivity check.
       conformalize on a disjoint k_target labeled target calibration set,
       then test on the remaining target cells.

Inputs:
    data/intermediate/features_sop12_combined.csv
    splits/sop_v2/{matr,hust}_{seed}.json

Outputs:
    outputs/results_v2_conformal/results_detailed.csv
    outputs/results_v2_conformal/results_summary.csv
    outputs/results_v2_conformal/results_conformal.json

Usage:
    python 3_analysis/conformal_prediction.py
    python 3_analysis/conformal_prediction.py --models catboost --seeds 42
    python 3_analysis/conformal_prediction.py --target-k-values 10 15 20
    python 3_analysis/conformal_prediction.py --adapter-k-values 10 15 20
    python 3_analysis/conformal_prediction.py --confidence-levels 0.90 0.95
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd
import sklearn
import mapie
from mapie.regression import SplitConformalRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(PROJECT_ROOT / "2_models"))
from metrics_utils import compute_metrics  # noqa: E402
from run_experiments import (  # noqa: E402
    META_COLS,
    SEEDS,
    fit_catboost,
    fit_elastic_net,
    fit_gaussian_process,
    fit_pls,
    fit_random_forest,
    fit_stacking,
    fit_xgboost,
)

warnings.filterwarnings("ignore", category=ConvergenceWarning)

FEATURES_PATH = PROJECT_ROOT / "data" / "intermediate" / "features_sop12_combined.csv"
SPLITS_DIR = PROJECT_ROOT / "splits" / "sop_v2"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "results_v2_conformal"

ALL_MODELS = [
    "elastic_net",
    "pls",
    "random_forest",
    "xgboost",
    "catboost",
    "gaussian_process",
    "stacking",
]
DEFAULT_MODELS = ["catboost", "random_forest"]
DEFAULT_DATASETS = ["matr", "hust"]
DEFAULT_WINDOWS = [100]
DEFAULT_TARGET_KS = [10, 15, 20]
DEFAULT_TARGET_REPEATS = 20
DEFAULT_CONFIDENCE_LEVELS = [0.90, 0.95]
ALL_ADAPTER_TYPES = ["residual_mean", "linear"]
DEFAULT_ADAPTER_TYPES = ["residual_mean"]

FITTERS = {
    "elastic_net": fit_elastic_net,
    "pls": fit_pls,
    "random_forest": fit_random_forest,
    "gaussian_process": fit_gaussian_process,
    "xgboost": fit_xgboost,
    "catboost": fit_catboost,
    "stacking": fit_stacking,
}


@dataclass
class FittedPredictor:
    model_name: str
    model: object
    scaler: StandardScaler
    feature_cols: list[str]
    log_target: bool


@dataclass
class TargetPointAdapter:
    """Scientific target-domain point adapter before CP calibration."""

    adapter_type: str
    slope: float
    intercept: float

    @classmethod
    def fit(cls, y_pred: np.ndarray, y_true: np.ndarray, adapter_type: str) -> "TargetPointAdapter":
        y_pred = np.asarray(y_pred, dtype=float)
        y_true = np.asarray(y_true, dtype=float)
        if len(y_pred) == 0:
            return cls(adapter_type, 1.0, 0.0)
        if adapter_type == "residual_mean":
            return cls(adapter_type, 1.0, float(np.mean(y_true - y_pred)))
        if adapter_type != "linear":
            raise ValueError(f"Unknown adapter_type: {adapter_type}")
        if np.std(y_pred) < 1e-12:
            return cls(adapter_type, 1.0, float(np.mean(y_true - y_pred)))
        slope, intercept = np.polyfit(y_pred, y_true, 1)
        return cls(adapter_type, float(slope), float(intercept))

    def predict(self, y_pred: np.ndarray) -> np.ndarray:
        adapted = self.slope * np.asarray(y_pred, dtype=float) + self.intercept
        return np.clip(np.nan_to_num(adapted, nan=1.0, posinf=1e9, neginf=1.0), 1.0, 1e9)


class MapieCycleRegressor(RegressorMixin, BaseEstimator):
    """sklearn-compatible regressor exposing cycle-space predictions to MAPIE."""

    def __init__(self, predictor: FittedPredictor, target_adapter: TargetPointAdapter | None = None):
        self.predictor = predictor
        self.target_adapter = target_adapter
        self.is_fitted_ = True
        self.fitted_ = True
        self.n_features_in_ = len(predictor.feature_cols)
        self.feature_names_in_ = np.asarray(predictor.feature_cols, dtype=object)

    def fit(self, X, y=None):
        self.is_fitted_ = True
        self.fitted_ = True
        return self

    def __sklearn_is_fitted__(self) -> bool:
        return True

    def predict(self, X):
        X_s = self.predictor.scaler.transform(np.asarray(X, dtype=float))
        X_s = np.clip(np.nan_to_num(X_s, nan=0.0, posinf=0.0, neginf=0.0), -1e6, 1e6)
        pred = to_cycles(safe_pred(self.predictor.model, X_s), log_target=self.predictor.log_target)
        if self.target_adapter is not None:
            return self.target_adapter.predict(pred)
        return pred


def safe_pred(model: object, X: np.ndarray) -> np.ndarray:
    raw = model.predict(X)
    if hasattr(raw, "ravel"):
        raw = raw.ravel()
    raw = np.nan_to_num(raw, nan=0.0, posinf=1e9, neginf=-1e9)
    return np.clip(raw, -1e9, 1e9)


def to_cycles(pred: np.ndarray, *, log_target: bool) -> np.ndarray:
    if log_target:
        pred = np.clip(pred, np.log(1.0), np.log(1e9))
        out = np.exp(pred)
    else:
        out = pred
    return np.clip(np.nan_to_num(out, nan=1.0, posinf=1e9, neginf=1.0), 1.0, 1e9)


def fit_predictor(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    *,
    model_name: str,
    seed: int,
    log_target: bool,
) -> FittedPredictor:
    X_train = train_df[feature_cols].to_numpy(dtype=float)
    y_train = train_df["cycle_life"].to_numpy(dtype=float)
    y_train_fit = np.log(y_train) if log_target else y_train

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_train_s = np.clip(np.nan_to_num(X_train_s, nan=0.0, posinf=0.0, neginf=0.0), -1e6, 1e6)

    result = FITTERS[model_name](X_train_s, y_train_fit, seed=seed)
    model = result[0] if isinstance(result, tuple) else result
    return FittedPredictor(model_name, model, scaler, feature_cols, log_target)


def predict_cycles(predictor: FittedPredictor, df: pd.DataFrame) -> np.ndarray:
    X = df[predictor.feature_cols].to_numpy(dtype=float)
    X_s = predictor.scaler.transform(X)
    X_s = np.clip(np.nan_to_num(X_s, nan=0.0, posinf=0.0, neginf=0.0), -1e6, 1e6)
    return to_cycles(safe_pred(predictor.model, X_s), log_target=predictor.log_target)


def mapie_split_interval(
    predictor: FittedPredictor,
    cal_df: pd.DataFrame,
    test_df: pd.DataFrame,
    confidence_level: float,
    target_adapter: TargetPointAdapter | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, bool, int]:
    estimator = MapieCycleRegressor(predictor, target_adapter=target_adapter)
    X_cal = cal_df[predictor.feature_cols].to_numpy(dtype=float)
    y_cal = cal_df["cycle_life"].to_numpy(dtype=float)
    X_test = test_df[predictor.feature_cols].to_numpy(dtype=float)

    conformal = SplitConformalRegressor(
        estimator=estimator,
        confidence_level=confidence_level,
        conformity_score="absolute",
        prefit=True,
    )
    conformal.conformalize(X_cal, y_cal)
    pred_test, intervals = conformal.predict_interval(X_test, allow_infinite_bounds=True)

    lower = intervals[:, 0, 0]
    upper = intervals[:, 1, 0]

    alpha = 1.0 - confidence_level
    pred_cal = estimator.predict(X_cal)
    q_hat, finite_q, quantile_rank = finite_sample_quantile(np.abs(y_cal - pred_cal), alpha)
    return pred_test, lower, upper, q_hat, finite_q, quantile_rank


def finite_sample_quantile(abs_residuals: np.ndarray, alpha: float) -> tuple[float, bool, int]:
    """Split-CP residual quantile with finite-sample rank correction.

    Rank = ceil((n_cal + 1) * (1 - alpha)). If this rank exceeds n_cal, the
    exact finite-sample interval is infinite; this happens for 90% CP with
    very tiny calibration sets, e.g. k <= 8.
    """
    residuals = np.sort(np.asarray(abs_residuals, dtype=float))
    n = len(residuals)
    if n == 0:
        return float("nan"), False, 0
    rank = int(math.ceil((n + 1) * (1.0 - alpha)))
    if rank > n:
        return float("inf"), False, rank
    return float(residuals[rank - 1]), True, rank


def winkler_score(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray, alpha: float) -> np.ndarray:
    width = upper - lower
    below = y_true < lower
    above = y_true > upper
    score = width.copy()
    score[below] += (2.0 / alpha) * (lower[below] - y_true[below])
    score[above] += (2.0 / alpha) * (y_true[above] - upper[above])
    return score


def wilson_interval(successes: int, n: int, confidence_level: float = 0.95) -> tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    z = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
    phat = successes / n
    denom = 1.0 + (z**2 / n)
    center = (phat + (z**2 / (2.0 * n))) / denom
    half = z * math.sqrt((phat * (1.0 - phat) / n) + (z**2 / (4.0 * n**2))) / denom
    return float(max(0.0, center - half)), float(min(1.0, center + half))


def coverage_stats(covered: np.ndarray) -> dict[str, float | int]:
    n = int(len(covered))
    successes = int(np.sum(covered)) if n else 0
    lo, hi = wilson_interval(successes, n)
    return {
        "n": n,
        "covered_count": successes,
        "coverage": float(successes / n) if n else float("nan"),
        "coverage_wilson95_lower": lo,
        "coverage_wilson95_upper": hi,
    }


def stratified_coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> dict[str, float | int]:
    if len(y_true) < 3:
        return {
            "coverage_short": float("nan"),
            "coverage_mid": float("nan"),
            "coverage_long": float("nan"),
            "n_short": 0,
            "n_mid": 0,
            "n_long": 0,
        }
    q1, q2 = np.quantile(y_true, [1.0 / 3.0, 2.0 / 3.0])
    masks = {
        "short": y_true <= q1,
        "mid": (y_true > q1) & (y_true <= q2),
        "long": y_true > q2,
    }
    covered = (y_true >= lower) & (y_true <= upper)
    out: dict[str, float | int] = {}
    for label, mask in masks.items():
        n = int(mask.sum())
        out[f"n_{label}"] = n
        out[f"coverage_{label}"] = float(np.mean(covered[mask])) if n else float("nan")
    return out


def size_stratified_coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> dict[str, float | int]:
    """Coverage on shorter-lived vs longer-lived halves of the evaluation set."""
    y_true = np.asarray(y_true, dtype=float)
    covered = (y_true >= lower) & (y_true <= upper)
    if len(y_true) < 2:
        return {
            "n_short_life": 0,
            "n_long_life": 0,
            "coverage_short_life": float("nan"),
            "coverage_long_life": float("nan"),
            "coverage_short_life_wilson95_lower": float("nan"),
            "coverage_short_life_wilson95_upper": float("nan"),
            "coverage_long_life_wilson95_lower": float("nan"),
            "coverage_long_life_wilson95_upper": float("nan"),
            "short_long_coverage_gap": float("nan"),
            "lifetime_split_threshold": float("nan"),
        }

    order = np.argsort(y_true)
    split = len(y_true) // 2
    short_idx = order[:split]
    long_idx = order[split:]

    out: dict[str, float | int] = {}
    for label, idx in {"short_life": short_idx, "long_life": long_idx}.items():
        stats = coverage_stats(covered[idx])
        out[f"n_{label}"] = stats["n"]
        out[f"coverage_{label}"] = stats["coverage"]
        out[f"coverage_{label}_wilson95_lower"] = stats["coverage_wilson95_lower"]
        out[f"coverage_{label}_wilson95_upper"] = stats["coverage_wilson95_upper"]
    out["short_long_coverage_gap"] = (
        float(abs(out["coverage_short_life"] - out["coverage_long_life"]))
        if not pd.isna(out["coverage_short_life"]) and not pd.isna(out["coverage_long_life"])
        else float("nan")
    )
    out["lifetime_split_threshold"] = float(y_true[long_idx[0]]) if len(long_idx) else float("nan")
    return out


def evaluate_interval(
    *,
    scenario: str,
    source: str,
    target: str,
    calibration_domain: str,
    model: str,
    n_cycles: int,
    seed: int,
    repeat: int | None,
    k_target: int | None,
    k_adapter: int | None,
    n_train: int,
    n_adapter: int,
    n_calibration: int,
    adapter_type: str,
    adapter_slope: float | None,
    adapter_intercept: float | None,
    y_test: np.ndarray,
    pred_test: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    q_hat: float,
    finite_q: bool,
    quantile_rank: int,
    confidence_level: float,
) -> dict:
    alpha = 1.0 - confidence_level
    covered = (y_test >= lower) & (y_test <= upper)
    width = upper - lower
    wis = winkler_score(y_test, lower, upper, alpha)
    point_metrics = compute_metrics(y_test, pred_test)
    cov = coverage_stats(covered)

    row = {
        "scenario": scenario,
        "source": source,
        "target": target,
        "calibration_domain": calibration_domain,
        "model": model,
        "n_cycles": int(n_cycles),
        "seed": int(seed),
        "repeat": float("nan") if repeat is None else int(repeat),
        "k_target": float("nan") if k_target is None else int(k_target),
        "k_adapter": float("nan") if k_adapter is None else int(k_adapter),
        "n_train": int(n_train),
        "n_adapter": int(n_adapter),
        "n_calibration": int(n_calibration),
        "n_test": int(len(y_test)),
        "adapter_type": adapter_type,
        "adapter_slope": float("nan") if adapter_slope is None else float(adapter_slope),
        "adapter_intercept": float("nan") if adapter_intercept is None else float(adapter_intercept),
        "confidence_level": float(confidence_level),
        "alpha": float(alpha),
        "quantile_rank": int(quantile_rank),
        "q_hat": float(q_hat),
        "finite_q": bool(finite_q),
        "covered_count": int(cov["covered_count"]),
        "coverage": cov["coverage"],
        "coverage_gap": float(abs(confidence_level - cov["coverage"])) if not pd.isna(cov["coverage"]) else float("nan"),
        "coverage_wilson95_lower": cov["coverage_wilson95_lower"],
        "coverage_wilson95_upper": cov["coverage_wilson95_upper"],
        "mean_width": float(np.mean(width)) if len(width) else float("nan"),
        "median_width": float(np.median(width)) if len(width) else float("nan"),
        "winkler_mean": float(np.mean(wis)) if len(wis) else float("nan"),
        "MAE": point_metrics["MAE"],
        "SMAPE": point_metrics["SMAPE"],
        "R2": point_metrics["R2"],
    }
    row.update(stratified_coverage(y_test, lower, upper))
    row.update(size_stratified_coverage(y_test, lower, upper))
    return row


def load_split(dataset: str, seed: int) -> dict:
    split_path = SPLITS_DIR / f"{dataset}_{seed}.json"
    if not split_path.exists():
        raise FileNotFoundError(f"Missing split file: {split_path}")
    with split_path.open() as f:
        return json.load(f)


def dataset_window(df: pd.DataFrame, dataset: str, n_cycles: int) -> pd.DataFrame:
    return df[(df["dataset"] == dataset) & (df["n_cycles"] == n_cycles) & (df["is_censored"] == 0)].copy()


def split_frames(sub: pd.DataFrame, split: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = sub[sub["cell_id"].isin(split["train"])].copy()
    cal = sub[sub["cell_id"].isin(split["calibration"])].copy()
    test = sub[sub["cell_id"].isin(split["test"])].copy()
    return train, cal, test


def stable_seed(*parts: object) -> int:
    text = "|".join(str(p) for p in parts)
    return sum((i + 1) * ord(ch) for i, ch in enumerate(text)) % (2**32 - 1)


def within_split_cp(
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    datasets: list[str],
    windows: list[int],
    models: list[str],
    seeds: list[int],
    confidence_levels: list[float],
    log_target: bool,
) -> list[dict]:
    rows: list[dict] = []
    for dataset in datasets:
        for seed in seeds:
            split = load_split(dataset, seed)
            for n_cycles in windows:
                sub = dataset_window(df, dataset, n_cycles)
                train_df, cal_df, test_df = split_frames(sub, split)
                if len(train_df) < 5 or len(cal_df) < 2 or len(test_df) < 2:
                    continue
                for model_name in models:
                    print(f"  within {dataset} seed={seed} N={n_cycles} model={model_name}")
                    predictor = fit_predictor(
                        train_df,
                        feature_cols,
                        model_name=model_name,
                        seed=seed,
                        log_target=log_target,
                    )
                    for confidence_level in confidence_levels:
                        pred_test, lower, upper, q_hat, finite_q, quantile_rank = mapie_split_interval(
                            predictor,
                            cal_df,
                            test_df,
                            confidence_level,
                        )
                        rows.append(
                            evaluate_interval(
                                scenario="within_split_cp",
                                source=dataset,
                                target=dataset,
                                calibration_domain=dataset,
                                model=model_name,
                                n_cycles=n_cycles,
                                seed=seed,
                                repeat=None,
                                k_target=None,
                                k_adapter=None,
                                n_train=len(train_df),
                                n_adapter=0,
                                n_calibration=len(cal_df),
                                adapter_type="none",
                                adapter_slope=None,
                                adapter_intercept=None,
                                y_test=test_df["cycle_life"].to_numpy(dtype=float),
                                pred_test=pred_test,
                                lower=lower,
                                upper=upper,
                                q_hat=q_hat,
                                finite_q=finite_q,
                                quantile_rank=quantile_rank,
                                confidence_level=confidence_level,
                            )
                        )
    return rows


def cross_cp(
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    datasets: list[str],
    windows: list[int],
    models: list[str],
    seeds: list[int],
    confidence_levels: list[float],
    log_target: bool,
    target_k_values: list[int],
    adapter_k_values: list[int],
    adapter_types: list[str],
    target_repeats: int,
) -> list[dict]:
    rows: list[dict] = []
    pairs = [(s, t) for s in datasets for t in datasets if s != t]
    for src, tgt in pairs:
        for seed in seeds:
            src_split = load_split(src, seed)
            for n_cycles in windows:
                src_sub = dataset_window(df, src, n_cycles)
                tgt_sub = dataset_window(df, tgt, n_cycles)
                src_train, src_cal, _ = split_frames(src_sub, src_split)
                if len(src_train) < 5 or len(src_cal) < 2 or len(tgt_sub) < 3:
                    continue
                for model_name in models:
                    print(f"  cross {src}->{tgt} seed={seed} N={n_cycles} model={model_name}")
                    predictor = fit_predictor(
                        src_train,
                        feature_cols,
                        model_name=model_name,
                        seed=seed,
                        log_target=log_target,
                    )
                    target_y = tgt_sub["cycle_life"].to_numpy(dtype=float)
                    for confidence_level in confidence_levels:
                        src_pred_test, src_lower, src_upper, src_q_hat, src_finite_q, src_quantile_rank = mapie_split_interval(
                            predictor,
                            src_cal,
                            tgt_sub,
                            confidence_level,
                        )
                        rows.append(
                            evaluate_interval(
                                scenario="cross_source_calibrated_cp",
                                source=src,
                                target=tgt,
                                calibration_domain=src,
                                model=model_name,
                                n_cycles=n_cycles,
                                seed=seed,
                                repeat=None,
                                k_target=None,
                                k_adapter=None,
                                n_train=len(src_train),
                                n_adapter=0,
                                n_calibration=len(src_cal),
                                adapter_type="none",
                                adapter_slope=None,
                                adapter_intercept=None,
                                y_test=target_y,
                                pred_test=src_pred_test,
                                lower=src_lower,
                                upper=src_upper,
                                q_hat=src_q_hat,
                                finite_q=src_finite_q,
                                quantile_rank=src_quantile_rank,
                                confidence_level=confidence_level,
                            )
                        )

                    n_target = len(tgt_sub)
                    target_indices = np.arange(n_target)
                    for k in target_k_values:
                        if k >= n_target - 1:
                            continue
                        for repeat in range(target_repeats):
                            rng = np.random.default_rng(stable_seed(src, tgt, seed, n_cycles, model_name, k, repeat))
                            cal_idx = rng.choice(target_indices, size=k, replace=False)
                            test_idx = np.setdiff1d(target_indices, cal_idx)
                            target_cal_df = tgt_sub.iloc[cal_idx]
                            target_test_df = tgt_sub.iloc[test_idx]
                            for confidence_level in confidence_levels:
                                (
                                    tgt_pred_test,
                                    tgt_lower,
                                    tgt_upper,
                                    tgt_q_hat,
                                    tgt_finite_q,
                                    tgt_quantile_rank,
                                ) = mapie_split_interval(
                                    predictor,
                                    target_cal_df,
                                    target_test_df,
                                    confidence_level,
                                )
                                rows.append(
                                    evaluate_interval(
                                        scenario="cross_target_calibrated_cp",
                                        source=src,
                                        target=tgt,
                                        calibration_domain=tgt,
                                        model=model_name,
                                        n_cycles=n_cycles,
                                        seed=seed,
                                        repeat=repeat,
                                        k_target=k,
                                        k_adapter=None,
                                        n_train=len(src_train),
                                        n_adapter=0,
                                        n_calibration=len(target_cal_df),
                                        adapter_type="none",
                                        adapter_slope=None,
                                        adapter_intercept=None,
                                        y_test=target_y[test_idx],
                                        pred_test=tgt_pred_test,
                                        lower=tgt_lower,
                                        upper=tgt_upper,
                                        q_hat=tgt_q_hat,
                                        finite_q=tgt_finite_q,
                                        quantile_rank=tgt_quantile_rank,
                                        confidence_level=confidence_level,
                                    )
                                )

                    for adapter_type in adapter_types:
                        for k_adapter in adapter_k_values:
                            for k_cp in target_k_values:
                                if k_adapter + k_cp >= n_target - 1:
                                    continue
                                for repeat in range(target_repeats):
                                    rng = np.random.default_rng(
                                        stable_seed(
                                            "adapted",
                                            adapter_type,
                                            src,
                                            tgt,
                                            seed,
                                            n_cycles,
                                            model_name,
                                            k_adapter,
                                            k_cp,
                                            repeat,
                                        )
                                    )
                                    adapter_idx = rng.choice(target_indices, size=k_adapter, replace=False)
                                    remaining_idx = np.setdiff1d(target_indices, adapter_idx)
                                    cal_idx = rng.choice(remaining_idx, size=k_cp, replace=False)
                                    test_idx = np.setdiff1d(remaining_idx, cal_idx)

                                    target_adapter_df = tgt_sub.iloc[adapter_idx]
                                    target_cal_df = tgt_sub.iloc[cal_idx]
                                    target_test_df = tgt_sub.iloc[test_idx]
                                    base_adapter_pred = predict_cycles(predictor, target_adapter_df)
                                    target_adapter = TargetPointAdapter.fit(
                                        base_adapter_pred,
                                        target_adapter_df["cycle_life"].to_numpy(dtype=float),
                                        adapter_type=adapter_type,
                                    )
                                    for confidence_level in confidence_levels:
                                        (
                                            adapted_pred_test,
                                            adapted_lower,
                                            adapted_upper,
                                            adapted_q_hat,
                                            adapted_finite_q,
                                            adapted_quantile_rank,
                                        ) = mapie_split_interval(
                                            predictor,
                                            target_cal_df,
                                            target_test_df,
                                            confidence_level,
                                            target_adapter=target_adapter,
                                        )
                                        rows.append(
                                            evaluate_interval(
                                                scenario="cross_target_adapted_cp",
                                                source=src,
                                                target=tgt,
                                                calibration_domain=tgt,
                                                model=model_name,
                                                n_cycles=n_cycles,
                                                seed=seed,
                                                repeat=repeat,
                                                k_target=k_cp,
                                                k_adapter=k_adapter,
                                                n_train=len(src_train),
                                                n_adapter=len(target_adapter_df),
                                                n_calibration=len(target_cal_df),
                                                adapter_type=target_adapter.adapter_type,
                                                adapter_slope=target_adapter.slope,
                                                adapter_intercept=target_adapter.intercept,
                                                y_test=target_y[test_idx],
                                                pred_test=adapted_pred_test,
                                                lower=adapted_lower,
                                                upper=adapted_upper,
                                                q_hat=adapted_q_hat,
                                                finite_q=adapted_finite_q,
                                                quantile_rank=adapted_quantile_rank,
                                                confidence_level=confidence_level,
                                            )
                                        )
    return rows


def summarize(rows: list[dict]) -> pd.DataFrame:
    detailed = pd.DataFrame(rows)
    if detailed.empty:
        return detailed
    group_cols = [
        "scenario",
        "source",
        "target",
        "calibration_domain",
        "model",
        "n_cycles",
        "confidence_level",
        "adapter_type",
        "k_adapter",
        "k_target",
    ]
    numeric_cols = [
        "coverage",
        "coverage_gap",
        "coverage_wilson95_lower",
        "coverage_wilson95_upper",
        "mean_width",
        "median_width",
        "winkler_mean",
        "MAE",
        "SMAPE",
        "R2",
        "q_hat",
        "finite_q",
        "adapter_slope",
        "adapter_intercept",
        "coverage_short",
        "coverage_mid",
        "coverage_long",
        "coverage_short_life",
        "coverage_long_life",
        "coverage_short_life_wilson95_lower",
        "coverage_short_life_wilson95_upper",
        "coverage_long_life_wilson95_lower",
        "coverage_long_life_wilson95_upper",
        "short_long_coverage_gap",
        "lifetime_split_threshold",
        "n_train",
        "n_adapter",
        "n_calibration",
        "n_test",
        "n_short_life",
        "n_long_life",
    ]
    agg_spec = {}
    for col in numeric_cols:
        if col in detailed.columns:
            agg_spec[col] = (
                ["mean", "std"]
                if col not in {"finite_q", "n_train", "n_adapter", "n_calibration", "n_test"}
                else ["mean"]
            )
    grouped = detailed.groupby(group_cols, dropna=False)
    summary = grouped.agg(agg_spec)
    summary.columns = [f"{base}_{stat}" for base, stat in summary.columns]
    summary = summary.reset_index()
    summary["n_runs"] = grouped.size().to_numpy()
    return summary


def json_clean(value):
    if isinstance(value, dict):
        return {k: json_clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_clean(v) for v in value]
    if isinstance(value, tuple):
        return [json_clean(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if pd.isna(value):
        return None
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS, choices=DEFAULT_DATASETS)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, choices=ALL_MODELS)
    parser.add_argument("--windows", type=int, nargs="+", default=DEFAULT_WINDOWS)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--features-from", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--confidence-levels",
        type=float,
        nargs="+",
        default=DEFAULT_CONFIDENCE_LEVELS,
        help="One or more conformal confidence levels. Default: 0.90 0.95.",
    )
    parser.add_argument(
        "--confidence-level",
        type=float,
        default=None,
        help="Backward-compatible single confidence level; overrides --confidence-levels if provided.",
    )
    parser.add_argument("--log-target", action="store_true", default=True)
    parser.add_argument("--no-log-target", action="store_false", dest="log_target")
    parser.add_argument("--target-k-values", type=int, nargs="+", default=DEFAULT_TARGET_KS)
    parser.add_argument(
        "--adapter-k-values",
        type=int,
        nargs="+",
        default=DEFAULT_TARGET_KS,
        help="Target cells used only to fit the point adapter in cross_target_adapted_cp.",
    )
    parser.add_argument(
        "--adapter-types",
        nargs="+",
        default=DEFAULT_ADAPTER_TYPES,
        choices=ALL_ADAPTER_TYPES,
        help=(
            "Point adapters to fit before separate CP calibration in cross_target_adapted_cp. "
            "Default: residual_mean. Add linear for a sensitivity check."
        ),
    )
    parser.add_argument("--target-repeats", type=int, default=DEFAULT_TARGET_REPEATS)
    parser.add_argument("--within-only", action="store_true")
    parser.add_argument("--cross-only", action="store_true")
    return parser.parse_args()


def resolve_features(df: pd.DataFrame, features_from: Path | None) -> tuple[list[str], str]:
    available = [c for c in df.columns if c not in META_COLS]
    if features_from is None:
        return list(available), f"auto-detected from CSV ({len(available)} features)"
    if not features_from.exists():
        raise FileNotFoundError(f"--features-from {features_from} not found")
    listed = [line.strip() for line in features_from.read_text().splitlines() if line.strip()]
    unknown = [f for f in listed if f not in available]
    if unknown:
        raise ValueError(f"features-from contains columns not in the CSV: {unknown}")
    if not listed:
        raise ValueError(f"--features-from {features_from} is empty")
    return listed, f"loaded from {features_from} ({len(listed)} features)"


def main() -> int:
    args = parse_args()
    if args.within_only and args.cross_only:
        print("[error] choose at most one of --within-only / --cross-only")
        return 1
    if not FEATURES_PATH.exists():
        print(f"[error] {FEATURES_PATH} missing — run build_features.py first.")
        return 1
    confidence_levels = [float(args.confidence_level)] if args.confidence_level is not None else args.confidence_levels
    confidence_levels = sorted(dict.fromkeys(confidence_levels))
    bad_conf = [c for c in confidence_levels if c <= 0.0 or c >= 1.0]
    if bad_conf:
        print(f"[error] confidence levels must be in (0, 1): {bad_conf}")
        return 1

    df = pd.read_csv(FEATURES_PATH)
    feature_cols, feature_source = resolve_features(df, args.features_from)
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[setup] features: {feature_source}")
    print(f"[setup] output_dir: {out_dir}")
    print(f"[setup] models={args.models}, windows={args.windows}, seeds={args.seeds}")
    print(
        f"[setup] confidence_levels={confidence_levels}, "
        f"target_k={args.target_k_values}, adapter_k={args.adapter_k_values}, "
        f"adapter_types={args.adapter_types}, repeats={args.target_repeats}"
    )

    rows: list[dict] = []
    if not args.cross_only:
        print("\n========== within-dataset split CP ==========")
        rows.extend(
            within_split_cp(
                df,
                feature_cols,
                datasets=args.datasets,
                windows=args.windows,
                models=args.models,
                seeds=args.seeds,
                confidence_levels=confidence_levels,
                log_target=args.log_target,
            )
        )
    if not args.within_only:
        print("\n========== cross-dataset split CP ==========")
        rows.extend(
            cross_cp(
                df,
                feature_cols,
                datasets=args.datasets,
                windows=args.windows,
                models=args.models,
                seeds=args.seeds,
                confidence_levels=confidence_levels,
                log_target=args.log_target,
                target_k_values=args.target_k_values,
                adapter_k_values=args.adapter_k_values,
                adapter_types=args.adapter_types,
                target_repeats=args.target_repeats,
            )
        )

    detailed_df = pd.DataFrame(rows)
    summary_df = summarize(rows)

    detailed_path = out_dir / "results_detailed.csv"
    summary_path = out_dir / "results_summary.csv"
    json_path = out_dir / "results_conformal.json"
    detailed_df.to_csv(detailed_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    payload = {
        "protocol": "mapie_standard_split_conformal_prediction_v3",
        "conformal_library": "MAPIE SplitConformalRegressor",
        "mapie_version": mapie.__version__,
        "sklearn_version": sklearn.__version__,
        "feature_set": feature_source,
        "feature_columns": feature_cols,
        "confidence_levels": confidence_levels,
        "log_target": bool(args.log_target),
        "datasets": args.datasets,
        "windows": args.windows,
        "models": args.models,
        "seeds": args.seeds,
        "target_k_values": args.target_k_values,
        "adapter_k_values": args.adapter_k_values,
        "adapter_types": args.adapter_types,
        "target_repeats": args.target_repeats,
        "notes": [
            "Intervals are generated with MAPIE's SplitConformalRegressor using absolute residual conformity scores.",
            "cross_source_calibrated_cp is diagnostic because source-calibration and target-test residuals are not exchangeable under dataset shift.",
            "cross_target_calibrated_cp is standard split CP with a labeled target-domain calibration set.",
            "cross_target_adapted_cp fits a target-domain point adapter on k_adapter target labels, then conformalizes on disjoint k_target target labels.",
            "residual_mean is the default scientific adapter; linear is available via --adapter-types as a sensitivity check.",
            "Very small calibration sets can produce infinite exact finite-sample intervals; finite_q records this explicitly.",
            "Coverage Wilson intervals are 95% Wilson score confidence intervals for empirical coverage.",
            "Size-stratified coverage splits each evaluation set into shorter-lived and longer-lived halves by observed target lifetime.",
        ],
        "summary": json_clean(summary_df.to_dict(orient="records")),
    }
    with json_path.open("w") as f:
        json.dump(json_clean(payload), f, indent=2, allow_nan=False)

    print(f"\n[save] {detailed_path}")
    print(f"[save] {summary_path}")
    print(f"[save] {json_path}")
    if not summary_df.empty:
        show_cols = [
            "scenario",
            "source",
            "target",
            "calibration_domain",
            "model",
            "n_cycles",
            "adapter_type",
            "k_adapter",
            "k_target",
            "confidence_level",
            "coverage_mean",
            "coverage_wilson95_lower_mean",
            "coverage_wilson95_upper_mean",
            "coverage_short_life_mean",
            "coverage_long_life_mean",
            "median_width_mean",
            "winkler_mean_mean",
            "finite_q_mean",
            "n_runs",
        ]
        show_cols = [c for c in show_cols if c in summary_df.columns]
        print("\n=== CONFORMAL SUMMARY ===")
        print(
            summary_df[show_cols]
            .sort_values(["scenario", "source", "target", "model", "k_adapter", "k_target"])
            .to_string(index=False)
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())

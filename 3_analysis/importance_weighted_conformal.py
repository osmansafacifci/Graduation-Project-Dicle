"""
Importance-weighted conformal prediction under covariate shift.

This is a falsifier for the conditional-shift story. Weighted CP is the
standard covariate-shift repair when P(Y|X) is assumed invariant and only P(X)
changes. If source-calibrated weighted CP still fails, or only achieves
coverage through low-ESS / infinite intervals, the result supports the claim
that target-side conditional calibration is needed.

The density ratio w(x) = p_target(x) / p_source(x) is estimated by a
cross-fitted logistic dataset discriminator using cell-level folds. The script
reports effective sample size (ESS) of the source calibration weights:

    ESS = (sum_i w_i)^2 / sum_i w_i^2

alongside coverage and interval width.

Inputs:
    data/intermediate/features_sop12_combined.csv
    splits/sop_v2/{matr,hust}_{seed}.json

Outputs:
    outputs/results_v2_importance_weighted_cp/results_detailed.csv
    outputs/results_v2_importance_weighted_cp/results_summary.csv
    outputs/results_v2_importance_weighted_cp/results_importance_weighted_cp.json
    outputs/results_v2_importance_weighted_cp/paper_iwcp_comparison.csv
    outputs/results_v2_importance_weighted_cp/paper_iwcp_comparison_90.png

Usage:
    python 3_analysis/importance_weighted_conformal.py
    python 3_analysis/importance_weighted_conformal.py --models catboost
    python 3_analysis/importance_weighted_conformal.py --clip-values 5 10 20 inf
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import warnings
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from plot_style import apply_science_style

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(PROJECT_ROOT / "2_models"))
sys.path.insert(0, str(PROJECT_ROOT))
from shared.constants import META_COLS, SEEDS  # noqa: E402
from shared.battery_utils import dataset_window, load_split as _load_split  # noqa: E402
from metrics_utils import compute_metrics  # noqa: E402
sys.path.insert(0, str(HERE))
from conformal_prediction import (  # noqa: E402
    DEFAULT_MODELS,
    DEFAULT_WINDOWS,
    fit_predictor,
    predict_cycles,
    finite_sample_quantile,
    winkler_score,
)

warnings.filterwarnings("ignore", category=ConvergenceWarning)

FEATURES_PATH = PROJECT_ROOT / "data" / "intermediate" / "features_sop12_combined.csv"
SPLITS_DIR = PROJECT_ROOT / "splits" / "sop_v2"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "results_v2_importance_weighted_cp"
DEFAULT_CONFORMAL_DIR = PROJECT_ROOT / "outputs" / "results_v2_conformal"
DEFAULT_CONFIDENCE_LEVELS = [0.90, 0.95]
DEFAULT_CLIPS = ["5", "10", "20", "inf"]


def parse_clip(value: str) -> float:
    text = str(value).strip().lower()
    if text in {"inf", "infinite", "none", "unclipped"}:
        return float("inf")
    out = float(text)
    if out <= 0:
        raise argparse.ArgumentTypeError("clip values must be positive or 'inf'")
    return out


def clip_label(value: float) -> str:
    return "inf" if math.isinf(value) else str(int(value) if float(value).is_integer() else value)


def wilson_interval(successes: int, n: int, confidence_level: float = 0.95) -> tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    z = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
    phat = successes / n
    denom = 1.0 + (z**2 / n)
    center = (phat + (z**2 / (2.0 * n))) / denom
    half = z * math.sqrt(phat * (1.0 - phat) / n + z**2 / (4.0 * n**2)) / denom
    return float(max(0.0, center - half)), float(min(1.0, center + half))


def coverage_row(covered: np.ndarray) -> dict[str, float | int]:
    n = int(len(covered))
    count = int(np.sum(covered)) if n else 0
    lo, hi = wilson_interval(count, n)
    return {
        "covered_count": count,
        "coverage": float(count / n) if n else float("nan"),
        "coverage_wilson95_lower": lo,
        "coverage_wilson95_upper": hi,
    }


def load_split(dataset: str, seed: int) -> dict:
    return _load_split(SPLITS_DIR, dataset, seed)


def split_frames(sub: pd.DataFrame, split: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = sub[sub["cell_id"].isin(split["train"])].copy()
    cal = sub[sub["cell_id"].isin(split["calibration"])].copy()
    test = sub[sub["cell_id"].isin(split["test"])].copy()
    return train, cal, test


def cross_fitted_density_ratio(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    feature_cols: list[str],
    *,
    seed: int,
    n_splits: int,
) -> tuple[pd.Series, pd.Series, float]:
    source = source_df.copy()
    target = target_df.copy()
    source["_domain"] = 0
    target["_domain"] = 1
    pooled = pd.concat([source, target], axis=0)
    x = pooled[feature_cols].to_numpy(dtype=float)
    y = pooled["_domain"].to_numpy(dtype=int)

    min_class = int(min(np.sum(y == 0), np.sum(y == 1)))
    folds = max(2, min(n_splits, min_class))
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    proba = np.full(len(y), np.nan, dtype=float)
    for train_idx, test_idx in cv.split(x, y):
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000),
        )
        model.fit(x[train_idx], y[train_idx])
        proba[test_idx] = model.predict_proba(x[test_idx])[:, 1]

    proba = np.clip(np.nan_to_num(proba, nan=float(np.mean(y))), 1e-6, 1.0 - 1e-6)
    odds = proba / (1.0 - proba)
    prior_source = float(np.mean(y == 0))
    prior_target = float(np.mean(y == 1))
    ratios = odds * (prior_source / prior_target)

    try:
        auc = float(roc_auc_score(y, proba))
    except ValueError:
        auc = float("nan")

    ratio_series = pd.Series(ratios, index=pooled.index)
    return ratio_series.loc[source_df.index], ratio_series.loc[target_df.index], auc


def apply_ratio_clip(weights: np.ndarray, clip: float) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    weights = np.clip(np.nan_to_num(weights, nan=1.0, posinf=1e12, neginf=1e-12), 1e-12, 1e12)
    if math.isinf(clip):
        return weights
    return np.clip(weights, 1.0 / clip, clip)


def effective_sample_size(weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=float)
    denom = float(np.sum(weights**2))
    if denom <= 1e-12:
        return 0.0
    return float(np.sum(weights) ** 2 / denom)


def weighted_quantile_with_test_mass(
    scores: np.ndarray,
    cal_weights: np.ndarray,
    test_weight: float,
    confidence_level: float,
) -> tuple[float, bool]:
    scores = np.asarray(scores, dtype=float)
    cal_weights = np.asarray(cal_weights, dtype=float)
    order = np.argsort(scores)
    scores_sorted = scores[order]
    weights_sorted = cal_weights[order]
    total_cal_weight = float(np.sum(weights_sorted))
    threshold = confidence_level * (total_cal_weight + float(test_weight))
    if threshold > total_cal_weight:
        return float("inf"), False
    cum = np.cumsum(weights_sorted)
    idx = int(np.searchsorted(cum, threshold, side="left"))
    idx = min(idx, len(scores_sorted) - 1)
    return float(scores_sorted[idx]), True


def evaluate_weighted_intervals(
    *,
    source: str,
    target: str,
    model_name: str,
    n_cycles: int,
    seed: int,
    confidence_level: float,
    clip: float,
    train_df: pd.DataFrame,
    cal_df: pd.DataFrame,
    test_df: pd.DataFrame,
    y_cal: np.ndarray,
    y_test: np.ndarray,
    pred_cal: np.ndarray,
    pred_test: np.ndarray,
    cal_weights_raw: np.ndarray,
    test_weights_raw: np.ndarray,
    discriminator_auc: float,
) -> dict:
    scores = np.abs(y_cal - pred_cal)
    cal_weights = apply_ratio_clip(cal_weights_raw, clip)
    test_weights = apply_ratio_clip(test_weights_raw, clip)
    cal_weight_sum = float(np.sum(cal_weights))
    target_mass_fraction = test_weights / (cal_weight_sum + test_weights)

    q_values = []
    finite = []
    for w_test in test_weights:
        q, is_finite = weighted_quantile_with_test_mass(scores, cal_weights, w_test, confidence_level)
        q_values.append(q)
        finite.append(is_finite)
    q_values = np.asarray(q_values, dtype=float)
    finite = np.asarray(finite, dtype=bool)
    lower = pred_test - q_values
    upper = pred_test + q_values
    lower[~finite] = -np.inf
    upper[~finite] = np.inf

    covered = (y_test >= lower) & (y_test <= upper)
    width = upper - lower
    finite_width = width[np.isfinite(width)]
    alpha = 1.0 - confidence_level
    wis = winkler_score(y_test, lower, upper, alpha)
    point = compute_metrics(y_test, pred_test)
    cov = coverage_row(covered)
    ess = effective_sample_size(cal_weights)
    raw_ess = effective_sample_size(cal_weights_raw)

    row = {
        "scenario": "importance_weighted_source_cp",
        "source": source,
        "target": target,
        "model": model_name,
        "n_cycles": int(n_cycles),
        "seed": int(seed),
        "confidence_level": float(confidence_level),
        "clip": clip_label(clip),
        "n_train": int(len(train_df)),
        "n_calibration": int(len(cal_df)),
        "n_test": int(len(test_df)),
        "discriminator_auc": discriminator_auc,
        "cal_weight_ess": ess,
        "cal_weight_ess_fraction": float(ess / max(len(cal_weights), 1)),
        "raw_cal_weight_ess": raw_ess,
        "raw_cal_weight_ess_fraction": float(raw_ess / max(len(cal_weights_raw), 1)),
        "cal_weight_sum": cal_weight_sum,
        "cal_weight_min": float(np.min(cal_weights)),
        "cal_weight_median": float(np.median(cal_weights)),
        "cal_weight_max": float(np.max(cal_weights)),
        "target_weight_median": float(np.median(test_weights)),
        "target_weight_max": float(np.max(test_weights)),
        "target_mass_fraction_median": float(np.median(target_mass_fraction)),
        "target_mass_fraction_max": float(np.max(target_mass_fraction)),
        "finite_interval_fraction": float(np.mean(finite)) if len(finite) else float("nan"),
        "n_infinite_intervals": int(np.sum(~finite)),
        "q_hat_median": float(np.median(q_values[np.isfinite(q_values)])) if np.any(np.isfinite(q_values)) else float("inf"),
        "mean_width": float(np.mean(width)) if np.all(np.isfinite(width)) else float("inf"),
        "median_width": float(np.median(width)) if np.all(np.isfinite(width)) else float("inf"),
        "median_finite_width": float(np.median(finite_width)) if len(finite_width) else float("inf"),
        "winkler_mean": float(np.mean(wis)) if np.all(np.isfinite(wis)) else float("inf"),
        "MAE": point["MAE"],
        "SMAPE": point["SMAPE"],
        "R2": point["R2"],
    }
    row.update(cov)
    return row


def evaluate_unweighted_source_cp(
    *,
    source: str,
    target: str,
    model_name: str,
    n_cycles: int,
    seed: int,
    confidence_level: float,
    train_df: pd.DataFrame,
    cal_df: pd.DataFrame,
    test_df: pd.DataFrame,
    y_cal: np.ndarray,
    y_test: np.ndarray,
    pred_cal: np.ndarray,
    pred_test: np.ndarray,
) -> dict:
    q_hat, finite_q, _ = finite_sample_quantile(np.abs(y_cal - pred_cal), 1.0 - confidence_level)
    lower = pred_test - q_hat
    upper = pred_test + q_hat
    if not finite_q:
        lower = np.full_like(y_test, -np.inf, dtype=float)
        upper = np.full_like(y_test, np.inf, dtype=float)

    covered = (y_test >= lower) & (y_test <= upper)
    width = upper - lower
    alpha = 1.0 - confidence_level
    wis = winkler_score(y_test, lower, upper, alpha)
    point = compute_metrics(y_test, pred_test)
    cov = coverage_row(covered)
    row = {
        "scenario": "unweighted_source_cp",
        "source": source,
        "target": target,
        "model": model_name,
        "n_cycles": int(n_cycles),
        "seed": int(seed),
        "confidence_level": float(confidence_level),
        "clip": "unweighted",
        "n_train": int(len(train_df)),
        "n_calibration": int(len(cal_df)),
        "n_test": int(len(test_df)),
        "discriminator_auc": float("nan"),
        "cal_weight_ess": float(len(cal_df)),
        "cal_weight_ess_fraction": 1.0,
        "raw_cal_weight_ess": float(len(cal_df)),
        "raw_cal_weight_ess_fraction": 1.0,
        "cal_weight_sum": float(len(cal_df)),
        "cal_weight_min": 1.0,
        "cal_weight_median": 1.0,
        "cal_weight_max": 1.0,
        "target_weight_median": 1.0,
        "target_weight_max": 1.0,
        "target_mass_fraction_median": float(1.0 / (len(cal_df) + 1.0)),
        "target_mass_fraction_max": float(1.0 / (len(cal_df) + 1.0)),
        "finite_interval_fraction": 1.0 if finite_q else 0.0,
        "n_infinite_intervals": 0 if finite_q else int(len(test_df)),
        "q_hat_median": float(q_hat),
        "mean_width": float(np.mean(width)) if np.all(np.isfinite(width)) else float("inf"),
        "median_width": float(np.median(width)) if np.all(np.isfinite(width)) else float("inf"),
        "median_finite_width": float(np.median(width[np.isfinite(width)])) if np.any(np.isfinite(width)) else float("inf"),
        "winkler_mean": float(np.mean(wis)) if np.all(np.isfinite(wis)) else float("inf"),
        "MAE": point["MAE"],
        "SMAPE": point["SMAPE"],
        "R2": point["R2"],
    }
    row.update(cov)
    return row


def summarize(detailed: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["scenario", "source", "target", "model", "n_cycles", "confidence_level", "clip"]
    numeric = [
        col for col in detailed.select_dtypes(include=[np.number]).columns
        if col not in {"seed"}
    ]
    return (
        detailed.groupby(group_cols, dropna=False)[numeric]
        .agg(["mean", "std"])
        .reset_index()
        .pipe(lambda df: df.set_axis([
            "_".join([str(x) for x in col if str(x)])
            if isinstance(col, tuple) else str(col)
            for col in df.columns
        ], axis=1))
    )


def _as_float(value, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out


def build_paper_comparison(
    iw_summary: pd.DataFrame,
    *,
    conformal_dir: Path,
    output_dir: Path,
    weighted_clip: str = "10",
    k_target: int = 20,
) -> pd.DataFrame:
    rows: list[dict] = []

    def append_iw_rows(mask: pd.Series, protocol: str, protocol_order: int) -> None:
        for _, row in iw_summary[mask].iterrows():
            rows.append(
                {
                    "protocol_order": protocol_order,
                    "protocol": protocol,
                    "source": row["source"],
                    "target": row["target"],
                    "direction": f"{str(row['source']).upper()} -> {str(row['target']).upper()}",
                    "model": row["model"],
                    "n_cycles": int(row["n_cycles"]),
                    "confidence_level": _as_float(row["confidence_level"]),
                    "coverage_mean": _as_float(row.get("coverage_mean")),
                    "coverage_std": _as_float(row.get("coverage_std")),
                    "median_width_mean": _as_float(row.get("median_width_mean")),
                    "median_finite_width_mean": _as_float(row.get("median_finite_width_mean")),
                    "finite_interval_fraction_mean": _as_float(row.get("finite_interval_fraction_mean")),
                    "discriminator_auc_mean": _as_float(row.get("discriminator_auc_mean")),
                    "cal_weight_ess_fraction_mean": _as_float(row.get("cal_weight_ess_fraction_mean")),
                    "raw_cal_weight_ess_fraction_mean": _as_float(row.get("raw_cal_weight_ess_fraction_mean")),
                    "target_mass_fraction_median_mean": _as_float(row.get("target_mass_fraction_median_mean")),
                    "source_file": "importance_weighted_conformal",
                }
            )

    append_iw_rows(iw_summary["scenario"].eq("unweighted_source_cp"), "Unweighted source CP", 1)
    append_iw_rows(
        iw_summary["scenario"].eq("importance_weighted_source_cp") & iw_summary["clip"].astype(str).eq(str(weighted_clip)),
        f"Importance-weighted source CP (clip={weighted_clip})",
        2,
    )

    conformal_summary_path = conformal_dir / "results_summary.csv"
    if conformal_summary_path.exists():
        cp = pd.read_csv(conformal_summary_path)
        target = cp[
            cp["scenario"].eq("cross_target_adapted_cp")
            & cp["adapter_type"].eq("residual_mean")
            & cp["k_adapter"].fillna(-1).astype(float).eq(float(k_target))
            & cp["k_target"].fillna(-1).astype(float).eq(float(k_target))
        ].copy()
        for _, row in target.iterrows():
            rows.append(
                {
                    "protocol_order": 3,
                    "protocol": f"Target-adapted CP, residual mean (k={k_target})",
                    "source": row["source"],
                    "target": row["target"],
                    "direction": f"{str(row['source']).upper()} -> {str(row['target']).upper()}",
                    "model": row["model"],
                    "n_cycles": int(row["n_cycles"]),
                    "confidence_level": _as_float(row["confidence_level"]),
                    "coverage_mean": _as_float(row.get("coverage_mean")),
                    "coverage_std": _as_float(row.get("coverage_std")),
                    "median_width_mean": _as_float(row.get("median_width_mean")),
                    "median_finite_width_mean": _as_float(row.get("median_width_mean")),
                    "finite_interval_fraction_mean": _as_float(row.get("finite_q_mean")),
                    "discriminator_auc_mean": float("nan"),
                    "cal_weight_ess_fraction_mean": float("nan"),
                    "raw_cal_weight_ess_fraction_mean": float("nan"),
                    "target_mass_fraction_median_mean": float("nan"),
                    "source_file": "conformal_prediction",
                }
            )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(
            ["confidence_level", "direction", "model", "protocol_order"],
            ignore_index=True,
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_dir / "paper_iwcp_comparison.csv", index=False)
    return out


def write_paper_comparison_plot(comparison: pd.DataFrame, output_dir: Path, *, confidence_level: float = 0.90) -> Path | None:
    if comparison.empty:
        return None
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.info("matplotlib not available; skipping comparison plot")
        return None

    apply_science_style()
    plot_df = comparison[np.isclose(comparison["confidence_level"].astype(float), confidence_level)].copy()
    if plot_df.empty:
        return None

    panels = []
    for (direction, model), sub in plot_df.groupby(["direction", "model"], sort=True):
        panels.append((direction, model, sub.sort_values("protocol_order")))
    if not panels:
        return None

    fig, axes = plt.subplots(len(panels), 3, figsize=(13.5, 3.4 * len(panels)), squeeze=False)
    colors = ["#4c78a8", "#f58518", "#54a24b"]
    label_map = {
        "Unweighted source CP": "Source\nCP",
        "Importance-weighted source CP (clip=10)": "Weighted\nsource CP\nclip=10",
        "Target-adapted CP, residual mean (k=20)": "Target-adapted\nCP\nk=20",
    }

    def pct_label(value: float) -> str:
        if not np.isfinite(value):
            return "NA"
        if value <= 0.001:
            return "0%"
        if value < 0.01:
            return f"{100 * value:.1f}%"
        return f"{100 * value:.0f}%"

    for row_idx, (direction, model, sub) in enumerate(panels):
        labels = [label_map.get(protocol, str(protocol).replace(", ", "\n")) for protocol in sub["protocol"]]
        x = np.arange(len(sub))

        ax = axes[row_idx][0]
        ax.bar(x, sub["coverage_mean"], color=colors[: len(sub)])
        ax.axhline(confidence_level, color="#333333", linestyle="--", linewidth=1.0)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Coverage")
        ax.set_title(f"{direction}, {model}")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0, fontsize=8)

        ax = axes[row_idx][1]
        finite_fraction = sub["finite_interval_fraction_mean"].astype(float).to_numpy()
        ax.bar(x, finite_fraction, color=colors[: len(sub)])
        for xi, value in zip(x, finite_fraction, strict=False):
            if value >= 0.95:
                ax.text(
                    xi,
                    0.965,
                    pct_label(float(value)),
                    ha="center",
                    va="top",
                    fontsize=8,
                    color="white",
                    fontweight="bold",
                )
            else:
                label_y = max(float(value) + 0.035, 0.055)
                ax.text(xi, label_y, pct_label(float(value)), ha="center", va="bottom", fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Finite interval fraction")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0, fontsize=8)

        ax = axes[row_idx][2]
        widths = sub["median_width_mean"].astype(float).to_numpy()
        finite_widths = widths[np.isfinite(widths)]
        cap = float(np.nanmax(finite_widths)) * 1.15 if len(finite_widths) else 1.0
        heights = np.where(np.isfinite(widths), widths, cap)
        ax.bar(x, heights, color=colors[: len(sub)])
        for xi, raw, height in zip(x, widths, heights, strict=False):
            if not np.isfinite(raw):
                ax.text(xi, height, "inf", ha="center", va="bottom", fontsize=9)
        ax.set_ylabel("Median width (cycles)")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0, fontsize=8)
        ax.set_ylim(0, max(1.0, float(np.nanmax(heights)) * 1.18))

    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"paper_iwcp_comparison_{int(round(confidence_level * 100))}.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def json_clean(value):
    if isinstance(value, dict):
        return {k: json_clean(v) for k, v in value.items()}
    if isinstance(value, list):
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
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--windows", type=int, nargs="+", default=DEFAULT_WINDOWS)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--confidence-levels", type=float, nargs="+", default=DEFAULT_CONFIDENCE_LEVELS)
    parser.add_argument("--clip-values", nargs="+", default=DEFAULT_CLIPS)
    parser.add_argument("--features-from", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--conformal-dir", type=Path, default=DEFAULT_CONFORMAL_DIR)
    parser.add_argument("--comparison-clip", default="10")
    parser.add_argument("--comparison-k-target", type=int, default=20)
    parser.add_argument("--log-target", action="store_true", default=True)
    parser.add_argument("--no-log-target", action="store_false", dest="log_target")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not FEATURES_PATH.exists():
        print(f"[error] missing {FEATURES_PATH}")
        return 1
    df = pd.read_csv(FEATURES_PATH)
    feature_cols = [c for c in df.columns if c not in META_COLS]
    if args.features_from is not None and args.features_from.exists():
        feature_cols = [line.strip() for line in args.features_from.read_text().splitlines() if line.strip()]
    clips = [parse_clip(v) for v in args.clip_values]

    print(
        f"[setup] n_features={len(feature_cols)}, models={args.models}, "
        f"windows={args.windows}, seeds={args.seeds}, clips={[clip_label(c) for c in clips]}"
    )

    rows = []
    for n_cycles in args.windows:
        for source, target in [("matr", "hust"), ("hust", "matr")]:
            src = dataset_window(df, source, n_cycles)
            tgt = dataset_window(df, target, n_cycles)
            source_weight_pool = src.copy()
            for seed in args.seeds:
                split = load_split(source, seed)
                train_df, cal_df, _ = split_frames(src, split)
                test_df = tgt.copy()
                source_ratios, target_ratios, discriminator_auc = cross_fitted_density_ratio(
                    source_weight_pool,
                    test_df,
                    feature_cols,
                    seed=seed,
                    n_splits=5,
                )
                cal_weights_raw = source_ratios.loc[cal_df.index].to_numpy(dtype=float)
                test_weights_raw = target_ratios.loc[test_df.index].to_numpy(dtype=float)
                for model_name in args.models:
                    print(f"  {source}->{target} seed={seed} model={model_name}")
                    predictor = fit_predictor(
                        train_df,
                        feature_cols,
                        model_name=model_name,
                        seed=seed,
                        log_target=args.log_target,
                    )
                    pred_cal = predict_cycles(predictor, cal_df)
                    pred_test = predict_cycles(predictor, test_df)
                    y_cal = cal_df["cycle_life"].to_numpy(dtype=float)
                    y_test = test_df["cycle_life"].to_numpy(dtype=float)
                    for confidence_level in args.confidence_levels:
                        rows.append(
                            evaluate_unweighted_source_cp(
                                source=source,
                                target=target,
                                model_name=model_name,
                                n_cycles=n_cycles,
                                seed=seed,
                                confidence_level=confidence_level,
                                train_df=train_df,
                                cal_df=cal_df,
                                test_df=test_df,
                                y_cal=y_cal,
                                y_test=y_test,
                                pred_cal=pred_cal,
                                pred_test=pred_test,
                            )
                        )
                        for clip in clips:
                            rows.append(
                                evaluate_weighted_intervals(
                                    source=source,
                                    target=target,
                                    model_name=model_name,
                                    n_cycles=n_cycles,
                                    seed=seed,
                                    confidence_level=confidence_level,
                                    clip=clip,
                                    train_df=train_df,
                                    cal_df=cal_df,
                                    test_df=test_df,
                                    y_cal=y_cal,
                                    y_test=y_test,
                                    pred_cal=pred_cal,
                                    pred_test=pred_test,
                                    cal_weights_raw=cal_weights_raw,
                                    test_weights_raw=test_weights_raw,
                                    discriminator_auc=discriminator_auc,
                                )
                            )

    detailed = pd.DataFrame(rows)
    summary = summarize(detailed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_detailed = args.output_dir / "results_detailed.csv"
    out_summary = args.output_dir / "results_summary.csv"
    out_json = args.output_dir / "results_importance_weighted_cp.json"

    detailed.to_csv(out_detailed, index=False)
    summary.to_csv(out_summary, index=False)
    comparison = build_paper_comparison(
        summary,
        conformal_dir=args.conformal_dir,
        output_dir=args.output_dir,
        weighted_clip=str(args.comparison_clip),
        k_target=args.comparison_k_target,
    )
    comparison_plot = write_paper_comparison_plot(comparison, args.output_dir, confidence_level=0.90)
    payload = {
        "protocol": "importance_weighted_conformal_v1",
        "features": feature_cols,
        "models": args.models,
        "windows": args.windows,
        "seeds": args.seeds,
        "confidence_levels": args.confidence_levels,
        "clip_values": [clip_label(c) for c in clips],
        "paper_comparison_rows": comparison.to_dict(orient="records"),
        "summary": summary.to_dict(orient="records"),
    }
    out_json.write_text(json.dumps(json_clean(payload), indent=2))

    print(f"[save] {out_detailed}")
    print(f"[save] {out_summary}")
    print(f"[save] {out_json}")
    print(f"[save] {args.output_dir / 'paper_iwcp_comparison.csv'}")
    if comparison_plot is not None:
        print(f"[save] {comparison_plot}")
    print("\n=== IMPORTANCE-WEIGHTED CP SUMMARY ===")
    cols = [
        "scenario",
        "source",
        "target",
        "model",
        "confidence_level",
        "clip",
        "coverage_mean",
        "median_width_mean",
        "finite_interval_fraction_mean",
        "cal_weight_ess_fraction_mean",
        "raw_cal_weight_ess_fraction_mean",
        "target_mass_fraction_median_mean",
        "discriminator_auc_mean",
    ]
    cols = [c for c in cols if c in summary.columns]
    print(summary[cols].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

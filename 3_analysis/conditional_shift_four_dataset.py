"""
Four-dataset conditional-shift diagnostics for the paper extension.

This extends the MATR<->HUST conditional-shift decomposition to MATR, HUST,
Sandia, and Luh/KIT. It uses the committed four-dataset feature table and
splits, then writes compact reviewer-facing artifacts:

1. Pairwise centered-log per-feature slope tests.
2. Source->target alpha/beta diagnostics using the naive-best cross model for
   each direction.
3. Direction summaries that classify rank-signal preservation and calibration
   behavior.

Usage:
    python 3_analysis/conditional_shift_four_dataset.py
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import HuberRegressor, TheilSenRegressor

from plot_style import apply_science_style

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(PROJECT_ROOT / "2_models"))
sys.path.insert(0, str(PROJECT_ROOT))
from shared.constants import META_COLS, SEEDS  # noqa: E402
from shared.battery_utils import (  # noqa: E402
    display_path as _display_path,
    load_split,
    safe_pred,
)
from metrics_utils import compute_metrics, fit_with_threaded_joblib, to_cycles  # noqa: E402
from run_experiments import (  # noqa: E402
    FITTERS,
    fit_catboost,
    fit_elastic_net,
    fit_gaussian_process,
    fit_pls,
    fit_random_forest,
    fit_stacking,
    fit_xgboost,
)

warnings.filterwarnings("ignore", category=ConvergenceWarning)

INTERMEDIATE_DIR = PROJECT_ROOT / "data" / "intermediate"
DEFAULT_FEATURES = INTERMEDIATE_DIR / "features_sop12_four_dataset_capnorm.csv"
DEFAULT_SPLITS = PROJECT_ROOT / "splits" / "sop_v2_four_dataset"
DEFAULT_CROSS_SUMMARY = PROJECT_ROOT / "outputs" / "results_v2_four_dataset_cross_34feat_capnorm_log" / "results_summary.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "results_v2_four_dataset_conditional_shift"

DATASETS = ["matr", "hust", "sandia", "luh"]
ALL_MODELS = ["elastic_net", "pls", "random_forest", "xgboost", "catboost", "gaussian_process", "stacking"]
def display_path(path: Path) -> str:
    return _display_path(path, PROJECT_ROOT)


def bh_fdr(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    q = np.full_like(p, np.nan)
    valid = np.isfinite(p)
    if not valid.any():
        return q
    pv = p[valid]
    order = np.argsort(pv)
    ranked = pv[order]
    m = len(ranked)
    adj = ranked * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out = np.empty_like(pv)
    out[order] = np.clip(adj, 0.0, 1.0)
    q[valid] = out
    return q


def bootstrap_p_value(samples: np.ndarray, null: float = 0.0) -> float:
    samples = np.asarray(samples, dtype=float)
    samples = samples[np.isfinite(samples)]
    if len(samples) == 0:
        return float("nan")
    centered = samples - null
    p = 2.0 * min(float(np.mean(centered <= 0.0)), float(np.mean(centered >= 0.0)))
    return float(min(1.0, max(p, 1.0 / (len(samples) + 1))))


def within_dataset_zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    std = float(np.std(values, ddof=0))
    if std < 1e-12:
        return np.zeros_like(values, dtype=float)
    return (values - float(np.mean(values))) / std


def centered_log_life(y: np.ndarray) -> tuple[np.ndarray, float]:
    y_log = np.log(np.asarray(y, dtype=float))
    mean_log = float(np.mean(y_log))
    return y_log - mean_log, mean_log


def univariate_slope(x: np.ndarray, y_centered_log: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y_centered_log = np.asarray(y_centered_log, dtype=float)
    if len(x) < 2 or np.std(x) < 1e-12:
        return 0.0
    slope, _ = np.polyfit(x, y_centered_log, 1)
    return float(slope)


def feature_slope_pairwise(
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    n_cycles: int,
    n_boot: int,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(random_seed)
    prepared: dict[str, dict[str, np.ndarray | float]] = {}
    for dataset in DATASETS:
        sub = df[(df["dataset"] == dataset) & (df["n_cycles"] == n_cycles) & (df["is_censored"] == 0)].copy()
        y = sub["cycle_life"].to_numpy(dtype=float)
        y_centered, mean_log = centered_log_life(y)
        prepared[dataset] = {
            "frame": sub,
            "y": y,
            "y_centered": y_centered,
            "mean_log_life": mean_log,
        }

    rows = []
    for dataset_a, dataset_b in combinations(DATASETS, 2):
        a = prepared[dataset_a]
        b = prepared[dataset_b]
        idx_a = np.arange(len(a["y"]))
        idx_b = np.arange(len(b["y"]))
        pair_rows = []

        for feature in feature_cols:
            x_a = within_dataset_zscore(a["frame"][feature].to_numpy(dtype=float))
            x_b = within_dataset_zscore(b["frame"][feature].to_numpy(dtype=float))
            slope_a = univariate_slope(x_a, a["y_centered"])
            slope_b = univariate_slope(x_b, b["y_centered"])
            boot = []
            for _ in range(n_boot):
                ba = rng.choice(idx_a, size=len(idx_a), replace=True)
                bb = rng.choice(idx_b, size=len(idx_b), replace=True)
                yca, _ = centered_log_life(a["y"][ba])
                ycb, _ = centered_log_life(b["y"][bb])
                boot.append(univariate_slope(x_b[bb], ycb) - univariate_slope(x_a[ba], yca))
            boot = np.asarray(boot, dtype=float)
            pair_rows.append(
                {
                    "pair": f"{dataset_a}_vs_{dataset_b}",
                    "dataset_a": dataset_a,
                    "dataset_b": dataset_b,
                    "feature": feature,
                    "n_cycles": int(n_cycles),
                    "n_a": int(len(idx_a)),
                    "n_b": int(len(idx_b)),
                    "mean_log_life_a": float(a["mean_log_life"]),
                    "mean_log_life_b": float(b["mean_log_life"]),
                    "log_life_offset_b_minus_a": float(b["mean_log_life"] - a["mean_log_life"]),
                    "life_ratio_b_over_a": float(np.exp(b["mean_log_life"] - a["mean_log_life"])),
                    "slope_a": slope_a,
                    "slope_b": slope_b,
                    "delta_slope_b_minus_a": slope_b - slope_a,
                    "delta_slope_ci95_low": float(np.percentile(boot, 2.5)),
                    "delta_slope_ci95_high": float(np.percentile(boot, 97.5)),
                    "delta_slope_boot_p": bootstrap_p_value(boot),
                }
            )

        pair_df = pd.DataFrame(pair_rows)
        pair_df["delta_slope_fdr_bh"] = bh_fdr(pair_df["delta_slope_boot_p"].to_numpy(dtype=float))
        pair_df["slope_shift_significant"] = (
            pair_df["delta_slope_fdr_bh"].lt(0.05)
            & ((pair_df["delta_slope_ci95_low"] > 0.0) | (pair_df["delta_slope_ci95_high"] < 0.0))
        )
        pair_df["shift_class"] = np.where(pair_df["slope_shift_significant"], "slope_shifted", "slope_stable")
        rows.append(pair_df)

    feature_rows = pd.concat(rows, ignore_index=True)
    pair_summary = (
        feature_rows.groupby(["pair", "dataset_a", "dataset_b"], as_index=False)
        .agg(
            n_features=("feature", "count"),
            slope_shifted_features=("slope_shift_significant", "sum"),
            median_abs_delta_slope=("delta_slope_b_minus_a", lambda s: float(np.median(np.abs(s)))),
            log_life_offset_b_minus_a=("log_life_offset_b_minus_a", "first"),
            life_ratio_b_over_a=("life_ratio_b_over_a", "first"),
        )
        .assign(slope_shifted_share=lambda x: x["slope_shifted_features"] / x["n_features"])
        .sort_values(["slope_shifted_share", "pair"], ascending=[False, True])
    )
    return feature_rows, pair_summary


def selected_models(cross_summary_path: Path, *, n_cycles: int) -> dict[str, str]:
    cross = pd.read_csv(cross_summary_path)
    sub = cross[cross["n_cycles"].eq(n_cycles)].copy()
    best = sub.sort_values(["experiment", "R2_mean"], ascending=[True, False]).groupby("experiment", as_index=False).head(1)
    return dict(zip(best["experiment"], best["model"], strict=False))


# safe_pred and load_split imported from shared.battery_utils


def fit_source_predict_target(
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    source: str,
    target: str,
    model_name: str,
    seed: int,
    n_cycles: int,
    splits_dir: Path,
    log_target: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from sklearn.preprocessing import StandardScaler

    src = df[(df["dataset"] == source) & (df["n_cycles"] == n_cycles) & (df["is_censored"] == 0)].copy()
    tgt = df[(df["dataset"] == target) & (df["n_cycles"] == n_cycles) & (df["is_censored"] == 0)].copy()
    split = load_split(splits_dir, source, seed)
    train = src[src["cell_id"].isin(split["train"])].copy()

    x_train = train[feature_cols].to_numpy(dtype=float)
    y_train = train["cycle_life"].to_numpy(dtype=float)
    y_train_fit = np.log(y_train) if log_target else y_train
    x_target = tgt[feature_cols].to_numpy(dtype=float)
    y_target = tgt["cycle_life"].to_numpy(dtype=float)
    target_cell_ids = tgt["cell_id"].to_numpy()

    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_target_s = scaler.transform(x_target)
    x_train_s = np.clip(np.nan_to_num(x_train_s, nan=0.0, posinf=0.0, neginf=0.0), -1e6, 1e6)
    x_target_s = np.clip(np.nan_to_num(x_target_s, nan=0.0, posinf=0.0, neginf=0.0), -1e6, 1e6)

    result = fit_with_threaded_joblib(FITTERS[model_name], x_train_s, y_train_fit, seed=seed)
    model = result[0] if isinstance(result, tuple) else result
    y_pred = to_cycles(safe_pred(model, x_target_s), log_target=log_target)
    return target_cell_ids, y_target, y_pred


def fit_alpha_beta(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    if len(y_true) < 2 or np.std(y_pred) < 1e-12:
        return 1.0, float(np.mean(y_true - y_pred))
    alpha, beta = np.polyfit(y_pred, y_true, 1)
    return float(alpha), float(beta)


def pearson_summary(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    if len(y_true) < 3 or np.std(y_true) < 1e-12 or np.std(y_pred) < 1e-12:
        return float("nan"), float("nan")
    try:
        res = pearsonr(y_pred, y_true)
        return float(res.statistic), float(res.pvalue)
    except Exception:
        return float("nan"), float("nan")


def robust_alpha_beta(y_true: np.ndarray, y_pred: np.ndarray, *, method: str, seed: int) -> tuple[float, float, float]:
    if len(y_true) < 2 or np.std(y_pred) < 1e-12:
        beta = float(np.mean(y_true - y_pred))
        corrected = y_pred + beta
        return 1.0, beta, compute_metrics(y_true, corrected)["R2"]
    x = np.asarray(y_pred, dtype=float).reshape(-1, 1)
    y = np.asarray(y_true, dtype=float)
    try:
        if method == "theil_sen":
            reg = TheilSenRegressor(random_state=seed, max_subpopulation=10000)
        elif method == "huber":
            reg = HuberRegressor(max_iter=1000)
        else:
            raise ValueError(method)
        reg.fit(x, y)
        alpha = float(reg.coef_[0])
        beta = float(reg.intercept_)
        corrected = np.clip(alpha * y_pred + beta, 1.0, 1e9)
        return alpha, beta, compute_metrics(y_true, corrected)["R2"]
    except Exception:
        alpha, beta = fit_alpha_beta(y_true, y_pred)
        corrected = np.clip(alpha * y_pred + beta, 1.0, 1e9)
        return alpha, beta, compute_metrics(y_true, corrected)["R2"]


def classify_rank_signal(r: float) -> str:
    if not np.isfinite(r):
        return "undefined"
    if r >= 0.50:
        return "strong_rank_signal"
    if r >= 0.25:
        return "moderate_rank_signal"
    if r >= 0.10:
        return "weak_rank_signal"
    if r > -0.10:
        return "rank_signal_collapsed"
    return "negative_or_inverted_signal"


def classify_adapter(row: pd.Series) -> str:
    residual_gain = float(row["residual_R2"] - row["raw_R2"])
    linear_gain = float(row["linear_R2"] - row["raw_R2"])
    linear_vs_residual = float(row["linear_R2"] - row["residual_R2"])
    if row["linear_R2"] > 0.25 and linear_vs_residual > 0.05:
        return "linear_recovers_predictive_transfer"
    if row["residual_R2"] >= -0.10 and linear_vs_residual <= 0.05:
        return "offset_dominant_repair"
    if linear_gain > 0.50 or residual_gain > 0.50:
        return "center_repaired_but_low_rank"
    return "limited_repair"


def alpha_beta_diagnostics(
    df: pd.DataFrame,
    feature_cols: list[str],
    model_by_direction: dict[str, str],
    *,
    n_cycles: int,
    seeds: list[int],
    n_boot: int,
    random_seed: int,
    splits_dir: Path,
    log_target: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(random_seed + 177)
    rows = []
    predictions = []

    for source in DATASETS:
        for target in DATASETS:
            if source == target:
                continue
            experiment = f"{source}_to_{target}"
            model_name = model_by_direction[experiment]
            print(f"  {experiment}: {model_name}")
            for split_seed in seeds:
                cell_ids, y_true, y_pred = fit_source_predict_target(
                    df,
                    feature_cols,
                    source=source,
                    target=target,
                    model_name=model_name,
                    seed=split_seed,
                    n_cycles=n_cycles,
                    splits_dir=splits_dir,
                    log_target=log_target,
                )
                alpha, beta = fit_alpha_beta(y_true, y_pred)
                residual_bias = float(np.mean(y_true - y_pred))
                raw = compute_metrics(y_true, y_pred)
                residual = compute_metrics(y_true, np.clip(y_pred + residual_bias, 1.0, 1e9))
                linear = compute_metrics(y_true, np.clip(alpha * y_pred + beta, 1.0, 1e9))
                ts_alpha, ts_beta, ts_r2 = robust_alpha_beta(y_true, y_pred, method="theil_sen", seed=split_seed)
                huber_alpha, huber_beta, huber_r2 = robust_alpha_beta(y_true, y_pred, method="huber", seed=split_seed)
                pearson_r, pearson_p = pearson_summary(y_true, y_pred)

                for cell_id, actual, pred in zip(cell_ids, y_true, y_pred, strict=False):
                    predictions.append(
                        {
                            "experiment": experiment,
                            "source": source,
                            "target": target,
                            "model": model_name,
                            "seed": int(split_seed),
                            "cell_id": cell_id,
                            "cycle_life": float(actual),
                            "source_prediction": float(pred),
                            "residual": float(actual - pred),
                        }
                    )

                boot_alpha, boot_r, boot_constant_share = [], [], []
                indices = np.arange(len(y_true))
                for _ in range(n_boot):
                    b = rng.choice(indices, size=len(indices), replace=True)
                    ba, _ = fit_alpha_beta(y_true[b], y_pred[b])
                    br, _ = pearson_summary(y_true[b], y_pred[b])
                    residuals = y_true[b] - y_pred[b]
                    ss_total = float(np.sum(residuals**2))
                    ss_after_constant = float(np.sum((residuals - np.mean(residuals)) ** 2))
                    share = 1.0 - ss_after_constant / ss_total if ss_total > 1e-12 else float("nan")
                    boot_alpha.append(ba)
                    boot_r.append(br)
                    boot_constant_share.append(share)

                boot_alpha = np.asarray(boot_alpha, dtype=float)
                boot_r = np.asarray(boot_r, dtype=float)
                boot_constant_share = np.asarray(boot_constant_share, dtype=float)
                rows.append(
                    {
                        "experiment": experiment,
                        "source": source,
                        "target": target,
                        "model": model_name,
                        "seed": int(split_seed),
                        "n_cycles": int(n_cycles),
                        "n_target": int(len(y_true)),
                        "alpha": alpha,
                        "alpha_ci95_low": float(np.percentile(boot_alpha, 2.5)),
                        "alpha_ci95_high": float(np.percentile(boot_alpha, 97.5)),
                        "alpha_minus_1_boot_p": bootstrap_p_value(boot_alpha, null=1.0),
                        "beta": beta,
                        "theil_sen_alpha": ts_alpha,
                        "theil_sen_beta": ts_beta,
                        "theil_sen_R2": ts_r2,
                        "huber_alpha": huber_alpha,
                        "huber_beta": huber_beta,
                        "huber_R2": huber_r2,
                        "pearson_r": pearson_r,
                        "pearson_r_ci95_low": float(np.nanpercentile(boot_r, 2.5)),
                        "pearson_r_ci95_high": float(np.nanpercentile(boot_r, 97.5)),
                        "pearson_p": pearson_p,
                        "constant_share_of_ss": float(np.nanmean(boot_constant_share)),
                        "constant_share_ci95_low": float(np.nanpercentile(boot_constant_share, 2.5)),
                        "constant_share_ci95_high": float(np.nanpercentile(boot_constant_share, 97.5)),
                        "raw_MAE": raw["MAE"],
                        "raw_R2": raw["R2"],
                        "residual_MAE": residual["MAE"],
                        "residual_R2": residual["R2"],
                        "linear_MAE": linear["MAE"],
                        "linear_R2": linear["R2"],
                    }
                )

    alpha_rows = pd.DataFrame(rows)
    summary = (
        alpha_rows.groupby(["experiment", "source", "target", "model"], as_index=False)
        .agg(
            n_target=("n_target", "first"),
            alpha=("alpha", "mean"),
            alpha_ci95_low=("alpha_ci95_low", "mean"),
            alpha_ci95_high=("alpha_ci95_high", "mean"),
            theil_sen_alpha=("theil_sen_alpha", "mean"),
            huber_alpha=("huber_alpha", "mean"),
            pearson_r=("pearson_r", "mean"),
            pearson_r_ci95_low=("pearson_r_ci95_low", "mean"),
            pearson_r_ci95_high=("pearson_r_ci95_high", "mean"),
            pearson_p=("pearson_p", "mean"),
            constant_share_of_ss=("constant_share_of_ss", "mean"),
            raw_MAE=("raw_MAE", "mean"),
            raw_R2=("raw_R2", "mean"),
            residual_MAE=("residual_MAE", "mean"),
            residual_R2=("residual_R2", "mean"),
            linear_MAE=("linear_MAE", "mean"),
            linear_R2=("linear_R2", "mean"),
        )
        .sort_values("experiment")
        .reset_index(drop=True)
    )
    summary["rank_signal_class"] = summary["pearson_r"].map(classify_rank_signal)
    summary["adapter_class"] = summary.apply(classify_adapter, axis=1)
    return alpha_rows, pd.DataFrame(predictions), summary


def add_pair_context(direction_summary: pd.DataFrame, pair_summary: pd.DataFrame) -> pd.DataFrame:
    lookup = {}
    for row in pair_summary.to_dict(orient="records"):
        a = row["dataset_a"]
        b = row["dataset_b"]
        lookup[(a, b)] = {
            "pair": row["pair"],
            "slope_shifted_share": row["slope_shifted_share"],
            "n_slope_shifted_features": row["slope_shifted_features"],
            "log_life_offset_target_minus_source": row["log_life_offset_b_minus_a"],
            "life_ratio_target_over_source": row["life_ratio_b_over_a"],
        }
        lookup[(b, a)] = {
            "pair": row["pair"],
            "slope_shifted_share": row["slope_shifted_share"],
            "n_slope_shifted_features": row["slope_shifted_features"],
            "log_life_offset_target_minus_source": -row["log_life_offset_b_minus_a"],
            "life_ratio_target_over_source": 1.0 / row["life_ratio_b_over_a"],
        }
    extras = []
    for _, row in direction_summary.iterrows():
        extras.append(lookup[(row["source"], row["target"])])
    return pd.concat([direction_summary.reset_index(drop=True), pd.DataFrame(extras)], axis=1)


def markdown_table(df: pd.DataFrame, float_digits: int | None = None) -> str:
    formatted = df.copy()
    if float_digits is not None:
        for col in formatted.select_dtypes(include=[np.number]).columns:
            formatted[col] = formatted[col].map(lambda x: "" if pd.isna(x) else f"{x:.{float_digits}f}")
    formatted = formatted.astype(object).where(pd.notna(formatted), "")
    header = "| " + " | ".join(map(str, formatted.columns)) + " |"
    separator = "| " + " | ".join(["---"] * len(formatted.columns)) + " |"
    body = ["| " + " | ".join(map(str, row)) + " |" for row in formatted.to_numpy()]
    return "\n".join([header, separator, *body])


def write_heatmaps(direction_summary: pd.DataFrame, pair_summary: pd.DataFrame, output_dir: Path) -> Path | None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import TwoSlopeNorm
    except Exception:
        return None
    apply_science_style()

    def matrix(column: str) -> np.ndarray:
        out = np.full((len(DATASETS), len(DATASETS)), np.nan)
        for _, row in direction_summary.iterrows():
            i = DATASETS.index(row["source"])
            j = DATASETS.index(row["target"])
            out[i, j] = row[column]
        return out

    mats = [
        ("Naive cross R2", matrix("raw_R2"), "coolwarm", (-2.0, 1.0), None),
        ("Pearson rank signal", matrix("pearson_r"), "coolwarm", (-0.5, 0.8), None),
        ("Linear calibrated R2", matrix("linear_R2"), "coolwarm", (-0.5, 1.0), None),
        ("Slope-shifted feature share", matrix("slope_shifted_share"), "viridis", (0.0, 1.0), None),
        (
            "Life-ratio (target / source)",
            matrix("life_ratio_target_over_source"),
            "RdBu_r",
            None,
            TwoSlopeNorm(vmin=0.25, vcenter=1.0, vmax=4.0),
        ),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 9.2), squeeze=False)
    for ax, (title, arr, cmap, limits, norm) in zip(axes.ravel(), mats, strict=False):
        if norm is None:
            im = ax.imshow(arr, cmap=cmap, vmin=limits[0], vmax=limits[1])
        else:
            im = ax.imshow(arr, cmap=cmap, norm=norm)
        ax.set_title(title)
        ax.set_xticks(range(len(DATASETS)), [d.upper() for d in DATASETS], rotation=35, ha="right")
        ax.set_yticks(range(len(DATASETS)), [d.upper() for d in DATASETS])
        ax.set_xlabel("Target")
        ax.set_ylabel("Source")
        for i in range(len(DATASETS)):
            for j in range(len(DATASETS)):
                if i == j:
                    ax.text(j, i, "-", ha="center", va="center", color="#333333")
                elif np.isfinite(arr[i, j]):
                    ax.text(j, i, f"{arr[i, j]:.2f}", ha="center", va="center", color="black", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for ax in axes.ravel()[len(mats):]:
        ax.axis("off")
    fig.suptitle("Four-Dataset Conditional-Shift Diagnostics", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "four_dataset_conditional_shift_heatmaps.png"
    fig.savefig(out, dpi=200)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    return out


def write_report(pair_summary: pd.DataFrame, direction_summary: pd.DataFrame, out_path: Path) -> None:
    pair_cols = [
        "pair",
        "log_life_offset_b_minus_a",
        "life_ratio_b_over_a",
        "slope_shifted_features",
        "n_features",
        "slope_shifted_share",
    ]
    dir_cols = [
        "experiment",
        "model",
        "raw_R2",
        "pearson_r",
        "rank_signal_class",
        "residual_R2",
        "linear_R2",
        "adapter_class",
        "slope_shifted_share",
        "life_ratio_target_over_source",
    ]
    top_linear = direction_summary.sort_values("linear_R2", ascending=False).head(4)[dir_cols]
    weak = direction_summary.sort_values("pearson_r", ascending=True).head(4)[dir_cols]
    lines = [
        "# Four-Dataset Conditional-Shift Diagnostics",
        "",
        "Feature table: `features_sop12_four_dataset_capnorm.csv`; N=100; censored cells excluded from regression diagnostics.",
        "",
        "## Pairwise Feature-Slope Shift",
        markdown_table(pair_summary[pair_cols], float_digits=3),
        "",
        "## Direction-Level Source-Prediction Diagnostics",
        markdown_table(direction_summary[dir_cols], float_digits=3),
        "",
        "## Strongest Calibrated Directions",
        markdown_table(top_linear, float_digits=3),
        "",
        "## Weakest Rank-Signal Directions",
        markdown_table(weak, float_digits=3),
        "",
        "Interpretation: `pearson_r` is the source model's rank signal on the target dataset. Linear calibration can exploit positive rank signal; residual-mean calibration mostly repairs the target center. Directions with low or negative rank signal are concept/conditional-shift failures even if target calibration reduces MAE.",
    ]
    out_path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features-path", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--splits-dir", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--cross-summary", type=Path, default=DEFAULT_CROSS_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-cycles", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--n-bootstrap", type=int, default=300)
    parser.add_argument("--random-seed", type=int, default=20260511)
    parser.add_argument("--log-target", action="store_true", default=True)
    parser.add_argument("--no-log-target", action="store_false", dest="log_target")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    features_path = args.features_path if args.features_path.is_absolute() else PROJECT_ROOT / args.features_path
    splits_dir = args.splits_dir if args.splits_dir.is_absolute() else PROJECT_ROOT / args.splits_dir
    cross_summary_path = args.cross_summary if args.cross_summary.is_absolute() else PROJECT_ROOT / args.cross_summary
    output_dir = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir

    for path in [features_path, splits_dir, cross_summary_path]:
        if not path.exists():
            print(f"[error] missing {path}")
            return 1

    df = pd.read_csv(features_path)
    feature_cols = [c for c in df.columns if c not in META_COLS]
    print(f"[setup] features={display_path(features_path)}")
    print(f"[setup] splits={display_path(splits_dir)}")
    print(f"[setup] n_features={len(feature_cols)}, n_bootstrap={args.n_bootstrap}, seeds={args.seeds}")

    print("\n========== pairwise centered-log feature slopes ==========")
    feature_rows, pair_summary = feature_slope_pairwise(
        df,
        feature_cols,
        n_cycles=args.n_cycles,
        n_boot=args.n_bootstrap,
        random_seed=args.random_seed,
    )
    out_feature = INTERMEDIATE_DIR / "four_dataset_conditional_shift_feature_slopes.csv"
    out_pair = INTERMEDIATE_DIR / "four_dataset_conditional_shift_pair_summary.csv"
    feature_rows.to_csv(out_feature, index=False)
    pair_summary.to_csv(out_pair, index=False)
    print(f"[save] {display_path(out_feature)}")
    print(f"[save] {display_path(out_pair)}")
    print(pair_summary[["pair", "slope_shifted_features", "n_features", "slope_shifted_share", "life_ratio_b_over_a"]].to_string(index=False))

    print("\n========== direction-level alpha/beta and rank signal ==========")
    model_by_direction = selected_models(cross_summary_path, n_cycles=args.n_cycles)
    alpha_rows, prediction_rows, direction_summary = alpha_beta_diagnostics(
        df,
        feature_cols,
        model_by_direction,
        n_cycles=args.n_cycles,
        seeds=args.seeds,
        n_boot=args.n_bootstrap,
        random_seed=args.random_seed,
        splits_dir=splits_dir,
        log_target=args.log_target,
    )
    direction_summary = add_pair_context(direction_summary, pair_summary)

    out_alpha = INTERMEDIATE_DIR / "four_dataset_conditional_shift_alpha_beta.csv"
    out_predictions = INTERMEDIATE_DIR / "four_dataset_conditional_shift_predictions.csv"
    out_direction = INTERMEDIATE_DIR / "four_dataset_conditional_shift_direction_summary.csv"
    out_report = INTERMEDIATE_DIR / "four_dataset_conditional_shift_report.md"
    alpha_rows.to_csv(out_alpha, index=False)
    prediction_rows.to_csv(out_predictions, index=False)
    direction_summary.to_csv(out_direction, index=False)
    write_report(pair_summary, direction_summary, out_report)
    print(f"[save] {display_path(out_alpha)}")
    print(f"[save] {display_path(out_predictions)}")
    print(f"[save] {display_path(out_direction)}")
    print(f"[save] {display_path(out_report)}")

    heatmap = write_heatmaps(direction_summary, pair_summary, output_dir)
    if heatmap is not None:
        print(f"[save] {display_path(heatmap)}")

    payload = {
        "protocol": "four_dataset_conditional_shift_v1",
        "features_path": display_path(features_path),
        "cross_summary": display_path(cross_summary_path),
        "n_cycles": int(args.n_cycles),
        "n_features": int(len(feature_cols)),
        "n_bootstrap": int(args.n_bootstrap),
        "seeds": [int(s) for s in args.seeds],
        "model_by_direction": model_by_direction,
        "pair_summary": pair_summary.replace({np.nan: None}).to_dict(orient="records"),
        "direction_summary": direction_summary.replace({np.nan: None}).to_dict(orient="records"),
    }
    out_json = INTERMEDIATE_DIR / "four_dataset_conditional_shift_summary.json"
    out_json.write_text(json.dumps(payload, indent=2, allow_nan=False))
    print(f"[save] {display_path(out_json)}")
    print("\n" + out_report.read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
Conditional-shift decomposition for the MATR <-> HUST study.

This script adds two reviewer-facing diagnostics to the existing concept-shift
analysis:

  1. Per-feature slope comparison:
       Within each dataset, z-score every feature, center log(cycle_life), and
       fit
           centered_log_life = intercept + slope * z_feature
       Compare HUST - MATR slope differences with bootstrap confidence
       intervals and Benjamini-Hochberg FDR correction. The universal
       HUST-MATR log-life offset is reported separately so it is not mistaken
       for a feature-specific intercept shift.

  2. Alpha/beta source-prediction calibration:
       For every source -> target direction, fit the source model, predict the
       target cells, then bootstrap
           y_target = alpha * y_source_pred + beta
       If alpha ~= 1, a residual-mean adapter is a scientifically reasonable
       first-order target correction. If alpha differs from 1, the linear
       adapter should be treated as more than a sensitivity check. Pearson r
       is reported alongside alpha to separate real rank-transfer signal from
       noisy slopes around near-zero correlation.

Inputs:
    data/intermediate/features_sop12_combined.csv
    splits/sop_v2/{matr,hust}_{seed}.json

Outputs:
    data/intermediate/conditional_shift_feature_slopes.csv
    data/intermediate/conditional_shift_alpha_beta.csv
    data/intermediate/conditional_shift_alpha_beta_predictions.csv
    data/intermediate/conditional_shift_summary.json
    data/intermediate/conditional_shift_report.txt
    outputs/results_v2_conditional_shift/conditional_shift_alpha_beta_scatter_seed42.png

Usage:
    python 3_analysis/conditional_shift_decomposition.py
    python 3_analysis/conditional_shift_decomposition.py --models catboost
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import HuberRegressor, TheilSenRegressor

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
INTERMEDIATE_DIR = PROJECT_ROOT / "data" / "intermediate"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "results_v2_conditional_shift"

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

FITTERS = {
    "elastic_net": fit_elastic_net,
    "pls": fit_pls,
    "random_forest": fit_random_forest,
    "gaussian_process": fit_gaussian_process,
    "xgboost": fit_xgboost,
    "catboost": fit_catboost,
    "stacking": fit_stacking,
}


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


def bootstrap_p_value(samples: np.ndarray) -> float:
    samples = np.asarray(samples, dtype=float)
    samples = samples[np.isfinite(samples)]
    if len(samples) == 0:
        return float("nan")
    p = 2.0 * min(float(np.mean(samples <= 0.0)), float(np.mean(samples >= 0.0)))
    return float(min(1.0, max(p, 1.0 / (len(samples) + 1))))


def center_log_life(y: np.ndarray) -> tuple[np.ndarray, float]:
    y_log = np.log(np.asarray(y, dtype=float))
    mean_log = float(np.mean(y_log)) if len(y_log) else float("nan")
    return y_log - mean_log, mean_log


def fit_univariate_centered_log_slope(x: np.ndarray, y_centered_log: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    y_centered_log = np.asarray(y_centered_log, dtype=float)
    if len(x) == 0 or np.std(x) < 1e-12:
        return float(np.mean(y_centered_log)) if len(y_centered_log) else float("nan"), 0.0
    slope, intercept = np.polyfit(x, y_centered_log, 1)
    return float(intercept), float(slope)


def within_dataset_zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    std = float(np.std(values, ddof=0))
    if std < 1e-12:
        return np.zeros_like(values, dtype=float)
    return (values - float(np.mean(values))) / std


def feature_slope_table(
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    n_cycles: int,
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    matr = df[(df["dataset"] == "matr") & (df["n_cycles"] == n_cycles) & (df["is_censored"] == 0)].copy()
    hust = df[(df["dataset"] == "hust") & (df["n_cycles"] == n_cycles) & (df["is_censored"] == 0)].copy()
    rng = np.random.default_rng(seed)
    rows = []

    for feature in feature_cols:
        x_m = within_dataset_zscore(matr[feature].to_numpy(dtype=float))
        y_m = matr["cycle_life"].to_numpy(dtype=float)
        y_m_centered, mean_log_m = center_log_life(y_m)
        x_h = within_dataset_zscore(hust[feature].to_numpy(dtype=float))
        y_h = hust["cycle_life"].to_numpy(dtype=float)
        y_h_centered, mean_log_h = center_log_life(y_h)

        intercept_m, slope_m = fit_univariate_centered_log_slope(x_m, y_m_centered)
        intercept_h, slope_h = fit_univariate_centered_log_slope(x_h, y_h_centered)

        boot_delta_slope = []
        m_idx = np.arange(len(y_m))
        h_idx = np.arange(len(y_h))
        for _ in range(n_boot):
            bm = rng.choice(m_idx, size=len(m_idx), replace=True)
            bh = rng.choice(h_idx, size=len(h_idx), replace=True)
            ybm_centered, _ = center_log_life(y_m[bm])
            ybh_centered, _ = center_log_life(y_h[bh])
            _, bs_m = fit_univariate_centered_log_slope(x_m[bm], ybm_centered)
            _, bs_h = fit_univariate_centered_log_slope(x_h[bh], ybh_centered)
            boot_delta_slope.append(bs_h - bs_m)

        bds = np.asarray(boot_delta_slope, dtype=float)
        rows.append(
            {
                "feature": feature,
                "n_cycles": int(n_cycles),
                "n_matr": int(len(y_m)),
                "n_hust": int(len(y_h)),
                "mean_log_life_matr": mean_log_m,
                "mean_log_life_hust": mean_log_h,
                "universal_log_life_offset_hust_minus_matr": mean_log_h - mean_log_m,
                "centered_intercept_matr": intercept_m,
                "centered_intercept_hust": intercept_h,
                "delta_centered_intercept_hust_minus_matr": intercept_h - intercept_m,
                "slope_matr": slope_m,
                "slope_hust": slope_h,
                "delta_slope_hust_minus_matr": slope_h - slope_m,
                "delta_slope_ci95_low": float(np.percentile(bds, 2.5)),
                "delta_slope_ci95_high": float(np.percentile(bds, 97.5)),
                "delta_slope_boot_p": bootstrap_p_value(bds),
            }
        )

    out = pd.DataFrame(rows)
    out["delta_slope_fdr_bh"] = bh_fdr(out["delta_slope_boot_p"].to_numpy(dtype=float))

    slope_sig = (
        out["delta_slope_fdr_bh"].lt(0.05)
        & ((out["delta_slope_ci95_low"] > 0.0) | (out["delta_slope_ci95_high"] < 0.0))
    )
    out["slope_shift_significant"] = slope_sig
    out["shift_class"] = np.where(slope_sig, "slope_shifted", "slope_stable")
    return out.sort_values(
        ["shift_class", "delta_slope_fdr_bh", "feature"],
        ignore_index=True,
    )


def safe_pred(model: object, x: np.ndarray) -> np.ndarray:
    raw = model.predict(x)
    if hasattr(raw, "ravel"):
        raw = raw.ravel()
    raw = np.nan_to_num(raw, nan=0.0, posinf=1e9, neginf=-1e9)
    return np.clip(raw, -1e9, 1e9)


def to_cycles(pred: np.ndarray, *, log_target: bool) -> np.ndarray:
    if log_target:
        pred = np.clip(pred, np.log(1.0), np.log(1e9))
        pred = np.exp(pred)
    return np.clip(np.nan_to_num(pred, nan=1.0, posinf=1e9, neginf=1.0), 1.0, 1e9)


def load_split(dataset: str, seed: int) -> dict:
    with (SPLITS_DIR / f"{dataset}_{seed}.json").open() as f:
        return json.load(f)


def fit_source_predict_target(
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    source: str,
    target: str,
    model_name: str,
    seed: int,
    n_cycles: int,
    log_target: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from sklearn.preprocessing import StandardScaler

    src = df[(df["dataset"] == source) & (df["n_cycles"] == n_cycles) & (df["is_censored"] == 0)].copy()
    tgt = df[(df["dataset"] == target) & (df["n_cycles"] == n_cycles) & (df["is_censored"] == 0)].copy()
    split = load_split(source, seed)
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

    result = FITTERS[model_name](x_train_s, y_train_fit, seed=seed)
    model = result[0] if isinstance(result, tuple) else result
    y_pred = to_cycles(safe_pred(model, x_target_s), log_target=log_target)
    return target_cell_ids, y_target, y_pred


def fit_alpha_beta(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    if len(y_true) < 2 or np.std(y_pred) < 1e-12:
        return 1.0, float(np.mean(y_true - y_pred))
    alpha, beta = np.polyfit(y_pred, y_true, 1)
    return float(alpha), float(beta)


def pearson_summary(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) < 3 or np.std(y_true) < 1e-12 or np.std(y_pred) < 1e-12:
        return float("nan"), float("nan")
    try:
        res = pearsonr(y_pred, y_true)
        return float(res.statistic), float(res.pvalue)
    except Exception:
        return float("nan"), float("nan")


def fit_robust_alpha_beta(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    method: str,
    seed: int,
) -> tuple[float, float, float]:
    if len(y_true) < 2 or np.std(y_pred) < 1e-12:
        beta = float(np.mean(y_true - y_pred))
        pred = y_pred + beta
        return 1.0, beta, compute_metrics(y_true, pred)["R2"]

    x = np.asarray(y_pred, dtype=float).reshape(-1, 1)
    y = np.asarray(y_true, dtype=float)
    try:
        if method == "theil_sen":
            reg = TheilSenRegressor(random_state=seed, max_subpopulation=10000)
        elif method == "huber":
            reg = HuberRegressor(max_iter=1000)
        else:
            raise ValueError(f"unknown robust method: {method}")
        reg.fit(x, y)
        alpha = float(reg.coef_[0])
        beta = float(reg.intercept_)
        pred = np.clip(alpha * y_pred + beta, 1.0, 1e9)
        return alpha, beta, compute_metrics(y_true, pred)["R2"]
    except Exception:
        alpha, beta = fit_alpha_beta(y_true, y_pred)
        pred = np.clip(alpha * y_pred + beta, 1.0, 1e9)
        return alpha, beta, compute_metrics(y_true, pred)["R2"]


def alpha_beta_table(
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    n_cycles: int,
    seeds: list[int],
    models: list[str],
    n_boot: int,
    seed: int,
    log_target: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed + 991)
    rows = []
    pred_rows = []
    for source, target in [("matr", "hust"), ("hust", "matr")]:
        for model_name in models:
            for split_seed in seeds:
                cell_ids, y_true, y_pred = fit_source_predict_target(
                    df,
                    feature_cols,
                    source=source,
                    target=target,
                    model_name=model_name,
                    seed=split_seed,
                    n_cycles=n_cycles,
                    log_target=log_target,
                )
                alpha, beta = fit_alpha_beta(y_true, y_pred)
                ts_alpha, ts_beta, ts_r2 = fit_robust_alpha_beta(
                    y_true, y_pred, method="theil_sen", seed=split_seed
                )
                huber_alpha, huber_beta, huber_r2 = fit_robust_alpha_beta(
                    y_true, y_pred, method="huber", seed=split_seed
                )
                constant_bias = float(np.mean(y_true - y_pred))
                raw = compute_metrics(y_true, y_pred)
                constant = compute_metrics(y_true, y_pred + constant_bias)
                linear = compute_metrics(y_true, np.clip(alpha * y_pred + beta, 1.0, 1e9))
                pearson_r, pearson_p = pearson_summary(y_true, y_pred)

                for cell_id, actual, pred in zip(cell_ids, y_true, y_pred, strict=False):
                    pred_rows.append(
                        {
                            "source": source,
                            "target": target,
                            "direction": f"{source.upper()} -> {target.upper()}",
                            "model": model_name,
                            "seed": int(split_seed),
                            "cell_id": cell_id,
                            "cycle_life": float(actual),
                            "source_prediction": float(pred),
                            "residual": float(actual - pred),
                        }
                    )

                boot_alpha = []
                boot_beta = []
                boot_constant_share = []
                boot_pearson_r = []
                indices = np.arange(len(y_true))
                for _ in range(n_boot):
                    b = rng.choice(indices, size=len(indices), replace=True)
                    ba, bb = fit_alpha_beta(y_true[b], y_pred[b])
                    br, _ = pearson_summary(y_true[b], y_pred[b])
                    residuals = y_true[b] - y_pred[b]
                    ss_total = float(np.sum(residuals**2))
                    ss_after_constant = float(np.sum((residuals - np.mean(residuals)) ** 2))
                    share = 1.0 - ss_after_constant / ss_total if ss_total > 1e-12 else float("nan")
                    boot_alpha.append(ba)
                    boot_beta.append(bb)
                    boot_constant_share.append(share)
                    boot_pearson_r.append(br)

                ba = np.asarray(boot_alpha, dtype=float)
                bb = np.asarray(boot_beta, dtype=float)
                bs = np.asarray(boot_constant_share, dtype=float)
                br = np.asarray(boot_pearson_r, dtype=float)
                rows.append(
                    {
                        "source": source,
                        "target": target,
                        "direction": f"{source.upper()} -> {target.upper()}",
                        "model": model_name,
                        "seed": int(split_seed),
                        "n_cycles": int(n_cycles),
                        "n_target": int(len(y_true)),
                        "alpha": alpha,
                        "alpha_ci95_low": float(np.percentile(ba, 2.5)),
                        "alpha_ci95_high": float(np.percentile(ba, 97.5)),
                        "alpha_ci95_contains_1": bool(np.percentile(ba, 2.5) <= 1.0 <= np.percentile(ba, 97.5)),
                        "alpha_minus_1_boot_p": bootstrap_p_value(ba - 1.0),
                        "beta": beta,
                        "beta_ci95_low": float(np.percentile(bb, 2.5)),
                        "beta_ci95_high": float(np.percentile(bb, 97.5)),
                        "theil_sen_alpha": ts_alpha,
                        "theil_sen_beta": ts_beta,
                        "theil_sen_R2": ts_r2,
                        "huber_alpha": huber_alpha,
                        "huber_beta": huber_beta,
                        "huber_R2": huber_r2,
                        "pearson_r": pearson_r,
                        "pearson_r_ci95_low": float(np.nanpercentile(br, 2.5)),
                        "pearson_r_ci95_high": float(np.nanpercentile(br, 97.5)),
                        "pearson_p": pearson_p,
                        "constant_bias": constant_bias,
                        "constant_share_of_ss": float(np.mean(bs)),
                        "constant_share_ci95_low": float(np.percentile(bs, 2.5)),
                        "constant_share_ci95_high": float(np.percentile(bs, 97.5)),
                        "raw_MAE": raw["MAE"],
                        "raw_R2": raw["R2"],
                        "constant_MAE": constant["MAE"],
                        "constant_R2": constant["R2"],
                        "linear_MAE": linear["MAE"],
                        "linear_R2": linear["R2"],
                    }
                )
    out = pd.DataFrame(rows)
    out["alpha_minus_1_fdr_bh"] = bh_fdr(out["alpha_minus_1_boot_p"].to_numpy(dtype=float))
    return out, pd.DataFrame(pred_rows)


def write_alpha_beta_scatter(pred_rows: pd.DataFrame, alpha_rows: pd.DataFrame, output_dir: Path, *, seed: int) -> Path | None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None

    sub = pred_rows[pred_rows["seed"] == seed].copy()
    if sub.empty:
        return None

    directions = ["MATR -> HUST", "HUST -> MATR"]
    models = sorted(sub["model"].unique())
    fig, axes = plt.subplots(len(directions), len(models), figsize=(5.2 * len(models), 4.4 * len(directions)), squeeze=False)
    for i, direction in enumerate(directions):
        for j, model in enumerate(models):
            ax = axes[i][j]
            panel = sub[(sub["direction"] == direction) & (sub["model"] == model)]
            if panel.empty:
                ax.axis("off")
                continue
            ax.scatter(panel["source_prediction"], panel["cycle_life"], s=28, alpha=0.75, edgecolor="none")
            x_min = float(panel["source_prediction"].min())
            x_max = float(panel["source_prediction"].max())
            y_min = float(panel["cycle_life"].min())
            y_max = float(panel["cycle_life"].max())
            lo = min(x_min, y_min)
            hi = max(x_max, y_max)
            xs = np.linspace(x_min, x_max, 100)
            fit_row = alpha_rows[
                (alpha_rows["direction"] == direction) & (alpha_rows["model"] == model) & (alpha_rows["seed"] == seed)
            ]
            if not fit_row.empty:
                row = fit_row.iloc[0]
                ax.plot(xs, row["alpha"] * xs + row["beta"], color="#d62728", linewidth=2.0, label=f"OLS alpha={row['alpha']:.2f}")
                ax.plot(
                    xs,
                    row["theil_sen_alpha"] * xs + row["theil_sen_beta"],
                    color="#2ca02c",
                    linewidth=1.7,
                    linestyle="--",
                    label=f"Theil-Sen alpha={row['theil_sen_alpha']:.2f}",
                )
                ax.text(
                    0.98,
                    0.97,
                    f"r={row['pearson_r']:.2f} "
                    f"[{row['pearson_r_ci95_low']:.2f}, {row['pearson_r_ci95_high']:.2f}]\n"
                    f"p={row['pearson_p']:.2g}",
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=9,
                    bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.85},
                )
            ax.plot([lo, hi], [lo, hi], color="#666666", linewidth=1.0, linestyle=":", label="identity")
            ax.set_title(f"{direction}, {model}")
            ax.set_xlabel("Source-model prediction (cycles)")
            ax.set_ylabel("Target cycle life")
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"conditional_shift_alpha_beta_scatter_seed{seed}.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def write_report(feature_rows: pd.DataFrame, alpha_rows: pd.DataFrame, path: Path) -> None:
    class_counts = feature_rows["shift_class"].value_counts().to_dict()
    n_features = len(feature_rows)
    slope_shifted = int(class_counts.get("slope_shifted", 0))
    slope_stable = int(class_counts.get("slope_stable", 0))
    universal_offset = float(feature_rows["universal_log_life_offset_hust_minus_matr"].iloc[0])

    lines = [
        "Conditional Shift Decomposition",
        "================================",
        "",
        f"Features analyzed: {n_features}",
        f"Universal HUST-MATR log-life offset: {universal_offset:.3f} (life ratio {np.exp(universal_offset):.2f}x)",
        f"Slope-stable features after log-life centering: {slope_stable} ({slope_stable / max(n_features, 1):.1%})",
        f"Slope-shifted features after log-life centering: {slope_shifted} ({slope_shifted / max(n_features, 1):.1%})",
        "",
        "Feature shift classes:",
    ]
    for klass, count in sorted(class_counts.items()):
        lines.append(f"  - {klass}: {count}")

    lines.extend(["", "Alpha/beta calibration summary:"])
    group_cols = ["direction", "model"]
    for (direction, model), sub in alpha_rows.groupby(group_cols):
        lines.append(
            "  - "
            f"{direction} {model}: alpha={sub['alpha'].mean():.3f} "
            f"[{sub['alpha_ci95_low'].mean():.3f}, {sub['alpha_ci95_high'].mean():.3f}], "
            f"constant_share={100 * sub['constant_share_of_ss'].mean():.1f}%, "
            f"constant_R2={sub['constant_R2'].mean():+.3f}, "
            f"linear_R2={sub['linear_R2'].mean():+.3f}, "
            f"Pearson r={sub['pearson_r'].mean():+.3f} "
            f"[{sub['pearson_r_ci95_low'].mean():+.3f}, {sub['pearson_r_ci95_high'].mean():+.3f}], "
            f"p={sub['pearson_p'].mean():.3g}, "
            f"Theil-Sen alpha={sub['theil_sen_alpha'].mean():+.3f}, "
            f"Huber alpha={sub['huber_alpha'].mean():+.3f}"
        )

    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-cycles", type=int, default=100)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, choices=ALL_MODELS)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--features-from", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--scatter-seed", type=int, default=42)
    parser.add_argument("--log-target", action="store_true", default=True)
    parser.add_argument("--no-log-target", action="store_false", dest="log_target")
    parser.add_argument("--random-seed", type=int, default=20260507)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not FEATURES_PATH.exists():
        print(f"[error] missing {FEATURES_PATH}")
        return 1
    df = pd.read_csv(FEATURES_PATH)
    available = [c for c in df.columns if c not in META_COLS]
    if args.features_from is not None and args.features_from.exists():
        feature_cols = [line.strip() for line in args.features_from.read_text().splitlines() if line.strip()]
    else:
        feature_cols = available

    print(
        f"[setup] n_features={len(feature_cols)}, models={args.models}, "
        f"seeds={args.seeds}, boot={args.n_bootstrap}"
    )

    print("\n========== per-feature centered-log slope decomposition ==========")
    feature_rows = feature_slope_table(
        df,
        feature_cols,
        n_cycles=args.n_cycles,
        n_boot=args.n_bootstrap,
        seed=args.random_seed,
    )
    out_feature = INTERMEDIATE_DIR / "conditional_shift_feature_slopes.csv"
    feature_rows.to_csv(out_feature, index=False)
    print(f"[save] {out_feature}")
    print(feature_rows["shift_class"].value_counts().to_string())

    print("\n========== alpha/beta source-prediction calibration ==========")
    alpha_rows, prediction_rows = alpha_beta_table(
        df,
        feature_cols,
        n_cycles=args.n_cycles,
        seeds=args.seeds,
        models=args.models,
        n_boot=args.n_bootstrap,
        seed=args.random_seed,
        log_target=args.log_target,
    )
    out_alpha = INTERMEDIATE_DIR / "conditional_shift_alpha_beta.csv"
    alpha_rows.to_csv(out_alpha, index=False)
    print(f"[save] {out_alpha}")
    out_predictions = INTERMEDIATE_DIR / "conditional_shift_alpha_beta_predictions.csv"
    prediction_rows.to_csv(out_predictions, index=False)
    print(f"[save] {out_predictions}")
    scatter_path = write_alpha_beta_scatter(prediction_rows, alpha_rows, args.output_dir, seed=args.scatter_seed)
    if scatter_path is not None:
        print(f"[save] {scatter_path}")
    print(
        alpha_rows.groupby(["direction", "model"])[
            [
                "alpha",
                "theil_sen_alpha",
                "huber_alpha",
                "pearson_r",
                "pearson_r_ci95_low",
                "pearson_r_ci95_high",
                "pearson_p",
                "constant_share_of_ss",
                "constant_R2",
                "linear_R2",
            ]
        ].mean()
    )

    universal_offset = float(feature_rows["universal_log_life_offset_hust_minus_matr"].iloc[0])
    payload = {
        "protocol": "conditional_shift_decomposition_v2_centered_log_life",
        "n_cycles": int(args.n_cycles),
        "feature_count": int(len(feature_cols)),
        "models": args.models,
        "seeds": args.seeds,
        "n_bootstrap": int(args.n_bootstrap),
        "universal_log_life_offset_hust_minus_matr": universal_offset,
        "universal_life_ratio_hust_over_matr": float(np.exp(universal_offset)),
        "shift_class_counts": feature_rows["shift_class"].value_counts().to_dict(),
        "alpha_beta_summary": (
            alpha_rows.groupby(["direction", "model"])[
                [
                    "alpha",
                    "theil_sen_alpha",
                    "huber_alpha",
                    "pearson_r",
                    "pearson_r_ci95_low",
                    "pearson_r_ci95_high",
                    "pearson_p",
                    "constant_share_of_ss",
                    "raw_R2",
                    "constant_R2",
                    "linear_R2",
                    "theil_sen_R2",
                    "huber_R2",
                ]
            ]
            .mean()
            .reset_index()
            .to_dict(orient="records")
        ),
    }
    out_json = INTERMEDIATE_DIR / "conditional_shift_summary.json"
    out_json.write_text(json.dumps(payload, indent=2))
    print(f"[save] {out_json}")

    out_report = INTERMEDIATE_DIR / "conditional_shift_report.txt"
    write_report(feature_rows, alpha_rows, out_report)
    print(f"[save] {out_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

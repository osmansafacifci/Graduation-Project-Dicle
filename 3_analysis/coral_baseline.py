"""Shallow CORAL covariance-alignment diagnostic for four-dataset transfer.

This is a reviewer-facing diagnostic baseline, not a full domain-adaptation
benchmark. For each of the 12 four-dataset transfer directions, the script:

1. uses the official five source-training splits at N=100,
2. trains a fixed RidgeCV log-life regressor on source features,
3. estimates a CORAL transform from source-train X to full unlabeled target X,
4. retrains the same RidgeCV regressor on CORAL-aligned source features, and
5. evaluates naive vs CORAL predictions on the full target labels.

The target labels are used only for scoring. Target covariates are used for
unsupervised moment alignment, which is the standard shallow CORAL setting.
Ridge predictions are clipped to the source-training log-life range before
conversion back to cycles; this is a conservative extrapolation policy that
prevents a linear diagnostic from being dominated by unphysical cycle counts.

Outputs
-------
data/intermediate/coral_baseline_summary.csv
data/intermediate/coral_baseline_summary.md

Usage
-----
python 3_analysis/coral_baseline.py
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MODELS_DIR = ROOT / "2_models"
sys.path.insert(0, str(MODELS_DIR))

from metrics_utils import compute_metrics, to_cycles  # noqa: E402
from run_experiments import META_COLS, SEEDS  # noqa: E402


FEATURES_PATH = ROOT / "data" / "intermediate" / "features_sop12_four_dataset_capnorm.csv"
SPLITS_DIR = ROOT / "splits" / "sop_v2_four_dataset"
DIRECTION_SUMMARY_PATH = ROOT / "data" / "intermediate" / "four_dataset_conditional_shift_direction_summary.csv"
OUT_CSV = ROOT / "data" / "intermediate" / "coral_baseline_summary.csv"
OUT_MD = ROOT / "data" / "intermediate" / "coral_baseline_summary.md"

N_CYCLES = 100
LOG_TARGET = True
RIDGE_SCALE = 1e-4
ALPHAS = np.logspace(-4, 4, 17)


def super_regime(rank_signal_class: str) -> str:
    """Map fine-grained rank-signal classes to deployment super-regimes."""
    if rank_signal_class in {"strong_rank_signal", "moderate_rank_signal"}:
        return "salvageable_linear_recovers"
    if rank_signal_class == "weak_rank_signal":
        return "offset_dominant_residual_only"
    return "cp_interval_only"


def feature_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c not in META_COLS]
    if not cols:
        raise ValueError("No feature columns found after dropping metadata columns.")
    return cols


def clean_scaled(X: np.ndarray) -> np.ndarray:
    return np.clip(np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0), -1e6, 1e6)


def covariance_with_ridge(X: np.ndarray, *, ridge_scale: float = RIDGE_SCALE) -> tuple[np.ndarray, float]:
    """Empirical covariance with deterministic trace-scaled ridge shrinkage."""
    if X.ndim != 2:
        raise ValueError("X must be 2D")
    n, d = X.shape
    if n < 2:
        raise ValueError("Need at least two rows for covariance.")
    cov = np.cov(X, rowvar=False, bias=False)
    cov = np.asarray(cov, dtype=float)
    cov = 0.5 * (cov + cov.T)
    trace = float(np.trace(cov))
    ridge = ridge_scale if (not np.isfinite(trace) or trace <= 0) else ridge_scale * trace / max(d, 1)
    cov = cov + ridge * np.eye(d)
    return cov, float(ridge)


def symmetric_power(mat: np.ndarray, power: float) -> np.ndarray:
    """Symmetric matrix fractional power with eigenvalue clipping."""
    mat = 0.5 * (mat + mat.T)
    vals, vecs = np.linalg.eigh(mat)
    vals = np.clip(vals, 1e-12, None)
    return (vecs * (vals ** power)) @ vecs.T


def coral_align_source(
    X_source: np.ndarray,
    X_target: np.ndarray,
    *,
    ridge_scale: float = RIDGE_SCALE,
) -> tuple[np.ndarray, dict[str, float]]:
    """Align source train features to target first and second moments.

    Row-vector convention:
        X_source_aligned = (X_source - mu_s) C_s^{-1/2} C_t^{1/2} + mu_t

    This is a shallow unsupervised CORAL transform. It uses target covariates
    but no target labels.
    """
    X_source = np.asarray(X_source, dtype=float)
    X_target = np.asarray(X_target, dtype=float)
    mu_s = X_source.mean(axis=0)
    mu_t = X_target.mean(axis=0)
    Xs = X_source - mu_s
    Xt = X_target - mu_t

    cov_s, ridge_s = covariance_with_ridge(Xs, ridge_scale=ridge_scale)
    cov_t, ridge_t = covariance_with_ridge(Xt, ridge_scale=ridge_scale)

    transform = symmetric_power(cov_s, -0.5) @ symmetric_power(cov_t, 0.5)
    aligned = Xs @ transform + mu_t
    info = {
        "ridge_source": float(ridge_s),
        "ridge_target": float(ridge_t),
        "source_cov_condition": float(np.linalg.cond(cov_s)),
        "target_cov_condition": float(np.linalg.cond(cov_t)),
    }
    return aligned, info


def fit_ridge(X_train: np.ndarray, y_train_log: np.ndarray) -> RidgeCV:
    model = RidgeCV(alphas=ALPHAS)
    model.fit(X_train, y_train_log)
    return model


def predict_cycles(model: RidgeCV, X: np.ndarray, *, log_bounds: tuple[float, float]) -> np.ndarray:
    pred = model.predict(X)
    pred = np.clip(pred, log_bounds[0], log_bounds[1])
    return to_cycles(pred, log_target=LOG_TARGET)


def evaluate_seed(
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    source: str,
    target: str,
    seed: int,
    ridge_scale: float,
) -> dict[str, float]:
    split_path = SPLITS_DIR / f"{source}_{seed}.json"
    if not split_path.exists():
        raise FileNotFoundError(split_path)
    split = json.loads(split_path.read_text())

    subset = df[(df["n_cycles"] == N_CYCLES) & (df["is_censored"] == 0)].copy()
    source_all = subset[subset["dataset"] == source]
    target_all = subset[subset["dataset"] == target]
    train_df = source_all[source_all["cell_id"].isin(split["train"])]
    if len(train_df) < 5:
        raise ValueError(f"Too few source train cells for {source}->{target}, seed={seed}: {len(train_df)}")
    if len(target_all) < 2:
        raise ValueError(f"Too few target cells for {source}->{target}: {len(target_all)}")

    X_train = train_df[feature_cols].to_numpy(dtype=float)
    y_train = train_df["cycle_life"].to_numpy(dtype=float)
    y_train_log = np.log(y_train)
    train_log_bounds = (float(y_train_log.min()), float(y_train_log.max()))
    X_target = target_all[feature_cols].to_numpy(dtype=float)
    y_target = target_all["cycle_life"].to_numpy(dtype=float)

    # Naive source baseline: z-score fitted on source train, target transformed
    # with the same scaler.
    scaler = StandardScaler()
    X_train_s = clean_scaled(scaler.fit_transform(X_train))
    X_target_s = clean_scaled(scaler.transform(X_target))
    naive_model = fit_ridge(X_train_s, y_train_log)
    naive_pred = predict_cycles(naive_model, X_target_s, log_bounds=train_log_bounds)
    naive_metrics = compute_metrics(y_target, naive_pred)

    # CORAL: source-train covariates are aligned to full unlabeled target
    # covariates, then the normal train-only scaler is fit on aligned source.
    X_train_coral, coral_info = coral_align_source(X_train, X_target, ridge_scale=ridge_scale)
    coral_scaler = StandardScaler()
    X_train_coral_s = clean_scaled(coral_scaler.fit_transform(X_train_coral))
    X_target_coral_s = clean_scaled(coral_scaler.transform(X_target))
    coral_model = fit_ridge(X_train_coral_s, y_train_log)
    coral_pred = predict_cycles(coral_model, X_target_coral_s, log_bounds=train_log_bounds)
    coral_metrics = compute_metrics(y_target, coral_pred)

    return {
        "seed": int(seed),
        "source_train_cells": int(len(train_df)),
        "n_target": int(len(target_all)),
        "naive_MAE": naive_metrics["MAE"],
        "naive_SMAPE": naive_metrics["SMAPE"],
        "naive_R2": naive_metrics["R2"],
        "coral_MAE": coral_metrics["MAE"],
        "coral_SMAPE": coral_metrics["SMAPE"],
        "coral_R2": coral_metrics["R2"],
        "ridge_source": coral_info["ridge_source"],
        "ridge_target": coral_info["ridge_target"],
        "source_cov_condition": coral_info["source_cov_condition"],
        "target_cov_condition": coral_info["target_cov_condition"],
    }


def aggregate_direction(seed_rows: list[dict[str, float]]) -> dict[str, float]:
    df = pd.DataFrame(seed_rows)
    out: dict[str, float] = {
        "n_seeds": int(len(df)),
        "source_train_cells_mean": float(df["source_train_cells"].mean()),
        "n_target": int(df["n_target"].iloc[0]),
    }
    for prefix in ("naive", "coral"):
        for metric in ("MAE", "SMAPE", "R2"):
            col = f"{prefix}_{metric}"
            out[f"{col}_mean"] = float(df[col].mean())
            out[f"{col}_std"] = float(df[col].std(ddof=0))
    out["delta_MAE_coral_minus_naive"] = out["coral_MAE_mean"] - out["naive_MAE_mean"]
    out["delta_SMAPE_coral_minus_naive"] = out["coral_SMAPE_mean"] - out["naive_SMAPE_mean"]
    out["delta_R2_coral_minus_naive"] = out["coral_R2_mean"] - out["naive_R2_mean"]
    out["ridge_source_mean"] = float(df["ridge_source"].mean())
    out["ridge_target_mean"] = float(df["ridge_target"].mean())
    return out


def format_num(value: float, digits: int = 3) -> str:
    if not np.isfinite(value):
        return "nan"
    return f"{value:.{digits}f}"


def write_markdown(summary: pd.DataFrame, path: Path) -> None:
    rows = [
        "# CORAL Covariance-Alignment Diagnostic Baseline",
        "",
        "Shallow unsupervised CORAL is included as one representative covariate-alignment baseline, not as an exhaustive domain-adaptation benchmark.",
        "",
        "Protocol: N=100, 34-feature Q0-normalised table; five official source-training splits; fixed RidgeCV log-life regressor; full unlabeled target covariates for CORAL moment alignment; target labels used only for scoring. Predictions are clipped to the source-training log-life range as a conservative extrapolation policy.",
        "",
        "This diagnostic intentionally keeps the downstream learner fixed. The comparison is therefore naive Ridge transfer vs CORAL-aligned Ridge transfer, not a replacement for the model-champion cross-dataset table.",
        "",
        "## Summary",
        "",
    ]

    improved = int((summary["delta_R2_coral_minus_naive"] > 0).sum())
    positive = int((summary["coral_R2_mean"] > 0).sum())
    rank_collapsed = summary[summary["super_regime"] == "cp_interval_only"]
    rank_collapsed_positive = int((rank_collapsed["coral_R2_mean"] > 0).sum())
    rows.extend(
        [
            f"- CORAL improves R2 over naive Ridge transfer in **{improved}/12** directions.",
            f"- CORAL yields positive R2 in **{positive}/12** directions.",
            f"- In rank-collapsed / CP-only directions, CORAL yields positive R2 in **{rank_collapsed_positive}/{len(rank_collapsed)}** directions.",
            "- Interpretation should be directional and regime-aware: CORAL modifies unlabeled covariate geometry; it does not use target labels and does not directly repair conditional shift.",
            "",
            "## Direction-Level Results",
            "",
            "| Direction | Regime | Naive Ridge R2 | CORAL Ridge R2 | Delta R2 | Naive MAE | CORAL MAE | Delta MAE | Reference Table 3 model | Reference Table 3 R2 |",
            "|---|---|---:|---:|---:|---:|---:|---:|---|---:|",
        ]
    )

    for _, row in summary.sort_values(["super_regime", "experiment"]).iterrows():
        rows.append(
            "| {experiment} | {super_regime} | {naive_r2} | {coral_r2} | {delta_r2} | {naive_mae} | {coral_mae} | {delta_mae} | {reference_model} | {reference_r2} |".format(
                experiment=row["experiment"].replace("_to_", " -> "),
                super_regime=row["super_regime"],
                naive_r2=format_num(row["naive_R2_mean"]),
                coral_r2=format_num(row["coral_R2_mean"]),
                delta_r2=format_num(row["delta_R2_coral_minus_naive"]),
                naive_mae=format_num(row["naive_MAE_mean"], 1),
                coral_mae=format_num(row["coral_MAE_mean"], 1),
                delta_mae=format_num(row["delta_MAE_coral_minus_naive"], 1),
                reference_model=row["reference_model"],
                reference_r2=format_num(row["reference_raw_R2"]),
            )
        )

    rows.extend(
        [
            "",
            "## Caveat",
            "",
            "CORAL is a mean/covariance-alignment diagnostic. It is useful as a standard unsupervised alignment falsifier, but it is not a claim that all modern domain-adaptation methods have been exhausted.",
        ]
    )
    path.write_text("\n".join(rows) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-path", type=Path, default=FEATURES_PATH)
    parser.add_argument("--direction-summary", type=Path, default=DIRECTION_SUMMARY_PATH)
    parser.add_argument("--output-csv", type=Path, default=OUT_CSV)
    parser.add_argument("--output-md", type=Path, default=OUT_MD)
    parser.add_argument("--ridge-scale", type=float, default=RIDGE_SCALE)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    features_path = args.features_path if args.features_path.is_absolute() else ROOT / args.features_path
    direction_path = args.direction_summary if args.direction_summary.is_absolute() else ROOT / args.direction_summary
    out_csv = args.output_csv if args.output_csv.is_absolute() else ROOT / args.output_csv
    out_md = args.output_md if args.output_md.is_absolute() else ROOT / args.output_md

    df = pd.read_csv(features_path)
    directions = pd.read_csv(direction_path)
    cols = feature_columns(df)
    directions = directions.sort_values(["source", "target"]).reset_index(drop=True)

    summary_rows: list[dict] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _, direction in directions.iterrows():
            source = str(direction["source"])
            target = str(direction["target"])
            experiment = f"{source}_to_{target}"
            print(f"[coral] {experiment}", flush=True)
            seed_rows = [
                evaluate_seed(
                    df,
                    cols,
                    source=source,
                    target=target,
                    seed=seed,
                    ridge_scale=float(args.ridge_scale),
                )
                for seed in SEEDS
            ]
            agg = aggregate_direction(seed_rows)
            summary_rows.append(
                {
                    "experiment": experiment,
                    "source": source,
                    "target": target,
                    "estimator": "ridge_cv",
                    "reference_model": direction["model"],
                    "reference_raw_R2": float(direction["raw_R2"]),
                    "reference_raw_MAE": float(direction["raw_MAE"]),
                    "rank_signal_class": direction["rank_signal_class"],
                    "super_regime": super_regime(str(direction["rank_signal_class"])),
                    "slope_shifted_share": float(direction["slope_shifted_share"]),
                    "life_ratio_target_over_source": float(direction["life_ratio_target_over_source"]),
                    **agg,
                }
            )

    summary = pd.DataFrame(summary_rows)
    summary = summary.sort_values(["super_regime", "experiment"]).reset_index(drop=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_csv, index=False)
    write_markdown(summary, out_md)

    print(f"[save] {out_csv.relative_to(ROOT)}", flush=True)
    print(f"[save] {out_md.relative_to(ROOT)}", flush=True)
    print("\n=== CORAL diagnostic summary ===", flush=True)
    print(
        summary[
            [
                "experiment",
                "super_regime",
                "naive_R2_mean",
                "coral_R2_mean",
                "delta_R2_coral_minus_naive",
                "naive_MAE_mean",
                "coral_MAE_mean",
                "delta_MAE_coral_minus_naive",
                "reference_model",
                "reference_raw_R2",
            ]
        ].to_string(index=False),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

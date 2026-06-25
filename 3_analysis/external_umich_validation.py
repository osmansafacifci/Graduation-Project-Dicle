"""
Held-out UMich external validation of the frozen rank-signal regime thresholds.

This script is intentionally additive: it does not recompute the original
four-dataset benchmark. It consumes the UMich feature table and the eight
new UMich<->existing cross-dataset summaries, refits only the naive-best
model per new direction, computes Pearson rank signal and simple target-side
calibration diagnostics, then applies the frozen thresholds in
configs/regime_thresholds_frozen.yaml.

Usage:
    python 3_analysis/external_umich_validation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(PROJECT_ROOT / "2_models"))
sys.path.insert(0, str(PROJECT_ROOT / "3_analysis"))
sys.path.insert(0, str(PROJECT_ROOT))
from shared.constants import META_COLS, SEEDS  # noqa: E402
from shared.battery_utils import display_path as _display_path  # noqa: E402
from metrics_utils import compute_metrics  # noqa: E402
from conditional_shift_four_dataset import (  # noqa: E402
    classify_adapter,
    classify_rank_signal,
    fit_alpha_beta,
    fit_source_predict_target,
    pearson_summary,
)

FEATURES_PATH = PROJECT_ROOT / "data/intermediate/features_sop12_four_dataset_plus_umich_capnorm.csv"
SPLITS_DIR = PROJECT_ROOT / "splits/sop_v2_five_dataset_external"
WITHIN_SUMMARY = PROJECT_ROOT / "outputs/results_v2_external_umich_within_34feat_capnorm_log/results_summary.csv"
CROSS_ROOT = PROJECT_ROOT / "outputs/results_v2_external_umich_cross_34feat_capnorm_log"
OUTPUT_DIR = PROJECT_ROOT / "outputs/results_v2_external_umich_validation"
INTERMEDIATE_DIR = PROJECT_ROOT / "data/intermediate"
THRESHOLD_CONFIG = PROJECT_ROOT / "configs/regime_thresholds_frozen.yaml"

HOLDOUT = "umich"
REFERENCE_DATASETS = ["matr", "hust", "sandia", "luh"]
N_CYCLES = 100
N_BOOT = 1000
RANDOM_SEED = 20260525


def display_path(path: Path) -> str:
    return _display_path(path, PROJECT_ROOT)


def bootstrap_ci(values: list[float] | np.ndarray, *, q=(2.5, 97.5)) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float("nan"), float("nan")
    return float(np.percentile(arr, q[0])), float(np.percentile(arr, q[1]))


def pearson_bootstrap(y_true: np.ndarray, y_pred: np.ndarray, *, rng: np.random.Generator) -> tuple[float, float]:
    idx = np.arange(len(y_true))
    boot = []
    for _ in range(N_BOOT):
        b = rng.choice(idx, size=len(idx), replace=True)
        r, _ = pearson_summary(y_true[b], y_pred[b])
        boot.append(r)
    return bootstrap_ci(boot)


def load_cross_summary() -> pd.DataFrame:
    frames = []
    for path in sorted(CROSS_ROOT.glob("pair_*/results_summary.csv")):
        part = pd.read_csv(path)
        part["source_file"] = display_path(path)
        frames.append(part)
    if not frames:
        raise FileNotFoundError(f"No pair summaries found under {CROSS_ROOT}")
    out = pd.concat(frames, ignore_index=True)
    out = out[out["experiment"].str.contains(HOLDOUT)].copy()
    return out


def select_best_models(cross: pd.DataFrame) -> pd.DataFrame:
    sub = cross[cross["n_cycles"].eq(N_CYCLES)].copy()
    best = (
        sub.sort_values(["experiment", "R2_mean"], ascending=[True, False])
        .groupby("experiment", as_index=False)
        .head(1)
        .sort_values("experiment")
        .reset_index(drop=True)
    )
    expected = {f"{a}_to_{HOLDOUT}" for a in REFERENCE_DATASETS} | {f"{HOLDOUT}_to_{a}" for a in REFERENCE_DATASETS}
    missing = sorted(expected - set(best["experiment"]))
    if missing:
        raise RuntimeError(f"Missing external validation directions: {missing}")
    return best


def deployment_super_regime(rank_signal_class: str) -> str:
    if rank_signal_class in {"strong_rank_signal", "moderate_rank_signal"}:
        return "salvageable_linear_recovers"
    if rank_signal_class == "weak_rank_signal":
        return "offset_dominant_residual_only"
    return "cp_interval_only"


def validation_match(row: pd.Series) -> bool:
    predicted = row["predicted_super_regime"]
    if predicted == "salvageable_linear_recovers":
        return bool(row["linear_R2"] > 0.25)
    if predicted == "offset_dominant_residual_only":
        return bool(row["residual_R2"] >= -0.10 and row["linear_R2"] <= 0.25)
    return bool(row["linear_R2"] <= 0.25)


def life_ratio_lookup(df: pd.DataFrame) -> dict[tuple[str, str], float]:
    means = {}
    for dataset, sub in df[(df["n_cycles"].eq(N_CYCLES)) & (df["is_censored"].eq(0))].groupby("dataset"):
        means[dataset] = float(np.mean(np.log(sub["cycle_life"].to_numpy(dtype=float))))
    out = {}
    for source, mean_source in means.items():
        for target, mean_target in means.items():
            if source != target:
                out[(source, target)] = float(np.exp(mean_target - mean_source))
    return out


def markdown_table(df: pd.DataFrame, float_digits: int = 3) -> str:
    formatted = df.copy()
    for col in formatted.select_dtypes(include=[np.number]).columns:
        formatted[col] = formatted[col].map(lambda x: "" if pd.isna(x) else f"{x:.{float_digits}f}")
    formatted = formatted.astype(object).where(pd.notna(formatted), "")
    header = "| " + " | ".join(map(str, formatted.columns)) + " |"
    sep = "| " + " | ".join(["---"] * len(formatted.columns)) + " |"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in formatted.to_numpy()]
    return "\n".join([header, sep, *rows])


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)

    if not THRESHOLD_CONFIG.exists():
        raise FileNotFoundError(f"Frozen threshold config missing: {THRESHOLD_CONFIG}")

    df = pd.read_csv(FEATURES_PATH)
    feature_cols = [c for c in df.columns if c not in META_COLS]
    cross = load_cross_summary()
    best = select_best_models(cross)
    ratios = life_ratio_lookup(df)
    rng = np.random.default_rng(RANDOM_SEED)

    seed_rows = []
    pred_rows = []
    for row in best.to_dict(orient="records"):
        experiment = row["experiment"]
        source, target = experiment.split("_to_")
        model_name = row["model"]
        print(f"[diagnostic] {experiment}: {model_name}")
        for seed in SEEDS:
            cell_ids, y_true, y_pred = fit_source_predict_target(
                df,
                feature_cols,
                source=source,
                target=target,
                model_name=model_name,
                seed=seed,
                n_cycles=N_CYCLES,
                splits_dir=SPLITS_DIR,
                log_target=True,
            )
            alpha, beta = fit_alpha_beta(y_true, y_pred)
            residual_bias = float(np.mean(y_true - y_pred))
            raw = compute_metrics(y_true, y_pred)
            residual = compute_metrics(y_true, np.clip(y_pred + residual_bias, 1.0, 1e9))
            linear = compute_metrics(y_true, np.clip(alpha * y_pred + beta, 1.0, 1e9))
            pearson_r, pearson_p = pearson_summary(y_true, y_pred)
            r_low, r_high = pearson_bootstrap(y_true, y_pred, rng=rng)
            seed_rows.append(
                {
                    "experiment": experiment,
                    "source": source,
                    "target": target,
                    "model": model_name,
                    "seed": int(seed),
                    "n_cycles": N_CYCLES,
                    "n_target": int(len(y_true)),
                    "life_ratio_target_over_source": ratios[(source, target)],
                    "pearson_r": pearson_r,
                    "pearson_r_ci95_low": r_low,
                    "pearson_r_ci95_high": r_high,
                    "pearson_p": pearson_p,
                    "alpha": alpha,
                    "beta": beta,
                    "raw_MAE": raw["MAE"],
                    "raw_R2": raw["R2"],
                    "residual_MAE": residual["MAE"],
                    "residual_R2": residual["R2"],
                    "linear_MAE": linear["MAE"],
                    "linear_R2": linear["R2"],
                }
            )
            for cell_id, actual, pred in zip(cell_ids, y_true, y_pred, strict=False):
                pred_rows.append(
                    {
                        "experiment": experiment,
                        "source": source,
                        "target": target,
                        "model": model_name,
                        "seed": int(seed),
                        "cell_id": cell_id,
                        "cycle_life": float(actual),
                        "source_prediction": float(pred),
                    }
                )

    seed_df = pd.DataFrame(seed_rows)
    direction = (
        seed_df.groupby(["experiment", "source", "target", "model"], as_index=False)
        .agg(
            n_target=("n_target", "first"),
            life_ratio_target_over_source=("life_ratio_target_over_source", "first"),
            pearson_r=("pearson_r", "mean"),
            pearson_r_ci95_low=("pearson_r_ci95_low", "mean"),
            pearson_r_ci95_high=("pearson_r_ci95_high", "mean"),
            pearson_p=("pearson_p", "mean"),
            alpha=("alpha", "mean"),
            beta=("beta", "mean"),
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
    direction["rank_signal_class"] = direction["pearson_r"].map(classify_rank_signal)
    direction["predicted_super_regime"] = direction["rank_signal_class"].map(deployment_super_regime)
    direction["adapter_class"] = direction.apply(classify_adapter, axis=1)
    direction["frozen_threshold_match"] = direction.apply(validation_match, axis=1)

    best_out = OUTPUT_DIR / "external_umich_best_cross_models.csv"
    seed_out = OUTPUT_DIR / "external_umich_seed_diagnostics.csv"
    pred_out = OUTPUT_DIR / "external_umich_predictions.csv"
    summary_out = OUTPUT_DIR / "external_umich_direction_summary.csv"
    report_out = INTERMEDIATE_DIR / "external_umich_validation_report.md"

    best.to_csv(best_out, index=False)
    seed_df.to_csv(seed_out, index=False)
    pd.DataFrame(pred_rows).to_csv(pred_out, index=False)
    direction.to_csv(summary_out, index=False)

    within = pd.read_csv(WITHIN_SUMMARY)
    within_best = (
        within[within["n_cycles"].eq(N_CYCLES)]
        .sort_values("R2_mean", ascending=False)
        .head(3)[["model", "MAE_mean", "SMAPE_mean", "R2_mean"]]
    )
    report_cols = [
        "experiment",
        "model",
        "life_ratio_target_over_source",
        "raw_R2",
        "pearson_r",
        "rank_signal_class",
        "linear_R2",
        "predicted_super_regime",
        "frozen_threshold_match",
    ]
    n_match = int(direction["frozen_threshold_match"].sum())
    report = [
        "# UMich Held-Out External Validation",
        "",
        f"Frozen thresholds: `{display_path(THRESHOLD_CONFIG)}`.",
        f"Feature table: `{display_path(FEATURES_PATH)}`.",
        "",
        "UMich is treated as held-out external validation, not as a fifth member of the main benchmark.",
        "",
        "## Within-UMich Baseline",
        "",
        markdown_table(within_best, float_digits=3),
        "",
        "## Eight New External Directions",
        "",
        markdown_table(direction[report_cols], float_digits=3),
        "",
        f"Frozen-threshold agreement: **{n_match}/8 directions**.",
        "",
        "Interpretation: a match means the rank-signal threshold predicted the deployment response class: "
        "strong/moderate should permit meaningful linear k-shot recovery, weak should be offset-dominant, "
        "and collapsed/inverted should remain CP-interval-only for point prediction.",
        "",
        "## Outputs",
        "",
        f"- `{display_path(best_out)}`",
        f"- `{display_path(summary_out)}`",
        f"- `{display_path(seed_out)}`",
        f"- `{display_path(pred_out)}`",
    ]
    report_out.write_text("\n".join(report) + "\n")

    print(f"[save] {display_path(best_out)}")
    print(f"[save] {display_path(summary_out)}")
    print(f"[save] {display_path(seed_out)}")
    print(f"[save] {display_path(pred_out)}")
    print(f"[save] {display_path(report_out)}")
    print(direction[report_cols].to_string(index=False))
    print(f"[match] {n_match}/8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

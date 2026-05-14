"""
Leave-one-dataset-out pooled training and k-shot source-expert adaptation.

This experiment asks the deployment question that follows from the four-dataset
shift results:

    Given a new target dataset and only k labeled target cells, should we
    transfer from a single historical source, pool all non-target sources, or
    adaptively select/weight source experts?

Protocols:
  1. pooled_erm:
       Train one model on the pooled train splits from all non-target datasets.
  2. source_expert_single:
       Train one fixed primary expert per source dataset; score each expert.
  3. source_expert_uniform:
       Average the three source-primary expert predictions.
  4. source_expert_select:
       Use k target labels to select the source expert with lowest calibration
       MAE, then optionally fit a residual/linear target adapter.
  5. source_expert_convex:
       Use k target labels to learn convex weights over the three source
       experts, then optionally fit a residual/linear target adapter.
  6. source_model_select:
       Use k target labels to select the best source+model expert from the
       full source-model pool, then optionally fit a residual/linear adapter.

The source-primary expert mapping is fixed from prior within-dataset winners:
MATR=CatBoost, HUST=Random Forest, Sandia=XGBoost, Luh=Gaussian Process.
The primary source-expert protocols only choose or weight sources. The
source_model_select protocol is the stronger deployment variant: the target
calibration cells choose both which historical source and which source-trained
model family to trust.

Inputs:
    data/intermediate/features_sop12_four_dataset_capnorm.csv
    splits/sop_v2_four_dataset/{dataset}_{seed}.json

Outputs:
    outputs/results_v2_four_dataset_lodo_source_expert/results_detailed.csv
    outputs/results_v2_four_dataset_lodo_source_expert/results_summary.csv
    outputs/results_v2_four_dataset_lodo_source_expert/results_config.json
    data/intermediate/four_dataset_lodo_source_expert_k20.csv
    data/intermediate/four_dataset_lodo_source_expert_report.md

Usage:
    python 3_analysis/lodo_source_expert_transfer.py
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(PROJECT_ROOT / "2_models"))
from metrics_utils import compute_metrics, fit_with_threaded_joblib, to_cycles  # noqa: E402
from run_experiments import (  # noqa: E402
    ALL_MODELS,
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

INTERMEDIATE_DIR = PROJECT_ROOT / "data" / "intermediate"
DEFAULT_FEATURES_PATH = INTERMEDIATE_DIR / "features_sop12_four_dataset_capnorm.csv"
DEFAULT_SPLITS_DIR = PROJECT_ROOT / "splits" / "sop_v2_four_dataset"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "results_v2_four_dataset_lodo_source_expert"
ALL_DATASETS = ["matr", "hust", "sandia", "luh"]
DEFAULT_K_VALUES = [5, 10, 15, 20]
DEFAULT_ADAPTERS = ["none", "residual_mean", "linear"]
SOURCE_PRIMARY_MODEL = {
    "matr": "catboost",
    "hust": "random_forest",
    "sandia": "xgboost",
    "luh": "gaussian_process",
}
FITTERS = {
    "elastic_net": fit_elastic_net,
    "pls": fit_pls,
    "random_forest": fit_random_forest,
    "gaussian_process": fit_gaussian_process,
    "xgboost": fit_xgboost,
    "catboost": fit_catboost,
    "stacking": fit_stacking,
}


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def stable_seed(*parts: object) -> int:
    text = "|".join(str(part) for part in parts)
    return sum((i + 1) * ord(ch) for i, ch in enumerate(text)) % (2**32 - 1)


def load_split(splits_dir: Path, dataset: str, seed: int) -> dict:
    with (splits_dir / f"{dataset}_{seed}.json").open() as f:
        return json.load(f)


def dataset_window(df: pd.DataFrame, dataset: str, n_cycles: int) -> pd.DataFrame:
    return df[(df["dataset"] == dataset) & (df["n_cycles"] == n_cycles) & (df["is_censored"] == 0)].copy()


def train_rows_for_dataset(
    df: pd.DataFrame,
    dataset: str,
    n_cycles: int,
    seed: int,
    splits_dir: Path,
) -> pd.DataFrame:
    sub = dataset_window(df, dataset, n_cycles)
    split = load_split(splits_dir, dataset, seed)
    return sub[sub["cell_id"].isin(split["train"])].copy()


def fit_model_predict(
    train_df: pd.DataFrame,
    target_df: pd.DataFrame,
    feature_cols: list[str],
    model_name: str,
    *,
    seed: int,
    log_target: bool,
) -> np.ndarray:
    fitter = FITTERS[model_name]
    X_train = train_df[feature_cols].to_numpy(dtype=float)
    y_train = train_df["cycle_life"].to_numpy(dtype=float)
    X_target = target_df[feature_cols].to_numpy(dtype=float)
    y_fit = np.log(y_train) if log_target else y_train

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_target_s = scaler.transform(X_target)
    X_train_s = np.clip(np.nan_to_num(X_train_s, nan=0.0, posinf=0.0, neginf=0.0), -1e6, 1e6)
    X_target_s = np.clip(np.nan_to_num(X_target_s, nan=0.0, posinf=0.0, neginf=0.0), -1e6, 1e6)

    result = fit_with_threaded_joblib(fitter, X_train_s, y_fit, seed=seed)
    model = result[0] if isinstance(result, tuple) else result
    return to_cycles(model.predict(X_target_s), log_target=log_target).ravel()


def fit_point_adapter(y_pred_cal: np.ndarray, y_true_cal: np.ndarray, adapter_type: str) -> tuple[float, float]:
    if adapter_type == "none":
        return 1.0, 0.0
    if adapter_type == "residual_mean":
        return 1.0, float(np.mean(y_true_cal - y_pred_cal))
    if adapter_type != "linear":
        raise ValueError(f"unknown adapter_type={adapter_type}")
    if len(y_pred_cal) < 2 or np.std(y_pred_cal) < 1e-12:
        return 1.0, float(np.mean(y_true_cal) - np.mean(y_pred_cal))
    slope, intercept = np.polyfit(y_pred_cal, y_true_cal, 1)
    return float(slope), float(intercept)


def apply_point_adapter(y_pred: np.ndarray, slope: float, intercept: float) -> np.ndarray:
    pred = slope * y_pred + intercept
    return np.clip(np.nan_to_num(pred, nan=1.0, posinf=1e9, neginf=1.0), 1.0, 1e9)


def convex_weight_grid(n_experts: int, step: float) -> np.ndarray:
    if n_experts < 1:
        raise ValueError("convex_weight_grid expects at least one expert")
    if n_experts == 1:
        return np.asarray([[1.0]], dtype=float)
    ticks = np.arange(0.0, 1.0 + step / 2.0, step)
    if n_experts == 2:
        return np.asarray([[w0, max(0.0, 1.0 - w0)] for w0 in ticks], dtype=float)
    if n_experts != 3:
        raise ValueError("convex_weight_grid currently supports 1, 2, or 3 experts")
    weights = []
    for w0 in ticks:
        for w1 in ticks:
            w2 = 1.0 - w0 - w1
            if w2 < -1e-9:
                continue
            if abs(round(w2 / step) * step - w2) > 1e-8:
                continue
            weights.append([w0, w1, max(0.0, w2)])
    return np.asarray(weights, dtype=float)


def choose_convex_weights(pred_matrix_cal: np.ndarray, y_true_cal: np.ndarray, weights: np.ndarray) -> np.ndarray:
    preds = weights @ pred_matrix_cal.T
    mse = np.mean((preds - y_true_cal[None, :]) ** 2, axis=1)
    return weights[int(np.argmin(mse))]


def metrics_row(
    *,
    protocol: str,
    target: str,
    source: str | None,
    source_set: list[str],
    model: str,
    n_cycles: int,
    seed: int,
    k: int,
    repeat: int,
    adapter_type: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    extra: dict | None = None,
) -> dict:
    metrics = compute_metrics(y_true, y_pred)
    row = {
        "protocol": protocol,
        "target": target,
        "source": source or "",
        "source_set": "+".join(source_set),
        "model": model,
        "n_cycles": int(n_cycles),
        "seed": int(seed),
        "k": int(k),
        "repeat": int(repeat),
        "adapter_type": adapter_type,
        "n_test": int(len(y_true)),
        **metrics,
    }
    if extra:
        row.update(extra)
    return row


def aggregate_summary(detailed: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["protocol", "target", "source", "source_set", "model", "n_cycles", "k", "adapter_type"]
    numeric = ["MAE", "SMAPE", "R2", "n_test"]
    grouped = detailed.groupby(group_cols, dropna=False)
    rows = []
    for key, block in grouped:
        row = dict(zip(group_cols, key))
        for col in numeric:
            row[f"{col}_mean"] = float(block[col].mean())
            row[f"{col}_std"] = float(block[col].std(ddof=0))
        row["n_runs"] = int(len(block))
        if "selected_source" in block.columns:
            nonempty = block["selected_source"].replace("", np.nan).dropna()
            row["selected_source_mode"] = str(nonempty.mode().iloc[0]) if not nonempty.empty else ""
        if "selected_model" in block.columns:
            nonempty = block["selected_model"].replace("", np.nan).dropna()
            row["selected_model_mode"] = str(nonempty.mode().iloc[0]) if not nonempty.empty else ""
        for src in ALL_DATASETS:
            col = f"weight_{src}"
            if col in block.columns:
                row[f"{col}_mean"] = float(block[col].mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["target", "protocol", "n_cycles", "k", "adapter_type", "R2_mean"],
        ascending=[True, True, True, True, True, False],
    )


def build_k20_report(summary: pd.DataFrame, k_report: int, out_prefix: str) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    candidates = summary[
        (summary["n_cycles"].eq(100))
        & (
            (summary["k"].eq(0) & summary["adapter_type"].eq("none"))
            | (summary["k"].eq(k_report))
        )
    ].copy()
    best_rows = (
        candidates.sort_values(["target", "R2_mean", "MAE_mean"], ascending=[True, False, True])
        .groupby("target", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    k_sweep = (
        summary[(summary["n_cycles"].eq(100)) & (summary["k"].gt(0))]
        .sort_values(["target", "k", "R2_mean", "MAE_mean"], ascending=[True, True, False, True])
        .groupby(["target", "k"], as_index=False)
        .head(1)
        .reset_index(drop=True)
    )

    lines = [
        f"# Four-Dataset LODO Source-Expert Transfer (k={k_report})",
        "",
        "Best protocol per held-out target at N=100:",
        "",
        "| Target | Protocol | Model | Adapter | k | MAE | sMAPE | R2 | Notes |",
        "|---|---|---|---|---:|---:|---:|---:|---|",
    ]
    for _, row in best_rows.iterrows():
        note_parts = []
        selected = row.get("selected_source_mode", "")
        if isinstance(selected, str) and selected:
            note_parts.append(f"selected={selected}")
        selected_model = row.get("selected_model_mode", "")
        if isinstance(selected_model, str) and selected_model:
            note_parts.append(f"model={selected_model}")
        for src in ALL_DATASETS:
            col = f"weight_{src}_mean"
            if col in row and pd.notna(row[col]) and row[col] > 1e-6:
                note_parts.append(f"w_{src}={row[col]:.2f}")
        lines.append(
            f"| {row['target']} | {row['protocol']} | {row['model']} | {row['adapter_type']} | "
            f"{int(row['k'])} | {row['MAE_mean']:.1f} | {row['SMAPE_mean']:.2f} | "
            f"{row['R2_mean']:.3f} | {'; '.join(note_parts)} |"
        )

    lines.extend([
        "",
        "## Best Protocol by Target-Calibration Size",
        "",
        "| Target | k | Protocol | Model | Adapter | MAE | R2 |",
        "|---|---:|---|---|---|---:|---:|",
    ])
    for _, row in k_sweep.iterrows():
        lines.append(
            f"| {row['target']} | {int(row['k'])} | {row['protocol']} | {row['model']} | "
            f"{row['adapter_type']} | {row['MAE_mean']:.1f} | {row['R2_mean']:.3f} |"
        )

    lines.extend([
        "",
        "Protocol counts in the k-report candidate set:",
        "",
    ])
    counts = candidates.sort_values(["target", "R2_mean"], ascending=[True, False]).groupby("target").head(3)
    for target, block in counts.groupby("target"):
        lines.append(f"## {target}")
        for _, row in block.iterrows():
            lines.append(
                f"- {row['protocol']} / {row['model']} / {row['adapter_type']} k={int(row['k'])}: "
                f"R2={row['R2_mean']:.3f}, MAE={row['MAE_mean']:.1f}"
            )
        lines.append("")

    return best_rows, k_sweep, "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features-path", type=Path, default=DEFAULT_FEATURES_PATH)
    parser.add_argument("--splits-dir", type=Path, default=DEFAULT_SPLITS_DIR)
    parser.add_argument("--datasets", nargs="+", default=ALL_DATASETS, choices=ALL_DATASETS)
    parser.add_argument("--windows", type=int, nargs="+", default=[100])
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--models", nargs="+", default=ALL_MODELS, choices=ALL_MODELS)
    parser.add_argument("--k-values", type=int, nargs="+", default=DEFAULT_K_VALUES)
    parser.add_argument("--adapter-types", nargs="+", default=DEFAULT_ADAPTERS, choices=DEFAULT_ADAPTERS)
    parser.add_argument("--n-repeats", type=int, default=20)
    parser.add_argument("--convex-step", type=float, default=0.05)
    parser.add_argument("--log-target", action="store_true", default=True)
    parser.add_argument("--no-log-target", action="store_false", dest="log_target")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--k-report", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    features_path = resolve_path(args.features_path)
    splits_dir = resolve_path(args.splits_dir)
    out_dir = resolve_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not features_path.exists():
        print(f"[error] missing feature table: {features_path}")
        return 1
    if len(args.datasets) < 2:
        print("[error] LODO needs at least 2 datasets.")
        return 1

    df = pd.read_csv(features_path)
    feature_cols = [col for col in df.columns if col not in META_COLS]
    grid_weights = convex_weight_grid(len(args.datasets) - 1, args.convex_step)
    detailed_rows: list[dict] = []

    print(f"[setup] features_path: {display_path(features_path)}")
    print(f"[setup] splits_dir: {display_path(splits_dir)}")
    print(f"[setup] output_dir: {display_path(out_dir)}")
    print(f"[setup] datasets={args.datasets}, models={args.models}, k_values={args.k_values}")

    for n_cycles in args.windows:
        for target in args.datasets:
            source_set = [dataset for dataset in args.datasets if dataset != target]
            target_df = dataset_window(df, target, n_cycles)
            y_target = target_df["cycle_life"].to_numpy(dtype=float)
            if len(target_df) < max(args.k_values) + 2:
                print(f"[skip] target={target} N={n_cycles}: too few target cells")
                continue

            print(f"\n========== held-out target={target} N={n_cycles} ==========")
            for seed in args.seeds:
                print(f"  seed={seed}")
                source_train_frames = [
                    train_rows_for_dataset(df, source, n_cycles, seed, splits_dir)
                    for source in source_set
                ]
                pooled_train = pd.concat(source_train_frames, ignore_index=True)

                pooled_predictions: dict[str, np.ndarray] = {}
                for model_name in args.models:
                    pred = fit_model_predict(
                        pooled_train,
                        target_df,
                        feature_cols,
                        model_name,
                        seed=seed,
                        log_target=args.log_target,
                    )
                    pooled_predictions[model_name] = pred
                    detailed_rows.append(
                        metrics_row(
                            protocol="pooled_erm",
                            target=target,
                            source=None,
                            source_set=source_set,
                            model=model_name,
                            n_cycles=n_cycles,
                            seed=seed,
                            k=0,
                            repeat=-1,
                            adapter_type="none",
                            y_true=y_target,
                            y_pred=pred,
                        )
                    )

                source_model_predictions: dict[tuple[str, str], np.ndarray] = {}
                expert_predictions: dict[str, np.ndarray] = {}
                for source in source_set:
                    train_df = train_rows_for_dataset(df, source, n_cycles, seed, splits_dir)
                    source_model_names = sorted(set(args.models) | {SOURCE_PRIMARY_MODEL[source]})
                    for model_name in source_model_names:
                        pred = fit_model_predict(
                            train_df,
                            target_df,
                            feature_cols,
                            model_name,
                            seed=seed,
                            log_target=args.log_target,
                        )
                        source_model_predictions[(source, model_name)] = pred
                        if model_name != SOURCE_PRIMARY_MODEL[source]:
                            continue
                        expert_predictions[source] = pred
                        detailed_rows.append(
                            metrics_row(
                                protocol="source_expert_single",
                                target=target,
                                source=source,
                                source_set=[source],
                                model=model_name,
                                n_cycles=n_cycles,
                                seed=seed,
                                k=0,
                                repeat=-1,
                                adapter_type="none",
                                y_true=y_target,
                                y_pred=pred,
                            )
                        )

                expert_sources = list(expert_predictions)
                pred_matrix = np.vstack([expert_predictions[source] for source in expert_sources]).T
                uniform_pred = pred_matrix.mean(axis=1)
                detailed_rows.append(
                    metrics_row(
                        protocol="source_expert_uniform",
                        target=target,
                        source=None,
                        source_set=expert_sources,
                        model="source_primary_experts",
                        n_cycles=n_cycles,
                        seed=seed,
                        k=0,
                        repeat=-1,
                        adapter_type="none",
                        y_true=y_target,
                        y_pred=uniform_pred,
                        extra={f"weight_{source}": 1.0 / len(expert_sources) for source in expert_sources},
                    )
                )

                target_indices = np.arange(len(target_df))
                for k, repeat in product(args.k_values, range(args.n_repeats)):
                    rng = np.random.default_rng(stable_seed("lodo", target, n_cycles, seed, k, repeat))
                    cal_idx = rng.choice(target_indices, size=k, replace=False)
                    test_idx = np.setdiff1d(target_indices, cal_idx)
                    y_cal = y_target[cal_idx]
                    y_test = y_target[test_idx]

                    for model_name, pred in pooled_predictions.items():
                        for adapter_type in args.adapter_types:
                            slope, intercept = fit_point_adapter(pred[cal_idx], y_cal, adapter_type)
                            adapted = apply_point_adapter(pred[test_idx], slope, intercept)
                            detailed_rows.append(
                                metrics_row(
                                    protocol="pooled_erm_kshot",
                                    target=target,
                                    source=None,
                                    source_set=source_set,
                                    model=model_name,
                                    n_cycles=n_cycles,
                                    seed=seed,
                                    k=k,
                                    repeat=repeat,
                                    adapter_type=adapter_type,
                                    y_true=y_test,
                                    y_pred=adapted,
                                    extra={"adapter_slope": slope, "adapter_intercept": intercept},
                                )
                            )

                    cal_mae_by_source = {
                        source: float(np.mean(np.abs(y_cal - expert_predictions[source][cal_idx])))
                        for source in expert_sources
                    }
                    selected_source = min(cal_mae_by_source, key=cal_mae_by_source.get)
                    selected_pred = expert_predictions[selected_source]
                    for adapter_type in args.adapter_types:
                        slope, intercept = fit_point_adapter(selected_pred[cal_idx], y_cal, adapter_type)
                        adapted = apply_point_adapter(selected_pred[test_idx], slope, intercept)
                        detailed_rows.append(
                                metrics_row(
                                    protocol="source_expert_select",
                                    target=target,
                                    source=None,
                                    source_set=expert_sources,
                                    model="source_primary_experts",
                                n_cycles=n_cycles,
                                seed=seed,
                                k=k,
                                repeat=repeat,
                                adapter_type=adapter_type,
                                y_true=y_test,
                                y_pred=adapted,
                                extra={
                                    "selected_source": selected_source,
                                    "calibration_MAE": cal_mae_by_source[selected_source],
                                    "adapter_slope": slope,
                                    "adapter_intercept": intercept,
                                    **{
                                        f"weight_{source}": 1.0 if source == selected_source else 0.0
                                        for source in expert_sources
                                    },
                                },
                            )
                        )

                    cal_mae_by_source_model = {
                        (source, model_name): float(np.mean(np.abs(y_cal - pred[cal_idx])))
                        for (source, model_name), pred in source_model_predictions.items()
                    }
                    selected_source_model = min(cal_mae_by_source_model, key=cal_mae_by_source_model.get)
                    selected_source2, selected_model2 = selected_source_model
                    selected_pred2 = source_model_predictions[selected_source_model]
                    for adapter_type in args.adapter_types:
                        slope, intercept = fit_point_adapter(selected_pred2[cal_idx], y_cal, adapter_type)
                        adapted = apply_point_adapter(selected_pred2[test_idx], slope, intercept)
                        detailed_rows.append(
                            metrics_row(
                                protocol="source_model_select",
                                target=target,
                                source=None,
                                source_set=expert_sources,
                                model="source_model_experts",
                                n_cycles=n_cycles,
                                seed=seed,
                                k=k,
                                repeat=repeat,
                                adapter_type=adapter_type,
                                y_true=y_test,
                                y_pred=adapted,
                                extra={
                                    "selected_source": selected_source2,
                                    "selected_model": selected_model2,
                                    "calibration_MAE": cal_mae_by_source_model[selected_source_model],
                                    "adapter_slope": slope,
                                    "adapter_intercept": intercept,
                                    **{
                                        f"weight_{source}": 1.0 if source == selected_source2 else 0.0
                                        for source in expert_sources
                                    },
                                },
                            )
                        )

                    best_weights = choose_convex_weights(pred_matrix[cal_idx], y_cal, grid_weights)
                    convex_pred = pred_matrix @ best_weights
                    for adapter_type in args.adapter_types:
                        slope, intercept = fit_point_adapter(convex_pred[cal_idx], y_cal, adapter_type)
                        adapted = apply_point_adapter(convex_pred[test_idx], slope, intercept)
                        detailed_rows.append(
                            metrics_row(
                                protocol="source_expert_convex",
                                target=target,
                                source=None,
                                source_set=expert_sources,
                                model="source_primary_experts",
                                n_cycles=n_cycles,
                                seed=seed,
                                k=k,
                                repeat=repeat,
                                adapter_type=adapter_type,
                                y_true=y_test,
                                y_pred=adapted,
                                extra={
                                    "adapter_slope": slope,
                                    "adapter_intercept": intercept,
                                    **{
                                        f"weight_{source}": float(weight)
                                        for source, weight in zip(expert_sources, best_weights)
                                    },
                                },
                            )
                        )

    detailed = pd.DataFrame(detailed_rows)
    summary = aggregate_summary(detailed)
    best_k, k_sweep, report = build_k20_report(summary, args.k_report, "four_dataset_lodo_source_expert")

    detail_path = out_dir / "results_detailed.csv"
    summary_path = out_dir / "results_summary.csv"
    config_path = out_dir / "results_config.json"
    paper_dir = INTERMEDIATE_DIR if out_dir == resolve_path(DEFAULT_OUTPUT_DIR) else out_dir
    best_path = paper_dir / f"four_dataset_lodo_source_expert_k{args.k_report}.csv"
    k_sweep_path = paper_dir / "four_dataset_lodo_source_expert_k_sweep.csv"
    report_path = paper_dir / "four_dataset_lodo_source_expert_report.md"

    detailed.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    best_k.to_csv(best_path, index=False)
    k_sweep.to_csv(k_sweep_path, index=False)
    report_path.write_text(report)
    with config_path.open("w") as f:
        json.dump(
            {
                "protocol": "four_dataset_lodo_source_expert_v1",
                "features_path": display_path(features_path),
                "splits_dir": display_path(splits_dir),
                "datasets": args.datasets,
                "windows": args.windows,
                "seeds": args.seeds,
                "models": args.models,
                "source_primary_model": SOURCE_PRIMARY_MODEL,
                "k_values": args.k_values,
                "adapter_types": args.adapter_types,
                "n_repeats": args.n_repeats,
                "convex_step": args.convex_step,
                "log_target": bool(args.log_target),
            },
            f,
            indent=2,
        )

    print(f"[save] {display_path(detail_path)}")
    print(f"[save] {display_path(summary_path)}")
    print(f"[save] {display_path(config_path)}")
    print(f"[save] {display_path(best_path)}")
    print(f"[save] {display_path(k_sweep_path)}")
    print(f"[save] {display_path(report_path)}")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

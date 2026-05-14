#!/usr/bin/env python3
"""Small dependency-light 1D-CNN baseline for early-capacity trajectories.

This is a reviewer-facing deep-learning sanity check, not a new headline
architecture. It uses the same four-dataset split protocol as the tabular
experiments, but consumes early capacity sequences directly:

    channel 0: Q_discharge / q0 over cycles 2..N
    channel 1: first difference of channel 0

The network is implemented in NumPy to avoid adding a heavyweight Torch or
TensorFlow dependency to the reproducibility stack:

    Conv1D -> ReLU -> temporal average pooling + global pooling -> dense ReLU
    -> scalar log-life, clipped to the source train log-life range by default

Outputs:
    outputs/results_v2_four_dataset_cnn_baseline/results_detailed.csv
    outputs/results_v2_four_dataset_cnn_baseline/results_predictions.csv
    outputs/results_v2_four_dataset_cnn_baseline/results_summary.csv
    data/intermediate/four_dataset_cnn_baseline_report.md

Usage:
    python 2_models/run_cnn_baseline.py
    python 2_models/run_cnn_baseline.py --datasets sandia luh --seeds 42
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from metrics_utils import bootstrap_metric_ci, compute_metrics, to_cycles  # noqa: E402


FEATURES_PATH = PROJECT_ROOT / "data/intermediate/features_sop12_four_dataset.csv"
SPLITS_DIR = PROJECT_ROOT / "splits/sop_v2_four_dataset"
OUTPUT_DIR = PROJECT_ROOT / "outputs/results_v2_four_dataset_cnn_baseline"
INTERMEDIATE_DIR = PROJECT_ROOT / "data/intermediate"

ALL_DATASETS = ["matr", "hust", "sandia", "luh"]
SEEDS = [42, 123, 456, 789, 1011]
DEFAULT_WINDOWS = [100]
META_COLS = {"dataset", "cell_id", "n_cycles", "q0", "cycle_life", "is_censored", "capacity_normalized"}


@dataclass
class TrainInfo:
    best_epoch: int
    epochs_run: int
    best_val_loss: float
    final_train_loss: float


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def dataframe_to_markdown(df: pd.DataFrame, *, floatfmt: str = ".3f") -> str:
    headers = [str(c) for c in df.columns]
    formatted_rows = []
    for row in df.itertuples(index=False):
        formatted = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                formatted.append(format(float(value), floatfmt))
            else:
                formatted.append(str(value))
        formatted_rows.append(formatted)
    widths = [
        max([len(headers[i])] + [len(row[i]) for row in formatted_rows])
        for i in range(len(headers))
    ]
    header = "| " + " | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))) + " |"
    sep = "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |"
    body = [
        "| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) + " |"
        for row in formatted_rows
    ]
    return "\n".join([header, sep, *body])


def load_split(splits_dir: Path, dataset: str, seed: int) -> dict:
    path = splits_dir / f"{dataset}_{seed}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")
    return json.loads(path.read_text())


def cycle_table_path(dataset: str) -> Path:
    return PROJECT_ROOT / f"data/intermediate/{dataset}_cycles_tidy.csv"


def load_cycle_tables(datasets: list[str]) -> pd.DataFrame:
    frames = []
    for dataset in datasets:
        path = cycle_table_path(dataset)
        if not path.exists():
            raise FileNotFoundError(f"Missing cycle table: {path}")
        df = pd.read_csv(path, usecols=["cell_id", "cycle", "Q_discharge"])
        df["dataset"] = dataset
        if dataset == "hust":
            df["cell_id"] = "hust_" + df["cell_id"].astype(str)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["cycle"] = pd.to_numeric(out["cycle"], errors="coerce")
    out["Q_discharge"] = pd.to_numeric(out["Q_discharge"], errors="coerce")
    out = out.dropna(subset=["cycle", "Q_discharge"])
    return out


def build_sequence_dataset(
    feature_df: pd.DataFrame,
    cycles_df: pd.DataFrame,
    *,
    datasets: list[str],
    n_cycles: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Return X=(cells, channels, cycles 2..N), y=cycle_life, and metadata."""
    feature_subset = feature_df[
        feature_df["dataset"].isin(datasets)
        & feature_df["n_cycles"].eq(n_cycles)
        & feature_df["is_censored"].eq(0)
    ].copy()
    required_cycles = np.arange(2, n_cycles + 1, dtype=float)
    grouped = {
        (dataset, cell_id): block.sort_values("cycle")
        for (dataset, cell_id), block in cycles_df.groupby(["dataset", "cell_id"], sort=False)
    }

    seqs: list[np.ndarray] = []
    rows: list[dict] = []
    skipped: list[str] = []
    for row in feature_subset.itertuples(index=False):
        key = (str(row.dataset), str(row.cell_id))
        block = grouped.get(key)
        if block is None or len(block) < 2:
            skipped.append(str(row.cell_id))
            continue
        cycle = block["cycle"].to_numpy(dtype=float)
        q = block["Q_discharge"].to_numpy(dtype=float)
        order = np.argsort(cycle)
        cycle = cycle[order]
        q = q[order]
        uniq_cycle, uniq_idx = np.unique(cycle, return_index=True)
        q = q[uniq_idx]
        if uniq_cycle[0] > required_cycles[0] or uniq_cycle[-1] < required_cycles[-1]:
            skipped.append(str(row.cell_id))
            continue
        q_interp = np.interp(required_cycles, uniq_cycle, q)
        q0 = float(row.q0)
        if not np.isfinite(q0) or q0 <= 0:
            skipped.append(str(row.cell_id))
            continue
        retention = q_interp / q0
        diff = np.concatenate([[0.0], np.diff(retention)])
        seq = np.stack([retention, diff], axis=0)
        seqs.append(seq.astype(float))
        rows.append(
            {
                "dataset": str(row.dataset),
                "cell_id": str(row.cell_id),
                "n_cycles": int(n_cycles),
                "q0": q0,
                "cycle_life": float(row.cycle_life),
            }
        )

    if skipped:
        print(f"[warn] skipped {len(skipped)} cells at N={n_cycles}: {skipped[:5]}")
    if not seqs:
        raise ValueError(f"No sequence rows built for N={n_cycles}")
    return np.stack(seqs, axis=0), np.asarray([r["cycle_life"] for r in rows], dtype=float), pd.DataFrame(rows)


def fit_sequence_scaler(X_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = X_train.mean(axis=(0, 2), keepdims=True)
    std = X_train.std(axis=(0, 2), keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return mean, std


def apply_sequence_scaler(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    out = (X - mean) / std
    return np.clip(np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0), -20.0, 20.0)


def pooled_length(length: int, pool_size: int) -> int:
    return int(np.ceil(length / pool_size))


def init_params(
    rng: np.random.Generator,
    *,
    channels: int,
    length: int,
    filters: int,
    kernel: int,
    hidden: int,
    pool_size: int,
) -> dict:
    pooled_features = filters * pooled_length(length, pool_size) + 2 * filters
    return {
        "Wc": rng.normal(0.0, np.sqrt(2.0 / (channels * kernel)), size=(filters, channels, kernel)),
        "bc": np.zeros(filters),
        "W1": rng.normal(0.0, np.sqrt(2.0 / pooled_features), size=(pooled_features, hidden)),
        "b1": np.zeros(hidden),
        "W2": rng.normal(0.0, np.sqrt(2.0 / hidden), size=(hidden, 1)),
        "b2": np.zeros(1),
    }


def conv1d_same(X: np.ndarray, W: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    batch, channels, length = X.shape
    filters, _, kernel = W.shape
    pad_left = kernel // 2
    pad_right = kernel - 1 - pad_left
    Xp = np.pad(X, ((0, 0), (0, 0), (pad_left, pad_right)), mode="constant")
    Z = np.empty((batch, filters, length), dtype=float)
    for t in range(length):
        patch = Xp[:, :, t : t + kernel]
        Z[:, :, t] = np.einsum("bck,fck->bf", patch, W) + b
    return Z, Xp


def average_pool1d(A: np.ndarray, pool_size: int) -> tuple[np.ndarray, list[tuple[int, int]]]:
    length = A.shape[2]
    segments = []
    blocks = []
    for start in range(0, length, pool_size):
        end = min(length, start + pool_size)
        segments.append((start, end))
        blocks.append(A[:, :, start:end].mean(axis=2))
    return np.stack(blocks, axis=2), segments


def forward(params: dict, X: np.ndarray, *, pool_size: int) -> tuple[np.ndarray, dict]:
    Z, Xp = conv1d_same(X, params["Wc"], params["bc"])
    A = np.maximum(Z, 0.0)
    pooled_temporal, pool_segments = average_pool1d(A, pool_size)
    pooled_flat = pooled_temporal.reshape(X.shape[0], -1)
    mean_pool = A.mean(axis=2)
    max_idx = A.argmax(axis=2)
    max_pool = A.max(axis=2)
    pooled = np.concatenate([pooled_flat, mean_pool, max_pool], axis=1)
    Hpre = pooled @ params["W1"] + params["b1"]
    H = np.maximum(Hpre, 0.0)
    pred = (H @ params["W2"] + params["b2"]).ravel()
    cache = {
        "X": X,
        "Xp": Xp,
        "Z": Z,
        "A": A,
        "max_idx": max_idx,
        "pool_segments": pool_segments,
        "pooled_temporal_shape": pooled_temporal.shape,
        "pooled": pooled,
        "Hpre": Hpre,
        "H": H,
    }
    return pred, cache


def loss_value(pred: np.ndarray, y: np.ndarray, params: dict, l2: float) -> float:
    mse = float(np.mean((pred - y) ** 2))
    reg = 0.0
    for key in ["Wc", "W1", "W2"]:
        reg += float(np.sum(params[key] ** 2))
    return mse + l2 * reg


def backward(params: dict, cache: dict, pred: np.ndarray, y: np.ndarray, l2: float) -> dict:
    X = cache["X"]
    Xp = cache["Xp"]
    Z = cache["Z"]
    A = cache["A"]
    max_idx = cache["max_idx"]
    pool_segments = cache["pool_segments"]
    pooled_temporal_shape = cache["pooled_temporal_shape"]
    pooled = cache["pooled"]
    Hpre = cache["Hpre"]
    H = cache["H"]
    batch, channels, length = X.shape
    filters, _, kernel = params["Wc"].shape

    dpred = (2.0 / batch) * (pred - y)
    grads = {
        "W2": H.T @ dpred[:, None] + 2.0 * l2 * params["W2"],
        "b2": np.asarray([dpred.sum()]),
    }
    dH = dpred[:, None] @ params["W2"].T
    dHpre = dH * (Hpre > 0.0)
    grads["W1"] = pooled.T @ dHpre + 2.0 * l2 * params["W1"]
    grads["b1"] = dHpre.sum(axis=0)
    dpooled = dHpre @ params["W1"].T
    pooled_flat_size = int(np.prod(pooled_temporal_shape[1:]))
    dpooled_temporal = dpooled[:, :pooled_flat_size].reshape(pooled_temporal_shape)
    dmean = dpooled[:, pooled_flat_size : pooled_flat_size + filters]
    dmax = dpooled[:, pooled_flat_size + filters :]

    dA = np.repeat((dmean / length)[:, :, None], length, axis=2)
    for p, (start, end) in enumerate(pool_segments):
        dA[:, :, start:end] += dpooled_temporal[:, :, p][:, :, None] / (end - start)
    for i in range(batch):
        for f in range(filters):
            dA[i, f, int(max_idx[i, f])] += dmax[i, f]
    dZ = dA * (Z > 0.0)

    dWc = np.zeros_like(params["Wc"])
    dbc = dZ.sum(axis=(0, 2))
    for t in range(length):
        patch = Xp[:, :, t : t + kernel]
        dWc += np.einsum("bf,bck->fck", dZ[:, :, t], patch)
    grads["Wc"] = dWc + 2.0 * l2 * params["Wc"]
    grads["bc"] = dbc
    return grads


def adam_update(params: dict, grads: dict, state: dict, *, lr: float, t: int) -> None:
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    for key, grad in grads.items():
        state["m"][key] = beta1 * state["m"][key] + (1.0 - beta1) * grad
        state["v"][key] = beta2 * state["v"][key] + (1.0 - beta2) * (grad * grad)
        m_hat = state["m"][key] / (1.0 - beta1**t)
        v_hat = state["v"][key] / (1.0 - beta2**t)
        params[key] -= lr * m_hat / (np.sqrt(v_hat) + eps)


def train_cnn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    seed: int,
    filters: int,
    kernel: int,
    hidden: int,
    epochs: int,
    patience: int,
    batch_size: int,
    lr: float,
    l2: float,
    pool_size: int,
) -> tuple[dict, TrainInfo]:
    rng = np.random.default_rng(seed)
    params = init_params(
        rng,
        channels=X_train.shape[1],
        length=X_train.shape[2],
        filters=filters,
        kernel=kernel,
        hidden=hidden,
        pool_size=pool_size,
    )
    state = {"m": {k: np.zeros_like(v) for k, v in params.items()}, "v": {k: np.zeros_like(v) for k, v in params.items()}}
    best_params = {k: v.copy() for k, v in params.items()}
    best_val = float("inf")
    best_epoch = 0
    wait = 0
    step = 0
    n = len(X_train)

    for epoch in range(1, epochs + 1):
        order = rng.permutation(n)
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            pred, cache = forward(params, X_train[idx], pool_size=pool_size)
            grads = backward(params, cache, pred, y_train[idx], l2)
            step += 1
            adam_update(params, grads, state, lr=lr, t=step)

        train_pred, _ = forward(params, X_train, pool_size=pool_size)
        val_pred, _ = forward(params, X_val if len(X_val) else X_train, pool_size=pool_size)
        val_y = y_val if len(y_val) else y_train
        train_loss = loss_value(train_pred, y_train, params, l2)
        val_loss = loss_value(val_pred, val_y, params, l2)
        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_epoch = epoch
            best_params = {k: v.copy() for k, v in params.items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    final_pred, _ = forward(best_params, X_train, pool_size=pool_size)
    return best_params, TrainInfo(
        best_epoch=int(best_epoch),
        epochs_run=int(epoch),
        best_val_loss=float(best_val),
        final_train_loss=loss_value(final_pred, y_train, best_params, l2),
    )


def predict_cycles(
    params: dict,
    X: np.ndarray,
    *,
    y_mean: float,
    y_std: float,
    pool_size: int,
    log_lower: float,
    log_upper: float,
) -> np.ndarray:
    pred_std, _ = forward(params, X, pool_size=pool_size)
    pred_log = np.clip(pred_std * y_std + y_mean, log_lower, log_upper)
    return to_cycles(pred_log, log_target=True, min_cycle=1.0, max_cycle=1e9)


def subset_indices(meta: pd.DataFrame, cells: list[str]) -> np.ndarray:
    mask = meta["cell_id"].isin(cells).to_numpy()
    return np.flatnonzero(mask)


def evaluate_within(
    X: np.ndarray,
    y: np.ndarray,
    meta: pd.DataFrame,
    split: dict,
    *,
    dataset: str,
    n_cycles: int,
    seed: int,
    args: argparse.Namespace,
) -> tuple[dict, list[dict]]:
    train_idx = subset_indices(meta, split["train"])
    cal_idx = subset_indices(meta, split["calibration"])
    test_idx = subset_indices(meta, split["test"])
    if len(train_idx) < 5 or len(test_idx) < 2:
        raise ValueError(f"Too few cells for within {dataset} seed={seed}")

    x_mean, x_std = fit_sequence_scaler(X[train_idx])
    X_train = apply_sequence_scaler(X[train_idx], x_mean, x_std)
    X_cal = apply_sequence_scaler(X[cal_idx], x_mean, x_std)
    X_test = apply_sequence_scaler(X[test_idx], x_mean, x_std)
    y_train_log = np.log(y[train_idx])
    y_mean = float(y_train_log.mean())
    y_std = float(y_train_log.std() if y_train_log.std() > 1e-8 else 1.0)
    y_train_fit = (y_train_log - y_mean) / y_std
    y_cal_fit = (np.log(y[cal_idx]) - y_mean) / y_std
    clip_margin = args.prediction_clip_std_margin * y_std
    log_lower = float(y_train_log.min() - clip_margin)
    log_upper = float(y_train_log.max() + clip_margin)

    params, info = train_cnn(
        X_train,
        y_train_fit,
        X_cal,
        y_cal_fit,
        seed=seed,
        filters=args.filters,
        kernel=args.kernel_size,
        hidden=args.hidden_units,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        lr=args.learning_rate,
        l2=args.l2,
        pool_size=args.pool_size,
    )
    pred = predict_cycles(
        params,
        X_test,
        y_mean=y_mean,
        y_std=y_std,
        pool_size=args.pool_size,
        log_lower=log_lower,
        log_upper=log_upper,
    )
    metrics = compute_metrics(y[test_idx], pred)
    row = {
        "experiment": f"{dataset}_to_{dataset}",
        "scenario": "within_split",
        "source": dataset,
        "target": dataset,
        "model": "numpy_1d_cnn",
        "n_cycles": int(n_cycles),
        "seed": int(seed),
        "train_cells": int(len(train_idx)),
        "calibration_cells": int(len(cal_idx)),
        "test_cells": int(len(test_idx)),
        **metrics,
        "best_epoch": info.best_epoch,
        "epochs_run": info.epochs_run,
        "best_val_loss": info.best_val_loss,
        "final_train_loss": info.final_train_loss,
        "prediction_log_lower": log_lower,
        "prediction_log_upper": log_upper,
    }
    pred_rows = prediction_rows(meta.iloc[test_idx], y[test_idx], pred, row)
    return row, pred_rows


def evaluate_cross(
    X: np.ndarray,
    y: np.ndarray,
    meta: pd.DataFrame,
    source_split: dict,
    *,
    source: str,
    target: str,
    n_cycles: int,
    seed: int,
    args: argparse.Namespace,
) -> tuple[dict, list[dict]]:
    train_idx = subset_indices(meta[(meta["dataset"] == source)], source_split["train"])
    source_positions = np.flatnonzero(meta["dataset"].eq(source).to_numpy())
    train_idx = source_positions[train_idx]
    cal_idx = subset_indices(meta[(meta["dataset"] == source)], source_split["calibration"])
    cal_idx = source_positions[cal_idx]
    test_idx = np.flatnonzero(meta["dataset"].eq(target).to_numpy())
    if len(train_idx) < 5 or len(test_idx) < 2:
        raise ValueError(f"Too few cells for cross {source}->{target} seed={seed}")

    x_mean, x_std = fit_sequence_scaler(X[train_idx])
    X_train = apply_sequence_scaler(X[train_idx], x_mean, x_std)
    X_cal = apply_sequence_scaler(X[cal_idx], x_mean, x_std)
    X_test = apply_sequence_scaler(X[test_idx], x_mean, x_std)
    y_train_log = np.log(y[train_idx])
    y_mean = float(y_train_log.mean())
    y_std = float(y_train_log.std() if y_train_log.std() > 1e-8 else 1.0)
    y_train_fit = (y_train_log - y_mean) / y_std
    y_cal_fit = (np.log(y[cal_idx]) - y_mean) / y_std
    clip_margin = args.prediction_clip_std_margin * y_std
    log_lower = float(y_train_log.min() - clip_margin)
    log_upper = float(y_train_log.max() + clip_margin)

    params, info = train_cnn(
        X_train,
        y_train_fit,
        X_cal,
        y_cal_fit,
        seed=seed,
        filters=args.filters,
        kernel=args.kernel_size,
        hidden=args.hidden_units,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        lr=args.learning_rate,
        l2=args.l2,
        pool_size=args.pool_size,
    )
    pred = predict_cycles(
        params,
        X_test,
        y_mean=y_mean,
        y_std=y_std,
        pool_size=args.pool_size,
        log_lower=log_lower,
        log_upper=log_upper,
    )
    metrics = compute_metrics(y[test_idx], pred)
    row = {
        "experiment": f"{source}_to_{target}",
        "scenario": "naive_cross",
        "source": source,
        "target": target,
        "model": "numpy_1d_cnn",
        "n_cycles": int(n_cycles),
        "seed": int(seed),
        "train_cells": int(len(train_idx)),
        "calibration_cells": int(len(cal_idx)),
        "test_cells": int(len(test_idx)),
        **metrics,
        "best_epoch": info.best_epoch,
        "epochs_run": info.epochs_run,
        "best_val_loss": info.best_val_loss,
        "final_train_loss": info.final_train_loss,
        "prediction_log_lower": log_lower,
        "prediction_log_upper": log_upper,
    }
    pred_rows = prediction_rows(meta.iloc[test_idx], y[test_idx], pred, row)
    return row, pred_rows


def prediction_rows(meta: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray, base_row: dict) -> list[dict]:
    rows = []
    for cell_id, dataset, true, pred in zip(meta["cell_id"], meta["dataset"], y_true, y_pred):
        rows.append(
            {
                "experiment": base_row["experiment"],
                "scenario": base_row["scenario"],
                "source": base_row["source"],
                "target": base_row["target"],
                "model": base_row["model"],
                "n_cycles": base_row["n_cycles"],
                "seed": base_row["seed"],
                "dataset": dataset,
                "cell_id": cell_id,
                "y_true": float(true),
                "y_pred": float(pred),
            }
        )
    return rows


def aggregate_summary(detailed: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["scenario", "experiment", "source", "target", "model", "n_cycles"]
    rows = []
    for keys, block in detailed.groupby(group_cols, sort=True):
        row = dict(zip(group_cols, keys))
        for metric in ["MAE", "SMAPE", "R2"]:
            row[f"{metric}_mean"] = float(block[metric].mean())
            row[f"{metric}_std"] = float(block[metric].std(ddof=1)) if len(block) > 1 else 0.0
        row["train_cells_mean"] = float(block["train_cells"].mean())
        row["calibration_cells_mean"] = float(block["calibration_cells"].mean())
        row["test_cells_mean"] = float(block["test_cells"].mean())
        row["best_epoch_mean"] = float(block["best_epoch"].mean())
        row["n_runs"] = int(len(block))

        pred_block = predictions
        for col, val in row.items():
            if col in group_cols:
                pred_block = pred_block[pred_block[col].eq(val)]
        if len(pred_block) >= 3:
            ci = bootstrap_metric_ci(
                pred_block["y_true"].to_numpy(dtype=float),
                pred_block["y_pred"].to_numpy(dtype=float),
                seed=42,
            )
            for metric in ["MAE", "SMAPE", "R2"]:
                row[f"{metric}_pooled_ci95_lower"] = ci[metric]["lower"]
                row[f"{metric}_pooled_ci95_upper"] = ci[metric]["upper"]
            row["pooled_prediction_rows"] = int(len(pred_block))
            row["pooled_distinct_cells"] = int(pred_block["cell_id"].nunique())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def build_report(summary: pd.DataFrame) -> str:
    within = summary[summary["scenario"].eq("within_split")].copy()
    cross = summary[summary["scenario"].eq("naive_cross")].copy()
    lines = [
        "# Four-Dataset 1D-CNN Baseline",
        "",
        "Model: dependency-light NumPy Conv1D -> ReLU -> temporal average pooling + global mean/max pooling -> dense ReLU.",
        "Input: cycles 2..N, channels `[Q_discharge/q0, diff(Q_discharge/q0)]`; target: standardized log(cycle_life).",
        "",
        "## Within-Dataset N=100",
        "",
    ]
    if len(within):
        table = within[["target", "MAE_mean", "SMAPE_mean", "R2_mean", "R2_pooled_ci95_lower", "R2_pooled_ci95_upper"]].copy()
        lines.append(dataframe_to_markdown(table))
    lines += ["", "## Naive Cross-Dataset N=100", ""]
    if len(cross):
        best = cross.sort_values(["target", "R2_mean"], ascending=[True, False]).groupby("target", as_index=False).head(3)
        table = best[["experiment", "MAE_mean", "SMAPE_mean", "R2_mean"]].copy()
        lines.append(dataframe_to_markdown(table))
    lines += [
        "",
        "Interpretation: use this as a deep-learning baseline only. The main paper claim remains the transfer-regime and calibration protocol, not CNN architecture novelty.",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-path", type=Path, default=FEATURES_PATH)
    parser.add_argument("--splits-dir", type=Path, default=SPLITS_DIR)
    parser.add_argument("--datasets", nargs="+", default=ALL_DATASETS, choices=ALL_DATASETS)
    parser.add_argument("--windows", type=int, nargs="+", default=DEFAULT_WINDOWS)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--within-only", action="store_true")
    parser.add_argument("--cross-only", action="store_true")
    parser.add_argument("--filters", type=int, default=16)
    parser.add_argument("--kernel-size", type=int, default=5)
    parser.add_argument("--pool-size", type=int, default=4)
    parser.add_argument("--hidden-units", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=700)
    parser.add_argument("--patience", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument(
        "--prediction-clip-std-margin",
        type=float,
        default=0.0,
        help="Clip predicted log-life to source train min/max plus this many train log-life stds.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    features_path = resolve_path(args.features_path)
    splits_dir = resolve_path(args.splits_dir)
    out_dir = resolve_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.within_only and args.cross_only:
        raise SystemExit("--within-only and --cross-only are mutually exclusive")

    feature_df = pd.read_csv(features_path)
    cycles_df = load_cycle_tables(args.datasets)
    detailed_rows = []
    pred_rows = []

    print(f"[setup] features_path: {display_path(features_path)}")
    print(f"[setup] splits_dir: {display_path(splits_dir)}")
    print(f"[setup] output_dir: {display_path(out_dir)}")
    print(f"[setup] datasets={args.datasets}, windows={args.windows}, seeds={args.seeds}")
    print("[setup] model=numpy_1d_cnn channels=[retention, diff], log-target=True")

    for n_cycles in args.windows:
        X, y, meta = build_sequence_dataset(feature_df, cycles_df, datasets=args.datasets, n_cycles=n_cycles)
        for seed in args.seeds:
            if not args.cross_only:
                for dataset in args.datasets:
                    split = load_split(splits_dir, dataset, seed)
                    dataset_idx = np.flatnonzero(meta["dataset"].eq(dataset).to_numpy())
                    print(f"within {dataset} seed={seed} N={n_cycles}")
                    row, preds = evaluate_within(
                        X[dataset_idx],
                        y[dataset_idx],
                        meta.iloc[dataset_idx].reset_index(drop=True),
                        split,
                        dataset=dataset,
                        n_cycles=n_cycles,
                        seed=seed,
                        args=args,
                    )
                    detailed_rows.append(row)
                    pred_rows.extend(preds)
            if not args.within_only:
                for source in args.datasets:
                    source_split = load_split(splits_dir, source, seed)
                    for target in args.datasets:
                        if source == target:
                            continue
                        print(f"cross {source}->{target} seed={seed} N={n_cycles}")
                        row, preds = evaluate_cross(
                            X,
                            y,
                            meta,
                            source_split,
                            source=source,
                            target=target,
                            n_cycles=n_cycles,
                            seed=seed,
                            args=args,
                        )
                        detailed_rows.append(row)
                        pred_rows.extend(preds)

    detailed = pd.DataFrame(detailed_rows)
    predictions = pd.DataFrame(pred_rows)
    summary = aggregate_summary(detailed, predictions)

    detail_path = out_dir / "results_detailed.csv"
    pred_path = out_dir / "results_predictions.csv"
    summary_path = out_dir / "results_summary.csv"
    config_path = out_dir / "results_config.json"
    paper_dir = INTERMEDIATE_DIR if out_dir == resolve_path(OUTPUT_DIR) else out_dir
    report_path = paper_dir / "four_dataset_cnn_baseline_report.md"

    detailed.to_csv(detail_path, index=False)
    predictions.to_csv(pred_path, index=False)
    summary.to_csv(summary_path, index=False)
    report_path.write_text(build_report(summary))
    with config_path.open("w") as f:
        json.dump(
            {
                "protocol": "four_dataset_numpy_1d_cnn_baseline_v1",
                "features_path": display_path(features_path),
                "splits_dir": display_path(splits_dir),
                "datasets": args.datasets,
                "windows": args.windows,
                "seeds": args.seeds,
                "input_cycles": "2..N",
                "channels": ["Q_discharge/q0", "first_difference_Q_discharge/q0"],
                "architecture": "Conv1D-ReLU-global_mean_max_pool-dense_ReLU-output",
                "filters": args.filters,
                "kernel_size": args.kernel_size,
                "pool_size": args.pool_size,
                "hidden_units": args.hidden_units,
                "epochs": args.epochs,
                "patience": args.patience,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "l2": args.l2,
                "prediction_clip_std_margin": args.prediction_clip_std_margin,
            },
            f,
            indent=2,
        )
    for path in [detail_path, pred_path, summary_path, config_path, report_path]:
        print(f"[save] {display_path(path)}")
    print(summary[["scenario", "experiment", "MAE_mean", "SMAPE_mean", "R2_mean", "n_runs"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

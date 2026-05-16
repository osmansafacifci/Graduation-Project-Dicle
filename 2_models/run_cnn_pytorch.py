#!/usr/bin/env python3
"""PyTorch 1D-CNN baseline with inner-CV HP search and resumable checkpointing.

Replacement for the NumPy `run_cnn_baseline.py`. Same data contract, same
splits, same output schema (so downstream scripts keep working), but with:

  - PyTorch backend with automatic MPS / CUDA / CPU device selection
  - Inner 5-fold CV hyperparameter search over (filters, learning_rate),
    parallelling the XGBoost/CatBoost grid in run_experiments.py
  - Two-layer 1D-CNN with dropout (slightly less shallow than the NumPy version)
  - Cluster bootstrap by cell_id for pooled CI (fixes the row-level bootstrap
    inflation in cross-dataset, where the same target cells are evaluated
    across all seeds)
  - Per-unit checkpointing: after each (dataset, seed) within-result or
    (source -> target, seed) cross-result is computed, its JSON is written
    immediately to `<output-dir>/checkpoints/`. On restart, completed units
    are skipped. So if MPS crashes, you lose at most one unit of work.

Architecture (~2.5K parameters, sized for n <= 135 cells/dataset):

    Conv1D(2 -> F, k=5, pad=same) -> ReLU
    Conv1D(F -> F, k=5, pad=same) -> ReLU
    AdaptiveAvgPool1d(1) + AdaptiveMaxPool1d(1) -> concat (2F-dim)
    Linear(2F -> H) -> ReLU -> Dropout(0.2)
    Linear(H -> 1)

Inputs:
    data/intermediate/features_sop12_four_dataset.csv
    data/intermediate/<dataset>_cycles_tidy.csv
    splits/sop_v2_four_dataset/<dataset>_<seed>.json

Outputs:
    outputs/results_v2_four_dataset_cnn_pytorch/checkpoints/<unit>.json
    outputs/results_v2_four_dataset_cnn_pytorch/results_detailed.csv
    outputs/results_v2_four_dataset_cnn_pytorch/results_predictions.csv
    outputs/results_v2_four_dataset_cnn_pytorch/results_summary.csv
    outputs/results_v2_four_dataset_cnn_pytorch/results_config.json
    data/intermediate/four_dataset_cnn_pytorch_report.md

Usage:
    python 2_models/run_cnn_pytorch.py
    python 2_models/run_cnn_pytorch.py --datasets sandia luh --seeds 42
    python 2_models/run_cnn_pytorch.py --hp-grid quick  # 4 configs instead of 9
    python 2_models/run_cnn_pytorch.py --aggregate-only  # rebuild CSVs from checkpoints
    python 2_models/run_cnn_pytorch.py --force-rerun     # ignore existing checkpoints

Resume semantics:
    Re-running the script with the same arguments skips any unit whose
    checkpoint file already exists. To rerun a unit, delete its checkpoint
    file (or pass --force-rerun to wipe all). To rebuild the summary CSVs
    without retraining, pass --aggregate-only.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as exc:
    raise SystemExit(
        "PyTorch is required. On M1: `pip install --no-cache-dir torch`\n"
        f"(import failed with: {exc})"
    )

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from metrics_utils import compute_metrics, to_cycles  # noqa: E402


# -------------------- constants --------------------

FEATURES_PATH = PROJECT_ROOT / "data/intermediate/features_sop12_four_dataset.csv"
SPLITS_DIR = PROJECT_ROOT / "splits/sop_v2_four_dataset"
OUTPUT_DIR = PROJECT_ROOT / "outputs/results_v2_four_dataset_cnn_pytorch"
INTERMEDIATE_DIR = PROJECT_ROOT / "data/intermediate"

ALL_DATASETS = ["matr", "hust", "sandia", "luh"]
SEEDS = [42, 123, 456, 789, 1011]
DEFAULT_WINDOWS = [100]

HP_GRIDS = {
    "full": [(f, lr) for f in (8, 16, 32) for lr in (1e-3, 3e-3, 1e-2)],  # 9 configs
    "quick": [(8, 3e-3), (16, 3e-3), (16, 1e-2), (32, 3e-3)],             # 4 configs
}

MODEL_NAME = "pytorch_1d_cnn"
SCHEMA_VERSION = "pytorch_1d_cnn_v1"


# -------------------- path helpers --------------------

def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def cycle_table_path(dataset: str) -> Path:
    return PROJECT_ROOT / f"data/intermediate/{dataset}_cycles_tidy.csv"


# -------------------- device --------------------

def select_device(requested: str | None) -> torch.device:
    """Pick a PyTorch device, with explicit-override and auto-detect modes.

    Priority when ``requested`` is ``None`` (auto): MPS → CUDA → CPU.
    Apple Silicon laptops (M1/M2/M3) hit the MPS branch; NVIDIA workstations
    hit CUDA; everything else falls back to CPU. Passing ``"mps"`` or
    ``"cuda"`` explicitly raises if the requested backend is not available,
    which is the desired behaviour in CI / reproducibility scripts.

    MPS results are not fully deterministic between sessions because some
    reduction kernels are non-deterministic on Apple GPUs; pass
    ``"cpu"`` for bit-exact reproduction at ~5–6× wall-clock cost.
    """
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise SystemExit("--device mps requested but MPS not available")
        return torch.device("mps")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("--device cuda requested but CUDA not available")
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# -------------------- data --------------------

def load_split(splits_dir: Path, dataset: str, seed: int) -> dict:
    path = splits_dir / f"{dataset}_{seed}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")
    return json.loads(path.read_text())


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
    """Convert per-cycle ``Q_discharge`` records into model-ready sequence tensors.

    For each modelled (uncensored) cell present in ``feature_df`` and in
    one of the requested ``datasets``:

    1. Look up its per-cycle ``Q_discharge`` series from ``cycles_df``.
    2. Interpolate the series onto integer cycles ``2..n_cycles``.
    3. Normalize by the cell's ``q0`` to get retention ``q_discharge/q0``.
    4. Compute the first difference of retention.
    5. Stack ``(retention, diff)`` as two channels.

    Cells missing cycle coverage, with non-positive ``q0``, or otherwise
    malformed are skipped (and reported via ``[warn]``); this matches the
    behaviour of the previous NumPy CNN so the two histories of results
    are directly comparable.

    Parameters
    ----------
    feature_df : pandas.DataFrame
        Capacity-only feature table (``features_sop12_four_dataset.csv``).
        We only read its ``cell_id``, ``dataset``, ``n_cycles``, ``q0``,
        ``cycle_life``, ``is_censored`` columns.
    cycles_df : pandas.DataFrame
        Per-cycle tidy table from :func:`load_cycle_tables`.
    datasets : list of str
        Subset of ``{"matr", "hust", "sandia", "luh"}`` to include.
    n_cycles : int
        Length of the cycle window (50 or 100). Sequences span cycles
        ``2..n_cycles`` (length ``n_cycles - 1``).

    Returns
    -------
    X : np.ndarray, shape (n_cells, 2, n_cycles - 1), float32
        Sequence tensor.
    y : np.ndarray, shape (n_cells,), float32
        Cycle-life targets.
    meta : pandas.DataFrame
        Row-aligned metadata (``dataset``, ``cell_id``, ``n_cycles``,
        ``q0``, ``cycle_life``).
    """
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
        seqs.append(seq.astype(np.float32))
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
    return (
        np.stack(seqs, axis=0).astype(np.float32),
        np.asarray([r["cycle_life"] for r in rows], dtype=np.float32),
        pd.DataFrame(rows),
    )


def subset_indices(meta: pd.DataFrame, cells: list[str]) -> np.ndarray:
    mask = meta["cell_id"].isin(cells).to_numpy()
    return np.flatnonzero(mask)


def fit_sequence_scaler(X_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel mean/std fit on the *train* set only.

    Returns broadcastable ``(1, 2, 1)`` arrays so :func:`apply_sequence_scaler`
    can z-score retention and diff channels independently. Zero-std
    channels are protected with a floor of 1.0 to avoid division blow-up.
    """
    mean = X_train.mean(axis=(0, 2), keepdims=True)
    std = X_train.std(axis=(0, 2), keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def apply_sequence_scaler(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Apply a fitted scaler to a sequence tensor, with sanitization and clipping.

    Replaces NaN/inf with zero (a post-z-score zero is mean-of-distribution
    in the absence of better information) and clips to ``[-20, 20]`` to
    prevent extreme-tail inputs from destabilizing CNN training. The clip
    bound is comfortably outside the empirical range of standardized
    retention values on all four datasets.
    """
    out = (X - mean) / std
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(out, -20.0, 20.0).astype(np.float32)


# -------------------- model --------------------

class CNN1D(nn.Module):
    """Small two-conv 1D CNN baseline. ~2.5k params with default sizing.

    Architecture::

        Conv1d(channels → filters, k=5, padding=same) → ReLU
        Conv1d(filters → filters, k=5, padding=same) → ReLU
        mean-pool + max-pool over time → concat (2*filters)
        Linear(2*filters → hidden) → ReLU → Dropout(0.2)
        Linear(hidden → 1)

    Output is a single standardized log-cycle scalar per cell; downstream
    code multiplies back by the train-set log-life std and adds the mean
    before exp-ing into cycle space.

    Intentionally small (~2.5k parameters with defaults) to match the
    n ≤ 135 cells per dataset; a heavier architecture would overfit and
    its results would not be comparable to the seven classical baselines
    that use 5-fold CV grids of similar (small) complexity. The two-layer
    convolutional stack is the minimum that lets the receptive field span
    a meaningful chunk of the 99-cycle sequence without growing into
    Transformer territory.
    """

    def __init__(
        self,
        *,
        channels: int = 2,
        filters: int = 16,
        kernel: int = 5,
        hidden: int = 32,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        pad = kernel // 2
        self.conv1 = nn.Conv1d(channels, filters, kernel_size=kernel, padding=pad)
        self.conv2 = nn.Conv1d(filters, filters, kernel_size=kernel, padding=pad)
        self.fc1 = nn.Linear(2 * filters, hidden)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        mean_pool = h.mean(dim=2)
        max_pool = h.amax(dim=2)
        h = torch.cat([mean_pool, max_pool], dim=1)
        h = F.relu(self.fc1(h))
        h = self.dropout(h)
        return self.fc2(h).squeeze(-1)


@dataclass
class TrainInfo:
    """Convergence telemetry for one CNN training run.

    Attributes
    ----------
    best_epoch : int
        Epoch index (1-based) where the best validation MSE was observed.
    epochs_run : int
        Total epochs actually run before early-stopping or hitting
        ``epochs`` cap.
    best_val_loss : float
        Best validation MSE in standardized log-life space.
    final_train_loss : float
        Training MSE of the best-state model on the full training set.
    """

    best_epoch: int
    epochs_run: int
    best_val_loss: float
    final_train_loss: float


def seed_everything(seed: int) -> None:
    """Seed NumPy and PyTorch for reproducibility within one process.

    Note that MPS has non-deterministic kernels (some reductions); for
    bit-exact reproduction across sessions, run with ``--device cpu``.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    device: torch.device,
    seed: int,
    filters: int,
    kernel: int,
    hidden: int,
    dropout: float,
    epochs: int,
    patience: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
) -> tuple[CNN1D, TrainInfo]:
    """Train one :class:`CNN1D` with Adam, MSE loss, and early stopping on val.

    The training loop is intentionally vanilla: full-batch when n is small
    enough, mini-batches of ``batch_size`` otherwise; Adam with ``lr`` and
    L2 ``weight_decay``; MSE loss in *standardized* log-life space (caller
    pre-standardizes ``y``); early stopping after ``patience`` epochs
    without ≥1e-5 improvement on validation MSE; ``epochs`` cap as an
    upper bound.

    On exit the model holds the best-validation-state weights (not the
    final-epoch weights). When ``X_val`` is empty (very small datasets)
    train MSE acts as a fallback validation signal, which is a known weak
    spot but lets the code path keep working for the smallest splits
    rather than skipping them entirely.

    Returns the model and a :class:`TrainInfo` with convergence telemetry.
    """
    seed_everything(seed)
    model = CNN1D(
        channels=X_train.shape[1],
        filters=filters,
        kernel=kernel,
        hidden=hidden,
        dropout=dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    x_train_t = torch.from_numpy(X_train).to(device)
    y_train_t = torch.from_numpy(y_train).to(device)
    has_val = len(X_val) > 0
    if has_val:
        x_val_t = torch.from_numpy(X_val).to(device)
        y_val_t = torch.from_numpy(y_val).to(device)

    dataset = TensorDataset(x_train_t, y_train_t)
    n = len(dataset)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=min(batch_size, n),
        shuffle=True,
        generator=generator,
        drop_last=False,
    )

    best_val = float("inf")
    best_state: dict | None = None
    best_epoch = 0
    wait = 0
    epoch_run = 0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_run = epoch
        train_loss_sum = 0.0
        train_count = 0
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = F.mse_loss(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.detach()) * xb.shape[0]
            train_count += xb.shape[0]

        model.eval()
        with torch.no_grad():
            if has_val:
                val_pred = model(x_val_t)
                val_loss = float(F.mse_loss(val_pred, y_val_t).detach())
            else:
                tr_pred = model(x_train_t)
                val_loss = float(F.mse_loss(tr_pred, y_train_t).detach())

        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        best_epoch = epoch_run

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        final_train_loss = float(F.mse_loss(model(x_train_t), y_train_t).detach())

    return model, TrainInfo(
        best_epoch=int(best_epoch),
        epochs_run=int(epoch_run),
        best_val_loss=float(best_val),
        final_train_loss=float(final_train_loss),
    )


def predict_cycles(
    model: CNN1D,
    X: np.ndarray,
    *,
    device: torch.device,
    y_mean: float,
    y_std: float,
    log_lower: float,
    log_upper: float,
) -> np.ndarray:
    """Run a trained :class:`CNN1D` and convert standardized log-life back to cycles.

    Parameters
    ----------
    model : CNN1D
        Trained model.
    X : np.ndarray, shape (n, channels, time)
        Pre-scaled sequence tensor.
    device : torch.device
        Device to run inference on.
    y_mean, y_std : float
        Train-set mean and std of ``log(cycle_life)``, used to de-standardize
        the raw model output.
    log_lower, log_upper : float
        Clip bounds applied in *log* space before exp-ing. Set by the
        caller to the source-train log-life range plus a margin
        (``args.prediction_clip_std_margin * y_std``); this is the
        "source-range clipping policy" referenced in the manuscript.

    Returns
    -------
    np.ndarray, shape (n,)
        Cycle-life predictions in cycles, clipped to ``[1, 1e9]``.
    """
    if len(X) == 0:
        return np.empty(0, dtype=np.float64)
    model.eval()
    with torch.no_grad():
        x_t = torch.from_numpy(X).to(device)
        raw = model(x_t).detach().cpu().numpy()
    pred_log = np.clip(raw * y_std + y_mean, log_lower, log_upper)
    return to_cycles(pred_log, log_target=True, min_cycle=1.0, max_cycle=1e9)


# -------------------- inner HP search --------------------

def inner_cv_hp_search(
    X_train: np.ndarray,
    y_train_log_std: np.ndarray,
    *,
    device: torch.device,
    seed: int,
    hp_grid: list[tuple[int, float]],
    kernel: int,
    hidden: int,
    dropout: float,
    epochs: int,
    patience: int,
    batch_size: int,
    weight_decay: float,
    n_folds: int,
) -> tuple[tuple[int, float], dict]:
    """Inner ``n_folds``-fold CV over ``(filters, learning_rate)``.

    Mirrors the XGBoost / CatBoost CV grid in
    :file:`2_models/run_experiments.py` so the CNN HP-tuning rigour
    matches the classical lineup. Scoring is mean fold MAE in *standardized
    log-life* space (the model's training space); this is consistent with
    the loss and avoids exp() in the inner loop, which would otherwise
    amplify variance in fold MAEs and bias selection toward high-bias /
    low-variance configurations.

    For each (filters, lr) configuration, the routine runs ``n_folds``
    independent training runs (each on n_folds-1 / n_folds of the train
    set, validated on the held-out fold). The configuration with lowest
    mean fold MAE wins; a final model is then retrained by the caller
    on the full train set with that configuration.

    Parameters
    ----------
    X_train, y_train_log_std : np.ndarray
        Train-set features (already scaled) and standardized log-life
        targets.
    hp_grid : list of (int, float)
        ``(filters, lr)`` pairs. Default grid in the paper is the 3×3
        product {8, 16, 32} × {1e-3, 3e-3, 1e-2}.
    kernel, hidden, dropout, weight_decay : non-tuned hyperparameters
        Held fixed across the grid; matches classical CV grids which only
        tune 2 hyperparameters and leave architecture-level choices fixed.
    epochs, patience, batch_size : training control
        Passed through to :func:`train_one_model` for every fold.
    n_folds : int
        Number of CV folds, clamped to ``[2, n-1]``.

    Returns
    -------
    best_config : tuple of (int, float)
        Winning ``(filters, lr)``.
    info : dict
        Per-fold MAE records, the full grid considered, and best-config
        diagnostics. Persisted in the checkpoint for audit.
    """
    n = len(y_train_log_std)
    n_folds = max(2, min(n_folds, n - 1))
    rng = np.random.default_rng(seed + 1000)
    perm = rng.permutation(n)
    folds = np.array_split(perm, n_folds)

    fold_records: list[dict] = []
    best_score = float("inf")
    best_config: tuple[int, float] = hp_grid[0]
    for filters, lr in hp_grid:
        fold_maes: list[float] = []
        for fi in range(n_folds):
            val_idx = folds[fi]
            train_mask = np.ones(n, dtype=bool)
            train_mask[val_idx] = False
            train_idx = np.flatnonzero(train_mask)
            if len(train_idx) < 5 or len(val_idx) < 1:
                continue
            model, _info = train_one_model(
                X_train[train_idx],
                y_train_log_std[train_idx],
                X_train[val_idx],
                y_train_log_std[val_idx],
                device=device,
                seed=seed + 10 * fi,
                filters=filters,
                kernel=kernel,
                hidden=hidden,
                dropout=dropout,
                epochs=epochs,
                patience=patience,
                batch_size=batch_size,
                lr=lr,
                weight_decay=weight_decay,
            )
            model.eval()
            with torch.no_grad():
                val_pred = model(torch.from_numpy(X_train[val_idx]).to(device)).detach().cpu().numpy()
            fold_maes.append(float(np.mean(np.abs(val_pred - y_train_log_std[val_idx]))))

        if not fold_maes:
            continue
        mean_mae = float(np.mean(fold_maes))
        fold_records.append(
            {
                "filters": filters,
                "learning_rate": lr,
                "cv_mean_mae_std_log": mean_mae,
                "cv_fold_maes_std_log": fold_maes,
                "n_folds": len(fold_maes),
            }
        )
        if mean_mae < best_score:
            best_score = mean_mae
            best_config = (filters, lr)

    info = {
        "hp_grid": [(int(f), float(lr)) for f, lr in hp_grid],
        "fold_records": fold_records,
        "best_filters": int(best_config[0]),
        "best_learning_rate": float(best_config[1]),
        "best_cv_mae_std_log": float(best_score),
        "n_folds": int(n_folds),
    }
    return best_config, info


# -------------------- evaluation --------------------

def prediction_rows(
    meta: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray, base_row: dict
) -> list[dict]:
    """Per-cell prediction-record rows for downstream pooled-bootstrap CIs.

    Emits one row per test cell with ``y_true`` and ``y_pred`` plus the
    identifying metadata from ``base_row`` (scenario, source, target,
    seed, etc.). These rows accumulate into ``results_predictions.csv``
    and are consumed by :func:`cluster_bootstrap_ci` in
    :func:`aggregate_summary` to compute cell_id-level cluster bootstrap
    CIs (rather than naive row-level CIs that would double-count cells
    appearing in cross-dataset evaluations across multiple seeds).
    """
    rows = []
    for cell_id, dataset, true, pred in zip(
        meta["cell_id"], meta["dataset"], y_true, y_pred, strict=False
    ):
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


def _train_and_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    X_test: np.ndarray,
    y_test_cycle: np.ndarray,
    *,
    device: torch.device,
    seed: int,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict, dict]:
    """Shared core for :func:`evaluate_within` and :func:`evaluate_cross`.

    Encapsulates the parts that are identical between within-dataset and
    cross-dataset evaluation:

    1. Standardize ``y_train`` by ``log(y).mean()`` and ``log(y).std()``.
    2. Compute the source-train log-life range plus
       ``prediction_clip_std_margin × std`` margin for the prediction clip.
    3. Run :func:`inner_cv_hp_search` (or use fixed HPs if
       ``--no-hp-search``).
    4. Train one final model on the full ``X_train`` with the winning
       configuration, using ``X_cal`` for early-stopping signal.
    5. Predict on ``X_test`` and clip into source-range cycle space.

    Returns ``(y_pred_cycles, train_block_dict, _aux)``. ``train_block``
    carries the y-standardization parameters, the chosen HPs, the
    convergence telemetry, and the full HP-search audit record so the
    checkpoint JSON can be re-read offline.
    """
    y_train_log = np.log(y_train)
    y_mean = float(y_train_log.mean())
    y_std = float(y_train_log.std() if y_train_log.std() > 1e-8 else 1.0)
    y_train_fit = ((y_train_log - y_mean) / y_std).astype(np.float32)
    y_cal_fit = (
        ((np.log(y_cal) - y_mean) / y_std).astype(np.float32)
        if len(y_cal)
        else np.empty(0, dtype=np.float32)
    )
    clip_margin = args.prediction_clip_std_margin * y_std
    log_lower = float(y_train_log.min() - clip_margin)
    log_upper = float(y_train_log.max() + clip_margin)

    hp_grid = HP_GRIDS[args.hp_grid]
    if args.no_hp_search:
        best_config = (args.filters, args.learning_rate)
        hp_info = {"skipped": True, "filters": args.filters, "learning_rate": args.learning_rate}
    else:
        best_config, hp_info = inner_cv_hp_search(
            X_train,
            y_train_fit,
            device=device,
            seed=seed,
            hp_grid=hp_grid,
            kernel=args.kernel_size,
            hidden=args.hidden_units,
            dropout=args.dropout,
            epochs=args.cv_epochs,
            patience=args.cv_patience,
            batch_size=args.batch_size,
            weight_decay=args.l2,
            n_folds=args.cv_folds,
        )

    best_filters, best_lr = best_config
    model, info = train_one_model(
        X_train,
        y_train_fit,
        X_cal,
        y_cal_fit,
        device=device,
        seed=seed,
        filters=best_filters,
        kernel=args.kernel_size,
        hidden=args.hidden_units,
        dropout=args.dropout,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        lr=best_lr,
        weight_decay=args.l2,
    )
    pred = predict_cycles(
        model,
        X_test,
        device=device,
        y_mean=y_mean,
        y_std=y_std,
        log_lower=log_lower,
        log_upper=log_upper,
    )
    train_block = {
        "y_mean_log": y_mean,
        "y_std_log": y_std,
        "log_lower": log_lower,
        "log_upper": log_upper,
        "best_filters": int(best_filters),
        "best_learning_rate": float(best_lr),
        "best_epoch": info.best_epoch,
        "epochs_run": info.epochs_run,
        "best_val_loss": info.best_val_loss,
        "final_train_loss": info.final_train_loss,
        "hp_search": hp_info,
    }
    return pred, train_block, {"y_test_cycle": y_test_cycle}


def evaluate_within(
    X: np.ndarray,
    y: np.ndarray,
    meta: pd.DataFrame,
    split: dict,
    *,
    dataset: str,
    n_cycles: int,
    seed: int,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[dict, list[dict]]:
    """Within-dataset CNN evaluation for one (dataset, seed) unit of work.

    Slices ``X`` / ``y`` / ``meta`` to the dataset's train / calibration /
    test cells via the seed's split JSON, fits the sequence scaler on the
    train rows only, then routes through :func:`_train_and_predict`.

    Raises ``ValueError`` (caught upstream as a unit failure) when train
    has fewer than 5 cells or test fewer than 2.

    Returns ``(row, pred_rows)``: a single results-detailed row and a
    list of per-cell prediction rows.
    """
    train_idx = subset_indices(meta, split["train"])
    cal_idx = subset_indices(meta, split["calibration"])
    test_idx = subset_indices(meta, split["test"])
    if len(train_idx) < 5 or len(test_idx) < 2:
        raise ValueError(f"Too few cells for within {dataset} seed={seed}")

    x_mean, x_std = fit_sequence_scaler(X[train_idx])
    X_train = apply_sequence_scaler(X[train_idx], x_mean, x_std)
    X_cal = apply_sequence_scaler(X[cal_idx], x_mean, x_std)
    X_test = apply_sequence_scaler(X[test_idx], x_mean, x_std)

    pred, train_block, _ = _train_and_predict(
        X_train,
        y[train_idx],
        X_cal,
        y[cal_idx],
        X_test,
        y[test_idx],
        device=device,
        seed=seed,
        args=args,
    )
    metrics = compute_metrics(y[test_idx], pred)
    row = {
        "experiment": f"{dataset}_to_{dataset}",
        "scenario": "within_split",
        "source": dataset,
        "target": dataset,
        "model": MODEL_NAME,
        "n_cycles": int(n_cycles),
        "seed": int(seed),
        "train_cells": int(len(train_idx)),
        "calibration_cells": int(len(cal_idx)),
        "test_cells": int(len(test_idx)),
        **metrics,
        **{k: v for k, v in train_block.items() if k != "hp_search"},
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
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[dict, list[dict]]:
    """Cross-dataset CNN evaluation: train on ``source`` train, test on full ``target``.

    Train + calibration cells come from the source dataset (selected by
    the seed's source-split JSON); test cells are *all* uncensored cells
    of the target dataset (no train/cal partitioning on the target side
    — that is what the target-adapter and CP scenarios in
    :file:`conformal_prediction.py` add).

    Sequence scaler is fit on the source training rows only and applied
    unchanged to target test rows; predictions are clipped to the source
    train log-life range plus margin (the "source-range clipping" policy).

    Returns ``(row, pred_rows)`` like :func:`evaluate_within`.
    """
    source_positions = np.flatnonzero(meta["dataset"].eq(source).to_numpy())
    source_meta = meta.iloc[source_positions].reset_index(drop=True)
    train_local = subset_indices(source_meta, source_split["train"])
    cal_local = subset_indices(source_meta, source_split["calibration"])
    train_idx = source_positions[train_local]
    cal_idx = source_positions[cal_local]
    test_idx = np.flatnonzero(meta["dataset"].eq(target).to_numpy())
    if len(train_idx) < 5 or len(test_idx) < 2:
        raise ValueError(f"Too few cells for cross {source}->{target} seed={seed}")

    x_mean, x_std = fit_sequence_scaler(X[train_idx])
    X_train = apply_sequence_scaler(X[train_idx], x_mean, x_std)
    X_cal = apply_sequence_scaler(X[cal_idx], x_mean, x_std)
    X_test = apply_sequence_scaler(X[test_idx], x_mean, x_std)

    pred, train_block, _ = _train_and_predict(
        X_train,
        y[train_idx],
        X_cal,
        y[cal_idx],
        X_test,
        y[test_idx],
        device=device,
        seed=seed,
        args=args,
    )
    metrics = compute_metrics(y[test_idx], pred)
    row = {
        "experiment": f"{source}_to_{target}",
        "scenario": "naive_cross",
        "source": source,
        "target": target,
        "model": MODEL_NAME,
        "n_cycles": int(n_cycles),
        "seed": int(seed),
        "train_cells": int(len(train_idx)),
        "calibration_cells": int(len(cal_idx)),
        "test_cells": int(len(test_idx)),
        **metrics,
        **{k: v for k, v in train_block.items() if k != "hp_search"},
    }
    pred_rows = prediction_rows(meta.iloc[test_idx], y[test_idx], pred, row)
    return row, pred_rows


# -------------------- checkpoint I/O --------------------

def checkpoint_filename(scenario: str, source: str, target: str, seed: int, n_cycles: int) -> str:
    """Deterministic checkpoint filename per unit of work.

    ``within_{dataset}_seed{seed}_N{n_cycles}.json`` for within-dataset
    units; ``cross_{source}_to_{target}_seed{seed}_N{n_cycles}.json`` for
    cross-dataset units. The filename is what the resume logic uses to
    decide whether to skip a unit on rerun, so it must be stable across
    invocations.
    """
    if scenario == "within_split":
        return f"within_{source}_seed{seed}_N{n_cycles}.json"
    return f"cross_{source}_to_{target}_seed{seed}_N{n_cycles}.json"


def save_checkpoint(checkpoint_dir: Path, row: dict, pred_rows: list[dict]) -> Path:
    """Atomically write a unit-of-work checkpoint JSON.

    Writes to ``<filename>.tmp`` first, then renames to ``<filename>.json``.
    The rename is atomic on POSIX filesystems, which means a crash mid-write
    (out-of-memory, MPS reset, Ctrl-C) cannot leave a half-written
    checkpoint that would mis-trigger the resume logic.

    Stores ``{"schema": SCHEMA_VERSION, "detailed": row, "predictions":
    pred_rows}`` so the schema version can be checked on load and we can
    bump the version if the dict layout changes incompatibly.
    """
    fname = checkpoint_filename(
        row["scenario"], row["source"], row["target"], int(row["seed"]), int(row["n_cycles"])
    )
    path = checkpoint_dir / fname
    payload = {
        "schema": SCHEMA_VERSION,
        "detailed": row,
        "predictions": pred_rows,
    }
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=_json_safe))
    tmp.replace(path)  # atomic write
    return path


def load_checkpoint(path: Path) -> tuple[dict, list[dict]] | None:
    """Load a single checkpoint, returning ``None`` for missing / corrupt files.

    Returns ``None`` (treated as "not yet computed") in three cases:
    file missing, malformed JSON, or schema-version mismatch. The caller
    re-runs the unit in that case.
    """
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except Exception:
        return None
    if data.get("schema") != SCHEMA_VERSION:
        return None
    return data["detailed"], data["predictions"]


def _json_safe(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Not JSON serializable: {type(obj)}")


def load_all_checkpoints(checkpoint_dir: Path) -> tuple[list[dict], list[dict]]:
    detailed, predictions = [], []
    for path in sorted(checkpoint_dir.glob("*.json")):
        loaded = load_checkpoint(path)
        if loaded is None:
            continue
        row, pred_rows = loaded
        detailed.append(row)
        predictions.extend(pred_rows)
    return detailed, predictions


# -------------------- cluster bootstrap --------------------

def cluster_bootstrap_ci(
    pred_block: pd.DataFrame, *, n_bootstrap: int = 1000, seed: int = 42
) -> dict[str, dict[str, float]]:
    """Cluster bootstrap by ``cell_id``, with full-row expansion per sample.

    Sampling unit is the *cell*, not the row. Each of the ``n_bootstrap``
    iterations samples ``n_cells`` cell IDs with replacement, then
    concatenates *all* prediction rows belonging to those cells. Metrics
    are recomputed on the expanded block.

    This is the correct bootstrap protocol for two reasons:

    1. **Cross-dataset rows duplicate cells across seeds.** The same
       target cell ``X`` is predicted by 5 different seeds in cross-
       dataset evaluations. A naive row-level bootstrap would treat the
       same cell as 5 independent observations and produce an
       artificially tight CI.
    2. **Within-dataset test rows are mostly distinct across seeds** but
       not always (some cells happen to land in test for multiple seeds
       under lifetime-quartile stratification). Cluster bootstrap is
       agnostic — it works correctly whether cells repeat across seeds or
       not.

    Returns ``{metric: {"lower": x, "upper": y}}`` for MAE / SMAPE / R²;
    NaN bounds when fewer than 3 unique cells exist (e.g., a tiny test
    split).
    """
    unique_cells = pred_block["cell_id"].unique()
    n_cells = len(unique_cells)
    blank = {"lower": float("nan"), "upper": float("nan")}
    if n_cells < 3:
        return {m: dict(blank) for m in ["MAE", "SMAPE", "R2"]}

    cell_to_pos = {
        cell: pred_block.index[pred_block["cell_id"].eq(cell)].to_numpy()
        for cell in unique_cells
    }
    y_true_all = pred_block["y_true"].to_numpy(dtype=float)
    y_pred_all = pred_block["y_pred"].to_numpy(dtype=float)

    rng = np.random.default_rng(seed)
    scores: dict[str, list[float]] = {"MAE": [], "SMAPE": [], "R2": []}
    cells_array = np.asarray(unique_cells)
    for _ in range(n_bootstrap):
        sampled = rng.choice(cells_array, size=n_cells, replace=True)
        positions = np.concatenate([cell_to_pos[c] for c in sampled])
        m = compute_metrics(y_true_all[positions], y_pred_all[positions])
        for k in scores:
            scores[k].append(m[k])

    return {
        k: {
            "lower": float(np.percentile(scores[k], 2.5)),
            "upper": float(np.percentile(scores[k], 97.5)),
        }
        for k in scores
    }


def aggregate_summary(detailed: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-seed detailed rows into the paper-facing summary table.

    For each ``(scenario, experiment, source, target, model, n_cycles)``
    group:

    - Seed-aggregate metrics: ``MAE_mean``, ``MAE_std`` (and similarly for
      SMAPE, R²); ``n_runs`` = number of seeds in the group.
    - Cell counts (``train_cells_mean``, ``calibration_cells_mean``,
      ``test_cells_mean``) and convergence (``best_epoch_mean``) as audit
      columns.
    - Pooled cluster-bootstrap CI on the predictions DataFrame restricted
      to the same group, via :func:`cluster_bootstrap_ci`. Provides
      ``MAE_cluster_ci95_lower/upper`` (and similarly for SMAPE, R²)
      alongside ``bootstrap_prediction_rows`` and
      ``bootstrap_distinct_cells`` for transparency.

    Output is sorted by ``group_cols`` for stable diffs across reruns.
    """
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
            ci = cluster_bootstrap_ci(pred_block.reset_index(drop=True), seed=42)
            for metric in ["MAE", "SMAPE", "R2"]:
                row[f"{metric}_cluster_ci95_lower"] = ci[metric]["lower"]
                row[f"{metric}_cluster_ci95_upper"] = ci[metric]["upper"]
            row["bootstrap_prediction_rows"] = int(len(pred_block))
            row["bootstrap_distinct_cells"] = int(pred_block["cell_id"].nunique())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


# -------------------- report --------------------

def dataframe_to_markdown(df: pd.DataFrame, *, floatfmt: str = ".3f") -> str:
    headers = [str(c) for c in df.columns]
    formatted_rows = []
    for row in df.itertuples(index=False):
        cells = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                cells.append(format(float(value), floatfmt))
            else:
                cells.append(str(value))
        formatted_rows.append(cells)
    widths = [
        max([len(headers[i])] + [len(r[i]) for r in formatted_rows])
        for i in range(len(headers))
    ]
    header = "| " + " | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))) + " |"
    sep = "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |"
    body = [
        "| " + " | ".join(r[i].ljust(widths[i]) for i in range(len(headers))) + " |"
        for r in formatted_rows
    ]
    return "\n".join([header, sep, *body])


def build_report(summary: pd.DataFrame) -> str:
    within = summary[summary["scenario"].eq("within_split")].copy()
    cross = summary[summary["scenario"].eq("naive_cross")].copy()
    lines = [
        "# Four-Dataset 1D-CNN Baseline (PyTorch)",
        "",
        "Model: PyTorch Conv1D x2 -> ReLU -> global mean/max pool -> dense + dropout -> scalar log-life.",
        "Input: cycles 2..N, channels `[Q_discharge/q0, diff(Q_discharge/q0)]`.",
        "Inner 5-fold CV over (filters, learning rate). Cluster bootstrap by cell_id for pooled CIs.",
        "",
        "## Within-Dataset N=100",
        "",
    ]
    if len(within):
        table = within[
            [
                "target",
                "MAE_mean",
                "SMAPE_mean",
                "R2_mean",
                "R2_cluster_ci95_lower",
                "R2_cluster_ci95_upper",
            ]
        ].copy()
        lines.append(dataframe_to_markdown(table))
    lines += ["", "## Naive Cross-Dataset N=100 (best per target by R2)", ""]
    if len(cross):
        best = (
            cross.sort_values(["target", "R2_mean"], ascending=[True, False])
            .groupby("target", as_index=False)
            .head(3)
        )
        table = best[["experiment", "MAE_mean", "SMAPE_mean", "R2_mean"]].copy()
        lines.append(dataframe_to_markdown(table))
    lines += [
        "",
        "Interpretation: this is the PyTorch deep-learning baseline. Within-dataset HP search",
        "via inner CV mirrors the protocol used for XGBoost/CatBoost in the classical lineup.",
        "Cross-dataset failure pattern is the relevant axis for the rank-signal regime claim.",
    ]
    return "\n".join(lines)


# -------------------- main --------------------

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
    parser.add_argument(
        "--device", default=None, choices=["auto", "mps", "cuda", "cpu"], help="Default: auto"
    )

    parser.add_argument("--hp-grid", default="full", choices=list(HP_GRIDS.keys()))
    parser.add_argument("--no-hp-search", action="store_true", help="Use fixed --filters/--learning-rate, skip CV")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--cv-epochs", type=int, default=200)
    parser.add_argument("--cv-patience", type=int, default=30)
    parser.add_argument("--kernel-size", type=int, default=5)
    parser.add_argument("--hidden-units", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--filters", type=int, default=16, help="Used only with --no-hp-search")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--patience", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-3, help="Used only with --no-hp-search")
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument(
        "--prediction-clip-std-margin",
        type=float,
        default=0.0,
        help="Clip predicted log-life to source train min/max plus this many train log-life stds.",
    )

    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Skip training; rebuild summary CSVs from existing checkpoints only.",
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Ignore existing checkpoints and rerun every unit.",
    )
    return parser.parse_args()


def write_outputs(
    out_dir: Path,
    detailed: list[dict],
    predictions: list[dict],
    config: dict,
    args: argparse.Namespace,
) -> None:
    if not detailed:
        print("[warn] no completed checkpoints found; skipping summary write")
        return
    detailed_df = pd.DataFrame(detailed)
    predictions_df = pd.DataFrame(predictions)
    summary = aggregate_summary(detailed_df, predictions_df)

    detail_path = out_dir / "results_detailed.csv"
    pred_path = out_dir / "results_predictions.csv"
    summary_path = out_dir / "results_summary.csv"
    config_path = out_dir / "results_config.json"
    paper_dir = INTERMEDIATE_DIR if out_dir == resolve_path(OUTPUT_DIR) else out_dir
    report_path = paper_dir / "four_dataset_cnn_pytorch_report.md"

    detailed_df.to_csv(detail_path, index=False)
    predictions_df.to_csv(pred_path, index=False)
    summary.to_csv(summary_path, index=False)
    report_path.write_text(build_report(summary))
    config_path.write_text(json.dumps(config, indent=2, default=_json_safe))
    for path in [detail_path, pred_path, summary_path, config_path, report_path]:
        print(f"[save] {display_path(path)}")
    print(
        summary[
            ["scenario", "experiment", "MAE_mean", "SMAPE_mean", "R2_mean", "n_runs"]
        ].to_string(index=False)
    )


def main() -> int:
    """Entry point: enumerate units of work, run + checkpoint, then aggregate.

    Plans the full set of units (all enabled scenarios × seeds × windows ×
    source/target pairs), skips any whose checkpoint JSON already exists
    (resume), and runs the rest serially through :func:`evaluate_within` /
    :func:`evaluate_cross`. After every unit, the checkpoint is written
    atomically so a mid-run interruption loses at most one unit.

    On completion (or after ``--aggregate-only``) reads every checkpoint
    in the directory, calls :func:`aggregate_summary` for the paper-facing
    summary CSV, writes the detailed and predictions CSVs, regenerates
    the markdown report, and prints the summary head.

    Return code is 0 on full success, 1 if any unit raised an exception
    (the run continues past failures; the failures are listed at the end).
    """
    args = parse_args()
    features_path = resolve_path(args.features_path)
    splits_dir = resolve_path(args.splits_dir)
    out_dir = resolve_path(args.output_dir)
    checkpoint_dir = out_dir / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if args.within_only and args.cross_only:
        raise SystemExit("--within-only and --cross-only are mutually exclusive")

    device = select_device(args.device)
    print(f"[setup] device: {device}")

    config = {
        "protocol": SCHEMA_VERSION,
        "features_path": display_path(features_path),
        "splits_dir": display_path(splits_dir),
        "datasets": args.datasets,
        "windows": args.windows,
        "seeds": args.seeds,
        "device": str(device),
        "hp_grid": args.hp_grid,
        "hp_configs": HP_GRIDS[args.hp_grid] if not args.no_hp_search else "fixed",
        "kernel_size": args.kernel_size,
        "hidden_units": args.hidden_units,
        "dropout": args.dropout,
        "epochs": args.epochs,
        "patience": args.patience,
        "cv_epochs": args.cv_epochs,
        "cv_patience": args.cv_patience,
        "cv_folds": args.cv_folds,
        "batch_size": args.batch_size,
        "l2": args.l2,
        "prediction_clip_std_margin": args.prediction_clip_std_margin,
        "torch_version": torch.__version__,
    }

    if args.aggregate_only:
        detailed, predictions = load_all_checkpoints(checkpoint_dir)
        print(f"[aggregate-only] loaded {len(detailed)} units from {display_path(checkpoint_dir)}")
        write_outputs(out_dir, detailed, predictions, config, args)
        return 0

    if args.force_rerun:
        wiped = 0
        for path in checkpoint_dir.glob("*.json"):
            path.unlink()
            wiped += 1
        print(f"[force-rerun] wiped {wiped} checkpoints")

    feature_df = pd.read_csv(features_path)
    cycles_df = load_cycle_tables(args.datasets)

    print(f"[setup] features_path: {display_path(features_path)}")
    print(f"[setup] splits_dir: {display_path(splits_dir)}")
    print(f"[setup] output_dir: {display_path(out_dir)}")
    print(f"[setup] datasets={args.datasets}, windows={args.windows}, seeds={args.seeds}")
    print(f"[setup] model={MODEL_NAME} hp_grid={args.hp_grid} hp_search={not args.no_hp_search}")

    # Enumerate all units of work
    units: list[dict] = []
    for n_cycles in args.windows:
        for seed in args.seeds:
            if not args.cross_only:
                for dataset in args.datasets:
                    units.append(
                        {
                            "scenario": "within_split",
                            "source": dataset,
                            "target": dataset,
                            "seed": seed,
                            "n_cycles": n_cycles,
                        }
                    )
            if not args.within_only:
                for source in args.datasets:
                    for target in args.datasets:
                        if source == target:
                            continue
                        units.append(
                            {
                                "scenario": "naive_cross",
                                "source": source,
                                "target": target,
                                "seed": seed,
                                "n_cycles": n_cycles,
                            }
                        )

    # Filter out completed units
    todo: list[dict] = []
    for unit in units:
        fname = checkpoint_filename(
            unit["scenario"], unit["source"], unit["target"], unit["seed"], unit["n_cycles"]
        )
        if (checkpoint_dir / fname).exists():
            continue
        todo.append(unit)

    print(
        f"[plan] {len(units)} total units, {len(units) - len(todo)} already complete, "
        f"{len(todo)} to run"
    )

    # Build sequence dataset(s) once per window
    cached_sequences: dict[int, tuple[np.ndarray, np.ndarray, pd.DataFrame]] = {}
    for n_cycles in args.windows:
        cached_sequences[n_cycles] = build_sequence_dataset(
            feature_df, cycles_df, datasets=args.datasets, n_cycles=n_cycles
        )

    # Run loop
    overall_start = time.time()
    failed_units: list[tuple[dict, str]] = []
    for i, unit in enumerate(todo, start=1):
        n_cycles = unit["n_cycles"]
        X, y, meta = cached_sequences[n_cycles]
        seed = int(unit["seed"])
        unit_start = time.time()
        try:
            if unit["scenario"] == "within_split":
                dataset = unit["source"]
                split = load_split(splits_dir, dataset, seed)
                dataset_idx = np.flatnonzero(meta["dataset"].eq(dataset).to_numpy())
                row, pred_rows = evaluate_within(
                    X[dataset_idx],
                    y[dataset_idx],
                    meta.iloc[dataset_idx].reset_index(drop=True),
                    split,
                    dataset=dataset,
                    n_cycles=n_cycles,
                    seed=seed,
                    device=device,
                    args=args,
                )
            else:
                source = unit["source"]
                target = unit["target"]
                source_split = load_split(splits_dir, source, seed)
                row, pred_rows = evaluate_cross(
                    X,
                    y,
                    meta,
                    source_split,
                    source=source,
                    target=target,
                    n_cycles=n_cycles,
                    seed=seed,
                    device=device,
                    args=args,
                )
            save_checkpoint(checkpoint_dir, row, pred_rows)
            elapsed = time.time() - unit_start
            mae = row["MAE"]
            r2 = row["R2"]
            print(
                f"[{i}/{len(todo)}] {unit['scenario']} {unit['source']}->{unit['target']} "
                f"seed={seed} | MAE={mae:.1f} R2={r2:+.3f} | {elapsed:.1f}s "
                f"| elapsed {time.time() - overall_start:.0f}s"
            )
        except Exception as exc:
            failed_units.append((unit, str(exc)))
            print(f"[ERROR] {unit} failed: {exc}")

    if failed_units:
        print(f"[warn] {len(failed_units)} units failed:")
        for unit, msg in failed_units[:10]:
            print(f"  - {unit}: {msg}")

    # Aggregate from all checkpoints (includes any from previous runs)
    detailed, predictions = load_all_checkpoints(checkpoint_dir)
    print(f"[aggregate] loaded {len(detailed)} units from {display_path(checkpoint_dir)}")
    write_outputs(out_dir, detailed, predictions, config, args)
    return 0 if not failed_units else 1


if __name__ == "__main__":
    raise SystemExit(main())

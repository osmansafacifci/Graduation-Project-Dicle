#!/usr/bin/env python3
"""Small PyTorch Transformer backbone check for the four-dataset benchmark.

This is intentionally a thin wrapper around :mod:`run_cnn_pytorch`: it reuses
the same sequence construction, official splits, source-range clipping,
checkpointing, cluster bootstrap, and aggregation. The only substantive change
is the model class:

    Linear(2 -> d_model) + sinusoidal position encoding
    TransformerEncoderLayer x 4
    global mean + max pooling
    Linear -> GELU -> Dropout -> Linear

The goal is not to start a model-zoo contest; it is a bounded second deep
backbone for the "regime taxonomy is not CNN-specific" reviewer check.

Outputs:
    outputs/results_v2_four_dataset_transformer_pytorch/checkpoints/<unit>.json
    outputs/results_v2_four_dataset_transformer_pytorch/results_detailed.csv
    outputs/results_v2_four_dataset_transformer_pytorch/results_predictions.csv
    outputs/results_v2_four_dataset_transformer_pytorch/results_summary.csv
    outputs/results_v2_four_dataset_transformer_pytorch/results_config.json
    data/intermediate/four_dataset_transformer_pytorch_report.md

Usage:
    python 2_models/run_transformer_pytorch.py --hp-grid quick
    python 2_models/run_transformer_pytorch.py --aggregate-only
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
INTERMEDIATE_DIR = PROJECT_ROOT / "data" / "intermediate"
OUTPUT_DIR = PROJECT_ROOT / "outputs/results_v2_four_dataset_transformer_pytorch"

sys.path.insert(0, str(HERE))
import run_cnn_pytorch as base  # noqa: E402

MODEL_NAME = "pytorch_simple_transformer"
SCHEMA_VERSION = "pytorch_simple_transformer_v1"

TRANSFORMER_HP_GRIDS = {
    "full": [(d, lr) for d in (16, 32, 64) for lr in (1e-3, 3e-3, 1e-2)],
    "quick": [(16, 1e-3), (32, 1e-3), (32, 3e-3), (64, 1e-3)],
}


class SimpleTransformer(nn.Module):
    """Four-layer encoder over early Q/Q0 trajectories.

    The inherited training loop calls the constructor with the same argument
    names as the CNN baseline. Here ``filters`` becomes ``d_model`` and
    ``hidden`` becomes the feed-forward / dense hidden width.
    """

    def __init__(
        self,
        *,
        channels: int = 2,
        filters: int = 32,
        kernel: int = 5,
        hidden: int = 64,
        dropout: float = 0.2,
        max_len: int = 256,
        layers: int = 4,
    ) -> None:
        super().__init__()
        del kernel  # kept for API compatibility with run_cnn_pytorch
        d_model = int(filters)
        n_heads = 4 if d_model % 4 == 0 else 2 if d_model % 2 == 0 else 1
        ff_dim = max(int(hidden), 2 * d_model)

        self.input_proj = nn.Linear(channels, d_model)
        self.register_buffer("positional_encoding", self._sinusoidal_encoding(max_len, d_model), persistent=False)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(2 * d_model, ff_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(ff_dim, 1)

    @staticmethod
    def _sinusoidal_encoding(max_len: int, d_model: int) -> torch.Tensor:
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model > 1:
            pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        return pe.unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input is (batch, channels, time); Transformer expects batch-first time.
        h = x.transpose(1, 2)
        h = self.input_proj(h)
        h = h + self.positional_encoding[:, : h.shape[1], :].to(h.device)
        h = self.encoder(h)
        h = self.norm(h)
        mean_pool = h.mean(dim=1)
        max_pool = h.amax(dim=1)
        h = torch.cat([mean_pool, max_pool], dim=1)
        h = F.gelu(self.fc1(h))
        h = self.dropout(h)
        return self.fc2(h).squeeze(-1)


def dataframe_to_markdown(df: pd.DataFrame, *, floatfmt: str = ".3f") -> str:
    return base.dataframe_to_markdown(df, floatfmt=floatfmt)


def build_report(summary: pd.DataFrame) -> str:
    within = summary[summary["scenario"].eq("within_split")].copy()
    cross = summary[summary["scenario"].eq("naive_cross")].copy()
    lines = [
        "# Four-Dataset Simple Transformer Baseline (PyTorch)",
        "",
        "Model: Linear input projection + sinusoidal position encoding + TransformerEncoderLayer x4 + global mean/max pooling.",
        "Input: cycles 2..N, channels `[Q_discharge/q0, diff(Q_discharge/q0)]`.",
        "Purpose: bounded second deep-backbone check; not a new headline architecture or domain-adaptation method.",
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
        "Interpretation: use this as SI-level defensive evidence that the rank-signal regime taxonomy is not unique to a convolutional sequence model. The quick grid is intentionally small to avoid architecture fishing.",
    ]
    return "\n".join(lines)


def write_outputs(
    out_dir: Path,
    detailed: list[dict],
    predictions: list[dict],
    config: dict,
    args,
) -> None:
    if not detailed:
        print("[warn] no completed checkpoints found; skipping summary write")
        return
    detailed_df = pd.DataFrame(detailed)
    predictions_df = pd.DataFrame(predictions)
    summary = base.aggregate_summary(detailed_df, predictions_df)

    detail_path = out_dir / "results_detailed.csv"
    pred_path = out_dir / "results_predictions.csv"
    summary_path = out_dir / "results_summary.csv"
    config_path = out_dir / "results_config.json"
    paper_dir = INTERMEDIATE_DIR if out_dir == base.resolve_path(OUTPUT_DIR) else out_dir
    report_path = paper_dir / "four_dataset_transformer_pytorch_report.md"

    config = dict(config)
    config.update(
        {
            "architecture": "SimpleTransformer",
            "model_name": MODEL_NAME,
            "schema_version": SCHEMA_VERSION,
            "transformer_layers": 4,
            "hp_grid_note": "filters is d_model for the Transformer wrapper",
        }
    )
    detailed_df.to_csv(detail_path, index=False)
    predictions_df.to_csv(pred_path, index=False)
    summary.to_csv(summary_path, index=False)
    report_path.write_text(build_report(summary))
    config_path.write_text(json.dumps(config, indent=2, default=base._json_safe))
    for path in [detail_path, pred_path, summary_path, config_path, report_path]:
        print(f"[save] {base.display_path(path)}")
    print(
        summary[
            ["scenario", "experiment", "MAE_mean", "SMAPE_mean", "R2_mean", "n_runs"]
        ].to_string(index=False)
    )


def install_transformer_backend() -> None:
    base.CNN1D = SimpleTransformer
    base.MODEL_NAME = MODEL_NAME
    base.SCHEMA_VERSION = SCHEMA_VERSION
    base.OUTPUT_DIR = OUTPUT_DIR
    base.HP_GRIDS = TRANSFORMER_HP_GRIDS
    base.build_report = build_report
    base.write_outputs = write_outputs


def main() -> int:
    install_transformer_backend()
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())

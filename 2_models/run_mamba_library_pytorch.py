#!/usr/bin/env python3
"""Library-backed Mamba sequence backbone for the four-dataset benchmark.

This runner intentionally reuses the PyTorch CNN baseline harness for sequence
construction, official splits, source-range clipping, checkpointing, cluster
bootstrap, and aggregation. The model block is the official ``mamba-ssm``
Mamba implementation. This script is meant primarily for Colab/GPU runs because
the Mamba package requires Linux + NVIDIA CUDA for normal installation.

Usage:
    python 2_models/run_mamba_library_pytorch.py --hp-grid quick
    python 2_models/run_mamba_library_pytorch.py --aggregate-only
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
INTERMEDIATE_DIR = PROJECT_ROOT / "data" / "intermediate"

sys.path.insert(0, str(HERE))
import run_cnn_pytorch as base  # noqa: E402


MAMBA_OUTPUT_DIR = PROJECT_ROOT / "outputs/results_v2_four_dataset_mamba_library_pytorch"

LIBRARY_HP_GRIDS = {
    "full": [(d, lr) for d in (16, 32, 64) for lr in (1e-3, 3e-3, 1e-2)],
    "quick": [(16, 1e-3), (32, 1e-3), (32, 3e-3), (64, 1e-3)],
}


class LibraryMambaRegressor(nn.Module):
    """Small regression head around the official mamba-ssm Mamba block.

    The inherited training loop calls this constructor with the same names as
    the CNN baseline. Here ``filters`` is the Mamba ``d_model``. The block
    itself is the library-backed Mamba implementation, not a hand-written SSM.
    """

    def __init__(
        self,
        *,
        channels: int = 2,
        filters: int = 32,
        kernel: int = 5,
        hidden: int = 64,
        dropout: float = 0.2,
        layers: int = 2,
    ) -> None:
        super().__init__()
        try:
            from mamba_ssm import Mamba
        except Exception as exc:  # pragma: no cover - depends on Colab CUDA install
            raise RuntimeError(
                "Could not import mamba_ssm.Mamba. Install with "
                "`pip install mamba-ssm[causal-conv1d] --no-build-isolation` "
                "on a Linux/NVIDIA CUDA runtime."
            ) from exc

        d_model = int(filters)
        d_conv = max(2, min(int(kernel), 4))
        self.input_proj = nn.Linear(channels, d_model)
        self.blocks = nn.ModuleList(
            [
                Mamba(
                    d_model=d_model,
                    d_state=16,
                    d_conv=d_conv,
                    expand=2,
                )
                for _ in range(layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(2 * d_model, int(hidden))
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(int(hidden), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input is (batch, channels, time); Mamba expects batch-first time.
        h = x.transpose(1, 2)
        h = self.input_proj(h)
        for block in self.blocks:
            h = h + block(h)
        h = self.norm(h)
        h = torch.cat([h.mean(dim=1), h.amax(dim=1)], dim=1)
        h = F.gelu(self.fc1(h))
        h = self.dropout(h)
        return self.fc2(h).squeeze(-1)


def dataframe_to_markdown(df: pd.DataFrame, *, floatfmt: str = ".3f") -> str:
    return base.dataframe_to_markdown(df, floatfmt=floatfmt)


def build_report(summary: pd.DataFrame) -> str:
    within = summary[summary["scenario"].eq("within_split")].copy()
    cross = summary[summary["scenario"].eq("naive_cross")].copy()
    lines = [
        "# Four-Dataset Mamba Library Backbone (PyTorch)",
        "",
        "Input: cycles 2..N, channels `[Q_discharge/q0, diff(Q_discharge/q0)]`.",
        "Purpose: official mamba-ssm deep sequence backbone sensitivity check using the same splits, metrics, clipping, and checkpointing as the CNN/Transformer runners.",
        "Run details: hyperparameter grid, epoch caps, and clipping policy are recorded in the output `results_config.json`.",
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
    lines += ["", "## Naive Cross-Dataset N=100", ""]
    if len(cross):
        table = cross[["experiment", "MAE_mean", "SMAPE_mean", "R2_mean"]].copy()
        lines.append(dataframe_to_markdown(table))
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
    paper_dir = INTERMEDIATE_DIR if out_dir == base.resolve_path(MAMBA_OUTPUT_DIR) else out_dir
    report_path = paper_dir / "four_dataset_mamba_library_pytorch_report.md"

    config = dict(config)
    config.update(
        {
            "architecture": "LibraryMambaRegressor",
            "library_backbone": "mamba-ssm",
            "model_name": base.MODEL_NAME,
            "schema_version": base.SCHEMA_VERSION,
            "hp_grid_note": "filters is d_model for the library-backed sequence wrapper",
        }
    )
    detailed_df.to_csv(detail_path, index=False)
    predictions_df.to_csv(pred_path, index=False)
    summary.to_csv(summary_path, index=False)
    report_path.write_text(build_report(summary) + "\n")
    config_path.write_text(json.dumps(config, indent=2, default=base._json_safe))
    for path in [detail_path, pred_path, summary_path, config_path, report_path]:
        print(f"[save] {base.display_path(path)}")
    print(
        summary[
            ["scenario", "experiment", "MAE_mean", "SMAPE_mean", "R2_mean", "n_runs"]
        ].to_string(index=False)
    )


def install_library_backend() -> None:
    base.CNN1D = LibraryMambaRegressor
    base.MODEL_NAME = "pytorch_mamba_library"
    base.SCHEMA_VERSION = "pytorch_mamba_library_v1"
    base.OUTPUT_DIR = MAMBA_OUTPUT_DIR
    base.HP_GRIDS = LIBRARY_HP_GRIDS
    base.build_report = build_report
    base.write_outputs = write_outputs


def main() -> int:
    install_library_backend()
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())

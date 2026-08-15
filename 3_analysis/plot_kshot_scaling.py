#!/usr/bin/env python3
"""Build paper-facing k-shot scaling figure from existing four-dataset outputs.

This script does not rerun any models. It consumes:
  - outputs/results_v2_four_dataset_conformal/paper_cp_k_sweep.csv
  - data/intermediate/four_dataset_lodo_source_expert_k_sweep.csv

and writes a compact manuscript figure plus summary tables.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile
from typing import Iterable

import numpy as np
import pandas as pd

from plot_style import apply_science_style


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CP_PATH = ROOT / "outputs/results_v2_four_dataset_conformal/paper_cp_k_sweep.csv"
DEFAULT_LODO_PATH = ROOT / "data/intermediate/four_dataset_lodo_source_expert_k_sweep.csv"
DEFAULT_OUTPUT_DIR = ROOT / "outputs/results_v2_four_dataset_kshot_scaling"
DEFAULT_INTERMEDIATE_DIR = ROOT / "data/intermediate"

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))


PROTOCOL_LABELS = {
    "Target CP": "Target CP",
    "Adapted CP": "Residual-adapted CP",
    "Residual-adapted CP": "Residual-adapted CP",
    "Linear-adapted CP": "Linear-adapted CP",
}

TARGET_LABELS = {
    "matr": "MATR",
    "hust": "HUST",
    "sandia": "Sandia",
    "luh": "Luh/KIT",
}

COLORS = {
    "Target CP": "#4C78A8",
    "Residual-adapted CP": "#F58518",
    "Linear-adapted CP": "#5F3DC4",
    "MATR": "#4C78A8",
    "HUST": "#2E7D32",
    "Sandia": "#E45756",
    "Luh/KIT": "#B279A2",
}


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _quantile(values: Iterable[float], q: float) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    return float(np.quantile(arr, q))


def summarize_cp(cp: pd.DataFrame) -> pd.DataFrame:
    df = cp.copy()
    df["protocol_clean"] = df["protocol"].map(PROTOCOL_LABELS).fillna(df["protocol"])
    group_cols = ["confidence_level", "protocol_clean", "k_target"]
    rows = []
    for keys, group in df.groupby(group_cols, dropna=False):
        confidence, protocol, k = keys
        rows.append(
            {
                "block": "conformal_prediction",
                "confidence_level": confidence,
                "protocol": protocol,
                "k": int(k),
                "n_directions": group["direction"].nunique(),
                "coverage_mean": group["coverage_mean"].mean(),
                "coverage_median": group["coverage_mean"].median(),
                "coverage_q25": _quantile(group["coverage_mean"], 0.25),
                "coverage_q75": _quantile(group["coverage_mean"], 0.75),
                "coverage_min": group["coverage_mean"].min(),
                "coverage_max": group["coverage_mean"].max(),
                "median_width_mean": group["median_width_mean"].mean(),
                "median_width_median": group["median_width_mean"].median(),
                "median_width_q25": _quantile(group["median_width_mean"], 0.25),
                "median_width_q75": _quantile(group["median_width_mean"], 0.75),
                "finite_interval_fraction_mean": group[
                    "finite_interval_fraction_mean"
                ].mean(),
                "short_long_coverage_gap_mean": group[
                    "short_long_coverage_gap_mean"
                ].mean(),
            }
        )
    return pd.DataFrame(rows).sort_values(["confidence_level", "protocol", "k"])


def summarize_lodo(lodo: pd.DataFrame) -> pd.DataFrame:
    df = lodo.copy()
    df["target_label"] = df["target"].map(TARGET_LABELS).fillna(df["target"])
    cols = [
        "target_label",
        "k",
        "protocol",
        "model",
        "adapter_type",
        "MAE_mean",
        "MAE_std",
        "SMAPE_mean",
        "SMAPE_std",
        "R2_mean",
        "R2_std",
        "n_test_mean",
        "n_runs",
        "selected_source_mode",
        "selected_model_mode",
    ]
    cols = [c for c in cols if c in df.columns]
    out = df[cols].rename(columns={"target_label": "target"})
    return out.sort_values(["target", "k"])


def set_style() -> None:
    import matplotlib as mpl

    apply_science_style()
    mpl.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
            "lines.linewidth": 2.0,
        }
    )


def plot_kshot_scaling(cp_summary: pd.DataFrame, lodo_summary: pd.DataFrame, out_png: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    set_style()
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.2), constrained_layout=True)
    ax_cov, ax_width, ax_mae, ax_r2 = axes.ravel()

    cp90 = cp_summary[cp_summary["confidence_level"].round(6).eq(0.9)].copy()
    for protocol, group in cp90.groupby("protocol"):
        group = group.sort_values("k")
        color = COLORS.get(protocol, "#333333")
        x = group["k"].to_numpy(dtype=float)
        ax_cov.plot(x, group["coverage_mean"], marker="o", label=protocol, color=color)
        ax_cov.fill_between(
            x,
            group["coverage_q25"].to_numpy(dtype=float),
            group["coverage_q75"].to_numpy(dtype=float),
            color=color,
            alpha=0.14,
            linewidth=0,
        )
        ax_width.plot(
            x,
            group["median_width_median"],
            marker="o",
            label=protocol,
            color=color,
        )
        ax_width.fill_between(
            x,
            group["median_width_q25"].to_numpy(dtype=float),
            group["median_width_q75"].to_numpy(dtype=float),
            color=color,
            alpha=0.14,
            linewidth=0,
        )

    ax_cov.axhline(0.9, color="#222222", linestyle="--", linewidth=1.1, alpha=0.8)
    ax_cov.text(20.4, 0.9, "90% target", va="center", ha="left", fontsize=8)
    ax_cov.set_title("A. CP coverage reaches nominal by k=10-20")
    ax_cov.set_xlabel("Target labels used for calibration (k)")
    ax_cov.set_ylabel("Empirical coverage")
    ax_cov.set_xticks([5, 10, 15, 20])
    ax_cov.set_ylim(0.78, 0.97)
    ax_cov.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax_cov.legend(frameon=False, loc="lower right")

    ax_width.set_title("B. Target adaptation narrows intervals")
    ax_width.set_xlabel("Target labels used for calibration (k)")
    ax_width.set_ylabel("Median interval width (cycles)")
    ax_width.set_xticks([5, 10, 15, 20])
    ax_width.legend(frameon=False, loc="upper left")

    target_order = ["MATR", "HUST", "Sandia", "Luh/KIT"]
    for target in target_order:
        group = lodo_summary[lodo_summary["target"].eq(target)].sort_values("k")
        if group.empty:
            continue
        color = COLORS.get(target, "#333333")
        ax_mae.plot(group["k"], group["MAE_mean"], marker="o", label=target, color=color)
        ax_r2.plot(group["k"], group["R2_mean"], marker="o", label=target, color=color)

    ax_mae.set_title("C. LODO adaptation improves MAE with more target labels")
    ax_mae.set_xlabel("Target labels used for adaptation (k)")
    ax_mae.set_ylabel("MAE (cycles)")
    ax_mae.set_xticks([5, 10, 15, 20])
    ax_mae.legend(frameon=False, ncol=2, loc="upper right")

    ax_r2.axhline(0.0, color="#222222", linestyle="--", linewidth=1.0, alpha=0.7)
    ax_r2.set_title("D. Rank signal remains target dependent")
    ax_r2.set_xlabel("Target labels used for adaptation (k)")
    ax_r2.set_ylabel(r"$R^2$")
    ax_r2.set_xticks([5, 10, 15, 20])
    ax_r2.legend(frameon=False, ncol=2, loc="lower right")

    fig.suptitle(
        "K-shot target labels repair reliability quickly; point accuracy improves unevenly",
        fontsize=13,
        fontweight="bold",
    )
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def write_report(
    cp_summary: pd.DataFrame,
    lodo_summary: pd.DataFrame,
    path: Path,
    cp_input: Path,
    lodo_input: Path,
) -> None:
    cp90 = cp_summary[cp_summary["confidence_level"].round(6).eq(0.9)].copy()
    cp20 = cp90[cp90["k"].eq(20)].sort_values("protocol")
    lodo_wide = lodo_summary.pivot(index="target", columns="k", values="MAE_mean")
    lodo_gain = (lodo_wide[5] - lodo_wide[20]).sort_values(ascending=False)

    lines = [
        "# K-Shot Scaling Figure Summary",
        "",
        "Inputs:",
        f"- CP k sweep: `{display_path(cp_input)}`",
        f"- LODO k sweep: `{display_path(lodo_input)}`",
        "",
        "Main 90% CP result at k=20:",
        "",
        "| Protocol | Mean coverage | Median width | Finite interval fraction |",
        "|---|---:|---:|---:|",
    ]
    for _, row in cp20.iterrows():
        lines.append(
            f"| {row['protocol']} | {row['coverage_mean']:.3f} | "
            f"{row['median_width_median']:.0f} | "
            f"{row['finite_interval_fraction_mean']:.3f} |"
        )
    lines.extend(
        [
            "",
            "LODO MAE reduction from k=5 to k=20:",
            "",
            "| Target | MAE at k=5 | MAE at k=20 | Reduction |",
            "|---|---:|---:|---:|",
        ]
    )
    for target, gain in lodo_gain.items():
        lines.append(
            f"| {target} | {lodo_wide.loc[target, 5]:.1f} | "
            f"{lodo_wide.loc[target, 20]:.1f} | {gain:.1f} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cp-k-sweep", type=Path, default=DEFAULT_CP_PATH)
    parser.add_argument("--lodo-k-sweep", type=Path, default=DEFAULT_LODO_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--intermediate-dir", type=Path, default=DEFAULT_INTERMEDIATE_DIR)
    args = parser.parse_args()

    cp = pd.read_csv(args.cp_k_sweep)
    lodo = pd.read_csv(args.lodo_k_sweep)
    cp_summary = summarize_cp(cp)
    lodo_summary = summarize_lodo(lodo)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.intermediate_dir.mkdir(parents=True, exist_ok=True)

    cp_path = args.intermediate_dir / "paper_kshot_cp_scaling_summary.csv"
    lodo_path = args.intermediate_dir / "paper_kshot_lodo_scaling_summary.csv"
    report_path = args.intermediate_dir / "paper_kshot_scaling_report.md"
    figure_path = args.output_dir / "paper_kshot_scaling.png"

    cp_summary.to_csv(cp_path, index=False)
    lodo_summary.to_csv(lodo_path, index=False)
    plot_kshot_scaling(cp_summary, lodo_summary, figure_path)
    write_report(cp_summary, lodo_summary, report_path, args.cp_k_sweep, args.lodo_k_sweep)

    print(f"[save] {display_path(cp_path)}")
    print(f"[save] {display_path(lodo_path)}")
    print(f"[save] {display_path(report_path)}")
    print(f"[save] {display_path(figure_path)}")
    print(f"[save] {display_path(figure_path.with_suffix('.pdf'))}")


if __name__ == "__main__":
    main()

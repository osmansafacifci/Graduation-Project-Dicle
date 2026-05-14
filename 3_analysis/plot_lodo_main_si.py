#!/usr/bin/env python3
"""Build the LODO main-text panel and SI detail tables.

This script does not rerun the leave-one-dataset-out experiment. It consumes
the committed LODO summary outputs and writes:
  - one main-text MAE panel comparing no-target baselines, k=20 LODO, and the
    within-dataset oracle;
  - SI-ready summary/ranking tables; and
  - a compact markdown report.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INTERMEDIATE_DIR = ROOT / "data/intermediate"
DEFAULT_LODO_SUMMARY = ROOT / "outputs/results_v2_four_dataset_lodo_source_expert/results_summary.csv"
DEFAULT_WITHIN_SUMMARY = ROOT / "outputs/results_v2_four_dataset_within_34feat_log/results_summary.csv"
DEFAULT_OUTPUT_DIR = ROOT / "outputs/results_v2_four_dataset_lodo_source_expert"

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))


TARGET_ORDER = ["matr", "hust", "sandia", "luh"]
TARGET_LABELS = {
    "matr": "MATR",
    "hust": "HUST",
    "sandia": "Sandia",
    "luh": "Luh/KIT",
}

SERIES_LABELS = {
    "naive_single_source": "Naive single-source",
    "pooled_no_target": "Pooled ERM, k=0",
    "lodo_k20": "LODO, k=20",
    "within_oracle": "Within-dataset oracle",
}

SERIES_COLORS = {
    "naive_single_source": "#9E9E9E",
    "pooled_no_target": "#4C78A8",
    "lodo_k20": "#F58518",
    "within_oracle": "#54A24B",
}


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def best_row(df: pd.DataFrame, *, sort_metric: str = "MAE_mean") -> pd.Series:
    if df.empty:
        raise ValueError("Cannot choose best row from an empty DataFrame.")
    return df.sort_values(sort_metric, ascending=True).iloc[0]


def protocol_descriptor(row: pd.Series) -> str:
    protocol = str(row.get("protocol", ""))
    model = str(row.get("model", ""))
    adapter = str(row.get("adapter_type", ""))
    source = row.get("source", np.nan)
    selected_source = row.get("selected_source_mode", np.nan)
    selected_model = row.get("selected_model_mode", np.nan)

    parts = [protocol]
    if source == source:
        parts.append(f"source={source}")
    if model and model != "nan":
        parts.append(f"model={model}")
    if adapter and adapter != "nan":
        parts.append(f"adapter={adapter}")
    if selected_source == selected_source:
        parts.append(f"selected_source={selected_source}")
    if selected_model == selected_model:
        parts.append(f"selected_model={selected_model}")
    return "; ".join(parts)


def build_main_summary(lodo: pd.DataFrame, within: pd.DataFrame, k_report: int) -> pd.DataFrame:
    rows = []
    within100 = within[within["n_cycles"].eq(100)].copy()
    lodo100 = lodo[lodo["n_cycles"].eq(100)].copy()

    for target in TARGET_ORDER:
        target_lodo = lodo100[lodo100["target"].eq(target)]
        if target_lodo.empty:
            continue

        naive = best_row(target_lodo[target_lodo["protocol"].eq("source_expert_single")])
        pooled = best_row(target_lodo[target_lodo["protocol"].eq("pooled_erm")])
        lodo_k20 = best_row(
            target_lodo[
                target_lodo["k"].eq(k_report)
                & target_lodo["protocol"].isin(
                    [
                        "pooled_erm_kshot",
                        "source_expert_convex",
                        "source_expert_select",
                        "source_model_select",
                    ]
                )
            ]
        )
        oracle = best_row(within100[within100["dataset"].eq(target)])

        baseline_best_mae = min(float(naive["MAE_mean"]), float(pooled["MAE_mean"]))
        lodo_mae = float(lodo_k20["MAE_mean"])
        rows.append(
            {
                "target": target,
                "target_label": TARGET_LABELS[target],
                "naive_single_source_MAE": float(naive["MAE_mean"]),
                "naive_single_source_R2": float(naive["R2_mean"]),
                "naive_single_source_detail": protocol_descriptor(naive),
                "pooled_no_target_MAE": float(pooled["MAE_mean"]),
                "pooled_no_target_R2": float(pooled["R2_mean"]),
                "pooled_no_target_detail": protocol_descriptor(pooled),
                "lodo_k20_MAE": lodo_mae,
                "lodo_k20_R2": float(lodo_k20["R2_mean"]),
                "lodo_k20_SMAPE": float(lodo_k20["SMAPE_mean"]),
                "lodo_k20_detail": protocol_descriptor(lodo_k20),
                "within_oracle_MAE": float(oracle["MAE_mean"]),
                "within_oracle_R2": float(oracle["R2_mean"]),
                "within_oracle_detail": protocol_descriptor(
                    pd.Series(
                        {
                            "protocol": "within_dataset",
                            "model": oracle["model"],
                            "adapter_type": "none",
                        }
                    )
                ),
                "best_no_target_MAE": baseline_best_mae,
                "lodo_vs_best_no_target_MAE_delta": lodo_mae - baseline_best_mae,
                "lodo_vs_best_no_target_MAE_reduction_pct": (
                    100.0 * (baseline_best_mae - lodo_mae) / baseline_best_mae
                ),
                "lodo_minus_within_oracle_MAE": lodo_mae - float(oracle["MAE_mean"]),
            }
        )
    return pd.DataFrame(rows)


def build_best_by_k(lodo: pd.DataFrame) -> pd.DataFrame:
    candidates = lodo[
        lodo["protocol"].isin(
            [
                "pooled_erm_kshot",
                "source_expert_convex",
                "source_expert_select",
                "source_model_select",
            ]
        )
    ].copy()
    rows = []
    for (target, k), group in candidates.groupby(["target", "k"], sort=True):
        row = best_row(group)
        out = row.to_dict()
        out["target_label"] = TARGET_LABELS.get(target, target)
        out["detail"] = protocol_descriptor(row)
        rows.append(out)
    return pd.DataFrame(rows).sort_values(["target", "k"])


def build_k20_rankings(lodo: pd.DataFrame, k_report: int, n_per_target: int) -> pd.DataFrame:
    rank_pool = lodo[
        (
            lodo["k"].eq(k_report)
            & lodo["protocol"].isin(
                [
                    "pooled_erm_kshot",
                    "source_expert_convex",
                    "source_expert_select",
                    "source_model_select",
                ]
            )
        )
        | lodo["protocol"].isin(["source_expert_single", "pooled_erm", "source_expert_uniform"])
    ].copy()
    rank_pool["target_label"] = rank_pool["target"].map(TARGET_LABELS).fillna(rank_pool["target"])
    rank_pool["detail"] = rank_pool.apply(protocol_descriptor, axis=1)
    rank_pool["rank_by_target"] = (
        rank_pool.groupby("target")["MAE_mean"].rank(method="first", ascending=True).astype(int)
    )
    rank_pool = rank_pool[rank_pool["rank_by_target"].le(n_per_target)].copy()
    cols = [
        "target",
        "target_label",
        "rank_by_target",
        "protocol",
        "source",
        "model",
        "adapter_type",
        "k",
        "MAE_mean",
        "MAE_std",
        "SMAPE_mean",
        "R2_mean",
        "R2_std",
        "n_runs",
        "selected_source_mode",
        "selected_model_mode",
        "detail",
    ]
    return rank_pool[[c for c in cols if c in rank_pool.columns]].sort_values(
        ["target", "rank_by_target"]
    )


def build_protocol_family_summary(lodo: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (target, protocol), group in lodo.groupby(["target", "protocol"]):
        row = best_row(group)
        out = row.to_dict()
        out["target_label"] = TARGET_LABELS.get(target, target)
        out["detail"] = protocol_descriptor(row)
        rows.append(out)
    cols = [
        "target",
        "target_label",
        "protocol",
        "source",
        "model",
        "adapter_type",
        "k",
        "MAE_mean",
        "MAE_std",
        "SMAPE_mean",
        "R2_mean",
        "R2_std",
        "n_runs",
        "detail",
    ]
    return pd.DataFrame(rows)[cols].sort_values(["target", "MAE_mean"])


def plot_main_panel(main: pd.DataFrame, out_png: Path) -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
        }
    )

    series = ["naive_single_source", "pooled_no_target", "lodo_k20", "within_oracle"]
    labels = [SERIES_LABELS[s] for s in series]
    colors = [SERIES_COLORS[s] for s in series]
    x = np.arange(len(main))
    width = 0.19

    fig, ax = plt.subplots(figsize=(9.6, 4.8), constrained_layout=True)
    for idx, key in enumerate(series):
        values = main[f"{key}_MAE"].to_numpy(dtype=float)
        offset = (idx - (len(series) - 1) / 2) * width
        bars = ax.bar(x + offset, values, width=width, label=labels[idx], color=colors[idx])
        if key == "lodo_k20":
            for bar, row in zip(bars, main.itertuples(index=False)):
                r2_value = 0.0 if abs(row.lodo_k20_R2) < 0.005 else row.lodo_k20_R2
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 18,
                    f"R2={r2_value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    rotation=90 if bar.get_height() > 700 else 0,
                )

    ax.set_title("Leave-one-dataset-out adaptation: k=20 target labels close much of the transfer gap")
    ax.set_ylabel("MAE (cycles)")
    ax.set_xticks(x)
    ax.set_xticklabels(main["target_label"].tolist())
    ax.set_ylim(0, max(main["naive_single_source_MAE"].max(), main["pooled_no_target_MAE"].max()) * 1.15)
    ax.legend(frameon=False, ncol=2, loc="upper right")

    for idx, row in enumerate(main.itertuples(index=False)):
        gain = row.lodo_vs_best_no_target_MAE_reduction_pct
        ax.text(
            idx,
            max(row.lodo_k20_MAE, row.within_oracle_MAE) + 75,
            f"{gain:.0f}% vs best k=0",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#333333",
        )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def md_table(df: pd.DataFrame, columns: list[str], formats: dict[str, str] | None = None) -> list[str]:
    formats = formats or {}
    lines = []
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("|" + "|".join(["---" for _ in columns]) + "|")
    for _, row in df.iterrows():
        vals = []
        for col in columns:
            val = row[col]
            if col in formats and pd.notna(val):
                vals.append(format(float(val), formats[col]))
            elif pd.isna(val):
                vals.append("")
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return lines


def write_report(
    main: pd.DataFrame,
    best_by_k: pd.DataFrame,
    rankings: pd.DataFrame,
    family: pd.DataFrame,
    report_path: Path,
) -> None:
    lines = [
        "# LODO Main Panel and Supplementary Tables",
        "",
        "Main-text panel: `outputs/results_v2_four_dataset_lodo_source_expert/paper_lodo_main_panel.png`.",
        "",
        "## Main Panel Summary",
        "",
    ]
    main_view = main[
        [
            "target_label",
            "naive_single_source_MAE",
            "pooled_no_target_MAE",
            "lodo_k20_MAE",
            "within_oracle_MAE",
            "lodo_vs_best_no_target_MAE_reduction_pct",
            "lodo_k20_R2",
        ]
    ].rename(
        columns={
            "target_label": "Target",
            "naive_single_source_MAE": "Naive single-source MAE",
            "pooled_no_target_MAE": "Pooled k=0 MAE",
            "lodo_k20_MAE": "LODO k=20 MAE",
            "within_oracle_MAE": "Within oracle MAE",
            "lodo_vs_best_no_target_MAE_reduction_pct": "Reduction vs best k=0 (%)",
            "lodo_k20_R2": "LODO k=20 R2",
        }
    )
    lines.extend(
        md_table(
            main_view,
            list(main_view.columns),
            {
                "Naive single-source MAE": ".1f",
                "Pooled k=0 MAE": ".1f",
                "LODO k=20 MAE": ".1f",
                "Within oracle MAE": ".1f",
                "Reduction vs best k=0 (%)": ".1f",
                "LODO k=20 R2": ".3f",
            },
        )
    )

    lines.extend(["", "## Best Protocol by k", ""])
    sweep_view = best_by_k[
        ["target_label", "k", "protocol", "model", "adapter_type", "MAE_mean", "R2_mean"]
    ].rename(
        columns={
            "target_label": "Target",
            "k": "k",
            "protocol": "Protocol",
            "model": "Model",
            "adapter_type": "Adapter",
            "MAE_mean": "MAE",
            "R2_mean": "R2",
        }
    )
    lines.extend(md_table(sweep_view, list(sweep_view.columns), {"MAE": ".1f", "R2": ".3f"}))

    lines.extend(["", "## Top k=20 / Baseline Protocols by Target", ""])
    rank_view = rankings[
        ["target_label", "rank_by_target", "protocol", "model", "adapter_type", "k", "MAE_mean", "R2_mean"]
    ].rename(
        columns={
            "target_label": "Target",
            "rank_by_target": "Rank",
            "protocol": "Protocol",
            "model": "Model",
            "adapter_type": "Adapter",
            "MAE_mean": "MAE",
            "R2_mean": "R2",
        }
    )
    lines.extend(md_table(rank_view, list(rank_view.columns), {"MAE": ".1f", "R2": ".3f"}))

    lines.extend(["", "## Protocol Family Winners", ""])
    family_view = family[
        ["target_label", "protocol", "model", "adapter_type", "k", "MAE_mean", "R2_mean"]
    ].rename(
        columns={
            "target_label": "Target",
            "protocol": "Protocol",
            "model": "Model",
            "adapter_type": "Adapter",
            "MAE_mean": "MAE",
            "R2_mean": "R2",
        }
    )
    lines.extend(md_table(family_view, list(family_view.columns), {"MAE": ".1f", "R2": ".3f"}))
    lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lodo-summary", type=Path, default=DEFAULT_LODO_SUMMARY)
    parser.add_argument("--within-summary", type=Path, default=DEFAULT_WITHIN_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--intermediate-dir", type=Path, default=INTERMEDIATE_DIR)
    parser.add_argument("--k-report", type=int, default=20)
    parser.add_argument("--ranking-rows", type=int, default=6)
    args = parser.parse_args()

    lodo = pd.read_csv(args.lodo_summary)
    within = pd.read_csv(args.within_summary)

    main_summary = build_main_summary(lodo, within, args.k_report)
    best_by_k = build_best_by_k(lodo)
    rankings = build_k20_rankings(lodo, args.k_report, args.ranking_rows)
    family = build_protocol_family_summary(lodo)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.intermediate_dir.mkdir(parents=True, exist_ok=True)

    main_path = args.intermediate_dir / "paper_lodo_main_panel_summary.csv"
    best_by_k_path = args.intermediate_dir / "paper_lodo_si_best_by_k.csv"
    ranking_path = args.intermediate_dir / "paper_lodo_si_k20_protocol_rankings.csv"
    family_path = args.intermediate_dir / "paper_lodo_si_protocol_family_summary.csv"
    report_path = args.intermediate_dir / "paper_lodo_si_report.md"
    figure_path = args.output_dir / "paper_lodo_main_panel.png"

    main_summary.to_csv(main_path, index=False)
    best_by_k.to_csv(best_by_k_path, index=False)
    rankings.to_csv(ranking_path, index=False)
    family.to_csv(family_path, index=False)
    plot_main_panel(main_summary, figure_path)
    write_report(main_summary, best_by_k, rankings, family, report_path)

    for path in [
        main_path,
        best_by_k_path,
        ranking_path,
        family_path,
        report_path,
        figure_path,
        figure_path.with_suffix(".pdf"),
    ]:
        print(f"[save] {display_path(path)}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build a regime-stratified CP figure for the four-dataset paper extension.

Rows are ordered by conditional-shift/rank-signal regime. Columns show the
three reviewer-facing reliability diagnostics: empirical coverage, median
interval width, and finite-interval fraction. This keeps the transfer regime,
rather than the CP scenario, as the organizing axis.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CP_SUMMARY = ROOT / "outputs/results_v2_four_dataset_conformal/paper_cp_summary.csv"
DEFAULT_REGIME_SUMMARY = ROOT / "data/intermediate/four_dataset_conditional_shift_direction_summary.csv"
DEFAULT_OUTPUT_DIR = ROOT / "outputs/results_v2_four_dataset_conformal"

SCENARIO_KEEP = [
    "cross_source_calibrated_cp",
    "cross_target_calibrated_cp",
    "cross_target_adapted_cp",
]
PROTOCOL_ORDER = [
    "source_calibrated",
    "target_domain",
    "residual_adapted",
    "linear_adapted",
]
PROTOCOL_LABELS = {
    "source_calibrated": "Source-calibrated",
    "target_domain": "Target-domain",
    "residual_adapted": "Residual-adapted",
    "linear_adapted": "Linear-adapted",
}
PROTOCOL_COLORS = {
    "source_calibrated": "#9aa0a6",
    "target_domain": "#4c78a8",
    "residual_adapted": "#f58518",
    "linear_adapted": "#5f3dc4",
}
REGIME_ORDER = {
    "strong_rank_signal": 0,
    "moderate_rank_signal": 1,
    "weak_rank_signal": 2,
    "rank_signal_collapsed": 3,
    "negative_or_inverted_signal": 4,
}
REGIME_LABELS = {
    "strong_rank_signal": "strong",
    "moderate_rank_signal": "moderate",
    "weak_rank_signal": "weak",
    "rank_signal_collapsed": "collapsed",
    "negative_or_inverted_signal": "negative",
}


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    headers = [str(c) for c in df.columns]
    rows = [[str(v) for v in row] for row in df.to_numpy()]
    widths = [
        max([len(headers[i])] + [len(row[i]) for row in rows])
        for i in range(len(headers))
    ]
    header = "| " + " | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))) + " |"
    sep = "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |"
    body = [
        "| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) + " |"
        for row in rows
    ]
    return "\n".join([header, sep, *body])


def protocol_key(row: pd.Series) -> str:
    scenario = str(row.get("scenario", ""))
    adapter = str(row.get("adapter_type", ""))
    if scenario == "cross_source_calibrated_cp":
        return "source_calibrated"
    if scenario == "cross_target_calibrated_cp":
        return "target_domain"
    if scenario == "cross_target_adapted_cp" and adapter == "linear":
        return "linear_adapted"
    if scenario == "cross_target_adapted_cp":
        return "residual_adapted"
    return scenario


def load_regime_cp(cp_path: Path, regime_path: Path, confidence: float) -> pd.DataFrame:
    cp = pd.read_csv(cp_path)
    regimes = pd.read_csv(regime_path)

    cp = cp[
        cp["scenario"].isin(SCENARIO_KEEP)
        & np.isclose(cp["confidence_level"].astype(float), float(confidence))
    ].copy()
    if cp.empty:
        raise ValueError(f"No cross-CP rows found for confidence={confidence}")

    keep = [
        "source",
        "target",
        "rank_signal_class",
        "adapter_class",
        "raw_R2",
        "pearson_r",
        "linear_R2",
    ]
    keep = [c for c in keep if c in regimes.columns]
    cp = cp.merge(regimes[keep], on=["source", "target"], how="left", validate="many_to_one")
    if cp["rank_signal_class"].isna().any():
        missing = cp.loc[cp["rank_signal_class"].isna(), ["source", "target"]].drop_duplicates()
        raise ValueError(f"Missing regime labels for directions:\n{missing.to_string(index=False)}")

    cp["protocol_key"] = cp.apply(protocol_key, axis=1)
    cp["scenario_short"] = cp["protocol_key"].map(PROTOCOL_LABELS).fillna(cp["protocol_key"])
    cp["direction"] = cp["source"].str.upper() + " -> " + cp["target"].str.upper()
    cp["regime_short"] = cp["rank_signal_class"].map(REGIME_LABELS).fillna(cp["rank_signal_class"])
    cp["regime_order"] = cp["rank_signal_class"].map(REGIME_ORDER).fillna(99).astype(int)
    naive = (
        cp[cp["scenario"].eq("cross_source_calibrated_cp")]
        [["source", "target", "MAE_mean", "coverage_mean"]]
        .rename(columns={"MAE_mean": "source_cp_MAE", "coverage_mean": "source_cp_coverage"})
    )
    cp = cp.merge(naive, on=["source", "target"], how="left", validate="many_to_one")
    cp["scenario_order"] = cp["protocol_key"].map({name: i for i, name in enumerate(PROTOCOL_ORDER)})
    cp = cp.sort_values(
        ["regime_order", "source_cp_MAE", "source", "target", "scenario_order"],
        ascending=[True, True, True, True, True],
        ignore_index=True,
    )
    return cp


def write_markdown(df: pd.DataFrame, path: Path, confidence: float) -> None:
    # R²_median is the robust per-direction aggregator across the (5 seeds × 20
    # adapter repeats) per-run set. The linear adapter occasionally fits a
    # near-singular slope on k_adapter=20 cells, producing extreme per-run R²
    # that dominates an arithmetic mean (e.g., LUH→MATR linear-adapted R²_mean
    # ≈ −3.5e5 vs R²_median ≈ −0.04). The mean column is kept for audit.
    pivot_cols = [
        "direction",
        "rank_signal_class",
        "scenario_short",
        "coverage_mean",
        "coverage_wilson95_lower_mean",
        "coverage_wilson95_upper_mean",
        "median_width_mean",
        "finite_interval_fraction_mean",
        "MAE_median" if "MAE_median" in df.columns else "MAE_mean",
        "R2_median" if "R2_median" in df.columns else "R2_mean",
    ]
    table = df[pivot_cols].copy()
    r2_col = "R2_median" if "R2_median" in df.columns else "R2_mean"
    mae_col = "MAE_median" if "MAE_median" in df.columns else "MAE_mean"
    for col in [
        "coverage_mean",
        "coverage_wilson95_lower_mean",
        "coverage_wilson95_upper_mean",
        "finite_interval_fraction_mean",
        r2_col,
    ]:
        table[col] = table[col].map(lambda x: "" if pd.isna(x) else f"{x:.3f}")
    table["median_width_mean"] = table["median_width_mean"].map(lambda x: "" if pd.isna(x) else f"{x:.0f}")
    table[mae_col] = table[mae_col].map(lambda x: "" if pd.isna(x) else f"{x:.0f}")

    lines = [
        "# Regime-Stratified CP Summary",
        "",
        f"Confidence level: {confidence:.0%}. Rows are ordered by rank-signal regime, then naive source-calibrated CP MAE.",
        f"MAE and R² columns are per-direction *medians* across the 5 seeds × adapter-repeat runs. Arithmetic means are retained in `results_summary.csv` for audit.",
        "",
        dataframe_to_markdown(table),
        "",
        "Interpretation: source-calibrated CP under-covers in most regimes; target-domain and residual-adapted CP restore near-nominal coverage with finite intervals. Linear-adapted CP behaves similarly to residual-mean on most directions but occasionally produces degenerate adapter fits on small k_adapter samples — visible in `results_summary.csv` as large R²_std and very negative R²_mean rows, and tamed by reporting R²_median here.",
    ]
    path.write_text("\n".join(lines))


def plot_regime_cp(df: pd.DataFrame, path: Path, confidence: float) -> None:
    import matplotlib.pyplot as plt

    direction_order = (
        df[["direction", "regime_short", "rank_signal_class", "source_cp_MAE", "regime_order"]]
        .drop_duplicates()
        .sort_values(["regime_order", "source_cp_MAE", "direction"])
        .reset_index(drop=True)
    )
    direction_order["y"] = np.arange(len(direction_order))[::-1]
    direction_order["y_label"] = (
        direction_order["direction"] + "  [" + direction_order["regime_short"] + "]"
    )
    plot_df = df.merge(direction_order[["direction", "y"]], on="direction", how="left")

    protocols = [name for name in PROTOCOL_ORDER if name in set(plot_df["protocol_key"])]
    offset_step = 0.18 if len(protocols) > 3 else 0.22
    offsets = {
        name: offset_step * (i - (len(protocols) - 1) / 2.0)
        for i, name in enumerate(protocols)
    }

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "figure.dpi": 140,
            "savefig.dpi": 220,
        }
    )
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(16.0, 8.8),
        sharey=True,
        gridspec_kw={"width_ratios": [1.0, 1.28, 1.0], "wspace": 0.12},
    )
    metrics = [
        ("coverage_mean", "Coverage", (0.0, 1.03)),
        ("median_width_mean", "Median width (cycles)", None),
        ("finite_interval_fraction_mean", "Finite intervals", (0.0, 1.03)),
    ]

    for ax, (metric, title, xlim) in zip(axes, metrics):
        for protocol in protocols:
            block = plot_df[plot_df["protocol_key"].eq(protocol)]
            y = block["y"].to_numpy(dtype=float) + offsets[protocol]
            x = block[metric].to_numpy(dtype=float)
            ax.scatter(
                x,
                y,
                s=52,
                color=PROTOCOL_COLORS[protocol],
                label=PROTOCOL_LABELS[protocol],
                edgecolor="white",
                linewidth=0.6,
                zorder=3,
            )
            if metric == "coverage_mean":
                lower = block["coverage_wilson95_lower_mean"].to_numpy(dtype=float)
                upper = block["coverage_wilson95_upper_mean"].to_numpy(dtype=float)
                ax.errorbar(
                    x,
                    y,
                    xerr=np.vstack([x - lower, upper - x]),
                    fmt="none",
                    ecolor=PROTOCOL_COLORS[protocol],
                    elinewidth=1.0,
                    alpha=0.45,
                    zorder=2,
                )
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.25)
        ax.set_axisbelow(True)
        if xlim is not None:
            ax.set_xlim(*xlim)
        if metric == "coverage_mean":
            ax.axvline(confidence, color="#222222", linestyle="--", linewidth=1.0, alpha=0.8)
            ax.text(confidence + 0.01, len(direction_order) - 0.15, f"{confidence:.0%}", va="top", fontsize=8)
        if metric == "median_width_mean":
            max_width = np.nanmax(plot_df[metric].to_numpy(dtype=float))
            ax.set_xlim(0, max_width * 1.08)
        if metric == "finite_interval_fraction_mean":
            ax.axvline(1.0, color="#222222", linestyle=":", linewidth=1.0, alpha=0.6)

    axes[0].set_yticks(direction_order["y"])
    axes[0].set_yticklabels(direction_order["y_label"])
    axes[0].set_ylabel("Direction [rank-signal regime]")

    # Light row bands by regime to make the regime ordering visible without
    # turning the plot into a color legend puzzle.
    for _, group in direction_order.groupby("rank_signal_class", sort=False):
        y_min = group["y"].min() - 0.48
        y_max = group["y"].max() + 0.48
        for ax in axes:
            ax.axhspan(y_min, y_max, color="#f6f8fa", alpha=0.6 if len(group) > 1 else 0.35, zorder=0)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles[: len(protocols)], labels[: len(protocols)], loc="lower center", ncol=len(protocols), frameon=False, bbox_to_anchor=(0.53, 0.01))
    fig.suptitle(
        f"Cross-Dataset Conformal Prediction by Transfer Regime ({confidence:.0%} intervals)",
        x=0.53,
        y=0.985,
        fontsize=13,
        fontweight="bold",
    )
    fig.text(
        0.53,
        0.045,
        "Rows sorted by rank-signal regime and naive source-calibrated MAE; horizontal bars on coverage show Wilson 95% CI.",
        ha="center",
        fontsize=8.5,
        color="#444444",
    )
    fig.subplots_adjust(left=0.24, right=0.985, top=0.90, bottom=0.13, wspace=0.13)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cp-summary", type=Path, default=DEFAULT_CP_SUMMARY)
    parser.add_argument("--regime-summary", type=Path, default=DEFAULT_REGIME_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--confidence", type=float, nargs="+", default=[0.90, 0.95])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cp_summary = args.cp_summary if args.cp_summary.is_absolute() else ROOT / args.cp_summary
    regime_summary = args.regime_summary if args.regime_summary.is_absolute() else ROOT / args.regime_summary
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for confidence in args.confidence:
        df = load_regime_cp(cp_summary, regime_summary, confidence)
        all_rows.append(df)
        suffix = f"{int(round(confidence * 100))}"
        plot_path = output_dir / f"paper_cp_regime_stratified_{suffix}.png"
        md_path = output_dir / f"paper_cp_regime_stratified_{suffix}.md"
        plot_regime_cp(df, plot_path, confidence)
        write_markdown(df, md_path, confidence)
        print(f"[save] {display_path(plot_path)}")
        print(f"[save] {display_path(md_path)}")

    out_csv = output_dir / "paper_cp_regime_stratified.csv"
    pd.concat(all_rows, ignore_index=True).to_csv(out_csv, index=False)
    print(f"[save] {display_path(out_csv)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

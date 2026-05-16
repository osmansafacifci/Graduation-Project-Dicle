"""
Build paper-facing summaries from MAPIE conformal prediction outputs.

Primary manuscript policy:
  - confidence levels: 90% and 95% if present
  - target-domain CP k_target = 20
  - target-adapted CP shows each available adapter type
  - adapter set size k_adapter = 20

Inputs:
    outputs/results_v2_conformal/results_summary.csv

Outputs:
    outputs/results_v2_conformal/paper_cp_summary.csv
    outputs/results_v2_conformal/paper_cp_summary.md
    outputs/results_v2_conformal/paper_cp_delta_summary.csv
    outputs/results_v2_conformal/paper_cp_k_sweep.csv
    outputs/results_v2_conformal/paper_cp_coverage_width.png  (if matplotlib exists)
    outputs/results_v2_conformal/paper_cp_k_sweep_coverage.png  (if matplotlib exists)

Usage:
    python 3_analysis/summarize_conformal_results.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from plot_style import apply_science_style

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "outputs" / "results_v2_conformal"


SCENARIO_LABELS = {
    "within_split_cp": "Within-dataset CP",
    "cross_source_calibrated_cp": "Naive source-calibrated cross CP",
    "cross_target_calibrated_cp": "Target-domain CP, no adapter",
    "cross_target_adapted_cp": "Target-adapted CP",
}

ADAPTER_LABELS = {
    "residual_mean": "Residual-mean target-adapted CP",
    "linear": "Linear target-adapted CP",
}


def fmt(x: float, digits: int = 3) -> str:
    if pd.isna(x):
        return ""
    return f"{x:.{digits}f}"


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


def select_primary(summary: pd.DataFrame, k_target: int, k_adapter: int) -> pd.DataFrame:
    keep = []
    keep.append(summary[summary["scenario"].eq("within_split_cp")])
    keep.append(summary[summary["scenario"].eq("cross_source_calibrated_cp")])
    keep.append(
        summary[
            summary["scenario"].eq("cross_target_calibrated_cp")
            & summary["k_target"].eq(float(k_target))
        ]
    )
    keep.append(
        summary[
            summary["scenario"].eq("cross_target_adapted_cp")
            & summary["k_target"].eq(float(k_target))
            & summary["k_adapter"].eq(float(k_adapter))
        ]
    )
    primary = pd.concat(keep, ignore_index=True)
    primary = primary.copy()
    primary["scenario_label"] = primary["scenario"].map(SCENARIO_LABELS)
    adapted_mask = primary["scenario"].eq("cross_target_adapted_cp")
    primary.loc[adapted_mask, "scenario_label"] = (
        primary.loc[adapted_mask, "adapter_type"]
        .map(ADAPTER_LABELS)
        .fillna(primary.loc[adapted_mask, "scenario_label"])
    )
    primary["direction"] = primary["source"].str.upper() + " -> " + primary["target"].str.upper()
    primary.loc[primary["scenario"].eq("within_split_cp"), "direction"] = primary["target"].str.upper()
    if "confidence_level" not in primary.columns:
        primary["confidence_level"] = 0.90
    return primary


def build_delta(primary: pd.DataFrame) -> pd.DataFrame:
    no_adapter = primary[primary["scenario"].eq("cross_target_calibrated_cp")]
    adapted = primary[primary["scenario"].eq("cross_target_adapted_cp")]
    rows: list[dict] = []
    for _, base in no_adapter.iterrows():
        match = adapted[
            adapted["source"].eq(base["source"])
            & adapted["target"].eq(base["target"])
            & adapted["model"].eq(base["model"])
            & adapted["confidence_level"].eq(base["confidence_level"])
        ]
        if match.empty:
            continue
        for _, row in match.iterrows():
            rows.append(
                {
                    "direction": f"{str(base['source']).upper()} -> {str(base['target']).upper()}",
                    "model": base["model"],
                    "confidence_level": base["confidence_level"],
                    "adapter_type": row.get("adapter_type", ""),
                    "coverage_no_adapter": base["coverage_mean"],
                    "coverage_adapted": row["coverage_mean"],
                    "median_width_no_adapter": base["median_width_mean"],
                    "median_width_adapted": row["median_width_mean"],
                    "median_width_reduction_pct": 100.0 * (1.0 - row["median_width_mean"] / base["median_width_mean"]),
                    "finite_interval_fraction_no_adapter": base.get("finite_interval_fraction_mean", float("nan")),
                    "finite_interval_fraction_adapted": row.get("finite_interval_fraction_mean", float("nan")),
                    "MAE_no_adapter": base["MAE_mean"],
                    "MAE_adapted": row["MAE_mean"],
                    "MAE_reduction_pct": 100.0 * (1.0 - row["MAE_mean"] / base["MAE_mean"]),
                    "R2_no_adapter": base.get("R2_median", base["R2_mean"]),
                    "R2_adapted": row.get("R2_median", row["R2_mean"]),
                    "R2_mean_no_adapter": base["R2_mean"],
                    "R2_mean_adapted": row["R2_mean"],
                    "winkler_no_adapter": base["winkler_mean_mean"],
                    "winkler_adapted": row["winkler_mean_mean"],
                    "winkler_reduction_pct": 100.0 * (1.0 - row["winkler_mean_mean"] / base["winkler_mean_mean"]),
                    "short_life_coverage_adapted": row.get("coverage_short_life_mean", float("nan")),
                    "long_life_coverage_adapted": row.get("coverage_long_life_mean", float("nan")),
                    "short_long_coverage_gap_adapted": row.get("short_long_coverage_gap_mean", float("nan")),
                }
            )
    return pd.DataFrame(rows)


def build_k_sweep(summary: pd.DataFrame) -> pd.DataFrame:
    target = summary[summary["scenario"].eq("cross_target_calibrated_cp")].copy()
    adapted = summary[
        summary["scenario"].eq("cross_target_adapted_cp")
        & summary["k_target"].eq(summary["k_adapter"])
    ].copy()
    k_sweep = pd.concat([target, adapted], ignore_index=True)
    if k_sweep.empty:
        return k_sweep

    k_sweep["scenario_label"] = k_sweep["scenario"].map(SCENARIO_LABELS)
    adapted_mask = k_sweep["scenario"].eq("cross_target_adapted_cp")
    k_sweep.loc[adapted_mask, "scenario_label"] = (
        k_sweep.loc[adapted_mask, "adapter_type"]
        .map(ADAPTER_LABELS)
        .fillna(k_sweep.loc[adapted_mask, "scenario_label"])
    )
    k_sweep["protocol"] = k_sweep["scenario"].map(
        {
            "cross_target_calibrated_cp": "Target CP",
            "cross_target_adapted_cp": "Adapted CP",
        }
    )
    k_sweep.loc[adapted_mask, "protocol"] = (
        k_sweep.loc[adapted_mask, "adapter_type"]
        .map({"residual_mean": "Residual-adapted CP", "linear": "Linear-adapted CP"})
        .fillna("Adapted CP")
    )
    k_sweep["direction"] = k_sweep["source"].str.upper() + " -> " + k_sweep["target"].str.upper()
    cols = [
        "scenario_label",
        "protocol",
        "direction",
        "model",
        "confidence_level",
        "k_target",
        "k_adapter",
        "adapter_type",
        "coverage_mean",
        "coverage_wilson95_lower_mean",
        "coverage_wilson95_upper_mean",
        "finite_interval_fraction_mean",
        "coverage_short_life_mean",
        "coverage_long_life_mean",
        "short_long_coverage_gap_mean",
        "median_width_mean",
        "winkler_mean_mean",
        "MAE_median",
        "MAE_mean",
        "SMAPE_median",
        "SMAPE_mean",
        "R2_median",
        "R2_mean",
        "finite_q_mean",
        "n_runs",
    ]
    cols = [c for c in cols if c in k_sweep.columns]
    return k_sweep[cols].sort_values(
        ["confidence_level", "direction", "model", "protocol", "k_target"],
        ignore_index=True,
    )


def write_markdown(primary: pd.DataFrame, delta: pd.DataFrame, path: Path) -> None:
    cols = [
        "scenario_label",
        "confidence_level",
        "direction",
        "model",
        "coverage_mean",
        "coverage_wilson95_lower_mean",
        "coverage_wilson95_upper_mean",
        "finite_interval_fraction_mean",
        "coverage_short_life_mean",
        "coverage_long_life_mean",
        "short_long_coverage_gap_mean",
        "median_width_mean",
        "winkler_mean_mean",
        "MAE_median",
        "SMAPE_median",
        "R2_median",
        "n_runs",
    ]
    cols = [c for c in cols if c in primary.columns]
    table = primary[cols].copy()
    table.columns = [
        "Scenario",
        "Confidence",
        "Direction",
        "Model",
        "Coverage",
        "Wilson low",
        "Wilson high",
        "Finite interval frac.",
        "Short-life cov.",
        "Long-life cov.",
        "Short/long gap",
        "Median width",
        "Winkler",
        "MAE (median)",
        "sMAPE (median)",
        "R2 (median)",
        "Runs",
    ][: len(table.columns)]
    for col in ["Confidence", "Coverage", "Wilson low", "Wilson high", "Finite interval frac.", "Short-life cov.", "Long-life cov.", "Short/long gap", "R2 (median)"]:
        if col in table.columns:
            table[col] = table[col].map(lambda x: fmt(x, 3))
    for col in ["Median width", "Winkler", "MAE (median)", "sMAPE (median)"]:
        if col in table.columns:
            table[col] = table[col].map(lambda x: fmt(x, 1))
    if "Runs" in table.columns:
        table["Runs"] = table["Runs"].astype(int)

    table = table.sort_values(["Confidence", "Scenario", "Direction", "Model"])

    lines = ["# Paper CP Summary", ""]
    lines.append("Primary policy: MAPIE split CP at 90% and 95% if present; target rows use k_target=20; adapted rows use k_adapter=20 and include each available point adapter.")
    lines.append("Wilson columns are 95% Wilson score intervals for empirical coverage; short/long coverage splits each test set by observed lifetime.")
    lines.append("MAE, sMAPE and R² in this table are per-direction *medians* across the 5 seeds × adapter-repeat runs (robust to occasional degenerate linear-adapter fits on small k_adapter samples); arithmetic means stay in `results_summary.csv`.")
    lines.append("")
    lines.append(dataframe_to_markdown(table))
    lines.append("")
    if not delta.empty:
        d = delta.copy()
        for col in ["confidence_level", "coverage_no_adapter", "coverage_adapted", "finite_interval_fraction_no_adapter", "finite_interval_fraction_adapted", "R2_no_adapter", "R2_adapted", "short_life_coverage_adapted", "long_life_coverage_adapted", "short_long_coverage_gap_adapted"]:
            if col in d.columns:
                d[col] = d[col].map(lambda x: fmt(x, 3))
        for col in [
            "median_width_no_adapter",
            "median_width_adapted",
            "median_width_reduction_pct",
            "MAE_no_adapter",
            "MAE_adapted",
            "MAE_reduction_pct",
            "winkler_no_adapter",
            "winkler_adapted",
            "winkler_reduction_pct",
        ]:
            if col in d.columns:
                d[col] = d[col].map(lambda x: fmt(x, 1))
        lines.append("## Adapter Improvement Over Target-Only CP")
        lines.append("")
        lines.append(dataframe_to_markdown(d.sort_values(["confidence_level", "direction", "model"])))
        lines.append("")
    path.write_text("\n".join(lines))


def maybe_write_plot(primary: pd.DataFrame, path: Path, confidence_level: float) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    apply_science_style()
    if "confidence_level" in primary.columns:
        primary = primary[primary["confidence_level"].round(6).eq(round(confidence_level, 6))].copy()
    if primary.empty:
        return

    cross = primary[primary["scenario"].isin([
        "cross_source_calibrated_cp",
        "cross_target_calibrated_cp",
        "cross_target_adapted_cp",
    ])].copy()
    cross["protocol"] = cross["scenario"].map(
        {
            "cross_source_calibrated_cp": "Source CP",
            "cross_target_calibrated_cp": "Target CP",
            "cross_target_adapted_cp": "Adapted CP",
        }
    )
    adapted_mask = cross["scenario"].eq("cross_target_adapted_cp")
    cross.loc[adapted_mask, "protocol"] = (
        cross.loc[adapted_mask, "adapter_type"]
        .map({"residual_mean": "Residual-adapted CP", "linear": "Linear-adapted CP"})
        .fillna("Adapted CP")
    )
    cross["label"] = cross["source"].str.upper() + "->" + cross["target"].str.upper() + " " + cross["model"].str.replace("_", " ")

    labels = list(dict.fromkeys(cross["label"]))
    protocols = [p for p in ["Source CP", "Target CP", "Residual-adapted CP", "Linear-adapted CP"] if p in set(cross["protocol"])]
    colors = {
        "Source CP": "#9b2226",
        "Target CP": "#ca6702",
        "Residual-adapted CP": "#0a9396",
        "Linear-adapted CP": "#5f3dc4",
    }

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    x = range(len(labels))
    width = 0.18 if len(protocols) > 3 else 0.24
    offset_values = [width * (i - (len(protocols) - 1) / 2.0) for i in range(len(protocols))]
    offsets = dict(zip(protocols, offset_values))
    for protocol in protocols:
        sub = cross[cross["protocol"].eq(protocol)].set_index("label")
        cov = [sub.loc[label, "coverage_mean"] if label in sub.index else float("nan") for label in labels]
        wid = [sub.loc[label, "median_width_mean"] if label in sub.index else float("nan") for label in labels]
        axes[0].bar([i + offsets[protocol] for i in x], cov, width=width, label=protocol, color=colors[protocol])
        axes[1].bar([i + offsets[protocol] for i in x], wid, width=width, label=protocol, color=colors[protocol])
    axes[0].axhline(confidence_level, color="#333333", linestyle="--", linewidth=1)
    axes[0].set_ylabel("Empirical coverage")
    axes[0].set_ylim(0, 1.05)
    axes[1].set_ylabel("Median interval width (cycles)")
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(labels, rotation=25, ha="right")
    axes[0].legend(ncols=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def maybe_write_k_sweep_plot(k_sweep: pd.DataFrame, path: Path, confidence_level: float) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    apply_science_style()
    if k_sweep.empty:
        return
    plot_df = k_sweep[k_sweep["confidence_level"].round(6).eq(round(confidence_level, 6))].copy()
    if plot_df.empty:
        return

    plot_df["line_label"] = (
        plot_df["direction"]
        + " "
        + plot_df["model"].str.replace("_", " ")
        + " "
        + plot_df["protocol"]
    )
    labels = list(dict.fromkeys(plot_df["line_label"]))
    fig, ax = plt.subplots(figsize=(11, 5.5))
    colors = {"Target CP": "#ca6702", "Residual-adapted CP": "#0a9396", "Linear-adapted CP": "#5f3dc4"}
    markers = {"Target CP": "o", "Residual-adapted CP": "s", "Linear-adapted CP": "^"}
    for label in labels:
        sub = plot_df[plot_df["line_label"].eq(label)].sort_values("k_target")
        protocol = str(sub["protocol"].iloc[0])
        ax.plot(
            sub["k_target"],
            sub["coverage_mean"],
            marker=markers.get(protocol, "o"),
            linewidth=1.6,
            color=colors.get(protocol, "#333333"),
            alpha=0.75,
            label=label,
        )
    ax.axhline(confidence_level, color="#333333", linestyle="--", linewidth=1)
    ax.set_xlabel("Target calibration cells (k)")
    ax.set_ylabel("Empirical coverage")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(sorted(plot_df["k_target"].dropna().unique()))
    ax.legend(fontsize=7, ncols=2, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def write_stratified_summary(primary: pd.DataFrame, path: Path) -> None:
    cols = [
        "scenario_label",
        "confidence_level",
        "direction",
        "model",
        "coverage_mean",
        "coverage_short_life_mean",
        "coverage_long_life_mean",
        "short_long_coverage_gap_mean",
        "coverage_wilson95_lower_mean",
        "coverage_wilson95_upper_mean",
        "finite_interval_fraction_mean",
        "n_runs",
    ]
    cols = [c for c in cols if c in primary.columns]
    primary[cols].to_csv(path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--k-target", type=int, default=20)
    parser.add_argument("--k-adapter", type=int, default=20)
    parser.add_argument("--plot-confidence-level", type=float, default=0.90)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary_path = args.results_dir / "results_summary.csv"
    if not summary_path.exists():
        print(f"[error] missing {summary_path}")
        return 1

    summary = pd.read_csv(summary_path)
    primary = select_primary(summary, args.k_target, args.k_adapter)
    delta = build_delta(primary)

    out_summary = args.results_dir / "paper_cp_summary.csv"
    out_delta = args.results_dir / "paper_cp_delta_summary.csv"
    out_k_sweep = args.results_dir / "paper_cp_k_sweep.csv"
    out_stratified = args.results_dir / "paper_cp_stratified_coverage.csv"
    out_md = args.results_dir / "paper_cp_summary.md"
    out_png = args.results_dir / "paper_cp_coverage_width.png"
    out_k_sweep_png = args.results_dir / "paper_cp_k_sweep_coverage.png"

    primary.to_csv(out_summary, index=False)
    delta.to_csv(out_delta, index=False)
    k_sweep = build_k_sweep(summary)
    k_sweep.to_csv(out_k_sweep, index=False)
    write_stratified_summary(primary, out_stratified)
    write_markdown(primary, delta, out_md)
    maybe_write_plot(primary, out_png, args.plot_confidence_level)
    maybe_write_k_sweep_plot(k_sweep, out_k_sweep_png, args.plot_confidence_level)

    print(f"[save] {out_summary}")
    print(f"[save] {out_delta}")
    print(f"[save] {out_k_sweep}")
    print(f"[save] {out_stratified}")
    print(f"[save] {out_md}")
    if out_png.exists():
        print(f"[save] {out_png}")
    if out_k_sweep_png.exists():
        print(f"[save] {out_k_sweep_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

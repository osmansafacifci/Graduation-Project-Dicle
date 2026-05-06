"""
Build paper-facing summaries from MAPIE conformal prediction outputs.

Primary manuscript policy:
  - confidence level: 90%
  - target-domain CP k_target = 20
  - target-adapted CP uses residual_mean adapter
  - adapter set size k_adapter = 20
  - linear adapter rows, if present, are sensitivity only and excluded here

Inputs:
    outputs/results_v2_conformal/results_summary.csv

Outputs:
    outputs/results_v2_conformal/paper_cp_summary.csv
    outputs/results_v2_conformal/paper_cp_summary.md
    outputs/results_v2_conformal/paper_cp_delta_summary.csv
    outputs/results_v2_conformal/paper_cp_coverage_width.png  (if matplotlib exists)

Usage:
    python 3_analysis/summarize_conformal_results.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "outputs" / "results_v2_conformal"


SCENARIO_LABELS = {
    "within_split_cp": "Within-dataset CP",
    "cross_source_calibrated_cp": "Naive source-calibrated cross CP",
    "cross_target_calibrated_cp": "Target-domain CP, no adapter",
    "cross_target_adapted_cp": "Residual-mean target-adapted CP",
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
            & summary["adapter_type"].eq("residual_mean")
            & summary["k_target"].eq(float(k_target))
            & summary["k_adapter"].eq(float(k_adapter))
        ]
    )
    primary = pd.concat(keep, ignore_index=True)
    primary = primary.copy()
    primary["scenario_label"] = primary["scenario"].map(SCENARIO_LABELS)
    primary["direction"] = primary["source"].str.upper() + " -> " + primary["target"].str.upper()
    primary.loc[primary["scenario"].eq("within_split_cp"), "direction"] = primary["target"].str.upper()
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
        ]
        if match.empty:
            continue
        row = match.iloc[0]
        rows.append(
            {
                "direction": f"{str(base['source']).upper()} -> {str(base['target']).upper()}",
                "model": base["model"],
                "coverage_no_adapter": base["coverage_mean"],
                "coverage_adapted": row["coverage_mean"],
                "median_width_no_adapter": base["median_width_mean"],
                "median_width_adapted": row["median_width_mean"],
                "median_width_reduction_pct": 100.0 * (1.0 - row["median_width_mean"] / base["median_width_mean"]),
                "MAE_no_adapter": base["MAE_mean"],
                "MAE_adapted": row["MAE_mean"],
                "MAE_reduction_pct": 100.0 * (1.0 - row["MAE_mean"] / base["MAE_mean"]),
                "R2_no_adapter": base["R2_mean"],
                "R2_adapted": row["R2_mean"],
                "winkler_no_adapter": base["winkler_mean_mean"],
                "winkler_adapted": row["winkler_mean_mean"],
                "winkler_reduction_pct": 100.0 * (1.0 - row["winkler_mean_mean"] / base["winkler_mean_mean"]),
            }
        )
    return pd.DataFrame(rows)


def write_markdown(primary: pd.DataFrame, delta: pd.DataFrame, path: Path) -> None:
    cols = [
        "scenario_label",
        "direction",
        "model",
        "coverage_mean",
        "median_width_mean",
        "MAE_mean",
        "SMAPE_mean",
        "R2_mean",
        "n_runs",
    ]
    table = primary[cols].copy()
    table.columns = ["Scenario", "Direction", "Model", "Coverage", "Median width", "MAE", "sMAPE", "R2", "Runs"]
    for col in ["Coverage", "R2"]:
        table[col] = table[col].map(lambda x: fmt(x, 3))
    for col in ["Median width", "MAE", "sMAPE"]:
        table[col] = table[col].map(lambda x: fmt(x, 1))
    table["Runs"] = table["Runs"].astype(int)

    lines = ["# Paper CP Summary", ""]
    lines.append("Primary policy: 90% MAPIE split CP; target rows use k_target=20; adapted rows use residual-mean k_adapter=20.")
    lines.append("")
    lines.append(dataframe_to_markdown(table))
    lines.append("")
    if not delta.empty:
        d = delta.copy()
        for col in ["coverage_no_adapter", "coverage_adapted", "R2_no_adapter", "R2_adapted"]:
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
            d[col] = d[col].map(lambda x: fmt(x, 1))
        lines.append("## Adapter Improvement Over Target-Only CP")
        lines.append("")
        lines.append(dataframe_to_markdown(d))
        lines.append("")
    path.write_text("\n".join(lines))


def maybe_write_plot(primary: pd.DataFrame, path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
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
    cross["label"] = cross["source"].str.upper() + "->" + cross["target"].str.upper() + " " + cross["model"].str.replace("_", " ")

    labels = list(dict.fromkeys(cross["label"]))
    protocols = ["Source CP", "Target CP", "Adapted CP"]
    colors = {"Source CP": "#9b2226", "Target CP": "#ca6702", "Adapted CP": "#0a9396"}

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    x = range(len(labels))
    width = 0.24
    offsets = {"Source CP": -width, "Target CP": 0.0, "Adapted CP": width}
    for protocol in protocols:
        sub = cross[cross["protocol"].eq(protocol)].set_index("label")
        cov = [sub.loc[label, "coverage_mean"] if label in sub.index else float("nan") for label in labels]
        wid = [sub.loc[label, "median_width_mean"] if label in sub.index else float("nan") for label in labels]
        axes[0].bar([i + offsets[protocol] for i in x], cov, width=width, label=protocol, color=colors[protocol])
        axes[1].bar([i + offsets[protocol] for i in x], wid, width=width, label=protocol, color=colors[protocol])
    axes[0].axhline(0.9, color="#333333", linestyle="--", linewidth=1)
    axes[0].set_ylabel("Empirical coverage")
    axes[0].set_ylim(0, 1.05)
    axes[1].set_ylabel("Median interval width (cycles)")
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(labels, rotation=25, ha="right")
    axes[0].legend(ncols=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--k-target", type=int, default=20)
    parser.add_argument("--k-adapter", type=int, default=20)
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
    out_md = args.results_dir / "paper_cp_summary.md"
    out_png = args.results_dir / "paper_cp_coverage_width.png"

    primary.to_csv(out_summary, index=False)
    delta.to_csv(out_delta, index=False)
    write_markdown(primary, delta, out_md)
    maybe_write_plot(primary, out_png)

    print(f"[save] {out_summary}")
    print(f"[save] {out_delta}")
    print(f"[save] {out_md}")
    if out_png.exists():
        print(f"[save] {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

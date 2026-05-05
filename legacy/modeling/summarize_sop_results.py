"""Summarize SOP experiment JSON files into compact CSV and Markdown outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"
DEFAULT_CSV = DEFAULT_RESULTS_DIR / "results_sop_summary.csv"
DEFAULT_MD = DEFAULT_RESULTS_DIR / "results_sop_summary.md"
DEFAULT_INPUTS = [
    DEFAULT_RESULTS_DIR / "results_sop_stored_cycle_life.json",
    DEFAULT_RESULTS_DIR / "results_sop_eol_80pct_q0_observed_only.json",
    DEFAULT_RESULTS_DIR / "results_sop_eol_88ah_observed_only.json",
]

SCENARIO_LABELS = {
    "results_sop_stored_cycle_life": "stored_cycle_life",
    "results_sop_eol_80pct_q0_observed_only": "eol_80pct_q0_observed_only",
    "results_sop_eol_88ah_observed_only": "eol_88ah_observed_only",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize SOP result JSON files.")
    parser.add_argument("--inputs", nargs="+", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--csv-output", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--md-output", type=Path, default=DEFAULT_MD)
    return parser.parse_args()


def scenario_name_from_path(path: Path) -> str:
    return SCENARIO_LABELS.get(path.stem, path.stem)


def summarize_json(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    rows: list[dict] = []
    scenario = scenario_name_from_path(path)
    censor_summary = data.get("censor_summary") or {}

    for scope_name in ("within_dataset", "cross_dataset"):
        scope = data.get(scope_name, {})
        for direction, models in scope.items():
            for model_name, cycles in models.items():
                for window, metrics in cycles.items():
                    if not metrics:
                        continue
                    row = {
                        "scenario": scenario,
                        "scope": scope_name,
                        "direction": direction,
                        "model": model_name,
                        "window": int(window),
                        "target_column": data.get("target_column"),
                        "feature_set": data.get("feature_set"),
                        "censor_column": censor_summary.get("column"),
                        "censor_rate": censor_summary.get("censor_rate"),
                        "num_seeds": metrics.get("num_seeds"),
                        "mae_mean": metrics["MAE"]["mean"],
                        "mae_std": metrics["MAE"]["std"],
                        "smape_mean": metrics["SMAPE"]["mean"],
                        "smape_std": metrics["SMAPE"]["std"],
                        "r2_mean": metrics["R2"]["mean"],
                        "r2_std": metrics["R2"]["std"],
                    }
                    per_seed = metrics.get("per_seed") or []
                    if per_seed:
                        row["train_rows"] = per_seed[0].get("train_rows")
                        row["test_rows"] = per_seed[0].get("test_rows")
                    else:
                        row["train_rows"] = None
                        row["test_rows"] = None
                    rows.append(row)
    return rows


def format_metric(value: float | None, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{value:.{digits}f}"


def dataframe_to_markdown_table(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    rows = [[str(value) for value in row] for row in df.to_numpy().tolist()]
    table = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        table.append("| " + " | ".join(row) + " |")
    return "\n".join(table)


def build_markdown(summary_df: pd.DataFrame) -> str:
    lines: list[str] = ["# SOP Results Summary", ""]
    if summary_df.empty:
        lines.append("No non-empty experiment results were found.")
        return "\n".join(lines)

    scenario_order = [
        "stored_cycle_life",
        "eol_80pct_q0_observed_only",
        "eol_88ah_observed_only",
    ]
    for scenario in scenario_order:
        subset = summary_df[summary_df["scenario"] == scenario].copy()
        if subset.empty:
            continue
        lines.append(f"## {scenario}")
        lines.append("")

        censor_rate = subset["censor_rate"].dropna().unique().tolist()
        if censor_rate:
            lines.append(f"- Censor rate: {format_metric(censor_rate[0], 3)}")

        within = subset[subset["scope"] == "within_dataset"].copy()
        if not within.empty:
            lines.append("- Within-dataset results:")
            best = within.sort_values("mae_mean").iloc[0]
            lines.append(
                f"  Best MAE: {best['direction']} | {best['model']} | "
                f"{int(best['window'])} cycles | MAE {format_metric(best['mae_mean'])} | "
                f"SMAPE {format_metric(best['smape_mean'])}"
            )

            display_cols = ["direction", "model", "window", "mae_mean", "smape_mean", "r2_mean", "train_rows", "test_rows"]
            pretty = within[display_cols].copy()
            pretty.rename(
                columns={
                    "direction": "Direction",
                    "model": "Model",
                    "window": "Window",
                    "mae_mean": "MAE",
                    "smape_mean": "SMAPE",
                    "r2_mean": "R2",
                    "train_rows": "TrainRows",
                    "test_rows": "TestRows",
                },
                inplace=True,
            )
            pretty["MAE"] = pretty["MAE"].map(format_metric)
            pretty["SMAPE"] = pretty["SMAPE"].map(format_metric)
            pretty["R2"] = pretty["R2"].map(format_metric)
            lines.append("")
            lines.append(dataframe_to_markdown_table(pretty))
            lines.append("")

        cross = subset[subset["scope"] == "cross_dataset"].copy()
        if not cross.empty:
            lines.append("- Cross-dataset results:")
            pretty = cross[["direction", "model", "window", "mae_mean", "smape_mean", "r2_mean"]].copy()
            pretty.rename(
                columns={
                    "direction": "Direction",
                    "model": "Model",
                    "window": "Window",
                    "mae_mean": "MAE",
                    "smape_mean": "SMAPE",
                    "r2_mean": "R2",
                },
                inplace=True,
            )
            pretty["MAE"] = pretty["MAE"].map(format_metric)
            pretty["SMAPE"] = pretty["SMAPE"].map(format_metric)
            pretty["R2"] = pretty["R2"].map(format_metric)
            lines.append("")
            lines.append(dataframe_to_markdown_table(pretty))
            lines.append("")

    stored = summary_df[(summary_df["scenario"] == "stored_cycle_life") & (summary_df["scope"] == "within_dataset")]
    eol80 = summary_df[(summary_df["scenario"] == "eol_80pct_q0_observed_only") & (summary_df["scope"] == "within_dataset")]
    eol88 = summary_df[(summary_df["scenario"] == "eol_88ah_observed_only") & (summary_df["scope"] == "within_dataset")]
    lines.append("## Short Report Text")
    lines.append("")
    if not stored.empty:
        best = stored.sort_values("mae_mean").iloc[0]
        lines.append(
            "SOP12 transition feature set ile `stored_cycle_life` hedefinde en iyi within-dataset "
            f"sonuc `xgboost` ailesinde elde edildi; en dusuk MAE {format_metric(best['mae_mean'])} "
            f"ve {int(best['window'])} cycle penceresinde goruldu."
        )
    if not eol80.empty:
        best = eol80.sort_values("mae_mean").iloc[0]
        lines.append(
            "%80 Q0 observed-only deneyinde sansur oranina ragmen kullanilabilir alt kumeden sonuc alinabildi; "
            f"en iyi MAE {format_metric(best['mae_mean'])} ve SMAPE {format_metric(best['smape_mean'])} oldu."
        )
    if not eol88.empty:
        best = eol88.sort_values("mae_mean").iloc[0]
        lines.append(
            "0.88Ah observed-only deneyinde de benzer sekilde kucuk test ornegiyle sonuc uretildi; "
            f"en iyi MAE {format_metric(best['mae_mean'])} seviyesinde kaldi."
        )
    lines.append(
        "Observed-only senaryolarda her seed icin testte tek hucre bulundugu icin R2 degeri anlamsizdir; "
        "bu sonuclar pilot/yon gosterici olarak yorumlanmalidir."
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    rows: list[dict] = []
    for input_path in args.inputs:
        if not input_path.exists():
            continue
        rows.extend(summarize_json(input_path))

    summary_df = pd.DataFrame(rows).sort_values(["scenario", "scope", "direction", "model", "window"])
    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(args.csv_output, index=False)
    args.md_output.write_text(build_markdown(summary_df), encoding="utf-8")
    print(f"Saved CSV summary to {args.csv_output}")
    print(f"Saved Markdown summary to {args.md_output}")


if __name__ == "__main__":
    main()

"""
Create presentation-friendly comparison tables for model metrics.

Usage
-----
python 2_modeling_featuring/make_model_metric_tables.py
"""

from __future__ import annotations

import json
from numbers import Number
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"
RESULTS_PATH = RESULTS_DIR / "results_top8_metrics.json"
NAIVE_PATH = RESULTS_DIR / "results_naive_baselines.json"

WINDOWS = [25, 50, 100]
METRICS = ["MAE", "R2", "MAPE", "SMAPE"]
METRIC_LABELS = {
    "MAE": "MAE",
    "R2": "R2",
    "MAPE": "MAPE",
    "SMAPE": "SMAPE",
}

TABLE_CONFIGS = [
    {
        "type": "single",
        "model": "random_forest",
        "title": "Random Forest: hold-out metrics",
        "output": PROJECT_ROOT / "plots" / "table_random_forest.png",
    },
    {
        "type": "single",
        "model": "xgboost",
        "title": "XGBoost: hold-out metrics",
        "output": PROJECT_ROOT / "plots" / "table_xgboost.png",
    },
    {
        "type": "single",
        "model": "catboost",
        "title": "CatBoost: hold-out metrics",
        "output": PROJECT_ROOT / "plots" / "table_catboost.png",
    },
    {
        "type": "summary",
        "metrics": ("MAE", "R2"),
        "title": "MAE & R² comparison (with vs without Qd_std)",
        "output": PROJECT_ROOT / "plots" / "table_results_mae_r2.png",
    },
    {
        "type": "summary",
        "metrics": ("MAPE", "SMAPE"),
        "title": "MAPE & SMAPE comparison (with vs without Qd_std)",
        "output": PROJECT_ROOT / "plots" / "table_results_mape_smape.png",
    },
    {
        "type": "naive",
        "title": "Naive baseline metrics (test split)",
        "output": PROJECT_ROOT / "plots" / "naive_baselines_metrics.png",
    },
]


def load_metrics() -> dict:
    if not RESULTS_PATH.exists():
        raise SystemExit(f"Missing metrics file: {RESULTS_PATH}")
    return json.loads(RESULTS_PATH.read_text())


def load_naive_df() -> pd.DataFrame:
    if not NAIVE_PATH.exists():
        raise SystemExit(f"Missing baselines file: {NAIVE_PATH}")
    data = json.loads(NAIVE_PATH.read_text())
    df = pd.DataFrame(data)
    df.rename(columns={"Baseline": "Baseline predictor"}, inplace=True)
    return df


def build_single_model_df(data: dict, model_key: str) -> pd.DataFrame:
    entries = []
    for n in WINDOWS:
        row = {"Cycle window": n}
        for variant, label in [
            ("with_qd_std", "With Qd_std"),
            ("without_qd_std", "No Qd_std"),
        ]:
            metrics = data[model_key][variant][str(n)]
            for metric in METRICS:
                col = f"{label}\n{metric}"
                row[col] = metrics[metric]
        entries.append(row)
    return pd.DataFrame(entries)


def build_summary_df(data: dict, metric_pair: tuple[str, str]) -> pd.DataFrame:
    m1, m2 = metric_pair
    rows = []
    for model_key, payload in data.items():
        row = {"Model": model_key.replace("_", " ").title()}
        for n in WINDOWS:
            with_vals = payload["with_qd_std"][str(n)]
            without_vals = payload["without_qd_std"][str(n)]
            row[f"{n} cycles\nWith Qd_std"] = "\n".join(
                [
                    f"{METRIC_LABELS[m1]} {with_vals[m1]:.2f}",
                    f"{METRIC_LABELS[m2]} {with_vals[m2]:.2f}",
                ]
            )
            row[f"{n} cycles\nNo Qd_std"] = "\n".join(
                [
                    f"{METRIC_LABELS[m1]} {without_vals[m1]:.2f}",
                    f"{METRIC_LABELS[m2]} {without_vals[m2]:.2f}",
                ]
            )
        rows.append(row)
    return pd.DataFrame(rows)


def format_table(df: pd.DataFrame, title: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axis("off")

    header_color = "#111d34"
    row_colors = ["#f2f4f7", "white"]
    text_color = "#111d34"

    formatted = df.copy()
    for col in formatted.columns[1:]:
        formatted[col] = formatted[col].apply(
            lambda v: f"{v:.2f}" if isinstance(v, Number) else v
        )

    wide_summary = df.columns[0] == "Model"
    if wide_summary:
        cell_scale = (1.9, 2.0)
    else:
        cell_scale = (1.25, 1.35)

    table = ax.table(
        cellText=formatted.values,
        colLabels=formatted.columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(*cell_scale)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#d7d9e1")
        cell.get_text().set_ha("center")
        cell.get_text().set_va("center")
        if row == 0:
            cell.set_facecolor(header_color)
            cell.set_text_props(color="white", weight="bold", size=11)
            if wide_summary:
                cell.set_height(cell.get_height() * 2.4)
                cell.set_width(cell.get_width() * 1.35)
            else:
                cell.set_height(cell.get_height() * 2.8)
                cell.set_width(cell.get_width() * 0.92)
        else:
            cell.set_facecolor(row_colors[(row - 1) % 2])
            cell.set_text_props(color=text_color)
            if wide_summary:
                cell.set_height(cell.get_height() * 1.8)
                cell.set_width(cell.get_width() * 1.35)
            else:
                cell.set_height(cell.get_height() * 1.3)
                cell.set_width(cell.get_width() * 0.92)

    ax.set_title(title, fontsize=14, pad=20, color=text_color)
    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.95])
    output_path.parent.mkdir(exist_ok=True)
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    metrics = load_metrics()
    naive_df = load_naive_df()

    for cfg in TABLE_CONFIGS:
        table_type = cfg["type"]
        if table_type == "single":
            df = build_single_model_df(metrics, cfg["model"])
        elif table_type == "summary":
            df = build_summary_df(metrics, cfg["metrics"])
        elif table_type == "naive":
            df = naive_df.copy()
        else:
            raise ValueError(f"Unknown table type: {table_type}")

        format_table(df, cfg["title"], cfg["output"])
        print(f"Wrote {cfg['output']}")


if __name__ == "__main__":
    main()

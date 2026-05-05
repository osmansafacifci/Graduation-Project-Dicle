"""Create a styled table comparing hold-out vs grouped CV metrics."""

from __future__ import annotations

import json
from numbers import Number
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"
HOLDOUT_PATH = RESULTS_DIR / "results_top8_metrics.json"
CV_PATH = RESULTS_DIR / "results_top8_cv_metrics.json"
OUTPUT = PROJECT_ROOT / "plots" / "table_holdout_vs_cv.png"

MODELS = [
    ("Random Forest", "random_forest"),
    ("XGBoost", "xgboost"),
    ("CatBoost", "catboost"),
    ("ElasticNet", "elastic_net"),
]
CYCLES = ("25", "50", "100")
FEATURE_KEY = "with_qd_std"


def load_json(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"Missing JSON file: {path}")
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def format_cv(value_dict: dict | None, metric: str) -> str:
    if not value_dict or metric not in value_dict:
        return "-"
    mean = value_dict[metric].get("mean")
    std = value_dict[metric].get("std")
    if mean is None:
        return "-"
    if std is None or std == 0:
        return f"{mean:.2f}"
    return f"{mean:.2f} +/- {std:.2f}"


def build_dataframe(holdout: dict, cv: dict) -> pd.DataFrame:
    rows = []
    for model_label, key in MODELS:
        holdout_model = holdout.get(key, {}).get(FEATURE_KEY, {})
        cv_model = cv.get(key, {}).get(FEATURE_KEY, {})
        for cycle in CYCLES:
            holdout_metrics = holdout_model.get(cycle)
            cv_metrics = cv_model.get(int(cycle)) or cv_model.get(cycle)
            if not holdout_metrics and not cv_metrics:
                continue
            rows.append(
                {
                    "Model": model_label,
                    "n_cycles": cycle,
                    "Hold-out MAE": holdout_metrics.get("MAE") if holdout_metrics else None,
                    "CV MAE (mean +/- std)": format_cv(cv_metrics, "MAE"),
                    "Hold-out R2": holdout_metrics.get("R2") if holdout_metrics else None,
                    "CV R2 (mean +/- std)": format_cv(cv_metrics, "R2"),
                    "Hold-out MAPE": holdout_metrics.get("MAPE") if holdout_metrics else None,
                    "CV MAPE (mean +/- std)": format_cv(cv_metrics, "MAPE"),
                    "Hold-out SMAPE": holdout_metrics.get("SMAPE") if holdout_metrics else None,
                    "CV SMAPE (mean +/- std)": format_cv(cv_metrics, "SMAPE"),
                }
            )
    return pd.DataFrame(rows)


def format_table(df: pd.DataFrame, title: str, output_path: Path) -> None:
    if df.empty:
        raise SystemExit("No data to render for hold-out vs CV table.")

    formatted = df.copy()
    for col in formatted.columns:
        formatted[col] = formatted[col].apply(
            lambda v: f"{v:.2f}" if isinstance(v, Number) else v
        )

    header_color = "#111d34"
    row_colors = ["#f2f4f7", "white"]
    text_color = "#111d34"

    fig_height = max(3.0, 0.45 * len(formatted) + 1.5)
    fig, ax = plt.subplots(figsize=(14, fig_height))
    ax.axis("off")
    table = ax.table(
        cellText=formatted.values,
        colLabels=formatted.columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.3)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#d7d9e1")
        cell.get_text().set_ha("center")
        cell.get_text().set_va("center")
        if row == 0:
            cell.set_facecolor(header_color)
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
            cell.get_text().set_size(11)
            cell.set_height(cell.get_height() * 2.0)
        else:
            cell.set_facecolor(row_colors[(row - 1) % 2])
            cell.get_text().set_color(text_color)
            cell.set_height(cell.get_height() * 1.2)

    ax.set_title(title, fontsize=14, pad=20, color=text_color)
    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.95])
    output_path.parent.mkdir(exist_ok=True)
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def main() -> None:
    holdout = load_json(HOLDOUT_PATH)
    cv = load_json(CV_PATH)
    df = build_dataframe(holdout, cv)
    format_table(df, "Hold-out vs grouped CV (Qd_std retained)", OUTPUT)


if __name__ == "__main__":
    main()

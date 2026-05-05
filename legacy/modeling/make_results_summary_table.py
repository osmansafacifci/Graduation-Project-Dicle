"""Generate summary tables (PNG) for model metrics."""

from pathlib import Path
import json

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"
RESULTS_PATH = RESULTS_DIR / "results_top8_metrics.json"
PLOTS_DIR = PROJECT_ROOT / "plots"

MODELS = [
    ("Random Forest", "random_forest"),
    ("XGBoost", "xgboost"),
    ("CatBoost", "catboost"),
    ("ElasticNet", "elastic_net"),
]
CYCLES = ("25", "50", "100")
TABLE_SPECS = [
    ("mae_r2", ["MAE", "R2"]),
    ("mape_smape", ["MAPE", "SMAPE"]),
]


def load_results(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"Missing metrics JSON: {path}")
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def build_rows(data: dict, metrics: list[str]) -> list[list[str]]:
    rows: list[list[str]] = []
    for label, key in MODELS:
        model_data = data.get(key)
        if not model_data:
            continue
        for cycle in CYCLES:
            with_metrics = model_data.get("with_qd_std", {}).get(cycle)
            without_metrics = model_data.get("without_qd_std", {}).get(cycle)
            if not with_metrics or not without_metrics:
                continue
            row = [label, cycle]
            for metric in metrics:
                row.append(f"{with_metrics[metric]:.2f}")
                row.append(f"{without_metrics[metric]:.2f}")
            rows.append(row)
    return rows


def render_table(rows: list[list[str]], metrics: list[str], suffix: str) -> None:
    if not rows:
        print(f"Skipping table {suffix}: no rows to render.")
        return

    PLOTS_DIR.mkdir(exist_ok=True)
    header = ["Model", "n_cycles"]
    for metric in metrics:
        header.extend([f"{metric} (Qd_std var)", f"{metric} (Qd_std yok)"])

    table_data = [header] + rows
    fig_height = max(2.5, 0.33 * len(rows) + 1.0)
    fig, ax = plt.subplots(figsize=(10, fig_height))
    ax.axis("off")
    table = ax.table(cellText=table_data, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.1)
    metrics_label = " / ".join(metrics)
    ax.set_title(f"Model vs n_cycles - {metrics_label}", pad=12)
    fig.tight_layout()
    output_path = PLOTS_DIR / f"table_results_{suffix}.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved {output_path}")


def main() -> None:
    results = load_results(RESULTS_PATH)
    for suffix, metrics in TABLE_SPECS:
        rows = build_rows(results, metrics)
        render_table(rows, metrics, suffix)


if __name__ == "__main__":
    main()

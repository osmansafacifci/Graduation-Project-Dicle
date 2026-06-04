"""Plot manuscript Figures 2 and 3 from committed result artifacts.

Outputs
-------
Figure 2:
    outputs/results_v2_four_dataset_within_34feat_log/
        within_preds_vs_truth_classical.{pdf,png}

Figure 3:
    outputs/results_v2_four_dataset_geometric_shift/
        mahal_raw_vs_capnorm.{pdf,png}

The script does not refit models. It reads pooled held-out prediction rows from
the within-dataset JSON files and Mahalanobis summaries from the geometric-shift
CSVs.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "graduation_dicle_mpl"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import scienceplots  # noqa: F401

    plt.style.use(["science", "no-latex"])
except Exception:
    plt.style.use("default")


ROOT = Path(__file__).resolve().parents[1]

WITHIN_DIR = ROOT / "outputs" / "results_v2_four_dataset_within_34feat_log"
GEOM_DIR = ROOT / "outputs" / "results_v2_four_dataset_geometric_shift"
RAW_SHIFT = ROOT / "data" / "intermediate" / "four_dataset_geometric_shift_raw_summary.csv"
CAP_SHIFT = ROOT / "data" / "intermediate" / "four_dataset_geometric_shift_capnorm_summary.csv"

DATASET_LABELS = {
    "matr": "MATR",
    "hust": "HUST",
    "sandia": "Sandia",
    "luh": "Luh / KIT",
}

CHAMPION_MODELS = {
    "matr": "catboost",
    "hust": "random_forest",
    "sandia": "xgboost",
    "luh": "gaussian_process",
}

MODEL_LABELS = {
    "catboost": "CatBoost",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "gaussian_process": "Gaussian Process",
}

PAIR_ORDER = [
    ("matr", "hust"),
    ("matr", "sandia"),
    ("matr", "luh"),
    ("hust", "sandia"),
    ("hust", "luh"),
    ("sandia", "luh"),
]


def load_prediction_rows(dataset: str, model: str, n_cycles: int = 100) -> pd.DataFrame:
    result_path = WITHIN_DIR / f"results_within_{dataset}.json"
    data = json.loads(result_path.read_text())
    rows: list[dict] = []
    for seed, seed_block in data["per_seed"].items():
        model_block = seed_block[str(n_cycles)][model]
        for row in model_block["prediction_rows"]:
            rows.append(
                {
                    **row,
                    "seed": int(seed),
                    "dataset": dataset,
                    "model": model,
                }
            )
    return pd.DataFrame(rows)


def lookup_summary(dataset: str, model: str, n_cycles: int = 100) -> pd.Series:
    summary = pd.read_csv(WITHIN_DIR / "results_summary.csv")
    mask = (
        (summary["dataset"] == dataset)
        & (summary["model"] == model)
        & (summary["n_cycles"] == n_cycles)
    )
    if not mask.any():
        raise ValueError(f"No summary row for {dataset}/{model}/N={n_cycles}")
    return summary.loc[mask].iloc[0]


def plot_within_predictions() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 6.4), constrained_layout=True)
    axes = axes.ravel()

    color = "#1B4F72"
    edge = "#0B1F2A"

    for ax, dataset in zip(axes, CHAMPION_MODELS, strict=True):
        model = CHAMPION_MODELS[dataset]
        pred = load_prediction_rows(dataset, model)
        row = lookup_summary(dataset, model)

        x = pred["y_true"].to_numpy(dtype=float)
        y = pred["y_pred"].to_numpy(dtype=float)
        lo = max(0.0, min(x.min(), y.min()) * 0.92)
        hi = max(x.max(), y.max()) * 1.08

        ax.scatter(
            x,
            y,
            s=18,
            alpha=0.72,
            color=color,
            edgecolor=edge,
            linewidth=0.25,
        )
        ax.plot([lo, hi], [lo, hi], color="#7A1F1F", linewidth=1.0, linestyle="--")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.22, linewidth=0.5)

        title = f"{DATASET_LABELS[dataset]} - {MODEL_LABELS[model]}"
        ax.set_title(title, fontsize=10)
        ax.text(
            0.05,
            0.95,
            f"$R^2$ = {row['R2_mean']:.3f}\nMAE = {row['MAE_mean']:.1f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8.5,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#B0B0B0", "alpha": 0.85},
        )

    for ax in axes[2:]:
        ax.set_xlabel("Ground-truth cycle life")
    for ax in axes[::2]:
        ax.set_ylabel("Predicted cycle life")

    fig.suptitle("Within-dataset predictions vs ground truth", fontsize=12)
    out_base = WITHIN_DIR / "within_preds_vs_truth_classical"
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {out_base.with_suffix('.pdf')}")
    print(f"[save] {out_base.with_suffix('.png')}")


def load_mahalanobis_table() -> pd.DataFrame:
    raw = pd.read_csv(RAW_SHIFT)
    cap = pd.read_csv(CAP_SHIFT)
    raw = raw[(raw["n_cycles"] == 100) & (raw["feature_set"] == 34)]
    cap = cap[(cap["n_cycles"] == 100) & (cap["feature_set"] == 34)]
    merged = raw[["pair", "dataset_a", "dataset_b", "Mahalanobis"]].merge(
        cap[["pair", "Mahalanobis"]],
        on="pair",
        suffixes=("_raw", "_capnorm"),
        validate="one_to_one",
    )
    merged["pair_key"] = [
        tuple(sorted((a, b))) for a, b in zip(merged["dataset_a"], merged["dataset_b"], strict=False)
    ]
    order_map = {tuple(sorted(pair)): i for i, pair in enumerate(PAIR_ORDER)}
    merged["order"] = merged["pair_key"].map(order_map)
    merged = merged.sort_values("order").reset_index(drop=True)
    merged["label"] = [
        f"{DATASET_LABELS[a]}\n{DATASET_LABELS[b]}"
        for a, b in zip(merged["dataset_a"], merged["dataset_b"], strict=False)
    ]
    merged["reduction_pct"] = (
        100.0
        * (merged["Mahalanobis_raw"] - merged["Mahalanobis_capnorm"])
        / merged["Mahalanobis_raw"]
    )
    return merged


def plot_mahalanobis_raw_vs_capnorm() -> None:
    GEOM_DIR.mkdir(parents=True, exist_ok=True)
    data = load_mahalanobis_table()

    x = np.arange(len(data))
    width = 0.38
    fig, ax = plt.subplots(figsize=(7.6, 4.6), constrained_layout=True)

    raw_color = "#8C2D04"
    cap_color = "#2E7D32"
    raw_bars = ax.bar(
        x - width / 2,
        data["Mahalanobis_raw"],
        width,
        label="Raw 34 features",
        color=raw_color,
        alpha=0.88,
    )
    cap_bars = ax.bar(
        x + width / 2,
        data["Mahalanobis_capnorm"],
        width,
        label="$Q_0$-normalised",
        color=cap_color,
        alpha=0.88,
    )

    ax.set_ylabel("Mahalanobis distance")
    ax.set_xticks(x)
    ax.set_xticklabels(data["label"], rotation=0)
    ax.set_title("Mahalanobis distance: raw vs $Q_0$-normalised features", fontsize=12)
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    ax.legend(frameon=True, loc="upper right")

    ymax = float(max(data["Mahalanobis_raw"].max(), data["Mahalanobis_capnorm"].max()))
    ax.set_ylim(0, ymax * 1.18)

    for bars in (raw_bars, cap_bars):
        ax.bar_label(bars, fmt="%.1f", fontsize=7, padding=2)

    for i, row in data.iterrows():
        change = float(row["reduction_pct"])
        change_label = f"{abs(change):.0f}% {'lower' if change >= 0 else 'higher'}"
        ax.text(
            i,
            max(row["Mahalanobis_raw"], row["Mahalanobis_capnorm"]) + ymax * 0.055,
            change_label,
            ha="center",
            va="bottom",
            fontsize=7.5,
            color="#333333",
        )

    out_base = GEOM_DIR / "mahal_raw_vs_capnorm"
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {out_base.with_suffix('.pdf')}")
    print(f"[save] {out_base.with_suffix('.png')}")


def main() -> None:
    plot_within_predictions()
    plot_mahalanobis_raw_vs_capnorm()


if __name__ == "__main__":
    main()
